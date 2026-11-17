import argparse, hashlib, json, random, tempfile
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import onnx, onnxruntime as ort
from torchvision import models, transforms
from PIL import Image, ImageOps

ARCHS = ['custom_cnn', 'resnet18', 'efficientnet_b0', 'mobilenet_v3_large', 'deit_tiny']
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []; c = 3
        for d in [32, 64, 128, 192]:
            layers += [nn.Conv2d(c, d, 3, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = d
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.3), nn.Linear(192, 1))
    def forward(self, x): return self.classifier(self.features(x)).flatten()


def build_model(name):
    if name == 'custom_cnn': return CustomCNN()
    if name == 'resnet18':
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif name == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif name == 'deit_tiny':
        import timm
        m = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=1)
    else: raise ValueError(name)
    return m


def preprocess(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / np.float32(255)
    x = (x - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -np.asarray(logits, dtype=np.float64)))

TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def sample_images(folder, n, seed):
    if folder is None:
        print('WARNING: no --images folder; parity check uses random synthetic images only.')
        rng = np.random.default_rng(seed)
        return [Image.fromarray(rng.integers(0, 256, (rng.integers(200, 640), rng.integers(200, 640), 3), dtype=np.uint8))
                for _ in range(n)]
    if not Path(folder).is_dir():
        raise FileNotFoundError(f'{folder} does not exist. In Colab, /content is wiped on every new session: '
                                f'rerun the notebook\'s dataset-preparation cell first, or pass --images "{{DATA_ROOT}}/LCC_FASD_development".')
    files = sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    if not files: raise FileNotFoundError(f'No .jpg/.png/.bmp/.webp images under {folder}')
    files = random.Random(seed).sample(files, min(n, len(files)))
    images = []
    for f in files:
        with Image.open(f) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
    return images


def export_graph(model, path):
    dummy = torch.zeros(1, 3, 224, 224)
    try:
        torch.onnx.export(model, dummy, str(path), input_names=['images'], output_names=['logits'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
        return 'torchscript-opset17'
    except Exception as exc:
        print(f'  legacy exporter failed ({type(exc).__name__}: {exc}); trying dynamo exporter')
    for leftover in path.parent.glob(path.name + '*'): leftover.unlink()
    torch.onnx.export(model, (dummy,), str(path), input_names=['images'], output_names=['logits'],
                      opset_version=18, dynamo=True, external_data=False)
    return 'dynamo-opset18'


def export_one(arch, runs, images, out, atol):
    folder = runs / arch
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    threshold = float(json.loads((folder / 'threshold.json').read_text())['threshold'])
    if not 0 <= threshold <= 1: raise ValueError(f'{arch}: invalid threshold {threshold}')
    test = json.loads((folder / 'test_metrics.json').read_text()) if (folder / 'test_metrics.json').exists() else {}
    model = build_model(arch); model.load_state_dict(ck['model']); model.eval()
    model_id = f'lcc_{arch}'
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / f'{model_id}.onnx'
        with torch.no_grad(): exporter = export_graph(model, path)
        if any(p != path for p in path.parent.iterdir()):
            raise RuntimeError('Exporter wrote external data files; the website needs a single .onnx file')
        onnx.checker.check_model(str(path))
        sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        if inp.shape != [1, 3, 224, 224] or inp.type != 'tensor(float)':
            raise RuntimeError(f'Unexpected ONNX input {inp.shape} {inp.type}')
        ref, got = [], []
        for image in images:
            x = preprocess(image)
            np.testing.assert_allclose(TF(image).unsqueeze(0).numpy(), x, atol=1e-6, rtol=1e-6)
            with torch.inference_mode(): ref.append(float(model(torch.from_numpy(x)).reshape(-1).sigmoid().item()))
            got.append(float(sigmoid(sess.run(None, {inp.name: x})[0]).reshape(-1)[0]))
        ref, got = np.array(ref), np.array(got)
        max_err = float(np.abs(ref - got).max())
        flips = int(((ref >= threshold) != (got >= threshold)).sum())
        if max_err > atol: raise RuntimeError(f'{arch}: PyTorch vs ONNX score error {max_err:.2e} > {atol}')
        if flips: raise RuntimeError(f'{arch}: {flips} predictions change at the saved threshold')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = dict(model_id=model_id, architecture=arch, dataset='LCC-FASD', threshold=threshold,
                    preprocessing='pil_bilinear_imagenet_v1', sha256=sha, input_shape=[1, 3, 224, 224],
                    output='logits', precision='float32', exporter=exporter,
                    best_epoch=int(ck.get('epoch', -1)) + 1, val_auc=ck.get('val_auc'),
                    test_metrics={k: test[k] for k in ['roc_auc', 'apcer', 'bpcer', 'acer', 'accuracy', 'n'] if k in test},
                    validation_samples=len(images), max_score_error=max_err, prediction_changes=flips,
                    torch_version=str(torch.__version__), ort_version=ort.__version__)
        path.replace(out / f'{model_id}.onnx')
    (out / f'{model_id}.json').write_text(json.dumps(meta, indent=2, default=float))
    size = (out / f'{model_id}.onnx').stat().st_size / 2**20
    print(f'OK {model_id}: {size:.1f} MiB, threshold {threshold:.4f}, max score error {max_err:.1e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, type=Path, help='RESULTS_DIR from the notebook (runs_lcc_fasd)')
    p.add_argument('--images', type=Path, help='LCC_FASD_development folder for the parity check (recommended)')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--models', nargs='+', default=ARCHS, choices=ARCHS)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--atol', type=float, default=1e-4)
    a = p.parse_args()
    torch.manual_seed(42); torch.set_num_threads(2)
    if hasattr(torch.backends, 'mha'): torch.backends.mha.set_fastpath_enabled(False)
    a.out.mkdir(parents=True, exist_ok=True)
    images = sample_images(a.images, a.samples, 42)
    failed = []
    for arch in a.models:
        if not (a.runs / arch / 'best.pt').exists():
            print(f'SKIP {arch}: no best.pt'); continue
        try: export_one(arch, a.runs, images, a.out, a.atol)
        except Exception as exc:
            failed.append(arch); print(f'FAILED {arch}: {type(exc).__name__}: {exc}')
    print('Output folder:', a.out)
    if failed: raise SystemExit(f'Failed: {failed}')


if __name__ == '__main__':
    main()
