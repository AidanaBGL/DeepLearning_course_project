import os,time,json,threading,urllib.request,tempfile,hashlib
from pathlib import Path
import onnxruntime as ort
from preprocess import preprocess,sigmoid

BUNDLED_ROOT=Path(__file__).resolve().parent/'models'
ROOT=Path(os.getenv('MODEL_DIR') or BUNDLED_ROOT).expanduser().resolve()

def local_file(name,ext):
    for folder in (ROOT,BUNDLED_ROOT):
        path=folder/f'{name}.{ext}'
        if path.is_file(): return path
    return None

# Model id -> display info. Files: models/<id>.onnx + models/<id>.json,
# or env vars <ID>_ONNX_URL + <ID>_META_URL (e.g. LCC_MOBILENET_V3_LARGE_ONNX_URL).
MODELS={
    'resnet18':dict(label='ResNet18',dataset='CelebA-Spoof'),
    'vit_b_16':dict(label='ViT-B/16',dataset='CelebA-Spoof'),
    'lcc_mobilenet_v3_large':dict(label='MobileNetV3-Large',dataset='LCC-FASD'),
    'lcc_efficientnet_b0':dict(label='EfficientNet-B0',dataset='LCC-FASD'),
    'lcc_resnet18':dict(label='ResNet18',dataset='LCC-FASD'),
    'lcc_deit_tiny':dict(label='DeiT-Tiny',dataset='LCC-FASD'),
    'lcc_custom_cnn':dict(label='Custom CNN',dataset='LCC-FASD'),
}

def display_name(name):
    info=MODELS[name]; return f"{info['label']} · {info['dataset']}"

# Settings from the user's matching ResNet18 ONNX export.
# ResNet18 never reads or downloads a metadata JSON file.
BUILTIN_META = {
    'resnet18': {
        'architecture': 'resnet18',
        'threshold': 0.1945590376853943,
        'preprocessing': 'pil_bilinear_imagenet_v1',
        'sha256': 'f0968bce5cab45b568bc5655f56590efe4e9329000f23bc1ad22aa785f470fe6',
    }
}

def available(name):
    if name in BUILTIN_META:
        return bool(local_file(name,'onnx') is not None or os.getenv(name.upper()+'_ONNX_URL'))
    return all(local_file(name,ext) is not None or os.getenv(name.upper()+'_'+suffix) for ext,suffix in [('onnx','ONNX_URL'),('json','META_URL')])

def fetch(name,ext,suffix,limit,emit=lambda message:None):
    local=local_file(name,ext)
    if local is not None:
        emit(f'Found local {ext.upper()} file in {local.parent}')
        return local
    path=ROOT/f'{name}.{ext}'
    url=os.getenv(name.upper()+'_'+suffix)
    if not url or not url.startswith('https://'): raise ValueError('Provide local model files or trusted HTTPS URLs')
    ROOT.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=ROOT,suffix='.download'); os.close(fd)
    try:
        emit(f'Downloading {ext.upper()} file')
        with urllib.request.urlopen(url,timeout=120) as r,open(tmp,'wb') as f:
            total=0; last_report=0
            while chunk:=r.read(1024*1024):
                total+=len(chunk)
                if total>limit: raise ValueError('Download exceeds size limit')
                f.write(chunk)
                if total-last_report>=16*1024*1024:
                    emit(f'Downloaded {total/2**20:.0f} MiB'); last_report=total
        os.replace(tmp,path)
        emit(f'{ext.upper()} download complete')
    finally: Path(tmp).unlink(missing_ok=True)
    return path

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def load_model(name,emit=lambda message:None):
    if name not in MODELS: raise ValueError('Unknown architecture')
    if name in BUILTIN_META:
        meta = BUILTIN_META[name].copy()
        emit('Using built-in ResNet18 settings; JSON not required')
    else:
        meta=json.loads(fetch(name,'json','META_URL',1024*1024,emit).read_text())
    threshold=float(meta['threshold'])
    # CelebA-Spoof JSONs have no model_id; their architecture is the id.
    if meta.get('model_id',meta['architecture'])!=name or not 0<=threshold<=1: raise ValueError('Invalid metadata')
    emit('Metadata loaded; threshold validated')
    if meta['preprocessing']!='pil_bilinear_imagenet_v1': raise ValueError('Unsupported preprocessing')
    path=fetch(name,'onnx','ONNX_URL',600*1024*1024,emit)
    if name == 'resnet18':
        emit('Checksum verification disabled for ResNet18')
    else:
        emit('Checking model SHA256')
        if digest(path)!=meta['sha256']: raise ValueError('Model and metadata hash mismatch')
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=int(os.getenv('ORT_NUM_THREADS','2'))
    opt.inter_op_num_threads=1
    opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    emit('Initializing ONNX Runtime CPU session')
    session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
    inp=session.get_inputs()[0]
    if inp.shape!=[1,3,224,224] or inp.type!='tensor(float)': raise ValueError('Unexpected model input')
    emit('Session initialized; input shape verified')
    return session,threshold,threading.Lock()

def predict(image,bundle,emit=lambda message:None):
    session,threshold,lock=bundle
    emit('Waiting for inference slot')
    with lock:
        emit('Preprocessing image')
        start=time.perf_counter()
        x=preprocess(image)
        emit('Running ONNX inference')
        score=float(sigmoid(session.run(None,{session.get_inputs()[0].name:x})[0]).item())
        ms=(time.perf_counter()-start)*1000
    return dict(prediction='Spoof' if score>=threshold else 'Genuine',spoof_score=score,threshold=threshold,processing_ms=ms)


def public_error(exc):
    import urllib.error
    if isinstance(exc,urllib.error.HTTPError):
        return f'Model download failed (HTTP {exc.code}). Check direct download links and repository access.'
    if isinstance(exc,(TimeoutError,urllib.error.URLError)):
        return 'Model download could not connect or timed out. Check the URL and retry.'
    if isinstance(exc,PermissionError):
        return 'Model cache directory is not writable. Check MODEL_DIR permissions.'
    if isinstance(exc,json.JSONDecodeError):
        return 'Metadata is not valid JSON. Use the direct /resolve/ download link, not a web page.'
    if isinstance(exc,KeyError):
        return 'Metadata fields are missing. Use the JSON created by export_onnx.py.'
    known={
        'Model and metadata hash mismatch':'ONNX checksum mismatch. ResNet18 must match the export configured in inference.py; replace any outdated cached ONNX. ViT requires its matching export JSON.',
        'Unexpected model input':'Wrong ONNX input shape/type. Export the model using the provided script.',
        'Invalid metadata':'Architecture or threshold metadata is invalid.',
        'Unsupported preprocessing':'Unsupported preprocessing metadata. Re-export using the provided script.',
        'Provide local model files or trusted HTTPS URLs':'Missing model files or HTTPS download links.',
        'Download exceeds size limit':'Model download exceeds the configured size limit.'
    }
    return known.get(str(exc),f'{type(exc).__name__}: failed at the last recorded stage. Check model format and service memory.')
