import argparse,json,hashlib,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch,onnx,onnxruntime as ort
from torchvision import models,transforms
from PIL import Image,ImageOps
from preprocess import preprocess,sigmoid

def build(architecture):
    if architecture=='resnet18':
        m=models.resnet18(weights=None); m.fc=torch.nn.Linear(m.fc.in_features,1)
    elif architecture=='vit_b_16':
        m=models.vit_b_16(weights=None); m.heads.head=torch.nn.Linear(m.heads.head.in_features,1)
    else: raise ValueError('Unsupported architecture')
    return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--csv',required=True,help='Validation CSV with path column; use the same images as training')
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--out',default='models')
    p.add_argument('--atol',type=float,default=1e-4)
    a=p.parse_args()
    if a.samples<1: p.error('--samples must be positive')
    torch.set_num_threads(2); torch.manual_seed(42)
    # Disables the fused MHA fast path which can obstruct legacy ViT ONNX export.
    torch.backends.mha.set_fastpath_enabled(False)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    name=ck.get('architecture','resnet18'); threshold=float(ck['threshold'])
    if not 0<=threshold<=1: raise ValueError('Invalid threshold')
    m=build(name); m.load_state_dict(ck['state']); m.eval()
    df=pd.read_csv(a.csv)
    if df.empty: raise ValueError('Empty validation CSV')
    paths=df.sample(n=min(a.samples,len(df)),random_state=42).path.tolist()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        path=Path(tmp)/f'{name}.onnx'
        with torch.no_grad():
            torch.onnx.export(m,torch.zeros(1,3,224,224),str(path),input_names=['images'],output_names=['logits'],opset_version=17,dynamo=False,do_constant_folding=True)
        onnx.checker.check_model(str(path))
        options=ort.SessionOptions(); options.intra_op_num_threads=2
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        max_error=0.; flips=0; reference=[]; actual=[]
        for file in paths:
            with Image.open(file) as im:
                image=ImageOps.exif_transpose(im).convert('RGB')
                original=tf(image).unsqueeze(0)
                x=preprocess(image)
            np.testing.assert_allclose(original.numpy(),x,atol=1e-6,rtol=1e-6)
            with torch.inference_mode(): ref=m(original).sigmoid().item()
            score=float(sigmoid(session.run(None,{'images':x})[0]).item())
            max_error=max(max_error,abs(ref-score)); flips+=int((ref>=threshold)!=(score>=threshold))
            reference.append(ref); actual.append(score)
        np.testing.assert_allclose(reference,actual,atol=a.atol,rtol=0)
        if flips: raise RuntimeError(f'{flips} predictions changed at saved threshold; inspect before deployment')
        h=hashlib.sha256()
        with open(path,'rb') as f:
            while chunk:=f.read(1024*1024): h.update(chunk)
        meta=dict(architecture=name,threshold=threshold,preprocessing='pil_bilinear_imagenet_v1',sha256=h.hexdigest(),input_shape=[1,3,224,224],output='logits',precision='float32',validation_samples=len(paths),max_score_error=max_error,prediction_changes=flips,torch_version=str(torch.__version__),ort_version=ort.__version__)
        path.replace(out/f'{name}.onnx')
        (out/f'{name}.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2)); print('Export and validation passed:',out/name)
if __name__=='__main__': main()
