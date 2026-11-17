import os,time,threading,urllib.request,tempfile
from pathlib import Path
import torch
from torchvision import models,transforms
from PIL import ImageOps

torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS','2')))
ROOT=Path(os.getenv('MODEL_DIR','models'))

def available(name):
    return (ROOT/f'{name}.pt').is_file() or bool(os.getenv(name.upper()+'_URL'))

def load_model(name):
    if name not in ['resnet18','vit_b_16']: raise ValueError('Unsupported architecture')
    path=ROOT/f'{name}.pt'
    if not path.is_file():
        url=os.getenv(name.upper()+'_URL')
        if not url: raise FileNotFoundError(f'Add {path} or configure {name.upper()}_URL.')
        if not url.startswith('https://'): raise ValueError('Checkpoint URL must use HTTPS')
        ROOT.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=ROOT,suffix='.download'); os.close(fd)
        try:
            with urllib.request.urlopen(url,timeout=120) as response,open(tmp,'wb') as out:
                total=0
                while chunk:=response.read(1024*1024):
                    total+=len(chunk)
                    if total>600*1024*1024: raise ValueError('Checkpoint exceeds 600 MiB')
                    out.write(chunk)
            os.replace(tmp,path)
        finally:
            Path(tmp).unlink(missing_ok=True)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    architecture=ck.get('architecture','resnet18')
    if architecture!=name: raise ValueError('Checkpoint architecture does not match selected model')
    threshold=float(ck['threshold'])
    if not 0<=threshold<=1: raise ValueError('Invalid checkpoint threshold')
    if name=='resnet18':
        net=models.resnet18(weights=None); net.fc=torch.nn.Linear(net.fc.in_features,1)
    else:
        net=models.vit_b_16(weights=None); net.heads.head=torch.nn.Linear(net.heads.head.in_features,1)
    net.load_state_dict(ck['state']); net.eval()
    return net,threshold,threading.Lock()

TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def predict(image,bundle):
    net,threshold,lock=bundle
    with lock,torch.inference_mode():
        start=time.perf_counter()
        x=TF(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
        score=net(x).sigmoid().item()
        ms=(time.perf_counter()-start)*1000
    return dict(prediction='Spoof' if score>=threshold else 'Genuine',spoof_score=score,threshold=threshold,processing_ms=ms)
