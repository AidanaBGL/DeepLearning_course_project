import numpy as np
import torch
from torchvision import models,transforms
from PIL import Image,ImageOps
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix

def transform(train=False):
    ops=[transforms.Resize((224,224))]
    if train: ops += [transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.15,contrast=.15)]
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def model(pretrained=False,architecture='resnet18'):
    if architecture=='resnet18':
        m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc=torch.nn.Linear(m.fc.in_features,1)
    elif architecture=='vit_b_16':
        m=models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head=torch.nn.Linear(m.heads.head.in_features,1)
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    return m
class Dataset(torch.utils.data.Dataset):
    def __init__(self,df,train=False): self.df,self.tf=df.reset_index(drop=True),transform(train)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.path) as im: x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
        return x,torch.tensor(float(r.label))
def metrics(y,s,t):
    y,s=np.asarray(y),np.asarray(s)
    tn,fp,fn,tp=confusion_matrix(y,s>=t,labels=[0,1]).ravel()
    ap=fn/(tp+fn); bp=fp/(tn+fp)
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.argmin(abs(fpr-fnr))
    return dict(AUC=float(roc_auc_score(y,s)),EER_percent=float(50*(fpr[k]+fnr[k])),APCER_percent=float(100*ap),BPCER_percent=float(100*bp),ACER_percent=float(50*(ap+bp)),threshold=float(t),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])
@torch.inference_mode()
def predict(m,loader,device):
    m.eval(); ys=[]; scores=[]
    for x,y in loader:
        scores.extend(m(x.to(device)).flatten().sigmoid().cpu().tolist()); ys.extend(y.tolist())
    return ys,scores
def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=model(architecture=ck.get('architecture','resnet18')); m.load_state_dict(ck['state']); m.eval()
    return m,ck
