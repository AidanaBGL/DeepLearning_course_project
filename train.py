import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm
from core import model,Dataset,predict,metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifests',default='data/manifests'); p.add_argument('--out',default='runs/baseline')
    p.add_argument('--loss',choices=['bce','focal'],default='bce'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--gamma',type=float,default=2); p.add_argument('--alpha',type=float,default=.5); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',choices=['resnet18','vit_b_16'],default='resnet18')
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    tr=DataLoader(Dataset(pd.read_csv(Path(a.manifests)/'train.csv'),True),batch_size=a.batch_size,shuffle=True)
    va=DataLoader(Dataset(pd.read_csv(Path(a.manifests)/'val.csv')),batch_size=a.batch_size)
    m=model(True,a.model).to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr); best=float('inf'); history=[]
    for epoch in range(a.epochs):
        m.train(); total=0
        for x,y in tqdm(tr,desc=f'Epoch {epoch+1}'):
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=m(x).flatten()
            loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
            if a.loss=='focal': loss=(y*a.alpha+(1-y)*(1-a.alpha))*(1-torch.exp(-loss))**a.gamma*loss
            loss=loss.mean(); loss.backward(); opt.step(); total+=loss.item()*len(y)
        y,s=predict(m,va,device); fpr,tpr,ts=roc_curve(y,s); ok=np.isfinite(ts)
        t=float(ts[ok][np.argmin((fpr+1-tpr)[ok])]); stats=metrics(y,s,t); stats.update(epoch=epoch+1,train_loss=total/len(tr.dataset))
        history.append(stats); print(stats); (out/'history.json').write_text(json.dumps(history,indent=2))
        if stats['ACER_percent']<best:
            best=stats['ACER_percent']; torch.save(dict(state=m.state_dict(),architecture=a.model,threshold=t,label_mapping={'genuine':0,'spoof':1}),out/'best.pt')
if __name__=='__main__': main()
