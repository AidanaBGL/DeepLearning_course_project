import argparse,json
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from core import load,Dataset,predict,metrics
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--csv',default='data/manifests/test.csv'); p.add_argument('--out',default='runs/evaluation')
a=p.parse_args(); m,ck=load(a.checkpoint); df=pd.read_csv(a.csv)
y,s=predict(m,DataLoader(Dataset(df),batch_size=32),'cpu')
r=metrics(y,s,ck['threshold']); r['architecture']=ck.get('architecture','resnet18'); r['MParam']=sum(p.numel() for p in m.parameters())/1e6; df['spoof_score']=s; df['prediction']=(df.spoof_score>=ck['threshold']).astype(int)
r['attack_APCER_percent']={str(k):float(100*(g.prediction==0).mean()) for k,g in df[df.label==1].groupby('attack')}
out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'predictions.csv',index=False); (out/'metrics.json').write_text(json.dumps(r,indent=2)); print(r)
