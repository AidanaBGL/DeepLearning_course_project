import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('results',nargs='+',help='Evaluation output folders')
p.add_argument('--out',default='runs/comparison.csv')
a=p.parse_args(); rows=[]
for folder in a.results:
    r=json.loads((Path(folder)/'metrics.json').read_text())
    rows.append({k:r[k] for k in ['architecture','AUC','EER_percent','APCER_percent','BPCER_percent','ACER_percent','MParam','threshold']})
df=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True)
df.to_csv(a.out,index=False); print(df.to_string(index=False))
