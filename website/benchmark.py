import argparse,json,time,threading
from pathlib import Path
import numpy as np
import psutil
from PIL import Image
from inference import load_model
from preprocess import preprocess

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['resnet18','vit_b_16'],default='resnet18')
p.add_argument('--image',required=True); p.add_argument('--runs',type=int,default=100)
p.add_argument('--out',default='benchmark.json'); a=p.parse_args()
if a.runs<1: p.error('--runs must be positive')
proc=psutil.Process(); peak=[proc.memory_info().rss]; stop=threading.Event()
def monitor():
    while not stop.wait(.01): peak[0]=max(peak[0],proc.memory_info().rss)
t=threading.Thread(target=monitor,daemon=True); t.start()
try:
    before=proc.memory_info().rss
    start=time.perf_counter(); session,_,_=load_model(a.model); load_ms=(time.perf_counter()-start)*1000
    with Image.open(a.image) as im: image=im.convert('RGB'); x=preprocess(image)
    key=session.get_inputs()[0].name
    for _ in range(10): session.run(None,{key:x})
    inference=[]; end_to_end=[]
    for _ in range(a.runs):
        start=time.perf_counter(); session.run(None,{key:x}); inference.append((time.perf_counter()-start)*1000)
        start=time.perf_counter(); data=preprocess(image); session.run(None,{key:data}); end_to_end.append((time.perf_counter()-start)*1000)
    report=dict(model=a.model,runs=a.runs,device='CPU',batch=1,model_load_ms=load_ms,process_RSS_before_MiB=before/2**20,process_RSS_after_MiB=proc.memory_info().rss/2**20,sampled_peak_process_RSS_MiB=peak[0]/2**20)
    for name,values in [('inference',inference),('preprocess_plus_inference',end_to_end)]:
        report[name+'_mean_ms']=float(np.mean(values)); report[name+'_p95_ms']=float(np.percentile(values,95))
    Path(a.out).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
finally: stop.set(); t.join()
