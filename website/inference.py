import os,time,json,threading,urllib.request,tempfile,hashlib
from pathlib import Path
import onnxruntime as ort
from preprocess import preprocess,sigmoid

ROOT=Path(os.getenv('MODEL_DIR','models'))

def available(name):
    return all((ROOT/f'{name}.{ext}').is_file() or os.getenv(name.upper()+'_'+suffix) for ext,suffix in [('onnx','ONNX_URL'),('json','META_URL')])

def fetch(name,ext,suffix,limit):
    path=ROOT/f'{name}.{ext}'
    if path.is_file(): return path
    url=os.getenv(name.upper()+'_'+suffix)
    if not url or not url.startswith('https://'): raise ValueError('Provide local model files or trusted HTTPS URLs')
    ROOT.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=ROOT,suffix='.download'); os.close(fd)
    try:
        with urllib.request.urlopen(url,timeout=120) as r,open(tmp,'wb') as f:
            total=0
            while chunk:=r.read(1024*1024):
                total+=len(chunk)
                if total>limit: raise ValueError('Download exceeds size limit')
                f.write(chunk)
        os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)
    return path

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def load_model(name):
    if name not in ['resnet18','vit_b_16']: raise ValueError('Unknown architecture')
    meta=json.loads(fetch(name,'json','META_URL',1024*1024).read_text())
    threshold=float(meta['threshold'])
    if meta['architecture']!=name or not 0<=threshold<=1: raise ValueError('Invalid metadata')
    if meta['preprocessing']!='pil_bilinear_imagenet_v1': raise ValueError('Unsupported preprocessing')
    path=fetch(name,'onnx','ONNX_URL',600*1024*1024)
    if digest(path)!=meta['sha256']: raise ValueError('Model and metadata hash mismatch')
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=int(os.getenv('ORT_NUM_THREADS','2'))
    opt.inter_op_num_threads=1
    opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
    inp=session.get_inputs()[0]
    if inp.shape!=[1,3,224,224] or inp.type!='tensor(float)': raise ValueError('Unexpected model input')
    return session,threshold,threading.Lock()

def predict(image,bundle):
    session,threshold,lock=bundle
    with lock:
        start=time.perf_counter()
        x=preprocess(image)
        score=float(sigmoid(session.run(None,{session.get_inputs()[0].name:x})[0]).item())
        ms=(time.perf_counter()-start)*1000
    return dict(prediction='Spoof' if score>=threshold else 'Genuine',spoof_score=score,threshold=threshold,processing_ms=ms)
