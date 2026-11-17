import time
import streamlit as st
import torch
from PIL import Image,ImageOps
from core import load,transform
st.title('Face Anti-Spoofing Detection')
st.caption('Course prototype. Scores are not guarantees of authenticity.')
path=st.sidebar.text_input('Checkpoint','runs/baseline/best.pt')
@st.cache_resource
def get_model(path): return load(path)
mode=st.radio('Input',['Upload','Webcam'])
f=st.file_uploader('Image',type=['jpg','jpeg','png']) if mode=='Upload' else st.camera_input('Capture a frame')
st.info('Frame one face prominently. The baseline processes the complete image; automatic face detection is not implemented. Webcam input is a snapshot.')
if f:
    im=ImageOps.exif_transpose(Image.open(f)).convert('RGB'); st.image(im,width=360)
    if st.button('Analyze'):
        try:
            m,ck=get_model(path); st.caption('Model: '+ck.get('architecture','resnet18')); start=time.perf_counter()
            with torch.inference_mode(): score=m(transform()(im).unsqueeze(0)).sigmoid().item()
            elapsed=(time.perf_counter()-start)*1000
            st.subheader('Spoof' if score>=ck['threshold'] else 'Genuine')
            st.metric('Spoof score',f'{score:.4f}'); st.write(f'Threshold: {ck["threshold"]:.4f} | Processing: {elapsed:.1f} ms')
        except Exception as e: st.error(str(e))
