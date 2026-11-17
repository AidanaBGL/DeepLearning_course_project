import io,json,warnings
import pandas as pd
import streamlit as st
from PIL import Image,ImageOps,UnidentifiedImageError
from inference import available,load_model,predict

st.set_page_config(page_title='FaceGuard | Anti-Spoofing Lab',page_icon='◉',layout='wide')
st.markdown('''<style>
.stApp {background:#f5f7fb;color:#172b46}
.block-container {max-width:1200px;padding-top:2.5rem}
h1 {letter-spacing:-1.5px;font-weight:800!important}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px}
[data-testid="stSidebar"] {background:#edf2f8}
.eyebrow {color:#237b80;letter-spacing:3px;font-size:12px;font-weight:700}
.hero {background:linear-gradient(115deg,#102d4b,#185b68);color:white;padding:32px;border-radius:22px;margin-bottom:24px}
.hero h2 {color:white;margin:0}.hero p {color:#d8e6ee;margin-bottom:0}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">COMPUTER VISION · RESEARCH DEMO</div>',unsafe_allow_html=True)
st.title('FaceGuard — Anti-Spoofing Lab')
st.caption('Aidana · Diana · Temirlan | Genuine faces, printed photos, and screen presentations')
page=st.sidebar.radio('Explore',['Live demo','Model results','About the project'])
st.sidebar.caption('Binary classification · ONNX Runtime CPU')

@st.cache_resource(max_entries=1)
def cached_model(name): return load_model(name)

if page=='Live demo':
    st.markdown('<div class="hero"><h2>A real face. Or a presentation attack?</h2><p>Upload a photo or capture a webcam frame to test the trained model.</p></div>',unsafe_allow_html=True)
    names=[n for n in ['resnet18','vit_b_16'] if available(n)]
    if not names:
        st.warning('Model setup required: add your trained checkpoint to enable predictions.')
        st.code('models/resnet18.onnx + models/resnet18.json\nmodels/vit_b_16.onnx + models/vit_b_16.json  # optional')
        st.caption('Deployment owner: alternatively set RESNET18_ONNX_URL + RESNET18_META_URL (or VIT_B_16 equivalents) to a trusted direct HTTPS checkpoint download.')
    else:
        name=st.selectbox('Model',names,format_func=lambda x:{'resnet18':'ResNet18 · recommended baseline','vit_b_16':'ViT-B/16 · comparison model'}[x])
        left,right=st.columns([1.15,1],gap='large')
        with left:
            mode=st.radio('Input source',['Upload image','Webcam snapshot'],horizontal=True)
            f=st.file_uploader('JPG, PNG or WEBP · up to 10 MB',type=['jpg','jpeg','png','webp']) if mode=='Upload image' else st.camera_input('Capture a frame')
            image=None
            if f:
                try:
                    if f.size>10*1024*1024: raise ValueError('Please use an image smaller than 10 MB.')
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(f.getvalue())) as im:
                            if im.width*im.height>20_000_000: raise ValueError('Please resize the image to at most 20 megapixels.')
                            im.load(); image=ImageOps.exif_transpose(im).convert('RGB')
                    st.image(image,use_container_width=True)
                except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombWarning,Image.DecompressionBombError):
                    st.error('Cannot read this image. Use a valid JPG/PNG/WEBP under 10 MB and 20 megapixels.')
        with right:
            st.subheader('Detection result')
            st.caption('Frame one face prominently, then run the analysis.')
            if st.button('Analyze image',type='primary',disabled=image is None,use_container_width=True):
                try:
                    with st.spinner('Loading model and analyzing…'):
                        result=predict(image,cached_model(name))
                    if result['prediction']=='Spoof': st.error('SPOOF — presentation attack predicted')
                    else: st.success('GENUINE — live face predicted')
                    st.metric('Spoof score',f"{result['spoof_score']:.4f}")
                    st.progress(min(1.,max(0.,result['spoof_score'])))
                    a,b=st.columns(2)
                    a.metric('Decision threshold',f"{result['threshold']:.4f}")
                    b.metric('Processing time',f"{result['processing_ms']:.0f} ms")
                    st.caption('Spoof is predicted when score ≥ threshold. A score is not a calibrated probability. Timing excludes model loading and network transfer.')
                    st.download_button('Download result JSON',json.dumps({'model':name,**result},indent=2),file_name='prediction.json',mime='application/json')
                except Exception:
                    st.error('Model could not be loaded or evaluated. Check that the configured file is a matching ONNX model and metadata JSON and that the service has enough memory.')
            else:
                st.info('Choose an image and click Analyze image. Predictions are never simulated.')
        st.warning('Research prototype: processes the full frame without face detection. Images without a face can still receive a prediction. Webcam mode captures individual frames, not continuous video.')
        st.caption('Use images with consent. Uploaded images are processed in server memory; this application does not save them to disk.')
elif page=='Model results':
    st.subheader('Measured performance on the held-out subset')
    st.caption('5,200 images · 46 identities · thresholds selected on validation')
    a,b,c=st.columns(3); a.metric('ResNet18 ACER','1.59%'); b.metric('ViT-B/16 ACER','3.24%'); c.metric('Test images','5,200')
    table=pd.DataFrame([
        ['ResNet18',.998153,1.44,.94,2.24,1.59,11.18],
        ['ViT-B/16',.994211,3.66,4.14,2.33,3.24,85.80]
    ],columns=['Model','AUC','EER %','APCER %','BPCER %','ACER %','M parameters'])
    st.dataframe(table,hide_index=True,use_container_width=True)
    st.bar_chart(table.set_index('Model')[['APCER %','BPCER %','ACER %']])
    st.markdown('**ResNet18:** 39 accepted attacks and 24 rejected genuine samples. **ViT:** 171 accepted attacks and 25 rejected genuine samples.')
    st.info('Custom identity-separated split of a partially recovered CelebA-Spoof subset. These are not official benchmark results or measured live-camera performance.')
else:
    st.subheader('From dataset to live demonstration')
    st.write('We compare ResNet18 and ViT-B/16 for binary face presentation attack detection. The selected checkpoints were chosen at epoch 4 using validation ACER.')
    st.dataframe(pd.DataFrame({'Split':['Training','Validation','Test'],'Images':[20198,3903,5200],'People':[210,45,46]}),hide_index=True,use_container_width=True)
    st.markdown('### Demo checklist\n1. Capture a real face.\n2. Show a printed photograph to the webcam.\n3. Display a photograph on a phone/tablet.\n4. Try a prerecorded face video on the screen.\n5. Compare scores under different lighting and angles.')
    st.write('The current models do not predict print/display/mask subtypes. Face cropping and multiclass training are future experiments.')
    st.link_button('CelebA-Spoof dataset','https://github.com/ZhangYuanhan-AI/CelebA-Spoof')
st.divider()
st.caption('FaceGuard · Course project · Experimental results, not identity verification or certification')
