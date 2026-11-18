import io,json,os,time,uuid,hashlib,warnings,logging,html
from urllib.parse import urlparse
import streamlit as st
from PIL import Image,ImageOps
from inference import MODELS,available,display_name,load_model,predict

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
log=logging.getLogger('face_antispoof')
st.set_page_config(page_title='FACE ANTISPOOF',page_icon='◌',layout='wide')
st.markdown('''<style>
.stApp {background:#f8f4ed;color:#302d29}
.block-container {max-width:1200px;padding:1.7rem 2rem 3rem}
header[data-testid="stHeader"] {background:transparent}
.topbar {display:flex;align-items:center;justify-content:space-between;gap:24px;padding:14px 0 25px;border-bottom:1px solid #ded7cc}
.brand {font-size:22px;font-weight:800;letter-spacing:-.8px;white-space:nowrap}
.nav {display:flex;gap:24px;flex-wrap:wrap}.nav a {color:#403d37;text-decoration:none;font-size:14px}.nav a:hover {color:#ce642b}.nav span {color:#a29a8e;font-size:14px}
.hero {padding:30px 0 20px;max-width:100%}.eyebrow {font-size:12px;letter-spacing:2px;color:#9b795a;font-weight:600}
.hero h1 {font-size:clamp(28px,3vw,40px);line-height:1.15;letter-spacing:-1.3px;font-weight:600;margin:8px 0 10px;color:#302d29}
.hero p {font-size:18px;color:#82786d;max-width:900px;line-height:1.7}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:20px!important}
[data-testid="stFileUploader"] {background:#fffdf8;border:1px dashed #d8cbbb;border-radius:15px;padding:12px}
.stButton button[kind="primary"] {background:#df7138;border:0;border-radius:12px;min-height:52px;font-size:17px;font-weight:600;color:white}
.stButton button[kind="primary"]:hover {background:#c95d27;color:white}
[data-testid="stMetric"] {background:#fffdf8;border:1px solid #e6ded2;border-radius:14px;padding:16px}
.empty {min-height:250px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;color:#9e9487;padding:28px}
.empty b {font-size:21px;color:#635b51;margin:12px}.symbol {font-size:50px;color:#c9bba9}
.result {padding:22px;border-radius:16px;background:#eaf0e5;border:1px solid #cedbc2;margin:15px 0}.result.spoof {background:#fae9db;border-color:#ebc6a8}.result h2 {margin:5px 0;color:#302d29}.result small {letter-spacing:2px;color:#84715c}
.foot {padding-top:25px;margin-top:30px;border-top:1px solid #ded7cc;color:#9d9284;font-size:12px}
@media(max-width:700px){.topbar{align-items:flex-start;flex-direction:column}.hero{padding:24px 0}.block-container{padding:1rem}.nav{gap:16px}}

.scorebox {background:#f6eee3;border-radius:12px;text-align:center;padding:23px 12px;margin:12px 0 18px}.scorevalue {font-size:58px;font-weight:750;line-height:1.1;color:#bd5525;letter-spacing:-2px}.scorebox p {font-size:12px;color:#8b7b67;margin:12px 0 0}
.samplecard {border:1px solid #e4dbcf;background:#fffdf8;border-radius:12px;padding:16px 18px;min-height:85px}.samplecard strong{display:block;color:#403a32;font-size:15px}.samplecard span{color:#8b8072;font-size:13px}.sectiontitle {font-size:18px;font-weight:650;margin-bottom:5px}
a.resource {display:block;border:1px solid #e4dbcf;background:#fffdf8;border-radius:12px;padding:20px;color:#403a32;text-decoration:none;min-height:130px}a.resource strong{font-size:18px}a.resource p{font-size:14px;color:#8b8072;margin:7px 0}a.resource span{font-size:13px;color:#bd5525}
</style>''',unsafe_allow_html=True)
links=[]
for label,key in [('GitHub','GITHUB_URL'),('Hugging Face','HUGGINGFACE_URL'),('Presentation','PRESENTATION_URL'),('Paper','PAPER_URL')]:
    url=os.getenv(key,'').strip()
    parsed=urlparse(url)
    if parsed.scheme=='https' and parsed.netloc:
        links.append(f'<a href="{html.escape(url,quote=True)}" target="_blank" rel="noopener noreferrer">{label} ↗</a>')
    else: links.append(f'<span title="Link coming soon">{label}</span>')
st.markdown('<div class="topbar"><div class="brand">FACE ANTISPOOF<span style="color:#df7138">.</span></div><nav class="nav">' + ''.join(links)+'</nav></div>',unsafe_allow_html=True)
st.markdown('''<section class="hero"><div class="eyebrow">FACE PRESENTATION ATTACK DETECTION</div><h1>Test a face. Inspect the result.</h1><p>Upload a sample or capture a webcam frame to detect presentation attacks.</p></section>''',unsafe_allow_html=True)

@st.cache_resource(max_entries=1)
def cached_model(name,_emit): return load_model(name,emit=_emit)

for key,default in [('last_result',None),('events',[]),('error',None),('input_key',None)]:
    if key not in st.session_state: st.session_state[key]=default

left,right=st.columns([1.65,1],gap='medium')
with left,st.container(border=True):
    st.subheader('Input sample')
    # List only configured models; if none are configured, list all so the setup warning is visible.
    choices=[n for n in MODELS if available(n)] or list(MODELS)
    name=st.selectbox('Model',choices,format_func=lambda n:display_name(n)+('' if available(n) else ' · not configured'))
    mode=st.radio('Input',['Upload image','Webcam snapshot'],horizontal=True,label_visibility='collapsed')
    f=st.file_uploader('Upload image',type=['jpg','jpeg','png','webp'],help='Maximum 10 MB and 20 megapixels.') if mode=='Upload image' else st.camera_input('Take a snapshot')
    image=None; raw=f.getvalue() if f else b''
    current=(name,mode,hashlib.sha256(raw).hexdigest())
    if st.session_state.input_key!=current:
        st.session_state.last_result=None; st.session_state.events=[]; st.session_state.error=None; st.session_state.input_key=current
    if f:
        try:
            if len(raw)>10*1024*1024: raise ValueError('Image exceeds 10 MB')
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width*im.height>20_000_000: raise ValueError('Image exceeds 20 megapixels')
                    im.load(); image=ImageOps.exif_transpose(im).convert('RGB')
            preview=image.copy(); preview.thumbnail((820,390)); st.image(preview,use_container_width=True)
            st.caption(f'Ready · {image.width} × {image.height} px')
        except Exception as exc:
            st.error('Cannot read this image. Choose a valid JPG, PNG or WEBP under 10 MB and 20 megapixels.')
            log.warning('image_decode_failed type=%s',type(exc).__name__)
    if not available(name): st.warning('This model is not configured. Open Setup details below.')
    send=st.button('Отправить →',type='primary',use_container_width=True,disabled=image is None or not available(name))
    if image is None: st.caption('Upload an image or take a snapshot to enable the button.')
with right,st.container(border=True):
    st.subheader('Detection result')
    result_slot=st.empty()
    status_slot=st.empty()

if send:
    request_id=uuid.uuid4().hex[:10]; started=time.perf_counter()
    st.session_state.events=[]; st.session_state.error=None; st.session_state.last_result=None
    with status_slot.container():
        with st.status('Processing your image…',expanded=True) as status:
            def emit(message):
                elapsed=time.perf_counter()-started
                line=f'{elapsed:6.1f}s  {message}'
                st.session_state.events.append(line); st.write(line)
                log.info('request=%s model=%s stage=%s',request_id,name,message)
            try:
                emit('Request accepted')
                emit('Loading model or retrieving cached session')
                bundle=cached_model(name,emit)
                emit('Model ready')
                result=predict(image,bundle,emit=emit)
                result.update(model=name,request_id=request_id,total_ms=(time.perf_counter()-started)*1000)
                st.session_state.last_result=result
                emit('Completed')
                status.update(label='Analysis complete',state='complete',expanded=False)
            except Exception as exc:
                from inference import public_error
                reason=public_error(exc)
                st.session_state.error=f'{reason} Reference: {request_id}'
                emit('FAILED: '+reason)
                log.error('request=%s model=%s failure_type=%s',request_id,name,type(exc).__name__)
                status.update(label='Analysis failed — see processing log',state='error',expanded=True)

with result_slot.container():
    result=st.session_state.last_result
    if result:
        spoof=result['prediction']=='Spoof'; score=result['spoof_score']
        st.markdown(f'<div class="result {"spoof" if spoof else ""}"><small>MODEL PREDICTION</small><h2>{"Spoof detected" if spoof else "Genuine face"}</h2><div>{"Possible presentation attack" if spoof else "Classified as a genuine presentation"}</div></div>',unsafe_allow_html=True)
        # Show the actual output score, not an invented confidence percentage.
        st.markdown(f'<div class="scorebox"><div class="scorevalue">{score*100:.2f}%</div><strong>Spoof score</strong><p>A model score, not a guarantee.</p></div>',unsafe_allow_html=True)
        st.caption('Model score, not a calibrated confidence probability. Prediction uses the saved threshold, which may differ from 50%.')
        a,b=st.columns(2)
        a.metric('Decision threshold',f"{result['threshold']*100:.2f}%")
        b.metric('Processing',f"{result['processing_ms']:.0f} ms")
        st.caption(f"Total request: {result['total_ms']/1000:.2f} s · {display_name(result['model'])} · ID {result['request_id']}")
        st.download_button('Download result',json.dumps(result,indent=2),'prediction.json','application/json')
    elif st.session_state.error: st.error(st.session_state.error)
    else: st.markdown('<div class="empty"><div class="symbol">◌</div><b>Your result appears here</b><span>Add an image and press Отправить.<br>Each processing step will be shown below.</span></div>',unsafe_allow_html=True)

with st.expander('Processing log',expanded=bool(st.session_state.error)):
    st.code('\n'.join(st.session_state.events) or 'No request yet. Choose an image and press Отправить.',language='text')
with st.expander('Setup details'):
    st.write({n:('Configured — download/inference not yet verified' if available(n) else 'Not configured') for n in MODELS})
    st.caption('ResNet18 (CelebA-Spoof) settings are built into the app: only its ONNX file or URL is required. Other models need matching local .onnx + .json files or model URL variables. First inference initializes ONNX Runtime; subsequent requests reuse the session.')
    st.code('\n'.join(f'{n.upper()}_{s}' for n in MODELS for s in (['ONNX_URL'] if n=='resnet18' else ['ONNX_URL','META_URL'])))
    st.caption('Navigation links: GITHUB_URL, HUGGINGFACE_URL, PRESENTATION_URL, PAPER_URL. Missing links stay gray until configured.')
with st.expander('Research results & limitations'):
    st.markdown('''| Model | AUC | APCER | BPCER | ACER |
|---|---:|---:|---:|---:|
| ResNet18 | 0.99815 | 0.94% | 2.24% | 1.59% |
| ViT-B/16 | 0.99421 | 4.14% | 2.33% | 3.24% |

Previously measured PyTorch results on a custom 5,200-image CelebA-Spoof test subset. Not official benchmark or live-camera results.

| Model (LCC-FASD) | AUC | APCER | BPCER | ACER |
|---|---:|---:|---:|---:|
| MobileNetV3-Large | 0.9932 | 2.39% | 8.92% | 5.66% |
| EfficientNet-B0 | 0.9817 | 3.36% | 10.51% | 6.93% |
| ResNet18 | 0.9661 | 2.08% | 14.97% | 8.52% |
| DeiT-Tiny | 0.9694 | 4.69% | 15.92% | 10.31% |
| Custom CNN | 0.9005 | 18.58% | 13.69% | 16.14% |

PyTorch results on the official LCC-FASD evaluation split (7,580 images: 314 real, 7,266 spoof); threshold chosen on the development split. Models trained on different datasets are not directly comparable.

Full-frame binary classification; no face detector or attack subtype prediction. Non-face images can receive predictions. Webcam mode takes snapshots, not continuous video. Scores are not calibrated probabilities. Uploads are processed in server memory and not saved by the application.''')
st.markdown('<div class="foot">Aidana · Diana · Temirlan &nbsp; / &nbsp; Research prototype · ONNX Runtime CPU</div>',unsafe_allow_html=True)
