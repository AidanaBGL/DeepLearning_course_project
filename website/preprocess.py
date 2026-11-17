import numpy as np
from PIL import Image,ImageOps

def preprocess(image):
    image=ImageOps.exif_transpose(image).convert('RGB')
    image=image.resize((224,224),Image.Resampling.BILINEAR)
    x=np.asarray(image,dtype=np.float32)/np.float32(255)
    x=(x-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2,0,1)[None])

def sigmoid(logits):
    x=np.asarray(logits,dtype=np.float64)
    return np.exp(-np.logaddexp(0,-x))
