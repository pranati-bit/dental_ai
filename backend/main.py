from __future__ import annotations
import base64, io, json, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CLASSES = ['no_need', 'borderline', 'braces_likely']
app = FastAPI(title='Dental AI Lab API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/assets', StaticFiles(directory=ROOT / 'results'), name='assets')
app.mount('/patient-images', StaticFiles(directory=ROOT / 'data' / 'pics'), name='patient-images')


def load_braces_model():
    html = (ROOT / 'web' / 'demo.html').read_text(encoding='utf-8')
    match = re.search(r'const M = (\{.*?\});\s*\n', html, re.S)
    if not match:
        raise RuntimeError('Could not locate the original exported braces model in web/demo.html')
    return json.loads(match.group(1))

BRACES = load_braces_model()
CROWD = np.load(RESULTS / 'crowding_model.npz')
BOX = np.load(RESULTS / 'box_model.npz')
PCA_MEAN = np.frombuffer(base64.b64decode(BRACES['pca_mean']), dtype='<f4').copy()
PCA_COMP = np.frombuffer(base64.b64decode(BRACES['pca_components']), dtype='<f4').copy().reshape(BRACES['k'], BRACES['D'])
BRACES_MU = np.frombuffer(base64.b64decode(BRACES['mu']), dtype='<f4').copy()
BRACES_LAYERS = []
for layer in BRACES['layers']:
    w = np.frombuffer(base64.b64decode(layer['W']), dtype='<f4').copy().reshape(layer['inp'], layer['out'])
    b = np.frombuffer(base64.b64decode(layer['b']), dtype='<f4').copy()
    BRACES_LAYERS.append((w, b))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def box_for_image(im: Image.Image):
    small = np.asarray(im.convert('RGB').resize(tuple(int(x) for x in BOX['size']), Image.Resampling.BICUBIC), dtype=np.float32).reshape(-1) / 255.0
    pred = (small - BOX['mu']) @ BOX['W'] + BOX['b']
    pred = np.clip(pred, 0, 1)
    pad = float(BOX['pad'])
    l, t, r, b = pred
    return [max(0., float(l)-pad), max(0., float(t)-pad), min(1., float(r)+pad), min(1., float(b)+pad)]


def preprocess(im: Image.Image):
    rgb = im.convert('RGB')
    im = im.convert('L')
    l, t, r, b = box_for_image(rgb)
    w, h = im.size
    crop = im.crop((int(l*w), int(t*h), int(r*w), int(b*h)))
    crop = crop.resize((BRACES['W'], BRACES['H']), Image.Resampling.BICUBIC)
    crop = ImageOps.equalize(crop)
    arr = np.asarray(crop, dtype=np.float32).reshape(-1) / 255.0
    z = ((arr - PCA_MEAN) @ PCA_COMP.T - BRACES_MU) / float(BRACES['sd'])
    return arr.reshape(BRACES['H'], BRACES['W']), z, [l,t,r,b]


def dense_predict(z, layers):
    a = z
    for i, (w, b) in enumerate(layers):
        a = a @ w + b
        if i < len(layers)-1:
            a = np.maximum(a, 0)
    return softmax(a)


def crowd_predict(z):
    a = (z - CROWD['mu']) / float(CROWD['sd'])
    for i in range(3):
        a = a @ CROWD[f'W{i}'] + CROWD[f'b{i}'].reshape(-1)
        if i < 2:
            a = np.maximum(a, 0)
    return softmax(a)


def crowd_probability(img_arr):
    flat = img_arr.reshape(-1)
    z = ((flat - PCA_MEAN) @ PCA_COMP.T)
    return float(crowd_predict(z)[1])


def make_occlusion(img_arr, patch=(12, 8), stride=(6, 4)):
    h, w = img_arr.shape
    base = crowd_probability(img_arr)
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    ph, pw = patch[1], patch[0]
    for y in range(0, max(1, h-ph+1), stride[1]):
        for x in range(0, max(1, w-pw+1), stride[0]):
            altered = img_arr.copy()
            altered[y:y+ph, x:x+pw] = float(img_arr.mean())
            delta = max(0.0, base - crowd_probability(altered))
            heat[y:y+ph, x:x+pw] += delta
            counts[y:y+ph, x:x+pw] += 1
    heat /= np.maximum(counts, 1)
    if heat.max() > 0:
        heat /= heat.max()
    return heat, base


def heat_layer_data_url(heat):
    stops = np.array([[20,10,70],[90,25,130],[190,45,110],[245,125,55],[255,220,100]], dtype=np.float32)
    q = np.clip(heat, 0, 1) * (len(stops)-1)
    lo = np.floor(q).astype(int); hi = np.minimum(lo+1, len(stops)-1); f=(q-lo)[...,None]
    rgb = (stops[lo]*(1-f)+stops[hi]*f).astype(np.uint8)
    rgba = np.dstack([rgb, (heat*255).astype(np.uint8)])
    buf=io.BytesIO(); Image.fromarray(rgba, 'RGBA').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def overlay_data_url(img_arr, heat, opacity=0.62):
    gray = np.uint8(np.clip(img_arr, 0, 1) * 255)
    base = Image.fromarray(gray).convert('RGBA')
    # Magma-like false color generated from a compact RGB ramp.
    stops = np.array([[20,10,70],[90,25,130],[190,45,110],[245,125,55],[255,220,100]], dtype=np.float32)
    q = np.clip(heat, 0, 1) * (len(stops)-1)
    lo = np.floor(q).astype(int); hi = np.minimum(lo+1, len(stops)-1); f=(q-lo)[...,None]
    rgb = (stops[lo]*(1-f)+stops[hi]*f).astype(np.uint8)
    rgba = np.dstack([rgb, (heat*255*opacity).astype(np.uint8)])
    overlay = Image.alpha_composite(base, Image.fromarray(rgba, 'RGBA'))
    buf=io.BytesIO(); overlay.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

@app.get('/api/health')
def health():
    return {'status':'ok','models':['braces_3class','crowding_binary'],'xai':'occlusion_sensitivity'}

@app.get('/api/metrics')
def metrics():
    crowd = json.loads((RESULTS/'crowding_metrics.json').read_text())
    return {'braces':BRACES['eval'], 'crowding':crowd}

@app.post('/api/infer')
async def infer(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415, 'Upload a JPG, PNG or WebP dental image.')
    raw = await file.read()
    if len(raw) > 12*1024*1024:
        raise HTTPException(413, 'Image must be 12 MB or smaller.')
    try:
        im = Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception:
        raise HTTPException(400, 'The uploaded file is not a readable image.')
    arr, z, box = preprocess(im)
    braces_p = dense_predict(z, BRACES_LAYERS)
    braces_i = int(np.argmax(braces_p))
    heat, crowd_p = make_occlusion(arr)
    crowd_pred = 'crowding' if crowd_p >= 0.5 else 'no_crowding'
    return {
      'filename':file.filename,
      'braces':{'class':CLASSES[braces_i], 'probabilities':{c:round(float(p),4) for c,p in zip(CLASSES,braces_p)}, 'confidence':round(float(braces_p[braces_i]),4)},
      'crowding':{'class':crowd_pred,'probabilities':{'no_crowding':round(1-crowd_p,4),'crowding':round(crowd_p,4)},'confidence':round(max(crowd_p,1-crowd_p),4)},
      'heatmap_overlay':heat_layer_data_url(heat),
      'original_preview':overlay_data_url(arr,np.zeros_like(heat),0),
      'teeth_box':[round(float(x),4) for x in box],
      'xai_method':'Occlusion sensitivity: each patch is neutralized and the drop in crowding probability is mapped. This explains model sensitivity, not clinical evidence.',
      'warning':'Research prototype only. Model confidence is not calibrated clinical risk. Do not use this result to make treatment decisions.'
    }

@app.get('/api/gallery')
def gallery():
    exp=RESULTS/'explain'
    idx=pd.read_csv(RESULTS/'index.csv', dtype={'patient_id':str})
    scored=pd.read_csv(exp/'explained_patients.csv', dtype={'patient':str})
    items=[]
    for p in sorted(exp.glob('*_crowding.png')):
        # The archived XAI figure and source photo are joined by the original patient key.
        patient=p.name.split('_')[0]
        row=idx[idx.patient_id==patient]
        if row.empty:
            continue
        source=ROOT / row.iloc[0]['path']
        if not source.exists():
            continue
        score=scored[scored.patient==patient]
        prob=float(score.iloc[0].p_crowding) if not score.empty else None
        items.append({'id':p.stem,'image':'/assets/explain/'+p.name,
                      'photo':'/patient-images/'+str(row.iloc[0]['path']).replace('data/pics/',''),
                      'prediction':('crowding' if prob>=0.5 else 'no crowding') if prob is not None else None,
                      'confidence':round(prob if prob>=0.5 else 1-prob,3) if prob is not None else None,
                      'label':'Existing held-out explanation figure'})
    return {'items':items}
