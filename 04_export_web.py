"""
STEP 4: Export the trained model into a single web page (web/demo.html).

Needs results/pca_params.npz, results/model_pca.npz, results/box_model.npz (made by 02_pca.py, 03_ann.py, 02a_teeth_box.py),
and results/features.npz. Nothing is uploaded anywhere: the page runs the model in the browser.
Run:  python 04_export_web.py
"""
import base64
import json
from pathlib import Path

import numpy as np

from config import CLASSES, IMG_H, IMG_W, OUT

WEB = Path("web")
WEB.mkdir(exist_ok=True)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


pca = np.load(OUT / "pca_params.npz")
box = np.load(OUT / "box_model.npz")          # learned teeth box (from 02a_teeth_box.py)
net = np.load(OUT / "model_pca.npz")
feats = np.load(OUT / "features.npz", allow_pickle=True)

n_layers = sum(1 for k in net.files if k.startswith("W"))
layers = []
for i in range(n_layers):
    W, bias = net[f"W{i}"], net[f"b{i}"].reshape(-1)
    layers.append(dict(inp=int(W.shape[0]), out=int(W.shape[1]), W=b64(W), b=b64(bias)))


def forward(Z):
    a = Z
    for i in range(n_layers):
        z = a @ net[f"W{i}"] + net[f"b{i}"]
        if i == n_layers - 1:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            a = e / e.sum(axis=1, keepdims=True)
        else:
            a = np.maximum(0, z)
    return a


# held-out test numbers, computed here so the page shows what the model really scored
Xte = (feats["test_pca"] - net["mu"]) / net["sd"]
yte = np.array([CLASSES.index(v) for v in feats["test_y"]])
pred = forward(Xte).argmax(axis=1)
cm = np.zeros((len(CLASSES), len(CLASSES)), int)
for t, p in zip(yte, pred):
    cm[t, p] += 1
per_class = {}
for i, c in enumerate(CLASSES):
    prec = cm[i, i] / max(cm[:, i].sum(), 1)
    rec = cm[i, i] / max(cm[i].sum(), 1)
    per_class[c] = dict(precision=round(float(prec), 2), recall=round(float(rec), 2))

model = dict(
    classes=CLASSES, W=IMG_W, H=IMG_H,
    box=dict(W=b64(box["W"]), b=b64(box["b"]), mu=b64(box["mu"]),
             size=[int(v) for v in box["size"]], pad=float(box["pad"])),
    k=int(pca["components"].shape[0]), D=int(pca["components"].shape[1]),
    pca_mean=b64(pca["mean"]), pca_components=b64(pca["components"]),
    mu=b64(net["mu"].reshape(-1)), sd=float(net["sd"]), layers=layers,
    eval=dict(accuracy=round(float((pred == yte).mean()), 2), n_test=int(len(yte)),
              n_train=int(feats["train_pca"].shape[0]), components=int(pca["components"].shape[0]),
              confusion=cm.tolist(), per_class=per_class),
)
js = json.dumps(model)
tpl = (WEB / "template.html").read_text(encoding="utf-8")
(WEB / "demo.html").write_text(tpl.replace("__MODEL_JSON__", js), encoding="utf-8")
print(f"Wrote {WEB / 'demo.html'} ({(WEB / 'demo.html').stat().st_size / 1e6:.1f} MB)")
print(f"Test accuracy from this export: {model['eval']['accuracy']} on {len(yte)} patients")
print("Confusion (rows true, cols predicted):", cm.tolist())
