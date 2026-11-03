"""
STEP 2a: learn where the teeth are, so every photo is cropped the same way automatically.

The OMNI annotations give a tooth box for each tooth. We fit a ridge regression that maps a small
photo (48 x 32 colour pixels) to the box around all the teeth. It is FIT ON TRAIN ONLY and its
accuracy is reported on val. After this step, every photo (train, val, test, and new uploads) is
cropped with its predicted box: no manual crop, no annotations needed for new photos.

Writes results/box_model.npz and results/teeth_boxes.csv.
Needs data/annotations/*.json and the index from 01_build_index.py.
Run:  python 02a_teeth_box.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge

from config import ANNOT_DIR, OUT, SEED, VIEW

TEETH = {"HT", "TT", "TM", "TE"}      # tooth status codes (not appliances)
SIZE = (48, 32)                        # tiny input image (width, height)
ALPHA = 10.0                           # ridge strength (val median IoU was flat from 0.1 to 10)
PAD = 0.02                             # small margin added around the predicted box


def feat(path):
    im = Image.open(path).convert("RGB").resize(SIZE)
    return np.asarray(im, dtype=np.float32).reshape(-1) / 255.0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ---- ground-truth teeth boxes from the annotations (fractions of the photo) ----
gt = {}
for split in ["train", "val", "test"]:
    jf = ANNOT_DIR / f"instances_{split}.json"
    if not jf.exists():
        continue
    d = json.load(open(jf, encoding="utf-8"))
    id2c = {c["id"]: c["name"] for c in d["categories"]}
    boxes = defaultdict(list)
    for a in d["annotations"]:
        if id2c[a["category_id"]] in TEETH:
            boxes[a["image_id"]].append(a["bbox"])
    for im in d["images"]:
        bs = boxes.get(im["id"])
        if not bs:
            continue
        W, H = im["width"], im["height"]
        gt[Path(im["file_name"]).stem] = (
            min(b[0] for b in bs) / W, min(b[1] for b in bs) / H,
            max(b[0] + b[2] for b in bs) / W, max(b[1] + b[3] for b in bs) / H)

idx = pd.read_csv(OUT / "index.csv", dtype={"patient_id": str})
idx = idx[idx.view == VIEW].reset_index(drop=True)
idx["stem"] = idx.path.map(lambda p: Path(p.replace("\\", "/")).stem)
idx = idx[idx.stem.isin(gt)].reset_index(drop=True)

tr = idx[idx.split == "train"].reset_index(drop=True)
va = idx[idx.split == "val"].reset_index(drop=True)
Xtr = np.stack([feat(p) for p in tr.path])
Ytr = np.array([gt[s] for s in tr.stem])
Xva = np.stack([feat(p) for p in va.path])
Yva = np.array([gt[s] for s in va.stem])

mu = Xtr.mean(axis=0)
model = Ridge(alpha=ALPHA, random_state=SEED).fit(Xtr - mu, Ytr)


def predict_boxes(X):
    P = model.predict(X - mu)
    P = np.clip(P, 0, 1)
    l, t, r, b = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    out = np.stack([np.clip(l - PAD, 0, 1), np.clip(t - PAD, 0, 1),
                    np.clip(r + PAD, 0, 1), np.clip(b + PAD, 0, 1)], axis=1)
    return out


pv = predict_boxes(Xva)
ious = np.array([iou(p, g) for p, g in zip(pv, Yva)])
print(f"val (not used for fitting): {len(va)} photos | median IoU {np.median(ious):.2f} | "
      f"IoU>0.5 {100 * (ious > 0.5).mean():.0f}% | IoU>0.7 {100 * (ious > 0.7).mean():.0f}%")

# ---- predicted box for EVERY frontal photo in the index (this is what the crop uses) ----
allidx = pd.read_csv(OUT / "index.csv", dtype={"patient_id": str})
allidx = allidx[allidx.view == VIEW].reset_index(drop=True)
Xall = np.stack([feat(p) for p in allidx.path])
B = predict_boxes(Xall)
out = allidx[["split", "patient_id", "path"]].copy()
out[["l", "t", "r", "b"]] = B
out.to_csv(OUT / "teeth_boxes.csv", index=False)

np.savez(OUT / "box_model.npz", W=model.coef_.T, b=model.intercept_, mu=mu,
         size=np.array(SIZE), pad=PAD)
print(f"Saved {OUT / 'teeth_boxes.csv'} ({len(out)} photos) and {OUT / 'box_model.npz'}")
