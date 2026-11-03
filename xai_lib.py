"""
Shared helpers for the crowding model and its explanations (steps 06 and 07).

Nothing here changes the earlier steps. It only READS the files they wrote in results/:
  results/pca_params.npz   (02_pca.py)        PCA mean + components
  results/box_model.npz    (02a_teeth_box.py) learned teeth-box regressor
  results/teeth_boxes.csv  (02a_teeth_box.py) teeth box for every frontal photo
  results/crowding_model.npz (06_train_crowding.py) the crowding network

The preprocessing is copied step by step from 02a_teeth_box.py and 02_pca.py, so a new
photo goes through exactly the same steps as the training photos.
"""
import numpy as np
from PIL import Image, ImageOps

from config import IMG_H, IMG_W, OUT

PAD_NOTE = "box padding is read from box_model.npz"


# ----------------------------- loading -----------------------------
def load_pca():
    p = np.load(OUT / "pca_params.npz")
    return p["mean"], p["components"]


def load_box_model():
    b = np.load(OUT / "box_model.npz")
    return dict(W=b["W"], b=b["b"], mu=b["mu"],
                size=tuple(int(v) for v in b["size"]), pad=float(b["pad"]))


def load_crowding_net():
    d = np.load(OUT / "crowding_model.npz")
    n = sum(1 for k in d.files if k.startswith("W"))
    return dict(mu=d["mu"], sd=float(d["sd"]),
                layers=[(d[f"W{i}"], d[f"b{i}"]) for i in range(n)])


# ----------------------------- preprocessing (same as 02a / 02_pca) -----------------------------
def predict_teeth_box(photo_path, bm):
    """Same as predict_boxes() in 02a_teeth_box.py, for ONE new photo."""
    im = Image.open(photo_path).convert("RGB").resize(bm["size"])
    x = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    p = np.clip((x - bm["mu"]) @ bm["W"] + bm["b"], 0, 1)
    pad = bm["pad"]
    l, t, r, b = p
    return (float(np.clip(l - pad, 0, 1)), float(np.clip(t - pad, 0, 1)),
            float(np.clip(r + pad, 0, 1)), float(np.clip(b + pad, 0, 1)))


def crop_teeth(photo_path, box):
    """Same steps as load_image() in 02_pca.py: gray -> crop -> resize -> equalize.
    Returns a (IMG_H, IMG_W) float array in [0, 1]."""
    im = Image.open(photo_path).convert("L")
    l, t, r, b = box
    w, h = im.size
    im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    im = im.resize((IMG_W, IMG_H))
    im = ImageOps.equalize(im)
    return np.asarray(im, dtype=np.float32).reshape(IMG_H, IMG_W) / 255.0


# ----------------------------- the network -----------------------------
def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward_probs(net, Z):
    """Z: standardised PCA features, shape (N, k). Returns class probabilities (N, 2)."""
    a = Z
    last = len(net["layers"]) - 1
    for i, (W, b) in enumerate(net["layers"]):
        z = a @ W + b
        a = _softmax(z) if i == last else np.maximum(0, z)
    return a


def pixels_to_probs(net, pca, X):
    """X: flattened crops (N, IMG_H*IMG_W) in [0, 1]. Full pipeline: PCA -> standardise -> network."""
    mean, C = pca
    Z = ((X - mean) @ C.T - net["mu"]) / net["sd"]
    return forward_probs(net, Z)


# ----------------------------- explanations -----------------------------
def occlusion_map(net, pca, img, win=8, stride=4):
    """Occlusion sensitivity. Slide a grey patch over the crop; where hiding a region LOWERS
    P(crowding) a lot, that region supported the crowding prediction.
    Returns (heat, p_base): heat has the crop's shape, positive = supports crowding."""
    H, W = img.shape
    x = img.reshape(-1).astype(np.float32)
    p_base = float(pixels_to_probs(net, pca, x[None])[0, 1])
    fill = float(img.mean())                      # neutral grey, so the patch adds no edge
    wins = [(i, j) for i in range(0, H - win + 1, stride) for j in range(0, W - win + 1, stride)]
    batch = np.repeat(img[None].astype(np.float32), len(wins), axis=0)
    for k, (i, j) in enumerate(wins):
        batch[k, i:i + win, j:j + win] = fill
    p_occ = pixels_to_probs(net, pca, batch.reshape(len(wins), -1))[:, 1]
    drop = p_base - p_occ
    heat, cnt = np.zeros((H, W)), np.zeros((H, W))
    for k, (i, j) in enumerate(wins):
        heat[i:i + win, j:j + win] += drop[k]
        cnt[i:i + win, j:j + win] += 1
    return heat / np.maximum(cnt, 1), p_base


def input_gradient(net, pca, img, target=1):
    """Exact gradient of P(target class) with respect to every pixel of the crop.
    Backpropagation through the network, then through the PCA (a linear map)."""
    mean, C = pca
    x = img.reshape(1, -1).astype(np.float64)
    pc = (x - mean) @ C.T
    z0 = (pc - net["mu"]) / net["sd"]
    acts, zs, a = [z0], [], z0
    last = len(net["layers"]) - 1
    for i, (W, b) in enumerate(net["layers"]):
        z = a @ W + b
        zs.append(z)
        a = _softmax(z) if i == last else np.maximum(0, z)
        acts.append(a)
    p = a
    onehot = np.zeros_like(p)
    onehot[0, target] = 1.0
    delta = p[:, [target]] * (onehot - p)        # d p_target / d logits (softmax rule)
    for i in reversed(range(len(net["layers"]))):
        W, _ = net["layers"][i]
        delta = delta @ W.T                       # gradient w.r.t. this layer's input
        if i > 0:
            delta = delta * (zs[i - 1] > 0)       # ReLU derivative of the layer before
    g = (delta[0] / net["sd"]) @ C                # chain rule through standardise + PCA
    return g.reshape(img.shape)


def normalise_positive(a):
    """Keep only what SUPPORTS the prediction (positive part), scaled to 0..1 for display."""
    a = np.clip(a, 0, None)
    m = a.max()
    return a / m if m > 0 else a


def hot_spot_text(heat, frac=0.10):
    """Where in the teeth crop the strongest support lies, in plain words."""
    H, W = heat.shape
    thr = np.quantile(heat, 1 - frac)
    rows, cols = np.nonzero(heat >= thr)
    r, c = rows.mean() / H, cols.mean() / W
    horiz = "left" if c < 1 / 3 else ("centre" if c < 2 / 3 else "right")
    vert = "upper" if r < 0.5 else "lower"
    return f"{vert} {horiz} part of the teeth row (as seen in the photo)"


# ----------------------------- ground-truth overlap (for checking only) -----------------------------
def boxes_to_crop_mask(boxes_px, photo_wh, box):
    """Rasterise OMNI tooth boxes (pixel x,y,w,h) into the model's crop grid (IMG_H x IMG_W).
    Used only to CHECK the explanation against dentist annotations."""
    Wp, Hp = photo_wh
    l, t, r, b = box
    mask = np.zeros((IMG_H, IMG_W), bool)
    cu = np.arange(IMG_W) + 0.5
    cv = np.arange(IMG_H) + 0.5
    for x, y, w, h in boxes_px:
        u0 = (x / Wp - l) / (r - l); u1 = ((x + w) / Wp - l) / (r - l)
        v0 = (y / Hp - t) / (b - t); v1 = ((y + h) / Hp - t) / (b - t)
        cols = (cu / IMG_W >= u0) & (cu / IMG_W <= u1)
        rows = (cv / IMG_H >= v0) & (cv / IMG_H <= v1)
        mask |= np.outer(rows, cols)
    return mask
