"""
STEP 2: Preprocess frontal images and run PCA.

Preprocessing: grayscale -> resize -> histogram equalization -> flatten.
PCA is FIT ON TRAIN ONLY, then applied to val/test (no leakage).

Run:  python 02_pca.py        (labels.csv is optional for this step)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

from config import CLASSES, IMG_H, IMG_W, LABELS_FILE, OUT, SEED, SPLITS, VIEW


BOXES = pd.read_csv(OUT / "teeth_boxes.csv").set_index("path")   # predicted teeth box per photo


def load_image(path):
    im = Image.open(path).convert("L")
    l, t, r, b = BOXES.loc[path, ["l", "t", "r", "b"]].to_numpy(dtype=float)
    w, h = im.size                                   # keep only the predicted teeth area
    im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    im = im.resize((IMG_W, IMG_H))
    im = ImageOps.equalize(im)                       # contrast normalization
    return np.asarray(im, dtype=np.float32).reshape(-1) / 255.0


idx = pd.read_csv(OUT / "index.csv", dtype={"patient_id": str})
idx = idx[idx.view == VIEW].reset_index(drop=True)

if LABELS_FILE.exists():
    lab = pd.read_csv(LABELS_FILE, dtype={"patient_id": str})
    lab = lab[lab.label.isin(CLASSES)][["patient_id", "label"]]
    if len(lab):   # keep only labelled patients (excluded ones are dropped everywhere)
        idx = idx.merge(lab, on="patient_id", how="inner")
    else:
        idx["label"] = np.nan
else:
    idx["label"] = np.nan
idx["label"] = idx["label"].fillna("")

data = {}
for s in SPLITS:
    sub = idx[idx.split == s]
    if len(sub):
        data[s] = (np.stack([load_image(p) for p in sub.path]),
                   sub.label.to_numpy(), sub.patient_id.to_numpy())
if "train" not in data:
    raise SystemExit("No training images found.")

Xtr = data["train"][0]
print("Train matrix:", Xtr.shape, "(patients x pixels)")

# ---- PCA (keep 95% of the variance) ----
pca = PCA(n_components=0.95, svd_solver="full", random_state=SEED).fit(Xtr)
k = pca.n_components_
print(f"Components kept for 95% variance: {k}")
print("(Max possible = number of train patients - 1. More patients -> more components.)")

# ---- plots ----
plt.figure(figsize=(4, 3))
plt.imshow(pca.mean_.reshape(IMG_H, IMG_W), cmap="gray")
plt.title("Mean tooth image"); plt.axis("off")
plt.savefig(OUT / "pca_mean.png", dpi=120, bbox_inches="tight"); plt.close()

plt.figure(figsize=(5, 3.5))
plt.plot(np.arange(1, k + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("number of components"); plt.ylabel("cumulative variance")
plt.title("PCA variance curve"); plt.grid(alpha=0.3)
plt.savefig(OUT / "pca_variance.png", dpi=120, bbox_inches="tight"); plt.close()

n_show = min(10, k)
fig, axes = plt.subplots(2, 5, figsize=(12, 4.2))
for i, ax in enumerate(axes.ravel()):
    ax.axis("off")
    if i < n_show:
        ax.imshow(pca.components_[i].reshape(IMG_H, IMG_W), cmap="gray")
        ax.set_title(f"eigen-teeth {i + 1}", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "eigen_teeth.png", dpi=120); plt.close()

if k >= 2:
    Z = pca.transform(Xtr)
    labs = data["train"][1]
    plt.figure(figsize=(5, 4))
    for c in (CLASSES if any(labs) else [""]):
        m = labs == c
        if m.any():
            plt.scatter(Z[m, 0], Z[m, 1], label=c or "train")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.title("Train in PCA space")
    plt.savefig(OUT / "pca_scatter.png", dpi=120, bbox_inches="tight"); plt.close()

# ---- save features for the ANN ----
save = {}
for s, (X, y, ids) in data.items():
    save[f"{s}_raw"] = X
    save[f"{s}_pca"] = pca.transform(X)
    save[f"{s}_y"] = y.astype(str)
    save[f"{s}_ids"] = ids.astype(str)
np.savez(OUT / "features.npz", **save)
# the PCA itself (needed by the web demo to project new photos the same way)
np.savez(OUT / "pca_params.npz", mean=pca.mean_, components=pca.components_)
print(f"Saved plots and {OUT / 'features.npz'} and {OUT / 'pca_params.npz'}")
