"""
Preview the teeth crop BEFORE running PCA.

Shows 12 random training photos with the crop box drawn in red, and the cropped result
underneath. If teeth are cut off or too much lip is inside, change CROP_BOX in config.py
and run this again.

Run:  python crop_preview.py
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import CROP_BOX, IMG_H, IMG_W, OUT, SEED, VIEW

if not CROP_BOX:
    raise SystemExit("CROP_BOX is None in config.py, nothing to preview.")
idx = pd.read_csv(OUT / "index.csv")
paths = list(idx[(idx.split == "train") & (idx.view == VIEW)].path)
random.Random(SEED).shuffle(paths)
paths = paths[:12]

fig, axes = plt.subplots(4, 6, figsize=(16, 8))
for k, p in enumerate(paths):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    l, t, r, b = CROP_BOX
    box = (int(l * w), int(t * h), int(r * w), int(b * h))
    row, col = (k // 6) * 2, k % 6
    ax = axes[row][col]
    ax.imshow(im); ax.axis("off")
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               fill=False, edgecolor="red", linewidth=2))
    ax2 = axes[row + 1][col]
    ax2.imshow(im.crop(box).resize((IMG_W * 2, IMG_H * 2))); ax2.axis("off")
plt.suptitle(f"Red box = crop {CROP_BOX}. Row below each photo = what PCA will see")
plt.tight_layout()
plt.savefig(OUT / "crop_preview.png", dpi=100)
print(f"Saved {OUT / 'crop_preview.png'}  <- open it and check the crops")
