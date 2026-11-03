"""
STEP 1: Scan the image folders and build an index.

- patient_id = file name without the last digit
- view       = last digit of the file name (1 = frontal)
  (NOTE: a date folder can hold TWO patients, so we never use the folder name.)
- Creates labels.csv (empty) with one row per patient, for you to fill in.

Run:  python 01_build_index.py
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import CLASSES, DATA_ROOT, LABELS_FILE, OUT, SPLITS, VIEW

rows = []
for split in SPLITS:
    folder = DATA_ROOT / split
    if not folder.exists():
        print(f"[info] no '{split}' folder, skipping")
        continue
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = re.fullmatch(r"(\d+)(\d)", p.stem)
        if not m:
            print("[skip] unexpected file name:", p.name)
            continue
        rows.append(dict(split=split, patient_id=m.group(1),
                         view=int(m.group(2)), path=str(p)))

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"No images found in {DATA_ROOT}. Check DATA_ROOT in config.py")
df.to_csv(OUT / "index.csv", index=False)

print("\nImages per split:\n", df["split"].value_counts().to_string())
print("\nPatients per split:\n", df.groupby("split")["patient_id"].nunique().to_string())
print("\nViews per patient (patients with N images):",
      dict(df.groupby("patient_id").size().value_counts()))

leak = (df.groupby("patient_id")["split"].nunique() > 1).sum()
print("Patients appearing in more than one split (must be 0):", int(leak))

has_front = set(df[df.view == VIEW].patient_id)
no_front = sorted(set(df.patient_id) - has_front)
if no_front:
    print(f"[warning] {len(no_front)} patient(s) have no frontal view and will be "
          f"ignored: {no_front}")

# ---- labels template ----
if LABELS_FILE.exists():
    print(f"\n{LABELS_FILE} already exists, not overwritten.")
else:
    t = (df[df.view == VIEW][["split", "patient_id"]]
         .drop_duplicates().sort_values(["split", "patient_id"]))
    t["label"] = ""
    t.to_csv(LABELS_FILE, index=False)
    print(f"\nCreated {LABELS_FILE}. Fill the 'label' column with one of: {CLASSES}")

# ---- contact sheet: all views of 2 patients (check what each view number shows) ----
pats = df["patient_id"].drop_duplicates().iloc[:2]
fig, axes = plt.subplots(len(pats), 5, figsize=(15, 3.2 * len(pats)), squeeze=False)
for r, pid in enumerate(pats):
    sub = df[df.patient_id == pid].drop_duplicates("view").set_index("view")
    for v in range(1, 6):
        ax = axes[r][v - 1]
        ax.axis("off")
        if v in sub.index:
            ax.imshow(Image.open(sub.loc[v, "path"]))
            ax.set_title(f"{pid}  view {v}", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "contact_sheet.png", dpi=110)
print(f"Saved {OUT / 'contact_sheet.png'}")
