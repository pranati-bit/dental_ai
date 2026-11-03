"""
STEP 0b: Build the dataset automatically from the FULL OMNI download.

For every patient it
  1. finds the frontal photo (file name ends in 1),
  2. skips patients who already wear appliances (braces, aligner attachments, ...),
  3. gives the patient a class (no_need / borderline / braces_likely) using label_rules.py,
  4. picks a BALANCED number of patients per class, per official split,
  5. copies the frontal photos into data/pics/<split>/ and writes labels.csv.

Before running: move your old data/pics and labels.csv away (rename them).
Run:  python 00b_pick_subset.py
"""
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from config import (ANNOT_DIR, BORDERLINE_MIN, CLASSES, DATA_ROOT, LABELS_FILE,
                    LIKELY_MIN, OMNI_ROOT, PER_CLASS, SEED, USE_SEVERE_FLAG, VIEW)
from label_rules import label_patient

IMG_EXT = {".jpg", ".jpeg", ".png"}
PAPER_FRONTAL = {"train": 534, "val": 187, "test": 182}   # frontal images reported in the OMNI paper

if DATA_ROOT.exists() and any(p.suffix.lower() in IMG_EXT for p in DATA_ROOT.rglob("*")):
    raise SystemExit(f"{DATA_ROOT} already has images. Rename or delete it first, then run again.")
if LABELS_FILE.exists():
    raise SystemExit(f"{LABELS_FILE} already exists. Rename or delete it first, then run again.")

print(f"OMNI_ROOT = {OMNI_ROOT}  (exists: {OMNI_ROOT.exists()})")
print(f"ANNOT_DIR = {ANNOT_DIR}  (exists: {ANNOT_DIR.exists()})")
if not OMNI_ROOT.exists():
    print("[warning] OMNI_ROOT does not exist -> set it in config.py to the folder that "
          "contains train/, val/ and test/")

rng = random.Random(SEED)
rows = []
for split in ["train", "val", "test"]:
    jf = ANNOT_DIR / f"instances_{split}.json"
    if not jf.exists():
        print(f"[skip] {jf} not found")
        continue
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    id2cat = {c["id"]: c["name"] for c in data["categories"]}
    per_img = defaultdict(Counter)
    for a in data["annotations"]:
        per_img[a["image_id"]][id2cat[a["category_id"]]] += 1

    patients = defaultdict(dict)                      # patient_id -> {view: image entry}
    for im in data["images"]:
        m = re.fullmatch(r"(\d+)(\d)", Path(im["file_name"]).stem)
        if m:
            patients[m.group(1)][int(m.group(2))] = im
    n_front = sum(1 for v in patients.values() if VIEW in v)
    print(f"\n[{split}] patients: {len(patients)} | frontal images (ending in {VIEW}): {n_front}"
          f" | OMNI paper reports {PAPER_FRONTAL[split]} frontal images")

    on_disk = {}
    if (OMNI_ROOT / split).exists():
        on_disk = {p.stem: p for p in (OMNI_ROOT / split).rglob("*")
                   if p.suffix.lower() in IMG_EXT}
    print(f"  image files found on disk in {OMNI_ROOT / split}: {len(on_disk)}")

    labelled, skipped, hist = defaultdict(list), Counter(), Counter()
    for pid, views in patients.items():
        if VIEW not in views:
            skipped["no frontal photo"] += 1
            continue
        front = views[VIEW]
        allc = Counter()
        for im in views.values():
            allc.update(per_img[im["id"]])
        r = label_patient(per_img[front["id"]], allc, BORDERLINE_MIN, LIKELY_MIN,
                          USE_SEVERE_FLAG)
        if r["label"] is None:
            skipped[r["excluded"]] += 1
            continue
        hist[min(r["problem"], 8)] += 1
        labelled[r["label"]].append((pid, front, r))

    # labels are known for all of these; now keep only those whose photo we really have
    found = defaultdict(list)
    for c in CLASSES:
        for pid, front, r in labelled[c]:
            src = on_disk.get(Path(front["file_name"]).stem)
            if src is None:
                skipped["image file missing"] += 1
            else:
                found[c].append((pid, src, front, r))

    print("  skipped:", dict(skipped))
    print("  rotated/misaligned teeth per frontal photo (8 = 8 or more):",
          dict(sorted(hist.items())))
    print("  patients per class by the rule (before checking photos):",
          {c: len(labelled[c]) for c in CLASSES})
    avail = {c: len(found[c]) for c in CLASSES}
    n = min(PER_CLASS[split], min(avail.values()))
    print(f"  available per class: {avail} -> picking {n} per class")
    if n == 0:
        print("  [warning] a class is empty in this split; nothing picked here.")
    for c in CLASSES:
        for pid, src, front, r in rng.sample(found[c], n):
            dst = DATA_ROOT / split / front["subfolder1"] / front["subfolder2"] / front["file_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rows.append(dict(split=split, patient_id=pid, label=c,
                             problem_teeth=r["problem"], deep_overjet_or_retrusion=r["severe"]))

if not rows:
    raise SystemExit("Nothing was picked. Check OMNI_ROOT and ANNOT_DIR in config.py")
out = pd.DataFrame(rows)
out.to_csv(LABELS_FILE, index=False)
print("\nPicked patients:")
print(out.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
print(f"\nCopied photos to {DATA_ROOT}/  and wrote {LABELS_FILE}")
