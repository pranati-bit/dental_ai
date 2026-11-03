"""Shared settings. Change values here, not inside the other scripts."""
import os
from pathlib import Path

# Folder that contains train/ (and test/, optionally val/).
# Default: data/pics  ->  put your unzipped "pics" folder there.
# Or set an environment variable PICS_ROOT to point somewhere else.
DATA_ROOT = Path(os.environ.get("PICS_ROOT", "data/pics"))

OUT = Path("results")            # all plots / tables / arrays are saved here
OUT.mkdir(exist_ok=True)
LABELS_FILE = Path("labels.csv")  # you fill this in (see 01_build_index.py)

SPLITS = ["train", "val", "test"]  # val is optional, skipped if the folder is missing
VIEW = 1                           # 1 = frontal view (we checked the images)
IMG_W, IMG_H = 144, 64             # every image is resized to this (width, height)

# Crop to the teeth area BEFORE resizing, as fractions of the photo:
# (left, top, right, bottom). Set to None to use the whole photo.
# Check it with:  python crop_preview.py
CROP_BOX = (0.0, 0.04, 0.99, 0.98)   # set from train+val tooth annotations (2nd-98th percentile): cuts teeth in ~6% of photos

# Your 3 classes. Labels in labels.csv must be written exactly like this.
CLASSES = ["no_need", "borderline", "braces_likely"]

SEED = 42

# Folder with instances_train.json / instances_val.json / instances_test.json
ANNOT_DIR = Path(os.environ.get("ANNOT_DIR", "data/annotations"))

# ---- Full OMNI download (the folder that contains annotations/, train/, val/, test/) ----
OMNI_ROOT = Path(os.environ.get("OMNI_ROOT", Path.home() / "Downloads" / "OMNI_COCO" / "omni_coco"))

# ---- How many patients to pick per class, per split (capped by what exists) ----
PER_CLASS = {"train": 70, "val": 15, "test": 15}

# ---- Labelling rule (OUR design choice, not a clinical standard; state it in the report) ----
# problem teeth in the frontal photo = rotated teeth (TT) + misaligned teeth (TM)
BORDERLINE_MIN = 1   # 1..(LIKELY_MIN-1) problem teeth -> borderline
LIKELY_MIN = 3       # >= this many problem teeth       -> braces_likely
# DO / MR (deep overjet, mandibular retrusion) are mainly seen from the SIDE, so a
# front-photo model cannot see them. Keep False for a frontal-only model.
USE_SEVERE_FLAG = False
