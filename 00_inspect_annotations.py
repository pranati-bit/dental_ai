"""
STEP 0: Look inside the OMNI annotation files (nothing is changed or saved except a summary).

Prints: label categories, how many images carry each category, example file names,
and extra fields. Send the printed output (or results/annotation_summary.txt) to your guide.

Put the 3 json files in  data/annotations/  (or set ANNOT_DIR in config.py).
Run:  python 00_inspect_annotations.py
"""
import json
from collections import Counter

from config import ANNOT_DIR, OUT

files = sorted(ANNOT_DIR.glob("instances_*.json"))
if not files:
    raise SystemExit(f"No instances_*.json found in {ANNOT_DIR}. Check ANNOT_DIR in config.py")

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


for jf in files:
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    say("=" * 60)
    say(f"{jf.name}")
    say(f"top-level keys: {list(data.keys())}")
    imgs, anns, cats = data.get("images", []), data.get("annotations", []), data.get("categories", [])
    say(f"images: {len(imgs)} | annotations: {len(anns)} | categories: {len(cats)}")

    id2name = {c["id"]: c["name"] for c in cats}
    say("\nCategories (id: name):")
    for cid, name in sorted(id2name.items()):
        say(f"  {cid}: {name}")

    # how many different IMAGES carry each category
    per_cat = Counter()
    cats_per_img = {}
    for a in anns:
        cats_per_img.setdefault(a["image_id"], set()).add(a["category_id"])
    for s in cats_per_img.values():
        for cid in s:
            per_cat[cid] += 1
    say("\nImages carrying each category:")
    for cid, n in per_cat.most_common():
        say(f"  {id2name.get(cid, cid)}: {n}")
    say(f"\nImages with NO annotation at all: {len(imgs) - len(cats_per_img)}")
    say("Labels per image (how many images have N different categories): "
        f"{dict(sorted(Counter(len(s) for s in cats_per_img.values()).items()))}")

    say("\nExample image entries:")
    for im in imgs[:3]:
        say(f"  {im}")
    if anns:
        say("Example annotation entry (keys only): " + str(list(anns[0].keys())))
    say()

(OUT / "annotation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {OUT / 'annotation_summary.txt'}")
