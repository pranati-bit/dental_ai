"""
Check whether OMNI contains the SAME patient at several visits (follow-up photos).

File name = 6-digit patient code + 8-digit visit date (YYYYMMDD) + 1-digit view.
This script only reads the annotation files and prints a summary. Nothing is changed.

Run:  python 05_explore_visits.py
"""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

from config import ANNOT_DIR, VIEW

APPLIANCE = {"IOA", "CFOA", "OB", "FOD"}
visits = defaultdict(dict)            # code -> {date -> info}
bad = 0
for split in ["train", "val", "test"]:
    jf = ANNOT_DIR / f"instances_{split}.json"
    if not jf.exists():
        continue
    data = json.load(open(jf, encoding="utf-8"))
    id2cat = {c["id"]: c["name"] for c in data["categories"]}
    per_img = defaultdict(Counter)
    for a in data["annotations"]:
        per_img[a["image_id"]][id2cat[a["category_id"]]] += 1
    for im in data["images"]:
        m = re.fullmatch(r"(\d{6})(\d{8})(\d)", Path(im["file_name"]).stem)
        if not m:
            bad += 1
            continue
        code, date, view = m.group(1), m.group(2), int(m.group(3))
        v = visits[code].setdefault(date, dict(split=split, appl=False, frontal=False, problem=0))
        c = per_img[im["id"]]
        if any(c.get(a) for a in APPLIANCE):
            v["appl"] = True
        if view == VIEW:
            v["frontal"] = True
            v["problem"] = c.get("TT", 0) + c.get("TM", 0)

print(f"file names not matching the pattern: {bad}")
print(f"distinct 6-digit patient codes: {len(visits)}   (the OMNI paper says 384 patients)")
vc = Counter(len(d) for d in visits.values())
print("visits per patient code (patients with N visits):", dict(sorted(vc.items())))

in_many = [c for c, d in visits.items() if len({v['split'] for v in d.values()}) > 1]
print(f"patient codes appearing in more than one split (must be 0): {len(in_many)}")

gaps, trans = [], Counter()
examples = []
for code, d in visits.items():
    dates = sorted(d)
    if len(dates) < 2:
        continue
    t = [datetime.strptime(x, "%Y%m%d") for x in dates]
    gaps += [(b - a).days for a, b in zip(t, t[1:])]
    first, last = d[dates[0]], d[dates[-1]]
    trans[(("appliance" if first["appl"] else "none"),
           ("appliance" if last["appl"] else "none"))] += 1
    if not first["appl"] and last["appl"] and first["frontal"] and len(examples) < 5:
        examples.append((code, dates[0], dates[-1], d[dates[0]]["problem"]))

multi = sum(1 for d in visits.values() if len(d) >= 2)
print(f"\npatients with 2 or more visits: {multi}")
if gaps:
    print(f"days between visits: min {min(gaps)}, median {int(median(gaps))}, max {max(gaps)}")
print("appliance status  first visit -> last visit  (patients):")
for (a, b), n in sorted(trans.items()):
    print(f"   {a:9s} -> {b:9s}: {n}")
ok = sum(1 for d in visits.values() if len(d) >= 2
         and not d[sorted(d)[0]]["appl"] and d[sorted(d)[0]]["frontal"])
print(f"patients usable for 'photo now -> later outcome' (no appliance at first visit, "
      f"frontal photo, 2+ visits): {ok}")
for e in examples:
    print("  example (no appliance first, appliance later):", e)
