"""
STEP 7 (Feature 1): explainable AI. Show WHY the crowding model said what it said.

For a photo it shows:
  * the crowding probability (a confidence, not a calibrated risk: only 276 patients)
  * a heat map of the regions that pushed the prediction towards "crowding"
    - occlusion: grey out each patch and measure how much P(crowding) drops
    - gradient: the exact sensitivity of P(crowding) to each pixel (sharper, noisier)
  * a plain-words hot spot, e.g. "upper right part of the teeth row"
  * for PHOTOS FROM OUR DATASET ONLY: the dentist's tooth boxes (red = rotated or misaligned,
    green = healthy). New photos have no tooth boxes, so the page cannot name individual teeth.

--test also checks the explanation against the dentist's tooth boxes on held-out
val and test patients with crowding: does the hot region fall on the red teeth more often
than chance? This is a "pointing game" check, the usual honest way to test a heat map.

Writes: results/explain/<name>_crowding.png and .json, and (with --test) results/explain/pointing_check.json
Run:
  python 07_explain_crowding.py path/to/new_photo.jpg      # one new photo
  python 07_explain_crowding.py --test                      # held-out patients + pointing check
  python 07_explain_crowding.py --test --max-figs 8         # how many figures to save
Run from the project folder (paths are relative, like the other steps).
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from config import ANNOT_DIR, LABELS_FILE, OUT, VIEW
from xai_lib import (boxes_to_crop_mask, crop_teeth, hot_spot_text, input_gradient,
                     load_box_model, load_crowding_net, load_pca, normalise_positive,
                     occlusion_map, predict_teeth_box)

EXP = OUT / "explain"
EXP.mkdir(parents=True, exist_ok=True)
TEETH_PROBLEM = {"TT", "TM"}     # rotated / misaligned = the crowding proxy
TEETH_HEALTHY = {"HT"}
CAVEAT = ("Crowding here means 'rotated or misaligned front teeth' (a proxy). "
          "The model is weak (test AUC about 0.7), so read the heat map as "
          "'what the model looked at', not as proof of crowding.")


# ----------------------------- one photo -----------------------------
def explain_photo(photo_path, net, pca, box, gt=None):
    """gt: optional (photo_wh, [(cat, x, y, w, h), ...]) for dataset photos."""
    img = crop_teeth(photo_path, box)
    occ, p_crowd = occlusion_map(net, pca, img)
    grad = input_gradient(net, pca, img, target=1)
    p = np.array([1 - p_crowd, p_crowd])
    return dict(img=img, occ=occ, grad=grad, p=p, box=box, hot=hot_spot_text(occ))


def draw(photo_path, res, out_png, title, gt=None):
    photo = np.asarray(Image.open(photo_path).convert("RGB"))
    H, W = photo.shape[:2]
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.25], hspace=0.35, wspace=0.12)
    l, t, r, b = res["box"]

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(photo); ax.set_title("Photo and teeth area", fontsize=10)
    ax.add_patch(Rectangle((l * W, t * H), (r - l) * W, (b - t) * H, fill=False, ec="#2b6cb0", lw=2))
    if gt is not None:
        for cat, x, y, w, h in gt[1]:
            if cat in TEETH_PROBLEM:
                col = "#e53e3e"
            elif cat in TEETH_HEALTHY:
                col = "#38a169"
            else:
                continue
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=1.1))
        ax.set_xlabel("red = dentist: rotated/misaligned   green = healthy", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(res["img"], cmap="gray"); ax.set_title("Teeth crop the model sees", fontsize=10)
    ax.axis("off")

    occ_n = normalise_positive(res["occ"])
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(res["img"], cmap="gray")
    ax.imshow(np.ma.masked_where(occ_n < 0.15, occ_n), cmap="magma", alpha=0.7, vmin=0, vmax=1)
    ax.set_title("Occlusion: regions that support 'crowding'", fontsize=10); ax.axis("off")

    g_n = normalise_positive(res["grad"])
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(res["img"], cmap="gray")
    ax.imshow(np.ma.masked_where(g_n < 0.15, g_n), cmap="magma", alpha=0.7, vmin=0, vmax=1)
    ax.set_title("Gradient: pixel sensitivity", fontsize=10); ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    names = ["no crowding", "crowding (proxy)"]
    bars = ax.barh(names, res["p"], color=["#a0aec0", "#c05621"])
    ax.set_xlim(0, 1); ax.set_xlabel("model confidence (not calibrated)")
    for bar, v in zip(bars, res["p"]):
        ax.text(min(v + 0.02, 0.9), bar.get_y() + bar.get_height() / 2, f"{v:.0%}", va="center")
    ax.set_title(title, fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
    lines = [f"Prediction: {'crowding' if res['p'][1] >= 0.5 else 'no crowding'} "
             f"({res['p'][1]:.0%} crowding)",
             f"Strongest support: {res['hot']}",
             "Individual teeth: " + ("marked by the dentist (red boxes)" if gt is not None
                                     else "not available for new photos (no tooth detector yet)"),
             ""] + textwrap.wrap(CAVEAT, 70)
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=9.5)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- pointing-game check -----------------------------
def dataset_gt():
    """Tooth boxes from the OMNI annotations, keyed by file stem."""
    out = {}
    for split in ["val", "test"]:
        jf = ANNOT_DIR / f"instances_{split}.json"
        d = json.load(open(jf, encoding="utf-8"))
        id2c = {c["id"]: c["name"] for c in d["categories"]}
        per = {}
        for a in d["annotations"]:
            per.setdefault(a["image_id"], []).append((id2c[a["category_id"]], *a["bbox"]))
        for im in d["images"]:
            out[Path(im["file_name"]).stem] = ((im["width"], im["height"]), per.get(im["id"], []))
    return out


def pointing_check(res, gt_wh, boxes):
    """Share of the strongest 20% of the heat map that lands on the problem teeth,
    compared with the share of the whole crop those teeth cover (= chance)."""
    box = res["box"]
    problem = [(x, y, w, h) for c, x, y, w, h in boxes if c in TEETH_PROBLEM]
    if not problem:
        return None
    mask = boxes_to_crop_mask(problem, gt_wh, box)
    occ = res["occ"]
    thr = np.quantile(occ, 0.80)
    hot = occ >= thr
    hit = float(mask[hot].mean())
    chance = float(mask.mean())
    return dict(hit_rate=hit, chance=chance, lift=(hit / chance) if chance > 0 else None)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?", help="a new frontal photo")
    ap.add_argument("--test", action="store_true", help="run on held-out patients + pointing check")
    ap.add_argument("--max-figs", type=int, default=6)
    args = ap.parse_args()

    net, pca, bm = load_crowding_net(), load_pca(), load_box_model()

    if args.photo:
        photo = Path(args.photo)
        box = predict_teeth_box(photo, bm)
        res = explain_photo(photo, net, pca, box)
        stem = photo.stem
        out_png = EXP / f"{stem}_crowding.png"
        draw(photo, res, out_png, title=f"New photo: {photo.name}")
        summary = dict(photo=str(photo), p_crowding=round(float(res["p"][1]), 3),
                       prediction="crowding" if res["p"][1] >= 0.5 else "no crowding",
                       strongest_support=res["hot"], teeth_box_fractions=[round(v, 3) for v in box],
                       note=CAVEAT, figure=str(out_png))
        (EXP / f"{stem}_crowding.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    if not args.test:
        ap.error("give a photo path, or use --test")

    # ---- held-out dataset patients ----
    idx = pd.read_csv(OUT / "index.csv", dtype={"patient_id": str})
    idx = idx[(idx.view == VIEW) & idx.split.isin(["val", "test"])]
    boxes_csv = pd.read_csv(OUT / "teeth_boxes.csv", dtype={"patient_id": str}).set_index("path")
    lab = pd.read_csv(LABELS_FILE, dtype={"patient_id": str})[["patient_id", "label"]]
    idx = idx.merge(lab, on="patient_id", how="inner")
    gts = dataset_gt()

    rows, checks, n_fig = [], [], 0
    for _, r in idx.iterrows():
        stem = Path(r.path.replace("\\", "/")).stem
        if stem not in gts:
            continue
        box = tuple(boxes_csv.loc[r.path, ["l", "t", "r", "b"]].astype(float))
        res = explain_photo(r.path, net, pca, box)
        crowd_true = int(r.label != "no_need")
        rows.append(dict(patient=r.patient_id, split=r.split, true=crowd_true,
                         p_crowding=round(float(res["p"][1]), 3)))
        if crowd_true == 1:
            pc = pointing_check(res, gts[stem][0], gts[stem][1])
            if pc:
                checks.append(dict(patient=r.patient_id, split=r.split, **pc))
        if n_fig < args.max_figs and (crowd_true == 1 or n_fig % 2 == 0):
            draw(r.path, res, EXP / f"{r.patient_id}_{r.split}_crowding.png",
                 title=f"Held-out {r.split} patient {r.patient_id} (dentist label: "
                       f"{'crowding' if crowd_true else 'no crowding'})",
                 gt=(gts[stem][0], gts[stem][1]))
            n_fig += 1

    hits = np.array([c["hit_rate"] for c in checks])
    chance = np.array([c["chance"] for c in checks])
    lifts = np.array([c["lift"] for c in checks if c["lift"] is not None])
    summary = dict(
        note=CAVEAT,
        patients_checked=len(rows),
        crowding_patients_with_problem_boxes=len(checks),
        mean_share_of_hot_region_on_problem_teeth=round(float(hits.mean()), 3) if len(hits) else None,
        mean_chance_share=round(float(chance.mean()), 3) if len(chance) else None,
        median_lift=round(float(np.median(lifts)), 2) if len(lifts) else None,
        share_with_lift_above_1=round(float((lifts > 1).mean()), 2) if len(lifts) else None,
        interpretation=("lift > 1 means the hot region sits on the dentist-marked teeth more often "
                        "than a random region of the same size. Small sample: read it as a sanity check."),
        per_patient=checks,
    )
    (EXP / "pointing_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(EXP / "explained_patients.csv", index=False)
    print(f"Checked {len(rows)} held-out patients; {len(checks)} crowding patients with marked teeth.")
    if len(hits):
        print(f"Hot region on problem teeth: {hits.mean():.0%} vs chance {chance.mean():.0%} "
              f"(median lift {np.median(lifts):.2f}; {(lifts > 1).mean():.0%} of patients above 1)")
    print(f"Figures + pointing_check.json saved in {EXP}/")


if __name__ == "__main__":
    main()
