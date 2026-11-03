# Crowding model + explainable AI (added on top of the braces project)

Your earlier files are unchanged. Everything new is in these files:

| File | What it does |
|---|---|
| `xai_lib.py` | Shared helpers: same preprocessing as steps 02a/02, the network, occlusion and gradient explanations, the pointing-game check. |
| `06_train_crowding.py` | Trains a yes/no crowding model on the features from `02_pca.py` (same network design as `03_ann.py`). |
| `07_explain_crowding.py` | Explains a photo: confidence, heat map of the regions behind the prediction, plain-words hot spot. `--test` runs the held-out check. |
| `requirements.txt` | Python packages (numpy, pandas, pillow, matplotlib, scikit-learn). |
| `XAI_README.md` | This file. |

Your original `requirements.txt.txt` is left as it was (its name has a double extension, so pip will not read it; `requirements.txt` is the working one).

## Run it

From the project folder:

```bash
pip install -r requirements.txt

python 06_train_crowding.py                       # train the crowding model (about 2 s)
python 07_explain_crowding.py --test              # held-out patients + pointing check
python 07_explain_crowding.py path/to/photo.jpg   # explain one new frontal photo
```

Outputs go to `results/`:
- `crowding_model.npz`, `crowding_metrics.json`, `crowding_loss.png`, `crowding_confusion.png`
- `explain/<name>_crowding.png` and `.json` (one figure per patient or photo)
- `explain/pointing_check.json`, `explain/explained_patients.csv`

## What "crowding" means here (important)

OMNI has no separate labels for crowding, spacing, open bite or deep bite:

- **Crowding and spacing** share one code, `TM` (tooth misalignment). Those are the only frontal teeth marked for this, 85 boxes across the whole training split. The two cannot be told apart.
- **Open bite** has no code at all. `label_rules.py` treats `OB` as an orthodontic appliance, and I could not confirm from the data what it stands for. **Please check `OB`**: if it means open bite, the exclusion rule is removing the wrong patients.
- **Deep bite** is not labelled either. `DO` is *deep overjet*, a side-view feature, which a frontal photo cannot show (you already set `USE_SEVERE_FLAG = False` for this reason).

So the model is a **crowding proxy**: `problem_teeth >= 1`, meaning one or more rotated (`TT`) or misaligned (`TM`) front teeth. This is your existing `borderline` and `braces_likely` classes, so no new labelling was needed. It is not a dental diagnosis.

## Results (be careful with these)

| | accuracy | majority baseline | AUC | sensitivity | specificity |
|---|---|---|---|---|---|
| val (45) | 0.62 | 0.67 | 0.65 | 0.70 | 0.47 |
| test (39) | 0.64 | 0.67 | 0.70 | 0.65 | 0.61 |

- The model is **weak**. On test it is not better than always saying "crowding", which is right 67% of the time. It separates the groups only a little (AUC 0.70).
- The 192-patient training set is the main limit. With 39 test patients, one mistake moves accuracy by about 2.6 points.
- **The explanation check is negative.** For held-out crowding patients whose problem teeth the dentist marked, the strongest 20% of the heat map lands on those teeth 15% of the time, against 13% expected by chance (median lift 0.95, so no better than random). In plain words: **the model is not reliably looking at the teeth the dentist marked.** Its explanations show what it used, which may not be the teeth.
- The confidence shown is **not calibrated** and is overconfident: on the 84 held-out patients, 46% of predictions are at 5% or below, or 95% or above, even though the model separates the groups only weakly (AUC 0.70). Do not show these numbers to patients.

## How the explanation works

1. **Crop**: the photo goes through the same teeth-box model and crop as your training data.
2. **Occlusion**: a grey 8×8 patch slides over the crop. Where hiding a patch lowers P(crowding), that patch supported the prediction. This is the main heat map.
3. **Gradient**: the exact derivative of P(crowding) with respect to each pixel, through the network and the PCA. Sharper but noisier.
4. **Hot spot**: the strongest 10% of the occlusion map, described in words ("upper centre part of the teeth row").
5. **Teeth**: for photos from your dataset only, the dentist's tooth boxes are drawn (red = rotated or misaligned, green = healthy). New photos have no tooth boxes, so the page cannot say which individual teeth are involved.

Checks run while building (all passed): the crop reproduces the saved training features exactly; the pipeline reproduces the test AUC; the backprop gradient matches finite differences to 3e-7; the network's gradient check is OK.

## Not done, and why

- **Four-class output (crowding, spacing, open bite, deep bite).** Not built, because the labels do not exist (see above). Adding them would mean labelling photos by hand; that is a decision for you and a dentist.
- **Naming individual teeth on new photos.** This needs a tooth detector. OMNI's tooth boxes could train one, which is the natural next step. It is a larger job, so I did not start it.
- **Web demo.** The browser page (`web/demo.html`) is untouched. Adding the explanation there would mean re-implementing occlusion in JavaScript; say if you want it.

## Next steps that would most improve this

1. Get more labelled patients, especially crowding and spacing labelled separately by a dentist.
2. Train a tooth detector on OMNI's tooth boxes, so explanations can name teeth.
3. Calibrate the confidence (e.g. temperature scaling on the validation set) before showing it to anyone.
