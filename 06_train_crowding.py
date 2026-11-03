"""
STEP 6: a crowding model (yes / no) on the SAME PCA features as step 3.

Why binary: OMNI has no separate labels for crowding, spacing, open bite or deep bite.
The only frontal signal is the tooth code TM (misalignment) and TT (rotation), and your
labels.csv already counts them in the column problem_teeth. So, as a PROXY:
    crowding = 1  if problem_teeth >= 1   (= your 'borderline' + 'braces_likely' patients)
    crowding = 0  if problem_teeth == 0   (= your 'no_need' patients)
This is a proxy, not a dental diagnosis. Crowding vs spacing cannot be told apart from these
labels, and open bite / deep bite are not labelled at all (see XAI_README.md).

The network is the same design as 03_ann.py (copied below, not imported, because 03_ann.py
runs its whole experiment on import). Hyper-parameters are the same and were NOT tuned on test.

Writes:  results/crowding_model.npz, results/crowding_metrics.json, results/crowding_loss.png,
         results/crowding_confusion.png
Needs:   results/features.npz (from 02_pca.py) and labels.csv.
Run:     python 06_train_crowding.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from config import OUT, SEED

HIDDEN = (16, 8)
LR = 0.1
EPOCHS = 500
CLASSES2 = ["no_crowding", "crowding"]


# ----------------------------- network (same as 03_ann.py) -----------------------------
def init(sizes, rng):
    return [[rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])),
             np.zeros((1, sizes[i + 1]))] for i in range(len(sizes) - 1)]


def forward(P, X):
    acts, zs, a = [X], [], X
    for l, (W, b) in enumerate(P):
        z = a @ W + b
        zs.append(z)
        if l == len(P) - 1:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            a = e / e.sum(axis=1, keepdims=True)
        else:
            a = np.maximum(0, z)
        acts.append(a)
    return acts, zs


def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def backward(P, acts, zs, Y):
    n = Y.shape[0]
    grads = [None] * len(P)
    delta = (acts[-1] - Y) / n
    for l in reversed(range(len(P))):
        grads[l] = (acts[l].T @ delta, delta.sum(axis=0, keepdims=True))
        if l > 0:
            delta = (delta @ P[l][0].T) * (zs[l - 1] > 0)
    return grads


def train(X, y, sizes, C, seed=SEED):
    rng = np.random.default_rng(seed)
    P, Y, history = init(sizes, rng), np.eye(C)[y], []
    for _ in range(EPOCHS):
        acts, zs = forward(P, X)
        history.append(cross_entropy(acts[-1], Y))
        for (W, b), (dW, db) in zip(P, backward(P, acts, zs, Y)):
            W -= LR * dW
            b -= LR * db
    return P, history


def gradient_check(C):
    """Backprop vs finite differences on a tiny random net (same test as 03_ann.py)."""
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(6, 5)), rng.integers(0, C, 6)
    Y, P = np.eye(C)[y], init([5, 4, C], rng)
    acts, zs = forward(P, X)
    grads, worst = backward(P, acts, zs, Y), 0.0
    for l, (W, _) in enumerate(P):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                old = W[i, j]
                W[i, j] = old + 1e-5; lp = cross_entropy(forward(P, X)[0][-1], Y)
                W[i, j] = old - 1e-5; lm = cross_entropy(forward(P, X)[0][-1], Y)
                W[i, j] = old
                num, ana = (lp - lm) / 2e-5, grads[l][0][i, j]
                worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    print(f"Gradient check: worst relative error = {worst:.2e}  "
          f"({'OK' if worst < 1e-4 else 'CHECK THE CODE'})")


# ----------------------------- data -----------------------------
d = np.load(OUT / "features.npz", allow_pickle=True)


def get(split):
    y = d[f"{split}_y"]
    keep = y != ""
    yb = (y[keep] != "no_need").astype(int)        # 1 = crowding proxy present
    return d[f"{split}_pca"][keep], yb, d[f"{split}_ids"][keep]


Xtr, ytr, _ = get("train")
Xva, yva, _ = get("val")
Xte, yte, _ = get("test")
print(f"Train: {len(ytr)} patients ({ytr.sum()} crowding)  |  val: {len(yva)} ({yva.sum()})  |  "
      f"test: {len(yte)} ({yte.sum()})")

mu, sd = Xtr.mean(axis=0), Xtr.std() + 1e-8        # fitted on TRAIN only
Xtr_s, Xva_s, Xte_s = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

gradient_check(2)

P, hist = train(Xtr_s, ytr, [Xtr_s.shape[1], *HIDDEN, 2], C=2)


def probs(X):
    return forward(P, X)[0][-1][:, 1]


def report(name, y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    out = dict(n=int(len(y)), accuracy=round(float((pred == y).mean()), 3),
               balanced_accuracy=round(float((sens + spec) / 2), 3),
               sensitivity=round(float(sens), 3), specificity=round(float(spec), 3),
               auc=round(float(roc_auc_score(y, p)), 3),
               majority_baseline_accuracy=round(float(max(y.mean(), 1 - y.mean())), 3),
               confusion=[[tn, fp], [fn, tp]])
    print(f"{name:6s} n={out['n']:3d}  acc {out['accuracy']:.2f} (majority {out['majority_baseline_accuracy']:.2f})  "
          f"bal-acc {out['balanced_accuracy']:.2f}  sens {out['sensitivity']:.2f}  "
          f"spec {out['specificity']:.2f}  AUC {out['auc']:.2f}")
    return out


print("\n==== Crowding proxy model (no tuning; the test set was not used for any choice) ====")
metrics = dict(
    definition="crowding proxy = problem_teeth >= 1 (rotated TT or misaligned TM front teeth)",
    train=dict(n=int(len(ytr)), positives=int(ytr.sum())),
    val=report("val", yva, probs(Xva_s)),
    test=report("test", yte, probs(Xte_s)),
    train_accuracy=round(float(((probs(Xtr_s) >= 0.5) == ytr).mean()), 3),
    note="Small data: 39 test patients means each error moves accuracy by about 2.6 points. "
         "Treat these as a pipeline check, not as clinical accuracy.",
)

# ----------------------------- save -----------------------------
save = dict(mu=mu, sd=sd)
for i, (W, b) in enumerate(P):
    save[f"W{i}"], save[f"b{i}"] = W, b
np.savez(OUT / "crowding_model.npz", **save)
(OUT / "crowding_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

plt.figure(figsize=(5.5, 3.8))
plt.plot(hist)
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("Crowding proxy: loss vs epochs")
plt.grid(alpha=0.3)
plt.savefig(OUT / "crowding_loss.png", dpi=120, bbox_inches="tight"); plt.close()

cm = np.array(metrics["test"]["confusion"])
plt.figure(figsize=(4.3, 3.8))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xticks([0, 1], CLASSES2, rotation=20); plt.yticks([0, 1], CLASSES2)
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Crowding proxy confusion (test)")
plt.savefig(OUT / "crowding_confusion.png", dpi=120, bbox_inches="tight"); plt.close()

print(f"\nSaved {OUT / 'crowding_model.npz'}, {OUT / 'crowding_metrics.json'} and plots.")
