"""
STEP 3: ANN built from scratch in NumPy.

forward propagation -> cross-entropy loss -> backpropagation -> gradient descent.
Compares: ANN on PCA features  vs  ANN on raw pixels (with vs without PCA).

Needs labels.csv filled in (train AND test patients).
Run:  python 03_ann.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES, OUT, SEED

C = len(CLASSES)
HIDDEN = (16, 8)       # two small hidden layers (data is small)
LR = 0.1               # learning rate
EPOCHS = 500


# ----------------------------- the network -----------------------------
def init(sizes, rng):
    return [[rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])),
             np.zeros((1, sizes[i + 1]))] for i in range(len(sizes) - 1)]


def forward(P, X):
    """Returns activations and pre-activations of every layer."""
    acts, zs, a = [X], [], X
    for l, (W, b) in enumerate(P):
        z = a @ W + b
        zs.append(z)
        if l == len(P) - 1:                                  # softmax output
            e = np.exp(z - z.max(axis=1, keepdims=True))
            a = e / e.sum(axis=1, keepdims=True)
        else:                                                # ReLU hidden layers
            a = np.maximum(0, z)
        acts.append(a)
    return acts, zs


def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def backward(P, acts, zs, Y):
    """Backpropagation: chain rule from the output layer back to the input."""
    n = Y.shape[0]
    grads = [None] * len(P)
    delta = (acts[-1] - Y) / n                   # softmax + cross-entropy gradient
    for l in reversed(range(len(P))):
        grads[l] = (acts[l].T @ delta, delta.sum(axis=0, keepdims=True))
        if l > 0:
            delta = (delta @ P[l][0].T) * (zs[l - 1] > 0)    # ReLU derivative
    return grads


def train(X, y, sizes, seed=SEED):
    rng = np.random.default_rng(seed)
    P, Y, history = init(sizes, rng), np.eye(C)[y], []
    for _ in range(EPOCHS):
        acts, zs = forward(P, X)
        history.append(cross_entropy(acts[-1], Y))
        for (W, b), (dW, db) in zip(P, backward(P, acts, zs, Y)):
            W -= LR * dW                          # gradient descent update
            b -= LR * db
    return P, history


def predict(P, X):
    return forward(P, X)[0][-1].argmax(axis=1)


def gradient_check():
    """Compare backprop gradients with numerical gradients (proves the code is right)."""
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


def metrics(y, pred):
    cm = np.zeros((C, C), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    prec = cm.diagonal() / np.maximum(cm.sum(axis=0), 1)
    rec = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return cm, prec, rec, f1


# ----------------------------- data -----------------------------
d = np.load(OUT / "features.npz", allow_pickle=True)


def get(split, kind):
    y = d[f"{split}_y"]
    keep = np.isin(y, CLASSES)
    return d[f"{split}_{kind}"][keep], np.array([CLASSES.index(v) for v in y[keep]])


if "test_y" not in d.files:
    raise SystemExit("No test split found.")
ytr_all = get("train", "pca")[1]
yte_all = get("test", "pca")[1]
if len(ytr_all) == 0 or len(yte_all) == 0:
    raise SystemExit("Fill the 'label' column in labels.csv (train AND test) first. "
                     f"Allowed labels: {CLASSES}")
print(f"Labelled patients -> train: {len(ytr_all)}, test: {len(yte_all)}")
print("Train class counts:", dict(zip(CLASSES, np.bincount(ytr_all, minlength=C))))
print("Test  class counts:", dict(zip(CLASSES, np.bincount(yte_all, minlength=C))))
if len(ytr_all) < 60:
    print("[warning] very small training set: treat the numbers as a pipeline test, "
          "not as real results.")

gradient_check()

# ----------------------------- experiments -----------------------------
results, curves = {}, {}
for name, kind in [("with PCA", "pca"), ("without PCA (raw pixels)", "raw")]:
    Xtr, ytr = get("train", kind)
    Xte, yte = get("test", kind)
    mu, sd = Xtr.mean(axis=0), Xtr.std() + 1e-8      # centre, one global scale
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    t0 = time.time()
    P, hist = train(Xtr, ytr, [Xtr.shape[1], *HIDDEN, C])
    secs = time.time() - t0
    pred = predict(P, Xte)
    cm, prec, rec, f1 = metrics(yte, pred)
    results[name] = dict(features=Xtr.shape[1], train_acc=(predict(P, Xtr) == ytr).mean(),
                         test_acc=(pred == yte).mean(), secs=secs, cm=cm,
                         prec=prec, rec=rec, f1=f1)
    curves[name] = hist
    if kind == "pca":                                # keep the trained network for the web demo
        np.savez(OUT / "model_pca.npz", mu=mu, sd=sd,
                 **{f"W{i}": W for i, (W, _) in enumerate(P)},
                 **{f"b{i}": b for i, (_, b) in enumerate(P)})

print("\n==== With vs without PCA ====")
print(f"{'setup':28s}{'features':>10s}{'train acc':>11s}{'test acc':>10s}{'time(s)':>9s}")
for n, r in results.items():
    print(f"{n:28s}{r['features']:>10d}{r['train_acc']:>11.2f}{r['test_acc']:>10.2f}{r['secs']:>9.1f}")

r = results["with PCA"]
print("\n==== Per-class results (with PCA, test set) ====")
print(f"{'class':16s}{'precision':>10s}{'recall':>8s}{'F1':>6s}")
for i, c in enumerate(CLASSES):
    print(f"{c:16s}{r['prec'][i]:>10.2f}{r['rec'][i]:>8.2f}{r['f1'][i]:>6.2f}")
print("Confusion matrix (rows = true, columns = predicted):\n", r["cm"])

# ----------------------------- plots -----------------------------
plt.figure(figsize=(5.5, 3.8))
for n, h in curves.items():
    plt.plot(h, label=n)
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend()
plt.title("Gradient descent: loss vs epochs"); plt.grid(alpha=0.3)
plt.savefig(OUT / "loss_curve.png", dpi=120, bbox_inches="tight"); plt.close()

plt.figure(figsize=(4.3, 3.8))
plt.imshow(r["cm"], cmap="Blues")
for i in range(C):
    for j in range(C):
        plt.text(j, i, r["cm"][i, j], ha="center", va="center")
plt.xticks(range(C), CLASSES, rotation=20); plt.yticks(range(C), CLASSES)
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion matrix (test)")
plt.savefig(OUT / "confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close()
print(f"\nSaved plots in {OUT}/")
