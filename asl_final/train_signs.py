"""
Train the motion-sign classifier from data/signs.npz.

Run:   python train_signs.py          (needs scikit-learn)

It prints cross-validated accuracy (each example is tested by a model that never
saw it), the per-sign results, and the most confused pairs, then saves
signs_model.joblib for live_signs.py.
"""

import os
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from motion_features import model_inputs
from common import HERE, SEQ_T, WINDOW_SECONDS

NPZ_PATH = os.path.join(HERE, "data", "signs.npz")
MODEL_PATH = os.path.join(HERE, "signs_model.joblib")
AUG_COPIES = 2        # noisy copies of each training example
AUG_NOISE = 0.02
SHIFTS = (-2, -1, 1, 2)   # time shifts in frames (1 frame = WINDOW/16 = ~0.09s)


def make_model():
    return ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1)


def shift_in_time(X, k, pad):
    """Slide every sequence k frames later (k>0) or earlier (k<0).

    Live, a sign is rarely lined up perfectly with the 1.5s window, so we teach the
    model what slightly early/late versions look like. The gap is filled either by
    holding the edge pose ("edge") or with hands absent ("zero").
    """
    n = X.shape[0]
    S = X.reshape(n, SEQ_T, -1)
    out = np.empty_like(S)
    if k > 0:
        out[:, k:] = S[:, :-k]
        out[:, :k] = S[:, :1] if pad == "edge" else 0.0
    else:
        out[:, :k] = S[:, -k:]
        out[:, k:] = S[:, -1:] if pad == "edge" else 0.0
    return out.reshape(n, -1)


def augment(X, y, rng):
    Xs, ys = [X], [y]
    for _ in range(AUG_COPIES):
        Xs.append(X + rng.normal(scale=AUG_NOISE, size=X.shape).astype(np.float32))
        ys.append(y)
    for k in SHIFTS:
        for pad in ("edge", "zero"):
            Xs.append(shift_in_time(X, k, pad))
            ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def main(path=NPZ_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"No data at {path}. Run collect_signs.py first.")
    d = np.load(path)
    X, y = d["X"].astype(np.float32), np.array([str(v) for v in d["y"]])
    counts = Counter(y.tolist())
    print("Examples per label:", dict(sorted(counts.items())))
    if len(counts) < 2:
        raise SystemExit("Need at least 2 labels (record NONE too).")
    smallest = min(counts.values())
    if smallest < 3:
        weak = sorted(k for k, v in counts.items() if v < 3)
        raise SystemExit(f"Record at least 3 examples of {weak} (20+ is much better).")
    if "NONE" not in counts:
        print("Warning: no NONE examples. The model will name a sign for any movement.")

    rng = np.random.default_rng(0)
    folds = min(5, smallest)
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(X, y):
        Xa, ya = augment(X[tr], y[tr], rng)
        pred[te] = make_model().fit(model_inputs(Xa, SEQ_T), ya).predict(model_inputs(X[te], SEQ_T))

    print(f"\nCross-validated accuracy ({folds}-fold): {float((pred == y).mean()):.1%}\n")
    print(classification_report(y, pred, zero_division=0))
    confused = Counter((t, p) for t, p in zip(y.tolist(), pred.tolist()) if t != p).most_common(8)
    if confused:
        print("Most confused (true -> predicted : count):")
        for (t, p), n in confused:
            print(f"  {t} -> {p} : {n}")
    else:
        print("No mistakes in cross-validation.")

    Xa, ya = augment(X, y, rng)
    final = make_model().fit(model_inputs(Xa, SEQ_T), ya)
    final.n_jobs = 1  # Avoid worker scheduling overhead for single live predictions.
    joblib.dump({"model": final, "classes": [str(c) for c in final.classes_],
                 "motion_deltas": True, "T": SEQ_T, "window": WINDOW_SECONDS}, MODEL_PATH)
    print(f"\nTrained on all {len(y)} examples. Saved {MODEL_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else NPZ_PATH)
