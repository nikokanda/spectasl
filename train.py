"""
Step 3: train a letter classifier from data/landmarks.csv.

Run:   python train.py
Needs: pip install scikit-learn

What it does:
  1. Loads your recorded samples.
  2. Holds out ONE whole burst per letter as an honest test. (Frames inside a
     burst are nearly identical, so a random split would score ~100% and tell
     you nothing.)
  3. Trains a random forest, prints accuracy, per-letter results, and the most
     confused letter pairs.
  4. Retrains on ALL data and saves model.joblib for the live translator.
"""

import csv
import os
import sys
from collections import Counter, defaultdict

from pathlib import Path
from letter_models import ExtendedLetters, save_training, select_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "landmarks.csv")
MODEL_PATH = os.path.join(HERE, "model.joblib")
BURST_SIZE = 150  # must match SAMPLES_PER_BURST in collect_data.py


def load(path):
    labels, feats = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
            labels.append(line[0])
            feats.append([float(x) for x in line[1:]])
    return np.array(labels), np.array(feats, dtype=np.float32)


def burst_ids(labels):
    """Group consecutive rows into bursts. Normally every burst is BURST_SIZE rows."""
    if len(labels) % BURST_SIZE == 0:
        return np.arange(len(labels)) // BURST_SIZE
    # Fallback: split on label changes (merges back-to-back bursts of one letter)
    ids, current = [], 0
    for i, lab in enumerate(labels):
        if i and lab != labels[i - 1]:
            current += 1
        ids.append(current)
    return np.array(ids)


def make_model():
    return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)


def main(path=CSV_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"No data found at {path}. Run collect_data.py first.")
    y, X = load(path)
    if X.ndim != 2 or X.shape[1] != 63 or not np.isfinite(X).all():
        raise SystemExit("Expected recorded samples with 63 finite landmark coordinates.")
    groups = burst_ids(y)

    counts = Counter(y.tolist())
    print("Samples per label:", dict(sorted(counts.items())))
    active = Path(MODEL_PATH)
    if active.exists() or (active.parent / "recovery" / "original_letters.joblib").exists():
        previous = select_model(active)
        if set(previous["classes"]) - counts.keys():
            extended = ExtendedLetters(previous, X, y)
            saved = save_training({"model": extended, "classes": extended.classes_}, active)
            print("Combined vocabulary:", " ".join(extended.classes_))
            print(f"Saved {saved}")
            return
    if len(counts) < 2:
        raise SystemExit("Need at least 2 labels (include NONE) to train.")

    # Hold out the last burst of each label that has 2+ bursts
    bursts_by_label = defaultdict(list)
    for g, lab in zip(groups, y):
        if g not in bursts_by_label[lab]:
            bursts_by_label[lab].append(g)
    held = {bs[-1] for lab, bs in bursts_by_label.items() if len(bs) >= 2}
    short = [lab for lab, bs in bursts_by_label.items() if len(bs) < 2]
    if short:
        print(f"Only 1 burst for {sorted(short)}: record at least 2 so they can be tested honestly.")

    test_mask = np.isin(groups, list(held))
    if test_mask.sum() == 0 or (~test_mask).sum() == 0:
        print("Not enough bursts to hold any out, so skipping the honest test.")
    else:
        model = make_model().fit(X[~test_mask], y[~test_mask])
        pred = model.predict(X[test_mask])
        acc = float((pred == y[test_mask]).mean())
        print(f"\nAccuracy on held-out bursts (never seen in training): {acc:.1%}\n")
        print(classification_report(y[test_mask], pred, zero_division=0))

        confused = Counter(
            (str(t), str(p)) for t, p in zip(y[test_mask], pred) if t != p
        ).most_common(8)
        if confused:
            print("Most confused (true -> predicted : count):")
            for (t, p), n in confused:
                print(f"  {t} -> {p} : {n}")
        else:
            print("No mistakes on the held-out bursts.")

    final = make_model().fit(X, y)
    saved = save_training({"model": final, "classes": list(final.classes_)}, MODEL_PATH)
    print(f"\nTrained on all {len(y)} samples. Saved {saved}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_PATH)
