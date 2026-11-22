"""Keep the bundled fingerspelling vocabulary available across retraining."""
from pathlib import Path
import os
import tempfile

import joblib
import numpy as np


def select_model(active):
    active = Path(active)
    backup = active.parent / "recovery" / "original_letters.joblib"
    original = joblib.load(backup) if backup.is_file() else None
    try:
        current = joblib.load(active)
        labels = set(current["classes"])
        if original is None or set(original["classes"]) <= labels:
            return current
        reason = "missing letters: " + ", ".join(sorted(set(original["classes"]) - labels))
    except Exception as exc:
        if original is None:
            raise
        reason = str(exc)
    print(f"Using original letter model ({reason}).")
    return original


def save_training(bundle, active):
    active = Path(active)
    backup = active.parent / "recovery" / "original_letters.joblib"
    previous = set()
    for path in (active, backup):
        if path.exists():
            previous.update(joblib.load(path)["classes"])
    missing = previous - set(bundle["classes"])
    destination = active.with_name("letters_candidate.joblib") if missing else active
    with tempfile.NamedTemporaryFile(dir=active.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        joblib.dump(bundle, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    if missing:
        print("Kept active model. Training examples are missing:", ", ".join(sorted(missing)))
    return destination


class ExtendedLetters:
    """Recognize newly recorded hand shapes locally around their examples."""

    def __init__(self, previous, X, y):
        model = previous["model"]
        if isinstance(model, ExtendedLetters):
            self.base = model.base
            self.base_classes = model.base_classes
            self.examples = dict(model.examples)
        else:
            self.base = model
            self.base_classes = list(previous["classes"])
            self.examples = {}

        for label in sorted(set(y)):
            rows = X[y == label]
            if label in self.examples:
                rows = np.vstack((self.examples[label], rows))
            self.examples[label] = np.unique(rows, axis=0)
        self.classes_ = sorted(set(self.base_classes) | self.examples.keys())
        self.base_columns = [self.classes_.index(c) for c in self.base_classes]

    def predict_proba(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        probs = np.zeros((len(rows), len(self.classes_)))
        probs[:, self.base_columns] = self.base.predict_proba(rows)
        for row, votes in zip(rows, probs):
            # RMS distance uses the same wrist-relative units as the saved data.
            distances = []
            for label, examples in self.examples.items():
                squared = np.mean((examples - row) ** 2, axis=1)
                k = min(3, len(squared))
                nearest = np.partition(squared, k - 1)[:k]
                distances.append((float(np.sqrt(nearest.mean())), label))
            distances.sort()
            distance, label = distances[0]
            # Fade to the original prediction for unfamiliar or ambiguous poses.
            weight = float(np.clip(1 - (distance / .20) ** 2, 0, 1))
            if len(distances) > 1:
                weight *= min(1, (distances[1][0] - distance) / .04)
            votes *= 1 - weight
            votes[self.classes_.index(label)] += weight
        return probs
