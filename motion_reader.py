"""Motion recognition on tracked frames from either live application."""
import os
from collections import deque

import joblib

from common import HERE, presence_fraction, resample_sequence, window_motion
from motion_features import model_inputs

MODEL_PATH = os.path.join(HERE, "signs_model.joblib")

CONF_MIN = 0.45
PREDICT_EVERY = 0.10     # seconds between predictions
MIN_PRESENCE = 0.30      # ignore windows where hands were visible less than this
MIN_MOTION = 0.05        # ignore windows where the hands barely moved (a resting hand is not a sign).
                         # Watch the "motion" number on screen: resting should read ~0.00-0.03 and
                         # a real sign well above this. Lower it if real signs get ignored.


class SignCommitter:
    """Commit the first qualifying word and suppress repeats during cooldown."""

    def __init__(self, conf_min=CONF_MIN, cooldown=1.5):
        self.conf_min = conf_min
        self.cooldown = cooldown
        self.last_seen = {}

    def update(self, label, conf, now):
        if not label or label == "NONE" or conf <= self.conf_min:
            return None
        previous = self.last_seen.get(label, float("-inf"))
        self.last_seen[label] = now
        if now - previous >= self.cooldown:
            return label
        return None


def predict_window(bundle, buf, now):
    """Classify the last `window` seconds of frames. Returns (label, confidence)."""
    window = bundle["window"]
    if not buf or buf[0][0] > now - window * 0.9:
        return None, 0.0                       # not enough history yet
    recent = [f for t, f in buf if t >= now - window]
    if presence_fraction(recent) < MIN_PRESENCE:
        return None, 0.0
    if window_motion(recent) < MIN_MOTION:
        return None, 0.0
    x = resample_sequence([t for t, _ in buf], [f for _, f in buf], now - window, now, bundle["T"])
    inputs = model_inputs([x], bundle["T"]) if bundle.get("motion_deltas") else [x]
    probs = bundle["model"].predict_proba(inputs)[0]
    i = int(probs.argmax())
    return str(bundle["classes"][i]), float(probs[i])


class MotionReader:
    def __init__(self, bundle):
        self.bundle = bundle
        self.reset()

    def reset(self):
        self.frames = deque()
        self.committer = SignCommitter(cooldown=self.bundle["window"])
        self.next_predict = 0
        self.status = "Ready to sign"

    def update(self, features, now):
        self.frames.append((now, features))
        while self.frames and self.frames[0][0] < now - self.bundle["window"] - 0.5:
            self.frames.popleft()
        if now < self.next_predict:
            return None
        self.next_predict = now + PREDICT_EVERY
        label, confidence = predict_window(self.bundle, self.frames, now)
        self.status = f"{label}   {confidence:.0%}" if label and label != "NONE" else "Ready to sign"
        return self.committer.update(label, confidence, now)


def load_motion_reader():
    return MotionReader(joblib.load(MODEL_PATH))
