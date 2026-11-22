"""Shared helpers: model download, landmarker setup, drawing, feature normalization."""

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

NUM_FEATURES = 63  # 21 landmarks * (x, y, z)
SIGNING_HAND = "Right"  # "Right" or "Left": which hand you sign letters with


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def make_landmarker(num_hands: int = 4):
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_mp_image(bgr_frame):
    # The saved classifiers were trained with mirrored MediaPipe input.
    bgr_frame = cv2.flip(bgr_frame, 1)
    # Keep the display at capture resolution. Landmarks are normalized to the frame.
    h, w = bgr_frame.shape[:2]
    if w > 960:
        bgr_frame = cv2.resize(bgr_frame, (960, max(1, round(h * 960 / w))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_hand(frame, landmarks, line_color=(0, 200, 0), dot_color=(0, 0, 255)):
    h, w, _ = frame.shape
    pts = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, dot_color, -1)
    return pts


def normalize(landmarks, handedness_label: str):
    """Turn 21 landmarks into 63 numbers that ignore hand position and size.

    - wrist becomes the origin
    - left hands are mirrored so one model handles both hands
    - everything is scaled by wrist -> middle-finger-base distance
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness_label == "Left":
        pts[:, 0] *= -1
    scale = float(np.linalg.norm(pts[9]))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()


# ---- Motion (sequence) features, used by collect_signs.py / train_signs.py / live_signs.py ----
SEQ_T = 16               # frames each sign is resampled to
WINDOW_SECONDS = 2.5     # how long one sign attempt lasts
HAND_BLOCK = 67          # 63 shape numbers + wrist x, wrist y, hand size, "present" flag
SEQ_FEATS = 2 * HAND_BLOCK
PRESENT_IDX = (HAND_BLOCK - 1, 2 * HAND_BLOCK - 1)  # positions of the two "present" flags


def hand_block(landmarks, label, w, h):
    """Shape (63) + where the hand is and how big it looks (4) for one hand."""
    shape = normalize(landmarks, label)
    scale = float(max(w, h))
    size = math.hypot((landmarks[9].x - landmarks[0].x) * w,
                      (landmarks[9].y - landmarks[0].y) * h) / scale
    extra = np.array([landmarks[0].x, landmarks[0].y, size, 1.0], dtype=np.float32)
    return np.concatenate([shape, extra])


def frame_features(lock, assign, hands, handedness, w, h):
    """134 numbers for one video frame: [Right hand block, Left hand block].

    A hand that isn't tracked this frame is all zeros (its present flag is 0).
    """
    blocks = {"Right": np.zeros(HAND_BLOCK, np.float32), "Left": np.zeros(HAND_BLOCK, np.float32)}
    if lock.state == "locked":
        for j, k in assign.items():
            label = lock.tracks[j]["label"] or handedness[k][0].category_name
            if label in blocks:
                blocks[label] = hand_block(hands[k], label, w, h)
    return np.concatenate([blocks["Right"], blocks["Left"]])


def presence_fraction(feats_list):
    """Fraction of frames in which at least one locked hand was visible."""
    if not len(feats_list):
        return 0.0
    return float(np.mean([max(f[PRESENT_IDX[0]], f[PRESENT_IDX[1]]) for f in feats_list]))


def window_motion(feats_list):
    """How much the hand(s) moved over a window: position range + depth change + finger movement.

    A hand resting in view scores near 0, so live recognition can ignore it.
    Uses whichever visible hand moved the most.
    """
    F = np.asarray(feats_list, dtype=np.float32)
    best = 0.0
    for off in (0, HAND_BLOCK):
        P = F[F[:, off + HAND_BLOCK - 1] > 0.5]
        if len(P) < 3:
            continue
        travel = float(np.ptp(P[:, off + 63]) + np.ptp(P[:, off + 64]) + np.ptp(P[:, off + 65]))
        fingers = float(np.mean(np.std(P[:, off:off + 63], axis=0)))
        best = max(best, travel + 0.5 * fingers)
    return best


def resample_sequence(times, feats, start, end, T=SEQ_T):
    """Resample irregular (time, features) frames onto T evenly spaced moments."""
    ts = np.asarray(times, dtype=float)
    F = np.asarray(feats, dtype=np.float32)
    targets = np.linspace(start, end, T)
    out = np.empty((T, F.shape[1]), np.float32)
    for d in range(F.shape[1]):
        out[:, d] = np.interp(targets, ts, F[:, d])
    return out.flatten()


class HandLock:
    """Lock onto ONE person's hands (up to two) and ignore every other hand.

    MediaPipe has no notion of identity, so each locked hand is a "track" that
    we follow by position:

    1. Calibration: hold your hand(s) steady in view for `calib_seconds`. Show
       both hands to track both, or just one. If several hands are visible, the
       biggest (closest to the camera) wins, plus a second hand of similar size.
       Calibrate with only your own hands in view for best results. The timer
       resets if a hand moves too far or an extra hand appears.
    2. Locked: each frame, every track claims the detected hand closest to where
       it was last seen (and of similar size). Hands outside the gate are ignored.
    3. If a tracked hand disappears, its gate slowly widens so it can be picked
       up again near where it left. If that fails, recalibrate (SPACE).
    """

    def __init__(self, max_hands=2, calib_seconds=2.0, base_gate=0.25, max_gate=0.6,
                 gate_growth=0.15, size_min=0.45, size_max=2.2, second_hand_min_ratio=0.5):
        self.max_hands = max_hands
        self.calib_seconds = calib_seconds
        self.base_gate = base_gate        # allowed jump per frame, in units of max(w, h)
        self.max_gate = max_gate
        self.gate_growth = gate_growth    # extra gate per second while a hand is lost
        self.size_min = size_min
        self.size_max = size_max
        self.second_hand_min_ratio = second_hand_min_ratio
        self.start_calibration()

    def start_calibration(self):
        self.state = "calibrating"        # calibrating | locked
        self.tracks = []                  # each: center, size, label, lost_since
        self._calib_start = None
        self._calib_best = []             # candidate hands seen in this calibration window

    @property
    def calib_progress(self):
        if self._calib_start is None:
            return 0.0
        return min(1.0, (time.monotonic() - self._calib_start) / self.calib_seconds)

    @property
    def tracked_count(self):
        """How many hands the lock is following (1 or 2 once calibrated)."""
        return len(self.tracks) if self.state == "locked" else len(self._calib_best)

    @property
    def lost(self):
        """True when every locked hand is currently missing."""
        return self.state == "locked" and all(t["lost_since"] is not None for t in self.tracks)

    @staticmethod
    def _describe(landmarks, w, h):
        scale = float(max(w, h))
        idx = (0, 5, 9, 13, 17)  # wrist + knuckles = stable palm center
        cx = sum(landmarks[i].x for i in idx) / len(idx) * w / scale
        cy = sum(landmarks[i].y for i in idx) / len(idx) * h / scale
        size = math.hypot((landmarks[9].x - landmarks[0].x) * w,
                          (landmarks[9].y - landmarks[0].y) * h) / scale
        return (cx, cy), size

    @staticmethod
    def _label(handedness, k):
        if handedness is None or k >= len(handedness):
            return None
        return handedness[k][0].category_name

    def update(self, hand_landmarks, w, h, handedness=None, now=None):
        """Return {track_id: hand_index} for the locked hands visible this frame.

        While calibrating, the returned hands are just the candidates (for drawing);
        `self.state` stays "calibrating" until the lock is established.
        """
        now = time.monotonic() if now is None else now
        desc = [self._describe(lm, w, h) for lm in hand_landmarks]
        if self.state == "calibrating":
            return self._calibrate(desc, handedness, now)
        return self._track(desc, now)

    def _calibrate(self, desc, handedness, now):
        if not desc:
            self._calib_start = None
            self._calib_best = []
            return {}
        order = sorted(range(len(desc)), key=lambda k: desc[k][1], reverse=True)
        chosen = [order[0]]
        for k in order[1:self.max_hands]:
            if desc[k][1] >= self.second_hand_min_ratio * desc[order[0]][1]:
                chosen.append(k)
        cands = [(desc[k][0], desc[k][1], self._label(handedness, k)) for k in chosen]

        far = bool(self._calib_best) and any(
            min(math.dist(c[0], b[0]) for b in self._calib_best) > self.base_gate for c in cands
        )
        more = len(cands) > len(self._calib_best)
        if self._calib_start is None or far or more:
            self._calib_start = now          # start a fresh window
            self._calib_best = cands
        elif len(cands) == len(self._calib_best):
            self._calib_best = cands         # keep positions fresh; a one-frame dropout won't reset

        if now - self._calib_start >= self.calib_seconds:
            self.tracks = [
                {"center": c, "size": s, "label": lab, "lost_since": None}
                for c, s, lab in self._calib_best
            ]
            self.state = "locked"
        return {j: k for j, k in enumerate(chosen)}

    def _track(self, desc, now):
        gates = []
        for t in self.tracks:
            lost_for = 0.0 if t["lost_since"] is None else now - t["lost_since"]
            gates.append(min(self.max_gate, self.base_gate + self.gate_growth * lost_for))

        pairs = []
        for j, t in enumerate(self.tracks):
            for k, (c, s) in enumerate(desc):
                d = math.dist(c, t["center"])
                ratio = s / max(t["size"], 1e-6)
                if d <= gates[j] and self.size_min <= ratio <= self.size_max:
                    pairs.append((d, j, k))
        pairs.sort()

        used_t, used_h, assign = set(), set(), {}
        for d, j, k in pairs:
            if j in used_t or k in used_h:
                continue
            used_t.add(j)
            used_h.add(k)
            assign[j] = k

        for j, t in enumerate(self.tracks):
            if j in assign:
                c, s = desc[assign[j]]
                t["lost_since"] = None
                t["center"] = (0.5 * t["center"][0] + 0.5 * c[0], 0.5 * t["center"][1] + 0.5 * c[1])
                t["size"] = 0.9 * t["size"] + 0.1 * s
            elif t["lost_since"] is None:
                t["lost_since"] = now
        return assign

    def pick_signing(self, assign, preferred="Right"):
        """Choose which locked hand to read letters from. Returns a hand index or None.

        With one locked hand, that hand is used. With two, the one whose
        Left/Right label matches `preferred` is used (None if it isn't visible).
        """
        if self.state != "locked" or not assign:
            return None
        if len(self.tracks) == 1:
            return next(iter(assign.values()))
        for j, k in assign.items():
            if self.tracks[j]["label"] == preferred:
                return k
        return None
