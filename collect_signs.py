"""
Record MOTION signs (whole words) for training.

Run from a normal terminal:   python collect_signs.py

Calibrate first (hold your hand(s) steady ~2s; show both hands if your signs use both).

Press a number key to record a BATCH of examples of that sign. For each example
you get a three-second countdown, then SIGN NOW appears and you have 2.5 seconds to
perform the whole sign. Repeat it naturally each time, with small differences in
speed and position. Edit SIGNS below to choose your words (up to 9).

  1-9        record a batch of that sign
  0          record a batch of NONE: NOT a sign. Rest, hands down, random waving,
             fidgeting, fingerspelling, mid-gesture. Record lots of this.
  ENTER      start the countdown for the next take
  BACKSPACE  delete the last example
  SPACE      recalibrate
  ESC        cancel the current batch / quit

Examples are saved to data/signs.npz and accumulate across sessions.
Aim for 20-30 examples per sign and 40+ NONE examples.
"""

import os
import time
from collections import Counter

import cv2
from camera import camera_options, open_camera
import numpy as np

from common import (HERE, WINDOW_SECONDS, HandLock, draw_hand, frame_features,
                    make_landmarker, presence_fraction, resample_sequence, to_mp_image)

SIGNS = ["HELLO", "THANKS", "HELP", "YES", "NO", "PLEASE"]   # keys 1-6; edit freely
BATCH = 5              # examples recorded per key press
COUNTDOWN = 3.0        # seconds before each example
MIN_PRESENCE = 0.4     # reject a sign example if hands were visible less than this
DATA_DIR = os.path.join(HERE, "data")
NPZ_PATH = os.path.join(DATA_DIR, "signs.npz")


def load():
    if os.path.exists(NPZ_PATH):
        d = np.load(NPZ_PATH)
        return list(d["X"]), [str(v) for v in d["y"]]
    return [], []


def save(X, y):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(NPZ_PATH, X=np.array(X, dtype=np.float32), y=np.array(y))


def put(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main(camera_index: int = 0, quality="native"):
    X, y = load()
    print(f"Loaded {len(X)} examples." if X else "Starting a new sign dataset.")
    print("Keys:", {str(i + 1): s for i, s in enumerate(SIGNS[:9])}, "| 0 = NONE")

    cap = open_camera(camera_index, quality)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}. Try camera_index=1 and close other apps using the webcam.")

    lock = HandLock()
    mode = "idle"           # idle | countdown | recording
    label = None
    batch_left = 0
    batch_total = 0
    t0 = 0.0
    rec_start = 0.0
    rec_times, rec_feats = [], []
    message, message_until = "", 0.0
    start = time.monotonic()

    with make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed, stopping.")
                break
            fh, fw, _ = frame.shape
            now = time.monotonic()

            result = landmarker.detect_for_video(to_mp_image(frame), int((now - start) * 1000))
            hands = result.hand_landmarks
            assign = lock.update(hands, fw, fh, handedness=result.handedness)
            locked = lock.state == "locked"
            mine = set(assign.values())
            feats = frame_features(lock, assign, hands, result.handedness, fw, fh)

            for i, lm in enumerate(hands):
                if i in mine and locked:
                    draw_hand(frame, lm)
                elif i in mine:
                    draw_hand(frame, lm, (0, 200, 255), (0, 200, 255))
                else:
                    draw_hand(frame, lm, (140, 140, 140), (140, 140, 140))

            if not locked:
                mode, batch_left = "idle", 0
                put(frame, "CALIBRATING: hold your hand(s) steady", (10, 35), 0.7, (0, 255, 255))
                bar = int(300 * lock.calib_progress)
                cv2.rectangle(frame, (10, 50), (310, 68), (80, 80, 80), 1)
                cv2.rectangle(frame, (10, 50), (10 + bar, 68), (0, 255, 255), -1)
            elif mode == "ready":
                put(frame, f"{label}  [{batch_total - batch_left + 1}/{batch_total}]   Enter when ready",
                    (10, 35), 0.7, (0, 255, 255))
            elif mode == "countdown":
                remaining = COUNTDOWN - (now - t0)
                if remaining <= 0:
                    mode, rec_start, rec_times, rec_feats = "recording", now, [], []
                else:
                    put(frame, f"Get ready: {label}  ({remaining:.1f})  [{batch_total - batch_left + 1}/{batch_total}]",
                        (10, 35), 0.8, (0, 255, 255))
            elif mode == "recording":
                rec_times.append(now)
                rec_feats.append(feats)
                frac = min(1.0, (now - rec_start) / WINDOW_SECONDS)
                put(frame, f"SIGN NOW: {label}  [{batch_total - batch_left + 1}/{batch_total}]", (10, 35), 0.9, (0, 0, 255))
                cv2.rectangle(frame, (10, 50), (310, 68), (80, 80, 80), 1)
                cv2.rectangle(frame, (10, 50), (10 + int(300 * frac), 68), (0, 0, 255), -1)
                if now - rec_start >= WINDOW_SECONDS:
                    pres = presence_fraction(rec_feats)
                    if label != "NONE" and pres < MIN_PRESENCE:
                        message, message_until = "Hands not visible enough, try that one again", now + 2
                    else:
                        X.append(resample_sequence(rec_times, rec_feats, rec_start, rec_start + WINDOW_SECONDS))
                        y.append(label)
                        save(X, y)
                        batch_left -= 1
                        print(f"Saved {label} ({y.count(label)} total, {len(y)} overall)")
                    if batch_left > 0:
                        mode = "ready"
                    else:
                        mode = "idle"
            else:
                put(frame, "Press 1-9 (or 0 = NONE) to record a batch", (10, 35), 0.65, (0, 255, 0))

            if now < message_until:
                put(frame, message, (10, 85), 0.6, (0, 165, 255))

            # Key legend and counts
            legend = "  ".join(f"{i + 1}={s}" for i, s in enumerate(SIGNS[:9])) + "  0=NONE"
            put(frame, legend, (10, fh - 45), 0.5, (200, 200, 200), 1)
            counts = Counter(y)
            put(frame, "  ".join(f"{k}:{counts[k]}" for k in sorted(counts)) or "no data yet",
                (10, fh - 15), 0.55, (255, 255, 0), 1)

            cv2.imshow("Collect signs", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                mode, batch_left = "idle", 0
                lock.start_calibration()
            elif key == 27:
                if mode != "idle":
                    mode, batch_left = "idle", 0
                    print("Batch cancelled.")
                else:
                    break
            elif locked and mode == "ready" and key in (10, 13):
                mode, t0 = "countdown", now
            elif locked and mode == "idle" and key == 8:
                if X:
                    X.pop()
                    removed = y.pop()
                    save(X, y) if X else (os.path.exists(NPZ_PATH) and os.remove(NPZ_PATH))
                    print(f"Deleted last example ({removed}). {len(X)} left.")
            elif locked and mode == "idle" and ord("0") <= key <= ord("9"):
                n = key - ord("0")
                if n == 0:
                    label = "NONE"
                elif n <= len(SIGNS):
                    label = SIGNS[n - 1]
                else:
                    label = None
                if label:
                    mode, batch_left, batch_total = "ready", BATCH, BATCH

    if X:
        save(X, y)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(X)} examples to {NPZ_PATH}")


if __name__ == "__main__":
    main(**camera_options())
