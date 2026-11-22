"""
Step 2: record labeled hand-shape samples for training.

Run from a normal terminal:   python collect_data.py

On start it CALIBRATES: hold your hand(s) steady in view for 2 seconds. Show both
hands to track both, or just one. It then follows only your hands, so other
people's hands (shown in gray) are ignored while you record.

Letters are read from ONE hand, set by SIGNING_HAND in common.py. If you calibrated
with only that one hand, it simply uses that hand. Your other locked hand is
drawn in cyan and not recorded.

Controls (click the video window first so it has keyboard focus):
  SPACE      recalibrate (lock onto your hands again)
  a-z        record a burst of samples for that letter (J and Z skipped, see below)
  0          record a burst for NONE (relaxed hand, random non-letter shapes)
  BACKSPACE  undo the last burst
  ESC        during a burst: cancel it.  Otherwise: save and quit

How a burst works: press a key, you get a 1 second countdown, then it records
150 samples while the signing hand is visible. Slowly move/tilt/rotate your hand
a little and change distance during the burst. Variety makes the model better.

J and Z are skipped because they involve motion. A single-frame classifier
can't learn them. We'll handle them later or leave them out of the first demo.

Samples are saved to data/landmarks.csv and accumulate across sessions.
"""

import csv
import os
import time
from collections import Counter

import cv2
from camera import camera_options, open_camera

from common import HERE, NUM_FEATURES, SIGNING_HAND, HandLock, draw_hand, make_landmarker, normalize, to_mp_image

DATA_DIR = os.path.join(HERE, "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
SAMPLES_PER_BURST = 150
COUNTDOWN_SECONDS = 1.0
SKIP_LETTERS = {"j", "z"}


def load_rows():
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for line in reader:
                rows.append((line[0], [float(x) for x in line[1:]]))
    return rows


def save_rows(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        for label, feats in rows:
            writer.writerow([label] + [f"{v:.5f}" for v in feats])


def put(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main(camera_index: int = 0, quality="native"):
    rows = load_rows()
    bursts = []  # how many rows each burst added, for undo
    print(f"Loaded {len(rows)} existing samples from {CSV_PATH}" if rows else "Starting a new dataset.")

    cap = open_camera(camera_index, quality)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera_index}. Try camera_index=1 and close other "
            "apps using the webcam."
        )

    lock = HandLock()
    mode = "idle"  # idle | countdown | recording
    label = None
    t0 = 0.0
    added = 0
    start = time.monotonic()

    with make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed, stopping.")
                break
            fh, fw, _ = frame.shape

            ts_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(to_mp_image(frame), ts_ms)
            hands = result.hand_landmarks
            assign = lock.update(hands, fw, fh, handedness=result.handedness)
            locked = lock.state == "locked"
            sign_idx = lock.pick_signing(assign, SIGNING_HAND)
            have_hand = sign_idx is not None
            mine = set(assign.values())

            for i, lm in enumerate(hands):
                if i == sign_idx:
                    draw_hand(frame, lm)                                    # green: being recorded
                elif i in mine and locked:
                    draw_hand(frame, lm, (255, 200, 0), (255, 200, 0))      # cyan: yours, not recorded
                elif i in mine:
                    draw_hand(frame, lm, (0, 200, 255), (0, 200, 255))      # yellow: calibrating
                else:
                    draw_hand(frame, lm, (140, 140, 140), (140, 140, 140))  # gray: ignored

            now = time.monotonic()
            if not locked:
                mode = "idle"
                put(frame, "CALIBRATING: hold your hand(s) steady (show both to track both)",
                    (10, 35), 0.6, (0, 255, 255))
                bar = int(300 * lock.calib_progress)
                cv2.rectangle(frame, (10, 50), (310, 68), (80, 80, 80), 1)
                cv2.rectangle(frame, (10, 50), (10 + bar, 68), (0, 255, 255), -1)
                put(frame, f"Seeing {lock.tracked_count} hand(s)", (320, 65), 0.55, (0, 255, 255), 1)
            elif mode == "countdown":
                remaining = COUNTDOWN_SECONDS - (now - t0)
                if remaining <= 0:
                    mode = "recording"
                    added = 0
                else:
                    put(frame, f"Get ready: {label}  ({remaining:.1f})", (10, 35), 0.9, (0, 255, 255))
            elif mode == "recording":
                if have_hand:
                    handed = result.handedness[sign_idx][0].category_name
                    feats = normalize(hands[sign_idx], handed)
                    rows.append((label, feats.tolist()))
                    added += 1
                put(
                    frame,
                    f"RECORDING {label}: {added}/{SAMPLES_PER_BURST}"
                    + ("" if have_hand else "  (signing hand not visible!)"),
                    (10, 35), 0.9, (0, 0, 255),
                )
                if added >= SAMPLES_PER_BURST:
                    bursts.append(added)
                    save_rows(rows)
                    print(f"Saved burst for {label}. Total samples: {len(rows)}")
                    mode = "idle"
            else:
                put(frame, "Press a-z (or 0 for NONE) to record. SPACE = recalibrate. ESC = save & quit",
                    (10, 35), 0.55, (0, 255, 0))
                if not have_hand:
                    put(frame, f"Signing hand ({SIGNING_HAND}) not visible (SPACE to recalibrate)",
                        (10, 65), 0.6, (0, 165, 255))

            # Per-label counts, wrapped into rows along the bottom
            counts = Counter(lbl for lbl, _ in rows)
            entries = [f"{k}:{counts[k]}" for k in sorted(counts)]
            per_line = 9
            lines = [entries[i:i + per_line] for i in range(0, len(entries), per_line)]
            for i, chunk in enumerate(lines[-4:]):
                y = fh - 15 - 24 * (len(lines[-4:]) - 1 - i)
                put(frame, "  ".join(chunk), (10, y), 0.55, (255, 255, 0), 1)

            cv2.imshow("Collect data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE = recalibrate
                if mode == "recording" and added:
                    del rows[-added:]
                mode = "idle"
                lock.start_calibration()
            elif not locked:
                if key == 27:
                    break
            elif key == 27:  # ESC
                if mode == "idle":
                    break
                if mode == "recording" and added:
                    del rows[-added:]
                mode = "idle"
                print("Burst cancelled.")
            elif mode == "idle" and key == 8:  # BACKSPACE
                if bursts:
                    n = bursts.pop()
                    del rows[-n:]
                    save_rows(rows)
                    print(f"Undid last burst ({n} samples). Total: {len(rows)}")
            elif mode == "idle" and 0 < key < 128:
                ch = chr(key).lower()
                if ch == "0":
                    label, mode, t0 = "NONE", "countdown", now
                elif ch.isalpha() and ch not in SKIP_LETTERS:
                    label, mode, t0 = ch.upper(), "countdown", now
                elif ch in SKIP_LETTERS:
                    print(f"{ch.upper()} needs motion and is skipped for now.")

    save_rows(rows)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(rows)} samples to {CSV_PATH}")


if __name__ == "__main__":
    main(**camera_options())
