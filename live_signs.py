"""
Live motion-sign recognizer (whole words). Run after train_signs.py:

    python live_signs.py

Calibrate first (hold your hand(s) steady ~2s). Then just sign. The last 1.5
seconds of movement are checked ~10 times a second; when the same sign is
recognized confidently twice within a short span, it's added to the sentence.
Then there is a 1.5s cooldown so one sign isn't typed twice.

Keys:  ENTER = speak the words and clear     BACKSPACE = delete last word
       c = clear     r = recalibrate     q or ESC = quit
"""

import os
import time

from live_view import LiveView
from terminal_view import event, session
import cv2
from camera import camera_options, open_camera

from common import (HERE, HandLock, draw_hand, frame_features, make_landmarker,
                    to_mp_image)
from live_translate import speak

from motion_reader import MODEL_PATH, load_motion_reader


def main(camera_index: int = 0, quality="native", call_audio=False):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("No signs_model.joblib. Run collect_signs.py, then train_signs.py.")
    reader = load_motion_reader()
    bundle = reader.bundle
    session("Motion signs", camera_index, quality, bundle["classes"])

    cap = open_camera(camera_index, quality)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}. Try camera_index=1 and close other apps using the webcam.")

    view = LiveView("MOTION SIGNS", "Live signs")
    lock = HandLock()
    words = []
    start = time.monotonic()

    with make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                event("Camera", "Feed lost. Session stopped.")
                break
            fh, fw, _ = frame.shape
            now = time.monotonic()

            result = landmarker.detect_for_video(to_mp_image(frame), int((now - start) * 1000))
            hands = result.hand_landmarks
            assign = lock.update(hands, fw, fh, handedness=result.handedness)
            locked = lock.state == "locked"
            mine = set(assign.values())

            for i, lm in enumerate(hands):
                if i in mine and locked:
                    draw_hand(frame, lm)
                elif i in mine:
                    draw_hand(frame, lm, (0, 200, 255), (0, 200, 255))
                else:
                    draw_hand(frame, lm, (140, 140, 140), (140, 140, 140))

            if not locked:
                reader.reset()
                status = "Hold hands steady to calibrate"
                progress = lock.calib_progress
            else:
                feats = frame_features(lock, assign, hands, result.handedness, fw, fh)
                out = reader.update(feats, now)
                if out:
                    words.append(out)
                    event("Signed", " ".join(words))
                status = reader.status
                if lock.lost:
                    status = "Bring hands back into view"
                progress = 0

            cv2.imshow(view.title, view.render(frame, now, status, " ".join(words[-50:]), progress,
                       "Enter speak   T audio test   Backspace delete   C clear   R calibrate   F fullscreen   Q quit"))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("h"), ord("H")):
                view.help_visible = not view.help_visible
            elif key in (ord("f"), ord("F")):
                view.toggle_fullscreen()
            elif key in (ord("t"), ord("T")):
                speak("Howdy. This is the glasses audio test.", call_audio)
            elif key == 13:
                sentence = " ".join(words)
                if sentence:
                    event("Speak", sentence)
                speak(sentence, call_audio)
                words.clear()
            elif key == 8 and words:
                words.pop()
            elif key == ord("c"):
                words.clear()
            elif key == ord("r"):
                lock.start_calibration()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(**camera_options(with_speech=True))
