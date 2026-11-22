"""
Step 4: live fingerspelling translator.

Run from a normal terminal (after train.py has created model.joblib):
    python live_translate.py
Optional speech:   pip install pyttsx3

On start it CALIBRATES (hold your hand(s) steady ~2s), then reads letters from
your signing hand (SIGNING_HAND in common.py) and builds text.

How letters are committed:
  - A letter must be held steady for HOLD_SECONDS before it is typed.
  - To type the SAME letter twice (like the L's in HELLO), drop your hand or
    relax it (NONE) for a moment between them.
  - Press Space between words. Pauses leave the current word open.

Keys:  ENTER = speak the text and clear it     SPACE = add a space
       BACKSPACE = delete last character       c = clear
       r = recalibrate                         q or ESC = quit
"""

import os
import threading
import time
from collections import Counter, deque

from live_view import LiveView
from motion_reader import load_motion_reader
from common import frame_features
from terminal_view import event, session
import cv2
from camera import camera_options, open_camera
from letter_models import select_model

from common import HERE, SIGNING_HAND, HandLock, draw_hand, make_landmarker, normalize, to_mp_image

MODEL_PATH = os.path.join(HERE, "model.joblib")

CONF_MIN = 0.60            # ignore predictions less confident than this
WINDOW = 3                 # recent frames considered when smoothing
MIN_VOTES = 2              # frames in the window that must agree
HOLD_SECONDS = 0.20         # how long a letter must stay steady to be typed
RELEASE_SECONDS = 0.25     # gap needed before the same letter can repeat
MAX_TEXT = 200


class LetterCommitter:
    """Turns noisy per-frame predictions into typed characters."""

    def __init__(self, window=WINDOW, min_votes=MIN_VOTES, hold_seconds=HOLD_SECONDS,
                 release_seconds=RELEASE_SECONDS):
        self.votes = deque(maxlen=window)
        self.min_votes = min_votes
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.reset()

    def reset(self):
        self.votes.clear()
        self.hold_label = None
        self.hold_start = 0.0
        self.committed_this_hold = False
        self.last_committed = None
        self.armed = True            # True once the hand has been released since the last letter
        self.none_since = None
        self.stable = None

    def update(self, label, now):
        """`label` is a letter, or None for no hand / NONE / low confidence.

        Returns a character to append ("A", "B", ...) or None.
        """
        self.votes.append(label)
        counts = Counter(v for v in self.votes if v is not None)
        self.stable = None
        if counts:
            top, n = counts.most_common(1)[0]
            if n >= self.min_votes:
                self.stable = top

        if self.stable is None:
            if self.none_since is None:
                self.none_since = now
            gap = now - self.none_since
            self.hold_label = None
            self.committed_this_hold = False
            if gap >= self.release_seconds:
                self.armed = True
            return None

        self.none_since = None
        if self.stable != self.hold_label:
            self.hold_label = self.stable
            self.hold_start = now
            self.committed_this_hold = False

        if label == self.stable and not self.committed_this_hold and now - self.hold_start >= self.hold_seconds:
            if self.stable == self.last_committed and not self.armed:
                return None          # same letter without a release in between
            self.committed_this_hold = True
            self.last_committed = self.stable
            self.armed = False
            return self.stable
        return None

    def progress(self, now):
        if self.hold_label is None:
            return 0.0
        if self.committed_this_hold:
            return 1.0
        return min(1.0, (now - self.hold_start) / self.hold_seconds)


def predict_letter(bundle, feats):
    """Return (label, confidence) for one 63-number feature vector."""
    probs = bundle["model"].predict_proba([feats])[0]
    i = int(probs.argmax())
    return str(bundle["classes"][i]), float(probs[i])


_speak_lock = threading.Lock()


def configure_speech(engine):
    """Prefer an available English natural voice, then familiar Windows voices."""
    voices = engine.getProperty("voices") or []
    for preferred in ("natural", "zira", "hazel", "david"):
        for voice in voices:
            name = str(voice.name).lower()
            if preferred not in name:
                continue
            if preferred == "natural":
                languages = " ".join(str(lang).lower() for lang in (voice.languages or []))
                if "english" not in name and "en-" not in languages and "en_" not in languages:
                    continue
            engine.setProperty("voice", voice.id)
            engine.setProperty("rate", 155)
            return
    engine.setProperty("rate", 155)


def speak(text, call_audio=False):
    text = " ".join(text.split())
    if not text:
        return

    def run():
        with _speak_lock:
            try:
                if call_audio:
                    from call_audio import speak_to_call
                    speak_to_call(text)
                    return
                import pyttsx3
                engine = pyttsx3.init()
                configure_speech(engine)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:  # missing package, no audio device, etc.
                event("Audio", str(e))

    threading.Thread(target=run, daemon=True).start()


def main(camera_index: int = 0, quality="native", call_audio=False):
    bundle = select_model(MODEL_PATH)
    if hasattr(bundle["model"], "n_jobs"):
        bundle["model"].n_jobs = 1
    session("Fingerspelling", camera_index, quality, bundle["classes"])
    if "NONE" not in bundle["classes"]:
        print("Note: no NONE class in this model. It will guess a letter for any hand shape, "
              "so record a few NONE bursts and retrain for better results.")

    cap = open_camera(camera_index, quality)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera_index}. Try camera_index=1 and close other "
            "apps using the webcam."
        )

    view = LiveView("FINGERSPELLING", "Live translator")
    lock = HandLock()
    committer = LetterCommitter()
    text = ""
    motion = None
    signs_mode = False
    event("Mode", "Letters | Click the camera window and press M to switch to Signs.")
    notice_until = 0.0
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
            sign_idx = lock.pick_signing(assign, SIGNING_HAND)
            mine = set(assign.values())

            for i, lm in enumerate(hands):
                if i == sign_idx:
                    draw_hand(frame, lm)
                elif i in mine and locked:
                    draw_hand(frame, lm, (255, 200, 0), (255, 200, 0))
                elif i in mine:
                    draw_hand(frame, lm, (0, 200, 255), (0, 200, 255))
                else:
                    draw_hand(frame, lm, (140, 140, 140), (140, 140, 140))

            label, conf = None, 0.0
            if not locked:
                committer.reset()
                status = "Hold hands steady to calibrate"
                progress = lock.calib_progress
            elif signs_mode:
                features = frame_features(lock, assign, hands, result.handedness, fw, fh)
                word = motion.update(features, now)
                if word:
                    text = (text + word)[-MAX_TEXT:]
                    event("Signed", text)
                status = "Bring hands back into view" if lock.lost else motion.status
                progress = 0
            else:
                if sign_idx is not None:
                    handed = result.handedness[sign_idx][0].category_name
                    raw, conf = predict_letter(bundle, normalize(hands[sign_idx], handed))
                    if raw.strip() and raw != "NONE" and conf >= CONF_MIN:
                        label = raw
                    shown = raw if conf >= CONF_MIN else f"? ({raw})"
                    status = f"{shown}   {conf:.0%}"
                else:
                    status = f"Show signing hand ({SIGNING_HAND})"

                out = committer.update(label, now)
                if out is not None:
                    text = (text + out)[-MAX_TEXT:]
                    event("Typed", text)

                progress = committer.progress(now)

            status = ("Signs" if signs_mode else "Letters") + "  /  " + status
            if now < notice_until:
                status = "Train signs first: run train_signs.py"
            cv2.imshow(view.title, view.render(frame, now, status, text, progress,
                       "M letters/signs   Enter speak   T audio test   Space word break   Backspace delete   C clear   R calibrate   F fullscreen   Q quit"))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("m"), ord("M")):
                if motion is None:
                    try:
                        motion = load_motion_reader()
                    except FileNotFoundError:
                        event("Signs", "Still in Letters: signs_model.joblib is missing. Run train_signs.py in this project folder.")
                        notice_until = now + 3
                        continue
                signs_mode = not signs_mode
                view.mode = "MOTION SIGNS" if signs_mode else "FINGERSPELLING"
                event("Mode", "Signs" if signs_mode else "Letters")
                notice_until = 0
                committer.reset()
                motion.reset()
            elif key in (ord("h"), ord("H")):
                view.help_visible = not view.help_visible
            elif key in (ord("f"), ord("F")):
                view.toggle_fullscreen()
            elif key in (ord("t"), ord("T")):
                speak("Howdy. This is the glasses audio test.", call_audio)
            elif key == 13:                      # ENTER
                if text.strip():
                    event("Speak", text.strip())
                speak(text, call_audio)
                text = ""
                if motion is not None:
                    motion.reset()
                committer.reset()
            elif key == 32:                      # SPACE
                if text and not text.endswith(" "):
                    text += " "
            elif key == 8:                       # BACKSPACE
                text = text[:-1]
            elif key == ord("c"):
                text = ""
                if motion is not None:
                    motion.reset()
                committer.reset()
            elif key == ord("r"):
                lock.start_calibration()
                if motion is not None:
                    motion.reset()
                committer.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(**camera_options(with_speech=True))
