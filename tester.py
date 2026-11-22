"""
Live hand tracking with a one-person lock (needs common.py in the same folder).

Run from a normal terminal:   python hand_tracker.py

On start it CALIBRATES: hold your hand(s) steady in view for 2 seconds. Show
both hands to track both, or just one. After that it only follows your hands.
Other people's hands show up in gray and are ignored.

Keys:  SPACE = recalibrate     p = print one locked hand's 21 landmarks     q = quit
"""

import time

import cv2
from camera import camera_options, open_camera

from common import HandLock, draw_hand, make_landmarker, to_mp_image


def put(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main(camera_index: int = 0, quality="native"):
    cap = open_camera(camera_index, quality)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera_index}. Try camera_index=1, "
            "and close other apps that use the webcam (Zoom, Teams, etc.)."
        )

    lock = HandLock()
    start = time.monotonic()
    prev = start

    with make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed, stopping.")
                break
            h, w, _ = frame.shape

            ts_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(to_mp_image(frame), ts_ms)
            hands = result.hand_landmarks
            assign = lock.update(hands, w, h, handedness=result.handedness)
            mine = set(assign.values())
            locked = lock.state == "locked"

            # Locked hands green, calibration candidates yellow, everything else gray
            for i, lm in enumerate(hands):
                if i in mine and locked:
                    pts = draw_hand(frame, lm)
                    label = result.handedness[i][0].category_name
                    put(frame, f"LOCKED ({label})", (pts[0][0], pts[0][1] + 25), 0.6, (0, 255, 0))
                elif i in mine:
                    draw_hand(frame, lm, (0, 200, 255), (0, 200, 255))
                else:
                    draw_hand(frame, lm, (140, 140, 140), (140, 140, 140))

            if not locked:
                put(frame, "CALIBRATING: hold your hand(s) steady (show both to track both)",
                    (10, 65), 0.6, (0, 255, 255))
                bar = int(300 * lock.calib_progress)
                cv2.rectangle(frame, (10, 80), (310, 98), (80, 80, 80), 1)
                cv2.rectangle(frame, (10, 80), (10 + bar, 98), (0, 255, 255), -1)
                put(frame, f"Seeing {lock.tracked_count} hand(s)", (320, 95), 0.55, (0, 255, 255), 1)
            elif lock.lost:
                put(frame, "Locked hands lost (SPACE to recalibrate)", (10, 65), 0.65, (0, 165, 255))
            else:
                put(frame, f"Tracking {len(assign)} of {lock.tracked_count} locked hand(s)",
                    (10, 65), 0.6, (0, 255, 0), 1)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            put(frame, f"FPS: {fps:.0f}", (10, 30), 0.8, (255, 255, 0))

            cv2.imshow("Hand tracker (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:  # SPACE
                lock.start_calibration()
            if key == ord("p") and locked and assign:
                k = next(iter(assign.values()))
                for i, lm in enumerate(hands[k]):
                    print(f"{i:2d}: x={lm.x:.3f} y={lm.y:.3f} z={lm.z:.3f}")
                print("-" * 30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(**camera_options())