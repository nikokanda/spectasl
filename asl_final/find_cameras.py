"""
List which camera numbers work and show each one briefly, so you can tell which is which
(laptop webcam, OBS Virtual Camera, a Pi stream, ...).

Run:   python find_cameras.py
Each camera is shown for ~2.5 seconds. Press any key to skip to the next one.
Then run any script with the number you want, for example:   python live_translate.py 1
"""

import time

import cv2

from camera import open_camera


def main():
    found = []
    for i in range(8):
        cap = open_camera(i)
        if not cap.isOpened():
            print(f"camera {i}: not available")
            continue
        frame = None
        for _ in range(8):                      # let the camera warm up
            ok, frame = cap.read()
        if not ok or frame is None:
            print(f"camera {i}: opens but gives no picture (is the source running?)")
            cap.release()
            continue
        h, w = frame.shape[:2]
        print(f"camera {i}: {w}x{h}")
        found.append(i)
        start = time.time()
        while time.time() - start < 2.5:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"camera {i}  {w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 4)
            cv2.putText(frame, f"camera {i}  {w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)
            cv2.imshow("find_cameras (any key = next)", frame)
            if cv2.waitKey(30) != -1:
                break
        cap.release()
    cv2.destroyAllWindows()
    print("Working cameras:", found or "none found")


if __name__ == "__main__":
    main()
