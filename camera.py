"""Shared camera selection for the OBS preview, capture and recognition tools."""
import argparse
import cv2


CAPTURE_SIZES = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}


def open_camera(index, quality="native"):
    cap = cv2.VideoCapture(index)
    if quality != "native" and cap.isOpened():
        width, height = CAPTURE_SIZES[quality]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Requested mode returned no picture. Retrying native settings.")
            cap.release()
            cap = cv2.VideoCapture(index)
        elif frame.shape[:2] != (height, width):
            print(f"Requested {width}x{height}. Camera delivered {frame.shape[1]}x{frame.shape[0]}.")
    return cap


def camera_options(with_speech=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_index", type=int, nargs="?", default=0)
    parser.add_argument("--quality", choices=("native", *CAPTURE_SIZES), default="native",
                        help="Request a camera size at 30 FPS; native keeps the existing settings")
    if with_speech:
        parser.add_argument("--call-audio", action="store_true",
                            help="Send spoken captions to VB-CABLE for a WhatsApp call")
    options = parser.parse_args()
    if options.camera_index < 0:
        parser.error("Camera number must be zero or greater")
    return vars(options)
