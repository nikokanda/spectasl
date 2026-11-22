"""Keep captions, controls and measured frame rate outside the camera picture."""
from collections import deque

import cv2


class LiveView:
    def __init__(self, mode, title):
        self.title = title
        self.fullscreen = False
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(title, 960, 640)
        self.toggle_fullscreen()
        self.mode = mode
        self.help_visible = False
        self.times = deque(maxlen=25)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        state = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, state)

    def render(self, frame, now, status, transcript, progress, controls):
        self.times.append(now)
        _, _, width, height = cv2.getWindowImageRect(self.title)
        view = fit_window(frame, width, height)
        h, w = view.shape[:2]
        unit = max(.55, min(1.8, w / 1000, h / 650))
        inset = max(4, round(17 * unit))
        font = cv2.FONT_HERSHEY_DUPLEX
        dark, white, ice = (32, 26, 22), (248, 247, 242), (219, 221, 125)

        def label(text, x, y, size, *, tail=False):
            scale = size * unit
            weight = max(1, round(unit * size))
            available = max(1, w - x - 3 * inset)
            def measure(value):
                return cv2.getTextSize(value, font, scale, weight)[0]
            while text and measure(text)[0] > available:
                text = text[1:] if tail else text[:-1]
            tw, th = measure(text)
            baseline = y + inset // 2 + th
            end = x + tw + inset
            bottom = baseline + inset // 2 + round(4 * unit)
            cv2.rectangle(view, (x, y), (end, bottom), dark, -1)
            cv2.putText(view, text, (x + inset // 2, baseline), font, scale,
                        white, weight, cv2.LINE_AA)
            return end, bottom

        end, bottom = label(status, inset, inset, .61)
        if progress > 0:
            cv2.line(view, (inset, bottom), (inset + round((end-inset)*min(1, progress)), bottom),
                     ice, max(2, round(2 * unit)))
        if transcript:
            label(transcript, inset, h - round(64 * unit), 1.05, tail=True)

        if self.help_visible:
            elapsed = self.times[-1] - self.times[0]
            fps = round((len(self.times) - 1) / elapsed) if elapsed > 0 else 0
            fh, fw = frame.shape[:2]
            rows = [f"{self.mode}   {fw}x{fh}   {fps} FPS", *controls.split("   "), "H   close help"]
            y = bottom + inset // 2
            for row in rows:
                _, y = label(row, inset, y, .48)
                y += 1
        return view


def fit_window(view, width, height):
    """Scale the complete view uniformly, centering it in the drawable area."""
    if width <= 0 or height <= 0:
        return view
    h, w = view.shape[:2]
    factor = min(width / w, height / h)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    if size != (w, h):
        method = cv2.INTER_LINEAR if factor > 1 else cv2.INTER_AREA
        view = cv2.resize(view, size, interpolation=method)
    dx, dy = width - size[0], height - size[1]
    if dx or dy:
        view = cv2.copyMakeBorder(view, dy // 2, dy - dy // 2,
                                  dx // 2, dx - dx // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return view
