"""Small, event-driven terminal layout for live ASL sessions."""
import os
import sys


def ink(text, style):
    color = (sys.stdout.isatty() and "NO_COLOR" not in os.environ
             and (os.name != "nt" or bool(os.environ.get("WT_SESSION"))))
    return f"\033[{style}m{text}\033[0m" if color else text


def session(mode, camera, quality, classes):
    print()
    print(ink("ASL", "1;36") + "  /  " + ink(mode, "1"))
    print(f"Camera {camera}   |   {quality} capture   |   {len(classes)} labels")
    print(ink("  ".join(map(str, classes)), "2"))
    print()


def event(label, text):
    print(ink(f"{label:<8}", "36") + "  " + text)
