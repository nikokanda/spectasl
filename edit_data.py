"""
Fix your recorded data without touching the CSV by hand.

Run from the project folder:
    python edit_data.py list            show how many samples each label has
    python edit_data.py delete H        delete ALL samples labeled H
    python edit_data.py last 150        delete the last 150 samples (one burst)

A backup is saved as data/landmarks.csv.bak before anything is changed, and
you're asked to confirm. After editing, run train.py again.
"""

import argparse
import csv
import os
import shutil
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "landmarks.csv")


def load(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def save(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def show_counts(rows):
    counts = Counter(r[0] for r in rows)
    print(f"{len(rows)} samples:", dict(sorted(counts.items())))


def confirm(message):
    return input(f"{message} [y/N] ").strip().lower() == "y"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["list", "delete", "last"])
    p.add_argument("value", nargs="?", help="a label (delete) or a number of samples (last)")
    p.add_argument("--csv", default=DEFAULT_CSV, help="path to landmarks.csv")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"No data found at {args.csv}")
    header, rows = load(args.csv)
    show_counts(rows)

    if args.command == "list":
        return

    if args.command == "delete":
        if not args.value:
            raise SystemExit("Say which label, e.g.  python edit_data.py delete H")
        label = args.value.upper() if args.value.lower() != "none" else "NONE"
        keep = [r for r in rows if r[0] != label]
        removed = len(rows) - len(keep)
        if removed == 0:
            raise SystemExit(f"No samples labeled {label}.")
        question = f"Delete {removed} samples labeled {label}?"
    else:  # last N
        try:
            n = int(args.value)
        except (TypeError, ValueError):
            raise SystemExit("Give a number, e.g.  python edit_data.py last 150")
        if n <= 0 or n > len(rows):
            raise SystemExit(f"N must be between 1 and {len(rows)}.")
        labels = Counter(r[0] for r in rows[-n:])
        keep = rows[:-n]
        removed = n
        question = f"Delete the last {n} samples (labels: {dict(labels)})?"

    if not confirm(question):
        print("Cancelled. Nothing changed.")
        return

    shutil.copyfile(args.csv, args.csv + ".bak")
    save(args.csv, header, keep)
    print(f"Deleted {removed} samples. Backup saved as {args.csv}.bak")
    show_counts(keep)
    print("Now run train.py again.")


if __name__ == "__main__":
    main()
