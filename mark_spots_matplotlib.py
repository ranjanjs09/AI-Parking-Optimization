"""
mark_spots_matplotlib.py
--------------------------
BACKUP/ALTERNATIVE to mark_spots.py.

Agar mark_spots.py (OpenCV wali window) tumhare Mac pe khulti nahi ya focus
nahi hoti, to ye wahi kaam matplotlib ke through karta hai — jo macOS pe
generally zyada reliably window kholta hai.

HOW TO USE:
    python mark_spots_matplotlib.py --image reference.jpg --output spots.json

CONTROLS:
    - Click and drag  -> ek rectangle draw karo (ek parking spot)
    - Rectangle chhod dene ke baad wo automatically list me add ho jaata hai
    - Window band karo (red X / Cmd+W)  -> spots.json automatically save ho jaata hai
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg

boxes = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="spots.json")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = mpimg.imread(args.image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_title(
        "Click + drag to mark each parking spot. Close window when done "
        "(spots.json saves automatically)."
    )

    def on_select(eclick, erelease):
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w > 5 and h > 5:
            idx = len(boxes) + 1
            boxes.append((x, y, w, h))
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
            ax.text(x, max(y - 6, 10), f"spot_{idx}", color="lime", fontsize=9, weight="bold")
            fig.canvas.draw_idle()
            print(f"Marked spot_{idx}: x={x}, y={y}, w={w}, h={h}")

    selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1], minspanx=5, minspany=5,
        spancoords="pixels", interactive=False
    )

    def on_close(event):
        save_spots(args.output)

    fig.canvas.mpl_connect("close_event", on_close)

    plt.tight_layout()
    plt.show()  # window yahan khulti hai; band karne pe on_close chalega


def save_spots(output_path):
    data = {
        f"spot_{idx+1}": {"x": x, "y": y, "w": w, "h": h}
        for idx, (x, y, w, h) in enumerate(boxes)
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(boxes)} spots to {output_path}")


if __name__ == "__main__":
    main()
