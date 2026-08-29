"""
mark_spots.py
--------------
STEP 1 of the pipeline.

Ye script tumhe apni parking lot ki EK reference image (ya video ka pehla
frame) pe mouse se click karke har parking spot ko rectangle se mark karne
deta hai. Output ek "spots.json" file hai jisme har spot ki ID aur
coordinates (x, y, w, h) save ho jaate hain.

Ye coordinates baad me detect_occupancy.py use karega taaki wo har spot ko
crop karke CNN model ko de sake.

HOW TO USE:
    python mark_spots.py --image reference.jpg --output spots.json

CONTROLS:
    - Left click + drag  -> ek rectangle draw karo (ek parking spot)
    - 'z'                -> last drawn box undo karo
    - 's'                -> spots.json save karo
    - 'q'                -> save karke exit karo
"""

import cv2
import json
import argparse
import os

drawing = False
ix, iy = -1, -1
boxes = []          # list of (x, y, w, h)
img = None
img_display = None


def redraw():
    """Redraw the image with all boxes + their spot numbers."""
    global img_display
    img_display = img.copy()
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_display, f"spot_{idx+1}", (x, max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            redraw()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 200, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = min(ix, x), min(iy, y)
        w, h = abs(x - ix), abs(y - iy)
        if w > 5 and h > 5:  # ignore accidental tiny clicks
            boxes.append((x0, y0, w, h))
        redraw()


def main():
    global img, img_display

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to reference image of the parking lot")
    parser.add_argument("--output", default="spots.json", help="Where to save the marked spot coordinates")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("Could not read image. Check the file format/path.")

    redraw()
    cv2.namedWindow("Mark Parking Spots", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Mark Parking Spots", 100, 100)
    cv2.setMouseCallback("Mark Parking Spots", mouse_callback)

    print("Instructions: click+drag to draw a box around each parking spot.")
    print("Press 'z' to undo, 's' to save, 'q' to save and quit.")

    # macOS-specific fix: OpenCV window kabhi kabhi Dock me ban jaati hai lekin
    # focus/render nahi hoti. Ye force karta hai Python app ko foreground me laane ke liye.
    import sys
    import subprocess
    if sys.platform == "darwin":
        try:
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to set frontmost of '
                'every process whose unix id is {} to true'.format(os.getpid())
            ], timeout=2, capture_output=True)
        except Exception:
            pass  # agar ye fail ho to bhi window kaam karegi, bas focus manually lana padega

    while True:
        cv2.imshow("Mark Parking Spots", img_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('z') and boxes:
            boxes.pop()
            redraw()

        elif key == ord('s'):
            save_spots(args.output)

        elif key == ord('q'):
            save_spots(args.output)
            break

    cv2.destroyAllWindows()


def save_spots(output_path):
    data = {
        f"spot_{idx+1}": {"x": x, "y": y, "w": w, "h": h}
        for idx, (x, y, w, h) in enumerate(boxes)
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(boxes)} spots to {output_path}")


if __name__ == "__main__":
    main()
