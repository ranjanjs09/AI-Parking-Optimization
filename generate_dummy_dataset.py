"""
generate_dummy_dataset.py
---------------------------
YE OPTIONAL/BONUS SCRIPT HAI - sirf pipeline TEST karne ke liye.

Real PKLot dataset download karne me time lagta hai (Kaggle/Roboflow account
chahiye). Isliye ye script FAKE (synthetic) parking spot images bana deta hai
taaki tum turant train_model.py, mark_spots.py, aur detect_occupancy.py chala
ke dekh sako ki pura pipeline chal raha hai ya nahi.

FAKE images kaise banti hain:
    - "empty" spot   = gray/asphalt background + white parking lines
    - "occupied" spot = same background + ek colored rectangle (car ka
      substitute) upar draw kiya hua, random color/position ke saath

NOTE: Ye REAL accuracy dikhane ke liye NAHI hai (real cars jaisa nahi dikhta).
Iska use sirf ye confirm karne ke liye hai ki code sahi chal raha hai.
Final report/demo ke liye REAL PKLot dataset use karna (neeche instructions
README me hain).

HOW TO USE:
    python generate_dummy_dataset.py --out_dir data --train_per_class 100 --val_per_class 20

Isse ye structure ban jaayega (train_model.py isi format ko expect karta hai):
    data/
        train/
            empty/      (100 images)
            occupied/   (100 images)
        val/
            empty/      (20 images)
            occupied/   (20 images)

Ye script ek bada "reference.jpg" bhi banata hai (ek poori parking lot ka
top-view, 12 spots ke saath, kuch occupied kuch empty) - isko mark_spots.py
aur detect_occupancy.py test karne ke liye use karo.
"""

import argparse
import os
import random
from PIL import Image, ImageDraw


def make_asphalt_background(w, h, base_gray=90):
    """Thoda noisy gray background banata hai, jaise asphalt/road texture."""
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            noise = random.randint(-12, 12)
            g = max(0, min(255, base_gray + noise))
            pixels[x, y] = (g, g, g)
    return img


def make_empty_spot(size=64):
    img = make_asphalt_background(size, size)
    draw = ImageDraw.Draw(img)
    # White parking-line borders (left and right), jaise real parking bay marking
    draw.line([(2, 0), (2, size)], fill=(230, 230, 230), width=2)
    draw.line([(size - 2, 0), (size - 2, size)], fill=(230, 230, 230), width=2)
    return img


def make_occupied_spot(size=64):
    img = make_empty_spot(size)
    draw = ImageDraw.Draw(img)

    # Ek "car" jaisa colored rectangle + windshield stripe, random color/position
    car_colors = [
        (180, 30, 30), (30, 60, 160), (200, 200, 60),
        (40, 40, 40), (230, 230, 230), (30, 130, 90),
    ]
    color = random.choice(car_colors)

    margin = random.randint(4, 8)
    top = random.randint(6, 12)
    bottom = size - random.randint(6, 12)

    draw.rounded_rectangle(
        [(margin, top), (size - margin, bottom)],
        radius=8, fill=color, outline=(15, 15, 15)
    )
    # Windshield-jaisi halki stripe (thoda realism ke liye)
    draw.rectangle(
        [(margin + 6, top + 6), (size - margin - 6, top + 16)],
        fill=(150, 190, 210)
    )
    return img


def generate_split(out_dir, split_name, per_class, size=64):
    for label, gen_fn in [("empty", make_empty_spot), ("occupied", make_occupied_spot)]:
        folder = os.path.join(out_dir, split_name, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(per_class):
            img = gen_fn(size)
            img.save(os.path.join(folder, f"{label}_{i:04d}.jpg"))
    print(f"  {split_name}: {per_class} empty + {per_class} occupied images created")


def generate_reference_image(path, rows=2, cols=6, spot_size=100, gap=10):
    """
    Ek poori 'parking lot' image banata hai (grid of spots) jisse mark_spots.py
    aur detect_occupancy.py test kiya ja sake, jaise real CCTV top-view.
    """
    w = cols * (spot_size + gap) + gap
    h = rows * (spot_size + gap) + gap
    canvas = make_asphalt_background(w, h, base_gray=100)

    occupied_flags = []
    for r in range(rows):
        for c in range(cols):
            x = gap + c * (spot_size + gap)
            y = gap + r * (spot_size + gap)
            is_occupied = random.random() < 0.5
            occupied_flags.append(is_occupied)

            spot_img = make_occupied_spot(spot_size) if is_occupied else make_empty_spot(spot_size)
            canvas.paste(spot_img, (x, y))

    canvas.save(path)
    return occupied_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--train_per_class", type=int, default=100)
    parser.add_argument("--val_per_class", type=int, default=20)
    parser.add_argument("--reference_out", default="reference.jpg")
    parser.add_argument("--test_out", default="test.jpg")
    args = parser.parse_args()

    random.seed(42)

    print("Generating synthetic training data...")
    generate_split(args.out_dir, "train", args.train_per_class)
    generate_split(args.out_dir, "val", args.val_per_class)

    print("Generating a fake 'parking lot' reference image (for mark_spots.py)...")
    generate_reference_image(args.reference_out)
    print(f"  saved to {args.reference_out}")

    print("Generating a second fake image (for detect_occupancy.py testing)...")
    flags = generate_reference_image(args.test_out)
    print(f"  saved to {args.test_out}")
    print(f"  (ground truth for test.jpg, left-to-right/top-to-bottom): {['occupied' if f else 'empty' for f in flags]}")

    print("\nDone! Ab ye chalao:")
    print(f"  python train_model.py --data_dir {args.out_dir} --epochs 5")
    print(f"  python mark_spots.py --image {args.reference_out}")
    print(f"  python detect_occupancy.py --mode image --source {args.test_out} --spots spots.json --model best_model.pth")


if __name__ == "__main__":
    main()
