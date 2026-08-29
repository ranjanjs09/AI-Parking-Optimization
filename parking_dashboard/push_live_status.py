"""
push_live_status.py
----------------------
Ye Module 1 (CV model) ko Module 2 (dashboard) se JODTA hai.

Ye script:
    1. Tumhare trained model.pth aur spots.json (Module 1 se) load karta hai
    2. Video/webcam/image ko har N seconds me process karta hai
    3. Result ko dashboard ke /api/status endpoint pe POST kar deta hai

Isse dashboard REAL-TIME me update hota rehta hai, jaise real production
system me hota - CCTV feed -> CV model -> backend -> live dashboard.

HOW TO USE:

Static image ko baar-baar "refresh" karke bhejna (demo ke liye achha hai,
jaise tumhara test.jpg):
    python push_live_status.py --mode image --source test.jpg \
        --spots spots.json --model model.pth --interval 5

Webcam/video ko continuously process karna:
    python push_live_status.py --mode video --source 0 \
        --spots spots.json --model model.pth --interval 5

PEHLE Flask server (app.py) ek alag terminal me chala hua hona chahiye.
"""

import argparse
import json
import time

import cv2
import requests
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = ["empty", "occupied"]
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_spot(model, frame_bgr, box, device):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return "unknown"

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        pred_idx = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()

    return CLASS_NAMES[pred_idx]


def run_inference(model, frame, spots, device):
    return {spot_id: predict_spot(model, frame, box, device) for spot_id, box in spots.items()}


def push_to_dashboard(api_url, status_dict):
    try:
        res = requests.post(api_url, json=status_dict, timeout=3)
        if res.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] Pushed: {status_dict}")
        else:
            print(f"Dashboard returned status {res.status_code}: {res.text}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not reach dashboard. Is 'python app.py' running in another terminal?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--source", required=True, help="Image/video path, 0 for webcam, or rtsp:// URL")
    parser.add_argument("--spots", default="spots.json")
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--api_url", default="http://localhost:5001/api/status")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between each push")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    with open(args.spots, "r") as f:
        spots = json.load(f)

    print(f"Pushing updates to {args.api_url} every {args.interval}s. Press Ctrl+C to stop.")

    if args.mode == "image":
        # Same static image ko baar baar re-process karke "live feed" simulate karta hai
        # (demo/testing ke liye - real deployment me --mode video use hoga)
        while True:
            frame = cv2.imread(args.source)
            if frame is None:
                raise ValueError("Could not read source image.")
            status = run_inference(model, frame, spots, device)
            push_to_dashboard(args.api_url, status)
            time.sleep(args.interval)

    else:  # video
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError("Could not open video source.")

        last_push = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # video khatam ho gaya to loop se dobara shuru karo (demo ke liye)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            now = time.time()
            if now - last_push >= args.interval:
                status = run_inference(model, frame, spots, device)
                push_to_dashboard(args.api_url, status)
                last_push = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
