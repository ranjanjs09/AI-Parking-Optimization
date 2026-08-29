"""
detect_occupancy.py
---------------------
STEP 3 of the pipeline (this is the "live" module).

Ye script:
    1. spots.json (mark_spots.py se banaya hua) load karta hai
    2. trained model.pth (train_model.py se banaya hua) load karta hai
    3. Ek image ya video frame lekar har spot ko crop karta hai
    4. Har crop ko model ko dekar "empty" ya "occupied" predict karta hai
    5. Result ko:
         a) JSON me print/save karta hai   -> {"spot_1": "occupied", ...}
         b) Image pe green/red box draw karke overlay dikhata hai

HOW TO USE (single image):
    python detect_occupancy.py --mode image --source test.jpg \
        --spots spots.json --model best_model.pth --output result.jpg

HOW TO USE (video / webcam / RTSP CCTV stream):
    python detect_occupancy.py --mode video --source parking_feed.mp4 \
        --spots spots.json --model best_model.pth --output result.mp4

    (For a webcam use --source 0, for an RTSP CCTV feed use its rtsp:// URL)

This JSON output is exactly what your Flask/Firebase backend will read and
push to the dashboard in real time (Module 2).
"""

import argparse
import json
import time

import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = ["empty", "occupied"]  # must match the folder order used in training

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


def load_spots(spots_path):
    with open(spots_path, "r") as f:
        return json.load(f)


def predict_spot(model, frame_bgr, box, device):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    crop = frame_bgr[y:y + h, x:x + w]

    if crop.size == 0:
        return "unknown", 0.0

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    return CLASS_NAMES[pred_idx.item()], conf.item()


def annotate_frame(frame, spots, results):
    for spot_id, box in spots.items():
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        status, conf = results[spot_id]
        color = (0, 0, 255) if status == "occupied" else (0, 200, 0)  # red = occupied, green = empty
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{spot_id}: {status}", (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return frame


def run_on_frame(model, frame, spots, device):
    results = {}
    for spot_id, box in spots.items():
        status, conf = predict_spot(model, frame, box, device)
        results[spot_id] = (status, conf)

    json_output = {spot_id: status for spot_id, (status, conf) in results.items()}
    return results, json_output


def process_image(args, model, spots, device):
    frame = cv2.imread(args.source)
    if frame is None:
        raise ValueError("Could not read source image.")

    results, json_output = run_on_frame(model, frame, spots, device)
    print(json.dumps(json_output, indent=2))

    annotated = annotate_frame(frame.copy(), spots, results)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to {args.output}")

    with open(args.output.rsplit(".", 1)[0] + "_status.json", "w") as f:
        json.dump(json_output, f, indent=2)


def process_video(args, model, spots, device):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Could not open video source.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    last_json = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For speed, re-run the CNN only every N frames (occupancy doesn't change every frame anyway)
        if frame_count % args.infer_every == 0:
            results, last_json = run_on_frame(model, frame, spots, device)
            # This is the point where, in the full system, you'd push last_json
            # to Firebase/your backend API for the live dashboard to consume.
            print(f"[frame {frame_count}] {json.dumps(last_json)}")
        else:
            results = {sid: (status, 1.0) for sid, status in last_json.items()}

        annotated = annotate_frame(frame, spots, results)
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--source", required=True, help="Path to image/video, or 0 for webcam, or an rtsp:// URL")
    parser.add_argument("--spots", default="spots.json")
    parser.add_argument("--model", default="best_model.pth")
    parser.add_argument("--output", default="result.jpg")
    parser.add_argument("--infer_every", type=int, default=10,
                         help="Run CNN every N frames for video (speed vs accuracy tradeoff)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    spots = load_spots(args.spots)

    if args.mode == "image":
        process_image(args, model, spots, device)
    else:
        process_video(args, model, spots, device)


if __name__ == "__main__":
    main()
