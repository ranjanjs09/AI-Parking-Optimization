"""
train_model.py
----------------
STEP 2 of the pipeline.

Ye script ek CNN (ResNet18, transfer learning) train karta hai jo ek
parking spot ki cropped image dekh kar bata sake: "empty" ya "occupied".

DATASET FORMAT (PKLot / CNRPark style):
    data/
        train/
            empty/       <- empty spot images yahan daalo
            occupied/    <- occupied spot images yahan daalo
        val/
            empty/
            occupied/

Free datasets jo tum use kar sakte ho:
    - PKLot dataset      (~12,000 images, multiple parking lots, weather conditions)
    - CNRPark+EXT dataset (~150,000 images, Italy parking lot camera)
Dono Kaggle / official research pages pe free available hain.

HOW TO USE:
    python train_model.py --data_dir data --epochs 10 --output model.pth

OUTPUT:
    model.pth  -> trained weights, used later by detect_occupancy.py
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(data_dir, batch_size=32, img_size=64):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # helps with day/night, sunny/cloudy variation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    print(f"Classes found: {train_ds.classes}  (should be ['empty', 'occupied'])")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes=2):
    # ResNet18 transfer learning -> chhota dataset ke liye best, fast train hota hai.
    # Kabhi kabhi macOS pe SSL certificate issue ki wajah se pretrained weights
    # download nahi ho paate (common Python.org install problem, code ka bug nahi).
    # Aisa hua to hum bina pretrained weights ke train kar lete hain - synthetic
    # dummy dataset ke liye ye kaafi hai. Real PKLot dataset pe pretrained
    # weights milne se accuracy thodi better hogi, isliye SSL fix README me hai.
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pretrained = True
    except Exception as e:
        print(f"WARNING: Pretrained weights download fail ho gaya ({type(e).__name__}).")
        print("Bina pretrained weights ke train kar rahe hain (thodi kam accuracy aayegi, "
              "lekin pipeline chalegi). SSL fix ke liye README dekho.")
        model = models.resnet18(weights=None)
        pretrained = False

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False  # freeze backbone, sirf last layer train karo (fast + kam data me bhi accurate)
    # pretrained na mile to poora model hi train hoga (freeze nahi karenge)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(model, train_loader, val_loader, device, epochs, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    # sirf un parameters ko train karo jinka requires_grad=True hai
    # (agar backbone frozen hai to sirf fc layer, warna poora model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}/{epochs} | loss: {running_loss/total:.4f} "
              f"| train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f} "
              f"| time: {time.time()-start:.1f}s")

    print(f"Best validation accuracy: {best_acc:.3f}  (saved to best_model.pth)")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset folder (with train/ and val/ subfolders)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)
    model = build_model(num_classes=len(classes))

    train(model, train_loader, val_loader, device, args.epochs)

    # final save (also best_model.pth is saved automatically during training)
    torch.save(model.state_dict(), args.output)
    print(f"Final model saved to {args.output}")


if __name__ == "__main__":
    main()