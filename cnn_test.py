import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def class_order_key(name):
    if not name.startswith("T"):
        return (99, 0, name)
    bits = name[1:]
    if bits.isdigit() and set(bits).issubset({"0", "1"}):
        return (len(bits), int(bits, 2), name)
    return (99, 0, name)


class OrderedImageFolder(datasets.ImageFolder):
    def __init__(self, root, class_names, transform=None):
        self._class_names = list(class_names)
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory):
        classes = [
            d
            for d in self._class_names
            if os.path.isdir(os.path.join(directory, d))
        ]
        if not classes:
            raise FileNotFoundError(f"No class folders found in {directory} for provided class list")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_test_loader(data_root, image_size, batch_size, num_workers, class_names):
    test_dir = os.path.join(data_root, "test")
    test_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_ds = OrderedImageFolder(test_dir, class_names=class_names, transform=test_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, test_ds.classes


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    eval_start = time.perf_counter()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing", unit="batch")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            cur_loss = total_loss / max(total, 1)
            cur_acc = correct / max(total, 1)
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    eval_time = time.perf_counter() - eval_start
    return avg_loss, acc, eval_time


def main():
    parser = argparse.ArgumentParser(description="Deeper CNN test demo")
    parser.add_argument("--data-root", default="./spectrogram_dataset", help="Dataset root with test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--checkpoint", default="./baseline_cnn_deeper.pth", help="Model checkpoint")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes")
    image_size = checkpoint.get("image_size", args.image_size)

    if not classes:
        raise ValueError("Checkpoint does not contain classes. Please retrain with updated cnn_train.py")
    classes = sorted(classes, key=class_order_key)

    test_loader, ds_classes = build_test_loader(
        args.data_root, image_size, args.batch_size, args.num_workers, classes
    )
    if classes != ds_classes:
        print("[WARN] Some classes in checkpoint are missing in test split folders.")

    model = DeeperCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_time = evaluate(model, test_loader, device, criterion)

    print(f"[INFO] Classes({len(classes)}): {classes}")
    print(f"Test loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"[TIME] Total test time: {format_seconds(test_time)}")


if __name__ == "__main__":
    main()
