import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


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


def class_order_key(name):
    if not name.startswith("T"):
        return (99, 0, name)
    bits = name[1:]
    if bits.isdigit() and set(bits).issubset({"0", "1"}):
        return (len(bits), int(bits, 2), name)
    return (99, 0, name)


def discover_ordered_classes(split_dir):
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    classes = [
        d
        for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and d.startswith("T")
    ]
    classes = sorted(classes, key=class_order_key)
    if not classes:
        raise ValueError(f"No class folders found in: {split_dir}")
    return classes


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


def build_train_loader(data_root, image_size, batch_size, num_workers):
    train_dir = os.path.join(data_root, "train")
    classes = discover_ordered_classes(train_dir)
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    train_ds = OrderedImageFolder(train_dir, class_names=classes, transform=train_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, train_ds.classes


def main():
    parser = argparse.ArgumentParser(description="Deeper CNN training demo")
    parser.add_argument("--data-root", default="./spectrogram_dataset", help="Dataset root with train/test")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--save-path", default="./baseline_cnn_deeper.pth", help="Model path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, classes = build_train_loader(
        args.data_root, args.image_size, args.batch_size, args.num_workers
    )
    print(f"[INFO] Classes({len(classes)}): {classes}")

    model = DeeperCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", unit="batch")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            cur_loss = running_loss / max(total, 1)
            cur_acc = correct / max(total, 1)
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"epoch_time={format_seconds(epoch_time)}"
        )

    total_time = time.perf_counter() - train_start

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "image_size": args.image_size,
        },
        args.save_path,
    )
    print(f"[TIME] Total training time: {format_seconds(total_time)}")
    print(f"[DONE] Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
