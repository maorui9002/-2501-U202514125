from time import perf_counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cnn_model import SceneCNN
from fast_loader import ThreadedDataLoader as DataLoader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    torch.manual_seed(42)
    task_dir = Path(__file__).resolve().parent
    output_dir = task_dir / "outputs" / "cnn_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_num_threads(1)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = datasets.ImageFolder(
        task_dir / "dataset" / "seg_train",
        transform=transform,
    )

    # 从原训练数据中划分验证集
    train_size = int(len(dataset) * 0.8)
    train_data, val_data = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True,
        num_workers=0, persistent_workers=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_data, batch_size=64, shuffle=False,
        num_workers=0, persistent_workers=False,
        pin_memory=False,
    )

    # 每轮记录测试准确率；最佳模型仍只根据验证集选择。
    test_data = datasets.ImageFolder(
        task_dir / "dataset" / "seg_test",
        transform=transform,
    )
    if test_data.class_to_idx != dataset.class_to_idx:
        raise ValueError("训练集与测试集的类别编号不一致")

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        # 使用独立随机数生成器，避免新增评价改变训练的随机数序列。
        generator=torch.Generator().manual_seed(123),
    )

    model = SceneCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )

    epochs = 20
    best_val_acc = -1.0
    history = []

    print("设备：", device)
    print("训练数量：", len(train_data), "验证数量：", len(val_data))
    print("开始训练……", flush=True)

    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        _, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "classes": dataset.classes,
                "epoch": epoch,
                "val_acc": val_acc,
            }, output_dir / "best.pth")

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv", index=False
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Test Acc: {test_acc:.2%} | "
            f"Time: {perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    print(f"\n最佳验证准确率：{best_val_acc:.2%}")
    print("结果目录：", output_dir)


if __name__ == "__main__":
    main()