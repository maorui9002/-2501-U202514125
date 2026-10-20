from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import SceneCNN
from fast_loader import ThreadedDataLoader as DataLoader


def main():
    task_dir = Path(__file__).resolve().parent
    output_dir = task_dir / "outputs" / "cnn_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_num_threads(1)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    test_data = datasets.ImageFolder(
        task_dir / "dataset" / "seg_test",
        transform=transform,
    )

    loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    comparison = []

    for version in ["cnn_baseline", "cnn_augmented"]:
        checkpoint = torch.load(
            task_dir / "outputs" / version / "best.pth",
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["classes"] != test_data.classes:
            raise ValueError("模型与测试集的类别顺序不一致")

        model = SceneCNN(len(test_data.classes)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in loader:
                predictions = model(
                    images.to(device)
                ).argmax(dim=1).cpu()

                true_labels.extend(labels.tolist())
                predicted_labels.extend(predictions.tolist())

        correct = sum(
            true == pred
            for true, pred in zip(true_labels, predicted_labels)
        )
        accuracy = correct / len(test_data)

        comparison.append({
            "version": version,
            "best_epoch": checkpoint["epoch"],
            "val_accuracy": checkpoint["val_acc"],
            "test_accuracy": accuracy,
        })

        # 保存每张图片的预测，后面用于绘图
        pd.DataFrame({
            "image": [path for path, label in test_data.samples],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
        }).to_csv(
            output_dir / f"{version}_predictions.csv",
            index=False,
        )

        print(
            f"{version} | "
            f"最佳轮次：{checkpoint['epoch']} | "
            f"测试准确率：{accuracy:.2%}"
        )

    pd.DataFrame(comparison).to_csv(
        output_dir / "test_comparison.csv",
        index=False,
    )
    print("结果已保存：", output_dir)


if __name__ == "__main__":
    main()