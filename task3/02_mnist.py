from pathlib import Path
import io

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

task_dir = Path(__file__).resolve().parent
output_dir = task_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 读取数据
data_path = task_dir / "train-00000-of-00001.parquet"
df = pd.read_parquet(data_path)

print("样本数量：", len(df))
print("数据列名：", df.columns.tolist())
print("各数字的样本数量：")
print(df["label"].value_counts().sort_index())

# 显示前三张图片，确认解码结果
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for index, ax in enumerate(axes):
    row = df.iloc[index]

    image = Image.open(
        io.BytesIO(row["image"]["bytes"])
    ).convert("L")

    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {row['label']}")
    ax.axis("off")

fig.tight_layout()
fig.savefig(output_dir / "mnist_samples.png", dpi=150)
plt.show()
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 解码全部图片
images = []

for item in df["image"]:
    with Image.open(io.BytesIO(item["bytes"])) as image:
        images.append(np.array(image.convert("L")))

# 像素从 0～255 缩放到 0～1，并展平
X = torch.from_numpy(np.stack(images)).float() / 255.0
X = X.reshape(len(df), -1)

# 分类标签必须为整数类型
y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

# 2. 将当前文件划分为训练集和测试集
dataset = TensorDataset(X, y)
train_size = int(len(dataset) * 0.8)

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, len(dataset) - train_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset, batch_size=512, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=512, shuffle=False
)

# 3. 搭建网络：784 → 256 → 128 → 10
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

# 4. 检查输入输出形状
batch_images, batch_labels = next(iter(train_loader))

with torch.no_grad():
    outputs = model(batch_images.to(device))

print("运行设备：", device)
print("训练样本：", len(train_dataset))
print("测试样本：", len(test_dataset))
print("输入形状：", batch_images.shape)
print("输出形状：", outputs.shape)
# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15
train_losses = []
train_accs = []
test_accs = []


# 计算当前模型在指定数据集上的准确率
def evaluate_accuracy(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


print("\n开始训练……")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    # 每轮训练结束后，用同一个模型评价两个数据集
    average_loss = total_loss / len(train_dataset)
    train_acc = evaluate_accuracy(train_loader)
    test_acc = evaluate_accuracy(test_loader)

    train_losses.append(average_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Test Acc: {test_acc:.2%}"
    )

# 保存最后一轮的模型参数
torch.save(model.state_dict(), output_dir / "mnist_mlp.pth")
print("训练完成，模型参数已保存")
# 1. 保存训练曲线
epoch_numbers = range(1, epochs + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

curves = [
    (train_losses, "Training Loss", "Cross Entropy"),
    (train_accs, "Train Accuracy", "Accuracy"),
    (test_accs, "Test Accuracy", "Accuracy"),
]

for ax, (values, title, ylabel) in zip(axes, curves):
    ax.plot(epoch_numbers, values)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    if ylabel == "Accuracy":
        ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(output_dir / "mnist_curves.png", dpi=200)
plt.close(fig)

# 2. 从测试集中随机选择 6 张图片
generator = torch.Generator().manual_seed(123)
selected = torch.randperm(
    len(test_dataset), generator=generator
)[:6].tolist()

examples = [test_dataset[index] for index in selected]
sample_images = torch.stack([image for image, label in examples])
sample_labels = torch.stack([label for image, label in examples])

model.eval()
with torch.no_grad():
    predictions = model(
        sample_images.to(device)
    ).argmax(dim=1).cpu()

# 3. 绘制真实标签与预测标签
fig, axes = plt.subplots(2, 3, figsize=(9, 6))

for index, ax in enumerate(axes.flat):
    true_label = sample_labels[index].item()
    predicted_label = predictions[index].item()
    color = "green" if true_label == predicted_label else "red"

    ax.imshow(sample_images[index].reshape(28, 28).numpy(), cmap="gray")
    ax.set_title(
        f"True: {true_label} | Pred: {predicted_label}",
        color=color,
    )
    ax.axis("off")

fig.tight_layout()
fig.savefig(output_dir / "mnist_predictions.png", dpi=200)
plt.close(fig)

# 4. 保存每一轮的数据，方便写报告
pd.DataFrame({
    "epoch": list(epoch_numbers),
    "train_loss": train_losses,
    "train_accuracy": train_accs,
    "test_accuracy": test_accs,
}).to_csv(output_dir / "mnist_history.csv", index=False)

print("曲线、预测图和训练记录已保存")