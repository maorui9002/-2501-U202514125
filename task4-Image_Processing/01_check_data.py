from pathlib import Path

from torchvision.datasets import ImageFolder

task_dir = Path(__file__).resolve().parent

train_data = ImageFolder(task_dir / "dataset" / "seg_train")
test_data = ImageFolder(task_dir / "dataset" / "seg_test")

print("类别编号：", train_data.class_to_idx)
print("训练图片数量：", len(train_data))
print("测试图片数量：", len(test_data))

if train_data.class_to_idx != test_data.class_to_idx:
    raise ValueError("训练集与测试集的类别编号不一致")

image, label = train_data[0]
print("第一张图片尺寸：", image.size)
print("第一张图片类别：", train_data.classes[label])