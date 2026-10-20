"""复核现有最佳权重，生成图表、逐图预测、分类指标和中文报告；不重新训练。"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms

from cnn_model import SceneCNN
from fast_loader import ThreadedDataLoader


VERSIONS = ("cnn_baseline", "cnn_augmented")
DISPLAY = {"cnn_baseline": "Baseline CNN", "cnn_augmented": "Augmented CNN"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def savefig(fig, output, name):
    fig.savefig(output / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def classification_metrics(truth, prediction, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    support = cm.sum(1)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(len(classes)), where=cm.sum(0) > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes)), where=precision + recall > 0)
    rows = [{"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(classes)]
    return cm, rows, {"test_accuracy": float(cm.trace() / cm.sum()), "macro_precision": float(precision.mean()),
                      "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
                      "weighted_f1": float(np.average(f1, weights=support)), "correct": int(cm.trace()),
                      "test_count": int(cm.sum())}


def draw_architecture(output):
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set(xlim=(0, 13), ylim=(0, 2.6))
    ax.axis("off")
    blocks = [("RGB input", "3 x 128 x 128"), ("Conv + BN + ReLU\nMaxPool", "32 x 64 x 64"),
              ("Conv + BN + ReLU\nMaxPool", "64 x 32 x 32"), ("Conv + BN + ReLU\nMaxPool", "128 x 16 x 16"),
              ("AdaptiveAvgPool\nFlatten", "128 x 2 x 2 -> 512"), ("FC + ReLU\nDropout(0.3) + FC", "512 -> 128 -> 6")]
    for i, (name, shape) in enumerate(blocks):
        x = 0.12 + i * 2.16
        ax.add_patch(FancyBboxPatch((x, .7), 1.92, 1.3, boxstyle="round,pad=.06", facecolor="#edf3fb", edgecolor="#2a638c"))
        ax.text(x + .96, 1.55, name, ha="center", va="center", fontsize=10)
        ax.text(x + .96, .94, shape, ha="center", fontsize=9)
        if i < 5:
            ax.annotate("", xy=(x + 2.12, 1.35), xytext=(x + 1.96, 1.35), arrowprops={"arrowstyle": "->"})
    ax.set_title("SceneCNN | 159,910 trainable parameters | Conv kernels: 3 x 3", pad=8)
    savefig(fig, output, "01_architecture.png")


def draw_curves(histories, metrics, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    for r, version in enumerate(VERSIONS):
        h = histories[version]
        epoch = metrics[version]["best_epoch"]
        for col, title in enumerate(("Loss", "Train / validation accuracy", "Test accuracy")):
            ax = axes[r, col]
            ax.set_title(f"{DISPLAY[version]}: {title}")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.2)
            ax.axvline(epoch, color="#888888", ls=":", label="Selected by validation")
        axes[r, 0].plot(h.epoch, h.train_loss, label="Train")
        axes[r, 0].plot(h.epoch, h.val_loss, label="Validation")
        axes[r, 0].set_ylabel("Cross entropy")
        axes[r, 1].plot(h.epoch, h.train_acc * 100, label="Train")
        axes[r, 1].plot(h.epoch, h.val_acc * 100, label="Validation")
        axes[r, 1].set_ylabel("Accuracy (%)")
        axes[r, 2].plot(h.epoch, h.test_acc * 100, color="#0a8a83", label="Test (monitor only)")
        axes[r, 2].set_ylabel("Accuracy (%)")
        for ax in axes[r]:
            ax.legend(fontsize=8)
    savefig(fig, output, "02_training_curves.png")


def draw_confusion(matrices, classes, output, normalized=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    vmax = 1 if normalized else max(m.max() for m in matrices.values())
    for ax, version in zip(axes, VERSIONS):
        cm = matrices[version]
        values = cm / cm.sum(1, keepdims=True) if normalized else cm
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
               xlabel="Predicted class", ylabel="True class", title=DISPLAY[version])
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = f"{values[i, j]:.1%}" if normalized else str(values[i, j])
                ax.text(j, i, value, ha="center", va="center", fontsize=9,
                        color="white" if values[i, j] > vmax * .55 else "#17324d")
    fig.colorbar(im, ax=axes, shrink=.8, label="Row proportion" if normalized else "Image count")
    savefig(fig, output, "04_confusion_normalized.png" if normalized else "03_confusion_counts.png")


def draw_examples(data, truth, predictions, confidences, indices, title, output, name):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), layout="constrained")
    fig.suptitle(title, fontsize=14)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        with Image.open(data.samples[int(index)][0]) as image:
            ax.imshow(image.convert("RGB"))
        correct = truth[index] == predictions[index]
        ax.set_title(f"True: {data.classes[truth[index]]}\nPred: {data.classes[predictions[index]]} ({confidences[index]:.1%})",
                     color="#267237" if correct else "#be3939", fontsize=10)
    savefig(fig, output, name)


def write_markdown(output, summary, histories):
    metrics, classes = summary["models"], summary["classes"]
    a, b = (metrics[v] for v in VERSIONS)
    diff = (b["test_accuracy"] - a["test_accuracy"]) * 100
    conclusion = ("本次增强模型准确率高于基础模型，但单次实验不足以证明稳定提升。" if diff > 0 else
                  "本次增强模型未超过基础模型，不能把数据增强写成已验证的性能提升。")
    text = f"""# 图像分类实验报告

## 1. 任务与数据

使用 Intel Image Classification 的六类场景图片：{', '.join(classes)}。训练目录共 {summary['source_train_count']} 张，固定随机种子 42 按 80%/20% 划分为 {summary['train_count']} 张训练图与 {summary['validation_count']} 张验证图；独立测试目录共 {summary['test_count']} 张。两种实验使用相同划分、网络和优化参数。

输入转换为 RGB，缩放至 128×128，像素先除以 255，再按均值 0.5、标准差 0.5 归一化，即 x'=2x-1。类别编号和各集合计数见 dataset_counts.csv。未进行跨目录重复图片审计，因此此处“独立”指目录与训练更新分离。

## 2. 模型结构与原理

![模型结构](01_architecture.png)

三组 Conv2d(3×3, padding=1, bias=False)、BatchNorm2d、ReLU、MaxPool2d(2)，通道数为 3→32→64→128。空间尺寸依次为 128→64→32→16。卷积用共享权重提取局部边缘与纹理；批归一化调整中间特征分布；ReLU 引入非线性；池化降低空间尺寸与计算量。

随后 AdaptiveAvgPool2d((2,2)) 得到 128×2×2 特征，展平为 512 维，经 Linear(512,128)、ReLU、Dropout(0.3)、Linear(128,6) 输出六个 logits。总参数量 {summary['parameter_count']:,}。训练直接将 logits 传入交叉熵，预测取 argmax；图中的置信度来自 softmax，不能解释为已校准的正确概率。

## 3. 训练方案与增强实验

PyTorch {summary['environment']['torch']}，torchvision {summary['environment']['torchvision']}，设备 {summary['environment']['device_name']}。AdamW 学习率 0.001、weight_decay=0.0001，交叉熵损失，batch_size=64，各训练 20 轮。损失按样本数加权汇总。仅以验证准确率最高保存 best.pth；本次复核未重新训练。

增强版仅改变训练变换：RandomResizedCrop(128, scale=(0.8,1.0), ratio=(0.9,1.1))、概率 0.5 的水平翻转、brightness/contrast/saturation 均为 0.15 的 ColorJitter。它们增加位置、尺度和光照变化；过强裁剪也可能损失场景线索。验证与测试始终使用确定性预处理。

Windows 采用同进程四线程、最多四批预读，避免多进程共享内存错误；GPU 模式限制 PyTorch CPU 线程为 1。随机增强由线程执行，固定种子不保证逐位复现。训练准确率是在训练模式、各批更新过程中统计，增强版还使用增强输入，不能与固定模型的验证准确率直接等同。

## 4. 训练曲线

![训练曲线](02_training_curves.png)

曲线分别为训练/验证交叉熵、训练/验证准确率、每轮测试准确率。竖线标记验证集选择的轮次。测试集每轮只用于监测，不参与反向传播或最佳权重选择；由于训练期间查看过测试曲线，不能把它描述为完全盲测。最后一轮不一定等于最佳轮次。

## 5. 当前最佳权重的测试结果

| 模型 | 最佳轮次 | 验证准确率 | 测试准确率 | 正确/总数 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| 基础 CNN | {a['best_epoch']} | {a['val_accuracy']:.2%} | {a['test_accuracy']:.2%} | {a['correct']}/{a['test_count']} | {a['macro_f1']:.4f} |
| 增强 CNN | {b['best_epoch']} | {b['val_accuracy']:.2%} | {b['test_accuracy']:.2%} | {b['correct']}/{b['test_count']} | {b['macro_f1']:.4f} |

测试准确率差（增强减基础）为 {diff:+.2f} 个百分点。{conclusion} 两者只进行了当前这一组训练，未做多随机种子或显著性检验，不宣称统计优势。根据验证集选择，本次建议使用 **{summary['recommended_model']}**。

Accuracy=正确数/总数；每类 Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)。Macro F1 对六类等权平均；weighted F1 按真实样本数加权。若分母为零，代码将相应值定义为零。详见 model_comparison.csv 与各模型 classification_report.csv。

![混淆矩阵计数](03_confusion_counts.png)

![混淆矩阵归一化](04_confusion_normalized.png)

矩阵行是真实类别，列是预测类别。计数版展示错误数量；归一化版每行合计 100%，对角线即该类召回率。

![每类 F1](05_class_f1.png)

"""
    for version in VERSIONS:
        cm = np.array(metrics[version]["confusion_matrix"])
        off = cm.copy()
        np.fill_diagonal(off, 0)
        top = np.argsort(off.ravel())[::-1][:3]
        pairs = [f"{classes[i]}→{classes[j]}：{off[i,j]} 张" for i, j in (divmod(int(k), len(classes)) for k in top) if off[i,j] > 0]
        text += f"\n{DISPLAY[version]} 主要混淆：" + "；".join(pairs) + "。这是根据预测统计得到的现象，单靠混淆矩阵不能确定成因。\n"
    text += """
## 6. 分类结果与错误案例

固定随机种子 2026，从完整测试集随机选取六张，同一批图片分别展示两种模型结果；绿色标题为正确，红色为错误。另从增强版全部错误中固定随机抽取六张用于分析，不把错误案例当作随机整体表现。sample_indices.json 记录抽样索引，逐图 CSV 包含文件、标签及 softmax 置信度。

![基础模型示例](06_baseline_predictions.png)

![增强模型示例](07_augmented_predictions.png)

![增强模型错误示例](08_augmented_errors.png)

相近场景可能共享颜色和纹理，例如冰川与山地、街道与建筑。误判分析应结合具体图片，避免仅凭类别名称断言原因。后续可以只使用验证集比较更温和的裁剪、学习率调度、早停、轻量注意力模块，再对锁定配置进行最终测试。

## 7. ViT 原理理解

Vision Transformer 将图像切分为固定大小的 patch，每个 patch 展平后线性映射成 token，加入位置编码，再由 Transformer Encoder 的多头自注意力与前馈层处理。自注意力根据 Q、K、V 计算 token 间关联，形式为 softmax(QK^T/sqrt(d))V，使不同位置的信息相互交换。原始 ViT 使用可学习的分类 token 汇聚全局信息，再由分类头预测。

与卷积的局部感受野和权重共享相比，自注意力可以直接关联远距离 patch；常规全局注意力的计算/内存成本随 token 数近似二次增长。ViT 的效果依赖数据规模、训练和正则化方案，不能简单保证优于 CNN。本项目按题目要求理解原理，实际实现 CNN 与数据增强对照，没有训练或虚构 ViT 指标。

参考：[Dosovitskiy 等，An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)。数据集参考：[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)。

## 8. 使用与复核

在 VS Code 选择 comsen-answer 环境。已有权重时直接运行 05_finish_classification.py，可重新生成本报告的图表与指标；06_predict_image.py 可预测自己的图片。两者都不启动训练。README.md 给出命令与目录。

summary.json 保存环境、划分、完整混淆矩阵与当前权重/历史 SHA-256；本报告每次以实际载入的 best.pth 重新计算结果，不沿用过往训练的旧测试表。报告 PDF 是同次结果的排版导出；重新训练后需重跑整理脚本并重新导出 PDF。
"""
    (output / "图像分类实验报告.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output or project / "outputs" / "cnn_final_report").resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    data = datasets.ImageFolder(project / "dataset" / "seg_test", transform=transform)
    source = datasets.ImageFolder(project / "dataset" / "seg_train")
    if source.class_to_idx != data.class_to_idx:
        raise ValueError("训练与测试类别编号不一致")
    classes = data.classes
    train_size = int(len(source) * .8)
    split = torch.randperm(len(source), generator=torch.Generator().manual_seed(42)).numpy()
    labels = np.array(source.targets)
    test_labels = np.array(data.targets)
    counts = pd.DataFrame({"class": classes,
                           "train": np.bincount(labels[split[:train_size]], minlength=len(classes)),
                           "validation": np.bincount(labels[split[train_size:]], minlength=len(classes)),
                           "test": np.bincount(test_labels, minlength=len(classes))})
    counts.to_csv(output / "dataset_counts.csv", index=False)
    loader = ThreadedDataLoader(data, batch_size=128, shuffle=False)
    metrics, matrices, histories, predictions, confidence = {}, {}, {}, {}, {}
    for version in VERSIONS:
        print(f"正在复核 {version}：{len(data)} 张图片……", flush=True)
        weight_path = project / "outputs" / version / "best.pth"
        history_path = project / "outputs" / version / "history.csv"
        weight_hash = sha256(weight_path)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        history = pd.read_csv(history_path)
        needed = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "test_acc"}
        if not needed.issubset(history.columns) or not np.isfinite(history[list(needed)].to_numpy()).all():
            raise ValueError(f"{version} 训练记录缺列或包含非有限值")
        selected = history.loc[history.epoch == checkpoint["epoch"]]
        if len(selected) != 1 or not np.isclose(selected.iloc[0].val_acc, checkpoint["val_acc"], atol=1e-8):
            raise ValueError(f"{version} 历史记录与最佳权重不一致")
        if int(history.loc[history.val_acc.idxmax(), "epoch"]) != int(checkpoint["epoch"]):
            raise ValueError(f"{version} 权重并非验证集最佳轮次")
        if checkpoint["classes"] != classes:
            raise ValueError("模型与数据类别不一致")
        model = SceneCNN(len(classes)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        all_prob = []
        truth = []
        with torch.inference_mode():
            for images, batch_labels in loader:
                all_prob.append(model(images.to(device)).softmax(1).cpu().numpy())
                truth.extend(batch_labels.tolist())
        truth = np.array(truth)
        np.testing.assert_array_equal(truth, test_labels)
        prob = np.concatenate(all_prob)
        pred = prob.argmax(1)
        conf = prob.max(1)
        cm, report, score = classification_metrics(truth, pred, classes)
        if abs(score["test_accuracy"] - float(selected.iloc[0].test_acc)) > 1 / len(data) + 1e-8:
            raise ValueError(f"{version} 当前测试成绩与对应轮次记录不同，请核对数据或权重")
        if sha256(weight_path) != weight_hash:
            raise RuntimeError("测试期间权重发生变化，请停止训练后重新整理")
        score.update(best_epoch=int(checkpoint["epoch"]), val_accuracy=float(checkpoint["val_acc"]),
                     weight_sha256=weight_hash, history_sha256=sha256(history_path),
                     confusion_matrix=cm.tolist(), class_report=report)
        metrics[version], matrices[version], histories[version] = score, cm, history
        predictions[version], confidence[version] = pred, conf
        pd.DataFrame(report).to_csv(output / f"{version}_classification_report.csv", index=False)
        frame = pd.DataFrame({"image": [str(Path(path).relative_to(project)).replace('\\', '/') for path, _ in data.samples],
                              "true_label": truth, "predicted_label": pred,
                              "true_class": [classes[i] for i in truth], "predicted_class": [classes[i] for i in pred],
                              "confidence": conf, "correct": truth == pred})
        for i, name in enumerate(classes):
            frame[f"prob_{name}"] = prob[:, i]
        frame.to_csv(output / f"{version}_predictions.csv", index=False)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(output / f"{version}_confusion.csv", index_label="true/pred")
        print(f"{version} | 最佳轮次 {score['best_epoch']} | 测试准确率 {score['test_accuracy']:.2%} | Macro F1 {score['macro_f1']:.4f}", flush=True)
    summary = {"classes": classes, "source_train_count": len(source), "train_count": train_size,
               "validation_count": len(source) - train_size, "test_count": len(data),
               "parameter_count": sum(p.numel() for p in model.parameters()), "split_seed": 42,
               "recommended_model": max(VERSIONS, key=lambda v: metrics[v]["val_accuracy"]), "models": metrics,
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                               "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU"}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = pd.DataFrame([{"version": v, **{k: metrics[v][k] for k in ("best_epoch", "val_accuracy", "test_accuracy", "correct", "test_count", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}} for v in VERSIONS])
    comparison.to_csv(output / "model_comparison.csv", index=False)
    comparison[["version", "best_epoch", "val_accuracy", "test_accuracy"]].to_csv(output / "test_comparison.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_architecture(output)
    draw_curves(histories, metrics, output)
    draw_confusion(matrices, classes, output)
    draw_confusion(matrices, classes, output, normalized=True)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.arange(len(classes))
    for i, version in enumerate(VERSIONS):
        ax.bar(x + (i - .5) * .36, [r["f1"] for r in metrics[version]["class_report"]], .36, label=DISPLAY[version])
    ax.set(xticks=x, xticklabels=classes, ylim=(0, 1), ylabel="F1 score", title="Per-class F1 on the same 3,000 test images")
    ax.legend(loc="lower right")
    savefig(fig, output, "05_class_f1.png")
    rng = np.random.default_rng(2026)
    sample = rng.choice(len(data), size=min(6, len(data)), replace=False)
    for version, filename in zip(VERSIONS, ("06_baseline_predictions.png", "07_augmented_predictions.png")):
        draw_examples(data, test_labels, predictions[version], confidence[version], sample,
                      DISPLAY[version] + " | Same randomly selected test images", output, filename)
    errors = np.flatnonzero(predictions["cnn_augmented"] != test_labels)
    chosen = rng.choice(errors, size=min(6, len(errors)), replace=False)
    draw_examples(data, test_labels, predictions["cnn_augmented"], confidence["cnn_augmented"], chosen,
                  "Augmented CNN | Random sample of errors only", output, "08_augmented_errors.png")
    (output / "sample_indices.json").write_text(json.dumps({"random_seed": 2026, "test_sample_indices": sample.tolist(), "augmented_error_indices": chosen.tolist()}, indent=2), encoding="utf-8")
    write_markdown(output, summary, histories)
    print(f"报告与图表已保存：{output}", flush=True)


if __name__ == "__main__":
    main()
