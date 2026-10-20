"""月度客流预测：LSTM、注意力与季节残差改进、独立测试、递归未来预测。"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class Config:
    lookback: int = 24
    hidden_size: int = 32
    epochs: int = 600
    patience: int = 80
    learning_rate: float = 0.004
    weight_decay: float = 0.0001
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    horizon: int = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 此数据集很小，避免大量 CPU 线程与多进程加载的开销。
    torch.set_num_threads(1)


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"Month", "Passengers"}.issubset(frame.columns):
        raise ValueError("CSV 必须包含 Month、Passengers 两列")
    frame = frame[["Month", "Passengers"]].copy()
    frame["Month"] = pd.to_datetime(frame["Month"], format="%Y-%m", errors="raise")
    frame["Passengers"] = pd.to_numeric(frame["Passengers"], errors="raise")
    frame = frame.sort_values("Month").reset_index(drop=True)
    if len(frame) == 0 or frame["Month"].duplicated().any():
        raise ValueError("月份为空或存在重复月份")
    expected = pd.date_range(frame["Month"].iloc[0], periods=len(frame), freq="MS")
    if not expected.equals(pd.DatetimeIndex(frame["Month"])):
        raise ValueError("月份必须连续，且使用每月第一天表示")
    values = frame["Passengers"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Passengers 必须是有限的正数，不能含缺失值")
    return frame


def features(values: np.ndarray, months: pd.DatetimeIndex, kind: str) -> np.ndarray:
    """所有特征只使用该行以及更早的数据，不做居中滚动或向后填充。"""
    logs = np.log(np.asarray(values, dtype=np.float64))
    if kind == "lstm":
        return logs[:, None]
    growth = np.zeros(len(logs), dtype=np.float64)
    available = np.zeros(len(logs), dtype=np.float64)
    growth[12:] = logs[12:] - logs[:-12]
    available[12:] = 1.0
    phase = 2 * np.pi * (months.month.to_numpy() - 1) / 12
    return np.column_stack([logs, growth, np.sin(phase), np.cos(phase), available])


def fit_normalizer(values: np.ndarray) -> dict:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.maximum(np.std(values, axis=0), 1e-6).tolist(),
    }


def normalize(values: np.ndarray, scaler: dict) -> np.ndarray:
    return (values - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])


def prepare(frame: pd.DataFrame, config: Config, kind: str) -> dict:
    n = len(frame)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))
    if config.lookback < 12 or train_end <= config.lookback + 3 or not train_end < val_end < n:
        raise ValueError("数据不足或划分不合法：lookback 至少为 12，训练窗口至少为 4")
    values = frame["Passengers"].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(frame["Month"])
    raw_features = features(values, dates, kind)
    # 两个模型都只使用训练时间段拟合归一化参数。
    input_scaler = fit_normalizer(raw_features[:train_end])
    normalized = normalize(raw_features, input_scaler)
    indices = np.arange(config.lookback, n)
    logs = np.log(values)
    bases = logs[indices - 12] if kind == "attention_seasonal" else np.zeros(len(indices))
    targets = logs[indices] - bases
    masks = {
        "train": indices < train_end,
        "val": (indices >= train_end) & (indices < val_end),
        "test": indices >= val_end,
    }
    target_scaler = fit_normalizer(targets[masks["train"]])
    windows = np.stack([normalized[i - config.lookback:i] for i in indices]).astype(np.float32)
    standardized_targets = normalize(targets, target_scaler).astype(np.float32)[:, None]
    partitions = {}
    for split, mask in masks.items():
        partitions[split] = {
            "x": windows[mask], "y": standardized_targets[mask],
            "indices": indices[mask], "base": bases[mask],
            "truth": values[indices[mask]],
        }
    return {
        "partitions": partitions,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "train_end": train_end,
        "val_end": val_end,
        "input_size": raw_features.shape[1],
    }


class ForecastNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, attention: bool = False):
        super().__init__()
        self.use_attention = attention
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        if attention:
            self.score = nn.Sequential(nn.Linear(hidden_size, 16), nn.Tanh(), nn.Linear(16, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_size * (2 if attention else 1), 16),
            nn.SiLU(), nn.Linear(16, 1),
        )

    def encode(self, x):
        sequence, _ = self.encoder(x)
        if not self.use_attention:
            return sequence[:, -1], None
        weights = torch.softmax(self.score(sequence), dim=1)
        context = (sequence * weights).sum(dim=1)
        return torch.cat([sequence[:, -1], context], dim=1), weights.squeeze(-1)

    def forward(self, x):
        representation, _ = self.encode(x)
        return self.head(representation)


def decode(prediction: np.ndarray, base: np.ndarray, scaler: dict) -> np.ndarray:
    log_values = prediction.reshape(-1) * float(scaler["std"]) + float(scaler["mean"]) + base
    result = np.exp(log_values)
    if not np.isfinite(result).all():
        raise FloatingPointError("模型产生了非有限预测，请检查数据或学习率")
    return result


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = prediction - truth
    mse = float(np.mean(error ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse, "RMSE": math.sqrt(mse),
        "MAPE_percent": float(np.mean(np.abs(error) / truth) * 100),
        "R2": 1 - float(np.sum(error ** 2)) / total if total > 0 else None,
    }


def latency_ms(model, sample, device) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(100):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (perf_counter() - start) * 1000 / 100


def train_model(frame, config, kind, device, directory):
    seed_everything(config.seed)
    prepared = prepare(frame, config, kind)
    partitions = prepared["partitions"]
    tensors = {
        name: (torch.as_tensor(part["x"], device=device), torch.as_tensor(part["y"], device=device))
        for name, part in partitions.items() if name != "test"
    }
    model = ForecastNet(prepared["input_size"], config.hidden_size, kind == "attention_seasonal").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    history = []
    started = perf_counter()
    print(f"\nTraining {kind}: train={len(partitions['train']['x'])}, val={len(partitions['val']['x'])}, test={len(partitions['test']['x'])}", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tensors["train"][0]), tensors["train"][1])
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失不是有限值")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(tensors["val"][0]), tensors["val"][1]).item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}: train={loss.item():.6f}, val={val_loss:.6f}", flush=True)
        if stale >= config.patience:
            break
    training_seconds = perf_counter() - started
    if best_state is None:
        raise RuntimeError("没有获得有效的模型权重")
    model.load_state_dict(best_state)
    model.eval()
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    checkpoint = {
        "state_dict": best_state, "kind": kind, "config": asdict(config),
        "input_size": prepared["input_size"],
        "input_scaler": prepared["input_scaler"], "target_scaler": prepared["target_scaler"],
        "train_end": prepared["train_end"], "val_end": prepared["val_end"],
        "data_start": frame["Month"].iloc[0].strftime("%Y-%m"),
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, directory / "best.pt")
    predictions, weights = {}, None
    for split, part in partitions.items():
        with torch.inference_mode():
            inputs = torch.as_tensor(part["x"], device=device)
            outputs = model(inputs).cpu().numpy()
            if split == "test" and model.use_attention:
                _, weights = model.encode(inputs)
                weights = weights.cpu().numpy()
        prediction = decode(outputs, part["base"], prepared["target_scaler"])
        predictions[split] = prediction
        pd.DataFrame({
            "Month": frame["Month"].iloc[part["indices"]].dt.strftime("%Y-%m").to_numpy(),
            "Actual": part["truth"], "Predicted": prediction,
            "Error": prediction - part["truth"],
        }).to_csv(directory / f"predictions_{split}.csv", index=False)
    row = {"model": kind, **metrics(partitions["test"]["truth"], predictions["test"])}
    row.update({
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_RMSE": metrics(partitions["val"]["truth"], predictions["val"])["RMSE"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "network_ms_per_sample": latency_ms(model, torch.as_tensor(partitions["test"]["x"][:1], device=device), device),
    })
    (directory / "metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Best epoch={best_epoch}; TEST MAE={row['MAE']:.3f}, RMSE={row['RMSE']:.3f}", flush=True)
    return {"model": model, "checkpoint": checkpoint, "prepared": prepared, "history": pd.DataFrame(history), "predictions": predictions, "metrics": row, "attention": weights}


def forecast(frame, model, checkpoint, horizon, device):
    """递归预测：未来窗口只能加入模型预测值，不使用不存在的未来真值。"""
    history = frame["Passengers"].to_numpy(dtype=np.float64).tolist()
    dates = pd.DatetimeIndex(frame["Month"])
    start = pd.Timestamp(checkpoint["data_start"])
    if dates[0] != start:
        raise ValueError("预测数据的开始月份必须与保存模型的数据一致")
    lookback = checkpoint["config"]["lookback"]
    if len(history) < lookback:
        raise ValueError("历史数据长度小于模型窗口")
    future_months = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    predictions = []
    model.eval()
    for month in future_months:
        raw = features(np.asarray(history), dates, checkpoint["kind"])
        window = normalize(raw[-lookback:], checkpoint["input_scaler"]).astype(np.float32)[None]
        base = np.log(history[-12]) if checkpoint["kind"] == "attention_seasonal" else 0.0
        with torch.inference_mode():
            prediction = model(torch.as_tensor(window, device=device)).cpu().numpy()
        value = float(decode(prediction, np.asarray([base]), checkpoint["target_scaler"])[0])
        predictions.append(value)
        history.append(value)
        dates = dates.append(pd.DatetimeIndex([month]))
    return pd.DataFrame({"Month": future_months, "Predicted": predictions})


def save_plot(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(frame, results, comparison, future, directory):
    first = results["lstm"]["prepared"]
    dates = pd.DatetimeIndex(frame["Month"])
    values = frame["Passengers"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, values, color="#26364a", linewidth=1.6)
    for left, right, label, color in [
        (0, first["train_end"], "Train", "#b7d5ed"),
        (first["train_end"], first["val_end"], "Validation", "#f9db9d"),
        (first["val_end"], len(frame), "Test", "#b9dfcb"),
    ]:
        end = dates[right] if right < len(frame) else dates[-1] + pd.offsets.MonthBegin(1)
        ax.axvspan(dates[left], end, color=color, alpha=.45, label=label)
    ax.set(title="Monthly passengers and chronological split", ylabel="Passengers (CSV units)")
    ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate()
    save_plot(fig, directory / "01_dataset_split.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (kind, result) in zip(axes, results.items()):
        history = result["history"]
        ax.plot(history["epoch"], history["train_loss"], label="Train")
        ax.plot(history["epoch"], history["val_loss"], label="Validation")
        ax.axvline(result["checkpoint"]["best_epoch"], color="gray", linestyle="--", linewidth=1, label="Selected epoch")
        ax.set(title=kind, xlabel="Epoch", ylabel="MSE of standardized target")
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); save_plot(fig, directory / "02_loss_curves.png")

    test_indices = first["partitions"]["test"]["indices"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates[test_indices], values[test_indices], "o-", color="#202a35", label="Actual", linewidth=2)
    colors = {"last_value": "#999999", "seasonal_naive": "#d3a84b", "seasonal_growth": "#9a77aa", "lstm": "#377eb8", "attention_seasonal": "#d85049"}
    for name in comparison.columns.drop(["Month", "Actual"]):
        ax.plot(dates[test_indices], comparison[name], label=name, color=colors[name], linestyle="--" if name not in results else "-", alpha=.85)
    ax.set(title="Rolling one-step predictions on the held-out test period", ylabel="Passengers (CSV units)")
    ax.legend(ncol=2); ax.grid(alpha=.2); fig.autofmt_xdate()
    save_plot(fig, directory / "03_test_predictions.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (kind, result) in zip(axes, results.items()):
        error = result["predictions"]["test"] - values[test_indices]
        ax.bar(np.arange(len(error)), error, color=colors[kind])
        ax.axhline(0, color="black", linewidth=1)
        ax.set(title=kind, xlabel="Test month index", ylabel="Predicted - actual")
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); save_plot(fig, directory / "04_test_errors.png")

    metric_table = pd.read_csv(directory / "metrics_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.barh(metric_table["model"], metric_table[metric], color=[colors[name] for name in metric_table["model"]])
        ax.set(title=f"Test {metric} (lower is better)", xlabel="Passengers (CSV units)")
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); save_plot(fig, directory / "05_metrics_comparison.png")

    weights = results["attention_seasonal"]["attention"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(np.arange(-weights.shape[1], 0), weights.mean(axis=0), color="#467c98")
    ax.set(title="Mean temporal attention weights across test windows", xlabel="Months before prediction", ylabel="Attention weight")
    ax.grid(axis="y", alpha=.2)
    save_plot(fig, directory / "06_attention_weights.png")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates[-48:], values[-48:], color="#202a35", label="Observed history")
    for kind in results:
        ax.plot(pd.to_datetime(future["Month"]), future[kind], "o--", color=colors[kind], label=f"Future: {kind}")
    ax.axvline(dates[-1], color="gray", linestyle=":")
    ax.set(title="Recursive future forecast (future actual values are unavailable)", ylabel="Passengers (CSV units)")
    ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate()
    save_plot(fig, directory / "07_future_forecast.png")


def write_report(frame, config, results, metric_table, directory, device):
    prepared = results["lstm"]["prepared"]
    periods = []
    for split, part in prepared["partitions"].items():
        selected = frame["Month"].iloc[part["indices"]]
        periods.append(f"- {split}: {len(selected)} 个预测目标，{selected.iloc[0]:%Y-%m} 至 {selected.iloc[-1]:%Y-%m}")
    table = ["| Model | MAE | MSE | RMSE | MAPE (%) | R2 |", "|---|---:|---:|---:|---:|---:|"]
    for row in metric_table.to_dict("records"):
        r2 = "N/A" if row["R2"] is None else f"{row['R2']:.4f}"
        table.append(f"| {row['model']} | {row['MAE']:.3f} | {row['MSE']:.3f} | {row['RMSE']:.3f} | {row['MAPE_percent']:.3f} | {r2} |")
    base = results["lstm"]["metrics"]["RMSE"]
    improved = results["attention_seasonal"]["metrics"]["RMSE"]
    change = (base - improved) / base * 100
    best_comparison = metric_table.loc[metric_table["RMSE"].idxmin()]
    text = f"""# 时序预测实验结果

数据来自随题目提供的 airline-passengers.csv，共 {len(frame)} 个月。数值使用 CSV 原单位。

## 划分与评价协议

训练/验证/测试按时间顺序划分为 {config.train_ratio:.0%}/{config.val_ratio:.0%}/{1-config.train_ratio-config.val_ratio:.0%}。输入窗口为 {config.lookback} 个月。
{chr(10).join(periods)}

所有归一化参数均由训练时间段拟合。最佳轮次由验证损失选择，测试集在选定权重后评价，不参与梯度更新或模型选择。
测试采用滚动单步预测：预测某月时，可以使用此前月份已观测到的真实值，包括更早的验证/测试月份；不能使用当前月或更晚的值。此结果不等同于从训练结束点一次预测整个测试期。
未来 {config.horizon} 个月使用递归预测，把前一步预测值加入后续窗口，没有未来真实值，也没有把单步测试误差当成未来误差。

## 模型与对照

- lstm：单变量对数客流，经 LSTM 后预测下一月对数值。
- attention_seasonal：LSTM 时间注意力、同比增长及月份特征、以去年同月为基准的对数残差预测。
- last_value：预测等于上月真实值。
- seasonal_naive：预测等于去年同月真实值。
- seasonal_growth：去年同月乘以由训练段估计的平均年度增长倍数。

改进版同时改变特征、预测目标和注意力结构，属于组合改进实验，不能把全部变化单独归因于注意力。
两版损失均为各自标准化目标的 MSE，因为预测目标不同，损失数值不适合跨模型直接比较。

## 独立测试结果

{chr(10).join(table)}

相对基础 LSTM，改进版 RMSE 的下降比例为 {change:.2f}%（负数表示变差）。保留所有基线和失败结果。
本次全部对照中，最低 RMSE 来自 {best_comparison['model']}，为 {best_comparison['RMSE']:.3f}。如果简单季节基线更好，应如实报告，不能宣称神经网络全面优于基线。
R2 可以为负；这表示该测试段上的平方误差大于以测试真值均值作为常数预测的误差。基础 LSTM 在测试期存在明显低估，结果保留用于分析趋势外推的局限。

## 复现条件

- Device: {device}
- PyTorch: {torch.__version__}
- Seed: {config.seed}
- 最大轮数: {config.epochs}；早停等待轮数: {config.patience}
- AdamW 学习率: {config.learning_rate}；权重衰减: {config.weight_decay}
- 梯度裁剪: 1.0；整批训练；无多进程 DataLoader。

每个模型的 best.pt 包含网络权重、网络配置和归一化参数。history.csv 保存训练记录；predictions_test.csv 保存原单位的逐月预测。
network_ms_per_sample 只测模型前向计算，预热 10 次后测 100 次，不包含读取、特征生成或写图。
注意力图只是权重分布展示，不代表因果解释。单个小数据集、单个随机种子的结果不能保证在其他数据上同样有效。
"""
    (directory / "实验结果说明.md").write_text(text, encoding="utf-8")


def run_training(frame, config, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    # 两版实验配置预先固定；最佳权重只根据验证损失选择，测试指标不参与选择。
    for kind in ["lstm", "attention_seasonal"]:
        results[kind] = train_model(frame, config, kind, device, directory / kind)
    prepared = results["lstm"]["prepared"]
    indices = prepared["partitions"]["test"]["indices"]
    values = frame["Passengers"].to_numpy(dtype=np.float64)
    logs = np.log(values)
    growth = float(np.mean(logs[12:prepared["train_end"]] - logs[:prepared["train_end"] - 12]))
    baseline_predictions = {
        "last_value": values[indices - 1],
        "seasonal_naive": values[indices - 12],
        "seasonal_growth": values[indices - 12] * np.exp(growth),
    }
    rows = [{"model": name, **metrics(values[indices], prediction)} for name, prediction in baseline_predictions.items()]
    rows.extend(result["metrics"] for result in results.values())
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(directory / "metrics_comparison.csv", index=False)
    comparison = pd.DataFrame({"Month": frame["Month"].iloc[indices].dt.strftime("%Y-%m").to_numpy(), "Actual": values[indices]})
    for name, prediction in baseline_predictions.items():
        comparison[name] = prediction
    for kind, result in results.items():
        comparison[kind] = result["predictions"]["test"]
    comparison.to_csv(directory / "test_predictions_comparison.csv", index=False)
    future = None
    for kind, result in results.items():
        prediction = forecast(frame, result["model"], result["checkpoint"], config.horizon, device)
        if future is None:
            future = pd.DataFrame({"Month": prediction["Month"].dt.strftime("%Y-%m")})
        future[kind] = prediction["Predicted"].to_numpy()
    future.to_csv(directory / "future_forecast.csv", index=False)
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    make_figures(frame, results, comparison, future, directory)
    write_report(frame, config, results, metric_table, directory, device)
    print("\nTEST comparison (original CSV units):")
    print(metric_table[["model", "MAE", "RMSE", "MAPE_percent", "R2"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {directory.resolve()}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "forecast"], default="train")
    parser.add_argument("--data", type=Path, default=root / "dataset" / "airline-passengers.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "final_run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.004)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.hidden_size, args.horizon) < 1 or args.lr <= 0:
        parser.error("epochs、patience、hidden-size、horizon、lr 必须为正数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请改用 --device cpu")
    seed_everything(args.seed)
    device = torch.device(args.device)
    frame = load_data(args.data)
    config = Config(lookback=args.lookback, hidden_size=args.hidden_size, epochs=args.epochs, patience=args.patience, learning_rate=args.lr, seed=args.seed, horizon=args.horizon)
    if args.mode == "train":
        run_training(frame, config, device, args.output)
    else:
        if args.checkpoint is None:
            parser.error("forecast 模式必须指定 --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = ForecastNet(checkpoint["input_size"], checkpoint["config"]["hidden_size"], checkpoint["kind"] == "attention_seasonal").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        prediction = forecast(frame, model, checkpoint, args.horizon, device)
        args.output.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(args.output / "future_forecast.csv", index=False)
        print(prediction.to_string(index=False))
        print("Saved to:", args.output / "future_forecast.csv")


if __name__ == "__main__":
    main()
