"""检查时间因果性、训练期归一化、权重加载和递归预测。"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timeseries_forecasting import Config, ForecastNet, decode, features, forecast, load_data, metrics, prepare, seed_everything


def main():
    root = Path(__file__).resolve().parent
    seed_everything(42)
    frame = load_data(root / "dataset" / "airline-passengers.csv")
    config = Config()
    expected_scores = pd.read_csv(root / "outputs" / "final_run" / "metrics_comparison.csv").set_index("model")

    for kind in ["lstm", "attention_seasonal"]:
        original = prepare(frame, config, kind)
        changed_frame = frame.copy()
        # 大幅修改整个验证/测试段，不允许改变任何训练统计量或训练窗口。
        changed_frame.loc[original["train_end"]:, "Passengers"] *= 10
        changed = prepare(changed_frame, config, kind)
        assert original["input_scaler"] == changed["input_scaler"]
        assert original["target_scaler"] == changed["target_scaler"]
        np.testing.assert_array_equal(original["partitions"]["train"]["x"], changed["partitions"]["train"]["x"])
        np.testing.assert_array_equal(original["partitions"]["train"]["y"], changed["partitions"]["train"]["y"])

        # 修改测试当前月及以后的值，不应改变该月预测的输入或季节基准。
        test_start = original["val_end"]
        future_changed = frame.copy()
        future_changed.loc[test_start:, "Passengers"] *= 2
        causal = prepare(future_changed, config, kind)
        np.testing.assert_array_equal(original["partitions"]["test"]["x"][0], causal["partitions"]["test"]["x"][0])
        np.testing.assert_array_equal(original["partitions"]["test"]["base"][0], causal["partitions"]["test"]["base"][0])

        path = root / "outputs" / "final_run" / kind / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = ForecastNet(checkpoint["input_size"], checkpoint["config"]["hidden_size"], kind == "attention_seasonal")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        part = original["partitions"]["test"]
        with torch.inference_mode():
            inputs = torch.from_numpy(part["x"])
            raw_prediction = model(inputs).numpy()
            if kind == "attention_seasonal":
                _, weights = model.encode(inputs)
                np.testing.assert_allclose(weights.sum(dim=1).numpy(), 1, atol=1e-6)
                assert (weights >= 0).all()
        prediction = decode(raw_prediction, part["base"], checkpoint["target_scaler"])
        np.testing.assert_allclose(metrics(part["truth"], prediction)["RMSE"], expected_scores.loc[kind, "RMSE"], rtol=1e-5)
        future = forecast(frame, model, checkpoint, 12, torch.device("cpu"))
        assert len(future) == 12
        assert future["Month"].iloc[0] == pd.Timestamp("1961-01-01")
        assert np.isfinite(future["Predicted"]).all() and (future["Predicted"] > 0).all()
        print(f"PASS {kind}: no training-statistic leakage, causal windows, checkpoint reload, 12-month forecast")

    known = metrics(np.array([1., 2., 3.]), np.array([1., 2., 4.]))
    np.testing.assert_allclose(known["MSE"], 1 / 3)
    np.testing.assert_allclose(known["RMSE"], np.sqrt(1 / 3))
    np.testing.assert_allclose(known["R2"], .5)
    print("PASS metrics formulas and attention normalization")


if __name__ == "__main__":
    main()
