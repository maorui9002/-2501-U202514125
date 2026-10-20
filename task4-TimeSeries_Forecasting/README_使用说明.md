# 时序预测完整项目

本项目使用题目提供的月度客流数据，完成时序预测、改进实验、模型保存和全部主要图表。代码与参考答案重新组织实现。

## 使用

在 VS Code 打开本目录，选择已经配置好的 `comsen-answer` Python 解释器。当前环境已有所需库，无需重新安装 PyTorch。

直接运行 `timeseries_forecasting.py`，或在 PowerShell 输入：

```powershell
& "D:\anaconda\envs\comsen-answer\python.exe" .\timeseries_forecasting.py
```

数据只有 144 个月，默认使用 CPU 单线程和整批训练，避免小模型的 GPU 调用开销及 Windows 多进程共享内存问题。需要比较 GPU 时可加 `--device cuda`。

## 已包含的功能

- 检查月份连续性、重复值、缺失值和客流正值。
- 按时间划分训练、验证、测试，比例为 70%/15%/15%。
- 24 个月历史窗口；归一化参数仅从训练时间段拟合。
- 基础 LSTM；LSTM 注意力＋月份/同比特征＋季节残差改进。
- 上月值、去年同月、训练段估计的季节增长三个简单基线。
- AdamW、梯度裁剪、验证早停、保存最佳模型。
- 原始单位的 MAE、MSE、RMSE、MAPE、R2。
- 七张图片、逐月预测 CSV、训练记录、配置与实验说明。
- 保存模型后，可直接重新加载并递归预测未来，无需重新训练。

## 默认输出

```text
outputs/final_run/
├─ lstm/
│  ├─ best.pt
│  ├─ history.csv
│  ├─ metrics.json
│  └─ predictions_train/val/test.csv
├─ attention_seasonal/
│  └─ 同样的模型、训练和预测文件
├─ 01_dataset_split.png
├─ 02_loss_curves.png
├─ 03_test_predictions.png
├─ 04_test_errors.png
├─ 05_metrics_comparison.png
├─ 06_attention_weights.png
├─ 07_future_forecast.png
├─ metrics_comparison.csv
├─ test_predictions_comparison.csv
├─ future_forecast.csv
├─ config.json
└─ 实验结果说明.md
```

`predictions_train/val/test.csv` 在上面的树中是简写，实际为 `predictions_train.csv`、`predictions_val.csv`、`predictions_test.csv` 三个文件。

重复运行默认会覆盖 `outputs/final_run` 中的同名结果。保留一次实验可指定新目录：

```powershell
& "D:\anaconda\envs\comsen-answer\python.exe" .\timeseries_forecasting.py --output .\outputs\run_02
```

重新加载改进模型，仅预测未来 12 个月：

```powershell
& "D:\anaconda\envs\comsen-answer\python.exe" .\timeseries_forecasting.py --mode forecast --checkpoint .\outputs\final_run\attention_seasonal\best.pt --horizon 12 --output .\outputs\forecast_only
```

## 评价口径

测试为滚动单步预测：每个月的预测只使用此前已知月份的实际观测，不使用当前月或更晚的值。它与从训练截止月开始一次性预测整个测试区间不同。

未来预测采用递归方式，后续输入中使用此前预测值，没有未来真值，不能声称具有与单步测试相同的误差。

改进版同时改变注意力、特征和预测目标，因此只能报告组合方法的效果，不能把全部变化归功于注意力。无论是否优于简单基线，代码都会保留实际结果。

两版训练损失对应不同的标准化目标：基础模型预测对数客流，改进模型预测对数同比残差。损失数值不适合直接跨模型比较，效果应比较原单位测试指标。

`network_ms_per_sample` 是预热后的单样本网络前向耗时，不包含 CSV 读取、特征生成和绘图。

## 已验证结果

随附的 `outputs/final_run` 是在当前 Python 3.11、PyTorch 2.10.0、CPU 单线程环境下实际生成的结果。

- 基础 LSTM 测试 RMSE：102.743。
- 注意力＋季节残差模型测试 RMSE：23.657。
- 简单季节增长基线测试 RMSE：17.920，优于本次神经网络结果。

保留基础模型较差和简单基线更好的结果，便于在报告中分析小样本、趋势外推和季节性建模，不保证每个复杂模型都优于简单模型。

运行 `check_correctness.py` 可检查训练统计量是否受到验证/测试数据影响、单步输入是否包含未来信息、保存权重能否复现指标、注意力归一化以及递归预测月份。
