# 图像分类完整项目

已完成六类场景 CNN、训练集数据增强对照、20 轮 Loss / Train Acc / Test Acc 曲线、独立目录测试、混淆矩阵、分类示例、错误案例、逐类指标及实验报告。沿用当前已训练权重，无需再次训练。

当前结果：基础 CNN 测试准确率 86.10%，增强 CNN 85.63%，两者最佳轮次均为 19。模型按验证集选择，建议使用基础 CNN。历史训练的 86.80% / 87.20% 属于另一组权重，本项目最终报告使用当前权重重新测试所得数据。

## 在现有项目中使用

VS Code 打开 `D:/comsen/comsen-answer`，解释器选 `D:/anaconda/envs/comsen-answer/python.exe`。以下为 PowerShell 命令。

重新整理现有权重的全部图表、指标和 Markdown 报告：

```powershell
& D:/anaconda/envs/comsen-answer/python.exe D:/comsen/comsen-answer/task4-Image_Processing/05_finish_classification.py
```

预测自己的图片，替换图片路径即可：

```powershell
& D:/anaconda/envs/comsen-answer/python.exe D:/comsen/comsen-answer/task4-Image_Processing/06_predict_image.py "D:/你的图片.jpg"
```

增强模型可加 `--model cnn_augmented`。图片路径不存在时会明确提示。

## 文件与结果

- `01_check_data.py`：检查六类图片。
- `cnn_model.py`：159,910 参数的 SceneCNN。
- `02_train_cnn.py`：基础实验，训练 20 轮。
- `03_train_augmented.py`：相同划分上的增强实验，训练 20 轮。
- `04_test_cnn.py`：两组最佳权重的简洁测试。
- `05_finish_classification.py`：复核权重与历史是否匹配，重新测试并生成八张结果图、CSV、JSON、中文报告。
- `06_predict_image.py`：自己的单张图片推理。
- `07_export_report_pdf.py`：将最终结果排版为 PDF。
- `fast_loader.py`：单进程、四线程、有界预读，避免 Windows 多进程共享内存问题。
- `outputs/cnn_baseline` 和 `outputs/cnn_augmented`：两组 `best.pth`、20 轮 `history.csv`。
- `outputs/cnn_final_report`：八张图、PDF/Markdown 报告、测试比较表、逐类指标、每张测试图的预测以及完整混淆矩阵。
- `outputs/cnn_comparison`：兼容旧测试脚本的最新比较表和预测 CSV。

`summary.json` 保存环境、当前权重及训练历史 SHA-256。`sample_indices.json` 记录随机示例索引；两种模型展示同一批随机测试图片。错误图从增强模型的错误集合中抽样，不代表测试集整体分布。softmax 分数未进行置信度校准。

## 数据集与迁移

提交包不包含大体积数据集。现有电脑直接沿用该目录里的 `dataset`，不要删除。换电脑时使用 [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)，整理为：

```text
task4-Image_Processing/
  dataset/
    seg_train/
      buildings/ forest/ glacier/ mountain/ sea/ street/
    seg_test/
      buildings/ forest/ glacier/ mountain/ sea/ street/
  cnn_model.py
  05_finish_classification.py
  outputs/
    cnn_baseline/best.pth
    cnn_augmented/best.pth
```

原训练目录 14,034 张，以种子 42 划分为 11,227 训练、2,807 验证，测试目录 3,000 张。数据必须与训练时相同，整理脚本会核对复算测试准确率与最佳轮次历史记录；数据变化时请重新训练，不要修改表格凑结果。未执行跨目录重复图片审计。

## 环境与重新训练

现有 `comsen-answer` 环境已具备训练与绘图依赖，不用再次安装。复核环境为 Python 3.11、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128。若更换环境，请按显卡安装匹配的 PyTorch/torchvision，再安装 `requirements.txt` 中其余包。

需要重训时依次运行 02、03、05；02、03 会更新各自输出目录的权重和训练记录，运行前按需备份。训练集随机增强采用线程执行，不能保证逐位复现。

训练每轮记录测试准确率，但只按验证准确率保存模型。训练准确率是在更新过程中统计，增强版还使用随机变换，不能与固定模型的验证成绩直接等同。增强效果可能受裁剪幅度和随机性影响，本次结果不支持已提高准确率的结论。

## PDF 更新

已附带本次结果的 PDF。05 重新生成的是图表与 Markdown；若重新训练或更改结果，需要另运行 07 导出 PDF，避免保留旧报告。07 额外需要 `reportlab`，使用 Windows 微软雅黑字体；其他系统可通过 `--font` 指定支持中文的 TrueType 字体。

```powershell
& D:/anaconda/envs/comsen-answer/python.exe -m pip install reportlab
& D:/anaconda/envs/comsen-answer/python.exe D:/comsen/comsen-answer/task4-Image_Processing/07_export_report_pdf.py
```

ViT 的原理说明和参考文献已写入报告；本项目实际训练 CNN 与增强对照，没有虚构 Transformer 实验。
