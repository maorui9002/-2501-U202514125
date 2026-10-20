"""将 05 的最终图表与指标排版成中文 PDF。依赖 reportlab 与中文字体。"""
import argparse
from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "cnn_final_report")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    root = args.report_dir.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    pdfmetrics.registerFont(TTFont("Chinese", str(args.font), subfontIndex=0))
    body = ParagraphStyle("Body", fontName="Chinese", fontSize=10, leading=17, textColor=colors.HexColor("#253b50"), spaceAfter=8, wordWrap="CJK")
    heading = ParagraphStyle("Heading", parent=body, fontSize=15, leading=23, textColor=colors.HexColor("#16496c"), spaceBefore=5, spaceAfter=12)
    title = ParagraphStyle("Title", parent=heading, fontSize=23, leading=32, spaceAfter=14)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=14, textColor=colors.HexColor("#526577"))
    table_text = ParagraphStyle("Table", parent=body, fontSize=9, leading=15, spaceAfter=0)
    doc = SimpleDocTemplate(str(root / "图像分类实验报告.pdf"), pagesize=A4,
                            rightMargin=42, leftMargin=42, topMargin=45, bottomMargin=40,
                            title="图像分类实验报告", author="comsen-answer")
    width = A4[0] - 84
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def picture(name, max_height):
        image = Image(str(root / name))
        scale = min(width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        image.hAlign = "CENTER"
        story.extend([image, Spacer(1, 9)])

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), table_text) for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f7")),
                               ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#cbd7e2")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.extend([t, Spacer(1, 10)])

    a, b = (summary["models"][v] for v in ("cnn_baseline", "cnn_augmented"))
    p("图像分类实验报告", title)
    p("Intel 六类场景 · CNN 基础模型与数据增强对照", heading)
    p(f"当前权重复核结果：基础模型 {a['test_accuracy']:.2%}，增强模型 {b['test_accuracy']:.2%}。按验证准确率分别选择第 {a['best_epoch']}、{b['best_epoch']} 轮；本次整理不重新训练。")
    p("1  数据与实验协议", heading)
    p(f"类别为 {', '.join(summary['classes'])}。原训练目录 {summary['source_train_count']:,} 张，以种子 42 按 80% / 20% 划分：训练 {summary['train_count']:,} 张、验证 {summary['validation_count']:,} 张；测试目录 {summary['test_count']:,} 张。两组实验使用同一划分。")
    p("图片转为 RGB 并缩放至 128×128；ToTensor 后以均值 0.5、标准差 0.5 归一化。验证、测试采用确定性变换。此处独立测试指测试目录不参与训练梯度更新；未进行跨目录重复图片审计。")
    p("2  模型结构", heading)
    picture("01_architecture.png", 135)
    p("三个卷积块依次包含 3×3 卷积、BatchNorm、ReLU 和 2×2 最大池化，通道为 32、64、128。卷积共享权重提取局部纹理，非线性与池化逐步形成场景特征。自适应平均池化至 2×2 后展平，接 512→128→6 分类头，中间使用 Dropout(0.3)。共 159,910 个参数。")
    p("交叉熵直接接收 logits，推理取 argmax。图片标题显示 softmax 最大分数，未作置信度校准，不能当成已验证的正确概率。", small)
    story.append(PageBreak())

    p("3  训练设置与过程", heading)
    table([["配置", "两种实验共同设置"], ["优化", "AdamW；lr=0.001；weight_decay=0.0001；交叉熵"],
           ["训练", "20 轮；batch_size=64；仅按验证准确率保存最佳模型"],
           ["设备", summary["environment"]["device_name"]]], [75, width - 75])
    picture("02_training_curves.png", 335)
    p("增强版仅改变训练变换：随机裁剪缩放（scale 0.8~1.0、ratio 0.9~1.1）、概率 0.5 的水平翻转、亮度/对比度/饱和度幅度均为 0.15 的颜色扰动。它们增加场景变化，也可能裁掉重要信息。")
    p("竖线为验证集选中的轮次。训练指标在训练模式及批次更新中统计，增强版使用增强输入，因此与固定权重的验证指标存在口径差别。损失按样本数加权平均。最后一轮不一定为最佳轮次。")
    p("测试准确率每轮仅用于监测，不参与梯度更新或权重选择；训练期间已经查看测试曲线，不能描述为完全盲测。Windows 使用单进程四线程、有界四批预读，避免多进程共享内存错误；线程随机增强不保证逐位复现。", small)
    story.append(PageBreak())

    p("4  当前最佳模型的最终测试", heading)
    rows = [["模型", "轮次", "验证 Acc", "测试 Acc", "Macro F1"]]
    for label, m in (("基础 CNN", a), ("增强 CNN", b)):
        rows.append([label, m["best_epoch"], f"{m['val_accuracy']:.2%}", f"{m['test_accuracy']:.2%}", f"{m['macro_f1']:.4f}"])
    table(rows, [100, 45, 118, 118, width - 381])
    delta = (b["test_accuracy"] - a["test_accuracy"]) * 100
    effect = "本次增强未带来测试准确率提升，不能写成改进成功。" if delta <= 0 else "本次增强测试准确率更高，仍需多次实验确认其稳定性。"
    recommended = "基础 CNN" if summary["recommended_model"] == "cnn_baseline" else "增强 CNN"
    p(f"基础模型正确 {a['correct']}/{a['test_count']} 张，增强模型正确 {b['correct']}/{b['test_count']} 张；增强减基础为 {delta:+.2f} 个百分点。{effect}按照验证成绩，当前建议使用{recommended}。单次训练不能支持统计优势结论。")
    picture("03_confusion_counts.png", 245)
    picture("05_class_f1.png", 185)
    p("混淆矩阵行是真实类别，列是预测类别。Accuracy 为正确比例；Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)。Macro F1 对类别等权，weighted F1 按真实样本数加权；零分母时定义为零。逐类指标已保存为 CSV。", small)
    story.append(PageBreak())

    p("5  错误分布与改进分析", heading)
    picture("04_confusion_normalized.png", 265)
    p("上图按真实类别逐行归一化，各行合计 100%，对角线即该类召回率。它可以在类样本数不同的情况下比较漏分情况。")
    for label, m in (("基础模型", a), ("增强模型", b)):
        cm = m["confusion_matrix"]
        pairs = sorted(((cm[i][j], summary["classes"][i], summary["classes"][j]) for i in range(6) for j in range(6) if i != j), reverse=True)[:3]
        p(label + "主要混淆：" + "；".join(f"{x}→{y} {count} 张" for count, x, y in pairs) + "。")
    p("这些统计说明错误出现的位置，不能单独证明成因。相近场景可能具有相似颜色、纹理或混合主体，应结合原图判断。当前增强可能改变关键场景线索，也可能受训练波动影响，不能只凭一次对照下因果结论。")
    p("后续可使用验证集比较更温和的裁剪、学习率调度、早停或轻量注意力模块；固定方案后再做最终评价。应增加多随机种子实验，并同时记录准确率、F1 和实际推理耗时，避免仅报告有利指标。")
    p("6  ViT 原理理解", heading)
    p("ViT 将图像分成固定大小的 patch，展平并线性映射为 token，加入位置编码后送入 Transformer Encoder。多头自注意力通过 Q、K、V 建模不同位置之间的关联，计算形式为 softmax(QK^T/sqrt(d))V；原始 ViT 使用可学习分类 token 汇聚信息，再由分类头预测。")
    p("卷积具有局部感受野与权重共享，自注意力可直接关联远距离 patch；常规全局注意力的计算和内存成本随 token 数近似二次增长。效果依赖数据规模与训练方案。本项目仅说明原理，实际完成 CNN 和增强对照，没有训练 ViT 或编造其成绩。", small)
    p('<link href="https://arxiv.org/abs/2010.11929" color="#216b9d">参考：Dosovitskiy 等，An Image is Worth 16x16 Words，arXiv:2010.11929。</link>', small)
    story.append(PageBreak())

    p("7  同一批随机测试图片的分类结果", heading)
    p("固定种子 2026，从全部测试图片随机抽取六张，两种模型展示相同图片。绿色为预测正确，红色为预测错误；括号内为 softmax 分数。", small)
    picture("06_baseline_predictions.png", 298)
    picture("07_augmented_predictions.png", 298)
    story.append(PageBreak())

    p("8  错误案例与复核方法", heading)
    picture("08_augmented_errors.png", 290)
    p("上图从增强模型全部错误中固定随机抽取六张，仅用于错误分析，不代表整体准确率。随机索引见 sample_indices.json；每张测试图的真实类别、预测类别及六类 softmax 分数见 predictions.csv。")
    p("现有权重直接运行 05_finish_classification.py，即可重新测试并生成八张图、全部 CSV、summary.json 和 Markdown 报告。06_predict_image.py 接收一张本地图片，可选择基础或增强权重；完整命令见项目 README。")
    p("05 会核对最佳轮次、验证准确率及对应轮次的测试成绩，记录权重与历史 SHA-256，避免混用旧记录。若模型或数据更改，需重训或重新复核，不应手工修改指标。05 不自动更新 PDF；请再运行 07_export_report_pdf.py 导出。")
    p(f"环境：Python {summary['environment']['python']}；PyTorch {summary['environment']['torch']}；torchvision {summary['environment']['torchvision']}。提交包包含源码、两组最佳权重、20 轮历史、图表、指标和报告；不包含大体积原始图片，当前电脑可直接沿用 dataset。", small)
    p('<link href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" color="#216b9d">数据集参考：Intel Image Classification（Kaggle）。</link>', small)
    p("结果仅对应本次保存权重与现有数据；两次实验属于单次训练比较，没有多随机种子验证、置信度校准或跨目录重复图片审计。", small)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e1ea"))
        canvas.line(42, 30, A4[0] - 42, 30)
        canvas.setFont("Chinese", 8)
        canvas.setFillColor(colors.HexColor("#6d8193"))
        canvas.drawString(42, 18, "comsen-answer | 图像分类实验")
        canvas.drawRightString(A4[0] - 42, 18, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(root / "图像分类实验报告.pdf")


if __name__ == "__main__":
    main()
