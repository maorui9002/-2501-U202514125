"""使用已训练的 CNN 预测一张图片，不启动训练。"""
from pathlib import Path
import argparse

from PIL import Image
import torch
from torchvision import transforms
from cnn_model import SceneCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="待分类图片的路径")
    parser.add_argument("--model", choices=["cnn_baseline", "cnn_augmented"], default="cnn_baseline")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(__file__).resolve().parent / "outputs" / args.model / "best.pth", map_location="cpu", weights_only=True)
    model = SceneCNN(len(checkpoint["classes"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    with Image.open(args.image) as source:
        inputs = transform(source.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = model(inputs).softmax(1)[0].cpu()
    print(f"模型：{args.model} | 最佳轮次：{checkpoint['epoch']}")
    for i in probs.argsort(descending=True)[:3].tolist():
        print(f"{checkpoint['classes'][i]}：{probs[i]:.2%}")
    print("以上为模型 softmax 分数，未进行置信度校准。")


if __name__ == "__main__":
    main()
