import torch
from torch import nn


class SceneCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        layers = []
        channels = [3, 32, 64, 128]

        for input_channels, output_channels in zip(
            channels[:-1], channels[1:]
        ):
            layers.extend([
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneCNN().to(device)

    # 模拟 4 张 RGB 图片
    sample = torch.randn(4, 3, 128, 128, device=device)

    with torch.no_grad():
        result = model(sample)

    print("输入形状：", sample.shape)
    print("输出形状：", result.shape)
    print("参数数量：", sum(p.numel() for p in model.parameters()))