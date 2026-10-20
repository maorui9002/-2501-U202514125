import torch

# 固定随机种子，让重复运行得到相同的随机数据
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义不含噪声的真实函数
def true_function(x1, x2):
    return (
        2 * x1**2
        + 1.5 * x1 * x2
        + 3 * torch.sin(x1)
        + 0.5 * torch.cos(2 * x2)
        + torch.exp(-0.5 * x1)
        + 0.8 * x2
        + 2026
    )


# 生成 2000 组输入，两个变量均在 [-3, 3) 内随机取值
sample_count = 2000

x1 = torch.rand(sample_count, 1, device=device) * 6 - 3
x2 = torch.rand(sample_count, 1, device=device) * 6 - 3

# 干净的函数值
y_clean = true_function(x1, x2)

# 在 [-3, 3) 内生成均匀随机噪声
noise = torch.rand(sample_count, 1, device=device) * 6 - 3

# 模拟实际观测数据
y_observed = y_clean + noise

print("运行设备：", device)
print("输入形状：", x1.shape, x2.shape)
print("输出形状：", y_observed.shape)
print(f"噪声范围：{noise.min().item():.3f} ～ {noise.max().item():.3f}")

print("\n前 5 组数据：")
for i in range(5):
    print(
        f"x1={x1[i].item():7.3f}, "
        f"x2={x2[i].item():7.3f}, "
        f"真实值={y_clean[i].item():9.3f}, "
        f"观测值={y_observed[i].item():9.3f}"
    )
    import torch.nn as nn


class FormulaModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 七个待学习的参数，初始值不是最终答案
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        self.a4 = nn.Parameter(torch.tensor(0.0))
        self.a5 = nn.Parameter(torch.tensor(-0.1))
        self.a6 = nn.Parameter(torch.tensor(0.0))

        # 偏置初值接近输出数值的量级，后续仍会更新
        self.b = nn.Parameter(torch.tensor(2000.0))

    def forward(self, x1, x2):
        return (
            self.a1 * x1**2
            + self.a2 * x1 * x2
            + self.a3 * torch.sin(x1)
            + self.a4 * torch.cos(2 * x2)
            + torch.exp(self.a5 * x1)
            + self.a6 * x2
            + self.b
        )


# 将模型参数放到与输入相同的设备上
model = FormulaModel().to(device)

print("\n模型初始参数：")
for name, parameter in model.named_parameters():
    print(f"{name} = {parameter.item():.4f}")

# 先检查模型是否能完成预测
with torch.no_grad():
    initial_prediction = model(x1, x2)

print("预测结果形状：", initial_prediction.shape)
# 1. 随机划分：80% 用于训练，20% 留作测试
generator = torch.Generator().manual_seed(123)
indices = torch.randperm(sample_count, generator=generator).to(device)

train_count = int(sample_count * 0.8)
train_indices = indices[:train_count]
test_indices = indices[train_count:]

x1_train = x1[train_indices]
x2_train = x2[train_indices]
y_train = y_observed[train_indices]

x1_test = x1[test_indices]
x2_test = x2[test_indices]
y_test = y_observed[test_indices]

# 2. 根据训练数据的均值初始化偏置，加快训练
with torch.no_grad():
    model.b.copy_(y_train.mean())

# 3. 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

epochs = 2000
loss_history = []

model.train()
print("\n开始训练……")

# 4. 重复预测、计算误差、求导和更新参数
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    prediction = model(x1_train, x2_train)
    loss = criterion(prediction, y_train)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch == 1 or epoch % 250 == 0:
        print(f"Epoch {epoch:4d}/{epochs}，Loss = {loss.item():.6f}")

# 5. 查看训练后的参数
print("\n训练后的参数：")
for name, parameter in model.named_parameters():
    print(f"{name} = {parameter.item():.4f}")
from pathlib import Path

import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 在测试集上评价模型，不再更新参数
model.eval()

with torch.no_grad():
    test_prediction = model(x1_test, x2_test)

    test_mse = criterion(test_prediction, y_test).item()

    # 模型预测误差的平方和
    ss_res = torch.sum((y_test - test_prediction) ** 2)

    # 以测试集均值预测时，误差的平方和
    ss_tot = torch.sum((y_test - y_test.mean()) ** 2)

    r2 = (1 - ss_res / ss_tot).item()

print(f"\n测试集 MSE：{test_mse:.6f}")
print(f"测试集 R²：{r2:.6f}")

# 2. 保存参数及测试结果
lines = [
    f"{name} = {parameter.item():.6f}"
    for name, parameter in model.named_parameters()
]
lines.extend([
    f"Test MSE = {test_mse:.6f}",
    f"Test R2 = {r2:.6f}",
])

(output_dir / "fit_results.txt").write_text(
    "\n".join(lines),
    encoding="utf-8",
)

# 3. 绘制训练损失曲线
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(1, epochs + 1), loss_history)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training MSE")
ax.set_title("Function Fitting: Training Loss")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "fit_loss.png", dpi=200)
plt.close(fig)

print("参数、评价指标和损失曲线已保存到：", output_dir) 
# 1. 在 [-3, 3] 范围内生成规则网格
axis_values = torch.linspace(-3, 3, 80, device=device)

grid_x1, grid_x2 = torch.meshgrid(
    axis_values,
    axis_values,
    indexing="ij",
)

# 2. 分别计算无噪声真实值和模型预测值
model.eval()

with torch.no_grad():
    grid_true = true_function(grid_x1, grid_x2)
    grid_fitted = model(grid_x1, grid_x2)

# Matplotlib 使用 CPU 上的 NumPy 数组
X1 = grid_x1.cpu().numpy()
X2 = grid_x2.cpu().numpy()
Z_true = grid_true.cpu().numpy()
Z_fitted = grid_fitted.cpu().numpy()

# 3. 两张图使用相同的高度范围和颜色范围
z_min = min(Z_true.min(), Z_fitted.min())
z_max = max(Z_true.max(), Z_fitted.max())

fig = plt.figure(figsize=(13, 6), layout="constrained")

panels = [
    (Z_true, "True Function (Without Noise)"),
    (Z_fitted, f"Fitted Function (Test R2 = {r2:.4f})"),
]

for index, (Z, title) in enumerate(panels, start=1):
    ax = fig.add_subplot(1, 2, index, projection="3d")

    ax.plot_surface(
        X1, X2, Z,
        cmap="viridis",
        vmin=z_min,
        vmax=z_max,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y", labelpad=10)
    ax.set_title(title)
    ax.set_zlim(z_min, z_max)

    # 使用相同视角，方便比较
    ax.view_init(elev=25, azim=-55)

    # 直接显示完整数值，避免出现偏移量标记
    ax.ticklabel_format(axis="z", style="plain", useOffset=False)

# 4. 保存曲面对比图
surface_path = output_dir / "fit_surfaces.png"
fig.savefig(surface_path, dpi=200)
plt.close(fig)

print("曲面对比图已保存：", surface_path)