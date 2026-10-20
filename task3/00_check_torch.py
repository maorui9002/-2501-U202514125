import torch

print("PyTorch 版本：", torch.__version__)
print("CUDA 运行时版本：", torch.version.cuda)
print("GPU 是否可用：", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch 未能使用显卡，请把输出发给我")

print("显卡名称：", torch.cuda.get_device_name(0))

# 在显卡上创建矩阵，并进行矩阵乘法
matrix = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    device="cuda",
)

result = matrix @ matrix

# 取回结果，确认 GPU 计算已完成
print("矩阵乘法结果：\n", result.cpu())
print("GPU 计算验证通过")