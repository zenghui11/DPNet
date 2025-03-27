
import torch
from models.Fusion import MBHFuse
import torch.nn as nn
from thop import profile
import os

# 初始化模型并加载到 GPU
model = MBHFuse().cuda()
device = torch.device("cuda:0")
model.to(device)
model.eval()  # 切换到评估模式

# 定义输入张量
random_input = torch.randn(1, 2, 256, 256).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(random_input,))
print('FLOPs = ' + str(flops/1000**3) + ' G')
print('Params = ' + str(params/1000**2) + ' M')

# GPU 预热
for _ in range(50):
    _ = model(random_input)

# 测速
iterations = 300   # 重复计算的轮次
times = torch.zeros(iterations)  # 存储每轮 iteration 的时间

with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()

        # 同步 GPU 时间
        torch.cuda.synchronize()

        curr_time = starter.elapsed_time(ender)  # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f} ms, FPS: {:.2f}".format(mean_time, 1000/mean_time))