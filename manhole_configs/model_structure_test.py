import torch
from mmpretrain import get_model

# 获取模型
model = get_model('resnet50_8xb32_in1k', device='cpu')

# 创建一个字典来存储中间层的输出
outputs = {}

def hook(module, input, output):
    outputs[module] = output

# 注册钩子到模型的每个阶段
for name, module in model.backbone.named_children():
    if 'layer' in name:
        module.register_forward_hook(hook)

# 创建一个虚拟输入
x = torch.randn(1, 3, 224, 224)

# 前向传播
model.eval()
with torch.no_grad():
    _ = model(x)

# 打印输出的维度
for module, output in outputs.items():
    print(f"{module}: {output.shape}")

