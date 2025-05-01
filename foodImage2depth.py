import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

img_path = 'test1.jpg'
img = Image.open(img_path).convert('RGB')

# 转为 Tensor 并 padding 到 832x832
transform = T.ToTensor()
img_tensor = transform(img)  # [3, 513, 513]

# 计算 padding 尺寸（对称 pad）
pad_h = (832 - 513) // 2  # 159
pad_w = (832 - 513) // 2  # 159
img_padded = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h))  # [3, 832, 832]

# 加 batch 维
img_padded = img_padded.unsqueeze(0)  # [1, 3, 832, 832]

# 滑窗参数
patch_h, patch_w = 256, 832
stride_h = 128  # 可以调小更密集
_, _, H, W = img_padded.shape

# dummy 模型
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))  # 假设输出与输入大小一致

model = DummyModel()

# 保存路径
os.makedirs("model_inputs", exist_ok=True)

# 滑窗推理
depth_full = torch.zeros((1, 1, H, W))
count_map = torch.zeros((1, 1, H, W))

idx = 0
for y in range(0, H - patch_h + 1, stride_h):
    patch = img_padded[:, :, y:y+patch_h, :]  # shape: [1, 3, 256, 832]

    # 显示或保存 patch
    patch_to_show = patch[0].permute(1, 2, 0).numpy()  # [H, W, C]
    plt.imsave(f"patch_{idx}.png", patch_to_show)
    print(f"Patch {idx} saved: shape = {patch.shape}")

    # 模型推理（替换成你的模型）
    with torch.no_grad():
        depth_patch = model(patch)

    # 加入总图 & 记录累计次数
    depth_full[:, :, y:y+patch_h, :] += depth_patch
    count_map[:, :, y:y+patch_h, :] += 1

    idx += 1

# 平均融合滑窗结果
depth_final = depth_full / (count_map + 1e-6)

# 恢复原图区域 (从 padded 图里截取回 513×513 区域)
depth_final_cropped = depth_final[:, :, pad_h:pad_h+513, pad_w:pad_w+513]


from PIL import Image

# 加载原图
img_path = 'test1.jpg'
img = Image.open(img_path).convert('RGB')

# Resize 成 256 × 832（高 × 宽）
resized_img = img.resize((832, 256), resample=Image.BILINEAR)

# 保存结果
resized_img.save('resized_input_256x832.jpg')
print("保存完成: resized_input_256x832.jpg")
