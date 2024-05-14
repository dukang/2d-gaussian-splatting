import os
import sys
from PIL import Image
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.render_utils import save_img_f32, save_img_u8
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

image_normal = Image.open("DSC05573_normal.png")
resolution = (int(1264), int(832))
resized_image_normal = PILtoTorch(image_normal, resolution)

# 定义旋转矩阵
rotation_matrix = torch.tensor(
    [
        [9.9978e-01, -1.9624e-02, 7.3681e-03, 0.0000e00],
        [2.0917e-02, 9.5678e-01, -2.9005e-01, 0.0000e00],
        [-1.3578e-03, 2.9014e-01, 9.5698e-01, 0.0000e00],
        [-3.3662e00, -3.2633e-01, 3.1848e00, 1.0000e00],
    ],
)

rotation_2 = torch.tensor(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)

# 将normal_map的维度从(3, 600, 800)变换为(3, 600*800)，以便进行矩阵乘法
normal_map_flat = resized_image_normal.view(3, -1)

# 转到-1,1
normal_map_flat = normal_map_flat * 2 - 1


rotation_matrix = rotation_matrix[:3, :3]
# 应用旋转矩阵self.R到每一个法线向量
normal_map_flat = torch.matmul(
    torch.tensor(rotation_2, dtype=torch.float32), normal_map_flat
)

normal_map_flat = torch.matmul(
    torch.tensor(rotation_matrix, dtype=torch.float32), normal_map_flat
)

# 将旋转后的法线图维度变回(3, 600, 800)
rotated_normal_map = normal_map_flat.view(resized_image_normal.shape)

# 保存图片
save_img_u8(
    rotated_normal_map.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5,
    "normal_1.png",
)
