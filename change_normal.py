import numpy as np
from PIL import Image

def normal_map_to_world_space(normal_map):
    normal_world_space = (normal_map - 128) / 128
    return normal_world_space

def world_space_to_normal_map(normal_world_space):
    normal_map = (normal_world_space * 128) + 128
    return normal_map


# 读取normal map
normal_map = Image.open("normal_00026.png")
normal_map_np = np.array(normal_map, dtype=np.float32)

# 将normal map转换为世界空间坐标
normal_world_space = normal_map_to_world_space(normal_map_np)

# 定义旋转矩阵
rotation_matrix = np.array([[-0.17111025, -0.5680907 ,  0.80498089],
                            [ 0.43227552,  0.69090543,  0.5794718 ],
                            [-0.88535821,  0.4471271 ,  0.1273507 ]])

# 旋转normal值
rotated_normal_world_space = np.einsum('ij,hwj->hwi', rotation_matrix, normal_world_space)

# 将旋转后的normal值转换回normal map
rotated_normal_map_np = world_space_to_normal_map(rotated_normal_world_space)

# 保存旋转后的normal map
rotated_normal_map = Image.fromarray(rotated_normal_map_np.astype(np.uint8))
rotated_normal_map.save("rotated_normal_map.png")