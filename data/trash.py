import numpy as np

# 替换成你硬盘里任意一个真实存在的 .npy 文件路径
test_file = "./dataset/squat/01.npy"

data = np.load(test_file)
print(f"文件原始存储形状: {data.shape}")
print()