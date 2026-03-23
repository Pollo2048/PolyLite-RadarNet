import numpy as np

# 加载 npy 文件
data = np.load(r'C:\Users\zmh\Desktop\polylite-repo\PolyLite-RadarNet\data\dataset\standup\01.npy')

# 查看数据内容
print(data)

# 查看数据的维度（这对于雷达信号处理非常重要，比如确定距离多普勒图的尺寸）
print(data.shape)

# 查看数据类型（如 float32, int64 等）
print(data.dtype)