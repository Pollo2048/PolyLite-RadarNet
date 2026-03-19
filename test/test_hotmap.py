import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_npy_data(file_path, frame_idx=0):
    """
    可视化雷达 .npy 数据中的 RVP, RAP, REP 热力图
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    # 1. 加载数据
    # 原始形状通常是 (Frames, Channels, Height, Width)
    data = np.load(file_path)

    # 2. 转换维度为 (Frames, Height, Width, Channels)
    # 对应仓库 dataset.py 中的逻辑: data.transpose(0, 2, 3, 1)
    buffer = data.transpose(0, 2, 3, 1)

    print(f"数据形状: {buffer.shape} (帧数, 高度, 宽度, 通道数)")

    if frame_idx >= buffer.shape[0]:
        print(f"错误: 索引 {frame_idx} 超出总帧数 {buffer.shape[0]}")
        return

    # 3. 提取指定帧的三个视图 (RV, RA, RE)
    # 根据论文，通道 0, 1, 2 分别对应 RVP, RAP, REP
    rv_view = buffer[frame_idx, :, :, 0]
    ra_view = buffer[frame_idx, :, :, 1]
    re_view = buffer[frame_idx, :, :, 2]

    # 4. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"File: {os.path.basename(file_path)} - Frame: {frame_idx}", fontsize=16)

    # 绘制距离-速度图 (RV)
    im0 = axes[0].imshow(rv_view, aspect='auto', cmap='viridis')
    axes[0].set_title("Range-Velocity (RV)")
    axes[0].set_xlabel("Velocity")
    axes[0].set_ylabel("Range")
    plt.colorbar(im0, ax=axes[0])

    # 绘制距离-方位角图 (RA)
    im1 = axes[1].imshow(ra_view, aspect='auto', cmap='viridis')
    axes[1].set_title("Range-Azimuth (RA)")
    axes[1].set_xlabel("Azimuth Angle")
    axes[1].set_ylabel("Range")
    plt.colorbar(im1, ax=axes[1])

    # 绘制距离-仰角图 (RE)
    im2 = axes[2].imshow(re_view, aspect='auto', cmap='viridis')
    axes[2].set_title("Range-Elevation (RE)")
    axes[2].set_xlabel("Elevation Angle")
    axes[2].set_ylabel("Range")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

# 使用示例：替换为您本地的 .npy 文件路径
# file_to_show = "./data/dataset/walk/1.npy"
# visualize_npy_data(file_to_show, frame_idx=10)
# ... 您之前的函数定义 ...

if __name__ == '__main__':
    # 1. 请确保这里的路径指向您电脑上真实的 .npy 文件
    # 例如：'data/datasets/class1/1.npy'
    test_file = r'C:\Users\pollo\Desktop\new file\polylite-repo\PolyLite-RadarNet\data\dataset\squat\01.npy'

    # 2. 调用函数
    visualize_npy_data(test_file, frame_idx=10)

    # 3. 显式调用 plt.show() 确保窗口保持开启
    import matplotlib.pyplot as plt

    plt.show()