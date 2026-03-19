import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_npy_single(file_path, frame_idx=10):
    """
    可视化单个 .npy 文件的 TRIview 热力图
    """
    try:
        # 加载并转置数据
        data = np.load(file_path)
        buffer = data.transpose(0, 2, 3, 1)

        # 提取 RV, RA, RE 视图 [cite: 214, 215]
        rv_view = buffer[frame_idx, :, :, 0]
        ra_view = buffer[frame_idx, :, :, 1]
        re_view = buffer[frame_idx, :, :, 2]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # 获取文件名用于标题显示
        file_name = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        fig.suptitle(f"Action: {parent_dir} | File: {file_name}", fontsize=14)

        axes[0].imshow(rv_view, aspect='auto', cmap='viridis')
        axes[0].set_title("Range-Velocity (RV)")

        axes[1].imshow(ra_view, aspect='auto', cmap='viridis')
        axes[1].set_title("Range-Azimuth (RA)")

        axes[2].imshow(re_view, aspect='auto', cmap='viridis')
        axes[2].set_title("Range-Elevation (RE)")

        plt.tight_layout()
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def auto_browse_dataset(root_path):
    """
    自动遍历文件夹并顺序显示
    """
    npy_files = []
    # 递归查找所有 .npy 文件
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".npy"):
                full_path = os.path.join(root, file)
                npy_files.append(full_path)

    # 按路径名称排序，确保显示顺序一致
    npy_files.sort()

    if not npy_files:
        print(f"在路径 {root_path} 下未找到 .npy 文件，请检查路径。")
        return

    print(f"共发现 {len(npy_files)} 个文件。")
    print("操作提示：关闭当前弹出的图片窗口，即可查看下一个文件。")

    for i, file_path in enumerate(npy_files):
        print(f"[{i + 1}/{len(npy_files)}] 正在显示: {file_path}")
        visualize_npy_single(file_path)
        plt.show()  # 这里会暂停程序，直到你手动关闭窗口


if __name__ == '__main__':
    # 使用原始字符串 r'' 防止 Windows 路径转义错误
    # 请根据你的实际存放位置修改此处的根目录路径
    dataset_root = r'C:\Users\zmh\Desktop\polylite-repo\PolyLite-RadarNet\data\dataset'

    auto_browse_dataset(dataset_root)