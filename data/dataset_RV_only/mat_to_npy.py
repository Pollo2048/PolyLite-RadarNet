import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# =========================
# 默认参数
# =========================
FRAMES = 7475
CHIRPS = 17
RX = 2
ADC_SAMPLES = 64

RANGE_BINS = 32
CLIP_LEN = 64
STRIDE = 32

EPS = 1e-6


def find_main_array(mat_dict):
    """
    从 loadmat 返回结果中找到主数据数组
    """
    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            candidates.append((k, v))

    if not candidates:
        raise ValueError("在 .mat 文件中没有找到有效 ndarray 变量")

    candidates.sort(key=lambda x: x[1].size, reverse=True)
    return candidates[0][0], candidates[0][1]


def reshape_repetition(rep_1d: np.ndarray) -> np.ndarray:
    """
    把一个 repetition 的一维复数流 reshape 为:
    [frame, chirp, rx, adc_sample] = [7475, 17, 2, 64]
    """
    expected = FRAMES * CHIRPS * RX * ADC_SAMPLES
    rep_1d = np.asarray(rep_1d).reshape(-1)

    if rep_1d.size != expected:
        raise ValueError(
            f"repetition 长度不匹配: got {rep_1d.size}, expected {expected}"
        )

    cube = rep_1d.reshape(FRAMES, CHIRPS, RX, ADC_SAMPLES)
    return cube


def stationary_clutter_removal(cube: np.ndarray) -> np.ndarray:
    """
    静态杂波去除：按 frame 维求背景并相减
    """
    bg = cube.mean(axis=0, keepdims=True)
    return cube - bg


def compute_rv(cube: np.ndarray) -> np.ndarray:
    """
    输入:
        cube: [F, C, RX, A]
    输出:
        rv: [F, 32, 17]
    """
    # Range FFT
    range_win = np.hanning(ADC_SAMPLES).reshape(1, 1, 1, ADC_SAMPLES)
    cube_win = cube * range_win

    range_cube = np.fft.fft(cube_win, axis=-1)
    range_cube = range_cube[..., :RANGE_BINS]   # [F, C, RX, 32]

    # Doppler FFT
    doppler_win = np.hanning(CHIRPS).reshape(1, CHIRPS, 1, 1)
    range_cube = range_cube * doppler_win

    rd_cube = np.fft.fft(range_cube, axis=1)
    rd_cube = np.fft.fftshift(rd_cube, axes=1)  # [F, 17, RX, 32]

    # RX 聚合
    rv = np.mean(np.abs(rd_cube), axis=2)       # [F, 17, 32]
    rv = np.transpose(rv, (0, 2, 1))            # [F, 32, 17]

    return rv.astype(np.float32)


def split_clips(rv: np.ndarray, clip_len: int = CLIP_LEN, stride: int = STRIDE) -> np.ndarray:
    """
    输入:
        rv: [F, 32, 17]
    输出:
        clips: [N, 64, 32, 17]
    """
    total_frames = rv.shape[0]
    clips = []

    for start in range(0, total_frames - clip_len + 1, stride):
        clip = rv[start:start + clip_len]
        clips.append(clip)

    if len(clips) == 0:
        return np.empty((0, clip_len, rv.shape[1], rv.shape[2]), dtype=np.float32)

    return np.stack(clips, axis=0).astype(np.float32)


def normalize_per_clip(clips: np.ndarray) -> np.ndarray:
    """
    每个 clip 做:
        log1p -> z-score
    输入/输出:
        [N, 64, 32, 17]
    """
    clips = np.log1p(clips)

    mean = clips.mean(axis=(1, 2, 3), keepdims=True)
    std = clips.std(axis=(1, 2, 3), keepdims=True)

    clips = (clips - mean) / (std + EPS)
    return clips.astype(np.float32)


def to_uint8_for_original_loader(clips: np.ndarray) -> np.ndarray:
    """
    如果要兼容旧 dataloader 的 /255 逻辑，可转成 uint8
    """
    out = []
    for i in range(clips.shape[0]):
        x = clips[i]
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min + EPS)
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
        out.append(x)
    return np.stack(out, axis=0)


def save_clips(
    clips: np.ndarray,
    save_dir: Path,
    mat_stem: str,
    rep_idx: int,
    save_uint8: bool = False,
):
    """
    保存为单样本 npy:
        [64, 1, 32, 17]
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_uint8:
        clips = to_uint8_for_original_loader(clips)

    for i in range(clips.shape[0]):
        clip = clips[i]            # [64, 32, 17]
        clip = clip[:, None, :, :] # [64, 1, 32, 17]

        save_name = f"{mat_stem}_rep{rep_idx:02d}_clip{i:04d}.npy"
        np.save(save_dir / save_name, clip)


def process_one_mat(mat_path: Path, save_dir: Path, save_uint8: bool = False):
    print(f"\n[INFO] Processing: {mat_path}")

    mat = loadmat(mat_path)
    key, raw = find_main_array(mat)
    raw = np.asarray(raw)

    print(f"[INFO] Main variable: {key}, shape={raw.shape}, dtype={raw.dtype}")

    if raw.ndim != 2:
        raise ValueError(f"暂只支持二维主数组，当前 shape={raw.shape}")

    expected_len = FRAMES * CHIRPS * RX * ADC_SAMPLES

    # 兼容两种布局:
    # (num_rep, expected_len) 或 (expected_len, num_rep)
    if raw.shape[1] == expected_len:
        repetitions = raw
    elif raw.shape[0] == expected_len:
        repetitions = raw.T
    else:
        raise ValueError(
            f"无法识别 repetition 维度，raw.shape={raw.shape}, expected one dim == {expected_len}"
        )

    total_saved = 0

    for rep_idx in range(repetitions.shape[0]):
        rep = repetitions[rep_idx]
        cube = reshape_repetition(rep)
        cube = stationary_clutter_removal(cube)
        rv = compute_rv(cube)                        # [7475, 32, 17]
        clips = split_clips(rv, CLIP_LEN, STRIDE)   # [N, 64, 32, 17]
        clips = normalize_per_clip(clips)

        save_clips(
            clips=clips,
            save_dir=save_dir,
            mat_stem=mat_path.stem,
            rep_idx=rep_idx,
            save_uint8=save_uint8,
        )

        total_saved += clips.shape[0]
        print(f"[INFO] rep {rep_idx}: rv={rv.shape}, clips={clips.shape}")

    print(f"[DONE] {mat_path.name}: saved {total_saved} clips -> {save_dir}")


def process_one_fa_folder(fa_dir: Path, save_uint8: bool = False):
    """
    处理一个类别目录，例如:
        .../FA1
    读取:
        .../FA1/raw/*.mat
    保存到:
        .../FA1/rv_npy/
    """
    raw_dir = fa_dir / "raw"
    save_dir = fa_dir / "rv_npy"

    if not raw_dir.exists():
        print(f"[WARN] Skip {fa_dir.name}: raw 文件夹不存在 -> {raw_dir}")
        return

    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        print(f"[WARN] Skip {fa_dir.name}: raw 下没有 .mat 文件")
        return

    print(f"\n========== {fa_dir.name} ==========")
    print(f"[INFO] raw_dir : {raw_dir}")
    print(f"[INFO] save_dir: {save_dir}")
    print(f"[INFO] found {len(mat_files)} mat files")

    for mat_path in mat_files:
        try:
            process_one_mat(mat_path, save_dir=save_dir, save_uint8=save_uint8)
        except Exception as e:
            print(f"[ERROR] Failed on {mat_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=r"FA 数据根目录，例如 C:\_D_file\har_40_data\fa_raw"
    )
    parser.add_argument(
        "--save_uint8",
        action="store_true",
        help="若加此参数，则保存为 uint8；默认保存 float32"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"根目录不存在: {root}")

    # 仅处理 FA1 ~ FA8
    for i in range(1, 9):
        fa_dir = root / f"FA{i}"
        if not fa_dir.exists():
            print(f"[WARN] Skip FA{i}: 文件夹不存在 -> {fa_dir}")
            continue
        process_one_fa_folder(fa_dir, save_uint8=args.save_uint8)

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()