import os
import re
import scipy.io as sio
import numpy as np

mat_path = r'C:\_D_file\har_40_data\fa_raw\FA1\raw\FA_20210407_lab_2_1_1.mat'
save_dir = r'C:\_D_file\har_40_data\processed_rv'
os.makedirs(save_dir, exist_ok=True)

filename = os.path.basename(mat_path)
m = re.match(r'FA_\d+_[^_]+_(\d+)_(\d+)_(\d+)\.mat', filename)
if m is None:
    raise ValueError(f'Filename format not recognized: {filename}')

action_label = int(m.group(1)) - 1
user_label = int(m.group(2)) - 1
set_idx = int(m.group(3))

print('filename:', filename)
print('action_label:', action_label)
print('user_label:', user_label)
print('set_idx:', set_idx)

mat = sio.loadmat(mat_path)
x = mat['raw_data']
print('raw_data shape:', x.shape)

all_clips = []
all_action_labels = []
all_user_labels = []
all_rep_idxs = []
all_clip_idxs = []

for rep_idx in range(x.shape[0]):
    print(f'\n=== rep_idx {rep_idx} start ===')
    rep = x[rep_idx]

    cube = rep.reshape(7475, 17, 2, 64)
    print('cube shape:', cube.shape)

    bg = cube.mean(axis=0, keepdims=True)
    cube_denoised = cube - bg
    print('denoise done')

    range_win = np.hanning(64).reshape(1, 1, 1, 64)
    range_cube = np.fft.fft(cube_denoised * range_win, axis=-1)
    range_cube = range_cube[..., :32]
    print('range fft done, shape:', range_cube.shape)

    doppler_win = np.hanning(17).reshape(1, 17, 1, 1)
    rd_cube = np.fft.fftshift(np.fft.fft(range_cube * doppler_win, axis=1), axes=1)
    print('doppler fft done, shape:', rd_cube.shape)

    rv = np.mean(np.abs(rd_cube), axis=2)
    rv = np.transpose(rv, (0, 2, 1))
    print('rv shape:', rv.shape)

    clip_len = 64
    stride = 32
    clips = []
    for start in range(0, rv.shape[0] - clip_len + 1, stride):
        clip = rv[start:start + clip_len]
        clips.append(clip)

    clips = np.stack(clips, axis=0)
    print('clips shape before norm:', clips.shape)

    clips = np.log1p(clips)
    mean_val = clips.mean()
    std_val = clips.std() + 1e-8
    clips = (clips - mean_val) / std_val
    print('norm done')

    for clip_idx in range(clips.shape[0]):
        all_clips.append(clips[clip_idx].astype(np.float32))
        all_action_labels.append(action_label)
        all_user_labels.append(user_label)
        all_rep_idxs.append(rep_idx)
        all_clip_idxs.append(clip_idx)

    print(f'=== rep_idx {rep_idx} done, accumulated samples: {len(all_clips)} ===')

all_clips = np.stack(all_clips, axis=0)
print('\nfinal x shape:', all_clips.shape)

save_path = os.path.join(save_dir, filename.replace('.mat', '_rv.npz'))
np.savez_compressed(
    save_path,
    x=all_clips,
    action_label=np.array(all_action_labels, dtype=np.int64),
    user_label=np.array(all_user_labels, dtype=np.int64),
    rep_idx=np.array(all_rep_idxs, dtype=np.int64),
    clip_idx=np.array(all_clip_idxs, dtype=np.int64),
    source_file=filename,
)

print('saved to:', save_path)
print('num samples:', len(all_action_labels))
print('action labels unique:', np.unique(all_action_labels))
print('user labels unique:', np.unique(all_user_labels))
print('rep_idx unique:', np.unique(all_rep_idxs))