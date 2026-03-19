import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Dataset for loading video data from numpy files.
    Each video is expected to be stored as a .npy file containing frame sequences.
    """

    def __init__(self, directory, clip_len=64, crop_size=112):
        """
        Args:
            directory (str): Root directory containing video data
            clip_len (int): Number of frames per clip
            crop_size (int): Size of the frame crop
        """
        self.clip_len = clip_len
        self.crop_size = crop_size

        # Get all categories (subdirectories)
        self.categories = sorted(os.listdir(directory))
        self.label_map = {cat: idx for idx, cat in enumerate(self.categories)}

        # Get all video files
        self.video_files = []
        self.labels = []

        for category in self.categories:
            category_path = os.path.join(directory, category)
            video_files = glob.glob(os.path.join(category_path, "*.npy"))
            self.video_files.extend(video_files)
            self.labels.extend([self.label_map[category]] * len(video_files))

        # Save category mapping
        self._save_category_mapping()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (video, label) where label is index of the target class
        """
        # Load video data
        video_path = self.video_files[index]
        buffer = self._load_video(video_path)

        # Process frames
        buffer = self._crop(buffer)
        buffer = self._normalize(buffer)
        buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))

        # Get label
        label = self.labels[index]

        return buffer, label

    def __len__(self):
        """Returns the total number of video files"""
        return len(self.video_files)

    def _load_video(self, video_path):
        """Load video frames from .npy file"""
        try:
            # Load numpy file and transpose to (frames, height, width, channels)
            data = np.load(video_path)
            buffer = data.transpose(0, 2, 3, 1)
            return buffer
        except Exception as e:
            print(f"Error loading video file {video_path}: {str(e)}")
            # Return zero tensor of correct shape as fallback
            return np.zeros((self.clip_len, self.crop_size, self.crop_size, 3))

    def _crop(self, buffer):
        """Crop video frames to specified size"""
        # Assuming center crop
        if buffer.shape[1] > self.crop_size:
            h_offset = (buffer.shape[1] - self.crop_size) // 2
            w_offset = (buffer.shape[2] - self.crop_size) // 2
            buffer = buffer[:,
                     h_offset:h_offset + self.crop_size,
                     w_offset:w_offset + self.crop_size, :]
        return buffer

    def _normalize(self, buffer):
        """Normalize pixel values to float32 in range [0, 1]"""
        return buffer.astype(np.float32) / 255.0

    def _save_category_mapping(self):
        """Save category to label mapping for reference"""
        mapping_file = 'category_mapping.txt'
        with open(mapping_file, 'w') as f:
            f.write("Category to Label Mapping:\n")
            for category, label in self.label_map.items():
                f.write(f"{category}: {label}\n")

    def get_class_count(self):
        """Returns the number of classes"""
        return len(self.categories)


# Example usage:
if __name__ == "__main__":
    # Example directory structure:
    # data_dir/
    #   category1/
    #     video1.npy
    #     video2.npy
    #   category2/
    #     video3.npy
    #     video4.npy

    dataset = VideoDataset(
        directory="dataset",
        clip_len=64,
        crop_size=112
    )

    # Get a sample
    video, label = dataset[0]
    print(f"Video tensor shape: {video.shape}")
    print(f"Label: {label}")