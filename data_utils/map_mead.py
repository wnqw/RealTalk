# combined_dataset.py

import os
import cv2
import json
import torch
from torch.utils.data import Dataset

class CombinedVideoDataset(Dataset):
    """
    Loads frames from `long_video.mp4`,
    referencing `metadata.json` to get person_id, emotion, etc.
    """
    def __init__(
        self,
        video_path,
        metadata_path,
        downscale=1,
        skip=1,
        transform=None
    ):
        """
        Args:
            video_path (str): Path to the concatenated .mp4 file.
            metadata_path (str): Path to the metadata JSON from video_concat.py.
            downscale (int): Downscale factor for frames (1 => no downscale).
            skip (int): Only load every `skip` frames.
            transform (callable): Optional transform on the frame (e.g. ToTensor).
        """
        super().__init__()
        self.video_path = video_path
        self.metadata_path = metadata_path
        self.downscale = downscale
        self.skip = skip
        self.transform = transform

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Build label array: for each frame index => label dict
        self.frame_labels = [{} for _ in range(self.total_frames)]
        for entry in self.metadata:
            start = entry['frame_start']
            end = entry['frame_end']
            for fid in range(start, end + 1):
                self.frame_labels[fid] = {
                    'person_id': entry['person_id'],
                    'emotion': entry['emotion'],
                    'intensity': entry['intensity'],
                    'utterance_id': entry['utterance_id'],
                    'video_path' : entry['video_path']
                }

        # We'll skip frames with step = `skip`
        self.indices = list(range(0, self.total_frames, skip))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Downscale
        if self.downscale > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (w//self.downscale, h//self.downscale),
                interpolation=cv2.INTER_AREA
            )

        # Convert to float [0,1]
        frame = frame.astype('float32') / 255.0

        if self.transform:
            frame = self.transform(frame)

        # Retrieve labels
        lbl = self.frame_labels[frame_idx]
        sample = {
            'frame': frame,        # Typically [H, W, 3] or [C, H, W]
            'frame_idx': frame_idx,
            'person_id': lbl.get('person_id', 'NA'),
            'emotion': lbl.get('emotion', 'neutral'),
            'intensity': lbl.get('intensity', 0),
            'utterance_id': lbl.get('utterance_id', '000'),
        }
        return sample

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
