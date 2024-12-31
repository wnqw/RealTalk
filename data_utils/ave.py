import os
import cv2
import glob
import json
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from nerf_triplane.network import AudioEncoder
import trimesh
import sys
import torch
from torch.utils.data import DataLoader
from nerf_triplane.utils import get_audio_features, get_rays, get_bg_coords, AudDataset
from argparse import ArgumentParser





def get_model_and_device(ckpt_path: str = './nerf_triplane/checkpoints/audio_visual_encoder.pth'):
    """Load model, checkpoint, and return (model, device)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioEncoder().to(device).eval()

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    # The checkpoint keys are prefixed with "audio_encoder." in the script,
    # so we adapt them accordingly:
    model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()}, strict=False)
    
    return model, device

# ---------------------------------------------------------------------
# Processing function for a single WAV file
# ---------------------------------------------------------------------
def process_wav(wav_path, model, device, batch_size=64):
    """
    1. Create a dataset from the .wav file (AudDataset).
    2. Run inference with the audio model to get the embeddings.
    3. Post-process the embeddings to get aud_features.
    4. Save them as a .npy file, replacing the .wav extension.
    """
    # 1. Create dataset and dataloader
    dataset = AudDataset(wav_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    outputs = []
    # 2. Inference
    with torch.no_grad():
        for mel in data_loader:
            mel = mel.to(device)
            out = model(mel)
            outputs.append(out)

    # 3. Post-process the embeddings
    outputs = torch.cat(outputs, dim=0).cpu()
    # first_frame and last_frame
    first_frame, last_frame = outputs[:1], outputs[-1:]
    aud_features = torch.cat(
        [first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
        dim=0
    ).numpy()

    # 4. Save as .npy
    npy_path = wav_path.replace('.wav', '_ave.npy')
    np.save(npy_path, aud_features)

# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, help='Root folder containing .wav files', required=True)
    parser.add_argument('--ckpt', type=str, default='./nerf_triplane/checkpoints/audio_visual_encoder.pth',
                        help='Path to the audio encoder checkpoint')
    args = parser.parse_args()

    root_folder = args.root
    ckpt_path = args.ckpt

    # 1. Load model and device once
    model, device = get_model_and_device(ckpt_path)

    # 2. Gather all .wav files
    wav_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(subdir, file))

    # 3. Process each .wav file with a progress bar
    for wav_path in tqdm(wav_files, desc="Processing WAV files"):
        process_wav(wav_path, model, device)
        # print(wav_path)