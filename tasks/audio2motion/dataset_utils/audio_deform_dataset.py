# mead_audio_deform_dataset.py
import sys
import tqdm
import os
import glob
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from utils.commons.indexed_datasets import IndexedDataset

class MeadAudioDeformDataset(Dataset):

    def __init__(self, prefix='train'):
        super().__init__()
        self.root_dir = '/mnt/sda4/mead_data/mead'
        self.emotion_short_map = {
                'ang': 'angry',
                'hap': 'happy',
                'sad': 'sad',
                'fea': 'fear',
                'dis': 'disgust',
                'sur': 'surprise',
                'neu': 'neutral',
                'con': 'contempt'
            }

        self.all_emotions = list(set(self.emotion_short_map.values()))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.all_emotions)

        self.pairs = []

        all_npy = glob.glob(os.path.join(self.root_dir, "**", "*_hubert.npy"), recursive=True)

        self.index_dict = {}
        pattern = re.compile(r"^(M\d+|W\d+)_([a-z]{3})_(\d+)_(\d+)_hubert\.npy$")

        for npy_path in all_npy:
            base = os.path.basename(npy_path)  # e.g. M003_ang_3_001_hu.npy
            match = pattern.match(base)
            if not match:
                print(f'{npy_path} dont match')
                sys.exit()
                continue
            subject = match.group(1)       # e.g. M003
            emo_code = match.group(2)     # e.g. ang
            intensity = match.group(3)    # e.g. 3
            utt = match.group(4)          # e.g. 001

            key = (subject, utt)

            if key not in self.index_dict:
                self.index_dict[key] = {}
            self.index_dict[key][emo_code] = npy_path

        for key, emo_map in self.index_dict.items():
            neu_path= None
            if 'neu' in emo_map:
                neu_path = emo_map['neu']
            for code, path_emo in emo_map.items():
                if code != 'neu' and neu_path:
                    emotion_str = self.emotion_short_map[code]
                    self.pairs.append((neu_path, path_emo, emotion_str))

        # done building
        # print(f"[INFO] Found {len(self.pairs)} (neutral,emotional) pairs in '{root_dir}'.")


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        neu_path, emo_path, emo_str = self.pairs[idx]

        # Load .npy files
        neutral_tensor = np.load(neu_path)   
        emotional_tensor = np.load(emo_path)   
# neutral_tensor (67, 16, 29), emotional_tensor (177, 16, 29)

        # print(f'neutral_tensor {neutral_tensor.shape}, emotional_tensor {emotional_tensor.shape}')
        # sys.exit()
        
        neutral_tensor   = torch.from_numpy(neutral_tensor).float() #.permute(0,2,1)
        emotional_tensor = torch.from_numpy(emotional_tensor).float() #.permute(0,2,1)     
# neutral_tensor torch.Size([294, 1024]), emotional_tensor torch.Size([234, 1024])
        
        
        neutral_tensor = neutral_tensor.reshape(neutral_tensor.shape[0], -1)[0] # 1024
        emotional_tensor = emotional_tensor.reshape(emotional_tensor.shape[0], -1)[0] # 1024
        # print(f'neutral_tensor {neutral_tensor.shape}, emotional_tensor {emotional_tensor.shape}')
        # sys.exit()



        # Get integer emotion label
        emo_id_arr = self.label_encoder.transform([emo_str])  # Shape: (1,)
        emo_label  =  torch.from_numpy(emo_id_arr) # [1]  

        # print(f'emo_id_arr {emo_label.shape}')
        # sys.exit()

        return {
        'neutral_auds': neutral_tensor,   # [512]
        'emotional_auds': emotional_tensor, # [512]
        'emotion_label': emo_label        # [1]
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None

        neutral_auds = torch.stack([s['neutral_auds'] for s in samples], dim=0)  # [B, auddim]
        emotional_auds = torch.stack([s['emotional_auds'] for s in samples], dim=0)  # [B, auddim]
        emotion_label = torch.tensor([s['emotion_label'] for s in samples], dtype=torch.long)  # [B]

        return {
            'neutral_auds': neutral_auds,
            'emotional_auds': emotional_auds,
            'emotion_label': emotion_label
        }


    def get_dataloader(self):
        loader = DataLoader(
            self,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collater
        )
        return loader


if __name__ == "__main__":
    # Example usage
    mead_path = "/mnt/sda4/mead_data/mead"  # Update this to your MEAD root directory
    dataset = MeadAudioDeformDataset(mead_path)
    loader = dataset.get_dataloader()

    for batch in loader:
        print("neutral_auds:",   batch['neutral_auds'].shape)    
        print("emotional_auds:", batch['emotional_auds'].shape)   
        print("emotion_label:",  batch['emotion_label'].shape)     
        break

# neutral_auds: torch.Size([1, 464])
# emotional_auds: torch.Size([1, 464])
# emotion_label: torch.Size([1])

    # print("\nProcessing dataset batches with tqdm:")
    # total_batches = len(loader)
    # pbar = tqdm.tqdm(total=total_batches, desc="Loading batches")

    # for batch in loader:
    #     # Simulate processing each batch
    #     # You can add your training/inference step here
    #     pbar.update(1)

    # pbar.close()

