import os
import tqdm
import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rich.console import Console
import tensorboardX  # optional

from deformation_nw import AudioFeatureDeformationModel
from deform_dataset import (
    MeadAudioDeformDataset,
    # mead_audio_deform_collate_fn,
)
from datetime import datetime
import sys

class AudioDeformTrainer:
    """
    A simplified Trainer class that:
      1) Directly trains the AudioFeatureDeformationModel to transform
         neutral audio features -> emotional audio features.
      2) Saves model checkpoints & logs to a workspace directory.
      3) Includes evaluate() and test() methods.
    """

    def __init__(
        self,
        name="deformation",
        workspace="model/deformation_ave/",
        device='cuda',
        lr=1e-3,
        max_epochs=3001,
        batch_size=1,
        num_workers=0,
        local_rank=0,
        fp16=False,
        use_tensorboard=False,
        save_interval_steps=1000,
    ):
        """
        Args:
          name (str): Name of this experiment.
          workspace (str): Directory for logs & checkpoints.
          device: torch.device or None => auto pick cuda if available.
          lr (float): learning rate
          max_epochs (int): number of training epochs
          batch_size (int): dataloader batch size
          num_workers (int): dataloader workers
          local_rank (int): for multi-GPU
          fp16 (bool): whether to use half precision
          use_tensorboard (bool): whether to log with tensorboard
          save_interval_steps (int): checkpoint save interval in steps
        """
        self.name = name
        self.workspace = workspace+"fixedloader"
        self.console = Console()

        # pick device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.local_rank = local_rank
        self.fp16 = fp16
        self.use_tensorboard = use_tensorboard
        self.save_interval_steps = save_interval_steps

        # create workspace
        os.makedirs(self.workspace, exist_ok=True)
        self.ckpt_dir = os.path.join(self.workspace, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # setup logging
        self.log_file = open(os.path.join(self.workspace, f"log_{self.name}.txt"), "a")
        self.log(f"Trainer initialized: {self.name} / {self.workspace}, device={self.device}")

        # other variables
        self.epoch = 0
        self.global_step = 0
        self.seed_everything(42)

        # placeholders
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.writer = None

    def log(self, msg):
        """Print & log to file"""
        self.console.print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def seed_everything(self, seed=42):
        """Seed everything for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def build_dataloader(self, data_root):
        """
        Build the training dataloader (no PCA).
        """
        dataset = MeadAudioDeformDataset(root_dir=data_root)
        # loader = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     collate_fn=dataset.collater
        # )
        loader = dataset.get_mead_audio_deform_loader(batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def build_model(self, T=1, freq_dim=512):
        """
        Build AudioFeatureDeformationModel with dimension = T*freq_dim.
        For example, if T=8 and freq_dim=512, then audio_feature_dim=4096.
        """
        audio_feature_dim = T * freq_dim
        num_emotions = 8  # or get from dataset label encoder
        model = AudioFeatureDeformationModel(
            audio_feature_dim=audio_feature_dim,
            num_emotions=num_emotions,
            emotion_embedding_dim=16,
            hidden_dim=256
        ).to(self.device)
        self.log(f"[INFO] Model created with audio_feature_dim={audio_feature_dim}")
        return model

    def train_loop(self, loader):
        """
        The main training loop for self.max_epochs on the provided loader.
        We'll flatten [B, T, freq_dim] => [B, T*freq_dim].
        """

        # Build the model for a known T=8, freq_dim=512 (change if needed)
        self.model = self.build_model(T=1, freq_dim=512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # (Optional) Setup TensorBoard
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.workspace, "tensorboard_logs")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)

        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            n_steps = 0

            loader_iter = iter(loader)
            pbar = tqdm.tqdm(total=len(loader), desc=f"Epoch {epoch}/{self.max_epochs}", ncols=100)

            while True:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                if batch is None:
                    continue

                # push to device
                neutral_auds = batch["neutral_auds"].to(self.device)    # [B, T, 512]
                emotional_auds = batch["emotional_auds"].to(self.device)
                emotion_label = batch["emotion_label"].to(self.device)   # [B]
                # torch.Size([1, 512]) torch.Size([1, 512]) torch.Size([1])
                
                # B, T, Fr = neutral_auds.shape  # e.g. B=4, T=8, Fr=512
                # # Flatten => [B, T*Fr] => e.g. [B, 4096]
                # neu_flat = neutral_auds.view(B, -1)
                # emo_flat = emotional_auds.view(B, -1)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    pred_emo = self.model(neutral_auds, emotion_label, delta_scale=1.0)
                    loss = F.mse_loss(pred_emo, emotional_auds)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                n_steps += 1
                self.global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)

                # save checkpoint every X steps
                if (self.global_step % self.save_interval_steps) == 0:
                    self.save_checkpoint(step=self.global_step, epoch=epoch, prefix="step")

                # optionally log to TensorBoard
                if self.use_tensorboard and (self.global_step % 50 == 0):
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)

            pbar.close()

            avg_loss = total_loss / max(1, n_steps)
            self.log(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")

            # Evaluate after each epoch (if you have a validation loader)
            # val_metrics = self.evaluate(val_loader, desc="Validation", epoch=epoch)
            # self.log(f"[Epoch {epoch}] val_mse={val_metrics['mse']:.4f}")

            # Save checkpoint at end of epoch
            self.save_checkpoint(step=self.global_step, epoch=epoch, prefix="epoch")

        self.log("[INFO] Training completed.")
        if self.writer:
            self.writer.close()
        self.log_file.close()

    def evaluate(self, loader, desc="Eval", epoch=0):
        """
        Evaluate the model on a given loader. 
        We compute an average MSE and return a dict with 'mse'.
        """
        self.model.eval()
        total_mse = 0.0
        total_count = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(loader, desc=desc, ncols=100)
            for batch in pbar:
                if batch is None:
                    continue
                neutral_auds = batch["neutral_auds"].to(self.device)
                emotional_auds = batch["emotional_auds"].to(self.device)
                emotion_label = batch["emotion_label"].to(self.device)

                # B, T, Fr = neutral_auds.shape
                # neu_flat = neutral_auds.view(B, -1)
                # emo_flat = emotional_auds.view(B, -1)

                # forward
                pred_emo = self.model(neutral_auds, emotion_label, delta_scale=1.0)
                mse_val = F.mse_loss(pred_emo, emotional_auds, reduction="sum").item()
                total_mse += mse_val
                total_count += B
            pbar.close()
        avg_mse = total_mse / (total_count if total_count > 0 else 1)
        self.log(f"[Eval] epoch={epoch}, mse={avg_mse:.4f}")
        return {"mse": avg_mse}

    def test(self, loader):
        """
        A 'test' method that runs the model on a test loader,
        collects predictions and computes average MSE.
        """
        self.model.eval()
        total_mse = 0.0
        total_count = 0
        predictions = []
        with torch.no_grad():
            pbar = tqdm.tqdm(loader, desc="Test", ncols=100)
            for batch in pbar:
                if batch is None:
                    continue
                neutral_auds = batch["neutral_auds"].to(self.device)
                emotional_auds = batch["emotional_auds"].to(self.device)
                emotion_label = batch["emotion_label"].to(self.device)

                # B, T, Fr = neutral_auds.shape
                # neu_flat = neutral_auds.view(B, -1)
                # emo_flat = emotional_auds.view(B, -1)

                pred_emo = self.model(neutral_auds, emotion_label, delta_scale=1.0)
                mse_val = F.mse_loss(pred_emo, emotional_auds, reduction="sum").item()
                total_mse += mse_val
                total_count += B
                predictions.append(pred_emo.cpu().numpy())

            pbar.close()
        avg_mse = total_mse / (total_count if total_count > 0 else 1)
        self.log(f"[Test] avg_mse={avg_mse:.4f}")
        return {"mse": avg_mse, "predictions": predictions}


    def save_checkpoint(self, step=0, epoch=0, prefix="latest"):
        """
        Saves a checkpoint with model + optimizer + step/epoch info.
        Only the latest checkpoint is saved as 'latest.pth'.
        """
        ckpt_filename = "latest.pth"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_filename)
        self.log(f"Saving checkpoint to {ckpt_path}")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        }
        torch.save(checkpoint, ckpt_path)

    def run_train(self, data_root):
        """
        Public method to build the loader, then run the training loop.
        """
        loader = self.build_dataloader(data_root)
        self.train_loop(loader)

    def run_test(self, data_root):
        """
        Simple test procedure after training. 
        You can build a separate test loader, or reuse the dataset with a different split.
        """
        loader = self.build_dataloader(data_root)
        test_res = self.test(loader)
        self.log(f"[INFO] Test done. MSE={test_res['mse']:.4f}")


if __name__ == '__main__':
    trainer = AudioDeformTrainer()
    trainer.run_train('/mnt/sda4/mead_data/mead')




# # audio_deform_trainer.py

# import os
# import tqdm
# import copy
# import time
# import numpy as np

# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# import joblib  # for saving/loading PCA
# from sklearn.decomposition import PCA

# from rich.console import Console
# from packaging import version as pver
# import tensorboardX

# from deformation_nw import AudioFeatureDeformationModel
# from deform_dataset import (
#     MeadAudioDeformDataset,
#     get_mead_audio_deform_loader,
#     mead_audio_deform_collate_fn,
# )


# class AudioDeformTrainer:
#     """
#     A Trainer class that:
#       1) Optionally fits PCA on the 1024-dim features to reduce them to 64-dim.
#       2) Trains AudioFeatureDeformationModel to transform neutral audio features -> emotional audio features.
#       3) Saves model checkpoints & logs to a workspace directory.
#       4) Includes evaluate() and test() methods.
#     """

#     def __init__(
#         self,
#         name="deformation",
#         workspace="model/deformation/",
#         device='cuda',
#         lr=1e-3,
#         max_epochs=1002,
#         batch_size=4,
#         num_workers=0,
#         n_components=64,
#         local_rank=0,
#         fp16=False,
#         use_tensorboard=False,
#         pca_path="pca_model.pkl",
#         save_interval_steps=1000,
#     ):
#         """
#         Args:
#           name (str): Name of this experiment.
#           workspace (str): Directory for logs & checkpoints.
#           device: torch.device or None => auto pick cuda if available.
#           lr (float): learning rate
#           max_epochs (int): number of training epochs
#           batch_size (int): dataloader batch size
#           num_workers (int): dataloader workers
#           n_components (int): number of principal components after PCA
#           local_rank (int): for multi-GPU
#           fp16 (bool): whether to use half precision
#           use_tensorboard (bool): whether to log with tensorboard
#           pca_path (str): file path to load/save the PCA model
#           save_interval_steps (int): checkpoint save interval in steps
#         """
#         self.name = name
#         self.workspace = workspace
#         self.console = Console()

#         # pick device
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)

#         self.lr = lr
#         self.max_epochs = max_epochs
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.n_components = n_components
#         self.local_rank = local_rank
#         self.fp16 = fp16
#         self.use_tensorboard = use_tensorboard
#         self.pca_path = pca_path
#         self.save_interval_steps = save_interval_steps  # <-- for saving checkpoints every X steps

#         # create workspace
#         os.makedirs(self.workspace, exist_ok=True)
#         self.ckpt_dir = os.path.join(self.workspace, "checkpoints")
#         os.makedirs(self.ckpt_dir, exist_ok=True)

#         # setup logging
#         self.log_file = open(os.path.join(self.workspace, f"log_{self.name}.txt"), "a")
#         self.log(f"Trainer initialized: {self.name} / {self.workspace}, device={self.device}")

#         # other variables
#         self.epoch = 0
#         self.global_step = 0
#         self.seed_everything(42)

#         # placeholders
#         self.model = None
#         self.optimizer = None
#         self.scaler = None
#         self.writer = None

#     def log(self, msg):
#         """Print & log to file"""
#         self.console.print(msg)
#         self.log_file.write(msg + "\n")
#         self.log_file.flush()

#     def seed_everything(self, seed=42):
#         """Seed everything for reproducibility."""
#         import random
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)

#     def fit_pca(self, data_root):
#         """
#         Fits a PCA model to reduce feature dimension from 1024 => n_components.
#         Saves it to self.pca_path if not found, else loads from file.
#         """
#         if os.path.exists(self.pca_path):
#             self.log(f"[INFO] Loading existing PCA from {self.pca_path}")
#             pca = joblib.load(self.pca_path)
#             return pca

#         self.log(f"[INFO] Fitting PCA (-> {self.n_components} comps) on dataset in {data_root} ...")
#         dataset = MeadAudioDeformDataset(root_dir=data_root)
#         loader = DataLoader(
#             dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=mead_audio_deform_collate_fn
#         )
#         all_features = []
#         for batch in loader:
#             if batch is None:
#                 continue
#             neu = batch["neutral_auds"]  # [B, T, 1024]
#             emo = batch["emotional_auds"]  # [B, T, 1024]
#             B, T, Fr = neu.shape

#             # flatten => [B*T, F]
#             neu_flat = neu.view(-1, Fr).cpu().numpy()
#             emo_flat = emo.view(-1, Fr).cpu().numpy()

#             all_features.append(neu_flat)
#             all_features.append(emo_flat)

#         all_features = np.concatenate(all_features, axis=0)
#         self.log(f"[PCA] All features shape for PCA: {all_features.shape}")
#         pca = PCA(n_components=self.n_components)
#         pca.fit(all_features)

#         joblib.dump(pca, self.pca_path)
#         self.log(f"[INFO] PCA saved to {self.pca_path}")
#         return pca

#     def build_dataloader(self, data_root):
#         """
#         Build the training dataloader with integrated PCA or a collate that applies PCA.
#         """
#         # If you haven't fit PCA yet, do so
#         pca = self.fit_pca(data_root)

#         # Build dataset (and do PCA in collate or in dataset's __getitem__)
#         dataset = MeadAudioDeformDataset(root_dir=data_root)  # no pca here
#         # We'll do a collate function that transforms with pca
#         def collate_with_pca(batch):
#             out = mead_audio_deform_collate_fn(batch)
#             if out is None:
#                 return None
#             neu = out["neutral_auds"]  # [B, T, 1024]
#             emo = out["emotional_auds"]  # [B, T, 1024]
#             B, T, Fr = neu.shape
#             # flatten => [B*T, F]
#             neu_flat = neu.view(B * T, Fr).cpu().numpy()
#             emo_flat = emo.view(B * T, Fr).cpu().numpy()

#             # apply pca => [B*T, n_components]
#             neu_pca = pca.transform(neu_flat)
#             emo_pca = pca.transform(emo_flat)

#             # reshape => [B, T, n_components]
#             neu_torch = torch.from_numpy(neu_pca).float().reshape(B, T, self.n_components)
#             emo_torch = torch.from_numpy(emo_pca).float().reshape(B, T, self.n_components)

#             out["neutral_auds"] = neu_torch
#             out["emotional_auds"] = emo_torch
#             return out

#         loader = DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             collate_fn=collate_with_pca
#         )
#         return loader

#     def build_model(self, max_T=158):
#         """
#         Build AudioFeatureDeformationModel. The dimension => max_T * n_components.
#         """
#         audio_feature_dim = max_T * self.n_components
#         num_emotions = 8  # or get from dataset label_encoder
#         model = AudioFeatureDeformationModel(
#             audio_feature_dim=audio_feature_dim,
#             num_emotions=num_emotions,
#             emotion_embedding_dim=16,
#             hidden_dim=256
#         ).to(self.device)
#         self.log(f"[INFO] Model created with audio_feature_dim={audio_feature_dim}")
#         return model

#     def train_loop(self, loader):
#         """
#         The main training loop for self.max_epochs on the provided loader.
#         We'll flatten [B, T, n_components] => [B, T*n_components].
#         Also includes evaluating after each epoch, and checkpointing.
#         """

#         max_T = 158  # <--- fix the max time dimension
#         audio_feature_dim = max_T * self.n_components
#         self.model = self.build_model(max_T=max_T)  # ensures input_dim = audio_feature_dim + 16
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

#         if self.use_tensorboard:
#             from torch.utils.tensorboard import SummaryWriter
#             tb_dir = os.path.join(self.workspace, "tensorboard_logs")
#             os.makedirs(tb_dir, exist_ok=True)
#             self.writer = SummaryWriter(tb_dir)

#         for epoch in range(1, self.max_epochs + 1):
#             self.epoch = epoch
#             self.model.train()
#             total_loss = 0.0
#             n_steps = 0

#             loader_iter = iter(loader)
#             pbar = tqdm.tqdm(total=len(loader), desc=f"Epoch {epoch}/{self.max_epochs}", ncols=100)

#             while True:
#                 try:
#                     batch = next(loader_iter)
#                 except StopIteration:
#                     break
#                 if batch is None:
#                     continue

#                 # push to device
#                 neutral_auds = batch["neutral_auds"].to(self.device)  # [B, T, n_components]
#                 emotional_auds = batch["emotional_auds"].to(self.device)
#                 emotion_label = batch["emotion_label"].to(self.device)

#                 B, T, Fr = neutral_auds.shape
#                 # 1) If T > 158, truncate
#                 if T > max_T:
#                     neutral_auds = neutral_auds[:, :max_T, :]     # [B, 158, F]
#                     emotional_auds = emotional_auds[:, :max_T, :] # [B, 158, F]
#                     T = max_T
#                 # (Optional) If T < 158, you could also pad up to 158
#                 if T < max_T:
#                     pad_len = max_T - T
#                     pad_shape = (B, pad_len, Fr)
#                     neutral_pad = torch.zeros(pad_shape, dtype=neutral_auds.dtype, device=neutral_auds.device)
#                     emotional_pad = torch.zeros(pad_shape, dtype=emotional_auds.dtype, device=emotional_auds.device)
#                     neutral_auds = torch.cat([neutral_auds, neutral_pad], dim=1)
#                     emotional_auds = torch.cat([emotional_auds, emotional_pad], dim=1)
#                     T = max_T

#                 # 2) Flatten => [B, T*F] => [B, 158*64]
#                 neu_flat = neutral_auds.view(B, -1)
#                 emo_flat = emotional_auds.view(B, -1)

#                 with torch.cuda.amp.autocast(enabled=self.fp16):
#                     pred_emo = self.model(neu_flat, emotion_label, delta_scale=1.0)
#                     loss = F.mse_loss(pred_emo, emo_flat)

#                 self.scaler.scale(loss).backward()
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#                 self.optimizer.zero_grad()

#                 total_loss += loss.item()
#                 n_steps += 1
#                 self.global_step += 1

#                 pbar.set_postfix({"loss": f"{loss.item():.4f}"})
#                 pbar.update(1)

#                 # save checkpoint every X steps
#                 if (self.global_step % self.save_interval_steps) == 0:
#                     self.save_checkpoint(step=self.global_step, epoch=epoch, prefix="step")

#                 # optionally log to TB
#                 if self.use_tensorboard and (self.global_step % 50 == 0):
#                     self.writer.add_scalar("train/loss", loss.item(), self.global_step)

#             pbar.close()

#             avg_loss = total_loss / max(1, n_steps)
#             self.log(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")

#             # Evaluate after each epoch
#             # val_metrics = self.evaluate(loader, desc="Validation", epoch=epoch)
#             # self.log(f"[Epoch {epoch}] val_mse={val_metrics['mse']:.4f}")

#             # Save checkpoint at end of epoch
#             self.save_checkpoint(step=self.global_step, epoch=epoch, prefix="epoch")

#         self.log("[INFO] Training completed.")
#         if self.writer:
#             self.writer.close()
#         self.log_file.close()

#     def evaluate(self, loader, desc="Eval", epoch=0):
#         """
#         Evaluate the model on a given loader. 
#         We compute an average MSE and return a dict with 'mse' for logging.
#         """
#         self.model.eval()
#         total_mse = 0.0
#         total_count = 0
#         with torch.no_grad():
#             pbar = tqdm.tqdm(loader, desc=desc, ncols=100)
#             for batch in pbar:
#                 if batch is None:
#                     continue
#                 neutral_auds = batch["neutral_auds"].to(self.device)
#                 emotional_auds = batch["emotional_auds"].to(self.device)
#                 emotion_label = batch["emotion_label"].to(self.device)

#                 B, T, _ = neutral_auds.shape
#                 neu_flat = neutral_auds.view(B, -1)
#                 emo_flat = emotional_auds.view(B, -1)

#                 # forward
#                 pred_emo = self.model(neu_flat, emotion_label, delta_scale=1.0)
#                 mse_val = F.mse_loss(pred_emo, emo_flat, reduction="sum").item()
#                 total_mse += mse_val
#                 total_count += B
#             pbar.close()
#         avg_mse = total_mse / (total_count if total_count > 0 else 1)
#         self.log(f"[Eval] epoch={epoch}, mse={avg_mse:.4f}")
#         return {"mse": avg_mse}

#     def test(self, loader):
#         """
#         A simple 'test' method that runs the model on a test loader,
#         collects predictions and computes an average MSE (or store them).
#         """
#         self.model.eval()
#         total_mse = 0.0
#         total_count = 0
#         predictions = []  # if you want to store them
#         with torch.no_grad():
#             pbar = tqdm.tqdm(loader, desc="Test", ncols=100)
#             for batch in pbar:
#                 if batch is None:
#                     continue
#                 neutral_auds = batch["neutral_auds"].to(self.device)
#                 emotional_auds = batch["emotional_auds"].to(self.device)
#                 emotion_label = batch["emotion_label"].to(self.device)

#                 B, T, _ = neutral_auds.shape
#                 neu_flat = neutral_auds.view(B, -1)
#                 emo_flat = emotional_auds.view(B, -1)

#                 pred_emo = self.model(neu_flat, emotion_label, delta_scale=1.0)
#                 # compute MSE
#                 mse_val = F.mse_loss(pred_emo, emo_flat, reduction="sum").item()
#                 total_mse += mse_val
#                 total_count += B

#                 # store predictions if needed
#                 predictions.append(pred_emo.cpu().numpy())

#             pbar.close()
#         avg_mse = total_mse / (total_count if total_count > 0 else 1)
#         self.log(f"[Test] avg_mse={avg_mse:.4f}")
#         return {"mse": avg_mse, "predictions": predictions}

#     def save_checkpoint(self, step=0, epoch=0, prefix="epoch"):
#         """
#         Saves a checkpoint with model + optimizer + step/epoch info.
#         """
#         ckpt_path = os.path.join(self.ckpt_dir, f"{prefix}_ckpt_{self.name}_{step}.pth")
#         self.log(f"Saving checkpoint to {ckpt_path}")
#         checkpoint = {
#             "model": self.model.state_dict(),
#             "optimizer": self.optimizer.state_dict(),
#             "step": step,
#             "epoch": epoch,
#         }
#         torch.save(checkpoint, ckpt_path)

#     def run_train(self, data_root):
#         """
#         Public method to build the loader, then run the training loop.
#         """
#         loader = self.build_dataloader(data_root)
#         self.train_loop(loader)

#     def run_test(self, data_root):
#         """
#         Simple test procedure after training. You can build a separate test loader, or reuse the dataset with a different split.
#         """
#         loader = self.build_dataloader(data_root)
#         test_res = self.test(loader)
#         self.log(f"[INFO] Test done. MSE={test_res['mse']:.4f}")
#         # further usage of test_res if needed


# if __name__ == '__main__':
#         """
#         Full pipeline:
#           1) build loader with PCA
#           2) train loop
#         """
#         trainer = AudioDeformTrainer()
#         trainer.run_train('/mnt/sda4/mead_data/mead')




