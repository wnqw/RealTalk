import torch
import torch.nn as nn
import numpy as np
import os

from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np
from utils.nn.model_utils import print_arch, get_device_of_model, not_requires_grad
from utils.nn.schedulers import ExponentialSchedule
from utils.nn.grad import get_grad_norm
from utils.commons.meters import AvgrageMeter

from modules.audio_deformation.aud_deformation_res_att import ResAttAudioDeformationModel
from tasks.audio2motion.dataset_utils.audio_deform_dataset import MeadAudioDeformDataset

class AudioDeformationTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_cls = MeadAudioDeformDataset

    def build_model(self):
        self.model = ResAttAudioDeformationModel(
            aud_dim=1024,
            num_emotions=8,
            emotion_embedding_dim=16,
            hidden_dim=256,
            num_res_blocks=2,
            dropout_prob=0.0,
            use_attention=True,        # <-- NEW ARG: if True, add an attention block
            n_heads=4   
        )
        print_arch(self.model)
        return self.model

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2'])
        )
        return self.optimizer

    def build_scheduler(self, optimizer):
        return ExponentialSchedule(optimizer, hparams['lr'], hparams['warmup_updates'])

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix='train')
        self.train_dl = train_dataset.get_dataloader()
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        val_dataset = self.dataset_cls(prefix='val')
        self.val_dl = val_dataset.get_dataloader()
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix='test')
        self.test_dl = test_dataset.get_dataloader()
        return self.test_dl

    ##########################
    # Training and Validation
    ##########################
    def run_model(self, sample, infer=False):
        """
        Runs the model on the given sample.
        If infer is True, runs in evaluation mode.
        """
        neutral_auds = sample['neutral_auds'].to(self.device)
        emotion_label = sample['emotion_label'].squeeze().to(self.device)
        if infer:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(neutral_auds, emotion_label)
        else:
            self.model.train()
            outputs = self.model(neutral_auds, emotion_label)
        return outputs

    def _training_step(self, sample, batch_idx, optimizer_idx):
        neutral_auds = sample['neutral_auds'].to(self.device)
        emotional_auds = sample['emotional_auds'].to(self.device)
        emotion_label = sample['emotion_label'].squeeze().to(self.device)

        outputs = self.run_model(sample, infer=False)

        # Compute loss
        loss = self.mse_loss(outputs, emotional_auds)

        # Prepare outputs for logging
        loss_output = {
            'mse_loss': loss,
        }
        total_loss = loss

        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        neutral_auds = sample['neutral_auds'].to(self.device)
        emotional_auds = sample['emotional_auds'].to(self.device)
        emotion_label = sample['emotion_label'].squeeze().to(self.device)

        outputs = self.run_model(sample, infer=True)

        # Compute loss
        loss = self.mse_loss(outputs, emotional_auds)

        # Prepare outputs for logging
        outputs = {
            'losses': {
                'val_mse_loss': loss,
            },
            'nsamples': neutral_auds.size(0),
        }
        return outputs

    def validation_end(self, outputs):
        all_losses_meter = {'total_loss': AvgrageMeter()}
        for output in outputs:
            if output is None or len(output) == 0:
                continue
            losses = tensors_to_scalars(output['losses'])
            n = output.get('nsamples', 1)
            total_loss = sum(losses.values())
            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(total_loss, n)
        loss_output = {k: round(v.avg, 4) for k, v in all_losses_meter.items()}
        print(f"| Validation results at step {self.global_step}: {loss_output}")
        return {
            'tb_log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}')
        os.makedirs(self.gen_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        neutral_auds = sample['neutral_auds'].to(self.device)
        emotion_label = sample['emotion_label'].squeeze().to(self.device)

        outputs = self.run_model(sample, infer=True)

        # Save the outputs
        base_fn = f"generated_sample_{batch_idx}"
        self.save_result(outputs, base_fn, self.gen_dir)

        return {}

    def test_end(self, outputs):
        print(f"Generated samples saved in {self.gen_dir}")

    @staticmethod
    def save_result(emotional_auds, base_fname, gen_dir):
        emotional_auds_np = emotional_auds.cpu().numpy()
        np.save(os.path.join(gen_dir, f"{base_fname}.npy"), emotional_auds_np)

    #####################
    # Utility Functions
    #####################
    def mse_loss(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)
