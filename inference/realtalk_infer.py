import os
import sys
sys.path.append('./')
from argparse import Namespace
import torch
import torch.nn.functional as F
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import uuid
import traceback
# common utils
from utils.commons.hparams import hparams, set_hparams, set_hparams2, hparams2, set_hparams3, hparams3
from utils.commons.tensor_utils import move_to_cuda, convert_to_np, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478, index_lm131_from_lm478
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from utils.commons.pitch_utils import f0_to_coarse
# Dataset Related
from tasks.radnerfs.dataset_utils import RADNeRFDataset, get_boundary_mask, dilate_boundary_mask, get_lf_boundary_mask
# Method Related
from modules.audio2motion.vae import PitchContourVAEModel, VAEModel
from modules.lm_deformation.lm_deformation import LandmarkDeformationModel
from modules.lm_deformation.lm_deformation_res_att import ResAttLandmarkDeformationModel

# ablations



from modules.postnet.lle import compute_LLE_projection, find_k_nearest_neighbors
from modules.radnerfs.utils import get_audio_features, get_rays, get_bg_coords, convert_poses, nerf_matrix_to_ngp
from modules.radnerfs.radnerf import RADNeRF
from modules.radnerfs.radnerf_sr import RADNeRFwithSR
from modules.radnerfs.radnerf_torso import RADNeRFTorso
from modules.radnerfs.radnerf_torso_sr import RADNeRFTorsowithSR

from modules.syncnerfs.syncnerf import SyncNeRF
from modules.syncnerfs.provider import SyncNeRFDataset
# from modules.syncnerfs.provider_ori import SyncNeRFDataset
from modules.syncnerfs.utils import *

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')
    

import audioop
import wave
import soundfile as sf
face3d_helper = None

from sklearn.preprocessing import LabelEncoder

import sys

def vis_cano_lm3d_to_imgs(cano_lm3d, hw=512):
    cano_lm3d_ = cano_lm3d[:1, ].repeat([len(cano_lm3d),1,1])
    cano_lm3d_[:, 17:27] = cano_lm3d[:, 17:27] # brow
    cano_lm3d_[:, 36:48] = cano_lm3d[:, 36:48] # eye
    cano_lm3d_[:, 27:36] = cano_lm3d[:, 27:36] # nose
    cano_lm3d_[:, 48:68] = cano_lm3d[:, 48:68] # mouth
    cano_lm3d_[:, 0:17] = cano_lm3d[:, :17] # yaw
    
    cano_lm3d = cano_lm3d_

    cano_lm3d = convert_to_np(cano_lm3d)

    WH = hw
    cano_lm3d = (cano_lm3d * WH/2 + WH/2).astype(int)
    frame_lst = []
    for i_img in range(len(cano_lm3d)):
        # lm2d = cano_lm3d[i_img ,:, 1:] # [68, 2]
        lm2d = cano_lm3d[i_img ,:, :2] # [68, 2]
        img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            # color = (255,0,0)
            color = (0, 0, 255)
            img = cv2.circle(img, center=(x,y), radius=3, color=color, thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.flip(img, 0)
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            y = WH - y
            img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
        frame_lst.append(img)
    return frame_lst

def inject_blink_to_lm68(lm68, opened_eye_area_percent=0.6, closed_eye_area_percent=0.15):
    # [T, 68, 2]
    # lm68[:,36:48] = lm68[0:1,36:48].repeat([len(lm68), 1, 1])
    opened_eye_lm68 = copy.deepcopy(lm68)
    eye_area_percent = 0.6 * torch.ones([len(lm68), 1], dtype=opened_eye_lm68.dtype, device=opened_eye_lm68.device)

    eye_open_scale = (opened_eye_lm68[:, 41, 1] - opened_eye_lm68[:, 37, 1]) + (opened_eye_lm68[:, 40, 1] - opened_eye_lm68[:, 38, 1]) + (opened_eye_lm68[:, 47, 1] - opened_eye_lm68[:, 43, 1]) + (opened_eye_lm68[:, 46, 1] - opened_eye_lm68[:, 44, 1])
    eye_open_scale = eye_open_scale.abs()
    idx_largest_eye = eye_open_scale.argmax()
    # lm68[:, list(range(17,27))] = lm68[idx_largest_eye:idx_largest_eye+1, list(range(17,27))].repeat([len(lm68),1,1])
    # lm68[:, list(range(36,48))] = lm68[idx_largest_eye:idx_largest_eye+1, list(range(36,48))].repeat([len(lm68),1,1])
    lm68[:,[37,38,43,44],1] = lm68[:,[37,38,43,44],1] + 0.03
    lm68[:,[41,40,47,46],1] = lm68[:,[41,40,47,46],1] - 0.03
    closed_eye_lm68 = copy.deepcopy(lm68)
    closed_eye_lm68[:,37] = closed_eye_lm68[:,41] = closed_eye_lm68[:,36] * 0.67 + closed_eye_lm68[:,39] * 0.33
    closed_eye_lm68[:,38] = closed_eye_lm68[:,40] = closed_eye_lm68[:,36] * 0.33 + closed_eye_lm68[:,39] * 0.67
    closed_eye_lm68[:,43] = closed_eye_lm68[:,47] = closed_eye_lm68[:,42] * 0.67 + closed_eye_lm68[:,45] * 0.33
    closed_eye_lm68[:,44] = closed_eye_lm68[:,46] = closed_eye_lm68[:,42] * 0.33 + closed_eye_lm68[:,45] * 0.67
    
    T = len(lm68)
    period = 100
    # blink_factor_lst = np.array([0.4, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.4]) # * 0.9
    # blink_factor_lst = np.array([0.4, 0.7, 0.8, 1.0, 0.8, 0.6, 0.4]) # * 0.9
    blink_factor_lst = np.array([0.1, 0.5, 0.7, 1.0, 0.7, 0.5, 0.1]) # * 0.9
    dur = len(blink_factor_lst)
    for i in range(T):
        if (i + 25) % period == 0:
            for j in range(dur):
                idx = i+j
                if idx > T - 1: # prevent out of index error
                    break
                blink_factor = blink_factor_lst[j]
                lm68[idx, 36:48] = lm68[idx, 36:48] * (1-blink_factor) + closed_eye_lm68[idx, 36:48] * blink_factor
                eye_area_percent[idx] = opened_eye_area_percent * (1-blink_factor) + closed_eye_area_percent * blink_factor
    return lm68, eye_area_percent


class RealTalk2Infer:
    def __init__(self, audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, lm_deform_ckpt, inp, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.audio2secc_dir = audio2secc_dir
        self.postnet_dir = postnet_dir
        self.head_model_dir = head_model_dir
        self.torso_model_dir = torso_model_dir
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.postnet_model = self.load_postnet(postnet_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir)
        self.audio2secc_model.to(device).eval()
        if self.postnet_model is not None:
            self.postnet_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=True)
        hparams['infer_smooth_camera_path_kernel_size'] = 7

        # emotion
        self.label_encoder = LabelEncoder()
        self.emotions = ['angry', 'disgust', 'contempt', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.label_encoder.fit(self.emotions)

        self.secc2deform_model = self.load_secc2deform(lm_deform_ckpt)
        self.secc2deform_model.to(device).eval()

        self.inp = inp
        self.syncnerf = self.load_syncnerfs()
        opt = Namespace(**self.inp)
        self.test_set = SyncNeRFDataset(opt, device=self.device, type='trainval')
        self.test_set.training = False 
        self.test_set.num_rays = -1


    def load_syncnerfs(self):
        opt = Namespace(**self.inp)
        model = SyncNeRF(opt)

        if opt.torso and opt.syncnerf_head_ckpt != '':
            model_dict = torch.load(opt.syncnerf_head_ckpt, map_location='cpu')['model']

            missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

            if len(missing_keys) > 0:
                print(f"[WARN] missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"[WARN] unexpected keys: {unexpected_keys}")   

            # freeze these keys
            for k, v in model.named_parameters():
                if k in model_dict:
                    print(f'[INFO] freeze {k}, {v.shape}')
                    v.requires_grad = False

        return model


    def load_audio2secc(self, audio2secc_dir):
        set_hparams(f"{os.path.dirname(audio2secc_dir) if os.path.isfile(audio2secc_dir) else audio2secc_dir}/config.yaml")
        self.audio2secc_hparams = copy.deepcopy(hparams)

        self.in_out_dim =  64 # 64, 204
        audio_in_dim = 1024
        self.model = PitchContourVAEModel(hparams, in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim)
        load_ckpt(self.model, f"{audio2secc_dir}", model_name='model', strict=True)
        return self.model

    
    def load_secc2deform(self, secc2deform_dir):
        set_hparams3(f"{secc2deform_dir}/config.yaml")
        self.secc2deform_hparams = copy.deepcopy(hparams3)

        # self.model3 = LandmarkDeformationModel()
        self.model3 = ResAttLandmarkDeformationModel()
        load_ckpt(self.model3, f"{secc2deform_dir}", model_name='model', strict=True)
        return self.model3


    def load_postnet(self, postnet_dir):
        if postnet_dir == '':
            return None
        from modules.postnet.models import PitchContourCNNPostNet
        set_hparams(f"{os.path.dirname(postnet_dir) if os.path.isfile(postnet_dir) else postnet_dir}/config.yaml")
        self.postnet_hparams = copy.deepcopy(hparams)
        in_out_dim = 68*3
        pitch_dim = 128
        self.model = PitchContourCNNPostNet(in_out_dim=in_out_dim, pitch_dim=pitch_dim)
        load_ckpt(self.model, f"{postnet_dir}", steps=20000, model_name='model', strict=True)
        return self.model

    def load_secc2video(self, head_model_dir, torso_model_dir):

        if torso_model_dir != '':
            set_hparams(f"{os.path.dirname(torso_model_dir) if os.path.isfile(torso_model_dir) else torso_model_dir}/config.yaml")
            self.secc2video_hparams = copy.deepcopy(hparams)
            if hparams.get("with_sr"):
                model = RADNeRFTorsowithSR(hparams)
            else:
                model = RADNeRFTorso(hparams)
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=True)
        else:
            set_hparams(f"{os.path.dirname(head_model_dir) if os.path.isfile(head_model_dir) else head_model_dir}/config.yaml")
            self.secc2video_hparams = copy.deepcopy(hparams)
            if hparams.get("with_sr"):
                model = RADNeRFwithSR(hparams)
            else:
                model = RADNeRF(hparams)
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
        self.dataset_cls = RADNeRFDataset # the dataset only provides head pose 
        self.dataset = self.dataset_cls('trainval', training=False)
        eye_area_percents = torch.tensor(self.dataset.eye_area_percents)
        self.closed_eye_area_percent = torch.quantile(eye_area_percents, q=0.03).item()
        self.opened_eye_area_percent = torch.quantile(eye_area_percents, q=0.97).item()
        try:
            model = torch.compile(model)
        except:
            traceback.print_exc()
        return model

    def infer_once(self, inp):
        self.inp = inp
        samples = self.prepare_batch_from_inp(inp)
        self.forward_system(samples, inp) 
        # return out_name
        
    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        # print(self.dataset.ds_dict.keys())
        sample = {}
        # emotion
        emotion = inp['emotion']
        print(f'emotion text label: {emotion}')
        if inp['emotion'] != 'none':
            text_emotion = [inp['emotion']]
            int_emo_encoded = self.label_encoder.transform(text_emotion)
            sample['emotion'] = torch.tensor(int_emo_encoded, dtype=torch.long).unsqueeze(0).to(self.device)

        # Process Audio
        self.save_wav16k(inp['drv_audio_name'])
        hubert = self.get_hubert(self.wav16k_name)
        t_x = hubert.shape[0]
        x_mask = torch.ones([1, t_x]).float() # mask for audio frames
        y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
        f0 = self.get_f0(self.wav16k_name)
        if f0.shape[0] > len(hubert):
            f0 = f0[:len(hubert)]
        else:
            num_to_pad = len(hubert) - len(f0)
            f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))

        sample.update({
            'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
            'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
            'x_mask': x_mask.cuda(),
            'y_mask': y_mask.cuda(),
            })

        sample['audio'] = sample['hubert']
        sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()

        sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
        sample['mouth_amp'] = torch.ones([1, 1]).cuda() * float(inp['mouth_amp'])
        # sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:t_x//2]).cuda()
        sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:1]).cuda().repeat([t_x//2, 1])
        pose_lst = []
        euler_lst = []
        trans_lst = []
        rays_o_lst = []
        rays_d_lst = []

        # print(f"drv_pose {inp['drv_pose']}")
        # sys.exit()

        for i in range(t_x//2):
            euler = torch.tensor(self.dataset.ds_dict['euler'][i])
            trans = torch.tensor(self.dataset.ds_dict['trans'][i])

            pose = self.test_set.poses[i].to(self.device)
            pose_lst.append(pose)
            euler_lst.append(euler)
            trans_lst.append(trans)
        sample['poses'] = pose_lst # 80
        sample['euler'] = torch.stack(euler_lst).cuda()
        sample['trans'] = torch.stack(trans_lst).cuda()
        sample['bg_img'] = self.dataset.bg_img.reshape([1,-1,3]).cuda()
        sample['bg_coords'] = self.dataset.bg_coords.cuda()
        return sample

    def find_min_max_gap(self, tensor):
        # Compute cosine similarities between each contiguous pair of poses
        similarities = F.cosine_similarity(tensor[:-1].flatten(start_dim=1), tensor[1:].flatten(start_dim=1), dim=1)

        # Determine idle stretches (where similarity is high)
        thresholds = torch.ones_like(similarities)

        # Compare each element in similarities to the corresponding threshold
        is_idle = (similarities >= thresholds)

        # Find lengths of idle stretches
        idle_lengths = []
        current_length = 0
        min_idle_length = 0
        max_idle_length = 0
        avg_gap_length = 0
        idle_nums = 0

        for idle in is_idle:
            if idle:
                current_length += 1
            elif current_length > 0:
                idle_lengths.append(current_length)
                current_length = 0
        if current_length > 0:  # Append the last segment if it's idle
            idle_lengths.append(current_length)

        # Calculate statistics
        if idle_lengths:
            min_idle_length = min(idle_lengths)
            max_idle_length = max(idle_lengths)
            avg_gap_length = (tensor.shape[0] - sum(idle_lengths)) / len(idle_lengths)
            idle_nums = len(idle_lengths)

        return min_idle_length, max_idle_length, avg_gap_length, idle_nums


    def generate_subsets_with_random_lengths_fixed_gaps(self, tensor_length, min_subset_length, max_subset_length, fixed_gap, idle_nums):
        if min_subset_length > max_subset_length or min_subset_length <= 0:
            raise ValueError("Invalid subset length range.")
        
        if (fixed_gap < 0) or (fixed_gap >= tensor_length):
            raise ValueError("Invalid fixed gap.")

        subsets = []
        current_position = 0
        # avg_subset_len = (max_subset_length - min_subset_length) // idle_nums
        # subset_length = avg_subset_len
        
        while current_position < tensor_length:
            # Randomly determine the length of the next subset
            if min_subset_length > min(max_subset_length, tensor_length - current_position): break

            subset_length = random.randint(min_subset_length, min(max_subset_length, tensor_length - current_position))
            
            start = current_position
            end = start + subset_length - 1
            
            # Check if the end goes beyond the tensor's length
            if end + fixed_gap >= tensor_length:
                break

            # Add the new subset
            subsets.append((start, end))
            
            # Update the current position for the next iteration, taking into account the fixed gap
            current_position = end + 1 + fixed_gap
        
        return subsets
    
    
    def insert_subsets_into_tensor(self, original_tensor, subsets):
        # Ensure subsets are sorted to maintain correct insertion order
        subsets = sorted(subsets, key=lambda x: x[0])

        new_tensor_parts = []
        last_index = 0

        for start, end in subsets:
            # Validate the subset indices
            if start < 0 or end >= original_tensor.size(0) or start > end:
                print(f"Invalid start or end position for subset {start}-{end}")
                continue

            new_tensor_parts.append(original_tensor[last_index:start])

            # Create the subset tensor based on the first value of the subset
            subset_length = end - start + 1
            first_slice = original_tensor[start].unsqueeze(0)  # Add dimension to match
            subset_tensor = first_slice.repeat(subset_length, 1, 1)  # Replicate the slice
            new_tensor_parts.append(subset_tensor)

            # Update the last processed index
            last_index = start

        # Append the remaining part of the original tensor
        if last_index < original_tensor.size(0):
            new_tensor_parts.append(original_tensor[last_index:])

        # Concatenate all parts together along the first dimension
        new_tensor = torch.cat(new_tensor_parts, dim=0)

        return new_tensor
    


    @torch.no_grad()
    def get_hubert(self, wav16k_name):
        from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
        hubert = get_hubert_from_16k_wav(wav16k_name).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        hubert = hubert[:len(hubert)//8*8]
        # if len_mel % x_multiply == 0:
        #     num_to_pad = 0
        # else:
        #     num_to_pad = x_multiply - len_mel % x_multiply
        # hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0)))
        return hubert

    def get_f0(self, wav16k_name):
        from data_gen.utils.process_audio.extract_mel_f0 import extract_mel_from_fname, extract_f0_from_wav_and_mel
        wav, mel = extract_mel_from_fname(self.wav16k_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        f0 = f0.reshape([-1,1])
        return f0
    
    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        # forward the audio-to-motion
        ret={}
        pred = self.audio2secc_model.forward(batch, ret=ret, train=False, temperature=inp['temperature'])
        if pred.shape[-1] == 144:
            id = ret['pred'][0][:,:80]
            exp = ret['pred'][0][:,80:]
        else:
            id = batch['id']
            exp = ret['pred'][0]


        batch['exp'] = exp

        # render the SECC given the id,exp.
        # note that SECC is only used for visualization
        zero_eulers = torch.zeros([id.shape[0], 3]).to(id.device)
        zero_trans = torch.zeros([id.shape[0], 3]).to(exp.device)

        # get idexp_lm3d
        id_ds = torch.from_numpy(self.dataset.ds_dict['id']).float().cuda()
        exp_ds = torch.from_numpy(self.dataset.ds_dict['exp']).float().cuda()
        idexp_lm3d_ds = self.face3d_helper.reconstruct_idexp_lm3d(id_ds, exp_ds) # torch.Size([4384, 80]) torch.Size([4384, 64])
        idexp_lm3d_mean = idexp_lm3d_ds.mean(dim=0, keepdim=True)
        idexp_lm3d_std = idexp_lm3d_ds.std(dim=0, keepdim=True)
        if hparams.get("normalize_cond", True):
            idexp_lm3d_ds_normalized = (idexp_lm3d_ds - idexp_lm3d_mean) / idexp_lm3d_std
        else:
            idexp_lm3d_ds_normalized = idexp_lm3d_ds
        lower = torch.quantile(idexp_lm3d_ds_normalized, q=0.03, dim=0)
        upper = torch.quantile(idexp_lm3d_ds_normalized, q=0.97, dim=0)

        LLE_percent = inp['lle_percent']
            
        keypoint_mode = 'lm68' 
        
        # print(id.shape, exp.shape)id: [244,80], exp: [244,204]
        idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id, exp) # idexp_lm3d size torch.Size([240, 468, 3])

        if keypoint_mode == 'lm68':
            idexp_lm3d = idexp_lm3d[:, index_lm68_from_lm478]
            idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478]
            idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
            lower = lower[index_lm68_from_lm478]
            upper = upper[index_lm68_from_lm478]
        elif keypoint_mode == 'lm131':
            idexp_lm3d = idexp_lm3d[:, index_lm131_from_lm478]
            idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm131_from_lm478]
            idexp_lm3d_std = idexp_lm3d_std[:, index_lm131_from_lm478]
            lower = lower[index_lm131_from_lm478]
            upper = upper[index_lm131_from_lm478]
        elif keypoint_mode == 'lm468':
            idexp_lm3d = idexp_lm3d
        else:
            raise NotImplementedError()
        # idexp_lm3d size torch.Size([240, 68, 3])
        idexp_lm3d = idexp_lm3d.reshape([-1, 68*3]) # 204, 204
        # emo=batch['emotion'] # 1,1

        idexp_lm3d = self.secc2deform_model.forward(idexp_lm3d, batch['emotion'], float(inp['lm_deform_delta'])) #204,204



        idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
        feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d[:, :68*3], feat_database=idexp_lm3d_ds_lle[:, :68*3], K=10)
        # feat_fuse = smooth_features_xd(feat_fuse, kernel_size=3)
        
        idexp_lm3d[:, :68*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d[:,:68*3]
        idexp_lm3d = idexp_lm3d.reshape([-1, 68, 3])
        idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std
        # idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)

        cano_lm3d = (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized) / 10 + self.face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)
        eye_area_percent = self.opened_eye_area_percent * torch.ones([len(cano_lm3d), 1], dtype=cano_lm3d.dtype, device=cano_lm3d.device)
        
        if inp['blink_mode'] == 'period':
            cano_lm3d, eye_area_percent = inject_blink_to_lm68(cano_lm3d, self.opened_eye_area_percent, self.closed_eye_area_percent)
            print("Injected blink to idexp_lm3d by directly editting.")
        

        batch['eye_area_percent'] = eye_area_percent
        idexp_lm3d_normalized = ((cano_lm3d - self.face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)) * 10 - idexp_lm3d_mean) / idexp_lm3d_std
        idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)
        batch['cano_lm3d'] = cano_lm3d

        assert keypoint_mode == 'lm68'
        idexp_lm3d_normalized_ = idexp_lm3d_normalized[0:1, :].repeat([len(exp),1,1]).clone()
        idexp_lm3d_normalized_[:, 17:27] = idexp_lm3d_normalized[:, 17:27] # brow
        idexp_lm3d_normalized_[:, 36:48] = idexp_lm3d_normalized[:, 36:48] # eye
        idexp_lm3d_normalized_[:, 27:36] = idexp_lm3d_normalized[:, 27:36] # nose
        idexp_lm3d_normalized_[:, 48:68] = idexp_lm3d_normalized[:, 48:68] # mouth
        idexp_lm3d_normalized_[:, 0:17] = idexp_lm3d_normalized[:, :17] # yaw

        idexp_lm3d_normalized = idexp_lm3d_normalized_

        cond_win = idexp_lm3d_normalized.reshape([len(exp), 1, -1])
        cond_wins = [get_audio_features(cond_win, att_mode=2, index=idx) for idx in range(len(cond_win))]
        batch['cond_wins'] = cond_wins # 240 len

        # face boundark mask, for cond mask
        smo_euler = smooth_features_xd(batch['euler'])
        smo_trans = smooth_features_xd(batch['trans'])
        lm2d = self.face3d_helper.reconstruct_lm2d_nerf(id, exp, smo_euler, smo_trans)
        lm68 = lm2d[:, index_lm68_from_lm478, :]
        batch['lm68'] = lm68.reshape([lm68.shape[0], 68*2])

        # self.plot_lm(cano_lm3d, inp)
        return batch


    @torch.no_grad()
    def plot_lm(self, cano_lm3d, inp=None):
        cano_lm3d_frame_lst = vis_cano_lm3d_to_imgs(cano_lm3d, hw=512)
        cano_lm3d_frames = convert_to_tensor(np.stack(cano_lm3d_frame_lst)).permute(0, 3, 1, 2) / 127.5 - 1
        imgs = cano_lm3d_frames
        imgs = imgs.clamp(-1,1)

        print(inp['cano_lm_out_name'])

        try:
            os.makedirs(os.path.dirname(inp['cano_lm_out_name']), exist_ok=True)
        except: pass
        import imageio
        tmp_out_name = inp['cano_lm_out_name'].replace(".mp4", ".tmp.mp4")
        out_imgs = ((imgs.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
        writer = imageio.get_writer(tmp_out_name, fps = 25, format='FFMPEG', codec='h264')
        for i in tqdm.trange(len(out_imgs), desc=f"ImageIO is saving video using FFMPEG(h264) to {tmp_out_name}"):
            writer.append_data(out_imgs[i])
        writer.close()

        cmd = f"ffmpeg -i {tmp_out_name} -i {self.wav16k_name} -y -shortest -c:v libx264 -pix_fmt yuv420p -b:v 2000k -y -v quiet -shortest {inp['cano_lm_out_name']}"
        ret = os.system(cmd)
        if ret == 0:
            print(f"Saved at {inp['cano_lm_out_name']}")
            os.system(f"rm {tmp_out_name}")
        else:
            raise ValueError(f"error running {cmd}, please check ffmpeg installation, especially check whether it supports libx264!")


    @torch.no_grad()
    # todo: fp16 infer for faster inference. Now 192/128/64 are all 45fps
    def forward_secc2video(self, batch, inp=None):
        poses = batch['poses']
        poses = torch.stack(poses) # [80,4,4]


        cond_inp = batch['cond_wins'] # 6072
        cond_inp = torch.stack(cond_inp)[:,0,:,:] # torch.Size([6072, 3, 1, 204])

        opt = Namespace(**inp)

        self.test_set.poses = poses
        self.test_set.conds = cond_inp
        self.test_loader = self.test_set.dataloader()


        metrics = [PSNRMeter(), LPIPSMeter(device=self.device), LMDMeter(backend='fan')]
        criterion = torch.nn.L1Loss(reduction='none')
        trainer = Trainer('ngp', opt, self.syncnerf, device=self.device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        
        self.syncnerf.aud_features = self.test_loader._data.auds
        self.syncnerf.eye_areas = self.test_loader._data.eye_area
        
        # trainer.test(self.test_loader)
        if opt.output_dir and opt.output_name:
            trainer.test(self.test_loader, save_path=opt.output_dir, name=opt.output_name)
        else:
            trainer.test(self.test_loader)

        # if self.test_loader.has_gt:
        #     trainer.evaluate(self.test_loader)

    @torch.no_grad()
    def forward_system(self, batch, inp):
        self.forward_audio2secc(batch, inp)
        self.forward_secc2video(batch, inp)
        # return inp['out_name']

    @classmethod
    def example_run(cls, inp=None):
        inp_tmp = {
            'drv_audio_name': 'data/raw/val_wavs/zozo.wav',
            'src_image_name': 'data/raw/val_imgs/Macron.png'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp

        infer_instance = cls(inp['a2m_ckpt'], inp['postnet_ckpt'], inp['head_ckpt'], inp['torso_ckpt'], inp['lm_deform_ckpt'], inp=inp)
        infer_instance.infer_once(inp)

    ##############
    # IO-related
    ##############
    def save_wav16k(self, audio_name):
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert audio_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = audio_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {audio_name} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {audio_name} to {wav16k_name}.")

    def is_wav_silent(self, file_path, threshold=0.001):
        # Load the audio file
        data, samplerate = sf.read(file_path)
        
        # Calculate the RMS value of the audio data
        rms_value = np.sqrt(np.mean(data**2))
        
        # Normalize RMS by the maximum possible value (1.0 for float32 audio data)
        normalized_rms = rms_value / 1.0
        
        # Check if the normalized RMS value is below the threshold
        return normalized_rms < threshold
    
    def get_raysod_lists(self, new_poses):
        rays_o_lst = []
        rays_d_lst = []

        hubert = self.get_hubert(self.wav16k_name)
        t_x = hubert.shape[0]

        for i in range(t_x//2):
            ngp_pose = new_poses[i].unsqueeze(0)
            rays = get_rays(ngp_pose.cuda(), self.dataset.intrinsics, self.dataset.H, self.dataset.W, N=-1)
            rays_o_lst.append(rays['rays_o'].cuda())
            rays_d_lst.append(rays['rays_d'].cuda())
        
        return rays_o_lst, rays_d_lst
    

if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/audio2motion_vae')  # checkpoints/audio2motion_vae
    parser.add_argument("--head_ckpt", default='') 
    parser.add_argument("--postnet_ckpt", default='') 
    parser.add_argument("--torso_ckpt", default='')
    parser.add_argument("--drv_aud", default='data/raw/val_wavs/MacronSpeech.wav')
    parser.add_argument("--drv_pose", default='nearest', help="目前仅支持源视频的pose,包括从头开始和指定区间两种,暂时不支持in-the-wild的pose")
    parser.add_argument("--blink_mode", default='none') # none | period
    parser.add_argument("--lle_percent", default=0.2) # nearest | random
    parser.add_argument("--temperature", default=0.2) # nearest | random
    parser.add_argument("--mouth_amp", default=0.4) # nearest | random
    parser.add_argument("--raymarching_end_threshold", default=0.01, help="increase it to improve fps") # nearest | random
    parser.add_argument("--debug", action='store_true') 
    parser.add_argument("--fast", action='store_true') 
    parser.add_argument("--out_name", default='tmp.mp4') 
    parser.add_argument("--low_memory_usage", action='store_true', help='write img to video upon generated, leads to slower fps, but use less memory')
    # others
    parser.add_argument("--cano_lm_out_name", default='cano_lm.mp4') 
    parser.add_argument("--emotion", default='neutral') 
    parser.add_argument("--lm_deform_ckpt", default='checkpoints/mead/lm_deformation') # gfpp
    parser.add_argument("--lm_deform_delta", default=1) 

    # args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
    parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=200000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")
    parser.add_argument('--pyramid_loss', type=int, default=0, help="use perceptual loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--bg_img', type=str, default='', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")
    parser.add_argument('--bs_area', type=str, default="upper", help="upper or eye")
    parser.add_argument('--au45', action='store_true', help="use openface au45")
    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--syncnerf_head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")
    parser.add_argument('--portrait', action='store_true', help="only render face")
    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=20000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    parser.add_argument('--asr_model', type=str, default='deepspeech')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--cond_out_dim', type=int, default=64)
    parser.add_argument('--cond_win_size', type=int, default=1)
    parser.add_argument('--smo_win_size', type=int, default=3) # 5

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--output_name', type=str)


    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    if opt.test and False:
        opt.smooth_path = True
        opt.smooth_eye = True
        opt.smooth_lips = True

    opt.cuda_ray = True
    # assert opt.cuda_ray, "Only support CUDA ray mode."

    if opt.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."
    
    args = opt

    inp = {
        'cond_out_dim': args.cond_out_dim,
        'cond_win_size': args.cond_win_size,
        'smo_win_size': args.smo_win_size,
        'output_dir': args.output_dir,
        'output_name': args.output_name,
        'a2m_ckpt': args.a2m_ckpt,
        'postnet_ckpt': args.postnet_ckpt,
        'head_ckpt': args.head_ckpt,
        'syncnerf_head_ckpt': args.syncnerf_head_ckpt,
        'torso_ckpt': args.torso_ckpt,
        'drv_audio_name': args.drv_aud,
        'drv_pose': args.drv_pose,
        'blink_mode': args.blink_mode,
        'temperature': float(args.temperature),
        'mouth_amp': args.mouth_amp,
        'lle_percent': float(args.lle_percent),
        'debug': args.debug,
        'out_name': args.out_name,
        'raymarching_end_threshold': args.raymarching_end_threshold,
        'low_memory_usage': args.low_memory_usage,
        'cano_lm_out_name': args.cano_lm_out_name,
        'emotion': args.emotion,
        'lm_deform_ckpt': args.lm_deform_ckpt,
        'lm_deform_delta': args.lm_deform_delta,
        'path': args.path,
        'O': args.O,
        'test': args.test,
        'test_train': args.test_train,
        'data_range': args.data_range,
        'workspace': args.workspace,
        'seed': args.seed,
        'iters': args.iters,
        'lr': args.lr,
        'lr_net': args.lr_net,
        'ckpt': args.ckpt,
        'num_rays': args.num_rays,
        'cuda_ray': args.cuda_ray,
        'max_steps': args.max_steps,
        'num_steps': args.num_steps,
        'upsample_steps': args.upsample_steps,
        'update_extra_interval': args.update_extra_interval,
        'max_ray_batch': args.max_ray_batch,
        'warmup_step': args.warmup_step,
        'amb_aud_loss': args.amb_aud_loss,
        'amb_eye_loss': args.amb_eye_loss,
        'unc_loss': args.unc_loss,
        'lambda_amb': args.lambda_amb,
        'pyramid_loss': args.pyramid_loss,
        'fp16': args.fp16,
        'bg_img': args.bg_img,
        'fbg': args.fbg,
        'exp_eye': args.exp_eye,
        'fix_eye': args.fix_eye,
        'smooth_eye': args.smooth_eye,
        'bs_area': args.bs_area,
        'au45': args.au45,
        'torso_shrink': args.torso_shrink,
        'color_space': args.color_space,
        'preload': args.preload,
        'bound': args.bound,
        'scale': args.scale,
        'offset': args.offset,
        'dt_gamma': args.dt_gamma,
        'min_near': args.min_near,
        'density_thresh': args.density_thresh,
        'density_thresh_torso': args.density_thresh_torso,
        'patch_size': args.patch_size,
        'init_lips': args.init_lips,
        'finetune_lips': args.finetune_lips,
        'smooth_lips': args.smooth_lips,
        'torso': args.torso,
        'gui': args.gui,
        'W': args.W,
        'H': args.H,
        'radius': args.radius,
        'fovy': args.fovy,
        'max_spp': args.max_spp,
        'att': args.att,
        'aud': args.aud,
        'emb': args.emb,
        'portrait': args.portrait,
        'ind_dim': args.ind_dim,
        'ind_num': args.ind_num,
        'ind_dim_torso': args.ind_dim_torso,
        'amb_dim': args.amb_dim,
        'part': args.part,
        'part2': args.part2,
        'train_camera': args.train_camera,
        'smooth_path': args.smooth_path,
        'smooth_path_window': args.smooth_path_window,
        'asr': args.asr,
        'asr_wav': args.asr_wav,
        'asr_play': args.asr_play,
        'asr_model': args.asr_model,
        'asr_save_feats': args.asr_save_feats,
        'fps': args.fps,
        'l': args.l,
        'm': args.m,
        'r': args.r,
    }

    if args.fast:
        inp['raymarching_end_threshold'] = 0.05
    
    RealTalk2Infer.example_run(inp)
