# Get Audio2Motion Model

You can download the pre-trained Audio-to-Motion model (pretrained on voxceleb2, a 2000-hour lip reading dataset) in this [Google Drive](https://drive.google.com/drive/folders/1FqvNbQgOSkvVO8i-vCDJmKM4ppPZjUpL?usp=sharing).

Place the model in the directory `checkpoints/audio2motion_vae`.

# Train Landmark Deformation Model
```
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/audio_deformation/audio_deformation.yaml --exp_name=aud_deformation/aud_deformation_res_att
```
Place the model in `checkpoints/lm_deformation/lm_deformation_res_att`.

# Train Emotion2Video Model
We suppose you have prepared the dataset following `docs/prepare_data/guide.md` and you can find a binarized `.npy` file in `data/binary/videos/{Video_ID}/trainval_dataset.npy` 

```
# Train the Head NeRF
# model and tensorboard will be saved at `checkpoints/<exp_name>`
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/{Video_ID}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/{Video_ID}_head --reset

# Train the Torso NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/{Video_ID}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/{Video_ID}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/{Video_ID}_head --reset
```

## How to train on your own video: 
Suppose you have a video named `{Video_ID}.mp4`
- Step1: crop your video to 512x512 and 25fps, then place it into `data/raw/videos/{Video_ID}.mp4`
- Step2: copy a config folder `egs/datasets/{Video_ID}` 
- Step3: Process the video following `docs/process_data/guide.md`, then you can get a `data/binary/videos/{Video_ID}/trainval_dataset.npy`
- Step4: Use the commandlines above to train the NeRF.

# Inference
```
# we provide a inference script in infer.sh:

export PYTHONPATH=./
export WAV
export old_nerf= 


realtalk_name= 
dataset=processed/videos/$realtalk_name
workspace=checkpoints/realtalk/$realtalk_name
# workspace=checkpoints/realtalk/${realtalk_name}_torso
asr_model=hubert
cuda=0
realtalk_ckpt=checkpoints/realtalk/$realtalk_name/checkpoints/ngp.pth
# realtalk_ckpt=checkpoints/realtalk/${realtalk_name}_torso/checkpoints/ngp.pth

lm_deform_ckpt=checkpoints/lm_deformation/lm_deformation_res_att
delta=0.15
output_dir=results
output_name= 

# emotion: angry, disgust, contempt, fear, happy, sad, surprise, neutral

time CUDA_VISIBLE_DEVICES=$cuda  python inference/realtalk_infer.py --head_ckpt=checkpoints/motion2video_nerf/${old_nerf}_head --torso_ckpt=checkpoints/motion2video_nerf/${old_nerf}_torso \
        --drv_aud=$WAV  --emotion surprise --lm_deform_delta $delta --blink_mode period \
        --a2m_ckpt checkpoints/audio2motion_vae --lm_deform_ckpt $lm_deform_ckpt \
        --path data/$dataset --workspace $workspace -O --test --test_train --asr_model $asr_model \
        --portrait --aud $WAV --ckpt $realtalk_ckpt \
        --output_dir $output_dir --output_name $output_name


# CUDA_VISIBLE_DEVICES=$cuda  python inference/realtalk_infer.py --head_ckpt=checkpoints/motion2video_nerf/${old_nerf}_head --torso_ckpt=checkpoints/motion2video_nerf/${old_nerf}_torso \
#         --drv_aud=$WAV  --emotion happy --lm_deform_delta $delta --blink_mode period \
#         --a2m_ckpt checkpoints/audio2motion_vae --lm_deform_ckpt $lm_deform_ckpt \
#         --path data/$dataset --workspace ${workspace} -O --test --test_train --asr_model $asr_model \
#         --aud $WAV --ckpt $syncnerf_ckpt \
#         --output_dir $output_dir --output_name $output_name \
#         --syncnerf_head_ckpt $syncnerf_ckpt --torso
```
