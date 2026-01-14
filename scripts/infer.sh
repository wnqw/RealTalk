export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
set -e

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