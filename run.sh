export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
set -e
export PYTHONPATH=.

dataset=ber
workspace=model/ber_ave
asr_model=ave
target_aud=/home/wenqing/projs/fg_proj/others/GeneFacePlusPlus/data/raw/val_wavs/Kati_hap.wav
target_aud_npy=/home/wenqing/projs/fg_proj/SyncTalk/data/audio/Kati_hap_hu.npy
cuda=0

# CUDA_VISIBLE_DEVICES=$cuda python nerf_triplane/deform_dataset.py
CUDA_VISIBLE_DEVICES=$cuda python nerf_triplane/deform_trainer.py

# CUDA_VISIBLE_DEVICES=$cuda python data_utils/process.py data/$dataset/$dataset.mp4 --asr $asr_model

# CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --iters 60000 --asr_model $asr_model
# CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --iters 100000 --finetune_lips --patch_size 64 --asr_model $asr_model

# CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --test --test_train --asr_model $asr_model --portrait --aud $target_aud #_npy
#CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --test --asr_model $asr_model --portrait

# ffmpeg -i /home/wenqing/projs/fg_proj/SyncTalk/model/ber/results/ngp_ep0023.mp4 \
    #  -i $target_aud -c:v copy -c:a aac -strict experimental /home/wenqing/projs/fg_proj/SyncTalk/model/ber/results/ngp_ep0023_aud.mp4

# path=/home/wenqing/projs/fg_proj/SyncTalk/model/deformation_hu2/400000epoch/checkpoints/
# find $path -type f ! -name 'epoch_ckpt_deformation_5205032.pth' -delete
