export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
set -e

dataset=ber
workspace=model/ber
asr_model=ave
target_aud=/home/wenqing/projs/fg_proj/TalkingGaussian/data/kati.wav
cuda=0

CUDA_VISIBLE_DEVICES=$cuda python data_utils/process.py data/$dataset/$dataset.mp4 --asr $asr_model

CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --iters 60000 --asr_model $asr_model
CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --iters 100000 --finetune_lips --patch_size 64 --asr_model $asr_model

CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --test --test_train --asr_model $asr_model --portrait --aud $target_aud
#CUDA_VISIBLE_DEVICES=$cuda python main.py data/$dataset --workspace $workspace -O --test --asr_model $asr_model --portrait
 