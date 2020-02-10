#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u bin/import_custom_data.py
./bin/wav_preprocess.sh

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    #checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
    checkpoint_dir=`pwd`/checkpoints
fi

model_dir=`pwd`/saved_models

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --train_files data/custom_data/custom_data_train.csv \
  --dev_files data/custom_data/custom_data_dev.csv \
  --test_files data/custom_data/custom_data_test.csv \
  --n_hidden 512 \
  --epochs 200 \
  --learning_rate 0.0001 \
  --checkpoint_dir "$checkpoint_dir" \
  --export_dir "$model_dir" \
  "$@"
#--cudnn_checkpoint "$checkpoint_dir" \
