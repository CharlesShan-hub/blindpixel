#!/bin/bash

# Build Path

cd "$(dirname "$0")"
BASE_PATH=$(./check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../scripts/train.py"
RES_PATH="${BASE_PATH}/model/fastblindnet/ino"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"

python $PYTHON_SCRIPT \
    --comment "fastblindnet on noisy INO with ReduceLROnPlateau on SGD" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --width 328 \
    --height 254 \
    --seed 32 \
    --batch_size 16 \
    --lr 0.3 \
    --max_epoch 100 \
    --max_reduce 3 \
    --factor 0.1 \
    --train_mode "Holdout" \
    --val 0.1 \
    --test 0.1