#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("./check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../scripts/test.py"
RES_PATH="${BASE_PATH}/model/fastblindnet/ino"

python $PYTHON_SCRIPT \
    --comment "fastblindnet on noisy INO with ReduceLROnPlateau on SGD" \
    --model_path "${RES_PATH}/2024_12_26_19_55/checkpoints/18.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --width 328 \
    --height 254 \
    --batch_size 32 \
    --seed 32 \
    --val 0.1 \
    --test 0.1
