#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("./check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../scripts/inference.py"
RES_PATH="${BASE_PATH}/model/fastblindnet/ino"
ID_IMAGE=0
ID_CKPT=22
NAME_CKPT_FOLDER="2024_12_27_11_03"


python $PYTHON_SCRIPT \
    --comment "fastblindnet on noisy INO with ReduceLROnPlateau on SGD" \
    --model_path "${RES_PATH}/${NAME_CKPT_FOLDER}/checkpoints/${ID_CKPT}.pt" \
    --input_path "../assets/noise${ID_IMAGE}.png"\
    --gt_path "../assets/noise${ID_IMAGE}_gt.png"\
    --mask_path "../assets/noise${ID_IMAGE}_mask.png"\
    --width 640 \
    --height 512 \
    --use_mask False

    
# python $PYTHON_SCRIPT \
#     --comment "fastblindnet on noisy INO with ReduceLROnPlateau on SGD" \
#     --model_path "${RES_PATH}/${NAME_CKPT_FOLDER}/checkpoints/${ID_CKPT}.pt" \
#     --input_path "../assets/house.png"\
#     --gt_path "../assets/house.png"\
#     --mask_path "../assets/house_mask.png"\
#     --width 640 \
#     --height 512 \
#     --use_mask False
