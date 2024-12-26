#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("./check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../scripts/model.py"

python $PYTHON_SCRIPT \
    --input_path "../assets/noise.png"\
    --gt_path "../assets/noise_gt.png"\
    --mask_path "../assets/noise_mask.png"
