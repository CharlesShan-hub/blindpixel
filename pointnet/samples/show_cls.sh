#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/show_cls.py"

python $PYTHON_SCRIPT \
    --model "/home/vision/users/sht/project/blindpixel/pointnet/sample/cls/cls_model_9.pth" \
    --dataset "/home/vision/users/sht/data/shapenetcore_partanno_segmentation_benchmark_v0"