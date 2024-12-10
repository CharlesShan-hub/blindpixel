#!/bin/bash

# Run Script
python ../scripts/train_classification.py \
    --dataset "/home/vision/users/sht/data/shapenetcore_partanno_segmentation_benchmark_v0" \
    --nepoch 10 \
    --dataset_type shapenet