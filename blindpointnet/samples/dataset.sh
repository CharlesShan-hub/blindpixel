#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/dataset.py"

python $PYTHON_SCRIPT \
    --dataset_path "/Users/kimshan/Public/data/blindpoint/source" \
    --method "gb"
