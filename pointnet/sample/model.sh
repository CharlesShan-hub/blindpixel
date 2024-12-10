#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/model.py"

python $PYTHON_SCRIPT

# stn torch.Size([32, 3, 3])
# loss tensor(1.6484, grad_fn=<MeanBackward0>)
# stn64d torch.Size([32, 64, 64])
# loss tensor(127.8758, grad_fn=<MeanBackward0>)
# global feat torch.Size([32, 1024])
# point feat torch.Size([32, 1088, 2500])
# class torch.Size([32, 5])
# seg torch.Size([32, 2500, 3])