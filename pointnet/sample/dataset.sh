#!/bin/bash

# Run Script
PYTHON_SCRIPT="../scripts/dataset.py"

python $PYTHON_SCRIPT shapenet '/Users/kimshan/Public/data/shapenetcore_partanno_segmentation_benchmark_v0'

# {'Chair': 0}
# {'Airplane': 4, 'Bag': 2, 'Cap': 2, 'Car': 4, 'Chair': 4, 'Earphone': 3, 'Guitar': 3, 'Knife': 2, 'Lamp': 4, 'Laptop': 2, 'Motorbike': 6, 'Mug': 2, 'Pistol': 3, 'Rocket': 3, 'Skateboard': 3, 'Table': 3} 4
# 2658
# torch.Size([2500, 3]) torch.FloatTensor torch.Size([2500]) torch.LongTensor
# {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}
# {'Airplane': 4, 'Bag': 2, 'Cap': 2, 'Car': 4, 'Chair': 4, 'Earphone': 3, 'Guitar': 3, 'Knife': 2, 'Lamp': 4, 'Laptop': 2, 'Motorbike': 6, 'Mug': 2, 'Pistol': 3, 'Rocket': 3, 'Skateboard': 3, 'Table': 3} 4
# 12137
# torch.Size([2500, 3]) torch.FloatTensor torch.Size([1]) torch.LongTensor