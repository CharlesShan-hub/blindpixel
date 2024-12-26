"""
File: train.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: training script
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda import is_available

from trainer import LitModel

if __name__ == "__main__":
    data_dir = '/Users/kimshan/Public/data/vision/torchvision/ino'
    model = LitModel(data_dir=data_dir,
                     batch_size=1)
    ckpt_cb = ModelCheckpoint(dirpath='./ckpt', save_top_k=10,monitor='val_loss',mode='min',
                              save_weights_only=False)
    devices = 1 if is_available() else 0  # 1 GPU or 0 for CPU
    accelerator = 'gpu' if is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=accelerator,  # Use 'gpu' or 'cpu' based on availability
                         devices=devices,  # Number of devices (GPUs or CPU)
                         max_epochs=180,
                         callbacks=[ckpt_cb],  # Use callbacks list instead of checkpoint_callback
                         resume_from_checkpoint='./ckpt/epoch=128.ckpt',
                         reload_dataloaders_every_epoch=True)
    trainer.fit(model)
