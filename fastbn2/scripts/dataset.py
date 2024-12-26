# """
# File: mit_dataloader.py
# Author: Nrupatunga
# Email: nrupatunga.s@byjus.com
# Github: https://github.com/nrupatunga
# Description: mit dataset loaders
# """
# from pathlib import Path
# from typing import Union

# import cv2
# import numpy as np
# import torch
# from imutils import paths
# from torch.utils.data import DataLoader, Dataset
# from torchvision.utils import make_grid
# from tqdm import tqdm


# class MitData(Dataset):

#     """Docstring for MitData. """

#     def __init__(self,
#                  data_dir: Union[str, Path],
#                  is_train: bool):
#         """
#         @data_dir: path to the dataset
#         @shuffle: shuffle True/False
#         """
#         super().__init__()

#         self.data_dir = data_dir
#         self.is_train = is_train

#         input_dir = Path(data_dir).joinpath('input')
#         gt_dir = Path(data_dir).joinpath('gt')

#         input_imgs = list(paths.list_images(input_dir))
#         gt_imgs = list(paths.list_images(gt_dir))

#         self.pair_images = list(zip(input_imgs, gt_imgs))

#     def __len__(self):
#         return len(self.pair_images)

#     def __getitem__(self, idx):
#         img_path, gt_path = self.pair_images[idx]
#         img = cv2.imread(str(img_path))
#         gt = cv2.imread(str(gt_path))

#         img = np.transpose(img, axes=(2, 0, 1)) / 255.
#         gt = np.transpose(gt, axes=(2, 0, 1)) / 255.

#         img = torch.from_numpy(img).float()
#         gt = torch.from_numpy(gt).float()
#         return (img, gt)


# if __name__ == "__main__":
#     from utils import Visualizer

#     viz = Visualizer()
#     # data_dir = '/media/nthere/datasets/FastImageProcessing/data/train/set-1/'
#     data_dir = '/Users/kimshan/Public/data/vision/other'

#     mit = MitData(data_dir, is_train=True)
#     dataloader = DataLoader(mit, batch_size=1, shuffle=True,
#                             num_workers=6)

#     viz = Visualizer()
#     for i, (img, gt) in tqdm(enumerate(dataloader)):
#         data = torch.cat((img, gt), 0)
#         data = make_grid(data, nrow=2, pad_value=0)
#         viz.plot_images_np(data, 'data')

# """
# File: mit_dataloader.py
# Author: Nrupatunga
# Email: nrupatunga.s@byjus.com
# Github: https://github.com/nrupatunga
# Description: mit dataset loaders
# """
# from pathlib import Path
# from typing import Union

# import cv2
# import numpy as np
# import torch
# from imutils import paths
# from torch.utils.data import DataLoader, Dataset
# from torchvision.utils import make_grid
# from tqdm import tqdm


# class MitData(Dataset):

#     """Docstring for MitData. """

#     def __init__(self,
#                  data_dir: Union[str, Path],
#                  is_train: bool):
#         """
#         @data_dir: path to the dataset
#         @shuffle: shuffle True/False
#         """
#         super().__init__()

#         self.data_dir = data_dir
#         self.is_train = is_train

#         input_dir = Path(data_dir).joinpath('input')
#         gt_dir = Path(data_dir).joinpath('gt')

#         input_imgs = list(paths.list_images(input_dir))
#         gt_imgs = list(paths.list_images(gt_dir))

#         self.pair_images = list(zip(input_imgs, gt_imgs))

#     def __len__(self):
#         return len(self.pair_images)

#     def __getitem__(self, idx):
#         img_path, gt_path = self.pair_images[idx]
#         img = cv2.imread(str(img_path))
#         gt = cv2.imread(str(gt_path))

#         img = np.transpose(img, axes=(2, 0, 1)) / 255.
#         gt = np.transpose(gt, axes=(2, 0, 1)) / 255.

#         img = torch.from_numpy(img).float()
#         gt = torch.from_numpy(gt).float()
#         return (img, gt)


# if __name__ == "__main__":
#     from core.utils.vis_utils import Visualizer

#     viz = Visualizer()
#     # data_dir = '/media/nthere/datasets/FastImageProcessing/data/train/set-1/'
#     data_dir = '/Users/kimshan/Public/data/vision/other'

#     mit = MitData(data_dir, is_train=True)
#     dataloader = DataLoader(mit, batch_size=1, shuffle=True,
#                             num_workers=6)

#     viz = Visualizer()
#     for i, (img, gt) in tqdm(enumerate(dataloader)):
#         data = torch.cat((img, gt), 0)
#         data = make_grid(data, nrow=2, pad_value=0)
#         viz.plot_images_np(data, 'data')

import click
from clib.dataset.fusion import INO
import numpy as np
from PIL import Image

class MitData(INO):
    def __init__(self, root, transform = None, download = False, mode = 'image',\
                 salt_prob = 0.01, pepper_prob=0.01):
        super().__init__(root, transform = None, download = download, mode = mode)
        self.transform = transform
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
    
    def __getitem__(self, idx):
        ir_file = self.ir_image[idx]
        ir = Image.open(ir_file).convert("L")
        
        ir_noisy, noise_mask = self.add_salt_and_pepper_noise(ir)
    
        if self.transform:
            ir = self.transform(ir)
            noise_mask = self.transform(noise_mask)
            ir_noisy = self.transform(ir_noisy)

        # return ir_noisy,ir,noise_mask
        return ir_noisy,ir
    
    def __len__(self):
        return super().__len__()
    
    def add_salt_and_pepper_noise(self, image):
        """Add salt and pepper noise to the image."""
        # Create a random noise mask for salt and pepper

        noise = np.random.rand(image.size[1],image.size[0])
        image_array = np.array(image)/255.0
        noisy_image = image_array.copy()
        
        # Salt noise (white)
        salt_mask = noise < self.salt_prob
        noisy_image[salt_mask] = 1.0  # White (maximum intensity)
        
        # Pepper noise (black)
        pepper_mask = noise > (1 - self.pepper_prob)
        noisy_image[pepper_mask] = 0.0  # Black (minimum intensity)

        # Create a noise mask (1 where noise is added, 0 elsewhere)
        noise_mask = np.logical_or(salt_mask, pepper_mask).astype(np.float32)
        
        return Image.fromarray((noisy_image*255).astype(np.uint8), mode="L"), Image.fromarray(noise_mask * 255, mode="L") 


@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(**kwargs):
    # Test for INO
    # dataset = INO(root=kwargs['dataset_path'],download=True)
    # for (idx,(ir,vis,mask)) in enumerate(dataset):
    #     if idx // 100 == 0:
    #         print(ir.size,vis.size,mask.size if mask is not None else None)
    
    # Test for INO with blind point
    dataset = MitData(root=kwargs['dataset_path'],download=True)
    for (idx, (ir_noisy,ir,noise_mask)) in enumerate(dataset):
        if idx // 100 == 0:
            print(ir.size,noise_mask.size)

if __name__ == "__main__":
    main()