import click
from clib.dataset.fusion import INO
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_dilation

class BlindPointINO(INO):
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
            ir_noisy = self.transform(ir_noisy)
            noise_mask = self.transform(noise_mask)
            noise_mask = torch.where(noise_mask != 0.0, torch.tensor(1.0, dtype=noise_mask.dtype, device=noise_mask.device), noise_mask)

        return ir_noisy,ir,noise_mask
    
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

        # Dilation
        structure_element = np.ones((3, 3), dtype=np.uint8)
        dilated_salt_mask = binary_dilation(salt_mask, structure=structure_element)
        dilated_pepper_mask = binary_dilation(pepper_mask, structure=structure_element)

        # Create a noise mask (1 where noise is added, 0 elsewhere)
        noise_mask = np.logical_or(salt_mask, pepper_mask).astype(np.uint8)
        # noise_mask = np.logical_or(dilated_salt_mask, dilated_pepper_mask).astype(np.uint8)
        
        return Image.fromarray((noisy_image*255).astype(np.uint8), mode="L"), Image.fromarray((noise_mask * 255), mode="L") 


@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(**kwargs):
    # Test for INO
    # dataset = INO(root=kwargs['dataset_path'],download=True)
    # for (idx,(ir,vis,mask)) in enumerate(dataset):
    #     if idx // 100 == 0:
    #         print(ir.size,vis.size,mask.size if mask is not None else None)
    
    # Test for INO with blind point
    dataset = BlindPointINO(root=kwargs['dataset_path'],download=True)
    for (idx, (ir_noisy,ir,noise_mask)) in enumerate(dataset):
        if idx // 100 == 0:
            print(ir.size,noise_mask.size)
        # noise_mask.show()
        # breakpoint()

if __name__ == "__main__":
    main()