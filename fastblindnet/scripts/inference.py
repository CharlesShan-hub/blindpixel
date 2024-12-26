import click
import torch
from pathlib import Path
from transform import transform
from model import FastBlindNet, MedianPool2d
from config import InferenceOptions
from clib.inference import BaseInferencer
from clib.utils import *
from kornia.metrics import psnr,ssim


class FastBlindNetInferencer(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = FastBlindNet().to(opts.device)
        self.load_checkpoint()
        self.transform = transform(self.opts.width,self.opts.height)

    def inference(self):
        input_tensor = to_tensor(path_to_gray(self.opts.input_path)).to(torch.float32)
        mask_tensor = to_tensor(path_to_gray(self.opts.mask_path)).to(torch.float32)
        if Path(self.opts.gt_path).exists():
            gt_tensor = to_tensor(path_to_gray(self.opts.gt_path)).to(torch.float32)
        else:
            gt_tensor = None
        if self.opts.use_mask:
            stack_input_tensor = torch.cat((input_tensor,mask_tensor),dim=0).unsqueeze(0)
        else:
            stack_input_tensor = input_tensor.unsqueeze(0)
        output_tensor = self.model(stack_input_tensor).squeeze(0)
        out_add_input = torch.zeros_like(input_tensor)
        out_add_input[~mask_tensor.bool()] = input_tensor[~mask_tensor.bool()]
        out_add_input[mask_tensor.bool()] = output_tensor[mask_tensor.bool()]

        m7 = MedianPool2d(kernel_size=7)
        m5 = MedianPool2d(kernel_size=5)
        m3 = MedianPool2d(kernel_size=3)
        md_image = m3(m3(input_tensor.unsqueeze(0))).squeeze(0)

        psnr_md = psnr(md_image,gt_tensor,1.0)
        psnr_out = psnr(output_tensor,gt_tensor,1.0)
        ssim_md = ssim(md_image.unsqueeze(0),gt_tensor.unsqueeze(0),7).mean()
        ssim_out = ssim(output_tensor.unsqueeze(0),gt_tensor.unsqueeze(0),7).mean()
        # breakpoint()
        glance(
            image=[input_tensor, gt_tensor, md_image, output_tensor, out_add_input,
                   input_tensor, gt_tensor, md_image, output_tensor, out_add_input],
            title=['Input', 'GT', f'md\npsnr={psnr_md}\nssim={ssim_md}', f'Output\npsnr={psnr_out}\nssim={ssim_out}', 'In+Out', 
                   'Input', 'GT', f'md\npsnr={psnr_md}\nssim={ssim_md}', f'Output\npsnr={psnr_out}\nssim={ssim_out}', 'In+Out'],
            auto_contrast=[True,True,True,True,True,
                           False,False,False,False,False],
            shape=(2,5),
            suptitle=f'{self.opts.model_path}',
        )

@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--input_path", type=click.Path(exists=True), required=True)
@click.option("--gt_path", type=click.Path(exists=True), required=True)
@click.option("--mask_path", type=click.Path(exists=True), required=True)
@click.option("--height", type=int, default=254, show_default=True)
@click.option("--width", type=int, default=328, show_default=True)
@click.option("--use_mask", type=bool, default=True, show_default=True)
@click.option("--comment", type=str, default="", show_default=False)
def inference(**kwargs):
    opts = InferenceOptions().parse(kwargs,present=True)
    inferencer = FastBlindNetInferencer(opts)
    inferencer.inference()


if __name__ == "__main__":
    inference()
