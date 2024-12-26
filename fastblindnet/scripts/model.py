import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from clib.utils import *

__all__ = ['FastBlindNet']

def find_medians(x, k):
    # x: (N, C, H, W)
    #unfold the tensor into patches of size k x k
    patches = F.unfold(x, kernel_size=(k, k), stride=1, padding=k//2)
    #reshape patches to (N, C, k*k, H*W)
    patches = patches.view(x.size(0), x.size(1), k*k, -1)
    #sort patches along the k*k dimension
    patches, _ = torch.sort(patches, dim=2)
    #select the median value (k*k is odd, so we select the middle element)
    median = patches[:, :, k*k//2, :]
    return median

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size=3):
        super(MedianPool2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (N, C, H, W)
        #apply find_medians for each channel
        channels = torch.split(x, 1, dim=1)
        medians = [find_medians(channel, self.kernel_size) for channel in channels]
        #concatenate the channels back together
        median = torch.cat(medians, dim=1)
        #reshape to the original spatial dimensions
        H, W = x.size(2), x.size(3)
        median = median.view(-1, x.size(1), H, W)
        return median

# Example usage:
# model = MedianPool2d(kernel_size=3)
# input_tensor = torch.randn(1, 3, 28, 28)
# output_tensor = model(input_tensor)


class AdaptiveBatchNorm2d(nn.Module):

    """Adaptive batch normalization"""
    # Author: Nrupatunga
    # Email: nrupatunga.s@byjus.com
    # Github: https://github.com/nrupatunga

    def __init__(self, num_feat, eps=1e-5, momentum=0.1, affine=True):
        """Adaptive batch normalization"""
        super().__init__()
        self.bn = nn.BatchNorm2d(num_feat, eps, momentum, affine)
        self.a = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x) 
    
class ConvBlock(nn.Module):

    """Convolution head"""
    # Author: Nrupatunga
    # Email: nrupatunga.s@byjus.com
    # Github: https://github.com/nrupatunga

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 dilation: int,
                 norm_layer: nn.Module = AdaptiveBatchNorm2d):
        """
        @in_channels: number of input channels
        @out_channels: number of output channels
        @dilation: dilation factor @activation: 'relu'- relu,
        'lrelu': leaky relu
        @norm_layer: 'bn': batch norm, 'in': instance norm, 'gn': group
        norm, 'an': adaptive norm
        """
        super().__init__()
        convblk = []

        convblk.extend([
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation),
            nn.LeakyReLU(negative_slope=0.2),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()])

        self.convblk = nn.Sequential(*convblk)
        self.init_weights(self.convblk)

    def identity_init(self, shape):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[2] // 2, shape[3] // 2
        for i in range(np.minimum(shape[0], shape[1])):
            array[i, i, cx, cy] = 1

        return array

    def init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                weights = self.identity_init(m.weight.shape)
                with torch.no_grad():
                    m.weight.copy_(torch.from_numpy(weights).float())
                torch.nn.init.zeros_(m.bias)

    def forward(self, *inputs):
        return self.convblk(inputs[0])

class FastBlindNet(nn.Module):
    def __init__(self, 
                 unit_test: bool = False,
                 use_mask: bool = False):
        """Initialization """
        super().__init__()
        self.unit_test = unit_test
        self.input_channels = 2 if use_mask else 1

        nbLayers = 24

        self.median3 = MedianPool2d(3)
        self.median5 = MedianPool2d(5)
        self.conv1 = ConvBlock(self.input_channels, nbLayers, 3, 1, 1)
        self.conv2 = ConvBlock(nbLayers, nbLayers, 3, 2, 2)
        self.conv3 = ConvBlock(nbLayers, nbLayers, 3, 4, 4)
        self.conv4 = ConvBlock(nbLayers+self.input_channels, nbLayers+self.input_channels, 3, 8, 8)
        self.conv5 = ConvBlock(nbLayers+self.input_channels, nbLayers+self.input_channels, 3, 16, 16)
        self.conv6 = ConvBlock(nbLayers+self.input_channels, nbLayers+self.input_channels, 3, 32, 32)
        # self.conv7 = ConvBlock(nbLayers+self.input_channels, nbLayers+self.input_channels, 3, 64, 64)
        self.conv8 = ConvBlock(nbLayers+self.input_channels, 1 if unit_test else nbLayers, 3, 1, 1)
        self.conv9 = nn.Conv2d(nbLayers, 1, kernel_size=1, dilation=1)
        self.weights_init(self.conv9)

    def forward(self, x):
        y = self.median3(x)
        y = self.conv1(y)
        y = self.median3(y)
        y = self.conv2(y)
        y = self.conv3(y)
        x = torch.cat((y,x),dim=1)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.conv7(x)
        x = self.conv8(x)
        if self.unit_test:
            return x
        else:
            return self.conv9(x)


    def weights_init(self, m):
        """conv2d Init
        """
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

@click.command()
@click.option("--input_path", type=click.Path(exists=True), required=True)
@click.option("--gt_path", type=click.Path(exists=True), required=True)
@click.option("--mask_path", type=click.Path(exists=True), required=True)
def main(input_path,gt_path,mask_path):
    from clib.metrics.fusion import fused
    from clib.utils import glance,path_to_gray,to_tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = MedianPool2d(kernel_size=3).to(device)
    # output_tensor = model(fused)
    # glance([fused, output_tensor])

    from torchinfo import summary

    # Create model instance
    model = FastBlindNet(unit_test=True).to(device)

    # Print model architecture
    summary(model, input_size=(1, 2, 500, 500))

    # breakpoint()

    noisy = to_tensor(path_to_gray(input_path)).to(device)
    gt = to_tensor(path_to_gray(gt_path)).to(device)
    mask = to_tensor(path_to_gray(mask_path)).to(device)
    input_tensor = torch.cat((noisy,mask),dim=0).to(torch.float32).unsqueeze(0)

    # print(input_tensor.shape)
    # breakpoint()

    glance([noisy, model(input_tensor).squeeze(0)])

if __name__ == "__main__":
    main()
