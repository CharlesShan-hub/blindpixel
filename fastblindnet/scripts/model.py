import torch
import torch.nn as nn
import torch.nn.functional as F
from clib.utils import *

__all__ = ['FastBlindNet']

class FastBlindNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Leaky ReLU Activation
        self.negative_slope = 0.2
        
        # Custom Normalization Layer
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))

        # Define Convolutional Layers
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1, dilation=1)  # 膨胀率1
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=2, dilation=2)  # 膨胀率2
        self.conv3 = nn.Conv2d(24, 24, kernel_size=3, padding=4, dilation=4)  # 膨胀率4
        self.conv4 = nn.Conv2d(24, 24, kernel_size=3, padding=8, dilation=8)  # 膨胀率8
        self.conv5 = nn.Conv2d(24, 24, kernel_size=3, padding=16, dilation=16)  # 膨胀率16
        self.conv6 = nn.Conv2d(24, 24, kernel_size=3, padding=32, dilation=32)  # 膨胀率32
        self.conv7 = nn.Conv2d(24, 24, kernel_size=3, padding=64, dilation=64)  # 膨胀率64
        self.conv8 = nn.Conv2d(24, 24, kernel_size=3, padding=1, dilation=1)  # 膨胀率1
        self.conv9 = nn.Conv2d(24, 1, kernel_size=1)  # 输出层，卷积核为1x1

        # Define Batch Norm Layers
        self.batch_norm1 = nn.BatchNorm2d(24)
        self.batch_norm2 = nn.BatchNorm2d(24)
        self.batch_norm3 = nn.BatchNorm2d(24)
        self.batch_norm4 = nn.BatchNorm2d(24)
        self.batch_norm5 = nn.BatchNorm2d(24)
        self.batch_norm6 = nn.BatchNorm2d(24)
        self.batch_norm7 = nn.BatchNorm2d(24)
        self.batch_norm8 = nn.BatchNorm2d(24)
        self.batch_norm9 = nn.BatchNorm2d(1)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """Initialize weights with identity initializer"""
        if isinstance(m, nn.Conv2d):
            m.weight.data = self.identity_initializer(m.weight.data)

    def identity_initializer(self, tensor):
        """Identity initialization for the convolution weights"""
        if tensor.ndimension() == 4:
            n = tensor.shape[0]
            for i in range(min(tensor.shape[1], tensor.shape[2], tensor.shape[3])):
                tensor[0, i, i, i] = 1.0
        return tensor

    def forward(self, x):
        x = self.conv1(x)
        x = self.custom_norm(x, self.batch_norm1)
        x = self.conv2(x)
        x = self.custom_norm(x, self.batch_norm2)
        x = self.conv3(x)
        x = self.custom_norm(x, self.batch_norm3)
        x = self.conv4(x)
        x = self.custom_norm(x, self.batch_norm4)
        x = self.conv5(x)
        x = self.custom_norm(x, self.batch_norm5)
        x = self.conv6(x)
        x = self.custom_norm(x, self.batch_norm6)
        x = self.conv7(x)
        x = self.custom_norm(x, self.batch_norm7)
        x = self.conv8(x)
        x = self.custom_norm(x, self.batch_norm8)
        x = self.conv9(x)
        x = self.custom_norm(x, self.batch_norm9)

        return x
    
    def lrelu(self, x):
        """Leaky ReLU activation"""
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def custom_norm(self, x, batch_norm):
        """Custom normalization with learnable parameters"""
        x = F.relu(x)
        return self.w0 * x + self.w1 * batch_norm(x)

def main():
    # Create model instance
    model = FastBlindNet()

    # Print model architecture
    print(model)
    # ImageProcessingNet(
    # (conv_layers): ModuleList(
    #     (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (1): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #     (2): Conv2d(24, 24, kernel_size=(3, 3), stride=(4, 4), padding=(1, 1))
    #     (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(8, 8), padding=(1, 1))
    #     (4): Conv2d(24, 24, kernel_size=(3, 3), stride=(16, 16), padding=(1, 1))
    #     (5): Conv2d(24, 24, kernel_size=(3, 3), stride=(32, 32), padding=(1, 1))
    #     (6): Conv2d(24, 24, kernel_size=(3, 3), stride=(64, 64), padding=(1, 1))
    #     (7): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (8): Conv2d(24, 3, kernel_size=(1, 1), stride=(1, 1))
    # )
    # )

if __name__ == "__main__":
    main()
