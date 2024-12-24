import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 24, 3, 1, 1),  # Conv1
            nn.Conv2d(24, 24, 3, 2, 1),  # Conv2
            nn.Conv2d(24, 24, 3, 4, 1),  # Conv3
            nn.Conv2d(24, 24, 3, 8, 1),  # Conv4
            nn.Conv2d(24, 24, 3, 16, 1), # Conv5
            nn.Conv2d(24, 24, 3, 32, 1), # Conv6
            nn.Conv2d(24, 24, 3, 64, 1), # Conv7
            nn.Conv2d(24, 24, 3, 1, 1),  # Conv8
            nn.Conv2d(24, 3, 1, 1)       # Conv9 (output)
        ])
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
        # Apply all conv layers with LeakyReLU activation and custom normalization
        for conv in self.conv_layers[:-1]:  # Apply all conv layers except the last
            x = self.lrelu(self.custom_norm(conv(x)))
        return self.conv_layers[-1](x)  # Output layer
    
    def lrelu(self, x):
        """Leaky ReLU activation"""
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def custom_norm(self, x):
        """Custom normalization with learnable parameters"""
        return self.w0 * x + self.w1 * F.batch_norm(x, running_mean=None, running_var=None, weight=None, bias=None, training=self.training)

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
