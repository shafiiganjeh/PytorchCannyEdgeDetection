
import torch 
import torch.nn as nn
import torch.nn.functional as F

class GaussianFilter(nn.Module):
    """
    Gaussian filter via seperable convolution
    """
    def __init__(self, 
                 sigma: float = 1.0, 
                 kernel_size: int = 3,
                 padding_mode: str = 'reflect',
                 precision: torch.dtype = torch.float32):
        """
        Args:
            sigma: Standard deviation of Gaussian kernel
            kernel_size: Size of the kernel.
            padding_mode: 'zeros', 'reflect', 'replicate', 'circular'
            precision: 'torch.float32', 'torch.float16'
        """
        
        super().__init__()
        self.sigma = sigma
        self.padding_mode = padding_mode
        self.precision = precision
        self.pi = torch.acos(torch.zeros(1)).item() * 2
         
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self._create_kernel()

    def _create_kernel(self):
        # gaussian kernel is seperable and symmetric
        x = torch.arange(self.kernel_size, dtype=self.precision) - self.kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * self.sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        self.kernel_x = nn.Parameter(kernel_1d.view(1, 1, 1, self.kernel_size), requires_grad=False)
        self.kernel_y = nn.Parameter(kernel_1d.view(1, 1, self.kernel_size, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, 1, H, W)

        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode=self.padding_mode)
        x = F.conv2d(x, self.kernel_x, padding=0)
        x = F.conv2d(x, self.kernel_y, padding=0)

        return x

    def to(self, device):

        super().to(device)
        self.kernel_x.data = self.kernel_x.data.to(device)
        self.kernel_y.data = self.kernel_y.data.to(device)

        return self


    

    

    