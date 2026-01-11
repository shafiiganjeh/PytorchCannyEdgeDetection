
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class Sobel(nn.Module):
    
    """
    sobel gradient via seperable convolution
    """
    
    def __init__(self, 
                 kernel_size: int = 3,
                 padding_mode: str = 'reflect',
                 precision: torch.dtype = torch.float32):
        """
        Args:
            kernel_size: Size of the kernel.
            padding_mode: 'zeros', 'reflect', 'replicate', 'circular'
            precision: 'torch.float32', 'torch.float16'
        """

        super().__init__()

        self.padding_mode = padding_mode
        self.precision = precision

        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self._create_kernel()

    def _create_kernel(self):
        # sobel kernel is seperable
        dx = torch.arange(self.kernel_size, dtype=self.precision) - self.kernel_size // 2
        norm = torch.sum(torch.abs(dx)) / 2
        dx = dx / norm
        
        s = torch.tensor([self._nk(self.kernel_size-1,k) for k in range(self.kernel_size)], dtype=self.precision)
        s = s / torch.sum(s)
        # but not symmetric
        self.dx = nn.Parameter(dx.view( 1,1,1, self.kernel_size), requires_grad=False)
        self.sx = nn.Parameter(s.view( 1,1,self.kernel_size, 1), requires_grad=False)
        
        self.dy = nn.Parameter(dx.view( 1,1,self.kernel_size, 1), requires_grad=False)
        self.sy = nn.Parameter(s.view( 1,1,1, self.kernel_size), requires_grad=False)
        
    def _nk(self,n,k):
        return math.factorial(n)/(math.factorial(k) * math.factorial(n-k))
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor,torch.Tensor): # (B, 1, H, W)
            
        
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode=self.padding_mode)
        
        g_x = F.conv2d(x, self.sx, padding=0)
        g_x = F.conv2d(g_x, self.dx, padding=0)
        
        g_y = F.conv2d(x, self.sy, padding=0)
        g_y = F.conv2d(g_y, self.dy, padding=0)

        return (g_x,g_y)
    
    def to(self, device):

        super().to(device)
        self.dx.data = self.dx.data.to(device)
        self.dy.data = self.dy.data.to(device)
        
        self.sx.data = self.sx.data.to(device)
        self.sy.data = self.sy.data.to(device)

        return self
    
    

