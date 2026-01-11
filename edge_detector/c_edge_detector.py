
import torch
import torch.nn as nn
import sys
sys.path.append("..")

import functions as fs


class c_edge(nn.Module):
    
    """
    main canny edge detector module
    """
    
    def __init__(self, 
                 blur: bool = True,
                 kernel_size_gauss: int = 3,
                 sigma_gauss: float = 1.,
                 kernel_size_sobel: int = 3,
                 upper_treshold: float = 20,
                 lower_treshold: float = 10,
                 max_iterations: int = 20,
                 padding_mode: str = 'reflect',
                 precision: torch.dtype = torch.float16):
        """
        Args:
            kernel_size_gauss: int.
            sigma_gauss: float.
            kernel_size_sobel: int.
            upper_treshold: float.
            lower_treshold: float.
            max_iterations: int.
            padding_mode: 'zeros', 'reflect', 'replicate', 'circular'.
            precision: 'torch.float32', 'torch.float16'.
        """

        super().__init__()

        self.upper_treshold = upper_treshold
        self.lower_treshold = lower_treshold

        if blur:
            self.gauss_filter = fs.GaussianFilter(sigma_gauss,kernel_size_gauss,
                                                  padding_mode,precision)
        else:
            self.gauss_filter = nn.Identity(sigma_gauss,kernel_size_gauss,
                                            padding_mode,precision)
            
        self.sobel_derivative = fs.Sobel(kernel_size = kernel_size_sobel,
                                         precision = precision,
                                         padding_mode = padding_mode)
        self.grad_tresh = fs.threshold()
        self.hysteresis = fs.hysteresis(max_iterations = max_iterations)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.gauss_filter(x)
        (gx,gy) = self.sobel_derivative(x)

        G = torch.sqrt(gx**2 + gy**2 )
        theta = torch.atan2(gy, gx)
        
        x = self.grad_tresh(G,theta)
        
        line_mask = torch.zeros_like(x,dtype = torch.int8)
        # 1 maybe edge 2 defenitive edge
        temp_mask = (x > self.lower_treshold) & (x < self.upper_treshold)
        line_mask = line_mask + (2*temp_mask).to(torch.int8)

        temp_mask =  x > self.upper_treshold
        line_mask = line_mask + (1*temp_mask).to(torch.int8)
        
        x = self.hysteresis(line_mask)
        x = (x < x.shape[-1]*x.shape[-2])*255
        
        return x
    
