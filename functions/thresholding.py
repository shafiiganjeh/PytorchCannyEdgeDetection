
import torch
import torch.nn as nn
import torch.nn.functional as F

class threshold(nn.Module):
    
    """
    lower bound cut-off suppression
    """
    
    def __init__(self):

        super().__init__()

        pi = torch.acos(torch.zeros(1)).item() * 2
        
        self.d225 = pi / 8
        self.d1575 = 7 * pi / 8
        self.d180 = pi
        self.d675 = 3 * pi / 8
        self.d1125 = 5 * pi / 8

    
    def forward(self, g: torch.Tensor,t: torch.Tensor) -> torch.Tensor:

        #t: angle tensor
        #g: magnitude tensor

        t = torch.abs(t)

        h_mask = ((t >= 0) & (t < self.d225)) | ((t >= self.d1575) & (t <= self.d180)) 
        d45_mask = (t >= self.d225) & (t < self.d675)
        v_mask = (t >= self.d675) & (t < self.d1125)
        d135_mask = (t >= self.d1125) & (t < self.d1575)
        
        padded = F.pad(g, (1, 1, 1, 1), mode='replicate')
        
        # slicing creates tensor views instead of copies, so much faster than using convolutions
        left = padded[:,:,1:-1, :-2]
        right = padded[:,:,1:-1, 2:]
        top = padded[:,:,:-2, 1:-1]
        bottom = padded[:,:,2:, 1:-1]
        top_left = padded[:,:,:-2, :-2]
        top_right = padded[:,:,:-2, 2:]
        bottom_left = padded[:,:,2:, :-2]
        bottom_right = padded[:,:,2:, 2:]
        
        # gradient values masked
        g_mask = torch.zeros_like(t)
        
        temp_mask = h_mask & (g >= left) & (g >= right)
        g_mask[temp_mask] = g[temp_mask]

        temp_mask = v_mask & (g >= top) & (g >= bottom)
        g_mask[temp_mask] = g[temp_mask]
        
        temp_mask = d45_mask & (g >= top_right) & (g >= bottom_left)
        g_mask[temp_mask] = g[temp_mask]
        
        temp_mask = d135_mask & (g >= top_left) & (g >= bottom_right)
        g_mask[temp_mask] = g[temp_mask]

        return g_mask


