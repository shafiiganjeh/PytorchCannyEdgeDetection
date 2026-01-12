
import torch
import torch.nn as nn
import torch.nn.functional as F


class hysteresis(nn.Module):
    
    """
    lower bound cut-off suppression
    """
    
    def __init__(self, 
                 max_iterations: int = 15):
        """
        Args:
            max_iterations: 'int'
        """

        super().__init__()

        self.shape = [0,0,0,0]
        self.width = 0
        self.height = 0
        
        self.val = torch.arange(start=0,end = 1) 
        self.max_plus = 0
        
        self.max_iterations = max_iterations
        
    def add_masked(self, x,mask,val):
        return torch.einsum('bchw,chw->bchw', mask, val)
        
    def find_min_neighbour(self,vertex_values,b_idx,h_idx, w_idx):
        
        padded = F.pad(vertex_values, (1, 1, 1, 1), mode='constant', value=float(self.max_plus))
        device = vertex_values.device
        
        neighborhoods = []
        for height in [-1, 0, 1]:
            for width in [-1, 0, 1]:
                val = padded[b_idx,0, h_idx + height + 1, w_idx + width + 1]
                neighborhoods.append(val)
        
        neighborhoods_stacked = torch.stack(neighborhoods, dim=1)  # (num edge pixels, 9)
        min_vals = neighborhoods_stacked.min(dim=1)[0]
        
        result = torch.full(self.shape, self.max_plus,dtype = torch.int32, device=device)
        result[b_idx,0,h_idx, w_idx] = min_vals
        
        return result
    
        
    def pointer_jumping_op(self, x1,x2, mask, batch_idx, seq_idx):

        original_shape = x1.shape
        x1 = x1.view(-1, self.width * self.height)
        x2 = x2.view(x1.shape)
        mask = mask.view(x1.shape)
        
        values = x2[batch_idx, seq_idx]
        reduce_mask = values >= (self.width * self.height)
        
        reorder = values.clone()
        reorder[reduce_mask] -= (self.width * self.height)
        
        # pointer jumping P[i] = P[P[i]] only for i in edges ignore rest
        x1[batch_idx, seq_idx] = x1[batch_idx, reorder]
        
        # reform back
        x1 = x1.view(original_shape)
        x2 = x2.view(original_shape)
        mask = mask.view(original_shape)
        
        return x1
    
    
    def pointer_jumping_regular(self, x, mask, batch_idx, seq_idx):

        original_shape = x.shape
        x = x.view(-1, self.width * self.height)
        mask = mask.view(x.shape)
        
        values = x[batch_idx, seq_idx]
        reduce_mask = values >= (self.width * self.height)
        
        reorder = values.clone()
        reorder[reduce_mask] -= (self.width * self.height)
        
        # pointer jumping P[i] = P[P[i]] only for i in edges ignore rest
        x[batch_idx, seq_idx] = x[batch_idx, reorder]
        
        # reform back
        x = x.view(original_shape)
        mask = mask.view(original_shape)
        
        return x
    
    def tree_hooking(self, p1, p2, mask):

        n = self.width * self.height
        
        # essentially doing following operation for parent functions P and P':
        # P(u) = min{ P'(u) , min{P'(v) | P'(v) = u }}
        
        # Reshape for easier processing

        p1_flat = p1.view(self.shape[0], n)
        p2_flat = p2.view(self.shape[0], n)
        mask_flat = mask.view(self.shape[0], n)
        
        for b in range(self.shape[0]):
            batch_mask = mask_flat[b]
            
            p1_masked = p1_flat[b, batch_mask]
            p2_masked = p2_flat[b, batch_mask]

            has_offset = p1_masked >= n
            
            indices = p1_masked.clone()
            indices[has_offset] -= n
            
            p2_flat[b].scatter_reduce_(0, indices, p2_masked, reduce='amin', include_self=True)
        
        return p2
    
 
    def absolute_error_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                        batch_idx_flat: torch.Tensor, batches: int) -> torch.Tensor:
        l_abs = torch.abs(pred - target)
        loss = torch.zeros(batches, dtype = torch.int32, device=l_abs.device).scatter_reduce_(0,
                                                                  batch_idx_flat,
                                                                  l_abs,
                                                                  reduce="sum", 
                                                                  include_self=False)
                
        return loss
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        device = x.device
        self.shape = x.shape        
        self.width = self.shape[-1]
        self.height = self.shape[-2]
        
        self.val = torch.arange(start=0,end = self.width*self.height, device=device) 
        self.val = self.val.view(1,self.height, self.width)
        self.max_plus = 2*self.width*self.height

        indices = x.nonzero()
        b_idx = indices[:, 0]
        h_idx = indices[:, 2]
        w_idx = indices[:, 3]
        # make this in the def here only for testing
        vertex_values = torch.zeros(self.shape,dtype = torch.int32, device=device)
        M = (x > 1) # mb edge
        vertex_values.add_((self.width*self.height) * M) 
        M = (x == 0) # no edge
        vertex_values.add_(self.max_plus * M)  
        M = (x > 0) # def edge
        vertex_values += self.add_masked(vertex_values,M,self.val)

        M = M.view(-1,self.width*self.height)
        batch_idx_flat, seq_idx_flat = torch.where(M)
        M = M.view(self.shape)
        
        # initialize Parent function
        x = self.find_min_neighbour(vertex_values,b_idx,h_idx, w_idx)
        
        T_skip = torch.ones(self.shape[0], dtype = torch.int32, device=device)
        
        # connected component alg. simmilar to Shiloach-Vishkin algorithm but not the same.
        while torch.sum(T_skip) > 0 and self.max_iterations > 0:
            
            self.max_iterations = self.max_iterations - 1
            
            # get valid mask
            batch_valid = T_skip > 0
            M = M & batch_valid.view(-1, 1, 1, 1)
            
            x_old = x.clone()
            # first pointer jumping
            x_1 = x.clone()
            x_2 = self.find_min_neighbour(x_1,b_idx,h_idx, w_idx)
            x_1 = self.pointer_jumping_op(x_1,x_2,M,batch_idx_flat, seq_idx_flat)
            
            # hooking trees 
            x = self.tree_hooking(x,x_1,M)
            
            # second pointer jumping
            x = self.pointer_jumping_regular(x,M,batch_idx_flat, seq_idx_flat)
            # get the loss 
            T_skip = self.absolute_error_loss(x_old[b_idx,0,h_idx, w_idx], x[b_idx,0,h_idx, w_idx],batch_idx_flat,x.shape[0])

        return x