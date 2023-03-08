import torch.nn as nn
import torch
from torch.nn import MSELoss

class MSELossExcludeNotChangedTemp(nn.MSELoss):
    def __init__(self, size_average=None, reduction: str = 'mean', ignore_temp:float=0.0) -> None:
        super(nn.MSELoss, self).__init__(size_average, reduction)
        self.ignore_temp = ignore_temp

    def forward(self, inputs:torch.tensor, targets:torch.tensor):
        assert inputs.shape == targets.shape, f"inputs.shape: {inputs.shape}, targets.shape: {targets.shape}"
        mask_inputs = inputs != self.ignore_temp # check where temperature is not the ignore temperature in inputs
        mask_targets = targets != self.ignore_temp # check where temperature is not the ignore temperature in targets
        mask = torch.logical_or(mask_inputs, mask_targets)
        out = torch.mean((inputs[mask]-targets[mask])**2)
        return out
    
class MSELossExcludeYBoundary(nn.MSELoss):
    """ loss specifically excludes a little of the y-boundary because there are some artifacts there (from the simulation)"""
    def __init__(self, **kwargs) -> None:
        super(nn.MSELoss, self).__init__(**kwargs)

    def forward(self, inputs:torch.tensor, targets:torch.tensor):
        # expects input in shape (batch, channels, height, width)
        assert len(inputs.shape) == 4, f"inputs.shape: {inputs.shape}, expects (batch, channels, height, width)"
        out = torch.mean((inputs[:,:,1:-2,:]-targets[:,:,1:-2,:])**2)
        return out
    
class InfinityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs:torch.tensor, targets:torch.tensor):
        out = torch.max(torch.abs(inputs-targets))
        return out

def normalize(x, mean, std):
    return (x - mean) / std

if __name__ == "__main__":
    tensor1 = torch.tensor([[10.6, 12], [12, 10.6]])
    tensor2 = torch.tensor([[10.6, 12], [10.6, 12]])
    loss_fn = MSELossExcludeNotChangedTemp(ignore_temp=10.6)
    loss_value = loss_fn(tensor1, tensor2)
    print(loss_value)
