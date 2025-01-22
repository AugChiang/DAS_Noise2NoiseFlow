import torch
import random
import torchvision.transforms.functional as F

class ChannelNorm():
    """
    Normalize to mu=0, std=1 along width (col) given input tensor shape=(C,H,W).
    """
    def __call__(self, tensor:torch.Tensor):
        assert tensor.dim() == 3
        mu = torch.mean(tensor, dim=(0,1), keepdim=True)
        std = torch.std(tensor, dim=(0,1), keepdim=True)
        return (tensor - mu) / std
    
class LinearNorm01(): # this might not be appropiate to apply.
    """ Linearly normalize to [0,1]. """
    def __call__(self, tensor:torch.Tensor):
        assert tensor.dim() == 3
        m = torch.min(tensor)
        M = torch.max(tensor)
        return (tensor - m) / (M - m)
    
class LinearNorm1():
    """ Linearly normalize to [-1,1]. """
    def __call__(self, tensor):
        assert tensor.dim() == 3
        m = torch.min(tensor)
        M = torch.max(tensor)
        return ((tensor - m) / (M-m))*2-1

class RandomCropWithinBounds():
    """
    Crop a sub-tensor within the boundary (NO any paddings).
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, tensor:torch.Tensor):
        _, height, width = tensor.shape

        assert height > self.crop_size[0] or width > self.crop_size[1], "Crop size must be smaller than the image size"
        top = random.randint(0, height - self.crop_size[0])
        left = random.randint(0, width - self.crop_size[1])

        return F.crop(tensor, top, left, self.crop_size[0], self.crop_size[1])

class SampleClamp():
    """
    Clamp the outliers of the cropping.
    """
    def __init__(self, sample_clamp=5.0):
        self.sample_clamp = sample_clamp

    def __call__(self, tensor):
        tensor = torch.clamp(tensor,
                             min = -self.sample_clamp,
                             max =  self.sample_clamp)
        return tensor
