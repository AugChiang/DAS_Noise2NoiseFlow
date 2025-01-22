from torch.utils.data import Dataset
from glob import glob
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(
        self,
        root_dir:str,
        crop_size:tuple,
        transform=None
    ):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.transform = transform
        self.data_lists = glob(os.path.join(root_dir, "*.npy"))
    
    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, idx):
        arr = np.load(self.data_lists[idx]) # (Time,Channels,2)
        if self.transform:
            arr = self.transform(arr) # (2,Time,Channels)  
        return arr[0].unsqueeze(0), arr[1].unsqueeze(0)