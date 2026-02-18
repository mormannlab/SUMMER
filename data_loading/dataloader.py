"""
Setup for nwb dataloader to use DHV data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

class DHVDataset(Dataset):
    def __init__(self, nwb_file, transform=None):
        pass