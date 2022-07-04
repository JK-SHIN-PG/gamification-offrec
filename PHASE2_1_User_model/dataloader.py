import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
import numpy as np
import pandas as pd

class Transaction_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        #x = torch.tensor(self.data[idx], dtype = torch.float32)
        x = self.data[idx]
        label = self.labels[idx]
        #label = torch.tensor(self.labels[idx], dtype = torch.int64)
        return x, label

    def __len__(self):
        return len(self.labels)

def Data_load(data, index_label, BATCH_SIZE):
    
    train_dataset = Transaction_Dataset(data, index_label)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE,  num_workers = 8, persistent_workers=True)
    print("Data_loading : Complete!")
    return data.shape, train_loader
