import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import pandas as pd

class Transaction_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.int64)
    
    def __getitem__(self, idx):
        #x = torch.tensor(self.data[idx], dtype = torch.float32)
        x = self.data[idx]
        #label = torch.tensor(self.labels[idx], dtype = torch.int64)
        label = self.labels[idx]
        return x, label

    def __len__(self):
        return len(self.labels)

def Data_load(data, index_label, BATCH_SIZE):
    train_dataset = Transaction_Dataset(data, index_label)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE,  num_workers = 8, persistent_workers=True)
    print("Data_loading : Complete!")
    return data.shape, train_loader

def collect_data(data_path, mode):

    data_list = glob(data_path + "/*.npy")
    label_list = glob(data_path + "/*.csv")
    data_list = data_filter(data_list, mode)
    label_list = data_filter(label_list, mode)
    data = []
    for d_idx, d_path in enumerate(data_list):
        temp_data = np.load(d_path)
        if d_idx == 0:
            data = temp_data
        else:
            data = np.concatenate((data, temp_data), axis = 0)

    label = pd.DataFrame()
    for l_idx, l_path in enumerate(label_list):
        temp_label = pd.read_csv(l_path, "\t", header=None)
        if l_idx == 0:
            label = temp_label
        else:
            label = label.append(temp_label)

    print("Data_collecting : Complete!")
    return data, label

def data_filter(data_list, mode):
    temp_list = []
    for i in range(len(data_list)):
        par = data_list[i]
        strlist = par.split("/")
        filename = strlist[-1].split("_")
        file_mode = filename[-1].split(".")[0]
        if mode == "train":
            if file_mode not in ["test", "valid"]:
                temp_list.append(par)
        elif mode == "valid":
            if file_mode == "valid":
                temp_list.append(par)
        else:
            if file_mode == "test":
                temp_list.append(par)
    return temp_list


