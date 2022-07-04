from tqdm import tqdm
import numpy as np
import os
import shutil
import torch

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def save_model(model, name, path):
    torch.save(model.state_dict(), os.path.join(path, name + '.pth'))

def load_unique_label(path):
    import pandas as pd
    df = pd.read_csv(path)
    unique_list = df["0"].tolist()
    df['index'] = [i for i in range(len(df))]
    unique_dic = df.set_index("0").to_dict()["index"]
    return unique_list, unique_dic

def get_label_index(label):
    unique_label_list, unique_dic = load_unique_label("../Data/fixed_unique_label.csv")
    L_shape = label.shape
    label_array = np.empty(L_shape)
    for i in tqdm(range(len(label)),desc = "Label indexing: " ):
        for j in range(len(label.columns)):
            if label.iloc[i,j] == None:
                label_array[i][j] = 3 #1787
            else:
                label_array[i][j] = unique_dic[label.iloc[i,j]]
    return label_array, unique_label_list

