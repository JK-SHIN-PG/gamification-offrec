import torch
import shutil
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PHASE0_Data_preprocess.map_generator import *
from PHASE0_Data_preprocess.sequence_generator import *
from PHASE0_Data_preprocess.A_star import *

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def save_model(model, name, path):
    torch.save(model.state_dict(), os.path.join(path, name + '.pth'))


def get_params(MODEL_PATH):
    f = open(MODEL_PATH + "report.txt")
    param = f.readline()
    param = param.strip()
    param = eval(param)
    f.close()
    return param


def get_label_index(label):
    unique_label_list, unique_dic = load_unique_label(
        "../Data/fixed_unique_label.csv")
    L_shape = label.shape
    label_array = np.empty(L_shape)
    for i in tqdm(range(len(label)), desc="Label indexing: "):
        for j in range(len(label.columns)):
            label_array[i][j] = unique_dic[label.iloc[i, j]]
    return label_array, unique_label_list


def load_unique_label(path):
    import pandas as pd
    df = pd.read_csv(path)
    unique_list = df["0"].tolist()
    df['index'] = [i for i in range(len(df))]
    unique_dic = df.set_index("0").to_dict()["index"]
    return unique_list, unique_dic


def multi_onehot(label, n_class):
    labels = label.clone().detach()
    if label.dim() == 1:
        labels = labels.unsqueeze(1)
    else:
        labels = labels.transpose(1, 0)
    target = torch.zeros(labels.size(0), n_class).scatter_(1, labels, 1.)*1000
    return target


def TSP_solver(current_pos, sequence_trans, loc_df):
    Seq_Gen = Sequence_generation(loc_df)
    sequence = []
    for trans in sequence_trans:
        trans = eval(trans)
        sequence.append((int(trans["row"]), int(trans["column"])))
    sub_list = [seq for seq in sequence if seq != 0]
    sort_idx_list = Seq_Gen.Grid_TSP_based_sequence_generation(
        current_pos, sub_list, (14, 0), option="euc")
    sort_idx_list.remove(len(sort_idx_list)-1)
    sort_idx_list.remove(0)
    sort_idx_list = list(np.array(sort_idx_list)-1)
    seq_list = [sequence_trans[idx] for idx in sort_idx_list]
    return seq_list


def calculate_cnn_dim(in_shape, padding, kernel, stride):
    return int(((in_shape + 2*padding - kernel)/stride) + 1)


def calculate_flatten_dim(h, w, out_d, padding, kernel, stride, max_pooling_kernel):
    w_dim = calculate_cnn_dim(w, padding, kernel, stride)
    h_dim = calculate_cnn_dim(h, padding, kernel, stride)
    w_dim = int(w_dim/max_pooling_kernel)
    h_dim = int(h_dim/max_pooling_kernel)
    return w_dim * h_dim * out_d
