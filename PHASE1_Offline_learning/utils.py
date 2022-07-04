import os
import numpy as np
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
import shutil
import os.path as osp
from tqdm import tqdm
import random

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def save_model(model, name, path):
    torch.save(model.state_dict(), osp.join(path, name + '.pth'))

def get_whole_labels(label):
    unique_list = []
    for D in label.columns:
        unique_list = unique_list + label[D].unique().tolist()
        unique_list = list(set(unique_list))
    return unique_list

def get_label_index(label):
    unique_label_list, unique_dic = load_unique_label("../Data/fixed_unique_label.csv")
    L_shape = label.shape
    label_array = np.empty(L_shape)
    for i in tqdm(range(len(label)),desc = "Label indexing: " ):
        for j in range(len(label.columns)):
            label_array[i][j] = unique_dic[label.iloc[i,j]]
    return label_array, unique_label_list, unique_dic

def load_unique_label(path):
    import pandas as pd
    df = pd.read_csv(path)
    unique_list = df["0"].tolist()
    df['index'] = [i for i in range(len(df))]
    unique_dic = df.set_index("0").to_dict()["index"]
    return unique_list, unique_dic

def to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def multi_onehot(label, n_class):
    labels = label.clone().detach()
    if label.dim() == 1:
        labels = labels.unsqueeze(1)
    else:
        labels = labels.transpose(1,0)

    target = torch.zeros(labels.size(0), n_class).scatter_(1, labels, 1.)*1000
    return target

def generate_current_basket(BATCH_SIZE , unique_label_list, label):
    current_busket = torch.empty((BATCH_SIZE, len(unique_label_list)))
    for bidx in range(label.shape[0]):
        temp = []
        for iidx in range(label.shape[1]):
            if iidx == 0:
                current_busket[bidx][iidx] = torch.zeros(len(unique_label_list))
            else:
                temp.append(label[bidx][iidx-1].data.item())
                onehot = multi_onehot(temp, len(unique_label_list))
                current_busket[bidx][iidx] = onehot
    return current_busket

def calculate_cnn_dim(in_shape, padding, kernel, stride):
    return int(((in_shape + 2*padding - kernel)/stride) + 1)

def calculate_flatten_dim(h,w,out_d, padding, kernel, stride, max_pooling_kernel):
    w_dim = calculate_cnn_dim(w, padding, kernel, stride)
    h_dim = calculate_cnn_dim(h, padding, kernel, stride)
    w_dim = int(w_dim/max_pooling_kernel)
    h_dim = int(h_dim/max_pooling_kernel)
    return w_dim * h_dim * out_d

def plot(rewards,  y_title):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.title('')
    plt.ylabel(y_title)
    plt.xlabel('episode')
    plt.plot(rewards)
    plt.show()

def get_params(MODEL_PATH):

    f = open(MODEL_PATH + "report.txt")
    param = f.readline()
    param = param.strip()
    param = eval(param)
    f.close()

    return param

def masked_metric(metric_type, prediction, k, true_label_idx, padding_idx):

    mask_array = true_label_idx != padding_idx
    prec_list = []
    recall_list = []
    ndcg_list = []
    ndcg_with_rel_list = []
    map_list = []
    map_with_rel_list = []

    for j in range(len(true_label_idx)):
        precision, recall, ndcg, ndcg_with_rel, Map, Map_with_rel = metric_type(true_label_idx[j].item(), prediction[j], k)
        prec_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)
        ndcg_with_rel_list.append(ndcg_with_rel)
        map_list.append(Map)
        map_with_rel_list.append(Map_with_rel)
        
    mean_precision, metric_minus_masked_list = calculate_mean_metric(prec_list, mask_array)
    mean_recall, _ = calculate_mean_metric(recall_list,mask_array)
    mean_ndcg, _ = calculate_mean_metric(ndcg_list, mask_array)
    mean_ndcg_with_rel, _ = calculate_mean_metric(ndcg_with_rel_list, mask_array)
    mean_map, _ = calculate_mean_metric(map_list, mask_array)
    mean_map_with_rel, _ = calculate_mean_metric(map_with_rel_list, mask_array)

    return mean_precision, mean_recall, mean_ndcg,mean_ndcg_with_rel,mean_map, mean_map_with_rel, metric_minus_masked_list

def calculate_mean_metric(metric_list, mask_array):
    metric_list = torch.tensor(metric_list).type(torch.float32)
    metric_masked_list = metric_list.where(mask_array, torch.tensor(-1.0))
    metric_minus_masked_list = metric_masked_list != -1
    mean_metric= torch.div(((metric_masked_list*metric_minus_masked_list).sum()), metric_minus_masked_list.sum() + 1e-6)

    return mean_metric, metric_minus_masked_list


class multi_metric():
    def __init__(self, index_label_dic,option):
        self.index_label_dic = index_label_dic
        self.option = option

    def metrics(self,true_value, prediction_idx, k): #precision, 
        _, topk_item =torch.topk(prediction_idx, k)
        topk_item_list_idx = topk_item.squeeze(0).clone().detach().cpu().numpy()
        rel_score_list = self.calculate_rel_with_weight(true_value, topk_item_list_idx)
        equal_rel_score_list = self.calculate_rel(true_value, topk_item_list_idx)
        rel_score_list = np.array(rel_score_list)
        equal_rel_score_list = np.array(equal_rel_score_list)

        precision = self.precision(equal_rel_score_list)
        recall = self.recall(true_value, topk_item_list_idx, equal_rel_score_list)
        ndcg_with_rel = self.ndcg(rel_score_list)
        ndcg = self.ndcg(equal_rel_score_list)
        Map = self.mean_ap_rel(equal_rel_score_list)
        Map_with_rel = self.mean_ap_rel(rel_score_list)

        return precision, recall, ndcg, ndcg_with_rel, Map, Map_with_rel

    def precision(self, rel_score_list):
        finder = np.where(np.array(rel_score_list) != 0)
        return len(finder[0])/len(rel_score_list)

    def recall(self, true_value, topk_item_list_idx, rel_score_list):
        finder = np.where(np.array(rel_score_list) != 0)
        #finder = np.where(np.array(topk_item_list_idx) == true_value)
        if len(finder[0]) == 0:
            return 0
        else:
            return 1

    def precision_with_rel(self, rel_score_list, topk_item_list_idx):
        return np.sum(rel_score_list / len(topk_item_list_idx))
    
    def MRR(self, rel_score_list):
        return np.sum(rel_score_list/np.arange(1,rel_score_list.size+1))
    
    def dcg(self,rel_score_list):
        return np.sum(rel_score_list/ np.log2(np.arange(2,rel_score_list.size+2))), list(rel_score_list)

    def mean_ap_rel(self, rel_score_list):
        map = 0

        for k in range(len(rel_score_list)):
            temp = rel_score_list[:k+1]
            map += np.sum(temp) / len(temp)
        return map/len(rel_score_list)

    def mean_ap(self, rel_score_list):
        map = 0
        for k in range(len(rel_score_list)):
            temp = rel_score_list[:k+1]
            finder = np.where(np.array(temp) != 0)
            map += len(finder[0]) / len(temp)
        return map/len(rel_score_list)

    def ndcg(self,rel_score_list):
        dcg_value, rel_score_list = self.dcg(rel_score_list)
        ideal_rel_score_list = sorted(rel_score_list, reverse=True)
        ideal_rel_score_list = np.array(ideal_rel_score_list)
        idcg_value = np.sum(ideal_rel_score_list/ np.log2(np.arange(2,ideal_rel_score_list.size+2)))
        return dcg_value/(idcg_value+1e-6)
    

    def extract_product_class(self, item):
        true_product_inf = eval(self.index_label_dic[item])
        if str(true_product_inf["상품코드"]) in ["<bos>", "<eos>"]:
            high_class = "None"
            mid_class = "None"
            sml_class = "None"
            code = "None"
        else:
            high_class = true_product_inf["대분류"]
            mid_class = true_product_inf["중분류"]
            sml_class = true_product_inf["소분류"]
            code = true_product_inf["상품코드"]
        return high_class, mid_class, sml_class, code


    def calculate_rel(self, true_value, topk_item_list_idx):     # 추천상품에 대한 관련도 계산 - 옵션에 따라 다름

        rel_score_list = []
        t_high, t_mid, t_sml, t_code = self.extract_product_class(true_value)
        #print(topk_item_list_idx)
        #print(topk_item_list_idx.shape)
        for token in topk_item_list_idx:
            high, mid, sml, code = self.extract_product_class(token)
            score = 0
            if self.option == "대분류":
                if t_high == high:
                    score = 1
            elif self.option =="중분류":
                if t_mid == mid:
                    score = 1
            elif self.option =="소분류":
                if t_sml == sml:
                    score = 1
            else:
                if str(t_code) == str(code):
                    score = 1
            rel_score_list.append(score)
        
        return rel_score_list

    def calculate_rel_with_weight(self, true_value, topk_item_list_idx):
        rel_score_list = []
        t_high, t_mid, t_sml, t_code = self.extract_product_class(true_value)
        for token in topk_item_list_idx:
            high, mid, sml, code = self.extract_product_class(token)
            score = 0
            if t_high == high:
                score += 1
            if t_mid == mid:
                score += 1
            if t_sml == sml:
                score += 1
            if str(t_code) == str(code):
                score += 1
            rel_score_list.append(score)
        return rel_score_list

def generate_negative_sample(label):
    negative_tensor = torch.empty((label.shape[0], label.shape[1])).long()
    padding_mask = label!= 3
    negative_sample = label[padding_mask]
    for i in range(len(label)): 
        padding_mask = label!= 3
        negative_sample = label[padding_mask]
        col_idxs = list(range(negative_sample.shape[0]))
        random.shuffle(col_idxs)
        negative_tensor[i] = negative_sample[col_idxs][:label.shape[1]]
    return negative_tensor

def generate_negative_sample_NEW(label):
    negative_tensor = torch.empty((label.shape[0], 1)).long()
    padding_mask = label!= 3
    negative_sample = label[padding_mask]
    for i in range(len(label)): 
        padding_mask = label!= 3
        negative_sample = label[padding_mask]
        col_idxs = list(range(negative_sample.shape[0]))
        random.shuffle(col_idxs)
        negative_tensor[i] = negative_sample[col_idxs][0]
    return negative_tensor