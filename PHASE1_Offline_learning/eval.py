import warnings
from model import *
import torch
from utils import *
from dataloader import Data_load, collect_data
import sys
import os
import argparse
__file__ = "train.py"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--save", type=str, default='example')
    parser.add_argument("--bth", type=int, default=1024)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()

    RECSYS_MODEL_PATH = '../Saved_file/RCN_model/{}/'.format(args.save)
    DATA_SOURCE_PATH = '../Data/2021_MAP'
    GPU_NUM = args.gpu

    f = open(RECSYS_MODEL_PATH + "/validation_report.txt", "a")
    f.write(str(vars(args)) + "\n")
    f.close()

    device = torch.device(
        f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
    valid_data, valid_label = collect_data(DATA_SOURCE_PATH, mode="test")

    valid_index_label, unique_label_list, unique_dic = get_label_index(
        valid_label)

    index_label_dic = dict((y, x) for x, y in unique_dic.items())

    n_actions = len(unique_label_list)
    in_shape, valid_loader = Data_load(valid_data, valid_index_label, args.bth)

    r_params = get_params(RECSYS_MODEL_PATH)

    in_dim = in_shape[-1]
    h = in_shape[2]
    w = in_shape[3]

    cnn_outdim = r_params["odim"]
    cnn_padding = r_params["pad"]
    cnn_kernel = r_params["ker"]
    cnn_stride = r_params["std"]
    cnn_maxp_ker = r_params["mpker"]
    gru_hiddim = r_params["gruhd"]
    gru_dropout = r_params["grudp"]
    lr = r_params["lr"]
    gru_ly = r_params["gruly"]
    maxlen = r_params["maxlen"]

    flat_dim = calculate_flatten_dim(
        h, w, cnn_outdim, cnn_padding, cnn_kernel, cnn_stride, cnn_maxp_ker)

    Encoder = CNN(in_dim, cnn_outdim, cnn_padding, cnn_kernel,
                  cnn_stride, cnn_maxp_ker, device)
    Encoder.load_state_dict(torch.load(
        RECSYS_MODEL_PATH + 'CNN.pth'), strict=False)
    Encoder.to(device)

    Decoder = GRU_Network(flat_dim, n_actions, hidden_dim=gru_hiddim,
                          n_layers=gru_ly, device=device, drop_prob=gru_dropout)
    Decoder.load_state_dict(torch.load(RECSYS_MODEL_PATH + 'GRU.pth'))
    Decoder.to(device)

    Encoder.eval()
    Decoder.eval()

    for k in tqdm([1, 5, 20], desc="validation in progress"):
        print("k :", k)
        padding_idx = 3

        data_precision = 0
        data_recall = 0
        data_ndcg = 0
        data_ndcg_with_rel = 0
        data_map = 0
        data_map_with_rel = 0

        option = "상품"
        m_metric = multi_metric(index_label_dic, option)

        with torch.cuda.device(f"cuda:{GPU_NUM}"):
            for iter in range(5):  # 5 iteration
                for idx, (data, label) in enumerate(valid_loader):
                    with torch.no_grad():
                        data = data.permute(1, 0, 4, 2, 3)
                        data = data.to(device)
                        label = label.transpose(1, 0)
                        hidden = Decoder.init_hidden(data.shape[1])
                        loss = 0
                        token_loss = 0
                        correct = 0
                        empty_basket = torch.ones(
                            (data.shape[0], data.shape[1])) * int(3)  # embedding

                        batch_precision = 0
                        batch_recall = 0
                        batch_ndcg = 0
                        batch_ndcg_with_rel = 0
                        batch_map = 0
                        batch_map_with_rel = 0
                        count_without_padding = 0
                        for i in range(1, maxlen-1):
                            # Encode
                            embedding_tensor = Encoder(data[i-1])
                            # Decode
                            prediction, hidden = Decoder(
                                embedding_tensor.unsqueeze(1), hidden)

                            if i >= 3:
                                mean_precision, mean_recall, mean_ndcg, mean_ndcg_with_rel, mean_map, mean_map_with_rel, metric_minus_masked_list = masked_metric(
                                    m_metric.metrics, prediction, k, label[i], padding_idx)
                                if metric_minus_masked_list.sum() != 0:
                                    count_without_padding += 1

                                batch_precision += mean_precision
                                batch_recall += mean_recall
                                batch_ndcg += mean_ndcg
                                batch_ndcg_with_rel += mean_ndcg_with_rel
                                batch_map += mean_map
                                batch_map_with_rel += mean_map_with_rel

                        data_precision += batch_precision/count_without_padding
                        data_recall += batch_recall/count_without_padding
                        data_ndcg += batch_ndcg/count_without_padding
                        data_ndcg_with_rel += batch_ndcg_with_rel/count_without_padding
                        data_map += batch_map/count_without_padding
                        data_map_with_rel += batch_map_with_rel/count_without_padding

            #print("precision : ",data_precision/(len(valid_loader)*5))
            #print("hit_ratio : ", data_recall/(len(valid_loader)*5))
            #print("ndcg : ",data_ndcg/(len(valid_loader)*5))
            #print("ndcg with rel : ",data_ndcg_with_rel/(len(valid_loader)*5))
            #print("MAP : ",data_map/(len(valid_loader)*5))
            #print("MAP with rel : ",data_map_with_rel/(len(valid_loader)*5))

            HR = round(data_recall.item()/(len(valid_loader)*5), 4)
            prec = round(data_precision.item()/(len(valid_loader)*5), 4)
            NG = round(data_ndcg.item()/(len(valid_loader)*5), 4)
            NG_with_rel = round(data_ndcg_with_rel.item() /
                                (len(valid_loader)*5), 4)
            MAP = round(data_map.item()/(len(valid_loader)*5), 4)
            MAP_rel = round(data_map_with_rel.item()/(len(valid_loader)*5), 4)

            f = open(RECSYS_MODEL_PATH + "/validation_report.txt", "a")
            f.write("K : {}\t HR : {}\t prec : {} \n".format(k, HR, prec))
            f.write("NG : {}\t NG_product_level : {} \n".format(NG, NG_with_rel))
            f.write("MAP : {}\t MAP_product_level : {} \n".format(MAP, MAP_rel))
            f.write("\n")
            f.close()
