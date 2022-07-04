import json
import pandas as pd
from env import Environment
from utils import *
from PHASE2_1_User_model.model import *
from PHASE1_Offline_learning.model import *
import sys
import os
import argparse
import numpy as np
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--RPATH", type=str, default='example')
    parser.add_argument("--UPATH", type=str, default='example')
    parser.add_argument("--save", type=str, default="example")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--lr", type=float, default=5*1e-4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ep", type=int, default=500)
    parser.add_argument("--lambda_", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.5)
    args = parser.parse_args()

    RECSYS_MODEL_PATH = '../Saved_file/RCN_model/{}/'.format(args.RPATH)
    USER_MODEL_PATH = '../Saved_file/User_model/{}/'.format(args.UPATH)
    MODEL_STORAGE_PATH = '../Saved_file/Online_learning/'
    DATA_SOURCE_PATH = '../Data/'

    ensure_path(MODEL_STORAGE_PATH + args.save)

    f = open(MODEL_STORAGE_PATH + args.save + "/report.txt", "a")
    f.write(str(vars(args)) + "\n")
    f.close()

    r_params = get_params(RECSYS_MODEL_PATH)
    u_params = get_params(USER_MODEL_PATH)
    # seed setting
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cnn_outdim = r_params["odim"]
    cnn_padding = r_params["pad"]
    cnn_kernel = r_params["ker"]
    cnn_stride = r_params["std"]
    cnn_maxp_ker = r_params["mpker"]
    gru_hiddim = r_params["gruhd"]
    gru_dropout = r_params["grudp"]
    lr = r_params["lr"]
    gru_ly = r_params["gruly"]
    dense_outdim = r_params["dsod"]
    maxlen = r_params["maxlen"]
    user_lr = u_params["lr"]
    user_embdim = u_params["emdim"]
    GPU_NUM = args.gpu
    lr = args.lr
    lamdba_ = args.lambda_
    epsilon = args.epsilon
    h = 32
    w = 54

    flat_dim = calculate_flatten_dim(
        h, w, cnn_outdim, cnn_padding, cnn_kernel, cnn_stride, cnn_maxp_ker)

    unique_label_list, unique_dic = load_unique_label(
        "../Data/fixed_unique_label.csv")
    device = torch.device(
        f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")

    n_actions = len(unique_label_list)

    Encoder = CNN(3, cnn_outdim, cnn_padding, cnn_kernel,
                  cnn_stride, cnn_maxp_ker, device)
    Encoder.load_state_dict(torch.load(
        RECSYS_MODEL_PATH + 'CNN.pth', map_location=device), strict=False)
    Encoder.to(device)

    Decoder = GRU_Network(flat_dim, n_actions, hidden_dim=gru_hiddim,
                          n_layers=gru_ly, device=device, drop_prob=gru_dropout)
    Decoder.load_state_dict(torch.load(
        RECSYS_MODEL_PATH + 'GRU.pth', map_location=device))
    Decoder.to(device)

    user_model = UserDecision(n_actions, user_embdim, device)
    user_model.load_state_dict(torch.load(
        USER_MODEL_PATH + 'user_model.pth', map_location=device))
    user_model.to(device)

    optimizer = optim.Adam(Decoder.parameters(), lr=lr)

    # data load
    loc_df = pd.read_csv(
        DATA_SOURCE_PATH + "Store_POG_PixelWorld.csv", engine="python", encoding="euc-kr")

    with open("../Data/Dataset_usermodel/planned_items_transaction_UserModel.json", "r") as jsonfile:
        data = json.load(jsonfile)

    map_arr = np.load(DATA_SOURCE_PATH+"mapping_array.npy")

    freqent_items = []
    for key in data.keys():
        freqent_items = freqent_items + data[key]["planned_item_sets"]
    freqent_items = list(set(freqent_items))

    Encoder.eval()
    user_model.eval()
    Decoder.train()

    episode_logprob = []
    episode_J = []
    episode_reward_price = []
    episode_reward_path = []
    max_price = 100000
    min_path = 35
    max_path = 200
    for ep in range(args.ep):
        optimizer.zero_grad()
        batch_logprob = 0
        batch_reward_path = 0
        batch_reward_price = 0
        batch_path = 0
        batch_size = 10

        sum_price_list = []
        sum_path_list = []
        for iter in range(batch_size):

            candidate_label_list = unique_label_list.copy()
            planned_item_list = [freqent_items[index] for index in np.random.choice(
                len(freqent_items), 3, replace=False)]
            p_idx_list = [candidate_label_list.index(
                p_item_list) for p_item_list in planned_item_list]
            for idx in p_idx_list + [3, 20559]:
                del candidate_label_list[idx]

            random_item_num = np.random.randint(0, 3)
            random_item_list = [candidate_label_list[index] for index in np.random.choice(
                len(candidate_label_list), random_item_num, replace=False)]

            env = Environment(loc_df, map_arr, unique_label_list,
                             planned_item_list + random_item_list, view_range=3, epsilon=epsilon)
            env.hidden = Decoder.init_hidden(1)
            # simulated interaction
            decision_list = []
            logprob = 0
            while(env.terminal != True):
                prediction, rec_product, hidden = env.get_recommended_item(
                    Encoder, Decoder, device)
                decision = env.decide_acceptance(
                    rec_product, user_model, device)
                response_result = env.get_response(
                    decision, rec_product, hidden)
                decision_list.append(response_result)
                logprob += response_result * \
                    torch.log(prediction.squeeze(0).squeeze(0)[rec_product])
                env.move_customer(response_result)

            path_length_per_item = len(env.whole_past_path)
            reward_path = np.log(
                (path_length_per_item - min_path) / (max_path - min_path))*(-1)
            #reward_path = path_length_per_item*(-1)
            reward_total_price = np.sum(np.array(
                [eval(item)["price"] for item in env.unplanned_product_list]).astype(int))
            sum_price_list.append(reward_total_price)
            sum_path_list.append(path_length_per_item)
            batch_logprob += logprob
            # transform to match scale with length of shopping
            batch_reward_price += np.log((reward_total_price +
                                         1) / (max_price))
            batch_reward_path += reward_path
            batch_path += path_length_per_item

        mean_batch_logprob = batch_logprob/batch_size
        mean_batch_reward_price = batch_reward_price/batch_size
        mean_batch_reward_path = batch_reward_path/batch_size
        mean_batch_path_length = batch_path/batch_size
        J = (-1) * mean_batch_logprob * (mean_batch_reward_path *
                                         lamdba_ + mean_batch_reward_price*(1-lamdba_))
        J.backward()
        optimizer.step()

        f = open(MODEL_STORAGE_PATH + args.save + "/report.txt", "a")
        f.write("epoch : {}\t logprob : {}\t J : {}\t LOS : {}\t TPP(won) : {}\n".format(ep, round(mean_batch_logprob.item(
        ), 2), round(J.item(), 2), round(mean_batch_path_length, 2), round(np.exp(mean_batch_reward_price)*max_price - 1, 2)))
        f.close()

        batch_max_price = max(sum_price_list)
        batch_min_path = min(sum_path_list)
        batch_max_path = max(sum_path_list)
        if batch_max_price > max_price:
            max_price = batch_max_price
        if batch_min_path < min_path:
            min_path = batch_min_path
        if batch_max_path > max_path:
            max_path = batch_max_path

    save_model(Decoder, "Controlled_GRU", MODEL_STORAGE_PATH + args.save)
