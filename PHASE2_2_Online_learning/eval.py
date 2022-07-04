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
    parser.add_argument("--save", type=str, default='example')
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epsilon", type=float, default="0.3")
    args = parser.parse_args()

    DATA_SOURCE_PATH = '../Data/'
    MODEL_STORAGE_PATH = '../Saved_file/Online_learning/{}/'.format(args.save)
    online_params = get_params(MODEL_STORAGE_PATH)
    lambda_ = online_params["lambda_"]

    RECSYS_MODEL_PATH = '../Saved_file/RCN_model/{}/'.format(
        online_params["RPATH"])
    USER_MODEL_PATH = '../Saved_file/User_model/{}/'.format(
        online_params["UPATH"])
    r_params = get_params(RECSYS_MODEL_PATH)
    u_params = get_params(USER_MODEL_PATH)

    # seed and hyper-parameter setting
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

    # data load
    loc_df = pd.read_csv(
        DATA_SOURCE_PATH + "Store_POG_PixelWorld.csv", engine="python", encoding="euc-kr")
    map_arr = np.load(DATA_SOURCE_PATH+"mapping_array.npy")

    # test planned itemsets for validation
    test_planned = []
    f = open(DATA_SOURCE_PATH + "test_planned_itemsets.txt", "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        sets = line.split("\t")
        test_planned.append(sets)

    Encoder = CNN(3, cnn_outdim, cnn_padding, cnn_kernel,
                  cnn_stride, cnn_maxp_ker, device)
    Encoder.load_state_dict(torch.load(
        RECSYS_MODEL_PATH + 'CNN.pth'), strict=False)
    Encoder.to(device)

    Decoder = GRU_Network(flat_dim, n_actions, hidden_dim=gru_hiddim,
                          n_layers=gru_ly, device=device, drop_prob=gru_dropout)
    Decoder.load_state_dict(torch.load(
        MODEL_STORAGE_PATH + 'Controlled_GRU.pth', map_location=device))
    Decoder.to(device)

    user_model = UserDecision(n_actions, user_embdim, device)
    user_model.load_state_dict(torch.load(USER_MODEL_PATH + 'user_model.pth',map_location=device))
    user_model.to(device)

    Encoder.eval()
    Decoder.eval()
    user_model.eval()

    # test for online recommender
    eval_episode_logprob = []
    eval_episode_reward_price = []
    eval_episode_reward_path = []
    eval_episode_accept_rate = []

    for ep in range(100):
        env = Environment(loc_df, map_arr, unique_label_list,
                         test_planned[ep], view_range=3, epsilon=epsilon)
        env.hidden = Decoder.init_hidden(1)
        # simulated interaction
        decision_list = []
        logprob = 0
        while(env.terminal != True):
            prediction, rec_product, hidden = env.get_recommended_item(
                Encoder, Decoder, device)
            decision = env.decide_acceptance(rec_product, user_model, device)
            response_result = env.get_response(decision, rec_product, hidden)
            decision_list.append(response_result)
            logprob += response_result * \
                torch.log(prediction.squeeze(0).squeeze(0)[rec_product])
            env.move_customer(response_result)

        path_length_per_item = len(env.whole_past_path)
        reward_total_price = np.sum(np.array(
            [eval(item)["price"] for item in env.unplanned_product_list]).astype(int))

        eval_episode_logprob.append(logprob.item())
        eval_episode_reward_price.append(np.sqrt(reward_total_price*0.0001))
        eval_episode_reward_path.append(path_length_per_item)
        eval_episode_accept_rate.append(
            np.sum(decision_list)/len(decision_list))

    f = open(MODEL_STORAGE_PATH + "validation_report.txt", 'w')
    f.write("lambda : {} \n".format(lambda_))
    f.write("Acceptance Ratio : {}\t logprob : {}\n".format(round(np.mean(
        eval_episode_accept_rate)*100, 2), round(np.mean(eval_episode_logprob), 3)))
    f.write("LOS : {}\t TPP(won) : {}".format(round(np.mean(eval_episode_reward_path),
            2), round(np.mean(np.square(eval_episode_reward_price))*10000, 2)))
