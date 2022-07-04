import datetime
import time
import warnings
from model import *
import torch.optim as optim
import torch.nn as nn
import torch
from utils import get_label_index, calculate_flatten_dim, to_one_hot, ensure_path, save_model
from dataloader import Data_load, collect_data
import sys
import os
import argparse
__file__ = "train.py"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter(action='ignore', category=FutureWarning)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--ep", type=int, default=100)
    parser.add_argument("--bth", type=int, default=1024)
    parser.add_argument("--maxlen", type=int, default=20)
    parser.add_argument("--odim", type=int, default=2)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--ker", type=int, default=2)
    parser.add_argument("--std", type=int, default=1)
    parser.add_argument("--mpker", type=int, default=2)
    parser.add_argument("--gruhd", type=int, default=256)
    parser.add_argument("--gruly", type=int, default=2)
    parser.add_argument("--grudp", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save", default='trial-1')
    args = parser.parse_args()
    print(vars(args))

    DATA_SOURCE_PATH = '../Data/2021_MAP'
    MODEL_STORAGE_PATH = '../Saved_file/RCN_model/'
    BATCH_SIZE = args.bth
    GPU_NUM = args.gpu

    ensure_path(MODEL_STORAGE_PATH + args.save)

    f = open(MODEL_STORAGE_PATH + args.save + "/report.txt", "a")
    f.write(str(vars(args)) + "\n")
    f.close()

    device = torch.device(
        f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")

    with torch.cuda.device(f"cuda:{GPU_NUM}"):
        train_data, train_label = collect_data(DATA_SOURCE_PATH, mode="train")
        train_label = train_label[train_label.iloc[:, 3] !=
                                  "{'상품코드': '<eos>', '상품명': '<eos>', 'row': '14', 'column': '0', 'price': '0'}"]
        train_data = train_data[train_label.index]
        train_index_label, unique_label_list, unique_dic = get_label_index(
            train_label)

        n_actions = len(unique_label_list)

        in_shape, train_loader = Data_load(
            train_data, train_index_label, BATCH_SIZE)

        in_dim = in_shape[-1]
        h = in_shape[2]
        w = in_shape[3]

        flat_dim = calculate_flatten_dim(
            h, w, args.odim, args.pad, args.ker, args.std, args.mpker)

        Encoder = CNN(in_dim, args.odim, args.pad, args.ker,
                      args.std, args.mpker, device)
        Encoder.to(device)

        Decoder = GRU_Network(
            flat_dim, n_actions, hidden_dim=args.gruhd, n_layers=args.gruly, device=device, drop_prob=args.grudp)
        Decoder.to(device)

        learning_rate = args.lr

        optimizer_ec = optim.Adam(Encoder.parameters(), lr=learning_rate)
        optimizer_dc = optim.Adam(Decoder.parameters(), lr=learning_rate)

        Encoder.train()
        Decoder.train()
        best_loss = 1000000
        losses = []
        padding_idx = 3

        for ep in range(args.ep):
            start = time.time()
            episode_loss = 0
            episode_acc = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                optimizer_ec.zero_grad()
                optimizer_dc.zero_grad()
                data = data.permute(1, 0, 4, 2, 3)
                label = label.transpose(1, 0)
                data = data.to(device)
                hidden = Decoder.init_hidden(data.shape[1])
                loss = 0
                e_loss = 0
                correct = 0
                empty_basket = torch.ones(
                    (data.shape[0], data.shape[1])) * int(3)  # embedding
                count_without_padding = 0

                for i in range(1, args.maxlen):  # Given(current location) -> predict next item
                    # Encode
                    embedding_tensor = Encoder(data[i-1])  # current location
                    empty_basket[i-1] = label[i-1]
                    current_basket = empty_basket.transpose(
                        1, 0).type(torch.LongTensor)
                    # Decode
                    prediction, hidden = Decoder(
                        embedding_tensor.unsqueeze(1), hidden)

                    onehot_label = to_one_hot(label[i], num_classes=n_actions).type(
                        torch.float32).to(device)  # teacher force!
                    # padding masking (padding is excepted to calucate loss)
                    loss_mask = label[i] != padding_idx
                    prob = torch.matmul(
                        prediction, onehot_label.unsqueeze(-1).float())
                    prob = prob.squeeze(-1).squeeze(-1)
                    logprob = torch.log(prob)
                    logprob_masked = logprob.where(loss_mask.to(
                        device), torch.tensor(0.0).to(device))
                    logp_zero_mask = logprob_masked != 0
                    mean_logprob = torch.div(
                        (logprob_masked*logp_zero_mask).sum(dim=0), (logp_zero_mask.sum(dim=0) + 1e-6))
                    loss += -1 * mean_logprob
                    e_loss += loss.item()

                    # calculate accuracy
                    pred = prediction.clone().detach()
                    pred_onehot = torch.argmax(pred, dim=-1).squeeze(-1)
                    answer_onehot = torch.argmax(onehot_label, dim=-1)
                    pred_onehot_mask = pred_onehot != padding_idx
                    answer_onehot_mask = answer_onehot != padding_idx
                    pred_onehot_masked = pred_onehot.where(
                        pred_onehot_mask, torch.tensor(-1).to(device))
                    answer_onehot_masked = answer_onehot.where(
                        answer_onehot_mask, torch.tensor(-2).to(device))
                    answer_zero_mask = answer_onehot_masked != -2
                    correct += torch.div((answer_onehot_masked ==
                                         pred_onehot_masked).sum(), (answer_zero_mask.sum() + 1e-6))
                    if answer_zero_mask.sum() != 0:
                        count_without_padding += 1

                loss.backward()
                optimizer_dc.step()
                optimizer_ec.step()
                episode_loss += e_loss
                batch_acc = correct/count_without_padding
                episode_acc += batch_acc

            end = time.time()

            if ep % 5 == 0:
                sec = (end - start)
                sec_re = str(datetime.timedelta(seconds=sec)).split(".")

                f = open(MODEL_STORAGE_PATH + args.save + "/report.txt", "a")

                f.write("Train epoch : {0} \t Loss: {1:0.4f} \t Accuracy : {2:.4f}% \t Time/epoch (s)  : {3} \n".format(
                    ep, episode_loss/len(train_loader), (episode_acc*100)/len(train_loader), sec_re[0]))
                f.close()
                print("Report! (ep {})".format(ep))
                if ep % 50 == 0:
                    save_model(Decoder, "GRU{}".format(ep),
                               MODEL_STORAGE_PATH + args.save)
                    save_model(Encoder, "CNN{}".format(ep),
                               MODEL_STORAGE_PATH + args.save)

            if best_loss > episode_loss:
                best_loss = episode_loss
                save_model(Decoder, "GRU", MODEL_STORAGE_PATH + args.save)
                save_model(Encoder, "CNN", MODEL_STORAGE_PATH + args.save)

        print("Done")
