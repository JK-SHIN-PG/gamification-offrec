import sys
import os
import argparse
from model import *
from utils import *
import pandas as pd
from dataloader import *
import torch.optim as optim
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--ep", type=int, default=2000)
    parser.add_argument("--bth", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--save", type=str, default="example")
    parser.add_argument("--emdim", type=int, default=32)

    args = parser.parse_args()

    GPU_NUM = args.gpu
    BATCH_SIZE = args.bth
    learning_rate = args.lr
    epoch = args.ep
    embedding_dim = args.emdim
    MODEL_STORAGE_PATH = '../Saved_file/User_model/'

    ensure_path(MODEL_STORAGE_PATH + args.save)

    f = open(MODEL_STORAGE_PATH + args.save + "/report.txt", "a")
    f.write(str(vars(args)) + "\n")
    f.close()

    with open("../Data/Dataset_usermodel/planned_items_transaction_UserModel.json", "r") as jsonfile:
        data_plan = json.load(jsonfile)

    with open("../Data/Dataset_usermodel/random_items_transaction_UserModel.json", "r") as jsonfile:
        data_random = json.load(jsonfile)

    data_list = [data_plan, data_random]

    labels = torch.Tensor([])
    total_seq_tensor = torch.Tensor([])

    for idx, data in enumerate(data_list):
        transaction = []
        for key in data.keys():
            for t in data[key]["transactions"]:
                if t not in transaction:
                    transaction.append(t)
        transaction = pd.DataFrame(transaction)
        train_index_label, unique_label_list = get_label_index(transaction)
        train_index_label = torch.LongTensor(train_index_label)  # x_data
        train_shape = train_index_label.shape

        seq_tensor = torch.Tensor([])

        for i in range(train_shape[0]):
            empty_tensor = torch.ones(
                (train_shape[1], train_shape[1])) * int(3)  # eos padding
            for j in range(empty_tensor.shape[0]):
                for k in range(j):
                    if int(train_index_label[i][k].item()) == 3:
                        break
                    else:
                        empty_tensor[j][k] = train_index_label[i][k].item()
            ide = sum(train_index_label[0] != 3)
            seq_tensor = torch.cat((seq_tensor, torch.hstack(
                (train_index_label[i][:ide].unsqueeze(1), empty_tensor[:ide]))))
            if i > 4288:
                break
        if idx == 0:
            #label_data = torch.ones(seq_tensor.shape[0])
            label_data = torch.zeros(seq_tensor.shape[0])
        else:
            #label_data = torch.zeros(seq_tensor.shape[0])
            label_data = torch.ones(seq_tensor.shape[0])

        labels = torch.cat((labels, label_data))
        total_seq_tensor = torch.cat((total_seq_tensor, seq_tensor))

    n_actions = len(unique_label_list)
    total_seq_tensor = total_seq_tensor.type(torch.LongTensor)
    #labels = labels.type(torch.LongTensor)
    sep_idx = int(len(total_seq_tensor)*0.8)

    shuffle_index_list = np.arange(0, len(labels))
    train_idx_list = shuffle_index_list[:sep_idx]
    test_idx_list = shuffle_index_list[sep_idx:]

    train_seq = total_seq_tensor[train_idx_list]
    train_label = labels[train_idx_list]
    test_seq = total_seq_tensor[test_idx_list]  # test_data
    test_label = labels[test_idx_list]  # test_label

    torch.save(
        test_label, "../Data/Dataset_usermodel/Testset_usermodel/test_label.pt")
    torch.save(
        test_seq, "../Data/Dataset_usermodel/Testset_usermodel/test_data.pt")

    data_shape, train_loader = Data_load(train_seq, train_label, BATCH_SIZE)

    # Training step
    device = torch.device(
        f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")

    DCNet = UserDecision(n_actions, embedding_dim, device)
    DCNet.to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(DCNet.parameters(), lr=learning_rate)

    best_loss = 1000000
    DCNet.train()
    for ep in range(epoch):
        loss = 0
        correct = 0
        for train, label in train_loader:
            optimizer.zero_grad()
            prediction = DCNet(train.to(device))
            label = label.to(device)
            bceloss = criterion(prediction, label.unsqueeze(1))
            bceloss.backward()
            optimizer.step()
            loss += bceloss.item()
            correct += (torch.round(prediction.squeeze(1))
                        == label).sum().item() / BATCH_SIZE

        if ep % 5 == 0:
            f = open(MODEL_STORAGE_PATH + args.save + "/report.txt", "a")
            f.write("epoch : {}, \t loss : {} \t accuracy : {:.7f}% \n".format(
                ep, loss, (correct*100)/len(train_loader)))
            f.close()

        if best_loss > loss:
            best_loss = loss
            save_model(DCNet, "user_model", MODEL_STORAGE_PATH + args.save)
