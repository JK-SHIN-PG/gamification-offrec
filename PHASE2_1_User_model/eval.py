import torch
from model import *
from dataloader import *
import sys
import os
import argparse
__file__ = 'eval.py'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",type=str,default= 'example')
    parser.add_argument("--bth", type=int,default= 1024)
    parser.add_argument("--gpu", type=str, default = '0')
    args = parser.parse_args()

    seq = torch.load("../Data/Dataset_usermodel/Testset_usermodel/test_data.pt")
    labels = torch.load("../Data/Dataset_usermodel/Testset_usermodel/test_label.pt")

    GPU_NUM = args.gpu
    device = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
    BATCHSIZE = args.bth
    USER_MODEL_PATH = '../Saved_file/User_model/{}/'.format(args.save)

    f = open(USER_MODEL_PATH +  "/validation_report.txt", "a")
    f.write(str(vars(args)) + "\n")
    f.close()

    user_model = UserDecision(32940, 32, device)
    user_model.load_state_dict(torch.load(USER_MODEL_PATH + 'user_model.pth'))
    user_model.to(device)

    data_shape , test_loader = Data_load(seq, labels, BATCHSIZE)
    user_model.eval()
    criterion = nn.BCELoss().to(device)
    correct = 0
    for test, label in test_loader:
        with torch.no_grad():
            prediction = user_model(test.to(device))
            label = label.to(device)
            bceloss = criterion(prediction, label.unsqueeze(1))
            correct += (torch.round(prediction.squeeze(1)) == label).sum().item() / BATCHSIZE

    f = open(USER_MODEL_PATH +  "/validation_report.txt", "a")
    f.write("accuracy : {}%".format(round((correct*100)/len(test_loader),2)))
    f.close()

