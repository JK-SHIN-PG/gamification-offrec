import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#CNN <- Encoder
class CNN(nn.Module):
    def __init__(self, in_dim, out_dim, padding, kernel_size,  stride,max_p_ker, device):
        super(CNN, self).__init__()
        # ImgIn shape = (#batch, #height, #width, #channels)
        # ex) ImgIn shape = (10, 32, 54, 5)
        #    Conv         -> (1, 28, 36, 3)
        #    Pool         -> (1, 14, 18, 3)
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_p_ker = max_p_ker
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(),
            nn.MaxPool2d(self.max_p_ker)
        )

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        out = torch.flatten(out, start_dim=1)
        return out

#GRU <- Decoder
class GRU_Network(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, n_layers, device, drop_prob= 0.2):
        super(GRU_Network, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(obs_dim, hidden_dim, n_layers, batch_first=True, dropout = drop_prob)
        self.device = device
        self.fc = nn.Linear(hidden_dim, n_actions)
        self.relu = nn.ReLU()
       
    def forward(self, x, h):
        out, h = self.gru(x,h)
        out = self.fc(self.relu(out))
        out = F.softmax(out, dim=-1)
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
