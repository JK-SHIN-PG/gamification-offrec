import torch
import torch.nn as nn
import torch.nn.functional as F

class UserDecision(nn.Module):
    def __init__(self, n_action, num_emb, device):
        super(UserDecision, self).__init__()
        self.embedding = nn.Embedding(n_action, num_emb, padding_idx= 3)
        self.fc1 = nn.Linear(num_emb*19, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.dout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        self.device = device
    
    def forward(self, x):
        out = self.embedding(x)
        out = out.flatten(start_dim = 1)
        out = self.relu(self.fc1(out))
        out = self.dout(out)
        out = self.relu(self.fc2(out))
        out = self.dout(out)
        out = self.relu(self.fc3(out))
        out = self.dout(out)
        out = self.relu(self.fc4(out))
        out = torch.sigmoid(out)
        return out