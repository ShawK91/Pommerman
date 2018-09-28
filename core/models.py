import torch
from torch import nn
import torch.nn.functional as F
import os
from core import utils

os.environ["CUDA_VISIBLE_DEVICES"]='3'


class Conv_model(nn.Module):
    def __init__(self, z_dim):
        super(Conv_model, self).__init__()
        self.hw = 5 #Intermediate computation to track the HW dimension

        ## Encoder
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 48, 3, stride=1)
        self.conv3 = nn.Conv2d(48, 64, 3, stride=1)
        self.fc_encoder = nn.Linear(64 * self.hw * self.hw, z_dim)

        #Policy Net
        self.policy_fc1 = nn.Linear(z_dim, 200)
        self.policy_lnorm1 = LayerNorm(200)
        self.policy_fc2 = nn.Linear(200, 6)

        #Value Net
        self.value_fc1 = nn.Linear(z_dim, 200)
        self.value_lnorm1 = LayerNorm(200)
        self.value_fc2 = nn.Linear(200, 1)


    def encode(self, x):
        h = F.elu(self.conv1(x)); #print(h.shape)
        h = F.elu(self.conv2(h)) ; #print(h.shape)
        h = F.elu(self.conv3(h)); #print(h.shape)
        h = h.view(-1, 64 * self.hw * self.hw); #print(h.shape)
        h = F.elu(self.fc_encoder(h))
        return h


    def policy_value_net(self, z):
        #Compute policy head
        p = F.elu(self.policy_fc1(z))
        p = self.policy_lnorm1(p)
        p = F.softmax(self.policy_fc2(p))

        #Compute value head
        v = F.elu(self.value_fc1(z))
        v = self.value_lnorm1(v)
        v = F.tanh(self.value_fc2(v))
        return p, v


    def forward(self, x):
        z = self.encode(x)
        p, v = self.policy_value_net(z)
        return p, v

    #API FOR MCTS
    def predict(self,x):
        x = torch.Tensor(x).permute([0,3,1,2]).cuda()
        p, v = self.forward(x)
        return utils.to_numpy(p.cpu()), utils.to_numpy(v.cpu())

    def loss_fn(self, p, p_target, v, v_target):
        p_loss = torch.sum(torch.nn.functional.cross_entropy(p, p_target))
        v_loss = torch.sum(torch.nn.functional.mse_loss(v.squeeze(), v_target))
        total_loss = p_loss + v_loss
        return total_loss, p_loss.item(), v_loss.item()

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Imitation:
    def __init__(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    def imitate(self, model, x, p_target, v_target):
        x = torch.Tensor(x).permute([0,3,1,2]).cuda(); p_target=torch.Tensor(p_target).cuda(); v_target=torch.Tensor(v_target).cuda()
        p, v = model.forward(x)

        #Loss functions
        p_loss = torch.sum(torch.nn.functional.mse_loss(p, p_target))
        v_loss = torch.sum(torch.nn.functional.mse_loss(v.squeeze(), v_target))
        total_loss = p_loss + v_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return  model