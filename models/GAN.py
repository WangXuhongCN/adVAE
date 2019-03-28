import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_module import BasicModule

class NetD(BasicModule):
    def __init__(self,sample_dim):
        super().__init__()
        self.sample_dim = sample_dim
        self.fc1=nn.Linear(sample_dim,int(0.5*sample_dim))
        #self.bn1 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fc2=nn.Linear(int(0.5*sample_dim),int(0.5*sample_dim))
        #self.bn2 = nn.BatchNorm1d(int(0.25*sample_dim))
        self.fc3=nn.Linear(int(0.5*sample_dim),1)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        #x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

class NetG(BasicModule):
    def __init__(self,sample_dim:400,rep_dim:64):
        super().__init__()
        self.sample_dim = sample_dim
        self.rep_dim = rep_dim
        self.fc3=nn.Linear(rep_dim,int(0.5*sample_dim))
        #self.bn3 = nn.BatchNorm1d(int(0.25*sample_dim))
        self.fc4=nn.Linear(int(0.5*sample_dim),int(0.5*sample_dim))
        #self.bn4 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fcxmu=nn.Linear(int(0.5*sample_dim),sample_dim)
        #self.fcxlogvar=nn.Linear(int(0.5*sample_dim),sample_dim)

    def forward(self, x):
        x=self.fc3(x)
        #x=self.bn3(x)
        x = F.relu(x)
        x=self.fc4(x)
        #x=self.bn4(x)
        x = F.relu(x)
        x=self.fcxmu(x)
        x = torch.sigmoid(x)
        return x
