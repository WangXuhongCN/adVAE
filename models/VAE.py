import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_module import BasicModule

class VAE(BasicModule):
    def __init__(self,sample_dim:400,rep_dim:64):
        super().__init__()
        self.sample_dim = sample_dim
        self.rep_dim = rep_dim
        self.fc1=nn.Linear(sample_dim,int(0.5*sample_dim))
        self.bn1 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fc2=nn.Linear(int(0.5*sample_dim),int(0.5*sample_dim))
        self.bn2 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fcmu=nn.Linear(int(0.5*sample_dim),rep_dim)
        self.fclogvar=nn.Linear(int(0.5*sample_dim),rep_dim)

        self.fc3=nn.Linear(rep_dim,int(0.5*sample_dim))
        self.bn3 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fc4=nn.Linear(int(0.5*sample_dim),int(0.5*sample_dim))
        self.bn4 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fcxmu=nn.Linear(int(0.5*sample_dim),sample_dim)
        #self.fcxlogvar=nn.Linear(int(0.5*sample_dim),sample_dim)

    def encoder(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        #x = x.view(x.size(0), -1)
        mu = self.fcmu(x)
        logvar = self.fclogvar(x)
        return mu,logvar

    def decoder(self, x):
        x=self.fc3(x)
        #x=self.bn3(x)
        x = F.relu(x)
        x=self.fc4(x)
        #x=self.bn4(x)
        x = F.relu(x)
        mu_x=self.fcxmu(x)
        mu_x = torch.sigmoid(mu_x)
        return mu_x

    # def decoder(self, x):
    #     x=self.fc3(x)
    #     #x=self.bn3(x)
    #     x = F.relu(x)
    #     x=self.fc4(x)
    #     #x=self.bn4(x)
    #     x = F.relu(x)
    #     mu_x=self.fcxmu(x)
    #     mu_x = torch.sigmoid(mu_x)
    #     logvar_x=self.fcxlogvar(x)
    #     logvar_x = torch.sigmoid(logvar_x)
    #     return mu_x,  logvar_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        mu_x=self.decoder(z)
        #x_recon=self.reparameterize(mu_x, logvar_x)
        return mu_x,z,mu, logvar    