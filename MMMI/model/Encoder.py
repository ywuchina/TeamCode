import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations.Pooling import Pooling

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.fc1 = nn.Conv1d(1088, 512, 1)
        self.fc2 = nn.Conv1d(512, 256, 1)
        self.fc3 = nn.Conv1d(256, 128, 1)
        self.fc4 = nn.Conv1d(128, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


class Encoder(nn.Module):
    def __init__(self, feature_dim=3):
        super(Encoder, self).__init__()
        self.localdiscriminator = LocalDiscriminator()
        self.globaldiscriminator = GlobalDiscriminator()
        self.conv1 = nn.Conv1d(feature_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 1024)  
        self.fc2 = nn.Linear(1024, 1024)  

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,N,feature_dim]->[B,feature_dim,N]
        output = F.leaky_relu(self.bn1(self.conv1(x)))  # [B,64,N]
        output = F.leaky_relu(self.bn2(self.conv2(output)))  # [B,64,N]
        output_local = F.leaky_relu(self.bn3(self.conv3(output)))  # [B,64,N]
        output = F.leaky_relu(self.bn4(self.conv4(output_local)))  # [B,128,N]
        output = self.conv5(output)  # [B,1024,N]
        output =Pooling('max')(output)  # [B,1024]

        self.z_mean = self.fc1(output)  # mean[B,1024]
        self.z_var = self.fc2(output)  # variance[B,1024]
        z_sample = self.sampling(self.z_mean, self.z_var)  # [B,1024]
        KL_loss = -0.5 * torch.mean(1 + self.z_var - (self.z_mean ** 2) - torch.exp(self.z_var))

        z_shuffle = self.shuffling(z_sample)  # [B,1024]
        
        zz1 = torch.cat([z_sample, z_sample], dim=1)  # [B,2048]
        zz2 = torch.cat([z_sample, z_shuffle], dim=1)  # [B,2048]

        local_shuffle = self.shuffling(output_local)  # [B,64,N]
        z_samplemap = z_sample.unsqueeze(1).contiguous().repeat(1, x.shape[2], 1).permute(0, 2, 1)  # [B,1,1024]->[B,N,1024]->[B,1024,N]
        
        zl1 = torch.cat([z_samplemap, output_local], dim=1)  # [B,1024+64,N]
        zl2 = torch.cat([z_samplemap, local_shuffle], dim=1)  # [B,1024+64,N]

        zz1score = self.globaldiscriminator(zz1)
        zz2score = self.globaldiscriminator(zz2)
        global_loss = -torch.mean(torch.log(zz1score + 1e-6) + torch.log(1 - zz2score + 1e-6))

        zl1score = self.localdiscriminator(zl1)
        zl2score = self.localdiscriminator(zl2)
        local_loss = -torch.mean(torch.log(zl1score + 1e-6) + torch.log(1 - zl2score + 1e-6))
        encoder_loss = global_loss + local_loss
        
        return self.z_mean, encoder_loss

    
    def sampling(self, z_mean, z_var):
        u = torch.randn(z_mean.shape).to(z_mean)
        return z_mean + torch.exp(z_var / 2) * u

    def shuffling(self, x):
        index = np.arange(0, x.shape[0])
        np.random.shuffle(index)
        return x[index].clone()




