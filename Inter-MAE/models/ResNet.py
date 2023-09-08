import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, model, feat_dim = 2048):
        super(ResNet, self).__init__()

        self.resnet = model
        self.resnet.fc = nn.Identity()
        self.inv_head = nn.Sequential(
                            nn.Linear(feat_dim, 512, bias = False),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256, bias = False)
                            ) 
        
    def forward(self, x):
        # print(self.resnet)
        x = self.resnet(x)
        x = self.inv_head(x)
        return x
