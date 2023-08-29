import torch.nn as nn
import torch
import torch.nn.functional as F
from operations.Pooling import Pooling


class RGFI(nn.Module):
    def __init__(self):
        super(RGFI, self).__init__()
        self.fc1=nn.Linear(2048,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)


    def forward(self,Fx,Fy):
        inputF1=torch.cat([Fx,Fy],dim=1)
        inputF2=torch.cat([Fy,Fx],dim=1)

        outputF1=F.relu(self.fc1(inputF1))
        outputF1=F.relu(self.fc2(outputF1))
        HrX=F.relu(self.fc3(outputF1))

        outputF2 = F.relu(self.fc1(inputF2))
        outputF2 = F.relu(self.fc2(outputF2))
        HrY = self.fc3(outputF2)

        return HrX,HrY


class TGFI(nn.Module):
    def __init__(self):
        super(TGFI, self).__init__()
        self.fc1=nn.Linear(2048,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)

    def forward(self, Fx, Fy):
        inputF1 = torch.cat([Fx, Fy], dim=1)
        inputF2 = torch.cat([Fy, Fx], dim=1)

        outputF1 = F.relu(self.fc1(inputF1))
        outputF1 = F.relu(self.fc2(outputF1))
        HtY = self.fc3(outputF1)

        outputF2 = F.relu(self.fc1(inputF2))
        outputF2 = F.relu(self.fc2(outputF2))
        HtX = self.fc3(outputF2)

        return HtY,HtX


class Rotation_Branch(nn.Module):
    def __init__(self):
        super(Rotation_Branch, self).__init__()
        self.fc1 = nn.Linear(256 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 4)

    def forward(self, inputH):
        output = F.relu(self.fc1(inputH))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        quaternion = self.fc5(output)

        return quaternion

class Translation_Branch(nn.Module):
    def __init__(self):
        super(Translation_Branch, self).__init__()
        self.fc1 = nn.Linear(256 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)


    def forward(self,inputH1,inputH2):

        output = F.relu(self.fc1(inputH1))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        cx = self.fc5(output)
        
        output = F.relu(self.fc1(inputH2))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        cy = self.fc5(output)

        return cx-cy

class RegressionBranch(nn.Module):
    def __init__(self):
        super(RegressionBranch, self).__init__()
        self.rgfi = RGFI()
        self.tgfi = TGFI()
        self.rotation_branch = Rotation_Branch()
        self.translation_branch = Translation_Branch()


    def forward(self,template_feature,source_feature):
        Hrx,Hry = self.rgfi(template_feature,source_feature)
        Htx,Hty = self.tgfi(template_feature,source_feature)
        Hr_feature_concatenate = torch.cat([Hrx,Htx,Hry,Hty], dim=1)
        Ht_feature_concatenate_x = torch.cat([Hrx,Htx,Hty], dim=1)
        Ht_feature_concatenate_y = torch.cat([Hry,Hty,Htx], dim=1)


        quaternion = self.rotation_branch(Hr_feature_concatenate)
        translation = self.translation_branch(Ht_feature_concatenate_x,Ht_feature_concatenate_y)
        pose_7d = torch.cat([quaternion,translation], dim=1)
        return pose_7d
