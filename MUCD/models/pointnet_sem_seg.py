import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from models.pointnet2_sem_seg import gather_neighbour

class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        
        return x

class SiamPointNet(nn.Module):
    def __init__(self):
        super(SiamPointNet, self).__init__()
        self.net = get_model(64)
        self.mlp1 = nn.Conv1d(64, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop1 = nn.Dropout(0.5)
        self.mlp2 = nn.Conv1d(32, 1, kernel_size=1, bias=False)
        
    def forward(self, end_points0, end_points1, knearest_idx):
        end_points0 = end_points0.transpose(2, 1)
        end_points1 = end_points1.transpose(2, 1)
        out0= self.net(end_points0)
        out1 = self.net(end_points1)
        knearest_01, knearest_10 = knearest_idx
        
        out0 = out0.unsqueeze(-1)#B,C,N,1
        out1 = out1.unsqueeze(-1)
        fout0 = self.nearest_feature_difference(out0, out1, knearest_01)
        fout1 = self.nearest_feature_difference(out1, out0, knearest_10)
        fout0 = self.drop1(self.relu1(self.bn1(self.mlp1(fout0))))
        fout1 = self.drop1(self.relu1(self.bn1(self.mlp1(fout1))))
        fout0 = self.mlp2(fout0)
        fout1 = self.mlp2(fout1)
        fout0 = torch.sigmoid(fout0)
        fout1 = torch.sigmoid(fout1)
        return fout0, fout1
    
    @staticmethod
    def nearest_feature_difference(raw, query, nearest_idx):
        nearest_features = gather_neighbour(query, nearest_idx)
        fused_features = torch.mean(torch.abs(raw - nearest_features), -1)
        return fused_features

if __name__ == '__main__':
    import  torch
    loss_net = nn.L1Loss().cuda()
    model = SiamPointNet().cuda()
    xyz0 = torch.rand(2, 8192, 3).cuda()
    xyz1 = torch.rand(2, 8192, 3).cuda()
    xofy = torch.randint(0, 8192, (2, 8192, 3)).cuda()
    
    yofx = torch.randint(0, 8192, (2, 8192, 3)).cuda()
    nearest_idx = [xofy,yofx]
    fout1,fout2 = model(xyz0,xyz1,nearest_idx)
    lable1 = torch.rand(2, 1, 8192).cuda()
    lable2 = torch.rand(2, 1, 8192).cuda()
    loss = loss_net(fout1,lable1) + loss_net(fout2,lable2)
    loss.backward()