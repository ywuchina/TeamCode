import torch.nn as nn
import torch.nn.functional as F
import torch
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
def gather_neighbour(pc, neighbor_idx):
    """
    gather the coordinates or features of neighboring points
    pc: [B, C, N, 1]
    neighbor_idx: [B, N, K]
    """
    pc = pc.transpose(2, 1).squeeze(-1)
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  #b* n *k *d
    features = features.permute(0, 3, 1, 2) # b*c*n*k
    return features

class get_model(nn.Module):
    def __init__(self, out_c):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(2048, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(512, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(128, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(32, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv1d(128, out_c, 1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, xyz):
        
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        # x = self.conv2(x)
        # x = F.sigmoid(x)
        
        return x




class SiamPointNet2(nn.Module):
    def __init__(self):
        super(SiamPointNet2, self).__init__()
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
    model = SiamPointNet2().cuda()
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