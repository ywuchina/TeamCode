import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group, group, geometric_point_descriptor, square_distance, index_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.transfer = nn.Sequential(nn.Conv1d(in_channels, out_channels,kernel_size=1, bias=False),
                                  nn.BatchNorm1d(out_channels), nn.ReLU())
        # self.net1 = nn.Sequential(nn.Conv1d(out_channels, out_channels,kernel_size=1, bias=False),
        #                           nn.BatchNorm1d(out_channels), nn.ReLU())
        self.net2 = nn.Sequential(nn.Conv1d(out_channels, out_channels,kernel_size=1, bias=False),
                                  nn.BatchNorm1d(out_channels))

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   # torch.Size([32, 512, 6, 32]),(B,N,D,S)
        x = x.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        batch_size, _, N = x.size()
        x = self.transfer(x)
        # x = F.relu(self.net2(self.net1(x)) + x)
        x = F.relu(self.net2(x)) + x
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # torch.Size(B*N,D')
        x = x.reshape(b, n, -1).permute(0, 2, 1) # torch.Size(B,D',N)
        return x

class Sa_Layer(nn.Module):
    def __init__(self, channels):
        super(Sa_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1) # torch.Size([32, 256, 256/4]),(B,N,D/4)
        x_k = self.k_conv(x) # torch.Size([32, 256/4, 256]),(B,D/4,N)
        x_v = self.v_conv(x) # torch.Size([32, 256, 256])
        energy = torch.bmm(x_q, x_k) # torch.Size([32, 256, 256])
        attention = self.softmax(energy) # torch.Size([32, 256, 256])
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) # torch.Size([32, 256, 256])
        x_r = torch.bmm(x_v, attention) # torch.Size([32, 256, 256])
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r # torch.Size([32, 256, 256])
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        # self.q_conv = nn.Conv1d(channels, channels // 4, 1)
        # self.k_conv = nn.Conv1d(channels, channels // 4, 1)
        self.q_conv = nn.Conv1d(channels, channels, 1, bias = False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias = False)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x, xyz):
        xx = x + xyz
        # x_q = self.q_conv(x).permute(0, 2, 1) # torch.Size(B,N,D/4)
        # x_k = self.k_conv(x) # torch.Size(B,D/4+,N)
        x_q = self.q_conv(xx) # torch.Size([B, N, D])
        x_k = self.k_conv(xx)  # torch.Size([B, D, N])
        x_v = self.v_conv(x) # torch.Size([B, N, D])

        # energy = torch.bmm(x_q, x_k) # torch.Size([B, D, D])
        energy = x_q - x_k   # torch.Size([B, D, D])
        attention = self.softmax(energy) # torch.Size([B, D, D])
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) # torch.Size([B, D, N])
        # out = torch.bmm(attention, x_v) # torch.Size([B, D, N])
        out = torch.mul(attention, x_v + xyz)  # torch.Size([B, D, N])
        out = self.act(self.after_norm(self.trans_conv(xx - out))) + xx
        return out

class Pct_cls1(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls1, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(14, 256, 1)

        self.sa1 = SA_Layer(channels = 256)
        self.sa2 = SA_Layer(channels = 256)
        self.sa3 = SA_Layer(channels = 256)
        self.sa4 = SA_Layer(channels = 256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x)  # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 64, 256(128+128)])
        feature_1 = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N

        new_xyz = new_xyz.permute(0, 2, 1) # torch.Size([32, 3, 256])
        new_xyz = self.pos_xyz(new_xyz) # torch.Size([32, 256, 256]),(B,D,N)

        x1 = self.sa1(feature_1, new_xyz)
        x2 = self.sa2(feature_1, new_xyz)
        x3 = self.sa3(feature_1, new_xyz)
        x4 = self.sa4(feature_1, new_xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 1024, 256]),(B,D,N)

        x = torch.cat([x, feature_1], dim=1) # torch.Size([32, 1280(1024+256), 256])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls2(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls2, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        # self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(3, 128, 1)

        self.sa1 = Sa_Layer(channels = 128)
        self.sa2 = Sa_Layer(channels = 128)
        self.sa3 = Sa_Layer(channels = 128)
        self.sa4 = Sa_Layer(channels = 128)
        self.sa5 = Sa_Layer(channels = 128)
        self.sa6 = Sa_Layer(channels = 128)
        self.sa7 = Sa_Layer(channels = 128)
        self.sa8 = Sa_Layer(channels = 128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1152, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        # feature = feature_0.permute(0, 2, 1)
        # new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # # new_xyz: torch.Size([32, 256, 3])
        # # new_feature: d,n,s,d torch.Size([32, 256, 64, 256(128+128)])
        # feature_1 = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N

        new_xyz = new_xyz.permute(0, 2, 1) # torch.Size([32, 3, 512])
        new_xyz = self.pos_xyz(new_xyz) # torch.Size([32, 128, 512]),(B,D,N)

        x1 = self.sa1(feature_0, new_xyz)
        x2 = self.sa2(x1, new_xyz)
        x3 = self.sa3(x2, new_xyz)
        x4 = self.sa4(x3, new_xyz)
        x5 = self.sa5(x4, new_xyz)
        x6 = self.sa6(x5, new_xyz)
        x7 = self.sa7(x6, new_xyz)
        x8 = self.sa8(x7, new_xyz)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)  # torch.Size([32, 1024, 512]),(B,D,N)

        x = torch.cat([x, feature_0], dim=1) # torch.Size([32, 1152(1024+128), 512])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls3(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls3, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        # self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(3, 128, 1)

        self.sa1 = Sa_Layer(channels = 128)
        self.sa2 = Sa_Layer(channels = 128)
        self.sa3 = Sa_Layer(channels = 128)
        self.sa4 = Sa_Layer(channels = 128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(640, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        # feature = feature_0.permute(0, 2, 1)
        # new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # # new_xyz: torch.Size([32, 256, 3])
        # # new_feature: d,n,s,d torch.Size([32, 256, 64, 256(128+128)])
        # feature_1 = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N

        new_xyz = new_xyz.permute(0, 2, 1) # torch.Size([32, 3, 512])
        new_xyz = self.pos_xyz(new_xyz) # torch.Size([32, 128, 512]),(B,D,N)

        x1 = self.sa1(feature_0, new_xyz)
        x2 = self.sa2(x1, new_xyz)
        x3 = self.sa3(x2, new_xyz)
        x4 = self.sa4(x3, new_xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 512, 512]),(B,D,N)

        x = torch.cat([x, feature_0], dim=1) # torch.Size([32, 640(512+128), 512])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls4(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls4, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_2 = Local_op(in_channels=512, out_channels=512)

        self.pos_xyz = nn.Conv1d(3, 512, 1)

        self.sa1 = Sa_Layer(channels = 512)
        self.sa2 = Sa_Layer(channels = 512)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024+512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 64, 256(128+128)])
        feature = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        new_xyz, new_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 128, 3])
        # new_feature: d,n,s,d torch.Size([32, 128, 64, 512(256+256)])
        feature_1 = self.gather_local_2(new_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N

        new_xyz = new_xyz.permute(0, 2, 1) # torch.Size([32, 3, 128])
        new_xyz = self.pos_xyz(new_xyz) # torch.Size([32, 512, 128]),(B,D,N)

        x1 = self.sa1(feature_1, new_xyz)
        x2 = self.sa2(x1, new_xyz)
        x = torch.cat((x1, x2), dim=1)  # torch.Size([32, 1024, 128]),(B,D,N)

        x = torch.cat([x, feature_1], dim=1) # torch.Size([32, 1280(1024+512), 128])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls5(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls5, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(14, 256, 1)

        self.sa1 = SA_Layer(channels = 256)
        self.sa2 = SA_Layer(channels = 256)
        self.sa3 = SA_Layer(channels = 256)
        self.sa4 = SA_Layer(channels = 256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024+256, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=128, radius=0.2, nsample=64, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        feature_1 = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 128]),B,D,N

        new_xyz = new_xyz.permute(0, 2, 1) # torch.Size([32, 3, 128])
        new_xyz = self.pos_xyz(new_xyz) # torch.Size([32, 256, 128]),(B,D,N)

        x1 = self.sa1(feature_1, new_xyz)
        x2 = self.sa2(feature_1, new_xyz)
        x3 = self.sa3(feature_1, new_xyz)
        x4 = self.sa4(feature_1, new_xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 1024, 128]),(B,D,N)

        x = torch.cat([x, feature_1], dim=1) # torch.Size([32, 1280(1024+256), 128])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls6(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls6, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.local_pos_xyz = nn.Conv1d(14, 256, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_sa1 = SA_Layer(channels = 256)
        self.local_sa2 = SA_Layer(channels = 256)
        self.local_sa3 = SA_Layer(channels = 256)
        self.local_sa4 = SA_Layer(channels = 256)
        self.global_sa1 = SA_Layer(channels = 128)
        self.global_sa2 = SA_Layer(channels = 128)
        self.global_sa3 = SA_Layer(channels = 128)
        self.global_sa4 = SA_Layer(channels = 128)
        # self.conv_fuse = nn.Sequential(nn.Conv1d(1024+256, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024+512, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_0 = self.gather_local_0(local_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        local_feature = local_feature_0.permute(0, 2, 1)
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=64, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_1 = self.gather_local_1(local_feature) # feature_1: torch.Size([32, 256, 128]),B,D,N

        local_xyz = local_xyz.permute(0, 2, 1) # torch.Size([32, 3, 128])
        local_xyz = self.local_pos_xyz(local_xyz) # torch.Size([32, 256, 128]),(B,D,N)
        x1 = self.local_sa1(local_feature_1, local_xyz)
        x2 = self.local_sa2(local_feature_1, local_xyz)
        x3 = self.local_sa3(local_feature_1, local_xyz)
        x4 = self.local_sa4(local_feature_1, local_xyz)
        local_x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 256*4, 128]),(B,D,N)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        x1 = self.global_sa1(global_feature, global_xyz)
        x2 = self.global_sa2(global_feature, global_xyz)
        x3 = self.global_sa3(global_feature, global_xyz)
        x4 = self.global_sa4(global_feature, global_xyz)
        global_x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 128*4, 1024]),(B,D,N)

        # x = torch.cat([local_x, local_feature_1], dim=1) # torch.Size([32, 1280(1024+256), 128])
        # x = self.conv_fuse(x)
        local_x = F.adaptive_max_pool1d(local_x, 1).view(batch_size, -1) # torch.Size([32, 256*4]),(B,D)
        global_x = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*4]),(B,D)

        x = torch.cat([global_x, local_x], dim=1)  # torch.Size([32, 1024+512])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls7(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls7, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        self.global_sa = SA_Layer(channels = 128)
        # self.conv_fuse = nn.Sequential(nn.Conv1d(1024+256, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_x = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        local_x1 = self.local_scale1(local_feature_1, local_xyz1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        local_x2 = self.local_scale2(local_feature_2, local_xyz2)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        local_x3 = self.local_scale3(local_feature_3, local_xyz3)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        global_x = self.global_sa(global_x, global_xyz)

        # x = self.conv_fuse(x)
        local_x1 = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128]),(B,D)
        local_x2 = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        local_x3 = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512]),(B,D)
        global_x = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128]),(B,D)

        x = torch.cat([global_x, local_x1, local_x2, local_x3], dim=1)  # torch.Size([32, 128+128+256+512])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls8(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls8, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.global_sa = SA_Layer(channels = 128)
        # self.conv_fuse = nn.Sequential(nn.Conv1d(1024+256, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_x = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        local_x1 = self.local_scale1(local_feature_1, local_xyz1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        local_x2 = self.local_scale2(local_feature_2, local_xyz2)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        global_x = self.global_sa(global_x, global_xyz)

        # x = self.conv_fuse(x)
        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128]),(B,D)
        local_x1_avg = F.adaptive_avg_pool1d(local_x1, 1).view(batch_size, -1)  # torch.Size([32, 128]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        local_x2_avg = F.adaptive_avg_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128]),(B,D)
        global_x_avg = F.adaptive_avg_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128]),(B,D)

        x = torch.cat([global_x_max, global_x_avg, local_x1_max, local_x1_avg, local_x2_max, local_x2_avg], dim=1)  # torch.Size([32, 128+128+256+512])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls9(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls9, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.local_pos_xyz = nn.Conv1d(14, 256, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_sa1 = SA_Layer(channels = 256)
        self.local_sa2 = SA_Layer(channels = 256)
        self.local_sa3 = SA_Layer(channels = 256)
        self.local_sa4 = SA_Layer(channels = 256)
        self.local_delta1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.local_delta2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.local_delta3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.local_delta4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.global_sa1 = SA_Layer(channels = 128)
        self.global_sa2 = SA_Layer(channels = 128)
        self.global_sa3 = SA_Layer(channels = 128)
        self.global_sa4 = SA_Layer(channels = 128)
        self.global_delta1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.global_delta2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.global_delta3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.global_delta4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.conv_local_fuse = nn.Sequential(nn.Conv1d(1024+256, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv_global_fuse = nn.Sequential(nn.Conv1d(512+128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024+512, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_0 = self.gather_local_0(local_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        local_feature = local_feature_0.permute(0, 2, 1)
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=64, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_1 = self.gather_local_1(local_feature) # feature_1: torch.Size([32, 256, 128]),B,D,N

        local_xyz = local_xyz.permute(0, 2, 1) # torch.Size([32, 3, 128])
        local_xyz = self.local_pos_xyz(local_xyz) # torch.Size([32, 256, 128]),(B,D,N)
        x1 = self.local_sa1(local_feature_1, local_xyz)
        local_x1 = x1 - local_feature_1 # Back-projection signal
        x1 = x1 + self.local_delta1*local_x1 # Feedback

        x2 = self.local_sa2(local_feature_1, local_xyz)
        local_x2 = x2 - local_feature_1
        x2 = x2 + self.local_delta2*local_x2
        x3 = self.local_sa3(local_feature_1, local_xyz)
        local_x3 = x3 - local_feature_1
        x3 = x3 + self.local_delta3*local_x3
        x4 = self.local_sa4(local_feature_1, local_xyz)
        local_x4 = x4 - local_feature_1
        x4 = x4 + self.local_delta4*local_x4
        local_x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 256*4, 128]),(B,D,N)
        local_x = self.conv_local_fuse(torch.cat([local_x, local_feature_1], dim=1))

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        x1 = self.global_sa1(global_feature, global_xyz)
        global_x1 = x1 - global_feature     # Back-projection signal
        x1 = x1 + self.global_delta1*global_x1     # Feedback
        x2 = self.global_sa2(global_feature, global_xyz)
        global_x2 = x2 - global_feature
        x2 = x2 + self.global_delta2*global_x2
        x3 = self.global_sa3(global_feature, global_xyz)
        global_x3 = x3 - global_feature
        x3 = x3 + self.global_delta1*global_x3
        x4 = self.global_sa4(global_feature, global_xyz)
        global_x4 = x4 - global_feature
        x4 = x4 + self.global_delta4*global_x4
        global_x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 128*4, 1024]),(B,D,N)
        global_x = self.conv_global_fuse(torch.cat([global_x, global_feature], dim=1))

        local_x = F.adaptive_max_pool1d(local_x, 1).view(batch_size, -1) # torch.Size([32, 256*4]),(B,D)
        global_x = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*4]),(B,D)
        x = torch.cat([global_x, local_x], dim=1)  # torch.Size([32, 1024+512])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls10(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls10, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1, bias=False)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1, bias=False)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1, bias=False)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1, bias=False)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        self.global_sa = SA_Layer(channels = 128)

        self.linear1 = nn.Linear((128+128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 256, 1024]),(B,D,N)
        global_x = self.global_sa(global_feature, global_xyz)
        global_x = torch.cat([global_x, global_feature], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*2]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*2]),(B,D)
        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        x_max = torch.cat([global_x_max, local_x1_max, local_x2_max, local_x3_max], dim=1)  # torch.Size([32, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls11(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls11, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_sa1 = SA_Layer(channels = 128)
        self.local_sa2 = SA_Layer(channels = 256)
        self.local_sa3 = SA_Layer(channels = 512)
        self.global_sa1 = SA_Layer(channels = 128)
        self.global_sa2 = SA_Layer(channels = 128)
        self.global_sa3 = SA_Layer(channels = 128)
        self.global_sa4 = SA_Layer(channels = 128)
        # self.conv_fuse = nn.Sequential(nn.Conv1d(1024+256, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(128*5+(128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        x1 = self.local_sa1(local_feature_1, local_xyz1)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        x2 = self.local_sa2(local_feature_2, local_xyz2)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        x3 = self.local_sa3(local_feature_3, local_xyz3)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        global_x1 = self.global_sa1(global_feature, global_xyz)
        global_x2 = self.global_sa2(global_x1, global_xyz)
        global_x3 = self.global_sa3(global_x2, global_xyz)
        global_x4 = self.global_sa4(global_x3, global_xyz)
        # global_x = x - global_feature  # Back-projection signal
        # global_x = x + self.global_delta * global_x  # Feedback
        global_x = torch.cat([global_x1, global_x2, global_x3, global_x4, global_feature], dim=1) # torch.Size([32, 128 * 5, 1024]),(B,D,N)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*2]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*2]),(B,D)
        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*5]),(B,D)
        x_max = torch.cat([global_x_max, local_x1_max, local_x2_max, local_x3_max], dim=1)  # torch.Size([32, 128*5+(128+256+512)*2])
        # local_x1_avg = F.adaptive_avg_pool1d(local_x1, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        # local_x2_avg = F.adaptive_avg_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        # local_x3_avg = F.adaptive_avg_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*2]),(B,D)
        # global_x_avg = F.adaptive_avg_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        # x_avg = torch.cat([global_x_avg, local_x1_avg, local_x2_avg, local_x3_avg],dim=1)  # torch.Size([32, (128+128+256+512)*2])
        # x = torch.cat([x_max,x_avg], dim = 1) # torch.Size([32, (128+128+256+512)*2*2])

        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls12(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls12, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz = nn.Conv1d(14, 512, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_sa1 = SA_Layer(channels = 512)
        self.local_sa2 = SA_Layer(channels = 512)
        self.global_sa1 = SA_Layer(channels = 128)
        self.global_sa2 = SA_Layer(channels = 128)
        self.global_sa3 = SA_Layer(channels = 128)
        self.global_sa4 = SA_Layer(channels = 128)

        self.linear_fuse = nn.Sequential(nn.Linear(512*3 + 128 * 5, 1024, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz(local_xyz.permute(0, 2, 1))
        local_x1 = self.local_sa1(local_feature_3, local_xyz3)
        local_x2 = self.local_sa1(local_feature_3, local_xyz3)
        local_x = torch.cat([local_x1, local_x2, local_feature_3], dim = 1) # torch.Size([32, 512 * 3, 1024]),(B,D,N)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        global_x1 = self.global_sa1(global_feature, global_xyz)
        global_x2 = self.global_sa2(global_feature, global_xyz)
        global_x3 = self.global_sa3(global_feature, global_xyz)
        global_x4 = self.global_sa4(global_feature, global_xyz)
        global_x = torch.cat([global_x1, global_x2, global_x3, global_x4, global_feature], dim=1) # torch.Size([32, 128 * 5, 1024]),(B,D,N)

        local_x_max = F.adaptive_max_pool1d(local_x, 1).view(batch_size, -1)  # torch.Size([32, 512*3]),(B,D)
        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*5]),(B,D)

        x_max = torch.cat([global_x_max, local_x_max], dim=1) # torch.Size([32, 512*3 + 128*5]),(B,D)
        x_max = self.linear_fuse(x_max)
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls13(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls13, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.local_sa11 = SA_Layer(channels = 128)
        self.local_sa21 = SA_Layer(channels = 256)
        self.local_sa31 = SA_Layer(channels = 512)
        self.global_sa11 = SA_Layer(channels = 128)
        self.local_sa12 = SA_Layer(channels = 128)
        self.local_sa22 = SA_Layer(channels = 256)
        self.local_sa32 = SA_Layer(channels = 512)
        self.global_sa12 = SA_Layer(channels = 128)

        self.linear_fuse = nn.Sequential(nn.Linear(128*3+(128+256+512)*3, 1024, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, 64, 128
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        local_x11 = self.local_sa11(local_feature_1, local_xyz1)
        local_x12 = self.local_sa12(local_x11, local_xyz1)
        local_x1 = torch.cat([local_x11, local_x12, local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        local_x21 = self.local_sa21(local_feature_2, local_xyz2)
        local_x22 = self.local_sa22(local_x21, local_xyz2)
        local_x2 = torch.cat([local_x21, local_x22, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14])
        # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        local_x31 = self.local_sa31(local_feature_3, local_xyz3)
        local_x32 = self.local_sa32(local_x31, local_xyz3)
        local_x3 = torch.cat([local_x31, local_x32, local_feature_3], dim = 1)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        global_x1 = self.global_sa11(global_feature, global_xyz)
        global_x2 = self.global_sa12(global_x1, global_xyz)
        # global_x = x - global_feature  # Back-projection signal
        # global_x = x + self.global_delta * global_x  # Feedback
        global_x = torch.cat([global_x1, global_x2, global_feature], dim=1) # torch.Size([32, 128 * 3, 1024]),(B,D,N)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*3]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*3]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*3]),(B,D)
        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*3]),(B,D)
        x_max = torch.cat([global_x_max, local_x1_max, local_x2_max, local_x3_max], dim=1)  # torch.Size([32, 128*3+(128+256+512)*3])
        # local_x1_avg = F.adaptive_avg_pool1d(local_x1, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        # local_x2_avg = F.adaptive_avg_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        # local_x3_avg = F.adaptive_avg_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*2]),(B,D)
        # global_x_avg = F.adaptive_avg_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        # x_avg = torch.cat([global_x_avg, local_x1_avg, local_x2_avg, local_x3_avg],dim=1)  # torch.Size([32, (128+128+256+512)*2])
        # x = torch.cat([x_max,x_avg], dim = 1) # torch.Size([32, (128+128+256+512)*2*2])
        x_max = self.linear_fuse(x_max)
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls14(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls14, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)

        self.pos_xyz = nn.Conv1d(14, 256, 1, bias=False)

        self.sa1 = SA_Layer(channels = 256)
        self.sa2 = SA_Layer(channels = 256)
        self.sa3 = SA_Layer(channels = 256)
        self.sa4 = SA_Layer(channels = 256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x)  # B, 3, N => # B, 14, N
        xyz = x # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x_feature = F.relu(self.bn3(self.conv3(x)))  # B, 256, N

        new_xyz = self.pos_xyz(xyz) # torch.Size([32, 256, 1024]),(B,D,N)
        x1 = self.sa1(x_feature, new_xyz)
        x2 = self.sa2(x1, new_xyz)
        x3 = self.sa3(x2, new_xyz)
        x4 = self.sa4(x3, new_xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 1024, 1024]),(B,D,N)
        x = torch.cat([x, x_feature], dim=1) # torch.Size([32, 1280(1024+256), 1024])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls15(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls15, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1, bias=False)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1, bias=False)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1, bias=False)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)

        self.linear1 = nn.Linear((128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        # global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 256, 1024]),(B,D,N)
        # x = self.global_sa(global_feature, global_xyz)
        # global_x = x - global_feature  # Back-projection signal
        # global_x = global_feature + self.global_delta * global_x  # Feedback
        # global_x = torch.cat([global_x, global_feature], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*5]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*5]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*5]),(B,D)
        # global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1)  # torch.Size([32, (128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls16(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls16, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_xyz_1 = Local_op(in_channels=6, out_channels=128)
        self.gather_points_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_xyz_2 = Local_op(in_channels=6, out_channels=256)
        self.gather_points_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_xyz_3 = Local_op(in_channels=6, out_channels=512)
        self.gather_points_3 = Local_op(in_channels=512, out_channels=512)

        self.local_scale1 = SA_Layer(channels = 128*2)
        self.local_scale2 = SA_Layer(channels = 256*2)
        self.local_scale3 = SA_Layer(channels = 512*2)

        self.linear1 = nn.Linear((128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        x =x.permute(0, 2, 1) # B, N, 64
        sample_xyz1, local_xyz, local_feature = sample_and_group(npoint=512, nsample=16, xyz=xyz, points=x)
        #sample_xyz1:[32, 512, 14] local_xyz:[32, 512, 32, 28(14+14)] local_feature: d,n,s,d [32, 512, 32, 128(64+64)]
        # print(sample_xyz1.shape,local_xyz.shape,local_feature.shape)
        local_xyz1 = self.gather_xyz_1(local_xyz)  # torch.Size([32, 128, 512]),B,D,N
        local_feature1 = self.gather_points_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_x1 = self.local_scale1(torch.cat([local_feature1, local_xyz1],dim = 1)) # torch.Size([32, 128*2, 512]),B,D,N

        local_feature = local_feature1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        sample_xyz2, local_xyz, local_feature = sample_and_group(npoint=256, nsample=16, xyz=sample_xyz1, points=local_feature)
        local_xyz2 = self.gather_xyz_2(local_xyz)  # torch.Size([32, 256, 256]),B,D,N
        local_feature2 = self.gather_points_2(local_feature)  # torch.Size([32, 256, 256]),B,D,N
        local_x2 = self.local_scale2(torch.cat([local_feature2, local_xyz2],dim = 1)) # torch.Size([32, 256*2, 256]),B,D,N

        local_feature = local_feature2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        sample_xyz3, local_xyz, local_feature = sample_and_group(npoint=128, nsample=16, xyz=sample_xyz2, points=local_feature)
        local_xyz3 = self.gather_xyz_3(local_xyz)  # torch.Size([32, 512, 128]),B,D,N
        local_feature3 = self.gather_points_3(local_feature)  # torch.Size([32, 512, 128]),B,D,N
        local_x3 = self.local_scale3(torch.cat([local_feature3, local_xyz3],dim = 1)) # torch.Size([32, 512*2, 128]),B,D,N

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512]),(B,D)
        # global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1)  # torch.Size([32, (128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls17(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls17, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.gather_xyz_1 = Local_op(in_channels=6, out_channels=128)
        self.gather_points_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_xyz_2 = Local_op(in_channels=6, out_channels=128)
        self.gather_points_2 = Local_op(in_channels=128, out_channels=128)
        self.gather_xyz_3 = Local_op(in_channels=6, out_channels=256)
        self.gather_points_3 = Local_op(in_channels=256, out_channels=256)
        self.gather_xyz_4 = Local_op(in_channels=6, out_channels=512)
        self.gather_points_4 = Local_op(in_channels=512, out_channels=512)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 128)
        self.local_scale3 = SA_Layer(channels = 256)
        self.local_scale4 = SA_Layer(channels = 512)

        self.linear1 = nn.Linear(1024, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N
        x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N
        x4 = F.relu(self.bn4(self.conv4(x3)))  # B, 256, N
        xyz1, points1 = group(nsample=16, xyz=xyz, point=x1)
        #xyz1,points1: torch.Size([32, 1024, 8, 6]) torch.Size([32, 1024, 8, 128])
        local_xyz1 = self.gather_xyz_1(xyz1)  # torch.Size([32, 128, 1024]),B,D,N
        local_feature1 = self.gather_points_1(points1) # torch.Size([32, 128, 512]),B,D,N
        local_x1 = self.local_scale1(local_feature1, local_xyz1) # torch.Size([32, 128, 1024]),B,D,N

        xyz2, points2 = group(nsample=16, xyz=xyz, point=x2)
        local_xyz2 = self.gather_xyz_2(xyz2)  # torch.Size([32, 128, 1024]),B,D,N
        local_feature2 = self.gather_points_2(points2) # torch.Size([32, 128, 1024]),B,D,N
        local_x2 = self.local_scale2(local_feature2, local_xyz2) # torch.Size([32, 128, 1024]),B,D,N

        xyz3, points3 = group(nsample=16, xyz=xyz, point=x3)
        local_xyz3 = self.gather_xyz_3(xyz3)  # torch.Size([32, 256, 1024]),B,D,N
        local_feature3 = self.gather_points_3(points3) # torch.Size([32, 256, 1024]),B,D,N
        local_x3 = self.local_scale3(local_feature3, local_xyz3) # torch.Size([32, 256, 1024]),B,D,N

        xyz4, points4 = group(nsample=16, xyz=xyz, point=x4)
        local_xyz4 = self.gather_xyz_4(xyz4)  # torch.Size([32, 512, 1024]),B,D,N
        local_feature4 = self.gather_points_4(points4) # torch.Size([32, 512, 1024]),B,D,N
        local_x4 = self.local_scale4(local_feature4, local_xyz4) # torch.Size([32, 512, 1024]),B,D,N

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 128]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        local_x4_max = F.adaptive_max_pool1d(local_x4, 1).view(batch_size, -1)  # torch.Size([32, 512]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max, local_x4_max], dim=1)  # torch.Size([32, 1024])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls18(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls18, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        # self.gather_points_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_points_2 = Local_op(in_channels=128, out_channels=128)
        self.gather_points_3 = Local_op(in_channels=256, out_channels=256)
        self.gather_points_4 = Local_op(in_channels=512, out_channels=512)

        # self.xyzs_1 = nn.Conv1d(3, 128, 1, bias=False)
        self.xyzs_2 = nn.Conv1d(3, 128, 1, bias=False)
        self.xyzs_3 = nn.Conv1d(3, 256, 1, bias=False)
        self.xyzs_4 = nn.Conv1d(3, 512, 1, bias=False)

        self.gather_scale1 = SA_Layer(channels = 128)
        self.gather_scale2 = SA_Layer(channels = 128)
        self.gather_scale3 = SA_Layer(channels = 256)
        self.gather_scale4 = SA_Layer(channels = 512)

        self.linear1 = nn.Linear((128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N
        x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N
        x4 = F.relu(self.bn4(self.conv4(x3)))  # B, 256, N
        # points1 = group(nsample=16, xyz=xyz, point=x1)
        #xyz1,points1: torch.Size([32, 1024, 16, 6]) torch.Size([32, 1024, 16, 128])
        # xyz1 = self.xyzs_1(xyz)  # torch.Size([32, 128, 1024]),B,D,N
        # gather_feature1 = self.gather_points_1(points1) # torch.Size([32, 128, 512]),B,D,N
        # att_x1 = self.gather_scale1(gather_feature1, xyz1) # torch.Size([32, 128, 1024]),B,D,N
        # x1 = torch.cat([att_x1, gather_feature1], dim = 1)

        points2 = group(nsample=32, xyz=xyz, point=x2)
        xyz2 = self.xyzs_2(xyz)  # torch.Size([32, 128, 1024]),B,D,N
        gather_feature2 = self.gather_points_2(points2) # torch.Size([32, 128, 1024]),B,D,N
        att_x2 = self.gather_scale2(gather_feature2, xyz2) # torch.Size([32, 128, 1024]),B,D,N
        x2 = torch.cat([att_x2, gather_feature2], dim = 1)

        points3 = group(nsample=32, xyz=xyz, point=x3)
        xyz3 = self.xyzs_3(xyz)  # torch.Size([32, 256, 1024]),B,D,N
        gather_feature3 = self.gather_points_3(points3) # torch.Size([32, 256, 1024]),B,D,N
        att_x3 = self.gather_scale3(gather_feature3, xyz3) # torch.Size([32, 256, 1024]),B,D,N
        x3 = torch.cat([att_x3, gather_feature3], dim = 1)

        points4 = group(nsample=32, xyz=xyz, point=x4)
        xyz4 = self.xyzs_4(xyz)  # torch.Size([32, 512, 1024]),B,D,N
        gather_feature4 = self.gather_points_4(points4) # torch.Size([32, 512, 1024]),B,D,N
        att_x4 = self.gather_scale4(gather_feature4, xyz4) # torch.Size([32, 512, 1024]),B,D,N
        x4 = torch.cat([att_x4, gather_feature4], dim = 1)

        # x1_max = F.adaptive_max_pool1d(x1, 1).view(batch_size, -1) # torch.Size([32, 128*2]),(B,D)
        x2_max = F.adaptive_max_pool1d(x2, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        x3_max = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1)  # torch.Size([32, 256*2]),(B,D)
        x4_max = F.adaptive_max_pool1d(x4, 1).view(batch_size, -1)  # torch.Size([32, 512*2]),(B,D)
        x_max = torch.cat([x2_max, x3_max, x4_max], dim=1)  # torch.Size([32, 1024*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls19(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls19, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1, bias=False)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1, bias=False)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1, bias=False)
        # self.global_pos_xyz1 = nn.Conv1d(14, 64, 1, bias=False)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        # self.global_sa = SA_Layer(channels=64)

        self.linear1 = nn.Linear((128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_fea1 = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz, _, _, local_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, _, _, local_feature = sample_and_group(npoint=256, nsample=32, xyz=local_xyz, point=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, _, _, local_feature = sample_and_group(npoint=128, nsample=32, xyz=local_xyz, point=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)
        #
        # global_xyz1 = self.global_pos_xyz1(xyz.permute(0, 2, 1))  # torch.Size([32, 64, 1024]),(B,D,N)
        # global_x1 = self.global_sa(global_fea1, global_xyz1)
        # global_x = torch.cat([global_x1, global_fea1], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*5]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*5]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*5]),(B,D)
        # global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1)
        # torch.Size([32, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls20(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls20, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)
        self.gather_global = Local_op(in_channels=128, out_channels=128)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1, bias=False)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1, bias=False)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1, bias=False)
        self.global_pos_xyz = nn.Conv1d(14, 128, 1, bias=False)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        self.global_sa = SA_Layer(channels=128)

        self.linear1 = nn.Linear((128+128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_fea = x
        x = global_fea.permute(0, 2, 1)  # B, N, 128

        local_xyz, _, local_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 256(128+128)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = self.local_pos_xyz1(local_xyz.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz, _, local_feature = sample_and_group(npoint=256, nsample=32, xyz=local_xyz, point=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = self.local_pos_xyz2(local_xyz.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz, _, local_feature = sample_and_group(npoint=128, nsample=32, xyz=local_xyz, point=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = self.local_pos_xyz3(local_xyz.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        global_fea = group(nsample=32, xyz=xyz.permute(0, 2, 1), point=global_fea)
        global_fea = self.gather_global(global_fea) # global_fea: torch.Size([32, 128, 1024]),B,D,N
        global_xyz = self.global_pos_xyz(xyz.permute(0, 2, 1))  # torch.Size([32, 128, 1024]),(B,D,N)
        global_x = self.global_sa(global_fea, global_xyz)
        global_x = torch.cat([global_x, global_fea], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*5]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*5]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*5]),(B,D)
        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        x_max = torch.cat([global_x_max, local_x1_max, local_x2_max, local_x3_max], dim=1)
        # torch.Size([32, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls21(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls21, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)
        # self.gather_global = Local_op(in_channels=128, out_channels=128)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1, bias=False)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1, bias=False)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1, bias=False)
        # self.global_pos_xyz = nn.Conv1d(3, 128, 1, bias=False)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        # self.global_sa = SA_Layer(channels=128)

        self.linear1 = nn.Linear((128+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        # x = F.relu(self.bn3(self.conv3(x)))  # B, 128, N

        # global_fea = group(nsample=32, xyz=xyz.permute(0, 2, 1), point=x)
        # global_fea = self.gather_global(global_fea) # global_fea: torch.Size([32, 128, 1024]),B,D,N
        # # global_xyz = geometric_point_descriptor(xyz.permute(0, 2, 1))  # B, 3, N => # B, 14, N
        # global_xyz = self.global_pos_xyz(xyz.permute(0, 2, 1))  # torch.Size([32, 128, 1024]),(B,D,N)
        # global_x = self.global_sa(global_fea, global_xyz)
        # global_x = torch.cat([global_x, global_fea], dim=1) # torch.Size([32, 256, 1024]),(B,D,N)

        sample_xyz, sample_fea, _, local_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=x.permute(0, 2, 1))
        #sample_xyz: torch.Size([32, 512, 3]) sample_fea: torch.Size([32, 512, 64]) local_feature:torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1 = geometric_point_descriptor(sample_xyz.permute(0, 2, 1))  # B, 3, N => # B, 14, N
        local_xyz1 = self.local_pos_xyz1(local_xyz1)
        x1 = self.local_scale1(local_feature_1, local_xyz1)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1) # torch.Size([32, 512, 1024]),(B,D,N)

        sample_xyz, sample_fea, _, local_feature = sample_and_group(npoint=256, nsample=32, xyz=sample_xyz, point=local_feature_1.permute(0, 2, 1))
        local_feature_2 = self.gather_local_2(local_feature) # torch.Size([32, 256, 256]),B,D,N
        local_xyz2 = geometric_point_descriptor(sample_xyz.permute(0, 2, 1))  # B, 3, N => # B, 14, N
        local_xyz2 = self.local_pos_xyz2(local_xyz2)
        x2 = self.local_scale2(local_feature_2, local_xyz2)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1) # torch.Size([32, 1024, 1024]),(B,D,N)

        sample_xyz, sample_fea, _, local_feature = sample_and_group(npoint=128, nsample=32, xyz=sample_xyz, point=local_feature_2.permute(0, 2, 1))
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 128]),B,D,N
        local_xyz3 = geometric_point_descriptor(sample_xyz.permute(0, 2, 1))  # B, 3, N => # B, 14, N
        local_xyz3 = self.local_pos_xyz3(local_xyz3)
        x3 = self.local_scale3(local_feature_3, local_xyz3)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 128*5]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256*5]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512*5]),(B,D)
        # global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1)
        # torch.Size([32, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls22(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls22, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        self.geo_xyz1 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.local_scale1 = SA_Layer(channels = 256)
        self.geo_xyz2 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        self.geo_xyz3 = nn.Sequential(nn.Conv1d(10, 256, 1, bias=False), nn.BatchNorm1d(256))
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)

        self.linear1 = nn.Linear((256+256+512)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N
        x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=8, xyz=xyz, point=x1.permute(0, 2, 1))
        #sample_xyz: [32, 512, 3] grouped_xyz: [32, 512, 32, 3] sample_points: [32, 512, 128] grouped_points: [32, 512, 32, 128]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 8, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo1.size()  # torch.Size([32, 512, 32, 10])
        geo1 = geo1.permute(0, 1, 3, 2)  # torch.Size([32, 512, 10, 32]),(B,N,D,S)
        geo1 = geo1.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea1 = F.relu(self.geo_xyz1(geo1))  # torch.Size(B*N,D',S),D':128
        geo_fea1 = geo_fea1.reshape(b, n, s, -1)  # 32,512,32,128
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 8, 1), grouped_points), dim=-1) # 32,512,32,128
        local_fea1 = torch.cat((geo_fea1, sem_fea1), dim = -1)  # 32,512,32,256
        local_fea1 = self.gather_local_1(local_fea1) # 32,256,512
        local_att1 = self.local_scale1(local_fea1) # 32,256,512
        local_x1 = torch.cat([local_att1, local_fea1], dim = 1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=16, xyz=xyz, point=x2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo2.size()  # torch.Size([32, 512, 16, 10])
        geo2 = geo2.permute(0, 1, 3, 2)  # torch.Size([32, 512, 10, 16]),(B,N,D,S)
        geo2 = geo2.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea2 = F.relu(self.geo_xyz2(geo2))  # torch.Size(B*N,D',S),D':128
        geo_fea2 = geo_fea2.reshape(b, n, s, -1)  # 32,512,16,128
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 32,512,16,128
        local_fea2 = torch.cat((geo_fea2, sem_fea2), dim = -1)
        local_fea2 = self.gather_local_2(local_fea2)
        local_att2 = self.local_scale2(local_fea2)
        local_x2 = torch.cat([local_att2, local_fea2], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=32, xyz=xyz, point=x3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo3.size()  # torch.Size([32, 512, 8, 10])
        geo3 = geo3.permute(0, 1, 3, 2)  # torch.Size([32, 512, 10, 8]),(B,N,D,S)
        geo3 = geo3.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea3 = F.relu(self.geo_xyz3(geo3))  # torch.Size(B*N,D',S),D':256
        geo_fea3 = geo_fea3.reshape(b, n, s, -1)  # 32,512,8,256
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 32,512,8,256
        local_fea3 = torch.cat((geo_fea3, sem_fea3), dim = -1)
        local_fea3 = self.gather_local_3(local_fea3)
        local_att3 = self.local_scale3(local_fea3)
        local_x3 = torch.cat([local_att3, local_fea3], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 256]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1)
        # torch.Size([32, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls23(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls23, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        self.geo_xyz1 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_geo_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)

        self.geo_xyz2 = nn.Sequential(nn.Conv1d(10, 256, 1, bias=False), nn.BatchNorm1d(256))
        self.gather_geo_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)

        self.geo_xyz3 = nn.Sequential(nn.Conv1d(10, 512, 1, bias=False), nn.BatchNorm1d(512))
        self.gather_geo_3 = Local_op(in_channels=512, out_channels=512)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)

        self.linear1 = nn.Linear(256+512+1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N
        # x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N
        global_fea = x2
        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=global_fea.permute(0, 2, 1))
        #sample_xyz: [32, 512, 3] grouped_xyz: [32, 512, 32, 3] sample_points: [32, 512, 64] grouped_points: [32, 512, 8, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo1.size()  # torch.Size([32, 512, 32, 10])
        geo1 = geo1.permute(0, 1, 3, 2)  # torch.Size([32, 512, 10, 32]),(B,N,D,S)
        geo1 = geo1.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea1 = F.relu(self.geo_xyz1(geo1))  # torch.Size(B*N,D',S),D':128
        geo_fea1 = geo_fea1.reshape(b, n, s, -1)  # 32,512,8,128
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 32,512,8,128
        local_geo1 = self.gather_geo_1(geo_fea1)  # 32,128,512
        local_fea1 = self.gather_fea_1(sem_fea1) # 32,128,512
        local_att1 = self.local_scale1(local_fea1,local_geo1) # 32,128,512
        local_x1 = torch.cat([local_att1, local_fea1], dim=1) # 32,128*2,512

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=32, xyz=sample_xyz, point=local_att1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo2.size()  # torch.Size([32, 512, 16, 10])
        geo2 = geo2.permute(0, 1, 3, 2)  # torch.Size([32, 512, 10, 16]),(B,N,D,S)
        geo2 = geo2.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea2 = F.relu(self.geo_xyz2(geo2))  # torch.Size(B*N,D',S),D':128
        geo_fea2 = geo_fea2.reshape(b, n, s, -1)  # 32,256,16,256
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 32,256,16,256
        local_geo2 = self.gather_geo_2(geo_fea2)  # 32,256,256
        local_fea2 = self.gather_fea_2(sem_fea2) # 32,256,256
        local_att2 = self.local_scale2(local_fea2,local_geo2) # 32,256,256
        local_x2 = torch.cat([local_att2, local_fea2], dim=1) # 32,256*2,256

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=32, xyz=sample_xyz, point=local_att2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo3.size()  # torch.Size([32, 512, 8, 10])
        geo3 = geo3.permute(0, 1, 3, 2)  # torch.Size([32, 512, 10, 8]),(B,N,D,S)
        geo3 = geo3.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea3 = F.relu(self.geo_xyz3(geo3))  # torch.Size(B*N,D',S),D':256
        geo_fea3 = geo_fea3.reshape(b, n, s, -1)  # 32,512,8,256
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 32,512,8,256
        local_geo3 = self.gather_geo_3(geo_fea3)  # 32,512,128
        local_fea3 = self.gather_fea_3(sem_fea3) # 32,512,128
        local_att3 = self.local_scale3(local_fea3,local_geo3) # 32,1024,128
        local_x3 = torch.cat([local_att3, local_fea3], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1) # torch.Size([32, 256]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)  # torch.Size([32, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)  # torch.Size([32, 512]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1)
        # torch.Size([32, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls24(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls24, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        self.geo_xyz1 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_geo_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)
        # self.affine_alpha1 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        # self.affine_beta1 = nn.Parameter(torch.zeros([1, 1, 1, 64]))

        self.geo_xyz2 = nn.Sequential(nn.Conv1d(10, 256, 1, bias=False), nn.BatchNorm1d(256))
        self.gather_geo_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        # self.affine_alpha2 = nn.Parameter(torch.ones([1, 1, 1, 128]))
        # self.affine_beta2 = nn.Parameter(torch.zeros([1, 1, 1, 128]))

        self.geo_xyz3 = nn.Sequential(nn.Conv1d(10, 512, 1, bias=False), nn.BatchNorm1d(512))
        self.gather_geo_3 = Local_op(in_channels=512, out_channels=512)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)
        # self.affine_alpha3 = nn.Parameter(torch.ones([1, 1, 1, 256]))
        # self.affine_beta3 = nn.Parameter(torch.zeros([1, 1, 1, 256]))

        self.geo_xyz4 = nn.Sequential(nn.Conv1d(10, 1024, 1, bias=False), nn.BatchNorm1d(1024))
        self.gather_geo_4 = Local_op(in_channels=1024, out_channels=1024)
        self.gather_fea_4 = Local_op(in_channels=1024, out_channels=1024)
        self.local_scale4 = SA_Layer(channels = 1024)
        # self.affine_alpha4 = nn.Parameter(torch.ones([1, 1, 1, 512]))
        # self.affine_beta4 = nn.Parameter(torch.zeros([1, 1, 1, 512]))

        self.linear1 = nn.Linear(128+ 256 + 512 + 1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N
        # x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N
        global_fea = x2

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=global_fea.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo1.size()  # torch.Size([24, 512, 24, 10])
        geo1 = geo1.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 24]),(B,N,D,S)
        geo1 = geo1.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea1 = F.relu(self.geo_xyz1(geo1))  # torch.Size(B*N,D',S),D':128
        geo_fea1 = geo_fea1.reshape(b, n, s, -1)  # 24,512,8,128
        # std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo_fea1)  # 24,128,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_att1 = self.local_scale1(local_sem1,local_geo1) # 24,128,512

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=24, xyz=sample_xyz, point=local_sem1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 24, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo2.size()  # torch.Size([24, 512, 16, 10])
        geo2 = geo2.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 16]),(B,N,D,S)
        geo2 = geo2.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea2 = F.relu(self.geo_xyz2(geo2))  # torch.Size(B*N,D',S),D':128
        geo_fea2 = geo_fea2.reshape(b, n, s, -1)  # 24,256,16,256
        # std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 24, 1), grouped_points), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo_fea2)  # 24,256,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_att2 = self.local_scale2(local_sem2,local_geo2) # 24,256,256

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=24, xyz=sample_xyz, point=local_sem2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 24, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo3.size()  # torch.Size([24, 512, 8, 10])
        geo3 = geo3.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 8]),(B,N,D,S)
        geo3 = geo3.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea3 = F.relu(self.geo_xyz3(geo3))  # torch.Size(B*N,D',S),D':256
        geo_fea3 = geo_fea3.reshape(b, n, s, -1)  # 24,1024,8,256
        # std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 24, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo_fea3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_att3 = self.local_scale3(local_sem3,local_geo3) # 24,512,128

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=64, nsample=16, xyz=sample_xyz, point=local_sem3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 64, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz.view(B, 64, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        b, n, s, d = geo4.size()  # torch.Size([24, 512, 8, 10])
        geo4 = geo4.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 8]),(B,N,D,S)
        geo4 = geo4.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        geo_fea4 = F.relu(self.geo_xyz4(geo4))  # torch.Size(B*N,D',S),D':256
        geo_fea4 = geo_fea4.reshape(b, n, s, -1)  # 24,1024,8,256
        # std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea4 = torch.cat((sample_points.view(B, 64, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo_fea4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_att4 = self.local_scale4(local_sem4,local_geo4) # 24,1024,64

        local_x1_max = F.adaptive_max_pool1d(local_att1, 1).view(batch_size, -1) # torch.Size([24, 256]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_att2, 1).view(batch_size, -1)  # torch.Size([24, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_att3, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        local_x4_max = F.adaptive_max_pool1d(local_att4, 1).view(batch_size, -1)  # torch.Size([24, 1024]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max, local_x4_max], dim=1)
        # torch.Size([24, (128+128+256+512)*2])
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls25(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls25, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        # self.geo_xyz1 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_geo_1 = Local_op(in_channels=10, out_channels=10)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)
        # self.affine_alpha1 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        # self.affine_beta1 = nn.Parameter(torch.zeros([1, 1, 1, 64]))
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        # self.geo_xyz2 = nn.Sequential(nn.Conv1d(10, 256, 1, bias=False), nn.BatchNorm1d(256))
        self.gather_geo_2 = Local_op(in_channels=10, out_channels=10)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        # self.affine_alpha2 = nn.Parameter(torch.ones([1, 1, 1, 128]))
        # self.affine_beta2 = nn.Parameter(torch.zeros([1, 1, 1, 128]))
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
        )

        # self.geo_xyz3 = nn.Sequential(nn.Conv1d(10, 512, 1, bias=False), nn.BatchNorm1d(512))
        self.gather_geo_3 = Local_op(in_channels=10, out_channels=10)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)
        # self.affine_alpha3 = nn.Parameter(torch.ones([1, 1, 1, 256]))
        # self.affine_beta3 = nn.Parameter(torch.zeros([1, 1, 1, 256]))
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
        )

        # self.geo_xyz4 = nn.Sequential(nn.Conv1d(10, 1024, 1, bias=False), nn.BatchNorm1d(1024))
        self.gather_geo_4 = Local_op(in_channels=10, out_channels=10)
        self.gather_fea_4 = Local_op(in_channels=1024, out_channels=1024)
        self.local_scale4 = SA_Layer(channels = 1024)
        # self.affine_alpha4 = nn.Parameter(torch.ones([1, 1, 1, 512]))
        # self.affine_beta4 = nn.Parameter(torch.zeros([1, 1, 1, 512]))
        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
        )

        self.linear1 = nn.Linear(128+256+512+1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1)) + x1) # B, 64, N
        # x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N
        global_fea = x2

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=global_fea.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo1.size()  # torch.Size([24, 512, 24, 10])
        # geo1 = geo1.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 24]),(B,N,D,S)
        # geo1 = geo1.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea1 = F.relu(self.geo_xyz1(geo1))  # torch.Size(B*N,D',S),D':128
        # geo_fea1 = geo_fea1.reshape(b, n, s, -1)  # 24,512,8,128
        # mean = sample_points.unsqueeze(dim=-2)  # [B, npoint, 1, d]
        # std = torch.std(grouped_points - mean)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_fea1 = F.relu(self.net1(local_sem1) + local_sem1)
        local_att1 = self.local_scale1(local_fea1,local_geo1)  # 24,128,512

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=32, xyz=sample_xyz, point=local_att1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo2.size()  # torch.Size([24, 512, 16, 10])
        # geo2 = geo2.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 16]),(B,N,D,S)
        # geo2 = geo2.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea2 = F.relu(self.geo_xyz2(geo2))  # torch.Size(B*N,D',S),D':128
        # geo_fea2 = geo_fea2.reshape(b, n, s, -1)  # 24,256,16,256
        # mean = sample_points.unsqueeze(dim=-2)  # [B, npoint, 1, d]
        # std = torch.std(grouped_points - mean)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_fea2 = F.relu(self.net2(local_sem2) + local_sem2)
        local_att2 = self.local_scale2(local_fea2,local_geo2)  # 24,256,256

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=32, xyz=sample_xyz, point=local_att2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo3.size()  # torch.Size([24, 512, 8, 10])
        # geo3 = geo3.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 8]),(B,N,D,S)
        # geo3 = geo3.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea3 = F.relu(self.geo_xyz3(geo3))  # torch.Size(B*N,D',S),D':256
        # geo_fea3 = geo_fea3.reshape(b, n, s, -1)  # 24,1024,8,256
        # mean = sample_points.unsqueeze(dim=-2)  # [B, npoint, 1, d]
        # std = torch.std(grouped_points - mean)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_fea3 = F.relu(self.net3(local_sem3) + local_sem3)
        local_att3 = self.local_scale3(local_fea3,local_geo3)  # 24,512,128

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=64, nsample=32, xyz=sample_xyz, point=local_att3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 64, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz.view(B, 64, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo4.size()  # torch.Size([24, 512, 8, 10])
        # local_geo4 = geo4.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 8]),(B,N,D,S)
        # geo4 = geo4.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea4 = F.relu(self.geo_xyz4(geo4))  # torch.Size(B*N,D',S),D':256
        # geo_fea4 = geo_fea4.reshape(b, n, s, -1)  # 24,1024,8,256
        # std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha4 * grouped_points + self.affine_beta4
        sem_fea4 = torch.cat((sample_points.view(B, 64, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_fea4 = F.relu(self.net4(local_sem4) + local_sem4)
        local_att4 = self.local_scale4(local_fea4,local_geo4) # 24,1024,64

        local_x1_max = F.adaptive_max_pool1d(local_att1, 1).view(batch_size, -1) # torch.Size([24, 128]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_att2, 1).view(batch_size, -1)  # torch.Size([24, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_att3, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        local_x4_max = F.adaptive_max_pool1d(local_att4, 1).view(batch_size, -1)  # torch.Size([24, 1024]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max, local_x4_max], dim=1)
        # torch.Size([24, (128+128+256+512)*2])0
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls26(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls26, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        # self.geo_xyz1 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_geo_1 = Local_op(in_channels=10, out_channels=128)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)
        # self.affine_alpha1 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        # self.affine_beta1 = nn.Parameter(torch.zeros([1, 1, 1, 64]))
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        # self.geo_xyz2 = nn.Sequential(nn.Conv1d(10, 256, 1, bias=False), nn.BatchNorm1d(256))
        self.gather_geo_2 = Local_op(in_channels=10, out_channels=256)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        # self.affine_alpha2 = nn.Parameter(torch.ones([1, 1, 1, 128]))
        # self.affine_beta2 = nn.Parameter(torch.zeros([1, 1, 1, 128]))
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
        )

        # self.geo_xyz3 = nn.Sequential(nn.Conv1d(10, 512, 1, bias=False), nn.BatchNorm1d(512))
        self.gather_geo_3 = Local_op(in_channels=10, out_channels=512)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)
        # self.affine_alpha3 = nn.Parameter(torch.ones([1, 1, 1, 256]))
        # self.affine_beta3 = nn.Parameter(torch.zeros([1, 1, 1, 256]))
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
        )

        # self.geo_xyz4 = nn.Sequential(nn.Conv1d(10, 1024, 1, bias=False), nn.BatchNorm1d(1024))
        self.gather_geo_4 = Local_op(in_channels=10, out_channels=1024)
        self.gather_fea_4 = Local_op(in_channels=1024, out_channels=1024)
        self.local_scale4 = SA_Layer(channels = 1024)
        # self.affine_alpha4 = nn.Parameter(torch.ones([1, 1, 1, 512]))
        # self.affine_beta4 = nn.Parameter(torch.zeros([1, 1, 1, 512]))
        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
        )

        self.linear1 = nn.Linear((128+256+512+1024)*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1)) + x1) # B, 64, N
        # x3 = F.relu(self.bn3(self.conv3(x2)))  # B, 128, N

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=24, xyz=xyz, point=x2.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 24, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo1.size()  # torch.Size([24, 512, 24, 10])
        # geo1 = geo1.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 24]),(B,N,D,S)
        # geo1 = geo1.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea1 = F.relu(self.geo_xyz1(geo1))  # torch.Size(B*N,D',S),D':128
        # geo_fea1 = geo_fea1.reshape(b, n, s, -1)  # 24,512,8,128
        # mean = sample_points.unsqueeze(dim=-2)  # [B, npoint, 1, d]
        # std = torch.std(grouped_points - mean)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 24, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_sem1 = F.relu(self.net1(local_sem1) + local_sem1)
        local_att1 = self.local_scale1(local_sem1,local_geo1)  # 24,128,512
        local_fea1 = torch.cat([local_att1,local_sem1], dim = 1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=24, xyz=sample_xyz, point=local_sem1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 24, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo2.size()  # torch.Size([24, 512, 16, 10])
        # geo2 = geo2.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 16]),(B,N,D,S)
        # geo2 = geo2.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea2 = F.relu(self.geo_xyz2(geo2))  # torch.Size(B*N,D',S),D':128
        # geo_fea2 = geo_fea2.reshape(b, n, s, -1)  # 24,256,16,256
        # mean = sample_points.unsqueeze(dim=-2)  # [B, npoint, 1, d]
        # std = torch.std(grouped_points - mean)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 24, 1), grouped_points), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_sem2 = F.relu(self.net2(local_sem2) + local_sem2)
        local_att2 = self.local_scale2(local_sem2,local_geo2)  # 24,256,256
        local_fea2 = torch.cat([local_att2, local_sem2], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=24, xyz=sample_xyz, point=local_sem2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 24, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo3.size()  # torch.Size([24, 512, 8, 10])
        # geo3 = geo3.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 8]),(B,N,D,S)
        # geo3 = geo3.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea3 = F.relu(self.geo_xyz3(geo3))  # torch.Size(B*N,D',S),D':256
        # geo_fea3 = geo_fea3.reshape(b, n, s, -1)  # 24,1024,8,256
        # mean = sample_points.unsqueeze(dim=-2)  # [B, npoint, 1, d]
        # std = torch.std(grouped_points - mean)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 24, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_sem3 = F.relu(self.net3(local_sem3) + local_sem3)
        local_att3 = self.local_scale3(local_sem3,local_geo3)  # 24,512,128
        local_fea3 = torch.cat([local_att3, local_sem3], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=64, nsample=24, xyz=sample_xyz, point=local_sem3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 64, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz.view(B, 64, 1, -1).repeat(1, 1, 24, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # b, n, s, d = geo4.size()  # torch.Size([24, 512, 8, 10])
        # local_geo4 = geo4.permute(0, 1, 3, 2)  # torch.Size([24, 512, 10, 8]),(B,N,D,S)
        # geo4 = geo4.reshape(-1, d, s)  # torch.Size(B*N,D,S)
        # geo_fea4 = F.relu(self.geo_xyz4(geo4))  # torch.Size(B*N,D',S),D':256
        # geo_fea4 = geo_fea4.reshape(b, n, s, -1)  # 24,1024,8,256
        # std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        # grouped_points = (grouped_points - mean) / (std + 1e-5)
        # grouped_points = self.affine_alpha4 * grouped_points + self.affine_beta4
        sem_fea4 = torch.cat((sample_points.view(B, 64, 1, -1).repeat(1, 1, 24, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_fea4 = F.relu(self.net4(local_sem4) + local_sem4)
        local_att4 = self.local_scale4(local_fea4,local_geo4) # 24,1024,64
        local_fea4 = torch.cat([local_att4, local_sem4], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_fea1, 1).view(batch_size, -1) # torch.Size([24, 128]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_fea2, 1).view(batch_size, -1)  # torch.Size([24, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_fea3, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        local_x4_max = F.adaptive_max_pool1d(local_fea4, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max, local_x4_max], dim=1)
        # torch.Size([24, (128+128+256+512)*2])0
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls27(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls27, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        # self.geo_xyz1 = nn.Sequential(nn.Conv1d(10, 128, 1, bias=False), nn.BatchNorm1d(128))
        self.gather_geo_1 = Local_op(in_channels=10, out_channels=128)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)
        # self.affine_alpha1 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        # self.affine_beta1 = nn.Parameter(torch.zeros([1, 1, 1, 64]))
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        # self.geo_xyz2 = nn.Sequential(nn.Conv1d(10, 256, 1, bias=False), nn.BatchNorm1d(256))
        self.gather_geo_2 = Local_op(in_channels=10, out_channels=256)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        # self.affine_alpha2 = nn.Parameter(torch.ones([1, 1, 1, 128]))
        # self.affine_beta2 = nn.Parameter(torch.zeros([1, 1, 1, 128]))
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
        )

        # self.geo_xyz3 = nn.Sequential(nn.Conv1d(10, 512, 1, bias=False), nn.BatchNorm1d(512))
        self.gather_geo_3 = Local_op(in_channels=10, out_channels=512)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)
        # self.affine_alpha3 = nn.Parameter(torch.ones([1, 1, 1, 256]))
        # self.affine_beta3 = nn.Parameter(torch.zeros([1, 1, 1, 256]))
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
        )

        # self.geo_xyz4 = nn.Sequential(nn.Conv1d(10, 1024, 1, bias=False), nn.BatchNorm1d(1024))
        self.gather_geo_4 = Local_op(in_channels=10, out_channels=1024)
        self.gather_fea_4 = Local_op(in_channels=1024, out_channels=1024)
        self.local_scale4 = SA_Layer(channels = 1024)
        # self.affine_alpha4 = nn.Parameter(torch.ones([1, 1, 1, 512]))
        # self.affine_beta4 = nn.Parameter(torch.zeros([1, 1, 1, 512]))
        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
        )

        self.linear1 = nn.Linear(128+256+512+1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=32, xyz=xyz, point=x2.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        grouped_points_norm = grouped_points - sample_points.view(B, 512, 1, -1)  # [B, npoint, nsample, D]
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_points_norm), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_sem1 = F.relu(self.net1(local_sem1) + local_sem1)
        local_att1 = self.local_scale1(local_sem1,local_geo1)  # 24,128,512
        # local_fea1 = torch.cat([local_att1,local_sem1], dim = 1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=32, xyz=sample_xyz, point=local_sem1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist),dim=-1)
        grouped_points_norm = grouped_points - sample_points.view(B, 256, 1, -1)  # [B, npoint, nsample, D]
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_points_norm), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_sem2 = F.relu(self.net2(local_sem2) + local_sem2)
        local_att2 = self.local_scale2(local_sem2,local_geo2)  # 24,256,256
        # local_fea2 = torch.cat([local_att2, local_sem2], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=32, xyz=sample_xyz, point=local_sem2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        grouped_points_norm = grouped_points - sample_points.view(B, 128, 1, -1)  # [B, npoint, nsample, D]
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 32, 1), grouped_points_norm), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_sem3 = F.relu(self.net3(local_sem3) + local_sem3)
        local_att3 = self.local_scale3(local_sem3,local_geo3)  # 24,512,128
        # local_fea3 = torch.cat([local_att3, local_sem3], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=64, nsample=32, xyz=sample_xyz, point=local_sem3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 64, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz.view(B, 64, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        grouped_points_norm = grouped_points - sample_points.view(B, 64, 1, -1)  # [B, npoint, nsample, D]
        sem_fea4 = torch.cat((sample_points.view(B, 64, 1, -1).repeat(1, 1, 32, 1), grouped_points_norm), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_fea4 = F.relu(self.net4(local_sem4) + local_sem4)
        local_att4 = self.local_scale4(local_fea4,local_geo4) # 24,1024,64
        # local_fea4 = torch.cat([local_att4, local_sem4], dim=1)

        local_x1_max = F.adaptive_max_pool1d(local_att1, 1).view(batch_size, -1) # torch.Size([24, 128]),(B,D)
        local_x2_max = F.adaptive_max_pool1d(local_att2, 1).view(batch_size, -1)  # torch.Size([24, 256]),(B,D)
        local_x3_max = F.adaptive_max_pool1d(local_att3, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        local_x4_max = F.adaptive_max_pool1d(local_att4, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max, local_x4_max], dim=1)
        # torch.Size([24, (128+128+256+512)*2])0
        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls28(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls28, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.gather_geo_1 = Local_op(in_channels=10, out_channels=128)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)
        self.affine_alpha1 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        self.affine_beta1 = nn.Parameter(torch.zeros([1, 1, 1, 64]))
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        self.gather_geo_2 = Local_op(in_channels=10, out_channels=256)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        self.affine_alpha2 = nn.Parameter(torch.ones([1, 1, 1, 128]))
        self.affine_beta2 = nn.Parameter(torch.zeros([1, 1, 1, 128]))
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
        )

        self.gather_geo_3 = Local_op(in_channels=10, out_channels=512)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)
        self.affine_alpha3 = nn.Parameter(torch.ones([1, 1, 1, 256]))
        self.affine_beta3 = nn.Parameter(torch.zeros([1, 1, 1, 256]))
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
        )

        self.gather_geo_4 = Local_op(in_channels=10, out_channels=1024)
        self.gather_fea_4 = Local_op(in_channels=1024, out_channels=1024)
        self.local_scale4 = SA_Layer(channels = 1024)
        self.affine_alpha4 = nn.Parameter(torch.ones([1, 1, 1, 512]))
        self.affine_beta4 = nn.Parameter(torch.zeros([1, 1, 1, 512]))
        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
        )

        # self.linear = nn.Linear((128+256+512+1024)*2, 1024, bias=False)
        # self.bn = nn.BatchNorm1d(1024)
        # self.dp = nn.Dropout(0.5)
        self.linear1 = nn.Linear(128+256+512+1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) # B, 64, N

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=16, xyz=xyz, point=x2.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 512, 1, -1)  # [B, npoint, nsample, D]
        std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_sem1 = F.relu(self.net1(local_sem1)) + local_sem1
        local_att1 = self.local_scale1(local_sem1,local_geo1)  # 24,128,512
        # local_fea1 = torch.cat([local_att1,local_sem1], dim = 1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=16, xyz=sample_xyz, point=local_sem1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist),dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 256, 1, -1)  # [B, npoint, nsample, D]
        std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_sem2 = F.relu(self.net2(local_sem2)) + local_sem2
        local_att2 = self.local_scale2(local_sem2,local_geo2)  # 24,256,256
        # local_fea2 = torch.cat([local_att2, local_sem2], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=16, xyz=sample_xyz, point=local_sem2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 128, 1, -1)  # [B, npoint, nsample, D]
        std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_sem3 = F.relu(self.net3(local_sem3)) + local_sem3
        local_att3 = self.local_scale3(local_sem3,local_geo3)  # 24,512,128
        # local_fea3 = torch.cat([local_att3, local_sem3], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=64, nsample=16, xyz=sample_xyz, point=local_sem3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 64, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz.view(B, 64, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 64, 1, -1)  # [B, npoint, nsample, D]
        std, mean = torch.std_mean(grouped_points, dim=2, keepdim=True)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha4 * grouped_points + self.affine_beta4
        sem_fea4 = torch.cat((sample_points.view(B, 64, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_fea4 = F.relu(self.net4(local_sem4)) + local_sem4
        local_att4 = self.local_scale4(local_fea4,local_geo4) # 24,1024,64
        # local_fea4 = torch.cat([local_att4, local_sem4], dim=1)

        local_x1_max = F.adaptive_avg_pool1d(local_att1, 1).view(batch_size, -1) # torch.Size([24, 128]),(B,D)
        local_x2_max = F.adaptive_avg_pool1d(local_att2, 1).view(batch_size, -1)  # torch.Size([24, 256]),(B,D)
        local_x3_max = F.adaptive_avg_pool1d(local_att3, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        local_x4_max = F.adaptive_avg_pool1d(local_att4, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        x_max = torch.cat([local_x1_max, local_x2_max, local_x3_max, local_x4_max], dim=1)
        # torch.Size([24, (128+128+256+512)*2])0

        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls29(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls29, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.gather_geo_1 = Local_op(in_channels=10, out_channels=128)
        self.gather_fea_1 = Local_op(in_channels=128, out_channels=128)
        self.local_scale1 = SA_Layer(channels = 128)
        self.affine_alpha1 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        self.affine_beta1 = nn.Parameter(torch.zeros([1, 1, 1, 64]))
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        self.gather_geo_2 = Local_op(in_channels=10, out_channels=256)
        self.gather_fea_2 = Local_op(in_channels=256, out_channels=256)
        self.local_scale2 = SA_Layer(channels = 256)
        self.affine_alpha2 = nn.Parameter(torch.ones([1, 1, 1, 128]))
        self.affine_beta2 = nn.Parameter(torch.zeros([1, 1, 1, 128]))
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
        )

        self.gather_geo_3 = Local_op(in_channels=10, out_channels=512)
        self.gather_fea_3 = Local_op(in_channels=512, out_channels=512)
        self.local_scale3 = SA_Layer(channels = 512)
        self.affine_alpha3 = nn.Parameter(torch.ones([1, 1, 1, 256]))
        self.affine_beta3 = nn.Parameter(torch.zeros([1, 1, 1, 256]))
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
        )

        self.gather_geo_4 = Local_op(in_channels=10, out_channels=1024)
        self.gather_fea_4 = Local_op(in_channels=1024, out_channels=1024)
        self.local_scale4 = SA_Layer(channels = 1024)
        self.affine_alpha4 = nn.Parameter(torch.ones([1, 1, 1, 512]))
        self.affine_beta4 = nn.Parameter(torch.zeros([1, 1, 1, 512]))
        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
        )

        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) + x1 # B, 64, N

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=8, xyz=xyz, point=x2.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz.view(B, 512, 1, -1).repeat(1, 1, 8, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 512, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 8, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_sem1 = F.relu(self.net1(local_sem1)) + local_sem1
        local_att1 = self.local_scale1(local_sem1,local_geo1)  # 24,128,512
        # local_fea1 = torch.cat([local_att1,local_sem1], dim = 1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=8, xyz=sample_xyz, point=local_att1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz.view(B, 256, 1, -1).repeat(1, 1, 8, 1), grouped_xyz, grouped_xyz_norm, dist),dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 256, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 8, 1), grouped_points), dim=-1) # 24,256,24,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_sem2 = F.relu(self.net2(local_sem2)) + local_sem2
        local_att2 = self.local_scale2(local_sem2,local_geo2)  # 24,256,256
        # local_fea2 = torch.cat([local_att2, local_sem2], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=8, xyz=sample_xyz, point=local_att2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz.view(B, 128, 1, -1).repeat(1, 1, 8, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 128, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 8, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_sem3 = F.relu(self.net3(local_sem3)) + local_sem3
        local_att3 = self.local_scale3(local_sem3,local_geo3)  # 24,512,128
        # local_fea3 = torch.cat([local_att3, local_sem3], dim=1)

        sample_xyz, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=64, nsample=8, xyz=sample_xyz, point=local_att3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz.view(B, 64, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz.view(B, 64, 1, -1).repeat(1, 1, 8, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 64, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha4 * grouped_points + self.affine_beta4
        sem_fea4 = torch.cat((sample_points.view(B, 64, 1, -1).repeat(1, 1, 8, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_fea4 = F.relu(self.net4(local_sem4)) + local_sem4
        local_att4 = self.local_scale4(local_fea4,local_geo4) # 24,1024,64
        local_fea4 = torch.cat([local_att4, local_sem4], dim=1)

        # local_x1_max = F.adaptive_max_pool1d(local_fea1, 1).view(batch_size, -1) # torch.Size([24, 128]),(B,D)
        # local_x2_max = F.adaptive_max_pool1d(local_fea2, 1).view(batch_size, -1)  # torch.Size([24, 256]),(B,D)
        # local_x3_max = F.adaptive_max_pool1d(local_fea3, 1).view(batch_size, -1)  # torch.Size([24, 512]),(B,D)
        local_x4_max = F.adaptive_max_pool1d(local_fea4, 1).view(batch_size, -1)  # torch.Size([24, 1024]),(B,D)
        x_max = torch.cat([local_x4_max], dim=1)
        # torch.Size([24, (128+128+256+512)*2])0

        x = F.leaky_relu(self.bn6(self.linear1(x_max)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


