import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group, index_points, square_distance, geometric_point_descriptor


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PointNetFeaturePropagation, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, D1, N]
            points2: input points data, [B, D2, S]
        Return:
            new_points: upsampled points data, [B, D1+D2, N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # print('xyz1=>',xyz1.shape,'xyz2=>',xyz2.shape)
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-9)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = F.relu(self.bn1(self.conv1(new_points))) + new_points
        return new_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # 针对gather_local_0
        b, n, s, d = x.size() # torch.Size([32, 512, 32, 128])
        # print("b, n, s, d = x.size()=>", x.shape)
        x = x.permute(0, 1, 3, 2) # torch.Size([32, 512, 128, 32])
        x = x.reshape(-1, d, s) # torch.Size([16384, 128, 32])
        # print("x = x.reshape(-1, d, s)=>",x.shape)
        batch_size, _, N = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) + x  #torch.Size([16384, 128, 32])
        # print('x = F.relu(self.bn2(self.conv2(x)))=>',x.shape)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # torch.Size([16384, 128])
        # print('x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)=>', x.shape)
        x = x.reshape(b, n, -1).permute(0, 2, 1) # torch.Size([32, 128, 512]),(B,D,N)
        # print('x = x.reshape(b, n, -1).permute(0, 2, 1)=>', x.shape)
        return x

class Sa_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
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
        energy = x_q - x_k  # torch.Size([B, D, D])
        attention = self.softmax(energy) # torch.Size([B, D, D])
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) # torch.Size([B, D, N])
        # out = torch.bmm(attention, x_v) # torch.Size([B, D, N])
        out = torch.mul(attention, x_v + xyz)  # torch.Size([B, D, N])
        out = self.act(self.after_norm(self.trans_conv(xx - out))) + xx
        return out

class Pct_partseg1(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg1, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(3, 256, 1)
        self.sa1 = SA_Layer(channels = 256)
        self.sa2 = SA_Layer(channels = 256)
        self.sa3 = SA_Layer(channels = 256)
        self.sa4 = SA_Layer(channels = 256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 1024 + 256, mlp=[1024])


        self.linear1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 64])

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 64, 256(128+128)])
        feature_1 = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N

        new_xyz_feature = self.pos_xyz(new_xyz.permute(0, 2, 1)) # torch.Size([32, 256, 256]),(B,D,N)

        x1 = self.sa1(feature_1, new_xyz_feature)
        x2 = self.sa2(x1, new_xyz_feature)
        x3 = self.sa3(x2, new_xyz_feature)
        x4 = self.sa4(x3, new_xyz_feature)
        x_tr = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 1024, 256]),(B,D,N)

        x = torch.cat([x_tr, feature_1], dim=1)  # torch.Size([32, 1024+256(1280), 256]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)
        x_max_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        x = self.fp1(xyz, new_xyz, torch.cat([cls_label_feature, x_max_feature], 1), feature_1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.dp2(x)
        x = self.linear3(x) # torch.Size([32, 50, 2048])

        return x

class Pct_partseg2(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg2, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(3, 128, 1)
        self.sa1 = SA_Layer(channels = 128)
        self.sa2 = SA_Layer(channels = 128)
        self.sa3 = SA_Layer(channels = 128)
        self.sa4 = SA_Layer(channels = 128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(640, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 1024 + 128, mlp=[1024])

        self.linear1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 64])

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N

        new_xyz_feature = self.pos_xyz(new_xyz.permute(0, 2, 1)) # torch.Size([32, 128, 512]),(B,D,N)

        x1 = self.sa1(feature_0, new_xyz_feature)
        x2 = self.sa2(x1, new_xyz_feature)
        x3 = self.sa3(x2, new_xyz_feature)
        x4 = self.sa4(x3, new_xyz_feature)
        x_tr = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 512, 512]),(B,D,N)

        x = torch.cat([x_tr, feature_0], dim=1)  # torch.Size([32, 512+128(640), 256]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)
        x_max_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        x = self.fp1(xyz, new_xyz, torch.cat([cls_label_feature, x_max_feature], 1), feature_0)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.dp2(x)
        x = self.linear3(x) # torch.Size([32, 50, 2048])

        return x

class Pct_partseg3(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg3, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_xyz = nn.Conv1d(3, 128, 1)
        self.sa1 = SA_Layer(channels = 128)
        self.sa2 = SA_Layer(channels = 128)
        self.sa3 = SA_Layer(channels = 128)
        self.sa4 = SA_Layer(channels = 128)
        self.sa5 = SA_Layer(channels = 128)
        self.sa6 = SA_Layer(channels = 128)
        self.sa7 = SA_Layer(channels = 128)
        self.sa8 = SA_Layer(channels = 128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1240, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 1024 + 128, mlp=[1024])

        self.linear1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 64])

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N

        new_xyz_feature = self.pos_xyz(new_xyz.permute(0, 2, 1)) # torch.Size([32, 128, 512]),(B,D,N)

        x1 = self.sa1(feature_0, new_xyz_feature)
        x2 = self.sa2(x1, new_xyz_feature)
        x3 = self.sa3(x2, new_xyz_feature)
        x4 = self.sa4(x3, new_xyz_feature)
        x5 = self.sa5(x4, new_xyz_feature)
        x6 = self.sa6(x5, new_xyz_feature)
        x7 = self.sa7(x6, new_xyz_feature)
        x8 = self.sa8(x7, new_xyz_feature)
        x_tr = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)  # torch.Size([32, 1024, 512]),(B,D,N)

        x = torch.cat([x_tr, feature_0], dim=1)  # torch.Size([32, 1024+128(1240), 256]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)
        x_max_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        x = self.fp1(xyz, new_xyz, torch.cat([cls_label_feature, x_max_feature], 1), feature_0)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.dp2(x)
        x = self.linear3(x) # torch.Size([32, 50, 2048])

        return x

class Pct_partseg4(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg4, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pos_local_xyz = nn.Conv1d(3, 128, 1)
        self.pos_global_xyz = nn.Conv1d(3, 128, 1)

        self.local_sa1 = SA_Layer(channels = 128)
        self.local_sa2 = SA_Layer(channels = 128)
        self.local_sa3 = SA_Layer(channels = 128)
        self.local_sa4 = SA_Layer(channels = 128)

        self.global_sa1 = SA_Layer(channels = 128)
        self.global_sa2 = SA_Layer(channels = 128)
        self.global_sa3 = SA_Layer(channels = 128)
        self.global_sa4 = SA_Layer(channels = 128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(640, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fp1 = PointNetFeaturePropagation(in_channel=512+128+128, mlp=[512+128+128])

        self.linear1 = nn.Conv1d(640 +768 + 64, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        x_local = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        x_global = F.relu(self.bn3(self.conv3(x)))  # torch.Size([32, 128, 2048])
        x_local = x_local.permute(0, 2, 1) # torch.Size([32, 2048, 64])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])
        'local'
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x_local)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        new_xyz_feature = self.pos_local_xyz(new_xyz.permute(0, 2, 1)) # torch.Size([32, 128, 512]),(B,D,N)
        x1 = self.local_sa1(feature_0, new_xyz_feature)
        x2 = self.local_sa2(x1, new_xyz_feature)
        x3 = self.local_sa3(x2, new_xyz_feature)
        x4 = self.local_sa4(x3, new_xyz_feature)
        x_tr_local = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 512, 512]),(B,D,N)
        x_local = torch.cat([x_tr_local, feature_0], dim=1)  # torch.Size([32, 512+128(640), 256]),(B,D,N)
        # x_local = self.conv_fuse(x_local) # torch.Size([32, 1024, 256]),(B,D,N)
        x_local_max_feature = F.adaptive_max_pool1d(x_local, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 512+128(640), 2048])
        x_local_max_feature = self.fp1(xyz, new_xyz, x_local_max_feature,feature_0) # torch.Size([32, 640+128, 2048])

        'global'
        new_xyz_global_feature = self.pos_global_xyz(xyz.permute(0, 2, 1))  # torch.Size([32, 128, 2048]),(B,D,N)
        x1_global = self.global_sa1(x_global, new_xyz_global_feature)
        x2_global = self.global_sa2(x1_global, new_xyz_global_feature)
        x3_global = self.global_sa3(x2_global, new_xyz_global_feature)
        x4_global = self.global_sa4(x3_global, new_xyz_global_feature)
        x_tr_global = torch.cat((x1_global, x2_global, x3_global, x4_global), dim=1)  # torch.Size([32, 512, 2048]),(B,D,N)
        x_global = torch.cat([x_tr_global, x_global], dim=1)  # torch.Size([32, 512+128(640), 2048]),(B,D,N)
        x_global = F.adaptive_max_pool1d(x_global, 1)  # (B,512+128(640),1)
        x_global_max_frature = x_global.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # (B,512+128(640),N)

        x = torch.cat((x_global_max_frature, x_local_max_feature, cls_label_feature), dim = 1) # (32,640 +768 + 64,N)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.dp2(x)
        x = self.linear3(x) # torch.Size([32, 50, 2048])

        return x

class Pct_partseg5(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg5, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pos_xyz = nn.Conv1d(14, 128, 1)

        self.global_sa1 = SA_Layer(channels=128)
        self.global_sa2 = SA_Layer(channels=128)
        self.global_sa3 = SA_Layer(channels=128)
        self.global_sa4 = SA_Layer(channels=128)

        self.fuse_conv = nn.Sequential(nn.Conv1d(640, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024*2+64, 512, 1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Conv1d(256, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, N, 64

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        x1 = self.global_sa1(global_feature, global_xyz)
        x2 = self.global_sa2(global_feature, global_xyz)
        x3 = self.global_sa3(global_feature, global_xyz)
        x4 = self.global_sa4(global_feature, global_xyz)
        global_x = torch.cat([x1, x2, x3, x4, global_feature], dim=1) # torch.Size([32, 128*5, 2048]),(B,D,N)
        global_x = self.fuse_conv(global_x) # torch.Size([32, 1024, 2048]),(B,D,N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024,2048]),(B,D,N)

        x = torch.cat([global_x, global_x_max, cls_label_feature], dim=1)  # torch.Size([32, 1024*2+64,2048])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_partseg6(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg6, self).__init__()
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

        self.fp1 = PointNetFeaturePropagation(in_channel=256 + 256 + 64, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv = nn.Conv1d(128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature1 = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_xyz_feature1 = self.local_pos_xyz1(local_xyz1.permute(0,2,1)) # torch.Size([32, 128, 512]),B,D,N
        x1 = self.local_scale1(local_feature_1, local_xyz_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_x1 = torch.cat([x1,local_feature_1], dim = 1) # torch.Size([32, 256, 512]),B,D,N

        local_xyz2, local_feature2 = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature_1.permute(0, 2, 1))
        local_feature_2 = self.gather_local_2(local_feature2) # torch.Size([32, 256, 256]),B,D,N
        local_xyz_feature_2 = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1)) # torch.Size([32, 256, 256]),B,D,N
        x2 = self.local_scale2(local_feature_2, local_xyz_feature_2) # torch.Size([32, 256, 256]),B,D,N
        local_x2 = torch.cat([x2, local_feature_2], dim= 1) # torch.Size([32, 512, 256]),B,D,N

        local_xyz3, local_feature3 = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature_2.permute(0, 2, 1))
        local_feature_3 = self.gather_local_3(local_feature3) # torch.Size([32, 512, 128]),B,D,N
        local_xyz_feature_3 = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1)) # torch.Size([32, 512, 128]),B,D,N
        x3 = self.local_scale3(local_feature_3, local_xyz_feature_3) # torch.Size([32, 512, 128]),B,D,N
        local_x3 = torch.cat([x3, local_feature_3], dim = 1) # torch.Size([32, 1024, 128]),B,D,N

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 2048]),(B,D,N)
        x = self.global_sa(global_feature, global_xyz) # torch.Size([32, 128, 2048]),(B,D,N)
        global_x = torch.cat([x, global_feature], dim=1) # torch.Size([32, 256, 2048]),(B,D,N)

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3) # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x,cls_label_feature],dim = 1), x1)  # torch.Size([32, 256+64+256->128, 2048]),(B,D,N)

        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x

class Pct_partseg7(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg7, self).__init__()
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
        self.global_sa1 = SA_Layer(channels = 128)
        self.global_sa2 = SA_Layer(channels=128)
        self.global_sa3 = SA_Layer(channels=128)
        self.global_sa4 = SA_Layer(channels=128)

        self.fp1 = PointNetFeaturePropagation(in_channel=128+64+256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=128+512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256+512, mlp=[512])

        # self.conv_fuse = nn.Sequential(nn.Conv1d(640, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv = nn.Conv1d(128*4+128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature1 = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_xyz_feature1 = self.local_pos_xyz1(local_xyz1.permute(0,2,1)) # torch.Size([32, 128, 512]),B,D,N
        x1 = self.local_scale1(local_feature_1, local_xyz_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_x1 = x1 # torch.Size([32, 128, 512]),B,D,N

        local_xyz2, local_feature2 = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature_1.permute(0, 2, 1))
        local_feature_2 = self.gather_local_2(local_feature2) # torch.Size([32, 256, 256]),B,D,N
        local_xyz_feature_2 = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1)) # torch.Size([32, 256, 256]),B,D,N
        x2 = self.local_scale2(local_feature_2, local_xyz_feature_2) # torch.Size([32, 256, 256]),B,D,N
        local_x2 = x2 # torch.Size([32, 256, 256]),B,D,N

        local_xyz3, local_feature3 = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature_2.permute(0, 2, 1))
        local_feature_3 = self.gather_local_3(local_feature3) # torch.Size([32, 512, 128]),B,D,N
        local_xyz_feature_3 = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1)) # torch.Size([32, 512, 128]),B,D,N
        x3 = self.local_scale3(local_feature_3, local_xyz_feature_3) # torch.Size([32, 512, 128]),B,D,N
        local_x3 = x3 # torch.Size([32, 512, 128]),B,D,N

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 2048]),(B,D,N)
        global_x1 = self.global_sa1(global_feature, global_xyz) # torch.Size([32, 128, 2048]),(B,D,N)
        global_x = global_x1 # torch.Size([32, 128, 2048]),(B,D,N)

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3) # torch.Size([32, 256+512->512, 1024]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 128+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x,cls_label_feature],dim = 1), x1)  # torch.Size([32, 128+64+256->128, 2048]),(B,D,N)

        global_x2 = self.global_sa2(global_feature, global_xyz)
        global_x3 = self.global_sa3(global_feature, global_xyz)
        global_x4 = self.global_sa4(global_feature, global_xyz)
        global_x = torch.cat((global_x1, global_x2, global_x3, global_x4), dim=1)  # torch.Size([32, 128*4, 2048]),(B,D,N)
        # global_x = self.conv_fuse(global_x) # torch.Size([32, 1024, 2048]),(B,D,N)
        x = torch.cat([global_x,x],dim = 1) # torch.Size([32, 128*4 + 128, 2048]),(B,D,N)
        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x

class Pct_partseg8(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg8, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.local_pos_xyz = nn.Conv1d(14, 256, 1)
        self.global_pos_xyz = nn.Conv1d(14, 256, 1)

        self.local_sa1 = SA_Layer(channels = 256)
        self.local_sa2 = SA_Layer(channels = 256)
        self.local_sa3 = SA_Layer(channels = 256)
        self.local_sa4 = SA_Layer(channels = 256)
        self.global_sa1 = SA_Layer(channels = 256)
        self.global_sa2 = SA_Layer(channels = 256)
        self.global_sa3 = SA_Layer(channels = 256)
        self.global_sa4 = SA_Layer(channels = 256)
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1280+1280+64, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x)  # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        global_x = F.relu(self.bn3(self.conv3(x)))  # torch.Size([32, 256, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 64])

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature) # feature_0: torch.Size([32, 128, 512]),B,D,N
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature) # feature_1: torch.Size([32, 256, 256]),B,D,N
        local_xyz_feature = self.local_pos_xyz(new_xyz.permute(0, 2, 1)) # torch.Size([32, 256, 256]),(B,D,N)
        x1 = self.local_sa1(feature_1, local_xyz_feature)
        x2 = self.local_sa2(feature_1, local_xyz_feature)
        x3 = self.local_sa3(feature_1, local_xyz_feature)
        x4 = self.local_sa4(feature_1, local_xyz_feature)
        x_tr = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 1024, 256]),(B,D,N)
        local_x = torch.cat([x_tr, feature_1], dim=1)  # torch.Size([32, 1024+256(1280), 256]),(B,D,N)
        local_x_max_feature = F.adaptive_max_pool1d(local_x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        # torch.Size([32, 1280, 2048])
        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        global_xyz_feature = self.global_pos_xyz(xyz.permute(0, 2, 1)) # torch.Size([32, 256, 2048]),(B,D,N)
        x1 = self.global_sa1(global_x, global_xyz_feature)
        x2 = self.global_sa2(global_x, global_xyz_feature)
        x3 = self.global_sa3(global_x, global_xyz_feature)
        x4 = self.global_sa4(global_x, global_xyz_feature)
        x_tr = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([32, 1024, 2048]),(B,D,N)
        global_x = torch.cat([x_tr, global_x], dim=1)  # torch.Size([32, 1024+256(1280), 2048]),(B,D,N)

        x = torch.cat([global_x,local_x_max_feature,cls_label_feature], dim=1)  # torch.Size([32, 1280+1280+64, 2048]),(B,D,N)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.linear3(x) # torch.Size([32, 50, 2048])
        return x

class Pct_partseg9(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg9, self).__init__()
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
        self.global_sa1 = SA_Layer(channels=128)
        self.global_sa2 = SA_Layer(channels=128)
        self.global_sa3 = SA_Layer(channels=128)
        self.global_sa4 = SA_Layer(channels=128)

        self.fuse_conv = nn.Sequential(nn.Conv1d(640, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024 + (256+512+1024) + 64, 512, 1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature1 = sample_and_group(npoint=1024, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_xyz_feature1 = self.local_pos_xyz1(local_xyz1.permute(0,2,1)) # torch.Size([32, 128, 512]),B,D,N
        local_x1 = self.local_scale1(local_feature_1, local_xyz_feature1) # torch.Size([32, 128, 1024]),B,D,N
        local_x1 = torch.cat([local_x1,local_feature_1], dim = 1) # torch.Size([32, 256, 1024]),B,D,N

        local_xyz2, local_feature2 = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature_1.permute(0, 2, 1))
        local_feature_2 = self.gather_local_2(local_feature2) # torch.Size([32, 256, 512]),B,D,N
        local_xyz_feature_2 = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1)) # torch.Size([32, 256, 512]),B,D,N
        x2 = self.local_scale2(local_feature_2, local_xyz_feature_2) # torch.Size([32, 256, 512]),B,D,N
        local_x2 = torch.cat([x2, local_feature_2], dim= 1) # torch.Size([32, 512, 512]),B,D,N


        local_xyz3, local_feature3 = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature_2.permute(0, 2, 1))
        local_feature_3 = self.gather_local_3(local_feature3) # torch.Size([32, 512, 256]),B,D,N
        local_xyz_feature_3 = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1)) # torch.Size([32, 512, 256]),B,D,N
        x3 = self.local_scale3(local_feature_3, local_xyz_feature_3) # torch.Size([32, 512, 256]),B,D,N
        local_x3 = torch.cat([x3, local_feature_3], dim = 1) # torch.Size([32, 1024, 256]),B,D,N

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 1024]),(B,D,N)
        x1 = self.global_sa1(global_feature, global_xyz)
        x2 = self.global_sa2(global_feature, global_xyz)
        x3 = self.global_sa3(global_feature, global_xyz)
        x4 = self.global_sa4(global_feature, global_xyz)
        global_x = torch.cat([x1, x2, x3, x4, global_feature], dim=1) # torch.Size([32, 128*5, 2048]),(B,D,N)
        global_x = self.fuse_conv(global_x) # torch.Size([32, 1024, 2048]),(B,D,N)

        # x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3) # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        # x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        # x = self.fp1(xyz, local_xyz1, torch.cat([global_feature,cls_label_feature],dim = 1), x1)  # torch.Size([32, 128+64+256->128, 2048]),(B,D,N)

        # global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D,N)
        local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1)
        local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)
        local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)
        x = torch.cat([local_x1_max, local_x2_max, local_x3_max], dim=1) # torch.Size([32, 256+512+1024]),(B,D)
        x = x.unsqueeze(-1).repeat(1, 1, N) # torch.Size(32, 256+512+1024,2048),(B,D,N)
        x = torch.cat([global_x, x, cls_label_feature], dim=1) # torch.Size([32, 1024 + (256+512+1024) + 64],2048),(B,D,N)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.linear3(x)
        return x

class Pct_partseg10(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg10, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
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

        self.fp1 = PointNetFeaturePropagation(in_channel=256 + 128 + 64, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        #
        # self.local_delta1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.local_delta2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.local_delta3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.global_delta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # self.linear1 = nn.Conv1d(1024+128*2+128*2+256*2+512*2 + 64, 512, 1)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(0.5)
        # self.linear2 = nn.Conv1d(512, 256, 1)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.linear3 = nn.Conv1d(256, output_channels, 1)

        self.conv = nn.Conv1d(128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_xyz = xyz.permute(0, 2, 1) # B, 14, N
        global_feature = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        global_feature_pw = F.relu(self.bn4(self.conv4(global_feature)))  # B, 1024, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature1 = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_xyz_feature1 = self.local_pos_xyz1(local_xyz1.permute(0,2,1)) # torch.Size([32, 128, 512]),B,D,N
        x1 = self.local_scale1(local_feature_1, local_xyz_feature1) # torch.Size([32, 128, 512]),B,D,N
        local_x1 = torch.cat([x1,local_feature_1], dim = 1) # torch.Size([32, 256, 512]),B,D,N

        local_xyz2, local_feature2 = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature_1.permute(0, 2, 1))
        local_feature_2 = self.gather_local_2(local_feature2) # torch.Size([32, 256, 256]),B,D,N
        local_xyz_feature_2 = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1)) # torch.Size([32, 256, 256]),B,D,N
        x2 = self.local_scale2(local_feature_2, local_xyz_feature_2) # torch.Size([32, 256, 256]),B,D,N
        local_x2 = torch.cat([x2, local_feature_2], dim= 1) # torch.Size([32, 512, 256]),B,D,N


        local_xyz3, local_feature3 = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature_2.permute(0, 2, 1))
        local_feature_3 = self.gather_local_3(local_feature3) # torch.Size([32, 512, 128]),B,D,N
        local_xyz_feature_3 = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1)) # torch.Size([32, 512, 128]),B,D,N
        x3 = self.local_scale3(local_feature_3, local_xyz_feature_3) # torch.Size([32, 512, 128]),B,D,N
        local_x3 = torch.cat([x3, local_feature_3], dim = 1) # torch.Size([32, 1024, 128]),B,D,N

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 128, 2048]),(B,D,N)
        x = self.global_sa(global_feature, global_xyz) # torch.Size([32, 128, 2048]),(B,D,N)
        global_x = torch.cat([x, global_feature], dim=1) # torch.Size([32, 256, 2048]),(B,D,N)

        # data, [B, N, C]
        # data, [B, S, C]
        # data, [B, D1, N]
        # data, [B, D2, S]
        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3) # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_feature,cls_label_feature],dim = 1), x1)  # torch.Size([32, 128+64+256->128, 2048]),(B,D,N)

        # global_x_max = F.adaptive_max_pool1d(global_x, 1).view(batch_size, -1)  # torch.Size([32, 128*2]),(B,D,N)
        # local_x1_max = F.adaptive_max_pool1d(local_x1, 1).view(batch_size, -1)
        # local_x2_max = F.adaptive_max_pool1d(local_x2, 1).view(batch_size, -1)
        # local_x3_max = F.adaptive_max_pool1d(local_x3, 1).view(batch_size, -1)
        # x = torch.cat([global_x_max, local_x1_max, local_x2_max, local_x3_max], dim=1) # torch.Size([32, 128*2+128*2+256*2+512*2]),(B,D)
        # x = x.unsqueeze(-1).repeat(1, 1, N) # torch.Size([32, 128*2+128*2+256*2+512*2],2048),(B,D,N)
        # x = torch.cat([global_feature_pw, x, cls_label_feature], dim=1) # torch.Size([32, 1024+128*2+128*2+256*2+512*2 + 64],2048),(B,D,N)
        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.linear3(x)
        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x

class Pct_partseg11(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg11, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024 + 128, 512 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        global_x_wise = F.relu(self.bn3(self.conv3(x))) # B, 256, N
        global_x_wise = F.relu(self.bn4(self.conv4(global_x_wise))) # B, 1024, N
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)
        
        x = torch.cat([global_x_wise,x], dim=1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 1024+128, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg12(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg12, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv = nn.Conv1d(128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=1028, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 512]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 1024]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)

        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x

class Pct_partseg13(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg13, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        self.local_scale11 = SA_Layer(channels=128)
        self.local_scale21 = SA_Layer(channels=256)
        self.local_scale31 = SA_Layer(channels=512)
        self.local_scale12 = SA_Layer(channels=128)
        self.local_scale22 = SA_Layer(channels=256)
        self.local_scale32 = SA_Layer(channels=512)
        self.local_scale13 = SA_Layer(channels=128)
        self.local_scale23 = SA_Layer(channels=256)
        self.local_scale33 = SA_Layer(channels=512)

        self.fp1 = PointNetFeaturePropagation(in_channel= 64+64+128*5, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=128*5+256*5, mlp=[128*5])
        self.fp3 = PointNetFeaturePropagation(in_channel=256*5+512*5, mlp=[256*5])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv = nn.Conv1d(128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        x11 = self.local_scale11(local_feature_1, local_xyz1_feature)
        x12 = self.local_scale12(local_feature_1, local_xyz1_feature)
        x13 = self.local_scale13(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,x11,x12,x13,local_feature_1], dim = 1)# torch.Size([32, 128*5, 512]),B,D,N

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        x21 = self.local_scale21(local_feature_2, local_xyz2_feature)
        x22 = self.local_scale22(local_feature_2, local_xyz2_feature)
        x23 = self.local_scale23(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, x21, x22,x23,local_feature_2], dim= 1) # torch.Size([32, 256*5, 256]),B,D,N

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        x31 = self.local_scale31(local_feature_3, local_xyz3_feature)
        x32 = self.local_scale32(local_feature_3, local_xyz3_feature)
        x33 = self.local_scale33(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, x31, x32, x33,local_feature_3], dim = 1) # torch.Size([32, 512*5, 128]),B,D,N

        # global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 256, 1024]),(B,D,N)
        # x = self.global_sa(global_feature, global_xyz)
        # global_x = x - global_feature  # Back-projection signal
        # global_x = global_feature + self.global_delta * global_x  # Feedback
        # global_x = torch.cat([global_x, global_feature], dim=1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 256*5+512*5->256*5, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 128*5+256*5->128*5, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+128*5->128, 2048]),(B,D,N)

        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x

class Pct_partseg14(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg14, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        self.local_scale11 = SA_Layer(channels=128)
        self.local_scale21 = SA_Layer(channels=256)
        self.local_scale31 = SA_Layer(channels=512)

        self.fp1 = PointNetFeaturePropagation(in_channel= 64+64+128*3, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=128*3+256*3, mlp=[128*3])
        self.fp3 = PointNetFeaturePropagation(in_channel=256*3+512*3, mlp=[256*3])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        # self.linear1 = nn.Conv1d((128+256+512)*2 + 64, 512 , 1, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(0.5)
        # self.linear2 = nn.Conv1d(512, 256, 1)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.linear3 = nn.Conv1d(256, output_channels , 1)
        self.conv = nn.Conv1d(128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 512]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        x11 = self.local_scale11(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,x11,local_feature_1], dim = 1)# torch.Size([32, 128*3, 512]),B,D,N

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        x21 = self.local_scale21(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, x21, local_feature_2], dim= 1) # torch.Size([32, 256*3, 256]),B,D,N

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        x31 = self.local_scale31(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, x31, local_feature_3], dim = 1) # torch.Size([32, 512*3, 128]),B,D,N

        # global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 256, 1024]),(B,D,N)
        # x = self.global_sa(global_feature, global_xyz)
        # global_x = x - global_feature  # Back-projection signal
        # global_x = global_feature + self.global_delta * global_x  # Feedback
        # global_x = torch.cat([global_x, global_feature], dim=1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 256*3+512*3->256*3, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 128*3+256*3->128*3, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+128*3->128, 2048]),(B,D,N)

        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x


class Pct_partseg15(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg15, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        # self.linear1 = nn.Conv1d((128+256+512)*2 + 64, 512 , 1, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(0.5)
        # self.linear2 = nn.Conv1d(512, 256, 1)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.linear3 = nn.Conv1d(256, output_channels , 1)
        self.conv = nn.Conv1d(128, 128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Conv1d(128, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        # global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 256, 1024]),(B,D,N)
        # x = self.global_sa(global_feature, global_xyz)
        # global_x = x - global_feature  # Back-projection signal
        # global_x = global_feature + self.global_delta * global_x  # Feedback
        # global_x = torch.cat([global_x, global_feature], dim=1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)

        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)
        x = self.drop(x)
        x = self.out(x)
        return x


class Pct_partseg16(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg16, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels=128)
        self.local_scale2 = SA_Layer(channels=256)
        self.local_scale3 = SA_Layer(channels=512)

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024 + 128, 512, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels, 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x)  # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()  # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # B, 64, N
        global_x = x
        global_x_wise = F.relu(self.bn3(self.conv3(x)))  # B, 128, N
        global_x_wise = F.relu(self.bn4(self.conv4(global_x_wise)))  # B, 1024, N
        x = x.permute(0, 2, 1)  # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature)  # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0, 2, 1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1, local_feature_1], dim=1)

        local_feature = local_feature_1.permute(0, 2, 1)  # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1,
                                                     points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature)  # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim=1)

        local_feature = local_feature_2.permute(0, 2, 1)  # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2,
                                                     points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature)  # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim=1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)

        global_x = F.adaptive_max_pool1d(global_x_wise, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x = torch.cat([global_x, x], dim=1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024+128, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg17(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg17, self).__init__()
        self.conv1 = nn.Conv1d(14, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_2 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_3 = Local_op(in_channels=512, out_channels=512)

        self.local_pos_xyz1 = nn.Conv1d(14, 128, 1)
        self.local_pos_xyz2 = nn.Conv1d(14, 256, 1)
        self.local_pos_xyz3 = nn.Conv1d(14, 512, 1)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.attention_conv = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024*2, 512 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 256]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 512]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)
        x_att = self.attention_conv(x) # torch.Size([32, 128->1024, 2048]),(B,D,N)
        x_max = F.adaptive_max_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # torch.Size([32, 1024, 2048]),(B,D,N)
        x = torch.cat([x_att,x_max],dim = 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024+1024, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg18(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg18, self).__init__()
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

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 512, mlp=[256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 1024, mlp=[512])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[1024])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.attention_conv = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024*3, 512 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=1024, radius=0.2, nsample=16, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=16, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=16, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 128, 14]) # new_feature: d,n,s,d torch.Size([32, 128, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->1024, 1024]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->512, 1024]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->256, 2048]),(B,D,N)
        x_att = self.attention_conv(x) # torch.Size([32, 256->1024, 2048]),(B,D,N)
        x_max = F.adaptive_max_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # torch.Size([32, 1024, 2048]),(B,D,N)
        x_avg = F.adaptive_avg_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048]),(B,D,N)
        x = torch.cat([x_att,x_max,x_avg],dim = 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024*3, 2048])
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg19(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg19, self).__init__()
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

        self.fp1 = PointNetFeaturePropagation(in_channel=64 + 64 + 256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[512])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.attention_conv = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024*3, 512 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=1024, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 1024, 14]) # new_feature: d,n,s,d torch.Size([32, 1024, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 516, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 256, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 512]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 1024]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)
        x_att = self.attention_conv(x) # torch.Size([32, 128->1024, 2048]),(B,D,N)
        x_max = F.adaptive_max_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # torch.Size([32, 1024, 2048]),(B,D,N)
        x_avg = F.adaptive_avg_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048]),(B,D,N)
        x = torch.cat([x_att,x_max,x_avg],dim = 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024+1024, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg20(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg20, self).__init__()
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

        self.fp1 = PointNetFeaturePropagation(in_channel=64+64+128*2, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=64+64+256*2, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=64+64+512*2, mlp=[512])
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        #
        # self.attention_conv = nn.Sequential(nn.Conv1d(128+256+512, 1024, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(1024),
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d((128+256+512)*3, 128+256+512 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(128+256+512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(128+256+512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=1024, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 1024, 14]) # new_feature: d,n,s,d torch.Size([32, 1024, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 516, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 256, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp3(xyz, local_xyz3, torch.cat([global_x, cls_label_feature], dim=1), local_x3)  # torch.Size([32, 64+64+512*2->512, 2048]),(B,D,N)
        x1 = self.fp2(xyz, local_xyz2, torch.cat([global_x, cls_label_feature], dim=1), local_x2)  # torch.Size([32, 64+64+256*2->256, 2048]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1), local_x1)  # torch.Size([32, 64+64+128*2->128, 2048]),(B,D,N)
        x_att = torch.cat([x, x1, x2], dim=1)  # torch.Size([32, 128+256+512, 2048]),(B,D,N)
        # x_att = self.attention_conv(torch.cat([x,x1,x2],dim = 1)) # torch.Size([32, 128+256+512->1024, 2048]),(B,D,N)
        x_max = F.adaptive_max_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # torch.Size([32, 128+256+512, 2048]),(B,D,N)
        x_avg = F.adaptive_avg_pool1d(x_att, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 128+256+512, 2048]),(B,D,N)
        x = torch.cat([x_att,x_max,x_avg],dim = 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024*3, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg21(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg21, self).__init__()
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

        self.fp1 = PointNetFeaturePropagation(in_channel=64+64+256, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+512, mlp=[256])
        self.fp3 = PointNetFeaturePropagation(in_channel=512+1024, mlp=[512])
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        #
        # self.attention_conv = nn.Sequential(nn.Conv1d(128+256+512, 1024, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(1024),
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024, 256 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(256, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, local_feature = sample_and_group(npoint=1024, radius=0.2, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 1024, 14]) # new_feature: d,n,s,d torch.Size([32, 1024, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, local_feature = sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 516, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, local_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 256, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])
        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->512, 512]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512->256, 1024]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x, cls_label_feature], dim=1),x1)  # torch.Size([32, 64+64+256->128, 2048]),(B,D,N)

        x2_max = F.adaptive_max_pool1d(x2, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        x1_max = F.adaptive_max_pool1d(x1, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        x = torch.cat([x, x_max, x1_max, x2_max], dim=1)  # torch.Size([32, 128+128+256+512, 2048]),(B,D,N)
        # x_att = self.attention_conv(torch.cat([x,x1,x2],dim = 1)) # torch.Size([32, 128+256+512->1024, 2048]),(B,D,N)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg22(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg22, self).__init__()
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
        self.global_pos_xyz = nn.Conv1d(14, 64, 1, bias=False)

        self.local_scale1 = SA_Layer(channels = 128)
        self.local_scale2 = SA_Layer(channels = 256)
        self.local_scale3 = SA_Layer(channels = 512)
        self.global_sa = SA_Layer(channels=64)

        self.fp1 = PointNetFeaturePropagation(in_channel=64+128+256+512+1024, mlp=[1024])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+512+1024, mlp=[])
        self.fp3 = PointNetFeaturePropagation(in_channel=512+1024, mlp=[])
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024*2, 512 , 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        global_xyz = x
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size() # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # B, 64, N
        global_x = x
        x =x.permute(0, 2, 1) # B, N, 64
        local_xyz1, _, local_feature = sample_and_group(npoint=1024, nsample=32, xyz=xyz, points=x)
        # local_xyz: torch.Size([32, 1024,14]) # new_feature: d,n,s,d torch.Size([32, 1024, 32, 128(64+64)])
        local_feature_1 = self.gather_local_1(local_feature) # torch.Size([32, 128, 1024]),B,D,N
        local_xyz1_feature = self.local_pos_xyz1(local_xyz1.permute(0,2,1))
        x1 = self.local_scale1(local_feature_1, local_xyz1_feature)
        local_x1 = torch.cat([x1,local_feature_1], dim = 1)

        local_feature = local_feature_1.permute(0, 2, 1) # torch.Size([32, 512, 128]),B,N,D
        local_xyz2, _, local_feature = sample_and_group(npoint=512, nsample=32, xyz=local_xyz1, points=local_feature)
        # new_xyz: torch.Size([32, 512, 14]) # new_feature: d,n,s,d torch.Size([32, 516, 32, 256(128+128)])
        local_feature_2 = self.gather_local_2(local_feature) # feature_1: torch.Size([32, 256, 512]),B,D,N
        local_xyz2_feature = self.local_pos_xyz2(local_xyz2.permute(0, 2, 1))
        x2 = self.local_scale2(local_feature_2, local_xyz2_feature)
        local_x2 = torch.cat([x2, local_feature_2], dim= 1)

        local_feature = local_feature_2.permute(0, 2, 1) # torch.Size([32, 256, 256]),B,N,D
        local_xyz3, _, local_feature = sample_and_group(npoint=256, nsample=32, xyz=local_xyz2, points=local_feature)
        # new_xyz: torch.Size([32, 256, 14]) # new_feature: d,n,s,d torch.Size([32, 256, 32, 512(256+256)])
        local_feature_3 = self.gather_local_3(local_feature) # feature_1: torch.Size([32, 512, 256]),B,D,N
        local_xyz3_feature = self.local_pos_xyz3(local_xyz3.permute(0, 2, 1))
        x3 = self.local_scale3(local_feature_3, local_xyz3_feature)
        local_x3 = torch.cat([x3, local_feature_3], dim = 1)

        global_xyz = self.global_pos_xyz(global_xyz)  # torch.Size([32, 64, 2048]),(B,D,N)
        global_feature = self.global_sa(global_x, global_xyz)
        global_x = torch.cat([global_x, global_feature], dim=1)# torch.Size([32, 128, 2048]),(B,D,N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])
        x2 = self.fp3(local_xyz2, local_xyz3, local_x2, local_x3)  # torch.Size([32, 512+1024->, 512]),(B,D,N)
        x1 = self.fp2(local_xyz1, local_xyz2, local_x1, x2)  # torch.Size([32, 256+512+1024->, 1024]),(B,D,N)
        x = self.fp1(xyz, local_xyz1, torch.cat([global_x,cls_label_feature],dim = 1), x1)  # torch.Size([32, 64+128+256+512+1024->1024, 2048]),(B,D,N)

        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        x = torch.cat([x, x_max], dim=1)  # torch.Size([32, 1024*2, 2048]),(B,D,N)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # torch.Size([32, 1024, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # torch.Size([32, 256, 2048])
        x = self.linear3(x)
        return x

class Pct_partseg23(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg23, self).__init__()
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

        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels=64+64+128+256+512+1024, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
        )

        self.fp = PointNetFeaturePropagation(in_channel=64+64+128+256+512+1024, out_channel=64+64+128+256+512+1024)
        self.fp1 = PointNetFeaturePropagation(in_channel=128+256+512+1024, out_channel=128+256+512+1024)
        self.fp2 = PointNetFeaturePropagation(in_channel=256+512+1024, out_channel=256+512+1024)
        self.fp3 = PointNetFeaturePropagation(in_channel=512+1024, out_channel=512+1024)
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),nn.BatchNorm1d(64),nn.ReLU())

        self.linear1 = nn.Conv1d(1024*3, 512, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) + x1 # B, 64, N
        global_fea = x2
        sample_xyz1, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=1024, nsample=16, xyz=xyz, point=x2.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz1.view(B, 1024, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz1.view(B, 1024, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 512, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 1024, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_sem1 = F.relu(self.net1(local_sem1)) + local_sem1
        local_att1 = self.local_scale1(local_sem1,local_geo1)  # 24,128,512
        local_fea1 = torch.cat([local_att1,local_sem1], dim = 1)

        sample_xyz2, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=16, xyz=sample_xyz1, point=local_att1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz2.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz2.view(B, 512, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist),dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 256, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_sem2 = F.relu(self.net2(local_sem2)) + local_sem2
        local_att2 = self.local_scale2(local_sem2,local_geo2)  # 24,256,256
        # local_fea2 = torch.cat([local_att2, local_sem2], dim=1)

        sample_xyz3, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=16, xyz=sample_xyz2, point=local_att2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz3.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz3.view(B, 256, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 128, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_sem3 = F.relu(self.net3(local_sem3)) + local_sem3
        local_att3 = self.local_scale3(local_sem3,local_geo3)  # 24,512,128
        # local_fea3 = torch.cat([local_att3, local_sem3], dim=1)

        sample_xyz4, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=128, nsample=16, xyz=sample_xyz3, point=local_att3.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz4.view(B, 128, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo4 = torch.cat((sample_xyz4.view(B, 128, 1, -1).repeat(1, 1, 16, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 64, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha4 * grouped_points + self.affine_beta4
        sem_fea4 = torch.cat((sample_points.view(B, 128, 1, -1).repeat(1, 1, 16, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo4 = self.gather_geo_4(geo4)  # 24,1024,64
        local_sem4 = self.gather_fea_4(sem_fea4) # 24,1024,64
        local_sem4 = F.relu(self.net4(local_sem4)) + local_sem4
        local_att4 = self.local_scale4(local_sem4,local_geo4) # 24,1024,64
        # local_fea4 = torch.cat([local_att4, local_sem4], dim=1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])
        x3 = self.fp3(sample_xyz3, sample_xyz4, local_att3, local_att4)  # torch.Size([32, 512+1024->1024, 256]),(B,D,N)
        x2 = self.fp2(sample_xyz2, sample_xyz3, local_att2, x3)  # torch.Size([32, 256+1024->1024, 512]),(B,D,N)
        x1 = self.fp1(sample_xyz1, sample_xyz2, local_att1, x2) # torch.Size([32, 128+1024->1024, 1024]),(B,D,N)
        x = self.fp(xyz, sample_xyz1, torch.cat([global_fea, cls_label_feature], dim=1),x1) # torch.Size([32, 64+64+1024->1024, 1024]),(B,D,N)

        x = F.relu(self.fuse(x))
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x = torch.cat([x, x_max, x_avg], dim=1)  # torch.Size([32, 1024*3, 2048]),(B,D,N)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_partseg24(nn.Module):
    def __init__(self, output_channels=50):
        super(Pct_partseg24, self).__init__()
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

        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels=64+64+(128+256+512)*2, out_channels=512*2, kernel_size=1, bias=False),
            nn.BatchNorm1d(512*2),
        )

        self.fp = PointNetFeaturePropagation(in_channel=64+64+(128+256+512)*2, out_channel=64+64+(128+256+512)*2)
        self.fp1 = PointNetFeaturePropagation(in_channel=(128+256+512)*2, out_channel=(128+256+512)*2)
        self.fp2 = PointNetFeaturePropagation(in_channel=(256+512)*2, out_channel=(256+512)*2)
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),nn.BatchNorm1d(64),nn.ReLU())

        self.linear1 = nn.Conv1d(512*2*3, 512, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Conv1d(256, output_channels , 1)

    def forward(self, x, cls_label):
        B,C,N = x.shape #C:3
        xyz = x.permute(0, 2, 1)
        x = geometric_point_descriptor(x) # B, 3, N => # B, 14, N
        batch_size, _, _ = x.size() # B, D, N
        x1 = F.relu(self.bn1(self.conv1(x))) # B, 64, N
        x2 = F.relu(self.bn2(self.conv2(x1))) + x1 # B, 64, N
        global_fea = x2
        sample_xyz1, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=1024, nsample=32, xyz=xyz, point=x2.permute(0, 2, 1))
        #sample_xyz: [24, 512, 3] grouped_xyz: [24, 512, 24, 3] sample_points: [24, 512, 128] grouped_points: [24, 512, 24, 64]
        grouped_xyz_norm = grouped_xyz - sample_xyz1.view(B, 1024, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo1 = torch.cat((sample_xyz1.view(B, 1024, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 512, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha1 * grouped_points + self.affine_beta1
        sem_fea1 = torch.cat((sample_points.view(B, 1024, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,512,24,128
        local_geo1 = self.gather_geo_1(geo1)  # 24,10,512
        local_sem1 = self.gather_fea_1(sem_fea1) # 24,128,512
        local_sem1 = F.relu(self.net1(local_sem1)) + local_sem1
        local_att1 = self.local_scale1(local_sem1,local_geo1)  # 24,128,512
        local_fea1 = torch.cat([local_att1,local_sem1], dim = 1)

        sample_xyz2, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=512, nsample=32, xyz=sample_xyz1, point=local_sem1.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz2.view(B, 512, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo2 = torch.cat((sample_xyz2.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist),dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 256, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha2 * grouped_points + self.affine_beta2
        sem_fea2 = torch.cat((sample_points.view(B, 512, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,256,16,256
        local_geo2 = self.gather_geo_2(geo2)  # 24,64,256
        local_sem2 = self.gather_fea_2(sem_fea2) # 24,256,256
        local_sem2 = F.relu(self.net2(local_sem2)) + local_sem2
        local_att2 = self.local_scale2(local_sem2,local_geo2)  # 24,256,256
        local_fea2 = torch.cat([local_att2, local_sem2], dim=1)

        sample_xyz3, grouped_xyz, sample_points, grouped_points = sample_and_group(npoint=256, nsample=32, xyz=sample_xyz2, point=local_sem2.permute(0, 2, 1))
        grouped_xyz_norm = grouped_xyz - sample_xyz3.view(B, 256, 1, C)  # [B, npoint, nsample, 3]
        dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
        geo3 = torch.cat((sample_xyz3.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_xyz, grouped_xyz_norm, dist), dim=-1)
        # grouped_points_norm = grouped_points - sample_points.view(B, 128, 1, -1)  # [B, npoint, nsample, D]
        mean = sample_points.unsqueeze(dim=-2)
        std = torch.std(grouped_points - mean)
        grouped_points = (grouped_points - mean) / (std + 1e-9)
        grouped_points = self.affine_alpha3 * grouped_points + self.affine_beta3
        sem_fea3 = torch.cat((sample_points.view(B, 256, 1, -1).repeat(1, 1, 32, 1), grouped_points), dim=-1) # 24,1024,8,256
        local_geo3 = self.gather_geo_3(geo3)  # 24,512,128
        local_sem3 = self.gather_fea_3(sem_fea3) # 24,512,128
        local_sem3 = F.relu(self.net3(local_sem3)) + local_sem3
        local_att3 = self.local_scale3(local_sem3,local_geo3)  # 24,512,128
        local_fea3 = torch.cat([local_att3, local_sem3], dim=1)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # torch.Size([32, 64, 2048])

        x2 = self.fp2(sample_xyz2, sample_xyz3, local_fea2, local_fea3)  # torch.Size([32, 256+1024->1024, 512]),(B,D,N)
        x1 = self.fp1(sample_xyz1, sample_xyz2, local_fea1, x2) # torch.Size([32, 128+1024->1024, 1024]),(B,D,N)
        x = self.fp(xyz, sample_xyz1, torch.cat([global_fea, cls_label_feature], dim=1),x1) # torch.Size([32, 64+64+1024->1024, 1024]),(B,D,N)

        x = F.relu(self.fuse(x))
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x = torch.cat([x, x_max, x_avg], dim=1)  # torch.Size([32, 1024*3, 2048]),(B,D,N)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        x = self.linear3(x)
        return x

