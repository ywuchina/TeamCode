import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group, square_distance, index_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, D1, N]
            points2: input points data, [B, D2, S]
        Return:
            new_points: upsampled points data, [B, D1+s, N]
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

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class Point_Transformer_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(Point_Transformer_partseg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = Sa_Layer(128)
        self.sa2 = Sa_Layer(128)
        self.sa3 = Sa_Layer(128)
        self.sa4 = Sa_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1) # (B,512,N)
        x = self.conv_fuse(x) # (B,1024,N)

        x_avg = F.adaptive_max_pool1d(x,  1) # (B,1024,1)
        x_max = F.adaptive_avg_pool1d(x,  1) # (B,1024,1)

        # print('x_max=>', x_max.shape)
        # print('x_avg=>', x_avg.shape)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # (B,1024,N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # (B,1024,N)

        cls_label_one_hot = cls_label.view(batch_size,16,1) # (B,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # (B,64,N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), dim = 1) # (B,1024+1024+64,N)

        x = torch.cat((x, x_global_feature), dim = 1) # (B,1024*3 +64,N)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.dp2(x)
        x = self.convs3(x) # (B,50,N)
        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
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
        # print('x = F.relu(self.bn2(self.conv2(x)))=>',x.shape)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # torch.Size([16384, 128])
        # print('x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)=>', x.shape)
        x = x.reshape(b, n, -1).permute(0, 2, 1) # torch.Size([32, 128, 512]),(B,D,N)
        # print('x = x.reshape(b, n, -1).permute(0, 2, 1)=>', x.shape)
        return x

class Pct_partseg1(nn.Module):
    def __init__(self, args, part_num=50):
        super(Pct_partseg1, self).__init__()
        self.args = args
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024 * 2 + 64, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        # print('x = F.relu(self.bn1(self.conv1(x)))',x.shape)

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 64])
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.1, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])

        feature_0 = self.gather_local_0(new_feature) # torch.Size([32, 128, 512])
        # print("feature_0 = self.gather_local_0(new_feature)=>", feature_0.shape)

        feature = feature_0.permute(0, 2, 1) # torch.Size([32, 512, 128])
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 32, 256(128+128)])

        feature_1 = self.gather_local_1(new_feature) # torch.Size([32, 256, 256])
        # print("feature_1 = self.gather_local_1(new_feature)", feature_1.shape)

        x = self.pt_last(feature_1) # torch.Size([32, 1024, 256]),(B,D,N)

        x = torch.cat([x, feature_1], dim=1)  # torch.Size([32, 1024+256(1280), 256]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)

        x_adaptive_max_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])
        x_adaptive_avg_feature = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        x = torch.cat((x_adaptive_max_feature, x_adaptive_avg_feature, cls_label_feature), dim = 1) # (32,1024+1024+64,N)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.dp2(x)
        x = self.linear3(x) # torch.Size([32, 50, 2048])

        return x

class Pct_partseg2(nn.Module):
    def __init__(self, args, part_num=50):
        super(Pct_partseg2, self).__init__()
        self.args = args
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024 * 2 + 64, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        # print('x = F.relu(self.bn1(self.conv1(x)))',x.shape)

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 64])
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.1, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128(64+64)])

        feature_0 = self.gather_local_0(new_feature) # torch.Size([32, 128, 512])
        # print("feature_0 = self.gather_local_0(new_feature)=>", feature_0.shape)

        feature = feature_0.permute(0, 2, 1) # torch.Size([32, 512, 128])
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 32, 256(128+128)])

        feature_1 = self.gather_local_1(new_feature) # torch.Size([32, 256, 256])
        # print("feature_1 = self.gather_local_1(new_feature)", feature_1.shape)

        x = self.pt_last(feature_1) # torch.Size([32, 1024, 256]),(B,D,N)

        # x = torch.cat([x, feature_1], dim=1)  # torch.Size([32, 1024+256(1280), 256]),(B,D,N)
        # x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)

        x_adaptive_max_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])
        x_adaptive_avg_feature = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        x = torch.cat((x_adaptive_max_feature, x_adaptive_avg_feature, cls_label_feature), dim = 1) # (32,1024+1024+64,N)

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
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        self.sa1 = Sa_Layer(256)
        self.sa2 = Sa_Layer(256)
        self.sa3 = Sa_Layer(256)
        self.sa4 = Sa_Layer(256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1) # (B,1024,N)
        # x = self.conv_fuse(x) # (B,1024,N)

        x_avg = F.adaptive_max_pool1d(x,  1) # (B,1024,1)
        x_max = F.adaptive_avg_pool1d(x,  1) # (B,1024,1)

        # print('x_max=>', x_max.shape)
        # print('x_avg=>', x_avg.shape)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # (B,1024,N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # (B,1024,N)

        cls_label_one_hot = cls_label.view(batch_size,16,1) # (B,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # (B,64,N)

        x = torch.cat((x,x_max_feature, x_avg_feature, cls_label_feature), dim = 1) # (B,1024*3 +64,N)

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.dp2(x)
        x = self.convs3(x) # (B,50,N)
        return x

class Pct_partseg4(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg4, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        self.sa1 = Sa_Layer(256)
        self.sa2 = Sa_Layer(256)
        self.sa3 = Sa_Layer(256)
        self.sa4 = Sa_Layer(256)
        self.sa5 = Sa_Layer(256)
        self.sa6 = Sa_Layer(256)
        self.sa7 = Sa_Layer(256)
        self.sa8 = Sa_Layer(256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(2048 * 2 + 64, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.2)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.2)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = self.sa5(x4)
        x6 = self.sa6(x5)
        x7 = self.sa7(x6)
        x8 = self.sa8(x7)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1) # (B,2048,N)
        # x = self.conv_fuse(x) # (B,1024,N)

        # x_avg = F.adaptive_max_pool1d(x,  1) # (B,2048,1)
        x_max = F.adaptive_avg_pool1d(x,  1) # (B,2048,1)

        # print('x_max=>', x_max.shape)
        # print('x_avg=>', x_avg.shape)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # (B,2048,N)
        # x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # (B,2048,N)

        cls_label_one_hot = cls_label.view(batch_size,16,1) # (B,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # (B,64,N)

        x = torch.cat((x,x_max_feature, cls_label_feature), dim = 1) # (B,2048*2 +64,N)

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.dp2(x)
        x = self.convs3(x) # (B,50,N)
        return x

class Pct_partseg5(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg5, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(1024 + 64, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout()
        self.linear3 = nn.Conv1d(256, part_num, kernel_size=1)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        # print('x = F.relu(self.bn1(self.conv1(x)))',x.shape)

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 128, 2048])
        x = x.permute(0, 2, 1) # torch.Size([32, 2048, 128])
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.1, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 256(128+128)])

        # feature_0 = self.gather_local_0(new_feature) # torch.Size([32, 128, 512])
        # # print("feature_0 = self.gather_local_0(new_feature)=>", feature_0.shape)
        #
        # feature = feature_0.permute(0, 2, 1) # torch.Size([32, 512, 128])
        # new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 32, 256(128+128)])

        feature_1 = self.gather_local_1(new_feature) # torch.Size([32, 256, 512]),(b,d,n)
        # print("feature_1 = self.gather_local_1(new_feature)", feature_1.shape)

        x = self.pt_last(feature_1) # torch.Size([32, 1024, 512]),(B,D,N)

        x = torch.cat([x, feature_1], dim=1)  # torch.Size([32, 1024+256(1280), 512]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 512]),(B,D,N)

        x_adaptive_max_feature = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])
        # x_adaptive_avg_feature = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # torch.Size([32, 1024, 2048])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # torch.Size([32, 64, 2048])

        x = torch.cat((x_adaptive_max_feature, cls_label_feature), dim = 1) # (32,1024+64,N)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512, 2048])
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256, 2048])
        x = self.dp2(x)
        x = self.linear3(x) # torch.Size([32, 50, 2048])

        return x

class Pct_partseg6(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg6, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = Sa_Layer(128)
        self.sa2 = Sa_Layer(128)
        self.sa3 = Sa_Layer(128)
        self.sa4 = Sa_Layer(128)
        self.sa5 = Sa_Layer(128)
        self.sa6 = Sa_Layer(128)
        self.sa7 = Sa_Layer(128)
        self.sa8 = Sa_Layer(128)

        # self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(1024),
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 2 + 64, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.2)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.2)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = self.sa5(x4)
        x6 = self.sa6(x5)
        x7 = self.sa7(x6)
        x8 = self.sa8(x7)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1) # (B,1024,N)

        # x_avg = F.adaptive_max_pool1d(x,  1) # (B,2048,1)
        x_max = F.adaptive_avg_pool1d(x,  1) # (B,1024,1)

        # print('x_max=>', x_max.shape)
        # print('x_avg=>', x_avg.shape)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # (B,1024,N)
        # x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # (B,1024,N)

        cls_label_one_hot = cls_label.view(batch_size,16,1) # (B,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # (B,64,N)

        x = torch.cat((x,x_max_feature, cls_label_feature), dim = 1) # (B,1024*2 +64,N)

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.dp2(x)
        x = self.convs3(x) # (B,50,N)
        return x

class Pct_partseg7(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_partseg7, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        # self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.sa1 = Sa_Layer(128)
        self.sa2 = Sa_Layer(128)
        self.sa3 = Sa_Layer(128)
        self.sa4 = Sa_Layer(128)
        # self.sa5 = Sa_Layer(128)
        # self.sa6 = Sa_Layer(128)
        # self.sa7 = Sa_Layer(128)
        # self.sa8 = Sa_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fp1 = PointNetFeaturePropagation(in_channel=64+1024+256, mlp=[1024])

        self.convs1 = nn.Conv1d(1024, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        xyz = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 2048])
        x = self.relu(self.bn2(self.conv2(x))) # torch.Size([32, 128, 2048])

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        # x5 = self.sa5(x4)
        # x6 = self.sa6(x5)
        # x7 = self.sa7(x6)
        # x8 = self.sa8(x7)
        # x_tr = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1) # (B,1024,N)
        x_tr = torch.cat((x1, x2, x3, x4), dim=1)  # (B,512,N)
        x_tr = self.conv_fuse(x_tr) # (B,1024,N)
        x = x.permute(0, 2, 1)

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 256(128+128)])
        feature_0 = self.gather_local_1(new_feature) # torch.Size([32, 256, 512]),(B,D,S)
        # print("feature_0 = self.gather_local_0(new_feature)=>", feature_0.shape)
        # feature = feature_0.permute(0, 2, 1) # torch.Size([32, 512, 128]),(B,S,D)
        # new_xyz, new_feature = sample_and_group(npoint=128, radius=0.2, nsample=64, xyz=new_xyz, points=feature)
        # # new_xyz: torch.Size([32, 256, 3])
        # # new_feature: d,n,s,d torch.Size([32, 128, 64, 256(128+128)])
        # feature_1 = self.gather_local_1(new_feature) # torch.Size([32, 256, 128])

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)  # (B,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # (B,64,N)

        x = self.fp1(xyz, new_xyz, torch.cat([cls_label_feature, x_tr], 1), feature_0)
        # print('x=>',x.shape)

        # x_max = F.adaptive_avg_pool1d(x,  1) # (B,1024,1)
        # x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N) # (B,1024,N)

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        # x = self.dp2(x)
        x = self.convs3(x) # (B,50,N)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = Sa_Layer(channels)
        self.sa2 = Sa_Layer(channels)
        self.sa3 = Sa_Layer(channels)
        self.sa4 = Sa_Layer(channels)

    def forward(self, x):
        batch_size, _, N = x.size() # torch.Size([32, 256, 256]),(B,D,N)

        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 256, 256])
        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 256, 256])
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1) # torch.Size([32, 1024, 256])

        return x

class Sa_Layer(nn.Module):
    def __init__(self, channels):
        super(Sa_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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