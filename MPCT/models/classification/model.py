import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group

class Point_Transformer_cls(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_cls, self).__init__()

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

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        # print (x.size())
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # 针对gather_local_0
        b, n, s, d = x.size() # torch.Size([32, 512, 32, 256])
        # print("b, n, s, d = x.size()=>", x.shape)
        x = x.permute(0, 1, 3, 2) # torch.Size([32, 512, 256, 32])
        x = x.reshape(-1, d, s) # torch.Size([16384, 256, 32])
        # print("x = x.reshape(-1, d, s)=>",x.shape)
        batch_size, _, N = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        # print('x = F.relu(self.bn2(self.conv2(x)))=>',x.shape)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # torch.Size([16384, 256])
        # print('x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)=>', x.shape)
        x = x.reshape(b, n, -1).permute(0, 2, 1) # torch.Size([32, 256, 512]),(B,D,N)
        # print('x = x.reshape(b, n, -1).permute(0, 2, 1)=>', x.shape)
        return x

class Pct_cls1(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls1, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

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
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 1024])
        # print('x = F.relu(self.bn1(self.conv1(x)))',x.shape)

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 64, 1024])
        x = x.permute(0, 2, 1) # torch.Size([32, 1024, 64])
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 128])

        feature_0 = self.gather_local_0(new_feature) # torch.Size([32, 128, 512])
        # print("feature_0 = self.gather_local_0(new_feature)=>", feature_0.shape)

        feature = feature_0.permute(0, 2, 1) # torch.Size([32, 512, 128])
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature)
        # new_xyz: torch.Size([32, 256, 3])
        # new_feature: d,n,s,d torch.Size([32, 256, 32, 128])

        feature_1 = self.gather_local_1(new_feature) # torch.Size([32, 256, 256])
        # print("feature_1 = self.gather_local_1(new_feature)", feature_1.shape)

        x = self.pt_last(feature_1) # torch.Size([32, 1024, 256]),(B,D,N)
        # print("x = self.pt_last(feature_1)",x.shape)

        x = torch.cat([x, feature_1], dim=1)  # torch.Size([32, 1024+256(1280), 256]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # torch.Size([32, 1024])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512])
        x = self.dp1(x) # torch.Size([32, 512])
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256])
        x = self.dp2(x) # torch.Size([32, 256])
        x = self.linear3(x) # torch.Size([32, 40])

        return x

class Pct_cls2(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls2, self).__init__()

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

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        # print (x.size())
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls2sax(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls2sax, self).__init__()

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

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        # print (x.size())
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x)
        x3 = self.sa3(x)
        x4 = self.sa4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls3(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls3, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = Sa_Layer(128)
        self.sa2 = Sa_Layer(128)
        self.sa3 = Sa_Layer(128)
        self.sa4 = Sa_Layer(128)
        self.sa5 = Sa_Layer(128)
        self.sa6 = Sa_Layer(128)
        self.sa7 = Sa_Layer(128)
        self.sa8 = Sa_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        # print (x.size())
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = self.sa1(x4)
        x6 = self.sa2(x5)
        x7 = self.sa3(x6)
        x8 = self.sa4(x7)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
        # x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls3sax(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls3sax, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = Sa_Layer(128)
        self.sa2 = Sa_Layer(128)
        self.sa3 = Sa_Layer(128)
        self.sa4 = Sa_Layer(128)
        self.sa5 = Sa_Layer(128)
        self.sa6 = Sa_Layer(128)
        self.sa7 = Sa_Layer(128)
        self.sa8 = Sa_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        # print (x.size())
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x)
        x3 = self.sa3(x)
        x4 = self.sa4(x)
        x5 = self.sa1(x)
        x6 = self.sa2(x)
        x7 = self.sa3(x)
        x8 = self.sa4(x)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
        # x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Pct_cls4(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_cls4, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.gather_local_0 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

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
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 64, 1024])
        # print('x = F.relu(self.bn1(self.conv1(x)))',x.shape)

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([32, 128, 1024])
        x = x.permute(0, 2, 1) # torch.Size([32, 1024, 128])
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)
        # new_xyz: torch.Size([32, 512, 3])
        # new_feature: d,n,s,d torch.Size([32, 512, 32, 256])

        feature_0 = self.gather_local_0(new_feature) # torch.Size([32, 256, 512])
        # print("feature_0 = self.gather_local_0(new_feature)=>", feature_0.shape)

        x = self.pt_last(feature_0) # torch.Size([32, 1024, 512]),(B,D,N)
        # print("x = self.pt_last(feature_1)",x.shape)

        x = torch.cat([x, feature_0], dim=1)  # torch.Size([32, 1024+256(1280), 512]),(B,D,N)
        x = self.conv_fuse(x) # torch.Size([32, 1024, 256]),(B,D,N)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # torch.Size([32, 1024])
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # torch.Size([32, 512])
        x = self.dp1(x) # torch.Size([32, 512])
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([32, 256])
        x = self.dp2(x) # torch.Size([32, 256])
        x = self.linear3(x) # torch.Size([32, 40])

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = Sa_Layer(channels)
        self.sa2 = Sa_Layer(channels)
        self.sa3 = Sa_Layer(channels)
        self.sa4 = Sa_Layer(channels)

    def forward(self, x):
        batch_size, _, N = x.size() # torch.Size([32, 256, 512]),(B,D,N)

        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([32, 256, 512])
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1) # torch.Size([32, 1024, 512])

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