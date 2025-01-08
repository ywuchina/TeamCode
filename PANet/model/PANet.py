import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transforms3d as t3d
from torch.nn.modules.transformer import Transformer
from scipy.spatial.transform import Rotation
from ops.transform_functions import PCRNetTransform as transform
from ops import quaternion
# torch.set_printoptions(threshold=float('inf'))

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2, dim=0, keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):  # x: data(B, 3, N)  k: neighbors_num
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)    # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)     # 对第1维求平方和, keepdim:求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True`
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_neighbors(data, k=20):
    # xyz = data[:, :3, :]    # (B, 3, N)
    xyz = data.view(*data.size()[:3])
    idx = knn(xyz, k=k)  # (batch_size, num_points, k) 即: (B, N, n): 里面存的是N个点的n个邻居的下标
    batch_size, num_points, _ = idx.size()
    # device = torch.device('cuda')

    # idx_base: [B, 1, 1]
    idx_base = torch.arange(0, batch_size).to(xyz.device).view(-1, 1, 1) * num_points   # arange不包含batch_size
    nbrs = torch.tensor([]).to(xyz.device)

    idx = idx + idx_base    # 每个点n近邻的下标    (B, N, n)
    idx = idx.view(-1)  # idx: 0 ~ (batch_size * num_points -1)

    _, num_dims, _ = xyz.size()    # num_dims = 3

    xyz = xyz.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)

    # gxyz
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_gxyz = xyz.view(batch_size * num_points, -1)[idx, :]   # neighbor_gxyz.shape = (B*N*n, 3)
    neighbor_gxyz = neighbor_gxyz.view(batch_size, num_points, k, num_dims)     # (B, N, n, 3)
    # if 'gxyz' in feature_name:
    #     net_input = torch.cat((net_input, neighbor_gxyz), dim=3)

    # # xyz
    # xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # net_input = torch.cat((net_input, xyz), dim=3)

    nbrs = torch.cat((nbrs, neighbor_gxyz), dim=3)
    nbrs = nbrs.permute(0, 3, 1, 2).contiguous()

    return nbrs, idx    # (B, 3, N, n)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()
        # self.max_pool = nn.AdaptiveMaxPool1d(1)  # (B, C, N) -> (B, C, c)
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, N) -> (B, C, c)
        self.fc_1 = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.Linear(channel, channel // reduction), nn.ReLU(),
            nn.Linear(channel // reduction, channel)
            # # 若 a + b ≠ 1 则加上这句; 若 a + b = 1 则删除这句
            # nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(),
            nn.Linear(channel // reduction, channel)
            # # 若 a + b ≠ 1 则加上这句; 若 a + b = 1 则删除这句
            # nn.Sigmoid()
        )

    def forward(self, g, f):    # (B, C, N)
        b, c, _ = g.size()

        # a + b = 1
        features = torch.cat((g.unsqueeze(dim=1), f.unsqueeze(dim=1)), dim=1)     # (B, 2, C, N)
        features_U = torch.sum(features, dim=1)     # (B, C, N)
        feat_pool = torch.mean(features_U, dim=-1)
        # feat_pool = torch.max(features_U, dim=-1)
        # feat_pool = self.avg_pool(features_U).view(b, c)    # (B, C, N) -> (B, C, 1) -> (B, C)
        alpha = self.fc_1(feat_pool).view(b, 1, c)      # (B, C) -> (B, 1, C)
        beta = self.fc_2(feat_pool).view(b, 1, c)       # (B, C) -> (B, 1, C)
        matrix = torch.cat((alpha, beta), dim=1)   # (B, 2, C)
        matrix = F.softmax(matrix, dim=1).view(b, 2, c, 1)  # (B, 2, C) -> (B, 2, C, 1)
        return (matrix * features).sum(dim=1)   # (B, C, N)

        # # a + b ≠ 1
        # features = torch.cat((g.unsqueeze(dim=1), f.unsqueeze(dim=1)), dim=1)     # (B, 2, C, N)
        # features = torch.sum(features, dim=1)  # (B, C, N)
        # feat_pool = self.avg_pool(features).view(b, c)     # (B, C, 1) -> (B, C)
        # alpha = self.fc_1(feat_pool).view(b, c, 1)         # (B, C) -> (B, C, 1)
        # beta = self.fc_2(feat_pool).view(b, c, 1)
        # return alpha * g + beta * f


class PointAttention(nn.Module):
    def __init__(self, channel, reduction=4):   # channel = 1024
        super(PointAttention, self).__init__()
        # 这里可以考虑将 64 -> 1 的卷积换成平均池化再过bn和relu
        self.fcn_1 = nn.Sequential(
            # nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            # nn.Conv1d(channel // reduction, (channel // reduction) // reduction, 1), nn.BatchNorm1d((channel // reduction) // reduction), nn.ReLU(),
            # nn.Conv1d((channel // reduction) // reduction, 1, 1), nn.BatchNorm1d(1)
            nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), nn.BatchNorm1d(1)
        )
        self.fcn_2 = nn.Sequential(
            # nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            # nn.Conv1d(channel // reduction, (channel // reduction) // reduction, 1), nn.BatchNorm1d((channel // reduction) // reduction), nn.ReLU(),
            # nn.Conv1d((channel // reduction) // reduction, 1, 1), nn.BatchNorm1d(1)
            nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), nn.BatchNorm1d(1)
        )
        self.fcn_3 = nn.Sequential(
            # nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            # nn.Conv1d(channel // reduction, (channel // reduction) // reduction, 1), nn.BatchNorm1d((channel // reduction) // reduction), nn.ReLU(),
            # nn.Conv1d((channel // reduction) // reduction, 1, 1), nn.BatchNorm1d(1)
            nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), nn.BatchNorm1d(1)
        )

    # 局部全局融合(或多个局部融合)
    def forward(self, feature1, feature2):
        feature1 = feature1.unsqueeze(dim=1)
        feature2 = feature2.unsqueeze(dim=1)
        features = torch.cat((feature1, feature2), dim=1) # (B, 2, C, N)
        feature_U = torch.sum(features, dim=1)  # (B, C, N)

        # a + b = 1
        a = self.fcn_1(feature_U)   # (B, 1, N)
        b = self.fcn_2(feature_U)   # (B, 1, N)
        matrix = torch.cat((a, b), dim=1)   # (B, 2, N)
        matrix = F.softmax(matrix, dim=1)   # g -> a; f -> 1-a (B, 2, N)
        matrix = matrix.unsqueeze(dim=2)    # (B, 2, 1, N)
        features = (matrix * features).sum(dim=1)  # (B, C, N): a * g + (1 - a) * f

        # # a + b ≠ 1
        # a = F.softmax(self.fcn_1(feature_U), dim=1)
        # b = F.softmax(self.fcn_2(feature_U), dim=1)
        # matrix = torch.cat((a, b), dim=1)   # (B, 2, N)
        # matrix = matrix.unsqueeze(dim=2)    # (B, 2, 1, N)
        # features = (matrix * features).sum(dim=1)   # (B, C, N)

        return features

    # # 全局+局部多尺度 融合
    # def forward(self, m1, m2, mg):
    #     m1 = m1.unsqueeze(dim=1)
    #     m2 = m2.unsqueeze(dim=1)
    #     mg = mg.unsqueeze(dim=1)
    #     features = torch.cat((m1, m2, mg), dim=1)     # (B, 3, C, N) -> 3: 分支数
    #     feature_U = torch.sum(features, dim=1)          # (B, C, N)
    #
    #     # a + b + c = 1
    #     a = self.fcn_1(feature_U)   # (B, 1, N)
    #     b = self.fcn_2(feature_U)   # (B, 1, N)
    #     c = self.fcn_3(feature_U)   # (B, 1, N)
    #     matrix = torch.cat((a, b, c), dim=1)    # (B, 3, N)
    #     matrix = F.softmax(matrix, dim=1)       # a + b + c = 1
    #     matrix = matrix.unsqueeze(dim=2)        # (B, 3, 1, N)
    #     features = (matrix * features).sum(dim=1)  # (B, C, N): a * m1 + b * m2 + c * mg
    #
    #     return features

class LAGNet(nn.Module):
    def __init__(self, nbrs_num1=16, nbrs_num2=8):
        super(LAGNet, self).__init__()
        self.nbrs_num1 = nbrs_num1
        self.nbrs_num2 = nbrs_num2

        self.pa_layer1 = PointAttention(channel=64, reduction=4)
        self.pa_layer2 = PointAttention(channel=64, reduction=4)
        self.pa_layer3 = PointAttention(channel=128, reduction=4)
        self.pa_layer4 = PointAttention(channel=256, reduction=4)

        # layer_fuse
        self.conv1d_1 = nn.Conv1d(3, 64, 1)
        self.conv1d_2 = nn.Conv1d(64, 64, 1)
        self.conv1d_3 = nn.Conv1d(64, 128, 1)
        self.conv1d_4 = nn.Conv1d(128, 256, 1)
        self.bn1d_1 = nn.BatchNorm1d(64)
        self.bn1d_2 = nn.BatchNorm1d(64)
        self.bn1d_3 = nn.BatchNorm1d(128)
        self.bn1d_4 = nn.BatchNorm1d(256)

        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.bn2d_1 = nn.BatchNorm2d(64)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.bn2d_3 = nn.BatchNorm2d(128)
        self.bn2d_4 = nn.BatchNorm2d(256)

        self.conv1d_5 = nn.Conv1d(512, 512, 1)
        self.bn1d_5 = nn.BatchNorm1d(512)

        # # 1024
        # self.conv1d_5 = nn.Conv1d(512, 1024, 1)
        # self.bn1d_5 = nn.BatchNorm1d(1024)
        #
        # self.conv1d_6 = nn.Conv1d(1024, 1024, 1)
        # self.bn1d_6 = nn.BatchNorm1d(1024)

    def forward(self, pointcloud):
        pointcloud = pointcloud.permute(0, 2, 1).contiguous()   # (32, 3, 1024) 即:(B, 3, N)
        batch_size, num_dims, N = pointcloud.size()

        # # 全局局部融合(lf + gf)
        # lf, idx_lf = get_neighbors(pointcloud, k=self.nbrs_num1)
        # gf = lf[:, :, :, :1]
        #
        # lf = F.relu(self.bn2d_1(self.conv2d_1(lf)), inplace=True)     # (B, C, N, n1)
        # gf = F.relu(self.bn2d_1(self.conv2d_1(gf)), inplace=True)     # (B, C, N, 1)
        # mf = lf.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]                         # (B, C, N)
        # mg = (mg.max(dim=-1, keepdim=True)[0]).repeat(1, 1, N)        # (B, C, 1) -> (B, C, N)
        # fuse_1 = self.pa_layer1(mf, mg)
        #
        # lf = F.relu(self.bn2d_2(self.conv2d_2(lf)), inplace=True)
        # gf = F.relu(self.bn2d_2(self.conv2d_2(gf)), inplace=True)
        # mf = lf.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # mg = (mg.max(dim=-1, keepdim=True)[0]).repeat(1, 1, N)        # (B, C, 1) -> (B, C, N)
        # fuse_2 = self.pa_layer2(mf, mg)
        #
        # lf = F.relu(self.bn2d_3(self.conv2d_3(lf)), inplace=True)
        # gf = F.relu(self.bn2d_3(self.conv2d_3(gf)), inplace=True)
        # mf = lf.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # mg = (mg.max(dim=-1, keepdim=True)[0]).repeat(1, 1, N)        # (B, C, 1) -> (B, C, N)
        # fuse_3 = self.pa_layer3(mf, mg)
        #
        # lf = F.relu(self.bn2d_4(self.conv2d_4(lf)), inplace=True)
        # gf = F.relu(self.bn2d_4(self.conv2d_4(gf)), inplace=True)
        # mf = lf.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # mg = (mg.max(dim=-1, keepdim=True)[0]).repeat(1, 1, N)        # (B, C, 1) -> (B, C, N)
        # fuse_4 = self.pa_layer4(mf, mg)
        #
        # features_cat = torch.cat((fuse_1, fuse_2, fuse_3, fuse_4), dim=1)
        # pointcloud_features = F.relu(self.bn1d_5(self.conv1d_5(features_cat)), inplace=True)


        # 仅局部多尺度融合
        lf1, idx_lf1 = get_neighbors(pointcloud, k=self.nbrs_num1)  # (B, 3, N, n1)
        lf2 = lf1[:, :, :, :self.nbrs_num2]

        lf1 = F.relu(self.bn2d_1(self.conv2d_1(lf1)), inplace=True)     # (B, C, N, n1)
        lf2 = F.relu(self.bn2d_1(self.conv2d_1(lf2)), inplace=True)     # (B, C, N, n2)
        m1 = lf1.max(dim=-1, keepdim=False)[0]     # (B, C, N)
        m2 = lf2.max(dim=-1, keepdim=False)[0]     # (B, C, N)
        fuse_1 = self.pa_layer1(m1, m2)

        lf1 = F.relu(self.bn2d_2(self.conv2d_2(lf1)), inplace=True)
        lf2 = F.relu(self.bn2d_2(self.conv2d_2(lf2)), inplace=True)
        m1 = lf1.max(dim=-1, keepdim=False)[0]
        m2 = lf2.max(dim=-1, keepdim=False)[0]
        fuse_2 = self.pa_layer2(m1, m2)

        lf1 = F.relu(self.bn2d_3(self.conv2d_3(lf1)), inplace=True)
        lf2 = F.relu(self.bn2d_3(self.conv2d_3(lf2)), inplace=True)
        m1 = lf1.max(dim=-1, keepdim=False)[0]
        m2 = lf2.max(dim=-1, keepdim=False)[0]
        fuse_3 = self.pa_layer3(m1, m2)

        lf1 = F.relu(self.bn2d_4(self.conv2d_4(lf1)), inplace=True)
        lf2 = F.relu(self.bn2d_4(self.conv2d_4(lf2)), inplace=True)
        m1 = lf1.max(dim=-1, keepdim=False)[0]
        m2 = lf2.max(dim=-1, keepdim=False)[0]
        fuse_4 = self.pa_layer4(m1, m2)
        features_cat = torch.cat((fuse_1, fuse_2, fuse_3, fuse_4), dim=1)

        pointcloud_features = F.relu(self.bn1d_5(self.conv1d_5(features_cat)), inplace=True)


        # # 全局加局部多尺度融合(lf1 + lf2 + gf)
        # lf1, idx_lf1 = get_neighbors(pointcloud, k=self.nbrs_num1)
        # lf2 = lf1[:, :, :, :self.nbrs_num2]
        # gf = lf1[:, :, :, :1]
        #
        # lf1 = F.relu(self.bn2d_1(self.conv2d_1(lf1)), inplace=True)     # (B, C, N, n1)
        # lf2 = F.relu(self.bn2d_1(self.conv2d_1(lf2)), inplace=True)     # (B, C, N, n2)
        # gf = F.relu(self.bn2d_1(self.conv2d_1(gf)), inplace=True)       # (B, C, N, 1)
        # m1 = lf1.max(dim=-1, keepdim=False)[0]
        # m2 = lf2.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # fuse_1 = self.pa_layer1(m1, m2, mg)
        #
        # lf1 = F.relu(self.bn2d_2(self.conv2d_2(lf1)), inplace=True)
        # lf2 = F.relu(self.bn2d_2(self.conv2d_2(lf2)), inplace=True)
        # gf = F.relu(self.bn2d_2(self.conv2d_2(gf)), inplace=True)
        # m1 = lf1.max(dim=-1, keepdim=False)[0]
        # m2 = lf2.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # fuse_2 = self.pa_layer2(m1, m2, mg)
        #
        # lf1 = F.relu(self.bn2d_3(self.conv2d_3(lf1)), inplace=True)
        # lf2 = F.relu(self.bn2d_3(self.conv2d_3(lf2)), inplace=True)
        # gf = F.relu(self.bn2d_3(self.conv2d_3(gf)), inplace=True)
        # m1 = lf1.max(dim=-1, keepdim=False)[0]
        # m2 = lf2.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # fuse_3 = self.pa_layer3(m1, m2, mg)
        #
        # lf1 = F.relu(self.bn2d_4(self.conv2d_4(lf1)), inplace=True)
        # lf2 = F.relu(self.bn2d_4(self.conv2d_4(lf2)), inplace=True)
        # gf = F.relu(self.bn2d_4(self.conv2d_4(gf)), inplace=True)
        # m1 = lf1.max(dim=-1, keepdim=False)[0]
        # m2 = lf2.max(dim=-1, keepdim=False)[0]
        # mg = gf.max(dim=-1, keepdim=False)[0]
        # fuse_4 = self.pa_layer4(m1, m2, mg)
        #
        # features_cat = torch.cat((fuse_1, fuse_2, fuse_3, fuse_4), dim=1)
        # pointcloud_features = F.relu(self.bn1d_5(self.conv1d_5(features_cat)), inplace=True)

        return pointcloud_features


class PANet(nn.Module):
    def __init__(self, source_feature_size=512, template_feature_size=512, feature_model=LAGNet()):
        super(PANet, self).__init__()
        self.feature_model = feature_model
        input_size = source_feature_size + template_feature_size
        self.fc = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU(),
                                # nn.Linear(1024, 1024), nn.ReLU(),
                                nn.Linear(1024, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512, 256), nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, 7))

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()    # 判断输入数据是几维的, 若是(N, C)则ndim=2, 若是(B, N, C)则ndim=3
        # 生成真实值
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = pose_7d[:, :4].expand([N, -1])  # (N, 4)
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)

        # 生成预测值
        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = pose_7d[:, :4].unsqueeze(1).expand([-1, N, -1]).contiguous()    # (B, N, 4)
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)
        else:
            raise ValueError("quaternion_rotate dims error")

        return rotated_point_cloud

    @staticmethod
    def parameter_update(pose_pred_new, pose_pred_old):
        pose_quat = quaternion.qmul(pose_pred_new[:, :4], pose_pred_old[:, :4])         # (B, 4)
        pose_trans = quaternion.qrot(pose_pred_new[:, :4], pose_pred_old[:, 4:]) + pose_pred_new[:, 4:] # (B, 3)
        pose_pred = torch.cat([pose_quat, pose_trans], dim=1)
        return pose_pred

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    # source & template: (32, 1024, 3)
    def forward(self, source, template, num_iter=4):    # template -> source
        # init params
        B, src_N, _ = source.size()
        _, ref_N, _ = template.size()
        init_quat = t3d.euler.euler2quat(0., 0., 0., "sxyz")
        init_quat = torch.from_numpy(init_quat).expand(B, 4).to(source.device)
        init_translate = torch.from_numpy(np.array([[0., 0., 0.]])).expand(B, 3).to(source.device)
        pose_pred = torch.cat((init_quat, init_translate), dim=1).float().to(source.device)  # (B, 7)

        # rename template
        template_iter = template.clone().to(template.device)
        source_features = self.feature_model(source)    # (B, 1024, N)
        source_features = torch.max(source_features, dim=2)[0].contiguous()

        for i in range(num_iter):
            template_features = self.feature_model(template_iter)  # (B, 1024, N)
            template_features = torch.max(template_features, dim=2)[0].contiguous()

            fc_input = torch.cat((template_features, source_features), dim=1)
            pose_pred_iter = self.fc(fc_input)  # (B, 7)
            pose_pred_iter = self.create_pose_7d(pose_pred_iter)    # 对输出(四元数)归一化

            template_iter = self.quaternion_rotate(template_iter, pose_pred_iter) + pose_pred_iter[:, 4:].unsqueeze(dim=1)   # Pt" = R*Pt + t
            pose_pred = self.parameter_update(pose_pred_iter, pose_pred)

            pred_rot = torch.tensor([]).to(source.device)

            for i in range(B):
                tmp = self.quaternion_rotate(torch.eye(3).to(source.device), pose_pred[i:i+1, :]).permute(1, 0)
                pred_rot = torch.cat([pred_rot, tmp.unsqueeze(dim=0)], dim=0)
            transform_pred = torch.cat([pred_rot, pose_pred[:, 4:].unsqueeze(dim=1).permute(0, 2, 1)], dim=2)  # (B, 3, 4)

        result = {'pose_pred': pose_pred,               # (B, 7)
                  'transform_pred': transform_pred,     # (B, 3, 4)
                  'transformed_template': template_iter
                  }
        return result

if __name__ == '__main__':
    source, template = torch.rand(10, 1024, 3), torch.rand(10, 1024, 3)
    net = PANet()
    result = net(source, template)
    import ipdb; ipdb.set_trace()
