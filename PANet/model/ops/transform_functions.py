import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors
from . import quaternion

# Create Partial Point Cloud. [Code referred from PRNet paper.]
# def farthest_subsample_points(source_cloud, num_subsampled_points=1536):
def farthest_subsample_points(source_cloud, num_subsampled_points=768):
    source = source_cloud
    num_points = source.shape[0]    # 1024*3
    # nbrs: 768*3
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(source[:, :3])
    # np.random.random(size=(1, 3)) 产生[0,1)之间大小为 1*3 的浮点数
    # random_p1: -499.xx or 500.xx
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
    return source[idx1, :], gt_mask

# def jitter_pointcloud(source_cloud1, sigma=0.01, clip=0.05):
#     # N, C = pointcloud.shape
#     # np.random.normal()
#     sigma = 0.04*np.random.random_sample()
#     source_cloud1 += torch.empty(source_cloud1.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
#     # pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
#     return source_cloud1

def jitter_pointcloud(pointcloud, sigma=0.06, clip=0.05):
# def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.001):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def add_outliers(template_cloud, gt_mask):
    # pointcloud: 			Point Cloud (ndarray) [NxC]
    # output: 				Corrupted Point Cloud (ndarray) [(N+100)xC]
    N, C = template_cloud.shape
    outliers = 2*torch.rand(100, C)-1 					# Sample points in a cube [-0.5, 0.5]
    template_cloud = torch.cat([template_cloud, outliers], dim=0)   # +100
    gt_mask = torch.cat([gt_mask, torch.zeros(100)])    # +100
    idx = torch.randperm(template_cloud.shape[0])
    template_cloud, gt_mask = template_cloud[idx], gt_mask[idx]
    return template_cloud, gt_mask

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles(弧度制).
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        # result *= -1  # 对应pcrnet
        result *= 1    # 对应omnet

    return result.reshape(original_shape)

def npmat2euler(mats, seq='zyx'):
    """ return euler angles in radian """
    eulers = []
    r = Rotation.from_matrix(mats)
    eulers.append(r.as_euler(seq, degrees=False))
    return np.asarray(eulers, dtype='float32')

def mat2euler(mats, seq='zyx'):
    eulers = torch.tensor([])
    for i in range(mats.size(0)):
        r = Rotation.from_matrix(mats[i])
        temp = torch.tensor(r.as_euler(seq, degrees=False)).view(1, 3)[:, [2, 1, 0]]
        eulers = torch.cat((eulers, temp), dim=0)
    return eulers

class PCRNetTransform:
    # def __init__(self, data_size, angle_range=45, translation_range=0.5):
    def __init__(self, data_size, angle_range=45, translation_range=0.12):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        # self.transformations: 9840 * (1, 7): 初始化9840个变换参数
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.index = 0

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg

    # 由欧拉角到四元数再归一化
    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)    # 角度转弧度
        # euler = np.random.uniform(-max_rotation, max_rotation, [1, 3])    # ndarray(1,3) 真实弧度制欧拉角
        euler = np.random.uniform(0, max_rotation, [1, 3])    # ndarray(1,3) 真实弧度制欧拉角
        # print("ground truth euler angles in radian:\n", rot)
        trans = np.random.uniform(-max_translation, max_translation, [1, 3])
        quat = euler_to_quaternion(euler, "xyz")  # 返回的是已经归一化的结果

        vec = np.concatenate([quat, trans], axis=1)
        vec = torch.tensor(vec, dtype=dtype)
        return vec
    # -----end init-----

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
            return pose_7d[:, 4:]

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()    # 判断输入数据是几维的, 若是(N, C)则ndim=2, 若是(B, N, C)则ndim=3
        # 生成真实值
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = PCRNetTransform.get_quaternion(pose_7d).expand([N, -1])  # (N, 4)
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)

        # 生成预测值
        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = PCRNetTransform.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()    # (B, N, 4)
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)
        else:
            raise ValueError("quaternion_rotate dims error")

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = PCRNetTransform.quaternion_rotate(point_cloud, pose_7d) + PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix

    # template: (1024, 3)
    def __call__(self, template):   # 检验过，全部正确
        self.igt = self.transformations[self.index]     # (1, 7)
        igt = self.create_pose_7d(self.igt)     # F.normalize: 归一化后的pose_7d
        gt_quat = igt[:, :4].squeeze(dim=0)
        gt_trans = igt[:, 4:].squeeze(dim=0)

        source = self.quaternion_rotate(template, igt) + self.get_translation(igt)
        gt_rot = self.quaternion_rotate(torch.eye(3), igt).permute(1, 0)            # (3, 3)
        gt_T = torch.cat([gt_rot, gt_trans.unsqueeze(dim=0).permute(1, 0)], dim=1)  # (3, 4)
        return source, gt_quat, gt_trans, gt_T  # source: (1024, 3), gt_quat: (4), gt_trans: (3)


        # self.igt_rotation = self.quaternion_rotate(torch.eye(3), igt).permute(1, 0)        # 真实旋转矩阵("xyz"): [3x3]
        # self.igt_translation = self.get_translation(igt)                                   # 真实平移向量: [1x3]
        # source = self.quaternion_rotate(template, igt) + self.get_translation(igt)
        # return source, igt, self.igt_rotation, self.igt_translation



class Generate_transformed_source:
    def __init__(self, data_size, angle_range=0.5, translation_range=0.0005):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        # self.transformations: 9840 * (1, 7): 初始化9840个变换参数
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.index = 0

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg

    # 由欧拉角到四元数再归一化
    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)    # 角度转弧度
        # euler = np.random.uniform(-max_rotation, max_rotation, [1, 3])    # ndarray(1,3) 真实弧度制欧拉角
        euler = np.random.uniform(0, max_rotation, [1, 3])    # ndarray(1,3) 真实弧度制欧拉角
        # print("ground truth euler angles in radian:\n", rot)
        trans = np.random.uniform(-max_translation, max_translation, [1, 3])
        quat = euler_to_quaternion(euler, "xyz")  # 返回的是已经归一化的结果

        vec = np.concatenate([quat, trans], axis=1)
        vec = torch.tensor(vec, dtype=dtype)
        return vec
    # -----end init-----

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
            return pose_7d[:, 4:]

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()    # 判断输入数据是几维的, 若是(N, C)则ndim=2, 若是(B, N, C)则ndim=3
        # 生成真实值
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = PCRNetTransform.get_quaternion(pose_7d).expand([N, -1])  # (N, 4)
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)

        # 生成预测值
        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = PCRNetTransform.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()    # (B, N, 4)
            rotated_point_cloud = quaternion.qrot(quat, point_cloud)
        else:
            raise ValueError("quaternion_rotate dims error")

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = PCRNetTransform.quaternion_rotate(point_cloud, pose_7d) + PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix

    # template: (1024, 3)
    def __call__(self, template):   # 检验过，全部正确
        self.igt = self.transformations[self.index]     # (1, 7)
        igt = self.create_pose_7d(self.igt)     # F.normalize: 归一化后的pose_7d
        gt_quat = igt[:, :4].squeeze(dim=0)
        gt_trans = igt[:, 4:].squeeze(dim=0)

        transformed_source = self.quaternion_rotate(template, igt) + self.get_translation(igt)
        gt_rot = self.quaternion_rotate(torch.eye(3), igt).permute(1, 0)            # (3, 3)
        gt_T = torch.cat([gt_rot, gt_trans.unsqueeze(dim=0).permute(1, 0)], dim=1)  # (3, 4)
        return transformed_source, gt_quat, gt_trans, gt_T  # transformed_source: (768, 3), gt_quat: (4), gt_trans: (3)