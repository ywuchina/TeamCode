# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from scipy.spatial.transform import Rotation
# PyTorch-backed implementations

def torch_qmul(q1, q2):
    """
    Multiply quaternion(s) q2q1, rotate q1 first, rotate q2 second.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q1.view(-1, 4, 1), q2.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def torch_qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

# # OMNet原代码
# def torch_transform_pose(pose_old, pose_new):
#     quat_old, translate_old = pose_old[:, :4], pose_old[:, 4:]
#     quat_new, translate_new = pose_new[:, :4], pose_new[:, 4:]
#
#     quat = torch_qmul(quat_old, quat_new)
#     translate = torch_qrot(quat_new, translate_old) + translate_new
#     pose = torch.cat((quat, translate), dim=1)
#
#     return pose

# PANet代码：修改过
def torch_transform_pose(pose_old, pose_new):
    quat_old, translate_old = pose_old[:, :4], pose_old[:, 4:]
    quat_new, translate_new = pose_new[:, :4], pose_new[:, 4:]

    quat = torch_qmul(quat_new, quat_old)
    translate = torch_qrot(quat_new, translate_old) + translate_new
    pose = torch.cat((quat, translate), dim=1)

    return pose

def torch_quat_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
    ndim = point_cloud.dim()
    if ndim == 2:
        N, _ = point_cloud.shape
        assert pose_7d.shape[0] == 1
        # repeat transformation vector for each point in shape
        quat = pose_7d[:, 0:4].expand([N, -1])
        rotated_point_cloud = torch_qrot(quat, point_cloud)

    elif ndim == 3:
        B, N, _ = point_cloud.shape
        quat = pose_7d[:, 0:4].unsqueeze(1).expand([-1, N, -1]).contiguous()
        rotated_point_cloud = torch_qrot(quat, point_cloud)

    else:
        raise RuntimeError("point cloud dim must be 2 or 3 !")

    return rotated_point_cloud

# pc: (B, N, 3)
def torch_quat_transform(pose_7d: torch.Tensor, pc: torch.Tensor, normal: torch.Tensor = None):
    pc_t = torch_quat_rotate(pc, pose_7d) + pose_7d[:, 4:].view(-1, 1, 3).repeat(1, pc.shape[1], 1)  # Ps" = R*Ps + t
    return pc_t

def torch_quat2mat(pose):
    # Separate each quaternion value.
    q0, q1, q2, q3 = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    # Convert quaternion to rotation matrix.
    # Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
    R11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R12 = 2 * (q1 * q2 - q0 * q3)
    R13 = 2 * (q1 * q3 + q0 * q2)
    R21 = 2 * (q1 * q2 + q0 * q3)
    R22 = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    R23 = 2 * (q2 * q3 - q0 * q1)
    R31 = 2 * (q1 * q3 - q0 * q2)
    R32 = 2 * (q2 * q3 + q0 * q1)
    R33 = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2
    R = torch.stack((torch.stack((R11, R12, R13), dim=0), torch.stack((R21, R22, R23), dim=0), torch.stack((R31, R32, R33), dim=0)), dim=0)

    rot_mat = R.permute((2, 0, 1))  # (B, 3, 3)
    translation = pose[:, 4:].unsqueeze(2)  # (B, 3, 1)
    transform = torch.cat((rot_mat, translation), dim=2)
    return transform  # (B, 3, 4)

def mat2euler(mats, seq='zyx'):
    eulers = torch.tensor([])
    for i in range(mats.size(0)):
        r = Rotation.from_matrix(mats[i].detach().cpu().numpy())
        temp = torch.tensor(r.as_euler(seq, degrees=False)).view(1, 3)[:, [2, 1, 0]]
        eulers = torch.cat((eulers, temp), dim=0)
    return eulers

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


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


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


# Numpy-backed implementations


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


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
        result *= -1

    return result.reshape(original_shape)


def qinv(q):
    # expectes q in (w,x,y,z) format
    w = q[:, 0:1]
    v = q[:, 1:]
    inv = torch.cat([w, -v], dim=1)
    return inv
