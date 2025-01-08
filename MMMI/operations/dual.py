import torch
import numpy as np
import torch.nn as nn

def get_extrinsic(rot, trans):
    return np.hstack((rot, trans))

def get_rotmat_from_extrinsic(extrinsic):
    return extrinsic[: ,: , :3]

def get_trans_from_extrinsic(extrinsic):
    return extrinsic[:, :, 3:]

def rotmat_to_quat(rotmat):
    B,_,_ =  rotmat.size()
    tr = rotmat[:,0, 0] + rotmat[:,1, 1] + rotmat[:,2, 2]

    # if tr > 0:
    s = torch.sqrt(tr + 1.0) * 2
    qw = 0.25 * s
    qx = (rotmat[:,1, 2] - rotmat[:,2, 1]) / s
    qy = (rotmat[:,2, 0] - rotmat[:,0, 2]) / s
    qz = (rotmat[:,0, 1] - rotmat[:,1, 0]) / s

    return torch.cat((qw,qx,qy,qz),dim=0).reshape(B,4)

def quat_mult(q1, q2):
    B,_ = q1.size()
    qw = -q1[:,1] * q2[:,1] - q1[:,2] * q2[:,2] - q1[:,3] * q2[:,3] + q1[:,0] * q2[:,0]
    qx =  q1[:,1] * q2[:,0] + q1[:,2] * q2[:,3] - q1[:,3] * q2[:,2] + q1[:,0] * q2[:,1]
    qy = -q1[:,1] * q2[:,3] + q1[:,2] * q2[:,0] + q1[:,3] * q2[:,1] + q1[:,0] * q2[:,2]
    qz =  q1[:,1] * q2[:,2] - q1[:,2] * q2[:,1] + q1[:,3] * q2[:,0] + q1[:,0] * q2[:,3]
    return torch.cat((qw,qx,qy,qz),dim=0).reshape(B,4)


def extrinsic_to_dual_quat(extrinsic):
    rotmat = get_rotmat_from_extrinsic(extrinsic)   #R
    trans = get_trans_from_extrinsic(extrinsic)    #T
    B,_,_ = rotmat.size()

    rotmat = rotmat.cuda()
    trans = trans.cuda()

    zero = torch.zeros(B,1,1).cuda()
    trans = torch.cat((zero,trans),dim = 1).reshape(B,4)#给T增加一个维度0

    real_quat = rotmat_to_quat(rotmat)
    dual_quat = quat_mult(trans, real_quat) * 0.5

    return real_quat, dual_quat

def conj_quat(q):
    q0 = q[:,0].unsqueeze(-1)
    q1 = (q[:, 1] * -1).unsqueeze(-1)
    q2 = (-q[:, 2]* -1).unsqueeze(-1)
    q3 = (-q[:, 3]* -1).unsqueeze(-1)
    q = torch.cat((q0,q1,q2,q3),dim = 1)
    return q

def dual_quat_to_extrinsic(real_quat, dual_quat):
    w = real_quat[:, 0]
    x = real_quat[:, 1]
    y = real_quat[:, 2]
    z = real_quat[:, 3]

    B = real_quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z


    t = qmul(2 * dual_quat, conj_quat(real_quat))


    rot = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)


    trans = torch.stack([t[:,1],t[:,2],t[:,3]],dim = 1).reshape(B, 1, 3)


    return rot,trans

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

if __name__ == '__main__':
     x=torch.rand(10,4)
     y=torch.rand(10,4)
     a=dual_quat_to_extrinsic(x,y)
     #print(a)
     x = torch.rand(10, 3, 3)
     y = torch.rand(10, 3, 1)
     B,_,_ = x.size()
     z = torch.cat((x,y),dim=2)
     real, dual = extrinsic_to_dual_quat(z)



