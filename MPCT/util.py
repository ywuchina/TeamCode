import torch
import torch.nn.functional as F

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, nsample, xyz, point):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()
    fps_idx = farthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) # [B, npoint, 3]
    sample_xyz = new_xyz
    new_points = index_points(point, fps_idx) # [B, npoint, D]
    sample_points = new_points
    idx = knn_point(nsample, xyz, new_xyz) # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, 3]
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # [B, npoint, nsample, 3]
    # dist = torch.norm(grouped_xyz_norm, dim=3, keepdim=True)  #  [B, npoint, nsample, 1]
    # new_xyz = torch.cat((grouped_xyz_norm, new_xyz.view(B, S, 1, -1).repeat(1, 1, nsample, 1)), dim=-1)

    grouped_points = index_points(point, idx) # [B, npoint, nsample, D]
    # grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1) # [B, npoint, nsample, D]
    # new_points = torch.cat((grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)), dim=-1)
    return sample_xyz, grouped_xyz, sample_points, grouped_points

def group(nsample, xyz, point):
    # xyz: B,C,N, points:B,D,N
    batch_size = xyz.size(0)
    num_points = xyz.size(2)
    xyz = xyz.view(batch_size, -1, num_points)
    idx = knn(xyz, k=nsample)  # (batch_size, num_points, nsample)

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.LongTensor)
    idx = idx.type(torch.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = xyz.size()
    # xyz = xyz.transpose(2, 1).contiguous()
    # xyzs = xyz.view(batch_size * num_points, -1)[idx, :]
    # grouped_xyzs = xyzs.view(batch_size, num_points, nsample, num_dims) # B,N,nsample,C
    # grouped_xyzs_norm = grouped_xyzs - xyz.view(batch_size, num_points, 1, -1)  # [B, N, nsample, C]
    # new_xyzs = torch.cat((grouped_xyzs_norm, xyz.view(batch_size, num_points, 1, -1).repeat(1, 1, nsample, 1)), dim=-1)

    _, embidding, _ = point.size()
    points = point.view(batch_size * num_points, -1)[idx, :]
    grouped_points = points.view(batch_size, num_points, nsample, embidding) # B,N,nsample,D
    grouped_points_norm = grouped_points - point.view(batch_size, num_points, 1, -1)  # [B, N, nsample, D]
    new_points = torch.cat((grouped_points_norm, point.view(batch_size, num_points, 1, -1).repeat(1, 1, nsample, 1)),dim=-1)

    return new_points

def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    device = x.device
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.LongTensor)
    idx = idx.type(torch.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)
    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.LongTensor([1]).to(device)) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.LongTensor([2]).to(device)) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x
    edge2 = neighbor_2nd-org_x
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True) # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1) # B,14,N
    # new_pts = torch.cat((org_x, normals, edge1, edge2), 1)  # B,12,N
    # new_pts = torch.cat((org_x, dist1, dist2, edge1, edge2), 1)  # B,11,N
    # new_pts = torch.cat((org_x, normals, dist1, dist2), 1)  # B,8,N
    return new_pts
