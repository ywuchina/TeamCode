import torch
import torch.nn as nn


class ChamferLoss(nn.Module):

    def __init__(self,use_cuda = True):
        super(ChamferLoss, self).__init__()
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            self.use_cuda = True

    def forward(self, preds, gts):
        S1=preds.shape[1]
        S2=gts.shape[1]
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.div(torch.sum(mins),S1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.div(torch.sum(mins),S2)

        return loss_1 + loss_2


    def batch_pairwise_dist(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        xy = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xy.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(xy)
        P = (rx.transpose(2, 1) + ry - 2 * xy)
        return P

if __name__ == '__main__':

    source = torch.rand(10,1024,3)
    trm=torch.rand(10,1024,3)
    a=ChamferLoss()(source,trm)




