import os
import argparse

import wandb

from model import IDAM, FPFH, GNN
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# arg_bool = lambda x: x.lower() in ['true', 't', '1']
# parser = argparse.ArgumentParser(description='Point Cloud Registration')
# parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
#                     help='Name of the experiment')
# parser.add_argument('--num_iter', type=int, default=3, metavar='N',
#                     help='Number of iteration inside the network')
# parser.add_argument('--emb_nn', type=str, default='GNN', metavar='N',
#                     help='Feature extraction method. [GNN, FPFH]')
# parser.add_argument('--emb_dims', type=int, default=64, metavar='N',
#                     help='Dimension of embeddings. Must be 33 if emb_nn == FPFH')
# parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
#                     help='Size of batch)')
# parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
#                     help='Size of batch)')
# parser.add_argument('--epochs', type=int, default=40, metavar='N',
#                     help='number of episode to train ')
# parser.add_argument('--unseen', type=arg_bool, default='False',
#                     help='Test on unseen categories')
# parser.add_argument('--alpha', type=float, default=0.75, metavar='N',
#                     help='Fraction of points when sampling partial point cloud')
# parser.add_argument('--factor', type=float, default=4, metavar='N',
#                     help='Divided factor for rotations')
#
# args = parser.parse_args()
#
# # net = IDAM(GNN(args.emb_dims), args)
# args.emb_dims = 33
# net = IDAM(FPFH(), args)
# from thop import profile
# dummy_input1 = torch.randn(1, 3, 1024)
# dummy_input2 = torch.randn(1, 3, 1024)
# flops, params = profile(net, (dummy_input1,dummy_input2,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.4f G, params: %.4f M' % (flops / 1000000000.0, params / 1000000.0))
# assert 1 == 2

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40, RegistrationkittiData, Kitti
from util import npmat2euler
import numpy as np
from tqdm import tqdm


torch.backends.cudnn.enabled = False # fix cudnn non-contiguous error


def test_one_epoch(args, net, test_loader):
    net.eval()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    for src, target, R, t, euler in tqdm(test_loader):

        src = src.to(torch.device('cuda:{}'.format(args.gpu)))
        target = target.to(torch.device('cuda:{}'.format(args.gpu)))
        R = R.to(torch.device('cuda:{}'.format(args.gpu)))
        t = t.to(torch.device('cuda:{}'.format(args.gpu)))

        R_pred, t_pred, *_ = net(src, target)

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    for src, target, R, t, euler in tqdm(train_loader):
        src = src.to(torch.device('cuda:{}'.format(args.gpu)))
        target = target.to(torch.device('cuda:{}'.format(args.gpu)))
        R = R.to(torch.device('cuda:{}'.format(args.gpu)))
        t = t.to(torch.device('cuda:{}'.format(args.gpu)))

        opt.zero_grad()
        R_pred, t_pred, loss = net(src, target, R, t)

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

        loss.backward()
        opt.step()

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae


def train(args, net, train_loader, test_loader):
    opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = MultiStepLR(opt, milestones=[30], gamma=0.1)

    wandb.init(project='IDAM', name=args.exp_name)
    wandb.watch(net)
    for epoch in range(args.epochs):

        train_stats = train_one_epoch(args, net, train_loader, opt)
        r_rmse,r_mae,t_rmse,t_mae = test_one_epoch(args, net, test_loader)

        print('=====  EPOCH %d  =====' % (epoch+1))
        print('TRAIN, rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % train_stats)
        # print('TEST,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % test_stats)
        wandb_log = {}
        wandb_log['RMSE(R)'] = r_rmse
        wandb_log['MAE(R)'] = r_mae
        wandb_log['RMSE(t)'] = t_rmse
        wandb_log['MAE(t)'] = t_mae
        wandb.log(wandb_log)
        torch.save(net.state_dict(), 'checkpoints/others/idam/kitti_%s_%d.t7' % (args.exp_name, args.type, epoch))

        scheduler.step()

def main():
    arg_bool = lambda x: x.lower() in ['true', 't', '1']
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--num_iter', type=int, default=3, metavar='N',
                        help='Number of iteration inside the network')
    parser.add_argument('--emb_nn', type=str, default='GNN', metavar='N',
                        help='Feature extraction method. [GNN, FPFH]')
    parser.add_argument('--type', type=str, default='object', metavar='N',
                        help='KITTI type. [object, tracking]')
    parser.add_argument('--emb_dims', type=int, default=64, metavar='N',
                        help='Dimension of embeddings. Must be 33 if emb_nn == FPFH')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--noise', type=arg_bool, default='False',
                        help='Test on noise models')
    parser.add_argument('--unseen', type=arg_bool, default='False',
                        help='Test on unseen categories')
    parser.add_argument('--alpha', type=float, default=1, metavar='N',
                        help='Fraction of points when sampling partial point cloud')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.gpu))
    ##### load data #####
    max_angle = 45
    max_t = 1.0
    partial_source = False

    ##### load data #####
    trainset = RegistrationkittiData(Kitti(split='train', type=args.type), sample_point_num=2048, is_testing=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    testset = RegistrationkittiData(Kitti(split='test', type=args.type), sample_point_num=2048, is_testing=False)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=4)

    # testset = Registration3dmatchData(ThreeDMatch(split='val'), sample_point_num=2048, max_angle=max_angle,
    #                                   max_t=max_t, unseen=unseen, noise=noise,single=single,is_testing=True)
    # test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=numworkers)
    #
    # testset= RegistrationkittiData(Kitti(split='test',type='tracking'), sample_point_num=2048 ,max_angle=max_angle,
    #                                   max_t=max_t, unseen=unseen, noise=noise,single=single,is_testing=True)
    # test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=4)


    ##### load model #####
    if args.emb_nn == 'GNN':
        net = IDAM(GNN(args.emb_dims), args).to(device)
    elif args.emb_nn == 'FPFH':
        args.emb_dims == 33
        net = IDAM(FPFH(), args).to(device)
    ##### load model #####

    ##### train #####
    train(args, net, train_loader, test_loader)
    ##### train #####


if __name__ == '__main__':
    main()
