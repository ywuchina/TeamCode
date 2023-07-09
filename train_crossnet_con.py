from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18, resnet101
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.data import ShapeNetRender, ModelNet40SVM
from models.dgcnn import DGCNN_cls, DGCNN_partseg
from models.pointnet import PointNet_cls, PointNet_partseg
from models.resnet import ResNet
from util import IOStream, AverageMeter


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn_cls', metavar='N',
                        choices=['dgcnn_cls', 'dgcnn_seg', 'pointnet_cls', 'pointnet_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args()

def train(args, io):
    wandb.init(project="CrossNet1", name=args.exp_name)

    train_loader = DataLoader(ShapeNetRender(), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    #Try to load models
    if args.model == 'dgcnn_cls':
        point_model = DGCNN_cls(args).to(device)
    elif args.model == 'dgcnn_seg':
        point_model = DGCNN_partseg(args).to(device)
    elif args.model == 'pointnet_cls':
        point_model = PointNet_cls(args).to(device)
    elif args.model == 'pointnet_seg':
        point_model = PointNet_partseg(args).to(device)
    else:
        raise Exception("Not implemented")
        
    img_rgb_model = ResNet(resnet50(), feat_dim = 2048)
    img_rgb_model = img_rgb_model.to(device)

    img_gray_model = ResNet(resnet50(), feat_dim = 2048, type = 'gray')
    img_gray_model = img_gray_model.to(device)
        
    wandb.watch(point_model)
    
    if args.resume:
        point_model.load_state_dict(torch.load(args.model_path))
        print("Model Loaded !!")
        
    parameters = list(point_model.parameters()) + list(img_rgb_model.parameters()) + list(img_gray_model.parameters())
    # print(parameters)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(point_model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-5)
    else:
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature = 0.2).to(device)
    
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        train_intra_losses = AverageMeter()
        train_cross_losses = AverageMeter()
        
        point_model.train()
        img_rgb_model.train()
        img_gray_model.train()
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for (data_t1, data_t2), (img_rgb, img_gray) in tqdm(train_loader):
            data_t1, data_t2, img_rgb, img_gray = data_t1.to(device), data_t2.to(device), img_rgb.to(device), img_gray.to(device)
            # print('data_t1=>',data_t1.shape,'data_t2=>',data_t2.shape,'imgs=>',imgs.shape)
            batch_size = data_t1.size()[0]

            opt.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            point_feats, _ = point_model(data)
            img_rgb_feats = img_rgb_model(img_rgb)
            img_gray_feats = img_gray_model(img_gray)

            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size: , :]
            loss_intra = criterion(point_t1_feats, point_t2_feats)
            image_feats = torch.cat([img_rgb_feats, img_gray_feats],dim = 0)

            loss_cross = criterion(point_feats, image_feats)

            total_loss = loss_intra + loss_cross
            total_loss.backward()
            opt.step()
            lr_scheduler.step()

            train_losses.update(total_loss.item(), batch_size)
            train_intra_losses.update(loss_intra.item(), batch_size)
            train_cross_losses.update(loss_cross.item(), batch_size)

        outstr = 'Epoch (%d), Batch(%d), loss: %.6f, intra loss: %.6f, cross loss: %.6f' % \
                 (epoch, len(train_loader), train_losses.avg, train_intra_losses.avg, train_cross_losses.avg)
        io.cprint(outstr)

        wandb_log['Train Loss'] = train_losses.avg
        wandb_log['Train Intra Loss'] = train_intra_losses.avg
        wandb_log['Train Cross Loss'] = train_cross_losses.avg
        
        # Testing
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=64, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=64, shuffle=False)

        feats_train = []
        labels_train = []
        point_model.eval()

        for (data, label) in tqdm(train_val_loader):
            # print('data=>',data.shape,'label=>',label.shape) #[B,num_points,3]
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            # print('labels=>',labels) #[B,1]的label标签转换为一个大小为B的数组
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[-1]
            feats = feats.detach().cpu().numpy()
            # print('feats=>', feats.shape) #[B,2048(max1024+avg1024)]
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)
        # print('feats_train=>',feats_train.shape,'labels_train=>',labels_train.shape) #(9840, 2048),(9840,)

        feats_test = []
        labels_test = []

        for data, label in tqdm(test_val_loader):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[-1]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        
        model_tl = SVC(C = 0.01, kernel ='linear')
        # print('model_tl=>',model_tl)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        # wandb_log['No Fit Linear Accuracy'] = test_accuracy

        model_tl.fit(feats_train, labels_train)
        # print('model_tl=>', model_tl)
        test_accuracy = model_tl.score(feats_test, labels_test)
        wandb_log['Fit Linear Accuracy'] = test_accuracy

        io.cprint(f"Linear Accuracy : {test_accuracy}, Best Accuracy : {best_acc}")
        
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)
            
            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_rgb_model_best.pth')
            torch.save(img_rgb_model.state_dict(), save_img_model_file)
            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                               'img_gray_model_best.pth')
            torch.save(img_gray_model.state_dict(), save_img_model_file)

        wandb.log(wandb_log)
    
    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(point_model.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_rgb_model_last.pth')
    torch.save(img_rgb_model.state_dict(), save_img_model_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_gray_model_last.pth')
    torch.save(img_gray_model.state_dict(), save_img_model_file)

if __name__ == "__main__":
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)

    if not args.eval:
        train(args, io)
    else:
        test(args, io)


