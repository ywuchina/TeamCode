from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from thop import profile
from dataset.ModelNet40 import ModelNet40
from dataset.ScanObjectNN import ScanObjectNN
from models.classification.model_new import Pct_cls1, Pct_cls2, Pct_cls3, Pct_cls4, Pct_cls5, Pct_cls6, Pct_cls7, \
    Pct_cls8, Pct_cls9, Pct_cls10, Pct_cls11, Pct_cls12, Pct_cls13, Pct_cls14, Pct_cls15, Pct_cls16, Pct_cls17, \
    Pct_cls18, Pct_cls19, Pct_cls20, Pct_cls21, Pct_cls22, Pct_cls23, Pct_cls24, Pct_cls25, Pct_cls26, Pct_cls27, \
    Pct_cls28, Pct_cls29
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

import time 

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='train', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40','ScanObjectNN'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args()

def train(args, io):
    output_channels = 0
    if args.dataset == 'modelnet40':
        output_channels = 40
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=4,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'ScanObjectNN':
        output_channels = 15
        train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("Dataset Not supported")
    # model = Pct_cls(args).to(device)
    model = Pct_cls29(output_channels=output_channels).to(device)
    io.cprint(str(model))

    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(args.model_path))
    opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=0.9, weight_decay=1e-4)
    # opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        count = 0
        model.train()
        train_pred = []
        train_true = []
        total_time = 0.0
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device).squeeze() 
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        per_class_acc = recall_score(train_true, train_pred, average=None)
        outstr = 'Epoch: %d, Learning Rate: %s \ntrain loss: %.6f, train acc: %.6f, train avg acc: %.6f' \
                 % (epoch,opt.param_groups[0]['lr'],train_loss*1.0/count,
                    metrics.accuracy_score(train_true, train_pred),
                    metrics.balanced_accuracy_score(train_true, train_pred))
        io.cprint(outstr)
        print('train loss=>',train_loss,'count=>',count,'total time=>',total_time)
        print('train_per_class_acc=>', per_class_acc)

        #################
        # Test
        #################
        test_loss = 0.0
        count = 0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        per_class_acc = recall_score(test_true, test_pred, average=None)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'checkpoints/classification/{args.exp_name}_{args.scheduler}/models/model_{args.num_points}.t7')

        outstr = 'test loss: %.6f, test acc: %.6f, test avg acc: %.6f, best_test_acc: %.6f' \
                 % (test_loss * 1.0 / count, test_acc, avg_per_class_acc, best_test_acc)
        io.cprint(outstr)
        print('test loss=>', test_loss, 'count=>', count, 'total time=>',total_time)
        print('test_per_class_acc=>', per_class_acc)
    
def test(args, io):
    output_channels = 0
    if args.dataset == 'modelnet40':
        output_channels = 40
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'ScanObjectNN':
        output_channels = 15
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("Dataset Not supported")

    model = Pct_cls20(output_channels = output_channels).to(device)
    # model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for i,(data,label) in enumerate(tqdm(test_loader)):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)

        flops, params = profile(model, (data,))
        print('flops: %.2f M, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

        logits = model(data) #[B, 40]
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    per_class_acc = recall_score(test_true, test_pred, average=None)

    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f  \n %s'%(test_acc, avg_per_class_acc,per_class_acc)
    io.cprint(outstr)

if __name__ == "__main__":

    args = parse_args()
    if not os.path.exists('checkpoints/classification/' + args.exp_name + '_' + args.scheduler +'/models'):
        os.makedirs('checkpoints/classification/' + args.exp_name + '_' + args.scheduler + '/models')

    device = torch.device("cuda:1")

    io = IOStream('checkpoints/classification/' + args.exp_name + '_' + args.scheduler + f'/run_{args.num_points}.log')
    io.cprint(str(args))

    args.cuda = True
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint('Using GPU')
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        with torch.no_grad():
            test(args, io)
