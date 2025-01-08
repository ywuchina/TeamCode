
import argparse
import os
import time
from torch.utils.data import Dataset, DataLoader
import numpy
from torch import nn
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from tools.runner_pretrain import Acc_Metric
from utils.AverageMeter import AverageMeter
from utils.config import *
import hydra
import torch
import datetime
from utils.logger import *
from utils.tools_fun import search_k_neighbors
import sys
import importlib
import shutil
from tools import pretrain_run_net as pretrain, builder
from tools import test_svm_run_net_modelnet40 as test_svm_modelnet40
from tools import test_svm_run_net_scan as test_svm_scan
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from torch.utils.tensorboard import SummaryWriter
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
import logging
from datasets.shrec21cd import CDDataset
from torchvision import transforms
from datasets import data_transforms
from reference1 import gather_neighbour
from torch_geometric.nn import knn
train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)
sys.path.append("./")



def pc_normalize(pc):
    
    pc = pc.cpu().numpy()
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return torch.from_numpy(pc).cuda()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

k = 8

class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        # self.mse = nn.MSELoss()
        # self.loss_restruct = nn.MSELoss()
        self.loss_restruct = nn.L1Loss()   

    def forward(self, target, restruct, cmap):
       
        num_point = target.size()[2] 
        num_wnc = torch.sum(1 - cmap, (1, 2))

        # print(cmap)
        # print(1 - cmap.repeat(1, target.size()[1], 1))
        target_mask = target * (1 - cmap.repeat(1, target.size()[1], 1))
        restruct_mask = restruct * (1 - cmap.repeat(1, restruct.size()[1], 1))
        
        restruct_loss = 0
        for i in range(target.shape[0]):
            restruct_loss += self.loss_restruct(target_mask[i], restruct_mask[i]) * num_point / (num_wnc[i] +  1e-7)
            
        restruct_loss = restruct_loss / target.shape[0]

        # if generator_mask_switch is True, the cmp will be translated to binary change mask
        l1_loss = torch.mean(abs(cmap))
        

        return restruct_loss, l1_loss

def save_prediction(p0, p1, lb0, lb1, pred0, pred1, path, p0_name, p1_name):
    if os.path.exists(path):
        print("已保存")
        return
    else:
        os.mkdir(path)
    

    p0 = np.hstack((p0, lb0, pred0))
   
    p1 = np.hstack((p1, lb1, pred1))
    
    
    
    np.savetxt(os.path.join(path, p0_name), p0, fmt="%.8f %.8f %.8f %.0f %.0f")
    np.savetxt(os.path.join(path, p1_name), p1, fmt="%.8f %.8f %.8f %.0f %.0f")    
    
    head = '//X Y Z label prediction\n'
    with open(os.path.join(path,p0_name), 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(head + (str(len(p0))+'\n') + content)
    with open(os.path.join(path,p1_name), 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(head + (str(len(p1))+'\n') + content)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='models.pointnet2_sem_seg', help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()


def main():



    args = parser.get_args()

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    log = get_root_logger(log_file=log_file, name=args.log_name)

    config = get_config(args, logger = log)

    #dataset
    args.prepare_data = args.data_root + '/prapared_data_' + str(args.n_samples) + '_k' + str(k)
    train_data = CDDataset(args.datapath, args.txtpath, args.n_samples, 'train', args.prepare_data)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_data = CDDataset(args.test_datapath, args.test_txtpath, args.n_samples, 'test', args.prepare_data)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)
    val_data = CDDataset(args.datapath, args.val_txtpath, args.n_samples, 'val', args.prepare_data)
    val_dataloader = DataLoader(val_data, batch_size=2, shuffle=True,drop_last=True)

    #model
    base_model = builder.model_builder(config.model).cuda()
    

    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size


    if args.use_gpu:
        base_model.cuda()


    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    args.resume_path = os.path.join('./experiments/pre-training/point-m2ae/defaultk8/')

    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = log)
        best_metrics = Acc_Metric(best_metric)

        # DDP

    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            log.info('Using Synchronized BatchNorm ...')
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        log.info('Using Distributed Data parallel ...')
    else:
        log.info('Using Data parallel ...')
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=log)

    
    
    #classfier
    cargs = parse_args()
    '''classifier LOADING'''
    MODEL = importlib.import_module(cargs.model)
    # shutil.copy('models/%s.py' % cargs.model, str(args.experiment_path))
    # shutil.copy('models/pointnet2_utils.py', str(args.experiment_path))

    classifier = MODEL.SiamPointNet2().cuda()
    classifier.apply(inplace_relu)

   
    best_iou = 0
    # args.resume:
    checkpoint = torch.load(str(args.experiment_path) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    print_log('Use pretrain model',logger=log)
    best_iou = checkpoint['class_avg_iou']
    # best_iou = 
    print_log('bestmIOU:%f'%(best_iou),logger=log)
   
    

    # loss_sim = NetLoss().cuda()

    best_accuracy = 0.0
    global_epoch = 0
    
    # with torch.no_grad():
    #     num_batches = len(val_dataloader)
    #     total_correct = 0
    #     total_seen = 0
    #     loss_sum = 0
    #     total_seen_class = [0 for _ in range(3)]
    #     total_correct_class = [0 for _ in range(3)]
    #     total_iou_deno_class = [0 for _ in range(3)]
    #     classifier = classifier.eval()

    #     for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), smoothing=0.9):
    #         data0, data1, dir_name, pc0_name, pc1_name = data
    #         points0 = data0['xyz'].cuda()
    #         points1 = data1['xyz'].cuda()
            
    #         xofy = data0['knearst_idx_in_another_pc'].cuda()
    #         yofx = data1['knearst_idx_in_another_pc'].cuda()
    #         knearest_idx = [xofy, yofx]
    #         seg_mask0, seg_mask1 = classifier(points0,points1,knearest_idx)

    #         seg_pred0 = torch.zeros_like(seg_mask0).cuda()
    #         seg_pred1 = torch.zeros_like(seg_mask1).cuda()
    #         seg_pred0[seg_mask0>0.5] = 1
    #         seg_pred1[seg_mask1>0.5] = 1
    #         lables_batch0 = data0['label'].cuda().view(-1, 1)
    #         lables_batch1 = data1['label'].cuda().view(-1, 1)

    #         pred_val0 = seg_pred0.view(-1, 1)
    #         pred_val1 = seg_pred1.view(-1, 1)

    #         pred0 = pred_val0.cpu().numpy()
    #         pred1 = pred_val1.cpu().numpy()
    #         pred_valid = np.vstack((pred0, pred1*2))#0不变,1移除,2新增
            
    #         lb0 = lables_batch0.cpu().numpy()
    #         lb1 = lables_batch1.cpu().numpy()
    #         labels_valid = np.vstack((lb0, lb1*2))


    #         points0 = points0.cpu().numpy()
    #         points1 = points1.cpu().numpy()
            
           
    #         # save_prediction(points0[0],points1[0],
    #         #                 lb0[:8192,:],lb1[:8192,:],
    #         #                 pred0[:8192,:],pred1[:8192,:],
    #         #                 os.path.join(args.save_txt_path, str(dir_name[0])),
    #         #                 pc0_name[0],pc1_name[0])

    #         # save_prediction(points0[1],points1[1],
    #         #                 lb0[8192:,:],lb1[8192:,:],
    #         #                 pred0[8192:,:],pred1[8192:,:],
    #         #                 os.path.join(args.save_txt_path, str(dir_name[1])),
    #         #                 pc0_name[1],pc1_name[1])
            
    #         correct = np.sum(pred_valid == labels_valid)
    #         total_correct += correct
    #         total_seen += len(labels_valid)
    #         for l in range(3):#0不变,1移除,2新增
    #             total_correct_class[l] += np.sum((pred_valid == l) & (labels_valid == l))
    #             total_iou_deno_class[l] += np.sum(((pred_valid == l) | (labels_valid == l)))

    #     nochangeIou = total_correct_class[0] / (total_iou_deno_class[0] + 1e-6)
    #     removeIou = total_correct_class[1] / (total_iou_deno_class[1] + 1e-6)
    #     addIou = total_correct_class[2] / (total_iou_deno_class[2] + 1e-6)
    #     mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        
    #     print_log('eval point nochangeIou: %f' % (nochangeIou),logger=log)
    #     print_log('eval point removeIou: %f' % (removeIou),logger=log)
    #     print_log('eval point addIou: %f' % (addIou),logger=log)

    #     print_log('eval point avg class IoU: %f' % (mIoU),logger=log)
    #     print_log('eval point oa: %f' % (total_correct / float(total_seen)),logger=log)
   
    with torch.no_grad():
        num_batches = len(test_dataloader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(3)]
        total_correct_class = [0 for _ in range(3)]
        total_iou_deno_class = [0 for _ in range(3)]
        classifier = classifier.eval()

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
            data0, data1, dir_name, pc0_name, pc1_name = data
            points0 = data0['xyz'].cuda()
            points1 = data1['xyz'].cuda()
            
            xofy = data0['knearst_idx_in_another_pc'].cuda()
            yofx = data1['knearst_idx_in_another_pc'].cuda()
            knearest_idx = [xofy, yofx]
            seg_mask0, seg_mask1 = classifier(points0,points1,knearest_idx)

            seg_pred0 = torch.zeros_like(seg_mask0).cuda()
            seg_pred1 = torch.zeros_like(seg_mask1).cuda()
            seg_pred0[seg_mask0>0.5] = 1
            seg_pred1[seg_mask1>0.5] = 1
            lables_batch0 = data0['label'].cuda().view(-1, 1)
            lables_batch1 = data1['label'].cuda().view(-1, 1)

            pred_val0 = seg_pred0.view(-1, 1)
            pred_val1 = seg_pred1.view(-1, 1)

            pred0 = pred_val0.cpu().numpy()
            pred1 = pred_val1.cpu().numpy()
            pred_valid = np.vstack((pred0, pred1*2))#0不变,1移除,2新增
            
            lb0 = lables_batch0.cpu().numpy()
            lb1 = lables_batch1.cpu().numpy()
            labels_valid = np.vstack((lb0, lb1*2))


            points0 = points0.cpu().numpy()
            points1 = points1.cpu().numpy()
            
           
            # save_prediction(points0[0],points1[0],
            #                 lb0[:8192,:],lb1[:8192,:],
            #                 pred0[:8192,:],pred1[:8192,:],
            #                 os.path.join(args.save_txt_path, str(dir_name[0])),
            #                 pc0_name[0],pc1_name[0])

            # save_prediction(points0[1],points1[1],
            #                 lb0[8192:,:],lb1[8192:,:],
            #                 pred0[8192:,:],pred1[8192:,:],
            #                 os.path.join(args.save_txt_path, str(dir_name[1])),
            #                 pc0_name[1],pc1_name[1])
            
            correct = np.sum(pred_valid == labels_valid)
            total_correct += correct
            total_seen += len(labels_valid)
            for l in range(3):#0不变,1移除,2新增
                total_correct_class[l] += np.sum((pred_valid == l) & (labels_valid == l))
                total_iou_deno_class[l] += np.sum(((pred_valid == l) | (labels_valid == l)))

        nochangeIou = total_correct_class[0] / (total_iou_deno_class[0] + 1e-6)
        removeIou = total_correct_class[1] / (total_iou_deno_class[1] + 1e-6)
        addIou = total_correct_class[2] / (total_iou_deno_class[2] + 1e-6)
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        
        print_log('eval point nochangeIou: %f' % (nochangeIou),logger=log)
        print_log('eval point removeIou: %f' % (removeIou),logger=log)
        print_log('eval point addIou: %f' % (addIou),logger=log)

        print_log('eval point avg class IoU: %f' % (mIoU),logger=log)
        print_log('eval point oa: %f' % (total_correct / float(total_seen)),logger=log)
        

    global_epoch += 1        #                    ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=log)

        

   
    return 0






if __name__ == '__main__':
    main()

