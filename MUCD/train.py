
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

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    best_iou = 0
    if args.resume:
        checkpoint = torch.load(str(args.experiment_path) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print_log('Use pretrain model',logger=log)
        best_iou = checkpoint['class_avg_iou']
        # best_iou = 39.0
        print_log('bestmIOU:%f'%(best_iou),logger=log)
    else:
        classifier = classifier.apply(weights_init)
    
    classifier = classifier.apply(weights_init)
    
    if cargs.optimizer == 'Adam':
        optimizerC = torch.optim.Adam(
            classifier.parameters(),
            lr=cargs.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cargs.decay_rate
        )
    else:
        optimizerC = torch.optim.SGD(classifier.parameters(), lr=cargs.learning_rate, momentum=0.9)
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = cargs.step_size

    loss_sim = NetLoss().cuda()

    best_accuracy = 0.0
    global_epoch = 0
    

    
    total_trainable_params = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad)
    total_trainable_paramsC = sum(
        p.numel() for p in classifier.parameters() if p.requires_grad)

    print("params")
    print(total_trainable_params+total_trainable_paramsC)

    for epoch in range(start_epoch, 101):
        log.info('EPOCH %i / %i',epoch, 100)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0
        loss_value = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)


        lr = max(cargs.learning_rate * (cargs.lr_decay ** ((epoch-40) // cargs.step_size)), LEARNING_RATE_CLIP)
        print("lr:%f"%(lr))
        for param_group in optimizerC.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** ((epoch-40) // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()
        base_model.zero_grad()
        classifier.zero_grad()
        with Ctq(train_dataloader) as train_loader:
            for idx, data in enumerate(train_loader):
                
                n_itr = epoch * 2 * n_batches
                data_time.update(time.time() - batch_start_time)

                data0, data1, _, _, _ = data


                
                points0 = data0['xyz'].cuda()
                
                points1 = data1['xyz'].cuda()
                
                num_iter += 1
                n_itr += 1
                # remote
                    
                
                xofy = data0['knearst_idx_in_another_pc'].cuda()
                yofx = data1['knearst_idx_in_another_pc'].cuda()
                
                # print(xofy.shape)
                if(idx%3==1):
                    points1 = points1[[1,0],:,:]#交换batch
                    xofy = [None]*2
                    yofx = [None]*2
                    for i in range(points0.shape[0]):#重新计算k近邻
                        point0 = points0[i].cpu().numpy()
                        point1 = points1[i].cpu().numpy()
                        xofy[i] = search_k_neighbors(point1,point0,k)
                        yofx[i] = search_k_neighbors(point0,point1,k)
                    xofy = torch.from_numpy(numpy.stack(xofy,axis=0)).cuda()
                    yofx = torch.from_numpy(numpy.stack(yofx,axis=0)).cuda()
                    
                    points0 = train_transforms(points0)
                    points1 = train_transforms(points1)
                    x_feature = base_model(pts=points0, eval=True)
                    y_feature = base_model(pts=points1, eval=True)
                    x_plane_map = data0['is_plane'].cuda()
                    y_plane_map = data1['is_plane'].cuda()
                    # print(y_plane_map)
                    y_plane_map = y_plane_map[[1,0],:,:]#交换
                    
                    nearest_featuresx = gather_neighbour(y_feature,xofy).mean(-1)
                    nearest_featuresy = gather_neighbour(x_feature,yofx).mean(-1)
                    # print(y_plane_map)
                    # print(x_plane_map.shape)

                    loss_remotex,_ = loss_sim(x_feature,nearest_featuresx,x_plane_map)#地面点map为1，拉远操作时掩盖掉*0
                    loss_remotey,_ = loss_sim(y_feature,nearest_featuresy,y_plane_map)
                    
                    loss_remotex = (-loss_remotex).exp()
                    loss_remotey = (-loss_remotey).exp()  
                    loss = ( loss_remotex + loss_remotey )/2
                elif (idx%3==2):
                    if epoch<=40:#100epoch之前拉进操作使用地面为不变点
                        x_plane_map = data0['is_plane'].cuda()
                        y_plane_map = data1['is_plane'].cuda()
                        seg_pred0 = 1-x_plane_map
                        seg_pred1 = 1-y_plane_map
                    else:
                        xofy = data0['knearst_idx_in_another_pc'].cuda()
                        yofx = data1['knearst_idx_in_another_pc'].cuda()
                        knearest_idx = [xofy, yofx]
                        seg_pred0, seg_pred1 = classifier(points0,points1,knearest_idx)
                    

                    points0 = train_transforms(points0)
                    points1 = train_transforms(points1)
                    x_feature = base_model(pts=points0, eval=True)
                    y_feature = base_model(pts=points1, eval=True)#B,C,N
                    nearest_featuresx = gather_neighbour(y_feature,xofy).mean(-1)
                    nearest_featuresy = gather_neighbour(x_feature,yofx).mean(-1)#B,C,N
                    # print(x_plane_map.shape)#B,1,N
                    # print(seg_pred0.shape)
                    # print(seg_pred0)
                    loss_remotex,l1_lossx = loss_sim(x_feature,nearest_featuresx,seg_pred0)#变化点seg接近为1，拉近操作时权重小
                    loss_remotey,l1_lossy = loss_sim(y_feature,nearest_featuresy,seg_pred1)
                    loss = ( loss_remotex + loss_remotey )/4
                    netloss = ( loss_remotex + loss_remotey + 0.4 * l1_lossx + 0.4 * l1_lossy)/4
                    
                else:
                    points0 = train_transforms(points0)
                    loss0 = base_model(points0)
                    points1 = train_transforms(points1)
                    loss1 = base_model(points1)
                    loss = (loss0 + loss1)/2
                    # loss = loss0
                    
                
                
                
                loss = loss / 3
                loss_value += loss.item()
                try:
                    loss.backward(retain_graph=True)
                except:
                    loss = loss.mean()
                    loss.backward(retain_graph=True)
                
            

                # forward
                if num_iter == 3:
                    # print(loss_value)
                    losses.update([loss_value * 1000])
                    loss_value = 0
                    num_iter = 0
                    if epoch<=40:#只训练特征提取
                        optimizer.step()
                    elif epoch>40 and epoch<=50:#只训练掩码
                        optimizerC.zero_grad()
                        netloss.backward()
                        optimizerC.step()
                    else:
                        
                        optimizerC.zero_grad()
                        
                        print_log("NetLoss:%f"% (netloss.item()))
                        netloss.backward()
                        optimizer.step()
                    
                        
                        optimizerC.step()
                    optimizer.zero_grad()
                

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 20 == 0 and idx != 0:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=log)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=log)
        
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=log)
        if epoch % 10 == 0 :
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                        args,
                                        logger=log)
        if epoch <= 200:
            continue
        with torch.no_grad():
            num_batches = len(test_dataloader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(3)]
            total_correct_class = [0 for _ in range(3)]
            total_iou_deno_class = [0 for _ in range(3)]
            classifier = classifier.eval()

            print_log('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1),logger=log)
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
                data0, data1, _, _, _ = data
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
            
            print_log('Save model...',logger=log)
            savepath = str(args.experiment_path) + '/checkpoints/model40.pth'
            print_log('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

            print(mIoU)
            print(best_iou)
            print(mIoU >= best_iou)
            if mIoU >= best_iou:
                best_iou = mIoU
                print_log('Save model...',logger=log)
                savepath = str(args.experiment_path) + '/checkpoints/best_model40.pth'
                print_log('Saving at %s' % savepath,logger=log)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': best_iou,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                print_log('Saving model....',logger=log)
            print_log('Best mIoU: %f' % best_iou,logger=log)
        
        global_epoch += 1        #                    ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=log)

        

   
    return 0






if __name__ == '__main__':
    main()

