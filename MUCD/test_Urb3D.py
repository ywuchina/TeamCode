#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import utils as utl

import configs as cfg
import metrics as mc
from models.pointnet2_sem_seg import SiamPointNet2
from datasets.UrbDataset import UrbCDDataset
from tqdm import tqdm





def test_network(tcfg):
    test_txt = tcfg.path['test_txt']
    test_data = UrbCDDataset(tcfg.path['test_dataset_path'], tcfg.path['test_txt'], tcfg.n_samples, 'Test', tcfg.path.prepare_data)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    

    # resume_path = os.path.join('outputs_1-Lidar05_TrainLarge-1c/best_model.pth')
    # # resume_path = os.path.join('outputs_1-Lidar05_TrainLarge-1c/counterpart.pth')

    
  
   
    # classifier = SiamPointNet2().cuda()

    # checkpoint = torch.load(resume_path)
    # classifier.load_state_dict(checkpoint['model_state_dict'])
    # print('Use pretrain model')
    # best_iou = checkpoint['class_avg_iou']
  
    # print('bestmIOU:%f'%(best_iou))

    with torch.no_grad():
    #     num_batches = len(test_dataloader)
    #     total_correct = 0
    #     total_seen = 0
    #     loss_sum = 0
    #     total_seen_class = [0 for _ in range(3)]
    #     total_correct_class = [0 for _ in range(3)]
    #     total_iou_deno_class = [0 for _ in range(3)]
    #     classifier = classifier.eval()
    #     for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
    #         data0, data1, dir_name, pc0_name, pc1_name,raw_data = data
    #         points0 = data0['xyz'].cuda()
    #         points1 = data1['xyz'].cuda()
            
    #         p0_length = data0['raw_length'].cuda()
    #         p1_length = data1['raw_length'].cuda()
    #         p0_length[p0_length>8192] = 8192
    #         p1_length[p1_length>8192] = 8192
    #         # print(p0_length)
    #         # print(p1_length)
            
    #         xofy = data0['knearst_idx_in_another_pc'].cuda()
    #         yofx = data1['knearst_idx_in_another_pc'].cuda()
    #         knearest_idx = [xofy, yofx]
    #         seg_mask0, seg_mask1 = classifier(points0,points1,knearest_idx)

    #         seg_pred0 = torch.zeros_like(seg_mask0).cuda().long()
    #         seg_pred1 = torch.zeros_like(seg_mask1).cuda().long()
    #         seg_pred0[seg_mask0>0.5] = 1
    #         seg_pred1[seg_mask1>0.5] = 1
    #         lables_batch0 = data0['label'].cuda()
    #         lables_batch1 = data1['label'].cuda()


    #         # print(lables_batch0.shape)
    #         # print(seg_pred0.shape)
    #         # print(lables_batch0.dtype)
    #         # print(seg_pred0.dtype)

            
    #         if tcfg.save_prediction:
    #             utl.save_prediction2(raw_data[0], raw_data[1], 
    #                                     lables_batch0.squeeze(-1), lables_batch1.squeeze(-1), 
    #                                     seg_pred0.squeeze(1), seg_pred1.squeeze(1), 
    #                                     os.path.join(tcfg.path['test_prediction_PCs'], str(dir_name[0])),
    #                                     pc0_name, pc1_name,
    #                                     tcfg.path['test_dataset_path'])


    #         pred_val0 = seg_pred0.view(-1, 1)
    #         pred_val1 = seg_pred1.view(-1, 1)

    #         pred0 = pred_val0.cpu().numpy()
    #         pred1 = pred_val1.cpu().numpy()

    #         # if p0_length.shape[0]==2:
    #         #     pred0 = np.vstack((pred0[:p0_length[0]],pred0[8192:8192+p0_length[1]]))
    #         #     pred1 = np.vstack((pred1[:p1_length[0]],pred1[8192:8192+p1_length[1]]))
    #         # else:
    #         #     pred0 = pred0[:p0_length[0]]
    #         #     pred1 = pred1[:p1_length[0]]
    #         # print(pred0.shape)

    #         pred_valid = np.vstack((pred0, pred1*2))#0不变,1移除,2新增
            
    #         lb0 = lables_batch0.view(-1, 1).cpu().numpy()
    #         lb1 = lables_batch1.view(-1, 1).cpu().numpy()

    #         # if p0_length.shape[0]==2:
    #         #     lb0 = np.vstack((lb0[:p0_length[0]],lb0[8192:8192+p0_length[1]]))
    #         #     lb1 = np.vstack((lb1[:p1_length[0]],lb1[8192:8192+p1_length[1]]))
    #         # else:
    #         #     lb0 = lb0[:p0_length[0]]
    #         #     lb1 = lb1[:p1_length[0]]
    #         # print(lb0.shape)


    #         labels_valid = np.vstack((lb0, lb1*2))
    #         # print(labels_valid.shape)
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
        
    #     print('eval point nochangeIou: %f' % (nochangeIou))
    #     print('eval point removeIou: %f' % (removeIou))
    #     print('eval point addIou: %f' % (addIou))

    #     print('eval point avg class IoU: %f' % (mIoU))
    #     print('eval point oa: %f' % (total_correct / float(total_seen)))
        utl.combine_sub_pcs(tcfg.path.test_prediction_PCs)
    # torch.no_grad()
    # net.eval()
    # dur = 0
    # iou_calc = mc.IoUCalculator()
    # tqdm_loader = tqdm(test_dataloader, total=len(test_dataloader))
    # for _, data in enumerate(tqdm_loader):
    #     batch_data0, batch_data1, dir_name, pc0_name, pc1_name, raw_data = data   
    #     p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx, lb0, knearest_idx0, raw_length0 = [i for i in batch_data0.values()]
    #     p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx, lb1, knearest_idx1, raw_length1 = [i for i in batch_data1.values()]
    #     p0 = [i.to(device, dtype=torch.float) for i in p0]
    #     p0_neighbors_idx = [i.to(device, dtype=torch.long) for i in p0_neighbors_idx]
    #     p0_neighbors_idx = [i.to(device, dtype=torch.long) for i in p0_neighbors_idx]
    #     p0_pool_idx = [i.to(device, dtype=torch.long) for i in p0_pool_idx]
    #     p0_unsam_idx = [i.to(device, dtype=torch.long) for i in p0_unsam_idx]
    #     p1 = [i.to(device, dtype=torch.float) for i in p1]
    #     p1_neighbors_idx = [i.to(device, dtype=torch.long) for i in p1_neighbors_idx]
    #     p1_pool_idx = [i.to(device, dtype=torch.long) for i in p1_pool_idx]
    #     p1_unsam_idx = [i.to(device, dtype=torch.long) for i in p1_unsam_idx]
    #     knearest_idx = [knearest_idx0.to(device, dtype=torch.long), knearest_idx1.to(device, dtype=torch.long)]
        
    #     lb0 = lb0.squeeze(-1).to(device, dtype=torch.long)
    #     lb1 = lb1.squeeze(-1).to(device, dtype=torch.long)
    #     t0 = time.time()
    #     out0, out1 = net([p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx],
    #                           [p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx], 
    #                           knearest_idx)
    #     dur += time.time()-t0
    #     out0 = out0.max(dim=-1)[1]; out1 = out1.max(dim=-1)[1];
        
    #     iou_calc.add_data(out0.squeeze(0), out1.squeeze(0), lb0.squeeze(0), lb1.squeeze(0))
        
            
    # iou = iou_calc.metrics()
    # for k, v in iou.items():
    #     print(k, v)    
    # with open(os.path.join(tcfg.path['outputs'], 'test_IoU.txt'),'a') as f:
    #     f.write('Time:{},miou:{:.6f},oa:{:.6f},iou_list:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
    #             iou['miou'], iou['oa'], iou['iou_list']))
    #     f.write('\n')   
    # print('FPS: ', len(test_dataloader)/dur)
    # utl.combine_sub_pcs(tcfg.path.test_prediction_PCs)
    
if __name__ == '__main__':
    
    tcfg = cfg.CONFIGS['Train']
    

        
    test_network(tcfg)
    
    
    
    
    