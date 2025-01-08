import os
import torch
import os.path as osp
import random
import numpy as np
import sys
sys.path.append(os.path.abspath(".."))
import configs as cfg
import utils
from tqdm import tqdm
tcfg = cfg.CONFIGS['Train']
import time

def is_plane(points):
    zmin = np.min(points[:, 2])
    nthr = zmin + tcfg.plane_threshold

    return points[:, 2] < nthr
    

def norm_data(p16, p20):
    p16_raw, p20_raw = p16, p20
    # np.savetxt("/home/wzp/DATA/SHREC2020-CD/p16_raw.txt", p16_raw, fmt="%.8f %.8f %.8f %.8f %.8f %.8f")
    # idx = np.where(np.logical_not(np.all(p16 == 0, axis=1)))[0]
    # point_set = p16[idx, :]
    # np.savetxt("/home/wzp/DATA/SHREC2020-CD/point_set.txt", point_set, fmt="%.8f %.8f %.8f %.8f %.8f %.8f")
    point_pair = np.vstack((p16, p20))
    # idx = np.where(point_pair==0)
    idx = np.where(np.logical_not(np.all(point_pair == 0, axis=1)))[0]
    point_set = point_pair[idx, : 3]
    

    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist   # scale
    
    point_pair[idx, : 3] = point_set
    p16 = point_pair[:tcfg.n_samples,:]
    p20 = point_pair[tcfg.n_samples:,:]

    # np.savetxt("/home/wzp/DATA/SHREC2020-CD/p16.txt", p16, fmt="%.8f %.8f %.8f %.8f %.8f %.8f")
    return p16, p20, p16_raw, p20_raw


class PrepareData():
    def __init__(self, flag, data_path, txt_path, n_samples):
        super(PrepareData, self).__init__()
        self.flag = flag
        self.data_path = data_path
        self.txt_path = txt_path
        self.n_samples = n_samples
        with open(self.txt_path, 'r') as f:
            self.list = f.readlines()
            self.file_size = len(self.list)
            
    def load_and_pp_data(self):
        for idx in tqdm(range(self.file_size)):
            dir_name = self.list[idx].strip()
            query_path = os.path.join(tcfg.path.prepare_data, tcfg.sub_dataset, self.flag, str(dir_name)+'.npy')
            if os.path.exists(query_path):
                continue
            p16_name = 'pointCloud0.txt'
            p20_name = 'pointCloud1.txt'
            p16_path = osp.join(self.data_path, dir_name, p16_name)
            p20_path = osp.join(self.data_path, dir_name, p20_name)
            if not os.path.exists(p16_path):
                p16_path = p20_path
            if not os.path.exists(p20_path):
                p20_path = p16_path

            p16, p20, p16_raw_length, p20_raw_length = utils.align_length(p16_path, p20_path, self.n_samples)
            
            label16, label20 = self.generate_label(p16, p20)
            p16_data = p16[:, :-1]; p20_data = p20[:, :-1]

            

            if tcfg.norm_data:
               p16_data, p20_data, p16_raw, p20_raw = norm_data(p16_data, p20_data)
              
            
            
            
            p16ofp20 = utils.search_k_neighbors(p20_data[:, :3], p16_data[:, :3], 1)
            p20ofp16 = utils.search_k_neighbors(p16_data[:, :3], p20_data[:, :3], 1)
            p16ofp20k = utils.search_k_neighbors(p20_data[:, :3], p16_data[:, :3], tcfg.k_neighbors)
            p20ofp16k = utils.search_k_neighbors(p16_data[:, :3], p20_data[:, :3], tcfg.k_neighbors)
            inputs16 = {}; inputs20 = {}
        
            inputs16['xyz'] = torch.from_numpy(p16_data[:, :3]).float() 
            inputs16['label'] = torch.from_numpy(label16).long()
            inputs16['nearst_idx_in_another_pc'] = torch.from_numpy(p16ofp20).long()
            inputs16['raw_length'] = p16_raw_length
            
            inputs16['knearst_idx_in_another_pc'] = torch.from_numpy(p16ofp20k).long()

            
            inputs20['xyz'] = torch.from_numpy(p20_data[:, :3]).float() 
            inputs20['label'] = torch.from_numpy(label20).long()
            inputs20['nearst_idx_in_another_pc'] = torch.from_numpy(p20ofp16).long()
            inputs20['raw_length'] = p20_raw_length
            
            inputs20['knearst_idx_in_another_pc'] = torch.from_numpy(p20ofp16k).long()

            
            

            ppdata = inputs16, inputs20, dir_name, p16_name, p20_name, [p16_raw, p20_raw]
            save = os.path.join(tcfg.path.prepare_data, tcfg.sub_dataset, self.flag)
            if not os.path.exists(save):
                os.makedirs(save)
            np.save(os.path.join(save, str(dir_name)+'.npy'), ppdata)

        
    def process_data(self, pc_data, subsam_rate=tcfg.sub_sampling_ratio):
        if pc_data.shape[1] == 3:
            xyz = pc_data
        else:
            xyz = pc_data[:, :3]
        input_points = []
        input_points.append(pc_data)
        neighbors_idx = []
        pool_idx = []
        upsam_idx = []
        for i in range(tcfg.num_layers):
            k_neigh_idx = utils.search_k_neighbors(xyz, xyz, tcfg.k_neighbors)
            sub_pc_data = pc_data[:pc_data.shape[0]//subsam_rate[i], :]
            sub_xyz = xyz[:xyz.shape[0]//subsam_rate[i], :]
            sub_idx = k_neigh_idx[:pc_data.shape[0]//subsam_rate[i], :]
            up_idx = utils.search_k_neighbors(sub_xyz, xyz,  1)
            input_points.append(sub_pc_data)
            neighbors_idx.append(k_neigh_idx)
            pool_idx.append(sub_idx)
            upsam_idx.append(up_idx)
            pc_data = sub_pc_data
            xyz = sub_xyz
        inputs_list = [input_points, neighbors_idx, pool_idx, upsam_idx]
        
        return inputs_list
    
    def generate_label(self, p16, p20):        
        label16 = np.expand_dims(p16[:, -1], 1)
        label20 = np.expand_dims(p20[:, -1], 1)
        return label16, label20


def prepare_data():
    
    pptrain = PrepareData(tcfg.train_flag, tcfg.path['train_dataset_path'], tcfg.path['train_txt'], tcfg.n_samples)
    pptest = PrepareData(tcfg.test_flag, tcfg.path['test_dataset_path'], tcfg.path['test_txt'], tcfg.n_samples)
    ppval = PrepareData(tcfg.val_flag, tcfg.path['val_dataset_path'], tcfg.path['val_txt'], tcfg.n_samples)
    # print('processing training data......')
    # pptrain.load_and_pp_data()
    print('processing testing data......')
    pptest.load_and_pp_data()
    print('processing val data......')
    ppval.load_and_pp_data()
    print('data preparation finished.')
    
if __name__ == '__main__':
    prepare_data()