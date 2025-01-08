import os
import torch
import os.path as osp
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils
from tqdm import tqdm
import time

class CDDataset(Dataset):
    def __init__(self, data_path, txt_path, n_samples, flag, ppdata_path):
        super(CDDataset, self).__init__()
        
        self.txt_path = txt_path
        self.n_samples = n_samples
        self.flag = flag
        self.ppdata_path = ppdata_path
        # if self.flag=='train' or self.flag=='test':
        #     if tcfg.remove_plane:
        #         if self.flag == 'train':
        #             generate_removed_plane_dataset_train(tcfg.path.data_root)
        #         if self.flag == 'test':
        #             generate_removed_plane_dataset_test(tcfg.path.data_root)
        #     if tcfg.if_prepare_data:
        #         if self.flag == 'train':
        #             prepare_data_train()
        #         if self.flag == 'test':
        #             prepare_data_test()
        with open(self.txt_path, 'r') as f:
            self.list = f.readlines()
            self.file_size = len(self.list)
            
    def __getitem__(self, idx):
        
        ppdata = np.load(os.path.join(self.ppdata_path, self.flag, str(idx)+'.npy'), allow_pickle=True)
        inputs16, inputs20, dir_name, pc0_name, pc1_name = ppdata
        return inputs16, inputs20, dir_name, pc0_name, pc1_name
        
    def __len__(self):
        return self.file_size
    
if __name__ == '__main__':
    
    train_data = CDDataset(tcfg.path['train_dataset'], tcfg.path['train_txt'], tcfg.n_samples, 'train', tcfg.path.prepare_data)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False)
    for i, data in enumerate(train_dataloader):
        data0, data1, _, _, _, _ = data

        print(data0['xyz'][0].shape, 
              data0['neighbors_idx'][0].shape, 
              data0['pool_idx'][0].shape, 
              data0['unsam_idx'][0].shape, 
              data0['label'].shape, 
              data0['raw_length'])
        xyz = data0['xyz']
        for j in range(5):
            print(xyz[j].shape)
        break
"""
output:
torch.Size([2, 8192, 3]) torch.Size([2, 8192, 16]) torch.Size([2, 2048, 16]) torch.Size([2, 8192, 1]) torch.Size([2, 8192, 1]) tensor([8192, 8192])
torch.Size([2, 8192, 3])
torch.Size([2, 2048, 3])
torch.Size([2, 512, 3])
torch.Size([2, 128, 3])
torch.Size([2, 32, 3])

"""