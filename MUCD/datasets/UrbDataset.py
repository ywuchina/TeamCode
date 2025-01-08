import os
import torch
import os.path as osp
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils
from tqdm import tqdm
import time
import configs as cfg
tcfg = cfg.CONFIGS['Train']
class UrbCDDataset(Dataset):
    def __init__(self, data_path, txt_path, n_samples, flag, ppdata_path,):
        super(UrbCDDataset, self).__init__()
        self.data_path = data_path
        self.txt_path = txt_path
        self.n_samples = n_samples
        self.flag = flag
        self.ppdata_path = ppdata_path
        
        with open(self.txt_path, 'r') as f:
            self.list = f.readlines()
            self.file_size = len(self.list)
            
    def __getitem__(self, idx):
        dir_name = self.list[idx].strip()
        ppdata = np.load(os.path.join(self.ppdata_path, tcfg.sub_dataset, self.flag, str(dir_name)+'.npy'), allow_pickle=True)
        inputs16, inputs20, dir_name, pc0_name, pc1_name, [p16_raw, p20_raw] = ppdata
        
        return inputs16, inputs20, dir_name, pc0_name, pc1_name, [p16_raw, p20_raw]
        
    def __len__(self):
        return self.file_size
    
if __name__ == '__main__':
    
    test_data = UrbCDDataset(tcfg.path.test_dataset_path, tcfg.path.test_txt, tcfg.n_samples, 'Test', tcfg.path.prepare_data)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)
    for i, data in enumerate(test_dataloader):
        data0, data1, _, _, _, _ = data

        print(data0['xyz'][0].shape, 
              data0['nearst_idx_in_another_pc'][0].shape, 
               
              data0['label'].shape, 
              data0['raw_length'])
    
        print(data1['xyz'][0].shape, 
              data1['nearst_idx_in_another_pc'][0].shape, 
               
              data1['label'].shape, 
              data1['raw_length'])
        xyz = data1['xyz']
        for j in range(2):
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