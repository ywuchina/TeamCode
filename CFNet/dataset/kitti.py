import os
from os.path import join, exists
import pickle
import glob
import random
import torch
import torch.utils.data as data
import numpy as np
from operations.visualization import *
from operations.transform_functions import PCRNetTransform
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home/ljm'

class Kitti(data.Dataset):
    def __init__(self,split,type = 'object'):
        self.type = type
        self.root = os.path.join(BASE_DIR,f'data/kitti_{self.type}')
        # print(self.root) /home/ljm/data/kitti_object
        self.split = split
        DATA_FILES = {
            'train': os.path.join(self.root,'training/velodyne'),
            'test':os.path.join(self.root,'testing/velodyne'),
        }
        self.files = []
        self.length =0
        if type == 'object':
            fnames_txt = glob.glob(os.path.join(DATA_FILES[split],"*.bin"))
        else:
            fnames_txt = glob.glob(os.path.join(DATA_FILES[split], "*","*.bin"))
        for i,fname_txt in enumerate(fnames_txt):
            self.files.append(fname_txt)
            self.length += 1
            
    def __getitem__(self, index):
        template_file= self.files[index]
        template_data = np.fromfile(os.path.join(self.root,template_file),dtype=np.float32).reshape(-1, 4)
        template_pcd =template_data[:, :3]
        return torch.from_numpy(template_pcd).type(torch.float)

    def __len__(self):
        return self.length

class RegistrationkittiData(data.Dataset):
    def __init__(self, data_class,sample_point_num=2048,is_testing=False):
        super(RegistrationkittiData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        self.sample_point_num =sample_point_num
        self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

    def __getitem__(self, index):
        template_pcd = self.data_class[index]

        self.transforms.index = index
        source_pcd = self.transforms(template_pcd)
        igt = self.transforms.igt
        N_tmp =template_pcd.shape[0]
        if N_tmp<self.sample_point_num:
            return self.__getitem__(index+1)
        sel_ind = np.random.choice(int(N_tmp), self.sample_point_num)
        sample_template_pcd =template_pcd[sel_ind,:]
        sample_source_pcd  =source_pcd[sel_ind,:]
        # visual_3dmatch_pcd(sample_template_pcd.detach().cpu().numpy(),sample_source_pcd.detach().cpu().numpy())
        if self.is_testing:
            # visual_3dmatch_pcd(template_pcd[:15000,:].detach().cpu().numpy(),source_pcd[:15000,:].detach().cpu().numpy())
            return sample_template_pcd, sample_source_pcd,igt, self.transforms.igt_rotation, self.transforms.igt_translation,template_pcd[:100000,:],source_pcd[:100000,:]
        else:
            return sample_template_pcd, sample_source_pcd,igt, self.transforms.igt_rotation, self.transforms.igt_translation,

    def __len__(self):
        return len(self.data_class)

if __name__ == "__main__":
    test = Kitti(split='test')
    test  =RegistrationkittiData(test,is_testing=True,sample_point_num=2048)
    # test =Registration3dmatchData(test,is_testing=True,sample_point_num=2048)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    for data in test_loader:
        print(data[0].shape)

