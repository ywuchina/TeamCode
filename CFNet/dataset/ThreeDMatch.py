import os,sys
from os.path import join, exists
import pickle
import glob
import random

import torch
import torch.utils.data as data
import numpy as np
# from operations.SE3 import *
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from operations.transform_functions import PCRNetTransform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/ljm/data'

def farthest_subsample_points(pointcloud1, num_subsampled_points):
    # (num_points, 3)
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
    return pointcloud1[idx1, :], gt_mask

class ThreeDMatch(data.Dataset):
    def __init__(self,split):
        self.root = os.path.join(DATA_DIR, 'threedmatch')
        self.split = split
        DATA_FILES = {
            'train': os.path.join(BASE_DIR,'split/train_3dmatch.txt'),
            'val':os.path.join(BASE_DIR,'split/val_3dmatch.txt'),
        }
        subset_names = open(DATA_FILES[split]).read().split()
        print(subset_names)
        self.files = []
        self.length = 0
        for name in subset_names:
            fname = name + "*.npz"
            fnames_txt = glob.glob(os.path.join(self.root,fname))
            for fname_txt in fnames_txt:
                self.files.append(fname_txt)
                self.length += 1
        # print(self.files)
        
    def __getitem__(self, index):
        template_file= self.files[index]
        template_data = np.load(os.path.join(self.root,template_file))
        template_keypts = template_data['pcd']
        template_rgb = template_data['color']
        template =np.concatenate((template_keypts,template_rgb), axis=1)

        return torch.from_numpy(template).type(torch.float)

    def __len__(self):
        return self.length

class Registration3dmatchData(data.Dataset):
    def __init__(self, data_class,sample_point_num=2048,is_testing=False):
        super(Registration3dmatchData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        self.sample_point_num =sample_point_num
        self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

    def __getitem__(self, index):
        template= self.data_class[index] # [N,6]
        template_pcd = template[:, :3] #
        self.transforms.index = index
        source_pcd = self.transforms(template_pcd)
        source = torch.cat((source_pcd, template[:, 3:6]),dim=1)
        igt = self.transforms.igt
        N_tmp =template_pcd.shape[0]
        sel_ind = np.random.choice(N_tmp, self.sample_point_num)
        sample_template_pcd =template_pcd[sel_ind,:]
        sample_source_pcd  =source_pcd[sel_ind,:]
        # visual_3dmatch_pcd(template_pcd.detach().cpu().numpy(),source_pcd.detach().cpu().numpy())
        return sample_template_pcd, sample_source_pcd,igt, self.transforms.igt_rotation, self.transforms.igt_translation,

    def __len__(self):
        return len(self.data_class)

if __name__ == "__main__":
    test = ThreeDMatch(split='train')
    test =Registration3dmatchData(test,is_testing=True,sample_point_num=2048)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    for data in test_loader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        print(data[4].shape)
        print(data[5].shape)
        print(data[6].shape)


    # a = torch.rand(10000,3)
    # b,c= farthest_subsample_points(a,2048)
    # print(b.shape)
    # print(c[:10])
