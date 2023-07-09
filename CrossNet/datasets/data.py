import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image
from .plyfile import load_ply
from . import data_utils as d_utils
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

trans_pc_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])
    
trans_pc_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

trans_img_1 = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace = True)])

trans_img_2 = transforms.Compose([transforms.Grayscale(1),
                                transforms.Resize((224, 224)),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(0.485, 0.229, inplace = True)])

def load_modelnet_data(partition, cat = 40):
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data, new_all_data = [],[]
    all_label, new_all_label= [],[]
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    if (cat == 10):
        for i in range(len(all_label)):
            if all_label[i] in [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]:
                #bathtub,bed,chair,desk,dresser,monitor,night_stand,sofa,table,toilet
                new_all_data.append(all_data[i])
                new_all_label.append(all_label[i])
        all_data = np.array(new_all_data)
        all_label = np.array(new_all_label)
    return all_data, all_label

def load_ScanObjectNN(partition):
    BASE_DIR = 'data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

def load_shapenet_data():
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_filepath = []
    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    # print(len(all_filepath)) 43783
    return all_filepath

def get_render_imgs(pcd_path):
    path_lst = pcd_path.split('/')

    path_lst[1] = 'ShapeNetRendering'
    path_lst[-1] = path_lst[-1][:-4]
    path_lst.append('rendering')
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))
    # print(img_path_list)
    return img_path_list

class ShapeNetRender(Dataset):
    def __init__(self, n_imgs = 1):
        self.data = load_shapenet_data()
        self.n_imgs = n_imgs

    def __getitem__(self, item):
        pcd_path = self.data[item]
        render_img_path = random.choice(get_render_imgs(pcd_path))
        # print('pcd_path=>',pcd_path,'render_img_path=>',render_img_path)

        # render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)
        # render_img_rgb_list,render_img_gray_list = [],[]

        # for render_img_path in render_img_path_list:
        render_img_rgb = Image.open(render_img_path).convert('RGB')
        # os.makedirs('result/'+pcd_path.split('.')[0])
        # render_img_rgb.save('result/'+pcd_path.split('.')[0] + '_rgb.png')
        render_img_gray = Image.open(render_img_path).convert('L')
        # render_img_gray.save('result/'+pcd_path.split('.')[0] + '_gray.png')
        print('render_img_rgb0=>',render_img_rgb,np.array(render_img_rgb).shape)

        render_img_rgb = trans_img_1(render_img_rgb)
        render_img_gray = trans_img_2(render_img_gray) #Tensor无save方法
            # render_img_rgb_list.append(render_img_rgb)
            # render_img_gray_list.append(render_img_gray)

        print('render_img_rgb1=>',render_img_rgb,np.array(render_img_rgb).shape)
        pointcloud_1 = load_ply(self.data[item])

        # pointcloud_orig = pointcloud_1.copy()
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = trans_pc_1(pointcloud_1)
        point_t2 = trans_pc_2(pointcloud_2)

        # pointcloud = (pointcloud_orig, point_t1, point_t2)
        pointcloud = (point_t1, point_t2)
        render_img = (render_img_rgb, render_img_gray)
        return pointcloud, render_img # render_img_list

    def __len__(self):
        return len(self.data)

class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train', cat = 40):
        self.data, self.label = load_modelnet_data(partition, cat)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
        
        

