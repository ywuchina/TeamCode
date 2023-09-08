import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .build import DATASETS
from utils.logger import *
import glob
from PIL import Image
import random
from .plyfile import load_ply
trans_img = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

@DATASETS.register_module()
class ShapeNetRender(data.Dataset):
    def __init__(self, config, n_imgs = 1):
        self.data_path = self.load_shapenet_data()
        self.n_imgs = n_imgs
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.permutation = np.arange(self.npoints)

    def load_shapenet_data(self):
        BASE_DIR = '/home/ljm'
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        all_filepath = []
        for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
            pcs = glob.glob(os.path.join(cls, '*'))
            all_filepath += pcs
        # print(len(all_filepath)) #43783

        return all_filepath

    def get_render_imgs(self, pcd_path):
        pcd_path = pcd_path.replace("ShapeNet/", "ShapeNetRendering/")
        path_lst = pcd_path.split('/')

        path_lst[-1] = path_lst[-1][:-4] # remove .ply
        path_lst.append('rendering')
        DIR = '/'.join(path_lst)
        img_path_list = glob.glob(os.path.join(DIR, '*.png'))
        # print(img_path_list)
        return img_path_list

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, item):
        pcd_path = self.data_path[item]

        # render_img_path = random.choice(self.get_render_imgs(pcd_path))
        # render_img = Image.open(render_img_path)
        # render_img = trans_img(render_img.convert('RGB'))

        render_img_path_list = random.sample(self.get_render_imgs(pcd_path), self.n_imgs)
        render_img_list = []
        for render_img_path in render_img_path_list:
            render_img = Image.open(render_img_path)
            render_img = trans_img(render_img.convert('RGB'))
            render_img_list.append(render_img)

        # print('render_img=>', render_img, np.array(render_img).shape) (3, 224, 224)

        pc = load_ply(self.data_path[item])
        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()
        return pc, render_img_list # render_img,render_img_list

    def __len__(self):
        return len(self.data_path)
