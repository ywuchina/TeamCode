import random

import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import glob
from torch.utils.data import DataLoader

from operations.transform_functions import PCRNetTransform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/ljm/data'
def load_data(train):
    if train:
        partition = 'train'
    else:
        partition = 'test'
    # 存放数据和对应标签
    Data = []
    Label = []
    Category = [0 for i in range(41)]
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_{}*.h5'.format(partition))):
        with h5py.File(h5_name, 'r') as file:
            data = file['data'][:].astype('float32')
            label = file['label'][:].astype('int64')
            Data.append(data)
            Label.append(label)

    Data = np.concatenate(Data, axis=0)  # (9840, 2048, 3)  9840个训练样本，每个样本2048个点，每个点3维
    Label = np.concatenate(Label, axis=0)  # (9840, 1)
    Label = Label.squeeze(1)
    for i in Label:
        Category[i] = Category[i] + 1
    print(partition,Category)
    return Data, Label

def read_classed():
    # 读取所有类的类名
    with open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r') as file:
        shape_name = file.read()
        shape_name = np.array(shape_name.split('\n')[:-1])
        return shape_name

# 数据点随机抖动
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

class ModelNet40(Dataset):
    def __init__(self, train=True, num_points=1024, randomize_data=False, gaussian_noise=False, noise_number = 0.01, unseen=False, single = -1, percent=1.0):
        super(ModelNet40, self).__init__()
        self.data, self.labels = load_data(train)
        self.shapes = read_classed()
        self.num_points = num_points
        self.randomize_data = randomize_data
        self.train = train
        self.percent = percent
        self.gaussian_noise = gaussian_noise
        self.noise_number = noise_number
        self.unseen =unseen
        self.single = single

        if self.percent < 1:
            self.data, self.labels = self.sample_data()
            print("Remain data and label:", self.data.shape, self.labels.shape)

        if self.unseen:
                ######## simulate testing on first 20 categories while training on last 20 categories
            if self.train:
                self.data = self.data[self.labels.squeeze(1)<20]
                self.labels = self.labels[self.labels.squeeze(1)<20]
            else:
                self.data = self.data[self.labels.squeeze(1)>=20]
                self.labels = self.labels[self.labels.squeeze(1)>=20]

        if self.single>=0:
            self.data = self.data[self.labels.squeeze(1)==self.single]
            self.labels = self.labels[self.labels.squeeze(1)==self.single]

    def __getitem__(self, index):
        if self.randomize_data:
            current_points = self.randomize(index)  # 从该实例2048个点随机采样了1024个点
        else:
            current_points = self.data[index].copy()  # 直接使用该实例2048个点
        if self.gaussian_noise:
                current_points = jitter_pointcloud(current_points,self.noise_number)

        current_points = torch.from_numpy(current_points).float()
        label = torch.from_numpy(self.labels[index]).type(torch.LongTensor)
        return current_points, label  # 返回该实例（实例从2048个点随机采样了1024个点）以及标签

    def __len__(self):
        return self.data.shape[0]

    def get_shape(self, label):
        return self.shapes[label]

    def randomize(self, index):
        point_index = np.arange(0, self.num_points)  # 在0~num_points范围内生成索引
        np.random.shuffle(point_index)  # 打乱索引
        return self.data[index, point_index].copy()

    def sample_data(self):
        data_by_label = {}
        for i in range(len(self.data)):
            label = self.labels[i][0]
            if label not in data_by_label:
                data_by_label[label] = [self.data[i]]
            else:
                data_by_label[label].append(self.data[i])
        chosen_data = []
        chosen_labels = []
        all_data = []
        all_labels = []
        for label in data_by_label:
            idx = list(range(len(data_by_label[label])))
            cidx = np.random.choice(idx)
            chosen_data.append(data_by_label[label][cidx])
            chosen_labels.append(label)
            del data_by_label[label][cidx]
            all_data.extend(data_by_label[label])
            all_labels.extend([label]*len(data_by_label[label]))
        remain_num = int(round(len(self.data) * self.percent)) - len(chosen_data)
        idx = list(range(len(all_data)))
        cidx = random.sample(idx, remain_num)
        chosen_data = np.array(chosen_data)
        chosen_labels = np.array(chosen_labels)
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        chosen_data = np.concatenate([chosen_data, all_data[cidx]], 0)
        chosen_labels = np.concatenate([chosen_labels, all_labels[cidx]], 0)
        chosen_labels = chosen_labels.reshape(-1, 1)
        return chosen_data, chosen_labels

class RegistrationData(Dataset):
    def __init__(self, data_class=ModelNet40(), is_testing=False):
        super(RegistrationData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

    def __getitem__(self, index):
        # template_pc:模版点云 source_pc:源点云
        template_pc, label = self.data_class[index]
        self.transforms.index = index

        # 调用__call__对模版点云变换后获得源点云
        source_pc = self.transforms(template_pc)
        igt = self.transforms.igt

        return template_pc, source_pc, igt,self.transforms.igt_rotation, self.transforms.igt_translation

    def __len__(self):
        return len(self.data_class)

if __name__ == '__main__':
    data = RegistrationData(ModelNet40(train=False))
    # test_loader = DataLoader(data, batch_size=1, shuffle=False)
    # for i, data in enumerate(test_loader):
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     print(data[2].shape)
    #     print(data[3].shape)
    #     break
