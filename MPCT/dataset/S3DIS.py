import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def load_data_semseg(partition, test_area, train_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR = '/home/ljm/data'
    data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')

    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            for area in train_area:
                if "Area_" + area in room_name:
                    train_idxs.append(i)
                    break
    if partition == 'train':
        all_data = data_batches[train_idxs, ...] #(16733, 4096, 9)
        all_seg = seg_batches[train_idxs, ...] #(16733, 4096)
    else:
        all_data = data_batches[test_idxs, ...] # Area1:(3687, 4096, 9);Area5:(6852, 4096, 9)
        all_seg = seg_batches[test_idxs, ...] # Area5:(6852, 4096)
    print('all_data',all_data.shape)
    print('all_label',all_seg.shape)
    return all_data, all_seg

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='5', train_area=['1','2','3','4','5','6']):
        self.data, self.seg = load_data_semseg(partition, test_area, train_area)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]

        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    # train = S3DIS(4096)
    test = S3DIS(4096, 'test')
    print('Data reading completed!')
    from visual_shapenet import visual_sem
    # Area1:(3687, 4096, 9);Area2:(4440, 4096, 9);Area3:(1650, 4096, 9);Area4:(3662, 4096, 9);Area5:(6852, 4096, 9);Area6:(3294, 4096, 9)
    # 自定义颜色映射
    map = [[0,255,0], [0,0,255], [0,255,255], [255,255,0], [250,0,255],
     [100,100,255], [200,200,100], [170,120,200], [255,0,0], [200,100,100],
     [10,200,100], [200,200,200], [50,50,50]]

    area5_list = ['Area_5_conferenceRoom_1', 'Area_5_conferenceRoom_2', 'Area_5_conferenceRoom_3', 'Area_5_hallway_10', 'Area_5_hallway_11', 'Area_5_hallway_12', 'Area_5_hallway_13', 'Area_5_hallway_14', 'Area_5_hallway_15', 'Area_5_hallway_1', 'Area_5_hallway_2', 'Area_5_hallway_3', 'Area_5_hallway_4', 'Area_5_hallway_5', 'Area_5_hallway_6', 'Area_5_hallway_7', 'Area_5_hallway_8', 'Area_5_hallway_9', 'Area_5_lobby_1', 'Area_5_office_10', 'Area_5_office_11', 'Area_5_office_12', 'Area_5_office_13', 'Area_5_office_14', 'Area_5_office_15', 'Area_5_office_16', 'Area_5_office_17', 'Area_5_office_18', 'Area_5_office_19', 'Area_5_office_1', 'Area_5_office_20', 'Area_5_office_21', 'Area_5_office_22', 'Area_5_office_23', 'Area_5_office_24', 'Area_5_office_25', 'Area_5_office_26', 'Area_5_office_27', 'Area_5_office_28', 'Area_5_office_29', 'Area_5_office_2', 'Area_5_office_30', 'Area_5_office_31', 'Area_5_office_32', 'Area_5_office_33', 'Area_5_office_34', 'Area_5_office_35', 'Area_5_office_36', 'Area_5_office_37', 'Area_5_office_38', 'Area_5_office_39', 'Area_5_office_3', 'Area_5_office_40', 'Area_5_office_41', 'Area_5_office_42', 'Area_5_office_4', 'Area_5_office_5', 'Area_5_office_6', 'Area_5_office_7', 'Area_5_office_8', 'Area_5_office_9', 'Area_5_pantry_1', 'Area_5_storage_1', 'Area_5_storage_2', 'Area_5_storage_3', 'Area_5_storage_4', 'Area_5_WC_1', 'Area_5_WC_2']
    area5_cat = {'Area_5_conferenceRoom_1': 90, 'Area_5_conferenceRoom_2': 190, 'Area_5_conferenceRoom_3': 144, 'Area_5_hallway_10': 125, 'Area_5_hallway_11': 68, 'Area_5_hallway_12': 72, 'Area_5_hallway_13': 162, 'Area_5_hallway_14': 45, 'Area_5_hallway_15': 162, 'Area_5_hallway_1': 378, 'Area_5_hallway_2': 440, 'Area_5_hallway_3': 100, 'Area_5_hallway_4': 66, 'Area_5_hallway_5': 295, 'Area_5_hallway_6': 100, 'Area_5_hallway_7': 100, 'Area_5_hallway_8': 42, 'Area_5_hallway_9': 125, 'Area_5_lobby_1': 126, 'Area_5_office_10': 66, 'Area_5_office_11': 66, 'Area_5_office_12': 66, 'Area_5_office_13': 87, 'Area_5_office_14': 99, 'Area_5_office_15': 107, 'Area_5_office_16': 60, 'Area_5_office_17': 70, 'Area_5_office_18': 99, 'Area_5_office_19': 154, 'Area_5_office_1': 66, 'Area_5_office_20': 36, 'Area_5_office_21': 178, 'Area_5_office_22': 60, 'Area_5_office_23': 60, 'Area_5_office_24': 144, 'Area_5_office_25': 45, 'Area_5_office_26': 50, 'Area_5_office_27': 60, 'Area_5_office_28': 60, 'Area_5_office_29': 144, 'Area_5_office_2': 66, 'Area_5_office_30': 32, 'Area_5_office_31': 66, 'Area_5_office_32': 66, 'Area_5_office_33': 66, 'Area_5_office_34': 66, 'Area_5_office_35': 72, 'Area_5_office_36': 96, 'Area_5_office_37': 182, 'Area_5_office_38': 190, 'Area_5_office_39': 100, 'Area_5_office_3': 66, 'Area_5_office_40': 220, 'Area_5_office_41': 132, 'Area_5_office_42': 66, 'Area_5_office_4': 66, 'Area_5_office_5': 54, 'Area_5_office_6': 66, 'Area_5_office_7': 66, 'Area_5_office_8': 66, 'Area_5_office_9': 54, 'Area_5_pantry_1': 56, 'Area_5_storage_1': 48, 'Area_5_storage_2': 80, 'Area_5_storage_3': 9, 'Area_5_storage_4': 50, 'Area_5_WC_1': 72, 'Area_5_WC_2': 72}

    last = 0
    for i in range(0,len(area5_list)):
        data, label = [], []
        num = area5_cat[area5_list[i]]
        for j in range(last,last+num):
            pc, seg = test[j]
            # for k in range(0, len(pc)):
            #     pc[k, 3:6] = map[seg[k]]
            data.append(pc)
            label.append(seg)
        last += num
        print(area5_list[i],last-num,num)
        data = np.concatenate(data, 0)
        label = np.concatenate(label, 0)
        txt = np.concatenate((data[:, 0:6], np.expand_dims(label, axis=1)), 1)
        np.savetxt(f"dataset/{area5_list[i]}.txt", txt, fmt="%f", delimiter=" ")

    # visual_sem(data, label, f'gt_Area_1_conferenceRoom_1', 'sem')


