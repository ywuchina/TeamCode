# *_*coding:utf-8 *_*
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def load_data_partseg(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    DATA_DIR = '/home/ljm/data'
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0) # (14007/2874, 2048, 3)  14007/2874个训练/测试样本，每个样本2048个点，每个点3维
    all_label = np.concatenate(all_label, axis=0) # (14007/2874, 1)  14007/2874个训练/测试样本，每个样本代表1个类别
    all_seg = np.concatenate(all_seg, axis=0) # (14007/2874, 2048)  14007/2874个训练/测试样本，每个样本2048个点，每个点有一个分割标签
    print('all_data', all_data.shape)
    print('all_label', all_label.shape)
    print('all_seg',all_seg.shape)
    return all_data, all_label, all_seg

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

class ShapeNet(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None, class_test=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice
        self.class_test = class_test

        self.seg_num_all = 50
        self.seg_start_index = 0

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

            if self.class_test != None:
                id_choice = self.cat2id[self.class_test]
                indices = (self.label == id_choice).squeeze()
                self.data = self.data[indices]
                self.label = self.label[indices]
                self.seg = self.seg[indices]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    trainval = ShapeNet(2048, 'trainval')
    test = ShapeNet(2048, 'test')
    print('Data reading completed!')
    from visual_shapenet import visual_part
    id2cat = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
              'lamp', 'laptop', 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    id2num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #[341, 14, 11, 158, 704, 14, 159, 80, 286, 83, 51, 38, 44, 12, 31, 848]
    for i in range(0,len(test)):
        data, label, seg = test[i]
        # print(data)   print(label)  print(seg)
        label0 = label[0]
        if id2num[label0]>10:
            continue
        id2num[label0] = id2num[label0] + 1
        print(id2num)
        visual_part(data,seg,id2num[label0],id2cat[label0])
