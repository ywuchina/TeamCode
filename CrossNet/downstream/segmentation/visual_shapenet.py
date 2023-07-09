import random

import numpy as np
from matplotlib import pyplot as plt

def visual_part(data, seg, index, path):
    # lab2seg: 每个点云数据所对应的类别
    lab2cls = { 0:'airplane',  1:'bag',  2:'cap',  3:'car',  4:'chair',  5:'earphone',  6:'guitar',  7:'knife',  8:'lamp',  9:'laptop',
                10:'motor', 11:'mug', 12:'pistol', 13:'rocket', 14:'skateboard', 15:'table'}
    # lab2seg: 生成的颜色对应表，即每个种类的每个part类别所对应的颜色，作为输入是为了使得不同结果间保持一致
    lab2seg={'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43], 'car': [8, 9, 10, 11],
         'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37], 'guitar': [19, 20, 21],
         'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49], 'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40],
         'chair': [12, 13, 14, 15], 'knife': [22, 23]}

    #生成颜色映射表
    map = ['r', 'g', 'b', 'c', 'm', 'y']
    #官方的颜色映射
    map = [[0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65], [0.95, 0.95, 0.65], [0.95, 0.65, 0.05],
     [0.35, 0.05, 0.05], [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], [0.65, 0.05, 0.35],
     [0.05, 0.35, 0.35], [0.65, 0.65, 0.35], [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35],
     [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05], [0.95, 0.65, 0.95],
     [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95], [0.05, 0.05, 0.95],
     [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05],
     [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05],
     [0.05, 0.65, 0.35], [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], [0.05, 0.95, 0.35],
     [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35],
     [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95], [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]]

    # #随机生成颜色映射
    # idx = np.arange(0, len(map))
    # idx_all = []
    # np.random.seed(123)
    # for _ in range(len(lab2seg)):
    #     np.random.shuffle(idx)
    #     idx_all.append(copy.deepcopy(idx))
    # idx_all = np.array(idx_all)
    # print(idx_all)
    # #将生成的颜色映射表对应到不同种类的part类别上
    # for i,key in enumerate(lab2seg.keys()):
    #     lab2seg[key]=dict(zip(set(lab2seg[key]),[map[idx_all[i,j]] for j in range(len(set(lab2seg[key])))]))

    # 将数据点的part类别对应为lab2seg中设置的颜色
    colormap = [[] for _ in range(len(data))]
    for i in range(len(data)):
        # colormap[i] = lab2seg[lab2cls[cls]][seg[i]]
        colormap[i] = map[seg[i]]
        # print(colormap[i], seg[i])

    # 设置图片大小
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, projection='3d')
    # 设置视角
    # ax.view_init(elev=180, azim=90)
    ax.grid(False)
    # 关闭坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    # 设置坐标轴范围
    # ax.set_xlabel(r'$X$')
    # ax.set_ylabel(r'$Y$')
    # ax.set_zlabel(r'$Z$')
    ax.set_zlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_xlim3d(-1, 1)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colormap, s=6, marker='.')
    # plt.show()
    plt.savefig(f'result/part/mpct/{path}_mpct_{index}.png', dpi=500, bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    pc = np.loadtxt('airplane.txt')
    data = pc[:,:3].astype(float)
    seg = pc[:,-1].astype(int)
    index = 0
    visual_part(data, seg, 0,'part')
    #获取所要绘制点云数据的类别
    cls=['Earphone', 'Motorbike', 'Rocket', 'Car', 'Laptop', 'Cap', 'Skateboard', 'Mug', 'Guitar', 'Bag',
         'Lamp', 'Table', 'Airplane', 'Pistol', 'Chair', 'Knife']
