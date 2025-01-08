import os
import numpy as np
from plyfile import PlyData
from project_removed_label2pc1 import convert_removed_label
from crop_pc import save_sub_pc, write_txt
import sys
sys.path.append(os.path.abspath(".."))
import configs as cfg
tcfg = cfg.CONFIGS['Train']
"""
generate dataset from Ubr3DCD
    1. project label from PC2 into PC1
    2. crop PCs into subPCs using resolution of [cfg.vx, cfg.vy]
"""
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def convert_ply2txt(path):
    pcd = PlyData.read(path)
    x = pcd.elements[0].data['x']
    y = pcd.elements[0].data['y']
    z = pcd.elements[0].data['z']
    gt = pcd.elements[0].data['label_ch']
    pc = np.column_stack((x,y,z,gt))
    return pc

def read_ply(path, save):
    '''
    path: path of Ubr3DCD dataset
    '''
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('pointCloud0.ply'):
                ply1 = os.path.join(root, 'pointCloud0.ply')
                ply2 = os.path.join(root, 'pointCloud1.ply')
                pc1 = convert_ply2txt(ply1)
                pc2 = convert_ply2txt(ply2)
                cpc1, cpc2 = convert_removed_label(pc1, pc2)
                sp = os.path.join(save, 'after_project_label')
                spp = os.path.join(sp, root.split('/')[-3],  root.split('/')[-2], root.split('/')[-1])
                mkdir(spp)
                np.savetxt(os.path.join(spp, 'pointCloud0.txt'), cpc1, fmt="%.2f %.2f %.2f %.0f")
                np.savetxt(os.path.join(spp, 'pointCloud1.txt'), cpc2, fmt="%.2f %.2f %.2f %.0f")

def gen_dataset(tcfg):
    # print('***generating training val and test data from Ubr3DCD dataset, it will take some time.***')
    # print('   1. projecting label {removed: 2} from point cloud 2 into point cloud 1......')
    # read_ply(path=tcfg.Urb3DCD_path, save=tcfg.save_dataset_path)
    # print('   2. cropping Pcs......')
    # save_sub_pc(path=os.path.join(tcfg.save_dataset_path, 'after_project_label'), save=tcfg.path.dataset_root, vx=tcfg.vx, vy=tcfg.vy)
    print('   3. write train val and test .txt files......')
    write_txt(tcfg.path.dataset_root, tcfg.path.txt_path)
    print('***data preparation finished***')


if __name__ == '__main__':
    gen_dataset(tcfg)       
                