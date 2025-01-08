import os
import numpy as np
import glob


def VoxelPCs(pc0, pc1, vx, vy):
    """
    get voxel and its voxel coords from input PCs
    each voxel contains T points
    return: voxel (N, T, dim), coords (N, 3)
    """
    xyz0 = pc0[:, :3]
    xyz1 = pc1[:, :3]
    xyz = np.row_stack((xyz0, xyz1))
    xmin, ymin, zmin = np.amin(xyz, axis=0)
    xmax, ymax, zmax = np.amax(xyz, axis=0)

    vz = zmax
    
    voxel_coords0 = ((xyz0 - np.array([xmin, ymin, zmin])) / (vx, vy, vz)).astype(np.int32)
    #convert to (D, H, W)
    voxel_coords0 = voxel_coords0[:, [2, 0, 1]]
    voxel_coords0, inv_ind0, voxel_counts0 = np.unique(voxel_coords0, axis=0, \
                                           return_inverse=True, return_counts=True)
    voxel_features0 = []
    for i in range(len(voxel_coords0)):
        pts0 = pc0[inv_ind0 == i] # pts.shape[0]=voxel_counts[i]
        voxel_features0.append(pts0)   
        
    voxel_coords1 = ((xyz1 - np.array([xmin, ymin, zmin])) / (vx, vy, vz)).astype(np.int32)
    #convert to (D, H, W)
    voxel_coords1 = voxel_coords1[:, [2, 0, 1]]
    voxel_coords1, inv_ind1, voxel_counts1 = np.unique(voxel_coords1, axis=0, \
                                           return_inverse=True, return_counts=True)
    voxel_features1 = []
    for i in range(len(voxel_coords1)):
        pts1 = pc1[inv_ind1 == i] # pts.shape[0]=voxel_counts[i]
        voxel_features1.append(pts1)    

    return np.array(voxel_features0), voxel_coords0, np.array(voxel_features1), voxel_coords1 

def save_sub_pc(path, save, vx, vy):
    """
    path=os.path.join(cfg.save_dataset_path, 'after_project_label')
    save=cfg.path.dataset_root
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('0.txt'):
                pcp0 = os.path.join(root, file)
                pcp1 = os.path.join(root, file.replace('0', '1'))
                pc0 = np.loadtxt(pcp0)
                pc1 = np.loadtxt(pcp1)
                sub_pcs0, coords0, sub_pcs1, coords1 = VoxelPCs(pc0, pc1, vx, vy)
                         
                for i in range(len(sub_pcs0)):
                    sub_pc0 = sub_pcs0[i]
                    coord0 = coords0[i]
                    sp0 = os.path.join(save, root.split('/')[-3],  root.split('/')[-2], root.split('/')[-1])
                    sp0 = sp0 + '_{}_{}_{}'.format(coord0[0], coord0[1], coord0[2])
                    if not os.path.exists(sp0):
                        os.makedirs(sp0)
                    if os.path.exists(os.path.join(sp0, file)):
                        continue
                    else:
                        np.savetxt(os.path.join(sp0, file), sub_pc0, fmt="%.2f %.2f %.2f %.0f")
                         
                for i in range(len(sub_pcs1)):
                    sub_pc1 = sub_pcs1[i]
                    coord1 = coords1[i]
                    sp1 = os.path.join(save, root.split('/')[-3],  root.split('/')[-2], root.split('/')[-1])
                    sp1 = sp1 + '_{}_{}_{}'.format(coord1[0], coord1[1], coord1[2])
                    if not os.path.exists(sp1):
                        os.makedirs(sp1)
                    if os.path.exists(os.path.join(sp1, file.replace('0', '1'))):
                        continue
                    else:
                        np.savetxt(os.path.join(sp1, file.replace('0', '1')), sub_pc1, fmt="%.2f %.2f %.2f %.0f")
#                 print(root, 'have been cropped')
                
                
def write_txt(path, save):

#     cls = ['1-Lidar05', '2-Lidar10', '3-Lidar05Noisy', '4-Photogrametry', '5-MultiSensor']
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('0.txt'):
                cls, data_split, dir_name = root.split('/')[-3], root.split('/')[-2], root.split('/')[-1]
                if not os.path.exists(os.path.join(save, cls)):
                    os.makedirs(os.path.join(save, cls))
                with open(os.path.join(save, cls, data_split+'.txt'), 'a') as f:
                    f.write(dir_name + '\n')
                      
      
            
     
        
        