import os
import numpy as np
import glob
from sklearn.neighbors import KDTree

'''
project label 'removed: 2' from point cloud 2 into point cloud 1
'''

def search_k_neighbors(raw, query, k):
    search_tree = KDTree(raw, metric='manhattan')
    _, neigh_idx = search_tree.query(query, k)
    return neigh_idx

def convert_removed_label(pc1, pc2):
#     pc1 = np.loadtxt(pcp1)
#     pc2 = np.loadtxt(pcp2)   
    removed_idx = np.where(pc2[:, 3]==2)
    if len(removed_idx[0]) > 0:
        query = pc2[:, :2][removed_idx]
        revised_idx = search_k_neighbors(pc1[:, :2], query, 3)
        pc1[:, 3][revised_idx] = 1
        pc2[:, 3][removed_idx] = 0
    return pc1, pc2

                    
   