import scipy.io as io
import numpy as np
import cv2
import os
import pandas as pd
import patch_size
import image_path
import batch_size
BATCH_SIZE=batch_size.batch_size
import deepdish as dd

IMAGE_PATH=image_path.image_path
PATCH_SIZE=patch_size.patch_size

# input_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+str(PATCH_SIZE)+'_input.mat')['input_vecs']
# recon_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+str(PATCH_SIZE)+'_recon.mat')['recon_vecs']
# input_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+s+'_input.mat')['input_vecs']
# recon_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+s+'_recon.mat')['recon_vecs']

input_vecs = dd.io.load(IMAGE_PATH + '/caeae/data_vec_' + str(PATCH_SIZE) + '_input.h5')['input_vecs']
recon_vecs = dd.io.load(IMAGE_PATH + '/caeae/data_vec_' + str(PATCH_SIZE) + '_recon.h5')['recon_vecs']
print(input_vecs.shape)
print(recon_vecs.shape)
j=0
dist=[]
for i in range(int(input_vecs.shape[0])):
    dist.append(np.linalg.norm(input_vecs[j] - recon_vecs[j]))
    j=j+1

dist=(dist/max(dist))*255
print(len(dist))
image_path=IMAGE_PATH
change_map=cv2.imread(os.path.join(image_path,'im3.bmp'))
print(change_map.shape)
k=0
for i in range(change_map.shape[0]):
    for j in range(change_map.shape[1]):
        change_map[i][j]=dist[k]
        k=k+1
cv2.imwrite(os.path.join(IMAGE_PATH,'change_map_'+str(BATCH_SIZE)+'_s_'+str(PATCH_SIZE)+'.bmp'),change_map)
