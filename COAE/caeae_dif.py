import scipy.io as io
import numpy as np
import cv2
import os
import pandas as pd
import patch_size
import image_path
import batch_size
BATCH_SIZE=batch_size.batch_size

IMAGE_PATH=image_path.image_path
PATCH_SIZE=patch_size.patch_size

import deepdish as dd

# input_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+str(PATCH_SIZE)+'_input.mat')['input_vecs']
# recon_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+str(PATCH_SIZE)+'_recon.mat')['recon_vecs']
# input_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+s+'_input.mat')['input_vecs']
# recon_vecs = io.loadmat(IMAGE_PATH+'/caeae/data_vec_'+s+'_recon.mat')['recon_vecs']

# input_vecs = dd.io.load(IMAGE_PATH + '/caeae/data_vec_' + str(PATCH_SIZE) + '_input.h5')['input_vecs']
# recon_vecs = dd.io.load(IMAGE_PATH + '/caeae/data_vec_' + str(PATCH_SIZE) + '_recon.h5')['recon_vecs']
input_vecs = io.loadmat(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'_training_1.mat')['vec']
recon_vecs = io.loadmat(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'_training_2.mat')['vec']
j=0
dist=[]
for i in range(int(input_vecs.shape[0])):
    dist.append(np.linalg.norm(input_vecs[j] - recon_vecs[j]))
    j=j+1

#a=np.mean(dist)

image_path=IMAGE_PATH
dif_image=cv2.imread(image_path+'/change_map_1_s_'+str(PATCH_SIZE)+'.bmp')

threshold=0
threshold_add=0.01
PCC_before=0
flag=True
count=0
while(flag):

    k=0

    for i in range(dif_image.shape[0]):
        for j in range(dif_image.shape[1]):
            if dist[k]<threshold:
                dif_image[i, j] = 0
            else:
                dif_image[i, j] = 255
            k=k+1

    #cv2.imwrite(os.path.join(image_path,'caeae_dif_b_'+str(BATCH_SIZE)+'_s_'+str(PATCH_SIZE)+'_t_'+str(threshold)+'.bmp'),dif_image)
    #cv2.imwrite(os.path.join(image_path,'caeae_plus_dif_'+s+'_t_'+str(threshold)+'.bmp'),dif_image)

    pic1=dif_image
    pic2=cv2.imread(os.path.join(IMAGE_PATH,'im3.bmp'))

    width=pic2.shape[0]
    height=pic2.shape[1]

    Mc=0
    Mu=0
    TP=0
    TN=0

    for i in range(width):
        for j in range(height):
            if pic1[i,j,1]<127:
                pic1[i,j]=[0,0,0]
                TN=TN+1
            else:
                pic1[i,j]=[255,255,255]
                TP=TP+1

            if pic2[i,j,1]<10:
                pic2[i,j]=[0,0,0]
                Mu=Mu+1
            else:
                pic2[i,j]=[255,255,255]
                Mc=Mc+1

    picdif=pic2[:,:,1]-pic1[:,:,1]
    FN=0
    FP=0
    for i in range(width):
        for j in range(height):
            if picdif[i,j]==1:
                FP=FP+1
            if picdif[i,j]==255:
                FN=FN+1

    OE=FP+FN
    TP=TP-FP
    TN=TN-FN

    PCC=(float(TP)+float(TN))/(width*height)
    #print(PCC)

    PRE=(float(TP+FP)*float(Mc)+float(FN+TN)*float(Mu))/((TP+FP+TN+FN)**2)
    Kappa=(PCC-PRE)/(1-PRE)
    #print(Kappa)

    if PCC>=PCC_before:
        threshold=threshold+threshold_add
    else:
        flag=False

    PCC_before=PCC
    count=count+1
    print("完成{}/{}".format(count,k))

cv2.imwrite(os.path.join(IMAGE_PATH,'caeae_dif_b_'+str(BATCH_SIZE)+'_s_'+str(PATCH_SIZE)+'_t_'+'%.1f'%threshold+'_PCC_'+'%.5f'%PCC+'_Kappa_'+'%.5f'%Kappa+'_FP_'+str(FP)+'_FN_'+str(FN)+'_OE_'+str(OE)+'.bmp'),dif_image)
