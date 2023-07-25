import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from models import *
from dataset import DATASET  # this is the MNIST data manager that provides training/testing batches

import numpy as np
from random import shuffle
import cv2
import os
import tensorflow as tf
import dataset as input_data
import scipy.io as io
import pandas as pd
import patch_size
import image_path
import deepdish as dd

IMAGE_PATH=image_path.image_path
PATCH_SIZE=patch_size.patch_size

#training_vecs = io.loadmat(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'.mat')['vec']
training_vecs = dd.io.load(IMAGE_PATH + '/patchs/data_vec_' + str(PATCH_SIZE) + '.h5')['vec']

training_vecs_1=[]
training_vecs_2=[]

#num=training_vecs.shape[0]
num=len(training_vecs)
for i in range(num):
    if i%2==0:
        training_vecs_1.append(training_vecs[i])
    else:
        training_vecs_2.append(training_vecs[i])

vecs_1 = {}
# print(data_vec)
vecs_1["vec"] = training_vecs_1
io.savemat(os.path.join(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'_training_1.mat'), vecs_1)
# dd.io.save(os.path.join(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'_training_1.h5'), vecs_1)

vecs_2 = {}
# print(data_vec)
vecs_2["vec"] = training_vecs_2
io.savemat(os.path.join(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'_training_2.mat'), vecs_2)
# dd.io.save(os.path.join(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'_training_2.h5'), vecs_2)