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
from sklearn.cluster import KMeans
import image_path
import deepdish as dd

IMAGE_PATH=image_path.image_path
PATCH_SIZE=patch_size.patch_size
vec_len=20
flag=False

DATA_PATH=IMAGE_PATH+'/patchs'
batch_size = 100
config=tf.ConfigProto()
config.gpu_options.allow_growth = True

def fill_feed_dict(data_set, images_pl):

    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
      images_pl: images_feed,
    }
    return feed_dict

dataset=DATASET()

class Autoencoder(object):
    def __init__(self):

        # place holder of input data
        a = tf.placeholder(tf.float32, shape=[None, vec_len])  # [#batch, vec_len]
        x = tf.placeholder(tf.float32, shape=[None, vec_len])  # [#batch, vec_len]
        y = tf.placeholder(tf.float32, shape=[None, vec_len])  # [#batch, vec_len]

        # encoded = FullyConnected(12, activation=tf.nn.relu, scope='encode')(x)
        # reconstruction = FullyConnected(vec_len, activation=tf.nn.relu, scope='encode')(encoded)

        if flag:
            FullyConnect=FullyConnected(vec_len, activation=tf.nn.relu, scope='encode')(a)
            FullyConnecty = FullyConnected(16, activation=tf.nn.relu, scope='encode')(a)
        else:
            FullyConnect = FullyConnected(16, activation=tf.nn.relu, scope='encode')(x)
            FullyConnecty = FullyConnected(16, activation=tf.nn.relu, scope='encode')(y)
        encoded = FullyConnected(12, activation=tf.nn.relu, scope='encode')(FullyConnect)
        encodedy = FullyConnected(12, activation=tf.nn.relu, scope='encode')(FullyConnecty)
        decoded = FullyConnected(16, activation=tf.nn.relu, scope='decode')(encoded)
        decodedy = FullyConnected(16, activation=tf.nn.relu, scope='decode')(encodedy)
        reconstruction = FullyConnected(vec_len, activation=tf.nn.relu, scope='encode')(decoded)
        reconstructiony = FullyConnected(vec_len, activation=tf.nn.relu, scope='encode')(decodedy)
        if flag:
            loss = tf.nn.l2_loss(a - reconstruction)
        else:
            # loss = tf.reduce_sum(tf.math.abs(y - reconstruction))
            #loss = tf.nn.l2_loss(y - reconstruction)
            #loss = tf.nn.l2_loss(y - reconstruction)+tf.nn.l2_loss(x - reconstruction)
            loss = tf.nn.l2_loss(reconstructiony - reconstruction)+tf.nn.l2_loss(x - reconstruction)

        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self.a = a
        self.x = x
        self.y = y
        self.encoded = encoded
        self.encodedy = encodedy
        self.reconstruction = reconstruction
        self.reconstructiony = reconstructiony
        self.loss = loss
        self.training = training

    def train(self, batch_size, passes, new_training=True):
        """

        :param batch_size:
        :param passes:
        :param new_training:
        :return:
        """

        # data_sets = input_data.read_data_sets(os.path.join(DATA_PATH, 'train_dataset_'+str(PATCH_SIZE)+'.mat'))

        with tf.Session(config=config) as sess:
            # prepare session
            if new_training:
                saver, global_step = Model.start_new_session(sess)
            else:
                saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            # start training
            for step in range(1 + global_step, 1 + passes + global_step):

                a = dataset.vec_get_batch_all(batch_size)
                x_0, y_0 = dataset.vec_get_batch_shuffle(batch_size)

                # x=x_0
                # y=y_0
                # x=np.row_stack((x_0,y_0))
                # y=np.row_stack((y_0,y_0))
                x=np.row_stack((x_0,x_0))
                y=np.row_stack((y_0,x_0))

                if flag:
                    self.training.run(feed_dict={self.a: a})
                else:
                    self.training.run(feed_dict={self.x: x, self.y: y})

                if step % 10 == 0:
                    if flag:
                        loss = self.loss.eval(feed_dict={self.a: a})
                    else:
                        loss = self.loss.eval(feed_dict={self.x: x, self.y: y})
                    print("pass {}, training loss {}".format(step, loss))

                if step % 1000 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    print('checkpoint saved')

    def reconstruct(self):

        with tf.Session(config=config) as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')
            print("GOGOGOGO")
            #training_vecs = io.loadmat(IMAGE_PATH+'/patchs/data_vec_'+str(PATCH_SIZE)+'.mat')['vec']

            training_vecs = dd.io.load(IMAGE_PATH + '/patchs/data_vec_' + str(PATCH_SIZE) + '.h5')['vec']

            data_vec = []

            # pixel_num=training_vecs.shape[0]
            pixel_num=len(training_vecs)
            a = dataset.vec_get_batch_all(pixel_num)
            x, y = dataset.vec_get_batch(int(pixel_num/2))  #分为两个输入

            if flag:
                org, recon = sess.run((self.a, self.reconstruction), feed_dict={self.a: a})
                input_vecs = org
            else:
                org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: x, self.y: y})
                input_vecs = y

            recon_vecs = recon

            # print(input_vecs)
            # print(recon_vecs)

            # dataframe = pd.DataFrame({'input_vecs_0': input_vecs[0], 'input_vecs_0_recon': recon_vecs[0], 'input_vecs_1': input_vecs[1], 'input_vecs_1_recon': recon_vecs[1], 'input_vecs_2': input_vecs[2], 'input_vecs_2_recon': recon_vecs[2]})
            # dataframe.to_csv("data/Italy/vecs_test.csv")

            input = {}
            # print(data_vec)
            input["input_vecs"] = input_vecs
            #io.savemat(os.path.join(IMAGE_PATH+'/caeae/data_vec_'+str(PATCH_SIZE)+'_input.mat'), input, format='9.3')
            dd.io.save(os.path.join(IMAGE_PATH + '/caeae/data_vec_' + str(PATCH_SIZE) + '_input.h5'), input)

            recon = {}
            # print(data_vec)
            recon["recon_vecs"] = recon_vecs
            #io.savemat(os.path.join(IMAGE_PATH+'/caeae/data_vec_'+str(PATCH_SIZE)+'_recon.mat'), recon, format='9.3')
            dd.io.save(os.path.join(IMAGE_PATH + '/caeae/data_vec_' + str(PATCH_SIZE) + '_recon.h5'), recon)

def main():
    autoencoder = Autoencoder()
    autoencoder.train(batch_size=100, passes=10000, new_training=True)
    autoencoder.reconstruct()

if __name__ == '__main__':
    main()
