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
import patch_size
import batch_size
import image_path
from sklearn.cluster import KMeans
import deepdish as dd

IMAGE_PATH=image_path.image_path
DATA_PATH=IMAGE_PATH+'/patchs'
BATCH_SIZE=batch_size.batch_size
PATCH_SIZE = patch_size.patch_size
channels=3
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
def fill_feed_dict(data_set, images_pl):

    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
      images_pl: images_feed,
    }
    return feed_dict

dataset=DATASET()

class ConvolutionalAutoencoder(object):
    """

    """
    def __init__(self):
        """
        build the graph
        """
        # place holder of input data
        x = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, channels])  # [#batch, img_height, img_width, #channels]

        #encode
        conv1 = Convolution2D([3, 3, channels, 32], activation=tf.nn.relu, scope='conv_1')(x)
        pool1 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_1')(conv1)
        conv2 = Convolution2D([3, 3, 32, 64], activation=tf.nn.relu, scope='conv_2')(pool1)
        pool2 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_2')(conv2)
        unfold = Unfold(scope='unfold')(pool2)
        encoded = FullyConnected(20, activation=tf.nn.relu, scope='encode')(unfold)
        # decode
        decoded = FullyConnected(PATCH_SIZE*PATCH_SIZE*64, activation=tf.nn.relu, scope='decode')(encoded)
        fold = Fold([-1, PATCH_SIZE, PATCH_SIZE, 64], scope='fold')(decoded)
        unpool1 = UnPooling((1, 1), output_shape=tf.shape(conv2), scope='unpool_1')(fold)
        deconv1 = DeConvolution2D([3, 3, 32, 64], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        unpool2 = UnPooling((1, 1), output_shape=tf.shape(conv1), scope='unpool_2')(deconv1)
        reconstruction = DeConvolution2D([3, 3, channels, 32], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='deconv_2')(unpool2)

        # conv1 = Convolution2D([3, 3, channels, 64], activation=tf.nn.relu, scope='conv_1')(x)
        # pool1 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_1')(conv1)
        # conv2 = Convolution2D([3, 3, 64, 32], activation=tf.nn.relu, scope='conv_2')(pool1)
        # pool2 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_2')(conv2)
        # unfold = Unfold(scope='unfold')(pool2)
        # encoded = FullyConnected(20, activation=tf.nn.relu, scope='encode')(unfold)
        # # decode
        # decoded = FullyConnected(PATCH_SIZE*PATCH_SIZE*32, activation=tf.nn.relu, scope='decode')(encoded)
        # fold = Fold([-1, PATCH_SIZE, PATCH_SIZE, 32], scope='fold')(decoded)
        # unpool1 = UnPooling((1, 1), output_shape=tf.shape(conv2), scope='unpool_1')(fold)
        # deconv1 = DeConvolution2D([3, 3, 64, 32], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        # unpool2 = UnPooling((1, 1), output_shape=tf.shape(conv1), scope='unpool_2')(deconv1)
        # reconstruction = DeConvolution2D([3, 3, channels, 64], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='deconv_2')(unpool2)

        # conv1 = Convolution2D([3, 3, channels, 32], activation=tf.nn.relu, scope='conv_1')(x)
        # pool1 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_1')(conv1)
        # conv2 = Convolution2D([3, 3, 32, 128], activation=tf.nn.relu, scope='conv_2')(pool1)
        # pool2 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_2')(conv2)
        # conv3 = Convolution2D([3, 3, 128, 64], activation=tf.nn.relu, scope='conv_3')(pool2)
        # pool3 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_3')(conv3)
        # unfold = Unfold(scope='unfold')(pool3)
        # encoded = FullyConnected(20, activation=tf.nn.relu, scope='encode')(unfold)
        # # decode
        # decoded = FullyConnected(PATCH_SIZE*PATCH_SIZE*64, activation=tf.nn.relu, scope='decode')(encoded)
        # fold = Fold([-1, PATCH_SIZE, PATCH_SIZE, 64], scope='fold')(decoded)
        # unpool1 = UnPooling((1, 1), output_shape=tf.shape(conv3), scope='unpool_1')(fold)
        # deconv1 = DeConvolution2D([3, 3, 128, 64], output_shape=tf.shape(pool2), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        # unpool2 = UnPooling((1, 1), output_shape=tf.shape(conv2), scope='unpool_2')(deconv1)
        # deconv2 = DeConvolution2D([3, 3, 32, 128], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_2')(unpool2)
        # unpool3 = UnPooling((1, 1), output_shape=tf.shape(conv1), scope='unpool_3')(deconv2)
        # reconstruction = DeConvolution2D([3, 3, channels, 32], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='deconv_3')(unpool3)

        # conv1 = Convolution2D([3, 3, channels, 128], activation=tf.nn.relu, scope='conv_1')(x)
        # pool1 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_1')(conv1)
        # conv2 = Convolution2D([3, 3, 128, 64], activation=tf.nn.relu, scope='conv_2')(pool1)
        # pool2 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_2')(conv2)
        # conv3 = Convolution2D([3, 3, 64, 32], activation=tf.nn.relu, scope='conv_3')(pool2)
        # pool3 = MaxPooling(kernel_shape=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', scope='pool_3')(conv3)
        # unfold = Unfold(scope='unfold')(pool3)
        # encoded = FullyConnected(20, activation=tf.nn.relu, scope='encode')(unfold)
        # # decode
        # decoded = FullyConnected(PATCH_SIZE*PATCH_SIZE*32, activation=tf.nn.relu, scope='decode')(encoded)
        # fold = Fold([-1, PATCH_SIZE, PATCH_SIZE, 32], scope='fold')(decoded)
        # unpool1 = UnPooling((1, 1), output_shape=tf.shape(conv3), scope='unpool_1')(fold)
        # deconv1 = DeConvolution2D([3, 3, 64, 32], output_shape=tf.shape(pool2), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        # unpool2 = UnPooling((1, 1), output_shape=tf.shape(conv2), scope='unpool_2')(deconv1)
        # deconv2 = DeConvolution2D([3, 3, 128, 64], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_2')(unpool2)
        # unpool3 = UnPooling((1, 1), output_shape=tf.shape(conv1), scope='unpool_3')(deconv2)
        # reconstruction = DeConvolution2D([3, 3, channels, 128], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='deconv_3')(unpool3)

        # loss function
        # loss = tf.nn.l2_loss(x - reconstruction)  # L2 loss
        loss = tf.reduce_sum(tf.math.abs(x - reconstruction))

        # training
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        #
        self.x = x
        self.encoded = encoded
        self.reconstruction = reconstruction
        self.loss = loss
        self.training = training

    def train(self, batch_size, passes, new_training=True):
        #data_sets = input_data.read_data_sets(os.path.join(DATA_PATH, 'train_dataset_'+str(PATCH_SIZE)+'.mat'))
        with tf.Session(config=config) as sess:
            # prepare session
            if new_training:
                saver, global_step = Model.start_new_session(sess)
            else:
                saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            # start training
            for step in range(1+global_step, 1+passes+global_step):
                x= dataset.get_batch(batch_size)
                self.training.run(feed_dict={self.x: x})

                if step % 10 == 0:
                    loss = self.loss.eval(feed_dict={self.x: x})
                    print("pass {}, training loss {}".format(step, loss))

                if step % 1000 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    print('checkpoint saved')

    def reconstruct(self):

        def weights_to_grid(weights, rows, cols):
            """convert the weights tensor into a grid for visualization"""
            height, width, in_channel, out_channel = weights.shape
            padded = np.pad(weights, [(1, 1), (1, 1), (0, 0), (0, rows * cols - out_channel)],
                            mode='constant', constant_values=0)
            transposed = padded.transpose((3, 1, 0, 2))
            reshaped = transposed.reshape((rows, -1))
            grid_rows = [row.reshape((-1, height + 2, in_channel)).transpose((1, 0, 2)) for row in reshaped]
            grid = np.concatenate(grid_rows, axis=0)

            return grid.squeeze()

        #mnist = MNIST()
        with tf.Session(config=config) as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')
            print("GOGOGOGO")

            # visualize weights
            # encoded_layer_weights = tf.get_default_graph().get_tensor_by_name("encode/weights:0").eval()
            # grid_image = weights_to_grid(encoded_layer_weights, 1, 20)
            #
            # fig, ax0 = plt.subplots(ncols=1, figsize=(20, 1))
            # ax0.imshow(grid_image, cmap=plt.cm.gray, interpolation='nearest')
            # ax0.set_title('first conv layers weights')
            # plt.show()

            # visualize weights
            # first_layer_weights = tf.get_default_graph().get_tensor_by_name("conv_1/kernel:0").eval()
            # grid_image = weights_to_grid(first_layer_weights, 4, 8)
            #
            # fig, ax0 = plt.subplots(ncols=1, figsize=(8, 4))
            # ax0.imshow(grid_image, cmap=plt.cm.gray, interpolation='nearest')
            # ax0.set_title('first conv layers weights')
            # plt.show()

            # visualize encode
            # batch_size = 36
            # x, y = dataset.get_batch(batch_size)#, dataset='testing')
            # enc = sess.run((self.encoded), feed_dict={self.x: x})
            # print(enc)
            #
            # # visualize results
            # batch_size = 36
            # x = dataset.get_batch(batch_size)
            #
            #
            # training_images = io.loadmat(IMAGE_PATH+'/patchs/train_dataset_'+str(PATCH_SIZE)+'.mat')['patchs']
            training_images = dd.io.load(IMAGE_PATH + '/patchs/train_dataset_' + str(PATCH_SIZE) + '.h5')['patchs']
            images = np.transpose(training_images, (0, 2, 3, 1))

            data_vec=[]

            for i in range(images.shape[0]):
                x=images[[i]]

            # visualize encode
                vec = sess.run((self.encoded), feed_dict={self.x: x})
            #print(vec)
                a=vec[0]
                data_vec.append(vec[0])

            data_vec_dict={}
            #print(data_vec)
            data_vec_dict["vec"] = data_vec
            #io.savemat(os.path.join(DATA_PATH, 'data_vec_'+str(PATCH_SIZE)+'.mat'), data_vec_dict, format='9.3')
            dd.io.save(os.path.join(DATA_PATH,  'data_vec_'+str(PATCH_SIZE)+'.h5'), data_vec_dict)


def main():
    conv_autoencoder = ConvolutionalAutoencoder()
    conv_autoencoder.train(batch_size=BATCH_SIZE, passes=10000, new_training=True)
    conv_autoencoder.reconstruct()

if __name__ == '__main__':
    main()
