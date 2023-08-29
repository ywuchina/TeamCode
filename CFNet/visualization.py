import os

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import linalg
from drawnow import drawnow
# import cv2 as cv
import torch
from tqdm import tqdm
#import bop_toolkit_lib.renderer as renderer
class CloudVisualizer:

    def __init__(self, t_sleep=0.5, path ='', point_size=3, ortho=True):
        self.t_sleep = t_sleep
        self.point_size = point_size
        self.ortho = ortho
        self.pcd_src_2d, self.pcd_tgt_2d, self.pcd_est_2d = None, None, None
        self.path = path

    def reset(self, pcd_src, pcd_tgt, pcd_est):
        self.pcd_src_2d = self.project(pcd_src[:, :3])
        self.pcd_tgt_2d = self.project(pcd_tgt[:, :3])
        self.pcd_est_2d = self.project(pcd_est[:, :3])
        drawnow(self.plot)


    def update(self, new_est):
        self.pcd_est = new_est
        drawnow(self.plot)

    def plot(self):
        # get scale and center
        yx_min, yx_max = np.vstack([self.pcd_src_2d, self.pcd_tgt_2d]).min(axis=0),\
                         np.vstack([self.pcd_src_2d, self.pcd_tgt_2d]).max(axis=0)
        dimensions = yx_max - yx_min
        center = yx_min + dimensions/2

        # get appropriate x/y axes limits
        dimensions = np.array([dimensions.max()*1.1]*2)
        yx_min = center - dimensions/2
        yx_max = center + dimensions/2

        cmap = plt.get_cmap('tab20')
        magenta, gray, cyan = cmap.colors[12], cmap.colors[14], cmap.colors[18]

        plt.scatter(self.pcd_src_2d[:, 0], self.pcd_src_2d[:, 1], c=np.asarray(cyan)[None, :],
                    s=self.point_size, alpha=0.5)
        plt.scatter(self.pcd_tgt_2d[:, 0], self.pcd_tgt_2d[:, 1], c=np.asarray(magenta)[None, :],
                    s=self.point_size, alpha=0.5)
        plt.scatter(self.pcd_est_2d[:, 0], self.pcd_est_2d[:, 1], c=np.asarray(gray)[None, :],
                    s=self.point_size, alpha=0.7)
        plt.xlim([yx_min[0], yx_max[0]])
        plt.ylim([yx_min[1], yx_max[1]])
        plt.xticks([])
        plt.yticks([])
        self.capture(self.path+'.png')

    def project(self, points):
        Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2] + 3.0  # push back a little
        if self.ortho:
            xs = Xs
            ys = Ys
        else:
            xs = np.divide(Xs, Zs)
            ys = np.divide(Ys, Zs)
        points_2d = np.hstack([xs[:, None], ys[:, None]])
        return points_2d

    def capture(self, path):
        plt.savefig(path, dpi='figure')

class CloudVisualizer3d:

    def __init__(self, t_sleep=0.5, path ='', point_size=3, ortho=True):
        self.t_sleep = t_sleep
        self.point_size = point_size
        self.ortho = ortho
        self.pcd_src_2d, self.pcd_tgt_2d, self.pcd_est_2d = None, None, None
        self.path = path

    def draw(self,pcd_src, pcd_tgt, pcd_est):
        fig = plt.figure(figsize=(15,10))

        pcd_src_x,pcd_src_y,pcd_src_z = pcd_src[:,0],pcd_src[:,1],pcd_src[:,2]
        pcd_tgt_x, pcd_tgt_y, pcd_tgt_z = pcd_tgt[:, 0], pcd_tgt[:, 1], pcd_tgt[:, 2]
        pcd_est_x, pcd_est_y, pcd_est_z = pcd_est[:, 0], pcd_est[:, 1], pcd_est[:, 2]
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()

        ax = fig.add_subplot(111, projection='3d')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # ax
        ax.set_xlabel('X', fontsize=16)
        ax.set_ylabel('Y', fontsize=16)
        ax.set_zlabel('Z', fontsize=16)

        cmap = plt.get_cmap('tab20')
        magenta, gray, cyan = cmap.colors[12], cmap.colors[14], cmap.colors[18]
        area = np.pi * 1.5 ** 2
        ax.scatter(pcd_src_x, pcd_src_y, pcd_src_z, s=area, c=np.asarray(cyan)[None, :], marker='o', alpha=0.5, label='source')
        ax.scatter(pcd_tgt_x, pcd_tgt_y, pcd_tgt_z, s=area, c=np.asarray(magenta)[None, :], marker='o', alpha=0.5, label='target')
        ax.scatter(pcd_est_x, pcd_est_y, pcd_est_z, s=area, c=np.asarray(gray)[None, :], marker='o', alpha=0.7, label='transformed source')

        # plt.legend(loc='upper right',fontsize = 16,columnspacing = 0.2,labelspacing = 0.2,handletextpad =0.2)
        self.capture(self.path+'.png')

    def capture(self, path):
        plt.savefig(path, dpi='figure')


