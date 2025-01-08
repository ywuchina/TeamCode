# @Project  : GAN_for_Representation_Mapping
# @Author   : Finley_Jiang
# @File     : Image_Processing.py
# @Time     : 2018/4/9 16:49
# @Software : PyCharm

import cv2
import numpy as np


def sample_select(optical_data, sar_data, ref_data):
    num_r = optical_data.shape[0]
    select_index = []
    for i in range(num_r):
        if np.sum(ref_data, (1, 2))[i] == 0:
            select_index.append(i)
    s_optical_data = optical_data[select_index]
    s_sar_data = sar_data[select_index]
    return s_optical_data, s_sar_data


def image_cut(image, kernel_size):
    r, c = image.shape[0:2]
    if image.ndim > 2:
        d = image.shape[2]
    else:
        d = 1
        # 边缘扩充
    extd_lenth = kernel_size // 2
    extended_image = cv2.copyMakeBorder(image, extd_lenth, extd_lenth, extd_lenth, extd_lenth, cv2.BORDER_DEFAULT)
    data = np.zeros([r * c, kernel_size * kernel_size * d])
    for i in range(r):
        for j in range(c):
            data[i * c + j] = (extended_image[i:i + kernel_size, j:j + kernel_size]).flatten()
    return data


def image_recovery(data, kernel_size, r, c, d):
    if d > 1:
        recovery_data = data[:, [kernel_size ** 2 // 2, 3 * kernel_size ** 2 // 2, 5 * kernel_size ** 2 // 2]]
        recovery_image = recovery_data.reshape(r, c, d)
    else:
        recovery_data = data[:, kernel_size ** 2 // 2]
        recovery_image = recovery_data.reshape(r, c)
    return recovery_image


def main():
    kernel_size = 5
    # 数据读取
    optical = (cv2.imread('/Users/finley/Desktop/GAN_for_Representation_Mapping/Shuguang/optical.bmp'))
    sar = cv2.imread('/Users/finley/Desktop/GAN_for_Representation_Mapping/Shuguang/sar.bmp')
    # 数据切分
    # 保存为向量形式
    optical_data = image_cut(optical, kernel_size)
    sar_data = image_cut(sar, kernel_size)
    # ref_data = image_cut_vector(ref)

    # 训练数据筛选

    # 数据保存
    np.save('/Users/finley/Desktop/GAN_for_Representation_Mapping/Shuguang/optical_data.npy', optical_data / 255)
    np.save('/Users/finley/Desktop/GAN_for_Representation_Mapping/Shuguang/sar_data.npy', sar_data / 255)

if __name__ == '__main__':
    main()
