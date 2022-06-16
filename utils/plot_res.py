#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022-06-16 20:41 
# @Author : shi-d
# @File : plot_res.py
# @Software: PyCharm
# @GitHub : Shi-D
# @Homepage : https://shi-d.github.io/

import numpy as np
import os
import matplotlib.pyplot as plt

def load_gt(file_path):
    file_handler = open(file_path, 'r')
    content = file_handler.readlines()
    gts = []
    for con in content:
        gt = con
        gt = gt.strip('\n')
        gt = gt.split(' ')
        gt = np.array(gt).astype(float)
        gts.append(gt)
    return gts


seed_size = 5
seed_num = 10
gts = load_gt('../data/DY-BA/ba.{}.groundtruth'.format(seed_size))

mse_list = []
rmse_list = []
for seed_i in range(seed_num):
    load_path = '../output/DY-BA/output_{}_1/output_dyba_{}.npz'.format(seed_size, seed_i)

    pred = np.load(load_path, mmap_mode='r', allow_pickle=True)
    pred = pred['data']
    gt = gts[seed_i]

    tmp = []
    for i in pred:
        if i>1.:
            tmp.append(1.)
        elif i<0.:
            tmp.append(0.)
        else:
            tmp.append(i)
    pred = tmp


    a = np.abs(np.array(pred) - np.array(gt))
    mse = np.mean(a)
    rmse = np.mean(a * a)

    print('mse:', mse, 'rmse:', rmse)
    mse_list.append(mse)
    rmse_list.append(rmse)

    gt = gt.tolist()
    gt.append(1)
    pred.append(1)
    gt.append(0)
    pred.append(0)

    # plt.scatter(pred, gt, s=50, c='#7779A8', alpha=0.4)
    # plt.show()

meam_mse = np.mean(mse_list)
mean_rmse = np.mean(rmse_list)
best_mse = np.min(mse_list)
best_rmse = np.min(rmse_list)

print('-------------------------------------')
print('meam_mse: {:0.4f}'.format(meam_mse))
print('mean_rmse: {:0.4f}'.format(mean_rmse))
print('best_mse: {:0.4f}'.format(best_mse))
print('best_rmse: {:0.4f}'.format(best_rmse))