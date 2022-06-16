#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022-05-23 14:42 
# @Author : shi-d
# @File : load_emb.py
# @Software: PyCharm
# @GitHub : Shi-D
# @Homepage : https://shi-d.github.io/

import numpy as np
import dill
import matplotlib.pyplot as plt

load_path = './output/output_ba_0'
# emb = np.load(load_path, mmap_mode='r', allow_pickle=True)
# emb = emb['data']
# print(emb)
# print(emb.shape)

# 清洗数据
# with open(load_path, 'r') as f:
#     content = f.readlines()
#     tmp = []
#     for con in content:
#         con = con.strip('\n')
#         con = con.strip(' ')
#         con = con.split(' ')
#         print(con)
#         for c in con:
#             if c != ' ' and c != '' and c != '\n':
#                 print(c)
#                 tmp.append(c)
# print(len(tmp))
# tmp = np.array(tmp)
# tmp = tmp.astype(float)
# np.savez(load_path+str(0), data=tmp)
# print(tmp)


# 画图
path_pred = './output/output_ba_2.npz'
path_gt = './data/DY-BA/ba.4.groundtruth.1'
def load_gt(file_path):
    file_handler = open(file_path, 'r')
    content = file_handler.readlines()
    gt = content[0]
    gt = gt.strip('\n')
    gt = gt.split(' ')
    # print(len(gt))
    gt = np.array(gt).astype(float)
    return gt

gts = load_gt(path_gt)
pred = np.load(path_pred)['data']
print(pred)
print(gts)
pred = list(pred)
gts = list(gts)
pred.append(0)
pred.append(1)
gts.append(0)
gts.append(1)

plt.scatter(pred, gts, s=80, c='#7779A8', alpha=0.4)

plt.show()