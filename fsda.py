#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022-06-16 10:52 
# @Author : shi-d
# @File : fsda.py
# @Software: PyCharm
# @GitHub : Shi-D
# @Homepage : https://shi-d.github.io/

import numpy as np

a = np.load('./output/DY-BA/output_dyba_0.npz', mmap_mode='r', allow_pickle=True)
a = a['data']
print(a)
print(a.shape)