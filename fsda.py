#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022-06-16 10:52 
# @Author : shi-d
# @File : fsda.py
# @Software: PyCharm
# @GitHub : Shi-D
# @Homepage : https://shi-d.github.io/

import numpy as np
a = [0 for i in range(10)]
b = [2,4,5]
b = np.array(b)
a = np.array(a)
a[b]=1
print(a)