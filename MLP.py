#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022-05-23 14:48 
# @Author : shi-d
# @File : MLP.py
# @Software: PyCharm
# @GitHub : Shi-D
# @Homepage : https://shi-d.github.io/

import torch
import numpy as np
import torch.optim as optim
import random


class Classifier(torch.nn.Module):
    def __init__(self, dims=[64, 8]):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()
        activation1 = torch.nn.Tanh()

        self.mlp = torch.nn.Sequential()
        for i, dim in enumerate(dims):
            if i < len(dims)-1:
                self.mlp.add_module('linear'+str(i), torch.nn.Linear(in_features=dims[i], out_features=dims[i+1], bias=False))
                self.mlp.add_module('active'+str(i), activation)

        self.mlp.add_module('linear'+str(len(dims)-2), torch.nn.Linear(in_features=dims[len(dims)-2], out_features=dims[len(dims)-1], bias=False))
        self.mlp.add_module('active'+str(len(dims)-2), activation1)
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=in_features,
        #                                                out_features=hidden_features[0]),
        #                                activation,
        #                                torch.nn.Linear(in_features=hidden_features[0],
        #                                                out_features=hidden_features[1]),
        #                                activation,
        #                                torch.nn.Linear(in_features=hidden_features[1],
        #                                                out_features=out_features),
        #                                )

    def forward(self, x):
        return self.mlp(x)

    def loss(self, predictions, labels, λ):
        l1 = torch.mean(torch.pow(predictions - labels, 2))  # node-level error
        # l2 = torch.abs(torch.sum(predictions) - torch.sum(labels)) / (
        #             torch.sum(labels) + 1e-5)  # influence spread error
        l2 = 0
        return l1+λ*l2


random.seed(111)
def load_gt(file_path):
    file_handler = open(file_path, 'r')
    content = file_handler.readlines()
    gt = content[0]
    gt = gt.strip('\n')
    gt = gt.split(' ')
    print(len(gt))
    gt = np.array(gt).astype(float)
    return gt


load_path = './logs/DySAT_default/output/default_embs_ba_4.npz'
emb = np.load(load_path, mmap_mode='r', allow_pickle=True)
emb = emb['data']
# print(emb)
print(emb.shape)
gts = load_gt('./data/DY-BA/ba.4.groundtruth.1')
print(sum(gts))
assert 1==0, 'daf'
epochs = 100
lr = 0.1
λ = 0.001

emb = torch.from_numpy(emb)
gts = torch.Tensor(gts)

mlp = Classifier([128,64,8,1])
optimizer = optim.Adam(mlp.parameters(), lr=lr)
batch_size = 10
batch_i = 0
batch_num = int(100/batch_size)

for epoch in range(epochs):
    for i in range(batch_num):
        mlp.train()
        optimizer.zero_grad()
        output = mlp(emb[i*batch_size:(i+1)*batch_size])
        loss_train = mlp.loss(output, gts[i*batch_size:(i+1)*batch_size], λ)

        loss_train.backward()
        optimizer.step()

        print('epoch: {:04d}'.format(epoch),
              'loss: {:.4f}'.format(loss_train.item()))
        output = output.detach().numpy()
        output = np.squeeze(output)
        print(output)

    mlp.eval()
    optimizer.zero_grad()
    output = mlp(emb)
    loss_train = mlp.loss(output, gts, λ)

    loss_train.backward()
    optimizer.step()
    output = output.detach().numpy()
    output = np.squeeze(output)

print(output)

output = np.array(output)
np.savez('./output/output_ba_2', data=output)
