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
import os

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
        # print(predictions.shape)
        # print(labels.shape)
        # print('labels', labels)
        l1 = torch.mean(torch.pow(predictions - labels, 2))  # node-level error
        # l2 = torch.abs(torch.sum(predictions) - torch.sum(labels)) / (
        #             torch.sum(labels) + 1e-5)  # influence spread error
        l2 = 0
        return l1+λ*l2


random.seed(111)
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
epochs = 100
lr = 0.1
λ = 0.001

gts = load_gt('./data/DY-BA/ba.{}.groundtruth'.format(seed_size))

for seed_i in range(seed_num):
    load_path = './output/Dy-BA/emb_{}/emb_dyba_{}.npz'.format(seed_size, seed_i)
    out_path = './output/DY-BA/output_{}_1/output_dyba_{}'.format(seed_size, seed_i)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    emb = np.load(load_path, mmap_mode='r', allow_pickle=True)
    emb = emb['data']
    gt = gts[seed_i]
    # print(emb.shape)
    # print(gt.shape)
    # print(sum(gt))

    emb = torch.from_numpy(emb)
    gt = torch.Tensor(gt)

    mlp = Classifier([128,64,8,1])
    # mlp = Classifier([128,64,1])
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    batch_size = 10
    batch_i = 0
    batch_num = int(100/batch_size)

    for epoch in range(epochs):
        for batch_i in range(batch_num):
            mlp.train()
            optimizer.zero_grad()
            output = mlp(emb[batch_i*batch_size:(batch_i+1)*batch_size])
            loss_train = mlp.loss(output, gt[batch_i*batch_size:(batch_i+1)*batch_size], λ)

            loss_train.backward()
            optimizer.step()

            mse = torch.mean(torch.abs(output - gt[batch_i*batch_size:(batch_i+1)*batch_size]))

            print('epoch: {:03d}-{:03d}'.format(epoch, batch_i),
                  'loss: {:.4f}'.format(loss_train.item()),
                  'mse: {:.4f}'.format(mse.item()),)
            output = output.detach().numpy()
            output = np.squeeze(output)
            # print(output)

        # mlp.eval()
        # optimizer.zero_grad()
        output = mlp(emb)
        loss_test = mlp.loss(output, gt, λ)

        # loss_test.backward()
        # optimizer.step()
        mse = torch.mean(torch.abs(output - gt))
        print('test loss: {}'.format(loss_test.item()),
              'mse: {:.4f}'.format(mse.item()))
        output = output.detach().numpy()
        output = np.squeeze(output)
    # print(output)

    output = np.array(output)
    np.savez(out_path, data=output)
