#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/14 16:10
# @Author  : Wss
# @File    : subNet.py
# @Description : 时间区间[t0,T]的每个小区间上的子网络 / Sub-network on each small interval of the time interval [t0, T]
import torch
import torch.nn as nn
def generateNet(layers):
    layers = layers
    fc = nn.Sequential()
    fc.append(nn.BatchNorm1d(layers[0]))
    for i in range(len(layers)-2):
        Linear = nn.Linear(layers[i],layers[i+1],bias=False)
        nn.init.xavier_uniform_(Linear.weight)
        fc.append(Linear)
        fc.append(nn.BatchNorm1d(layers[i+1]))
        fc.append(nn.ReLU(inplace=True))
    fc.append(nn.Linear(layers[-2],layers[-1],bias=False))
    return fc #返回一个网络/Return a network