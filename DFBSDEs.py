#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/14 16:41
# @Author  : Wss
# @File    : DFBSDEs.py
# @Description : 定义了一个DFBSDEs类，可求解 Black-Scholes方程/Define a DFBSDEs class that can solve the Black-Scholes equation
import torch
import numpy as np
import torch.nn as nn
from BSM import Black_Scholes_Model
from subNet import generateNet

class DBSDE(nn.Module):
    def __init__(self,layers,N_partition,N_path,x0,t0,T,mu,sigma,r,K):
        super(DBSDE,self).__init__()
        self.layers = layers
        self.N_partition = N_partition
        self.model = nn.ModuleList(generateNet(self.layers) for _ in range(self.N_partition-1))
        self.equation = Black_Scholes_Model(N_path,x0,t0,T,N_partition,mu,sigma,r,K)
        self.equation.generate_brown_path
        self.x = self.equation.Euler_Maruyama
        y0 = torch.mean(self.equation.g(self.equation.T,self.x[:,-1,:,:])).reshape(1,self.equation.dim_y,1)
        #y0 = torch.mean(self.equation.g(self.equation.T,self.x[:,-1,:,:]),dim = 1).reshape(self.equation.N_path,self.equation.dim_y,1)
        y0.requires_grad = True
        self.y0 = nn.Parameter(y0)
        z0 = torch.rand(self.equation.N_path, self.equation.dim_y, self.equation.dim_w)
        z0.requires_grad = True
        self.z0 = nn.Parameter(z0)

    def forward(self):#求解正向随机微分方程/Solve the forward stochastic differential equation
        y = self.y0 * torch.ones(self.equation.N_path , self.equation.dim_x, 1)
        z = self.z0 * torch.ones(self.equation.N_path , self.equation.dim_y, self.equation.dim_w)
        for i, submodel in enumerate(self.model):
            y = y - self.equation.dt * \
                 self.equation.f(self.equation.t[i], self.x[:, i, :, :], y, z) + \
                 torch.matmul(z, self.equation.dW[:, i, :, :])
            z = submodel(self.x[:,i+1,:,:].squeeze(-1)).reshape(self.equation.N_path, self.equation.dim_y,self.equation.dim_w)
        self.y_terminal = y  - self.equation.dt * \
                 self.equation.f(self.equation.t[self.equation.N_partition-1], self.x[:, self.equation.N_partition-1, :, :], y, z) + \
                 torch.matmul(z, self.equation.dW[:, self.equation.N_partition-1, :, :])
        return self.y_terminal


    def train(self,epochs,lr):#训练函数/Training function
        optimizer = torch.optim.Adam(self.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)
        #y_true = self.equation.true_value(self.equation.T, self.x[:, -1, :, :])
        y_true = self.equation.g(self.equation.T, self.x[:, -1, :, :]).squeeze()
        optimizer.zero_grad()
        self.y_pred = self.forward().squeeze()
        loss = torch.mean((self.y_pred - y_true) ** 2)
        for i in range(epochs):
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.y_pred = self.forward().squeeze()
            loss = torch.mean((self.y_pred-y_true)**2)
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch:',i,'loss:',loss.item())
                print('y0:',self.y0)
            if loss < 1e-08:
                break

    def predict(self):
        y = torch.ones(self.equation.N_path, self.equation.N_partition + 1, self.equation.dim_y, 1)
        z = torch.ones(self.equation.N_path, self.equation.N_partition, self.equation.dim_y,
                       self.equation.dim_w)
        y[:, 0, :, :] = self.y0
        z[:, 0, :, :] = self.z0
        for i, submodel in enumerate(self.model):
            y[:, i + 1, :, :] = y[:, i, :, :] - self.equation.dt * \
                                self.equation.f(self.equation.t[i], self.x[:, i, :, :], y[:, i, :, :], z[:, i, :, :]) + \
                                torch.matmul(z[:, i, :, :], self.equation.dW[:, i, :, :])
            z[:, i + 1, :, :] = submodel(self.x[:, i + 1, :, :].squeeze(-1)).reshape(self.equation.N_path,
                                                                                     self.equation.dim_y,
                                                                                     self.equation.dim_w)
        y[:, self.equation.N_partition, :, :] = y[:, self.equation.N_partition - 1, :, :] - self.equation.dt * \
                                                self.equation.f(self.equation.t[self.equation.N_partition - 1],
                                                                self.x[:, self.equation.N_partition - 1, :, :],
                                                                y[:, self.equation.N_partition - 1, :, :],
                                                                z[:, self.equation.N_partition - 1, :, :]) + \
                                                torch.matmul(z[:, self.equation.N_partition - 1, :, :],
                                                             self.equation.dW[:, self.equation.N_partition - 1, :, :])
        return y, z
