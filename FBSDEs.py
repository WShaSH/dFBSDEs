#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/14 15:46
# @Author  : Wss
# @File    : FBSDEs.py
# @Description : 定义了一个正倒向随机微分方程组的基类 / Define a base class for forward-backward stochastic differential equations
import abc
import torch

class FBSDEs(object):#定义正倒向随机微分方程组基类/Define a base class for forward-backward stochastic differential equations
    def __init__(self,N_path,x0,t_0,T,N_partition,dim_x,dim_y,dim_w):
        self.N_path = N_path #路径数/path_num
        self.t0 = t_0 #初始时间/initial time
        self.T = T #终止时间/terminal time
        self.N_partition = N_partition #区间划分数/interval partition number
        self.dim_x = dim_x #X_t的维数/Dimension of X_t
        self.dim_y = dim_y #Y_t的维数/Dimension of Y_t
        self.dim_z = (dim_y,dim_w) #Z_t的维度/Dimension of Z_t
        self.dim_w = dim_w #Brown运动维数/Dimension of Brownian motion
        self.x0 = x0 #初始值/initial value
    @abc.abstractmethod
    def mu(self,t,x): #定义方程中的函数mu/Define mu of FBSDES in the equation
        pass

    @abc.abstractmethod
    def sigma(self,t,x): #定义方程中的函数sigma/Define sigma of FBSDES in the equation
        pass

    @abc.abstractmethod
    def f(self,t,x,y,z): #定义方程中的函数f/Define f of FBSDES in the equation
        pass

    @abc.abstractmethod
    def g(self,t,x): #定义方程中的函数g/Define g of FBSDES in the equation
        pass

    @property
    def generate_brown_path(self):#生成布朗运动路径/Generate Brownian motion path
        self.t = torch.linspace(self.t0,self.T,self.N_partition + 1).detach()
        self.dt = torch.tensor((self.T - self.t0) / self.N_partition).detach()
        self.dW = torch.sqrt(self.dt) * torch.randn(self.N_path * self.N_partition * self.dim_w)
        self.dW = self.dW.reshape(self.N_path, self.N_partition, self.dim_w,1)
        self.W = torch.cumsum(self.dW, axis = 1)

    @property
    def Euler_Maruyama(self):#欧拉-丸山方法求解正向随机微分方程/Euler-Maruyama method to solve the forward stochastic differential equation
        self.X = torch.zeros((self.N_path,self.N_partition+1,self.dim_x,1))
        self.X[:, 0, :, :] = self.x0
        for i in range(self.N_partition):
            self.X[:, i + 1, :, :] = self.X[:, i, :, :] + self.dt * self.mu(self.t[i],self.X[:, i, :, :]) + torch.matmul(self.sigma(self.t[i],self.X[:, i, :, :]),self.dW[:, i, :,:])
        return self.X

