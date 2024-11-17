#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/14 15:57
# @Author  : Wss
# @File    : Black-Scholes-Model.py
# @Description : 在fBSDEs的基础上定义了一个Black-Scholes方程/Define a Black-Scholes equation based on fBSDEs
import torch
from scipy.stats import norm as nm
from FBSDEs import FBSDEs
class Black_Scholes_Model(FBSDEs):
    def __init__(self,N_path,x0,t_0,T,N_partition,mu,sigma,r,K):
        super().__init__(N_path,x0,t_0,T,N_partition,1,1,1)
        self.mu_ = mu #mu为股票的收益率/mu is the return rate of the stock
        self.sigma_ = sigma #sigma为股票的波动率/sigma is the volatility of the stock
        self.r_ = r #r为无风险利率/r is the risk-free interest rate
        self.K = K #K为期权的执行价格/K is the exercise price of the option

    def mu(self, t, x): #重写方程中的函数mu/Rewrite the function mu in the equation
        return self.mu_ * x

    def sigma(self, t, x):
        return self.sigma_ * x #self.sigma_ * torch.tile(x,(1,1,self.dim_w))

    def f(self, t, x, y, z):
        return -((self.mu_ - self.r_) / self.sigma_ * z + self.r_ * y) #重写方程中的函数f/Rewrite the function f in the equation

    def g(self,t,x): #重写方程中的函数g/Rewrite the function g in the equation
        return torch.where(x - self.K > 0, x - self.K, 0)

    def true_value(self, t, x):#方程的解析解/Analytical solution of the equation
        n = len(x.shape)
        if n == 1:
            x.reshape(1,-1,1,1)
        else:
            x = x.reshape(x.shape[0],-1,1,1)
        T_t = torch.tile(torch.tensor(self.T-t).reshape(1,-1,1,1), (x.shape[0],1,1,1)).detach()
        d1 = 1 / (self.sigma_ * torch.sqrt(T_t)) * (
                    torch.log(x / self.K) + (self.r_ + self.sigma_ ** 2 / 2) * (T_t))
        d2 = 1 / (self.sigma_ * torch.sqrt(T_t)) * (
                    torch.log(x / self.K) + (self.r_ - self.sigma_ ** 2 / 2) * (T_t))
        y = x * torch.tensor(nm.cdf(d1)) - K * torch.exp(-self.r_ * (T_t)) * torch.tensor(nm.cdf(d2))
        return y
