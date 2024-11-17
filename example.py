
import torch
import numpy as np
from BSM import Black_Scholes_Model
from DFBSDEs import DBSDE
N = 6 # 区间划分数/Number of interval partitions
t0 = 0 # 起始时间/initial time
x0 = 400 # 初始值/initial value
tN = 3 # 终端时间/terminal time
K = 450 # 敲定价格/strike price
r = 0.04 # 无风险利率/risk-free interest rate
t = torch.linspace(t0,tN,N+1) # 时间划分/time partition
path_N = 1000 # 路径数/path number
mu = 0.06 # 收益率/return rate
sigma = 0.18 # 波动率/volatility
c = DBSDE([1,5,5,1],N,path_N,x0,t0,tN,mu,sigma,r,K)
c.train(2000,0.1)