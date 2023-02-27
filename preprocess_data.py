'''
预处理数据
'''

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader,Dataset
import xarray as xr
import numpy as np

def read_data():
    #所有数据大小都是[2132(164年*13个模式), 8(8个月), 121, 360]

    #首先读入输入数据

    #sst = torch.Tensor((np.load("process/sst.npz"))["sst"].reshape(2132, 1, 121, 360, 1))
    sst = (torch.Tensor((np.load("process/sst.npz"))["sst"].reshape(2132, 8, 121, 360, 1, 1))).permute(0, 1, 4, 5, 2, 3)

    thetao = torch.Tensor(np.load("process/thetao.npz")["thetao"].reshape(2132, 8, 121, 360, 1, 1)).permute(0, 1, 4, 5, 2, 3)
    ua = torch.Tensor(np.load("process/ua.npz")["ua"].reshape(2132, 8, 121, 360, 1, 1)).permute(0, 1, 4, 5, 2, 3)
    va = torch.Tensor(np.load("process/va.npz")["va"].reshape(2132, 8, 121, 360, 1, 1)).permute(0, 1, 4, 5, 2, 3)
    zg = torch.Tensor(np.load("process/zg.npz")["zg"].reshape(2132, 8, 121, 360, 1, 1)).permute(0, 1, 4, 5, 2, 3)

    input_data = torch.cat((sst,thetao,ua,va,zg),dim=3)
    #input_data = torch.cat((sst, thetao), dim=3)
    #print(input_data.shape)
    #shape is [2132,8,1,5,121,360]

    data = {
       "input": input_data
    }

    np.savez("process/test_data.npz", **data)

read_data()