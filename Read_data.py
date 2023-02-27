#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Read_data.py    
@Contact :   wccdontsleep@163.com

@Modify Time      @Author       @Version    @Desciption
------------      --------      --------    -----------
2022/9/18 13:24   Wangruoxuan      1.0         None
'''
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader,Dataset
import xarray as xr
import numpy as np

def read_data():
    #所有数据大小都是[1950(150年*13个模式), 121, 360]

    sst = (torch.Tensor((np.load("process/sst.npz"))["sst"].reshape(2132, 8, 121, 360, 1))).permute(1, 0, 4, 2, 3)

    thetao = torch.Tensor(np.load("process/thetao.npz")["thetao"].reshape(2132, 8, 121, 360, 1)).permute(1, 0, 4, 2, 3)
    ua = torch.Tensor(np.load("process/ua.npz")["ua"].reshape(2132, 8, 121, 360, 1)).permute(1, 0, 4, 2, 3)
    va = torch.Tensor(np.load("process/va.npz")["va"].reshape(2132, 8, 121, 360, 1)).permute(1, 0, 4, 2, 3)
    zg = torch.Tensor(np.load("process/zg.npz")["zg"].reshape(2132, 8, 121, 360, 1)).permute(1, 0, 4, 2, 3)

    input_data = torch.cat((sst, thetao, ua, va, zg), dim=2)
    #首先读入输入数据
    sst = torch.Tensor((np.load("process/data_sst_djf0.npz"))["sst"].reshape(1950,1,121,360,1))
    thetao = torch.Tensor(np.load("process/data_thetao_djf0.npz")["thetao"].reshape(1950,1,121,360,1))
    ua = torch.Tensor(np.load("process/data_ua_djf0.npz")["ua"].reshape(1950,1,121,360,1))
    va = torch.Tensor(np.load("process/data_va_djf0.npz")["va"].reshape(1950,1,121,360,1))
    zg = torch.Tensor(np.load("process/data_zg_djf0.npz")["zg"].reshape(1950,1,121,360,1))


    #读入target
    sst_8 = torch.Tensor(np.load("process/8_sst.npz")["sst"].reshape(1950,1,121,360,1))
    thetao_8 = torch.Tensor(np.load("process/8_thetao.npz")["thetao"].reshape(1950,1,121,360,1))
    ua_8 = torch.Tensor(np.load("process/8_ua.npz")["ua"].reshape(1950,1,121,360,1))
    va_8 = torch.Tensor(np.load("process/8_va.npz")["va"].reshape(1950,1,121,360,1))
    zg_8 = torch.Tensor(np.load("process/8_zg.npz")["zg"].reshape(1950,1,121,360,1))

    """
    print(sst.shape)
    print(thetao.shape)
    print(ua.shape)
    print(va.shape)
    print(zg.shape)

    print(sst_8.shape)
    print(thetao_8.shape)
    print(ua_8.shape)
    print(va_8.shape)
    print(zg_8.shape)
    """

    input_data = torch.cat((sst,thetao,ua,va,zg),dim=4)
    target = torch.cat((sst_8,thetao_8,ua_8,va_8,zg_8),dim=4)

    input_data = input_data.permute(0, 1, 4, 2, 3)
    target = target.permute(0, 1, 4, 2, 3)
    #print(input_data.shape)[1950,1,121,360,5]
    #print(target.shape)[1950,1,121,360,5]

    data = {
        "input" : input_data,
        "target" : target
    }

    np.savez("process/data.npz",**data)

read_data()


def test():
    test = np.zeros((2,3,1))
    test1 = np.zeros((2,3,1))
    test2 = np.zeros((2,3,1))


    test = [i+1 for i in test]
    test1 = [i + 2 for i in test1]
    test2 = [i + 3 for i in test2]
    print("test---------------")
    print(test)
    print(test1)
    print(test2)

    test = torch.Tensor(test)
    test1 = torch.Tensor(test1)
    test2 = torch.Tensor(test2)

    res = torch.cat((test,test1,test2),dim = 2)

    print(res)

