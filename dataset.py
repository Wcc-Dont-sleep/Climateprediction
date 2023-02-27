import numpy as np
import torch
import torch.nn as nn
import netCDF4 as nc
from torch.utils.data import DataLoader,Dataset

class MyData(Dataset):
    def __init__(self,input,target):
        """
        :param path: 获取输入数据的路径将其转变为dataset
        """

        self.input = input
        self.output = target

    def __getitem__(self, item):
        return self.input[item],self.output[item]
    def __len__(self):
        return self.input.shape[0]


