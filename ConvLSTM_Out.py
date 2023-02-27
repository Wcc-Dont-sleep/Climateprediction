#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvLSTM_Out.py
@Contact :   wccdontsleep@163.com

@Modify Time      @Author       @Version    @Desciption
------------      --------      --------    -----------
2022/10/29 18:40   Wangruoxuan      1.0         Main
'''

import numpy as np
import torch
import torch.nn as nn
import netCDF4 as nc
import xarray.backends.api
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from dataset import MyData
from Model import MyModel

if __name__ == '__main__':

    kernel, layer_num = (3, 3), 4
    hidden = [8, 16, 64]
    input = 2
    batch_size = 4

    model = MyModel(input_dim=input, output_dim=input, hidden_dim=hidden, length=batch_size)






    loss_fn = nn.MSELoss()

    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

    MAX_EPOCH = 10
    f_data = np.load('process/test_data.npz')["input"]
    f_data = torch.Tensor(f_data)
    train_data = torch.utils.data.DataLoader(dataset=MyData(f_data, f_data), batch_size=batch_size,
                                             shuffle=True)
    print_output = []
    for e in range(1):
        step = 0
        use_pre = True
        for data in train_data:

            input, _ = data
            pred = []
            hidden_c = []
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            model.train()
            loss = 0
            for i in range(1, 6):
                if i == 1:
                    model_input = torch.Tensor(input[:, i, ...])
                    pred, hidden_c = model(model_input)
                else:
                    pred, hidden_c = model(torch.Tensor(pred), hidden_c)
                loss += loss_fn(pred, input[:, i + 1, ...]) # [4,8,1,5,121,360]
            print(pred.shape)
            if step == 0:
                print_output = pred
            else:
                print_output = torch.cat((print_output,pred),dim=0)
            print(print_output.shape)
            step += 1
            loss.backward()
            optimizer.step()
            scheduler.step()

    data = {
        "output": print_output
    }
    np.savez("7month_output.npz", **data)

