# -*- coding: utf-8 -*-
# @File : train.py
# @Author : 秦博
# @Time : 2022/07/16
import os
os.environ["LOGURU_INFO_COLOR"] = "<green>"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import time
import torch
torch.cuda.manual_seed_all(42)
# torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import netCDF4 as nc
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progress.spinner import MoonSpinner
from loguru import logger

from data.preprocess import get_scaler
from data.dataset import TrainingSet, ForecastingSet
from data.postprocess import get_climatology, get_land, plot_helper, index_helper
from constrain_moments import K2M
from modelv3 import MyModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.add(f"./train.log", enqueue=True)
raw_path = os.path.join(r'./file/', 'HadISST_sst.nc')
save_path = os.path.join(r'./file/', 'model.pth')
data_path = os.path.join(r'./file/', 'data.npz')
ncfile = nc.Dataset(raw_path, mode='r')
lons = np.array(ncfile.variables['longitude'][:])
lats = np.array(ncfile.variables['latitude'][:])
climatology_scope=[1440, 1800]
epoch_num = 10000
batch_size = 6
length = 9
ground_length = 6
extend = 18
scaler = get_scaler(raw_path)
climatology = get_climatology(raw_path, scope=climatology_scope)
land = get_land(raw_path)
best_acc = np.inf
best_iter = 0
# constraints = torch.zeros((64, 8, 8)).to(device)
# ind = 0
# for i in range(0, 8):
#     for j in range(0, 8):
#         constraints[ind, i, j] = 1
#         ind += 1

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred), torch.log(actual)))

class I_RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log((1 - pred)), torch.log((1 - actual))))


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, pred, actual):
        delta = 2 * torch.abs(actual - 0.5)
        penalty = 1 / torch.clamp(torch.sigmoid(-torch.tan(self.pi * delta - self.pi / 2)), 0.0001, 0.9999)
        mse = torch.pow((pred - actual), 2)
        return torch.mean(penalty * mse)


if __name__=='__main__':
    train_loader = torch.utils.data.DataLoader(dataset=TrainingSet(data_path, './file/land.npz', length, 1740), batch_size=batch_size, shuffle=True)
    forecast_loader = torch.utils.data.DataLoader(dataset=ForecastingSet(data_path, './file/land.npz', length, 1800), batch_size=batch_size, shuffle=False)
    model = MyModel(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, factor=0.1, verbose=True)
    criterion_1 = nn.MSELoss().to(device)
    criterion_2 = nn.L1Loss().to(device)
    criterion_3 = RMSLELoss().to(device)
    criterion_4 = I_RMSLELoss().to(device)
    criterion_5 = Loss().to(device)

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        template = ("load model weights from: {}.")
        logger.info(template.format(save_path))

    for epoch in range(epoch_num):
        for step, input in enumerate(train_loader):

            if epoch == 0:
                break

            input, mask = input
            input = input.to(device).to(torch.float32).unsqueeze(2)
            mask = mask.to(device).to(torch.float32)

            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            model.to(device).train()
            loss = 0
            for i in range(length - 1):
                if i < ground_length:
                    pred = model(input[:, i, ...], (i==0))
                else:
                    pred = model(pred, (i==0))
                # loss += (criterion_1(pred, input[:, i+1, ...]) + criterion_2(pred, input[:, i+1, ...]))
                loss += criterion_5(pred, input[:, i+1, ...])
                # pred = pred * mask.unsqueeze(1)

            # k2m = K2M([8, 8]).to(device)
            # for b in range(0, model.phycell.cell_list[0].input_dim):
            #     filters = model.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
            #     m = k2m(filters.double())
            #     m = m.float()
            #     loss += criterion_1(m, constraints)

            if step % 10 == 0:
                template = ("epoch {} - step {}: loss is {:1.5f}.")
                logger.info(template.format(epoch, step, loss))

            loss.backward()
            optimizer.step()
            del pred, loss
            torch.cuda.empty_cache()

        # spinner = MoonSpinner('Testing ')
        # # test....
        # for step, input in enumerate(forecast_loader):
        #     input, mask = input
        #     input = input.to(device).to(torch.float32).unsqueeze(2)
        #     mask = mask.to(device).to(torch.float32)
        #     model.eval()
        #     torch.set_grad_enabled(False)
        #     shape = input[0, 0, 0, ...].shape
        #     val_acc = 0
        #     output = []
        #     for i in range(length + extend):
        #         if i < ground_length:
        #             pred = model(input[:, i, ...], (i==0))
        #             if i < ground_length - 1:
        #                 val_acc += (criterion_1(pred, input[:, i+1, ...]) + criterion_2(pred, input[:, i+1, ...]))
        #         else:
        #             pred = model(pred, (i==0))
        #             output.append(pred)
        #         # pred = pred * mask.unsqueeze(1)
        #
        #     template = ("epoch {} - forecast error is {:1.5f}.")
        #     logger.info(template.format(epoch, val_acc))
        #     pred = torch.cat(output, dim=1).detach().cpu().numpy()
        #     np.savez(f'pred-{length + extend - ground_length}.npz', data=pred)
        #     for i in range(length + extend - ground_length):
        #         year = 1870 + int((ncfile['sst'].shape[0] + i) / 12)
        #         month = (ncfile['sst'].shape[0] + i) % 12
        #         result = np.reshape(scaler.inverse_transform(np.reshape(pred[0, i], (1, -1))), shape)
        #         plot_helper(result, lons, lats, climatology=climatology[month], land=land, save=True, filename=f'pred_{year}_{month+1}.png')
        #         nino3, nino4, nino34 = index_helper(result, climatology=climatology[month])
        #         template = ("epoch {} - forecast year {} month {}: nino3 index is {:1.5f}, nino4 index is {:1.5f}, nino3.4 index is {:1.5f}.")
        #         logger.info(template.format(epoch, year, month + 1, nino3, nino4, nino34))
        #     spinner.next()
        #     del pred, output
        #     torch.cuda.empty_cache()
        # spinner.finish()
        # scheduler.step(val_acc)

        spinner = MoonSpinner('Testing ')
        # test....
        val_acc = 0
        for step, input in enumerate(forecast_loader):
            input, mask = input
            input = input.to(device).to(torch.float32).unsqueeze(2)
            mask = mask.to(device).to(torch.float32)
            model.eval()
            torch.set_grad_enabled(False)
            for i in range(length - 1):
                if i < ground_length:
                    pred = model(input[:, i, ...], (i==0))
                else:
                    pred = model(pred, (i==0))
                val_acc += (criterion_1(pred, input[:, i+1, ...]) + criterion_2(pred, input[:, i+1, ...]))

            spinner.next()
            del pred
            torch.cuda.empty_cache()

        template = ("epoch {} - forecast error is {:1.5f}.")
        logger.info(template.format(epoch, val_acc))
        spinner.finish()
        scheduler.step(val_acc)

        if val_acc < best_acc:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, save_path)
            print('Model saved successfully:', save_path)
            best_acc = val_acc
            best_iter = epoch

        template = ("-----------epoch {} finish!, current best results are from epoch {}.-----------")
        logger.info(template.format(epoch, best_iter))
