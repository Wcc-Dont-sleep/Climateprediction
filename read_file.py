# 读取冬季（十二月、一月、二月）海温、海表面热含量和位势高度场作为输入数据，以及次年夏季（六、七、八月）位势高度场作为输出数据：

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader,Dataset
import xarray as xr
import numpy as np

def read_data_file(SeaSurfaceTem,GeoHeight,HeatContent):
    """
    :param SeaSurfaceTem: 海表面热含量的nc文件，经纬度大小为181*360
    :param GeoHeight: 位势高度nc文件，经纬度大小为181*360
    :param HeatContent: 海表面热含量nc文件，经纬度大小为181*360
    :return: 储存处理好的数据在input_data.npz
    """
    # 读取冬季海温数据：

    f_sst = xr.open_dataset(SeaSurfaceTem)
    #12月1月2月的海温，从1850-1-1到2013-12-31号的温度
    sst_12 = f_sst.tos.loc[f_sst.time.dt.month.isin([12])].loc['1850-01-01':'2013-12-31', :, :]
    sst_1 = f_sst.tos.loc[f_sst.time.dt.month.isin([1])].loc['1851-01-01':'2014-12-01', :, :]
    sst_2 = f_sst.tos.loc[f_sst.time.dt.month.isin([2])].loc['1851-01-01':'2014-12-01', :, :]

    #大小是(164,181,360),其中year=164,lat=181,lon=360
    sst_12 = np.array(sst_12)
    sst_1 = np.array(sst_1)
    sst_2 = np.array(sst_2)

    sst_12 = np.nan_to_num(sst_12)
    sst_1 = np.nan_to_num(sst_1)
    sst_2 = np.nan_to_num(sst_2)

    # 计算海温冬季平均：
    sst_avg = (np.array(sst_12)+np.array(sst_1)+np.array(sst_2))/3
    #print("Avg",sst_avg.shape)

    # 读取冬季海表面热含量数据：
    f_hcont = xr.open_dataset(HeatContent)

    hcont_12 = f_hcont.hcont300.loc[f_hcont.time.dt.month.isin([12])].loc['1850-01-01':'2013-12-31', :, :]
    hcont_1 = f_hcont.hcont300.loc[f_hcont.time.dt.month.isin([1])].loc['1851-01-01':'2014-12-01', :, :]
    hcont_2 = f_hcont.hcont300.loc[f_hcont.time.dt.month.isin([2])].loc['1851-01-01':'2014-12-01', :, :]

    hcont_12 = np.nan_to_num(hcont_12)
    hcont_1 = np.nan_to_num(hcont_1)
    hcont_2 = np.nan_to_num(hcont_2)



    # 计算海表面热含量冬季平均：
    heat_cont_avg = (np.array(hcont_12)+np.array(hcont_1)+np.array(hcont_2))/3


    # 读取冬季位势高度数据：
    f_hgt = xr.open_dataset(GeoHeight)


    hgt_12 = f_hgt.zg.loc[f_hgt.time.dt.month.isin([12])].loc['1850-01-01':'2013-12-31', 50000, :, :]
    hgt_1 = f_hgt.zg.loc[f_hgt.time.dt.month.isin([1])].loc['1851-01-01':'2014-12-01', 50000, :, :]
    hgt_2 = f_hgt.zg.loc[f_hgt.time.dt.month.isin([2])].loc['1851-01-01':'2014-12-01', 50000, :, :]

    # 计算位势高度冬季平均：
    geo_height_avg = (np.array(hgt_12)+np.array(hgt_1)+np.array(hgt_2))/3
    #print(geo_height_avg.shape)


    # 读取夏季500hPa位势高度数据：
    hgt_678 = f_hgt.zg.loc[f_hgt.time.dt.month.isin([6, 7, 8])].loc['1851-01-01':'2014-12-31', 50000, :, :]
    #print(hgt_678.shape)

    # 计算500hPa位势高度夏季平均：
    hgt_678 = np.array(hgt_678).reshape((3, 164, 181, 360))
    hgt_jja_avg = hgt_678.mean(0)
    #print(hgt_jja.shape)

    #最大最小标准化
    sst_avg = sst_avg.reshape(164*181*360)
    sst_avg = (sst_avg-min(sst_avg))/(max(sst_avg)-min(sst_avg))
    sst_avg = sst_avg.reshape(164,181,360)

    heat_cont_avg = heat_cont_avg.reshape(164 * 181 * 360)
    heat_cont_avg = (heat_cont_avg - min(heat_cont_avg)) / (max(heat_cont_avg) - min(heat_cont_avg))
    heat_cont_avg = heat_cont_avg.reshape(164, 181, 360)

    geo_height_avg = geo_height_avg.reshape(164 * 181 * 360)
    geo_height_avg = (geo_height_avg - min(geo_height_avg)) / (max(geo_height_avg) - min(geo_height_avg))
    geo_height_avg = geo_height_avg.reshape(164, 181, 360)

    hgt_jja_avg = hgt_jja_avg.reshape(164 * 181 * 360)
    hgt_jja_avg = (hgt_jja_avg - min(hgt_jja_avg)) / (max(hgt_jja_avg) - min(hgt_jja_avg))
    hgt_jja_avg = hgt_jja_avg.reshape(164, 181, 360)



    #一共是164个月份，输入数据有三个特征，分别是海温，海表面热含量，冬季位势。
    # 处理完成的数据为
    # 海温：sst_avg
    # 海表面热含量：heat_cont_avg
    # 冬季位势：geo_height_avg
    # 大小均为（164，181，360）

    #print("sst_avg",sst_avg.shape)
    #print("heat",heat_cont_avg.shape)
    #print("geo",geo_height_avg.shape)


    sst_avg = np.array(sst_avg).reshape(164,181,360,1)
    heat_cont_avg = np.array(heat_cont_avg).reshape(164, 181, 360,1)
    geo_height_avg = np.array(geo_height_avg).reshape(164, 181, 360,1)

    #夏季位势高度，label
    hgt_jja_avg = np.array(hgt_jja_avg).reshape(164,1,181,360)

    sst_avg = torch.Tensor(sst_avg)
    heat_cont_avg = torch.Tensor(heat_cont_avg)
    geo_height_avg = torch.Tensor(geo_height_avg)
    hgt_jja_avg = torch.Tensor(hgt_jja_avg)

    input_data = torch.cat((sst_avg,heat_cont_avg,geo_height_avg),dim=3)
    #print(input_data.shape)

    input_data = np.array(input_data).reshape(164, 1, 181, 360, 3)
    input_data = torch.Tensor(input_data)
    input_data = input_data.permute(0, 1, 4, 2, 3)
    data = {
        "input": input_data,
        "output": hgt_jja_avg # 夏季位势高度
    }
    np.savez("process/input_data.npz", **data)
    return input_data

"""调用方法示例"""
#read_data_file('Data/tos_regrid.nc','Data/zg_interp.nc','Data/hcont_interp1.nc')


