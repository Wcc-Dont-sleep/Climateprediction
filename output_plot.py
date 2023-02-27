import os
import cmaps
import cartopy
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# conda install -c conda-forge cartopy


def plot_helper(data, lons, lats, save=True, filename='pred.png'):
    # fill land
    # data[data == 0] = np.nan

    x_extent = list(range(-180, 180, 30))
    y_extent = list(range(-90, 91, 20))

    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    ax.add_feature(cfeature.LAND)#添加陆地
    ax.add_feature(cfeature.COASTLINE, lw=0.3)#添加海岸线
    ax.add_feature(cfeature.RIVERS, lw=0.25)#添加河流
    ax.add_feature(cfeature.LAKES)#指定湖泊颜色为红色#添加湖泊
    ax.add_feature(cfeature.OCEAN)#添加海洋
    ax.set_xticks(x_extent, crs=ccrs.PlateCarree())
    ax.set_yticks(y_extent, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    m = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree(central_longitude=180), cmap=cmaps.GMT_panoply)

    posn = ax.get_position()
    cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.01, posn.height])

    ax.coastlines()
    plt.colorbar(m, cax=cbar_ax)
    if save:
        plt.savefig(filename)
        plt.cla(); plt.clf(); plt.close()


if __name__ == '__main__':

    data = np.load("process/7month_output.npz")["output"]
    print(data.shape)
    data = data[7,0,1]
    print(data.shape)
    lons = np.linspace(-180, 180, 360)
    lats = np.linspace(-60, 61, 121)
    plot_helper(data, lons, lats, save=True)
    plt.show()
