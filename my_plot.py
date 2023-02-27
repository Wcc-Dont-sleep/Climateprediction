"""
画图代码
"""
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_helper(data, lons, lats, filename='predict.png'):

    fig = plt.figure(figsize=(15, 15))
    proj = ccrs.PlateCarree(central_longitude=180)
    leftlon, rightlon, lowerlat, upperlat = (0, 359, -60, 60)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]

    # ax = fig.subplot(1, 1, 1, projection=proj)
    ax = fig.add_axes([0.36, 0.26, 0.30, 0.25], projection=proj)
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=1)
    # ax.add_feature(cfeature.LAND.with_scale('110m'))
    ax.set_xticks(np.arange(leftlon, rightlon + 60, 60), crs=ccrs.PlateCarree())  # 网格线标签显示范围稍大于绘制区域
    ax.set_yticks(np.arange(lowerlat, upperlat + 30, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(30))
    ax.set_ylabel('Latitude', fontsize=8)
    plt.tick_params(labelsize=8)
    c1 = ax.contourf(lons, lats, data, zorder=0,
                     # levels=[0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
                     extend='both', transform=ccrs.PlateCarree(), cmap=plt.cm.get_cmap('OrRd'))
    plt.grid(which='major', linestyle=':')
    ax.set_title('*****', loc='center')

    position = fig.add_axes([0.36, 0.25, 0.3, 0.015])
    plt.colorbar(c1, cax=position, fraction=0.05, extend='both',
                      # ticks=[0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
                      pad=0.2, orientation='horizontal')
    plt.savefig(filename, dpi=500, bbox_inches='tight')  # 保存图片


if __name__ == '__main__':
    data = np.ones((121, 360))
    lons = np.linspace(0, 360, 360)
    lats = np.linspace(-60, 60, 121)
    plot_helper(data, lons, lats)
    plt.show(block=True)
