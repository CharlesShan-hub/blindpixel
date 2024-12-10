import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from lib import *

r = 3
# root_path = Path('/Users/kimshan/Public/data/mang_yuan')
# base_path = root_path / '320_256_30/Test_fengzhuang/A2A12/5'
base_path = root_path / '320_256_30/Test-Z1406-C3-D12-SW/Test7'
# base_path = root_path / '320_256_30/Test-Z1406-B1-E13-SW-WL'
# base_path = root_path / '320_256_30/Test-Result/2024-03-19-14-50-25-Tint=36ms'
# base_path = root_path / '320_256_30/Test-Z1406-B2-D11-SW'
print(base_path)
bad_path = base_path / 'BadPixel.png' # 只有 255 和 0

def float_to_rgb16(value):
    """
    将0到1之间的小数映射到黑色到白色的RGB16进制字符串。
    
    参数:
    value -- 介于0到1之间的小数, 表示灰度值。
    
    返回:
    RGB16进制字符串,例如 '#FFFFFF' 表示白色，'#000000' 表示黑色。
    """
    # 将0到1的值映射到0到255的范围
    gray_value = int(value * 255)
    
    # 将灰度值转换为16进制字符串，并确保它是两位数
    hex_value = format(gray_value, '02x')
    
    # 返回RGB16进制字符串，由于是灰度，所以R、G、B值相同
    return f'#{hex_value}{hex_value}{hex_value}'

def get_bad_list(bad):
    bad_list = []
    for i in range(bad.shape[0]):
        for j in range(bad.shape[1]):
            if bad[i, j] == 255:
                bad_list.append((i,j))
    return bad_list

shape = get_shape(base_path)
noice = get_noice_image(base_path / '20C.dat')
average_noice = np.average(noice)
average_image = np.average(get_all_voltage(base_path / '20C.dat'),axis=0)
bad_image = read_png_to_array(bad_path)
bad_list = get_bad_list(bad_image)

for p in bad_list:
    fig, axs = plt.subplots(2*r+1, 2*r+1, figsize=(24, 24))
    color_image = average_image[max(p[0]-r,0):min(p[0]+r+1,shape[0]), max(p[1]-r,0):min(p[1]+r+1,shape[1])]
    color_image = (color_image - np.min(color_image)) / (np.max(color_image) - np.min(color_image))
    for i in range(7):
        for j in range(7):
            plt.subplot(7,7,1+7*i+j)
            point = (p[0]-r+i, p[1]-r+j)
            if point[0]<0 or point[0]>(shape[0]-1) or point[1]<0 or point[1]>(shape[1]-1):
                continue
            if point in bad_list:
                color = 'red'
            else:
                color = 'green'
            try:
                axs[i][j].set_facecolor(float_to_rgb16(color_image[i][j]))
            except:
                axs[i][j].set_facecolor('black')
            axs[i][j].plot(get_wave(base_path / '20C.dat', point), color=color)
            axs[i][j].set_ylim(average_image[point[0],point[1]]-average_noice*3,average_image[point[0],point[1]]+average_noice*3)
            axs[i][j].set_title(f'{point}')
    plt.savefig(f"./img/bad_{p[0]}_{p[1]}")
    plt.clf()