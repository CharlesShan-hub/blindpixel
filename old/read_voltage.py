import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from lib import *
import os

with open('/Users/kimshan/Public/data/mang_yuan/pathinfo.csv', 'r') as csv_file:
    # 创建一个 csv 读取器对象
    csv_reader = csv.reader(csv_file)
    # 跳过标题行
    next(csv_reader)
    # 读取 csv 文件的内容
    for row in csv_reader:
        # 获取索引和路径
        index = row[0]
        _path = row[2]
        path = root_path / _path
        shape = (int(row[3]),int(row[4]))
        num = (int(row[5]),int(row[6]))

        if (path / '像元噪声均值_V.txt').exists() == False:
            print('❌',index,path,shape,num)
            continue
        print(index,path)

        for filename in os.listdir(path):
            if filename.endswith('.ini'):
                print(f"✅Found .ini file: {filename}")
                break  # 找到一个.ini文件后可以停止循环

        # all_img = get_all_image(path / '20C.dat') * 2.01
        # noice = np.std(all_img, axis=0)
        all_vol = get_all_voltage(path / '20C.dat')
        noice = np.std(all_vol,axis=0)
        noice_gt = read_txt_to_matrix(path / '像元噪声均值_V.txt')
        times = np.average(noice_gt) / np.average(noice)
        # differ = np.abs(noice/noice_gt)
        # std = np.std(differ)
        print(index, times)#,times,std,np.nanmax(differ),np.nanmin(differ))#,path)

# import csv

# # 输入的 txt 文件路径
# input_txt_path = '/Users/kimshan/Public/data/mang_yuan/pathinfo.csv'
# # 输出的 csv 文件路径
# output_csv_path = '/Users/kimshan/Public/data/mang_yuan/pathinfo2.csv'

# with open('/Users/kimshan/Public/data/mang_yuan/pathinfo.csv', 'r') as csv_file:
#     with open(output_csv_path, 'w', newline='') as csv_2_file:
#         # 创建一个 csv 读取器对象
#         csv_reader = csv.reader(csv_file)
#         # 创建一个 csv 写入器对象
#         csv_writer2 = csv.writer(csv_2_file)
#         csv_writer2.writerow(['index', 'valid', 'path', 'height', 'width', '20C_num', '35C_num', 'scale'])
#         # 跳过标题行
#         next(csv_reader)
#         # 读取 csv 文件的内容
#         for row in csv_reader:
#             # 获取索引和路径
#             index = row[0]
#             _path = row[1]
#             path = root_path / _path
#             shape = get_shape(path)
#             num = get_num(path,shape)
#             valid = False
#             if (path / '像元噪声均值_V.txt').exists() == False:
#                 scale = 0
#             else:
#                 noice = get_noice_image(path / '20C.dat', num[0], shape)
#                 noice_gt = read_txt_to_matrix(path / '像元噪声均值_V.txt')
#                 times = np.average(noice_gt) / np.average(noice)
#                 scale = int(round(times,3) / 1.001)
#                 if scale > 1:
#                     valid = True


#             csv_writer2.writerow([index, valid, _path, shape[0], shape[1], num[0], num[1], scale])



# # 打开 txt 文件并读取内容
# with open(input_txt_path, 'r') as txt_file:
#     # 创建一个 csv 文件对象，指定输出的 csv 文件路径
#     with open(output_csv_path, 'w', newline='') as csv_file:
#         # 创建一个 csv 写入器对象
#         csv_writer = csv.writer(csv_file)
        
#         # 写入 CSV 文件的标题行
#         csv_writer.writerow(['index', 'path'])
        
#         # 读取 txt 文件的内容
#         for line in txt_file:
#             # 去除行尾的换行符
#             line = line.strip()
#             # 检查行是否为空
#             if not line:
#                 continue
            
#             # 按冒号分割行，得到索引和路径
#             index, path = line.split(':')
            
#             # 写入 CSV 文件
#             csv_writer.writerow([index, path])

# print(f'Conversion completed. CSV file saved to {output_csv_path}')
