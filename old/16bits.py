from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 图像的尺寸和每个像素点的字节数
image_width = 320
image_height = 256
pixel_size = 2  # 16位整数占用2个字节

base_path = Path("/Users/kimshan/Public/data/test/太原测试结果/BISTU_测试/Z0913-1-MW_TORNADO330/20240830")
low_data_path = base_path / '20C.dat'
low_img_path = base_path / '低温均值_V.txt'

def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    print(_min,_max)
    return (data - _min) / (_max - _min)

# 读取txt文件并保存为numpy矩阵
def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    print(data.shape, np.nanmin(data), np.nanmax(data))
    return data


def display_matrix(m1,m2):
    plt.subplot(1,2,1)
    plt.imshow(m1, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Average Temperature')
    plt.subplot(1,2,2)
    plt.imshow(m2, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Average Temperature')
    plt.show()

# 初始化一个数组来存储所有图像数据
all_images = np.zeros((100, image_height, image_width), dtype=np.float32)

# 读取所有图像
with open(low_data_path, 'rb') as f:
    for image_index in range(100):
        image_bytes = f.read(image_width * image_height * pixel_size)
        image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.float32)
        image_data = image_data.reshape((image_height, image_width))
        all_images[image_index] = image_data

# 计算所有图像的平均值
average_image = np.mean(all_images, axis=0)

display_matrix(image_data,read_txt_to_matrix(low_img_path))
