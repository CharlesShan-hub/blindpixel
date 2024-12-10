from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 图像的尺寸和每个像素点的字节数
image_width = 320
image_height = 256
pixel_size = 2  # 16位整数占用2个字节

base_path = Path("/Users/kimshan/Public/data/test/太原测试结果/BISTU_测试/Z0913-1-MW_TORNADO330/20240830")
data_path = base_path / '20C.dat'
txt_path = base_path / '低温均值_V.txt'

# 读取txt文件并保存为numpy矩阵
def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data

# 初始化一个数组来存储所有图像数据
all_images = np.zeros((100, image_height, image_width), dtype=np.float32)

# 读取所有图像
with open(data_path, 'rb') as f:
    for image_index in range(100):
        image_bytes = f.read(image_width * image_height * pixel_size)
        image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.float32)
        image_data = image_data.reshape((image_height, image_width))
        all_images[image_index] = image_data

# 计算所有图像的平均值
average_image = np.mean(all_images, axis=0)

# 计算 txt 里边的数
float_average = read_txt_to_matrix(txt_path)

def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    print(_min,_max)
    return (data - _min) / (_max - _min)
 
def display_matrix(m1,m2):
    plt.subplot(1,3,1)
    plt.imshow(m1, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Average from .dat\nMax={np.nanmax(m1)}\nMin={np.nanmin(m1)}')
    plt.subplot(1,3,2)
    plt.imshow(m2, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Average from .txt\nMax={np.nanmax(m2)}\nMin={np.nanmin(m2)}')
    plt.subplot(1,3,3)
    d = m1/m2
    plt.imshow(d, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Max={np.nanmax(d)}\nMin={np.nanmin(d)}')
    plt.show()

display_matrix(average_image,float_average)
