from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 图像的尺寸和每个像素点的字节数
image_width = 320
image_height = 256
pixel_size = 2  # 16位整数占用2个字节

base_path = Path("/Users/kimshan/Public/data/test/太原测试结果/BISTU_测试/Z0913-1-MW_TORNADO330/20240830")
data_path = base_path / '20C.dat'
src_path = base_path / '20C'
if src_path.exists() == False:
    src_path.mkdir()

def save_image(image_data, filename):
    plt.imsave(str(filename), image_data, cmap='gray', format='png')

def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    print(_min,_max)
    return (data - _min) / (_max - _min)

# 读取所有图像
with open(data_path, 'rb') as f:
    for image_index in range(100):
        image_bytes = f.read(image_width * image_height * pixel_size)
        image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.float32)
        image_data = image_data.reshape((image_height, image_width))
        save_image(image_data, src_path / f'{image_index:03d}.png')