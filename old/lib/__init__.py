import numpy as np
from pathlib import Path
import imageio
from PIL import Image

pixel_size=2
root_path = Path("/Users/kimshan/Public/data/test")
BASE_SCALE = 1.005035

def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    return (data - _min) / (_max - _min)

def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data

def read_png_to_array(file_path):
    img = imageio.imread(file_path)
    return img

def path_valid(path):
    # 确定目录结构
    if (path / '20C.dat').exists() == False:
        return False
    if (path / '35C.dat').exists() == False:
        return False
    return True

def get_shape(path):
    # 检查文件结构
    if path_valid(path) == False:
        return False
    # 确定大小
    if (path / 'BadPixel.png').exists():
        with Image.open(path / 'BadPixel.png') as img:
            width, height = img.size
    elif '320_256_30' in str(path):
        width, height = (320, 256)
    elif '640_512_MW' in str(path):
        width, height = (640, 512)
    elif '2024-03-14-15-54-21-积分时间36ms' in str(path):
        width, height = (320, 256)
    elif '2024-02-28-10-54-12' in str(path):
        width, height = (320, 256)
    else:
        return False
    return (height,width)

def get_num(path,shape=None):
    if shape==None:
        shape = get_shape(path)
        if shape == False:
            return False
    n1 = int((path / '20C.dat').stat().st_size / (shape[0] * shape[1] * pixel_size))
    n2 = int((path / '35C.dat').stat().st_size / (shape[0] * shape[1] * pixel_size))
    return (n1,n2)

def get_all_image(data_path,num=None,shape=None):
    if num == None or shape == None:
        shape = get_shape(data_path.parent)
        num = get_num(data_path.parent,shape)
        num = num[0] if data_path.name == '20C.dat' else num[1]
    # 初始化一个数组来存储所有图像数据
    all_images = np.zeros((num, shape[0], shape[1]), dtype=np.float32)

    # 读取所有图像
    with open(data_path, 'rb') as f:
        for image_index in range(num):
            image_bytes = f.read(shape[0] * shape[1] * pixel_size)
            image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.float32)
            image_data = image_data.reshape(shape)
            all_images[image_index] = image_data / 65535
            
    return all_images

def get_all_voltage(data_path,scale=None,num=None,shape=None):
    all_images = get_all_image(data_path, num, shape)
    if scale == None:
        noice = np.std(all_images, axis=0)
        noice_gt = read_txt_to_matrix(data_path.parent / '像元噪声均值_V.txt')
        times = np.average(noice_gt) / np.average(noice)
        scale = int(round(times,3) / 1.001)
    return all_images*scale*BASE_SCALE

def get_noice_image(data_path,scale=None,num=None,shape=None):
    return np.std(get_all_voltage(data_path,scale,num,shape), axis=0)

def get_wave(data_path,point,scale=None,num=None,shape=None):
    all_voltage = get_all_voltage(data_path,scale,num,shape)
    return [image[point[0],point[1]] for image in all_voltage]