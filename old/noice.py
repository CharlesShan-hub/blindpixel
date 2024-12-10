import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 图像的尺寸和每个像素点的字节数
image_width = [320,640][0]
image_height =[256,512][0]
pixel_size = 2  # 16位整数占用2个字节

# base_path = Path("/Users/kimshan/Public/data/test/太原测试结果/BISTU_测试/Z0913-1-MW_TORNADO330/20240830")
root_path = Path("/Users/kimshan/Public/data/test")
# data_path = './320_256_30/Test_fengzhuang/A2A12/20'                  # 2.010045037110002    100 100 320x256
# data_path = './320_256_30/Test_fengzhuang/A2A12/15'                  # 2.0100449190955034   100 100 320x256
# data_path = './320_256_30/Test_fengzhuang/A2A12/10'                  # 2.0100449248834704   100 100 320x256
# data_path = './320_256_30/Test_fengzhuang/A2A12/5'                   # 2.010045084648963    100 100 320x256
data_path = './320_256_30/Test_fengzhuang/1CH_9'                     # 2.0100450084267845   100 100 320x256
# data_path = './320_256_30/Test_fengzhuang/1CH_2'                     # 2.010044806472494    100 100 320x256
# data_path = './太原测试结果/BISTU_测试/Z0913-1-MW_TORNADO330/20240830'  # 1.0050223801574496   100 2   (2.010044760314899/2)  320x256
# data_path = './非正式测试结果/Test-Result/2024-03-14-10-48-57-0314-14MS'# 1.0050208181655982   100 100 (2.0100416363311964/2) 640x512
# data_path = './非正式测试结果/Test-Result/2024-03-14-09-01-58-0314'     # 1.0050208101021811   100 100  (2.0100416202043623) 640x512
data_path = './320_256_30/Test-Results-New/2024-04-17-Datavalid/NO.20-30pin（冷屏）'

# 1.0050179329231093
# data_path = './非正式测试结果/JF/短波/Test-Result/2024-02-28-11-32-01-36.67'

# 2.010042298894569
# data_path = './320_256_30/Test-Z1406-B2-D11-SW'

base_path = root_path / data_path
data_path1 = base_path / '20C.dat'
data_path2 = base_path / '35C.dat'
txt_path = base_path / '像元噪声均值_V.txt'
number1 = 100
number2 = 100

# 读取txt文件并保存为numpy矩阵
def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    print(data.shape, np.nanmin(data), np.nanmax(data))
    return data

def get_noice_image(data_path,size):
    # 初始化一个数组来存储所有图像数据
    all_images = np.zeros((size, image_height, image_width), dtype=np.float32)

    # 读取所有图像
    with open(data_path, 'rb') as f:
        for image_index in range(size):
            image_bytes = f.read(image_width * image_height * pixel_size)
            image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.float32)
            image_data = image_data.reshape((image_height, image_width))
            all_images[image_index] = image_data / 32767
            
    # 计算方差
    return np.std(all_images, axis=0)

# 计算 平均噪声
noice1_image = get_noice_image(data_path1,number1) 
noice2_image = get_noice_image(data_path2,number2)

# 计算 txt 里边的数
float_average = read_txt_to_matrix(txt_path)

def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    print(_min,_max)
    return (data - _min) / (_max - _min)

def display_matrix(m1,m2,n1,n2):
    # m1 = m1[:, :300]
    # m2 = m2[:, :300]
    # n1 = n1[:, :300]
    # n2 = n2[:, :300]

    plt.subplot(2,4,1)
    plt.imshow(norm(n1*2), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Noice from 20C.dat\nMax={np.nanmax(n1*2)}\nMin={np.nanmin(n1*2)}\nnumber={number1}')
    plt.subplot(2,4,5)
    plt.imshow(norm(n2), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Noice from 35C.dat\nMax={np.nanmax(n2)}\nMin={np.nanmin(n2)}\nnumber={number2}')

    plt.subplot(2,4,2)
    plt.imshow(norm(m1), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Average of .dat\nMax={np.nanmax(m1)}\nMin={np.nanmin(m1)}')
    plt.subplot(2,4,6)
    plt.imshow(norm(m2), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Noice from .txt\nMax={np.nanmax(m2)}\nMin={np.nanmin(m2)}')

    plt.subplot(2,4,3)
    d1 = np.abs(m1-m2)
    plt.imshow(d1, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Differ avg & txt\nMax={np.nanmax(d1)}\nMin={np.nanmin(d1)}\nAvg={np.average(d1)}')
    plt.subplot(2,4,7)
    d2 = np.abs(n1-m2)
    plt.imshow(d2, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Differ 20C & txt\nMax={np.nanmax(d2)}\nMin={np.nanmin(d2)}\nAvg={np.average(d2)}')
    plt.subplot(2,4,4)
    d2 = np.abs(n1*2-m2)
    plt.imshow(d2, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Differ 20C*2 & txt\nMax={np.nanmax(d2)}\nMin={np.nanmin(d2)}\nAvg={np.average(d2)}')
    plt.subplot(2,4,8)
    d2 = np.abs(n1*2.01-m2)
    plt.imshow(d2, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Differ 20C*2.01 & txt\nMax={np.nanmax(d2)}\nMin={np.nanmin(d2)}\nAvg={np.average(d2)}')

    plt.suptitle(data_path, fontsize=16, fontweight='bold')
    plt.show()

def display_hist(m1,m2):
    plt.subplot(4, 1, 1)
    plt.hist(m1.ravel(), bins=4000, color='gray', alpha=0.7, range=(0,0.002))
    plt.title('Histogram of Noice from .dat')

    plt.subplot(4, 1, 2)
    plt.hist(m1.ravel()*5.025078141748802, bins=4000, color='gray', alpha=0.7, range=(0,0.002))
    plt.title('Histogram of Noice from .dat (x2)')

    plt.subplot(4, 1, 3)
    plt.hist(m2.ravel(), bins=4000, color='gray', alpha=0.7, range=(0,0.002))
    plt.title('Histogram of Noice from .txt')

    plt.subplot(4, 1, 4)
    plt.hist(np.abs(m1.ravel()*5.025078141748802-m2.ravel()), bins=4000, color='gray', alpha=0.7)
    plt.title('Differ of .dat and .txt')

    plt.suptitle(data_path, fontsize=16, fontweight='bold')
    plt.show()


# display_matrix((noice1_image+noice2_image)/2,float_average,noice1_image,noice2_image)
display_hist(noice1_image,float_average)

print(np.average(float_average) / np.average(noice1_image))