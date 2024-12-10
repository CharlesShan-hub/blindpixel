import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    return (data - _min) / (_max - _min)

# 读取txt文件并保存为numpy矩阵
def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    print(data.shape, np.nanmin(data), np.nanmax(data))
    return data

def read_png_to_array(file_path):
    img = imageio.imread(file_path)
    return img

# 显示矩阵
def display_matrix(m):
    plt.subplot(2, 4, 1)
    plt.imshow(norm(m['low']), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Low Temperature Mean')

    plt.subplot(2, 4, 2)
    plt.imshow(norm(m['high']), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('High Temperature Mean')

    plt.subplot(2, 4, 3)
    plt.imshow(norm(m['noice']), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Noice')

    plt.subplot(2, 4, 4)
    plt.imshow(m['bad'], cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Bad')

    plt.subplot(2, 4, 5)
    plt.imshow(norm(m['n']), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Difference (High - Low) / Noice')

    plt.subplot(2, 4, 6)
    plt.imshow(np.isnan(m['n_normalized']), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Difference (High - Low) / Noice == Nan')

    plt.subplot(2, 4, 7)
    plt.imshow(m['color'], interpolation='nearest')
    plt.colorbar()
    plt.title('Difference (High - Low) / Noice')

    plt.subplot(2, 4, 8)
    plt.imshow(m['color_bg'], interpolation='nearest')
    plt.colorbar()
    plt.title('Difference + red')

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()



# 主函数
def main():
    base_path = Path("/Users/kimshan/Public/data/test/太原测试结果/BISTU_测试/Z0913-1-MW_TORNADO330/20240830")
    low_path = base_path / '低温均值_V.txt'
    high_path = base_path / '高温均值_V.txt'
    noice_path = base_path / '像元噪声均值_V.txt'
    bad_path = base_path / 'BadPixel.png'

    m = {
        'low': read_txt_to_matrix(low_path),
        'high': read_txt_to_matrix(high_path),
        'noice': read_txt_to_matrix(noice_path),
        'bad': read_png_to_array(bad_path)
    }
    
    m['sub'] = m['high'] - m['low']
    # print("Indices of NaN values (m['sub']):", np.where(np.isnan(m['sub'])))
    m['n'] = m['sub'] / m['noice']
    # print("Indices of NaN values (m['n']):", np.where(np.isnan(m['n'])))
    m['n_normalized'] = norm(m['n'])
    m['color'] = np.stack([m['n_normalized'], m['n_normalized'], m['n_normalized']], axis=-1)
    m['color'][np.isnan(m['n_normalized'])] = [0,1,0]  # 设置绿色标记
    m['color'][m['n_normalized'] > 0.98] = [0,0,1]  # 设置绿色标记
    m['color_bg'] = np.stack([m['n_normalized'], m['n_normalized'], m['n_normalized']], axis=-1)
    m['color_bg'][m['bad'] == 255] = [1,0,0]  # 设置红色标记

    display_matrix(m)

if __name__ == '__main__':
    main()
