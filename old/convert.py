from pathlib import Path
import click
from PIL import Image
import numpy as np
from matplotlib.pyplot import imsave

@click.command()
@click.option('--src', default='/Users/kimshan/Public/data/test')
@click.option('--dest', default='/Users/kimshan/Public/data/mang_yuan')
@click.option('--pixel_size', default=2)
def main(src,dest,pixel_size):
    count = 0
    with open(Path(dest) / 'pathinfo.txt', '+w') as log_f:
        for path in [i for i in Path(src).rglob('*') if i.is_dir()]:
            # 确定目录结构
            if (path / '20C.dat').exists() == False:
                continue
            if (path / '35C.dat').exists() == False:
                continue
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
                continue
            count += 1
            log_f.write(f'{count}:{path}\n')
            # continue
            # 创建目标文件夹
            if (Path(dest) / f'exp{count}').exists() == False:
                (Path(dest) / f'exp{count}').mkdir()
                (Path(dest) / f'exp{count}' / '20C').mkdir()
                (Path(dest) / f'exp{count}' / '35C').mkdir()
            # Log
            print(f"[{count}]")
            # 读入并保存
            for data_type in ['20C','35C']:
                with open(path / f'{data_type}.dat', 'rb') as f:
                    for image_index in range(int((path / f'{data_type}.dat').stat().st_size / (width * height * pixel_size))):
                        image_bytes = f.read(width * height * pixel_size)
                        image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.float32)
                        image_data = image_data.reshape((height, width))
                        imsave(str(Path(dest) / f'exp{count}' / f'{data_type}' / f'{image_index:03d}.png'), image_data, cmap='gray', format='png')

if __name__ == "__main__":
    main()
