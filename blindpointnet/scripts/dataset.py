import blindspot as bs
from torch.utils.data import Dataset
import click
import torch
import torch.nn.functional as TF

class BlindPoints(Dataset):
    def __init__(self, base_path, method):
        bs.BASE_PATH = base_path
        info = {}
        for k, v in bs.get_all_proj_info().items():
            if v['active'] and v['num_l'] == 100 and v['num_h'] == 100:
                info[k] = v
        self.data = list(info.values())
        self.method = method

    def __getitem__(self, idx):
        # 点云数据
        info = self.data[idx]
        bs.load_low_voltages(info)
        bs.load_high_voltages(info)
        vol_l_tensor = torch.tensor(info['vol_l'])
        vol_h_tensor = torch.tensor(info['vol_h'])
        padding_size = 1
        vol_l_padded = TF.pad(vol_l_tensor, (padding_size, padding_size, padding_size, padding_size),mode='reflect')
        vol_h_padded = TF.pad(vol_h_tensor, (padding_size, padding_size, padding_size, padding_size),mode='reflect')
        vol_l_unfold = vol_l_padded.unfold(dimension=1, size=3, step=1).unfold(dimension=2, size=3, step=1)
        vol_h_unfold = vol_h_padded.unfold(dimension=1, size=3, step=1).unfold(dimension=2, size=3, step=1)
        vol_l_shaped = vol_l_unfold.reshape(info['num_l'], info['width'] * info['height'], 3, 3).permute(1, 0, 2, 3)
        vol_h_shaped = vol_h_unfold.reshape(info['num_h'], info['width'] * info['height'], 3, 3).permute(1, 0, 2, 3)

        # 类别数据
        bs.load_bad_mask(info,self.method)
        bad_cls = info['bad'].reshape(info['width'] * info['height'])

        # 清除内存
        bs.delete_info(info)
        
        return torch.cat((vol_l_shaped, vol_h_shaped), dim=1), bad_cls, info
    
    def __len__(self):
        return len(self.data)

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--method", type=str, default="curve6")
def main(**kwargs):
    dataset = BlindPoints(kwargs['dataset_path'],kwargs['method'])
    for data,gt,info in dataset:
        print(info)
        print(data.shape,gt.shape,(gt == 1).sum())

if __name__ == '__main__':
    main()
