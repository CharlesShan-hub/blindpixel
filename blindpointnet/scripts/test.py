import blindspot as bs
from clib.utils import glance
import numpy as np

METHOD = "curve6"
bs.BASE_PATH = "/Users/kimshan/Public/data/blindpoint/source"

for k, info in bs.get_all_proj_info().items():
    bs.load_bad_mask(info,METHOD)
    bs.load_low_voltages(info)
    bs.load_high_voltages(info)
    glance(
        image = [np.average(info['vol_h'], axis=0), np.average(info['vol_l'], axis=0), info['bad']],
        title = [f"{k}-High", f"{k}-Low", f"({info['active']}) Bad:{(info['bad']==255).sum()}:{(info['bad']==0).sum()}"],
        figsize = (120,20)
    )  
    bs.delete_info(info)

