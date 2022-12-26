'''define "ds_trn" and "ds_val" in this file.'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm
from config import preset, args

# -----------------------------------------------------------------------------
# implement ds_trn and ds_val here
# -----------------------------------------------------------------------------
from datasets.icentia11k import Icentia
# ds_trn = Icentia(preset['ds_root'], shuffle=True, transforms=trn_tfms)
ds_trn = Icentia(preset['ds_root'], shuffle=True)
ds_val = Icentia(preset['ds_root_val'], shuffle=False)

def get_input_shape():
    return ds_trn[0][0].shape

classes_out = ds_trn.classes_out

train_dataloader = torch.utils.data.DataLoader(ds_trn,
    batch_size=preset['bs'], shuffle=True)

val_dataloader = torch.utils.data.DataLoader(ds_val,
    batch_size=preset['bs']*preset['test_bs_multiplier'], shuffle=False)

if __name__=='__main__':
    for i in [train_dataloader, val_dataloader]:
        loop = iter(i)
        # batch = loop.next()
        batch = next(loop)
        for i in batch:
            print(i.shape)
