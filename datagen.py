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

def get_input_shape():
    return ds_trn[0][0].shape

classes_out = ds_trn.classes_out

train_dataloader = torch.utils.data.DataLoader(ds_trn,
    batch_size=preset_['bs'], shuffle=True)

val_dataloader = torch.utils.data.DataLoader(ds_val,
    batch_size=preset_['bs']*preset_['test_bs_multiplier'], shuffle=False)

if __name__=='__main__':

    plot_channel = 0
    for i in [train_dataloader, val_dataloader]:
        loop = iter(i)
        batch = loop.next()
        imgs = batch[0]
        print(batch[1])
        print(batch[1].shape)
        print(imgs.shape)
