#!/usr/bin/python
# -*- encoding: utf-8 -*-
import config
from config import args

import torch

if args.prefered_device is not None: device = torch.device(args.prefered_device)
else: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from datagen import train_dataloader, val_dataloader
from model import net
net = net.to(device)


# class_wts = train_dataloader.dataset.class_weights().to(device)
# print(f'[ {__name__} ]', 'class wts:', class_wts)

# criterion = torch.nn.BCEWithLogitsLoss(weight=class_wts)
from criterion import CustomLoss, metrics
criterion = CustomLoss()

# from utils.metrics import m_config
# metrics = m_config['BCEWithLogitsLoss']

optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.scheduler_patience, factor=args.scheduler_factor, verbose=True)

from utils.trends import Trends_
trend = Trends_(prev_db_path=None, img_path=config.trend_plot_path)

from training_loop import fit
if __name__=='__main__':
    fit(
        net, args.nepoch, config.preset['bs'], criterion, 
        optimizer, train_dataloader, config.preset, args,
        val_dl=val_dataloader,
        scheduler=scheduler,
        ckpt_path=config.ckpt_path,
        save_to_checkpoint=config.preset['save_to_checkpoint'],
        trend=trend,
        metric=metrics,
        val_start=args.val_start,
        val_only=args.val_only,
        device = device
    )
