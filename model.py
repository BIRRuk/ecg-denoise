import torch
import torch.nn as nn

from config import classes_out, input_shape, ckpt_path, args, ckpt_dir, plot_dir

import torchvision as tv
net = tv.models.resnet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
net.fc = nn.Linear(in_features=512, out_features=classes_out, bias=True)

def modify_net_(model): # visualize seed output
    print(f'[ {__name__} ]', 'modifying net')
    return net

if __name__=="__main__":
    print(net)
    import os
    from utils.utils import save_str_net
    save_str_net(net, pathj=os.path.join(ckpt_dir, 'str_net--1_exp.txt'))
    print(net(torch.randn(1, classes_in, img_sze, img_sze)))
    # from torchsummary import summary
    # summary(net, tuple(input_shape)) # !cant display torchsummary of densenet

    from trn import train_dataloader, val_dataloader, device, optimizer, criterion

    dl = val_dataloader
    iter_ = iter(dl)
    # for _ in range(2):
    #     batch = iter_.next()
    batch = iter_.next()
    imgs = batch[0]
    print(f'[ {__name__} ]', 'labels:')
    print(batch[1])
    print(batch[1].shape)
    print(f'[ {__name__} ]', 'imgs shape', imgs.shape)

    devs = [ 'cpu' ]
    if torch.cuda.is_available(): devs.append('cuda:0')
    import time
    for dev in devs:
        device = torch.device(dev)
        print(f'[ {__name__} ]', 'running on device:', device)
        with torch.autograd.set_detect_anomaly(True):
            begin_time = time.time()
            net = net.to(device)
            imgs = imgs.to(device)

            out = net(imgs)
            print(f'[ {__name__} ]', 'out:')
            print(out)
            print(out.shape)
            loss = torch.nn.BCEWithLogitsLoss()(out, batch[1].to(device))
            print(f'[ {__name__} ]', 'loss:', loss)
            loss.backward()
            torch.optim.Adam(params=net.parameters(), lr=1e-4).step()

        print(f'[ {__name__} ]', 'forward and backward pass on %s succesful in %f secs.'%(device, time.time()-begin_time))
