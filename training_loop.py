import os
import torch
from tqdm import tqdm

from utils import checkpoint

def train(epoch, net, trn_dl, criterion, optimizer, trend=None, metric=None, device=None, **configs):
    loop = tqdm(trn_dl, ascii=True, ncols=configs['tqdm_ncols'], desc=(f'{epoch:>2} trn'))
    loss_ep = 0.0
    stats = metric(configs['classes_out'], device=device)
    net.train()

    for idx, (imgs, labels) in enumerate(loop):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        x = net(imgs)
        # labels = labels.type_as(x)
        loss = criterion(x, labels)
        loss.backward()
        optimizer.step()

        corrects, bs = stats(x, labels)
        loss_ep += (loss.item()*bs)

        loop.set_postfix(stat='%03d %.1f %.2f %.5f'%(
            100*corrects, 100*stats.accuracy(), loss, loss_ep/stats.total_count))
        
    return loss_ep, stats


def test(epoch, net, val_dl, criterion, optimizer=None, trend=None, metric=None, detail_print=False,
    scheduler=None, save_to_checkpoint=True, device=None, **configs):
    loop = tqdm(val_dl, ascii=True, ncols=configs['tqdm_ncols'], desc=(f'{epoch:>2} val'))
    loss_ep = 0.0
    stats = metric(configs['classes_out'], device=device)
    net.eval()

    for idx, (imgs, labels) in enumerate(loop):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            x = net(imgs)
            loss = criterion(x, labels)

        corrects, bs = stats(x, labels)
        loss_ep += (loss.item()*bs)

        loop.set_postfix(stat='%03d %.1f %.2f %.5f'%(
            100*corrects, 100*stats.accuracy(), loss, loss_ep/stats.total_count))

    return loss_ep, stats


def fit(
    net, eps, bs, criterion, optimizer, trn_dl, configs, args,
    val_dl=None, scheduler=None, trend=None, save_to_checkpoint=True,
    ckpt_path=None, val_start=False, val_only=False,
    metric='softmax_correct', device=None, **kwds,
    ):
    # import json; print(json.dumps(configs, indent=4))
    if device == None: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[ %s ]'%__name__, 'training on device:', device)

    global start_epoch, best_loss
    start_epoch = 0
    best_loss = float('inf')
    
    from utils.metrics import metrics
    assert metric in metrics.keys()
    metric_ = [metric]
    if 'val_metric' in kwds and kwds['val_metric'] is not None:
        metric_.append(kwds['val_metric'])
    else: metric_.append(metric)
    metric = [metrics[i] for i in metric_]

    if ckpt_path is not None:
        load_kwargs = {'filename':ckpt_path, 'net':net}
        if args.load_trend: load_kwargs['trend']=trend
        if args.load_optim: load_kwargs['optimizer']=optimizer
        if args.load_scheduler: load_kwargs['scheduler']=scheduler

        ckpt_loss, start_epoch, training_id = checkpoint.loadckpt2(['loss', 'epoch', 'training_id'], **load_kwargs)
        start_epoch += 1
        configs['training_id'] = training_id
        if args.use_ckpt_loss: best_loss = ckpt_loss

        if args.modify_net:
            from model import modify_net_
            net = modify_net_(net)
            for param in optimizer.param_groups:
                # param['lr'] = args.lr#*args.scheduler_factor
                pass

        if args.load_scheduler == False or args.load_optim == False:
            print('[ %s ]'%__name__, 'scheduler.__init__()')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=args.scheduler_patience, factor=args.scheduler_factor, verbose=True)

        print('[ %s ]'%__name__, 'optimizer stats:', optimizer)
        print('[ %s ]'%__name__, 'best_loss:', best_loss)

    print('\nstarting training for %s eopches, resuming from epoch %s' % (eps, start_epoch))
    print(r'stats: "%03d %.1f %.2f %.5f"%(instance@, cumulative@, instance$, cumulative$)')
    print()

    if val_start or val_only:
        if val_only: configs['save_to_checkpoint'] = False
        loss_ep_val, stats_val = test(-1, net, val_dl, criterion, metric=metric[1], detail_print=True, device=device, **configs)
        val_loss = loss_ep_val/val_dl.dataset.__len__()
        if trend is not None: trend.update(-1, loss_val = val_loss, 
            lr=torch.tensor([param['lr'] for param in optimizer.param_groups]), **stats_val.accuracy(mode='val'))
        if val_only: exit()

    for ep in range(start_epoch, start_epoch+eps):

        loss_ep_trn, stats_trn = train(ep, net, trn_dl, criterion, 
            optimizer, **configs, trend=trend, metric=metric[0], device=device)
        if trend is not None: 
            trend.update(ep, loss_trn = loss_ep_trn/trn_dl.dataset.__len__(), acc_trn=stats_trn.accuracy())


        if val_dl is not None:
            loss_ep_val, stats_val = test(ep, net, val_dl, criterion, optimizer=optimizer, trend=trend, metric=metric[1],
                scheduler=scheduler, detail_print=True, device=device, **configs, 
            )
            val_loss = loss_ep_val/val_dl.dataset.__len__()
            delta = (val_loss-best_loss)/best_loss
            if trend is not None: trend.update(ep, loss_val = val_loss, 
                lr=torch.tensor([param['lr'] for param in optimizer.param_groups]), **stats_val.accuracy(mode='val'))

            if val_loss < best_loss:
                print('\x1b[36mloss delta: %f%%, new best!\x1b[39m'%(delta*100))
                best_loss = val_loss
                if save_to_checkpoint: 
                    checkpoint.saveckpt2(id_=0, optimizer=optimizer, loss=best_loss, epoch=ep, net=net, 
                        trend=trend, scheduler=scheduler, training_id=configs['training_id'], path=configs['ckpt_dir'])
            else: 
                print('\x1b[38mloss delta: +%f%%\x1b[39m'%(delta*100))
                if save_to_checkpoint: checkpoint.saveckpt2(
                    id_=-1, optimizer=optimizer, loss=best_loss, epoch=ep, net=net, trend=trend, 
                    scheduler=scheduler, training_id=configs['training_id'], path=configs['ckpt_dir'])
    
            if trend is not None and ep!=0:
                trend.svplot(plot_groups=[('loss_trn', 'loss_val'), ('acc_trn', 'acc_val')], 
                    ignore=['dice', 'dices', 'accuracies'], twinx_groups = [['lr'], [], []])

            if scheduler is not None: scheduler.step(loss_ep_val)
