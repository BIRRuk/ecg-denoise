import sys
import time
import datetime
import argparse

todos = [
]
from utils.utils import pjson, tcolors, colorize
pjson(todos, title=colorize('#TODOs', tcolors.HEADER))

circle pentagon
# -----------------------------------------------------------------------------
# computer vision
# -----------------------------------------------------------------------------
# momentum = 0.5
img_sizes = [112, 160, 224, 384, 512]
img_sze = img_sizes[2]
# img_sze = img_sizes[3]
crop_ratio_trn = 0.2

# -----------------------------------------------------------------------------
# presets
# -----------------------------------------------------------------------------
machine = 'local'
import os
def _cfg(**kwargs):
    return {
        'ds_root' : './icentia11k/',
        'cache_dir' : 'cache',
        'tqdm_ncols' : 100,
        'bs' : 32,
        'save_to_checkpoint':True,
        'test_bs_multiplier':1,
        'trn_ds_epoch_limit':None,
        'val_ds_epoch_limit':None,
        **kwargs
    }

presets = dict(
    local = _cfg(
        cache_dir='/home/biruk/Documents/ml-cache/221206/icentia11k',
        ds_root='~/Downloads/torrents/icentia11k/',
        bs=4
    ),
    colab = _cfg(
        ds_root = '/content/icentia11k/', 
        cache_dir='/content/cache/icentia11k',
        tqdm_ncols=100, bs=32,
    ),
    win = _cfg(
        ds_root = 'C:\\ML\\datasets\\icentia11k', 
        cache_dir='C:\\ML\\cache\\icentia11k'
    ),
)
preset = presets[machine]

# configs = {
#         'csv_trn' : os.path.join(preset['ds_root'], preset['csv_trn']),
#         'csv_val' : os.path.join(preset['ds_root'], preset['csv_val']),
#         'imgs_root' : os.path.join(preset['ds_root'], preset['imgs_root']),
# }
# csv_trn, csv_val, imgs_root = configs.values()

def parse_args():
    parser = argparse.ArgumentParser(
        description='script to start training or evaluation of a model')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='Only run evaluaiton loop on --val_csv or config.val_csv')
    parser.add_argument('--val_start', action='store_true', default=False, 
                        help='Start training with validation loop')
    parser.add_argument('--verbose', '-q', action='store_false', default=True, 
                        help='verbose level')
   
    parser.add_argument('--ckpt_path', '-c', type=str, default=None, 
        help='previous model state_dict path')
    parser.add_argument('--load_optim', '-O', action='store_true', default=False)
    parser.add_argument('--load_trend', '-T', action='store_true', default=False)
    parser.add_argument('--load_scheduler', '-S', action='store_true', default=False)
    parser.add_argument('--modify_net', '-M', action='store_true', default=False)
    parser.add_argument('--use_ckpt_loss', '-L', action='store_true', default=False)
    
    # parser.add_argument('--csv_trn', type=str, default=csv_trn, help='training csv file')
    # parser.add_argument('--csv_val', type=str, default=csv_val, help='validation csv file')
    # parser.add_argument('--imgs_root', type=str, default=imgs_root)
    parser.add_argument('--ds_root', type=str, default=preset['ds_root'])
    parser.add_argument('--trn_ds_epoch_limit', type=int, default=None)

    parser.add_argument('--nepoch', '--ep', type=int, default=100, help='epoch num')
    parser.add_argument('--lr', type=float, default=1e-4, help='learing rate')
    parser.add_argument('--bs', type=int, default=preset['bs'], help='batch size')

    parser.add_argument('--scheduler_factor', type=float, default=.1)
    parser.add_argument('--scheduler_patience', type=float, default=1)
    parser.add_argument('--cache_dir', type=str, default=preset['cache_dir'])
    parser.add_argument('--img', type=str, default=None)

    parser.add_argument('--prefered_device', type=str, default=None)
    
    ipy = True if str(type(sys.stderr)).split('\'')[1].split('.')[0] == 'ipykernel' else False
    if ipy: args = parser.parse_args('') # !use this line when running from notebook
    else: args = parser.parse_args()

    return args


args = parse_args()

# if __name__!='__main__':
#     from datagen import get_input_shape, classes_out
#     input_shape = get_input_shape()
#     classes_in = input_shape[0]
#     preset['classes_out'] = classes_out
#     print(f'[ {__name__} ]', "input_shape:", input_shape, 'classes_out:', classes_out)


# cache_fldrs = [ 'checkpoint', 'plots' ]
# ckpt_dir, plot_dir = [os.path.join(preset['cache_dir'], i) for i in cache_fldrs]
# preset['ckpt_dir'] = ckpt_dir
# from utils.utils import check_folder
# for i in [preset['cache_dir'], ckpt_dir, plot_dir ]:
#     check_folder(i)

# if args.ckpt_path is not None and os.path.split(args.ckpt_path)[0] == '':
#     ckpt_path = os.path.join(ckpt_dir, args.ckpt_path)
# else: ckpt_path = args.ckpt_path
# trend_plot_path = os.path.join(ckpt_dir, 'trend_plot.png')

# now = str(datetime.datetime.now())[:-7]
# preset['training_id'] = time.time().__trunc__()

# pjson(preset, title=colorize('configs('+preset+')', tcolors.HEADER))
