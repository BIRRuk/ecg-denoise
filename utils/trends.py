import os
from tkinter import N
import numpy as np
# from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import math

import torch
class PB():
    """print a vetical progress grapy ascii"""
    def __init__(self, col=20, min=0.5):
        self.col = col
        self.min = min

    def plot(self, value):
        pr = value-self.min if value>self.min and value<1 else 0.0
        pr = round(self.col*pr/(1-self.min))
        print("["," "*(pr), "*", " "*(self.col-1-pr), "]", value)

    def ret(self, value):
        pr = value-self.min if value>self.min and value<1 else 0.0
        pr = round(self.col*pr/(1-self.min))
        return str("["+"-"*(pr)+">"+" "*(self.col-1-pr)+"]")

class Trends_():
    """
    '\'Trends 2.0\''
    keeps track of a trend,
    requieres numpy as np
    """
    def __init__(self, prev_db_path=None, img_path='./trend.png'):
        self.db = {}
        if prev_db_path is not None: self.db = self.load_prev(prev_db_path)
        self.img_path = img_path

        self.has_fig = False
        self.twinx_ax = {}

    def load_prev(self, prev_db_path) -> list:
        import pickle
        with open(prev_db_path, 'rb') as f:
            db = pickle.load(f, encoding="latin-1")
        self.db = db

    def save_db(self, db_save_path) -> None:
        import pickle
        with open(db_save_path, 'wb') as f:
            pickle.dump(self.db, f)

    def state_dict(self, *args, **kwargs):
        return self.db

    def load_state_dict(self, state):
        self.db=state

    def update(self, ep, **kwds):
        '''ep = epoch count'''
        for i in kwds:
            if isinstance(kwds[i], torch.Tensor): kwds[i]=kwds[i].tolist()
            print('->', i, ':', kwds[i])
            if i in self.db.keys():
                self.db[i][0].append(ep) # append to ep, append to value
                self.db[i][1].append(kwds[i]) # append to ep, append to value
            else:
                self.db[i] = [[ep], [kwds[i]]] #* list of epoch, data

    def print_db(self):
        for i in self.db.keys():
            print(i+':', self.db[i], '\n')

    def get_xy(self, key):
        y = []
        for k in self.db[key][1]:
            if isinstance(k, torch.Tensor): 
                if k.numel() == 1:
                    y.append(k.item())
                else: y.append(k.flatten().tolist())
            else: y.append(k)
        return self.db[key][0], y

    def plot(self, plot_groups=None, colors=None, scale=None, plot_all=True, ignore=None, twinx_groups=None, rows=1):
        '''
        plot_groups should be list of strings, or nested string list
        '''
        def dict_append(dict, key, value):
            if key in dict.keys():
                dict[key].append(value) # append to ep, append to value
            else:
                dict[key] = [value] #* list of epoch, data
            
        extra_colum: False
        remainder = list(self.db.keys())

        if twinx_groups is not None:
            if ignore is None: ignore = []
            if isinstance(twinx_groups, (list, tuple)):
                for idx, i in enumerate(twinx_groups):
                    if len(i) > 0:
                        if isinstance(i, str): i = [i]
                        for key in i:
                            ignore.append(key)

        if ignore is not None:
            for i in ignore: 
                if i in remainder:
                    remainder.remove(i)

        plot_idx = {}
        for idx, i in enumerate(plot_groups):
            if isinstance(i, (list, tuple)):
                for j in i:
                    dict_append(plot_idx, idx, j)
                    if plot_all: remainder.remove(j) # !remove if plot_all
            elif isinstance(i, str):
                dict_append(plot_idx, idx, i)
                if plot_all: remainder.remove(i)

        if plot_all and len(remainder) > 0:
            extra_colum: True 
            plot_groups.append(remainder)
            for i in remainder:
                dict_append(plot_idx, idx+1, i)

        cols = len(plot_groups) if plot_groups is not None else len(self.db)

        if self.has_fig:
            for axx in self.ax:
                axx.clear() 
        else:        
            self.fig, self.ax = plt.subplots(1, cols)
            self.fig.set_size_inches(4.4*cols, 4)
            self.colors = ['r', 'b', 'g', 'c', 'm', 'y']
            self.has_fig = True

        def plot_(ax, x, y, color='', pattern='+', smooth=True, label=None, **kwargs):
                ax.plot(x, y, pattern+color, label=label)
                if smooth: ax.plot(x, gaussian_filter1d(y, sigma=1), '-.'+color)
                ax.legend(loc='center left') #* opacity select

        for i in plot_idx.keys():
            if cols==1: ax = self.ax
            else: ax = self.ax[i]

            smooth = True
            pattern = '+'
            for idx, curve in enumerate(plot_idx[i]):
                color = self.colors[idx]
                # x, y = self.db[curve][0], self.db[curve][1]
                x, y = self.get_xy(curve)
                if isinstance(y[0], (list, tuple)):
                    smooth = False
                    pattern = ''
                    color = ''
                    for j in range(len(y[0])):
                        if len(y[0]) > 1:
                            label = curve+str(j)
                        else: label = curve
                        plot_(ax, x, [k[j] for k in y], pattern=pattern, color=color, label=label, smooth=smooth)

                else: plot_(ax, x, y, color=self.colors[idx], label=curve)

        if twinx_groups is not None:
            if isinstance(twinx_groups, (list, tuple)):
                for idx, i in enumerate(twinx_groups):
                    if isinstance(i, str):
                        i = [i]
                    if len(i) > 0:
                        if idx in self.twinx_ax.keys():
                            self.twinx_ax[idx].clear()
                        else: 
                            self.twinx_ax[idx] = self.ax[idx].twinx()

                        for key in i:
                            x, y = self.get_xy(key)
                            smooth = False
                            pattern = ''
                            color = ''
                            if isinstance(y[0], (list, tuple)):
                                for j in range(len(y[0])):
                                    if len(y[0]) > 1:
                                        label = key+str(j)
                                    else: label = key
                                    plot_(self.twinx_ax[idx], x, [k[j] for k in y], pattern=pattern, color=color, label=label, smooth=smooth)

                            else: plot_(self.twinx_ax[idx], x, y, pattern=pattern, color=color, label=key, smooth=smooth)
                            self.twinx_ax[idx].set_yscale('log')
                            self.twinx_ax[idx].legend(loc='center right') #* opacity select


        self.ax[0].set_yscale('log')
        # # self.ax[0].set_yticks(np.arange(0,101,20))
        for axx in self.ax:
            axx.grid()

        self.fig.tight_layout()
        return self.fig

    def svplot(self, *args, **kwargs):
        fig = self.plot(*args, **kwargs)
        # import os; os.system('feh \'%s\''%self.img_path) 
        fig.savefig(self.img_path) # (path, dpi=300)
        
    def showplot(self, *args, **kwargs):
        fig = self.plot(*args, **kwargs)
        fig.show() # (path, dpi=300)

# if __name__=='__main__':
    # trend = Trends_(prev_db_path=None, img_path='/home/biruk/Documents/ml-cache/220711/plots')
    # file = torch.load('/home/biruk/Documents/ml-cache/220711/checkpoint/ckpt-trend_-1.pth')#, map_location=device)
    # trend.load_state_dict(file)
    # # print(trend.db)
    # print(trend.get_xy('acc_val'))

    # trend.svplot(
    #             plot_groups=[('loss_trn', 'loss_val'), ('acc_trn', 'acc_val')], 
    #             ignore=['dice', 'dices', 'accuracies'], twinx_groups = [['lr'], [], []]
    #         )

if __name__=='__main__':
    trend = Trends_(prev_db_path=None, img_path='C:\\ML\\cache\\trend.png')
    file = torch.load('C:\\ML\\cache\\checkpoint\\ckpt-trend_-1.pth')#, map_location=device)
    # file = torch.load('C:\\ ML\\cache\\checkpoint\\resnet18\\ckpt-trend_-1.pth')#, map_location=device)
    trend.load_state_dict(file)
    # print(trend.db)
    for i in trend.db.items():
        print(i, '\n')

    # print(trend.db['loss_val'][1])
    trend.svplot(
                plot_groups=[('loss_trn', 'loss_val'), ('acc_trn', 'acc_val'), ('sensetivity', 'specificity'), ('accs', 'senss', 'specs'), ('dice', 'dices')], 
                ignore=None, 
                twinx_groups = [['lr'], [], [], [], []],
                rows = 2
            )
