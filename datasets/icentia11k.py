import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import collections
from tqdm import tqdm
import gzip
import pickle

def r_strip(r_waves, b_slice, e_slice):
    x = r_waves[np.logical_and(r_waves>b_slice, r_waves<e_slice)] - b_slice
    beats = np.zeros(e_slice-b_slice)
    for i in x:
        beats[i] = 1
    return beats

def np_pool(x, ker=2, operator=np.mean):
    bs, xs = x.shape
    # assert xs%ker == 0, 'dim -1 has to be a multiple of kernel_size'
    xs_ = xs//ker
    y = operator(x[:,:xs_*ker].reshape(bs, xs_, ker), axis=-1)
    return y

def r_tfms(x, ker = 8, repeat = 4):
    x = np_pool(x, ker = ker, operator=np.max)
    x = x.repeat(ker, axis=-1)
    return x

def avg_pool(x, ker = 2):
    x = np_pool(x, ker = ker, operator=np.mean)
    return x

class Icentia(Dataset):
    SAMPLE_RATE = 250
    BUFFER = 2500
    POOL_STRIDE = 2
    # SAMPLE_DURATION = 10 # SECONDS
    SAMPLE_DURATION = 2496 # 
    DURATION = 60*60 # 60 MINS TO BE SAFE
    SAMPLES_PER_STRIP = 50
    classes_out = 5
    TOTAL_SEGMENTS = 541_794
    
    '''requires csv in "file_pat, class_num" '''
    def __init__(self, folder, transforms=None, transforms_y=r_tfms, limit_len:int=None, shuffle=False, **kwargs):
        super().__init__()
        folder = os.path.realpath(os.path.expanduser(folder))
        self.folder = folder
        file_list = [x for _, _, x in os.walk(folder)][0]
        print(f'[ {__name__} ] len(file_list) in {folder}:', len(file_list))

        suf_file = '_batched.pkl.gz'
        suf_lbls = '_batched_lbls.pkl.gz'
        self.db = []
        self.db_names = []
        for i in tqdm(file_list, desc=('reading file list')):
            split_ = i.split('_')
            num = split_[0]
            # print(i)
            if len(split_)==2:
                if os.path.exists(os.path.join(folder,num+suf_file)) and os.path.exists(os.path.join(folder,num+suf_lbls)):
                    # print(num+suf_file, num+suf_lbls)
                    # self.db.append(num)
                    self.db_names.append([num+suf_file, num+suf_lbls])
                else:
                    print(f'[ {__name__} ] entery {num} does not have a matching file or label')
                    
        if shuffle: 
            # TODO: shuffle dataset properly
            np.random.shuffle(self.db_names)
            # np.random.shuffle(self.db)
        
        self.total = self.__len__()
        # print(self.db_names)
        self.transforms = transforms
        self.transforms_y = transforms_y
        self.slice_width = self.SAMPLE_DURATION * self.SAMPLE_RATE
        self.slice_width = self.SAMPLE_DURATION

        self.pt_idx = 0
        self.strip_idx = 0
        self.count = 0
        self.strips_per_file = 50
        self.signals = None
        self.labels_ = None
        self.load_file()

        print('database initialized with %i samples and %i dx'%(self.__len__(), self.classes_out))
        

    def __len__(self):
        # return len(self.db_names)*self.SAMPLES_PER_STRIP*self.strips_per_file
        return self.TOTAL_SEGMENTS*self.SAMPLES_PER_STRIP

    def load_pkl_gz(self, filename):
        file = os.path.join(self.folder, filename)
        print(__name__, 'loading file:', file)
        with gzip.open(file,'rb') as f:
            db = pickle.load(f)
        return db

    def load_file(self):
        filenames = self.db_names[self.pt_idx]
        self.signals = self.load_pkl_gz(filenames[0])
        self.labels_ = self.load_pkl_gz(filenames[1])
        self.strips_per_file = self.signals.shape[0]
            
    def get_sample(self):
        if self.count == self.SAMPLES_PER_STRIP:
            self.strip_idx += 1
            self.count = 0
    
        if self.strip_idx == self.strips_per_file:
            self.strip_idx = 0
            self.pt_idx += 1
            self.load_file()
            
        b_slice = self.count*self.SAMPLE_RATE+self.BUFFER
        e_slice = b_slice + self.slice_width

        b_type = self.labels_[self.strip_idx]['btype']
        x = self.signals[self.strip_idx][b_slice:e_slice]

        r_normal = r_strip(b_type[1], b_slice, e_slice) 
        r_pac = r_strip(b_type[2], b_slice, e_slice) 
        r_pvc = r_strip(b_type[4], b_slice, e_slice) 
        r_undef = r_strip(b_type[0], b_slice, e_slice) 

        y = np.stack((
            r_normal,
            r_pac,
            r_pvc,
            r_undef,
        ))

        self.count += 1
        return x.reshape(1,-1), y

    def __getitem__(self, idx):
        x, y = self.get_sample()
        # print(x.shape, y.shape)

        if self.transforms_y is not None:
            y = self.transforms_y(y)
        y = np.concatenate((y, x), axis=0)

        if self.transforms is not None:
            x = self.transforms(img)

        return x.reshape(1,-1), y


if __name__ == '__main__':
    ds_root = '~/Downloads/torrents/icentia11k/'
    ds = Icentia(ds_root, shuffle=True, transforms_y=r_tfms)
    print(ds[0])

    import matplotlib.pyplot as plt
    x, y = ds[1]
    x = np.concatenate((x,y), axis=0)

    fig, ax = plt.subplots(6,1)
    for idx, i in enumerate(x):
        ax[idx].plot(i)

    fig.set_size_inches(10,6)
    fig.savefig('cache/plots/ds_sample.png')