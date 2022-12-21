import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import collections
from tqdm import tqdm
import gzip
import pickle

class Icentia(Dataset):
    SAMPLE_RATE = 250
    BUFFER = 2500
    POOL_STRIDE = 2
    SAMPLE_DURATION = 10 # SECONDS
    DURATION = 60*60 # 60 MINS TO BE SAFE
    PTS_PER_FILE = 50
    SAMPLES_PER_PT = 50
    
    '''requires csv in "file_pat, class_num" '''
    def __init__(self, folder, transforms=None, transforms_y=None, limit_len:int=None, shuffle=False, **kwargs):
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
                    self.db.append(num)
                    self.db_names.append([num+suf_file, num+suf_lbls])
                else:
                    print(f'[ {__name__} ] entery {num} does not have a matching file or label')
                    
        if shuffle: 
            np.random.shuffle(self.db_names)
            # np.random.shuffle(self.db)
        # TODO: shuffle dataset
        
        self.samples_per_file = self.DURATION//self.SAMPLE_DURATION
        self.classes_out = 2 #(self.SAMPLE_RATE*self.DURATION)//self.POOL_STRIDE
        
        self.file_idx = 0
        self.pt = 0
        self.count = 0
        self.signals = None
        self.labels_ = None
        
        self.load_file()
        # print(self.signal)
            
        self.total = self.__len__()
        # print(self.db_names)
        self.transforms = transforms
        self.transforms_y = transforms_y
        self.slice_width = self.SAMPLE_DURATION * self.SAMPLE_RATE

        print('database initialized with %i samples and %i dx'%(self.__len__(), self.classes_out))
        

    def __len__(self):
        return len(self.db_names)*self.samples_per_file#*self.PTS_PER_FILE

    def load_pkl_gz(self, filename):
        file = os.path.join(self.folder, filename)
        print(__name__, 'loading file:', file)
        with gzip.open(file,'rb') as f:
            db = pickle.load(f)
        return db

    def load_file(self):
        filenames = self.db_names[self.file_idx]
        self.signals = self.load_pkl_gz(filenames[0])
        self.labels_ = self.load_pkl_gz(filenames[1])
            
    def get_sample(self):
        # if self.count % self.SAMPLES_PER_PT == 0 and self.count != 0:
        if self.count == self.SAMPLES_PER_PT:
            self.pt += 1
            self.count = 0
    
        if self.pt == self.PTS_PER_FILE:
            self.pt = 0
            self.file_idx += 1
            self.load_file()
            
        b_slice = self.count*self.SAMPLE_RATE+self.BUFFER
        x = self.signals[self.pt][b_slice:b_slice+self.slice_width]
        
        r_waves = self.labels_[self.pt]['btype'][1]
        r_waves = r_waves[np.logical_and(r_waves>b_slice, r_waves<b_slice+self.slice_width)] - b_slice
        # print(r_waves)
        # print(r_waves.shape)
        # y = np.zeros(x.shape, dtype=np.float16)
        y = np.zeros(x.shape, dtype=np.float32)
        for i in r_waves:
            y[i] = 1

        self.count += 1
        return x, y

    def __getitem__(self, idx):
        # signal = get_signal()
        # self.load_file()
        # x = self.get_signal()
        x, y = self.get_sample()

        if self.transforms_y is not None:
            y = self.transforms_y(y)

        y = np.stack((x,y))

        if self.transforms is not None:
            x = self.transforms(img)

        return x, y


if __name__ == '__main__':
    ds_root = '~/Downloads/torrents/icentia11k/'
    ds = Icentia(ds_root, shuffle=True,)
    print(ds[0])