import torch 
import torch.nn as nn 
import numpy as np 
import imageio
import os 

class LITS_dataset(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_label):
        self.path_raw = path_raw
        self.path_label = path_label
        self.filelist = os.listdir(self.path_label)

    def __len__(self):
        return len(self.filelist)

    def read_data(self, index):
        raw = imageio.imread(os.path.join(self.path_raw, self.filelist[index]))
        raw = torch.from_numpy(raw).float() / 255.
        label = imageio.imread(os.path.join(self.path_label, self.filelist[index]))
        label = torch.from_numpy(label).long()
        return raw, label
    
    def __getitem__(self, index):
        raw, label = self.read_data(index)
        raw = raw.unsqueeze(0)
        label[label>0] = 1
        # print(raw.shape, label.shape)
        return raw, label

def make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label, batch=4, n_workers=1):
    dataset_tr = LITS_dataset(tr_path_raw, tr_path_label)
    dataset_ts = LITS_dataset(ts_path_raw, ts_path_label)
    loader_tr = torch.utils.data.DataLoader(dataset=dataset_tr, batch_size=batch, shuffle=True, num_workers=n_workers)
    loader_ts = torch.utils.data.DataLoader(dataset=dataset_ts, batch_size=batch, shuffle=True, num_workers=n_workers)
    dataloaders = {'train': loader_tr, 'val': loader_ts}
    return dataloaders