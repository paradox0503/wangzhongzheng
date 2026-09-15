from torch.utils.data import Dataset
from utils.conf import Configuration

import numpy as np
import torch


class TSData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def getSamples(conf: Configuration):
    dataset_selected = conf.getEntry("dataset_selected")
    data_path = conf.getEntry("data_path")
    data_pos = data_path + dataset_selected + "/data.bin"
    
    train_path = conf.getEntry("train_path")
    val_path = conf.getEntry("val_path")
    test_path = conf.getEntry("test_path")
    train_indices_path = conf.getEntry("train_indices_path")
    val_indices_path = conf.getEntry("val_indices_path")
    test_indices_path = conf.getEntry("test_indices_path")
    len_series = conf.getEntry("len_series")
    data_size = conf.getEntry("data_size")
    train_size = conf.getEntry("train_size")
    val_size = conf.getEntry("val_size")
    test_size = conf.getEntry("test_size")
    
    train_sample_indices = np.random.randint(0, data_size, size=train_size, dtype=np.int64)
    val_sample_indices = np.random.randint(0, data_size, size=val_size, dtype=np.int64)
    test_sample_indices = np.random.randint(0, data_size, size=test_size, dtype=np.int64)
    
    train_sample_indices.tofile(train_indices_path)
    val_sample_indices.tofile(val_indices_path)
    test_sample_indices.tofile(test_indices_path)
    
    loaded = []
    for index in train_sample_indices:
        sequence = np.fromfile(data_pos, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        loaded.append(sequence)
    train_samples = np.asarray(loaded, dtype=np.float32)
    train_samples.tofile(train_path)
    
    loaded = []
    for index in val_sample_indices:
        sequence = np.fromfile(data_pos, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        loaded.append(sequence)
    val_samples = np.asarray(loaded, dtype=np.float32)
    val_samples.tofile(val_path)
    
    loaded = []
    for index in test_sample_indices:
        sequence = np.fromfile(data_pos, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        loaded.append(sequence)
    test_samples = np.asarray(loaded, dtype=np.float32)
    test_samples.tofile(test_path)
    
    train_samples, val_samples, test_samples = torch.from_numpy(train_samples), torch.from_numpy(val_samples), torch.from_numpy(test_samples)
    
    return train_samples, val_samples, test_samples