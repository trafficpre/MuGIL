from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np
import pandas as pd

def get_data_path(data_name):
    if data_name == 'pems08':
        data_path_set = '.\data\pems08.npz'

    elif data_name == 'pems03':
        data_path_set = '.\data\pems03.npz'

    elif data_name == 'pems04':
        data_path_set = '.\data\pems04.npz'

    elif data_name == 'pems07':
        data_path_set = '.\data\pems07.npz'

    else:
        data_path_set = []
        print('data_name error！')
    return data_path_set
'''
PEMS03：[16992, 358, 1]
PEMS04: [16992, 307, 3] # 流量，密度，速度
PEMS07：[16992, 883, 1]
PEMS08：[16992, 170, 3] # 流量，密度，速度
'data'+'adj'
'''

def get_data(file, task):
    if task in ['pems04', 'pems08']:
        B = np.load(file)
        data = B['data']
        data = data[:, :, 2]  # 取第三维度为流量数据
        adj = B['adj']
    else:
        B = np.load(file)
        data = B['data']
        data = data[:, :, 0]  # 取第一维度为流量数据
        adj = B['adj']
    return data, adj


class pems_Dataset(Dataset):
    def __init__(self, task_set, mode, seq_len, pre_len):
        self.task_set = task_set
        self.flow_norm = {}
        self.data_x, self.data_y = {}, {}
        self.adj = {}
        for task in self.task_set:
            file_path = get_data_path(task)
            data, adj = get_data(file_path, task)

            train_len = int(data.shape[0] * 0.6)
            val_len = int((data.shape[0] - train_len)/2)
            test_len = int(data.shape[0] - train_len - val_len)

            if mode == 'train':
                data = data[:train_len, :]
            elif mode == 'val':
                data = data[train_len:(train_len+val_len), :]
            elif mode == 'test':
                data = data[-test_len:, :]

            self.flow_norm[task], flow_data = pems_Dataset.pre_process_data(data=data, norm_dim=0)
            self.data_x[task], self.data_y[task] = pems_Dataset.slice_data(flow_data, seq_len, pre_len)
            self.adj[task] = adj


    def __getitem__(self, index):
        X_set, Y_set = {}, {}
        for task in self.task_set:
            x = self.data_x[task][index, :]
            y = self.data_y[task][index, :]
            x = pems_Dataset.to_tensor(x)
            y = pems_Dataset.to_tensor(y)
            X_set[task] = x
            Y_set[task] = y
        return X_set, Y_set

    def __len__(self):
        key, values = next(iter(self.data_x.items()))
        lengh = values.shape[0]
        return lengh

    @staticmethod
    def slice_data(data, seq_len, pre_len):     # 划分输入和输出
        data_x, data_y = [], []
        for i in range(data.shape[0] - seq_len - pre_len + 1):
            sliced_data = data[i: i + seq_len + pre_len, :]
            data_x.append(sliced_data[:seq_len, :])
            data_y.append(sliced_data[seq_len:seq_len+pre_len, :])
        return np.array(data_x), np.array(data_y)

    @staticmethod
    def pre_process_data(data, norm_dim) -> object:  # 预处理,归一化
        """
        :param data: np.array,原始的交通流量数据
        :param norm_dim: int,归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            norm_base: list, [max_data, min_data], 这个是归一化的基.
            norm_data: np.array, normalized traffic data.
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D], keepdims=True就保持了纬度一致
        min_data = np.min(data, norm_dim, keepdims=True)
        norm_base = (max_data, min_data)    # 返回基是为了恢复数据做准备的

        mid = min_data
        base = max_data - min_data
        norm_data = (data - mid) / (base+0.001)
        return norm_base, norm_data

    @staticmethod
    def recover_data(max_data, min_data, data):  # 恢复数据时使用的，为可视化比较做准备的
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        # recovered_data = data * base + mid
        recovered_data = data.squeeze() * base + mid

        return recovered_data   # np.expand_dims(recovered_data, 2)    # 这个就是原始的数据

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)

def pems_dataloader(tasks, batchsize, seq_len, pre_len):
    data_loader = {}
    for mode in ['train', 'val', 'test']:
        shuffle = True if mode == 'train' else False
        drop_last = True if mode == 'train' else False
        norm_dataset = pems_Dataset(tasks, mode, seq_len, pre_len)
        #             print(d, mode, len(txt_dataset))
        data_loader[mode] = DataLoader(norm_dataset,
                                              num_workers=0,  # num_workers=2,
                                              pin_memory=True,
                                              batch_size=batchsize,
                                              shuffle=shuffle,
                                              drop_last=drop_last)

    return data_loader


if __name__ == "__main__":
    task_set = ['pems03', 'pems04', 'pems07', 'pems08']
    data = pems_dataloader(tasks=task_set, batchsize=64, seq_len=6, pre_len=1)
    train_set = data['train']

    for x_set, y_set in train_set:
        for task in task_set:
            print(task)
            print('X.shape: ', x_set[task].shape, '\t', 'Y.shape: ', y_set[task].shape)
        break
