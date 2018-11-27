# -*- coding:utf-8 -*-
r"""
    learning data loader in pytorch 0.4.1
"""
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utilities.load_data import load_data


class DemoDataset(Dataset):

    def __init__(self, input_file):
        data = load_data(input_file)
        data = np.asarray(data, dtype=float)
        self.X, self.y = data[:, 0], data[:, 1]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


def main(input_file):
    dataset = DemoDataset(input_file)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, sampler=None, drop_last=True)
    for idx, (b_x, b_y) in enumerate(data_loader):
        print(f"{idx}, b_x:{b_x}, b_y:{b_y}")


if __name__ == '__main__':
    main(input_file='../data/samples.csv')
