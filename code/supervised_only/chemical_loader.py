import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class ChemicalLoader(Dataset):
    def __init__(self, bach_num, normalise=True, train=None, balance=True, seed=None):

        file = pd.read_csv("../../data/chemical/batch" + str(bach_num) + ".dat", sep=None, header=None).values
        for i in range(file.shape[0]):
            for j in range(1, file.shape[1]):
                file[i, j] = float(file[i, j].split(":")[-1])
        file = file.astype("float")

        self.data = file[:, 1:]
        self.gt = file[:, 0].astype('int') - 1

        if normalise:
            self.data = self.data - self.data.sum(0) / self.data.shape[0]
            self.data = self.data / np.sqrt((self.data**2).sum(0) / self.data.shape[0])

        if train is not None:
            if seed is not None:
                np.random.seed(seed)
            choice_arr = np.random.choice(a=[False, True], size=self.data.shape[0], p=[.5, 1 - .5])
            if train:
                self.data = self.data[choice_arr]
                self.gt = self.gt[choice_arr]
            else:
                self.data = self.data[np.logical_not(choice_arr)]
                self.gt = self.gt[np.logical_not(choice_arr)]

        if balance:
            # Find class with most samples
            largest_class = 0
            for i in range(max(self.gt)+1):
                if (self.gt == i).sum() > largest_class:
                    largest_class = (self.gt == i).sum()

            # oversample all classes smaller than largest
            data2 = np.zeros((0, self.data.shape[1]))
            gt_2 = np.zeros(0)
            for cls in range(max(self.gt)+1):
                data_cls = self.data[self.gt == cls]

                while data_cls.shape[0] * 2 < largest_class:
                    data_cls = np.vstack((data_cls, data_cls))

                p = (largest_class - data_cls.shape[0]) / data_cls.shape[0]
                if seed is not None:
                    np.random.seed(seed)
                choice_arr = np.random.choice(a=[False, True], size=data_cls.shape[0], p=[1 - p, p])
                data_cls = np.vstack((data_cls, data_cls[choice_arr]))
                data2 = np.vstack((data2, data_cls))
                gt_2 = np.hstack((gt_2, np.ones(data_cls.shape[0]) * cls))
            self.data = data2
            self.gt = gt_2

    def __getitem__(self, item):
        return {'data': torch.Tensor(self.data[item]), 'gt': torch.Tensor(self.gt[item])}

    def __len__(self):
        return self.data.shape[0]

    def get_weights(self):
        weights = [1] * self.data
        for i in range(1, max(self.gt)):
            weights = weights + [(self.gt == i).sum() / (self.gt == 0).sum()] * (self.gt == i).sum()
        return weights


def main():
    dataset = ChemicalLoader(1)
    for sample in dataset:
        print(sample)


if __name__ == "__main__":
    main()
