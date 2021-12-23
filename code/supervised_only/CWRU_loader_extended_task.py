import scipy.io
import torch
from torch.utils.data import Dataset
import random


class CWRU(Dataset):
    """Loads CWRU ball bearing dataset and splits it into training and testing data
    :param sample_length: The length of the time series the dataset should split the files into
    :param rpm: Cutoff of the domain. Source dataset will only include datapoints whose rpm is greater than rpm. If none
    no domain shift is applied.
    :param normalise: whether to normalise the dataset to have values between -1 and one
    :param train: whether to return the training or evaluation split"""
    def __init__(self, sample_length, normalise=True, rpms=None):
        if rpms is None:
            rpms = {'1797': '_0', '1772': '_1', '1750': '_2', '1730': '_3'}
        else:
            rpm_ls = rpms
            rpms = dict()
            if '1797' in rpm_ls:
                rpms['1797'] = '_0'
            if '1772' in rpm_ls:
                rpms['1772'] = '_1'
            if '1750' in rpm_ls:
                rpms['1750'] = '_2'
            if '1730' in rpm_ls:
                rpms['1730'] = '_3'

        self.sample_length = sample_length
        self.mean = torch.Tensor([0.0165, 0.0326, 0.0048]).unsqueeze(1)
        #self.variance = torch.Tensor([0.0378, 0.0264, 0.0033]).unsqueeze(1)
        self.variance = torch.Tensor([0.0881, 0.026, 0.0032]).unsqueeze(1)
        self.normalise = normalise
        self.data = {'healthy': [],
                     'B0_07': [],
                     'B0_14': [],
                     'B0_21': [],
                     'IR0_07': [],
                     'IR0_14': [],
                     'IR0_21': [],
                     'OR0_07': [],
                     'OR0_14': [],
                     'OR0_21': []}

        self.lengths = {'healthy': 0,
                        'B0_07': 0,
                        'B0_14': 0,
                        'B0_21': 0,
                        'IR0_07': 0,
                        'IR0_14': 0,
                        'IR0_21': 0,
                        'OR0_07': 0,
                        'OR0_14': 0,
                        'OR0_21': 0}

        self.gts = {'healthy': 0,
                    'B0_07': 1,
                    'B0_14': 2,
                    'B0_21': 3,
                    'IR0_07': 4,
                    'IR0_14': 5,
                    'IR0_21': 6,
                    'OR0_07': 7,
                    'OR0_14': 8,
                    'OR0_21': 9}

        fault_sizes = ['07', '14', '21']
        fault_locations = ['B0', 'IR0', 'OR0']

        # Load healthy data
        if '1797' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_0.mat")
            self.data['healthy'].append(torch.Tensor([h["X097_DE_time"][:, 0], h["X097_FE_time"][:, 0],
                                                      [0] * len(h["X097_DE_time"][:, 0])]))
            self.lengths['healthy'] += int(self.data['healthy'][-1].shape[1] / self.sample_length)

        if '1772' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_2.mat")
            self.data['healthy'].append(torch.Tensor([h["X098_DE_time"][:, 0], h["X098_FE_time"][:, 0],
                                                      [0] * len(h["X098_DE_time"][:, 0])]))
            self.lengths['healthy'] += int(self.data['healthy'][-1].shape[1] / self.sample_length)

        if '1750' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_2.mat")
            self.data['healthy'].append(torch.Tensor([h["X099_DE_time"][:, 0], h["X099_FE_time"][:, 0],
                                                      [0] * len(h["X099_DE_time"][:, 0])]))
            self.lengths['healthy'] += int(self.data['healthy'][-1].shape[1] / self.sample_length)

        if '1730' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_3.mat")
            self.data['healthy'].append(torch.Tensor([h["X100_DE_time"][:, 0], h["X100_FE_time"][:, 0],
                                                      [0] * len(h["X100_DE_time"][:, 0])]))
            self.lengths['healthy'] += int(self.data['healthy'][-1].shape[1] / self.sample_length)

        for fault_location in fault_locations:
            for rpm in rpms:
                for fault_size in fault_sizes:
                    h = scipy.io.loadmat('../../data/CWRU/' + fault_location + fault_size + rpms[rpm])
                    t_list = []
                    for key in h.keys():
                        if 'DE_time' in key:
                            t_list.append(h[key][:, 0])
                        if 'FE_time' in key:
                            t_list.append(h[key][:, 0])
                        if 'BA_time' in key:
                            t_list.append(h[key][:, 0])
                    self.data[fault_location + '_' + fault_size].append(torch.Tensor(t_list))
                    self.lengths[fault_location + '_' + fault_size] += int(torch.Tensor(t_list).shape[1] /
                                                                           self.sample_length)

    def __len__(self):

        length = 0
        for class_ in self.lengths:
            length += self.lengths[class_]
        return length

    def __getitem__(self, item):
        for key in self.data:
            if item >= self.lengths[key]:
                item -= self.lengths[key]
            else:
                for arr in self.data[key]:
                    if item >= int(arr.shape[1] / self.sample_length):
                        item -= int(arr.shape[1] / self.sample_length)
                    else:
                        if self.normalise:
                            return {'data': (arr[:, item * self.sample_length:(item+1)*self.sample_length] - self.mean) / self.variance,
                                    'gt': self.gts[key]}
                        else:
                            return {'data': arr[:, item * self.sample_length:(item + 1) * self.sample_length],
                                    'gt': self.gts[key]}

        return None

    def find_sampling_weights(self):
        sample_weights = []
        for class_ in self.lengths:
            sample_weights += [self.lengths['healthy'] / self.lengths[class_]] * self.lengths[class_]
        return sample_weights


def find_sampling_weights(dataset, n_classes):
    sample_weights = []
    lengths = [0] * n_classes
    for sample in dataset:
        lengths[sample['gt']] += 1

    for class_ in range(n_classes):
        if class_ != 0:
            sample_weights += [1 / lengths[class_]] * lengths[class_]
        else:
            sample_weights += [0] * lengths[class_]

    return sample_weights
