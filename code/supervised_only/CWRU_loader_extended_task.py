import scipy.io
import torch
from torch.utils.data import Dataset
import random


class CWRU(Dataset):
    """Loads CWRU ball bearing dataset and splits it into training and testing data
    :param sample_length: The length of the time series the dataset should split the files into
    :param rpms: List of Strings of which rpm should be included in the dataset. Available rpms are '1797', '1772',
    '1750' and '1730'
    :param normalise: whether to normalise the dataset to have values between -1 and one
    :param partial_da: Whether or not the dataset should only include healthy samples for training of on a partial DA
    task
    :param train: whether to return the training or evaluation split"""
    def __init__(self, sample_length, normalise=True, partial_da=False, rpms=None, train=None):
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
        # self.variance = torch.Tensor([0.0378, 0.0264, 0.0033]).unsqueeze(1)
        self.variance = torch.Tensor([0.0881, 0.026, 0.0032]).unsqueeze(1)
        self.normalise = normalise
        if not partial_da:
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
        else:
            self.data = {'healthy': []}
            self.lengths = {'healthy': 0}
            self.gts = {'healthy': 0}

        fault_sizes = ['07', '14', '21']
        fault_locations = ['B0', 'IR0', 'OR0']

        # Load healthy data
        if '1797' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_0.mat")
            h = torch.Tensor([h["X097_DE_time"][:, 0], h["X097_FE_time"][:, 0], [0] * len(h["X097_DE_time"][:, 0])])
            h = h[:, :int(h.shape[1] / sample_length) * sample_length]
            h = h.reshape(3, int(h.shape[1] / sample_length), sample_length)

            if train is not None:
                if train:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[:int(0.9 * h.shape[1])]]
                else:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[int(0.9 * h.shape[1]):]]

            self.data['healthy'].append(h)
            self.lengths['healthy'] += self.data['healthy'][-1].shape[1]

        if '1772' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_2.mat")
            h = torch.Tensor([h["X098_DE_time"][:, 0], h["X098_FE_time"][:, 0], [0] * len(h["X098_DE_time"][:, 0])])
            h = h[:, :int(h.shape[1] / sample_length) * sample_length]
            h = h.reshape(3, int(h.shape[1] / sample_length), sample_length)

            if train is not None:
                if train:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[:int(0.9 * h.shape[1])]]
                else:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[int(0.9 * h.shape[1]):]]

            self.data['healthy'].append(h)
            self.lengths['healthy'] += self.data['healthy'][-1].shape[1]

        if '1750' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_2.mat")
            h = torch.Tensor([h["X099_DE_time"][:, 0], h["X099_FE_time"][:, 0], [0] * len(h["X099_DE_time"][:, 0])])
            h = h[:, :int(h.shape[1] / sample_length) * sample_length]
            h = h.reshape(3, int(h.shape[1] / sample_length), sample_length)

            if train is not None:
                if train:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[:int(0.9 * h.shape[1])]]
                else:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[int(0.9 * h.shape[1]):]]

            self.data['healthy'].append(h)
            self.lengths['healthy'] += self.data['healthy'][-1].shape[1]

        if '1730' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_3.mat")
            h = torch.Tensor([h["X100_DE_time"][:, 0], h["X100_FE_time"][:, 0], [0] * len(h["X100_DE_time"][:, 0])])
            h = h[:, :int(h.shape[1] / sample_length) * sample_length]
            h = h.reshape(3, int(h.shape[1] / sample_length), sample_length)

            if train is not None:
                if train:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[:int(0.9 * h.shape[1])]]
                else:
                    torch.manual_seed(42)
                    h = h[:, torch.randperm(h.shape[1])[int(0.9 * h.shape[1]):]]

            self.data['healthy'].append(h)
            self.lengths['healthy'] += self.data['healthy'][-1].shape[1]

        if not partial_da:
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

                        t_list = torch.Tensor(t_list)
                        t_list = t_list[:, :int(t_list.shape[1] / sample_length) * sample_length]
                        t_list = t_list.reshape(3, int(t_list.shape[1] / sample_length), sample_length)
                        if train is not None:
                            if train:
                                torch.manual_seed(42)
                                t_list = t_list[:, torch.randperm(t_list.shape[1])[:int(0.9 * t_list.shape[1])]]
                            else:
                                torch.manual_seed(42)
                                t_list = t_list[:, torch.randperm(t_list.shape[1])[int(0.9 * t_list.shape[1]):]]
                        self.data[fault_location + '_' + fault_size].append(torch.Tensor(t_list))
                        self.lengths[fault_location + '_' + fault_size] += t_list.shape[1]

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
                    if item >= arr.shape[1]:
                        item -= arr.shape[1]
                    else:
                        if self.normalise:
                            return {'data': (arr[:, item] - self.mean) / self.variance,
                                    'gt': self.gts[key]}
                        else:
                            return {'data': arr[:, item], 'gt': self.gts[key]}

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


class StratifiedDataLoader:
    def __init__(self, dataset, batch_size):
        assert(batch_size % 10 == 0)
        self.dataset = dataset
        self.batch_size = batch_size
        self.idxs = self.shuffle_idxs()

    def shuffle_idxs(self):
        idxs = []
        idx0 = 0
        for key in self.dataset.lengths:
            idxs.append(list(range(idx0, idx0 + self.dataset.lengths[key])))
            random.shuffle(idxs[-1])
            idx0 += self.dataset.lengths[key]

        # oversample, assumes first class has most samples
        max_len = len(idxs[0])
        for i in range(len(idxs)):
            idxs[i] = idxs[i] * (int(max_len / len(idxs[i])) + 1)
            idxs[i] = idxs[i][:max_len]
        return idxs

    def __iter__(self):
        output_list = []

        for _ in range(int(len(self.idxs[0]) / self.batch_size * 10)):
            output = {'data': torch.Tensor([]), 'gt': torch.Tensor([])}
            for i in range(len(self.idxs)):
                for _ in range(int(self.batch_size / 10)):
                    sample = self.dataset[self.idxs[i].pop()]
                    output['data'] = torch.cat((output['data'], sample['data'].unsqueeze(0)))
                    output['gt'] = torch.cat((output['gt'], torch.Tensor([sample['gt']])), 0)
            output['gt'] = output['gt'].type(torch.LongTensor)
            output_list.append(output)
        self.idxs = self.shuffle_idxs()
        return iter(output_list)
