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
    def __init__(self, sample_length, rpm=None, normalise=True, rpms=None):
        if rpms is None:
            rpms = ['1797', '1772', '1750', '1730']
        self.sample_length = sample_length
        self.mean = torch.Tensor([0.0083, 0.0320, 0.0037]).unsqueeze(1)
        self.variance = torch.Tensor([0.0381, 0.0231, 0.0035]).unsqueeze(1)
        self.normalise = normalise

        # Load healthy data
        self.healthy_data = []
        if '1797' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_0.mat")
            self.healthy_data.append(torch.Tensor([h["X097_DE_time"][:, 0], h["X097_FE_time"][:, 0],
                                                   [0] * len(h["X097_DE_time"][:, 0])]))

        h = scipy.io.loadmat("../../data/CWRU/H_2.mat")
        if '1772' in rpms:
            self.healthy_data.append(torch.Tensor([h["X098_DE_time"][:, 0], h["X098_FE_time"][:, 0],
                                                   [0] * len(h["X098_DE_time"][:, 0])]))
        if '1750' in rpms:
            self.healthy_data.append(torch.Tensor([h["X099_DE_time"][:, 0], h["X099_FE_time"][:, 0],
                                                   [0] * len(h["X099_DE_time"][:, 0])]))

        if '1730' in rpms:
            h = scipy.io.loadmat("../../data/CWRU/H_3.mat")
            self.healthy_data.append(torch.Tensor([h["X100_DE_time"][:, 0], h["X100_FE_time"][:, 0],
                                                   [0] * len(h["X100_DE_time"][:, 0])]))

        # Load data of faulty balls
        ball_fault_paths = {'1797': "../../data/CWRU/B_007_0.mat", '1772': "../../data/CWRU/B_007_1.mat",
                            '1750': "../../data/CWRU/B_007_2.mat", '1730': "../../data/CWRU/B_007_3.mat"}

        self.ball_fault_data = []
        self.ball_fault_rpm = []

        for rpm in rpms:
            h = scipy.io.loadmat(ball_fault_paths[rpm])
            self.ball_fault_data.append(torch.Tensor([h[list(h)[3]][:, 0], h[list(h)[4]][:, 0], h[list(h)[5]][:, 0]]))
            self.ball_fault_rpm.append(h[list(h)[6]])

        # Load data of faulty inner rings
        ir_fault_paths = {'1797': "../../data/CWRU/IR_007_0.mat", '1772': "../../data/CWRU/IR_007_1.mat",
                          '1750': "../../data/CWRU/IR_007_2.mat", '1730': "../../data/CWRU/IR_007_3.mat"}

        self.ir_fault_data = []
        self.ir_fault_rpm = []

        for rpm in rpms:
            h = scipy.io.loadmat(ir_fault_paths[rpm])
            self.ir_fault_data.append(torch.Tensor([h[list(h)[3]][:, 0], h[list(h)[4]][:, 0], h[list(h)[5]][:, 0]]))
            self.ir_fault_rpm.append(h[list(h)[6]])

        # calculate length of healthy data
        self.healthy_length = 0
        for healthy_data_bit in self.healthy_data:
            self.healthy_length += healthy_data_bit.shape[1]

        # calculate length of faulty ball data
        self.ball_fault_length = 0
        for ball_fault_data_bit in self.ball_fault_data:
            self.ball_fault_length += ball_fault_data_bit.shape[1]

        # calculate length of faulty inner ring data
        self.ir_fault_length = 0
        for ir_fault_data_bit in self.ir_fault_data:
            self.ir_fault_length += ir_fault_data_bit.shape[1]

    def __len__(self):
        length = 0
        for healthy_data_bit in self.healthy_data:
            length += int(healthy_data_bit.shape[1] / self.sample_length)

        for ball_fault_data_bit in self.ball_fault_data:
            length += int(ball_fault_data_bit.shape[1] / self.sample_length)

        for ir_fault_data_bit in self.ir_fault_data:
            length += int(ir_fault_data_bit.shape[1] / self.sample_length)
        return length

    def __getitem__(self, item):

        series = [self.healthy_data, self.ir_fault_data, self.ball_fault_data]
        gt = -1
        # return one of healthy data
        for ts in series:
            gt += 1
            for healty_ts in ts:
                if item < healty_ts.shape[1] / self.sample_length - 1:
                    if not self.normalise:
                        return dict(data=healty_ts[:, item * self.sample_length:(item+1) * self.sample_length], gt=gt)
                    else:
                        return dict(
                            data=(healty_ts[:, item * self.sample_length:(item+1) * self.sample_length] - self.mean) /
                                 self.variance,
                            gt=gt)
                else:
                    item -= int(healty_ts.shape[1] / self.sample_length)
