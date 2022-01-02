import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../supervised_only')
import CWRU_loader_extended_task
from CWRU_loader_extended_task import find_sampling_weights
from scipy.io import savemat
import datetime
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class Classifier4(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier4, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 16, 10)
        self.l1_act = torch.nn.ReLU()
        self.l1_pool = torch.nn.MaxPool1d(2)

        self.l2 = torch.nn.Conv1d(16, 32, 5)
        self.l2_act = torch.nn.ReLU()
        self.l2_pool = torch.nn.MaxPool1d(2)

        self.l3 = torch.nn.Conv1d(32, 64, 5)
        self.l3_act = torch.nn.ReLU()

        self.l4 = torch.nn.Conv1d(64, 64, 5, stride=2)
        self.l4_act = torch.nn.ReLU()
        self.l4_pool = torch.nn.MaxPool1d(2)

        self.l5 = torch.nn.Conv1d(64, 128, 5, stride=2)
        self.l5_act = torch.nn.ReLU()
        self.l5_pool = torch.nn.MaxPool1d(2)

        self.l6 = torch.nn.Linear(896, 224)
        self.l6_act = torch.nn.ReLU()

        self.l7 = torch.nn.Linear(224, 10)
        self.out = torch.nn.Softmax(1)

        self.x1_act = None
        self.x2_act = None
        self.x3_act = None
        self.x4_act = None

    def forward(self, data):
        x1 = self.l1(data)
        self.x1_act = self.l1_act(x1)
        x1_pool = self.l1_pool(self.x1_act)  # self.l1_dropout(self.x1_act)

        x2 = self.l2(x1_pool)
        self.x2_act = self.l2_act(x2)
        x2_pool = self.l2_pool(self.x2_act)  # self.l2_dropout(self.x2_act)

        x3 = self.l3(x2_pool)
        self.x3_act = self.l3_act(x3)

        x4 = self.l4(self.x3_act)
        self.x4_act = self.l3_act(x4)
        x4_pool = self.l4_pool(self.x4_act)

        x5 = self.l4(x4_pool)
        self.x5_act = self.l5_act(x5)
        x5_pool = self.l5_pool(self.x5_act)
        self.x5_reshape = x5_pool.flatten(1)

        x6 = self.l6(self.x5_reshape)
        self.x6_act = self.l6_act(x6)

        x7 = self.l7(self.x6_act)
        output = self.out(x7)
        return output


def mmd(x, y, kernel='multiscale'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [2, 5, 9, 13]  # [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX = XX + torch.exp(-0.5 * dxx / a)
            YY = YY + torch.exp(-0.5 * dyy / a)
            XY = XY + torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def evaluate_model(clf, loader):
    n = 0
    m = 0

    for sample in enumerate(loader):
        output = clf.forward(sample[1]["data"].to(device))
        gt = sample[1]["gt"].to(device)
        m += 1
        if torch.argmax(output) == gt:
            n += 1

    return n / m


if __name__ == "__main__":

    # Initialise output files
    results = pd.DataFrame(columns=['1797->1772', '1797->1750', '1797->1730', '1772->1797', '1772->1750', '1772->1730',
                                    '1750->1797', '1750->1772', '1750->1730', '1730->1797', '1730->1772', '1730->1750'])
    results = results.append(pd.DataFrame(index=['1797', '1772', '1750', '1730']))

    for rpm in ['1797', '1772', '1750', '1730']:
        for rpm_target in ['1797', '1772', '1750', '1730']:
            if rpm == rpm_target:
                continue
            print('source rpm', str(rpm), ' -> target rpm', rpm_target)

            # Define what device to run on
            if torch.cuda.is_available():
                device = 'cuda:1'
            else:
                device = 'cpu'

            # Initialize source dataset
            sample_length = 1000
            dataset_s = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm], normalise=True, train=True)

            sampler = torch.utils.data.WeightedRandomSampler(dataset_s.find_sampling_weights(), len(dataset_s))
            loader_train_s = DataLoader(dataset_s, batch_size=100, shuffle=False, num_workers=1, sampler=sampler,
                                        drop_last=True)

            # Initialise target training dataset
            dataset_t = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm_target], normalise=True, train=True)

            sampler = torch.utils.data.WeightedRandomSampler(dataset_t.find_sampling_weights(), len(dataset_t))
            loader_train_t = DataLoader(dataset_t, batch_size=100, shuffle=False, num_workers=1, sampler=sampler,
                                        drop_last=True)

            # Initialise model and optimizer
            model = Classifier4(sample_length).to(device)
            model.train()
            lr = 0.001
            weight_decay = 0
            # weight_path = '../../models/CWRU/sup_only4_rpm' + rpm + '_lr' + str(lr) + '_weightdecay' + str(weight_decay) + '.pth'
            weight_path = None
            if weight_path is not None:
                model.load_state_dict(torch.load(weight_path))

            loss_function = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            N = 0
            num_epochs = 1501

            for epoch in range(num_epochs):
                for sample_s, sample_t in zip(enumerate(loader_train_s), enumerate(loader_train_t)):
                    data_s = sample_s[1]["data"].to(device)
                    gt = sample_s[1]["gt"].to(device)
                    data_t = sample_t[1]["data"].to(device)
                    output = model.forward(data_s)
                    features_s = model.x5_reshape
                    model.forward(data_t)
                    loss_classification = loss_function(output, gt)
                    if epoch > num_epochs / 3:
                        loss_mmd = 3 * mmd(features_s, model.x5_reshape)
                        loss = loss_classification + loss_mmd
                    else:
                        loss = loss_classification
                        loss_mmd = torch.Tensor([0])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    N += 1
                    # For debugging
                    if N % 50 == 0 and N != 0:
                        print('Loss after', N, 'iterations: ', round(loss.item(), 3), ' classification: ',
                              round(loss_classification.item(), 4), ', mmd: ', round(loss_mmd.item(), 3))

            # save model weights
            torch.save(model.state_dict(), '../../models/CWRU/mmd_rpm' + rpm + '_lr' + str(lr) + '_weightdecay' +
                       str(weight_decay) + '.pth')

            # evaluate the trained model
            model.eval()
            for rpm_eval in ['1797', '1772', '1750', '1730']:
                if rpm_eval == rpm:
                    dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[rpm_eval], train=False)
                else:
                    dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[rpm_eval])
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
                acc_target = round(evaluate_model(model, dataloader), 4)
                print('evaluation rpm', str(rpm_eval), acc_target)
                results[rpm + '->' + rpm_target][rpm_eval] = acc_target

    results.to_csv('../eval/results/CWRU/' + 'mmd' + rpm + '_lr' + str(lr) + '_epochs' + str(num_epochs) +
                    '_weightdecay' + str(weight_decay) + '.csv', ';')
