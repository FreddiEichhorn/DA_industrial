import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../supervised_only')
import CWRU_loader_extended_task
import pandas as pd
import argparse
import matplotlib as mpl
mpl.use('Agg')


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


class Classifier9(torch.nn.Module):
    def __init__(self, ss):
        super(Classifier9, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 20, 10)
        self.l1_act = torch.nn.ReLU()
        self.l1_pool = torch.nn.MaxPool1d(4)
        self.l1_da = torch.nn.Dropout(.1)

        self.l2 = torch.nn.Conv1d(20, 40, 5)
        self.l2_act = torch.nn.ReLU()
        self.l2_pool = torch.nn.MaxPool1d(4)
        self.l2_da = torch.nn.Dropout(.1)

        self.l3 = torch.nn.Conv1d(40, 80, 5)
        self.l3_act = torch.nn.ReLU()
        self.l3_pool = torch.nn.MaxPool1d(8)
        self.l3_da = torch.nn.Dropout(.1)

        self.l6 = torch.nn.Linear(560, 224)
        self.l6_act = torch.nn.ReLU()

        self.l7 = torch.nn.Linear(224, 10)
        self.out = torch.nn.Softmax(1)

        self.x1_act = None
        self.x2_act = None
        self.x2_pool = None
        self.x3_act = None
        self.x4_act = None

    def forward(self, data):
        x1 = self.l1(data)
        self.x1_act = self.l1_act(x1)
        x1_pool = self.l1_pool(self.x1_act)  # self.l1_dropout(self.x1_act)
        x1_da = self.l1_da(x1_pool)

        x2 = self.l2(x1_da)
        self.x2_act = self.l2_act(x2)
        self.x2_pool = self.l2_pool(self.x2_act)  # self# .l2_dropout(self.x2_act)
        x2_da = self.l2_da(self.x2_pool)

        x3 = self.l3(x2_da)
        self.x3_act = self.l3_act(x3)
        x3_pool = self.l3_pool(self.x3_act)
        x3_da = self.l3_da(x3_pool)
        self.x5_reshape = x3_da.flatten(1)

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


def mmd2(Xs, Xt, sigma=1):
    Z = torch.cat((Xs, Xt), 0)
    ZZT = torch.mm(Z, Z.T)
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T
    K = torch.exp(-exponent / (2 * sigma ** 2))

    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = torch.cat((1 / m * torch.ones(m, 1), -1 / m * torch.ones(m, 1)), 0)
    M = e * e.T
    tmp = torch.mm(torch.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = torch.trace(tmp).to(device)
    return loss


def mmd3(xs, xt, kernel='linear', sigma=1):

    def linear_kernel(a, b):
        helper = a @ b.T
        return -torch.diag(a @ a.T).expand_as(helper).T + 2 * helper - torch.diag(b @ b.T).expand_as(helper)

    def rbf_kernel(a, b):
        return torch.exp(linear_kernel(a, b) / sigma)

    if kernel == 'linear':
        k1 = linear_kernel(xs, xs)
        k2 = linear_kernel(xs, xt)
        k3 = linear_kernel(xt, xt)
    elif kernel == 'rbf':
        k1 = rbf_kernel(xs, xs)
        k2 = rbf_kernel(xs, xt)
        k3 = rbf_kernel(xt, xt)
    else:
        k1 = None
        k2 = None
        k3 = None

    return torch.mean(k1 - 2 * k2 + k3)


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


def loss_reg(output, n_classes=10):
    ls = (output.sum(0) - output.shape[0] / n_classes)**2
    return ls.sum(0) / n_classes


if __name__ == "__main__":

    # Initialise output files
    results = pd.DataFrame(columns=['1797->1772', '1797->1750', '1797->1730', '1772->1797', '1772->1750', '1772->1730',
                                    '1750->1797', '1750->1772', '1750->1730', '1730->1797', '1730->1772', '1730->1750'])
    results = results.append(pd.DataFrame(index=['1797', '1772', '1750', '1730']))

    # Hyperparameters of the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate of the model", default=0.001, required=False)
    parser.add_argument("--weight_decay", default=0, required=False)
    parser.add_argument("--num_epochs", default=1000, required=False)
    parser.add_argument("--regularize", default=True, required=False)
    parser.add_argument("--stratify", default=True, required=False)
    parser.add_argument("--kernel_type", default="linear", required=False)
    parser.add_argument("--device", default="cuda:0", required=False)
    args = parser.parse_args()
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    regularize = args.regularize
    stratify = args.stratify
    kernel_type = args.kernel_type

    for rpm in ['1797', '1772', '1750', '1730']:
        for rpm_target in ['1797', '1772', '1750', '1730']:
            if rpm == rpm_target:
                continue
            print('source rpm', str(rpm), ' -> target rpm', rpm_target)

            # Define what device to run on
            if torch.cuda.is_available():
                device = args.device
            else:
                device = 'cpu'

            # Initialize source dataset
            sample_length = 1000
            dataset_s = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm], normalise=True, train=True)

            if not stratify:
                sampler = torch.utils.data.WeightedRandomSampler(dataset_s.find_sampling_weights(), len(dataset_s))
                loader_train_s = DataLoader(dataset_s, batch_size=20, shuffle=False, num_workers=1, sampler=sampler)
            else:
                loader_train_s = CWRU_loader_extended_task.StratifiedDataLoader(dataset_s, 20)

            # Initialise target training dataset
            dataset_t = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm_target], normalise=True, train=True)

            sampler = torch.utils.data.WeightedRandomSampler(dataset_t.find_sampling_weights(), len(dataset_t))
            loader_train_t = DataLoader(dataset_t, batch_size=20, shuffle=False, num_workers=1, sampler=sampler,
                                        drop_last=True)

            # Initialise model and optimizer
            model = Classifier9(sample_length).to(device)
            model.train()
            weight_path = None
            if weight_path is not None:
                model.load_state_dict(torch.load(weight_path))

            loss_function = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            N = 0

            for epoch in range(num_epochs):
                for sample_s, sample_t in zip(enumerate(loader_train_s), enumerate(loader_train_t)):
                    data_s = sample_s[1]["data"].to(device)
                    gt = sample_s[1]["gt"].to(device)
                    data_t = sample_t[1]["data"].to(device)
                    output = model.forward(data_s)
                    features_s = model.x5_reshape
                    model.forward(data_t)
                    loss_classification = loss_function(output, gt)
                    loss_mmd = mmd3(features_s, model.x5_reshape, kernel_type, 10000)
                    if kernel_type == 'linear':
                        loss = loss_classification + loss_reg(output) / 1.5 + loss_mmd / 1000
                    else:
                        loss = loss_classification + loss_reg(output) / 1.5 + loss_mmd * 10
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
                       str(weight_decay) + '_reg' * regularize + '.pth')

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

    results.to_csv('../eval/results/CWRU/' + 'mmd' + '_lr' + str(lr) + '_epochs' + str(num_epochs) + '_weightdecay' +
                   str(weight_decay) + '_' + kernel_type + '_reg' * regularize + '_strat' * stratify + '.csv', ';')
