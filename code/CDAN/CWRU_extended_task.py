import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../supervised_only')
import CWRU_loader_extended_task
import pandas as pd
import matplotlib as mpl
import argparse
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
        self.x5_act = None
        self.x5_reshape = None
        self.x6_act = None

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


class Classifier8(torch.nn.Module):
    def __init__(self, ss):
        super(Classifier8, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 20, 10)
        self.l1_act = torch.nn.ReLU()
        self.l1_pool = torch.nn.MaxPool1d(4)

        self.l2 = torch.nn.Conv1d(20, 40, 5)
        self.l2_act = torch.nn.ReLU()
        self.l2_pool = torch.nn.MaxPool1d(4)

        self.l3 = torch.nn.Conv1d(40, 80, 5)
        self.l3_act = torch.nn.ReLU()
        self.l3_pool = torch.nn.MaxPool1d(8)

        self.l6 = torch.nn.Linear(560, 224)
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
        self.x2_pool = self.l2_pool(self.x2_act)  # self.l2_dropout(self.x2_act)

        x3 = self.l3(self.x2_pool)
        self.x3_act = self.l3_act(x3)
        x3_pool = self.l3_pool(self.x3_act)
        self.x5_reshape = x3_pool.flatten(1)

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
        self.x2_pool = None
        self.x3_act = None
        self.x4_act = None

    def forward(self, data):
        x1 = self.l1(data)
        self.x1_act = self.l1_act(x1)
        x1_pool = self.l1_pool(self.x1_act)  # self.l1_dropout(self.x1_act)
        x1_da = self.l1_da(x1_pool)

        x2 = self.l2(x1_da)
        x2_act = self.l2_act(x2)
        self.x2_pool = self.l2_pool(x2_act)  # self# .l2_dropout(self.x2_act)
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


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x #x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return grad_output * -1


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


class DomainCritic(torch.nn.Module):
    def __init__(self):
        super(DomainCritic, self).__init__()
        self.l6 = torch.nn.Linear(896, 224)
        self.l6_act = torch.nn.ReLU()

        self.l7 = torch.nn.Linear(224, 2)
        self.out = torch.nn.Softmax(1)

    def forward(self, input):
        rev = GRL.apply(input)
        x6 = self.l6(rev)
        x6_act = self.l6_act(x6)

        x7 = self.l7(x6_act)
        output = self.out(x7)
        return output


class DomainCritic2(torch.nn.Module):
    def __init__(self):
        super(DomainCritic2, self).__init__()

        self.l4 = torch.nn.Conv1d(64, 64, 5, stride=2)
        self.l4_act = torch.nn.ReLU()
        self.l4_pool = torch.nn.MaxPool1d(2)

        self.l5 = torch.nn.Conv1d(64, 128, 5, stride=2)
        self.l5_act = torch.nn.ReLU()
        self.l5_pool = torch.nn.MaxPool1d(2)

        self.l6 = torch.nn.Linear(896, 224)
        self.l6_act = torch.nn.ReLU()

        self.l7 = torch.nn.Linear(224, 2)
        self.out = torch.nn.Softmax(1)

        self.x1_act = None
        self.x2_act = None
        self.x3_act = None
        self.x4_act = None
        self.x5_act = None
        self.x5_reshape = None
        self.x6_act = None

    def forward(self, data):
        rev = GRL.apply(data)

        x4 = self.l4(rev)
        self.x4_act = self.l4_act(x4)
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


class DomainCritic3(torch.nn.Module):
    def __init__(self):
        super(DomainCritic3, self).__init__()
        self.l3 = torch.nn.Conv1d(40, 80, 5)
        self.l3_act = torch.nn.ReLU()
        self.l3_pool = torch.nn.MaxPool1d(8)

        self.l6 = torch.nn.Linear(560, 224)
        self.l6_act = torch.nn.ReLU()

        self.l7 = torch.nn.Linear(224, 2)
        self.out = torch.nn.Softmax(1)

        self.x1_act = None
        self.x2_act = None
        self.x3_act = None
        self.x4_act = None

    def forward(self, data):
        rev = GRL.apply(data)

        x3 = self.l3(rev)
        self.x3_act = self.l3_act(x3)
        x3_pool = self.l3_pool(self.x3_act)
        self.x5_reshape = x3_pool.flatten(1)

        x6 = self.l6(self.x5_reshape)
        self.x6_act = self.l6_act(x6)

        x7 = self.l7(self.x6_act)
        output = self.out(x7)
        return output


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


class ConditionalDomainCritic:
    def __init__(self, clf, n_classes):
        self.clf_list = []
        self.n_classes = n_classes
        for _ in range(n_classes):
            self.clf_list.append(clf().to(device).train())
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, data, domain_label, class_prediction):
        out = torch.Tensor([0] * data.shape[0]).to(device)
        class_prediction = class_prediction.detach()
        if class_prediction.ndim > 1:
            for i in range(self.n_classes):
                domain_pred = self.clf_list[i].forward(data)
                out = out + self.loss(domain_pred, domain_label) * class_prediction[:, i].detach()

        else:
            for i in range(self.n_classes):
                domain_pred = self.clf_list[i].forward(data)
                out = out + self.loss(domain_pred, domain_label) * (class_prediction == i)

        return torch.sum(out) / data.shape[0]

    def parameters(self):
        out = []
        for clf in self.clf_list:
            out = out + list(clf.parameters())
        return out


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
    parser.add_argument("--num_epochs", default=500, required=False, type=int)
    parser.add_argument("--regularize", default=True, required=False)
    parser.add_argument("--stratify", default=True, required=False)
    parser.add_argument("--device", default="cuda:0", required=False)
    parser.add_argument("--partial_da", default=False, required=False)
    parser.add_argument("--type_mdc", default="adversary", required=False, help="adversary, rbf or linear")
    parser.add_argument("--weight_cdc", default=.6, required=False, type=float)
    parser.add_argument("--weight_mdc", default=.3, required=False, type=float)
    parser.add_argument("--weight_reg", default=.7, required=False)
    args = parser.parse_args()
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    regularize = args.regularize
    stratify = args.stratify
    partial_da = args.partial_da
    type_mdc = args.type_mdc
    weight_cdc = float(args.weight_cdc)
    weight_mdc = float(args.weight_mdc)
    weight_reg = float(args.weight_reg)

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
                loader_train_s = DataLoader(dataset_s, batch_size=20, shuffle=False, num_workers=1, sampler=sampler,
                                            drop_last=True)
            else:
                loader_train_s = CWRU_loader_extended_task.StratifiedDataLoader(dataset_s, 20)

            # Initialise target training dataset
            dataset_t = CWRU_loader_extended_task.CWRU(sample_length, True, partial_da, [rpm_target], train=True)

            sampler = torch.utils.data.WeightedRandomSampler(dataset_t.find_sampling_weights(), len(dataset_t))
            loader_train_t = DataLoader(dataset_t, batch_size=20, shuffle=False, num_workers=1, sampler=sampler,
                                        drop_last=True)

            # Initialise model and optimizer
            model = Classifier9(sample_length).to(device)
            model.train()
            cdc = ConditionalDomainCritic(DomainCritic3, 10)
            if type_mdc == "adversary":
                mdc = DomainCritic3().to(device)
                mdc.train()

            loss_function = torch.nn.CrossEntropyLoss()
            loss_da = torch.nn.CrossEntropyLoss(reduction='none')
            if type_mdc == 'adversary':
                optimizer = torch.optim.SGD(list(model.parameters()) + cdc.parameters() + list(mdc.parameters()), lr=lr)
            else:
                optimizer = torch.optim.SGD(list(model.parameters()) + cdc.parameters(), lr=lr)
            N = 0

            for epoch in range(num_epochs):
                for sample_s, sample_t in zip(enumerate(loader_train_s), enumerate(loader_train_t)):
                    data_s = sample_s[1]["data"].to(device)
                    gt = sample_s[1]["gt"].to(device)
                    data_t = sample_t[1]["data"].to(device)
                    output_s = model.forward(data_s)
                    loss_classification = loss_function(output_s, gt)

                    loss_cdc_s = cdc.forward(model.x2_pool, torch.LongTensor([0] * data_s.shape[0]).to(device), gt)
                    features_x5_reshape = model.x5_reshape
                    features_x2_pool = model.x2_pool

                    output_t = model.forward(data_t)
                    loss_cdc_t = cdc.forward(model.x2_pool, torch.LongTensor([1] * data_s.shape[0]).to(device),output_t)

                    if type_mdc == "adversary":
                        domain_pred_s = mdc.forward(features_x2_pool)
                        loss_mdc_s = loss_da(domain_pred_s, torch.LongTensor([0] * domain_pred_s.shape[0]).
                                             to(device)).sum() / 20

                        domain_pred_t = mdc.forward(model.x2_pool)
                        loss_mdc_t = loss_da(domain_pred_t, torch.LongTensor([1] * domain_pred_t.shape[0]).
                                             to(device)).sum() / 20
                        loss_mdc = loss_mdc_s + loss_mdc_t
                    else:
                        loss_mdc = mmd3(features_x5_reshape, model.x5_reshape, type_mdc, 10000)

                    loss = (loss_cdc_s + loss_cdc_t) * weight_cdc + loss_classification + loss_mdc * weight_mdc + loss_reg(output_s, 10) * weight_reg

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    N += 1
                    # For debugging
                    if N % 50 == 0 or N < 50:
                        print('Loss after', N, 'iterations: ', round(loss.item(), 3), ' classification: ',
                              round(loss_classification.item(), 4), ', dc: ', round((loss_cdc_t + loss_cdc_s).item(), 3)
                              )

            # save model weights
            torch.save(model.state_dict(), '../../models/CWRU/dc_rpm' + rpm + '_lr' + str(lr) + '_weightdecay' +
                       str(weight_decay) + '_partial' * partial_da + '.pth')

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

    results.to_csv('../eval/results/CWRU/' + 'cdc' + rpm + '_lr' + str(lr) + '_epochs' + str(num_epochs) + '_reg' +
                   str(weight_reg) + '_cdcwght' + str(weight_cdc) + '_' + type_mdc + '_mdcwght' + str(weight_mdc) +
                   '.csv', ';')
