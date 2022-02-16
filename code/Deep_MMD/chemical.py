import sys

sys.path.append('../supervised_only')
import chemical_loader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch
from sklearn.preprocessing import StandardScaler
import mmd


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.l1 = torch.nn.Linear(128, 64)
        self.l1_act = torch.nn.ReLU()

        self.l2 = torch.nn.Linear(64, 32)
        self.l2_act = torch.nn.ReLU()
        self.l2_da = torch.nn.Dropout(.2)

        self.l3 = torch.nn.Linear(32, 16)
        self.l3_act = torch.nn.ReLU()

        self.l4 = torch.nn.Linear(16, 6)

        self.out = torch.nn.Softmax(1)

        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.x4 = None

    def forward(self, data):
        self.x1 = self.l1(data)
        x1_act = self.l1_act(self.x1)

        self.x2 = self.l2(x1_act)
        x2_act = self.l2_act(self.x2)
        x2_da = self.l2_da(x2_act)

        self.x3 = self.l3(x2_da)
        x3_act = self.l3_act(self.x3)

        self.x4 = self.l4(x3_act)
        output = self.out(self.x4)

        return output


class Classifier1(torch.nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()

        self.l1 = torch.nn.Linear(128, 200)
        self.l1_act = torch.nn.ReLU()

        self.l2 = torch.nn.Linear(200, 16)
        self.l2_act = torch.nn.ReLU()

        self.l3 = torch.nn.Linear(16, 6)
        self.out = torch.nn.Softmax(1)

    def forward(self, data):
        x1 = self.l1(data)
        x1_act = self.l1_act(x1)

        x2 = self.l2(x1_act)
        x2_act = self.l2_act(x2)

        x3 = self.l3(x2_act)
        output = self.out(x3)

        return output


def evaluate_model(model, dataloader, scaler, device):
    m = 0
    n = 0
    model.eval()

    for sample in dataloader:
        m += 1
        data = torch.Tensor(scaler.transform(sample['data'])).to(device)
        output = model.forward(data)
        if torch.argmax(output) == sample['gt'].to(device):
            n += 1
    return n/m


def main():
    results = pd.DataFrame(columns=["batch1+2"])
    seed = 42
    normalize_sep = False
    balance = True
    lr = 0.0001
    num_epochs = 1500
    batch_size = 1709
    kernel = 'rbf'
    weight_mmd = 10
    device = "cuda:0"

    if not torch.cuda.is_available():
        device = "cpu"

    dataset_s1 = chemical_loader.ChemicalLoader(1, train=True, normalise=False, balance=balance, seed=seed)
    dataset_s2 = chemical_loader.ChemicalLoader(2, train=True, normalise=False, balance=balance, seed=seed)
    dataset_s1.extend_dataset(len(dataset_s2))  # balance dataset_s1 and dataset_s2
    scaler = StandardScaler()
    scaler.fit(np.vstack((dataset_s1.data, dataset_s2.data)))
    dataset_s = torch.utils.data.ConcatDataset([dataset_s1, dataset_s2])

    dataloader_s = torch.utils.data.DataLoader(dataset_s1, shuffle=True, batch_size=batch_size, drop_last=True)
    dataloader_s_eval = torch.utils.data.DataLoader(dataset_s1, shuffle=True, batch_size=1)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for tgt in range(1, 11):
        model = Classifier().to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        dataset_t_train = chemical_loader.ChemicalLoader(tgt, train=True, normalise=False, balance=balance, seed=seed)
        dataset_t_train.extend_dataset(len(dataset_s))
        dataset_t_eval = chemical_loader.ChemicalLoader(tgt, train=False, normalise=False, balance=balance, seed=seed)
        dataloader_t_eval = torch.utils.data.DataLoader(dataset_t_eval, shuffle=False, batch_size=1)
        dataloader_t_train = torch.utils.data.DataLoader(dataset_t_train, shuffle=False, batch_size=batch_size,
                                                         drop_last=True)

        results = results.append(pd.DataFrame(index=['batch' + str(tgt)]))
        for epoch in range(num_epochs):
            for sample_s, sample_t in zip(dataloader_s, dataloader_t_train):
                data_s = torch.Tensor(scaler.transform(sample_s['data'])).to(device)
                data_t = torch.Tensor(scaler.transform(sample_t['data'])).to(device)
                output_s = model.forward(data_s)
                features_s = model.x2
                loss_class = loss_fn(output_s, sample_s['gt'][:, 0].to(device))
                output_t = model.forward(data_t)
                features_t = model.x2
                loss_mmd = mmd.mmd3(features_s, features_t, kernel=kernel, sigma=1000)
                loss = loss_class + loss_mmd * weight_mmd
                if loss < 1.1:
                    break
                loss.backward()
                optimizer.step()
                print(loss)
        acc = evaluate_model(model, dataloader_t_eval, scaler, device)
        print('batch1+2 --> batch' + str(tgt) + ' accuracy: ' + str(round(acc, 4)))
        results['batch1+2']['batch' + str(tgt)] = round(acc, 4)

    results.to_csv('../eval/results/chemical/mmd' + '_normalise_sep' * normalize_sep + '_balance' * balance + '_kernel'
                   + kernel + '_weightmmd' + str(round(weight_mmd, 2)) + '.csv', ';')

    # plot the results
    plt.plot(results['batch1+2'], label='nn')
    plt.legend()
    plt.axis([0, 10, 0, 1])
    plt.savefig('../eval/results/chemical/mmd' + '_normalise_sep' * normalize_sep + '_balance' * balance + '_kernel'
                + kernel + '_weightmmd' + str(round(weight_mmd, 2)) + '.png')


if __name__ == "__main__":
    main()
