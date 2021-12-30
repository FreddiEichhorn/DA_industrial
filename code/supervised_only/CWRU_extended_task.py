import torch
from torch.utils.data import DataLoader
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
    results = pd.DataFrame(columns=['1797', '1772', '1750', '1730'])
    results = results.append(pd.DataFrame(index=['1797', '1772', '1750', '1730']))
    loss_plots = {'1797': [], '1772': [], '1750': [], '1730': []}

    for rpm in ['1797', '1772', '1750', '1730']:
        print('source rpm', str(rpm))

        # Define what device to run on
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        # Initialize source dataset
        sample_length = 1000
        dataset = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm], normalise=True, train=True)

        sampler = torch.utils.data.WeightedRandomSampler(dataset.find_sampling_weights(), len(dataset))
        loader_train_s = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1, sampler=sampler)

        # Initialise model and optimizer
        model = Classifier4(sample_length).to(device)
        model.train()
        lr = 0.001
        weight_decay = 0
        #weight_path = '../../models/CWRU/sup_only4_rpm' + rpm + '_lr' + str(lr) + '_weightdecay' + str(weight_decay) + '.pth'
        weight_path = None
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path))

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        N = 0
        num_epochs = 1500

        for _ in range(num_epochs):
            for sample in enumerate(loader_train_s):
                data = sample[1]["data"].to(device)
                gt = sample[1]["gt"].to(device)
                output = model.forward(data)
                loss = loss_function(output, gt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                N += 1
                # For debugging
                if N % 50 == 0 and N != 0:
                    print('Loss after', N, 'iterations: ', loss)
                    loss_plots[rpm].append(loss.detach().cpu())
        # save model weights
        torch.save(model.state_dict(), '../../models/CWRU/sup_only4_rpm' + rpm + '_lr' + str(lr) +
                    '_weightdecay' + str(weight_decay) + '.pth')

        # evaluate the trained model
        for rpm_target in ['1797', '1772', '1750', '1730']:
            model.eval()
            if rpm_target == rpm:
                dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[rpm_target], train=False)
            else:
                dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[rpm_target])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            acc_target = round(evaluate_model(model, dataloader), 4)
            print('target rpm', str(rpm_target), acc_target)
            results[rpm][rpm_target] = acc_target

            # Generate deep features of network for classical DA
            dataset_s = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm])
            dataloader_s = DataLoader(dataset_s, batch_size=len(dataset_s), shuffle=False, num_workers=1)

            dataset_t = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm_target])
            dataloader_t = DataLoader(dataset_t, batch_size=len(dataset_t), shuffle=False, num_workers=1)
            # Save the features
            with torch.no_grad():
                for sample in dataloader_s:  # We need all of the data in the dataset for this
                    X_s = sample['data'].to(device)
                    y_s = sample['gt'].to(device)
                    model.forward(X_s)
                    features_s = model.x5_reshape
                    feat_dict_s = {'fts': features_s.cpu().numpy(), 'labels': y_s.unsqueeze(1).cpu().numpy()}
                    savemat('../../data/CWRU/deep_features/src2_' + rpm + '.mat',
                            feat_dict_s)

                for sample in dataloader_t:
                    X_t = sample['data'].to(device)
                    y_t = sample['gt'].to(device)
                    model.forward(X_t)
                    features_t = model.x5_reshape
                    feat_dict_t = {'fts': features_t.cpu().numpy(), 'labels': y_t.unsqueeze(1).cpu().numpy()}
                    savemat('../../data/CWRU/deep_features/src2_' + rpm + '_tgt_' + rpm_target + '.mat',
                            feat_dict_t)

    results.to_csv('../eval/results/CWRU/' + 'sup_only4' + rpm + '_lr' + str(lr) + '_epochs' + str(num_epochs) +
                    '_weightdecay' + str(weight_decay) + '.csv', ';')

    fig, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(loss_plots['1797'])
    axarr[0, 1].plot(loss_plots['1772'])
    axarr[1, 0].plot(loss_plots['1750'])
    axarr[1, 1].plot(loss_plots['1730'])
    fig.savefig('../eval/results/CWRU/' + 'sup_only4' + '_lr' + str(lr) + '_weightdecay' + str(weight_decay) +
                '.png')
