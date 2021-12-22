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


class Classifier(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 32, 3)
        self.l1_act = torch.nn.ReLU()
        self.l1_dropout = torch.nn.Dropout()
        self.l2 = torch.nn.Conv1d(32, 16, 3)
        self.l2_act = torch.nn.ReLU()
        self.l2_dropout = torch.nn.Dropout()
        self.l3 = torch.nn.Conv1d(16, 1, 3)
        self.l3_act = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(sample_length-6, 256)
        self.l4_act = torch.nn.ReLU()
        self.l5 = torch.nn.Linear(256, 128)
        self.l5_act = torch.nn.ReLU()
        self.l6 = torch.nn.Linear(128, 10)
        self.out = torch.nn.Softmax(2)

        self.x1_act = None
        self.x2_act = None
        self.x3_act = None
        self.x4_act = None

    def forward(self, data):
        x1 = self.l1(data)
        self.x1_act = self.l1_act(x1)
        x1_dropout = self.l1_dropout(self.x1_act)
        x2 = self.l2(x1_dropout)
        self.x2_act = self.l2_act(x2)
        x2_dropout = self.l2_dropout(self.x2_act)
        x3 = self.l3(x2_dropout)
        self.x3_act = self.l3_act(x3)
        x4 = self.l4(self.x3_act)
        self.x4_act = self.l4_act(x4)
        x5 = self.l5(self.x4_act)
        x5_act = self.l5_act(x5)
        x6 = self.l6(x5_act)
        output = self.out(x6)
        return output.squeeze(1)


class Classifier2(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier2, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 32, 10, stride=3)
        self.l1_act = torch.nn.ReLU()
        self.l1_dropout = torch.nn.Dropout()
        self.l2 = torch.nn.Conv1d(32, 16, 10, stride=3)
        self.l2_act = torch.nn.ReLU()
        self.l2_dropout = torch.nn.Dropout()
        self.l3 = torch.nn.Conv1d(16, 1, 10, stride=3)
        self.l3_act = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(33, 256)
        self.l4_act = torch.nn.ReLU()
        self.l5 = torch.nn.Linear(256, 128)
        self.l5_act = torch.nn.ReLU()
        self.l6 = torch.nn.Linear(128, 10)
        self.out = torch.nn.Softmax(2)

        self.x1_act = None
        self.x2_act = None
        self.x3_act = None
        self.x4_act = None

    def forward(self, data):
        x1 = self.l1(data)
        self.x1_act = self.l1_act(x1)
        x1_dropout = self.x1_act  # self.l1_dropout(self.x1_act)
        x2 = self.l2(x1_dropout)
        self.x2_act = self.l2_act(x2)
        x2_dropout = self.x2_act  # self.l2_dropout(self.x2_act)
        x3 = self.l3(x2_dropout)
        self.x3_act = self.l3_act(x3)
        x4 = self.l4(self.x3_act)
        self.x4_act = self.l4_act(x4)
        x5 = self.l5(self.x4_act)
        x5_act = self.l5_act(x5)
        x6 = self.l6(x5_act)
        output = self.out(x6)
        return output.squeeze(1)


class Classifier3(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier3, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 8, 10)
        self.l1_act = torch.nn.ReLU()
        self.l1_pool = torch.nn.MaxPool1d(2)
        self.l2 = torch.nn.Conv1d(8, 16, 5)
        self.l2_act = torch.nn.ReLU()
        self.l2_pool = torch.nn.MaxPool1d(2)
        self.l3 = torch.nn.Conv1d(16, 32, 5, stride=2)
        self.l3_act = torch.nn.ReLU()
        self.l3_pool = torch.nn.MaxPool1d(2)
        self.l4 = torch.nn.Conv1d(32, 64, 5, stride=2)
        self.l4_act = torch.nn.ReLU()
        self.l4_pool = torch.nn.MaxPool1d(2)
        self.l5 = torch.nn.Linear(896, 224)
        self.l5_act = torch.nn.ReLU()
        self.l6 = torch.nn.Linear(224, 10)
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
        x3_pool = self.l3_pool(self.x3_act)
        x4 = self.l4(x3_pool)
        self.x4_act = self.l4_act(x4)
        x4_pool = self.l4_pool(self.x4_act)
        x4_reshape = x4_pool.flatten(1)
        x5 = self.l5(x4_reshape)
        self.x5_act = self.l5_act(x5)
        x6 = self.l6(self.x5_act)
        output = self.out(x6)
        return output


class Classifier3(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier3, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 8, 10)
        self.l1_act = torch.nn.ReLU()
        self.l1_pool = torch.nn.MaxPool1d(2)
        self.l2 = torch.nn.Conv1d(8, 16, 5)
        self.l2_act = torch.nn.ReLU()
        self.l2_pool = torch.nn.MaxPool1d(2)
        self.l3 = torch.nn.Conv1d(16, 32, 5, stride=2)
        self.l3_act = torch.nn.ReLU()
        self.l3_pool = torch.nn.MaxPool1d(2)
        self.l4 = torch.nn.Conv1d(32, 64, 5, stride=2)
        self.l4_act = torch.nn.ReLU()
        self.l4_pool = torch.nn.MaxPool1d(2)
        self.l5 = torch.nn.Linear(896, 224)
        self.l5_act = torch.nn.ReLU()
        self.l6 = torch.nn.Linear(224, 10)
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
        x3_pool = self.l3_pool(self.x3_act)
        x4 = self.l4(x3_pool)
        self.x4_act = self.l4_act(x4)
        x4_pool = self.l4_pool(self.x4_act)
        x4_reshape = x4_pool.flatten(1)
        x5 = self.l5(x4_reshape)
        self.x5_act = self.l5_act(x5)
        x6 = self.l6(self.x5_act)
        output = self.out(x6)
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
    for weight_decay in [0.0, 0.1, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print('weight_decay: ', str(weight_decay))
        for lr in [.0002, .0001, .00005, .0005, .0008, .001]:
            print('lr: ' + str(lr))

            # Initialise output files
            results = pd.DataFrame(columns=['1797', '1772', '1750', '1730'])
            results = results.append(pd.DataFrame(index=['1797', '1772', '1750', '1730']))
            loss_plots = {'1797': [], '1772': [], '1750': [], '1730': [],
                          '1797_val': [], '1772_val': [], '1750_val': [], '1730_val': []}

            for rpm in ['1797', '1772', '1750', '1730']:
                print('source rpm', str(rpm))

                # Define what device to run on
                if torch.cuda.is_available():
                    device = 'cuda:1'
                else:
                    device = 'cpu'

                # Initialize source dataset and split it into training and testing subsets
                sample_length = 1000
                dataset = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm], normalise=True)
                dataset_train_s, dataset_test_s = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)),
                                                                                          len(dataset) -
                                                                                          int(0.9 * len(dataset))],
                                                                                generator=torch.Generator().manual_seed(
                                                                                  42))
                sampler = torch.utils.data.WeightedRandomSampler(find_sampling_weights(dataset_train_s, 10),
                                                                 len(dataset_train_s))
                loader_train_s = DataLoader(dataset_train_s, batch_size=1000, shuffle=False, num_workers=1, sampler=sampler)
                loader_test_s = DataLoader(dataset_test_s, batch_size=1, shuffle=False, num_workers=1)

                # Initialise model and optimizer
                model = Classifier3(sample_length).to(device)
                model.train()
                # weight_path = "../../models/CWRU/sup_only_ext_final_" + rpm + "rpms.pt"
                weight_path = None
                if weight_path is not None:
                    model.load_state_dict(torch.load(weight_path))

                loss_function = torch.nn.CrossEntropyLoss()
                old_acc = 0
                dcounter = 0
                #lr = .0001
                #weight_decay = .4
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
                        #print(loss)
                        optimizer.step()
                        N += 1
                        # For debugging
                        if N % 50 == 0 and N != 0:
                            with torch.no_grad():
                                new_acc = evaluate_model(model, loader_test_s)
                                print('testing accuracy after ', N, 'iterations: ', round(new_acc, 4))
                                loss_plots[rpm].append(loss.detach().cpu())
                                loss_plots[rpm + '_val'].append(new_acc)
                                if new_acc >= old_acc:
                                    best_model = model.state_dict()
                                    torch.save(model.state_dict(),
                                               "../../models/CWRU/sup_only_final_ext2_" + rpm + "rpms.pt")
                                    old_acc = new_acc
                                    dcounter = 0
                                else:
                                    dcounter += 1
                                # lower the learning rate and reset the model if for 500 steps no improvement was
                                # achieved
                                if dcounter > 5:
                                    model.load_state_dict(torch.load("../../models/CWRU/sup_only_final_ext2_" + rpm +
                                                                     "rpms.pt"))
                                    model.train()
                                    dcounter = 0
                                    optimizer = torch.optim.SGD(model.parameters(), 0.7 * optimizer.param_groups[0]['lr'],
                                                                weight_decay=weight_decay)
                                    print('model reset to best perormance (', old_acc, ') and learning rate is now',
                                          optimizer.param_groups[0]['lr'])

                # evaluate the trained model
                for rpm_target in ['1797', '1772', '1750', '1730']:
                    model.eval()
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
                        features_s = model.x4_act
                        feat_dict_s = {'fts': features_s.cpu().numpy(), 'labels': y_s.unsqueeze(1).cpu().numpy()}
                        savemat('../../data/CWRU/deep_features/src2_' + rpm + '.mat',
                                feat_dict_s)

                    for sample in dataloader_t:
                        X_t = sample['data'].to(device)
                        y_t = sample['gt'].to(device)
                        model.forward(X_t)
                        features_t = model.x4_act
                        feat_dict_t = {'fts': features_t.cpu().numpy(), 'labels': y_t.unsqueeze(1).cpu().numpy()}
                        savemat('../../data/CWRU/deep_features/src2_' + rpm + '_tgt_' + rpm_target + '.mat',
                                feat_dict_t)

            results.to_csv('../eval/results/CWRU/' + 'sup_only3' + rpm + '_lr' + str(lr) + '_epochs' + str(num_epochs) +
                           '_weightdecay' + str(weight_decay) + '.csv', ';')

            fig, axarr = plt.subplots(2, 2)
            axarr[0, 0].plot(loss_plots['1797'])
            axarr[0, 0].plot(loss_plots['1797_val'])
            axarr[0, 1].plot(loss_plots['1772'])
            axarr[0, 1].plot(loss_plots['1772_val'])
            axarr[1, 0].plot(loss_plots['1750'])
            axarr[1, 0].plot(loss_plots['1750_val'])
            axarr[1, 1].plot(loss_plots['1730'])
            axarr[1, 1].plot(loss_plots['1730_val'])
            fig.savefig('../eval/results/CWRU/' + 'sup_only3' + '_lr' + str(lr) + '_weightdecay' + str(weight_decay) + '.png')
