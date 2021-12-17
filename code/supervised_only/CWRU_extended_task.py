import torch
from torch.utils.data import DataLoader
import CWRU_loader_extended_task
from scipy.io import savemat
import datetime
import pandas as pd


class Classifier(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 16, 3)
        self.l1_act = torch.nn.ReLU()
        self.l1_dropout = torch.nn.Dropout()
        self.l2 = torch.nn.Conv1d(16, 1, 3)
        self.l2_act = torch.nn.ReLU()
        self.l2_dropout = torch.nn.Dropout()
        self.l3 = torch.nn.Linear(sample_length-4, 256)
        self.l3_act = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(256, 128)
        self.l4_act = torch.nn.ReLU()
        self.l5 = torch.nn.Linear(128, 6)
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
        output = self.out(x5)
        return output.squeeze(1)


if __name__ == "__main__":
    results = pd.DataFrame(columns=['1797', '1772', '1750', '1730'])
    for rpm in ['1797', '1772', '1750', '1730']:
        sample_length = 1000
        dataset = CWRU_loader_extended_task.CWRU(sample_length, rpms=[rpm])
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        sampler = torch.utils.data.WeightedRandomSampler([1.0] * int(dataset.healthy_length / sample_length)
                                                         + [dataset.healthy_length / dataset.ball_fault_length] *
                                                         int(dataset.ball_fault_length / sample_length) +
                                                         [dataset.healthy_length / dataset.ir_fault_length] *
                                                         int(dataset.ir_fault_length / sample_length) +
                                                         [dataset.healthy_length / dataset.or3_fault_length] *
                                                         int(dataset.or3_fault_length / sample_length) +
                                                         [dataset.healthy_length / dataset.or6_fault_length] *
                                                         int(dataset.or6_fault_length / sample_length) +
                                                         [dataset.healthy_length / dataset.or12_fault_length] *
                                                         int(dataset.or12_fault_length / sample_length),
                                                         len(dataset))
        dataloader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=1, sampler=sampler)
        model = Classifier(sample_length).to(device)
        model.train()
        # weight_path = "../../models/CWRU/sup_only_ext_final_" + rpm + "rpms.pt"
        weight_path = None
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path))

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        N = 0
        num_epochs = 800

        for _ in range(num_epochs):
            for sample in enumerate(dataloader):
                data = sample[1]["data"].to(device)
                gt = sample[1]["gt"].to(device)
                output = model.forward(data)
                loss = loss_function(output, gt)
                loss.backward()
                print(loss)
                optimizer.step()
                N += 1
                if N % 1000 == 0:
                    torch.save(model.state_dict(), "../../models/CWRU/sup_only_final_ext" +
                               datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pt")

        torch.save(model.state_dict(), "../../models/CWRU/sup_only_final_ext_" + rpm + "rpms.pt")

        # evaluate the trained model
        for rpm_target in ['1797', '1772', '1750', '1730']:
            results = results.append(pd.DataFrame(index=[rpm_target]))
            model.eval()
            dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[rpm_target])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            N = 0
            M = 0

            for sample in enumerate(dataloader):
                output = model.forward(sample[1]["data"].to(device))
                gt = sample[1]["gt"].to(device)
                prediction = torch.argmax(output)
                M += 1

                if torch.argmax(output) == gt:
                    N += 1

            results[rpm][rpm_target] = round(N / M, 4)

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
                savemat('../../data/CWRU/deep_features/src_' + rpm + '.mat',
                        feat_dict_s)

            for sample in dataloader_t:
                X_t = sample['data'].to(device)
                y_t = sample['gt'].to(device)
                model.forward(X_t)
                features_t = model.x4_act
                feat_dict_t = {'fts': features_t.cpu().numpy(), 'labels': y_t.unsqueeze(1).cpu().numpy()}
                savemat('../../data/CWRU/deep_features/src_' + rpm + '_tgt_' + rpm_target + '.mat',
                        feat_dict_t)

    results.to_csv('../eval/results/CWRU/' + 'sup_only' + '.csv', ';')
