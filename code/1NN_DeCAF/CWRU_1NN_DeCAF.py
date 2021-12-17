import torch
import sys
sys.path.append("../")
from supervised_only import CWRU_loader
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import savemat
import datetime


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
        # self.l4 = torch.nn.Linear(256, 128)
        # self.l4_act = torch.nn.ReLU()
        # self.l5 = torch.nn.Linear(128, 3)
        # self.out = torch.nn.Softmax(2)

    def forward(self, input):
        x1 = self.l1(input)
        x1_act = self.l1_act(x1)
        x1_dropout = self.l1_dropout(x1_act)
        x2 = self.l2(x1_dropout)
        x2_act = self.l2_act(x2)
        x2_dropout = self.l2_dropout(x2_act)
        x3 = self.l3(x2_dropout)
        x3_act = self.l3_act(x3)
        # x4 = self.l4(x3_act)
        # x4_act = self.l4_act(x4)
        # x5 = self.l5(x4_act)
        # output = self.out(x5)
        return x2_act.squeeze(1)


if __name__ == '__main__':
    for rpm_src in ['1797', '1772', '1750', '1730']:
        for rpm_tgt in ['1797', '1772', '1750', '1730']:

            sample_length = 1000
            model = Classifier(sample_length)
            model.load_state_dict(torch.load('../../models/CWRU/sup_only_final_' + rpm_src + 'rpms.pt'), strict=False)
            model.eval()

            dataset_s = CWRU_loader.CWRU(sample_length, rpms=[rpm_src])
            dataloader_s = DataLoader(dataset_s, batch_size=len(dataset_s), shuffle=False, num_workers=1)

            dataset_t = CWRU_loader.CWRU(sample_length, rpms=[rpm_tgt])
            dataloader_t = DataLoader(dataset_t, batch_size=len(dataset_t), shuffle=False, num_workers=1)

            for sample in dataloader_s:  # We need all of the data in the dataset for this
                X_s = sample['data']
                y_s = sample['gt']

            for sample in dataloader_t:
                X_t = sample['data']
                y_t = sample['gt']

            # Learn a 1NN classifier
            features_s = model.forward(X_s)
            features_t = model.forward(X_t)
            clf = KNeighborsClassifier(1)
            clf.fit(features_s.flatten(1).detach(), y_s)
            predict_t = clf.predict(features_t.flatten(1).detach())
            acc = (torch.Tensor(predict_t) == y_t).sum() / len(y_t)
            print(f"Raw Accuracy 1NN classifier " + rpm_src + "-->" + rpm_tgt + ": ", acc)

            with torch.no_grad():
                feat_dict_s = {'fts': features_s.numpy(), 'labels': y_s.unsqueeze(1).numpy()}
                savemat('../../data/CWRU/deep_features/src_' + rpm_src + '.mat',
                        feat_dict_s)

                feat_dict_t = {'fts': features_t.numpy(), 'labels': y_t.unsqueeze(1).numpy()}
                savemat('../../data/CWRU/deep_features/src_' + rpm_src + '_tgt_' + rpm_tgt + '.mat',
                        feat_dict_t)
