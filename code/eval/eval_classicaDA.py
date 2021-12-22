import sys
sys.path.append('../JGSA')
import JGSA
sys.path.append('../MEDA')
import MEDA
import GFK
sys.path.append('../SA')
import SA
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import datetime
import numpy as np
import torch

datapath = '../../data/SURF/'
#datapath = '../../data/CWRU/deep_features/'
normalize = False
use_train_test_split = True
source_domains = ['Caltech10_SURF_L10.mat', 'amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat']
target_domains = ['Caltech10_SURF_L10.mat', 'amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat']
#source_domains = ['src_1730.mat', 'src_1750.mat', 'src_1772.mat', 'src_1797.mat']
#target_domains = ['src_1730_tgt_1750.mat', 'src_1730_tgt_1772.mat', 'src_1730_tgt_1797.mat',
#                  'src_1750_tgt_1730.mat', 'src_1750_tgt_1772.mat', 'src_1750_tgt_1797.mat',
#                  'src_1772_tgt_1730.mat', 'src_1772_tgt_1750.mat', 'src_1772_tgt_1797.mat',
#                  'src_1797_tgt_1730.mat', 'src_1797_tgt_1750.mat', 'src_1797_tgt_1772.mat']
methods = {'1NN': None,
           'SA': SA.SA(),
           'PCA_src': SA.PCASource(),
           'PCA_tgt': SA.PCATarget(),
           'GFK': GFK.GFK(),
           'JGSA': JGSA.JGSA(),
           'MEDA': MEDA.MEDA()
           }
results = pd.DataFrame(columns=methods.keys())


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer = torch.nn.Linear(128, 10)
        self.out = torch.nn.Softmax(2)

    def forward(self, data):
        x1 = self.layer(data)
        output = self.out(x1)
        return output


for source_domain in source_domains:
    for target_domain in target_domains:
        if source_domain == target_domain:
            continue

        #if not source_domain.split('.')[0] in target_domain:
        #    continue

        results = results.append(pd.DataFrame(index=[source_domain + '-->' + target_domain]))
        for method_name in methods:
            source_path = datapath + source_domain
            target_path = datapath + target_domain
            source_data = scipy.io.loadmat(source_path)
            target_data = scipy.io.loadmat(target_path)
            source_labels = source_data['labels']
            target_labels = target_data['labels']

            if np.min(source_labels) == 0:
                source_labels += 1
                target_labels += 1
            if methods[method_name] is not None:
                if normalize:
                    source_normalised = methods[method_name].normalise_features(source_data)
                    target_normalised = methods[method_name].normalise_features(target_data)
                else:
                    source_normalised = source_data['fts']
                    target_normalised = target_data['fts']

                if use_train_test_split:
                    choice_arr = np.random.choice(a=[False, True], size=target_normalised.shape[0], p=[.15, 1-.15])
                    eval_split = target_normalised[np.logical_not(choice_arr)]
                    target_normalised = target_normalised[choice_arr]
                    target_labels = target_labels[np.logical_not(choice_arr)]
                else:
                    eval_split = target_normalised

                methods[method_name].fit(source_normalised, source_labels, target_normalised)
                predict_t = methods[method_name].inference(eval_split)

            else:  # fit a 1NN classifier on raw measurements
                source_normalised = source_data['fts']
                target_normalised = target_data['fts']
                clf = KNeighborsClassifier(1)
                clf.fit(source_data['fts'], source_labels[:, 0])
                predict_t = clf.predict(target_data['fts'])

            acc = (predict_t == target_labels[:, 0]).sum() / len(target_labels)
            print(source_domain + '-->' + target_domain + ' using ' + method_name + ' accuracy: ' + str(acc))
            results[method_name][source_domain + '-->' + target_domain] = round(acc, 4)

results = results.append(pd.DataFrame(index=['Average']))
results[[False] * (len(results)-1) + [True]] = results.mean(0)

results.to_csv('results/eval_classicalDA_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + '.csv', sep=';')
