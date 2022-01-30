"""Idea: in chemical.py we try to fit a classifier in the following way: We have labels for batch 1 and batch 2 (source
domain) and one other out of batch 3..10 for which there are no labels (target domain). The domain shift of this dataset
are sensor drifts over time. This way the adaptation performance decreases the further the target domain is from the
source domain. The reason for this could be limited expressability of the methods. In this script we treat the batch
from before using it's pseudo labels as the source domain."""

import sys
sys.path.append('../supervised_only')
import chemical_loader

sys.path.append('../JGSA')
import JGSA
sys.path.append('../MEDA')
import GFK
import MEDA
sys.path.append('../SA')
import SA

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    methods = {'1NN': None,
               'SA + 1NN': SA.SA(),
               'GFK + 1NN': GFK.GFK(),
               'JGSA + 1NN': JGSA.JGSA(),
               'MEDA + 1NN': MEDA.MEDA()
               }

    results = pd.DataFrame(columns=methods.keys())
    normalize_dataset = False
    normalize_sep = False
    seed = 42
    balance = True

    dataset_s1 = chemical_loader.ChemicalLoader(1, train=True, normalise=normalize_dataset, balance=balance, seed=seed)
    dataset_s2 = chemical_loader.ChemicalLoader(2, train=True, normalise=normalize_dataset, balance=balance, seed=seed)
    data_s = dict.fromkeys(methods.keys(), np.vstack((dataset_s1.data, dataset_s2.data)))
    gt_s = dict.fromkeys(methods.keys(), np.hstack((dataset_s1.gt, dataset_s2.gt)))

    for tgt in range(1, 11):
        dataset_t_train = chemical_loader.ChemicalLoader(tgt, train=True, normalise=normalize_dataset, balance=balance,
                                                         seed=seed)
        dataset_t_eval = chemical_loader.ChemicalLoader(tgt, train=False, normalise=normalize_dataset, balance=balance,
                                                        seed=seed)

        results = results.append(pd.DataFrame(index=['batch1+2 --> batch' + str(tgt)]))

        for method_name in methods:
            if methods[method_name] is None:
                if tgt == 1:
                    clf = KNeighborsClassifier(1)
                    clf.fit(data_s[method_name], gt_s[method_name])
                predict_t = clf.predict(dataset_t_eval.data)

            elif tgt > 2:

                if normalize_sep:
                    src_norm = methods[method_name].normalise_features({'fts': data_s[method_name]})
                    tgt_norm_eval = methods[method_name].normalise_features({'fts': dataset_t_eval.data})
                    tgt_norm_train = methods[method_name].normalise_features({'fts': dataset_t_eval.train})
                else:
                    all_normalised = methods[method_name].normalise_features({'fts': np.vstack((data_s[method_name],
                                                                                                dataset_t_eval.data,
                                                                                                dataset_t_train.data))})
                    src_norm = all_normalised[:data_s[method_name].shape[0]]
                    tgt_norm_eval = all_normalised[data_s[method_name].shape[0]:(data_s[method_name].shape[0] +
                                                                                 dataset_t_eval.data.shape[0])]
                    tgt_norm_train = all_normalised[(data_s[method_name].shape[0] + dataset_t_eval.data.shape[0]):]

                # BIG WARNING: Methods see unnormalised training data
                methods[method_name].fit(src_norm, np.expand_dims(gt_s[method_name], 1), tgt_norm_train)
                predict_t = methods[method_name].inference(tgt_norm_eval)

                data_s[method_name] = dataset_t_train.data
                gt_s[method_name] = methods[method_name].inference(tgt_norm_train)

            acc = (predict_t == dataset_t_eval.gt).sum() / len(dataset_t_eval)
            print('batch1+2 --> batch' + str(tgt) + ' using ' + method_name + ' accuracy: ' + str(round(acc, 4)))
            results[method_name]['batch1+2 --> batch' + str(tgt)] = round(acc, 4)

    results.to_csv('../eval/results/chemical/classical_sequential' + '_normalise' * normalize_dataset + '_normalise_sep'
                   * normalize_sep + '_balance' * balance + '.csv', ';')


if __name__ == "__main__":
    main()