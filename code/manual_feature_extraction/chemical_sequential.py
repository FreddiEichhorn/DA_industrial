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
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def balance_dataset(data, gt, seed=None):
    data = data.copy()
    gt = gt.copy()

    # Find class with most samples
    largest_class = 0
    for i in range(int(max(gt[:, 0]) + 1)):
        if (gt == i).sum() > largest_class:
            largest_class = (gt == i).sum()

    # oversample all classes smaller than largest
    data2 = np.zeros((0, data.shape[1]))
    gt_2 = np.zeros((0, 1))
    for cls in range(int(max(gt[:, 0]) + 1)):
        data_cls = data[gt[:, 0] == cls]

        while data_cls.shape[0] * 2 < largest_class:
            data_cls = np.vstack((data_cls, data_cls))

        p = (largest_class - data_cls.shape[0]) / data_cls.shape[0]
        if seed is not None:
            np.random.seed(seed)
        choice_arr = np.random.choice(a=[False, True], size=data_cls.shape[0], p=[1 - p, p])
        data_cls = np.vstack((data_cls, data_cls[choice_arr]))
        data2 = np.vstack((data2, data_cls))
        gt_2 = np.vstack((gt_2, np.ones((data_cls.shape[0], 1)) * cls))
    balanced_data = data2
    balanced_gt = gt_2
    return balanced_data, balanced_gt


def main():
    methods = {'1NN': None,
               #'SA + 1NN': SA.SA(),
               #'SA + 5NN': SA.SA(clf=KNeighborsClassifier(5)),
               #'SA + MLP': SA.SA(clf=MLPClassifier(alpha=1e-05, hidden_layer_sizes=(100, 15), random_state=1,
               #                                    solver='sgd', max_iter=600)),
               #'SA + SVM': SA.SA(clf=svm.SVC(gamma=2, C=1)),
               #'SA + DT': SA.SA(clf=DecisionTreeClassifier(max_depth=5)),
               #'PCA_src + 1NN': SA.PCASource(),
               #'PCA_src + SVM': SA.PCASource(clf=svm.SVC(gamma=2, C=1)),
               #'PCA_src + DT': SA.PCASource(clf=DecisionTreeClassifier(max_depth=5)),
               #'PCA_tgt + 1NN': SA.PCATarget(),
               #'PCA_tgt + SVM': SA.PCATarget(clf=svm.SVC(gamma=2, C=1)),
               #'PCA_tgt + DT': SA.PCATarget(clf=DecisionTreeClassifier(max_depth=5)),
               #'GFK + 1NN': GFK.GFK(),
               #'GFK + 5NN': GFK.GFK(clf=KNeighborsClassifier(5)),
               #'GFK + MLP': GFK.GFK(clf=MLPClassifier(alpha=1e-05, hidden_layer_sizes=(100, 15), random_state=1,
               #                                       solver='sgd', max_iter=600)),
               #'GFK + SVM': GFK.GFK(clf=svm.SVC(gamma=2, C=1)),
               #'GFK + SVM2': GFK.GFK(clf=svm.SVC(gamma=2, C=1)),
               #'GFK + DT': GFK.GFK(clf=DecisionTreeClassifier(max_depth=5)),
               #'JGSA + 1NN': JGSA.JGSA(),
               #'JGSA + 5NN': JGSA.JGSA(clf=KNeighborsClassifier(5)),
               #'JGSA + MLP': JGSA.JGSA(clf=MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=1,
                                                         #solver='sgd', max_iter=400)),
               #'JGSA + SVM': JGSA.JGSA(clf=svm.SVC(gamma=2, C=1)),
               #'JGSA + SVM2': JGSA.JGSA(clf=svm.SVC(gamma=2, C=1)),
               #'JGSA + DT': JGSA.JGSA(clf=DecisionTreeClassifier(max_depth=5)),
               'MEDA + 1NN': MEDA.MEDA(options={'mu': .7, 't': 20, 'lamb': 0, 'gamma': 1.6, 'rho': 0}),
               #'MEDA + 5NN': MEDA.MEDA(clf=KNeighborsClassifier(5)),
               #'MEDA + SVM': MEDA.MEDA(clf=svm.SVC(gamma='scale', C=2), options={'mu': .7, 't': 20, 'lamb': 2,
               #                                                                  'gamma': 1.6, 'rho': 0}),
               }

    results = pd.DataFrame(columns=methods.keys())
    last_x = 15000

    train_test_split = False

    dataset_s = chemical_loader.ChemicalLoader(1, train=None, normalise=False, balance=False, seed=None)

    # scaling for 1NN baseline method
    scaler = StandardScaler()
    scaler.fit(dataset_s.data)

    data_s = dict.fromkeys(methods.keys(), dataset_s.data)
    gt_s = dict.fromkeys(methods.keys(), dataset_s.gt)
    first_tgt_domain = 1

    for tgt in range(1, 11):
        dataset_t = chemical_loader.ChemicalLoader(tgt, train=True, normalise=False, balance=False, seed=None)

        results = results.append(pd.DataFrame(index=['batch' + str(tgt)]))

        for method_name in methods:

            if train_test_split:
                data_t_train = dataset_t.data_train
                data_t_test = dataset_t.data_test
                gt_t_test = dataset_t.gt_test
                gt_t_train = dataset_t.gt_train
            else:
                data_t_train = dataset_t.data
                data_t_test = dataset_t.data
                gt_t_test = dataset_t.gt

            if methods[method_name] is None:

                clf = KNeighborsClassifier(1)
                scaler.fit(data_s[method_name])
                clf.fit(scaler.transform(data_s[method_name]), gt_s[method_name])
                scaler.fit(data_t_train)
                data_t_test_norm = scaler.transform(data_t_test)

                predictions = clf.predict(data_t_test_norm)

            elif tgt > first_tgt_domain:
                scaler.fit(data_s[method_name])
                data_s_norm = scaler.transform(data_s[method_name])
                scaler.fit(data_t_train)
                data_t_train_norm = scaler.transform(data_t_train)
                data_t_test_norm = scaler.transform(data_t_test)
                methods[method_name].fit(data_s_norm, np.expand_dims(gt_s[method_name], 1), data_t_train_norm)
                predictions = methods[method_name].inference(data_t_test_norm)

                data_s[method_name] = np.vstack((data_t_train, data_s[method_name]))
                gt_s[method_name] = np.hstack((predictions, gt_s[method_name]))

                if len(data_s[method_name]) > last_x:
                    data_s[method_name] = data_s[method_name][:last_x]
                    gt_s[method_name] = gt_s[method_name][:last_x]

            acc = (predictions == gt_t_test).sum() / len(gt_t_test)
            print('batch1+2 --> batch' + str(tgt) + ' using ' + method_name + ' accuracy: ' + str(round(acc, 4)))
            results[method_name]['batch' + str(tgt)] = round(acc, 4)

    # Compute average performance
    results = results.append(pd.DataFrame(index=['Average']))
    for method_name in results.keys():
        avg = (results[method_name].sum(axis=0, skipna=True) - 1) / 9
        results[method_name]['Average'] = avg
        print('Average of ' + method_name + ' ' + str(round(avg, 4)))

    results.to_csv('../eval/results/chemical/classical_sequential' + '.csv', ';')

    # plot the results
    for method_name in methods:
        plt.plot(results[method_name], label=method_name)
    plt.legend()
    plt.axis([0, 10, 0, 1])
    plt.savefig('../eval/results/chemical/classical_sequential' + '.png')


if __name__ == "__main__":
    main()
