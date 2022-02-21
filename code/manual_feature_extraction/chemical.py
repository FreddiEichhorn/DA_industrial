import sys

sys.path.append('../supervised_only')
import chemical_loader

sys.path.append('../JGSA')
import JGSA

sys.path.append('../MEDA')
import MEDA
import GFK

sys.path.append('../SA')
import SA

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main():
    methods = {'1NN': KNeighborsClassifier(1),
               #'SA + 1NN': SA.SA(),
               #'SA + 5NN': SA.SA(clf=KNeighborsClassifier(5)),
               #'SA + SVM': SA.SA(clf=svm.SVC(gamma=2, C=1)),
               #'SA + DT': SA.SA(clf=DecisionTreeClassifier(max_depth=5)),
               #'PCA_src + 1NN': SA.PCASource(options={'subspace_dim': 10}),
               #'PCA_src + SVM': SA.PCASource(clf=svm.SVC(gamma=2, C=1)),
               #'PCA_src + DT': SA.PCASource(clf=DecisionTreeClassifier(max_depth=5)),
               #'PCA_tgt + 1NN': SA.PCATarget(),
               #'PCA_tgt + SVM': SA.PCATarget(clf=svm.SVC(gamma=2, C=1)),
               #'PCA_tgt + DT': SA.PCATarget(clf=DecisionTreeClassifier(max_depth=5)),
               #'GFK + 1NN': GFK.GFK(),
               #'GFK + 5NN': GFK.GFK(clf=KNeighborsClassifier(5)),
               #'GFK + SVM': GFK.GFK(clf=svm.SVC(gamma=2, C=1)),
               #'GFK + DT': GFK.GFK(clf=DecisionTreeClassifier(max_depth=5)),
               #'JGSA + 1NN': JGSA.JGSA(),
               #'JGSA + 5NN': JGSA.JGSA(clf=KNeighborsClassifier(5)),
               #'JGSA + SVM': JGSA.JGSA(clf=svm.SVC(gamma='scale', C=0.1)),
               #'JGSA + DT': JGSA.JGSA(clf=DecisionTreeClassifier(max_depth=5)),
               #'MEDA + 1NN': MEDA.MEDA(options={'mu': .7, 't': 20, 'lamb': 0.1, 'gamma': 1.6, 'rho': 0, 'eta': .3}),
               'MEDA + SVM': MEDA.MEDA(clf=svm.SVC(gamma='auto', C=2),
                                       options={'mu': .7, 't': 20, 'lamb': 0.1, 'gamma': 1.7, 'rho': 0, 'eta': .3}),
               'MEDA + tune + SVM': MEDA.MEDA(clf=svm.SVC(gamma='auto', C=2),
                                              options={'mu': .7, 't': 20, 'lamb': 0.1, 'gamma': 1.7, 'rho': 1,
                                                       'eta': .3, 'num_neighbors': 3}),
               }

    results = pd.DataFrame(columns=methods.keys())

    train_test_split = False
    use_all_target = False
    balance = False

    dataset_s = chemical_loader.ChemicalLoader(1, train=None, normalise=False, balance=balance, seed=None)
    data_s = dataset_s.data
    gt_s = dataset_s.gt
    data_t_train = np.zeros((0, 128))

    # scaling for 1NN baseline method
    scaler = StandardScaler()
    scaler.fit(data_s)
    data_s_norm = scaler.transform(data_s)
    methods['1NN'].fit(data_s_norm, gt_s)

    for tgt in range(2, 11):
        dataset_t = chemical_loader.ChemicalLoader(tgt, train=True, normalise=False, balance=balance, seed=None)

        if train_test_split:
            if use_all_target:
                data_t_train = np.vstack((dataset_t.data_train, data_t_train))
            else:
                data_t_train = dataset_t.data_train
            data_t_test = dataset_t.data_test
            gt_t_test = dataset_t.gt_test
            gt_t_train = dataset_t.gt_train
        else:
            if use_all_target:
                data_t_train = np.vstack((dataset_t.data, data_t_train))
            else:
                data_t_train = dataset_t.data
            data_t_test = dataset_t.data
            gt_t_test = dataset_t.gt

        results = results.append(pd.DataFrame(index=['batch' + str(tgt)]))

        for method_name in methods.keys():
            if method_name == '1NN':
                scaler.fit(data_t_train)
                predictions = methods[method_name].predict(scaler.transform(data_t_test))
            else:
                # scaler.fit(data_t_train)
                data_s_norm2 = methods[method_name].normalise_features({'fts': data_s})
                # data_t_train_norm = scaler.transform(data_t_train)
                data_t_train_norm = methods[method_name].normalise_features({'fts': data_t_train})
                # data_t_test_norm = scaler.transform(data_t_test)
                data_t_test_norm = methods[method_name].apply_normalisation({'fts': data_t_test})
                methods[method_name].fit(data_s_norm2, np.expand_dims(gt_s, 1), data_t_train_norm)
                predictions = methods[method_name].inference(data_t_test_norm)

            acc = (predictions == gt_t_test).sum() / len(predictions)
            print('batch1 -> batch' + str(tgt) + ' using ' + method_name + ': ' + str(round(acc, 4)))

            results[method_name]['batch' + str(tgt)] = round(acc, 4)

    # Compute average performance
    results = results.append(pd.DataFrame(index=['Average']))
    for method_name in results.keys():
        avg = (results[method_name].sum(axis=0, skipna=True)) / 9
        results[method_name]['Average'] = avg
        print('Average of ' + method_name + ' ' + str(round(avg, 4)))

    results.to_csv('../eval/results/chemical/classical' + '.csv', ';')

    # plot the results
    for method_name in methods:
        plt.plot(results[method_name], label=method_name)
    plt.legend()
    plt.axis([0, 10, 0, 1])
    plt.savefig('../eval/results/chemical/classical' + '.png')


if __name__ == "__main__":
    main()
