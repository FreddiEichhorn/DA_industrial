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


def main():
    methods = {'1NN': None,
               'SA + 1NN': SA.SA(),
               #'SA + 5NN': SA.SA(clf=KNeighborsClassifier(5)),
               #'SA + SVM': SA.SA(clf=svm.SVC(gamma=2, C=1)),
               #'SA + DT': SA.SA(clf=DecisionTreeClassifier(max_depth=5)),
               #'PCA_src + 1NN': SA.PCASource(),
               #'PCA_src + SVM': SA.PCASource(clf=svm.SVC(gamma=2, C=1)),
               #'PCA_src + DT': SA.PCASource(clf=DecisionTreeClassifier(max_depth=5)),
               #'PCA_tgt + 1NN': SA.PCATarget(),
               #'PCA_tgt + SVM': SA.PCATarget(clf=svm.SVC(gamma=2, C=1)),
               #'PCA_tgt + DT': SA.PCATarget(clf=DecisionTreeClassifier(max_depth=5)),
               'GFK + 1NN': GFK.GFK(),
               #'GFK + 5NN': GFK.GFK(clf=KNeighborsClassifier(5)),
               #'GFK + SVM': GFK.GFK(clf=svm.SVC(gamma=2, C=1)),
               #'GFK + DT': GFK.GFK(clf=DecisionTreeClassifier(max_depth=5)),
               'JGSA + 1NN': JGSA.JGSA(),
               'JGSA + 5NN': JGSA.JGSA(clf=KNeighborsClassifier(5)),
               #'JGSA + SVM': JGSA.JGSA(clf=svm.SVC(gamma=2, C=1)),
               #'JGSA + DT': JGSA.JGSA(clf=DecisionTreeClassifier(max_depth=5)),
               'MEDA + 1NN': MEDA.MEDA(),
               'MEDA + SVM': MEDA.MEDA(clf=svm.SVC(gamma='scale', C=2), options={'rho': 1.0, 'eta': .1, 'gamma': 1,
                                                                                 'lamb': 8, 'mu': .7}),
               'MEDA + MLP': MEDA.MEDA(clf=MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15, ), random_state=1,
                                                         solver='sgd', max_iter=1000), options={'rho': 1.0, 'eta': .1,
                                                                                               'gamma': 1, 'lamb': 8,
                                                                                               'mu': .7})
               }
    results = pd.DataFrame(columns=methods.keys())
    normalize_dataset = False
    seed = 51
    normalize_sep = False
    balance = False
    use_all_target = True
    use_batch2 = False

    dataset_s1 = chemical_loader.ChemicalLoader(1, train=True, normalise=normalize_dataset, balance=balance, seed=seed)
    dataset_s2 = chemical_loader.ChemicalLoader(2, train=True, normalise=normalize_dataset, balance=balance, seed=seed)

    if use_batch2:
        data_s = np.vstack((dataset_s1.data, dataset_s2.data))
        gt_s = np.hstack((dataset_s1.gt, dataset_s2.gt))
        first_tgt_domain = 2
    else:
        data_s = dataset_s1.data
        gt_s = dataset_s1.gt
        first_tgt_domain = 1

    data_t_train = np.zeros((0, 128))

    for tgt in range(1, 11):
        dataset_t_train = chemical_loader.ChemicalLoader(tgt, train=True, normalise=normalize_dataset, balance=balance,
                                                         seed=seed)
        dataset_t_eval = chemical_loader.ChemicalLoader(tgt, train=False, normalise=normalize_dataset, balance=balance,
                                                        seed=seed)

        if use_all_target:
            data_t_train = np.vstack((data_t_train, dataset_t_train.data))
        else:
            data_t_train = dataset_t_train.data
        data_t_eval = dataset_t_eval.data

        results = results.append(pd.DataFrame(index=['batch' + str(tgt)]))
        for method_name in methods:
            if methods[method_name] is None:
                # TODO: Normalise for this
                clf = KNeighborsClassifier(1)
                clf.fit(data_s, gt_s)
                predict_t = clf.predict(dataset_t_eval.data)

            elif tgt > first_tgt_domain:

                if normalize_sep:
                    src_norm = methods[method_name].normalise_features({'fts': data_s})
                    tgt_norm_train = methods[method_name].normalise_features({'fts': data_t_train.data})
                    tgt_norm_eval = methods[method_name].normalise_features({'fts': data_t_eval.data})
                else:
                    all_normalised = methods[method_name].normalise_features({'fts': np.vstack((data_s,
                                                                                                data_t_eval.data,
                                                                                                data_t_train.data))})
                    src_norm = all_normalised[:data_s.shape[0]]
                    tgt_norm_eval = all_normalised[data_s.shape[0]:(data_s.shape[0] + dataset_t_eval.data.shape[0])]
                    tgt_norm_train = all_normalised[(data_s.shape[0] + dataset_t_eval.data.shape[0]):]

                methods[method_name].fit(src_norm, np.expand_dims(gt_s, 1), tgt_norm_train)
                predict_t = methods[method_name].inference(tgt_norm_eval)

            acc = (predict_t == dataset_t_eval.gt).sum() / len(dataset_t_eval)
            print('batch1+2 --> batch' + str(tgt) + ' using ' + method_name + ' accuracy: ' + str(round(acc, 4)))
            results[method_name]['batch' + str(tgt)] = round(acc, 4)

    results.to_csv('../eval/results/chemical/classical' + '_normalise' * normalize_dataset + '_normalise_sep'
                   * normalize_sep + '_balance' * balance + '_alltarget' * use_all_target + '.csv', ';')

    # plot the results
    for method_name in methods:
        plt.plot(results[method_name], label=method_name)
    plt.legend()
    plt.axis([0, 10, 0, 1])
    plt.savefig('../eval/results/chemical/classical' + '_normalise' * normalize_dataset + '_normalise_sep' *
                normalize_sep + '_balance' * balance + '_alltarget' * use_all_target + '.png')


if __name__ == "__main__":
    main()
