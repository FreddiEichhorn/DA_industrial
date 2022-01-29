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


def main():
    methods = {'1NN': None,
               'SA': SA.SA(),
               'PCA_src': SA.PCASource(),
               'PCA_tgt': SA.PCATarget(),
               'GFK': GFK.GFK(),
               'JGSA': JGSA.JGSA(),
               'MEDA': MEDA.MEDA()
               }
    results = pd.DataFrame(columns=methods.keys())
    normalize_dataset = False
    normalize_sep = False

    dataset_s1 = chemical_loader.ChemicalLoader(1, train=True, normalise=normalize_dataset)
    dataset_s2 = chemical_loader.ChemicalLoader(2, train=True, normalise=normalize_dataset)
    data_s = np.vstack((dataset_s1.data, dataset_s2.data))
    gt_s = np.hstack((dataset_s1.gt, dataset_s2.gt))

    for tgt in range(1, 11):
        dataset_t_train = chemical_loader.ChemicalLoader(tgt, train=True, normalise=normalize_dataset)
        dataset_t_eval = chemical_loader.ChemicalLoader(tgt, train=False, normalise=normalize_dataset)

        results = results.append(pd.DataFrame(index=['batch1+2 --> batch' + str(tgt)]))
        for method_name in methods:
            if methods[method_name] is None:

                clf = KNeighborsClassifier(1)
                clf.fit(data_s, gt_s)
                predict_t = clf.predict(dataset_t_eval.data)

            elif tgt > 2:

                if normalize_sep:
                    source_normalised = methods[method_name].normalise_features({'fts': data_s})
                    target_normalised = methods[method_name].normalise_features({'fts': dataset_t_eval.data})
                else:
                    all_normalised = methods[method_name].normalise_features({'fts': np.vstack((data_s,
                                                                                                dataset_t_eval.data))})
                    source_normalised = all_normalised[:data_s.shape[0]]
                    target_normalised = all_normalised[data_s.shape[0]:]

                methods[method_name].fit(source_normalised, np.expand_dims(gt_s, 1), dataset_t_train.data)
                predict_t = methods[method_name].inference(target_normalised)

            acc = (predict_t == dataset_t_eval.gt).sum() / len(dataset_t_eval)
            print('batch1+2 --> batch' + str(tgt) + ' using ' + method_name + ' accuracy: ' + str(round(acc, 4)))
            results[method_name]['batch1+2 --> batch' + str(tgt)] = round(acc, 4)

    results.to_csv('../eval/results/chemical/classical' + '_normalise' * normalize_dataset + '_normalise_sep'
                   * normalize_sep + '.csv', ';')


if __name__ == "__main__":
    main()
