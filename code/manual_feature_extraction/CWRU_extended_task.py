import sys
sys.path.append('../supervised_only')
import CWRU_loader_extended_task

sys.path.append('../JGSA')
import JGSA
sys.path.append('../MEDA')
import MEDA
import GFK
sys.path.append('../SA')
import SA

import pandas as pd
import numpy as np
import tsfresh
from sklearn.neighbors import KNeighborsClassifier
import os
from tsfresh.utilities.dataframe_functions import impute


def dataset2dataframe(dataset):
    output = pd.DataFrame(columns=['id', 'time', 'acc1', 'acc2', 'acc3'])
    gt = pd.Series(dtype='int8')
    idx = 0
    for sample in dataset:
        if sample is None:
            break
        h = pd.DataFrame(np.hstack((np.array([[idx]] * 1000), np.expand_dims(np.arange(1000), 1),
                                    sample['data'].numpy().T)), columns=['id', 'time', 'acc1', 'acc2', 'acc3'])
        output = output.append(h)
        gt = gt.append(pd.Series([sample['gt']], index=[idx]))
        idx += 1
    return output, gt


def main():

    methods = {'1NN': None,
               'SA': SA.SA(),
               'PCA_src': SA.PCASource(),
               'PCA_tgt': SA.PCATarget(),
               'GFK': GFK.GFK(),
               'JGSA': JGSA.JGSA(),
               'MEDA': MEDA.MEDA()
               }
    normalize = True
    partial = [0, 1, 4]

    # Initialise output files
    results = pd.DataFrame(columns=methods.keys())
    b = []
    for i in ['1730', '1750', '1772', '1797']:
        b += [i + '->' + j for j in ['1730', '1750', '1772', '1797']]
    results = results.append(pd.DataFrame(index=b))

    for src_domain in ['1730', '1750', '1772', '1797']:
        for tgt_domain in ['1730', '1750', '1772', '1797']:

            # source domain training dataset
            if os.path.isfile('../../data/CWRU/manual_features/' + src_domain + '_train.csv'):
                ts_train = pd.read_csv('../../data/CWRU/manual_features/' + src_domain + '_train.csv')
                lbl_train = pd.read_csv('../../data/CWRU/manual_features/' + src_domain + '_train_lbl.csv', header=None,
                                        index_col=0, squeeze=True)[0:]
            else:
                dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[src_domain], normalise=True, train=True)
                data, lbl_train = dataset2dataframe(dataset)
                ts_train = tsfresh.extract_features(data, column_id="id", column_sort="time")
                ts_train.to_csv('../../data/CWRU/manual_features/' + src_domain + '_train.csv')
                lbl_train.to_csv('../../data/CWRU/manual_features/' + src_domain + '_train_lbl.csv')
            impute(ts_train)
            ts_train = tsfresh.select_features(ts_train, lbl_train)

            # source domain evaluation dataset
            if os.path.isfile('../../data/CWRU/manual_features/' + src_domain + '_eval.csv'):
                ts_eval_s = pd.read_csv('../../data/CWRU/manual_features/' + src_domain + '_eval.csv')
                lbl_eval_s = pd.read_csv('../../data/CWRU/manual_features/' + src_domain + '_eval_lbl.csv', header=None,
                                         index_col=0, squeeze=True)[0:]
            else:
                dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[src_domain], normalise=True, train=False)
                data, lbl_eval_s = dataset2dataframe(dataset)
                ts_eval_s = tsfresh.extract_features(data, column_id="id", column_sort="time")
                ts_eval_s.to_csv('../../data/CWRU/manual_features/' + src_domain + '_eval.csv')
                lbl_eval_s.to_csv('../../data/CWRU/manual_features/' + src_domain + '_eval_lbl.csv')
            ts_eval_s = ts_eval_s[ts_train.keys()]
            impute(ts_eval_s)

            # target domain training dataset
            if os.path.isfile('../../data/CWRU/manual_features/' + tgt_domain + '_train.csv'):
                ts_train_t = pd.read_csv('../../data/CWRU/manual_features/' + tgt_domain + '_train.csv')
                lbl_train_t = pd.read_csv('../../data/CWRU/manual_features/' + tgt_domain + '_train_lbl.csv',
                                          header=None, index_col=0, squeeze=True)[0:]
            else:
                dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[tgt_domain], normalise=True, train=True)
                data, lbl_train_t = dataset2dataframe(dataset)
                ts_train_t = tsfresh.extract_features(data, column_id="id", column_sort="time")
                ts_train_t.to_csv('../../data/CWRU/manual_features/' + tgt_domain + '_train.csv')
                lbl_train_t.to_csv('../../data/CWRU/manual_features/' + tgt_domain + '_train_lbl.csv')
            ts_train_t = ts_train_t[ts_train.keys()]
            impute(ts_train_t)

            # target domain evaluation dataset
            if os.path.isfile('../../data/CWRU/manual_features/' + tgt_domain + '_eval.csv'):
                ts_eval_t = pd.read_csv('../../data/CWRU/manual_features/' + tgt_domain + '_eval.csv')
                lbl_eval_t = pd.read_csv('../../data/CWRU/manual_features/' + tgt_domain + '_eval_lbl.csv',
                                         header=None, index_col=0, squeeze=True)[0:]
            else:
                dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[tgt_domain], normalise=True, train=False)
                data, lbl_eval_t = dataset2dataframe(dataset)
                ts_eval_t = tsfresh.extract_features(data, column_id="id", column_sort="time")
                ts_eval_t.to_csv('../../data/CWRU/manual_features/' + tgt_domain + '_eval.csv')
                lbl_eval_t.to_csv('../../data/CWRU/manual_features/' + tgt_domain + '_eval_lbl.csv')
            ts_eval_t = ts_eval_t[ts_train.keys()]
            impute(ts_eval_t)

            for method in methods:
                if methods[method] is None or tgt_domain is src_domain:
                    # fit a 1NN classifier
                    clf = KNeighborsClassifier(1)
                    clf.fit(ts_train, lbl_train)
                    predictions_s = clf.predict(ts_eval_s)
                    accuracy_s = (predictions_s == lbl_eval_s).sum() / ts_eval_s.shape[0]
                    print('accuracy of 1NN classifier on source domain: ', round(accuracy_s, 4))

                    predictions_t = clf.predict(ts_eval_t)
                    accuracy_t = (predictions_t == lbl_eval_t).sum() / ts_eval_t.shape[0]
                    print('accuracy of 1NN clf on domain shift ', src_domain, '->', tgt_domain, ': ', accuracy_t)

                    if src_domain is tgt_domain:
                        results['1NN'][src_domain + '->' + tgt_domain] = accuracy_s
                        continue
                    else:
                        results['1NN'][src_domain + '->' + tgt_domain] = accuracy_t

                else:
                    if normalize:
                        source_normalised = methods[method].normalise_features({'fts': np.asarray(ts_train)})
                        target_normalised_train = methods[method].normalise_features({'fts': np.asarray(ts_train_t)})
                        target_normalised_eval = methods[method].normalise_features({'fts': np.asarray(ts_eval_t)})
                    else:
                        source_normalised = np.asarray(ts_train)
                        target_normalised_train = np.asarray(ts_train_t)
                        target_normalised_eval = np.asarray(ts_eval_t)

                    classes = np.array(partial * target_normalised_train.shape[0]).reshape(
                        target_normalised_train.shape[0], len(partial))
                    lbl_train_t = np.asarray(lbl_train_t)
                    target_normalised_train = target_normalised_train[np.any(np.expand_dims(lbl_train_t, 1) == classes,
                                                                             axis=1)]

                    methods[method].fit(source_normalised, np.asarray(lbl_train)[:, np.newaxis],
                                        target_normalised_train)
                    predictions_t = methods[method].inference(target_normalised_eval)
                    accuracy_t = (predictions_t == lbl_eval_t).sum() / ts_eval_t.shape[0]
                    results[method][src_domain + '->' + tgt_domain] = accuracy_t
                    print('accuracy of ' + method + ' on domain shift ', src_domain, '->', tgt_domain, ': ', accuracy_t)

    results.to_csv('../eval/results/CWRU/' + 'manual_features_' + 'normalize' * normalize + '_missing' +
                   str(10-len(partial)) + '_classes.csv', ';')


if __name__ == '__main__':
    main()
