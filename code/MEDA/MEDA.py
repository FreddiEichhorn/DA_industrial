# encoding=utf-8
"""
    Created on 10:40 2018/11/14 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

import GFK


class MEDA:
    """Implementation of Manifold Embedded Distribution alignment, based on the official implementation"""
    def __init__(self, options=None, clf=None):
        if options is None:
            options = {}
        if 'kernel_type' in options.keys():
            self.kernel_type = options['kernel_type']
        else:
            self.kernel_type = 'rbf'

        if 'gamma' in options.keys():
            self.gamma = options['gamma']
        else:
            self.gamma = 1

        if 'T' in options.keys():
            self.t = options['T']
        else:
            self.t = 10

        if 'rho' in options.keys():
            self.rho = options['rho']
        else:
            self.rho = 1.0

        if 'eta' in options.keys():
            self.eta = options['eta']
        else:
            self.eta = 0.1

        if 'lamb' in options.keys():
            self.lamb = options['lamb']
        else:
            self.lamb = 10

        if 'mu' in options.keys():
            self.mu = options['mu']
            self.est_mu = False
        else:
            self.mu = .7
            self.est_mu = True

        # These are estimated using the fit method
        self.sq_g = None
        self.x = None
        self.beta = None
        self.clf = clf
        self.min_classes = None
        self.mean = None
        self.std = None

    @staticmethod
    def proxy_a_distance(source_x, target_x):
        """
        Compute the Proxy-A-Distance of a source/target representation
        """
        nb_source = np.shape(source_x)[0]
        nb_target = np.shape(target_x)[0]
        train_x = np.vstack((source_x, target_x))
        train_y = np.hstack((np.zeros(nb_source, dtype=int), np.ones(nb_target, dtype=int)))
        clf = svm.LinearSVC(random_state=0)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(train_x)
        error = metrics.mean_absolute_error(train_y, y_pred)
        dist = 2 * (1 - 2 * error)
        return dist

    def estimate_mu(self, _x1, _y1, _x2, _y2):
        adist_m = self.proxy_a_distance(_x1, _x2)
        C = len(np.unique(_y1))
        epsilon = 1e-3
        list_adist_c = []
        for i in range(1, C + 1):
            ind_i, ind_j = np.where(_y1 == i), np.where(_y2 == i)
            xsi = _x1[ind_i[0], :]
            xtj = _x2[ind_j[0], :]
            if len(xsi) == 0 or len(xtj) == 0:
                list_adist_c.append(2.0)
            else:
                adist_i = self.proxy_a_distance(xsi, xtj)
                list_adist_c.append(adist_i)
        adist_c = sum(list_adist_c) / C
        self.mu = adist_c / (adist_c + adist_m)
        if self.mu > 1:
            self.mu = 1
        if self.mu < epsilon:
            self.mu = 0
        return self.mu

    @staticmethod
    def kernel(ker, X, X2, gamma):
        if not ker or ker == 'primal':
            return X
        elif ker == 'linear':
            if not X2:
                K = np.dot(X.T, X)
            else:
                K = np.dot(X.T, X2)
        elif ker == 'rbf':
            n1sq = np.sum(X ** 2, axis=0)
            n1 = X.shape[1]
            if not X2:
                D = (np.ones((n1, 1)) * n1sq).T + np.ones((n1, 1)) * n1sq - 2 * np.dot(X.T, X)
            else:
                n2sq = np.sum(X2 ** 2, axis=0)
                n2 = X2.shape[1]
                D = (np.ones((n2, 1)) * n1sq).T + np.ones((n1, 1)) * n2sq - 2 * np.dot(X.T, X)
            K = np.exp(-gamma * D)
        elif ker == 'sam':
            if not X2:
                D = np.dot(X.T, X)
            else:
                D = np.dot(X.T, X2)
            K = np.exp(-gamma * np.arccos(D) ** 2)
        return K

    @staticmethod
    def kernel_inf(ker, X, x, gamma):
        if ker == 'rbf':
            k = np.exp(gamma * (2 * X.T @ x - np.tile(np.linalg.norm(X, axis=0), (x.shape[1], 1)).T -
                                np.tile(np.linalg.norm(x, axis=0), (X.shape[1], 1))))

            return k

    def normalise_features(self, h):
        fts = h["fts"]
        #fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        mean = np.mean(fts, 0)
        std = np.std(fts, 0)
        features = (fts - mean) / (std + 1e-9)
        self.mean = mean
        self.std = std
        return features

    def apply_normalisation(self, h):
        fts = h["fts"]
        #fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        return (fts - self.mean) / (self.std + 1e-9)

    def lapgraph(self, x, num_neighbors=10):
        # compute cosine distance of all samples
        w = x.T @ x / np.linalg.norm(x, 2, 0) ** 2
        sim2 = np.round(w.copy(), 6)
        neighbor_mask = np.zeros_like(w)
        for i in range(num_neighbors):
            nth_neighbor = np.logical_or(sim2 == np.max(sim2, 0), (sim2 == np.max(sim2, 0)).T)
            sim2[nth_neighbor] = -1
        w = w * neighbor_mask

        l = np.diag(np.sum(w, 1)) - w
        return l

    def fit(self, xs, ys, xt):
        """
        Transform and Predict
        :param xs: ns * n_feature, source feature
        :param ys: ns * 1, source label
        :param xt: nt * n_feature, target feature
        :return: acc, y_pred, list_acc
        """

        if min(ys) == 0:
            self.min_classes = 0
            ys = ys + 1
        else:
            self.min_classes = 1

        gfk = GFK.GFK()
        self.sq_g = gfk.fit(xs, ys, xt)

        # Transform features into new feature space
        xs_new = self.sq_g @ xs.T
        xt_new = self.sq_g @ xt.T

        self.x = np.hstack((xs_new, xt_new))
        n_source, n_target = xs_new.shape[1], xt_new.shape[1]
        n_classes = len(np.unique(ys))
        list_acc = []
        yy = np.zeros((n_source, n_classes))
        for c in range(1, n_classes + 1):
            ind = np.where(ys == c)
            yy[ind, c - 1] = 1
        yy = np.vstack((yy, np.zeros((n_target, n_classes))))
        yy[0, 1:] = 0

        self.x /= np.linalg.norm(self.x, axis=0) + 1e-9
        l = self.lapgraph(self.x)  # Graph Laplacian is on the way...
        if self.clf is None:
            self.clf = KNeighborsClassifier(n_neighbors=1)
        self.clf.fit(self.x[:, :n_source].T, ys.ravel())
        cls = self.clf.predict(self.x[:, n_source:].T)
        k = self.kernel(self.kernel_type, self.x, X2=None, gamma=self.gamma)
        E = np.diagflat(np.vstack((np.ones((n_source, 1)), np.zeros((n_target, 1)))))
        for iteration in range(1, self.t + 1):
            # estimating mu keeps throwing warnings and thus we just fix mu
            e = np.vstack((1 / n_source * np.ones((n_source, 1)), -1 / n_target * np.ones((n_target, 1))))
            m = e * e.T * n_classes
            n = 0
            for c in range(1, n_classes + 1):
                e = np.zeros((n_source + n_target, 1))
                e[np.where(ys == c)] = 1 / (len(ys[np.where(ys == c)]) + 1e-9)
                ind = np.where(cls == c)
                inds = [item + n_source for item in ind]
                e[tuple(inds)] = -1 / (len(cls[np.where(cls == c)]) + 1e-9)
                e[np.isinf(e)] = 0
                n += np.dot(e, e.T)
            if self.est_mu:
                self.mu = self.estimate_mu(xs_new.T, ys, xt_new.T, cls)
            m = (1 - self.mu) * m + self.mu * n
            m /= np.linalg.norm(m, 'fro') + 1e-9
            left = np.dot(E + self.lamb * m + self.rho * l, k) + self.eta * np.eye(n_source+n_target, n_source+n_target)
            self.beta = np.dot(np.linalg.inv(left), np.dot(E, yy))
            f = np.dot(k, self.beta)
            cls = np.argmax(f, axis=1) + 1
            cls = cls[n_source:]

        return self.x, self.beta, self.sq_g, cls

    def inference(self, sample):
        sample = self.sq_g @ sample.T
        sample /= np.linalg.norm(sample, axis=0) + 1e-9
        k = self.kernel_inf('rbf', sample, self.x, 1)
        if self.min_classes == 1:
            return np.argmax(k @ self.beta, axis=1) + 1
        else:
            return np.argmax(k @ self.beta, axis=1)


def main():
    domains = ['Caltech10', 'amazon', 'webcam', 'dslr']
    for i in range(4):
        for j in range(4):
            if i != j:
                src = 'data/' + domains[i] + '_SURF_L10.mat'
                tar = 'data/' + domains[j] + '_SURF_L10.mat'
                src_domain = scipy.io.loadmat(src)
                tar_domain = scipy.io.loadmat(tar)
                classifier = MEDA(options={'mu': .7})
                xs = classifier.normalise_features(src_domain)
                xt = classifier.normalise_features(tar_domain)
                train_test_split = np.random.choice([True, False], size=xt.shape[0], p=[.9, .1])
                xt_train = xt[train_test_split]
                xt_eval = xt[np.logical_not(train_test_split)]
                ys, yt = src_domain['labels'], tar_domain['labels']
                yt_train = yt[train_test_split]
                yt_eval = yt[np.logical_not(train_test_split)]
                x, beta, sq_g, cls1 = classifier.fit(xs, ys, xt_train)
                acc = np.mean(cls1 == yt_train.ravel())
                print(domains[i] + "-->" + domains[j] + " on training target domain data: " + str(acc))

                cls2 = classifier.inference(xt_eval)
                acc = np.mean(cls2 == yt_eval.ravel())
                print(domains[i] + "-->" + domains[j] + " on evaluation target domain data: " + str(acc))


if __name__ == '__main__':
    main()
