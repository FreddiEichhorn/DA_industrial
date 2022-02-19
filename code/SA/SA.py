# encoding=utf-8
"""
    Created on 10:40 2018/11/14 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
from sklearn.neighbors import KNeighborsClassifier


class SA:
    def __init__(self, options=None, clf=None):
        if options is None:
            options = {}

        self.Utrans = None
        self.Ut = None
        self.clf = clf

    @staticmethod
    def normalise_features(h):
        fts = h["fts"]
        fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        mean = np.mean(fts, 0)
        std = np.std(fts, 0)
        features = (fts - mean) / (std + 1e-9)
        return features

    @staticmethod
    def pca(x, n_components):
        cov = np.cov(x, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        return sorted_eigenvectors[:, 0:n_components]

    def fit(self, Xs, Ys, Xt):
        Us = self.pca(Xs, 50)
        self.Ut = self.pca(Xt, 50)
        self.Utrans = Us @ Us.T @ self.Ut
        if self.clf is None:
            self.clf = KNeighborsClassifier(1)
        self.clf.fit(Xs @ self.Utrans, Ys[:, 0])
        return self.Utrans, self.Ut

    def inference(self, xt):
        predict_t = self.clf.predict(xt @ self.Ut)
        return predict_t


class PCASource:
    def __init__(self, options=None, clf=None):
        if options is None:
            options = {}
        if 'subspace_dim' in options.keys():
            self.subspace_dim = options['subspace_dim']
        else:
            self.subspace_dim = 50
        self.Us = None
        self.clf = clf

    @staticmethod
    def normalise_features(h):
        fts = h["fts"]
        fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        mean = np.mean(fts, 0)
        std = np.std(fts, 0)
        features = (fts - mean) / (std + 1e-9)
        return features

    @staticmethod
    def pca(x, n_components):
        cov = np.cov(x, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        return sorted_eigenvectors[:, 0:n_components]

    def fit(self, xs, ys, xt):
        self.Us = self.pca(xs, self.subspace_dim)
        if self.clf is None:
            self.clf = KNeighborsClassifier(1)
        self.clf.fit(xs @ self.Us, ys[:, 0])
        return self.Us

    def inference(self, xt):
        predict_t = self.clf.predict(xt @ self.Us)
        return predict_t


class PCATarget:
    def __init__(self, options=None, clf=None):
        if options is None:
            options = {}
            self.subspace_size = 50
        else:
            self.subspace_size = options['subspace_size']
        self.Ut = None
        self.clf = clf

    @staticmethod
    def normalise_features(h):
        fts = h["fts"]
        fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        mean = np.mean(fts, 0)
        std = np.std(fts, 0)
        features = (fts - mean) / (std + 1e-9)
        return features

    @staticmethod
    def pca(x, n_components):
        cov = np.cov(x, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        return sorted_eigenvectors[:, 0:n_components]

    def fit(self, xs, ys, xt):
        self.Ut = self.pca(xt, self.subspace_size)
        if self.clf is None:
            self.clf = KNeighborsClassifier(1)
        self.clf.fit(xs @ self.Ut, ys[:, 0])
        return self.Ut

    def inference(self, xt):
        predict_t = self.clf.predict(xt @ self.Ut)
        return predict_t


def main():
    domains = ['Caltech10', 'amazon', 'webcam', 'dslr']
    for i in range(4):
        for j in range(4):
            if i != j:
                src = 'data/' + domains[i] + '_SURF_L10.mat'
                tar = 'data/' + domains[j] + '_SURF_L10.mat'
                src_domain = scipy.io.loadmat(src)
                tar_domain = scipy.io.loadmat(tar)
                #Xs = SA.normalise_features(src_domain)
                Xs = src_domain['fts']
                #Xt = SA.normalise_features(tar_domain)
                Xt = tar_domain['fts']
                Ys, Yt = src_domain['labels'], tar_domain['labels']
                classifier = SA()
                Utrans, Ut = classifier.fit(Xs, Ys, Xt)
                predict_t = classifier.inference(Xt)

                print(domains[i] + "-->" + domains[j] + ": ")


if __name__ == '__main__':
    main()
