# encoding=utf-8
"""
    Created on 10:40 2018/11/14 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
from sklearn.neighbors import KNeighborsClassifier


class JGSA:
    def __init__(self, clf=None, options=None):
        if options is None:
            options = {}

        # Collect hyperparameters
        if 'beta' in options.keys():
            self.beta = options['beta']
        else:
            self.beta = .1

        if 'mu' in options.keys():
            self.mu = options['mu']
        else:
            self.mu = 1

        if 'alpha' in options.keys():
            self.alpha = options['alpha']
        else:
            self.alpha = 1

        if 'subspace_dim' in options.keys():
            self.subspace_dim = options['subspace_dim']
        else:
            self.subspace_dim = 30

        if 'n_iterations' in options.keys():
            self.n_iterations = options['n_iterations']
        else:
            self.n_iterations = 2

        self.clf = clf
        self.transf_t = None
        self.transf_s = None

    @staticmethod
    def normalise_features(h):
        """Same normalization Zhang et al use in their implementation. It should be noted that the last line of this
        method has considerable influence on accuracy"""
        fts = h["fts"]
        fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        mean = np.mean(fts, 0)
        std = np.std(fts, 0)
        features = (fts - mean) / (std + 1e-9)
        features = features / np.expand_dims(np.sqrt(np.sum(features ** 2, 1)), 1)
        return features

    @staticmethod
    def construct_m(xs, xt, ys, yt):
        # This matrix is used for normalization of Ms, Mt, Mst and Mts, but is not mentioned in the paper,
        # only in the code
        e = np.vstack((np.ones((xs.shape[1], 1)) / xs.shape[1], -np.ones((xt.shape[1], 1)) / xt.shape[1]))
        norm = np.linalg.norm(e @ e.T * len(np.unique(ys)), ord='fro')

        # Compute Ms for distribution divergence minimization.
        ls = np.ones((xs.shape[1], xs.shape[1])) / xs.shape[1] ** 2 * len(np.unique(ys))
        lsc = np.zeros_like(ls)
        for c in np.unique(ys):
            x = xs[:, (ys == c)[:, 0]]
            idx = np.where(ys == c)[0]
            if not x.shape[0] == 0:
                lsc[np.ix_(idx, idx)] = 1 / x.shape[1] ** 2
        # No normalization is done in the paper, but in their implementation it is done
        ms = xs @ (ls + lsc) / norm @ xs.T

        # Compute Mt for distribution divergence minimization
        lt = np.ones((xt.shape[1], xt.shape[1])) / xt.shape[1] ** 2 * len(np.unique(ys))
        ltc = np.zeros_like(lt)
        for c in np.unique(yt):
            x = xt[:, (yt == c)[:, 0]]
            idx = np.where(yt == c)[0]
            if not x.shape[0] == 0:
                ltc[np.ix_(idx, idx)] = 1 / x.shape[1] ** 2
        mt = xt @ (lt + ltc) @ xt.T / norm

        # Compute Mst for distribution divergence minimization
        lst = -np.ones((xs.shape[1], xt.shape[1])) / xt.shape[1] / xs.shape[1] * len(np.unique(ys))
        lstc = np.zeros_like(lst)
        for c in np.unique(ys):
            idx_t = np.where(yt == c)[0]
            idx_s = np.where(ys == c)[0]
            if not idx_s.shape[0] == 0 and not idx_t.shape[0] == 0:
                lstc[np.ix_(idx_s, idx_t)] = -1 / idx_s.shape[0] / idx_t.shape[0]
        mst = xs @ (lst + lstc) @ xt.T / norm

        # Compute Mts for distribution divergence minimization
        mts = mst.T
        return ms, mt, mst, mts

    def fit(self, xs, ys, xt):
        """Perform jgsa domain adaptation on source Xs and target Xt. All variables are named as in the paper
        :param xs: Source domain data as numpy array NxD where N is the number of samples and D the number of features per
                    sample
        :param xt: Target domain data as NxD numpy array
        :param: ys: Ground truth source domain labels as Nx1 numpy array
        """

        # Calculate St (target variance maximization)
        xs = xs.T
        xt = xt.T
        ht = np.eye(xt.shape[1]) - np.ones((xt.shape[1], xt.shape[1])) / xt.shape[1]
        st = xt @ ht @ xt.T

        # Calculate Sb (between class scatter for source discriminative information preservation)
        sb = 0
        ms_mean = np.sum(xs, 1) / len(ys)
        for c in np.unique(ys):
            ns = np.sum(ys == c)
            ms = np.sum(xs[:, (ys == c)[:, 0]], 1) / ns
            sb += ns * np.expand_dims(ms - ms_mean, 1) @ np.expand_dims(ms - ms_mean, 1).T

        # Calculate Sw (within class scatter for source discriminative information preservation)
        sw = np.zeros((xs.shape[0], xs.shape[0]))
        for c in np.unique(ys):
            x = xs[:, (ys == c)[:, 0]]
            hs = np.eye(x.shape[1]) - np.ones((x.shape[1], x.shape[1])) / np.sum(ys == c)
            sw += x @ hs @ x.T

        zs = xs
        zt = xt
        for _ in range(self.n_iterations):
            # obtain pseudo-labels of target samples
            if self.clf is None:
                self.clf = KNeighborsClassifier(1)
            self.clf.fit(zs.T, ys[:, 0])
            yt = np.expand_dims(self.clf.predict(zt.T), 1)

            ms, mt, mst, mts = self.construct_m(xs, xt, ys, yt)

            # Assemble matrices for generalised eigendecomposition
            a = np.block([[self.beta * sb, np.zeros((sb.shape[0], st.shape[1]))],
                          [np.zeros((st.shape[0], sb.shape[1])), self.mu * st]])
            b = np.block([[ms + np.eye(ms.shape[0]) * self.alpha + self.beta*sw, mst - self.alpha*np.eye(ms.shape[0])],
                          [mts - self.alpha * np.eye(mts.shape[0]), mt + (self.alpha + self.mu) * np.eye(mt.shape[0])]])

            w, v = scipy.linalg.eig(a, b)
            v = np.real(v)
            self.transf_s = v[:int(v.shape[0] / 2), :self.subspace_dim]
            self.transf_t = v[int(v.shape[0] / 2):, :self.subspace_dim]
            zs = self.transf_s.T @ xs
            zt = self.transf_t.T @ xt

        if self.clf is None:
            self.clf = KNeighborsClassifier(1)
        self.clf.fit(zs.T, ys[:, 0])
        return zs, zt, self.transf_s, self.transf_t

    def inference(self, xt, from_target_domain=True):
        if from_target_domain:
            zt = self.transf_t.T @ xt.T
            return self.clf.predict(zt.T)
        else:
            zs = self.transf_s.T @ xt.T
            return self.clf.predict(zs.T)


def main():
    domains = ['Caltech10', 'amazon', 'webcam', 'dslr']
    for i in range(4):
        for j in range(4):
            if i != j:
                src = 'data/' + domains[i] + '_SURF_L10.mat'
                tar = 'data/' + domains[j] + '_SURF_L10.mat'
                src_domain = scipy.io.loadmat(src)
                tar_domain = scipy.io.loadmat(tar)
                Xs = JGSA.normalise_features(src_domain)
                Xt = JGSA.normalise_features(tar_domain)
                Ys, Yt = src_domain['labels'], tar_domain['labels']
                classifier = JGSA()
                zs, zt, transf_s, transf_t = classifier.run_jgsa(Xs, Xt, Ys)

                nearest_neigh = KNeighborsClassifier(1)
                nearest_neigh.fit(zs.T, Ys[:, 0])
                predict_t = nearest_neigh.predict(zt.T)
                acc = (predict_t == Yt[:, 0]).sum() / len(Yt)

                print(domains[i] + "-->" + domains[j] + " acc: " + str(acc))


if __name__ == '__main__':
    main()
