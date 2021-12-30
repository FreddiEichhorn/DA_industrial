import numpy as np
import scipy.linalg
from sklearn.neighbors import KNeighborsClassifier


def gsvd(d1, d2):
    """Implementation of GSVD, since scipy, numpy and torch do not provide bindings for LAPACK, and won't for the fore-
    seeable future. There are some fringe possibilities like bob and pygsvd, but these have various drawbacks regarding
    portability or usability. Credit for the underlying algorithm goes to Rui Luo 'Computing and Visualizing the
    Generalized Singular Value Decomposition in Python'. This implementation has been validated against Matlab's gsvd
    function and this should be repeated should this implementation of GFK give problems in the future
    :param d1: First input matrix
    :param d2: Second input matrix"""
    q, r = np.linalg.qr(np.concatenate((d1, d2), 0))
    # q = np.concatenate((d1, d2), 0)
    # r = np.eye(q.shape[1])
    q1 = q[:d1.shape[0]]
    q2 = q[d1.shape[0]:]
    uq1, sigmaq1, vq1 = np.linalg.svd(q1)
    uq2, sigmaq2, vq2 = np.linalg.svd(q2)

    # flip results to correspond to matlab implementation
    vq1 = np.flipud(vq1)
    uq1 = np.fliplr(uq1)
    sigmaq1 = np.flip(sigmaq1)

    # svd is unique up to a sign. We need for both SVDs to return the same vq in order to obtain a valid gsvd.
    trans = np.eye(uq2.shape[1])
    keep_sign = np.all(np.round(vq1, 2) == np.round(vq2, 2), 1)
    for i in range(len(keep_sign)):
        if not keep_sign[i]:
            trans[i, i] = -1

    uq2 = uq2 @ trans
    vq2 = trans[:vq2.shape[0], :vq2.shape[1]] @ vq2

    u1 = uq1
    u2 = uq2
    vq1r = vq1 @ r
    sigma1 = sigmaq1 * np.sqrt(np.diag(vq1r @ vq1r.T))
    sigma2 = sigmaq2 * np.sqrt(np.diag(vq1r @ vq1r.T))
    v = np.diag((1 / np.sqrt(np.diag(vq1r @ vq1r.T)))) @ vq1 @ r
    # d1 = u1 @ np.diag(np.flip(sigma1)) @ v
    # d2 = u2 @ np.concatenate((np.diag(sigma2), np.zeros(d2.shape)) @ np.flipud(v))
    return u1, u2, sigma1, sigma2, v


def gsvd2(d1, d2):
    """Only works if d1 and d2 have same number of columns"""
    s = (d1.T @ d1 @ np.linalg.inv(d2.T @ d2) + d2.T @ d2 @ np.linalg.inv(d1.T @ d1))
    w, v = np.linalg.eig(s)
    b1 = d1 @ np.linalg.inv(v.T)
    b2 = d2 @ np.linalg.inv(v.T)

    sigma1 = np.sqrt(np.sum(b1**2, 0))
    sigma2 = np.sqrt(np.sum(b2**2, 0))

    u1 = b1 / sigma1
    u2 = b2 / sigma2
    return u1, u2, sigma1, sigma2, v


class GFK:
    def __init__(self, options=None, clf_neural_net=None):
        if options is None:
            options = {}

        if 'dim' in options.keys():
            self.dim = options['dim']
        else:
            self.dim = 20

        self.clf = None
        self.sq_g = None
        self.clf_neural_net = clf_neural_net

    @staticmethod
    def normalise_features(h):
        fts = h["fts"]
        fts = fts / (np.expand_dims(np.sum(fts, 1), 1) + 1e-9)
        mean = np.mean(fts, 0)
        std = np.std(fts, 0)
        features = (fts - mean) / (std + 1e-9)
        return features

    def fit(self, features_s, labels_s, features_t):
        # determine subspace of source and target features
        _, _, p_s = np.linalg.svd(features_s)
        _, _, p_t = np.linalg.svd(features_t)

        q = p_s.T
        p_t = p_t.T[:, :self.dim]
        n = p_t.shape[0]

        # GFK core
        qp_t = q.T @ p_t
        u1, u2, gamma, sigma, v = gsvd(qp_t[:self.dim, :], qp_t[self.dim:, :])
        theta = np.arccos(gamma)
        theta[theta == 0] = 1e-20
        u2 = -u2

        # This is the ugliest implementation of anything ever, but oh well
        b1 = 0.5 * np.diag(1 + np.sin(2 * theta) / 2 / theta)
        b2 = 0.5 * np.diag((-1 + np.cos(2 * theta)) / 2 / theta)
        b4 = 0.5 * np.diag(1 - np.sin(2 * theta) / 2 / theta)

        a1 = np.concatenate((u1, np.zeros((self.dim, n-self.dim))), 1)
        a2 = np.concatenate((np.zeros((n-self.dim, self.dim)), u2), 1)
        a3 = np.concatenate((a1, a2), 0)

        a4 = np.concatenate((b1, b2, np.zeros((self.dim, n-2*self.dim))), 1)
        a5 = np.concatenate((b2, b4, np.zeros((self.dim, n-2*self.dim))), 1)
        a6 = np.concatenate((a4, a5, np.zeros((n-2*self.dim, n))), 0)

        a7 = np.concatenate((u1, np.zeros((self.dim, n-self.dim))), 1)
        a8 = np.concatenate((np.zeros((n-self.dim, self.dim)), u2), 1)
        a9 = np.concatenate((a7, a8), 0)

        g = q @ a3 @ a6 @ a9.T @ q.T
        self.sq_g = np.real(scipy.linalg.sqrtm(g))

        self.clf = KNeighborsClassifier(1)
        self.clf.fit((self.sq_g @ features_s.T).T, labels_s[:, 0])

        if self.clf_neural_net is not None:
            for _ in range(self.clf_neural_net.num_it):
                self.clf_neural_net.forward()
        return self.sq_g

    def inference(self, sample):
        return self.clf.predict((self.sq_g @ sample.T).T)

    def inference_neural_net(self, sample):
        if self.clf_neural_net is not None:
            return self.clf_neural_net.forward(sample)

