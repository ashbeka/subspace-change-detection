import cupy as cp
import numpy as np
from scipy.linalg import expm as npexpm
from cupyx.scipy.linalg import expm as cpexpm

class GRTool():

    def __init__(self, mode="numpy"):
        self.mode = mode
    
    def basisToGrassmannian(self, basis):

        P = None
        if self.mode == "numpy": P = basis@np.transpose(basis)
        elif self.mode == "cupy": P = basis@cp.transpose(basis)
        return P
    
    def grassmannianToBasis(self, P):

        basis = None
        if self.mode == "numpy":
            _, basis = np.linalg.eigh(P)
        elif self.mode == "cupy":
            _, basis = cp.linalg.eigh(P)
        return basis
    
    def _getSymmetric(self, X):

        if self.mode == "numpy": symX = 0.5 * (X + np.transpose(X))
        elif self.mode == "cupy": symX = 0.5 * (X + cp.transpose(X))
        return symX
    
    # X \in Grass(p, n) : point on grassmann manifold, spanning tangent space
    # D \in R^{n \times n} : any square matrix
    # This function can be utilized when we cannot assure that
    # D is on tangent space of X.
    # e.g. re-project euclid gradient D onto tangent space of X.
    def tangentProjection(self, X, D):
        if self.mode == "numpy": I = np.eye(X.shape[0])
        elif self.mode == "cupy": I = cp.eye(X.shape[0])
        tmp = X @ self._getSymmetric(D) @ (I - X)
        return 2 * self._getSymmetric(tmp)
    
    def tangentVectorInnerProduct(self, zeta, eta):
        if self.mode == "numpy": return np.trace(np.transpose(zeta)@eta)
        elif self.mode == "cupy": return cp.trace(cp.transpose(zeta)@eta)
        
    def tangentVectorNorm(self, zeta):
        if self.mode == "numpy": return np.sqrt(np.trace(np.transpose(zeta)@zeta))
        elif self.mode == "cupy": return cp.sqrt(cp.trace(cp.transpose(zeta)@zeta))
    
    def tangentVectorNormalize(self, X):
        return X / self.tangentVectorNorm(X)
    
    # Y \ in St(n, p) : point on stiefel manifold, such that P = Y Y^T
    # U \ in St(n, p) : point on stiefel manifold, such that X = U U^T
    # Output : point on horizental space of U, such that
    # its exponential mapping from tangent space of grassmannian X is P.
    # In short, output is tangent vector on tangent space of U, correspond with Y.
    def logarithmicMapping(self, X, P):

        U = self.grassmannianToBasis(X)
        Y = self.grassmannianToBasis(P)

        YTU = cp.transpose(Y)@U

        if self.mode == "numpy":
            _, Q_tilde = np.linalg.eigh(YTU@np.transpose(YTU))
            Q_tilde = Q_tilde[:,::-1]
            _, R_tilde = np.linalg.eigh(np.transpose(YTU)@YTU)
            R_tilde = R_tilde[:,::-1]
            Y_prime = Y @ Q_tilde @ np.transpose(R_tilde)

            result = (np.eye(U.shape[0]) - U@np.transpose(U)) @ Y_prime
            _, Q = np.linalg.eigh(result@np.transpose(result))
            Q = Q[:,::-1]
            _, R = np.linalg.eigh(np.transpose(result)@result)
            R = R[:,::-1]
            _, sigma, _ = np.linalg.svd(result)

            return Q[:,:sigma.shape[0]] @ np.diag(np.arcsin(sigma)) @ np.transpose(R)
        elif self.mode == "cupy":
            _, Q_tilde = cp.linalg.eigh(YTU@cp.transpose(YTU))
            Q_tilde = Q_tilde[:,::-1]
            _, R_tilde = cp.linalg.eigh(cp.transpose(YTU)@YTU)
            R_tilde = R_tilde[:,::-1]
            Y_prime = Y @ Q_tilde @ cp.transpose(R_tilde)

            result = (cp.eye(U.shape[0]) - U@cp.transpose(U)) @ Y_prime
            _, Q = cp.linalg.eigh(result@cp.transpose(result))
            Q = Q[:,::-1]
            _, R = cp.linalg.eigh(cp.transpose(result)@result)
            R = R[:,::-1]
            _, sigma, _ = cp.linalg.svd(result)

            return Q[:,:sigma.shape[0]] @ cp.diag(cp.arcsin(sigma)) @ cp.transpose(R)
    
    # X \in Grass(p, n) : point on grassmann manifold, spanning tangent space
    # eta \ in T_X Grass(p, n) : tangent vector on tangent space of X
    # Output : point on grassmann manifold Grass(p, n), such that
    # exponential mapping of eta, from tangent space of grassmannian X.
    def exponentialMapping(self, X, eta):

        L = None
        R = None
        if self.mode == "numpy":
            L = npexpm(eta@X - X@eta)
            R = npexpm(X@eta - eta@X)
        elif self.mode == "cupy":
            L = cpexpm(eta@X - X@eta)
            R = cpexpm(X@eta - eta@X)
        return L@X@R