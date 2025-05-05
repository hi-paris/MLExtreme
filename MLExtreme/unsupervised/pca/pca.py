"""
File: pca.py
Author: Anne Sabourin
Date: 2025-05-04
"""

import numpy as np
from scipy.linalg import svd
# from math import isclose
import matplotlib.pyplot as plt
# from copy import deepcopy
# from ...utils.EVT_basics import rank_transform, round_signif


class Xpca:
    def __init__(self, thresh_train=None, thresh_predict=None, beta=1,
                 centered=False):
        self.thresh_train = thresh_train
        self.thresh_predict = thresh_predict
        self.centered = centered
        self.beta = beta
        self.mean_angle = None
        self.principal_comp = None
        self.principal_axes = None
        self.eigenvalues = None


    def fit(self, X, max_dim=None):
        r"""
        Performs PCA by computing the singular value decomposition
        of the rescaled extreme points on the training dataset X,
        where each extreme point \(x\) is rescaled by the factor \(
        \\| x \\|^{-\\beta} \). Default value for beta is
        1. Admissible values are the interval \( (1-\\alpha/2, 1] \)
        where \( \\alpha\) is the regular variation index of the input
        X.

        """
        Xt = X
        R_squared = np.sum(Xt**2, axis=1)
        norm_Xt = R_squared**(1/2)
        # Set the training threshold if not provided
        if self.thresh_train is None:
            self.thresh_train = np.quantile(
                norm_Xt, 1 - 1 / np.sqrt(len(norm_Xt)))

        # set max_dim to ambient dimension if not provided
        if max_dim is None:
            max_dim = Xt.shape[1]
        # define extreme samples
        is_extreme = norm_Xt > self.thresh_train
        X_ext = Xt[is_extreme]
        # Update total mass accordingly
        self.number_extremes = len(X_ext)
        R_ext = norm_Xt[is_extreme]
        ang_ext = X_ext/(R_ext**(self.beta)).reshape(-1, 1)
        if self.centered:
            center = np.mean(ang_ext, axis=0)
            self.mean_angle = center
            Z = ang_ext - center
        else:
            Z = ang_ext
        U, s, Vh = svd(Z, full_matrices=False)
        self.principal_comp = U[:, :max_dim]
        self.principal_axes = (Vh.transpose())[:, :max_dim]
        self.eigenvalues = s[:max_dim]**2

    def scores(self, Xtest,  max_dim=None):
        r"""The method returns the scores, i.e. the coordinates of the
        points in the array Xtest, in the coordinate system given by
        the principal axes, of the extreme points (with norm greater
        than self.threh_predict) rescaled through the mapping
         \(z =\\|x\\|^{-\\beta} x \) where \( \\beta \)= self.beta. The model
        must have been fit beforehand. If Xtest is the training data,
        this gives the usual PCA scores of the rescaled extreme data. 

        """
        if self.principal_axes is None:
            raise ValueError("The model has not been fitted")
        if max_dim is None:
            p = np.shape(self.principal_axes)[1]
        else:
            p = max_dim

        if self.thresh_predict is None:
            self.thresh_predict = self.thresh_train

        Xt = Xtest
        norm_Xt = np.sum(Xt**2, axis=1)**(1/2)
        # define extreme samples
        is_extreme = norm_Xt > self.thresh_predict
        X_ext = Xt[is_extreme]
        R_ext = norm_Xt[is_extreme]
        ang_ext = X_ext/(R_ext**(self.beta)).reshape(-1, 1)
            
        if not self.centered:
            Zang = ang_ext
        else:
            Zang = ang_ext - self.mean_angle
        coords_in_pr_axes = Zang @ self.principal_axes[:, :p]
        mask = is_extreme
        return coords_in_pr_axes, X_ext, mask
    
    def predict(self, X_test, max_dim=None):
        """
        projects X_test onto the 

        """
        if self.principal_axes is None:
            raise ValueError("The model has not been fitted")
        if max_dim is None:
            p = np.shape(self.principal_axes)[1]
        else:
            p = max_dim
        Scores, X_ext, mask = self.scores(X_test, max_dim)
        recons_angle_intermediate = Scores @ (
            (self.principal_axes[:, :p]).transpose())
        if self.centered:
            recons_angle = recons_angle_intermediate + self.mean_angle
        else:
            recons_angle = recons_angle_intermediate
        R_ext = (np.sum(X_ext**2, axis=1))**(1/2)
        recons_X = recons_angle * (R_ext**self.beta).reshape(-1, 1)
        return recons_X, X_ext, mask

    def screeplot(self):
        if self.principal_axes is None:
            raise ValueError("The model has not been fitted")
        eigenvals = self.eigenvalues
        # Create the scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvals, 'bo-', markersize=8, label='Eigenvalues')
        # Add labels and title
        plt.xlabel('Principal Component Index')
        plt.ylabel('Eigenvalue')
        plt.title('Scree Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def recons_error(self, X_test, max_dim, test_beta=None):
        r"""The test data are rescaled to ensure existence of second
        moments, namely each test data is transformed following
        .. math::

        Z_i  = \\| Xtest_i\\|^{-\\beta} X_i

        where \( \\beta \\le 1 \) is given by the argument
        test_beta. Default is test_beta=None, it which case test_beta
        is set to the training value self.beta which default is 1. 

        """
        if test_beta is None:
            beta = self.beta
        else:
            beta = test_beta
        recons_X, X_ext, mask = self.predict(X_test, max_dim)
        
        X_ext_norm = np.sum(X_ext**2, axis=1)**(1/2)
        Z = X_ext / (X_ext_norm**beta).reshape(-1, 1)
        recons_Z  = recons_X / (X_ext_norm**beta).reshape(-1, 1)
        Err = np.sum((Z - recons_Z)**2, axis=1)
        return Err
