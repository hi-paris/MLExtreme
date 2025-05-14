# Author: Anne Sabourin
# Date: 2025
"""
Module for Supervised Learning with Heavy-Tailed Covariates

This module supports supervised binary classification and regression tasks with squared error loss, specifically designed for scenarios involving heavy-tailed (regularly varying) covariates.
"""

import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from copy import deepcopy


class xcovPredictor:
    """
    A prediction model for extreme value analysis, tailored for classification\
    or regression tasks on the tails of the covariate distribution.

    Parameters
    ----------
    task : str
        The type of prediction task, either 'regression' or 'classification'.
    model : object
        The model used for prediction, must have `.fit` and `.predict` methods.
    norm_func : callable, optional
        Function to compute the norm of input data. Defaults to L2 norm.

    Attributes
    ----------
    task : str
        The type of prediction task.
    model : object
        A deep copy of the model provided during initialization.
    norm_func : callable
        The norm function used for selecting extremes.
    thresh_train : float
        Training threshold, set after fitting.
    ratio_train : float
        Ratio of extreme training samples to total training samples, set after fitting.

    Methods
    -------
    fit(X_train, y_train, k=None, thresh_train=None):
        Fit the model using extreme points from the training data.
    predict(X_test, thresh_predict=None):
        Predict the target from extreme test covariates.
    cross_validate(X, y, thresh_train=None, thresh_predict=None, cv=5, scoring=None, random_state=None):
        Evaluate the generalization risk using cross-validation.
    """

    def __init__(self, task, model, norm_func=None):
        if task not in ["regression", "classification"]:
            raise ValueError("task must be either 'classification' or 'regression'")
        internal_model = deepcopy(model)
        self.task = task
        self.model = internal_model
        self.norm_func = norm_func if norm_func else lambda x: np.linalg.norm(x, axis=1)
        self.thresh_train = None
        self.ratio_train = None

    def fit(self, X_train, y_train, k=None, thresh_train=None):
        """
        Fit the model using extreme points from the training data.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training input samples.
        y_train : array-like, shape (n_samples,)
            Target values for training.
        k : int, optional
            Number of extreme samples to use for training.
        thresh_train : float, optional
            Threshold for considering extreme samples during training.

        Returns
        -------
        thresh_train : float
            The training threshold used.
        ratio_train : float
            Ratio of extreme points used for training.
        X_train_extreme : array-like, shape (n_extreme_samples, n_features)
            Extreme covariate points used for training.
        """
        if not callable(self.norm_func):
            raise ValueError("norm_func must be callable")

        if k is not None and thresh_train is not None:
            raise ValueError("k and thresh_train cannot both be set at the same time")

        Norm_X_train = self.norm_func(X_train)

        if k is None and thresh_train is None:
            thresh_train = np.quantile(Norm_X_train,
                                       1 - 1 / np.sqrt(len(Norm_X_train)))
        if thresh_train is None:
            thresh_train = np.quantile(Norm_X_train, 1 - k / len(Norm_X_train))

        id_extreme = Norm_X_train >= thresh_train
        if k is None:
            k = np.sum(id_extreme)

        self.thresh_train = thresh_train
        self.ratio_train = k / len(Norm_X_train)

        X_train_extreme = X_train[id_extreme]
        X_train_extreme_unit = X_train_extreme / Norm_X_train[id_extreme][:, np.newaxis]
        y_train_extreme = y_train[id_extreme]

        self.model.fit(X_train_extreme_unit, y_train_extreme)

        return thresh_train, k / len(Norm_X_train), X_train_extreme

    def predict(self, X_test, thresh_predict=None):
        """
        Predict the target from extreme test covariates.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test input samples.
        thresh_predict : float, optional
            Threshold for selecting extreme points during prediction.

        Returns
        -------
        y_pred_extreme : array-like, shape (n_extreme_samples,)
            Predicted labels for extreme points.
        X_test_extreme : array-like, shape (n_extreme_samples, n_features)
            Extreme points from the test data.
        mask_test : array-like, shape (n_samples,)
            Boolean mask indicating extreme points.
        """
        if self.thresh_train is None:
            raise ValueError("Model has not been fitted yet.")

        if thresh_predict is None:
            thresh_predict = self.thresh_train

        Norm_X_test = self.norm_func(X_test)
        mask_test = Norm_X_test >= thresh_predict
        X_test_extreme = X_test[mask_test]

        X_test_extreme_unit = X_test_extreme / Norm_X_test[mask_test][:, np.newaxis]
        y_pred_extreme = self.model.predict(X_test_extreme_unit)

        return y_pred_extreme, X_test_extreme, mask_test

    def cross_validate(
        self,
        X,
        y,
        thresh_train=None,
        thresh_predict=None,
        cv=5,
        scoring=None,
        random_state=None
    ):
        """
        Perform cross-validation and return the mean score, standard deviation, and scores.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        y : array-like, shape (n_samples,)
            Target values.
        thresh_train : float, optional
            Threshold for extreme values during training.
        thresh_predict : float, optional
            Threshold for extreme values during prediction.
        cv : int, optional
            Number of folds for cross-validation. Default is 5.
        scoring : callable, optional
            Scoring function to evaluate predictions.
        random_state : int, optional
            Seed for random number generation.

        Returns
        -------
        mean_score : float
            Mean score from cross-validation.
        std_err_mean : float
            Estimated standard deviation of the mean score.
        scores : list of float
            Scores from each fold of the cross-validation.
        """
        scores = []

        if not callable(self.norm_func):
            raise ValueError("norm_func must be callable")

        if scoring is None:
            scoring = (
                hamming_loss if self.task == "classification" else mean_squared_error
            )

        if not callable(scoring):
            raise ValueError("scoring must be callable or None")

        Norm_X = self.norm_func(X)

        if thresh_train is None:
            thresh_train = np.quantile(Norm_X, (1 - 1 / np.sqrt(len(Norm_X))))

        if thresh_predict is None:
            thresh_predict = thresh_train

        id_extreme_train = Norm_X >= thresh_train
        id_extreme_predict = Norm_X >= thresh_predict

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        for train_index, test_index in kf.split(X):
            size_train_ex = np.sum(id_extreme_train[train_index])
            size_predict_ex = np.sum(id_extreme_predict[test_index])
            if size_train_ex <= 2 or size_predict_ex <= 0:
                continue

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                self.fit(X_train, y_train, k=None, thresh_train=thresh_train)
            except Exception as e:
                ratio_train = size_train_ex / len(train_index)
                print(
                    f"Warning: An error occurred for extreme ratio (training) = {ratio_train:4f}. Error: {e}"
                )
                continue

            y_pred_extreme, _, mask_test = self.predict(
                X_test, thresh_predict=thresh_predict
            )
            result = scoring(y_test[mask_test], y_pred_extreme)
            scores.append(result)

        mean_size_train = np.sum(id_extreme_train) * (cv - 1) / cv
        mean_size_predict = np.sum(id_extreme_predict) / cv
        order_error = 1 / np.sqrt(mean_size_predict) + 1 / np.sqrt(mean_size_train)

        return np.mean(scores), order_error, scores


# import numpy as np
# from sklearn.metrics import hamming_loss
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import KFold
# from copy import deepcopy


# class xcovPredictor:
#     """
#     A prediction model (either classification or regression)
#      designed specifically for making predictions on the tails of the
#      covariate distribution.

#      Parameters
#     ----------

#     task: str The type of prediction task
#      envisioned.  Must be either 'prediction' or 'classification'.

#     model : object The model to be used for prediction. Must have a
#         `.fit` method and a `.predict' method.

#     norm_func : callable, optional A function that computes a norm of
#         the input data. Default is L2 norm.


#     Attributes
#     ----------

#     task : str The type of prediction task
#     envisioned.  Either 'prediction' or 'classification'.

#     model : object The model used for prediction.  It is a
#         `copy.deepcopy` copy of the argument passed to the contructor.

#     norm_func : callable The norm function used at training and
#         testing for selecting extremes.

#     thresh_train: float.  Set to None at initialization and later set
#         to the training threshold after fitting.

#      ratio_train: float, between 0 and 1 Set to None at initialization
#         and later set to the ratio k/n of extreme training sample
#         divided by the training sample size.

#     Methods
#     _________

#     .fit : fit the self.model object using the angle
#     of largest covariates and their associated targets

#     .predict : predicts the target based on new (extreme) covariates

#     .cross_validate : evaluates the generalization risk based on
#     cross-validation."""

#     def __init__(self, task, model, norm_func=None):
#         if task != "regression" and task != "classification":
#             raise ValueError("task must be either 'classification' or 'regression'")
#         internal_model = deepcopy(model)
#         self.task = task
#         self.model = internal_model
#         self.norm_func = norm_func if norm_func else lambda x: np.linalg.norm(x, axis=1)
#         self.thresh_train = None
#         self.ratio_train = None

#     def fit(self, X_train, y_train, k=None, thresh_train=None):
#         """
#         Fit the model using extreme points from the training data where
#         the covariate norm exceeds a high threshold.

#         Parameters
#         ----------
#         X_train : array-like of shape (n_samples, n_features)
#             The training input samples.

#         y_train : array-like of shape (n_samples,)
#             The target values for training.

#         k : int, optional
#             The number of extreme samples used to train the model. Data are
#             ordered according to their norm (`norm_func') and the k largest are
#             selected.

#         thresh_train:  float, optional.
#             The radial threshold above which training samples are considered
#             extreme for training

#         Details
#         ------

#         If `thresh_train=None` and `k=None`, `k` is set to sqrt(n_samples)
#           and thresh_train is set to the empirical 1 - k/n_samples quantile of
#           the norms of the input X_train

#         Setting both thresh_train and k at the same time raises an error.

#         Returns
#         -------
#         thresh_train : float
#             The default threshold for further prediction,
#             set to the training threshold. Also stored in model's
#             attribute .thresh_train

#         ratio_train: float
#             The ratio of extreme points used for training.
#             Also stored in model's attribute .ratio_train

#         X_train : array-like of shape (n_extreme_samples, n_features)
#             The  extreme covariate points actually  used to train the model.
#         """
#         if not callable(self.norm_func):
#             raise ValueError("norm_func must be callable")

#         # check and set k and thresh_train
#         if k is not None and thresh_train is not None:
#             raise ValueError("k and thresh_train cannot both be set at the same time")

#         Norm_X_train = self.norm_func(X_train)

#         if k is None and thresh_train is None:
#             thresh_train = np.quantile(Norm_X_train, 1 - 1 / np.sqrt(len(Norm_X_train)))
#         if thresh_train is None:  ## then k is not none
#             thresh_train = np.quantile(Norm_X_train, 1 - k / len(Norm_X_train))

#         id_extreme = Norm_X_train >= thresh_train
#         if k is None:
#             k = np.sum(id_extreme)
#         # update model with training threshold and k/n ratio
#         self.thresh_train = thresh_train
#         self.ratio_train = k / len(Norm_X_train)

#         X_train_extreme = X_train[id_extreme]
#         X_train_extreme_unit = X_train_extreme / Norm_X_train[id_extreme][:, np.newaxis]
#         y_train_extreme = y_train[id_extreme]

#         self.model.fit(X_train_extreme_unit, y_train_extreme)

#         return thresh_train, k / len(Norm_X_train), X_train_extreme

#     def predict(self, X_test, thresh_predict=None):
#         """
#         Predict the target from  extreme test covariates.

#         Parameters
#         ----------
#         X_test : array-like of shape (n_samples, n_features)
#             The test input (covariate) samples.

#         thresh_predict : float, optional
#             The threshold value to select extreme points. If not provided,
#             the threshold used during fitting will be used by default.


#         Returns
#         -------
#         y_pred : array-like of shape (n_extreme_samples,)
#             The predicted labels for the extreme points.

#         X_test : array-like of shape (n_extreme_samples, n_features)
#             The  extreme points from the test data.

#         mask_test : array-like of shape (n_samples,)
#             A boolean mask indicating which points are extreme.

#         """
#         if self.thresh_train is None:
#             raise ValueError("Model has not been fitted yet.")

#         if thresh_predict is None:
#             thresh_predict = self.thresh_train

#         Norm_X_test = self.norm_func(X_test)
#         mask_test = Norm_X_test >= thresh_predict
#         X_test_extreme = X_test[mask_test]

#         X_test_extreme_unit = X_test_extreme / Norm_X_test[mask_test][:, np.newaxis]
#         y_pred_extreme = self.model.predict(X_test_extreme_unit)

#         return y_pred_extreme, X_test_extreme, mask_test

#     def cross_validate(
#         self,
#         X,
#         y,
#         thresh_train=None,
#         thresh_predict=None,
#         cv=5,
#         scoring=None,
#         random_state=None,
#     ):
#         """Perform cross-validation and return the mean score
#         (defined by the scoring function), the estimated standard
#         deviation of the mean, and the array of scores.

#         Follows mostly Aghbalou et al's CV scheme (K-fold), see [1]
#         below.  Differently from the paper the radial threshold for
#         prediction/test may be chosen different from the radial
#         threshold for training.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             The input samples.

#         y : array-like of shape (n_samples,)
#             The target values.

#         thresh_train : float, optional
#             The threshold for considering extreme values during training.
#             By default, set to the 1- 1/sqrt(n) quantile of the covariates norms.

#         thresh_predict : float, optional
#             The threshold for considering extreme values during prediction.
#             By default, set to thresh_train
#         cv : int, optional
#             The number of folds for cross-validation. Default is 5.

#         scoring : callable, optional
#             The scoring function to evaluate predictions.
#             Default is `hamming_loss` (i.e. 0-1 loss) for classification tasks
#             and `mean_squared_error` for regression tasks.

#         random_state : int, optional
#             Seed for the random number generator for shuffling the data.

#         Returns
#         -------
#         mean_score : float
#             The mean score from cross-validation.

#         std_err_mean : float
#             The estimated standard deviation of the mean score.

#         scores : list of float
#             The scores from each fold of the cross-validation.

#         Details
#         ---------
#         see:

#         [1] Aghbalou, A., Bertail, P., Portier, F., & Sabourin, A. (2024).
#         Cross-validation on extreme regions. Extremes, 27(4), 505-555.

#         """
#         scores = []
#         # begin as in the fit method
#         if not callable(self.norm_func):
#             raise ValueError("norm_func must be callable")

#         # check scoring function
#         if scoring is None:
#             if self.task == "classification":
#                 scoring = hamming_loss
#             if self.task == "regression":
#                 scoring = mean_squared_error

#         if not callable(scoring):
#             raise ValueError("scoring must be callable or None")

#         # check and set thresh_train, as in fit method:
#         # Doing so outside the fit method permits to
#         # discard folds where there are too few extremes in the training and
#         # validation set, without fitting the model first and
#         # thus saving computational time.
#         Norm_X = self.norm_func(X)

#         if thresh_train is None:
#             thresh_train = np.quantile(Norm_X, (1 - 1 / np.sqrt(len(Norm_X))))

#         # id_extreme = Norm_X >= thresh_train
#         # k = np.sum(id_extreme)

#         if thresh_predict is None:
#             thresh_predict = thresh_train

#         # which data are considered extreme for training and testing:
#         # logical mask vectors of size len(y)
#         id_extreme_train = Norm_X >= thresh_train
#         id_extreme_predict = Norm_X >= thresh_predict

#         # K-fold train/test indices
#         kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
#         # CV loop
#         for train_index, test_index in kf.split(X):
#             size_train_ex = np.sum(id_extreme_train[train_index])
#             size_predict_ex = np.sum(id_extreme_predict[test_index])
#             if size_train_ex <= 2 or size_predict_ex <= 0:
#                 continue
#             # Split the data into training and testing sets
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]

#             # Fit the model on the training data
#             try:
#                 self.fit(X_train, y_train, k=None, thresh_train=thresh_train)

#             except Exception as e:
#                 ratio_train = size_train_ex / len(train_index)
#                 print(
#                     f"Warning: An error occurred for \
#                 extreme ratio (training) = {ratio_train:4f}. Error: {e}"
#                 )
#                 continue
#             # Predict on the testing data
#             y_pred_extreme, _, mask_test = self.predict(
#                 X_test, thresh_predict=thresh_predict
#             )

#             # Calculate the score
#             result = scoring(y_test[mask_test], y_pred_extreme)
#             scores.append(result)
#         # compute order of magnitude of the error following aghbalou's results
#         # using max(thresh_predict,thresh_train) as a conservative threshold

#         mean_size_train = np.sum(id_extreme_train) * (cv - 1) / cv
#         mean_size_predict = np.sum(id_extreme_predict) / cv
#         #        min_size = np.minimum(mean_size_predict, mean_size_train)
#         order_error = 1 / np.sqrt(mean_size_predict) + 1 / np.sqrt(mean_size_train)
#         #        return np.mean(scores), np.std(scores)/np.sqrt(len(scores)), scores
#         return np.mean(scores), order_error, scores
