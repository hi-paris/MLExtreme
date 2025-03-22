import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold



class Regressor:
    """
    A custom regressor that uses a specified model and a normalization function
    to predict continuous values.

    Parameters
    ----------
    model : object
        The model to be used for regression. Must have a `.fit` method.

    norm_func : callable, optional
        A function that computes a norm of the input data. Default is L2 norm.

    Attributes
    ----------
    model : object
        The model used for regression.

    norm_func : callable
        The normalization function.

    thresh_train: float.
        Set to None at initialization and later set to the training threshold
        after fitting.

    ratio_train: float, between 0 and 1
        Set to None at initialization and later set to the ratio k/n of extreme
        training sample divided by the training sample size.
    """

    def __init__(self, model, norm_func=None, k=0):
        self.model = model
        self.norm_func = (norm_func if norm_func
                          else lambda x: np.linalg.norm(x, axis=1))
        self.thresh_train = None
        self.ratio_train = None

    def fit(self, X_train,  y_train,  k=None, thresh_train=None):
        """
        Fit the model using extreme points from the training data where
        the covariate norm exceeds a high threshold.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.

        y_train : array-like of shape (n_samples,)
            The target values for training.
        
        k : int, optional
            The number of extreme samples used to train the model. Data are
            ordered according to their norm (`norm_func') and the k largest are
            selected.

        thresh_train:  float, optional.
            The radial threshold above which training samples are considered
            extreme for training

        Details
        ------

        If `thresh_train=None` and `k=None`, `k` is set to sqrt(n_samples)
          and thresh_train is set to the empirical 1 - k/n_samples quantile of
          the norms of the input X_train

        Setting both thresh_train and k at the same time raises an error.

        Returns
        -------
        thresh_train : float
            The default threshold for further prediction,
            set to the training threshold. Also stored in model's
            attribute .thresh_train

        ratio_train: float
            The ratio of extreme points used for training.
            Also stored in model's attribute .ratio_train

        X_train : array-like of shape (n_extreme_samples, n_features)
            The  extreme covariate points actually  used to train the model.
        """
        if not callable(self.norm_func):
            raise ValueError("norm_func must be callable")

        # check and set k and thresh_train
        if k is not None and thresh_train is not None:
            raise ValueError(
                "k and thresh_train cannot both be set at the same time")

        Norm_X_train = self.norm_func(X_train)

        if k is None and thresh_train is None:
            k = int(np.sqrt(len(X_train)))

        if thresh_train is None:
            thresh_train = np.percentile(Norm_X_train,
                                         100 * (1 - k / len(Norm_X_train)))

        id_extreme = Norm_X_train >= thresh_train
        if k is None:
            k = np.sum(id_extreme)

        # update model with training threshold and k/n ratio
        self.thresh_train = thresh_train
        self.ratio_train = k/len(Norm_X_train)

        X_train_extreme = X_train[id_extreme]
        X_train_extreme_unit = (X_train_extreme /
                                Norm_X_train[id_extreme][:, np.newaxis])
        y_train_extreme = y_train[id_extreme]

        self.model.fit(X_train_extreme_unit, y_train_extreme)

        return thresh_train, k/len(Norm_X_train), X_train_extreme

    def predict(self, X_test, thresh_predict=None):
        """
        Predict labels for the extreme test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.

        thresh_predict : float, optional
            The threshold value to select extreme points. If not provided,
            the threshold used during fitting


        Returns
        -------
        y_pred : array-like of shape (n_extreme_samples,)
            The predicted labels for the extreme points.
        
        X_test : array-like of shape (n_extreme_samples, n_features)
            The  extreme points from the test data.
        
        mask_test : array-like of shape (n_samples,)
            A boolean mask indicating which points are extreme.

        """
        if self.thresh_train is None:
            raise ValueError("Model has not been fitted yet.")

        if thresh_predict is None:
            thresh_predict = self.thresh_train

        Norm_X_test = self.norm_func(X_test)
        mask_test = Norm_X_test >= thresh_predict
        X_test_extreme = X_test[mask_test]

        X_test_extreme_unit = (X_test_extreme /
                               Norm_X_test[mask_test][:, np.newaxis])
        y_pred_extreme = self.model.predict(X_test_extreme_unit)

        return y_pred_extreme,  X_test_extreme, mask_test

    def plot_predictions(self, y_true, y_pred): 
        """
        Display predicted vs actual values.

        Parameters

        y_true : array-like of shape (n_samples,)
            The true values.

        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, color='blue', marker='o')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.show()

    def cross_validate(self, X, y, k=None, thresh_train=None,
                       thresh_predict=None, cv=5,
                       scoring=mean_squared_error, random_state=None):
        """
        Perform cross-validation and return the mean  score,
        the estimated standard deviation of the mean, and the  array of scores.
        Following Aghbalou et al's CV scheme (K-fold).
        Differently from the paper the radial threshold for prediction/test
        may be chosen different from the radial threshold for training.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        k : int, optional
            The number of extreme values to consider.

        thresh_train : float, optional
            The threshold for considering extreme values during training.

        thresh_predict : float, optional
            The threshold for considering extreme values during prediction.

        cv : int, optional
            The number of folds for cross-validation. Default is 5.

        scoring : callable, optional
            The scoring function to evaluate predictions.
            Default is accuracy_score.

        random_state : int, optional
            Seed for the random number generator for shuffling the data.

        Returns
        -------
        mean_score : float
            The mean accuracy score from cross-validation.

        std_err_mean : float
            The estimated standard deviation of the mean accuracy score.

        scores : list of float
            The accuracy scores from each fold of the cross-validation.

        Details
        ---------
        see TODO ADD REF
        """
        scores = []
        # begin as in the fit method
        if not callable(self.norm_func):
            raise ValueError("norm_func must be callable")

        # check scoring function
        if not callable(scoring):
            raise ValueError("scoring must be callable")

        Norm_X = self.norm_func(X)

        # check and set k and thresh_train, as in fit method:
        # necessary to do that outside the fit method to discard folds
        # where there are too few extremes without fitting the model
        if k is not None and thresh_train is not None:
            raise ValueError(
                "k and thresh_train cannot both be set at the same time")

        if k is None and thresh_train is None:
            k = int(np.sqrt(len(X)))

        if thresh_train is None:
            thresh_train = np.percentile(Norm_X,
                                         100 * (1 - k / len(Norm_X))
                                         )

        id_extreme = Norm_X >= thresh_train
        if k is None:
            k = np.sum(id_extreme)

        if thresh_predict is None:
            thresh_predict = thresh_train

        # which data are considered extreme for training and testing
        # logical mask vectors of size len(y)
        id_extreme_train = (Norm_X >= thresh_train)
        id_extreme_predict = (Norm_X >= thresh_predict)

        # K-fold train/test indices
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        # CV loop
        for train_index, test_index in kf.split(X):
            size_train_ex = np.sum(id_extreme_train[train_index])
            size_predict_ex = np.sum(id_extreme_predict[test_index])
            if size_train_ex <= 2 or size_predict_ex <= 0:
                continue
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            self.fit(X_train, y_train, thresh_train=thresh_train)

            # Predict on the testing data
            y_pred_extreme,  _, mask_test = \
                self.predict(X_test, thresh_predict=thresh_predict)

            # Calculate the score
            result = scoring(y_test[mask_test], y_pred_extreme)
            scores.append(result)
            
        return np.mean(scores), np.std(scores)/np.sqrt(len(scores)), scores
    # def evaluate(self,y_true, y_pred):
    #     """
    #     Evaluate the regression model using Mean Squared Error.

    #     Parameters
    #     ----------
    #     y_true : array-like of shape (n_samples,)
    #         The true values.

    #     y_pred : array-like of shape (n_samples,)
    #         The predicted values.

    #     Returns
    #     -------
    #     mse : float
    #         The mean squared error of the predictions.
    #     """
    #     return mean_squared_error(y_true, y_pred)

    # def cross_validate(self, X, y, cv=5, scoring='neg_mean_squared_error'):
    #     """
    #     Perform cross-validation and return the mean score.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #         The input samples.

    #     y : array-like of shape (n_samples,)
    #         The target values.

    #     cv : int, optional
    #         The number of folds for K-fold cross-validation.

    #     scoring : str, optional
    #         The scoring metric for cross-validation.

    #     Returns
    #     -------
    #     mean_score : float
    #         The mean score from cross-validation.
    #     """
    #     scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
    #     return np.mean(scores)
