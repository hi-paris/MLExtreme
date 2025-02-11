import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

class Regressor:
    """
    A custom regressor that uses a specified model and a normalization function to predict continuous values.

    Parameters
    ----------
    model : object
        The model to be used for regression. Must have a `.fit` method.

    norm_func : callable, optional
        A function that computes a norm of the input data. Default is L2 norm.

    k : int, optional
        A parameter used to determine the threshold for selecting extreme points.
        If k=0, it is automatically set to 4 * sqrt(n_samples).

    Attributes
    ----------
    threshold : float
        The threshold value used to select extreme points.

    model : object
        The model used for regression.

    norm_func : callable
        The normalization function.

    k : int
        The parameter k.
    """

    def __init__(self, model, norm_func=None, k=0):
        self.model = model
        self.norm_func = norm_func if norm_func else lambda x: np.linalg.norm(x, axis=1)
        self.k = k
        self.threshold = 0

    def fit(self, X_train, X_test, y_train):
        """
        Fit the model using extreme points from the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.

        X_test : array-like of shape (n_samples, n_features)
            The test input samples.

        y_train : array-like of shape (n_samples,)
            The target values for training.

        Returns
        -------
        threshold : float
            The computed threshold value.

        X_train_unit : array-like of shape (n_extreme_samples, n_features)
            The normalized extreme points from the training data.
        """
        if not callable(self.norm_func):
            raise ValueError("norm_func must be callable")

        if self.k == 0:
            self.k = 4 * int(np.sqrt(len(X_train) + len(X_test)))

        Norm_X_train = self.norm_func(X_train)
        self.threshold = np.percentile(Norm_X_train, 100 * (1 - self.k / len(Norm_X_train)))
        X_train_extrem = X_train[Norm_X_train >= self.threshold]

        X_train_unit = X_train_extrem / Norm_X_train[Norm_X_train >= self.threshold][:, np.newaxis]
        y_train_extrem = y_train[Norm_X_train >= self.threshold]

        self.model.fit(X_train_unit, y_train_extrem)

        return self.threshold, X_train_unit

    def predict(self, X_test, threshold=None):
        """
        Predict values for the test data using extreme points.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.

        threshold : float, optional
            The threshold value to select extreme points. If not provided,
            the threshold computed during fitting is used.

        Returns
        -------
        y_pred : array-like of shape (n_extreme_samples,)
            The predicted values for the extreme points.

        mask_test : array-like of shape (n_samples,)
            A boolean mask indicating which points are extreme.

        X_test_unit : array-like of shape (n_extreme_samples, n_features)
            The normalized extreme points from the test data.
        """
        if threshold is None:
            if self.threshold == 0:
                raise ValueError("Model has not been fitted yet.")
            threshold = self.threshold

        Norm_X_test = self.norm_func(X_test)
        mask_test = Norm_X_test >= threshold
        X_test_extrem = X_test[mask_test]

        X_test_unit = X_test_extrem / Norm_X_test[mask_test][:, np.newaxis]
        y_pred = self.model.predict(X_test_unit)

        return y_pred, mask_test, X_test_unit

    def plot_predictions(self, X, y_test, y_pred):
        """
        Display predicted vs actual values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y_test : array-like of shape (n_samples,)
            The true values.

        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', marker='o')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.show()

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the regression model using Mean Squared Error.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The true values.

        y_pred : array-like of shape (n_samples,)
            The predicted values.

        Returns
        -------
        mse : float
            The mean squared error of the predictions.
        """
        return mean_squared_error(y_true, y_pred)

    def cross_validate(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform cross-validation and return the mean score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        cv : int, optional
            The number of folds for cross-validation.

        scoring : str, optional
            The scoring metric for cross-validation.

        Returns
        -------
        mean_score : float
            The mean score from cross-validation.
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return np.mean(scores)
