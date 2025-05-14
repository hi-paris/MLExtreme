"""
Plotting utilities for the `supervised` module.
"""

import matplotlib.pyplot as plt

def plot_classif(X, y_test, y_pred, title=None):
    """
    Display points classified according to predictions and actual values.
    The covariate points are projected onto their first and last components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input samples for plotting.

    y_test : array-like, shape (n_samples,)
        True labels for the input samples.

    y_pred : array-like, shape (n_samples,)
        Predicted labels for the input samples.

    title : str, optional
        Title of the plot. Default is "Classification Results".
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X[:, 0][(y_pred == 0) & (y_test == 0)],
        X[:, -1][(y_pred == 0) & (y_test == 0)],
        color="blue",
        marker="o",
        label="True Negative",
        alpha=0.5,
    )
    plt.scatter(
        X[:, 0][(y_pred == 0) & (y_test == 1)],
        X[:, -1][(y_pred == 0) & (y_test == 1)],
        color="red",
        marker="x",
        label="False Negative",
        alpha=0.5,
    )
    plt.scatter(
        X[:, 0][(y_pred == 1) & (y_test == 0)],
        X[:, -1][(y_pred == 1) & (y_test == 0)],
        color="blue",
        marker="x",
        label="False Positive",
        alpha=0.5,
    )
    plt.scatter(
        X[:, 0][(y_pred == 1) & (y_test == 1)],
        X[:, -1][(y_pred == 1) & (y_test == 1)],
        color="red",
        marker="o",
        label="True Positive",
        alpha=0.5,
    )
    if title is None:
        title = "Classification Results"
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature p")
    plt.legend()
    plt.show()

def plot_predictions(y_true, y_pred, title=None):
    """
    Display predicted vs actual values.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True values for comparison.

    y_pred : array-like, shape (n_samples,)
        Predicted values for comparison.

    title : str, optional
        Title of the plot. Default is "True vs Predicted Values".
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color="blue", marker="o")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    if title is None:
        title = "True vs Predicted Values"
    plt.title(title)
    plt.show()
# """
# Plotting utilities for the `supervised` module
# """

# import matplotlib.pyplot as plt


# def plot_classif(X, y_test, y_pred, title=None):
#     """
#     Display points classified according to predictions and actual values.
#     the covariate points are projected onto there first and last component.

#     Parameters
#     ----------
#     X : array-like of shape (n_samples, n_features)
#         The input samples.

#     y_test : array-like of shape (n_samples,)
#         The true labels.

#     y_pred : array-like of shape (n_samples,)
#         The predicted labels.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         X[:, 0][(y_pred == 0) & (y_test == 0)],
#         X[:, -1][(y_pred == 0) & (y_test == 0)],
#         color="blue",
#         marker="o",
#         label="True Negative",
#         alpha=0.5,
#     )
#     plt.scatter(
#         X[:, 0][(y_pred == 0) & (y_test == 1)],
#         X[:, -1][(y_pred == 0) & (y_test == 1)],
#         color="red",
#         marker="x",
#         label="False Negative",
#         alpha=0.5,
#     )
#     plt.scatter(
#         X[:, 0][(y_pred == 1) & (y_test == 0)],
#         X[:, -1][(y_pred == 1) & (y_test == 0)],
#         color="blue",
#         marker="x",
#         label="False Positive",
#         alpha=0.5,
#     )
#     plt.scatter(
#         X[:, 0][(y_pred == 1) & (y_test == 1)],
#         X[:, -1][(y_pred == 1) & (y_test == 1)],
#         color="red",
#         marker="o",
#         label="True Positive",
#         alpha=0.5,
#     )
#     if title is None:
#         title = "Classification Results"
#     plt.title(title)
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature p")
#     plt.legend()
#     plt.show()


# def plot_predictions(y_true, y_pred, title=None):
#     """
#     Display predicted vs actual values.

#     Parameters

#     y_true : array-like of shape (n_samples,)
#         The true values.

#     y_pred : array-like of shape (n_samples,)
#         The predicted values.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_true, y_pred, color="blue", marker="o")
#     plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
#     plt.xlabel("True Values")
#     plt.ylabel("Predictions")
#     if title is None:
#         title = "True vs Predicted Values"
#     plt.title(title)
#     plt.show()
