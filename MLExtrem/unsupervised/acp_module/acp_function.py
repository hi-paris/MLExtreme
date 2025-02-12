import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def CV_recons_error(dataset, nexpe=100, p=3, idex=None, ratioTest=0.3, method='proposed'):
    """
    Calculate the cross-validated reconstruction error of extreme angles in a dataset.

    Parameters:
    -----------
    dataset : array-like of shape (n_samples, n_features)
        The input dataset.

    nexpe : int, optional
        Number of experiments to perform. Default is 100.

    p : int, optional
        The dimension of the projection space. Default is 3.

    idex : array-like, optional
        Index set of extreme observations.

    ratioTest : float, optional
        The ratio of observations among extremes to be put aside as a 'test set'. Default is 0.3.

    method : str, optional
        The method to use for reconstruction. Options are 'proposed', 'full', and 'part'. Default is 'proposed'.

    Returns:
    --------
    error : array of shape (nexpe,)
        The reconstruction errors for each experiment.
    """
    if idex is None:
        raise ValueError("idex must be provided and cannot be None.")

    k = len(idex)
    n, d = dataset.shape
    radii = np.sqrt(np.sum(dataset**2, axis=1))
    angles = dataset / radii[:, np.newaxis]

    error = np.zeros(nexpe)

    for iexp in range(nexpe):
        # Shuffle the indices and split into training and test sets
        shuffledIdex = np.random.permutation(idex)
        idexTest = shuffledIdex[:int(np.floor(k * ratioTest))]
        idexTrain = shuffledIdex[int(np.floor(k * ratioTest)):]

        exanglesTrain = angles[idexTrain, :]
        exanglesTest = angles[idexTest, :]

        if method == 'proposed':
            # Perform SVD on the training angles and project the test angles
            svTrain = svd(exanglesTrain, full_matrices=False)
            projectTest = exanglesTest @ svTrain[2][:, :p] @ svTrain[2][:, :p].T
            errorTest = np.sum((exanglesTest - projectTest)**2)
            error[iexp] = errorTest

        elif method == 'full':
            # Use the full dataset minus the test set for training
            idTrainFull = np.setdiff1d(np.arange(n), idexTest)
            anglesTrainFull = angles[idTrainFull, :]
            svTrainFull = svd(anglesTrainFull, full_matrices=False)
            projectTestFull = exanglesTest @ svTrainFull[2][:, :p] @ svTrainFull[2][:, :p].T
            errorTestFull = np.sum((exanglesTest - projectTestFull)**2)
            error[iexp] = errorTestFull

        elif method == 'part':
            # Use a subset of the full dataset for training
            idTrainFull_red = np.random.choice(np.setdiff1d(np.arange(n), idexTest), len(idexTrain), replace=False)
            anglesTrainFull_red = angles[idTrainFull_red, :]
            svTrainFull_red = svd(anglesTrainFull_red, full_matrices=False)
            projectTestFull_red = exanglesTest @ svTrainFull_red[2][:, :p] @ svTrainFull_red[2][:, :p].T
            errorTestFull_red = np.sum((exanglesTest - projectTestFull_red)**2)
            error[iexp] = errorTestFull_red

        else:
            raise ValueError("method must be 'proposed', 'full', or 'part'.")

    return error

def empMomentFourier(data, freq, exratiomax, exratiomin, graph=True, selectK=50):
    """
    Check the weak convergence of the distribution of angles conditional to a large radius.

    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        The input data.

    freq : int
        The frequency of the Fourier function.

    exratiomax : float
        The maximum ratio of extreme values to consider.

    exratiomin : float
        The minimum ratio of extreme values to consider.

    graph : bool, optional
        Whether to plot the results. Default is True.

    selectK : int, optional
        The value of k to highlight in the plot. Default is 50.

    Returns:
    --------
    values : array of shape (2, kmax - kmin + 1)
        The mean and standard deviation of the projections.
    """
    n, d = data.shape

    # Define the Fourier function based on the frequency
    if freq % 2 == 0:
        fourierFun = np.cos(freq * np.arange(1, d + 1) / d * 2 * np.pi) * np.sqrt(2)
    else:
        fourierFun = np.sin(freq * np.arange(1, d + 1) / d * 2 * np.pi) * np.sqrt(2)

    radii = np.sqrt(np.sum(data**2, axis=1))
    permut = np.argsort(radii)[::-1]  # Indices sorted by decreasing radii
    sradii = np.sort(radii)[::-1]  # Sorted radii
    angles = data / radii[:, np.newaxis]  # Scale data by radii

    kmin = int(np.floor(exratiomin * n))
    kmax = int(np.floor(exratiomax * n))

    values = np.zeros((2, kmax - kmin + 1))
    sortedAngles = angles[permut[:kmax], :]
    projections = np.abs(sortedAngles @ fourierFun)  # Absolute value of projection
    means = np.cumsum(projections) / np.arange(1, kmax + 1)
    variances = np.cumsum((projections - means)**2) / np.arange(1, kmax + 1)
    sdev_est = np.sqrt(variances) / np.sqrt(np.arange(1, kmax + 1))

    values[0, :] = means[kmin:kmax]
    values[1, :] = sdev_est[kmin:kmax]

    if graph:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(kmin, kmax + 1), values[0, :], label=f"Frequency = {freq}", color='black')
        plt.fill_between(np.arange(kmin, kmax + 1),
                         values[0, :] - 1.64 * values[1, :],
                         values[0, :] + 1.64 * values[1, :], color='blue', alpha=0.3)
        plt.axvline(x=selectK, linestyle='--', color='gray')
        plt.xlabel('k')
        plt.ylabel(r'$E | < \Theta_t, h_j > |$')
        plt.xticks([kmin, selectK // 2, selectK, selectK * 3 // 2, 2 * selectK, 3 * selectK, 4 * selectK, kmax])
        plt.yticks(np.round([np.min(values[0, :]), np.max(values[0, :])], 2))
        plt.grid(True)
        plt.legend()
        plt.show()

    return values
