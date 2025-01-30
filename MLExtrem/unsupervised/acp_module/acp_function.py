import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, norm
from sklearn.linear_model import LinearRegression


## Recons Part


def CV_recons_error(dataset, nexpe=100, p=3, idex=None, ratioTest=0.3, method='proposed'):
    ## p : the dimension of the projection space.
    ## idex: index set of extreme observations
    ## ratioTest: the ratio of observations among extremes to be put aside as a 'test set'.

    ## PA :  Add test raise error
    
    k = len(idex)
    n, d = dataset.shape
    radii = np.sqrt(np.sum(dataset**2, axis=1))
    angles = dataset / radii[:, np.newaxis]
    
    error = np.zeros(nexpe)
    
    for iexp in range(nexpe):
        shuffledIdex = np.random.permutation(idex)
        idexTest = shuffledIdex[:int(np.floor(k * ratioTest))]
        idexTrain = shuffledIdex[int(np.floor(k * ratioTest)):]
        
        exanglesTrain = angles[idexTrain, :]
        exanglesTest = angles[idexTest, :]
        if method=='proposed': ## PA : Rename à faire
            svTrain = np.linalg.svd(exanglesTrain, full_matrices=False)
            projectTest = exanglesTest @ svTrain[2][:, :p] @ svTrain[2][:, :p].T
            errorTest = np.sum((exanglesTest - projectTest)**2)
            error[iexp] = errorTest
        
        elif method=='full': ## PA : Rename à faire
            idTrainFull = np.setdiff1d(np.arange(n), idexTest)
            anglesTrainFull = angles[idTrainFull, :]
            svTrainFull = np.linalg.svd(anglesTrainFull, full_matrices=False)
            projectTestFull = exanglesTest @ svTrainFull[2][:, :p] @ svTrainFull[2][:, :p].T
            errorTestFull = np.sum((exanglesTest - projectTestFull)**2)
            error[iexp] = errorTestFull
        
        else : #elif method=='part' ## PA : Rename à faire 
            idTrainFull_red = np.random.choice(np.setdiff1d(np.arange(n), idexTest), len(idexTrain), replace=False)
            anglesTrainFull_red = angles[idTrainFull_red, :]
            svTrainFull_red = np.linalg.svd(anglesTrainFull_red, full_matrices=False)
            projectTestFull_red = exanglesTest @ svTrainFull_red[2][:, :p] @ svTrainFull_red[2][:, :p].T
            errorTestFull_red = np.sum((exanglesTest - projectTestFull_red)**2)
            error[iexp] = errorTestFull_red
    
    return error





##Fourier Part


def empMomentFourier(data, freq, exratiomax, exratiomin, graph=True, selectK=50):
    ## checking weak convergence of the distribution of angles conditional to a large radius, 
    ## as the radial threshold goes to infinity; Here we use proposition .. from the paper  
    ## and consider univariate projection on arbitrary functions $h$. 
    ## For simplicity here we restrict ourselves to h being a sinusoidal function with frequency 'freq'
    
    n, d = data.shape
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
    print(kmin,kmax)
    print(means)
    values[0, :] = means[kmin:kmax]
    values[1, :] = sdev_est[kmin:kmax]
    
    if graph:
        plt.plot(np.arange(kmin, kmax + 1), values[0, :], label=f"j = {freq}", color='black')
        plt.fill_between(np.arange(kmin, kmax + 1), 
                         values[0, :] - 1.64 * values[1, :], 
                         values[0, :] + 1.64 * values[1, :], color='blue', alpha=0.3)
        plt.axvline(x=selectK, linestyle='--', color='gray')
        plt.xlabel('k')
        plt.ylabel(r'$E | < \Theta_t, h_j > |$')
        plt.xticks([kmin, selectK // 2, selectK, selectK * 3 // 2, 2 * selectK, 3 * selectK, 4 * selectK, kmax])
        plt.yticks(np.round([np.min(values[0, :]), np.max(values[0, :])], 2))
        plt.grid(True)
        plt.show()

    return values




#######################
## begin NOT USED
#######################

#Recons moment

def CV_recons_error_X(dataset, nexpe=100, p=3, idex=None, ratioTest=0.3):
    ## reconstruction error of extreme data (error on X = R * Theta, not Theta) 
    k = len(idex)
    n = dataset.shape[0]
    radii = np.linalg.norm(dataset, axis=1)  # Calculate the radii
    angles = dataset / radii[:, np.newaxis]   # Normalize the dataset

    error_ex = np.zeros(nexpe)
    error_ext_X = np.zeros(nexpe) 
    error_full = np.zeros(nexpe)
    error_full_reduced = np.zeros(nexpe)

    for iexp in range(nexpe):
        ## 1  KLE of extreme angles -- promoted method
        shuffledIdex = np.random.permutation(idex)  # Shuffle the indices
        idexTest = shuffledIdex[:int(np.floor(k * ratioTest))]
        
        idexTrain = shuffledIdex[int(np.floor(k * ratioTest)):]  # Training indices
        exanglesTrain = angles[idexTrain, :]  # Training angles
        scaledExanglesTest = angles[idexTest, :]  # Test angles
        meanAnglesTest = np.mean(angles[idexTest, :], axis=0)  # Mean of test angles
        
        svTrain = np.linalg.svd(exanglesTrain, full_matrices=False)  # SVD on training angles
        projectTest = scaledExanglesTest @ svTrain[0][:, :p] @ svTrain[0][:, :p].T  # Project test angles
        
        reconsAnglesTest = projectTest  ##+ np.ones(len(idexTest)) @ meanAnglesTest
        reconsX = np.diag(radii[idexTest]) @ reconsAnglesTest  # Reconstructed X
        errorTest = np.sum((dataset[idexTest, :] - reconsX) ** 2)  # Reconstruction error
        error_ex[iexp] = errorTest
        
        ## 2. KLE of the extreme dataset with X (not theta)
        svTrain_ext_X = np.linalg.svd(dataset[idexTrain, :], full_matrices=False)  # SVD on training dataset
        scaledTest = dataset[idexTest, :]  # Test dataset
        projectTest_ext_X = scaledTest @ svTrain_ext_X[0][:, :p] @ svTrain_ext_X[0][:, :p].T  # Project test dataset
        errorTest_ext_X = np.sum((scaledTest - projectTest_ext_X) ** 2)  # Error calculation
        error_ext_X[iexp] = errorTest_ext_X
        
        ## 3 . KLE of the full dataset (minus the test set), with  X (not Theta)
        idTrainFull = np.setdiff1d(np.arange(1, n + 1), idexTest) - 1  # Full training indices
        svTrainFull = np.linalg.svd(dataset[idTrainFull, :], full_matrices=False)  # SVD on full training dataset
        projectTestFull = scaledTest @ svTrainFull[0][:, :p] @ svTrainFull[0][:, :p].T  # Project test dataset
        errorTestFull = np.sum((scaledTest - projectTestFull) ** 2)  # Error calculation
        error_full[iexp] = errorTestFull
        
        ## 4. KLE of the full dataset of reduced size (same training size as method 1), with X, not theta
        idTrainFull_red = np.random.choice(np.setdiff1d(np.arange(1, n + 1), idexTest), size=len(idexTrain), replace=False) - 1  # Reduced training indices
        svTrainFull_red = np.linalg.svd(dataset[idTrainFull_red, :], full_matrices=False)  # SVD on reduced training dataset
        projectTestFull_red = scaledTest @ svTrainFull_red[0][:, :p] @ svTrainFull_red[0][:, :p].T  # Project test dataset
        errorTestFull_red = np.sum((scaledTest - projectTestFull_red) ** 2)  # Error calculation
        error_full_reduced[iexp] = errorTestFull_red

    return {
        'error_ex': error_ex,
        'error_ext_X': error_ext_X,
        'error_full': error_full,
        'error_full_reduced': error_full_reduced
    }

## Fourrier 

def empMoment(data, Orders=[1, 2], j=None, exratiomax=1, exratiomin=0, graph=True):  ## NOT USED 
    ## checking weak convergence of the distribution of angles conditional to a large radius, 
    ## as the radial threshold goes to infinity; Here we use proposition .. from the paper
    ## and consider univariate projection on arbitrary functions $h$. 
    ## For simplicity here we restrict ourselves to h being an indicator function 
    ## of an interval around a time stamp, thus we investigate weak convergence 
    ## of single components (j) of the discretized angular variable.
    
    n, d = data.shape
    radii = np.sqrt(np.sum(data**2, axis=1))
    permut = np.argsort(radii)[::-1]  # Indices sorted by decreasing radii
    sradii = np.sort(radii)[::-1]  # Sorted radii
    angles = (1 / radii).reshape(-1, 1) * data  # Scaling the data by radii

    kmin = int(np.floor(exratiomin * n))
    kmax = int(np.floor(exratiomax * n))
    vect_k = np.arange(kmax, kmin - 1, -1)  # Vector of k values

    values = np.zeros((2 * len(Orders), len(vect_k)))  # Matrix to store results

    for idk in range(len(vect_k)):
        inds = permut[:vect_k[idk]]  # Indices of largest radii
        trigovec = np.sin(2 * np.pi * j * np.arange(1, d+1) / d)
        extr_angles = np.dot(angles[inds, :], trigovec)  # Projection on trigovec

        for i, order in enumerate(Orders):
            result = np.mean(extr_angles**order)
            sdev_est = np.std(extr_angles**order) / np.sqrt(len(extr_angles))
            values[2*i, idk] = result
            values[2*i + 1, idk] = sdev_est

    if graph:
        for i in range(len(Orders)):
            plt.plot(vect_k, values[2*i, :], label=f"Order {Orders[i]}")
            plt.fill_between(vect_k, 
                             values[2*i, :] - 1.64 * values[2*i + 1, :], 
                             values[2*i, :] + 1.64 * values[2*i + 1, :], 
                             color='blue', alpha=0.3)
            plt.title(f"Projection for component {j}")
            plt.xlabel("k")
            plt.ylabel(r"$E<\Theta_t, h_j>$")
            plt.grid(True)
            plt.legend()
            plt.show()

    return values


####################
### end  NOT USED
####################
