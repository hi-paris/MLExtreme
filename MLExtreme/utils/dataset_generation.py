# from scipy import stats as stat
# from scipy.stats import pareto
import numpy as np
from .EVT_basics import gen_dirimix, normalize_param_dirimix


def gen_rv_dirimix(alpha=1, Mu =np.array([ [0.5 ,  0.5]]),
                   wei=np.array([1]), lnu=np.array([2]),
                   scale_weight_noise=1,
                   index_weight_noise=None, Mu_bulk=None, size=1):
    """
    Generate `n` points where each point `X_i = R_i * Theta_i`.

    Parameters:
    -----------
    size : int (optional)
        Number of samples to generate.
    alpha : float
        Shape parameter of the Pareto distribution.
    Mu : ndarray, shape (k,d)
        Means for the Dirichlet mixture components.
    wei : ndarray, shape (k,)
        Weights for each mixture component.
    lnu : ndarray, shape (k,)
        Log-scale parameters for the Dirichlet mixture components.
    Mu_bulk: a parameter matrix of same dimension as Mu (optional)
        ruling the angular ditribution for small to moderate radii
    scale_weight_noise : float, optional
        Scaling factor (>0) for the angular noise. Default is 1.
    index_weight_noise : float, optional
        Exponent ruling how the noise vanishes for large R. Default is -alpha.

    Details
    ___________ 
        The noise influence vanishes proportional to
        ` (R/scale_weight_noise)**(-index_weight_noise) `
        where `R` is the  norm of the sample point.
    Returns:
    --------
    ndarray, shape (n, d)
        Array of points generated from the Pareto and Dirichlet mixture.
    """
    n = size
    if index_weight_noise is None:
        index_weight_noise = alpha

    R = np.random.pareto(alpha, size=n)+1
    Theta = gen_dirimix(Mu=Mu,  wei=wei, lnu=lnu, size=n)
    dim = np.shape(Mu)[1]
    # if Mu_bulk is None:
    #     Mu_bulk = (1 - Mu)/(dim - 1)
    if Mu_bulk is None:
        Mu_bulk_0 = np.maximum((2/dim - Mu), 10**(-5))
        r_0 = np.sum(Mu_bulk_0, axis=1)
        Mu_bulk = Mu_bulk_0 / r_0.reshape(-1,1)
    
    Noise = gen_dirimix(Mu=Mu_bulk, wei=wei, lnu=lnu, size=n)
    # gen_dirichlet(n, ) np.ones((n, np.shape(Mu)[0])))
    w = np.minimum(1,  (R/scale_weight_noise) ** (-index_weight_noise))
    newTheta = (1 - w[:, np.newaxis]) * Theta + w[:, np.newaxis] * Noise
    X = R[:, np.newaxis] * newTheta

    return X


def gen_classif_data_diriClasses(mu0=np.array([0.7, 0.3]),
                                 lnu=None, alpha=4,
                                 index_weight_noise=1,
                                 size=10):

    mu1 = 1-mu0
    Mu_temp = np.array([mu0, mu1])
    wei_temp = 0.5 * np.ones(2)
    Mu, wei = normalize_param_dirimix(Mu_temp, wei_temp)
    if lnu is None:
        lnu = np.log(2 / np.min(Mu, axis=1))        
    size0 = int(size*wei[0])
    size1 = int(size*wei[1])
    Mu0 = Mu[0, :].reshape(1, -1)
    Mu1 = Mu[1, :].reshape(1, -1)
    Dim = np.shape(Mu)[1]
    data0 = gen_rv_dirimix(alpha, Mu0,
                           wei=np.array([1]),
                           lnu=np.array([lnu[0]]),
                           scale_weight_noise= 10**(1/alpha), #np.sqrt(Dim),
                           index_weight_noise=index_weight_noise,
                           size=size0)
    data1 = gen_rv_dirimix(alpha, Mu1,
                           wei=np.array([1]),
                           lnu=np.array([lnu[1]]),
                           scale_weight_noise= 10**(1/alpha), #np.sqrt(Dim),
                           index_weight_noise=index_weight_noise,
                           size=size1)
    y = np.vstack((np.zeros(size0).reshape(-1, 1),
                  np.ones(size1).reshape(-1, 1))).flatten()
    X = np.vstack((data0, data1))
    permut = np.random.permutation(size0 + size1)
    X = X[permut, :]
    y = y[permut]
    return X,  y


# ## target generation for regression models:
# specific instance of additive noise nodel considered in Huet et al. 
def tail_reg_fun_default(angle):
    """
    Default tail regression function for generating target values.

    This function computes the dot product between the input angles and a
    predefined vector `beta`. The vector `beta` is designed to emphasize
    certain directions in the angular space, which is useful for generating
    target values that reflect the tail regression characteristics.

    Parameters
    ----------
    angle : np.ndarray
        A 1D or 2D array representing angular components of the data.
        Each element corresponds to an angle in the covariate space. If
        `angle` is a 2D array, each row represents a different observation.

    Returns
    -------
    np.ndarray
        The result of the dot product between the `angle` array and the
        vector `beta`. This output represents the tail regression values
        for the input angles, which can be used to generate target values
        in a regression model.

    Notes
    -----
    The `beta` vector is constructed by concatenating two segments: the
    first half consists of values of 10, and the second half consists of
    values of 0.1. 
    """
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([10 * np.ones(p1), 0.1 * np.ones(p - p1)])
    return np.dot(angle, beta)


def bulk_reg_fun_default(angle):
    """
    Default bulk regression function for generating target values.

    This function computes the dot product between the input angles and a
    predefined vector `beta`. 

    Parameters
    ----------
    angle : np.ndarray
        A 1D or 2D array representing angular components of the data.
        Each element corresponds to an angle in the covariate space. If
        `angle` is a 2D array, each row represents a different
        observation.

    Returns
    -------
    np.ndarray
        The result of the dot product between the `angle` array and the
        vector `beta`. This output represents the bulk regression values
        for the input angles, which can be used to generate target values
        in a regression model.

    Notes
    -----
    The `beta` vector is constructed by concatenating two segments: the
    first half consists of values of 0.1, and the second half consists of
    values of 10. 
    """
    if angle.ndim == 1:
        # angle is a 1D array
        p = angle.shape[0]
    elif angle.ndim == 2:
        # angle is a 2D array
        p = angle.shape[1]
    p1 = int(p / 2)
    beta = np.concatenate([0.1 * np.ones(p1), 10 * np.ones(p - p1)])
    return np.dot(angle, beta)

def bulk_decay_fun_default(radius, rv_index):
    """
    Default bulk decay function for generating target values.

    This function applies a decay transformation to the input radii,
    which can be used to generate target values in a regression model.
    The decay is determined by the `rv_index` parameter, allowing for
    customization of the decay rate.

    Parameters
    ----------
    radius : np.ndarray
        A 1D array representing the radial distances. Each element
        corresponds to the radius of a data point in the covariate space.

    rv_index : float
        A positive float that determines the rate of decay. Higher values
        result in faster decay. This parameter allows for adjustment of
        the decay function's sensitivity to changes in radius.

    Returns
    -------
    np.ndarray
        An array containing the result of applying the decay function to
        each element in the `radius` array. The output represents the
        decayed values, which can be used to weight the contributions of
        different data points in a regression model.
    """
    return 1 / (radius)**(rv_index)


def gen_target_CovariateRV(X, tail_reg_fun=tail_reg_fun_default,
                           bulk_reg_fun=bulk_reg_fun_default,
                           bulk_decay_fun=bulk_decay_fun_default,
                           param_decay_fun=2):
    """
    Generate target values for covariate random variables.

    This function generates target values for a given set of covariate data
    by combining tail and bulk regression functions with a decay function.
    The resulting target values are influenced by both the tail and bulk
    regression functions, weighted by the decay function.

    Parameters
    ----------
    X : np.ndarray
        A 2D array representing covariate data. Each row corresponds to a
        different observation, and each column corresponds to a different
        covariate.

    tail_reg_fun : function, optional
        A function that computes the tail regression values based on the
        angular components of the covariate data. Default is
        `tail_reg_fun_default`.

    bulk_reg_fun : function, optional
        A function that computes the bulk regression values based on the
        angular components of the covariate data. Default is
        `bulk_reg_fun_default`.

    bulk_decay_fun : function, optional
        A function that computes the decay weights based on the radial
        components of the covariate data. Default is
        `bulk_decay_fun_default`.

    param_decay_fun : any, optional
        An additional parameter to be passed to `bulk_decay_fun`. This can
        be used to customize the behavior of the decay function. Default is
        2.

    Returns
    -------
    y : np.ndarray
        An array of generated target values corresponding to the input
        covariate data. The target values are computed as a weighted
        combination of the tail and bulk regression values, with added
        Gaussian noise.
    """
    rad = np.linalg.norm(X, axis=1)
    ang = X / rad[:, np.newaxis]
    n = len(rad)
    noise = np.random.normal(loc=0, scale=0.01, size=n)
    tail_mean = tail_reg_fun(ang).flatten()
    bulk_mean = bulk_reg_fun(ang).flatten()
    bulk_weight = bulk_decay_fun(rad, param_decay_fun).flatten()
    ystar = (1 - bulk_weight)*tail_mean + bulk_weight * bulk_mean
    y = ystar + noise
    return y


# ## toy example for functional PCA
def gen_rv_functional_data(num_samples, grid, alpha, alphanoise, scalenoise=5,
                           om1=1, om2=2, om3=3, om4=4):
    """
    Generate random regularly varying functions on a grid using
    Pareto-distributed coefficients and noise components.

    Parameters:
    -----------
    num_samples : int
        Number of samples to generate.
    grid : array-like
        1D array representing the grid over which the basis functions are
        evaluated.
    alpha : float
        Shape parameter for the 'signal' Pareto-distributed variables.
    alphanoise : float
        Shape parameter for the noise Pareto-distributed variables.
    scalenoise : float, optional
        Scaling factor for the noise components. Default is 5.
    om1, om2, om3, om4 : float, optional
        Frequencies used in the sine and cosine basis functions.
        Defaults are 1, 2, 3, 4.

    Returns:
    --------
    ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row
    represents a generated functional data sample.
    """
    a1 = (np.random.pareto(alpha, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a2 = (np.random.pareto(alpha, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a3 = scalenoise * (np.random.pareto(alphanoise, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a4 = scalenoise * (np.random.pareto(alphanoise, size=num_samples) + 1) * \
        (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)

    result = np.dot(a1[:, None],
                    (1 + np.sin(2 * np.pi * om1 * grid))[None, :]) + \
            np.dot(a2[:, None],
                   (1 + np.cos(2 * np.pi * om2 * grid))[None, :]) + \
            np.dot(a3[:, None],
                    (1 + np.sin(2 * np.pi * om3 * grid))[None, :]) + \
            np.dot(a4[:, None],
                    (1 + np.cos(2 * np.pi * om4 * grid))[None, :])

    return result


def gen_rv_functional_data_gaussianNoise(num_samples, grid, alpha, sd,
                                         scalenoise=5, om1=1, om2=2,
                                         om3=3, om4=4, om5=5, om6=6):
    """
    Generate random regularly varying functions on a grid using
    Pareto-distributed coefficients and Gaussian noise components.

    Parameters:
    -----------
    num_samples : int
        Number of samples to generate.
    grid : array-like
        1D array representing the grid over which the basis functions are
        evaluated.
    alpha : float
        Shape parameter for the Pareto-distributed variables.
    sd : float
        Standard deviation for the Gaussian noise components.
    scalenoise : float, optional
        Scaling factor for the noise components. Default is 5.
    om1, om2, om3, om4, om5, om6 : float, optional
        Frequencies used in the sine and cosine basis functions.
        Defaults are 1, 2, 3, 4, 5, 6.

    Returns:
    --------
    ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row
        represents a generated data sample.
    """
    a1 = (np.random.pareto(alpha, size=num_samples) + 1)
    a2 = 0.8 * (np.random.pareto(alpha, size=num_samples) + 1)
    a3 = scalenoise * np.random.normal(0, sd, size=num_samples)
    a4 = 0.8 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a5 = 0.6 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a6 = 0.4 * scalenoise * np.random.normal(0, sd, size=num_samples)

    result = np.dot(a1[:, None],
                    (np.sqrt(2) * np.sin(2 * np.pi * om1 * grid))[None, :]) + \
             np.dot(a2[:, None],
                    (np.sqrt(2) * np.cos(2 * np.pi * om2 * grid))[None, :]) + \
             np.dot(a3[:, None],
                    (np.sqrt(2) * np.sin(2 * np.pi * om3 * grid))[None, :]) + \
             np.dot(a4[:, None],
                    (np.sqrt(2) * np.cos(2 * np.pi * om4 * grid))[None, :]) + \
             np.dot(a5[:, None],
                    (np.sqrt(2) * np.sin(2 * np.pi * om5 * grid))[None, :]) + \
             np.dot(a6[:, None],
                    (np.sqrt(2) * np.cos(2 * np.pi * om6 * grid))[None, :])

    return result
