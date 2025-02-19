import scipy.stats as stat
import numpy as np


######Multivariate symmetric logistic Model
####see "Simulating Multivariate Extreme Value Distributions of Logistic Type", 2003, A. Stephenson for more details


def gen_PositiveStable(n,alpha,seed=np.random.randint(2**31)): # From Nathan Huet
    """
    Generate positive stable random variables.

    Inputs:
    -----------------
    - n : sample size
    - alpha : dependence parameter
    - seed : random seed

    Output:
    ------------------
    - Sample of positive stable random variables
    """
    U=stat.uniform.rvs(0,np.pi,size=n,random_state = seed)
    W=stat.expon.rvs(size=n,random_state = seed)
    Term_1=(np.sin((1-alpha)*U)/W)**((1-alpha)/alpha)
    Term_2=np.sin(alpha*U)/((np.sin(U))**(1/alpha))
    return Term_1*Term_2


def gen_multilog(n,Dim,alpha,Hill_index,seed=np.random.randint(2**31)):
    # From Nathan Huet
    #
    
    """
    Generate multivariate logistic random variables.  

    Inputs:
    ---------------------
    - n : sample size
    - Dim : dimension
    - alpha : dependence parameter
    - Hill_index : Hill index ## to be removed 
    - seed : random seed

    Output:
    ----------------------
    - Sample of multivariate logistic random variables

    Details:
    ----------------------
     Implementation of Algorithm 2.1 in  "Stephenson, A. (2003). Simulating multivariate extreme value distributions of logistic type. Extremes, 6, 49-59.". 
    
    """
    W=stat.expon.rvs(size=(n,Dim),random_state = seed) ### Standard exponential Random Variables

    S=gen_PositiveStable(n,alpha,seed)### TO check --> OK
    Log=np.zeros((n,Dim))
    for ii in range(Dim):
        Log[:,ii]=(S/W[:,ii])**alpha
   # return Log  ## (anne)    
    return Log**(1/Hill_index) ##np.clip(Log**(1/Hill_index),a_min=0,a_max=1e3)


def gen_label(Matrix,angle=0.2): #PA
    #Gen label 2D
    Vect_angle=np.arctan2(Matrix[:,1],Matrix[:,0])
    label=[0 if (point_angle<(angle*np.pi/2) or (point_angle>((1-angle)*np.pi/2))) else 1 for point_angle in Vect_angle]
    return np.array(label)


def gen_label2(Matrix,angle=0.2): #PA
    #Gen label 2D
    Vect_angle=np.arctan2(Matrix[:,1],Matrix[:,0])
    label = [
        0 if (
            (0 <= point_angle < angle * np.pi / 4) or
            (1 - angle) * np.pi / 4 <= point_angle < np.pi / 4 or
            np.pi / 4 + angle * np.pi / 4 <= point_angle < np.pi / 4 + (1 - angle) * np.pi / 4
        ) else 1
        for point_angle in Vect_angle
    ]
    return np.array(label)

## new function(anne)
def rpareto(n, alpha, x_min=1):
    """
    Generate samples from a Pareto distribution with shape parameter `alpha` and 
    minimum value `x_min` (default is 1).
    
    Parameters
    ----------
    n : int
        The number of random samples to generate.
    alpha : float
        The shape parameter of the Pareto distribution (alpha > 0).
    x_min : float, optional
        The minimum value (scale) for the Pareto distribution (default is 1).
    
    Returns
    -------
    ndarray, shape (n,)
        A NumPy array of `n` random samples from the Pareto distribution.
    """
    # Generate uniform random variables U
    U = np.random.uniform(0, 1, n)
    
    # Apply inverse transform sampling to generate Pareto samples
    return x_min * (1 - U) ** (-1 / alpha)




def GenerateRVdata(num_samples, grid, alpha, alphanoise, scalenoise=5, om1=1, om2=2, om3=3, om4=4):
       """
    Generate random variate (RV) data using Pareto-distributed coefficients and sinusoidal basis functions.

    This function generates `num_samples` of data points using a combination of:
    - Pareto-distributed random variables (`rpareto`)
    - Sine and cosine basis functions with different frequencies (`om1`, `om2`, `om3`, `om4`)
    - Additional noise components controlled by `alphanoise` and `scalenoise`

    Parameters:
    ----------
    num_samples : int
        Number of samples to generate.

    grid : array-like
        A 1D array representing the grid over which the basis functions are evaluated.

    alpha : float
        Shape parameter for the primary Pareto-distributed variables (`a1`, `a2`).

    alphanoise : float
        Shape parameter for the noise Pareto-distributed variables (`a3`, `a4`).

    scalenoise : float, optional (default=5)
        Scaling factor for the noise components.

    om1, om2, om3, om4 : float, optional (default=1, 2, 3, 4)
        Frequencies used in the sine and cosine basis functions.

    Returns:
    -------
    result : ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row represents 
        a generated data sample.

    Notes:
    ------
    - `rpareto`  generates Pareto-distributed random variables.
    - The random coefficients `a1`, `a2`, `a3`, and `a4` are symmetrically distributed around zero.
    - The resulting dataset is constructed using sinusoidal basis functions with different frequencies.

    Example:
    --------
    ```python
    import numpy as np

    # Define grid
    grid = np.linspace(0, 1, 100)

    # Generate data
    data = GenerateRVdata(num_samples=500, grid=grid, alpha=2, alphanoise=3)

    print(data.shape)  # (500, 100)
    ```
    """
    a1 = rpareto(num_samples, alpha=alpha) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a2 = rpareto(num_samples, alpha=alpha) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a3 = scalenoise * rpareto(num_samples, alpha=alphanoise) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    a4 = scalenoise * rpareto(num_samples, alpha=alphanoise) * (2 * (np.random.uniform(size=num_samples) > 0.5) - 1)
    
    result = np.dot(a1[:, None], (1 + np.sin(2 * np.pi * om1 * grid))[None, :]) + \
             np.dot(a2[:, None], (1 + np.cos(2 * np.pi * om2 * grid))[None, :]) + \
             np.dot(a3[:, None], (1 + np.sin(2 * np.pi * om3 * grid))[None, :]) + \
             np.dot(a4[:, None], (1 + np.cos(2 * np.pi * om4 * grid))[None, :])
    
    return result

def GenerateRVdatabis(num_samples, grid, alpha, sd, scalenoise=5, om1=1, om2=2, om3=3, om4=4, om5=5, om6=6):#ACP
    """
    Generate random variate (RV) data using Pareto-distributed coefficients and Gaussian noise components
    with sinusoidal basis functions.

    This function generates `num_samples` of data points using a combination of:
    - Pareto-distributed random variables (`rpareto`)
    - Gaussian noise components controlled by `sd` and `scalenoise`
    - Sine and cosine basis functions with different frequencies (`om1` to `om6`)

    Parameters:
    ----------
    num_samples : int
        Number of samples to generate.

    grid : array-like
        A 1D array representing the grid over which the basis functions are evaluated.

    alpha : float (>0)
        Shape parameter for the Pareto-distributed variables (`a1`, `a2`).

    sd : float
        Standard deviation for the Gaussian noise components (`a3`, `a4`, `a5`, `a6`).

    scalenoise : float, optional (default=5)
        Scaling factor for the noise components.

    om1, om2, om3, om4, om5, om6 : float, optional (default=1, 2, 3, 4, 5, 6)
        Frequencies used in the sine and cosine basis functions.

    Returns:
    -------
    result : ndarray
        A 2D NumPy array of shape `(num_samples, len(grid))`, where each row represents 
        a generated data sample.

    Notes:
    ------
    - Unlike `GenerateRVdata`, this function includes **Gaussian noise** components instead of Pareto noise for variables `a3`, `a4`, `a5`, and `a6`.
    - The sine and cosine components are scaled by `np.sqrt(2)` for normalization.
    - The function combines six components (instead of four in `GenerateRVdata`) using different frequencies.
    
    Example:
    --------
    ```python
    import numpy as np

    # Define grid
    grid = np.linspace(0, 1, 100)

    # Generate data
    data = GenerateRVdatabis(num_samples=500, grid=grid, alpha=2, sd=0.5)

    print(data.shape)  # (500, 100)
    ```
    """
    a1 = rpareto(num_samples, alpha=alpha)
    a2 = 0.8 * rpareto(num_samples, alpha=alpha)
    a3 = scalenoise * np.random.normal(0, sd, size=num_samples)
    a4 = 0.8 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a5 = 0.6 * scalenoise * np.random.normal(0, sd, size=num_samples)
    a6 = 0.4 * scalenoise * np.random.normal(0, sd, size=num_samples)

    result = np.dot(a1[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om1 * grid))[None, :]) + \
             np.dot(a2[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om2 * grid))[None, :]) + \
             np.dot(a3[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om3 * grid))[None, :]) + \
             np.dot(a4[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om4 * grid))[None, :]) + \
             np.dot(a5[:, None], (np.sqrt(2) * np.sin(2 * np.pi * om5 * grid))[None, :]) + \
             np.dot(a6[:, None], (np.sqrt(2) * np.cos(2 * np.pi * om6 * grid))[None, :])
    
    return result

