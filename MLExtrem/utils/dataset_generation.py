import scipy.stats as stat
import numpy as np


######Multivariate symmetric logistic Model
####see "Simulating Multivariate Extreme Value Distributions of Logistic Type", 2003, A. Stephenson for more details


def gen_PositiveStable(n,alpha,seed=np.random.randint(2**31)): # From Nathan Huet
    """
    Generate positive stable random variables.

    Inputs:
    - n : sample size
    - alpha : dependence parameter
    - seed : random seed

    Output:
    - Sample of positive stable random variables
    """
    U=stat.uniform.rvs(0,np.pi,size=n,random_state = seed)
    W=stat.expon.rvs(size=n,random_state = seed)
    Term_1=(np.sin((1-alpha)*U)/W)**((1-alpha)/alpha)
    Term_2=np.sin(alpha*U)/((np.sin(U))**(1/alpha))
    return Term_1*Term_2


def gen_multilog(n,Dim,alpha,Hill_index,seed=np.random.randint(2**31)):  # From Nathan Huet
    #Assymetric ? ALgo 2.1 in the article
    
    """
    Generate multivariate logistic random variables.  

    Inputs:
    - n : sample size
    - Dim : dimension
    - alpha : dependence parameter
    - Hill_index : Hill index
    - seed : random seed

    Output:
    - Sample of multivariate logistic random variables
    """
    W=stat.expon.rvs(size=(n,Dim),random_state = seed) ### Standard exponential Random Variables

    S=gen_PositiveStable(n,alpha,seed)### TO check
    Log=np.zeros((n,Dim))
    for ii in range(Dim):
        Log[:,ii]=(S/W[:,ii])**alpha
    return np.clip(Log**(1/Hill_index),a_min=0,a_max=1e3)

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



def GenerateRVdata(num_samples, grid, alpha, alphanoise, scalenoise=5, om1=1, om2=2, om3=3, om4=4):#ACP
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

