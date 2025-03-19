## to be removed 

import numpy as np


def norm_L1(vector):
    return np.sum(np.abs(vector))
    
def norm_L2(vector):
    return np.sqrt(np.sum(vector**2))

def norm_Linf(vector):
    return np.max(np.abs(vector))

def norm_L3(vector):
    return np.sum(np.abs(vector)**3)**(1/3)

def norm_L0(vector):
    return np.sum(vector != 0)
    
