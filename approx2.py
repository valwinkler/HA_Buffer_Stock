#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:56:00 2023
Approximate log normal shocks by discrete distribution with equal weights
@author: valentinwinkler
"""

import numpy as np
import scipy.stats as sps

def approx_lognorm2(sd, n = 8):
    """
    Discretize log-normal random variable with mean 1 and a given variance for the
    logarithm subject to the condition that the pmf is constant (using the optimality
    result of Kennan 2006).
    
    Inputs:
        - sd    ... standard deviation of log of RV
        - n     ... number of discrete states
    
    Outputs:
        - v     ... n-dim numpy array of values
        - p     ... n-dim numpy array of probabilities
    """
    
    if sd == 0: # special case: RV is 1 for sure
        p = np.zeros(n)
        p[0] = 1
        v = np.zeros(n)
        v[0] = 1
    
    else:
        # normal distribution with sd as given that yields mean 1 for exp of RV
        dist = sps.norm(loc = - sd**2/2, scale = sd)
        
        p = np.ones(n) * 1/n
        v = np.zeros(n)
        for i in range(n):
            q    = (2*(i+1) - 1)/(2*n)
            v[i] = np.exp(dist.ppf(q))    # x_i = F^{-1}((2i - 1)/2n) i = 1,...,n
        
        # make sure mean is 1
        v = v + (1 - np.mean(v))
        
        return v, p