#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:34:28 2023
Compute the Gini coefficient
@author: valentinwinkler
"""

import numpy as np

def gini(x):
    """
    Compute and return gini coefficient of numpy array x
    """
    # Compute Lorentz Curve
    yax = np.sort(x)
    yax = np.cumsum(yax)/np.sum(x)
    
    # Compute Gini coefficient
    comp = np.ones(len(yax))                        # comparison: perfect equal distribution
    comp = np.cumsum(comp)/sum(comp)                # Lorentz y-axis for comparison
    gini = (np.sum(comp) - np.sum(yax))/sum(comp)   # compute Gini
    return(gini)