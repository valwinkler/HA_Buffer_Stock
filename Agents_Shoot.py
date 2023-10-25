#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:27:41 2023
Agent class and method for solving transitional dynamics problems in the Carroll
et al. (2017) model without aggregate shocks
@author: valentinwinkler
"""

import numpy as np
from numba import int64, float64
from numba.experimental import jitclass

### declare variable types for just in time compilation
agents_data2 = [
        ('beta', float64),          # discount factor
        ('rho', float64),           # CRRA
        ('alpha', float64),         # cap. share
        ('delta', float64),         # depr. rate
        ('l', float64),             # time worked per employee
        ('mu', float64),            # unemploym. insurance payment
        ('D', float64),             # prob. of death
        ('Dn', float64),            # prob. of survival
        ('Omega', float64),         # prob. of being unemployed
        ('tau', float64),           # tax rate
        ('v', float64[:,:]),       # time dependent value function 2d array
        ('C', float64[:,:]),       # time dependent policy function 1d array
        ('v_term', float64[:]),     # value function in terminal period
        ('C_term', float64[:]),     # policy function in terminal period
        ('periods', int64),         # number of periods used for shooting
        ('Rpath', float64[:]),      # path of interest rates
        ('Kpath', float64[:]),      # path of aggregate capital stock
        ('Wpath', float64[:]),      # path of wages
        ('thet_p', float64[:]),     # theta probabilities
        ('thet_val', float64[:]),   # theta values (discrete)
        ('psi_p', float64[:]),      # psi probabilities
        ('psi_val', float64[:]),    # psi (discrete)
        ('m', float64),             # cash on hand (normalized)
        ('k', float64),             # capital (normalized)
        ('t', int64),               # current time
        ('con', float64),           # last consumption (normalized)
        ('p', float64),             # current productivity
        ('W', float64),             # current wage
        ('R', float64),             # current interest rate
        ('a', float64),             # assets (normalized)
        ('K', float64),             # current aggregate capital
        ('mgrid', float64[:]),      # cash on hand grid
        ('nm', int64),              # cash-on-hand grid parameters
        ('mmin', float64),
        ('mmax', float64),
        ('eps', float64),            # tolerance parameter
        ('nthet', int64),            # number of theta states
        ('npsi', int64)              # number of psi states
]

##=========================================================================================
##===================== Agent Class & Methods =============================================
##=========================================================================================

@jitclass(agents_data2) # just in time compilation for speed
class Agent_Shoot:
    """"
    Agent/Consumer in the Carroll et al. (2017) model without aggregate shocks
    Get/Set methods, numerical solution for value/policy functions for given interest rate path
    """
    def __init__(self, beta, thet_p, thet_val, psi_p, psi_val, Rpath, v_term,
                 C_term, periods = 250, t = 0, rho = 2, alpha = 0.36, 
                 delta = 0.025, l = 1/0.9, mu = 0.15, D = 0.00625, Omega = 0.07,
                 nm = 600, mmin = 0.1, mmax = 250, m = 0, a = 0, k = 0, 
                 con = 0, p = 1, nthet = 8, npsi = 8):
        
        ## initialize stuff
        self.beta, self.rho, self.alpha, self.delta, self.l = beta, rho, alpha, delta, l
        self.thet_p, self.thet_val, self.psi_p, self.psi_val = thet_p, thet_val, psi_p, psi_val
        self.mu, self.D, self.Omega = mu, D,  Omega
        self.nm, self.mmin, self.mmax = nm, mmin, mmax, 
        self.m, self.k, self.a, self.con, self.p = m, k, a, con, p
        self.nthet, self.npsi = nthet, npsi
        self.Rpath, self.v_term = Rpath, v_term
        self.t = t
        self.periods = periods
        
        ## generate asset/capital grid
        self.mgrid      = np.exp(np.linspace(np.log(mmin), np.log(mmax), nm))
        
        ## get tax rate and survival probability
        self.tau        = (self.mu * self.Omega) / (self.l * (1-self.Omega))
        self.Dn         = 1 - self.D
        
        ## get paths for K and W
        self.Kpath      = np.zeros(periods, 'float64')
        self.Wpath      = np.zeros(periods, 'float64')
        
        for i in range(periods):
            self.Kpath[i] = (alpha/self.Rpath[i])**(1/(1-alpha)) * l * (1 - Omega)
            self.Wpath[i] = (1-alpha) * (self.Kpath[i]/(l*(1-Omega)))**alpha
        
        ## generate value and policy function together with initial guess
        self.v              = np.zeros((nm, periods), 'float64')
        self.v[:,periods-1] = v_term
        self.C              = np.zeros((nm, periods), 'float64')
        self.C[:,periods-1] = C_term
        
    ## ======================= get methods ==================================================
    def get_beta(self):
        """
        Get discount factor of agent
        """
        return(self.beta)
        
    def get_p(self):
        """
        Get current permanent productivity of agent
        """
        return(self.p)
    
    def get_m(self):
        """
        Get cash on hand of agent
        """
        return(self.m)
    
    def get_k(self):
        """
        Get capital (non-normalized) of agent
        """
        return(self.k * self.Wpath[self.t] * self.p)
    
    def get_con(self):
        """
        Get last consumption (non-normalized) of agent
        """
        return(self.con * self.Wpath[self.t] * self.p)
    
    def get_v(self):
        """
        Get value function of agent
        """
        return(self.v)
    
    def get_C(self):
        """
        Get policy function matrix
        """
        return(self.C)
        
    def get_mgrid(self):
        """
        Get cash on hand grid
        """
        return(self.mgrid)
    
    def c(self, m,t = 99):
        """
        Get consumption value for a given state m found via linear interpolation if not on grid
        """
        mgrid       = self.mgrid
        C           = self.C
        res = np.interp(m,mgrid,C[:,t])
        return(res)
    
    ## ======================= set methods ================================================== 
    def set_t(self, t):
        """
        Set current time, t
        """
        self.t = t
    
    def set_mp(self, m, p):
        """
        Set current normalized cash on hand, m, and permanent productivity, p
        """
        self.m = m
        self.p = p
    
    def set_v(self, v):
        """
        Set value function of agent
        """
        self.v = v
    
    def set_C(self, C):
        """
        Set policy function of agent
        """
        self.C = C
    
    def set_Rpath(self, Rpath):
        """
        Set the path of R and update the paths of K and W as well
        """
        Omega, l, alpha = self.Omega, self.l, self.alpha
        self.Rpath      = Rpath
        
        for i in range(self.periods):
            self.Kpath[i] = (alpha/self.Rpath[i])**(1/(1-alpha)) * l * (1 - Omega)
            self.Wpath[i] = (1-alpha) * (self.Kpath[i]/(l*(1-Omega)))**alpha
    
    def set_v_term(self, v_term):
        """
        Set terminal value function v
        """
        self.v_term            = v_term
        self.v[:,self.periods] = v_term
    
    def set_C_term(self, C_term):
        """
        Set terminal consumption function C
        """
        self.C_term            = C_term
        self.C[:,self.periods] = C_term
    
    ## ======================= Utility & value function methods =============================
    def u(self, c):
        """
        Utility of consumption c
        """
        rho = self.rho
        
        if rho == 1:
            return np.log(c)
        else:
            return(c**(1-rho)/(1-rho))
    
    
    def Q_fun(self, m, t, c, v):
        """
        Return state-action values according to value function; interpolate on m-grid
        """
        Dn, beta, delta, rho             = self.Dn, self.beta, self.delta, self.rho
        Omega, mu, tau, l                = self.Omega, self.mu, self.tau, self.l
        psi_val, thet_val, psi_p, thet_p = self.psi_val, self.thet_val, self.psi_p, self.thet_p
        mgrid                            = self.mgrid
        R                                = self.Rpath[t+1]
        W, W_next                        = self.Wpath[t], self.Wpath[t+1]
        
        if (c > m) | (c <= 0):
            return(-1000000000000000) # nonnegativity constraints!
        else:
            res     = self.u(c)                                  # instant. utility
            fac     = Dn*beta * (W_next/W)**(1-rho)              # factor to multiply expectation
            tmp_m = (1-delta+R)*(m-c)*W/(W_next*Dn)              # m_t+1 if psi = 1 and xi = 0
            
            for i in range(self.npsi):
                # unemployed
                mnext    = tmp_m/psi_val[i] + mu                                # next normalized cash-on-hand
                v_tmp    = np.interp(mnext, mgrid, v[:,t+1])                    # value fct of next m (interpolated)
                res     += fac * Omega*psi_p[i] * psi_val[i]**(1-rho) * v_tmp   # add to expected future benefit
                
                for j in range(self.nthet):
                    # employed
                    mnext   = tmp_m/psi_val[i] + (1-tau)*l*thet_val[j] # as above
                    v_tmp   = np.interp(mnext, mgrid, v[:,t+1])
                    res    += fac * (1-Omega)*thet_p[j]*psi_p[i] * psi_val[i]**(1-rho) * v_tmp
            
            return(res)
    
    def solve_VF(self, fine = 30, eta = 0.00001):
        """
        Compute time dependent value and consumption function depending on the path of R
        """
        mgrid, nm, = self.mgrid, self.nm
        periods    = self.periods
        psi1       = (3-np.sqrt(5))/2 # golden section parameter
        psi2       = (np.sqrt(5)-1)/2
        
        print("---------------------")
        print("Compute Value Function")
        
        # loop through T-2, T-3, ..., 0 [Remember: T-1 is last element in array]
        for k in range(periods-1):
            v  = np.copy(self.v)
            tc = (periods - 2) - k
            if tc%20 == 0:
                print("t =",tc)
            for i in range(nm):
                # Golden section search for maximum
                a     = 0
                b     = mgrid[i]
                x1    = a + psi1*(b-a)
                x2    = a + psi2*(b-a)
                
                while((b-a) > eta):
                    f1 = self.Q_fun(mgrid[i], tc, x1, v) # value at x1
                    f2 = self.Q_fun(mgrid[i], tc, x2, v)
                    
                    if f2 > f1:
                        a  = x1
                        x1 = x2
                        x2 = a + psi2*(b-a)
                    
                    else: 
                        b  = x2
                        x2 = x1
                        x1 = a + psi1*(b-a)
                
                self.C[i,tc] = (x1 + x2)/2
                self.v[i,tc] = self.Q_fun(mgrid[i], tc, (x1 + x2)/2, v)
                
    ## ======================= Simulation Methods ===============================================
    def consume(self):
        """
        Consume according to money on hand and current estimate of the value function
        Update consumption and assets of agent
        Return non-normalized current consumption
        """
        m, t     = self.m, self.t
        self.con = self.c(m, t)
        self.a   = m - self.con
        res      = self.get_con()
        return(res)
    
    def nextk(self, psi = 1):
        """
        Inflate assets by contribution of newly died agents and normalize by new permanent income shock
        Update capital stock of agent and permanent productivity p
        Return non-normalized capital and new permanent productivity
        """
        self.t  = self.t + 1
        t       = self.t
        self.k  = (self.a * self.Wpath[t-1]) / (self.Wpath[t] * self.Dn * psi)
        self.p  = self.p * psi
        
        # also re-normalize last consumption
        self.con = (self.con * self.Wpath[t-1]) / (self.Wpath[t] * psi)
        res     = self.get_k()
        return(res, self.p)
    
    def earn_income(self, xi = 1/0.9 * 0.93):
        """
        Update money on hand by returns on capital stocks and labor income
        Return normalized money on hand
        """
        self.m = (1- self.delta + self.Rpath[self.t]) * self.k + xi 
        if self.m > self.mmax:
            self.m = self.mmax
        return(self.m)
        