#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:21:55 2023
Agent class and methods for finding the steady state in the model of Carroll 
et al. 2017 without aggregate shocks
@author: valentinwinkler
"""

import numpy as np
from numba import int64, float64
from numba.experimental import jitclass

### declare variable types for just in time compilation
agents_data = [
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
        ('v', float64[:]),          # value function 1d array
        ('C', float64[:]),          # policy function 1d array
        ('thet_p', float64[:]),     # theta probabilities
        ('thet_val', float64[:]),   # theta values (discrete)
        ('psi_p', float64[:]),      # psi probabilities
        ('psi_val', float64[:]),    # psi (discrete)
        ('m', float64),             # cash on hand (normalized)
        ('k', float64),             # capital (normalized)
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

@jitclass(agents_data) # just in time compilation for speed
class Agent_Steady:
    """"
    Agent/Consumer in the Carroll et al. (2017) model without aggregate shocks
    Get/Set methods, numerical solution for value/policy functions (in steady state) and simulations
    """
    def __init__(self, beta, thet_p, thet_val, psi_p, psi_val, rho = 2, alpha = 0.36, 
                 delta = 0.025, l = 1/0.9, mu = 0.15, D = 0.00625, Omega = 0.07,
                 nm = 600, mmin = 0.1, mmax = 250, R = 0.01, m = 0, a = 0, k = 0, 
                 con = 0, p = 1, eps = 0.001, nthet = 8, npsi = 8):
        
        ## initialize stuff
        self.beta, self.rho, self.alpha, self.delta, self.l = beta, rho, alpha, delta, l
        self.thet_p, self.thet_val, self.psi_p, self.psi_val = thet_p, thet_val, psi_p, psi_val
        self.mu, self.D, self.Omega = mu, D,  Omega
        self.nm, self.mmin, self.mmax = nm, mmin, mmax, 
        self.R, self.m, self.k, self.a, self.con, self.p = R, m, k, a, con, p
        self.nthet, self.npsi = nthet, npsi
        
        ## generate asset/capital grid (log-linear)
        self.mgrid      = np.exp(np.linspace(np.log(mmin), np.log(mmax), nm))
        
        ## get tax rate and survival probability
        self.tau        = (self.mu * self.Omega) / (self.l * (1-self.Omega))
        self.Dn         = 1 - self.D
        
        ## set current K and W
        self.K          = (alpha/self.R)**(1/(1-alpha)) * l * (1 - Omega)
        self.W          = (1-alpha) * (self.K/(l*(1-Omega)))**alpha
        
        ## generate value and policy function 
        self.v          = np.zeros(nm, 'float64')
        self.C          = np.zeros(nm, 'float64')
    
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
        return(self.k * self.W * self.p)
    
    def get_con(self):
        """
        Get last consumption (non-normalized) of agent
        """
        return(self.con * self.W * self.p)
    
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
    
    def c(self, m):
        """
        Get consumption value for a given state m found via linear interpolation if not on grid
        """
        mgrid       = self.mgrid
        C           = self.C
        return(np.interp(m,mgrid,C))
            
    ## ======================= set methods ==================================================
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
    
    def set_R(self, R):
        """
        Set curent interest rate and update K as well as W
        """
        Omega, l, alpha = self.Omega, self.l, self.alpha
        W_old           = self.W                 # save for re-normalization
        
        self.R          = R
        self.K          = (alpha/R)**(1/(1-alpha)) * l * (1 - Omega)
        self.W          = alpha * ((l*(1-Omega))/self.K)**(1-alpha)
        
        # re-normalize
        self.k          = self.k * W_old / self.W
        self.con        = self.con * W_old / self.W
    
    def incr_m(self, incr, normalized = False):
        """
        Increase m by increment
        If normalized == False (default), increment will be normalized before being added to m
        """
        if normalized == False:
            self.m += incr/(self.W * self.p)
        else:
            self.m += incr
        return(self.m)
    
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
    
    
    def Q_fun(self, m, c, v):
        """
        Return state-action values according to value function; interpolate on m-grid
        """
        Dn, beta, delta, rho             = self.Dn, self.beta, self.delta, self.rho
        Omega, mu, tau, l                = self.Omega, self.mu, self.tau, self.l
        psi_val, thet_val, psi_p, thet_p = self.psi_val, self.thet_val, self.psi_p, self.thet_p
        mgrid                            = self.mgrid
        R                                = self.R
        
        if (c > m) | (c <= 0):
            return(-1000000000000000) # nonnegativity constraints!
        else:
            res     = self.u(c)                                  # instant. utility
            fac     = Dn*beta                                    # factor to multiply expectation
            tmp_m = (1-delta+R)*(m-c)/(Dn)                       # m_t+1 if psi = 1 and epsilon = 0
            
            for i in range(self.npsi):
                # unemployed
                mnext    = tmp_m/psi_val[i] + mu                                # next normalized cash-on-hand
                v_tmp    = np.interp(mnext, mgrid, v)                                  # value fct of next m (interpolated)
                res     += fac * Omega*psi_p[i] * psi_val[i]**(1-rho) * v_tmp   # add to expected future benefit
                
                for j in range(self.nthet):
                    # employed
                    mnext   = tmp_m/psi_val[i] + (1-tau)*l*thet_val[j] # as above
                    v_tmp    = np.interp(mnext, mgrid, v)
                    res    += fac * (1-Omega)*thet_p[j]*psi_p[i] * psi_val[i]**(1-rho) * v_tmp
            
            return(res)
    
    def VFI(self, it = 50, fine = 50, eta = 0.00001): 
        """
        VFI to compute value function and policy function of optimal policy;
        afterwards update these functions for the agent
        step_c.. step size in coarse grid
        it   ... no. of policy iterations before policy is improved
        fine ... no. of gridpoints between coarse gridpoints on finer grid
        """
        v         = np.copy(self.v)
        v_next    = np.empty_like(v)
        C         = np.copy(self.C)
        mgrid, nm = self.mgrid, self.nm
        psi1      = (3-np.sqrt(5))/2 # golden section parameter
        psi2      = (np.sqrt(5)-1)/2
        
        track = 0
        
        # guess for value function and policy function
        for i in range(nm):
                v[i]  = self.u(mgrid[i]/5)/0.02
                C[i]  = mgrid[i]/5
        
        ########### coarse grid #######################
        delt = 1000
        # iterate
        while delt > eta:
            track += 1
            print("Iteration:", track)
            # evaluate policy
            for it in range(it):
                for i in range(nm):
                    v_next[i] = self.Q_fun(mgrid[i], C[i], v)
                v[:] = v_next # copy contents into v
            
            # improve policy using golden section search
            for i in range(nm):
                # Golden section search for maximum
                a     = 0
                b     = mgrid[i]
                x1    = a + psi1*(b-a)
                x2    = a + psi2*(b-a)
                
                while((b-a) > eta):
                    f1 = self.Q_fun(mgrid[i], x1, v) # value at x1
                    f2 = self.Q_fun(mgrid[i], x2, v)
                    
                    if f2 > f1:
                        a  = x1
                        x1 = x2
                        x2 = a + psi2*(b-a)
                    
                    else: 
                        b  = x2
                        x2 = x1
                        x1 = a + psi1*(b-a)
                C[i]      = (x1 + x2)/2
                v_next[i] = self.Q_fun(mgrid[i], (x1 + x2)/2, v)
            delt   = np.max(np.abs(v - v_next))
            v[:]   = v_next
        
        ### update everything
        self.set_v(v)
        self.set_C(C)
    
    ## ======================= Simulation Methods ===============================================
    def consume(self):
        """
        Consume according to money on hand and current estimate of the value function
        Update consumption and assets of agent
        Return non-normalized current consumption
        """
        m        = self.m
        self.con = self.c(m)
        self.a   = m - self.con
        res      = self.get_con()
        return(res)
    
    def nextk(self, psi = 1):
        """
        Inflate assets by contribution of newly died agents and normalize by new permanent income shock
        Update capital stock of agent and permanent productivity p
        Return non-normalized capital and new permanent productivity
        """
        self.k  = self.a / (self.Dn * psi)
        self.p  = self.p * psi
        self.con= self.con/psi # re-normalize last consumption
        res     = self.get_k()
        return(res, self.p)
    
    def earn_income(self, xi = 1/0.9 * 0.93):
        """
        Update money on hand by returns on capital stocks and labor income
        Return normalized money on hand
        """
        self.m = (1- self.delta + self.R)*self.k + xi
        if self.m > self.mmax:
            self.m = self.mmax
        return(self.m)

