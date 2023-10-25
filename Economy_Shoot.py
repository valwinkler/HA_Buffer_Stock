#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:03:22 2023
Economy_Shoot class and methods for partly replicating transitional dynamics in 
Carroll et al. 2017 without aggregate shocks
@author: valentinwinkler
"""

import numpy as np
import matplotlib.pyplot as plt
from Agents_Shoot import Agent_Shoot

class Economy_Shoot:
    """
    Economy that is populated by consumers of type 'Agent_Steady'.
    Methods for getting the path of model variables when starting from an
    initial capital/productivity distribution and ending up in the steady state 
    of the b-point model.
    1/D has to be a natural number; 1/npsi has to be a natural number.
    psi_p has to be a vector with only 1/npsi as entries
    """
    def __init__(self, thet_p, thet_val, psi_p, psi_val, Rpath, v_term, C_term,
                 p_init, m_init, periods = 250, n = 8000, beta = 0.9894, rho = 2,
                 alpha = 0.36, delta = 0.025, l = 1/0.9, mu = 0.15, D = 0.00625, 
                 Omega = 0.07, nm = 600, mmin = 0.1, mmax = 250,  
                 Reps = 0.001, nthet = 8, npsi = 8):
        ### assign variables
        # Probabilities and model choice
        self.thet_p     = thet_p        # discrete approximation of shock process
        self.thet_val   = thet_val
        self.psi_p      = psi_p
        self.psi_val    = psi_val
        self.nthet      = nthet
        self.npsi       = npsi
        self.n          = n             # number of Agents in the economy
        self.periods    = periods
        
        # consumer parameters
        self.beta       = beta
        self.rho        = rho
        self.alpha      = alpha
        self.delta      = delta
        self.l          = l
        self.mu         = mu
        self.D          = D
        self.Omega      = Omega
        
        # approximation parameters
        self.nm, self.mmin, self.mmax = nm, mmin, mmax
        
        # macroeconomic aggregates/rules
        self.mdist          = np.zeros((n,periods))                        # distribution of normalized market ressources
        self.cdist          = np.zeros((n,periods))                        # distribution of consumption
        self.kdist          = np.zeros((n,periods))                        # distribution of capital
        self.pdist          = np.zeros((n,periods))                        # distribution of permanent productivity
        self.tau            = (mu * Omega) / (l * (1-Omega))               # tax rate
        
        ## set first column of kdist as real money on hand
        self.kdist[:,0] = p_init * m_init # just for ordering reasons to distribute shocks
        
        ## get paths for K, W and Y
        self.Rpath      = Rpath
        self.Kpath      = np.zeros(periods, 'float64')
        self.Wpath      = np.zeros(periods, 'float64')
        self.Ypath      = np.zeros(periods, 'float64')
        
        for i in range(periods):
            self.Kpath[i] = (alpha/self.Rpath[i])**(1/(1-alpha)) * l * (1 - Omega)
            self.Wpath[i] = (1-alpha) * (self.Kpath[i]/(l*(1-Omega)))**alpha
            self.Ypath[i] = self.Kpath[i]**alpha * (l*(1 - Omega))**(1-alpha)
        
        # initial/terminal variables
        self.p_init        = p_init
        self.m_init        = m_init
        self.pdist[:,0]    = p_init
        self.mdist[:,0]    = m_init
        self.v_term        = v_term
        self.C_term        = C_term
        
        # 'donor' agent to compute optimal policy/VF
        self.Donor = Agent_Shoot(beta, thet_p, thet_val, psi_p, psi_val, Rpath, v_term, C_term)
        self.C = self.Donor.get_C()
        self.v = self.Donor.get_v()
        
        # make list and append agents
        self.Agents = []
        t           = 0
        
        for i in range(n):
            self.Agents.append(Agent_Shoot(beta, thet_p, thet_val, psi_p, psi_val, Rpath, v_term,
                 C_term, periods, t, rho, alpha, delta, l, mu, D, Omega,
                 nm, mmin, mmax, m_init[i], 0, 0, 0, p_init[i], nthet, npsi))                          # create new agent
        
    ## ======================= get & set methods ===============================================
    def get_dist(self):
        """
        Return distributions of cash on hand (normalized), consumption c, capital k and 
        permanent productivity p
        """
        return(self.mdist, self.cdist, self.kdist, self.pdist)
    
    def get_Paths(self):
        """
        Return paths of R, W, K and Y
        """
        return(self.Rpath, self.Wpath, self.Kpath, self.Ypath)
    
    def get_Rpath_implied(self):
        """
        Get path of interest rates that is implied by simulated capital stock
        """
        Kpath_sim = np.mean(self.kdist, axis = 0) 
        Rpath_implied = np.zeros(self.periods)
        for i in range(len(Kpath_sim)):
            Rpath_implied[i] = self.alpha * ((self.l*(1-self.Omega))/Kpath_sim[i])**(1-self.alpha)
        return(Rpath_implied)
    
    def set_Rpath(self, Rpath):
        """
        Set interest rate path and derive paths of K, W and Y
        """
        alpha, l, Omega = self.alpha, self.l, self.Omega
        periods = self.periods
        
        self.Rpath = Rpath
        # update all other paths
        for i in range(periods):
            self.Kpath[i] = (alpha/self.Rpath[i])**(1/(1-alpha)) * l * (1-Omega)
            self.Wpath[i] = (1-alpha) * (self.Kpath[i]/(l*(1-Omega)))**alpha
            self.Ypath[i] = self.Kpath[i]**alpha * (l*(1 - Omega))**(1-alpha)
    
    def get_vC(self):
        """
        Return time-dependent state and policy function
        """
        return(self.v, self.C)
    
    ## ======================= Simulation methods ===============================================
    def Solve_VF(self):
        """
        Solve for value/policy function of current Path of R
        """
        self.Donor.set_Rpath(self.Rpath)
        print("solve VF...")
        self.Donor.solve_VF()
        print("...successful")
        self.v = self.Donor.get_v()
        self.C = self.Donor.get_C()
    
    def Shoot(self):
        """
        Simulate path of individual consumption/savings decisions and update distributional variables
        """
        n, periods = self.n, self.periods
        mu, tau, l, Omega, alpha, delta, D, rho = self.mu, self.tau, self.l, self.Omega, self.alpha, self.delta, self.D, self.rho
        beta, psi_val, thet_val, thet_p, psi_p, npsi, nthet = self.beta, self.psi_val, self.thet_val, self.thet_p, self.psi_p, self.npsi, self.nthet
        
        # set R-path, initial m/p distribution, t = 0, and value/policy functions
        for i in range(n):
            self.Agents[i].set_mp(self.m_init[i], self.p_init[i])
            self.Agents[i].set_t(0)
            self.Agents[i].set_Rpath(self.Rpath)
            self.Agents[i].set_v(self.v)
            self.Agents[i].set_C(self.C)
        
        # simulate *periods* periods
        for t2 in range((periods-1)):
            # draw (transitory) income shocks
            thet_tmp    = np.random.choice(thet_val, n, True, thet_p)       # transitory shocks
            u_tmp       = np.random.choice([0,1],n,True,[(1-Omega),Omega])  # unemployment shocks
            xi_tmp      = u_tmp*mu + (1-u_tmp)*(1-tau)*l*thet_tmp           # next normalized income
            
            ### Use simulation trick 1 of Carroll et al. (2015): Make sure death shocks are evenly distributed across wealth dist.
            dsize       = int(1/self.D)                                     # size of the group of which one pers. will die
            dgrno       = int(n/dsize)                                      # number of 'death groups'
            inds        = np.argsort(self.kdist[:,t2])                     # indices to sort list in asc. order of capital
            
            i = 0
            for j in range(dgrno):
                die = np.random.choice(dsize)                               # who will die?
                for k in range(dsize):
                    ind = inds[i]                                           # index of current person
                    i  += 1
                    # consume
                    self.cdist[ind,t2] = self.Agents[ind].consume()
                    
                    # Replace if died
                    if k == die: # replace person            
                        v_tmp = self.Agents[ind].get_v()
                        C_tmp = self.Agents[ind].get_C()
                        self.Agents[ind] = Agent_Shoot(beta = beta, thet_p = thet_p, thet_val = thet_val, psi_p = psi_p, psi_val = psi_val, 
                                                       Rpath = self.Rpath, v_term = self.v_term, C_term = self.C_term, periods = periods, t = t2, 
                                                       rho = rho, alpha = alpha, delta = delta, l = l, mu = mu, D = D, Omega = Omega, 
                                                       nm = self.nm, mmin = self.mmin, mmax = self.mmax, m = 0, a = 0, k = 0, con = 0, p = 1, nthet = nthet, npsi = npsi)  # create new agent
                        
                        self.Agents[ind].set_Rpath(self.Rpath)
                        self.Agents[ind].set_v(v_tmp)                       # set value & consumption function
                        self.Agents[ind].set_C(C_tmp)
            
            ### Use simulation trick 2 of Carroll et al. (2015): Make sure permanent inc. shocks are evently distr. across wealth dist
            psize       = npsi                                              # size of group across which each perm. income shock is assigned to one person
            pgrno       = int(n/psize)                                      # number of 'permanent inc. shock groups'
            
            i = 0
            for j in range(pgrno):
                p_perm = np.random.choice(psi_val, size = npsi, replace = False)
                for k in range(psize):
                    ind = inds[i]
                    i += 1
                    # Create next normalized capital
                    self.kdist[ind, t2+1], self.pdist[ind, t2+1] = self.Agents[ind].nextk(p_perm[k])
            
            # collect capital and labor income
            for i in range(n):
                self.mdist[i, t2+1] = self.Agents[i].earn_income(xi_tmp[i])
        
    
    def Iterate(self, eta = 0.00013, eta2 = 0.9):
        """
        Determine path of R where capital supply and demand balance
        """
        eta   = eta
        eta2  = eta2
        delt  = 1000
        track = 0
        stp   = 0      # step size already increased?
        
        while delt > eta:
            track = track + 1
            print("------------------------")
            print("Iteration ", track)
            self.Solve_VF()
            self.Shoot()
            R_set  = self.Rpath
            R_impl = self.get_Rpath_implied()
            
            # plot result of every iteration
            x = np.array(range(self.periods - 1)) + 1 # time axis
            plt.figure()
            plt.plot(x, R_set[1:], c = "green", label = "R_set")
            plt.plot(x, R_impl[1:], c = "red", label = "R_implied")
            plt.plot(x, np.ones(self.periods - 1)*R_set[self.periods - 1], c = "black", linewidth = 0.5)
            plt.legend()
            plt.show()
            
            delt   = np.max(np.abs(R_set[1:(self.periods - 1)] - R_impl[1:(self.periods - 1)]))
            print("Delta: ", delt)
            
            if (delt < 2*eta) & (stp == 0): # already close? decrease step towards implied R
                stp  = 1
                eta2 = 1 - (1-eta2)/3
            
            R_new  = R_set * eta2 + R_impl * (1-eta2)
            R_new[self.periods-1] = R_set[self.periods-1] # last period has to have ss R
            R_new[0] = R_set[0]                           # period 0 R was still R_ss (important for normalization with W!)
            
            self.set_Rpath(R_new)
        
        print("--------------------------")
        print("DONE")
        self.set_Rpath(R_set) # last set path that produced small error