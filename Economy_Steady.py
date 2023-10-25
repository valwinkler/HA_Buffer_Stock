#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:19:51 2023
Economy_Steady class and methods for partly replicating Carroll et al. 2017 without aggregate shocks
@author: valentinwinkler
"""
import numpy as np
from Agents_Steady import Agent_Steady

class Economy_Steady:
    """
    Economy that is populated by consumers of type 'Agent_Steady'.
    Methods for simulating the steady state b-point and b-dist model.
    Note that n modulo groups has to be zero for b-dist model.
    1/D has to be a natural number; 1/npsi has to be a natural number.
    psi_p has to be a vector with only 1/npsi as entries
    """
    def __init__(self, thet_p, thet_val, psi_p, psi_val, n = 8000, dist = False, 
                 beta = 0.9894, bet_p = 0.9867, bet_d = 0.0067, groups = 8, rho = 2, 
                 alpha = 0.36, delta = 0.025, l = 1/0.9, mu = 0.15, D = 0.00625, 
                 Omega = 0.07, nm = 600, mmin = 0.1, mmax = 250, 
                 Rmin = 0.03, Rmax = 0.04, nR = 8, R = 0.01, eps = 0.001, 
                 nthet = 8, npsi = 8):
        ### assign variables
        # Probabilities and model choice
        self.thet_p     = thet_p        # discrete approximation of shock process
        self.thet_val   = thet_val
        self.psi_p      = psi_p
        self.psi_val    = psi_val
        self.nthet      = nthet
        self.npsi       = npsi
        self.n          = n             # number of Agents in the economy
        self.dist       = dist          # (boolean) b-dist model? (else b-point model)
        self.groups     = groups        # number of groups to solve for b-dist model
        
        # consumer parameters
        self.rho        = rho
        self.alpha      = alpha
        self.delta      = delta
        self.l          = l
        self.mu         = mu
        self.D          = D
        self.Omega      = Omega
        
        # approximation parameters
        self.nm, self.mmin, self.mmax = nm, mmin, mmax
        self.nR, self.Rmin, self.Rmax = nR, Rmin, Rmax
        self.eps = eps
        
        # macroeconomic aggregates/rules
        self.R              = R                                            # current R
        self.K              = (alpha/R)**(1/(1-alpha)) * l * (1-Omega)     # capital implied by R
        self.W              = (1-alpha)*(self.K/(l*(1-Omega)))**alpha      # wage implied by R
        self.mdist          = np.zeros(n)                                  # distribution of normalized market ressources
        self.cdist          = np.zeros(n)                                  # distribution of consumption
        self.kdist          = np.zeros(n)                                  # distribution of capital
        self.pdist          = np.zeros(n)                                  # distribution of permanent productivity
        self.tau            = (mu * Omega) / (l * (1-Omega))               # tax rate
        
        #### initialize n agents
        k_start       = self.K/self.W                                      # everyone starts with equal capital stock
        thet_start    = np.random.choice(thet_val, n, True, thet_p)        # draw initial theta
        u_start       = np.random.choice([0,1],n,True,[(1-Omega),Omega])   # draw if unemployed
        xi_start      = u_start*mu + (1-u_start)*(1-self.tau)*l*thet_start # initial income
        m_start       = (1-delta+self.R)*k_start + xi_start                # next periods cash on hand
        
        self.kdist    = (self.kdist + k_start) * self.W
        self.mdist    = m_start
        
        # make list and append agents
        self.Agents = []
        
        if dist == False: # b-point model
            for i in range(n):
                self.Agents.append(Agent_Steady(beta = beta, thet_p = thet_p, thet_val = thet_val, 
                                         psi_p = psi_p, psi_val = psi_val, alpha = alpha, 
                                         delta = delta, l = l, mu = mu, D = D, Omega = Omega, nm = nm, mmin = mmin, 
                                         mmax = mmax, R = R, m = m_start[i], a = 0, k = k_start, con = 0, p = 1, 
                                         eps = eps, nthet = nthet, npsi = npsi))                             # create new agent
        
        if dist == True: # b-dist model
            step    = 2*bet_d/(groups-1)
            bet_st  = bet_p - bet_d
            
            for t in range(groups):
                b = bet_st + t*step
                for i in range(int(n/groups)):
                    ind = t*groups + i # current index
                    self.Agents.append(Agent_Steady(beta = b, thet_p = thet_p, thet_val = thet_val, 
                                         psi_p = psi_p, psi_val = psi_val, alpha = alpha, 
                                         delta = delta, l = l, mu = mu, D = D, Omega = Omega, nm = nm, mmin = mmin, 
                                         mmax = mmax, R = R, m = m_start[ind], a = 0, k = k_start, con = 0, p = 1, 
                                         eps = eps, nthet = nthet, npsi = npsi))                             # create new agent
    
    ## ======================= Get & Set Methods ==============================================
    def set_R(self, R):
        """
        Set curent interest rate and update K as well as W
        """
        Omega, l, alpha = self.Omega, self.l, self.alpha
        self.R          = R
        self.K          = (alpha/R)**(1/(1-alpha)) * l * (1 - Omega)
        self.W          = alpha * ((l*(1-Omega))/self.K)**(1-alpha)
    
    def get_dist(self):
        """
        Get current distribution of capital k, consumption c, cash on hand m
        and permanent productivity p, only m is normalized by W and k and c are not
        """
        return self.kdist, self.cdist, self.mdist, self.pdist
    
    def get_R(self):
        """
        get current interest rate R
        """
        return(self.R)

    
    def Lorentz(self, val = "k"):
        """
        Return x- and y-axis of Lorentz curve, y-axis 45d degree line + Gini coefficient for capital k, 
        consumption c or cash on hand m (the last one of which is normalized)
        input "val" must be in [k, c, m]
        """
        if val == "k":
            tmp = self.kdist
        if val == "c":
            tmp == self.cdist
        if val == "m":
            tmp == self.mdist
        
        # Compute Lorentz Curve
        yax = np.sort(tmp)
        yax = np.cumsum(yax)/np.sum(tmp)
        xax = np.linspace(0,1,num = len(tmp))
        
        # Compute Gini coefficient
        comp = np.ones(len(yax))                        # comparison: perfect equal distribution
        comp = np.cumsum(comp)/sum(comp)                # Lorentz y-axis for comparison
        gini = (np.sum(comp) - np.sum(yax))/sum(comp)   # compute Gini
        
        return xax, yax, comp, gini
    
    ## ====================== Simulation & Solution Methods =======================================
    def Iterate(self, eta = 0.00005, burn = 500, VFI = True):
        """
        Iterate the current Economy until convergence of the 1st and 2nd moment of the capital
        distribution and return the average capital stock at the end
        If VFI == True, value function iteration is conducted at the beginning of the simulation
        """
        mu, tau, l, Omega = self.mu, self.tau, self.l, self.Omega
        psi_val, thet_val, thet_p, npsi = self.psi_val, self.thet_val, self.thet_p, self.npsi
        n, groups = self.n, self.groups
        
        # new variables
        me_old = np.mean(self.kdist)
        s_old  = np.std(self.kdist)
        me     = me_old
        s      = s_old
        track  = 0
        
        ### Value Function Iteration and Updating of value and consumption function
        if VFI == True:
            # b-point model
            if self.dist == False: 
                A = self.Agents[1]
                A.set_R(self.R)     
                print("VFI")
                print("--------")
                A.VFI()             # value function iteration
                print("--------")
                print("VFI successful")
                v = A.get_v()       # get value function
                C = A.get_C()       # get policy function
                self.v = v
                self.C = C
                
                ## set value function and policy
                for i in range(n):
                    self.Agents[i].set_R(self.R)
                    self.Agents[i].set_v(v)
                    self.Agents[i].set_C(C)
            
            # b-dist model
            elif self.dist == True: 
                print("===")
                for i in range(groups):
                    print("Group ",i)
                    i1 = i*(int(n/groups))           # index of first agent in group
                    A  = self.Agents[i1]
                    A.set_R(self.R)
                    A.VFI()                     # value function iteration for given beta
                    print("--")
                    v = A.get_v()               # get value and policy function 
                    C = A.get_C()
                    
                    ## set value function and policy of remaining agents
                    for j in range(int(n/groups)):
                        ind = i*(int(n/groups)) + j  # get current index
                        self.Agents[ind].set_R(self.R)
                        self.Agents[ind].set_v(v)
                        self.Agents[ind].set_C(C)
        
        
        ### Iteration of the economy
        while (np.abs(me - me_old) > eta)| (track <= burn):
            track += 1
            me_old, s_old = me, s
            
            # draw (transitory) income shocks
            thet_tmp    = np.random.choice(thet_val, n, True, thet_p)       # transitory shocks
            u_tmp       = np.random.choice([0,1],n,True,[(1-Omega),Omega])  # unemployment shocks
            xi_tmp      = u_tmp*mu + (1-u_tmp)*(1-tau)*l*thet_tmp           # next normalized income
            
            ### Use simulation trick 1 of Carroll et al. (2015): Make sure death shocks are evenly distributed across wealth dist.
            dsize       = int(1/self.D)                                     # size of the group of which one pers. will die
            dgrno       = int(n/dsize)                                      # number of 'death groups'
            inds        = np.argsort(self.kdist)                            # indices to sort list in asc. order of capital
            
            i = 0
            for j in range(dgrno):
                die = np.random.choice(dsize)                               # who will die?
                for k in range(dsize):
                    ind = inds[i]                                           # index of current person
                    i  += 1
                    # consume
                    self.cdist[ind] = self.Agents[ind].consume()
                    
                    # Replace if died
                    if k == die: # replace person            
                        b_tmp = self.Agents[ind].get_beta()
                        v_tmp = self.Agents[ind].get_v()
                        C_tmp = self.Agents[ind].get_C()

                        self.Agents[ind] = Agent_Steady(beta = b_tmp, thet_p = self.thet_p, thet_val = self.thet_val, 
                                           psi_p = self.psi_p, psi_val = self.psi_val, alpha = self.alpha, 
                                           delta = self.delta, l = self.l, mu = self.mu, D = self.D, Omega = self.Omega, nm = self.nm, mmin = self.mmin, 
                                           mmax = self.mmax, R = self.R, m = 0, a = 0, k = 0, con = 0, p = 1, 
                                           eps = self.eps, nthet = self.nthet, npsi = self.npsi)                             # create new agent
                        
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
                    self.kdist[ind], self.pdist[ind] = self.Agents[ind].nextk(p_perm[k])
            
            # collect capital and labor income
            for i in range(n):
                self.mdist[i] = self.Agents[i].earn_income(xi_tmp[i])
            
            # get new delta
            me = np.mean(self.kdist)
            s  = np.std(self.kdist)
        return(me)
    
    def Steady(self, finer = 10):
        """
        Calculate Steady State of the Economy
        """
        l, alpha, Omega = self.l, self.alpha, self.Omega
        Rmin, Rmax, nR  = self.Rmin, self.Rmax, self.nR
        
        Rgrid = np.linspace(Rmin, Rmax, nR) # coarse grid at first
        R_sim = np.zeros(nR)                # vector of R implied by capital supply in simulation
        
        # try on coarse grid
        for i in range(nR):
            self.set_R(Rgrid[i])
            Ksup     = self.Iterate()
            R_sim[i] = alpha * (l*(1-Omega)/Ksup)**(1-alpha)
            print("--------")
            print("Coarse Iteration: ",i, "/", nR-1)
            print("R_d = ", np.round(Rgrid[i], decimals = 4))
            print("R_s = ", np.round(R_sim[i], decimals = 4))
        
        # continue on finer grid
        ind = np.argmin(np.abs(Rgrid - R_sim)) # where is capital market almost balanced?
        Rmin2, Rmax2 = Rgrid[ind-1], Rgrid[ind+1]
        Rgrid2 = np.linspace(Rmin2, Rmax2, finer)
        R_sim2 = np.zeros(finer)
        
        for i in range(finer):
            self.set_R(Rgrid2[i])
            Ksup     = self.Iterate()
            R_sim2[i] = alpha * (l*(1-Omega)/Ksup)**(1-alpha)
            print("Fine Iteration: ",i, "/", finer-1)
            print("R_d = ", np.round(Rgrid2[i], decimals = 4))
            print("R_s = ", np.round(R_sim2[i], decimals = 4))
        
        # optimal interest rate and final iteration
        ind_opt = np.argmin(np.abs(Rgrid2 - R_sim2))
        print("***===== D O N E =====***")
        print("R_d = ", np.round(Rgrid2[ind_opt], decimals = 4))
        print("R_s = ", np.round(R_sim2[ind_opt], decimals = 4))
        print("***===== D O N E =====***")
        R_final = (Rgrid2[ind_opt] + R_sim2[ind_opt])/2
        
        self.set_R(R_final)
        self.Iterate()
        
        