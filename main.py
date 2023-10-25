#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:20:39 2023
Main File for partly replicating Carroll et al. 2017
@author: valentinwinkler
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt

# auxiliary scripts
from Economy_Steady import Economy_Steady
from Economy_Shoot import Economy_Shoot
from Agents_Steady import Agent_Steady
from approx2 import approx_lognorm2
from Gini import gini

##=========================================================================================
##===================== Initialize Parameters (quarterly) =================================
##=========================================================================================

## model
beta    = 0.9894    # discount factor
rho     = 2         # CRRA
alpha   = 0.36      # cap. share
delta   = 0.025     # depr. rate
l       = 1/0.9     # time worked per employee
mu      = 0.15      # unemploym. insurance payment
D       = 0.00625   # prob. of death
s2_thet = 0.04      # var. of log of non-permanent idiosyncr. shock
s2_psi  = 0.04/11   # var. of log of permanent idiosyncr. shock
Omega   = 0.07      # unemployment rate
bet_p   = 0.9867    # mean of distr for b-dist model
bet_d   = 0.0067    # range of distr for b-dist model 

## approximation param.
nm, mmin, mmax      = 600, 0.01, 250       # cash-on-hand grid
nthet, npsi         = 8, 8              # no. of states for discrete shocks
n                   = 8000              # number of agents in the economy

## discretize shocks
thet_val, thet_p    = approx_lognorm2(m.sqrt(s2_thet), nthet)
psi_val, psi_p      = approx_lognorm2(m.sqrt(s2_psi), npsi)

### simulate and plot trajectories of permanent and non-permanent productivity
np.random.seed(118118)
tau       = (mu*Omega) / (l * (1-Omega))         # tax rate
lifetime  = np.random.geometric(p = D, size = 3)
ran       = np.max(lifetime)                     # time horizon
xi0       = np.zeros(lifetime[0])
xi1       = np.zeros(lifetime[1])
xi2       = np.zeros(lifetime[2])
p0        = np.zeros(lifetime[0])
p1        = np.zeros(lifetime[1])
p2        = np.zeros(lifetime[2])

u_tmp     = np.random.choice([0,1],(ran,3),True,[(1-Omega),Omega])
thet_tmp  = np.random.choice(thet_val, (ran,3), True, thet_p)
xi_tmp    = u_tmp*mu + (1-u_tmp)*(1-tau)*l*thet_tmp
psi_tmp   = np.random.choice(psi_val, (ran,3), True, psi_p)

# starting values
xi0[0]   = xi_tmp[0,0]
p0[0]    = psi_tmp[0,0]
xi1[0]   = xi_tmp[0,1]
p1[0]    = psi_tmp[0,1]
xi2[0]   = xi_tmp[0,2]
p2[0]    = psi_tmp[0,2]

# fill remaining values
t = 1
while t < ran:
    if t < lifetime[0]:
        xi0[t] = xi_tmp[t,0]
        p0[t]  = p0[t-1] * psi_tmp[t,0]
    
    if t < lifetime[1]:
        xi1[t] = xi_tmp[t,1]
        p1[t]  = p1[t-1] * psi_tmp[t,1]
    
    if t < lifetime[2]:
        xi2[t] = xi_tmp[t,2]
        p2[t]  = p2[t-1] * psi_tmp[t,2]
    t = t+1

# Plot
fig, axs = plt.subplots(2)
axs[0].plot(range(lifetime[0]), p0, c = "green")
axs[0].plot(range(lifetime[1]), p1, c = "violet")
axs[0].plot(range(lifetime[2]), p2, c = "blue")
axs[0].set_ylabel(r'$p_t$')
axs[1].plot(range(lifetime[0]), xi0, c = "green")
axs[1].plot(range(lifetime[1]), xi1, c = "violet")
axs[1].plot(range(lifetime[2]), xi2, c = "blue")
axs[1].set_ylabel(r'$\xi_t$')
axs[1].set_xlabel(r'Period $t$')
plt.savefig('figures/productivit.pdf')

##=========================================================================================
##======================== B-Point Model: Steady State ====================================
##=========================================================================================

B_point_S = Economy_Steady(thet_p, thet_val, psi_p, psi_val) # create steady state economy
B_point_S.set_R(0.03367)                                     # SS R
K_sim = B_point_S.Iterate()
R_sim = alpha*(l*(1-Omega)/K_sim)**(1-alpha)
print("R_sim = ", R_sim)
#B_point_S.Steady()                                           # compute steady state 

## get all distributions and steady state R
k_bp, c_bp, m_bp, p_bp = B_point_S.get_dist()
R_bp                   = B_point_S.get_R()

### Plot Lorentz Curve for Capital k
plt.figure()
x, kd, comp, kgini = B_point_S.Lorentz("k")
plt.plot(x, kd, c = "blue", linewidth = 2)
plt.plot(x, comp, c = "black", linewidth = 1)
plt.xlabel('Individuals')
plt.ylabel(r'Cumulative Capital $k$')
plt.savefig('figures/Lorentz_k_Bp.pdf')
print('Gini Coefficient b-point: ', np.round(kgini, decimals = 4))

### Plot distribution of cash on hand
plt.figure()
plt.hist(m_bp, bins = np.arange(0,80+2,2), color = 'black', lw = 0.3, density = True, alpha = 0.5)
plt.xlim([0,80])
plt.xlabel(r'Normalized cash on hand $m$')
plt.ylabel('density')
plt.savefig('figures/mdist_Bp.pdf')

### VFI and get consumption function
Indiv = Agent_Steady(beta, thet_p, thet_val, psi_p, psi_val) # initialize agent
Indiv.set_R(R_bp)                                            # set steady state interest rate
Indiv.VFI()                                                  # value function iteration
mgrid = Indiv.get_mgrid()
cfun  = Indiv.get_C()

# Plot consumption function
ind   = np.where(mgrid <= 50) # range to plot
x     = mgrid[ind]
y     = cfun[ind]
plt.figure()
plt.plot(x,y, c = "green", linewidth = 2)
plt.xlabel(r'Normalized cash on hand $m$')
plt.ylabel(r'Normalized consumption $c$')
plt.savefig('figures/cfun_Bp.pdf')

#=========================================================================================
#======================== B-Dist Model: Steady State =====================================
#=========================================================================================

B_dist_S = Economy_Steady(thet_p, thet_val, psi_p, psi_val, n = 8000, dist=True)
B_dist_S.set_R(0.03565) # SS R
B_dist_S.Iterate()
##B_dist_S.Steady()

#get all distributions and steady state R
k_bd, c_bd, m_bd, p_bd = B_dist_S.get_dist()
R_bd                   = B_dist_S.get_R()

### Plot Lorentz Curves and compare
plt.figure()
x, kd, comp, kgini = B_point_S.Lorentz("k")
x2, kd2, comp2, kgini2 = B_dist_S.Lorentz("k")
plt.plot(x, kd, c = "blue", linewidth = 2, label = r'$\beta$-point')
plt.plot(x, kd2, c = "green", linewidth = 2, label = r'$\beta$-dist')
plt.plot(x, comp, c = "black", linewidth = 1)
plt.xlabel('Individuals')
plt.ylabel(r'Cumulative Capital $k$')
plt.legend()
plt.savefig('figures/Lorentz_k_both.pdf')
print('Gini Coefficient b-dist: ', np.round(kgini2, decimals = 4))

### Plot distribution of cash on hand
impat = m_bd[:1000]
pat   = m_bd[7000:]
plt.figure()
plt.hist(impat, bins = np.arange(0,90+2,2), color = 'green', lw = 0.3, density = True, alpha = 0.4, label = r'$\beta = \beta^\prime - \Delta$')
plt.hist(pat, bins = np.arange(0,90+2,2), color = 'black', lw = 0.3, density = True, alpha = 0.4, label = r'$\beta = \beta^\prime + \Delta$')
plt.xlabel(r'Normalized cash on hand $m$')
plt.ylabel('density')
plt.legend()
plt.savefig('figures/mdist_Bd.pdf')

### VFI and get consumption function for patient and impatient
# impatient
Impat = Agent_Steady(bet_p - bet_d, thet_p, thet_val, psi_p, psi_val)
Impat.set_R(R_bd)
Impat.VFI()
mgrid  = Impat.get_mgrid()
cimpat = Impat.get_C()

# patient
Pat   = Agent_Steady(bet_p + bet_d, thet_p, thet_val, psi_p, psi_val)
Pat.set_R(R_bd)
Pat.VFI()
cpat = Pat.get_C()

# plot
ind   = np.where(mgrid <= 50) # range to plot
x     = mgrid[ind]
y1    = cimpat[ind]
y2    = cpat[ind]
plt.figure()
plt.plot(x,y1, c = "green", linewidth = 2, label = r'$\beta = \beta^\prime - \Delta$')
plt.plot(x,y2, c = "blue", linewidth = 2, label = r'$\beta = \beta^\prime + \Delta$')
plt.xlabel(r'Normalized cash on hand $m$')
plt.ylabel(r'Normalized consumption $c$')
plt.legend()
plt.savefig('figures/cfun_Bd.pdf')

##=========================================================================================
##======================== B-Point Model: Stimulus Check ==================================
##=========================================================================================

periods         = 250
K_bp            = (alpha/R_bp)**(1/(1-alpha)) * l * (1 - Omega) # steady state K
W_bp            = (1-alpha) * (K_bp/(l*(1-Omega)))**alpha       # steady state W
C_bp            = np.mean(c_bp)
Y_bp            = K_bp**alpha * (l*(1-Omega))**(1-alpha)
R_guess         = alpha * (l*(1-Omega)/(K_bp + 1))**(1-alpha)   # R for K_ss + 1
Rpath           = np.ones(periods) * R_bp
Rpath[0:250]     = np.linspace(R_guess, R_bp, 250)              # guess for path
Rpath[periods-1]= R_bp                                          # at the end economy has to be in SS
Rpath[0]        = R_bp                                          # element 0 needs to have ss R (because of normalization with W!!!)

# get cash on hand after stimulus check
m_init = m_bp + 1/(W_bp * p_bp) # add one asset, normalize with P and W

Stimulus = Economy_Shoot(thet_p, thet_val, psi_p, psi_val, Rpath, Indiv.get_v(),
                         Indiv.get_C(), p_bp, m_init)

## solve path with shooting method
Stimulus.Iterate()

# get simulated distribution path
m_stim, c_stim, k_stim, p_stim = Stimulus.get_dist()

# plot trajectories of R, W, capital and output

# plot trajectory of R
Rstim, Wstim, Kstim, Ystim = Stimulus.get_Paths()
x = np.array(range(periods - 1)) + 1 # time axis

fig, axs = plt.subplots(2,2)
axs[0,0].plot(x, Rstim[1:])
axs[0,0].plot(x, np.ones(periods-1)*R_bp, c = 'black')
axs[0,0].set_title(r'Interest rate $r$')
axs[0,0].tick_params(axis = 'x', colors = 'w')
axs[0,1].plot(x, Wstim[1:])
axs[0,1].plot(x, np.ones(periods-1)*W_bp, c = 'black')
axs[0,1].set_title(r'Wage $W$')
axs[0,1].tick_params(axis = 'x', colors = 'w')
axs[1,0].plot(x, Kstim[1:])
axs[1,0].plot(x, np.ones(periods-1)*K_bp, c = 'black')
axs[1,0].set_title(r'Capital $K$')
axs[1,0].set_xlabel(r'Quarters after stimulus $t$')
axs[1,1].plot(x, Ystim[1:])
axs[1,1].plot(x, np.ones(periods-1)*Y_bp, c = 'black')
axs[1,1].set_title(r'Output $Y$')
axs[1,1].set_xlabel(r'Quarters after stimulus $t$')
plt.savefig('figures/stim_transition.pdf')

# plot trajectory of consumption
plt.figure()
y = np.mean(c_stim[:,:(periods-1)], axis = 0)
x = range(periods-1)
plt.plot(x, y, c = "green", linewidth = 2)
plt.plot(x, np.ones(periods-1)*C_bp, c = "black")
plt.xlabel(r'Quarters after stimulus $t$')
plt.ylabel(r'Consumption $C$')
plt.savefig('figures/stim_consumption.pdf')

# compute MPC
mpc = np.sum(np.mean(c_stim[:,:4], axis = 0)  - C_bp)
print("Marginal Propensity to consume = ", mpc)

# compute effect on consumption of wealth quantiles
inds = np.argsort(k_bp)     # sort according to wealth before stimulus
mpcs = np.zeros(4)          # initialize MPCs array
css  = np.zeros(4)          # steady state consumption for quartiles

for i in range(4):
    css[i]  = np.mean(c_bp[ inds[(i*int(n/4)):((i+1)*int(n/4))] ])
    mpcs[i] = np.sum(np.mean(c_stim[ inds[(i*int(n/4)):((i+1)*int(n/4))],:4 ], axis = 0) - css[i])

# plot MPCs
x = ["Quartile 1", "Quartile 2", "Quartile 3", "Quartile 4",]
plt.figure()
plt.bar(x, mpcs, color = 'y')
plt.ylabel('MPC (first year)')
plt.xlabel('Wealth Quartile (before stimulus)')
plt.savefig('figures/MPCs.pdf')

# plot path of gini coefficients of capital and consumption
k_gini = np.zeros(periods - 1)
c_gini = np.zeros(periods - 1)

for i in range(periods - 1):
    k_gini[i] = gini(k_stim[:,i+1])
    c_gini[i] = gini(c_stim[:,i])

# gini path of k
xax = np.array(range(periods-1)) +1
plt.figure()
plt.plot(xax, np.ones(periods-1)*gini(k_bp), c = 'black')
plt.plot(xax, k_gini, c = 'blue')
plt.xlabel(r'Quarters after stimulus $t$')
plt.ylabel(r'Gini coefficient of $k_i$')
plt.savefig('figures/kgini.pdf')

# gini path of c
xax = np.array(range(periods-1)) 
plt.figure()
plt.plot(xax, np.ones(periods-1)*gini(c_bp), c = 'black')
plt.plot(xax, c_gini, c = 'green')
plt.xlabel(r'Quarters after stimulus $t$')
plt.ylabel(r'Gini coefficient of $c_i$')
plt.savefig('figures/cgini.pdf')

