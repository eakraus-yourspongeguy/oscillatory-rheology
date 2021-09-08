"""
@author: EAK
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import rheofuncs

#### Data Read-In ############################################################

# directory
dir = '/Users/Emile/Documents/Graduate/AntonPaarData/Porifera/summary/'+\
      'frequency data'
os.chdir(dir)
# file
datstr = 'freqsweepfullsummary'
dat    = pd.read_csv(datstr + '.csv').values 
n      = np.arange(0, np.int64(np.size(dat, 1)/8), 1) # number of species array
# columns
phi   = [dat[0, 8*i+1] for i in n] # mass porosity
phi_e = [dat[0, 8*i+2] for i in n] # mass porosity uncertainty
w     = [dat[:, 8*i+3].astype('float64') for i in n] # frequency, rad/s
Gp    = [dat[:, 8*i+4].astype('float64') for i in n] # storage modulus, Pa
Gpp   = [dat[:, 8*i+5].astype('float64') for i in n] # loss modulus, Pa
Gp_e  = [dat[:, 8*i+6].astype('float64') for i in n] # stored uncertainty, Pa
Gpp_e = [dat[:, 8*i+7].astype('float64') for i in n] # loss uncertainty, Pa

#### Calculations ############################################################

# loss tangent
tand = [Gpp[i]/Gp[i] for i in n]
# propagate its uncertainty from G' and G" uncertainties
tand_e = [np.sqrt((Gp_e[i]*Gpp[i]/Gp[i]**2)**2+(Gpp_e[i]/Gp[i])**2) for i in n]
# linear viscoelastic fit parameters file
fitstr, mdlstr = 'fitparams'+datstr[9:13], '_Zener' # choose which one
fit = pd.read_csv(fitstr + mdlstr + '.csv').values
params = [fit[:, i + 1].astype('float64') for i in n]
# get best fit values by inputting fit parameters into the model
Gpmodel  = getattr(rheofuncs, mdlstr[1:] + 'Gp')
Gppmodel = getattr(rheofuncs, mdlstr[1:] + 'Gpp')
Gpbest   = [ Gpmodel(w[i], *params[i][:-1]) for i in n]
Gppbest  = [Gppmodel(w[i], *params[i][:-1]) for i in n]
tandbest = [Gppbest[i]/Gpbest[i] for i in n]
# power law exponent of G'
m     = [np.polyfit(np.log10(w[i]), np.log10(Gp[i]), 1, cov = True) for i in n]
alpha = [m[i][0][0] for i in n]
alpha_e = [np.sqrt(np.diag(m[i][1])[0]) for i in n]

#### Plotting ################################################################
# font
mpl.rc('font', family = 'Courier New')
mpl.rc('text', usetex = True)
# full species list
spp   = ['Axinella polycapella','Callyspongia sp.','Cinachyrella apion',
         'Cliona celata','Tectitethya keyensis','Tethya aurantium', 
         'Aplysina fulva','Igernella notabilis']
spp   = [r'\textit{%s}' % spp[i] for i in n[:-1]] # italicize
spp.insert(len(spp), 'synthetic sponge')  # add synthetic sponge label
# colors
spixclrs   = [cm.cool(x) for x in np.linspace(0, 1, 6)]
nospixclrs = [cm.spring(x) for x in np.linspace(0.6, 0.89, 2)]
clrs       = [spixclrs[5], spixclrs[4], spixclrs[3], spixclrs[2], spixclrs[1],
              spixclrs[0], nospixclrs[0], nospixclrs[1]]
clrs.insert(len(spp), 'xkcd:dried blood')
# markers
mrkrs = ['s','s', 's','s','s','s','o','o','*']

# Gp vs omega
fig0, axs0 = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 10))
[axs0.errorbar(w[i], Gp[i], yerr = Gp_e[i], ms = 20, fmt = mrkrs[i], mew = 3,
               mfc = clrs[i], mec = 'black', ecolor = 'dimgray', 
               label = spp[i], capsize = 6) for i in n]
[axs0.plot(w[i], Gpbest[i], ls = '--', lw = 3, c = clrs[i]) for i in n]
axs0.set_xscale('log')
axs0.set_yscale('log')
axs0.set_ylim([35000, 2500000])
axs0.tick_params(axis = 'both', which = 'major', labelsize = 44)
axs0.set_xlabel('$\omega$ (rad/s)', fontsize = 44, labelpad = -6)
axs0.set_ylabel('$G^{\prime}$ (Pa)', fontsize = 44, labelpad = -10)
hndls, lbls = axs0.get_legend_handles_labels() # create legend
fig0.legend(hndls, lbls, loc = 'center right', bbox_to_anchor = (-0.075,0.5), 
            ncol = 1, fancybox = True, shadow = True, fontsize = 40)
fig0.savefig('plots/Gpvsomega'+datstr[9:13]+mdlstr+'.svg', dpi = 1200, 
             bbox_inches = 'tight')

# Gpp vs omega
fig1, axs1 = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 10))
[axs1.errorbar(w[i], Gpp[i], yerr = Gpp_e[i], ms = 18, fmt = mrkrs[i], 
               mew = 3, mfc = clrs[i], mec = 'black', ecolor = 'dimgray', 
               label = spp[i], capsize = 6) for i in n]
[axs1.plot(w[i], Gppbest[i], ls = '--', lw = 3, c = clrs[i]) for i in n]
axs1.plot(w[8], Gppbest[8], ls = '--', lw = 3, c = clrs[8])
axs1.set_xscale('log')
axs1.set_yscale('log')
axs1.set_ylim([2000, 400000])
axs1.tick_params(axis = 'both', which = 'major', labelsize = 44)
axs1.set_xlabel('$\omega$ (rad/s)', fontsize = 44, labelpad = -6)
axs1.set_ylabel('$G^{\prime \prime}$ (Pa)', fontsize = 44, labelpad = -10)
fig1.savefig('plots/Gppvsomega'+datstr[9:13]+mdlstr+'.svg', dpi = 1200, 
             bbox_inches = 'tight')

# loss tangent vs omega
fig2, axs2 = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 10))
[axs2.errorbar(w[i], tand[i], yerr = tand_e[i], ms = 18, fmt = mrkrs[i], 
               mew = 3, mfc = clrs[i], mec = 'black', ecolor = 'dimgray', 
               label = spp[i], capsize = 6) for i in n]
[axs2.plot(w[i], tandbest[i], ls = '--', lw = 3, c = clrs[i]) for i in n]
max2, min2 = np.max(axs2.get_ylim()), np.min(axs2.get_ylim())
axs2.set_xscale('log')
axs2.set_yscale('log')
axs2.set_ylim([0.03, 0.3])
axs2.tick_params(axis = 'both', which = 'major', labelsize = 44)
axs2.tick_params(axis = 'y', which = 'minor', labelsize = 28)
axs2.set_xlabel('$\omega$ (rad/s)', fontsize = 44, labelpad = -6)
axs2.set_ylabel(r'$\tan\delta$', fontsize = 44, labelpad = -10)
fig2.savefig('plots/tandelvsomega'+datstr[9:13]+mdlstr+'.svg', dpi = 1200, 
             bbox_inches = 'tight')

# Gpp vs Gp
fig3, axs3 = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
[axs3.errorbar(Gp[i][6], Gpp[i][6], xerr = Gp_e[i][6], yerr = Gpp_e[i][6], 
               ms = 18, fmt = mrkrs[i], mew = 3, mfc = clrs[i], 
               mec = 'black', ecolor = 'dimgray', capsize = 6) for i in n]
x = [10**4, 10**5, 10**6, 10**7]
axs3.plot(x,[0.06*x[i] for i in [0,1,2,3]], lw = 3, ls = '-.', c = 'silver',
          label = '$G^{\prime \prime}=0.06G^{\prime}$')
axs3.plot(x,[0.18*x[i] for i in [0,1,2,3]], lw = 3, ls = ':', c = 'silver',
          label = '$G^{\prime \prime}=0.18G^{\prime}$')
axs3.set_xscale('log')
axs3.set_yscale('log')
axs3.set_xlim([4*10**4, 2*10**6])
axs3.set_ylim([2*10**3, 3*10**5])
axs3.tick_params(axis = 'both', which = 'major', labelsize = 44)
axs3.set_xlabel('$G^{\prime}(\omega=\pi)$ (Pa)', fontsize = 44, labelpad = -6)
axs3.set_ylabel('$G^{\prime \prime}(\omega=\pi)$ (Pa)', fontsize = 44)
hndls, lbls = axs3.get_legend_handles_labels() # create legend
axs3.legend(hndls, lbls, loc = 'upper left', fancybox = True, shadow = True, 
            fontsize = 40)
fig3.savefig('plots/GppvsGp.svg', dpi = 1200, bbox_inches = 'tight')

# alpha vs. specimen
fig4, axs4 = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
X = np.argsort(-np.asarray(alpha))
[axs4.bar(n[i], alpha[X[i]], 0.65, yerr = alpha_e[X[i]], color = clrs[X[i]], 
          capsize = 6, ecolor = 'dimgray') for i in n]
axs4.set_xticks(n)
axs4.set_xticklabels([])
axs4.set_ylim(0, 0.076)
axs4.set_yticks([0.01, 0.03, 0.05, 0.07])
axs4.tick_params(axis = 'y', which = 'major', labelsize = 44)
axs4.set_ylabel(r'$\alpha$', fontsize = 44, labelpad = 20, rotation = 0)
fig4.savefig('plots/alphabarplot.svg', dpi = 300, bbox_inches = 'tight')

# porosity vs. specimen
fig5, axs5 = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
X = np.argsort(-np.asarray(phi))
[axs5.bar(n[i], phi[X[i]], 0.65, yerr = phi_e[X[i]], color = clrs[X[i]], 
          capsize = 6, ecolor = 'dimgray') for i in n]
spp_sort = [spp[X[i]] for i in n]
axs5.set_xticks(n)
axs5.set_xticklabels(spp_sort, fontdict = {'fontsize': 33}, rotation = 90)
axs5.set_ylim(0, 0.96)
axs5.set_yticks([0.2, 0.4, 0.6, 0.8])
axs5.tick_params(axis = 'y', which = 'major', labelsize = 44)
axs5.set_ylabel(r'$\phi_{w}$', fontsize = 44, labelpad = 20, rotation = 0)
fig5.savefig('plots/phibarplot.svg', dpi = 300, bbox_inches = 'tight')

# spicules bar plot
fig6, axs6 = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
spix   = [43900, 1540, 4020, 2000000, 2780, 5960, 0, 0, 0]  # number per cm^3
spix_e = [20000,  400, 2000, 1750000, 2000, 2000, 0, 0, 0]  # uncertainty
[axs6.bar(n[i], spix[X[i]], 0.65, yerr = spix_e[X[i]], color = clrs[X[i]], 
          capsize = 6, ecolor = 'dimgray', rasterized = True) for i in n]
axs6.set_xticks(n)
axs6.set_xticklabels([])
axs6.set_yticks([1000, 10000, 100000, 1000000])
axs6.set_yscale('log')
axs6.tick_params(axis = 'y', which = 'major', labelsize = 44)
axs6.set_ylabel('$N_{spicules}$/cm$^{3}$', fontsize = 44, labelpad = 10)
fig6.savefig('plots/spiculesbarplot.svg', dpi = 300)