"""
@author: EAK
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

#### data read-in ############################################################

dir='/Users/Emile/Documents/Graduate/AntonPaarData/Porifera/summary/shear data'
os.chdir(dir) # change to directory where data are stored

s  = ('Axinella', 'Callyspongia', 'Cliona') # shear summarized data by species
ns = len(s)                                 # number of species files
s_ = np.arange(0, ns, 1)                    # array of species data for looping

df = [(pd.read_csv(s[i]+'_shearsummary.csv')).values for i in s_] # read 'em

nr = np.int64(np.size(df, 2)/8) # number of distinct runs
r_ = np.arange(0, nr, 1)        # array of runs for loopin' over runs

na = np.int64(np.size(df, 1))   # number of amplitude points swept
a_ = np.arange(0, na, 1)        # array of points for loopin' over points

# variables, loop over species and runs to get all columns for each
# shear strains, dimensionless
gams = np.asarray([df[i][:, 8*j+1] for i in s_ for j in r_]).reshape(ns,nr,na)
# shear stresses, Pa
taus = np.asarray([df[i][:, 8*j+2] for i in s_ for j in r_]).reshape(ns,nr,na)
# storage moduli, Pa
Gps  = np.asarray([df[i][:, 8*j+3] for i in s_ for j in r_]).reshape(ns,nr,na)
# loss moduli, Pa
Gpps = np.asarray([df[i][:, 8*j+4] for i in s_ for j in r_]).reshape(ns,nr,na)
# minimum strain modulus, Pa
GpMs = np.asarray([df[i][:, 8*j+5] for i in s_ for j in r_]).reshape(ns,nr,na)
# maximum strain modulus, Pa
GpLs = np.asarray([df[i][:, 8*j+6] for i in s_ for j in r_]).reshape(ns,nr,na)
# normal stress, Pa
sigs = np.asarray([df[i][:, 8*j+7] for i in s_ for j in r_]).reshape(ns,nr,na)

#### calculations & averaging ################################################

# stiffening index, S
Ss = np.asarray([(GpLs[i, j, :] - GpMs[i, j, :])/GpLs[i, j, :] for i in s_ 
                 for j in r_]).reshape(ns,nr,na)
# tangent of G"/G'
tandels = np.asarray([Gpps[i, j, :]/Gps[i, j, :] for i in s_ 
                      for j in r_]).reshape(ns,nr,na)
# softening indices, WM and WL
WMs = np.asarray([(np.diff(GpMs[i, j, :]) / \
                   np.diff(gams[i, j, :]))/GpMs[i, j, 0] for i in s_ 
                  for j in r_]).reshape(ns,nr,na-1)
WLs = np.asarray([(np.diff(GpLs[i, j, :]) / \
                   np.diff(gams[i, j, :]))/GpLs[i, j, 0] for i in s_ 
                  for j in r_]).reshape(ns,nr,na-1)

# calculate onsets of different nonlinearity

# get first instances where S is >=0.02 for intracycle stiffening
ia = np.asarray([np.where(Ss[i, j, :] > 0.02)[0][0] for i in s_ 
                 for j in r_]).reshape(ns,nr)
# get shear strains at these instances
gams_a = np.asarray([gams[i,j,ia[i,j]] for i in s_ for j in r_]).reshape(ns,nr)

# get ratio of current GpM to GpM at previous amplitude
deltaGpM = np.asarray([(GpMs[i, j, k-1]-GpMs[i, j, k])/GpMs[i, j, k-1] 
                       for i in s_ for j in r_ 
                       for k in np.arange(1,na,1)]).reshape(ns,nr,na-1)
# get first instances where this ratio is >=0.05 for intercycle softening
ib = np.asarray([np.where(deltaGpM[i, j, :] > 0.05)[0][0] for i in s_ 
                 for j in r_]).reshape(ns,nr)
# get shear strains at these other instances
gams_b = np.asarray([gams[i,j,ib[i,j]] for i in s_ for j in r_]).reshape(ns,nr)

# check where WL is negative
checkc = np.asarray([np.where(WLs[i, j, :] < 0) for i in s_ 
                     for j in r_]).reshape(ns,nr)
# now, the sizes of these provides where, if, WL becomes positive for inter-
# cycle stiffening. if it never does, the value will be the full size of WL,
# or na-1
ic = np.asarray([np.size(checkc[i][j], axis = 0) for i in s_ 
                 for j in r_]).reshape(ns,nr)
# finally, get shear strains at these instances of intercycle stiffening
gams_c = np.asarray([gams[i,j,ic[i,j]] for i in s_ for j in r_]).reshape(ns,nr)

# averages and standard errors of critical strains for the first round
gam_a1   = [100*np.mean(gams_a[i, ::2], axis = 0) for i in s_]
gam_a1_e = [100*np.std(gams_a[i, ::2], axis = 0)/2 for i in s_]
gam_b1   = [100*np.mean(gams_b[i, ::2], axis = 0) for i in s_]
gam_b1_e = [100*np.std(gams_b[i, ::2], axis = 0)/2 for i in s_]
gam_c1   = [100*np.mean(gams_c[i, ::2], axis = 0) for i in s_]
gam_c1_e = [100*np.std(gams_c[i, ::2], axis = 0)/2 for i in s_]
# for the second round
gam_a2   = [100*np.mean(gams_a[i, 1::2], axis = 0) for i in s_]
gam_a2_e = [100*np.std(gams_a[i, 1::2], axis = 0)/2 for i in s_]
gam_b2   = [100*np.mean(gams_b[i, 1::2], axis = 0) for i in s_]
gam_b2_e = [100*np.std(gams_b[i, 1::2], axis = 0)/2 for i in s_]
gam_c2   = [100*np.mean(gams_c[i, 1::2], axis = 0) for i in s_]
gam_c2_e = [100*np.std(gams_c[i, 1::2], axis = 0)/2 for i in s_]

# calculate first round averages of all other variables
gam1    = [np.mean(100*gams[i, ::2, :], axis = 0) for i in s_] # change to %
tau1    = [np.mean(taus[i, ::2, :], axis = 0) for i in s_]
Gp1     = [np.mean(Gps[i, ::2, :], axis = 0) for i in s_]
Gpp1    = [np.mean(Gpps[i, ::2, :], axis = 0) for i in s_]
GpM1    = [np.mean(GpMs[i, ::2, :], axis = 0) for i in s_]
GpL1    = [np.mean(GpLs[i, ::2, :], axis = 0) for i in s_]
sig1    = [np.mean(sigs[i, ::2, :], axis = 0) for i in s_]
S1      = [np.mean(Ss[i, ::2, :], axis = 0) for i in s_]
tandel1 = [np.mean(tandels[i, ::2, :], axis = 0) for i in s_]
WM1     = [np.mean(WMs[i, ::2, :], axis = 0) for i in s_]
WL1     = [np.mean(WLs[i, ::2, :], axis = 0) for i in s_]
# calculate second round averages
gam2    = [np.mean(100*gams[i, 1::2, :], axis = 0) for i in s_]
tau2    = [np.mean(taus[i, 1::2, :], axis = 0) for i in s_]
Gp2     = [np.mean(Gps[i, 1::2, :], axis = 0) for i in s_]
Gpp2    = [np.mean(Gpps[i, 1::2, :], axis = 0) for i in s_]
GpM2    = [np.mean(GpMs[i, 1::2, :], axis = 0) for i in s_]
GpL2    = [np.mean(GpLs[i, 1::2, :], axis = 0) for i in s_]
sig2    = [np.mean(sigs[i, 1::2, :], axis = 0) for i in s_]
S2      = [np.mean(Ss[i, 1::2, :], axis = 0) for i in s_]
tandel2 = [np.mean(tandels[i, 1::2, :], axis = 0) for i in s_]
WM2     = [np.mean(WMs[i, 1::2, :], axis = 0) for i in s_]
WL2     = [np.mean(WLs[i, 1::2, :], axis = 0) for i in s_]
# calculate first round standard errors
gam1_e    = [np.std(100*gams[i, ::2, :], axis = 0)/2 for i in s_]
tau1_e    = [np.std(taus[i, ::2, :], axis = 0)/2 for i in s_]
Gp1_e     = [np.std(Gps[i, ::2, :], axis = 0)/2 for i in s_]
Gpp1_e    = [np.std(Gpps[i, ::2, :], axis = 0)/2 for i in s_]
GpM1_e    = [np.std(GpMs[i, ::2, :], axis = 0)/2 for i in s_]
GpL1_e    = [np.std(GpLs[i, ::2, :], axis = 0)/2 for i in s_]
sig1_e    = [np.std(np.abs(sigs[i, ::2, :]), axis = 0)/2 for i in s_]
S1_e      = [np.std(Ss[i, ::2, :], axis = 0)/2 for i in s_]
tandel1_e = [np.std(tandels[i, ::2, :], axis = 0)/2 for i in s_]
WM1_e     = [np.std(WMs[i, ::2, :], axis = 0)/2 for i in s_]
WL1_e     = [np.std(WLs[i, ::2, :], axis = 0)/2 for i in s_]
# calculate second round standard errors
gam2_e    = [np.std(100*gams[i, 1::2, :], axis = 0)/2 for i in s_]
tau2_e    = [np.std(taus[i, 1::2, :], axis = 0)/2 for i in s_]
Gp2_e     = [np.std(Gps[i, 1::2, :], axis = 0)/2 for i in s_]
Gpp2_e    = [np.std(Gpps[i, 1::2, :], axis = 0)/2 for i in s_]
GpM2_e    = [np.std(GpMs[i, 1::2, :], axis = 0)/2 for i in s_]
GpL2_e    = [np.std(GpLs[i, 1::2, :], axis = 0)/2 for i in s_]
sig2_e    = [np.std(np.abs(sigs[i, 1::2, :]), axis = 0)/2 for i in s_]
S2_e      = [np.std(Ss[i, 1::2, :], axis = 0)/2 for i in s_]
tandel2_e = [np.std(tandels[i, 1::2, :], axis = 0)/2 for i in s_]
WM2_e     = [np.std(WMs[i, 1::2, :], axis = 0)/2 for i in s_]
WL2_e     = [np.std(WLs[i, 1::2, :], axis = 0)/2 for i in s_]

#### plotting ################################################################

# set default font and activate Latex
mpl.rc('font', family = 'Courier New')
mpl.rc('text', usetex = True)
# colors
spixclrs = [cm.cool(x) for x in np.linspace(0, 1, 6)]
clrs     = [spixclrs[5],spixclrs[4],spixclrs[2]]
# labels
spp  = ('A. polycapella', 'Callyspongia sp.', 'C. celata')
spp  = [r'\textit{%s}' % spp[i] for i in [0,1,2]] # italicize species names

# GpM and GpL vs. strain amplitude figures and axes
plt0 = [plt.subplots(nrows = 1, ncols = 1, figsize = (12, 10)) for i in s_]
fig0 = [plt0[i][0] for i in s_]
ax0 = [plt0[i][1] for i in s_]
[ax0[j].errorbar(gam1[j], GpM1[j], yerr = GpM1_e[j], fmt = 'v', ms = 24, 
                 mew = 3, mfc = clrs[j], mec = 'black', ecolor = 'dimgray', 
                 capsize = 6, label = '$G^{\prime}_{M}$ 1st round') 
 for j in s_]
[ax0[j].plot([gam_b1[j],gam_b1[j],gam_b1[j]],[100,10000,100000000], ls = '-', 
              lw = 4, c = clrs[j], 
              label = '$\gamma_{a}=$ %.2f' % gam_b1[j] +'\%') for j in s_]
[ax0[j].plot([gam_a1[j],gam_a1[j],gam_a1[j]],[100,10000,100000000], ls = '--', 
              lw = 4, c = clrs[j], 
              label = '$\gamma_{b}=$ %.2f' % gam_a1[j] +'\%') for j in s_]
[ax0[j].plot([gam_c2[j],gam_c2[j],gam_c2[j]],[100,10000,100000000], ls = ':', 
              lw = 4, c = clrs[j], label = '$\gamma_{c}=$ %.2f'%gam_c2[j]+'\%') 
  for j in s_]
[ax0[j].errorbar(gam1[j], GpL1[j], yerr = GpL1_e[j], fmt = '^', ms = 24, 
                 mew = 3, mfc = clrs[j], mec = 'black', ecolor = 'dimgray', 
                 capsize = 6, label = '$G^{\prime}_{L}$ 1st round') 
 for j in s_]
[ax0[j].errorbar(gam2[j], GpM2[j], yerr = GpM2_e[j], fmt = 'v', ms = 24, 
                 mew = 3, mfc = 'gainsboro', mec = clrs[j], ecolor = 'dimgray', 
                 capsize = 6, label = '$G^{\prime}_{M}$ 2nd round') 
 for j in s_]
[ax0[j].errorbar(gam2[j], GpL2[j], yerr = GpL2_e[j], fmt = '^', ms = 24, 
                 mew = 3, mfc = 'gainsboro', mec = clrs[j], ecolor = 'dimgray', 
                 capsize = 6, label = '$G^{\prime}_{L}$ 2nd round') 
 for j in s_]
[ax0[j].set_xlabel('$\gamma_{0}$ (\%)', fontsize = 68) for j in s_]
[ax0[j].set_ylabel('$G^{\prime}_{X}$ (Pa)', fontsize = 68, labelpad = -10) 
 for j in s_]
[ax0[j].set_xscale('log') for j in s_]
[ax0[j].set_yscale('log') for j in s_]
[ax0[j].set_xlim([0.015, 25]) for j in s_]
[ax0[j].set_ylim([3.5*10**4, 2.5*10**6]) for j in s_]
[ax0[j].tick_params(axis = 'both', which = 'both', labelsize = 58) for j in s_]
hndls0, lbls0 = map(list,zip(*[ax0[j].get_legend_handles_labels() 
                               for j in s_]))
[ax0[j].legend(hndls0[j], lbls0[j], loc = 'best', fancybox = True, 
               shadow = True, fontsize = 28) for j in s_]
[fig0[j].savefig('plots/' + s[j] + '_GpMGpLvsgam.svg', bbox_inches = 'tight', 
                 dpi = 1200) for j in s_]

fig1, ax1 = plt.subplots(figsize = (12,10))
[ax1.errorbar(gam2[j], S2[j], yerr = S2_e[j], fmt = 's', ms = 20, mew = 3, 
              mfc = 'gainsboro', mec = clrs[j], ecolor = 'dimgray', 
              capsize = 6) for j in s_]
[ax1.errorbar(gam1[j], S1[j], yerr = S1_e[j], fmt = 's', ms = 20, mew = 3, 
              mfc = clrs[j], mec = 'black', ecolor = 'dimgray', capsize = 6, 
              label = spp[j]) for j in s_]
ax1.set_xlabel('$\gamma_{0}$ (\%)', fontsize = 68)
ax1.set_ylabel('$S$', fontsize = 68, rotation = 0, labelpad = 50)
ax1.set_xscale('log')
ax1.set_xlim([0.015, 25])
ax1.set_ylim([-0.03, 0.71])
ax1.tick_params(axis = 'both', which = 'major', labelsize = 58)
hndls1, lbls1 = ax1.get_legend_handles_labels()
ax1.legend(hndls1, lbls1, loc = 'upper left', fancybox = True, shadow = True, 
            fontsize = 28)
fig1.savefig('plots/Svsgam.svg', bbox_inches = 'tight', dpi = 1200)

fig2, ax2 = plt.subplots(figsize = (12,10))
[ax2.errorbar((gam2[j][:-1]+gam2[j][1:])/2, WL2[j], yerr = WL2_e[j], fmt = '^', 
              ms = 20, mew = 3, mfc = 'gainsboro', mec = clrs[j], 
              ecolor = 'dimgray', capsize = 6) for j in s_]
[ax2.errorbar((gam1[j][:-1]+gam1[j][1:])/2, WL1[j], yerr = WL1_e[j], fmt = '^', 
              ms = 20, mew = 3, mfc = clrs[j], mec = 'black', 
              ecolor = 'dimgray', capsize = 6, label = spp[j]) for j in s_]
[ax2.errorbar((gam2[j][:-1]+gam2[j][1:])/2, WM2[j], yerr = WM2_e[j], fmt = 'v', 
              ms = 20, mew = 3, mfc = 'gainsboro', mec = clrs[j], 
              ecolor = 'dimgray', capsize = 6) for j in s_]
[ax2.errorbar((gam1[j][:-1]+gam1[j][1:])/2, WM1[j], yerr = WM1_e[j], fmt = 'v', 
              ms = 20, mew = 3, mfc = clrs[j], mec = 'black', 
              ecolor = 'dimgray', capsize = 6, label = spp[j]) for j in s_]
ax2.plot([0.02,0.2,2,20],[0,0,0,0], lw = 3, c = 'silver', ls = '--')
ax2.set_xlabel('$\gamma_{0}$ (\%)', fontsize = 68)
ax2.set_ylabel('$W_{X}$', fontsize = 68)
ax2.set_xscale('log')
ax2.set_xlim([0.025, 19])
ax2.set_ylim([-295, 29.5])
ax2.tick_params(axis = 'both', which = 'major', labelsize = 58)
axins2 = inset_axes(ax2, width = '50%', height = '50%', loc = 4)
[axins2.errorbar((gam2[j][:-1]+gam2[j][1:])/2, WL2[j], yerr = WL2_e[j], 
                 fmt = '^', ms = 20, mew = 3, mfc = 'gainsboro', mec = clrs[j], 
                 ecolor = 'dimgray', capsize = 6) for j in s_]
[axins2.errorbar((gam1[j][:-1]+gam1[j][1:])/2, WL1[j], yerr = WL1_e[j], 
                 fmt = '^', ms = 20, mew = 3, mfc = clrs[j], mec = 'black', 
                 ecolor = 'dimgray', capsize = 6, label = spp[j]) for j in s_]
[axins2.errorbar((gam2[j][:-1]+gam2[j][1:])/2, WM2[j], yerr = WM2_e[j], 
                 fmt = 'v', ms = 20, mew = 3, mfc = 'gainsboro', mec = clrs[j], 
                 ecolor = 'dimgray', capsize = 6) for j in s_]
[axins2.errorbar((gam1[j][:-1]+gam1[j][1:])/2, WM1[j], yerr = WM1_e[j], 
                 fmt = 'v', ms = 20, mew = 3, mfc = clrs[j], mec = 'black', 
                 ecolor = 'dimgray', capsize = 6, label = spp[j]) for j in s_]
axins2.plot([0.02,0.2,2,20],[0,0,0,0], lw = 3, c = 'silver', ls = '--')
mark_inset(ax2, axins2, loc1 = 2, loc2 = 1, fc = 'none', ec = '0')
axins2.set_xlim(3.2, 16)
axins2.set_ylim(-6.4, 6.4)
axins2.xaxis.set_ticks_position('top') 
axins2.tick_params(axis = 'both', which = 'both', labelsize = 38)
fig2.savefig('plots/Wvsgam.svg', bbox_inches = 'tight', dpi = 1200)

fig3, ax3 = plt.subplots(figsize = (12,10))
[ax3.errorbar(gam2[j], sig2[j]/1000, xerr = gam2_e[j], yerr = sig2_e[j]/1000, 
              fmt = 's', ms = 20, mew = 3, mfc = 'gainsboro', mec = clrs[j], 
              ecolor = 'dimgray', capsize = 6) for j in s_]
[ax3.errorbar(gam1[j], sig1[j]/1000, xerr = gam1_e[j], yerr = sig1_e[j]/1000, 
              fmt = 's', ms = 20, mew = 3, mfc = clrs[j], mec = 'black', 
              ecolor = 'dimgray', capsize = 6, label = spp[j]) for j in s_]
ax3.plot([0.01,0.1,1,10,100],[0,0,0,0,0], lw = 3, c = 'silver', ls = '--')
ax3.set_xscale('log')
ax3.set_xlim([0.015, 25])
ax3.set_ylim([-16.5, 6.5])
ax3.set_yticks([-15, -10, -5, 0, 5])
ax3.set_xlabel('$\gamma_{0}$ (\%)', fontsize = 68)
ax3.set_ylabel(r'$\sigma_{zz}$ (kPa)', fontsize = 68)
ax3.tick_params(axis = 'both', which = 'major', labelsize = 58)
plt.savefig('plots/normvsgam.svg', bbox_inches = 'tight', dpi = 1200)

fig4, ax4 = plt.subplots(figsize = (12,10))
[ax4.errorbar(gam2[j], tandel2[j], xerr = gam2_e[j], yerr = tandel2_e[j], 
              fmt = 's', ms = 20, mew = 3, mfc = 'gainsboro', mec = clrs[j], 
              ecolor = 'dimgray', capsize = 6) for j in s_]
[ax4.errorbar(gam1[j], tandel1[j], xerr = gam1_e[j], yerr = tandel1_e[j], 
              fmt = 's', ms = 20, mew = 3, mfc = clrs[j], mec = 'black',
              ecolor = 'dimgray', capsize = 6, label = spp[j]) for j in s_]
ax4.set_xlabel('$\gamma_{0}$ (\%)', fontsize = 68)
ax4.set_ylabel(r'$\tan\delta$', fontsize = 68)
ax4.set_xscale('log')
ax4.set_xlim([0.015, 25])
ax4.set_ylim([0.03, 0.43])
ax4.tick_params(axis = 'both', which = 'major', labelsize = 58)
plt.savefig('plots/tandelvsgam.svg', bbox_inches = 'tight', dpi = 1200)