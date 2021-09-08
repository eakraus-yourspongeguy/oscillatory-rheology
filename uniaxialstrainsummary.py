"""
@author: EAK
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

#### Data Read-In ############################################################

dir = '/Users/Emile/Documents/Graduate/AntonPaarData/Porifera/summary/'+\
      'uniaxial data'
os.chdir(dir)

fs   = ['Axinella','Callyspongia','Cinachyrella','Cliona',
        'Tectitethya','Tethya','Aplysina','Igernella']    # sponge files
fs_  = np.arange(0, len(fs), 1)                           # array them
dfs  = [(pd.read_excel(fs[i] + '.xlsx', sheet_name = j)).values for i in fs_
        for j in [0,1]]                                   # dataframes
dfs_ = np.arange(0, len(dfs), 1)                          # array them
# average stress-strain-shear modulus curves and uncertainties
eps   = [100*dfs[i][:,0].astype('float64') for i in dfs_] # uniaxial strain, %
sig   = [dfs[i][:,1].astype('float64') for i in dfs_]     # uniaxial stress, Pa
Gp    = [dfs[i][:,2].astype('float64') for i in dfs_]     # G', Pa
Gpn   = [dfs[i][:,3].astype('float64') for i in dfs_]     # normalized G'
eps_e = [100*dfs[i][:,4].astype('float64') for i in dfs_]
sig_e = [dfs[i][:,5].astype('float64') for i in dfs_]
Gp_e  = [dfs[i][:,6].astype('float64') for i in dfs_]
# averaged moduli from each run
G0    = [dfs[i][0,8].astype('float64') for i in dfs_]   # shear modulus
G0_e  = [dfs[i][0,9].astype('float64') for i in dfs_]
Ec    = [dfs[i][0,10].astype('float64') for i in dfs_]  # compression modulus
Ec_e  = [dfs[i][0,11].astype('float64') for i in dfs_]
Es    = [dfs[i][0,12].astype('float64') for i in dfs_]  # extension modulus
Es_e  = [dfs[i][0,13].astype('float64') for i in dfs_]
E0    = [dfs[i][0,14].astype('float64') for i in dfs_]  # zero strain modulus
E0_e  = [dfs[i][0,15].astype('float64') for i in dfs_]

# other animal tissue and synthetic sponge data
dfo  = (pd.read_excel('mammalcompressstiff.xlsx')).values
fo   = np.arange(0, np.int64(np.size(dfo)/24), 1)
epso = [dfo[:,5*i+1].astype('float64') for i in fo]     # uniaxial strain, %
sigo = [dfo[:,5*i+2].astype('float64') for i in fo]     # uniaxial stress, Pa
Gpo  = [dfo[:,5*i+3].astype('float64') for i in fo]     # G', Pa
Eo   = [dfo[0,5*i+4].astype('float64') for i in fo]     # E, Pa
Eo_e = [dfo[1,5*i+4].astype('float64') for i in fo]

#### calculations ############################################################

# compression stiffening stress slope:
    
# for others
cso = [np.polyfit(-sigo[i][2:], Gpo[i][2:], 1, cov = True) for i in fo]

# indices to take fits at compression points
lsc = [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0]
lfc = [3, 5, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 3, 7, 7, 7]
# for sponges
cs  = [np.polyfit(-sig[i][lsc[i]:lfc[i]], Gp[i][lsc[i]:lfc[i]], 1, cov = True) 
       for i in dfs_]

# if the material follows classical Birch compression stiffening
nu  = [(9-2*cs[i][0][0])/(2*cs[i][0][0]+12) for i in dfs_] # Poisson's ratio
nuo = [(9-2*cso[i][0])/(2*cso[i][0]+12) for i in fo]

EG_birch = [2*(1+nu[i]) for i in dfs_]

EG   = [E0[i]/G0[i] for i in dfs_] # ratio of E to G at small uniaxial strain
EG_e = [np.sqrt(E0_e[j]**2/G0[j]**2 + G0_e[j]**2/G0[j]**4) for j in dfs_]

EGo   = [Eo[i]/Gpo[i][0] for i in fo]
EGo_e = [np.sqrt(Eo_e[j]**2/Gpo[j][0]**2) for j in fo]

#### Plotting ################################################################

# set default font and activate LaTeX rendering
mpl.rc('font', family = 'Courier New')
mpl.rc('text', usetex = True)
# living sponge colors, markers, and labels
spxclrs   = [cm.cool(x) for x in np.linspace(0, 1, 6)]
nospxclrs = [cm.spring(x) for x in np.linspace(0.6, 0.89, 2)]
clrs = [spxclrs[5],spxclrs[5],spxclrs[4],spxclrs[4],spxclrs[3],spxclrs[3],
        spxclrs[2],spxclrs[2],spxclrs[1],spxclrs[1],spxclrs[0],spxclrs[0],
        nospxclrs[0],nospxclrs[0],nospxclrs[1],nospxclrs[1]]
mrkrs = ['s','s','s','s','s','s','s','s','s','s','s','s','o','o','o','o']
spp   = ['A. polycapella','Callyspongia sp.','C. apion',
         'C. celata','T. keyensis','T. aurantium', 
         'A. fulva','I. notabilis']
spp   = [r'\textit{%s}' % spp[i] for i in fs_] # italicize
olbls  = ['brain', 'liver', 'adipose', 'synthetic']
oclrs  = [cm.Reds(x) for x in np.linspace(0.76, 1, 3)]
oclrs.insert(len(oclrs), 'xkcd:dried blood')
omrkrs = ['p', 'H', '8', '*']

# uniaxial stress vs. uniaxial strain figures and axes
# plts0 = [plt.subplots(nrows = 1, ncols = 2, figsize = (12, 10)) for i in fs_]
# figs0 = [plts0[i][0] for i in fs_]
# axes0 = [plts0[i][1] for i in fs_]
# # compression for second orientation
# [axes0[i][0].errorbar(eps[2*i+1][:7], sig[2*i+1][:7], xerr = eps_e[2*i+1][:7], 
#                       yerr = sig_e[2*i+1][:7], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = 'snow', mec = clrs[2*i], capsize = 6, 
#                       ecolor = 'dimgray', mew = 3) for i in fs_]
# [axes0[i][0].plot(eps[2*i+1][:6], Ec[2*i+1]*eps[2*i+1][:6]/100, c = clrs[2*i], 
#                   lw = 3, ls = '-') for i in fs_]
# [axes0[i][0].plot(eps[2*i+1][5:9], E0[2*i+1]*eps[2*i+1][5:9]/100, 
#                   c = clrs[2*i], lw = 3, ls = '--') for i in fs_]
# # compression for first orientation
# [axes0[i][0].errorbar(eps[2*i][:7], sig[2*i][:7], xerr = eps_e[2*i][:7], 
#                       yerr = sig_e[2*i][:7], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = clrs[2*i], mec = 'black', capsize = 6, mew = 3, 
#                       ecolor = 'dimgray') for i in fs_]
# [axes0[i][0].plot(eps[2*i][:6], Ec[2*i]*eps[2*i][:6]/100, c = clrs[2*i], 
#                   lw = 3, ls = ':') for i in fs_]
# [axes0[i][0].plot(eps[2*i][5:9], E0[2*i]*eps[2*i][5:9]/100, c = clrs[2*i], 
#                   lw = 3, ls = '-.') for i in fs_]
# # stretch for second orientation
# [axes0[i][1].errorbar(eps[2*i+1][7:], sig[2*i+1][7:], 
#                       xerr = eps_e[2*i+1][7:], yerr = sig_e[2*i+1][7:], 
#                       ms = 24, fmt = mrkrs[2*i], mfc = 'snow', mec = clrs[2*i], 
#                       capsize = 6, ecolor = 'dimgray', mew = 3) for i in fs_]
# [axes0[i][1].plot(eps[2*i+1][8:], Es[2*i+1]*eps[2*i+1][8:]/100, c = clrs[2*i], 
#                   lw = 3, ls = '-') for i in fs_]
# [axes0[i][1].plot(eps[2*i+1][5:9], E0[2*i+1]*eps[2*i+1][5:9]/100, 
#                   c = clrs[2*i], lw = 3, ls = '--') for i in fs_]
# # stretch for first orientation
# [axes0[i][1].errorbar(eps[2*i][7:], sig[2*i][7:], xerr = eps_e[2*i][7:], 
#                       yerr = sig_e[2*i][7:], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = clrs[2*i], mec = 'black', capsize = 6, mew = 3,
#                       ecolor = 'dimgray') for i in fs_]
# [axes0[i][1].plot(eps[2*i][8:], Es[2*i]*eps[2*i][8:]/100, c = clrs[2*i], 
#                   lw = 3, ls = ':') for i in fs_]
# [axes0[i][1].plot(eps[2*i][5:9], E0[2*i]*eps[2*i][5:9]/100, c = clrs[2*i], 
#                   lw = 3, ls = '-.') for i in fs_]
# # aesthetics
# [axes0[i][0].set_xlim(-13.1, 0) for i in fs_]
# [axes0[i][0].set_xticks([-10,-5,0]) for i in fs_]
# [axes0[i][1].set_xlim(0, 13.1) for i in fs_]
# [axes0[i][1].set_xticks([5,10]) for i in fs_]
# max0 = [np.max([np.abs(sig[2*i][:]),np.abs(sig[2*i+1][:])]) for i in fs_]
# [axes0[i][0].set_ylim(-max0[i]-0.26*max0[i], 0) for i in fs_]
# [axes0[i][1].set_ylim(0.001, max0[i]+0.26*max0[i]) for i in fs_]
# [axes0[i][1].set_xlabel('$\epsilon$ (\%)', fontsize = 72, labelpad = -10) 
#   for i in fs_]
# [axes0[i][1].set_ylabel('$\sigma_{zz}$ (Pa)', fontsize = 72) for i in fs_]
# [axes0[i][1].yaxis.set_ticks_position('right') for i in fs_]
# [axes0[i][j].yaxis.get_major_formatter().set_powerlimits((3, 3)) 
#   for i in fs_ for j in [0,1]]
# [axes0[i][j].yaxis.offsetText.set_fontsize(58) for i in fs_ for j in [0,1]]
# [axes0[i][1].yaxis.set_offset_position('right') for i in fs_]
# [axes0[i][j].tick_params(axis = 'both', which = 'major', labelsize = 58) 
#   for i in fs_ for j in [0,1]]
# [figs0[i].subplots_adjust(wspace = 0.001) for i in fs_]
# [figs0[i].savefig('plots/uniaxialstressvsstrain%s.svg' % i, 
#                   bbox_inches = 'tight', dpi = 1200) for i in fs_]

# shear modulus vs. uniaxial strain figures and axes
# plts1 = [plt.subplots(nrows = 1, ncols = 2, figsize = (12, 10)) for i in fs_]
# figs1 = [plts1[i][0] for i in fs_]
# axes1 = [plts1[i][1] for i in fs_]
# # compression for second orientation
# [axes1[i][0].errorbar(eps[2*i+1][:7], Gp[2*i+1][:7], xerr = eps_e[2*i+1][:7], 
#                       yerr = Gp_e[2*i+1][:7], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = 'snow', mec = clrs[2*i], capsize = 6, 
#                       ecolor = 'dimgray', mew = 3) for i in fs_]
# [axes1[i][0].plot(eps[2*i+1][:8], bc[2*i+1]+mc[2*i+1]*eps[2*i+1][:8]/100, 
#                   c = clrs[2*i], lw = 3, ls = '-') for i in fs_]
# # compression for first orientation
# [axes1[i][0].errorbar(eps[2*i][:7], Gp[2*i][:7], xerr = eps_e[2*i][:7], 
#                       yerr = Gp_e[2*i][:7], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = clrs[2*i], mec = 'black', capsize = 6, mew = 3,
#                       ecolor = 'dimgray') for i in fs_]
# [axes1[i][0].plot(eps[2*i][:8], bc[2*i]+mc[2*i]*eps[2*i][:8]/100, 
#                   c = clrs[2*i], lw = 3, ls = ':') for i in fs_]
# # stretch for second orientation
# [axes1[i][1].errorbar(eps[2*i+1][7:], Gp[2*i+1][7:], xerr = eps_e[2*i+1][7:], 
#                       yerr = Gp_e[2*i+1][7:], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = 'snow', mec = clrs[2*i], capsize = 6, 
#                       ecolor = 'dimgray', mew = 3) for i in fs_]
# [axes1[i][1].plot(eps[2*i+1][7:], be[2*i+1]+me[2*i+1]*eps[2*i+1][7:]/100, 
#                   c = clrs[2*i], lw = 3, ls = '-') for i in fs_]
# # stretch for first orientation
# [axes1[i][1].errorbar(eps[2*i][7:], Gp[2*i][7:], xerr = eps_e[2*i][7:], 
#                       yerr = Gp_e[2*i][7:], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = clrs[2*i], mec = 'black', capsize = 6, mew = 3,
#                       ecolor = 'dimgray') for i in fs_]
# [axes1[i][1].plot(eps[2*i][7:], be[2*i]+me[2*i]*eps[2*i][7:]/100, 
#                   c = clrs[2*i], lw = 3, ls = ':') for i in fs_]
# # aesthetics
# [axes1[i][0].set_xlim(-13.1, -0.131) for i in fs_]
# [axes1[i][1].set_xlim(0.131, 13.1) for i in fs_]
# min1 = [np.min([axes1[i][0].get_ylim(), axes1[i][1].get_ylim()]) for i in fs_]
# max1 = [np.max([axes1[i][0].get_ylim(), axes1[i][1].get_ylim()]) for i in fs_]
# [axes1[i][j].set_ylim(min1[i]-0.001*min1[i], max1[i]+0.001*max1[i]) 
#   for i in fs_ for j in [0,1]]
# [axes1[i][1].set_xlabel('$\epsilon$ (\%)', fontsize = 72) for i in fs_]
# [axes1[i][1].xaxis.set_label_coords(-0.1, -0.02) for i in fs_]
# [axes1[i][0].set_yticks([]) for i in fs_]
# [axes1[i][0].set_ylabel('$G^{\prime}$ (Pa)', fontsize = 72) for i in fs_]
# [axes1[i][j].tick_params(axis = 'both', which = 'major', labelsize = 58) 
#   for i in fs_ for j in [0,1]]
# [axes1[i][j].yaxis.get_major_formatter().set_powerlimits((6, 6)) 
#   for i in fs_ for j in [0,1]]
# [axes1[i][j].yaxis.offsetText.set_fontsize(58) for i in fs_ for j in [0,1]]
# [figs1[i].subplots_adjust(wspace = 0.27) for i in fs_]
# [figs1[i].savefig('plots/Gpvsstrain%s.svg' % i, bbox_inches = 'tight', 
#                   dpi = 1200) for i in fs_]

# E bar plot
# fig2, ax2 = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 18))
# [ax2[0].barh(fs_[i]+1.25*i+1, Ec[2*i], 1, xerr = Ec_e[2*i], color = clrs[2*i], 
#               lw = 3, capsize = 6, ecolor = 'dimgray') for i in fs_]
# [ax2[0].barh(fs_[i]+1.25*i, Ec[2*i+1], 1, xerr = Ec_e[2*i+1], color = 'snow', 
#               edgecolor = clrs[2*i], lw = 6, capsize = 6, ecolor = 'dimgray') 
#   for i in fs_]
# [ax2[1].barh(fs_[i]+1.25*i+1, Es[2*i], 1, xerr = Es_e[2*i], color = clrs[2*i], 
#               lw = 3, capsize = 6, ecolor = 'dimgray') for i in fs_]
# [ax2[1].barh(fs_[i]+1.25*i, Es[2*i+1], 1, xerr = Es_e[2*i+1], 
#               color = 'snow', edgecolor = clrs[2*i], lw = 6, 
#               capsize = 6, ecolor = 'dimgray') for i in fs_]
# [ax2[i].set_yticks([]) for i in [0,1]]
# [ax2[i].tick_params(axis = 'both', which = 'major', labelsize = 72) 
#   for i in [0,1]]
# [ax2[i].set_xlim(0.001, 1540000) for i in [0,1]]
# [ax2[i].set_ylim(-0.75, 17.5) for i in [0,1]]
# ax2[0].set_xlabel('$E_{c}$ (Pa)', fontsize = 72)
# ax2[1].set_xlabel('$E_{s}$ (Pa)', fontsize = 72)
# [ax2[i].xaxis.offsetText.set_fontsize(62) for i in [0,1]]
# ax2[0].invert_xaxis()
# [fig2.subplots_adjust(wspace = 0.04) for i in fs_]
# fig2.savefig('plots/Ebarplot.svg', bbox_inches = 'tight', dpi = 1200)

# EG ratio barplot
# fig3, ax3 = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 18))
# ax3.bar(0, EGo[2], 3, yerr = EGo_e[2], color = oclrs[2], capsize = 6, 
#         ecolor = 'dimgray')
# ax3.bar(5, EG[12], 3, yerr = EG_e[12], color = clrs[12], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(8, EG[13], 3, yerr = EG_e[13], color = 'snow', edgecolor = clrs[12], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(13, EG[0], 3, yerr = EG_e[0], color = clrs[0], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(16, EG[1], 3, yerr = EG_e[1], color = 'snow', edgecolor = clrs[0], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(21, EGo[0], 3, yerr = EGo_e[0], color = oclrs[0], capsize = 6, 
#         ecolor = 'dimgray')
# ax3.bar(26, EGo[1], 3, yerr = EGo_e[1], color = oclrs[1], capsize = 6, 
#         ecolor = 'dimgray')
# ax3.bar(31, EG[8], 3, yerr = EG_e[8], color = clrs[8], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(34, EG[9], 3, yerr = EG_e[9], color = 'snow', edgecolor = clrs[8], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(39, EG[14], 3, yerr = EG_e[14], color = clrs[14], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(42, EG[15], 3, yerr = EG_e[15], color = 'snow', edgecolor = clrs[14], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(47, EG[6], 3, yerr = EG_e[6], color = clrs[6], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(50, EG[7], 3, yerr = EG_e[7], color = 'snow', edgecolor = clrs[6], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(55, EG[4], 3, yerr = EG_e[4], color = clrs[4], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(58, EG[5], 3, yerr = EG_e[5], color = 'snow', edgecolor = clrs[5], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(63, EG[10], 3, yerr = EG_e[10], color = clrs[10], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(66, EG[11], 3, yerr = EG_e[11], color = 'snow', edgecolor = clrs[10], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(71, EG[2], 3, yerr = EG_e[2], color = clrs[2], capsize = 6, 
#         ecolor = 'dimgray', lw = 3)
# ax3.bar(74, EG[3], 3, yerr = EG_e[3], color = 'snow', edgecolor = clrs[2], 
#         capsize = 6, ecolor = 'dimgray', lw = 6)
# ax3.bar(79, EGo[3], 3, yerr = EGo_e[3], color = oclrs[3], capsize = 6, 
#         ecolor = 'dimgray')
# ax3.plot(np.arange(-3,100,2), 52*[1], ls = ':', lw = 6,  c = 'silver')
# ax3.plot(np.arange(-3,100,2), 52*[2], ls = '-.', lw = 6, c = 'silver')
# ax3.plot(np.arange(-3,100,2), 52*[3], ls = '--', lw = 6, c = 'silver')
# ax3.set_xticks([0, 6.5, 14.5, 21, 26, 32.5, 40.5, 48.5, 56.5, 64.5, 72.5, 79])
# ax3.set_xticklabels(['adipose', r'\textit{A. fulva}',
#                       r'\textit{A. polycapella}', 'brain', 'liver',
#                       r'\textit{T. keyensis}',r'\textit{I. notabilis}',
#                       r'\textit{C. celata}',r'\textit{C. apion}',
#                       r'\textit{T. aurantium}',r'\textit{Callyspongia sp.}',
#                       'synthetic'], fontdict = {'fontsize': 42}, rotation = 90)
# ax3.set_xlim(-3, 82)
# ax3.set_ylim(0.004, 4)
# ax3.set_ylabel('$E_{0}/G^{\prime}_{0}$', fontsize = 72, labelpad = 20)
# ax3.tick_params(axis = 'y', which = 'major', labelsize = 72)
# fig3.savefig('plots/EGbarplot.svg', bbox_inches = 'tight', dpi = 1200)

# compression stiffening
fig4, axs4 = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 18))
# sponges
[axs4.errorbar(sig[2*j+1][:7], Gp[2*j+1][:7], xerr = sig_e[2*j+1][:7], 
                yerr = Gp_e[2*j+1][:7], ms = 30, fmt = mrkrs[2*j], mfc = 'snow', 
                mec = clrs[2*j], capsize = 6, ecolor = 'dimgray', mew = 4) 
  for j in fs_]
[axs4.errorbar(sig[2*j][:7], Gp[2*j][:7], xerr = sig_e[2*j][:7], 
                yerr = Gp_e[2*j][:7], ms = 30, fmt = mrkrs[2*j], 
                mfc = clrs[2*j], mec = 'black', capsize = 6, label = spp[j],
                ecolor = 'dimgray', mew = 4) for j in fs_]
# axs4.plot(sig[7][lsc[7]:lfc[7]], 
#            -cs[7][0][0]*sig[7][lsc[7]:lfc[7]]+cs[7][0][1], 
#            c = clrs[6], lw = 3, ls = '-')#, label = '$m=%.0f$' % cs[7][0][0])
# axs4.plot(sig[6][lsc[6]:lfc[6]], 
#            -cs[6][0][0]*sig[6][lsc[6]:lfc[6]]+cs[6][0][1], 
#            c = clrs[6], lw = 3, ls = ':')#, label = '$m=%.0f$' % cs[6][0][0])
# others
[axs4.errorbar(sigo[j][:], Gpo[j][:], ms = 30, fmt = omrkrs[j], mfc = oclrs[j], 
               mec = 'black', mew = 4, label = olbls[j]) for j in fo]
axs4.set_xlabel('$\sigma_{zz}$ (Pa)', fontsize = 72, loc = 'left')
axs4.set_ylabel('$G^{\prime}$ (Pa)', fontsize = 72, loc = 'bottom')
axs4.tick_params(axis = 'both', which = 'major', labelsize = 72)
axs4.set_xlim([-200000, -100])
axs4.set_ylim([250, 12500000])
axs4.set_xscale('symlog')
axs4.set_yscale('log')
axs4.set_yticks([1000, 10000, 100000, 1000000, 10000000])
axs4.get_yaxis().set_major_formatter(mpl.ticker.LogFormatterSciNotation())
# axs4.set_ylim([1.25*10**2, 1.25*10**7])
# axins41 = axs4.inset_axes([1.125, 0.75, 0.4, 0.4])
# [axins41.errorbar(sig[2*j+1][:7], Gp[2*j+1][:7], xerr = sig_e[2*j+1][:7], 
#                   yerr = Gp_e[2*j+1][:7], ms = 30, fmt = mrkrs[2*j], 
#                   mfc = 'snow', mec = clrs[2*j], capsize = 6, 
#                   ecolor = 'dimgray', mew = 4) for j in fs_]
# [axins41.errorbar(sig[2*j][:7], Gp[2*j][:7], xerr = sig_e[2*j][:7], 
#                   yerr = Gp_e[2*j][:7], ms = 30, fmt = mrkrs[2*j], 
#                   mfc = clrs[2*j], mec = 'black', capsize = 6, 
#                   ecolor = 'dimgray', mew = 4) for j in fs_]
# axins41.plot(sig[7][lsc[7]:lfc[7]], 
#            -cs[7][0][0]*sig[7][lsc[7]:lfc[7]]+cs[7][0][1], 
#            c = clrs[6], lw = 3, ls = '-', label = '$m=%.1f$' 
#            % cs[7][0][0])
# axins41.plot(sig[6][lsc[6]:lfc[6]], 
#            -cs[6][0][0]*sig[6][lsc[6]:lfc[6]]+cs[6][0][1], 
#            c = clrs[6], lw = 3, ls = ':', label = '$m=%.0f$' 
#            % cs[6][0][0])
# mark_inset(axs4, axins41, loc1 = 2, loc2 = 3, fc = 'none', ec = '0')
# axins41.set_xlim(-75000, -75)
# axins41.set_ylim(400000, 3000000)
# axins41.xaxis.offsetText.set_fontsize(42)
# axins41.yaxis.offsetText.set_fontsize(42)
# axins41.xaxis.get_major_formatter().set_powerlimits((3, 3))
# axins41.yaxis.get_major_formatter().set_powerlimits((6, 6))
# axins41.tick_params(axis = 'both', which = 'both', labelsize = 52)

# axins42 = axs4.inset_axes([1.125, 0.25, 0.4, 0.4])
# [axins42.errorbar(sig[2*j+1][:7], Gp[2*j+1][:7], xerr = sig_e[2*j+1][:7], 
#                   yerr = Gp_e[2*j+1][:7], ms = 30, fmt = mrkrs[2*j], 
#                   mfc = 'snow', mec = clrs[2*j], capsize = 6, 
#                   ecolor = 'dimgray', mew = 4) for j in fs_]
# [axins42.errorbar(sig[2*j][:7], Gp[2*j][:7], xerr = sig_e[2*j][:7], 
#                   yerr = Gp_e[2*j][:7], ms = 30, fmt = mrkrs[2*j], 
#                   mfc = clrs[2*j], mec = 'black', capsize = 6, 
#                   ecolor = 'dimgray', mew = 4) for j in fs_]
# [axins42.errorbar(sigo[j][:], Gpo[j][:], ms = 30, fmt = omrkrs[j], 
#                   mfc = oclrs[j], mec = 'black', mew = 4, label = olbls[j])
#   for j in fo]
# mark_inset(axs4, axins42, loc1 = 2, loc2 = 3, fc = 'none', ec = '0')
# axins42.set_xlim(-15000, -15)
# axins42.set_ylim(43000, 860000)
# axins42.set_yticks([200000, 400000, 600000, 800000])
# axins42.xaxis.offsetText.set_fontsize(42)
# axins42.yaxis.offsetText.set_fontsize(42)
# axins42.xaxis.get_major_formatter().set_powerlimits((3, 3))
# axins42.yaxis.get_major_formatter().set_powerlimits((6, 6))
# axins42.tick_params(axis = 'both', which = 'both', labelsize = 52)

# axins43 = axs4.inset_axes([1.125, -0.25, 0.4, 0.4])
# [axins43.errorbar(sigo[j][:], Gpo[j][:], ms = 30, fmt = omrkrs[j], 
#                   mfc = oclrs[j], mec = 'black', mew = 4) for j in fo]
# [axins43.plot(sigo[2][:], -cso[i][0][0]*sigo[2][:]+cso[i][0][1], c = oclrs[i], 
#               lw = 3, ls = '--', label = '$m=%.2f$' % cso[i][0][0]) 
#  for i in fo[:-1]]
# mark_inset(axs4, axins43, loc1 = 2, loc2 = 3, fc = 'none', ec = '0')
# axins43.set_xlim(-2500, 0.025)
# axins43.set_xticks([-2000, -1000, 0])
# axins43.set_ylim(0.025, 2500)
# axins43.set_yticks([1000, 2000])
# axins43.xaxis.offsetText.set_fontsize(42)
# axins43.yaxis.offsetText.set_fontsize(42)
# axins43.xaxis.get_major_formatter().set_powerlimits((3, 3))
# axins43.yaxis.get_major_formatter().set_powerlimits((6, 6))
# axins43.tick_params(axis = 'both', which = 'both', labelsize = 52)
# hndls43, lbls43 = axins43.get_legend_handles_labels() # create legend
# axins43.legend(hndls43, lbls43, loc = 'upper right', fancybox = True, 
#                 shadow = True, fontsize = 38)
# legend
# hndls, lbls = axs4.get_legend_handles_labels() # create legend
# axs4.legend(hndls, lbls, loc = 'upper left', fancybox = True, shadow = True, 
#             fontsize = 42)
fig4.savefig('plots/compressstiff.svg', bbox_inches = 'tight', dpi = 1200)

# shear modulus vs. uniaxial stress figures and axes
# plts5 = [plt.subplots(nrows = 1, ncols = 2, figsize = (12, 10)) for i in fs_]
# figs5 = [plts5[i][0] for i in fs_]
# axes5 = [plts5[i][1] for i in fs_]
# # compression for second orientation
# [axes5[i][0].errorbar(sig[2*i+1][:7], Gp[2*i+1][:7], xerr = sig_e[2*i+1][:7], 
#                       yerr = Gp_e[2*i+1][:7], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = 'snow', mec = clrs[2*i], capsize = 6, 
#                       ecolor = 'dimgray', mew = 3) for i in fs_]
# # compression for first orientation
# [axes5[i][0].errorbar(sig[2*i][:7], Gp[2*i][:7], xerr = sig_e[2*i][:7], 
#                       yerr = Gp_e[2*i][:7], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = clrs[2*i], mec = 'black', capsize = 6, mew = 3,
#                       ecolor = 'dimgray') for i in fs_]
# # stretch for second orientation
# [axes5[i][1].errorbar(sig[2*i+1][7:], Gp[2*i+1][7:], xerr = sig_e[2*i+1][7:], 
#                       yerr = Gp_e[2*i+1][7:], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = 'snow', mec = clrs[2*i], capsize = 6, 
#                       ecolor = 'dimgray', mew = 3) for i in fs_]
# # stretch for first orientation
# [axes5[i][1].errorbar(sig[2*i][7:], Gp[2*i][7:], xerr = sig_e[2*i][7:], 
#                       yerr = Gp_e[2*i][7:], ms = 24, fmt = mrkrs[2*i], 
#                       mfc = clrs[2*i], mec = 'black', capsize = 6, mew = 3,
#                       ecolor = 'dimgray') for i in fs_]
# # aesthetics
# [axes5[i][j].yaxis.get_major_formatter().set_powerlimits((6, 6)) 
#   for i in fs_ for j in [0,1]]
# [axes5[i][j].yaxis.offsetText.set_fontsize(58) for i in fs_ for j in [0,1]]
# [axes5[i][1].set_xlabel('$\sigma_{zz}$ (Pa)', fontsize = 72) for i in fs_]
# [axes5[i][1].xaxis.set_label_coords(-0.1, -0.04) for i in fs_]
# [axes5[i][1].set_ylabel('$G^{\prime}$', fontsize = 72) for i in fs_]
# [axes5[i][1].yaxis.set_ticks_position('right') for i in fs_]
# [axes5[i][j].tick_params(axis = 'both', which = 'major', labelsize = 58) 
#   for i in fs_ for j in [0,1]]
# [figs5[i].subplots_adjust(wspace = 0.3) for i in fs_]
# [figs5[i].savefig('plots/Gpvsstress%s.svg' % i, bbox_inches = 'tight', 
#                   dpi = 1200) for i in fs_]