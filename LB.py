"""
@author: EAK
"""

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import os
import mainantonpaar

#### data read-in and preprocessing ##########################################

# change to directory where data are stored
dir='/Users/Emile/Documents/Graduate/AntonPaarData/Porifera/summary/shear data'
os.chdir(dir)

# first round LAOS
fn1    = '02082020_sponge12_samp3_strainsweep_8mm_3mm 0'
run1   = mainantonpaar.LAOS(fn1, r = 0.006)
t1     = run1.Timep
gam0_1 = run1.gam0
sig0_1 = run1.sig0
gamma1 = run1.Strain
sigma1 = run1.Stress
GpM1   = run1.Gpm
GpL1   = run1.Gpl
Gstar1 = run1.Gstarnorm
n1     = run1.n
# second round LAOS
fn2    = '02082020_sponge12_samp3_strainsweep_8mm_3mm 1'
run2   = mainantonpaar.LAOS(fn2, r = 0.006)
t2     = run2.Timep
sig0_2 = run2.sig0
gamma2 = run2.Strain
sigma2 = run2.Stress
GpM2   = run2.Gpm
GpL2   = run2.Gpl
Gstar2 = run2.Gstarnorm
n2     = run2.n
# number of amplitudes
Nf  = np.size(gam0_1)
# amplitude array for looping
Nf_ = np.arange(0, Nf, 1)
# strain amplitudes, change to percent
gam0 = [100*gam0_1[i] for i in Nf_]

#### plotting ################################################################

# color scheme
cwsubsec = np.linspace(0.7, 0.1, Nf)   # first round, blues and reds
clrscw   = [cm.pink(x) for x in cwsubsec]
grsubsec = np.linspace(0.2, 0.7, Nf) # second round, transparent grays
clrsgr   = [cm.gray(x) for x in grsubsec]
# strain amplitude labels
lbls = ['$\gamma_{0}=$ %.2f' % gam0[i] +'\%' for i in Nf_]

# time series
fig0, axs0 = plt.subplots(nrows = 1, ncols = 2, figsize = (22, 11))
[axs0[0].plot(t1[i], gamma1[i], linewidth = 3, color = clrscw[i]) 
 for i in Nf_[1::2]] # strain vs time, 1st run
[axs0[0].plot(t2[i], gamma2[i], linewidth = 3, color = clrsgr[i]) 
 for i in Nf_[1::2]] # strain vs time, 2nd run
[axs0[1].plot(t1[i], sigma1[i]/1000, linewidth = 3, color = clrscw[i]) # kPa
 for i in Nf_[1::2]] # stress vs time, 1st run
[axs0[1].plot(t2[i], sigma2[i]/1000, linewidth = 3, color = clrsgr[i]) # kPa
 for i in Nf_[1::2]] # stress vs time, 2nd run
axs0[0].set_ylabel(r'$\gamma$ (\%)', fontsize = 48, labelpad = -30)
axs0[1].set_ylabel(r'$\tau$ (kPa)', fontsize = 48, labelpad = -30)
maxx =  np.max(np.abs(axs0[0].get_xlim()))
maxy = [np.max(np.abs(axs0[i].get_ylim())) for i in [0,1]]
[axs0[i].set_ylim([-maxy[i]+0.06*maxy[i],maxy[i]-0.06*maxy[i]]) for i in [0,1]]
[axs0[j].set_xlabel(r'time (s)', fontsize = 48) for j in [0,1]]
[axs0[j].tick_params(axis = 'both', which = 'major', labelsize = 38) 
 for j in [0,1]]
# insets
axins = [inset_axes(axs0[i], width = '40%', height = '40%', loc = 1) 
         for i in [0,1]]
[axins[0].plot(t1[i], gamma1[i], lw = 3, c = clrscw[i]) for i in Nf_[1::2]]
[axins[0].plot(t2[i], gamma2[i], lw = 3, c = clrsgr[i]) for i in Nf_[1::2]]
[axins[i].tick_params(axis = 'both', labelsize = 28) for i in [0,1]]
[axins[i].set_xlim(0.5*maxx, maxx-0.1*maxx) for i in [0,1]]
[axins[i].set_ylim(-0.2*maxy[i], -0.001*maxy[i]) for i in [0,1]]
[axins[1].plot(t1[i], sigma1[i]/1000, lw = 3, c = clrscw[i]) 
 for i in Nf_[1::2]]
[axins[1].plot(t2[i], sigma2[i]/1000, lw = 3, c = clrsgr[i]) 
 for i in Nf_[1::2]]
[mark_inset(axs0[i], axins[i], loc1 = 4, loc2 = 3, fc = 'none', ec = '0')
 for i in [0,1]]
fig0.subplots_adjust(wspace = 0.2)
fig0.savefig('plots/'+fn1[:-2]+'_timeseries.svg', bbox_inches = 'tight', 
            dpi = 1200)

# Lissajous-Bowditch curves
fig1, ax1 = plt.subplots(figsize = (12,12))
x1 = np.asarray(gamma1, dtype = object)
y1 = np.asarray(sigma1, dtype = object)
x2 = np.asarray(gamma2, dtype = object)
y2 = np.asarray(sigma2, dtype = object)
# wrap-around so curves are totally closed
x1 = [np.append(x1[i][:], x1[i][0]) for i in Nf_]
y1 = [np.append(y1[i][:], y1[i][0]) for i in Nf_]
x2 = [np.append(x2[i][:], x2[i][0]) for i in Nf_]
y2 = [np.append(y2[i][:], y2[i][0]) for i in Nf_]
# plot
[ax1.plot(x2[i], y2[i]/1000, lw = 3, c = clrsgr[i]) for i in Nf_[1::2]]
[ax1.plot(x1[i], y1[i]/1000, lw = 3, c = clrscw[i], label = lbls[i])
 for i in Nf_[1::2]]
# lines illustrating minimum and maximum moduli
x  = np.array([-maxy[0]+0.06*maxy[0], 1.04*maxy[0]])
ym = GpM1[-1]*x/100
yl = GpL1[-1]*x/100
ax1.plot(x, ym/1000, linestyle = '--', lw = 3, c = 'black', marker = None)
ax1.plot(x, yl/1000, linestyle = '-.', lw = 3, c = 'black', marker = None)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 48)
ax1.set_xlabel(r'$\gamma$ (\%)',  fontsize = 58)
ax1.set_ylabel(r'$\tau$ (kPa)', fontsize = 58, labelpad = -30)
ax1.set_xlim([-1.008*maxy[0], 1.008*maxy[0]])
ax1.set_ylim([-1.008*maxy[1], 1.008*maxy[1]])
ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
hndls, lbls = ax1.get_legend_handles_labels()
fig1.legend(hndls, lbls, loc = 'center right', bbox_to_anchor = (1.32,0.32), 
            ncol = 1, fancybox = True, shadow = True, fontsize = 38)
# small strain inset
axins1 = inset_axes(ax1, width = '38%', height = '38%', loc = 4)
[axins1.plot(x2[i], y2[i]/1000, lw = 3, c = clrsgr[i]) for i in Nf_[1::2]]
[axins1.plot(x1[i], y1[i]/1000, lw = 3, c = clrscw[i]) for i in Nf_[1::2]]
axins1.set_xlim(-0.077*maxy[0], 0.077*maxy[0])
axins1.set_ylim(-0.077*maxy[1], 0.077*maxy[1])
axins1.xaxis.set_major_locator(ticker.MaxNLocator(3))
axins1.yaxis.set_major_locator(ticker.MaxNLocator(3))
axins1.xaxis.set_ticks_position('top') 
axins1.tick_params(axis = 'both', labelsize = 28)
mark_inset(ax1, axins1, loc1 = 2, loc2 = 1, fc = 'none', ec = '0')
# large strain
axins2 = inset_axes(ax1, width = '38%', height = '38%', loc = 2)
[axins2.plot(x2[i], y2[i]/1000, lw = 3, c = clrsgr[i]) for i in Nf_[1::2]]
[axins2.plot(x1[i], y1[i]/1000, lw = 3, c = clrscw[i]) for i in Nf_[1::2]]
axins2.set_xlim(0.21*maxy[0], maxy[0]-0.077*maxy[0])
axins2.set_ylim(0.21*maxy[1], maxy[1]-0.077*maxy[1])
axins2.xaxis.set_major_locator(ticker.MaxNLocator(2))
axins2.yaxis.set_major_locator(ticker.MaxNLocator(2))
axins2.yaxis.set_ticks_position('right') 
axins2.tick_params(axis = 'both', labelsize = 28)
mark_inset(ax1, axins2, loc1 = 4, loc2 = 1, fc = 'none', ec = '0')
# save
fig1.savefig('plots/'+fn1[:-2]+'_LB.svg', bbox_inches = 'tight', dpi = 1200)

# power spectra
fig2, ax2 = plt.subplots(figsize = (12,12))
[ax2.plot(n1[i], Gstar1[i], linewidth = 3, color = clrscw[i]) 
 for i in Nf_[1::2]]
[ax2.plot(n2[i], Gstar2[i], linewidth = 3, color = clrsgr[i]) 
 for i in Nf_[1::2]]
ax2.set_xlabel('$n$', fontsize = 42)
ax2.set_ylabel('$|G^{*}_{n}|$/$|G^{*}_{1}|$', fontsize = 42)
ax2.xaxis.set_major_locator(MaxNLocator(integer = True))     
ax2.tick_params(axis = 'both', which = 'major', labelsize = 48)
ax2.set_xlim([0, 12])
ax2.set_ylim([0.0002, 1])
ax2.set_yscale('log')
fig2.savefig('plots/'+fn1[:-2]+'_power.svg', bbox_inches = 'tight', dpi = 1200)