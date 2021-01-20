"""
@author: EAK
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

#==============================================================================
# These are a set of helper functions that get called from the 'main.py'
#==============================================================================

# set font for plots
mpl.rc('font', family = 'Courier New')
# activate latex text rendering
mpl.rc('text', usetex = True)

def calc_Gpm_Gpl(Gpn, hh, N_cyc):
    """
        Calculates the minimum and maximum elastic strain moduli from weighted 
        sums of Gpn and the highest harmonic excited to truncate the sum. 
    """
    
    # initialize the sums
    Gpm = 0
    Gpl = 0
    
    for k in np.arange(1, hh + 2, 2): # odds only
        # See definitions of below in literature
        Gpm = Gpm + k*Gpn[N_cyc*k]                   # tangent modulus
        Gpl = Gpl + Gpn[N_cyc*k]*(-1)**((k - 1) / 2) # secant modulus
        
    return Gpm, Gpl

def convert(r, theta, tau, f_n):
    """
        Convert the measurable variables displacements, torques, and forces, 
        to the rheological variables strains and stresses. Needed this on the
        Kinexus, don't on the AntonPaar.
    """
    
    if r == 0.004: # if the diameter is 8 mm of the Kinexus plate        
        # best to get these from rheometer software calibrations as there can
        # be adapters on rheometer hardware that change the moment of inertia
        # drastically and thus change the conversion factors from that of the 
        # theoretical values found in textbooks, ie Ferry 1980
        f_gam  = 0.71     # 1 / rad
        f_sigS = 7130274  # Pa / (N*m)
        f_sigN = 1 / (np.pi*r**2)
        
    if r == 0.0075: # if the diameter is 15 mm Bohlin plate w/ Kinexus adapter        
        f_gam  = 5.625
        f_sigS = 1131768
        f_sigN = 1 / (np.pi*r**2)
    
    # do the conversions
    gam  = f_gam * theta # shear strain, dimensionless
    sigS = f_sigS * tau  # shear stress, Pa
    sigN = f_n * f_sigN  # normal stress, Pa
        
    return gam, sigS, sigN

def cycle_averaging(timeseries, dt):
    """
        Averages cycles for signal processing. Needed this on the Kinexus, 
        AntonPaar seems to do some sort of inherent signal averaging.
    """

    timeseries1 = timeseries[-6400:]
    
    return timeseries1

def cycle_count(timeseries):
    """
        Counts how many times a signal crosses zero to return number of 
        cycles.
    """
    
    # ensure signal is centered around zero to count number of zero crossings
    timeseries = timeseries - np.mean(timeseries)
    
    # initialize a counter for the number of times the signal changes sign
    k = 0
    # get the sign of each element in the digitized signal
    sign = np.sign(timeseries)
    # initialize the list of indices where signal cross zero
    i_0 = []
    # for the whole time series signal
    for i in np.arange(0, np.size(timeseries, 0) - 1, 1):
        # if the sign changed
        if sign[i] != sign[i + 1]:
            # tick up the counter
            k = k + 1
            # and add at what index this happens at to i_0 list
            i_0.append(i + 1)
      
    # final number of sign changes
    N_sign = k
    # make i_0 into an object array, don't remember why just do it
    i_0 = np.array(i_0, dtype = np.int)

    # number of cycles
    if N_sign % 2 == 0:
        N_cycles = np.int(N_sign / 2)
    else:
        N_cycles = np.int((N_sign + 1) / 2)

    return N_cycles

def FT_exp_to_trig(f_m, dt):
    """
        Calculates the trigonometric Fourier coefficients from the exponential 
        coefficients calculated by Python's DFT and returns them. See Python
        notation from FFT documentation.
    """
    
    n = np.size(f_m)
    
    # do the FFT
    F_k = np.fft.fft(f_m)
    
    # the first term is the mean of the signal
    F0 = F_k[0]
    
    # scale the terms. See Python DFT definition.
    F_k = F_k / n
    
    # convert from exponential coefficients to trigonometric coefficients,
    # thank you Leonhard Euler
    An =  2*np.real(F_k)
    Bn = -2*np.imag(F_k) 
    
    # get the frequencies associated with the coefficients
    freq = np.fft.fftfreq(n, dt)
    
    return An, Bn, F0, freq

def FT_trig_to_G(gam, sig, dt, N_cycles):
    """
        Get the viscoelastic moduli G' and G" from the trig Fourier 
        coefficients of the shear stress output signal, and the strain and 
        stress amplitudes.
    """
    
    # trig coefficients, mean, and associated frequencies of the input signal
    gamAn_S, gamBn_S, gamA0, gamfreq = FT_exp_to_trig(gam, dt)
    
    # trig coefficients, mean, and associated frequencies of the output signal
    sigAn_S, sigBn_S, sigA0, sigfreq = FT_exp_to_trig(sig, dt)
    
    # unshift them from the inherent digital signal phase lag
    gamAn, gamBn, sigAn, sigBn = shift_strain(gamAn_S, gamBn_S, 
                                              sigAn_S, sigBn_S, N_cycles)
    
    # shear strain amplitude
    gam0 = np.sqrt(gamAn[N_cycles]**2 + gamBn[N_cycles]**2)
    
    # shear stress amplitude
    sig0 = np.sqrt(sigAn[N_cycles]**2 + sigBn[N_cycles]**2)
    
    # ensure that G' is positive, cause if not, you just dissed thermodynamics     
    if sigBn[N_cycles] > 0:
        Gpn = sigBn / gam0  # G' from sine terms
    else:
        Gpn = -sigBn / gam0
    
    # ensure that G" is positive, cause if not, you just dissed thermodynamics  
    if sigAn_S[N_cycles] > 0:
        Gppn = sigAn / gam0 # G" from cosine terms
    else: 
        Gppn = -sigAn / gam0

    return Gpn, Gppn, gam0, sig0

def load_antonpaar(fn):
    """
        Loads an Anton Paar rheometer data file.
    """
    
    # load the dataframe from the csv file
    df = pd.read_csv(fn)
    # get whole sheet
    df = df.values
    
    # number of final averaged data points
    N_f = pd.to_numeric(df[5,2])
    # array for looping over final data points
    N_f_ = np.arange(0, N_f, 1)
    
    # point number array
    f_pts  = pd.to_numeric(df[7:,1]) # the 7 is arbitrary and will depend on 
                                     # the size of the headers in the file
    
    # get the indices where the raw data for each final data point starts
    start = np.where(np.isfinite(f_pts))[0]
    start = start + 7 # takes care of arb headers that take up rows 0-7
    
    # raw variables
    time_p = pd.to_numeric(df[9:,2]) # time within an amplitude, s
    theta  = pd.to_numeric(df[9:,4]) # angular displacement, rad
    tau    = pd.to_numeric(df[9:,5]) # torque, N*m
    strain = pd.to_numeric(df[9:,15]) # waveform strain, %
    stress = pd.to_numeric(df[9:,16]) # waveform stress, Pa
    
    # software averaged values, smaller size than the raw variables
    N      = [pd.to_numeric(df[i,1]) for i in start] # point number
    time_a = [pd.to_numeric(df[i,3]) for i in start] # amplitude time, s
    f_N    = [pd.to_numeric(df[i,6]) for i in start] # normal force, N
    gap    = [pd.to_numeric(df[i,7]) for i in start] # gap, mm
    f      = [pd.to_numeric(df[i,8]) for i in start] # frequency, Hz
    gam = [pd.to_numeric(df[i,9]) for i in start]  # shear strain, %
    sig = [pd.to_numeric(df[i,10]) for i in start] # shear stress, Pa
    Gp  = [pd.to_numeric(df[i,11]) for i in start] # elastic modulus, Pa
    Gpp = [pd.to_numeric(df[i,12]) for i in start] # viscous modulus, Pa
    Jp  = [pd.to_numeric(df[i,13]) for i in start] # elastic compliance, 1/Pa
    Jpp = [pd.to_numeric(df[i,14]) for i in start] # viscous compliance, 1/Pa
    
    # split raw data that goes with each final data point
    # denote splitted arrays from unsplitted arrays w/ caps
    start = start - 7 # arbitrary thing for header
    start = np.append(start, np.size(time_p))
    Time_p = [time_p[start[i]:start[i+1]-2] for i in N_f_]
    Strain = [strain[start[i]:start[i+1]-2] for i in N_f_]
    Stress = [stress[start[i]:start[i+1]-2] for i in N_f_]

    return N, N_f_, time_p, theta, tau, strain, stress, time_a, f_N, gap, f, \
           gam, sig, Gp, Gpp, Jp, Jpp, Time_p, Strain, Stress

def load_kinexus(fn):
    """
        Loads a Kinexus rheometer data file.
    """
    
    # load the dataframe from the csv file
    df = pd.read_csv(fn)    
    df = df.values
    
    # raw data: time, angular displacement, torque, normal force, gap
    time  = df[:,0] # s
    theta = df[:,1] # rad
    tau   = df[:,2] # N*m
    f     = df[:,3] # N
    gap   = df[:,4] # mm
    
    # software columns, frequency, elastic modulus, viscous modulus, shear 
    # strain, shear stress, elastic compliance, viscous compliance                
    nu  = df[:,5]     # Hz
    Gp  = df[:,6]     # Pa
    Gpp = df[:,7]     # Pa
    gam = df[:,8]     # dimensionless
    sig = df[:,9]     # Pa
    Jp  = df[:,10]    # 1/Pa
    Jpp = df[:,11]    # 1/Pa
        
    # get where the calculated final data columns first populate with values
    start = np.where(np.isnan(nu))[0][-1] + 1

    return time, theta, tau, f, gap, nu, Gp, Gpp, gam, sig, Jp, Jpp, start

def plot_frequency_sweep(Gp, Gpp, w, fn):
    """
        Plots the viscoeastic moduli in the linear regime vs angular 
        frequency.
    """
    
    # change lists to arrays
    w   = np.asarray(w)
    Gp  = np.asarray(Gp)
    Gpp = np.asarray(Gpp)
    
    fig, ax = plt.subplots(figsize = (12,10))
    # plot elastic modulus vs angular frequency
    ax.scatter(w, Gp,  label = '$G\'$', s = 250, marker = 'o', 
               facecolors = 'black', edgecolors = 'black') 
    # plot viscous modulus vs angular frequency
    ax.scatter(w, Gpp, label = '$G^{\prime \prime}$', s = 250, marker = 'o',
               facecolors = 'white', edgecolors = 'black')
    ax.set_xlabel('$\omega$ (rad/s)', fontsize = 38)
    ax.set_ylabel('$G^{\prime}$, $G^{\prime \prime}$ (Pa)', fontsize = 38)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.xlim([0.1, 100])
    plt.ylim([10**4, 10**7])
    plt.legend(loc = 'lower right', fontsize = 26)
    plt.savefig('plots/' + fn + '_freqsweep.svg', dpi = 1200)    
    return

def plot_Lissajous_Bowditch(gamma, sigma, gam0, Gpm, Gpl, fn):
    """
        Creates Lissajous-Bowditch curves.
    """
    
    # number of amplitudes
    N_f  = np.size(gam0)
    # amplitude array
    N_f_ = np.arange(0, N_f, 1)
    
    # color scheme
    start, stop = 0, 1
    cm_subsec = np.linspace(start, stop, N_f)
    colors = [cm.coolwarm(x) for x in cm_subsec]
    
    gam0 = [100*gam0[i] for i in N_f_] # amplitudes
    lbls = ['$\gamma_{0}=$ %.2f' % gam0[i] +'\%' for i in N_f_] # amp labels
    fig, ax = plt.subplots(figsize = (14,14))  
    x = np.asarray(gamma)
    y = np.asarray(sigma)
    # wrap-around so LB curves are totally closed
    x = [np.append(x[i][:], x[i][0]) for i in N_f_]
    y = [np.append(y[i][:], y[i][0]) for i in N_f_]
    # plot time-resolved stress and strain
    [ax.plot(x[i], y[i], linewidth = 2.5, color = colors[i], label = lbls[i]) 
     for i in N_f_]
    # lines illustrating minimum and maximum moduli
    xm = np.array([-10.25, 10.25])
    ym = Gpm*xm/100
    xl = np.array([-10.25, 10.25])
    yl = Gpl*xl/100
    ax.plot(xm, ym, linestyle = 'dotted', linewidth = 2, color = 'black',
            marker = None)
    ax.plot(xl, yl, linestyle = 'dashed', linewidth = 2, color = 'black',
            marker = None)
    plt.text(xm[-1]+0.25, ym[-1]-2500, '$G^{\prime}_{M}$', fontsize = 42)
    plt.text(xl[-1]+0.25, yl[-1]-2500, '$G^{\prime}_{L}$', fontsize = 42)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 42)
    plt.legend(loc = 'upper left', fontsize = 28)
    
    # create zoomed inset axes
    axins = inset_axes(ax, width = '40%', height = '40%', loc = 4)
    [axins.plot(x[i], y[i], linewidth = 2.5, color = colors[i]) for i in N_f_]
    # specify zoomed region
    x1, x2, y1, y2 = -0.3, 0.3, -3000, 3000
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.xaxis.set_ticks_position('top') 
    axins.tick_params(axis = 'both', labelsize = 32)
    # inset x axes limits
    xtick_values = [-0.15, 0, 0.15]
    ytick_values = [-1500, 0, 1500]
    axins.set_xticks(xtick_values)
    axins.set_yticks(ytick_values)
    # zoom box effect
    mark_inset(ax, axins, loc1 = 2, loc2 = 1, fc = 'none', ec = '0.65')
    ax.set_xlabel(r'$\gamma$ (\%)',  fontsize = 52)
    ax.set_ylabel(r'$\sigma_{\theta z}$ (Pa)', fontsize = 52)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-140000, 140000])
    # scientific notation
    ax.yaxis.get_major_formatter().set_powerlimits((3, 3))
    ax.yaxis.offsetText.set_fontsize(42)
    plt.savefig('plots/' + fn + '_LB.svg', bbox_inches = 'tight', dpi = 1200)    
    return

def plot_nonlinear_sweep(Gpm, Gpl, gamma, fn):
    """
        Plots the nonlinear shear moduli vs shear strain amplitude.
    """
    
    # change to arrays
    gamma = np.asarray(gamma)
    Gpm   = np.asarray(Gpm)
    Gpl   = np.asarray(Gpl)
    
    fig, ax = plt.subplots(figsize = (12,10))
    # plot minimum strain modulus vs shear strain amplitude
    ax.errorbar(100*gamma, Gpm, label = '$G^{\prime}_{M}$', ms = 18, fmt = 's', 
                mfc = 'white', mec = 'black')
    # plot maximum strain modulus vs shear strain amplitude
    ax.errorbar(100*gamma, Gpl, label = '$G^{\prime}_{L}$', ms = 18, fmt = 's', 
                mfc = 'black', mec = 'black')
    ax.set_xlabel('$\gamma_{0}$ (\%)', fontsize = 62)
    ax.set_ylabel('$G^{\prime}_{M}$, $G^{\prime}_{L}$ (Pa)', fontsize = 62)    
    ax.set_xscale('log')
    ax.set_yscale('log')    
    plt.tick_params(axis = 'both', which = 'major', labelsize = 52)
    plt.xlim(0.04, 25)
    plt.ylim(10**5, 10**7)
    plt.legend(loc = 'best', fontsize = 38)
    plt.savefig('plots/' + fn + '_nonlinearsweep.svg', bbox_inches = 'tight', 
                dpi = 1200)
    return

def plot_power_spectrum(G_star, n, gam0, fn):
    """
        Plots the power spectrum of the input signal vs. harmonic number.
    """
    
    # number of amplitudes
    N_f = np.size(gam0)
    # amplitude array
    N_f_ = np.arange(0, N_f, 1)
    
    # create lines from colormap
    start, stop = 0, 1
    cm_subsec = np.linspace(start, stop, N_f)
    colors = [cm.coolwarm(x) for x in cm_subsec]
    
    fig, ax = plt.subplots(figsize = (12,10))
    # create amplitude labels
    gam0 = [np.round(100*gam0[i], 2) for i in N_f_]
    labels = ['$\gamma_{0}=$ %.2f' % gam0[i] +'\%' for i in N_f_]
    # plot complex modulus vs harmonic number
    [ax.plot(n[i], G_star[i], linewidth = 2.5, label = labels[i], color = \
             colors[i]) for i in N_f_]
    ax.set_xlabel('$n$', fontsize = 32)
    ax.set_ylabel('$|G^{*}_{n}|$/$|G^{*}_{1}|$', fontsize = 38)
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))     
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.xlim([0, 21])
    plt.ylim([0.0001, 1])
    ax.set_yscale('log') 
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    # plt.legend(loc = 'lower left', fontsize = 16)
    
    # create zoomed inset axes
    axins = inset_axes(ax, width = '70%', height = '50%', loc = 1)
    [axins.plot(n[i], G_star[i], linewidth = 2.5, color = colors[i]) for i in 
     N_f_]
    # specify zoomed region
    x1, x2, y1, y2 = 4.5, 11.5, 0.00012, 0.008
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yscale('log')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.savefig('plots/' + fn +'_power.svg', dpi = 1200) 
    return

def plot_strain_sweep(Gp, Gpp, gamma, fn):
    
    """
        Plots the first order viscoelastic moduli vs shear strain amplitude.
    """
    
    # change to arrays
    gamma = np.asarray(gamma)
    Gp    = np.asarray(Gp)
    Gpp   = np.asarray(Gpp)
    
    # check if gamma in percent
    if np.max(gamma) > 2:
        gamma = gamma
    else:
        gamma = 100*gamma # %
        
    fig, ax = plt.subplots(figsize = (12,10))
    # plot elastic modulus vs shear strain amplitude
    ax.scatter(gamma, Gp,  label = '$G\'$', s = 250, marker = 'o', 
               color = 'white', edgecolors = 'black')
    # plot viscous modulus vs shear strain amplitude
    ax.scatter(gamma, Gpp, label = '$G^{\prime\prime}$', s = 250, marker = '^',
               color = 'white', edgecolors = 'black')
    ax.set_xlabel('$\gamma_{0}$ (\%)', fontsize = 38)
    ax.set_ylabel('$G^{\prime}$, $G^{\prime \prime}$ (Pa)', fontsize = 38)    
    ax.set_xscale('log')
    ax.set_yscale('log')    
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.xlim(0.01, 100)
    plt.ylim(10000, 10000000)
    plt.legend(loc = 'best', fontsize = 26)
    plt.savefig('plots/' + fn + '_strainsweep.svg', dpi = 1200)  
    return

def plot_time_series(t, gam, sig, fn):
    """
        Plots input shear signal and output shear signal as a function of time.
    """
    
    fig = plt.figure(figsize = (12,12))
    
    ax1 = fig.add_subplot(211)
    ax1.plot(t, gam, linewidth = 2.5, color = 'black')
    ax1.set_ylabel(r'$\gamma$ (\%)', fontsize = 38)
    ax1.set_xticklabels([])
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 30) 
    
    ax2 = fig.add_subplot(212)
    ax2.plot(t, sig/1000, linewidth = 2.5, color = 'black')
    ax2.set_xlabel(r'time (s)', fontsize = 38)
    ax2.set_ylabel(r'$\sigma_{xz}$ (kPa)', fontsize = 38)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 30)
    
    return

def plot_time_series_LAOS(t, gam, sig, fn):
    """
        Plots input shear signal and output shear signal as a function of time
        for all amplitudes.
    """
    
    # number of amplitudes
    N_f = np.size(gam)
    # amplitude array
    N_f_ = np.arange(0, N_f, 1)
    
    # create lines from colormap
    start, stop = 0, 1
    cm_subsec = np.linspace(start, stop, N_f)
    colors = [cm.viridis(x) for x in cm_subsec]
    
    fig = plt.figure(figsize = (12,12))
    
    ax1 = fig.add_subplot(211)
    [ax1.plot(t[i], gam[i], linewidth = 2.5, color = colors[i]) for i in N_f_]
    ax1.set_ylabel(r'$\gamma$ (\%)', fontsize = 38)
    ax1.set_xticklabels([])
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 30) 
    
    ax2 = fig.add_subplot(212)
    [ax2.plot(t[i], sig[i]/1000, linewidth = 2.5, color = colors[i]) for i in 
     N_f_]
    ax2.set_ylabel(r'$\sigma_{xz}$ (kPa)', fontsize = 38)
    ax2.set_xlabel(r'time (s)', fontsize = 38)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 30)

    plt.savefig('plots/' + fn + '_timeseries.svg', dpi = 1200)
    return

def shift_strain(gamAn_S, gamBn_S, sigAn_S, sigBn_S, N_cycles):
    """
        Shifts the signal so that there is no inherent small phase lag. In 
        other words, trim the signal so that it starts as a positive or
        negative sinusoid. 
    """
    
    delta = np.arctan(gamAn_S[N_cycles] / gamBn_S[N_cycles])
    
    n = np.size(sigAn_S)
    
    sigAn = np.zeros((n))
    sigBn = np.zeros((n))
    
    for i in np.arange(0, n, 1):
        sigAn[i] = sigAn_S[i]*np.cos(i*delta/N_cycles) - \
            sigBn_S[i]*np.sin(i*delta/N_cycles)
                
        sigBn[i] = sigBn_S[i]*np.cos(i*delta/N_cycles) + \
            sigAn_S[i]*np.sin(i*delta/N_cycles)
                
    gamAn = np.zeros((n))
    gamBn = np.zeros((n))
    
    for i in np.arange(0, n, 1):
        gamAn[i] = gamAn_S[i]*np.cos(i*delta/N_cycles) - \
            gamBn_S[i]*np.sin(i*delta/N_cycles)
                
        gamBn[i] = gamBn_S[i]*np.cos(i*delta/N_cycles) + \
            gamAn_S[i]*np.sin(i*delta/N_cycles)
    
    return gamAn, gamBn, sigAn, sigBn