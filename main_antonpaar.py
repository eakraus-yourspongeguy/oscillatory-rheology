"""
@author: EAK
"""

# import Python modules and packages
import os
import numpy as np

# import subroutines
import rheofuncs

#==============================================================================
# data classes: various deformation protocols
# LAOS (Large Amplitude Oscillatory Shear)->amplitude sweeps (stress or strain)
# SAOS (Small Amplitude Oscillatory Shear)->frequency sweeps
#==============================================================================

class LAOS(object):
    """
        Describes a data file class of shear strain or stress oscillations at
        some single frequency sweeping the amplitude of strain or stress.
    """
    def __init__(self, fn, r):
        """
            Constructor of said data class with variables filename and radius.
        """
        # radius
        self.r = r # meters, input below

        # position control mode
        if 'strain' in fn:
            self.ctrl = 'strain' # strain-controlled input
        if 'stress' in fn:
            self.ctrl = 'stress' # stress-controlled input

        # load the data in the file
        [self.N, self.N_f_, self.time_p, self.theta, self.tau, self.strain,
         self.stress, self.time_a, self.f_N, self.gap, self.f, self.gam_soft,
         self.sig_soft, self.Gp_soft, self.Gpp_soft, self.Jp_soft,
         self.Jpp_soft, self.Time_p, self.Strain, self.Stress] = \
         rheofuncs.load_antonpaar(fn + '.csv')
            
        # calculate angular frequency rad/s, constant
        self.omega = 2*np.pi*self.f[0]
        
        # plot software reported values
        # rheofuncs.plot_strain_sweep(self.Gp_soft, self.Gpp_soft, self.gam_soft, 
        #                             fn + '_soft')
        
        # plot separately all amplitudes of stress and strain vs time
        # rheofuncs.plot_time_series_LAOS(self.Time_p, self.Strain, self.Stress, 
        #                                 fn)
        
        # count number of cycles
        N_cyc = [rheofuncs.cycle_count(self.Strain[i]) for i in self.N_f_]
        # dwell time, or sampling time, need this for FFT, equal to total time
        # of test divided by number of time points
        dt = [np.max(self.Time_p[i])/np.size(self.Time_p[i]) for i in 
              self.N_f_]
        
        # do it to 'em (do the actual FFT)
        self.Gpn, self.Gppn, self.gam0, self.sig0 = \
        map(list,zip(*[rheofuncs.FT_trig_to_G(self.Strain[i]/100,
                                              self.Stress[i], dt[i], N_cyc[i])
                       for i in self.N_f_]))
        # complex modulus       
        self.G_star = [np.sqrt(self.Gpn[i]**2 + self.Gppn[i]**2) for i in 
                       self.N_f_]
        # normalize it
        self.G_star_norm = [self.G_star[i][:] / self.G_star[i][N_cyc[i]] for i 
                            in self.N_f_]
        
        self.dn = [1/N_cyc[i] for i in self.N_f_]
        self.n  = [np.arange(0,self.dn[i]*np.size(self.G_star[i]),self.dn[i]) 
                   for i in self.N_f_]
        rheofuncs.plot_power_spectrum(self.G_star_norm, self.n, self.gam0, fn)
        
        # linear viscoelastic moduli
        self.Gp1  = [self.Gpn[i][N_cyc[i]] for i in self.N_f_]
        self.Gpp1 = [self.Gppn[i][N_cyc[i]] for i in self.N_f_]
        rheofuncs.plot_strain_sweep(self.Gp1, self.Gpp1, self.gam0, fn)
            
        # highest harmonics to use in summations for minimum and maximum 
        # strain moduli; these are a manual input! be careful, they change and
        # must be decided after inspecting the power spectra
        # see calc_Gpm_Gpl in rheofuncs for more info
        # Axinella
        hh = np.array([1, 1, 3, 5, 5, 5, 5, 7, 9, 9, 9, 9])
        # Cliona
        # hh = np.array([5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7])
        self.Gpm, self.Gpl = \
        map(list,zip(*[rheofuncs.calc_Gpm_Gpl(self.Gpn[i], hh[i], N_cyc[i]) 
                       for i in self.N_f_]))
        # plot 'em up
        rheofuncs.plot_nonlinear_sweep(self.Gpm, self.Gpl, self.gam0, fn)
        
        # Lissajous-Bowditch plots
        rheofuncs.plot_Lissajous_Bowditch(self.Strain, self.Stress, self.gam0, 
                                          self.Gpm[-1], self.Gpl[-1], fn)
            
class SAOS(object):
    """
        Describes a data file class of shear strain or stress oscillations at
        some single strain amplitude sweeping the frequency.
    """
    def __init__(self, fn, r):
        """
            Constructor of a data class with variables filename, radius.
        """
        # radius
        self.r = r # m
        
        # load the data in the file
        [self.N, self.N_f_, self.time_p, self.theta, self.tau, self.strain,
         self.stress, self.time_a, self.f_N, self.gap, self.f, self.gam_soft,
         self.sig_soft, self.Gp_soft, self.Gpp_soft, self.Jp_soft, 
         self.Jpp_soft, self.Time_p, self.Strain, self.Stress] = \
         rheofuncs.load_antonpaar(fn + '.csv')
        
        # angular frequency rad/s, sweeped for SAOS
        self.omega = [2*np.pi*self.f[i] for i in self.N_f_]
        
        # frequency sweep plot
        rheofuncs.plot_frequency_sweep(self.Gp_soft, self.Gpp_soft, self.omega, 
                                       fn)
        
#==============================================================================
# main
#==============================================================================

# change to directory where data are stored
dir = '/Users/Emile/Documents/Graduate/AntonPaarData/Porifera/Axinella sp2/'+\
      'shear/'
os.chdir(dir)

fn = '03232020_sponge3_samp4_strainsweep_2Hz 1'

# make a SAOS data object
# freq = SAOS(fn, r = 0.006)
# Gp  = freq.Gp_soft
# Gpp = freq.Gpp_soft

# make a LAOS data object
shear = LAOS(fn, r = 0.006)
gam   = shear.gam0
sig   = shear.sig0
Gp    = shear.Gp1
Gpp   = shear.Gpp1
Gpm   = shear.Gpm
Gpl   = shear.Gpl
f_n   = shear.f_N

# =============================================================================
# Goodbye, World.
# =============================================================================