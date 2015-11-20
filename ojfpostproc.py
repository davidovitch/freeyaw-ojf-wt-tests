# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:00:37 2012

@author: dave
"""

# STANDARD LIBRARY
from __future__ import division
import os
#import timeit

# COMMON 3TH PARTY
import numpy as np
import scipy.io
import scipy.integrate as integrate
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas
#import matplotlib.font_manager as mpl_font
#from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
#from matplotlib import tight_layout as tight_layout
import pylab as plt

# NOT SO COMMON 3TH PARTY
#from scikits.audiolab import play
import wafo

# CUSTOM
import plotting
import HawcPy
#import cython_func
#import ojfdesign
import bladeprop
import materials
import Simulations as sim
import misc
import ojfresult
import towercal
import bladecal
import yawcal
from ojfdb import ojf_db


RESDATA_CAL_02 = 'data/raw/02/calibration/'
RESDATA_CAL_04 = 'data/raw/04/calibration/'


## make list containing all the files in the folder
#files = []
#for f in os.walk(power_res):
#    files.append(f)
#
## load each file and merge into one airfoil_db dict
#for f in files[0][2]:
#    if f.startswith('bemv') and f.endswith('.dat'):
#        bem = np.loadtxt(power_res + f)
#        plotaoa(bem, f, power_res +'aoa/')
#        plotcpct(bem, f, power_res+'cpct/')
#        plot_dM(bem, f, power_res+'load_dM/')
#        plot_dF(bem, f, power_res+'load_dF/')

def _resistance_serie(ohms):
    """
    """

#    eq_ohm_grid = np.ndarray()
    x_range = np.arange(1,11)
    R_range = np.arange(0.05, 5, 0.05)
    # x: number of resistances in series
    # R: resistance value of one unit
    for x in x_range:
        for R in R_range:
            pass
    return ohms.prod() / ohms.sum()


def example_simple():
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-10/'
    resfile = '0210_run_005_freeyaw_init_0_stiffblades_pwm1000.mat'

    ojfmatfile = scipy.io.loadmat(respath+resfile)

    # ojfmatfile file structure: dictionary with keys
    # ['__version__', '__header__', 'run_005_freeyaw_init_0_sti','__globals__']
    # __globals__ seems to be empty

    # the actual data is behind the file name
    L1 = ojfmatfile[ojfmatfile.keys()[2]]
    # array([[ ([[(array([], dtype='<U1'), array([[4]], dtype=int32),
    L2 = L1[0,0]
    # >>> L2.dtype
    # dtype([('X', '|O8'), ('Y', '|O8'), ('Description', '|O8'),
    #       ('RTProgram', '|O8'), ('Capture', '|O8')])
    # --------------
    # >>> L2['X'].dtype
    # dtype([('Name', '|O8'), ('Type', '|O8'),('Data', '|O8'),('Unit', '|O8')])
    time = L2['X']['Data'][0,0]
    # --------------
    # >>> L2['Y'].dtype
    # dtype([('Name', '|O8'), ('Type', '|O8'),('Data', '|O8'),('Unit', '|O8'),
    #       ('XIndex', '|O8')])
    channel = 0
    data = L2['Y']['Data'][0,channel]
    label = L2['Y']['Name'][0,channel]
    # --------------
    print time.shape, data.shape, label.shape

def psd_example():
    """
    Example showing the effect of NFFT and Fs
    """
    dt1 = 0.001
    dt2 = 0.004
    freq = 2
#    omega = 10
    # omega = 2*pi*f
    # omega: angular frequency (rad/s)
    # f: frequency (Hz)
#    freq = omega / (2*np.pi)
    omega = 2*np.pi*freq
    time1 = np.arange(0,50,dt1)
    time2 = np.arange(0,50,dt2)
    data1 = np.sin(omega*time1) + (0.1*np.cos(omega*10*time1))
    data2 = np.sin(omega*time2) + (0.1*np.cos(omega*10*time2))

    plt.figure()
    plt.plot(time1, data1)
    plt.plot(time2, data2)

    plt.figure()
    # if you have 4* more points in data1, you will need to improve the
    # accuracy and increase NFFT. Or is that the padding they refer to??

    # NFFT: integer
    # The number of data points used in each block for the FFT.
    # Must be even; a power 2 is most efficient. The default value is 256.
    # This should NOT be used to get zero padding, or the scaling of the
    # result will be incorrect. Use pad_to for this instead.
    Pxx1, freqs1 = plt.psd(data1, NFFT=4096, Fs=1/dt1, label='data1')
    Pxx2, freqs2 = plt.psd(data2, NFFT=1024, Fs=1/dt2, label='data2')
    plt.xlim(0, 50)
    plt.legend()


###############################################################################
### CALIBRATION
###############################################################################

def compare_eigenfrq_ranges():
    """
    See what happens when comparing a selected and full range
    """
    # FOR THE BLADES
    respath = '/home/dave/PhD_data/OJF_data_edit/02/vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfrequencies/'
    # there was only one vibriation test in Februari
    # blade was just held tight at the root and then a deflection was
    # imposed and suddenly released
    resfile = '0213_run_099_straincalibration_blade12_tipdeflectionvibrations'
    channels = [0,1,2,3]
    blade = ojfresult.BladeStrainFile(respath+resfile)
    blade.plot_channel(figpath=figpath, channel=channels)

    # -----------------------------------------------------------------------
    # compare results from selected range and full range
    # selected range
    pa4 = plotting.A4Tuned()
    pa4.setup(figpath+resfile+'_COMPARE', nr_plots=4, grandtitle=resfile)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
    # make a numpy slice object
    ss = np.r_[9200:11000]
    ax1.plot(blade.time[ss], blade.data[ss,0])
    Pxx, freqs = ax2.psd(blade.data[ss,0], NFFT=512, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax2.psd(blade.data[ss,0], NFFT=1024, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax2.psd(blade.data[ss,0], NFFT=2048, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax2.psd(blade.data[ss,0], NFFT=4096, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    ax2.set_xlim([0,20])
    # full range
    ss = np.r_[1:len(blade.time)]
    ax3 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 3)
    ax4 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 4)
    ax3.plot(blade.time[ss], blade.data[ss,0])
    #Pxx, freqs = ax4.psd(blade.data[ss,0], NFFT=512, Fs=blade.sample_rate)
    #print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    #Pxx, freqs = ax4.psd(blade.data[ss,0], NFFT=1024, Fs=blade.sample_rate)
    #print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax4.psd(blade.data[ss,0], NFFT=2048, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax4.psd(blade.data[ss,0], NFFT=4096, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax4.psd(blade.data[ss,0], NFFT=8192, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    Pxx, freqs = ax4.psd(blade.data[ss,0], NFFT=16384, Fs=blade.sample_rate)
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    ax4.set_xlim([0,20])
    # ignore everything below 2Hz, give the highest peak
    print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]
    pa4.save_fig()
    # -----------------------------------------------------------------------

def eigenfreq_april():
    """
    Frequency runs done in April for tower and blade
    """

    nnfts = [8192,4096,2048]
    nnfts = [2048]

    respath = '/home/dave/PhD_data/OJF_data_edit/04/vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfrequencies/'

    # TOWER VIBRATIONS
    resfile = '0405_run_248_towercal_destroy_fish_wire'
    ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
    channels = ['Tower Strain Side-Side filtered',
                'Tower Top acc X (SS)',
                'Tower Strain For-Aft filtered',
                'Tower Top acc Y (FA)']
    ds.psd_plots(figpath, channels, nnfts=nnfts)

    resfile = '0405_run_253_towervibrations'
    ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
    eigenfreqs = ds.psd_plots(figpath, channels, nnfts=nnfts, saveresults=True)

    # BLADE VIBRATIONS
    #resfiles = ['257or258',
                #'257or258-2',
                #'0405_run_255or254a_bladecal_virbations_blade2',
                #'0405_run_255or254_virbations_bladecal_blade2',
                #'0405_run_257_bladecal_virbations_blad1']
    #for resfile in resfiles:
        #blade = ojfresult.BladeStrainFile(respath+resfile)
        #eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts)

    resfile = '257or258'
    cn = ['flex B2 root','fles B2 30%',
          'flex B1 root (excited)','flex B1 30% (excited)']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                                 channel_names=cn, saveresults=True)

    resfile = '257or258-2'
    cn = ['flex B2 root (excited)','fles B2 30% (excited)',
          'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                                 channel_names=cn, saveresults=True)

    resfile = '0405_run_255or254a_bladecal_virbations_blade2'
    cn = ['flex B2 root (excited)','fles B2 30% (excited)',
          'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                                 channel_names=cn, saveresults=True)

    resfile = '0405_run_255or254_virbations_bladecal_blade2'
    cn = ['flex B2 root','fles B2 30%',
          'flex B1 root (excited)','flex B1 30% (excited)']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                                 channel_names=cn, saveresults=True)

    resfile = '0405_run_257_bladecal_virbations_blad1'
    cn = ['flex B2 root','fles B2 30%',
          'flex B1 root (excited)','flex B1 30% (excited)']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                                 channel_names=cn, saveresults=True)


def eigenfreq_februari():
    """
    Plot all eigenfrequency test runs and determine the frequencies.

    You do not need to slice-out the vibration area (or select only datapoints
    for which the vibration is clearly present). Results are the same when
    taking the whole data range.
    """

    nnfts = [8192,4096,2048]
    nnfts = [2048]

    # FOR THE BLADES
    respath = '/home/dave/PhD_data/OJF_data_edit/02/vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfrequencies/'
    # there was only one vibriation test in Februari
    # blade was just held tight at the root and then a deflection was
    # imposed and suddenly released
    resfile = '0213_run_099_straincalibration_blade12_tipdeflectionvibrations'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)


    #Pxx, freqs = mpl.mlab.psd(blade.data[:,0], NFFT=8192, Fs=blade.sample_rate)

    # maximum frequency
    #print freqs[Pxx[[freqs.__ge__(2.5)]].argmax()]

def eigenfreq_august():
    """
    Plot all eigenfrequency test runs and determine the frequencies.

    You do not need to slice-out the vibration area (or select only datapoints
    for which the vibration is clearly present). Results are the same when
    taking the whole data range.
    """

    nnfts = [8192,4096,2048]
    nnfts = [2048]

    # FOR THE BLADES
    respath = '/home/dave/PhD_data/OJF_data_edit/08_vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfrequencies/'

    resfile = '500_trigger01_413314024'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '501_b1_stiff_trigger02_2664171792'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '502_b1_stiff_trigger03_504929400'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '503_b2_stiff_trigger01_507476419'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '504_b2_stiff_trigger02_2430537494'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '505_b2_stiff_trigger03_165311127'
    #channels = [0,1,2,3]
    channel_names = ['stiff B2 root','stiff B2 30%',
                     'stiff B1 root','stiff B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '506_b1_flex_trigger01_2990608704'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '507_b1_flex_trigger02_1468965218'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '508_b1_flex_trigger03_3392483542'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)


    resfile = '509_b1_flex_trigger01_2199793801'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '510_b1_flex_trigger02_2059868129'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '511_b1_flex_trigger03_3824274622'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '512_b2_flex_trigger01_3962726656'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '513_b2_flex_trigger02_1803468531'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '514_b2_flex_trigger03_2354543381'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '515_b2_flex_trigger01_2423128572'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '516_b2_flex_trigger02_157778266'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

    resfile = '517_b2_flex_trigger03_2028120981'
    #channels = [0,1,2,3]
    channel_names = ['flex B2 root','flex B2 30%',
                     'flex B1 root','flex B1 30%']
    blade = ojfresult.BladeStrainFile(respath+resfile)
    eigenfreqs = blade.psd_plots(figpath, [0,1,2,3], nnfts=nnfts,
                    channel_names=channel_names, fn_max=250, saveresults=True)

def eigenfreq_tower():
    """
    Manually obtained from 0405_run_253_towervibrations
    """

    path='/home/dave/PhD/Projects/PostProcessing/'
    path += 'OJF_tests/eigenfreq_damp_tower/'

    # is 24.41 the tower or something else? Remember the vibration measurements
    # where done on the complete wind turbine infrastructure

    fn = np.array([4.39, 24.41, 30.76])
    fname = 'eigenfreq_tower_ss'
    np.savetxt(path+fname, fn)

    fn = np.array([3.91, 24.41, 34.67])
    fname = 'eigenfreq_tower_fa'
    np.savetxt(path+fname, fn)


def eigenfreq_blade():
    """
    Because the blade eigenfrequency analysis is not very clear to handle,
    create manually the result files
    """

    path='/home/dave/PhD/Projects/PostProcessing/'
    path += 'OJF_tests/eigenfreq_damp_blade/'

    # blade 1 stiff
    struct = 'stiff'
    bladenr = 'B1'
    fn = np.array([17.0, 74.0])
    fname = 'eigenfreq_%s_%s' % (struct, bladenr)
    np.savetxt(path+fname, fn)

    bladenr = 'B2'
    fn = np.array([17.0, 72.0])
    fname = 'eigenfreq_%s_%s' % (struct, bladenr)
    np.savetxt(path+fname, fn)

    # blade 3 stiff
    struct = 'stiff'
    bladenr = 'B3'
    fn = np.array([17.0, 73.0])
    fname = 'eigenfreq_%s_%s' % (struct, bladenr)
    np.savetxt(path+fname, fn)

    # blade 1 flex
    struct = 'flex'
    bladenr = 'B1'
    fn = np.array([12.0, 54.0])
    fname = 'eigenfreq_%s_%s' % (struct, bladenr)
    np.savetxt(path+fname, fn)

    # blade 2 flex
    struct = 'flex'
    bladenr = 'B2'
    fn = np.array([11.0, 54.0])
    fname = 'eigenfreq_%s_%s' % (struct, bladenr)
    np.savetxt(path+fname, fn)

    # blade 3 flex
    struct = 'flex'
    bladenr = 'B3'
    fn = np.array([12.0, 54.0])
    fname = 'eigenfreq_%s_%s' % (struct, bladenr)
    np.savetxt(path+fname, fn)


def february_calibration():
    """
    Create all the calibration transformation polynomials for the February
    session
    """

    # -------------------------------------------------------------
    # Blade Strain Calibration
    # -------------------------------------------------------------
    bladecal.feb_bladestrain_calibration()

    # -------------------------------------------------------------
    # Tower Calibration
    # -------------------------------------------------------------
    # bad calibration results mean no tower calibration for February

    # -------------------------------------------------------------
    # Yaw calibration
    # -------------------------------------------------------------
    yawcal.feb_yawlaser_calibration()


def april_calibration():
    """
    All the calibrations for the April session
    """

    # -------------------------------------------------------------
    # Yaw calibration
    # -------------------------------------------------------------
    yawcal.apr_yawlaser_calibration()

    # -------------------------------------------------------------
    # Tower calibration
    # -------------------------------------------------------------
    towercal.all_tower_calibrations()

    # -------------------------------------------------------------
    # Blade calibration
    # -------------------------------------------------------------
    bladecal.apr_bladestrain_calibration()


def blade_extension_drag():
    """
    Blade extension calibration tests executed at Risoe
    """

    #bc = BladeCalibration()
    #bc.extension()
    #bc.drag()
    # get all the data points
    #bc.extension2()


    # load all the bc.extension2 generated data
    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'BladeStrainCalExt/'

    #stiff_B2_ch1 = np.loadtxt(figpath+'stiff_B2_ch1_allpoints')
    #stiff_B2_ch2 = np.loadtxt(figpath+'stiff_B2_ch2_allpoints')
    #flex_B2_ch3 = np.loadtxt(figpath+'flex_B2_ch3_allpoints')
    #flex_B2_ch4 = np.loadtxt(figpath+'flex_B2_ch4_allpoints')
    ## and plot them
    #plt.plot(stiff_B2_ch1[1,:], stiff_B2_ch1[0,:], 'bo')

    delta_ch1 = np.loadtxt(figpath+'delta_ch1')
    delta_ch2 = np.loadtxt(figpath+'delta_ch2')
    delta_ch3 = np.loadtxt(figpath+'delta_ch3')

    plt.figure(1)
    plt.plot(delta_ch1[0,:], delta_ch1[1,:], 'b^', label='root 7xx stiff')
    plt.plot(delta_ch2[0,:], delta_ch2[1,:], 'r^', label='30 7xx stiff')
    plt.plot(delta_ch3[0,:], delta_ch3[1,:], 'gv', label='root 7xx flex')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.0, 1.1), ncol=2)
    plt.xlabel('kg')
    plt.ylabel('strain/kg')

    ff = np.loadtxt(figpath+'ext_703_flex_B2_ch3_deltas')
    print ff
    ff = np.loadtxt(figpath+'ext_704_flex_B2_ch3_deltas')
    print ff
    ff = np.loadtxt(figpath+'ext_705_flex_B2_ch3_deltas')
    print ff

def blade_extension_3():
    """
    Final and last attempt to get the blade strain extensional influence
    """

    #bc = BladeCalibration()
    #bc.extension3_flex()
    #bc.extension3_stiff()

    # load all the bc.extension3 generated data
    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'BladeStrainCalExt/'

    #delta_ch1 = np.loadtxt(figpath+'delta_ch1_8xx')
    #delta_ch2 = np.loadtxt(figpath+'delta_ch2_8xx')
    delta_ch3 = np.loadtxt(figpath+'delta_ch3_8xx')
    delta_ch4 = np.loadtxt(figpath+'delta_ch4_8xx')

    #plt.figure(2)
    #plt.plot(delta_ch1[0,:], delta_ch1[1,:], 'bo', label='root B2 stiff')
    #plt.plot(delta_ch2[0,:], delta_ch2[1,:], 'rs', label='30 B2 stiff')
    plt.plot(delta_ch3[0,:], delta_ch3[1,:], 'g>', label='root 8xx flex')
    plt.plot(delta_ch4[0,:], delta_ch4[1,:], 'y<', label='30 8xx flex')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.0, 1.1), ncol=2)
    plt.xlabel('kg')
    plt.ylabel('strain/kg')

    #ff = np.loadtxt(figpath+'ext_703_flex_B2_ch3_deltas')
    #print ff
    #ff = np.loadtxt(figpath+'ext_704_flex_B2_ch3_deltas')
    #print ff
    #ff = np.loadtxt(figpath+'ext_705_flex_B2_ch3_deltas')
    #print ff

def plot_sync_blade_strain_dspace():
    """
    Plots to illustrate the syncing between dspace and blade strain and its
    importance.
    """

    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'sync_dspace_bladestrain/'
    respath = '/home/dave/PhD_data/OJF_data_edit/database/symlinks/'

    # illustrate the drifting of dSPACE and MicroStrain clocks, high rpm
    resfile = '0404_run_223_9.0ms_dc1_flexies_fixyaw_highrpm'
    res = ojfresult.ComboResults(respath, resfile, sync=False)
    res.overlap_pulse(figpath)
    res._sync_strain_dspace(checkplot=False)
    res.overlap_pulse(figpath)

    # illustrate the drifting of dSPACE and MicroStrain clocks, low rpm
    resfile = '0404_run_212_9.0ms_dc0.6_flexies_fixyaw_lowrpm'
    res = ojfresult.ComboResults(respath, resfile, sync=False)
    res.overlap_pulse(figpath)
    res._sync_strain_dspace(checkplot=False)
    res.overlap_pulse(figpath)

    # high rpm with free yawing
    r='0405_run_288_9.0ms_dc0_flexies_freeyawforced_yawerrorrange_fastside_highrpm'
    res = ojfresult.ComboResults(respath, r, sync=False)
    res.overlap_pulse(figpath)
    res._sync_strain_dspace(checkplot=False, min_h=0.20)
    res.overlap_pulse(figpath)
    res.dashboard_a3(figpath)

###############################################################################
### BLADE TESTS
###############################################################################

def structural_prop():
    """
    Based on st_from_etanna, but supplemented in order to have all data
    available for more advanced mass distribution estimations.

    It generates cross sectional data for the airfoil full of foam,
    a hybrid model where airfoil and etanna beam are merged together, and
    the etanna only model.

    this function was used to generate data during the ojf post
    processing phase

    Blade structural details tuned based on the experimental data: cg,
    eigenfrequency, mass including foam, stiffness from experiment

    An early version of this method lives in ojfdesign.structural_prop()
    """
    # desnity of the foam is 35 or 70 kg/m3. We used the heavier foam, but
    # that's probably only internally in the beam box.
    # It was the 70 that failed to come of the mould nicely and Nic
    # had to fall back to the 200 one. I remember him saying: "if only there
    # was a 120-150 or something, than it would have probably worked just fine"
    rho_foam = 200 # consistent with the PVC_H200 material density

    # load the general blade layout
    quasi_dat='/home/dave/PhD/Projects/Hawc2Models/ojf_post/blade_hawtopt/'
    blade = np.loadtxt(quasi_dat + 'blade.dat')
    blade[:,0] = blade[:,0] - blade[0,0]
    blade = misc._linear_distr_blade(blade)
    twist = blade[:,2]

    # prepare the st object
    md = sim.ModelData(silent=False)
    file_path = quasi_dat
    file_name = 'from-etanna-to-hawc2.st'
    md.prec_float = ' 10.06f'
    md.prec_exp = ' 10.4e'
    #md.load_st(file_path, file_name)

    # headers for the st_k are
    sti = HawcPy.ModelData.st_headers

    # calculate the case for the airfoil shape full of foam
    figpath = quasi_dat + 'foamonly_cross/'
    bl = bladeprop.Blade(figpath=figpath, plot=False)
    # blade: [r, chord, t/c, twist]
    s822, t822 = bladeprop.S822_coordinates()
    s823, t823 = bladeprop.S823_coordinates()
    airfoils = ['S823', s823, t823, 'S822', s822, t822]
    #['S823','S822']
    blade_hr, volume, area, x_na, y_na, Ixx, Iyy, st_arr, strainpos \
            = bl.build(blade[:,[0,1,3,2]],airfoils,res=None,step=None,\
                       plate=False, tw1_t=0, tw2_t=0)

    # and save in st-array format
    st_airf = np.ndarray((len(area),19))
    st_airf[:,:] = st_arr
    # material properties from PVC foam
    #st_airf[:,sti.E] = materials.Materials().cores['PVC_H200'][1]
    st_airf[:,sti.G] = materials.Materials().cores['PVC_H200'][3]
    # the chord
    st_airf[:,sti.r] = blade[:,0]
    # set shear factor
    st_airf[:,sti.k_x] = 0.7  # k_x what is shear coefficient for a plate?
    st_airf[:,sti.k_y] = 0.7  # k_y
    # set the pitch angle correct so that the structrural part is straight
    st_airf[:,sti.pitch] = twist

    # and safe to a file
    np.savetxt(quasi_dat + 'airfoil-foam-only.st', st_airf, fmt='%16.12f')
    # -------------------------------------------------------------------
    # and write to a nicely formatted st file
    m_airf_tot = integrate.trapz(st_airf[:,1], x=blade[:,0], axis=-1)
    # comments are placed in an iterable (tuple or list)
    details = md.column_header_line
    details += 'Blade foam only, volume %2.6f m3' % (volume)
    details += ' mass: %.5f kg\n' % (m_airf_tot)
    # and write to a nicely formatted st file
    # '07-00-0' : set number line in one peace
    # '07-01-a' : comments for set-subset nr 07-01 (str)
    # '07-01-b' : subset nr and number of data points, should be
    #             autocalculate every time you generate a file
    # '07-01-d' : data for set-subset nr 07-01 (ndarray(n,19))
    md.st_dict = dict()
    md.st_dict['007-003-a'] = details
    md.st_dict['007-003-b'] = '$3 18 Blade foam only\n'
    md.st_dict['007-003-d'] = st_airf

    r = blade[:,0]

    # the sets 37 and 38 are the beams final form
    for k in [41, 42]:
        # on the etanna st-37 and st-38, the core of the beam is filled
        # with 70 kg/m3 foam. However, this is only accounted for by the
        # cross sectional area and the mass. Other parameters are not
        # affected: no stiffness or moment of inertia contributions.
        # st-41 and st-42 are the same as 37/38, 39/40, but now A and m
        # only take the laminate into account. The latter is the correct
        # approach. Otherwise the extensional stiffness EA is way too high
        # and the radius of gyration is also incorrect.
        print 'converting to st format: ' + 'result-st-'+str(k)
        # for an octave saved text file
        tp = '/home/dave/Projects/0_TUDelft/cross_section_modeling/'
        data = np.loadtxt(tp+'result-st-'+str(k), skiprows=5)
        st_beam = data[:,0:19]

        # determine plate thickness. Used in modeller: tply = 0.1524e-3 as
        # ply thickness
        if k == 41:
            tw_t = 0.1524e-3*8
        elif k == 42:
            tw_t = 0.1524e-3*2
        else:
            tw_t = 0

        # first row is actually the chord, get rid of that. It should be radius
        st_beam[:,sti.r] = blade[:,0]

        # set shear factor
        st_beam[:,sti.k_x] = 0.7  # k_x what is shear coefficient for a plate?
        st_beam[:,sti.k_y] = 0.7  # k_y

        # set the pitch angle correct so that the structrural part is straight
        st_beam[:,sti.pitch] = twist

        # correct mass: add foam for the airfoil shape, LE coordinates
        #figpath = quasi_dat + 'foambeam_st%i_cross_LE/' %k
        # half chord point coordinates
        figpath = quasi_dat + 'foambeam_st%i_cross_c2/' %k
        bl = bladeprop.Blade(figpath=figpath, plot=True)
        s822, t822 = bladeprop.S822_coordinates()
        s823, t823 = bladeprop.S823_coordinates()
        airfoils = ['S823', s823, t823, 'S822', s822, t822]
        # blade: [r, chord, t/c, twist]
        blade_hr, volume, area, x_na, y_na, Ixx, Iyy, st_arr, strainpos \
            = bl.build(blade[:,[0,1,3,2]], airfoils, res=None,step=None,\
                       plate=True, tw1_t=tw_t, tw2_t=tw_t, st_arr_tw=st_beam)
        #blade_hr : radial stations, [radius, chord, t/c, twist]



        # hybrid model: merge foam only and beam together
        st_hybr = st_arr.copy()

        # -------------------------------------------------------------------
        # ORIGINAL BEAM FIBRE ONLY
        # and write to a nicely formatted st file
        m_beam_tot = integrate.trapz(st_beam[:,1], x=r, axis=-1)
        # comments are placed in an iterable (tuple or list)
        details = md.column_header_line
        details += 'from Etanna, source file: result-st-%i\n' % (k)
        details += 'beam mass: %.5f kg\n' % (m_beam_tot)
        details += 'ORIGINAL ETANNA OUTPUT\n'
        details += 'Here we assume that A,m of the beam only includes'
        details += 'the laminate. Internal foam is ignored for the beam'
        details += 'but included with the airfoil foam.\n'
        details += 'Note that this holds a small error: beam foam'
        details += 'was lighter than the airfoil foil.\n'
        if k == 41: ii=4
        else: ii=8
        md.st_dict['007-%03i-a' % (ii)] = details
        md.st_dict['007-%03i-b' % (ii)] = '$%i 18\n' % (ii)
        md.st_dict['007-%03i-d' % (ii)] = st_beam.copy()
        # and safe to a normal array txt file too
        filename = 'st-%i-beam.txt' % k
        np.savetxt(quasi_dat + filename, st_beam, fmt='%16.12f')

        # save the strain gauges positions as well
        filename = 'strainpos-%i-beam.txt' % k
        np.savetxt(quasi_dat + filename, strainpos, fmt='%16.12f')

        # -------------------------------------------------------------------
        # SAVE THE HYBRID APPROACH
        ## and write to a nicely formatted st file
        #m_hybr_tot = integrate.trapz(st_hybr[:,1], x=r, axis=-1)
        ## comments are placed in an iterable (tuple or list)
        #details = ['from Etanna, source file: result-st-%i' % (k)]
        #details[0] += '\nhybrid mass: %.5f kg' % (m_hybr_tot)
        #details[0] += '\nFor the hybrid beam and foam are added together'
        #details[0] += ' where Ixx, Iyy, rx, ry, Ip, A are scaled with the'
        #details[0] += ' ratio between E,G of foam and laminate'
        #st_dict['001' + '01' + '-' + '03' + '-data'] = st_hybr
        #st_dict['001' + '01' + '-' + '03' + '-comments'] = details
        ## and safe to a normal array txt file too
        #filename = 'st-%i-hybrid.txt' % k
        #np.savetxt(quasi_dat + filename, st_hybr, fmt='%16.12f')

        # -------------------------------------------------------------------
        # VERSION WITH CORRECTED BLADE MASS
        # also create mass inbalances:
        # Blade 1 stiff 202.34
        # Blade 2 stiff 198.1
        # Blade 3 stiff 222.7
        st_arr_cor = st_airf.copy()
        # WRONG FIRST TIME: it is the other way around: 41 is the FLEX BLADE
        # > it is corrected now
        if k == 42:
            ii = 5
            m_goal = 0.20234
            dm = (m_goal - (m_airf_tot + m_beam_tot))/r[-1]
            st_arr_cor[:,sti.m] = st_airf[:,sti.m] + st_beam[:,sti.m] + dm
            m_cor = integrate.trapz(st_arr_cor[:,sti.m], x=r, axis=-1)
            details = md.column_header_line
            details += 'BLADE 1 STIFF CORRECTED: stiffness foam only, \n'
            details += 'mass=beam+foam+correction=%3.6f kg ' % m_cor
            details += '(should match %3.6f)\n' % m_goal
            details += 'beam: Etanna source file: result-st-%i\n' % (k)
            md.st_dict['007-%03i-a' % (ii)] = details
            md.st_dict['007-%03i-b' % (ii)] = '$%i 18\n' % (ii)
            md.st_dict['007-%03i-d' % (ii)] = st_arr_cor.copy()

            m_goal = 0.1981
            dm = (m_goal - (m_airf_tot + m_beam_tot))/r[-1]
            st_arr_cor[:,sti.m] = st_airf[:,sti.m] + st_beam[:,sti.m] + dm
            m_cor = integrate.trapz(st_arr_cor[:,sti.m], x=r, axis=-1)
            details = md.column_header_line
            details += 'BLADE 2 STIFF CORRECTED: stiffness foam only, \n'
            details += 'mass=beam+foam+correction=%3.6f kg ' % m_cor
            details += '(should match %3.6f)\n' % m_goal
            details += 'beam: Etanna source file: result-st-%i\n' % (k)
            md.st_dict['007-%03i-a' % (ii+1)] = details
            md.st_dict['007-%03i-b' % (ii+1)] = '$%i 18\n' % (ii+1)
            md.st_dict['007-%03i-d' % (ii+1)] = st_arr_cor.copy()

            m_goal = 0.2227
            dm = (m_goal - (m_airf_tot + m_beam_tot))/r[-1]
            st_arr_cor[:,sti.m] = st_airf[:,sti.m] + st_beam[:,sti.m] + dm
            m_cor = integrate.trapz(st_arr_cor[:,sti.m], x=r, axis=-1)
            details = md.column_header_line
            details += 'BLADE 3 STIFF CORRECTED: stiffness foam only, \n'
            details += 'mass=beam+foam+correction=%3.6f kg ' % m_cor
            details += '(should match %3.6f)\n' % m_goal
            details += 'beam: Etanna source file: result-st-%i\n' % (k)
            md.st_dict['007-%03i-a' % (ii+2)] = details
            md.st_dict['007-%03i-b' % (ii+2)] = '$%i 18\n' % (ii+2)
            md.st_dict['007-%03i-d' % (ii+2)] = st_arr_cor.copy()
        # Blade 1 flex 162.6
        # Blade 2 flex 150.7
        # Blade 3 flex 141.12
        elif k == 41:
            ii=9
            m_goal = 0.1626
            dm = (m_goal - (m_airf_tot + m_beam_tot))/r[-1]
            st_arr_cor[:,sti.m] = st_airf[:,sti.m] + st_beam[:,sti.m] + dm
            m_cor = integrate.trapz(st_arr_cor[:,sti.m], x=r, axis=-1)
            details = md.column_header_line
            details += 'BLADE 1 FLEX CORRECTED: stiffness foam only, \n'
            details += 'mass=beam+foam+correction=%3.6f kg ' % m_cor
            details += '(should match %3.6f)\n' % m_goal
            details += 'beam: Etanna source file: result-st-%i\n' % (k)
            md.st_dict['007-%03i-a' % (ii)] = details
            md.st_dict['007-%03i-b' % (ii)] = '$%i 18\n' % (ii)
            md.st_dict['007-%03i-d' % (ii)] = st_arr_cor.copy()

            m_goal = 0.1507
            dm = (m_goal - (m_airf_tot + m_beam_tot))/r[-1]
            st_arr_cor[:,sti.m] = st_airf[:,sti.m] + st_beam[:,sti.m] + dm
            m_cor = integrate.trapz(st_arr_cor[:,sti.m], x=r, axis=-1)
            details = md.column_header_line
            details += 'BLADE 2 FLEX CORRECTED: stiffness foam only, \n'
            details += 'mass=beam+foam+correction=%3.6f kg ' % m_cor
            details += '(should match %3.6f)\n' % m_goal
            details += 'beam: Etanna source file: result-st-%i\n' % (k)
            md.st_dict['007-%03i-a' % (ii+1)] = details
            md.st_dict['007-%03i-b' % (ii+1)] = '$%i 18\n' % (ii+1)
            md.st_dict['007-%03i-d' % (ii+1)] = st_arr_cor.copy()

            m_goal = 0.14112
            dm = (m_goal - (m_airf_tot + m_beam_tot))/r[-1]
            st_arr_cor[:,sti.m] = st_airf[:,sti.m] + st_beam[:,sti.m] + dm
            m_cor = integrate.trapz(st_arr_cor[:,sti.m], x=r, axis=-1)
            details = md.column_header_line
            details += 'BLADE 3 FLEX CORRECTED: stiffness foam only, \n'
            details += 'mass=beam+foam+correction=%3.6f kg ' % m_cor
            details += '(should match %3.6f)\n' % m_goal
            details += 'beam: Etanna source file: result-st-%i\n' % (k)
            md.st_dict['007-%03i-a' % (ii+2)] = details
            md.st_dict['007-%03i-b' % (ii+2)] = '$%i 18\n' % (ii+2)
            md.st_dict['007-%03i-d' % (ii+2)] = st_arr_cor.copy()
        else:
            raise UserWarning, 'what st set are you talking about? %i ??' % k

        # start plotting -----------------------------------------------------
        # Compare the beam, foam only and hybrid models
        figfile = 'st-%i-beam-hybrid-airfoil' % k
        pa4 = plotting.A4Tuned()
        pa4.setup(quasi_dat+figfile, nr_plots=7, grandtitle=figfile)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
        ax3 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 3)
        ax4 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 4)
        ax5 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 5)
        ax6 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 6)
        ax7 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 7)
        #ax8 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 8)

        #ax1.set_title('area')
        ax1.plot(r, st_beam[:,sti.A], 'bx-', label='$A_{beam}$')
        ax1.plot(r, st_airf[:,sti.A], 'rs-', label='$A_{foam}$')
        ax1.plot(r, st_hybr[:,sti.A], 'go-', label='$A_{hybrid}$')
        ax1.legend()
        ax1.grid(True)

        #ax2.set_title('mass')
        ax2.plot(r, st_beam[:,sti.m], 'bx-', label='$m_{beam}$')
        ax2.plot(r, st_airf[:,sti.m], 'rs-', label='$m_{foam}$')
        ax2.plot(r, st_hybr[:,sti.m], 'go-', label='$m_{hybrid}$')
        ax2.plot(r, st_arr_cor[:,sti.m], 'yo-', label='$m_{cor}$')
        ax2.legend()
        ax2.grid(True)

        #ax3.set_title('Ixx')
        ax3.plot(r, st_beam[:,sti.Ixx], 'bx-', label='$I_{xx} beam$')
        ax3.plot(r, st_airf[:,sti.Ixx], 'rs-', label='$I_{xx} foam$')
        ax3.plot(r, st_hybr[:,sti.Ixx], 'go-', label='$I_{xx} hybrid$')
        ax3.legend()
        ax3.grid(True)

        #ax3.set_title('Iyy')
        ax4.plot(r, st_beam[:,sti.Iyy], 'bx-', label='$I_{yy} beam$')
        ax4.plot(r, st_airf[:,sti.Iyy], 'rs-', label='$I_{yy} foam$')
        ax4.plot(r, st_hybr[:,sti.Iyy], 'go-', label='$I_{yy} hybrid$')
        ax4.legend()
        ax4.grid(True)

        #ax3.set_title('x_e')
        ax5.plot(r, st_beam[:,sti.x_e], 'bx-', label='$x_e beam$')
        ax5.plot(r, st_airf[:,sti.x_e], 'rs-', label='$x_e foam$')
        ax5.plot(r, st_hybr[:,sti.x_e], 'go-', label='$x_e hybrid$')
        ax5.legend()
        ax5.grid(True)

        #ax3.set_title('y_e')
        ax6.plot(r, st_beam[:,sti.y_e], 'bx-', label='$y_e beam$')
        ax6.plot(r, st_airf[:,sti.y_e], 'rs-', label='$y_e foam$')
        ax6.plot(r, st_hybr[:,sti.y_e], 'go-', label='$y_e hybrid$')
        ax6.legend()
        ax6.grid(True)

        #ax3.set_title('I_p')
        ax7.plot(r, st_beam[:,sti.I_p], 'bx-', label='$I_p beam$')
        ax7.plot(r, st_airf[:,sti.I_p], 'rs-', label='$I_p foam$')
        ax7.plot(r, st_hybr[:,sti.I_p], 'go-', label='$I_p hybrid$')
        ax7.legend()
        ax7.grid(True)

        #ax3.set_title('I_p')
        #ax8.plot(r, fact_E, 'bx-', label='fact_E')
        #ax8.plot(r, fact_G, 'rs-', label='fact_G')
        ##ax8.plot(r, st_hybr[:,sti.I_p], 'go-', label='I_p hybrid')
        #ax8.legend()
        #ax8.grid(True)

        pa4.save_fig()
        # end plotting -------------------------------------------------------


    # -------------------------------------------------------------------
    # and write to a nicely formatted st file
    md.write_st(file_path, file_name)


def blade_cg():
    """
    Measurements originally carried out to determine the mass distribution,
    but after working out the solution I realised all the measurement points
    only gave away one parameter: the center of gravity.
    """

    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'blademassproperties/'

    cg = ojfresult.BladeCgMass(figpath=figpath)

    for blade in cg.bladedict:
        struct = blade.split(' ')[0]
        bladenr = blade.split(' ')[1]
        cg.read(struct, bladenr, plot=True)


def blade_aero_coeff():
    """
    CL, CD, CM curves for the blade only tests

    Indices
     0 : Time
     1 : Fx
     2 : Fy
     3 : Fz
     4 : Mx
     5 : My
     6 : Mz
     7 : Fan Speed
     8 : Air Temp
     9 : Atm Press
    10 : ??
    11 : wind speed
    12 : ??
    13 : AoA
    """

    def loadresults(path):
        res = np.loadtxt(path, skiprows=1)
        # invert the sign on the drag force
        res[:,1] *= -1.0
        return res

    # TODO: move function to ojfresult.py

    bladepath = '/home/dave/PhD_data/OJF_data_edit/bladeonly/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/bladeonly/'
    figname = 'flex-stiff-all-V'
    windi=11
    aoai=13

    chis = [1,2,3,4,5,6]
    # for the OJF coordinates
    titles = [r'$\frac{F_{x}}{0.5 \rho V^2}$', r'$\frac{F_{y}}{0.5 \rho V^2}$',
              r'$\frac{F_{z}}{0.5 \rho V^2}$', r'$\frac{M_{x}}{0.5 \rho V^2}$',
              r'$\frac{M_{y}}{0.5 \rho V^2}$', r'$\frac{M_{z}}{0.5 \rho V^2}$']
    names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # sensible aerodynamics, Lift, Drag and Torsion moment
    titles = [r'$\frac{D}{0.5 \rho V^2}$', r'$\frac{L}{0.5 \rho V^2}$',
              r'$\frac{F_{z}}{0.5 \rho V^2}$', r'$\frac{M_{L}}{0.5 \rho V^2}$',
              r'$\frac{M_{D}}{0.5 \rho V^2}$', r'$\frac{M_{T}}{0.5 \rho V^2}$']
    names = [ 'D',  'L', 'Fz','M_L','M_D','M_T']

    xlabel = 'Reference pitch angle [deg]'

    figsize_x = plotting.TexTemplate.pagewidth*0.5
    figsize_y = plotting.TexTemplate.pagewidth*0.5
    scale = 1.8

    # -------------------------------------------------------------------------
    # all Reynolds numbers included
    for i, chi in enumerate(chis):
        figname = 'flex-stiff-all-V-%s' % names[i]
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                       grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                       wstop_cm=1.0, wsbottom_cm=1.0,
                       figsize_x=figsize_x, figsize_y=figsize_y)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        bladefile = '001_flex_b1_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'r+', label='B1 flex')

        bladefile = '004_flex_b2_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'b*', label='B2 flex')

        bladefile = '005_flex_b3_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'k^', label='B3 flex', alpha=0.5)

        bladefile = '006_stiff_b3_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'yo', label='B3 stiff', alpha=0.5)

        ax1.set_title(titles[i])
        ax1.grid(True)
        ax1.legend(loc='best')
        ax1.set_xlabel(xlabel)

        pa4.save_fig()
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # only selected Reynolds numbers
    for i, chi in enumerate(chis):
        lowlim = 14.5
        uplim = 15.5
        figname = 'flex-stiff-V-%1.2f-%1.2f-ms-%s' % (lowlim, uplim, names[i])
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                       grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                       wstop_cm=1.0, wsbottom_cm=1.0,
                       figsize_x=figsize_x, figsize_y=figsize_y)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        bladefile = '001_flex_b1_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'r+', label='B1 flex')

        bladefile = '004_flex_b2_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'b*', label='B2 flex')

        bladefile = '005_flex_b3_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'k^', label='B3 flex', alpha=0.5)

        bladefile = '006_stiff_b3_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'yo', label='B3 stiff', alpha=0.5)

        ax1.set_title(titles[i])
        ax1.grid(True)
        ax1.legend(loc='best')
        ax1.set_xlabel(xlabel)

        pa4.save_fig()
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # only selected Reynolds numbers
    for i, chi in enumerate(chis):
        lowlim = 24.5
        uplim = 25.5
        figname = 'flex-stiff-V-%1.2f-%1.2f-ms-%s' % (lowlim, uplim, names[i])
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                       grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                       wstop_cm=1.0, wsbottom_cm=1.0,
                       figsize_x=figsize_x, figsize_y=figsize_y)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        bladefile = '001_flex_b1_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'r+', label='B1 flex')

        bladefile = '004_flex_b2_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'b*', label='B2 flex')

        bladefile = '005_flex_b3_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'k^', label='B3 flex', alpha=0.5)

        bladefile = '006_stiff_b3_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        res = res[res[:,windi].__ge__(lowlim),:]
        res = res[res[:,windi].__le__(uplim),:]
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'yo', label='B3 stiff', alpha=0.5)

        ax1.set_title(titles[i])
        ax1.grid(True)
        ax1.legend(loc='best')
        ax1.set_xlabel(xlabel)

        pa4.save_fig()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # SWEPT BLADES 1 and 2
    for i, chi in enumerate(chis):
        figname = 'swept-%s' % (names[i])
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                       grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                       wstop_cm=1.0, wsbottom_cm=1.0,
                       figsize_x=figsize_x, figsize_y=figsize_y)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        bladefile = '002_swept_b1_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'rs', label='B1 swept')

        bladefile = '003_swept_b2_t_0.1_n_30.txt'
        res = loadresults(bladepath+bladefile)
        CLS = -res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])
        ax1.plot(res[:,aoai],CLS, 'gs', label='B2 swept')

        ax1.set_title(titles[i])
        ax1.grid(True)
        ax1.legend(loc='best')
        ax1.set_xlabel(xlabel)

        pa4.save_fig()
    # -------------------------------------------------------------------------

def blade_contour_corr():
    """
    Checking the blade contour corrections, because something is not going
    ok.

    CONCLUSIONS:

    We will discard blade 1 flex, because there is something going on there.
    It is visible for all load cases, although 156 seems to have some ok
    results. Same holds for blade 3 stiff.

    Alternatively, take the average correction of the other two blades,
    that seems to work ok in both flex and stiff cases.

    However, as conclusion we note that we will not use B1 flex and B3 stiff.
    Just take the average of the two others for the stiffness.
    """

    testdatapath = '/home/dave/PhD_data/OJF_data_edit/blade_contour/'
    tipmass='256'
    struct='flex'

    bc = ojfresult.BladeContour()
    mean1, corr1 = bc.mean_defl(testdatapath, struct, 'B1', tipmass,
                                silent=False, correct=True)
    mean2, corr2 = bc.mean_defl(testdatapath, struct, 'B2', tipmass,
                                silent=False, correct=True)
    mean3, corr3 = bc.mean_defl(testdatapath, struct, 'B3', tipmass,
                                silent=False, correct=True, istart=5, iend=16)

    #b1=mean1[1,:]+corr1
    #b2=mean2[1,:]+corr2
    #b3=mean3[1,:]+corr3

    plt.figure()
    #plt.plot(mean1[0,:], mean1[1,:], 'r', label='B1 cor')
    plt.plot(mean1[0,:], mean1[1,:]+corr1-(corr2+corr3)/2., 'r', label='B1 cor')
    plt.plot(mean2[0,:], mean2[1,:], 'k', label='B2 cor')
    #plt.plot(mean3[0,:], mean3[1,:]+corr3-(corr1+corr2)/2., 'b', label='B3 cor')
    plt.plot(mean3[0,:], mean3[1,:], 'b', label='B3 cor')
    plt.grid()
    plt.plot(mean1[0,:], corr1, 'r.', label='cor1')
    plt.plot(mean2[0,:], corr2, 'k.', label='cor2')
    plt.plot(mean3[0,:], corr3, 'b.', label='cor3')

    mean1_nc, corr1 = bc.mean_defl(testdatapath, struct, 'B1', tipmass,
                                silent=False, correct=False)
    mean2_nc, corr2 = bc.mean_defl(testdatapath, struct, 'B2', tipmass,
                                silent=False, correct=False)
    mean3_nc, corr3 = bc.mean_defl(testdatapath, struct, 'B3', tipmass,
                                silent=False, correct=False)

    plt.plot(mean1_nc[0,:], mean1_nc[1,:], 'r--', label='B1 or')
    plt.plot(mean2_nc[0,:], mean2_nc[1,:], 'k--', label='B2 or')
    plt.plot(mean3_nc[0,:], mean3_nc[1,:], 'b--', label='B3 or')

    plt.plot(mean1[0,:], np.abs(mean1[1,:]-mean3[1,:]), 'y*')
    plt.plot(mean1_nc[0,:], np.abs(mean1_nc[1,:]-mean3_nc[1,:]), 'y+')

    # delta

    plt.legend(loc='upper left')


def save_contour_cg_mass():
    """
    Save the mean deflection curve for sharing data with others
    """

    bc = ojfresult.BladeContour()
    testdatapath = '/home/dave/PhD_data/OJF_data_edit/blade_contour/'

    base = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    massdata = base + 'blademassproperties/'
    countour = base + 'bladecontour/'

    flextip = ['56', '156', '256', '306']
    stifftip = ['106', '606', '706']
    for b in ['B1', 'B2', 'B3']:

        for w in flextip:
            struct = 'flex'
            bladenr = b
            tipmass = w
            mean, corr = bc.mean_defl(testdatapath, struct, bladenr, tipmass,
                                      silent=False, correct=True)
            tmp = np.append(mean, corr.reshape(1,len(corr)), axis=0)
            fname = 'mean_corr_%s_%s_%s' % (struct, bladenr, tipmass)
            np.savetxt(countour+fname, tmp)

        for w in stifftip:
            struct = 'stiff'
            bladenr = b
            tipmass = w
            mean, corr = bc.mean_defl(testdatapath, struct, bladenr, tipmass,
                                      silent=False, correct=True)
            tmp = np.append(mean, corr.reshape(1,len(corr)), axis=0)
            fname = 'mean_corr_%s_%s_%s' % (struct, bladenr, tipmass)
            np.savetxt(countour+fname, tmp)

        for struct in ['stiff', 'flex']:
            # and for mass and cg
            cg, mass = ojfresult.BladeCgMass().read(struct, bladenr)
            fname = 'cg_mass_%s_%s' % (struct, bladenr)
            # we don't have cg data for all blades....
            if type(cg).__name__ == 'NoneType':
                bladenr_next = int(bladenr[1])+1
                if bladenr_next > 3:
                    bladenr_next = 1
                bladenr_next = bladenr[0] + str(bladenr_next)
                cg, tmp = ojfresult.BladeCgMass().read(struct, bladenr_next)

            if type(cg).__name__ == 'NoneType':
                bladenr_next = int(bladenr_next[1])+1
                if bladenr_next > 3:
                    bladenr_next = 1
                bladenr_next = bladenr[0] + str(bladenr_next)
                cg, tmp = ojfresult.BladeCgMass().read(struct, bladenr_next)

            if type(cg).__name__ == 'NoneType':
                raise UserWarning, 'Couldn\'t find a cg reference value'
            np.savetxt(massdata+fname, np.array([cg, mass]))



def blade_contour():
    """
    As used for the thesis plotting
    """

    filepath = '/home/dave/PhD_data/OJF_data_edit/blade_contour/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/bladecontour/'

    # plot a single case
    #case = 'flex_B1_LE_0'
    #figname = case
    #grandtitle = 'flexible blade 1\nleading edge track, 0 tip load'
    #bc.plot_blade(filepath+case, figpath, figname, grandtitle)

    bc = ojfresult.BladeContour()
    bc.compare_blade_set(filepath, ['flex', '306'], figpath, scale=2.0)
    bc.compare_blade_set(filepath, ['flex', '256'], figpath, scale=2.0)
    bc.compare_blade_set(filepath, ['flex', '156'], figpath, scale=2.0)
    bc.compare_blade_set(filepath, ['flex', '56'], figpath, scale=2.0)
    bc.compare_blade_set(filepath, ['stiff', '706'], figpath, scale=2.0)
    bc.compare_blade_set(filepath, ['stiff', '606'], figpath, scale=2.0)
    bc.compare_blade_set(filepath, ['stiff', '106'], figpath, scale=2.0)

    flextip = ['56', '156', '256', '306']
    stifftip = ['106', '606', '706']
    for b in ['B1', 'B2', 'B3']:
        for w in flextip:
            search = ['flex', b, w]
            bc.compare_blade_set(filepath, search, figpath, correct=True,
                                 zoom=True, scale=2.0)
            bc.compare_blade_set(filepath, search, figpath, correct=True,
                                 zoom=False, scale=2.0)
        for w in stifftip:
            search = ['stiff', b, w]
            bc.compare_blade_set(filepath, search, figpath, correct=True,
                                 zoom=True, scale=2.0)
            bc.compare_blade_set(filepath, search, figpath, correct=True,
                                 zoom=False, scale=2.0)

    #fpath, struct, tipload, figpath
    bc.compare_blades(filepath, 'stiff', '606', figpath, scale=2.0)
    bc.compare_blades(filepath, 'flex', '256', figpath, scale=2.0)

    # following cases where changed

    # for two measurements the delfections at the tip where set to nan
    # because the deflection meter was not following the blade anymore
    # flex_B2_mid_306 and flex_B2_TE_306

    # cases removed because the measurement reference frame was not locked into
    # position
    # flex_B3_LE_256 and flex_B3_LE_306

def blade_contour_all_raw():
    """
    Just plot all the raw result files. However, the out of range values of
    the acquisition system are already replaced with nan's
    """

    # plot a single case
    #case = 'flex_B1_LE_0'
    #figname = case
    #grandtitle = 'flexible blade 1\nleading edge track, 0 tip load'
    #bc.plot_blade(filepath+case, figpath, figname, grandtitle)

    bc = ojfresult.BladeContour()

    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/bladecontour/'
    figpath += 'rawresults/'
    filepath = '/home/dave/PhD_data/OJF_data_edit/blade_contour/'
    for f in [f for f in os.walk(filepath)][0][2]:
        # ignore all the xls files
        if f.endswith('.xls'): continue

        figname = f
        tmp = f.replace('B', 'Blade ').split('_')
        tmp[-1] = 'tip load: %s gr' % tmp[-1]
        grandtitle = ' '.join(tmp)
        bc.plot_blade(filepath+f, figpath, figname, grandtitle)

def blade_damping_all(checkplot=True):
    """
    Determine the damping from the blade virbation tests.comboresults

    This is actually also done in plot_psd_blades, but the results are not
    saved
    """

    respath = '/home/dave/PhD_data/OJF_data_edit/08_vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfreq_damp_blade/'

    freqpath='/home/dave/PhD/Projects/PostProcessing/'
    freqpath += 'OJF_tests/eigenfreq_damp_blade/'

    # APPROACH: get all measurements for each blade and collect all calculated
    # damping values
    cases = [['501_b1_stiff_trigger02_2664171792', 'stiff_B1', [2,3]],
             ['502_b1_stiff_trigger03_504929400',  'stiff_B1', [2,3]],
             ['503_b2_stiff_trigger01_507476419',  'stiff_B2', [0,1]],
             ['504_b2_stiff_trigger02_2430537494', 'stiff_B2', [0,1]],
             ['505_b2_stiff_trigger03_165311127',  'stiff_B2', [0,1]],
             ['506_b1_flex_trigger01_2990608704',  'flex_B1',  [2,3]],
             ['508_b1_flex_trigger03_3392483542',  'flex_B1',  [2,3]],
             ['512_b2_flex_trigger01_3962726656',  'flex_B2',  [2,3]],
             ['513_b2_flex_trigger02_1803468531',  'flex_B2',  [2,3]]]
    # blade 2 flex is also on index 2-3 (instead of 0-1)

    # questionables: 510, 511, 515, 516, (517)
    cases2= [['500_trigger01_413314024',           'stiff_B1', [2,3]],
             ['507_b1_flex_trigger02_1468965218',  'flex_B1',  [2,3]],
             ['510_b1_flex_trigger02_2059868129',  'flex_B1',  [2,3]],
             ['511_b1_flex_trigger03_3824274622',  'flex_B1',  [2,3]],
             ['515_b2_flex_trigger01_2423128572',  'flex_B2',  [2,3]],
             ['516_b2_flex_trigger02_157778266',   'flex_B2',  [2,3]]]

    #check = True
    #cases = [ ['504_b2_stiff_trigger02_2430537494',  'stiff_B2',  [0]], ]

    #channels = [0,1,2,3]
    #channel_names = ['stiff B2 root','stiff B2 30%',
                     #'stiff B1 root','stiff B1 30%']

    # save all the measurements in a dictionary, and save each of them later
    damp_all = {}

    for case in cases:
        # load the blade strain result file
        resfile = case[0]
        blade = ojfresult.BladeStrainFile(respath+resfile)

        for chi in case[2]:
            if chi in [0,2]: add = '_root'
            if chi in [1,3]: add = '_30'
            print
            print 79*'-'
            print case[1] + add
            print 79*'-'

            # get the peaks, damping, exp fit for all blocks in the file
            i_p, i_bl, damp_bl, fit_i, fn5, psd = misc.damping(blade.time,
                                             blade.data[:,chi], verbose=True,
                                             offset_start=100, offset_end=200)

            # keep copy of all the results
            damp_all['damp_' + case[1] + add] = np.array(damp_bl)

    # save all the damp results
    for key, value in damp_all.iteritems():
        np.savetxt(freqpath+key, value)

    print
    print 'average damping per blade'

    # and come up with one number: average per damp per blade
    for freqfile in ['stiff_B1', 'stiff_B2', 'flex_B1', 'flex_B2']:
        mean = []
        for add in ['_30', '_root']:
            dampfile = 'damp_' + freqfile + add
            try:
                damp_arr = np.loadtxt(freqpath+dampfile)
                mean.append(damp_arr.mean())
            except IOError:
                pass
        # and get the global mean
        mean = np.array(mean).mean()
        # and save the file
        np.savetxt(freqpath+'damp_'+freqfile, np.array([mean]))
        print freqfile.rjust(10) + '%7.4f' % mean

    # and create for blade 3 the average of blade1 and blade2
    d1 = np.loadtxt(freqpath+'damp_stiff_B1')
    d2 = np.loadtxt(freqpath+'damp_stiff_B2')
    d3 = (d2+d1)/2.
    np.savetxt(freqpath+'damp_stiff_B3', (d3,))

    d1 = np.loadtxt(freqpath+'damp_flex_B1')
    d2 = np.loadtxt(freqpath+'damp_flex_B2')
    d3 = (d2+d1)/2.
    np.savetxt(freqpath+'damp_flex_B3', (d3,))


def plot_psd_blades():
    """
    Create for each blade a PSD of the root and 30% sections. For each blade
    overplot all the different blocks identified.

    As used for the plots in the thesis
    """
    respath = '/home/dave/PhD_data/OJF_data_edit/08_vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfreq_damp_blade/'

    freqpath='/home/dave/PhD/Projects/PostProcessing/'
    freqpath += 'OJF_tests/eigenfreq_damp_blade/'

    # APPROACH: get all measurements for each blade and collect all calculated
    # damping values
    cases = [['501_b1_stiff_trigger02_2664171792', 'stiff_B1', [2,3]],
             ['502_b1_stiff_trigger03_504929400',  'stiff_B1', [2,3]],
             ['503_b2_stiff_trigger01_507476419',  'stiff_B2', [0,1]],
             ['504_b2_stiff_trigger02_2430537494', 'stiff_B2', [0,1]],
             ['505_b2_stiff_trigger03_165311127',  'stiff_B2', [0,1]],
             ['506_b1_flex_trigger01_2990608704',  'flex_B1',  [2,3]],
             ['508_b1_flex_trigger03_3392483542',  'flex_B1',  [2,3]],
             ['512_b2_flex_trigger01_3962726656',  'flex_B2',  [2,3]],
             ['513_b2_flex_trigger02_1803468531',  'flex_B2',  [2,3]]]

#    cases = [['502_b1_stiff_trigger03_504929400',  'stiff_B1', [2,3]]]

    # save the results per blade
    blade_data = {}
    case_peaks = {}
    nrpeaks = 20
    thresh_occ = 0.5
    # with 2048, the precision on the found peaks is 1.0f
    NNFT = 2048

    for case in cases:

        print
        print 79*'-'
        print case[1]
        print 79*'-'

        # load the blade strain result file
        resfile = case[0]
        blade = ojfresult.BladeStrainFile(respath+resfile)

        for chi in case[2]:
            # get the peaks, damping, exp fit for all blocks in the file
            i_p, i_bl, damp_bl, fit_i, fn5, psd = misc.damping(blade.time,
                             blade.data[:,chi], verbose=True, checkplot=False,
                             offset_start=100, offset_end=200, NFFT=NNFT)
            # and the PSD from each block
            for Pxx, freqs in psd:
                try:
                    blade_data[case[1]].append([Pxx, freqs])
                except KeyError:
                    blade_data[case[1]] = [[Pxx, freqs]]

                # peak detection: only if strong peak occurs often
                # limit the Pxx search to up to 250 Hz for the blades
                imax = np.abs(freqs-250.0).argmin()
                imin = np.abs(freqs-10.0).argmin()
                # keep track of all the peaks, we want to select strong peaks
                # that occur often. In that way filter out peaks that only
                # occur once
                Pxx_log = 10.*np.log10(Pxx[imin:imax])
                # find all the peaks on the PSD
                ifns = wafo.misc.findpeaks(Pxx_log, n=len(Pxx_log), min_h=0)
                # and only consider the 20 strongest peaks, but convert the
                # frequency occurance to an int
                rounded_freqs = np.round(freqs[imin:imax][ifns[:nrpeaks]])
                appendix = np.array(rounded_freqs, dtype=np.int32)
                try:
                    q = case[1]
                    case_peaks[q] = np.append(case_peaks[q], appendix)
                except KeyError:
                    case_peaks[case[1]] = appendix

    # and now plot all the different measurements in one plot per blade
    for bladecase in blade_data:

        nr_tests = len(blade_data[bladecase])
        replace = (nr_tests, bladecase.replace('_B', ' blade '))
        title = 'Power spectra of %i decay tests, %s' % replace

        figsize_x = plotting.TexTemplate.pagewidth*0.99
        figsize_y = plotting.TexTemplate.pagewidth*0.30
        figname = bladecase + '_psd_all'
        scale = 1.8
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                       grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                       wstop_cm=0.8, wsbottom_cm=1.0,
                       figsize_x=figsize_x, figsize_y=figsize_y)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.set_yscale('log')
        ax1.grid()
        ax1.set_title(title)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_xlim([0, 200])
        ax1.set_ylim([1e-5, 1e5])
        # force some courser ticks, manually
#        xticks = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], num=11)
#        ax1.xaxis.set_ticks(xticks.tolist())
        # or using the ticker class
        locator = mpl.ticker.LinearLocator(numticks=11, presets=None)
        ax1.xaxis.set_major_locator( locator )

#        Pxx_mean = np.array(blade_data[bladecase][0][0].shape)
        for Pxx, freqs in blade_data[bladecase]:
            ax1.plot(freqs, Pxx)
#            Pxx_mean += Pxx

        Pxx, freqs = blade_data[bladecase][2]
        # only consider peaks that occur in at least 2/3 of the blocks
        count = np.bincount(case_peaks[bladecase])
        nrblocks = len(blade_data[bladecase])
        # and divide the sum with the number of blocks to get the mean
#        Pxx_mean /= nrblocks
        threshold = int(round(nrblocks*thresh_occ))
        freq_sort = count.argsort()
        freq_occ = np.sort(count)
        # select those that only occure more than the threshold
        iselect = freq_occ.__ge__(threshold)
        # and we have a number of occurances
        freq_select = freq_sort[iselect]
        xpos = [20,-20,-10,-10,-10,  0,  1,-10,  1,-10,-10,-10,-10,-10,-10,-10]
        ypos = [ 0, 40, 25, 40, 35, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
        # find the corresponding Pxx values, based on the average Pxx
        for i, fn in enumerate(sorted(freq_select)):
            iPxx = np.abs(freqs-fn).argmin()
            # and labels for only one PSD
            ax1.annotate(r'$%1.0f$' % fn,
                xy=(fn, Pxx[iPxx]), xycoords='data',
                xytext=(xpos[i], ypos[i]), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"),
                fontweight='bold', fontsize=12)

        pa4.save_fig()


def plot_damping_blade():
    """
    Take a random virbation test and make some plots for the thesis
    """

    # example input data ----------------------------------------------------
    respath = '/home/dave/PhD_data/OJF_data_edit/08_vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfreq_damp_blade/'
    freqpath='/home/dave/PhD/Projects/PostProcessing/'
    freqpath += 'OJF_tests/eigenfreq_damp_blade/'
    resfile='504_b2_stiff_trigger02_2430537494'
    freqfile='stiff_B2'
    checkplot = False
    i = 5 # block number
    k = 0 # blade channel nr
    # ------------------------------------------------------------------------

    freqfile = 'eigenfreq_' + freqfile

    blade = ojfresult.BladeStrainFile(respath+resfile)
    # get the peaks, damping, exp fit for all blocks in the file
    i_p, i_bl, damp_bl, fit_i, fn5, psd = misc.damping(blade.time,
                        blade.data[:,k], checkplot=checkplot, verbose=True)
    # indices to peaks of one cycle block
    ii = i_p[i]
    # the peaks for this block
    peaks = blade.data[ii,k]

    # center data around a zero mean for more convinient processing
    data = blade.data[i_bl[i],k] - blade.data[i_bl[i],k].mean()
    # also normalize the data so it is more general applicable
    data *= 1.0/(data.max() - data.min())

    # eigenfrequency as saved
    fn = np.loadtxt(freqpath + freqfile)

    # and now we can get the damping, take the first eigenfrequency
    zeta_fit = damp_bl[i]
    zeta_all = np.log(peaks[0]/peaks[-1])/len(peaks)
    zeta_rand = np.log(peaks[10]/peaks[30])/len(peaks[10:31])

    print '%i - %i' % (k,i)
    print 'fn, saved: %4.1f' % (float(fn[0]))
    print ' zeta_fit: %8.4f' % -zeta_fit
    print ' zeta_all: %8.4f' % zeta_all
    print 'zeta_rand: %8.4f' % zeta_rand

    # -------------------------------------------------------------------------
    # just plot the first found vibration test
    figsize_x = plotting.TexTemplate.pagewidth*0.99
    figsize_y = plotting.TexTemplate.pagewidth*0.35
    figname = resfile + '_block_%i' % i
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                   wstop_cm=0.8, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(blade.time[i_bl[i]], data)
    ax1.grid()
    ax1.set_title('Blade vibration decay test')
#    ax1.set_ylabel('blade strain sensor output [-]')
    ax1.set_xlabel('time [s]')
    ax1.set_xlim([blade.time[i_bl[i]].min(),blade.time[i_bl[i]].max()])
    pa4.save_fig()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # frequency domain plot:
    figsize_x = plotting.TexTemplate.pagewidth*0.99
    figsize_y = plotting.TexTemplate.pagewidth*0.35
    figname = resfile + '_block_%i_psd' % i
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                   wstop_cm=0.8, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    Pxx, freqs = psd[i]
    ax1.plot(freqs, Pxx, 'k')
    ax1.set_yscale('log')
    ypos = [1e-5, 1e3, 1e-5, 1e3, 1e-5, 1e3]
    # indicate the top 5 frequencies
    for ifn, fn in enumerate(sorted(fn5)):
        ax1.axvline(x=fn)
        y = ypos[ifn]
        textbox = r'$f_%i = %1.1f$' % (ifn+1, fn)
        bbox = dict(boxstyle="round", alpha=0.8, edgecolor=(1., 0.5, 0.5),
                    facecolor=(1., 0.8, 0.8),)
        ax1.text(fn, y, textbox, fontsize=12, verticalalignment='bottom',
                 horizontalalignment='center', bbox=bbox)

    ax1.grid()
    ax1.set_title('Blade vibration decay test frequency domain')
#    ax1.set_ylabel('blade strain')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_xlim([0, 250])
    pa4.save_fig()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # peak plot
    figsize_x = plotting.TexTemplate.pagewidth*0.99
    figsize_y = plotting.TexTemplate.pagewidth*0.35
    figname = resfile + '_block_%i_peaks_dampfit' % i
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                   wstop_cm=0.8, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(blade.time[i_bl[i]], blade.data[i_bl[i],k], 'k-', label='data')
    ax1.plot(blade.time[ii], blade.data[ii,k], 'ro', label='peaks')
    ax1.plot(blade.time[ii], fit_i[i], 'b', label='exp fit')
    ax1.set_xlim([blade.time[i_bl[i]].min(),blade.time[i_bl[i]].max()])
    ax1.legend(loc='best', ncol=3)
    ax1.grid()
    ax1.set_title('Derived damping from blade vibration decay tests')
    ax1.set_ylabel('blade strain sensor output [-]')
    ax1.set_xlabel('time [s]')
    ax1.set_ylim([-100, 300])
    pa4.save_fig()
    # -------------------------------------------------------------------------


###############################################################################
### TOWER FREQ
###############################################################################

def plot_damping_tower():
    """
    Take a random virbation test and make some plots for the thesis.
    This to show that the accelerometer was dirty and not very useful

    CAUTION: switch following check in misc.damping:
    'Eigenfreq: PSD max is not one of the first 5 freq. peaks'
    > because highest peak is on the 50 Hz, which is not in the beginning
    """

    respath = '/home/dave/PhD_data/OJF_data_edit/04/vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfreq_damp_tower/'

    freqpath  = '/home/dave/PhD/Projects/PostProcessing/'
    freqpath += 'OJF_tests/eigenfreq_damp_tower/'

    # TOWER VIBRATIONS
    channels = []
    channels.append('Tower Strain Side-Side filtered')
    channels.append('Tower Strain For-Aft filtered')
    # in order to get the acc meter to pass the analysis, misc.damping has to
    # disable the checking on the eigenfrequency:
    # 'Eigenfreq: PSD max is not one of the first 5 freq. peaks'
    # further, the damping will be rubbish
    channels.append('Tower Top acc Y (FA)')
    channels.append('Tower Top acc X (SS)')

    # there is only one case for tower vibrations
    resfile = '0405_run_253_towervibrations'
    #ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
    res = ojfresult.ComboResults(respath, resfile, sinc=False)
    res._calibrate_dspace(ojfresult.CalibrationData.caldict_dspace_04)
    ds = res.dspace

    # example input data ----------------------------------------------------
    checkplot = False
    i = 2 # block number
    k = ds.labels_ch['Tower Top acc Y (FA)'] # channel number
    # ------------------------------------------------------------------------

    # get the peaks, damping, exp fit for all blocks in the file
    i_p, i_bl, damp_bl, fit_i, fn5, psd = misc.damping(ds.time,
                        ds.data[:,k], checkplot=checkplot, verbose=True)
    # indices to peaks of one cycle block
    ii = i_p[i]
    # the peaks for this block
    peaks = ds.data[ii,k]

    # center data around a zero mean for more convinient processing
    data = ds.data[i_bl[i],k] - ds.data[i_bl[i],k].mean()
    # also normalize the data so it is more general applicable
    data *= 1.0/(data.max() - data.min())

#    # eigenfrequency as saved,
#    freqfile = 'eigenfreq_' + freqfile
#    fn = np.loadtxt(freqpath + freqfile)

    # and now we can get the damping, take the first eigenfrequency
    zeta_fit = damp_bl[i]
    zeta_all = np.log(peaks[0]/peaks[-1])/len(peaks)
    zeta_rand = np.log(peaks[10]/peaks[30])/len(peaks[10:31])

    print '%i - %i' % (k,i)
#    print 'fn, saved: %4.1f' % (float(fn[0]))
    print ' zeta_fit: %8.4f' % -zeta_fit
    print ' zeta_all: %8.4f' % zeta_all
    print 'zeta_rand: %8.4f' % zeta_rand

    # -------------------------------------------------------------------------
    # just plot the first found vibration test
    figsize_x = plotting.TexTemplate.pagewidth*0.99
    figsize_y = plotting.TexTemplate.pagewidth*0.35
    figname = resfile + '_chi_%i_block_%i' % (k, i)
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                   wstop_cm=0.8, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(ds.time[i_bl[i]], data)
    ax1.grid()
    ax1.set_title('Tower vibration decay test')
#    ax1.set_ylabel('blade strain sensor output [-]')
    ax1.set_xlabel('time [s]')
    ax1.set_xlim([ds.time[i_bl[i]].min(),ds.time[i_bl[i]].max()])
    pa4.save_fig()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # frequency domain plot:
    figsize_x = plotting.TexTemplate.pagewidth*0.99
    figsize_y = plotting.TexTemplate.pagewidth*0.35
    figname = resfile + '_chi_%i_block_%i_psd' % (k, i)
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                   wstop_cm=0.8, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    Pxx, freqs = psd[i]
    ax1.plot(freqs, Pxx, 'k')
    ax1.set_yscale('log')
    ypos = [1e-5, 1e3, 1e-5, 1e3, 1e-5, 1e3]
    # indicate the top 5 frequencies
    for ifn, fn in enumerate(sorted(fn5)):
        ax1.axvline(x=fn)
        y = ypos[ifn]
        textbox = r'$f_%i = %1.1f$' % (ifn+1, fn)
        bbox = dict(boxstyle="round", alpha=0.8, edgecolor=(1., 0.5, 0.5),
                    facecolor=(1., 0.8, 0.8),)
        ax1.text(fn, y, textbox, fontsize=12, verticalalignment='bottom',
                 horizontalalignment='center', bbox=bbox)

    ax1.grid()
    ax1.set_title('Tower vibration decay test frequency domain')
#    ax1.set_ylabel('blade strain')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_xlim([0, 250])
    pa4.save_fig()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # peak plot
    figsize_x = plotting.TexTemplate.pagewidth*0.99
    figsize_y = plotting.TexTemplate.pagewidth*0.35
    figname = resfile + '_chi_%i_block_%i_peaks_dampfit' % (k, i)
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                   wstop_cm=0.8, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(ds.time[i_bl[i]], ds.data[i_bl[i],k], 'k-', label='data')
    ax1.plot(ds.time[ii], ds.data[ii,k], 'ro', label='peaks')
    ax1.plot(ds.time[ii], fit_i[i], 'b', label='exp fit')
    ax1.set_xlim([ds.time[i_bl[i]].min(),ds.time[i_bl[i]].max()])
    ax1.legend(loc='best', ncol=3)
    ax1.grid()
    ax1.set_title('Derived damping from tower vibration decay tests')
    ax1.set_ylabel('blade strain sensor output [-]')
    ax1.set_xlabel('time [s]')
#    ax1.set_ylim([-100, 300])
    pa4.save_fig()
    # -------------------------------------------------------------------------

def plot_psd_tower():
    """
    Same methods used as for the blade

    We only have one result file for the tower vibrations. The file has been
    evaluated before in eigenfreq_april
    """

    respath = '/home/dave/PhD_data/OJF_data_edit/04/vibration/'
    figpath = '/home/dave/PhD/Projects/PostProcessing/'
    figpath += 'OJF_tests/eigenfreq_damp_tower/'

    freqpath  = '/home/dave/PhD/Projects/PostProcessing/'
    freqpath += 'OJF_tests/eigenfreq_damp_tower/'

    # TOWER VIBRATIONS
    channels = ['Tower Strain Side-Side filtered',
                'Tower Strain For-Aft filtered']
    # in order to get the acc meter to pass the analysis, misc.damping has to
    # disable the checking on the eigenfrequency:
    # 'Eigenfreq: PSD max is not one of the first 5 freq. peaks'
    # further, the damping will be rubbish
#    channels.append('Tower Top acc Y (FA)')
#    channels.append('Tower Top acc X (SS)')

    # there is only one case for tower vibrations
    resfile = '0405_run_253_towervibrations'
    #ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
    res = ojfresult.ComboResults(respath, resfile, sinc=False)
    res._calibrate_dspace(ojfresult.CalibrationData.caldict_dspace_04)
    ds = res.dspace

    #eigenfreqs = ds.psd_plots(figpath,channels,nnfts=[2048],saveresults=True)

    tower_data = {}
    case_peaks = {}
    nrpeaks = 10
    thresh_occ = 0.5
    NNFT = 4096
    damp_all = {}

    for ch in channels:
        chi = ds.labels_ch[ch]

        print
        print 79*'-'
        print ch
        print 79*'-'

        # load the actual measured eigenfrequency
        if ch.find('SS') > -1 or ch.find('Side-Side') > -1:
            freqfile = 'eigenfreq_tower_ss'
            i0, i1 = 68*ds.sample_rate, -1
        elif ch.find('FA') > -1 or ch.find('For-Aft') > -1:
            freqfile = 'eigenfreq_tower_fa'
            i0, i1 = 0, 68*ds.sample_rate
        fn = np.loadtxt(freqpath+freqfile)

        i_p, i_bl, damp_bl, fit_i, fn5, psd_bl \
                = misc.damping(ds.time[i0:i1], ds.data[i0:i1,chi],
                               checkplot=True, offset_start=100,
                               offset_end=200, verbose=True, NFFT=NNFT)

        # keep copy of all the results
        damp_all[freqfile.replace('eigenfreq', 'damp')] = np.array(damp_bl)

        print 'fn: %1.2f, fn1: %1.2f' % (float(fn[0]), float(fn5[0]))

        # and the PSD from each block
        for Pxx, freqs in psd_bl:
            try:
                tower_data[ch].append([Pxx, freqs])
            except KeyError:
                tower_data[ch] = [[Pxx, freqs]]

            # peak detection: only if strong peak occurs often
            # limit the Pxx search to up to 100 Hz for the tower
            ilim = np.abs(freqs-100.0).argmin()
            # keep track of all the peaks, we want to select strong peaks
            # that occur often. In that way filter out peaks that only
            # occur once
            Pxx_log = 10.*np.log10(Pxx[:ilim])
            # find all the peaks on the PSD
            ifns = wafo.misc.findpeaks(Pxx_log, n=len(Pxx_log), min_h=0)
            # and only consider the nrpeaks strongest peaks, but convert the
            # frequency occurance to an int, *10 to not lose accuracy
            rounded_freqs = np.round(10.0*freqs[:ilim][ifns[:nrpeaks]])
            appendix = np.array(rounded_freqs, dtype=np.int32)
            try:
                case_peaks[ch] = np.append(case_peaks[ch], appendix)
            except KeyError:
                case_peaks[ch] = appendix

        print 'damping mean: %1.2e' % np.array(damp_bl).mean()

    # save all the damp results
    for key, value in damp_all.iteritems():
        np.savetxt(freqpath+key+'blocks', value)
        np.savetxt(freqpath+key, np.array([value.mean()]))

    # and now plot all the different measurements in one plot per blade
    for towercase in tower_data:

        nr_tests = len(tower_data[towercase])
        replace = (nr_tests, towercase.replace(' filtered', ''))
        title = 'Power spectra of %i decay tests, %s' % replace

        figsize_x = plotting.TexTemplate.pagewidth*0.99
        figsize_y = plotting.TexTemplate.pagewidth*0.30
        figname = towercase.replace(' ', '_') + '_psd_all'
        scale = 1.8
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                       grandtitle=False, wsleft_cm=1.7, wsright_cm=0.7,
                       wstop_cm=0.8, wsbottom_cm=1.0,
                       figsize_x=figsize_x, figsize_y=figsize_y)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.set_yscale('log')
#        ax1.set_xscale('log')
#        ax1.set_xlim([1e0, 1e2])
        ax1.grid()
        ax1.set_title(title)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_xlim([0, 60])
        ax1.set_ylim([1e-11, 1e-1])
        # make sure to have enough ticks
        locator = mpl.ticker.LinearLocator(numticks=13, presets=None)
        ax1.xaxis.set_major_locator( locator )

        for Pxx, freqs in tower_data[towercase]:
            ax1.plot(freqs, Pxx)

        Pxx, freqs = tower_data[towercase][1]
        # only consider peaks that occur in at least 2/3 of the blocks
        count = np.bincount(case_peaks[towercase])
        nrblocks = len(tower_data[towercase])
        # and divide the sum with the number of blocks to get the mean
        threshold = int(round(nrblocks*thresh_occ))
        freq_sort = count.argsort()
        freq_occ = np.sort(count)
        # select those that only occure more than the threshold
        iselect = freq_occ.__ge__(threshold)
        # and we have a number of occurances, devide by 10 to get back to
        # original frequency
        freq_select = freq_sort[iselect]/10.0
        # find the corresponding Pxx values, based on the average Pxx
        #xpos = np.round(20.0*np.random.random(20)) + 10
        #ypos = np.round(50.0*(np.random.random(20)-0.5))
        # but that doesn't realy have nice results...
        xpos = [ 1,  1,-10,-10,  1, 10,  1,  1,  1,  1,  1]
        ypos = [10, 25, 25, 40, 25, 15, 25, 25, 25, 25, 25]
        for i, fn in enumerate(sorted(freq_select)):
            iPxx = np.abs(freqs-fn).argmin()
            # and labels for only one PSD
            ax1.annotate(r'$%1.1f$' % fn,
                xy=(fn, Pxx[iPxx]), xycoords='data',
                xytext=(xpos[i], ypos[i]), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"), weight='bold', fontsize=12)

        pa4.save_fig()


###############################################################################
### SOUND
###############################################################################

def sound_measurements():
    """
    check the sampling rates for the sound measurements!

    It seems we can't see anything, the PSD down't show anything ...
    """

    path_db = '/home/dave/PhD_data/OJF_data_edit/database/'
    db = ojf_db('steady_nocal', debug=False, path_db=path_db)
    # the below is much faster than str(num)!
    runs_inc = [`num` for num in xrange(335, 356)]
#    runs_inc += [`num` for num in xrange(359, 420)]
    data, cases, headers = db.select(['04'], [], [], runs_inc=runs_inc)

#    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/sound/'


    for case in cases:
        res = ojfresult.ComboResults(path_db+'symlinks/', case, silent=True)
#        isound = res.dspace.labels_ch['Sound']
        isoundg = res.dspace.labels_ch['Sound_gain']
        print res.dspace.sample_rate, case
        # do we need to account for an offset?
#        off = res.dspace.data[:,isound] - res.dspace.data[:,isound].mean()
#        sound = np.array([res.dspace.time, res.dspace.data[:,isound]] )
        soundg = np.array([res.dspace.time, res.dspace.data[:,isoundg]] )
#        sound_off = np.array([res.dspace.time, off] )

        plt.figure()
        plt.plot(res.dspace.time, res.dspace.data[:,isoundg])

        # play the stuff
        #play(sound, fs=1000)

        # plot PSD's, can we see anything at all?
        sampling = 1.0/(res.dspace.SamplingPeriod*res.dspace.Downsampling)
        #nn = [16384, 8192, 4096, 2048]
        plt.figure()
        Pxx, freqs = plt.psd(soundg, NFFT=8192, Fs=sampling)
#        pa4 = plotting.A4Tuned(scale=scale)
#        pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
#                       grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
#                       wstop_cm=1.0, wsbottom_cm=1.0,
#                       figsize_x=figsize_x, figsize_y=figsize_y)
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)


def sound_measurements_sweep():
    """
    What was the sample rate during the sweep series?
    """

#    source_folder = '/home/dave/PhD_data/OJF_data_edit/dc_sweep/'
    sf = '/home/dave/PhD_data/OJF_data_edit/dc_sweep/2012-04-12_stiff/'
    sf = '/home/dave/PhD_data/OJF_data_edit/dc_sweep/'
#    fname = 'full.mat'

    sf = '/home/dave/PhD_data/OJF_data_edit/dc_sweep/2012-04-13_stiff/'
    for root, dirs, files in os.walk(sf, topdown=True):
        print dirs
        for fname in files:
            if not fname.find('384') > -1:
                continue
            #dspace = ojfresult.DspaceMatFile(sf+fname, silent=False)
            #print dspace.sample_rate, fname
            fname = fname.replace('.mat', '')
            res = ojfresult.ComboResults(sf, fname, silent=True)
            try:
                isoundg = res.dspace.labels_ch['Sound_gain']
            except:
                continue
            print res.dspace.sample_rate, fname
            plt.figure()
            plt.plot(res.dspace.time, res.dspace.data[:,isoundg])



###############################################################################
### REPORT PLOTS
###############################################################################

def ojf_postproc_01():
    """
    Plots generated for the first preliminary report of the OJF tests in
    Februari. Goal is to see the measurements make any sense before going into
    the wind tunnel again in April.
    """

    calpath='/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath='/home/dave/PhD/Projects/PostProcessing/OJF_tests/dashboard-v2/'

    ycp = calpath + 'YawLaserCalibration/' + 'runs_050_051.yawcal-pol10'
    tfacp = calpath + 'TowerStrainCal/' + 'towercal-pol1_fa'
    tsscp = calpath + 'TowerStrainCal/' + 'towercal-pol1_ss'

    caldict_dspace = dict()
    caldict_dspace['Yaw Laser'] = ycp
    caldict_dspace['Tower Strain For-Aft'] = tfacp
    caldict_dspace['Tower Strain Side-Side'] = tsscp

    # -------------------------------------------------------------
    # Complete cycle: plot calibrated data for one case completely
    # -------------------------------------------------------------

    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-14/'
    resfile = '0214_run_158_7.0ms_dc0_flexies_fixyaw_pwm1000_highrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res.dashboard_a3(figpath, nr_rev=5)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath, nr_rev=5)

    resfile = '0214_run_149_10ms_dc1_flexies_fixyaw_pwm1000_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res.dashboard_a3(figpath, nr_rev=5)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath, nr_rev=5)

    # no ojf log file
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-09/'
    resfile = '0209_run_011_10ms_dc0.0_stiffblades_pwm10000'
    res = ojfresult.ComboResults(respath, resfile)
    res.dashboard_a3(figpath, nr_rev=5)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath, nr_rev=5)

    # stiff, coning
    resfile = '0212_run_077_8.0ms_dc0_fixyaw_spinuptippingpoint_stiffblades'
    resfile += '_coning_pwm1000_low-to-highrpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-12/'
    res = ojfresult.ComboResults(respath, resfile)
    res.dashboard_a3(figpath)
    res.dashboard_a3(figpath, time=[50,51])
    res.dashboard_a3(figpath, time=[0,3.5])
    res.dashboard_a3(figpath, time=[0,1.0])
    # only do the calirbation once!!
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.dashboard_a3(figpath, time=[50,51])
    res.dashboard_a3(figpath, time=[0,3.5])
    res.dashboard_a3(figpath, time=[0,1.0])

    # stiff, coning
    resfile = '0211_run_053_speeduptil_8ms_450_rpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-11/'
    res = ojfresult.ComboResults(respath, resfile)
    res.dashboard_a3(figpath, time=[35,45])
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath, time=[35,45])

    # flex, no coning, free yaw
    resfile = '0214_run_166_7.0ms_dc0_flexies_freeyawplaying_pwm1000_highrpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-14/'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.dashboard_a3(figpath, time=[25, 35])

    # flex, no coning, free yaw
    resfile = '0214_run_170_8.0ms_dc1_flexies_freeyawplaying_pwm1000_highrpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-14/'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.dashboard_a3(figpath, time=[40, 50])

#    # stiff, no blade signal
#    resfile = '0210_run_022_9ms_dc0_freeyaw_init_+ext_stiffblades_pwm1000'
#    resfile += '_accedental-speedup'

def torque2012_old():
    """
    This work lead to the torque2012_abstract selection
    """

    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath = calpath + 'Torque2012/abstract/'

    ycp = calpath + 'YawLaserCalibration/' + 'runs_050_051.yawcal-pol10'
    tfacp = calpath + 'TowerStrainCal/' + 'towercal-pol1_fa'
    tsscp = calpath + 'TowerStrainCal/' + 'towercal-pol1_ss'

    caldict_dspace = dict()
    caldict_dspace['Yaw Laser'] = ycp
    caldict_dspace['Tower Strain For-Aft'] = tfacp
    caldict_dspace['Tower Strain Side-Side'] = tsscp

    # -------------------------------------------------------------
    # Free yawing cases: FLEX NO CONING
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # Free yawing cases: STIFF NO CONING
    # -------------------------------------------------------------

    resfile = '0210_run_027_7.0ms_dc0_freeyawplaying_stiffblades_pwm1000'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-10/'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0210_run_029_6.0ms_dc0_freeyawplaying_stiffblades_pwm1000'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

#    resfile = '0210_run_017_8ms_dc0_freeyaw_init_-ext_stiffblades_pwm1000'
#    res = ojfresult.ComboResults(respath, resfile)
#    res.dashboard_a3(figpath)
#
#    resfile = '0210_run_019_8ms_dc0_freeyaw_init_+ext_stiffblades_pwm1000'
#    resfile += '_2ndattempt'
#    res = ojfresult.ComboResults(respath, resfile)
#    res.dashboard_a3(figpath)

    resfile = '0210_run_013_7ms_dc0_freeyaw_init_0_stiffblades_pwm1000'
    resfile += '_initsteadystate'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0210_run_014_7ms_dc0_freeyaw_init_-ext_stiffblades_pwm1000'
    resfile += '_initsteadystate'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0210_run_015_7ms_dc0_freeyaw_init_+ext_stiffblades_pwm1000'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    # -------------------------------------------------------------
    # Free yawing cases: STIFF WITH CONING
    # -------------------------------------------------------------

    resfile = '0212_run_068_5.5ms_dc0_freeyawplaying_stiffblades_coning'
    resfile += '_pwm1000_highrpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-12/'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0212_run_069_6.0ms_dc0_freeyawplaying_stiffblades_coning'
    resfile += '_pwm1000_highrpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-12/'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0212_run_071_7.0ms_dc0_freeyawplaying_stiffblades_coning'
    resfile += '_pwm1000_highrpm'
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-12/'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0212_run_056_7ms_dc0_freeyaw_init_0_stiffblades_coning_pwm1000'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0212_run_057_7ms_dc0_freeyaw_init_-ext_stiffblades_coning_pwm1000'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0212_run_058_7ms_dc0_freeyaw_init_+ext_stiffblades_coning_pwm1000'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    # -------------------------------------------------------------
    # Free yawing cases: SWEPT
    # -------------------------------------------------------------
    respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-13/'

    resfile = '0213_run_130_7.0ms_dc0_samoerai_freeyawplaying_pwm1000_highrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)

    resfile = '0213_run_128_7.0ms_dc0_samoerai_freeyawplaying_-ext-release'
    resfile += '_pwm1000_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath)
    res.freeyaw(figpath)


    # -------------------------------------------------------------
    # DIFFERENT STUFF
    # -------------------------------------------------------------

    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-04/0404_data/'
    resfile = '0404_run_222_9.5ms_dc0_flexies_fixyaw_speedup_200rpm'
    res = ojfresult.ComboResults(respath, resfile)
    res.dashboard_a3(figpath)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
#    res.dashboard_a3(figpath, time=[0, 10])
    res.dashboard_a3(figpath, time=[0, 1])
#    res.freeyaw(figpath)

    resfile = '0404_run_230_7.0ms_dc0.2_flexies_fixyaw_highrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.dashb_ts(figpath, time=[1, 1.6])

    #channel = kwargs.get('channel', 0)
    #figpath = kwargs.get('figpath', '')
    #figfile = self.matfile.split('/')[-1] + '_ch' + str(channel)
    #
    #plot = plotting.A4Tuned()
    #plot.plot_simple(figpath+figfile, self.time, self.data, self.labels,
                     #channels=[channel], grandtitle=figfile,
                     #figsize_y=10)

    resfile = '0404_run_204_7.0ms_dc0_flexies_fixyaw_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.dashb_ts(figpath, nr_rev=5)

def torque2012_abstract():
    """
    Exploratory plots for used for the Torque 2012 paper abstract
    """
    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath = calpath + 'Torque2012/abstract/'

    # -------------------------------------------------------------
    # APRIL YAW CALIBRATION
    # -------------------------------------------------------------
    ycp = calpath + 'YawLaserCalibration-04/' + 'runs_289_295.yawcal-pol10'
    caldict_dspace = dict()
    caldict_dspace['Yaw Laser'] = ycp

    # -------------------------------------------------------------
    # Free yawing cases: FLEXIES APRIL, LOW RPM, lowside
    # -------------------------------------------------------------

    #respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-10/0410_data/'
    #resfile = '0410_run_296_7ms_dc0_flexies_freeyaw_lowrpm'
    #res = ojfresult.ComboResults(respath, resfile)
    #res._sync_strain_dspace(min_h=0.3, checkplot=False)
    #res._calibrate_dspace(caldict_dspace, rem_wind=True)
    #res.dashboard_a3(figpath)
    #res.freeyaw_compact(figpath,'free yawing, 7 m/s, slow RPM and deep stall',
                        #time=[0, 60], )

    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-13/0413_data/'
    resfile = '0413_run_404_7ms_dc0_stiffblades_freeyaw_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._sync_strain_dspace(min_h=0.3, checkplot=False)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.freeyaw_compact(figpath, 'free yawing, 8 m/s,slow RPM and deep stall',
                        time=[0,60])

    # -------------------------------------------------------------
    # Free yawing cases: FLEXIES APRIL, LOW RPM, fastside
    # -------------------------------------------------------------

    #respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-10/0410_data/'
    #resfile = '0410_run_297_7ms_dc0_flexies_freeyaw_lowrpm'
    #res = ojfresult.ComboResults(respath, resfile)
    #res._sync_strain_dspace(min_h=0.3, checkplot=False)
    #res._calibrate_dspace(caldict_dspace, rem_wind=True)
    #res.dashboard_a3(figpath)
    #res.freeyaw_compact(figpath, 'free yawing, 7 m/s,slow RPM and deep stall',
                        #time=[0,60])

    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-13/0413_data/'
    resfile = '0413_run_403_7ms_dc0_stiffblades_freeyaw_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._sync_strain_dspace(min_h=0.3, checkplot=False)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.freeyaw_compact(figpath, 'free yawing, 8 m/s, slow RPM in deep stall',
                        time=[0,60])

    # -------------------------------------------------------------
    # Free yawing cases: FLEXIES APRIL, HIGH RPM
    # -------------------------------------------------------------

    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-05/0405_data/'
    resfile = '0405_run_266_8.0ms_dc0.4_flexies_freeyaw_highrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._sync_strain_dspace(min_h=0.3, checkplot=False)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    res.dashboard_a3(figpath)
    res.freeyaw_compact(figpath, 'free yawing, 8 m/s, high RPM', time=[0, 60])

    # -------------------------------------------------------------
    # Fix yawing cases: FLEXIES APRIL, compare TOWER SHADOW LOW RPM
    # -------------------------------------------------------------
    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-04/0404_data/'

    resfile = '0404_run_212_9.0ms_dc0.6_flexies_fixyaw_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._sync_strain_dspace(min_h=0.3, checkplot=False)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    #res.dashboard_a3(figpath)
    res.dashb_ts(figpath, 'fixed yaw, 9 m/s, low RPM', nr_rev=5)
    # and check the PSD analysis on the blade strain signal
    eigenfreqs = res.blade.psd_plots(figpath, [0,1,2,3], nnfts=[2048])
    print eigenfreqs

    # -------------------------------------------------------------
    # Fix yawing cases: FLEXIES APRIL, compare TOWER SHADOW HIGH RPM
    # -------------------------------------------------------------

    resfile = '0404_run_223_9.0ms_dc1_flexies_fixyaw_highrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res._sync_strain_dspace(min_h=0.3, checkplot=False)
    res._calibrate_dspace(caldict_dspace, rem_wind=True)
    #res.dashboard_a3(figpath)
    res.dashb_ts(figpath, 'fixed yaw, 9 m/s high RPM', time=[1, 1.6])
    # and check the PSD analysis on the blade strain signal
    eigenfreqs = res.blade.psd_plots(figpath, [0,1,2,3], nnfts=[4096])
    print eigenfreqs


def presentation_risoe_june():
    """
    Some additional plots for the big OJF presentation in june at Risø
    """

    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    ycp02 = calpath + 'YawLaserCalibration/' + 'runs_050_051.yawcal-pol10'
    ycp04 = calpath + 'YawLaserCalibration-04/' + 'runs_289_295.yawcal-pol10'
#    tfacp = calpath + 'TowerStrainCal/' + 'towercal-pol1_fa'
#    tsscp = calpath + 'TowerStrainCal/' + 'towercal-pol1_ss'

    tfacp = calpath + 'TowerStrainCal-04/' + 'towercal_249_250_251.cal-pol1'
    tsscp = calpath + 'TowerStrainCal-04/' + 'towercal_249_250_251.cal-pol1'

    caldict_dspace_02 = dict()
    caldict_dspace_02['Yaw Laser'] = ycp02

    caldict_dspace_04 = dict()
    caldict_dspace_04['Yaw Laser'] = ycp04
    caldict_dspace_04['Tower Strain For-Aft'] = tfacp
    caldict_dspace_04['Tower Strain Side-Side'] = tsscp

    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'sync_dspace_bladestrain/'
    # illustrate the drifting of dSPACE and MicroStrain clocks, high rpm
    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-04/0404_data/'
    resfile = '0404_run_223_9.0ms_dc1_flexies_fixyaw_highrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res.overlap_pulse(figpath)
    res._calibrate_dspace(caldict_dspace_04, rem_wind=True)
    res.dashboard_a3(figpath)
    # illustrate the drifting of dSPACE and MicroStrain clocks, low rpm
    resfile = '0404_run_212_9.0ms_dc0.6_flexies_fixyaw_lowrpm'
    res = ojfresult.ComboResults(respath, resfile)
    res.overlap_pulse(figpath)
    res._calibrate_dspace(caldict_dspace_04, rem_wind=True)
    res.dashboard_a3(figpath)


    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'spinup/'
    # illustrate yawing and blade loads
    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-05/0405_data/'
    resfile = '0405_run_270_9.0ms_dc0_flexies_freeyaw_spinupyawerror'
    res = ojfresult.ComboResults(respath, resfile)
    res._calibrate_dspace(caldict_dspace_04, rem_wind=True)
    res.dashb_blade_yaw(figpath, 'free yaw, 9 m/s speedup', time=[30, 60])
    res.dashb_blade_yaw(figpath, 'free yaw, 9 m/s speedup', time=[0, 30])


def speedup_cases():
    """
    """
    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'spinup/'

    # -------------------------------------------------------------------------
    #calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    #ycp02 = calpath + 'YawLaserCalibration/' + 'runs_050_051.yawcal-pol10'
    #tfacp02 = calpath + 'TowerStrainCal/' + 'towercal-pol1_fa'
    #tsscp02 = calpath + 'TowerStrainCal/' + 'towercal-pol1_ss'
    #caldict_dspace_02 = dict()
    #caldict_dspace_02['Yaw Laser'] = ycp02
    #caldict_dspace_02['Tower Strain For-Aft'] = tfacp02
    #caldict_dspace_02['Tower Strain Side-Side'] = tsscp02
    #
    ## THIS IS A CONED ROTOR!!
    #respath = '/home/dave/PhD_data/OJF_data_edit/02/2012-02-13/'
    #resfile='0213_run_144_9ms_dc0_flexies_fixyaw_spinup_pwm1000_low-to-highrpm'
    #res = ojfresult.ComboResults(respath, resfile)
    #res._calibrate_dspace(caldict_dspace_02, rem_wind=True)
    #res.dashb_blade_yaw(figpath, 'fix yaw, coning stiff, 8-9 m/s spinup')

    # -------------------------------------------------------------------------
#    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
#    ycp04 = calpath + 'YawLaserCalibration-04/' + 'runs_289_295.yawcal-pol10'
#    tfacp04 = calpath + 'TowerStrainCal-04/' + 'towercal_249_250_251.cal-pol1'
#    tsscp04 = calpath + 'TowerStrainCal-04/' + 'towercal_249_250_251.cal-pol1'
#    caldict_dspace_04 = dict()
#    caldict_dspace_04['Yaw Laser'] = ycp04
#    caldict_dspace_04['Tower Strain For-Aft'] = tfacp04
#    caldict_dspace_04['Tower Strain Side-Side'] = tsscp04

    # -------------------------------------------------------------------------
    figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    figpath += 'spinup/'
    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-12/0412_data/'
    resfile = '0412_run_372_10ms_stiff_fixyaw_speedup'
    res = ojfresult.ComboResults(respath, resfile, sync=True)
    res._calibrate_dspace(ojfresult.CalibrationData.caldict_dspace_04)
    res._calibrate_blade(ojfresult.CalibrationData.caldict_blade_04)
    res.dashb_blade_yaw(figpath, 'fix yaw stiff blade, 10 ms spinup',
                        time=[15, 40])

    # -------------------------------------------------------------------------
#    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-13/0413_data/'
#    resf='0413_run_408_0-10ms_dc0_stiffblades_freeyaw_lowrpm_startup_fastside'
#    res = ojfresult.ComboResults(respath, resf, sync=True)
#    res._calibrate_dspace(ojfresult.CalibrationData.caldict_dspace_04)
#    res.dashb_blade_yaw(figpath, 'free yaw stiff blade, 10 ms spinup')

    # -------------------------------------------------------------------------
#    # illustrate yawing and blade loads
#    respath = '/home/dave/PhD_data/OJF_data_edit/04/2012-04-05/0405_data/'
#    resfile = '0405_run_270_9.0ms_dc0_flexies_freeyaw_spinupyawerror'
#    res = ojfresult.ComboResults(respath, resfile, sync=True)
#    res._calibrate_dspace(ojfresult.CalibrationData.caldict_dspace_04)
#    res.dashb_blade_yaw(figpath, 'free yaw, 9 m/s speedup')
#    res.dashb_blade_yaw(figpath, 'free yaw, 9 m/s speedup', time=[30, 60])
#    res.dashb_blade_yaw(figpath, 'free yaw, 9 m/s speedup', time=[0, 30])

if __name__ == '__main__':

    dummy = None

#    speedup_cases()

    # -------------------------------------------------------------
    # Create proper model for the OJF blade
    # -------------------------------------------------------------

    #blade_cg()
    #structural_prop()
    #compare_eigenfrq_ranges()
    #eigenfreq_februari()
    #eigenfreq_april()
    #eigenfreq_august()

    # -------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------
#    april_calibration()
#    february_calibration()
#    BladeCalibration().compare_cal_feb_april()
#    BladeCalibration().thesis_plot_blade_strain()

    # -------------------------------------------------------------
    # BLADE STUFF
    # -------------------------------------------------------------
#    blade_aero_coeff()
    #blade_contour_all_raw()
#    blade_contour()
#    blade_extension_drag()
#    blade_extension_3()
#    blade_damping_all()
#    plot_damping_blade()
#    plot_psd_blades()
    #save_contour_cg_mass()
    #blade_contour_corr()
#    plot_sync_blade_strain_dspace()
#    plot_psd_tower()
#    plot_damping_tower()

    # -------------------------------------------------------------
    # Free yawing cases for Torque 2012 abstract
    # -------------------------------------------------------------

    #presentation_risoe_june()
    #presentation_3e_sept()
    #torque2012_abstract()

    # -------------------------------------------------------------
    # Complete cycle: plot calibrated data for one case completely
    # -------------------------------------------------------------

#    ojf_postproc_01()
#    make_cal_files()

    # -------------------------------------------------------------
    # Other stuff
    # -------------------------------------------------------------

#    sympy.mpmath.mp.dps = 3
#    t = timeit.Timer(stmt=ff)
#    #print t.timeit(number=20)
#    print "%.2f usec/pass" % (20 * t.timeit(number=20)/20)
