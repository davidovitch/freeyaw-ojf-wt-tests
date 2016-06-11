# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:03:47 2012

@author: dave
"""

from __future__ import division
import math
import os
import sys
#import timeit
import logging
import pickle

import numpy as np
import scipy.io
#from scipy.interpolate import UnivariateSpline
#import scipy.integrate as integrate
#import scipy.interpolate as sp_int
#import scipy.constants as spc
import scipy.interpolate as interpolate
import scipy as sp
from scipy.cluster import vq
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas
#import matplotlib.font_manager as mpl_font
#from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
#from matplotlib import tight_layout as tight_layout
#from matplotlib.ticker import FormatStrFormatter
import pylab as plt
import cv2
import pandas as pd

import progressbar as progbar

import wafo

import plotting
import misc
from staircase import StairCase

def calc_sample_rate(time, rel_error=1e-4):
    """
    the sample rate should be constant throughout the measurement serie
    define the maximum allowable relative error on the local sample rate

    rel_error = 1e-4 # 0.0001 = 0.01%
    """
    deltas = np.diff(time)
    # the sample rate should be constant throughout the measurement serie
    # define the maximum allowable relative error on the local sample rate
    if not (deltas.max() - deltas.min())/deltas.max() <  rel_error:
        print 'Sample rate not constant, max, min values:',
        print '%1.6f, %1.6f' % (1/deltas.max(), 1/deltas.min())
        raise AssertionError
    return 1/deltas.mean()

class CalibrationData:
    """
    This should be the final and correct calibration data
    """

    # ---------------------------------------------------------------------
    # definition of the calibration files for February
    calpath = 'data/calibration/'
    ycp = os.path.join(calpath, 'YawLaserCalibration/runs_050_051.yawcal-pol10')
    caldict_dspace_02 = {}
    caldict_dspace_02['Yaw Laser'] = ycp
    # do not calibrate tower strain in February, results are not reliable
    #tfacp = calpath + 'TowerStrainCal/towercal-pol1_fa'
    #tsscp = calpath + 'TowerStrainCal/towercal-pol1_ss'
    #caldict_dspace_02['Tower Strain For-Aft'] = tfacp
    #caldict_dspace_02['Tower Strain Side-Side'] = tsscp

    caldict_blade_02 = {}
    bcp = os.path.join(calpath, 'BladeStrainCal/')
    caldict_blade_02[0] = bcp + '0214_run_172_ch1_0214_run_173_ch1.pol1'
    caldict_blade_02[1] = bcp + '0214_run_172_ch2_0214_run_173_ch2.pol1'
    caldict_blade_02[2] = bcp + '0214_run_172_ch3_0214_run_173_ch3.pol1'
    caldict_blade_02[3] = bcp + '0214_run_172_ch4_0214_run_173_ch4.pol1'
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    tw_cal = 'opt'
    # definition of the calibration files for April
    ycp04 = os.path.join(calpath,
                         'YawLaserCalibration-04/runs_289_295.yawcal-pol10')
    # for the tower calibration, the yaw misalignment is already taken
    # into account in the calibration polynomial, no need to include the
    # yaw angle in the calibration. We always measure in the FA,SS dirs
    # if that needs to be converted, than do sin/cos psi to have the
    # components aligned with the wind
    tfacp  = calpath + 'TowerStrainCal-04/'
    tfacp += 'towercal_249_250_251_yawcorrect_fa-cal_pol1_%s' % tw_cal
    tsscp  = calpath + 'TowerStrainCal-04/'
    tsscp += 'towercal_249_250_251_yawcorrect_ss-cal_pol1_%s' % tw_cal
    caldict_dspace_04 = {}
    caldict_dspace_04['Yaw Laser'] = ycp04
    caldict_dspace_04['Tower Strain For-Aft'] = tfacp
    caldict_dspace_04['Tower Strain Side-Side'] = tsscp
    # and to convert to yaw coordinate frame of reference
    target_fa = calpath + 'TowerStrainCalYaw/psi_fa_max_%s' % tw_cal
    caldict_dspace_04['psi_fa_max'] = target_fa
    target_ss = calpath + 'TowerStrainCalYaw/psi_ss_0_%s' % tw_cal
    caldict_dspace_04['psi_ss_0'] = target_ss

    caldict_blade_04 = {}
    caldict_blade_04[0] = bcp + '0412_run_357_ch1_0412_run_358_ch1.pol1'
    caldict_blade_04[1] = bcp + '0412_run_357_ch2_0412_run_358_ch2.pol1'
    caldict_blade_04[2] = bcp + '0412_run_356_ch3_0412_run_358_ch3.pol1'
    caldict_blade_04[3] = bcp + '0412_run_356_ch4_0412_run_358_ch4.pol1'
    # ---------------------------------------------------------------------

class BladeStrainFile:

    def __init__(self, strainfile, Fs=512, verbose=False, debug=False,
                 norm_strain=False, silent=False, checkpulse=True):
        """
        Paramters
        ---------

        strainfile : str
            string containing the full path+filename to the blade strain file

        Fs : int, default=None
            Sampling frequency of the blade strain log file. Only relevant
            if the source is a streamed file. In case the source is a
            triggered file, the sample rate is given in the file header.

        verbose : boolean, default=False

        norm_strain : boolean, default=False

        silent : boolean, default=False

        checkpulse : boolean, default=True
            For the triggered file type (April and later) we can find the
            starting of the pulse signal and cut off everything before that
            point. This is helpful when syncing with dSpace.

        """
        # class properties
        self.__name__ = 'bladestrainfile'
        self.verbose = verbose
        self.silent = silent
        self.strainfile = strainfile
        self.Fs = Fs
        self.sample_rate = Fs
        self.norm_strain = norm_strain
        self.debug = debug

        if strainfile:
            if not self.silent:
                print '\n'+'*'*80
                print 'LOADING BLADE FILE'
            self.time, self.data, self.labels = self.read(strainfile)
            if self.filetype == 'streaming':
                self.time, self.data = self._tune_stream(self.time, self.data)

            elif self.filetype == 'triggered':
                self.time, self.data = self._tune_trig(self.time, self.data,
                                                       checkpulse=checkpulse)

            # split strainfile into respath and resfile
            #self.respath
            self.resfile = self.strainfile.split('/')[-1]

        self.cnames = ['blade2_root', 'blade2_30pc',
                       'blade1_root', 'blade1_30pc', 'blade_rpm_pulse']

    def _add_missing_point(self, time, data):
        """
        Insert missing data point
        =========================

        Insert a missing data point in between index k and k+1.

        .. math::
            t_{new} = \\frac{ t_k + t_{k+1} }{2}

        Why is there actually a missing time step? The time step counter
        missed a step or did the V-strain link failed to transmit/receive one
        time step? Communication error is more likely in my opinion.
        Conclusion: add the missing time step, obtain values by interpolation.

        Note that maybe many more time steps are missing. We don't know for
        sure if it is only one.

        Parameters
        ----------

        time : ndarray(n)
            1D array holding the time stamps

        data : ndarray(n)
            1D array holding the data points

        Returns
        -------

        time_new

        data_new

        missing

        """


        # see if there are some sweeps missing. This would destroy the
        # continuous sample time series
        deltas = time[1:] - time[:-1]
        # we are still in int mode, so all deltas need to be == 1
        d_check = np.abs(deltas-1).__gt__(0)
        if np.any(d_check):
            # find out how many missing points we have
            nr_missing = len(deltas[d_check])
#            ff = deltas[d_check.argsort()]
#            print ff[-nr_missing:]

            # the arguments that would sort d_check: containing False if ok,
            # True if delta > 1. The True values will be at the end of the
            # sort array
            tmp_i = d_check.argsort()
#            # reverse the arrays, so the True values are upfront
#            tmp_i = tmp_i[::-1]
#            d_check = d_check[::-1]
            sel_i = tmp_i[-nr_missing:]
            if self.verbose:
                print 'nr of missing data points: %i' % nr_missing
#                print 'deltas[sel_i], sel_i:', deltas[sel_i], sel_i

            # only do this for the first point, otherwise the indeces will
            # become a mess
            k = sel_i[0]
            # we need to add one additional point at in between k and k+1
            time_new = time[:k+1].copy()
            data_new = data[:k+1,:].copy()
            # interpolate the new value between k and k+1
            x = np.array([time[k], time[k+1]])
            y = np.array([data[k,:], data[k+1,:]])

            # in case there are many missing values, exclude start and end!
#            x_new = (time[k] + time[k+1])/2.
            x_new = np.arange(time[k]+1, time[k+1], 1)

            y_new = sp.interpolate.griddata(x, y, x_new)
            # add the missing point
            time_new = np.append(time_new, x_new)
            data_new = np.append(data_new, y_new, axis=0)
            # and the rest of the data
            time_new = np.append( time_new, time[k+1:].copy() )
            data_new = np.append( data_new, data[k+1:,:].copy(), axis=0 )
            if self.verbose:
                print 'added x_new.shape', x_new.shape,
                print 'added y_new.shape', y_new.shape
                print 'data_new.shape',data_new.shape,'data.shape',data.shape
                print '           (k-1)    (k)    (k+1)'
                print '    time:',time[k-1],time[k],time[k+1]
                print 'time_new:',time_new[k], time_new[k+1], time_new[k+2]
        else:
            if self.verbose:
                print 'no missing data points'
            return time, data, False

        return time_new, data_new, True

    def _tune_trig(self, time, data, checkpulse=True):
        """
        Some tuning for the triggered filetype
        """
        # make sure we have a continues series and check the sample rate
#        Fs_dummy = calc_sample_rate(time)
        deltas = time[1:] - time[:-1]
        rel_error = 5e-3 # 0.0050 = 0.50%
        assert (deltas.max() - deltas.min())/deltas.max() <  rel_error

        # compare sample rates
        Fs_ratio = (1/deltas.mean())/self.Fs
        if not (Fs_ratio > 0.9999 and Fs_ratio < 1.0001):
            msg = 'Sample rate from file header does not match actual data.'
            raise ValueError, msg

        # set zero load to zero strain instead of 2048
        # DO NOT CHANGE THIS. UNLESS YOU REDO ALL THE CALIBRATION AGAIN
        data[:,0:4] = (data[:,0:4] - 2048.)

        # convert to some other scale
        if self.norm_strain:
            data[:,0:4] = 100.*data[:,0:4]/2048.
            #data[:,0:4] = 100.*data[:,0:4]/data[:,0:4].max()

        if checkpulse:
            # normalise the pulse signal
            data[:,4] = 2.*data[:,4]/data[:,4].max()

            # and find the location of the first pulse
            self.i_first = (data[:,4] - 0.9).__ge__(0).argmax()
            # and cut off everything before that
            time = time[self.i_first:]
            time = time - time[0] + 0.002
            data = data[self.i_first:,:]

        if not self.silent:
            print '========> calibrated data.shape:', data.shape

        return time, data

    def _tune_stream(self, time, data):
        """
        Cleanup some stuff if the source is a streaming file

        Data calibration is dependent on the date and is specific for a certain
        dataset and even channel

        First 10 datapoints are always ignored due to some startup peaks.
        """
        # always ignore the first 10 points, they contain initial noise
        time, data = time[10:], data[10:,:]

        # add values for each missing point
        deltas = time[1:] - time[:-1]
        # we are still in int mode, so all deltas need to be == 1
        d_check = np.abs(deltas-1).__gt__(0)
        if np.any(d_check):
            # find out how many missing points we have
            nr_missing = len(deltas[d_check])
            if not self.silent:
                print '=======> missing points in blade.time: %i' % nr_missing

        missing = True
        while missing:
            time, data, missing = self._add_missing_point(time, data)

        # make sure we have a continues series, Fs will be 1, still in
        # counting time steps mode here
        Fs_dummy = calc_sample_rate(time)

        # set zero load to zero strain instead of 2048
        # DO NOT CHANGE THIS. UNLESS YOU REDO ALL THE CALIBRATION AGAIN
        data[:,0:4] = (data[:,0:4] - 2048.)

        # convert time signal from binary to seconds
        time *= (1./self.Fs)
        if not self.silent:
            print '=======> calibrated data.shape:', data.shape

        return time, data

    def read(self, strainfile):
        """
        Read the blade strain file which is saved from a streaming session.

        File header for a triggered session (used in April):
            Node,1259
            Opening Tag,65535
            Trigger ID,48
            # Sweeps,65500
            Trigger #,1
            Num Channels,5
            Channel Mask,47
            Clock Freq (Hz),1024


            Time (ms),Channel 1,Channel 2,Channel 3,Channel 4,Channel 6,

        Note that time is given in seconds, starting at zero.

        File header for a saved stream session:
            Port, 4
            Node, 1259
            Time, 02/09/2012 15:29:18
            Channel Mask, 15

            Acquisition Attempt, Channel 1, Channel 2, Channel 3, Channel 4,

        Note that time is given as an acquisition attempt number, not a time
        stamp.
        """

        # read the file header and determine which kind it is: a streamed
        # or triggered log file
        try:
            f = open(strainfile, 'r')
        except IOError:
            strainfile += '.csv'
            f = open(strainfile, 'r')

        # stuff for reading the new node commander software
        channel_info, data_start = False, False

        for i, line in enumerate(f):

            #print line.replace('\n', '')

            # determine streaming properties
            # ------------------------------

            # read the line containing the start Time (for stream version)
            #if line.startswith('Time,'):
                #starttime = line.replace('\n', '').replace('Time, ', '')
                #pass

            # reading stuff for the new Node Commander triggered file type
            # --------------------------------------
            if channel_info and not line.startswith('Channel'):
                self.Fs = int(line.split(';')[2].replace(' Hz', ''))
                self.sample_rate = self.Fs
                channel_info = False
                # FIXME: each channel has an entry in this section, but we
                # assume all channels have the same stuff
                if self.debug:
                    print 'sample rate: %i' % self.Fs
            elif data_start:
                header_lines = i+1
                # unicode array for the labels
                labels = np.ndarray((4), dtype='<U15')
                sep = ';'
                # skip the first label, that's time, and last entry is empty
                labels[:] = line.replace('\n', '').split(sep)[1:-1]
                if self.debug:
                    print 'header lines: %i' % header_lines
                    print 'labels: ', labels
                # stop reading, switch to faster np.loadtxt method for data
                break

            # --------------------------------------

            # original stream file on WinXP
            elif line.startswith('Acquisition'):
                self.filetype = 'streaming'
                header_lines = i+1

                # unicode array for the labels
                labels = np.ndarray((4), dtype='<U15')

                sep = ','

                # skip the first label, that's time, and last entry is empty
                labels[:] = line.replace('\n', '').split(sep)[1:-1]
                # stop reading, switch to faster np.loadtxt method for data
                break

            # with the new software Node Commander, the header looks different
            elif line.startswith('Sweeps;Channel'):
                self.filetype = 'streaming'
                header_lines = i+1

                sep = ';'
                # unicode array for the labels
                labels = np.ndarray((4), dtype='<U15')

                # skip the first label, that's time, and last entry is empty
                labels[:] = line.replace('\n', '').split(sep)[1:-1]

                # stop reading, switch to faster np.loadtxt method for data
                break

            # determine triggered session properties
            # --------------------------------------

            # triggered type with the new Node Commander software
            elif line.startswith('FILE_INFO'):
                self.filetype = 'triggered-v2'
            # with the new Node Commander, we also have the sample rate
            elif line.startswith('CHANNEL_INFO'):
                channel_info = True
            elif line.startswith('DATA_START'):
                data_start = True

            elif line.startswith('Clock Freq'):
                self.Fs = int(line.split(',')[1])
                self.sample_rate = self.Fs
                self.filetype = 'triggered'

            elif line.startswith('Num Channels'):
                sep = ','
                num_chan = int(line.split(sep)[1])

                # unicode array for the labels
                labels = np.ndarray((num_chan), dtype='<U15')

            elif line.startswith('Time (ms),'):
                header_lines = i
                # read the labels and ignore the last bogus entry
                # skip also the first label, that's time
                labels[:] = line.replace('\n', '').split(sep)[1:-1]
                # stop reading, switch to faster np.loadtxt method for data
                break

        f.close()

        # check: if the sample rate has not been given and we are talking
        # about a streamed file, raise an error.

        if not self.Fs:
            msg = 'Blade strain file has no sample rate set.'
            raise UserWarning, msg

        # read the data part
        #data = np.loadtxt(strainfile, delimiter=',', skiprows=6)

        # genfromtxt is faster, but skip the last line since in some cases
        # it might not be complete (especially for the triggered type)
        data = np.genfromtxt(strainfile, skip_header=header_lines,
                             delimiter=sep, dtype=np.float32, skip_footer=1)
        if not self.silent:
            print strainfile
            print 'loaded strain blade file, shape:', data.shape

        # seperate the time signal from the data array
        time = data[20:,0]
        data = data[20:,1::]

        return time, data, labels


    def plot_channel(self, **kwargs):
        """
        Plot a set of channels. Figure name is the same as the blade result
        file except that '_chX' is added. Figure is seized dynamically based
        on the number of plots.

        Arguments
        ---------

        channel : int or str, default=0
            Index of the channel or string of the short channel name

        figpath : str, default=None
            Full path of the to be saved figure

        """
        channel = kwargs.get('channel', 0)
        figpath = kwargs.get('figpath', None)

        # in case we have not an integer but a string: convert to index
        if type(channel).__name__ == 'int':
            channel = [channel]

        figfile = self.resfile + '_ch'
        for k in channel:
            figfile += '_' + str(k)

        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, self.time, self.data, self.labels,
                         channels=channel, grandtitle=figfile)

    def psd_plots(self, figpath, channels, **kwargs):
        """
        Power Spectral Density Analysis
        ===============================

        Do for all blade strain channels a PSD analysis, put besides
        the raw signal, and indicate the highest peak.
        """

        figfile = kwargs.get('figfile', self.resfile) + '_eigenfreqs'
        # take a few different number of samples for the NFFT
        nnfts = kwargs.get('nnfts', [16384, 8192, 4096, 2048])
        channel_names = kwargs.get('channel_names', None)
        fn_max = kwargs.get('fn_max', 100)
        saveresults = kwargs.get('saveresults', False)

        pa4 = plotting.A4Tuned()
        grandtitle = figfile.replace('_','\_')
        pa4.setup(figpath+figfile, nr_plots=len(channels)*2,
                  grandtitle=grandtitle)

        eigenfreqs = pa4.psd_eigenfreq(self.time, self.data, channels,
                      self.sample_rate, channel_names=channel_names,
                      nnfts=nnfts, fn_max=fn_max, saveresults=saveresults)

        pa4.save_fig()

        return eigenfreqs


class OJFLogFile:

    def __init__(self, ojffile=None, silent=False):
        """
        """
        self.silent = silent
        # class properties
        self.__name__ = 'ojflogfile'

        self.ojffile = ojffile

        self.cnames = ['RPM_fan', 'temperature', 'static_p', 'delta_p',
                       'wind_speed']

        if ojffile:
            self.time, self.data, self.labels = self.read(ojffile)
            self.cnames = [str(k).replace(' ', '_') for k in self.labels]

        # sample rate is fixed
        self.sample_rate = 4

    def _calibrate(self):
        """
        Data calibration is dependent on the date and is specific for a certain
        dataset and even channel
        """
        pass

    def _timestr2float(self, s):
        """
        Convert the time stamp string HH:MM:SS to a float in seconds where
        0 is 00:00:00 and 86400 is 23:59:59
        """
        s = s.split(':')
        # make sure that we have 3 elements back!
        assert len(s) == 3
        return (int(s[0])*60*60) + (int(s[1])*60) + (int(s[2]))

    def _correct_accuracy(self, time):
        """
        There are 4 log entries per second, however, the time stamps accuracy
        is only 1 second. Add 1/4 s to the corresponding entries
        """
        # figure out how many out of 4 time stamps are there in the beginning
        # of the file
        nr_ini = len(time[(time-time[0]).__eq__(0)])

        # this answer should always be lower than 4! There are only 4
        if nr_ini < 4:
            for k in range(0,nr_ini):
                time[k] += (3-k)*0.25
        elif nr_ini == 4:
            # switch ini back to zero, first second has a full set of 4
            # measurement points
            nr_ini = 0
        else:
            raise ValueError, 'OJF time series has unknown sample rate'

        # the first data point is only 1 second, the other ones are 4 per s
        time[nr_ini+1::4] += 0.25
        time[nr_ini+2::4] += 0.50
        time[nr_ini+3::4] += 0.75

        return time

    def read(self, ojffile):
        """
        """
        # define the labels
        labels = np.ndarray((5), dtype='<U15')
#        labels[0] = u'time'
        labels[0] = u'RPM fan'
        labels[1] = u'temperature'
        labels[2] = u'static_p'
        labels[3] = u'delta_p'
        labels[4] = u'wind speed'

        # and the inverse for easy lookup in dictionary
        self.labels_ch = dict()
        for index, item in enumerate(labels):
            self.labels_ch[item] = index

        self.ojffile = ojffile

        # in April the data did not hold a time column...
        # determine ojf log file type
        try:
            f = open(ojffile, 'r')
        # names have not consistantly saved with .log, anticipate troubles
        except IOError:
            f = open(ojffile + '.log', 'r')
            ojffile += '.log'

        # For the february session
        if f.readline().find(':') > 0:
            f.close()
            # reading the file as a recarray
            names=('time', 'rpm', 'temp', 'static_p', 'delta_p', 'windspeed')
            formats=('S8',  'f8',      'f8',       'f8',   'f8',        'f8')
            self.data_or = np.loadtxt(ojffile,
                              dtype={'names': names, 'formats': formats})

            # read as a simple array, convert the time stampts (HH:MM:SS) to a
            # float between 0 and 86400 (=24*60*60)
            data = np.loadtxt(ojffile, converters = {0: self._timestr2float})

            # split-up time, labels and actual data
            time = self._correct_accuracy(data[:,0])
            # ditch the time from the data signal
            data = data[:,1::]

        # and for the April session there is no time stamp HH:MM:SS
        else:
            f.close()
            # reading the file as a recarray
            names =   ('rpm', 'temp', 'static_p', 'delta_p', 'windspeed')
            formats = ( 'f8',      'f8',       'f8',   'f8',        'f8')
            self.data_or = np.loadtxt(ojffile,
                              dtype={'names': names, 'formats': formats})

            data = np.loadtxt(ojffile)
            # we don't know the sampling rate for sure, but it is probably
            # 4 Hz
            time = np.arange(0,len(data),0.25)

        if not self.silent:
            print '\n'+'*'*80
            print 'LOADING OJF FILE'
            print ojffile
            print 'ojffile.shape:', data.shape

        return time, data, labels


class DspaceMatFile:

    def __init__(self, matfile=None, silent=False):
        """
        Load a dSPACE generated matfile
        ===============================

        Load the measurement data aggregated by dSPACE and saved as a mat file.
        If the keyword matfile is a valid filename of a mat file, the latter
        will be loaded.

        keywords
        --------

        matfile : str, defaul=None
            file name of a valid dSPACE generated mat file.

        Members
        -------

        time : ndarray(n)
            array containing the time signal

        data : ndarray(n,m)
            array containing all m channels of measurement data

        labels : ndarray(m)
            array with the channel labels


        """

        # class properties
        self.__name__ = 'dspacematfile'
        self.silent = silent

        # declare the channel changes
        self.ch_dict = dict()
        self.ch_dict['"Model Root"/"Duty_Cycle"/"Value"'] = 'Duty Cycle'
        self.ch_dict['"Model Root"/"Rotorspeed_Estimator"/"Rotor_freq_RPM"']\
                    = 'RPM Estimator v1'
        # make it twice for easy RPM channel selection when mixing results
        # from April
        self.ch_dict['"Model Root"/"Rotorspeed_Estimator"/"Rotor_freq_RPM"']\
                    = 'RPM'
        self.ch_dict['"Model Root"/"Rotorspeed_Estimator"/"Rotor_freq_RPM1"'] \
                    = 'RPM Estimator v2'
        self.ch_dict['"Model Root"/"Rotorspeed_Estimator"/"Trigger"'] \
                    = 'HS trigger'
        self.ch_dict['"Model Root"/"Sensors"/"50mA_v"/"Out1"'] = 'Current'
        self.ch_dict['"Model Root"/"Sensors"/"Current_Filter"']='Current Filter'
        self.ch_dict['"Model Root"/"Sensors"/"FA_gain"/"Out1"'] \
                    = 'Tower Strain For-Aft'
        self.ch_dict['"Model Root"/"Sensors"/"Power"'] = 'Power'
        self.ch_dict['"Model Root"/"PWM_GEN"/"PWM_GEN_Out"'] = 'Power2'
        self.ch_dict['"Model Root"/"Sensors"/"SS_gain"/"Out1"'] \
                    = 'Tower Strain Side-Side'
        self.ch_dict['"Model Root"/"Sensors"/"TWRFA1"'] \
                    = 'Tower Strain For-Aft filtered'
        self.ch_dict['"Model Root"/"Sensors"/"TWRSS1"'] \
                    = 'Tower Strain Side-Side filtered'
        self.ch_dict['"Model Root"/"Sensors"/"Voltage_Filter"'] \
                    = 'Voltage filtered'
        # at least in april, accZ is actually SS
        self.ch_dict['"Model Root"/"Sensors"/"accX"'] = 'Tower Top acc Z'
        self.ch_dict['"Model Root"/"Sensors"/"accY"'] = 'Tower Top acc Y (FA)'
        self.ch_dict['"Model Root"/"Sensors"/"accZ"'] = 'Tower Top acc X (SS)'
        # RPM pulse is for April the wireless strain sensor pulse
        self.ch_dict['"Model Root"/"Sensors"/"pulse1"'] = 'RPM Pulse'
        self.ch_dict['"Model Root"/"Sensors"/"Dummy"'] = 'Yaw Laser'
        # for February 14, Dummy is renamed to Voltage_LP9
        self.ch_dict['"Model Root"/"Sensors"/"Voltage_LP9"/"Out1"']='Yaw Laser'
        # changes for April
        self.ch_dict['"Model Root"/"Sensors"/"Rotor_Speed_RPM"'] = 'RPM'
        self.ch_dict['"Model Root"/"Sensors"/"Rotor_Pos_deg"'] = 'Azimuth'
        self.ch_dict['"Model Root"/"Sensors"/"sound"'] = 'Sound'
        self.ch_dict['"Model Root"/"Sensors"/"sound_gain"/"Out1"']='Sound_gain'
        self.ch_dict['"Model Root"/"Tigger_Signal"/"Trigger"'] = 'HS trigger'
        self.ch_dict['"Model Root"/"Tigger_Signal"/"Trigger_signal"/"Dummy"']\
                    = 'HS trigger start-end'

        self.cnames = {}
        self.cnames['"Model Root"/"Duty_Cycle"/"Value"'] = 'duty_cycle'
        self.cnames['"Model Root"/"Rotorspeed_Estimator"/"Rotor_freq_RPM"']\
                    = 'rpm_est_v1'
        # make it twice for easy RPM channel selection when mixing results
        # from April
        self.cnames['"Model Root"/"Rotorspeed_Estimator"/"Rotor_freq_RPM"']\
                    = 'rpm'
        self.cnames['"Model Root"/"Rotorspeed_Estimator"/"Rotor_freq_RPM1"'] \
                    = 'rpm_est_v2'
        # or is this the blade strain trigger?
        self.cnames['"Model Root"/"Rotorspeed_Estimator"/"Trigger"'] \
                    = 'hs_trigger'
        self.cnames['"Model Root"/"Sensors"/"50mA_v"/"Out1"'] = 'current'
        self.cnames['"Model Root"/"Sensors"/"Current_Filter"']='current_filt'
        self.cnames['"Model Root"/"Sensors"/"FA_gain"/"Out1"'] \
                    = 'tower_strain_fa'
        self.cnames['"Model Root"/"Sensors"/"Power"'] = 'power'
        self.cnames['"Model Root"/"PWM_GEN"/"PWM_GEN_Out"'] = 'power2'
        self.cnames['"Model Root"/"Sensors"/"SS_gain"/"Out1"'] \
                    = 'tower_strain_ss'
        self.cnames['"Model Root"/"Sensors"/"TWRFA1"'] \
                    = 'tower_strain_fa_filt'
        self.cnames['"Model Root"/"Sensors"/"TWRSS1"'] \
                    = 'tower_strain_ss_filt'
        self.cnames['"Model Root"/"Sensors"/"Voltage_Filter"'] \
                    = 'voltage_filt'
        # at least in april, accZ is actually SS
        self.cnames['"Model Root"/"Sensors"/"accX"'] = 'towertop_acc_z'
        self.cnames['"Model Root"/"Sensors"/"accY"'] = 'towertop_acc_fa'
        self.cnames['"Model Root"/"Sensors"/"accZ"'] = 'towertop_acc_ss'
        # RPM pulse is for April the wireless strain sensor pulse
        self.cnames['"Model Root"/"Sensors"/"pulse1"'] = 'rpm_pulse'
        self.cnames['"Model Root"/"Sensors"/"Dummy"'] = 'yaw_angle'
        # for February 14, Dummy is renamed to Voltage_LP9
        self.cnames['"Model Root"/"Sensors"/"Voltage_LP9"/"Out1"']='yaw_angle'
        # changes for April
        self.cnames['"Model Root"/"Sensors"/"Rotor_Speed_RPM"'] = 'rpm'
        self.cnames['"Model Root"/"Sensors"/"Rotor_Pos_deg"'] = 'rotor_azimuth'
        self.cnames['"Model Root"/"Sensors"/"sound"'] = 'sound'
        self.cnames['"Model Root"/"Sensors"/"sound_gain"/"Out1"']='sound_gain'
        self.cnames['"Model Root"/"Tigger_Signal"/"Trigger"'] = 'hs_trigger'
        self.cnames['"Model Root"/"Tigger_Signal"/"Trigger_signal"/"Dummy"']\
                    = 'hs_trigger_start_end'

        self.rpm_spike_removed = False

        # they are named withouth quotes for the sweep files
        keys = self.ch_dict.keys()
        for key in keys:
            self.ch_dict[key.replace('"', '')] = self.ch_dict[key]
            self.cnames[key.replace('"', '')] = self.cnames[key]

        self.matfile = matfile
        if matfile:
            # file name only, excluding the path
            self.resfile = matfile.split('/')[-1]
            self.time, self.data, self.labels = self.read(matfile)

    def remove_rpm_spike(self, plot=False):
        """
        For some mistirious reason, the RPM signal can hold extreme big spikes.
        Remove only the spiked region and assume that ends after 0.6 seconds
        """
        irpm = self.labels_ch['RPM']
        rpm = self.data[:,irpm]
        # normalise by dividing by the time per sample
        # or multiply with sample rate
        diff_rpm = np.abs(np.diff(self.data[:,irpm]))*self.sample_rate
        if diff_rpm.max() > 1000:
            # find the insane high peak, and the next normal peak. Ignore
            # anything in between.
            # just make n as large as the array
            ids = wafo.misc.findpeaks(rpm, n=len(rpm), min_h=2, min_p=0)
            # sort, otherwise ordered according to significant wave height
            ids.sort()
            # seems the peak is often negative. So teak the peak before and
            # after the abs max value and replace values with interpolation
            imax = np.abs(rpm).argmax()
            # the first value bigger than the peak index is the end peak
            iiend = ids.__ge__(imax).argmax()
            iend = ids[iiend]
            istart = ids[iiend-1]

            # do some interactive plots to check
            if plot:
                plt.figure()
                plt.plot(self.time, rpm, 'b', label='rpm')
                plt.plot(self.time[1:], diff_rpm, 'r', label='diff')
                # plot all the RPM peaks
                plt.plot(self.time[ids], rpm[ids], 'k+')
                # plot the start end ending point of interpolation replacement
                plt.plot(self.time[[istart,iend]], rpm[[istart,iend]], 'ms')

            # data checks: time between peaks should not exceed 1 second
            # and the normalised difference between the two peaks should not
            # be too large
            tdelta = self.time[istart] - self.time[iend]
            ratio = np.abs((rpm[iend] - rpm[istart])/rpm[iend])

            if tdelta > 1.5:
                msg = 'remove peak failed: time between peaks %1.3f' % tdelta
                logging.warn(msg)
            elif ratio > 0.2:
                msg = 'remove peak failed: delta ratio peaks %1.3f' % ratio
                logging.warn(msg)
            # if tests when ok, save the results
            else:
                replace = np.linspace(rpm[istart], rpm[iend], iend-istart)
                self.data[istart:iend,irpm] = replace
                logging.warn('replaced peak in RPM signal')
                self.rpm_spike_removed = True

            if plot:
                # and plot the corrected value
                plt.plot(self.time[istart:iend], replace, 'y')


    def rpm_from_pulse(self, pulse_treshold=0.2, h=0.2, plot=False):
        """
        Derive the RPM from the pulse signal. However, note that this approach
        only works well if all pulses are of approximately the same height,
        and the mean level should'nt change.

        This function is quite similar to the RPM Estimater v1. There seems
        some averaging going on for the latter? The estimator is based on
        crossing a certain treshold maybe while this algorithm selects the
        signal peak!
        """

        ch = self.labels_ch['RPM Pulse']

        # use wafo to find the turning points: min/max positions on the waves
        wafo_sig = np.transpose(np.array([self.time, self.data[:,ch]]))
        sig_ts = wafo.objects.mat2timeseries(wafo_sig)
        self.sig_tp = sig_ts.turning_points(wavetype=None, h=h)
        # Select only the max peaks, not the min peaks. Make sure the initial
        # point has a value higher than pulse_treshold
        if   self.sig_tp.data[0] > pulse_treshold: start = 0
        elif self.sig_tp.data[1] > pulse_treshold: start = 1
        elif self.sig_tp.data[2] > pulse_treshold: start = 2
        else:
            msg = 'Can\'t find a peak value above treshold for first 3 points'
            raise UserWarning, msg

        end = len(self.sig_tp.data)
        # select only the signal peaks. Each peak seems to be accompinied by
        # a low peak, so skip every other peak found by wafo
        data_peaks = self.sig_tp.data[start:end:2]
        time_peaks = self.sig_tp.args[start:end:2]

        # and make sure we only have selected peaks above pulse_treshold
        if not data_peaks.__ge__(pulse_treshold).all():
            msg = 'selected peak range is not exclusively above threshold'
            raise ValueError, msg

        # convert to rpm, based on pulses per revolution
        # time per peak = sec/120deg = 3*sec/rev = 3*min/60*rev
        # carefull: in February we had 3 peaks per revolution,
        # somewhere in april they where made black though
        if self.campaign == 'February':
            pulses_per_rev = 3
        elif self.campaign == 'April':
            pulses_per_rev = 1
        else:
            msg='campaign has to be February or April, not %s' % self.campaign
            raise ValueError, msg

        self.data_rpm = 60.0/((time_peaks[1:]-time_peaks[:-1])*pulses_per_rev)
        self.time_rpm = time_peaks[1:]

        if plot:
            plt.figure()
            plt.plot(self.time, self.data[:,ch])
            plt.plot(time_peaks, data_peaks, 'g+')
            plt.title('original signal and selected peaks')
            plt.legend()

            plt.figure()
            plt.plot(self.time_rpm, self.data_rpm, 'r', label='peak derived')
            # overplot the RPM version1 signal
            try:
                ch_rpm_v1 = self.labels_ch['RPM']
                plt.plot(self.time, self.data[:,ch_rpm_v1], label='RPM est v1')
            except KeyError:
                pass
            try:
                ch_rpm_v1 = self.labels_ch['RPM Estimator v2']
                plt.plot(self.time, self.data[:,ch_rpm_v1], label='RPM est v2')
            except KeyError:
                pass
            plt.title('comparing rpm estimater with peak derived rpm')
            plt.legend()

            plt.figure()
            plt.title('RPM Pulse')
            plt.subplot(211)
            plt.plot(self.time, self.data[:,ch], label='data or')
            plt.plot(self.sig_tp.args, self.sig_tp.data, '+', label='wafo tp')
            plt.legend()
            plt.subplot(212)
            plt.title('PSD from original data')
            Fs = self.sample_rate
            Pxx, freqs = plt.psd(self.data[:,ch], NFFT=4*8192, Fs=Fs)

        return time_peaks, data_peaks

    def read_sweeps(self, dspacematfile):
        """
        The automated sweep recordings: where the dc was changed between
        0 and 1 and each value of dc is one measurement
        """

        L1 = dspacematfile['DATA']
        L2 = L1[0,0]

        # this we got by looking at the matlab file. Why is loadmat not able
        # to get the variable names?
        self.SamplingPeriod = L2[0][0,0]
#        fname = L2[1]
        self.DateTime = L2[2][0,0]
        x_var = L2[3]
        x_data = L2[4]
#        par_names = L2[5]
#        parameters = L2[6]
        N = L2[7]
#        measplan = L2[8]
#        windspeed = L2[9]
#        run = L2[10]
        self.Downsampling = L2[11][0,0]
#        temperature = L2[12]
#        density = L2[13]
#        comments = L2[14]

        time = np.ndarray(N, dtype=np.float64)
        time[:] = x_data[0,:]

        ychannels = x_var.shape[0]

        # in order to have the same result arrays for all sources (strain,
        # OJF wind tunnel data) break out of the recarray
        datashape = (time.shape[0], ychannels)
        data = np.ndarray(datashape, dtype=np.float64)

        labels = np.ndarray((ychannels), dtype='<U100')
        # and the inverse, the channel name coupled to its number
        self.labels_ch = dict()
        self.labels_cnames = {}
        # cycle through all the Y channels recorded
        for ch in range(ychannels):
            chname_long = x_var[ch,0][0][0]
            # change the label name if applicable
            try:
                labels[ch] = self.ch_dict[chname_long]
                # and the inverse: channel name coupled to its number
                self.labels_ch[labels[ch]] = ch
                # map the print name to C-compatible names
                self.labels_cnames[labels[ch]] = self.cnames[chname_long]
            except KeyError:
                labels[ch] = chname_long
            # channel 0 is the time and it is not in par_names
            data[:,ch] = x_data[ch+1,:]

        self.sample_rate = 20000/self.Downsampling
        self.nr_samples = len(time)

        self.campaign = 'dc-sweep'

        return time, data, labels

    def read(self, matfile):
        """
        Read the D-space generated Matlab file containing measurement data of
        one continues wind tunnel run.
        """
        self.matfile = matfile

        dspacematfile = scipy.io.loadmat(matfile)
        # print 'dspacematfile keys:', dspacematfile.keys()
        # the actual data is behind the file name, but the position on the
        # list varies. So find the correct one
        notinteresting = set(['__version__', '__header__', '__globals__'])
        filekey = (set(dspacematfile.keys()) - notinteresting).pop()
        # for all the regular cases, filekey is the file name
        # for the sweep cases, the key is simply DATA
        if filekey == 'DATA':
            return self.read_sweeps(dspacematfile)

        L1 = dspacematfile[filekey]
        # array([[ ([[(array([], dtype='<U1'), array([[4]], dtype=int32),
        L2 = L1[0,0]
        # >>> L2.dtype
        # dtype([('X', '|O8'), ('Y', '|O8'), ('Description', '|O8'),
        #       ('RTProgram', '|O8'), ('Capture', '|O8')])

        # make sure there is only one X data channel (=time)
        assert L2['X']['Data'].shape == (1,1)
        # >>> L2['X'].dtype
        # dtype([('Name', '|O8'), ('Type', '|O8'), ('Data', '|O8'),
        #       ('Unit', '|O8')]), ('XIndex', '|O8')])
        # not sure if going to float64 is usefull...data is saved in matlab
        # as float8
        time = np.ndarray(L2['X']['Data'][0,0][0,:].shape, dtype=np.float64)
        time[:] = L2['X']['Data'][0,0][0,:]

        # restruct the data: one array with time, result channels
        # another array with the channels labels
        ychannels = L2['Y']['Data'].shape[1]

        # other data we can extract: L2['Capture'].dtype
#        Length = L2['Capture']['Length'][0][0][0,0]
        self.Downsampling = L2['Capture']['Downsampling'][0][0][0,0]
        self.DateTime = L2['Capture']['DateTime'][0][0][0]
        self.SamplingPeriod = L2['Capture']['SamplingPeriod'][0][0][0,0]

        # in order to have the same result arrays for all sources (strain,
        # OJF wind tunnel data) break out of the recarray
        datashape = (time.shape[0], ychannels)
        data = np.ndarray(datashape, dtype=np.float64)
#        # or just keep the record array intact
#        data = L2['Y']['Data']
        # >>> L2['Y'].dtype
        # dtype([('Name', '|O8'), ('Type', '|O8'), ('Data', '|O8'),
        #       ('Unit', '|O8'), ('XIndex', '|O8')])

        if not self.silent:
            print '\n'+'*'*80
            print 'LOADING DSPACE FILE'
            print matfile
            print 'dspace.shape:', data.shape

        labels = np.ndarray((ychannels), dtype='<U100')
        # and the inverse, the channel name coupled to its number
        self.labels_ch = {}
        self.labels_cnames = {}
        # cycle through all the Y channels recorded
        for ch in range(ychannels):
            # change the label name if applicable
            try:
                chname_long = L2['Y']['Name'][0,ch][0]
                labels[ch] = self.ch_dict[chname_long]
                # and the inverse: channel name coupled to its number
                # make sure we're not overwriting any channel
                if labels[ch] in self.labels_ch:
                    msg = ('overwriting channel label in labels_ch '
                           'key, value: %s, %s' % (labels[ch], ch))
                    logging.warning(msg)
                self.labels_ch[labels[ch]] = ch
                # map the print name to C-compatible names
                self.labels_cnames[labels[ch]] = self.cnames[chname_long]
            except KeyError:
                labels[ch] = L2['Y']['Name'][0,ch][0]
            data[:,ch] = L2['Y']['Data'][0,ch][0,:]

        self.sample_rate = calc_sample_rate(time)
        self.nr_samples = len(time)

        # is it April or February data?
        if self.labels_ch.has_key('Azimuth'):
            self.campaign = 'April'
        else:
            self.campaign = 'February'

        return time, data, labels

    def plot_channel(self, **kwargs):
        """
        Plot a set of channels. Figure name is the same as the dSPACE result
        file except that '_chX' is added. Figure is seized dynamically based
        on the number of plots.

        Arguments
        ---------

        channel : int or str, default=0
            Index of the channel or string of the short channel name

        figpath : str, default=None
            Full path of the to be saved figure

        """
        channel = kwargs.get('channel', 0)
        figpath = kwargs.get('figpath', None)


        # in case we have not an integer but a string: convert to index
        if type(channel).__name__ == 'str':
            channel = [self.labels_ch[channel]]
        # for a list, see for each entry if it needs to be converted to index
        elif type(channel).__name__ == 'list':
            for k in range(len(channel)):
                if type(channel[k]).__name__ == 'str':
                    channel[k] = self.labels_ch[channel[k]]

        figfile = self.matfile.split('/')[-1] + '_ch'
        for k in channel:
            figfile += '_' + str(k)

        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, self.time, self.data, self.labels,
                         channels=channel, grandtitle=figfile)

    def psd_plots(self, figpath, channel_names, **kwargs):
        """
        Power Spectral Density Analysis
        ===============================

        Compare the signal and PSD analysis for all given channels.

        Parameters
        ----------

        figpath : str
            Path to where the figure should be saved. Figure file name is by
            default the name of the result file.

        channel_names : list of str


        figfile : str, default=self.resfile

        Returns
        -------

        eigenfreqs : dict with ndarray
            key is channeli, value is ndarray([freq peaks, Pxx_log values])

        """

        figfile = kwargs.get('figfile', self.resfile)
        # take a few different number of samples for the NFFT
        nnfts = kwargs.get('nnfts', [16384, 8192, 4096, 2048])
        fn_max = kwargs.get('fn_max', 100)
        saveresults = kwargs.get('saveresults', False)

        channels = []
        # convert to channel numbers
        for k in channel_names:
            channels.append(self.labels_ch[k])

        pa4 = plotting.A4Tuned()
        pa4.setup(figpath+figfile,nr_plots=len(channels)*2,grandtitle=figfile)

        eigenfreqs = pa4.psd_eigenfreq(self.time, self.data, channels,
                      self.sample_rate, channel_names=channel_names,
                      nnfts=nnfts, fn_max=fn_max, saveresults=saveresults)

        pa4.save_fig()

        return eigenfreqs


class ComboResults(BladeStrainFile, OJFLogFile, DspaceMatFile):

    def __init__(self, *args, **kwargs):
        """

        RPM spike is standard removed from the dpspace file

        Parameters for 2 arguments
        --------------------------

        respath

        resfile

        Parameters for 1 argument
        -------------------------

        runid : str
            If only one argument is passed on, it is assumed to be the runid

        database : str, default='symlinks_all'
            In 'database/' different symlink databases exist. Specify which one.

        Other Parameters
        -----------------

        hsanalysis_path : str, default=
            Path to the folder containing the HS analysis results

        hs_symlinks : str, default=
            Name of the symlink folder holding all the raw HS camera footage

        cal : boolean, default=False
            Standard calibration is applied, as set in CalibrationData. Default
            to False for backward compatibility!
        """

        sync = kwargs.get('sync', False)
        silent = kwargs.get('silent', False)
        # default path for the saved HS analysis
        hsanalysis_path = 'data/HighSpeedCamera/raw/'
        self.hsanalysis_path = kwargs.get('hsanalysis_path', hsanalysis_path)
        db_path = 'database/'
        hs_symlinks = kwargs.get('hs_symlinks', 'symlinks_hs_mimer')
        cal = kwargs.get('cal', False)
        self.hs_respath = db_path + hs_symlinks + '/'
        self.debug = kwargs.get('debug', False)
        checkplot = kwargs.get('checkplot', False)

        if len(args) == 2:
            respath = args[0]
            resfile = args[1]
            self.resfile = resfile
            self.respath = respath
        elif len(args) == 1:
            runid = args[0]
            path_db = 'database/'
            database = kwargs.get('database', 'symlinks_all')
            self.respath = path_db + database + '/'
            respath = self.respath
            # and load the full file name from the database index
            # load the database index for the dspace-strain-ojf cases
            FILE = open(path_db + 'db_index_%s_runid.pkl' % database)
            db_index_runid = pickle.load(FILE)
            FILE.close()
            resfile = db_index_runid[runid]
            self.resfile = resfile

            if self.debug:
                print '  runid: %s' % runid
                print 'respath: %s' % respath
                print 'resfile: %s' % resfile

        else:
            raise TypeError, 'ComboResults() takes either 1 or 2 arguments'

        # initialize stats dict
        self.stats = dict()

        # switch to True if the data gets calibrated
        self.dspace_is_cal = False
        self.blade_is_cal = False
        self.dspace_yawcal = False
        self.dspace_towercal = False
        self.dspace_towercal_psicor = False
        self.dspace_strain_is_synced = False
        self.nodspacefile = False
        self.data_resampled = False
        self.azimuth_resetted = False

        # if there is no dspace file, do not do anything
        try:
            self.dspace = DspaceMatFile(respath+resfile+'.mat', silent=silent)
        except IOError:
            logging.warning('no dspace file found, aborting...')
            self.nodspacefile = True
            return None

        self.dspace.remove_rpm_spike(plot=False)

        try:
            self.blade = BladeStrainFile(respath+resfile + '.csv', Fs=512,
                                         silent=silent)
            self.isbladedata = True
        except IOError:
            msg = ' no blade strain file found'
            logging.warning(msg)
            self.isbladedata = False

        # if it is a triggered file, we can synchronize with dspace
        if sync and self.isbladedata and self.blade.filetype == 'triggered':
            try:
                self._sync_strain_dspace(checkplot=checkplot)
            except:
                # print the error message we found
                print sys.exc_info()[0]
                logging.warning('syncing dspace and blade strain failed...')

        try:
            self.ojf = OJFLogFile(respath+resfile, silent=silent)
            self._reset_timings_ojf()
            self.isojfdata = True
            self.windspeed_mean = self.ojf.data[:,4].mean()
        except IOError:
            msg = ' no wind tunnel data found'
            logging.warning(msg)
            print respath+resfile
            self.isojfdata = False

        # default calibration?
        if cal:
            self.calibrate()

        # High speed camera: see if we can load the analyzed data
        try:
            # time from jpg file metadata is not reliable, convert it based
            # on the dspace trigger data
            self._hs_timings(silent=silent)
            self.ishsdata = True
        except IOError:
            logging.warning('no HS analysis found')
            self.ishsdata = False
        except IndexError:
            logging.warning('no HS time analysis available')
            self.ishsdata = True

        # or inherit directly from the classes
        # DspaceMatFile.__init__(self, resultfile + '.mat')
        # but in that case each of the classes methods should have different
        # names. Now they all have a read() method

        # determine average RPM for later reference
        ch_rpm = self.dspace.labels_ch['RPM']

        self.rpm_mean = self.dspace.data[:,ch_rpm].mean()

    def statistics(self, i1=0, i2=-1):
        """
        the selection only makes sense when blade and dspace have the same
        sampling!! Otherwise you have to define the time!
        """

        # data needs both to be synced and resampled when used in combination
        # with only a selection of the data!
        # however, when there is no blade data, this doesn't count of course
        if ( not (self.data_resampled and self.dspace_strain_is_synced)\
           and (i1>0 or i2>0) ) and self.isbladedata:
            msg = 'statistics of a selection only make sense after resampled,'
            msg += 'and synced dspace and strain signals'
            raise ValueError, msg

        self.stats['dspace mean'] = self.dspace.data[i1:i2,:].mean(axis=0)
        self.stats['dspace std'] = self.dspace.data[i1:i2,:].std(axis=0)
        self.stats['dspace min'] = self.dspace.data[i1:i2,:].min(axis=0)
        self.stats['dspace max'] = self.dspace.data[i1:i2,:].max(axis=0)
        self.stats['dspace range'] \
            = self.stats['dspace max'] - self.stats['dspace min']

        if self.isbladedata:
            self.stats['blade mean'] = self.blade.data[i1:i2,:].mean(axis=0)
            self.stats['blade std'] = self.blade.data[i1:i2,:].std(axis=0)
            self.stats['blade min'] = self.blade.data[i1:i2,:].min(axis=0)
            self.stats['blade max'] = self.blade.data[i1:i2,:].max(axis=0)
            self.stats['blade range'] \
                = self.stats['blade max'] - self.stats['blade min']

        if self.isojfdata:
            # it is assumed that the OJF data is always constant over the
            # entire range of the test
            self.stats['ojf mean'] = self.ojf.data.mean(axis=0)
            self.stats['ojf std'] = self.ojf.data.std(axis=0)
            self.stats['ojf min'] = self.ojf.data.min(axis=0)
            self.stats['ojf max'] = self.ojf.data.max(axis=0)
            self.stats['ojf range'] \
                = self.stats['ojf max'] - self.stats['ojf min']

    def _resample(self):
        """
        After Synchronizing blade and dspace, optionally set start and end
        time and sample rate exactly the same for both systems.
        The leading 4 seconds in dspace are cut off as well
        """

        # when there is only OJF and dSPACE data available
        if not self.isbladedata and self.isojfdata and not self.data_resampled:
            SPS = self.dspace.sample_rate
            # maximum time based on the shortest series, int() will basically also
            # floor the result: 60.9 becomes 60
            tmax = int(min([self.ojf.time[-1], self.dspace.time[-1]]))
            tnew = np.linspace(0,tmax,tmax*SPS)

            # interpollate dSPACE on new grid
            dnew = np.ndarray( (len(tnew),self.dspace.data.shape[1]),
                              dtype=np.float64)
            for i in range(self.dspace.data.shape[1]):
                dnew[:,i] = np.interp(tnew, self.dspace.time,self.dspace.data[:,i])
            self.dspace.time = tnew
            self.dspace.data = dnew

            # interpollate OJF on new grid
            dnew = np.ndarray( (len(tnew),self.ojf.data.shape[1]),
                              dtype=np.float64)
            # OJF time is not accurate and reliable
            if len(self.ojf.time) > self.ojf.data.shape[1]:
                self.ojf.time = self.ojf.time[0:self.ojf.data.shape[0]]
            for i in range(self.ojf.data.shape[1]):
                dnew[:,i] = np.interp(tnew, self.ojf.time, self.ojf.data[:,i])
            self.ojf.time = tnew
            self.ojf.data = dnew

            self.dspace.sample_rate = SPS
            self.ojf.sample_rate = SPS

            self.data_resampled = True

            return

        elif not self.isbladedata or not self.dspace_strain_is_synced \
            and self.data_resampled:
            return

        # take the highest sample rate as the prevailing one, and add
        # 20% to avoid AA. Is that realy necessary?
        SPS = int(2*max([self.blade.sample_rate, self.dspace.sample_rate]))
        # maximum time based on the shortest series, int() will basically also
        # floor the result: 60.9 becomes 60
        tmax = int(min([self.blade.time[-1], self.dspace.time[-1]]))
        tnew = np.linspace(0,tmax,tmax*SPS)

        # reset the blade data, only for the channels 1-4
        dnew = np.ndarray((len(tnew),self.blade.data.shape[1]))
        for i in range(self.blade.data.shape[1]):
            dnew[:,i] = np.interp(tnew, self.blade.time, self.blade.data[:,i])
        self.blade.time = tnew
        self.blade.data = dnew
        self.blade.sample_rate = SPS

        # and reset all the dspace data
        dnew = np.ndarray( (len(tnew),self.dspace.data.shape[1]),
                          dtype=np.float64)
        for i in range(self.dspace.data.shape[1]):
            dnew[:,i] = np.interp(tnew, self.dspace.time,self.dspace.data[:,i])
        self.dspace.time = tnew
        self.dspace.data = dnew
        self.dspace.sample_rate = SPS

        # OJF data: is irrelevant, but for the convience of having them in the
        # same DataFrame
        # and reset all the dspace data
        if self.isojfdata:
            dnew = np.ndarray( (len(tnew),self.ojf.data.shape[1]),
                              dtype=np.float64)
            # OJF time is not accurate and reliable
            if len(self.ojf.time) > self.ojf.data.shape[1]:
                self.ojf.time = self.ojf.time[0:self.ojf.data.shape[0]]
            for i in range(self.ojf.data.shape[1]):
                dnew[:,i] = np.interp(tnew, self.ojf.time,self.ojf.data[:,i])
            self.ojf.time = tnew
            self.ojf.data = dnew
            self.ojf.sample_rate = SPS

        self.data_resampled = True

    def _data_window(self, **kwargs):
        """
        Plotting strategy: plot 5 revolutions of a large time series but check
        as follows:

            * calculate averages, and other statistical poeha for the full
            series

            * calculate statistics on the 5 revolutions

            * compare 5 revolutions with full time serie statistics

            * do some frequency analysis on the full, confirm with the 5 rev

        Parameters
        ----------

        nr_rev : int, default=None

        time : list, default=None
            time = [time start, time stop]

        Returns
        -------

        slice_dspace

        slice_ojf

        slice_blade

        window_dspace

        window_ojf

        window_blade

        zoomtype

        time_range


        """

        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)

        # -------------------------------------------------
        # determine zome range if necesary
        # -------------------------------------------------
        time_range = None
        if nr_rev:
            # input is a number of revolutions, get RPM and sample rate to
            # calculate the required range
            time_range = nr_rev/(self.rpm_mean/60.)
            # convert to indices instead of seconds
            i_range = int(self.dspace.sample_rate*time_range)
            window_dspace = [0, time_range]
            # for April dspace starts at -4 seconds, start slice at 0
            i_zero = int(-self.dspace.time[0]*self.dspace.sample_rate)
            slice_dspace = np.r_[i_zero:i_range+i_zero]

            # for the OJF data
            if self.isojfdata:
                i_range = int(self.ojf.sample_rate*time_range)
                window_ojf = [0, time_range]
                # make it a bit larger, so we don't miss the last points.
                # This is due to the low sample rate. We might have a window
                # of 3.0 - 4.1 seconds, but than we only get 3.0-4.0
                slice_ojf = np.r_[0:i_range+5]
            else:
                i_range = None
                window_ojf = None
                slice_ojf = None

            # blade data
            if self.isbladedata:
                i_range = int(self.blade.sample_rate*time_range)
                window_blade = [0, time_range]
                slice_blade = np.r_[0:i_range]
            else:
                i_range = None
                window_blade = None
                slice_blade = None

            zoomtype = '_nrrev_' + format(nr_rev, '1.0f') + 'rev'

        # in some cases the time range is given as a numpy array
        #elif time.any():
        elif type(time).__name__ in ['ndarray', 'list']:
            time_range = time[1] - time[0]

            # for April dspace starts at -4 seconds, start slice at 0 seconds
            # and not at -4 seconds
            i_zero = int(-self.dspace.time[0]*self.dspace.sample_rate)

            i_start = int(time[0]*self.dspace.sample_rate) + i_zero
            i_end = int(time[1]*self.dspace.sample_rate) + i_zero
            slice_dspace = np.r_[i_start:i_end]
            window_dspace = [time[0], time[1]]

            if self.isojfdata:
                i_start = int(time[0]*self.ojf.sample_rate)
                i_end = int(time[1]*self.ojf.sample_rate)+5
                slice_ojf = np.r_[i_start:i_end]
                window_ojf = [time[0], time[1]]
            else:
                i_range = None
                window_ojf = None
                slice_ojf = None

            if self.isbladedata:
                #i_start = int(time[0]*self.blade.sample_rate)
                #i_end = int(time[1]*self.blade.sample_rate)
                # manually find the zero time index, because it is not reliable
                # after the blade syncing with dspace
                i_start = np.abs(self.blade.time - time[0]).argmin()
                i_end = np.abs(self.blade.time - time[1]).argmin()
                slice_blade = np.r_[i_start:i_end]
                window_blade = [time[0], time[1]]
            else:
                i_range = None
                window_blade = None
                slice_blade = None

            zoomtype = '_zoom_%1.1f-%1.1fsec' %  (time[0], time[1])
        else:
            zoomtype = ''
            # for April dspace starts at -4 seconds, start slice at 0
            i_zero = int(-self.dspace.time[0]*self.dspace.sample_rate)
            slice_dspace = np.r_[i_zero:len(self.dspace.data)]
            window_dspace = [0, self.dspace.time[-1]]

            if self.isojfdata:
                slice_ojf = np.r_[0:len(self.ojf.data)]
                window_ojf = [0, self.dspace.time[-1]]
            else:
                i_range = None
                window_ojf = None
                slice_ojf = None

            if self.isbladedata:
                slice_blade = np.r_[0:len(self.blade.data)]
                window_blade = [0, self.dspace.time[-1]]
            else:
                i_range = None
                window_blade = None
                slice_blade = None

        return slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range

    def calibrate(self):
        """
        Use the calibration settings as defined in CalibrationData
        """

        # define the proper Calibration dataset
        if self.resfile.startswith('02'):
            caldict_dspace = CalibrationData.caldict_dspace_02
            caldict_blade = CalibrationData.caldict_blade_02
        elif self.resfile.startswith('04'):
            caldict_dspace = CalibrationData.caldict_dspace_04
            caldict_blade = CalibrationData.caldict_blade_04
        else:
            raise ValueError, 'which month do you mean?? %s' % self.resfile

        # and only calibrate if there is actually data loaded
        if not self.nodspacefile:
            self._calibrate_dspace(caldict_dspace)
        if self.isbladedata:
            self._calibrate_blade(caldict_blade)

    def _calibrate_dspace(self, caldict_dspace, rem_wind=False):
        r"""

        If $\psi_{FA_{max}}$ and $\psi_{SS_0}$ are present in the caldict,
        convert the tower strain to the yaw angle coordinates instead of the
        misaligned FA/SS strain sensor coordinates. Use psi_fa_max and
        psi_ss_0 as keys, and set the path to the file as values. Always
        calibrate both FA and SS at the same time, do not try to split them
        since the $\psi$ tranfsormed coordinates rely on both.

        Parameters
        ----------

        caldict_dspace : dict
            Dictionary with the channel name as key, and corresponding to the
            labels_ch dictionary. Value is the path to the transformation
            polynomial.

        rem_wind : boolean, default=False
            Substract the contribution from wind drag on the tower strain.
            CAN NOT BE USED SINCE THE FEBRUARY TOWER STRAIN MEASUREMENTS
            ARE WORTHLESS

        """

        # if applicable, remove the wind drag contribution on the tower strain
        # do this before applying the force calibration
        if rem_wind:
            logging.warn('wind strain correction not available, remember?')
#            polpath =  'data/'
#            polpath += 'TowerWind/0203_tower_norotor_wire_0_4_20.cal-pol4'
#            pol = np.loadtxt(polpath)
#            # only consider the average wind speed
#            strain_wind = np.polyval(pol, self.windspeed_mean)
#            # substract the average strain value due to wind speed
#            if self.dspace.labels_ch.has_key('Tower Strain For-Aft filtered'):
#                ch = self.dspace.labels_ch['Tower Strain For-Aft filtered']
#                self.dspace.data[:,ch] -= strain_wind
#            if self.dspace.labels_ch.has_key('Tower Strain For-Aft'):
#                ch = self.dspace.labels_ch['Tower Strain For-Aft']
#                self.dspace.data[:,ch] -= strain_wind

        for k in caldict_dspace:
            # channel number, but sometimes the tower FA and SS only has the
            # filtered version
            try:
                ch = self.dspace.labels_ch[k]
            except KeyError:
                try:
                    ch = self.dspace.labels_ch[k + ' filtered']
                except KeyError:
                    # ignore any other keys, they will be used later
                    continue
            # load the transformation polynomial
            pol = np.loadtxt(caldict_dspace[k])
            self.dspace.data[:,ch] = np.polyval(pol, self.dspace.data[:,ch])

            # and mark which one is calibrated
            if caldict_dspace[k].find('towercal') > -1:
                self.dspace_towercal = True
                # keep a reference to them in case we will transfer to psi
                if k.find('For-Aft') > -1:
                    chi_fa = ch
                elif k.find('Side-Side') > -1:
                    chi_ss = ch

            elif caldict_dspace[k].find('yawcal') > -1:
                self.dspace_yawcal = True

        # if psi_fa_max and psi_ss_0 are present in the caldict, convert the
        # tower strain to the yaw angle coordinates instead of the misaligned
        # FA/SS strain sensor coordinates
        if self.dspace_towercal:
            try:
                # load the angles
                psi_fa_max_deg = np.loadtxt(caldict_dspace['psi_fa_max'])
                psi_ss_0_deg = np.loadtxt(caldict_dspace['psi_ss_0'])
                # convert to radians
                psi_fa_max = psi_fa_max_deg * np.pi / 180.0
                psi_ss_0 =   psi_ss_0_deg   * np.pi / 180.0
                # the origina data
                M_FA = self.dspace.data[:,chi_fa]
                M_SS = self.dspace.data[:,chi_ss]
                # and transform to psi and psi_90 coordinates
                M_psi = M_FA * np.cos(psi_fa_max) + M_SS * np.sin(psi_ss_0)
                M_psi90 = -M_FA * np.sin(psi_fa_max) + M_SS * np.cos(psi_ss_0)
                # and replace the original
                self.dspace.data[:,chi_fa] = M_psi
                self.dspace.data[:,chi_ss] = M_psi90
                self.dspace_towercal_psicor = True
            # if M_FA and M_SS doen't exists, they haven't been callibrated
            except NameError:
                msg = 'Psi correction only if FA and SS are callibrated'
                logging.warn(msg)
            # there is no psi transformation to be applied. Carry on plz!
            except KeyError:
                pass

        if self.dspace_yawcal or self.dspace_towercal:
            self.dspace_is_cal = True

    def _calibrate_blade(self, caldict_blade):
        """

        Parameters
        ----------

        caldict_blade : dict
            Dictionary with the channel number, an integer, as key. Value is
            the path to the transformation polynomial.

        """

        for k in caldict_blade:
            # load the transformation polynomial
            pol = np.loadtxt(caldict_blade[k])
            # if there is no blade data, do nothing
            try:
                self.blade.data[:,k] = np.polyval(pol, self.blade.data[:,k])
            except AttributeError:
                return

        self.blade_is_cal = True

    def _set_figure_borders(self, figsize_x, figsize_y, labelspace=1.):
        """
        Set the image borders in terms of cm's instead of %

        labelspace is the space the labels get on left and right, give in cm.
        """

        wsleft, wsright = labelspace/figsize_x, labelspace/figsize_y

        return wsleft, wsright

    def _reset_timings_ojf(self):
        """
        OJF log uses time stamps, let it start at 0 and make it last as long
        as the D-space mat file
        """
        endtime = self.dspace.time[-1]
        # sampling rate of the tunnel data is 4, and set initial value to 0
        self.ojf.time = self.ojf.time[0:int(endtime*4)] - self.ojf.time[0]
        self.ojf.data = self.ojf.data[0:int(endtime*4)]

    def _sync_strain_dspace(self, min_h=0.20, checkplot=False):
        """
        Synchronize MicroStrain signal based on dSPACE clock
        ====================================================

        There is a clock mismatch between the wireless MicroStrain blade strain
        receiver and the dSPACE system. Based on the same 1P pulse occuring in
        both systems, the MicroStrain timings are changed such that they match
        perfectly the dSPACE occurence times.

        Parameters
        ----------

        min_h : float, default=0.2
            Minimum wave height to be considered for peak finding algorithm.
            Both pulses are normalized by the max value of the pulse. Some
            of the noise or other light seen by the sensors stay well below
            the 20% mark. Variation due to yawing also stays above 80%.

        checkplot : boolean, default=False
            if True, present interactive plot of the original and calibrated
            blade/dspace pulse signal

        """
        if self.dspace_strain_is_synced:
            raise UserWarning, 'dspace and blade strain are already synced!'

        # first, check that we have the pulse signal living in both strain
        # and dspace datasets
        if not self.isbladedata \
            or not self.dspace.labels_ch.has_key('RPM Pulse'):
            msg = 'Can only sync blade strain if all data is present'
            raise UserWarning, msg

        # note that the sampling frequency is different in both systems

        # APPROACH 1: determine timings of all dSPACE peaks and bladestrain
        # this is probably the most robust approach
        # add a check to see if the time between each pulse is roughly the
        # same in the original timings.
        # verify if the drift over each pulse is the same

        # channel of the pulse signal in dspace
        ch = self.dspace.labels_ch['RPM Pulse']

        # dspace time
        t_dspace = self.dspace.time[:]
        # dspace pulse channel, nomralized, and only positive since some of
        # the noise goise down to negative and that increases their waveheight
        p_dspace = np.abs(self.dspace.data[:,ch]/self.dspace.data[:,ch].max())
        # just make n as large as the array
        ids=wafo.misc.findpeaks(p_dspace,n=len(p_dspace),min_h=min_h,min_p=0)
        # sort, otherwise it is ordered according to significant wave height
        ids.sort()

        # with the peaks we can now also evaluate the true blade azimuth
        # positions, but approach only works if no resampling has been done!!
        if not self.data_resampled:
            self._reset_azimuth(i_peaks=ids, checkplot=checkplot)

        t_blade = self.blade.time[:]
        p_blade = self.blade.data[:,4]/self.blade.data[:,4].max()
        ib = wafo.misc.findpeaks(p_blade,n=len(p_blade),min_h=min_h, min_p=0)
        ib.sort()

        # time per revolution, based on peak finding times
        # convert to RPM for more easy interpretation
        d_dspace = 60./np.diff(t_dspace[ids])
        d_blade  = 60./np.diff(t_blade[ib])
        # specify tolerances for comparing rotor speeds
        # atol refers to the time in seconds per revolution
        atol = 1.0e-02
        rtol = 4.0e-02
        errorandplot = False

        # only allow there to be a difference of one peak. Assume that this
        # difference appears at the end of the signal. For instance: dSPACE
        # registers the beginning of a new peak but blade strain just fails
        # to log it
        if len(ids) == len(ib):
            # the slices, select all the peaks all
            sld = np.s_[:]
            slb = np.s_[:]

        # in case dspace has one extra peak
        elif len(ids) == len(ib) + 1:
            # let the first peak be the matching one, ignore space last peak
            #t_peaks = t_dspace[ids[:-1]]
            # match last peak, ignore first dspace peak

            # Sometimes it looked like the first peak was lost on the blade
            # signal. By comparing dspace and blade calculated RPM, guess
            # the best match and assume that that can tell you whether or
            # not to cut of the first or last peak
            # which case is better, cutting of the first or last dspace peak?
            # figure out based on the max error found: find best match
#            case1 = np.abs(d_dspace[:-1] - d_blade).max()
#            case2 = np.abs(d_dspace[1:] - d_blade).max()
#            if case1 > case2:
#                # cutting of the first one is better
#                sld = np.s_[1:]
#            else:
#                # cutting of the last one is better
#                sld = np.s_[:-1]

            # why even consider cutting of the first peak? That should be the
            # calibration point. That's where it starts. Otherwise we have
            # one revolution out of sync stuff. But what if there is a small
            # delay between powerup and measure, so dpscae registers power on
            # but the blade sensor is not registering yet?
            sld = np.s_[:-1]
            # and take it all for the blade
            slb = np.s_[:]

        # in case we have two more on dspace, does that mean we missed a peak
        # at the beginning and the end? seems likely after watching the dt
        # per peak: there is no evidence of a missed peak! However, other
        # scenario is that the start is right, but that we miss 2-3 revs at
        # the end! The end is a more likely scenario: the blade sensor is still
        # running on some of the inductance power, so there is some inertia
        # in shutting down. Power up is more rapid.
        # If there where missing peaks on the blade in the middle, they will
        # be catched later on, when matching RPM's
        elif len(ids) >= len(ib) and len(ids) < len(ib)+5:
            # assume the losses are taken at the end of the series
            # cut off the start and end peak for dspace
            cutoff = len(ib) - len(ids)
            sld = np.s_[0:cutoff]
            # and take it all for the blade
            slb = np.s_[:]

        # in case blade has one extra peak, is that ever happening?
        elif len(ids) +1 == len(ib):
            # which case is better, cutting of the first or last dspace peak?
            # figure out based on the max error found
            case1 = np.abs(d_dspace - d_blade[:-1]).max()
            case2 = np.abs(d_dspace - d_blade[1:]).max()
            if case1 > case2:
                # cutting of the first one is better
                slb = np.s_[1:]
            else:
                # cutting of the last one is better
                slb = np.s_[:-1]
             # and take it all for dspace
            sld = np.s_[:]

        # TODO: select the missing peaks on the blade method based on
        # d_blade and d_space difference, it will be a factor of roughly 2
        else:
            print 'peaks dspace: %i' % len(ids)
            print ' peaks blade: %i' % len(ib)

            if checkplot:

                plt.figure()
                plt.title('original data')
                plt.plot(t_blade, p_blade, 'b', label='blade')
                plt.plot(t_blade[ib], p_blade[ib], 'b*')
                plt.plot(t_dspace, p_dspace, 'r', label='dpsace')
                plt.plot(t_dspace[ids], p_dspace[ids], 'r*')
                plt.axhline(y=min_h, color='k')
                plt.legend(loc='best')
                # and the deltas per peak, where is there a missed data point?
                plt.figure()
                plt.title('deltas per peak')
                plt.plot(d_blade, 'b', label='dt blade')
                plt.plot(d_dspace, 'r', label='dt dpsace')
                plt.legend(loc='best')

            raise UserWarning, 'more peaks are missing!'

        # gives each peak a time stamp from dspace
        t_peaks = t_dspace[ids[sld]]
        # and indices to the blade peaks
        i_peaks = ib[slb]

        # and check if time per revolution approximately matches between
        # dspace and blade
        # check that we didn't miss any peaks: one on one compare
        # the rotor speed at each peak occurence. They should be within
        # the specified tolerances
        if not np.allclose(d_dspace[sld], d_blade[slb], rtol=rtol, atol=atol):
        # allclose: absolute(a - b) <= (atol + rtol * absolute(b))
#            errorandplot = True
            logging.warn('pulse derived RPM differs between dspace and blade')
        if checkplot or errorandplot:
            abs_diff = np.abs(d_dspace[sld] - d_blade[slb])
            treshold = atol + rtol * np.abs(d_blade[slb])

        # plot the delta t's and throw an error
        if errorandplot:
            dss, bs = d_dspace.shape,d_blade.shape
            print 'shapes syncing peaks, dspace: %s blade %s' % (dss, bs)
            plt.figure()
            plt.title('comparing dt per revolution')
            plt.plot(d_dspace[sld], 'rs-', label='rpm dspace', alpha=0.6)
            plt.plot(d_blade[slb], 'bo-', label='rpm blade', alpha=0.6)
            plt.legend(loc='upper left')
            # plot the deltas on the left axis
            plt.twinx()
            plt.plot(abs_diff, 'g^-', label='abs diff')
            plt.plot(treshold, 'kd-', label='treshold')
            plt.legend(loc='lower right')
            msg = 'dspace-blade strain derived rotor speeds do not match'
            raise ValueError, msg

        # create all the indices for the blade file, consider them as x_new
        # for the interpolation scheme
        i_all_blade = np.arange(0, len(t_blade), 1, dtype=np.int32)
        # interpolate, consider time blade corrected (t_blade_new) as y_new,
        # all blade indices (i_all_blade) as x_new, indices to blade
        # peaks (i_peaks) as x, and dspace time peaks (t_peaks) as y
        t_blade_new = sp.interpolate.griddata(i_peaks, t_peaks, i_all_blade)

        # visual check on the data points
        if checkplot:
            plt.figure()
            plt.title('original data')
            plt.plot(t_blade, p_blade, 'b', label='blade')
            plt.plot(t_blade[ib], p_blade[ib], 'b*')
            plt.plot(t_dspace, p_dspace, 'r', label='dpsace')
            plt.plot(t_dspace[ids], p_dspace[ids], 'r*')
            plt.axhline(y=min_h, color='k')

            # do we count the same number of pulses
            print 'dspace:', len(ids), 'nr of peaks'
            print ' blade:', len(ib),  'nr of peaks'

            plt.figure()
            plt.title('corrected data')
            plt.plot(t_blade_new, p_blade, 'b', label='blade corrected')
            plt.plot(t_blade_new[ib], p_blade[ib], 'b*')
            plt.plot(t_dspace, p_dspace, 'r', label='dpsace')
            plt.plot(t_dspace[ids], p_dspace[ids], 'r*')

            # see how big the errors are and where
            plt.figure()
            plt.title('comparing dt per revolution')
            plt.plot(d_dspace[sld], 'rs-', label='rpm dspace', alpha=0.6)
            plt.plot(d_blade[slb], 'bo-', label='rpm blade', alpha=0.6)
            plt.legend(loc='upper left')
            # plot the deltas on the left axis
            plt.twinx()
            plt.plot(abs_diff, 'g^-', label='abs diff')
            plt.plot(treshold, 'kd-', label='treshold')
            plt.legend(loc='lower right')

        # overwrite the blade time with the new time, but ditch the first and
        # last points to avoid possibly nan values. BUT WHY ARE THERE ANY NANS?
        # FIXME: why can there by any nan's at the beginning? Probably because
        # of the interpolation that has been done?
        inew = np.isnan(t_blade_new).__invert__()
        self.blade.time = t_blade_new[inew]
        self.blade.data = self.blade.data[inew,:]
        # and reset the sample rate
        self.blade.Fs = calc_sample_rate(self.blade.time, rel_error=1e-1)

        self.dspace_strain_is_synced = True

    def _reset_azimuth(self, i_peaks=False, checkplot=False):
        """
        Use the same azimuthal angle definition as used for the HAWC2
        simulations: -180 degrees is down (in the tower shadow). However,
        convert both HAWC2 and OJF to not breaking the plot at the tower
        shadow. That is the most interesting part of the plot. So make a phase
        shift of 180 degrees

        This scenario is only valid for April, and ONLY WORKS if there hasn't
        been ANY RESAMPLING applied to the signal.

        Note that for both OJF and HAWC2, azimuth angle always increasing.
        However, the OJF is rotating vector is pointed upwind, while the HAWC2
        rotation vector is pointed downward. Make sure to revert the direction
        for the HAWC2 results.

        Parameters
        ----------

        i_peaks : ndarray(n), default=False
            Indices to the RPM Pulse peaks. These indicate the 3 o'clock
            position of blade 1. If False, the peaks will be obtained in the
            same way as used in sync_strain_dspace
        """

        if self.data_resampled:
            raise ValueError('reset_azimuth only works before resampling')

        if type(i_peaks).__name__ == 'bool':
            # channel of the pulse signal in dspace
            ch = self.dspace.labels_ch['RPM Pulse']
            # dspace pulse channel, nomralized, and only positive since some of
            # the noise goise down to negative and that increases waveheight
            pulse = np.abs(self.dspace.data[:,ch]/self.dspace.data[:,ch].max())
            # just make n as large as the array
            i_peaks = wafo.misc.findpeaks(pulse,n=len(pulse),min_h=0.2,min_p=0)
            # sort, otherwise ordered according to significant wave height
            i_peaks.sort()

        i_azi = self.dspace.labels_ch['Azimuth']
        # now see if each pulse corresponds to the same azi signal. It should!
        azi = self.dspace.data[:,i_azi]

        # in theory, azi should be one constant value. However, because of
        # the innacuracies, the azi position varies within a band of 3-4 deg at
        # high rpm's. Band at low RPM is much smaller: 1 degrees or less.
        # Additionally, a rpm peak spike causes a phase shift. In order to
        # anticipate all this, just consider the azi position as a variable.
        # Hence sample these positions to the same rate as dspace again
        time = self.dspace.time

        # peaks on the azimuth signal
        azip = wafo.misc.findpeaks(azi, n=len(azi), min_h=15, min_p=0)
        # sort, otherwise ordered according to significant wave height
        azip.sort()

        # except for the last peak, they should all be close to 360 degrees!
        # if it is lower, we now where the jump happens
        iijump = azi[azip[:-1]].argmin()
        ijump = azip[iijump]
        # take the next peak to see how many degrees the shift is
        inext = azip[iijump+1]

        # now only reset the azimuth angle if there actuall is a jump
        # other indication is if there was an RPM spike
        azi_reset = azi.copy()
        if self.dspace.rpm_spike_removed or azi[ijump] < 350:
            # this is why we have to reset the azimuth before any resampling
            # happens: the next point is good again, assume the azi change is
            # the same over the previous step
            d_azi_prev = azi[ijump] - azi[ijump-1]
            # the actual jump
            d_azi_jump = azi[ijump] - azi[ijump+1]
            # and apply the correction: anticipated delta + jump due to error
            azi_reset[ijump+1:] += d_azi_jump + d_azi_prev
            # and back to 0-360 boundaries
            azi_reset[azi_reset.__ge__(360.0)] -= 360.0

        # blade 3 was always synced with the pulse system, because on that
        # blade no strain gauge wires where routed into the hub. Change
        # azimuth such that blade 3 up means azi=0
        # but make sure the 90 deg azimuth position from the RPM pulse is
        # within an error range of 7 degrees.
        if azi_reset[i_peaks].max() - azi_reset[i_peaks].min() < 7.0:
            azi_reset += - azi_reset[i_peaks].mean() + 90.0
            azi_reset[azi_reset.__lt__(0.0)] += 360.0
            self.azimuth_resetted = True
        # there is a large difference between the different selected RPM pulse
        # positions (azi=90 deg)
        else:
            logging.warn('azimuth correction FAILED...')
            self.azimuth_resetted = False

        if checkplot:
            irpm = self.dspace.labels_ch['RPM']
            plt.figure('ComboResults._reset_azimuth')
            # also do blade 1 root bending for tower shadow detection
#            plt.plot(self.blade.time, self.blade.data[:,2]+100, 'k')
            plt.plot(self.dspace.time, self.dspace.data[:,i_azi], 'y*-')
            plt.plot(time[azip], azi[azip], 'bo')
            plt.plot(time[ijump], azi[ijump], 'rs')
            plt.plot(time[inext], azi[inext], 'rv')
            plt.plot(self.dspace.time, azi_reset, 'g--')
            plt.plot(self.dspace.time, self.dspace.data[:,irpm], 'b--')
            plt.plot(self.dspace.time[i_peaks], azi[i_peaks], 'r')

#            plt.plot(self.dspace.time[i_peaks], azi_reset[i_peaks], 'm')
            plt.grid()

        # and check that the measured 3 o'clock position is within a band with
        # of 5 degrees
        if self.azimuth_resetted:
            self.dspace.data[:,i_azi] = azi_reset


    def __opt_call(self, Fs):
        """
        call for the optimzer
        """

        data = self.blade.data[:,0]
#        rpm1 = self.rpm1

        Pxx, freqs = mpl.mlab.psd(data, NFFT=16*8192, Fs=Fs)
        # find the 1P from the PSD analysis
        sel = freqs.__ge__(self.rpm1/60. -1.)
        freqs_sel = freqs[sel]
        Pxx_sel = Pxx[sel]

#        print self.rpm1, Fs, data.shape, freqs.shape, freqs_sel.shape

        try:
            freq_1p = freqs_sel[Pxx_sel.argmax()]
        # in case we have a value error, return a large number so the
        # optimizer stays away from this area
        except ValueError:
            return 100.

        return abs(freq_1p - self.rpm1/60.)

    def _tune_blade_sample_rate(self):
        """
        Optimize for sample rate Fs, such that the PSD analysis of the blade
        strain signal has the same 1P frequency as the rotor speed
        initial value for Fs is set to 512
        """
#        # convert to wafo time object
#        data = self.blade.data[:,0]
#        wafo_sig = np.transpose(np.array([self.blade.time, data]))
#        sig_ts = wafo.objects.mat2timeseries(wafo_sig)
#        # the turning_points method in wafo.objects
#        # set h to appropriate value if only high amplitude cycle count
#        sig_tp = sig_ts.turning_points(wavetype=None, h=10.0)

        # mean rotor speed
        rpm1_ch = self.dspace.labels_ch['RPM Estimator v1']
        self.rpm1 = self.dspace.data[:,rpm1_ch].mean()

        # this only makes sense if the signal is always very close to the mean
        if self.dspace.data[:,rpm1_ch].std()/self.rpm1 > 0.02:
            msg = 'Std on RPM signal too large for tuning with blade Fs'
            raise ValueError, msg

        # optimize for sample rate Fs, such that the PSD analysis of the blade
        # strain signal has the same 1P frequency as the rotor speed
        # initial value for Fs is set to 512
        x0 = np.array([512])

        # fmin_cg finished ok, but we want to constrain the optimization and
        # than there was a ValueError raised on the unpacking of the function
        # variables.
#        xopt, fopt, func_calls, grad_calls, warnflag, allvecs \
#         = opt.fmin_cg(self.__opt_call, x0, gtol=1e-05, full_output=True,
#                       epsilon=1.4901161193847656e-08)

#        def optfunc(x):
#            return self.__opt_call(*x)

        x, f, d = sp.optimize.fmin_l_bfgs_b(self.__opt_call, x0, fprime=None,
                      args=(), approx_grad=True, bounds=[(400, 600)], m=10,
                      factr=1.0e7, pgtol=1e-05, epsilon=1e-08, maxfun=15000,
                      disp=None, iprint=False)

#        print opt.fmin_cg(self.__opt_call, x0, gtol=1e-05, full_output=True,
#                       epsilon=1.4901161193847656e-08)

        return x, f, d
#        return xopt, fopt, func_calls, grad_calls, warnflag, allvecs

    def _hs_timings(self, silent=False):
        """
        Based on the HS camera metadata and the triggers from dspace,
        estimate the time stamp of each frame

        hs : ndarray(10,n)
            [boxdeg, deg, chord, blade_nr, lex, ley, tex, tey,
             frame nr, trigger count, trigger frame number, time]
        """

        chtrigg_hs = 9
        chi_tfn_hs = 10
        chi_t_hs = 11
        chi_fn_hs = 8

        # load the analysis file
        runid = '_'.join(self.resfile.split('_')[0:3])
        self.hs = np.loadtxt(self.hsanalysis_path+runid)

        if not silent:
            print '\n'+'*'*80
            print 'LOADING HS ANALYSIS FILE'
            print self.hsanalysis_path+runid
            print 'hs.shape:', self.hs.shape

        # load the metadata file
        hsfiles = sorted(os.listdir(self.hs_respath+self.resfile))
        metapath = self.hs_respath + self.resfile + '/' + hsfiles[0]
        metadata = HighSpeedCamera().metadata(metapath)
        fps = metadata['FPS']

        if not silent:
            print '\n'+'*'*80
            print 'LOADING HS METADATA FILE'
            print metapath

        # get the timings of the start of a trigger session
        chtrig_ds = self.dspace.labels_ch['HS trigger']
        wafo_sig = np.transpose(np.array([self.dspace.time,
                                         self.dspace.data[:,chtrig_ds]]))
        # the turning points give the start (bottomg of the square wave) and
        # the end of the square wave. So only take 1 of 2 points. First point
        # is the starting point
        sig_ts = wafo.objects.mat2timeseries(wafo_sig)
        # signal goes from 0 to 1 in square waves
        sig_tp = sig_ts.turning_points(wavetype=None, h=0.2)
        trig_time = sig_tp.args[0::2]

        if self.debug:
            # check if we realy have taken the starting points
            plt.figure()
            plt.plot(self.dspace.time, self.dspace.data[:,chtrig_ds])
            plt.plot(trig_time, sig_tp.data[0::2], 'ro')

        # do we count the same number of triggers on the HS data?
        nrtrigg_hs = self.hs[chtrigg_hs,-1]
        nrtrigg_dspace = len(trig_time)
        # note that the trigger on dspace usually continued further, while
        # the camera's memory was already full
        if not nrtrigg_hs <= nrtrigg_dspace:
            print '    nrtrigg_hs:', nrtrigg_hs
            print 'nrtrigg_dspace:', nrtrigg_dspace
            logging.warn('more triggers counted in HS than in dspace...abort')
        else:
            # each trigger number refers to the index of the trigger time
            # in dspace
            itrigg = np.array(self.hs[chtrigg_hs,:]-1, dtype=np.int)
            times_trigg = trig_time[itrigg]
            # now the corrected time is time_dspace + (trigg_frame_count/FPS)
            self.hs[chi_t_hs,:] = times_trigg + (self.hs[chi_tfn_hs,:]/fps)

        # some more checkking: plot the triggers in dspace and the proposed
        # timings per frame
        if self.debug:
            plt.figure()
            # scale the trigger to the maxim frame number count
            aa = self.hs[chi_fn_hs,-1]
            plt.plot(self.dspace.time, aa*self.dspace.data[:,chtrig_ds])
            # plot the time vs frame number
            plt.plot(self.hs[chi_t_hs,:], self.hs[chi_fn_hs,:], 'ro')


    def blade1_azimuth_ts(self):
        """
        Mark blade 1 tower shadow positions
        ===================================

        Based on the pulse, we know when blade 1 is at the tower shadow.
        """
        # at t=0, we have the first pulse, meaning blade 3 at 3 o'clock
        ch = self.dspace.labels_ch['Azimuth']
        blade3_90deg = self.dspace.data[0,ch]
        blade1_ts = blade3_90deg - 30. - 120.

        if blade1_ts >   359.9999999:
            blade1_ts -= 359.999999
        elif blade1_ts < 0.0:
            blade1_ts += 359.999999

        # now determine at which indices, so at what moment in time, we have
        # blade1 tower shadow passage

        return blade1_ts

    def plot_azi_vs_bladeload(self):
        """
        Only works if azimuth is resetted and the signal has been resampled
        """

        if not self.azimuth_resetted or not self.data_resampled:
            print 'can only plot azi vs blade load if azi reset and resampled'
            return

        i_azi = self.dspace.labels_ch['Azimuth']

        plt.figure('ComboResults._reset_azimuth: azi vs blade 1 load')
        plt.plot(self.dspace.data[:,i_azi], self.blade.data[:,2], 'b+')
        # vertical lines for possible tower shadow events
        plt.axvline(x=180, color='k') # blade 3 tower shadow
        plt.axvline(x=60,  color='b') # blade 1 tower shadow
        plt.axvline(x=300, color='r') # blade 2 tower shadow
        plt.grid()

        plt.figure('ComboResults._reset_azimuth: azi vs blade 2 load')
        plt.plot(self.dspace.data[:,i_azi], self.blade.data[:,0], 'r+')
        # vertical lines for possible tower shadow events
        plt.axvline(x=180, color='k') # blade 3 tower shadow
        plt.axvline(x=60,  color='b') # blade 1 tower shadow
        plt.axvline(x=300, color='r') # blade 2 tower shadow
        plt.grid()

    def plot_azi_loads(self):
        """
        Plot loads vs azimuth angle, use sensible azimuth angle defintions:
        azi=0 -> up, 180 -> down

        This is based on ojfvshawc2.plot_blade_vs_azimuth
        """

        if not self.azimuth_resetted or not self.data_resampled:
            print 'can only plot azi vs blade load if azi reset and resampled'
            return

        def delay_time():
            # add a delay to the measurements of a fixed amount of time. This
            # will result in a azimuth phase shift as function of rotor speed.
            # benchmark case: 190 deg at 657 rpm results in a time delay of:
            timeoffset = 0.035514967021816335
            timeoffset = 140.0*np.pi/(180.0*657.0*np.pi/30.0)
            # which can be calculated with:
            #timeoffset = ojf_azi_delay*np.pi/(180.0*ojf_rpm.mean()*np.pi/30.0)
            # or explicitaly
            #190*np.pi/(180.0*657.0*np.pi/30.0)
            # and results in the following azimuth delay
            ojf_azi_delay = (timeoffset*ojf_rpm.mean()*np.pi/30.0)*180.0/np.pi
            ojf_azi_corr  = ojf_azi + ojf_azi_delay
            ojf_azi_corr[ojf_azi_corr.__ge__(360.0)] -= 360.0

            return ojf_azi_corr

        def plot_on_ax(ax, hawc2_azi, hawc2_blade, ojf_azi, ojf_blade):
            """
            These are generic for each blade plot
            """
            ax.plot(ojf_azi, ojf_blade, 'wo', label='OJF')
            ax.plot(hawc2_azi, hawc2_blade, 'r+', label='HAWC2')
            ax.axvline(x=180, color='k')
            ax.axvline(x=190, color='k', linestyle='--')
            ax.axvline(x=200, color='k', linestyle='--')
            ax.axvline(x=210, color='k', linestyle='--')
            ax.axvline(x=220, color='k', linestyle='--')

            # rotor speed ontop of it
            divider = plotting.make_axes_locatable(ax)
            height = pwy*0.2/pa4.oneinch
            ax2 = divider.append_axes("top", height, pad=0.2, sharex=ax)
            ax2.plot(ojf_azi, ojf_rpm, 'wo')
            ax2.grid()
            ax2.set_ylabel('RPM')
            # remove the xlabels (time)
            plotting.mpl.artist.setp(ax2.get_xticklabels(), visible=False)
            # reduce number of RPM ticks
            ylim2 = ax2.get_ylim()
            ax2.yaxis.set_ticks( np.linspace(ylim2[0], ylim2[1], 3).tolist() )

            ax.grid(True)
            ax.legend(loc='best')
            ax2.set_title(title)
            ax.set_xlabel('Azimuth position [deg]')
            if normalize:
                ax.set_ylabel('Normalized Bending moment')
            else:
                ax.set_ylabel('Bending moment [Nm]\nzero base offset')
            ax.set_xlim([0, 360])

        def next_blade(azi):
            # change azimuth to next blade
            azi += 120.0
            azi[azi.__ge__(360.0)] -= 360.0
            return azi

        i_azi = self.dspace.labels_ch['Azimuth']
        ojf_azi = self.dspace.data[:,i_azi]
        # and now to a more easy to read definition: 180 deg blade 1 down
        # now 60 degrees is tower shadow, we want 180 tower shadow, so we are
        # lagging 120 degrees
        ojf_azi   += 120.0
        ojf_azi[ojf_azi.__ge__(360.0)] -= 360.0
        if delay:
            ojf_azi_corr = delay_time()
            ojf_azi = ojf_azi_corr

        # ---------------------------------------------------------------------
        # plotting properties
        # ---------------------------------------------------------------------
        scale = 1.8
        eps = True
        pwx = plotting.TexTemplate.pagewidth*0.99
        pwy = plotting.TexTemplate.pagewidth*0.5
        # number of plots
        nr = 1
        wsb = 1.0
        wsl = 1.8
        wsr = 0.5
        hs  = 2.0
        wst = 1.1

        yawerror = 'Yaw error %1.1f deg' % cao.cases[cname]['[yaw_angle_misalign]']
        rpm = '%i rpm' % (cao.cases[cname]['[fix_wr]']*30.0/np.pi)
        wind = 'wind %1.1f m/s' % cao.cases[cname]['[windspeed]']
        wr = cao.cases[cname]['[fix_wr]']
        v = cao.cases[cname]['[windspeed]']
        tsr = '$\lambda = %1.1f$' % (ojf_post.model.blade_radius * wr / v)



    def plot_all_raw(self, figpath):
        """
        Simple plots for all 3 data sets in different plots
        """
        pa4 = plotting.A4Tuned()
        figfile = figpath + self.resfile + '_dspace'

        # do not plot all form dSPACE
        # TODO: use more robust approach with
        # channels.apend(self.dspace.ch_dict['RPM Estimater v1'])
        channels = [0, 1, 3, 6, 7, 9, 10, 11, 8, 13, 14, 15]
        pa4.plot_simple(figfile+'_p1', self.dspace.time, self.dspace.data,
                self.dspace.labels, channels=channels, grandtitle=self.resfile)

        channels = [2, 4, 5, 12, 16, 17]
        pa4.plot_simple(figfile+'_p2', self.dspace.time, self.dspace.data,
                self.dspace.labels, channels=channels, grandtitle=self.resfile)

        pa4.plot_simple(figfile, self.blade.time, self.blade.data,
                self.blade.labels, grandtitle=self.resfile)

        pa4.plot_simple(figfile, self.ojf.time, self.ojf.data,
                self.ojf.labels, grandtitle=self.resfile)

    def freeyaw_compact(self, figpath, grandtitle, **kwargs):
        """
        Compact free yaw plot for the Torque 2012 paper: rpm and yaw angle.
        """
        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)

        if not self.dspace_is_cal and not self.blade_is_cal:
            rawdata = '_raw'
        else:
            rawdata = '_cal'

        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self._data_window(nr_rev=nr_rev, time=time)

        # -------------------------------------------------
        # setup the figure
        # -------------------------------------------------
        nr_plots = 1
        pa4 = plotting.A4Tuned()
        fig_descr = rawdata + '_dashb_freeyaw_comp' + zoomtype
        pa4.setup(figpath + self.resfile + fig_descr, nr_plots=nr_plots,
                   grandtitle=grandtitle, wsleft_cm=2., wsright_cm=2.0,
                   hspace_cm=2., figsize_y=9., wstop_cm=1.6)

        # -------------------------------------------------
        # plotting dSPACE selection
        # -------------------------------------------------

        ch_rpm = self.dspace.labels_ch['RPM']
        ch_yaw = self.dspace.labels_ch['Yaw Laser']

        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.set_title('Rotor speed and yaw angle')

        ax1.plot(self.dspace.time, self.dspace.data[:,ch_yaw], 'r--',
                 label='Yaw error')
        ax1.set_ylabel('Yaw angle [deg]')
        ax1.grid(True)
        #ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(self.dspace.time, self.dspace.data[:,ch_rpm], 'b',
                 label='RPM')
        ax2.set_ylabel('rotor speed [RPM]')
        #ax2.legend(loc='lower right')

        # put both plot labels in one legend
        # plot on the 2nd axis so the legend is always on top!
        lines = ax1.lines + ax2.lines
        labels = [l.get_label() for l in lines]
        leg = ax2.legend(lines, labels, loc='best')

        # or alternatively ask for the plotted objects and their labels
        #lines, labels = ax1.get_legend_handles_labels()
        #lines2, labels2 = ax2.get_legend_handles_labels()

        ax1, ax2 = plotting.match_axis_ticks(ax1, ax2, ax2_format='%1.0f')

        if time_range:
            ax1.set_xlim(window_dspace)

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig()


    def freeyaw(self, figpath, **kwargs):
        """
        Plot OJF free yaw focus dashboard on A4 based size
        ==================================================
        """

        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)

        if not self.dspace_is_cal and not self.blade_is_cal:
            rawdata = '_raw'
        else:
            rawdata = '_cal'

        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self._data_window(nr_rev=nr_rev, time=time)

        # -------------------------------------------------
        # setup the figure
        # -------------------------------------------------
        nr_plots = 4
        pa4 = plotting.A4Tuned()
        fig_descr = rawdata + '_dashb_freeyaw' + zoomtype
        pa4.setup(figpath + self.resfile + fig_descr, nr_plots=nr_plots,
                   grandtitle=self.resfile, wsleft_cm=2., wsright_cm=1.5,
                   hspace_cm=2., figsize_y=19.)

        # -------------------------------------------------
        # data selection from dSPACE
        # -------------------------------------------------
        channels = []

        channels.append(self.dspace.labels_ch['RPM'])
#        channels.append(self.dspace.labels_ch['Power'])
        channels.append(self.dspace.labels_ch['Tower Strain For-Aft'])
        channels.append(self.dspace.labels_ch['Tower Strain Side-Side'])
#        channels.append(self.dspace.labels_ch['Tower Top acc Y (FA)'])
#        channels.append(self.dspace.labels_ch['Tower Top acc X (SS)'])
        channels.append(self.dspace.labels_ch['Yaw Laser'])
#        channels.append(self.dspace.labels_ch['Tower Top acc Z'])

        # labels for the calibrated signals
        ylabels = dict()
        ylabels['Yaw Laser'] = '[deg]'
        ylabels['Tower Strain For-Aft'] = 'Force tower top [N]'
        ylabels['Tower Strain Side-Side'] = 'Force tower top [N]'


        # -------------------------------------------------
        # plotting dSPACE selection
        # -------------------------------------------------
        time = self.dspace.time[slice_dspace]
        data = self.dspace.data[slice_dspace,:]
        plot_nr = 1
        for ch in channels:
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax1.plot(time, data[:,ch], 'b')
            ax1.set_title(self.dspace.labels[ch])
            ax1.grid(True)
            if plot_nr < len(channels)-1:
                # clear all x-axis values except the bottom subplots
                ax1.set_xticklabels([])

            ch_key = self.dspace.labels[ch]
            if ylabels.has_key(ch_key):# and caldict_dspace.has_key(ch_key):
                ax1.set_ylabel(ylabels[self.dspace.labels[ch]])

            if time_range:
                ax1.set_xlim(window_dspace)

            plot_nr += 1

#        # -------------------------------------------------
#        # plotting OJF selection
#        # -------------------------------------------------
#        time = self.ojf.time[slice_ojf]
#        data = self.ojf.data[slice_ojf,:]
#        # temperature
#        ch_t = 1
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,ch_t], 'r', label=self.ojf.labels[ch_t])
#        ax1.set_ylabel(self.ojf.labels[ch_t])
#        ax1.grid(True)
#        leg = ax1.legend(loc='upper left')
#        leg.get_frame().set_alpha(0.5)
#        # and wind speed on the right axes
#        ch_w = 4
#        ax2 = ax1.twinx()
#        ax2.plot(time, data[:,ch_w], 'b', label=self.ojf.labels[ch_w])
#        ax2.set_ylabel(self.ojf.labels[ch_w])
#        leg = ax2.legend(loc='upper right')
#        leg.get_frame().set_alpha(0.5)
#
#        ax1.set_title(self.ojf.labels[ch_t] + ', ' + self.ojf.labels[ch_w])
#
#        plot_nr += 1
#        if time_range:
#            ax1.set_xlim(window_ojf)

#        # -------------------------------------------------
#        # plotting Blade selection
#        # -------------------------------------------------
#        # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
#        # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)
#        time = self.blade.time[slice_blade]
#        data = self.blade.data[slice_blade,:]
#
#        # Blade root sections
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,2], 'm', label='blade 1 root')
#        ax1.plot(time, data[:,0], 'k', label='blade 2 root')
#        ax1.set_title('blade root sections')
#        ax1.grid(True)
#        leg = ax1.legend(loc='upper left')
#        leg.get_frame().set_alpha(0.5)
#        plot_nr += 1
#        if time_range:
#            ax1.set_xlim(window_blade)
#
#        # Blade mid sections
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,3], 'm', label='blade 1 30%')
#        ax1.plot(time, data[:,1], 'k', label='blade 2 30%')
#        ax1.set_title('blade 30% sections')
#        ax1.grid(True)
#        leg = ax1.legend(loc='upper left')
#        leg.get_frame().set_alpha(0.5)
#        if time_range:
#            ax1.set_xlim(window_blade)
#        plot_nr += 1

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig()

    def dashb_ts(self, figpath, grandtitle, **kwargs):
        """
        Plot OJF tower shadow dashboard
        ===============================

        Two plots: top RPM, lower blade 2 root and 30%

        """

        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)

        if not self.dspace_is_cal and not self.blade_is_cal:
            rawdata = '_raw'
        else:
            rawdata = '_cal'

        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self._data_window(nr_rev=nr_rev, time=time)

        # -------------------------------------------------
        # setup the figure
        # -------------------------------------------------
#        if self.dspace.campaign == 'February':
#            nr_plots = 10
#            nr_plots = 7
#        else:
#            nr_plots = 11
#            nr_plots = 8

        nr_plots = 2

        pa4 = plotting.A4Tuned()
        fig_descr = rawdata + '_dashb_TS' + zoomtype
        pa4.setup(figpath + self.resfile + fig_descr, nr_plots=nr_plots,
                   grandtitle=grandtitle, wsleft_cm=2., wsright_cm=1.5,
                   hspace_cm=2., figsize_y=16.)

        # -------------------------------------------------
        # data selection from dSPACE
        # -------------------------------------------------
        channels = []

        channels.append(self.dspace.labels_ch['RPM'])
#        channels.append(self.dspace.labels_ch['Yaw Laser'])
#        channels.append(self.dspace.labels_ch['Power'])
#        channels.append(self.dspace.labels_ch['Tower Strain For-Aft'])
#        channels.append(self.dspace.labels_ch['Tower Strain Side-Side'])
#        channels.append(self.dspace.labels_ch['Tower Top acc Y (FA)'])
#        channels.append(self.dspace.labels_ch['Tower Top acc X (SS)'])
#        channels.append(self.dspace.labels_ch['Yaw Laser'])
#        channels.append(self.dspace.labels_ch['Tower Top acc Z'])

        # convert labels for calibrated signals
        ylabels = dict()
        ylabels['Yaw Laser'] = '[deg]'
        ylabels['Tower Strain For-Aft'] = 'Force tower top [N]'
        ylabels['Tower Strain Side-Side'] = 'Force tower top [N]'

        # -------------------------------------------------
        # plotting dSPACE selection
        # -------------------------------------------------
        time = self.dspace.time[slice_dspace]
        data = self.dspace.data[slice_dspace,:]
        plot_nr = 1
        for ch in channels:
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax1.plot(time, data[:,ch], 'b')
            ax1.set_title(self.dspace.labels[ch])
            ax1.grid(True)
            if plot_nr < len(channels)-1:
                # clear all x-axis values except the bottom subplots
                ax1.set_xticklabels([])

            ch_key = self.dspace.labels[ch]
            if ylabels.has_key(ch_key):# and caldict_dspace.has_key(ch_key):
                ax1.set_ylabel(ylabels[self.dspace.labels[ch]])

            ax1.set_xlim(window_dspace)

            plot_nr += 1

        # overlapping pulse plotting
#        ax2 = ax1.twinx()
#        ch_pulse = self.dspace.labels_ch['RPM Pulse']
#        data_pulse = self.dspace.data[slice_dspace,ch_pulse]
#        ax2.plot(time, data_pulse)
#        ax2.plot(self.blade.time[slice_blade], self.blade.data[slice_blade,4])
#        ax1.set_xlim(window_dspace)

        # -------------------------------------------------
        # plotting OJF selection
        # -------------------------------------------------
#        time = self.ojf.time[slice_ojf]
#        data = self.ojf.data[slice_ojf,:]
#        # temperature
#        ch_t = 1
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,ch_t], 'r', label=self.ojf.labels[ch_t])
#        ax1.set_ylabel(self.ojf.labels[ch_t])
#        ax1.grid(True)
#        leg = ax1.legend(loc='upper left')
#        leg.get_frame().set_alpha(0.5)
#        # and wind speed on the right axes
#        ch_w = 4
#        ax2 = ax1.twinx()
#        ax2.plot(time, data[:,ch_w], 'b', label=self.ojf.labels[ch_w])
#        ax2.set_ylabel(self.ojf.labels[ch_w])
#        leg = ax2.legend(loc='upper right')
#        leg.get_frame().set_alpha(0.5)
#
#        ax1.set_title(self.ojf.labels[ch_t] + ', ' + self.ojf.labels[ch_w])
#
#        plot_nr += 1
#        ax1.set_xlim(window_ojf)

#        # windspeed
#        ch = 4
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(self.ojf.time, self.ojf.data[:,ch], 'r')
#        ax1.set_title(self.ojf.labels[ch_w])
#        ax1.grid(True)
#        plot_nr += 1
#        if time_range:
#            ax1.set_xlim(window_ojf)


#        # only relevant for April: overlap pulse
#        if self.dspace.campaign == 'April':
#            time = self.blade.time[slice_blade]
#            data = self.blade.data[slice_blade,:]
#            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#            ax1.plot(time, data[:,4], 'k', label='pulse strain')
#
#            ch = self.dspace.labels_ch['RPM Pulse']
#            time = self.dspace.time[slice_dspace]
#            data = self.dspace.data[slice_dspace,ch]
#
#            ax1.plot(time, data, 'm--', label='pulse dspace')
#            ax1.set_title('overlap pulse')
#
#            ch = self.dspace.labels_ch['Azimuth']
#            data = self.dspace.data[slice_dspace,ch]
#            ax1.plot(time, data/180., 'k', alpha=0.5)
#
#            # dirty hack to get al tower shadow pos for blade1
#            blade1_ts = self.blade1_azimuth_ts()
#            above = False
#            for k in xrange(len(data)):
#                if data[k] >= blade1_ts and not above:
#                    above = True
#                    ax1.axvline(x=time[k], linewidth=1, color='k',\
#                        linestyle='--', aa=False)
#                elif data[k] < 0.5:
#                    above = False
#
#
#            ax1.grid(True)
#            leg = ax1.legend(loc='lower right')
#            leg.get_frame().set_alpha(0.5)
#            ax1.set_xlim(window_blade)
#            plot_nr += 1

        # -------------------------------------------------
        # plotting Blade selection
        # -------------------------------------------------
        # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
        # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)
        time = self.blade.time[slice_blade]
        data = self.blade.data[slice_blade,:]

        # Blade root sections
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,2], 'k', label='blade 1 root')
#        ax1.plot(time, data[:,3], 'm', label='blade 1 30%')
        ax1.plot(time, data[:,0], 'k', label='blade 2 root')
        ax1.plot(time, data[:,1], 'm', label='blade 2 30\%')
        ax1.set_title('blade 2 strains')
        ax1.grid(True)
        leg = ax1.legend(loc='lower right')
        leg.get_frame().set_alpha(0.5)
        plot_nr += 1
        ax1.set_xlim(window_blade)

        # dirty hack to get al tower shadow pos for blade1
#        time_ds = self.dspace.time[slice_dspace]
#        ch_azi = self.dspace.labels_ch['Azimuth']
#        data_azi = self.dspace.data[slice_dspace,ch_azi]
#        blade1_ts = self.blade1_azimuth_ts()
#        print '  ------> blade1_ts:', blade1_ts
#        above = False
#        for k in xrange(len(data_azi)):
#            if data_azi[k] >= blade1_ts and not above:
#                above = True
#                ax1.axvline(x=time_ds[k], linewidth=1, color='k',\
#                    linestyle='--', aa=False)
#            elif data_azi[k] < 5.:
#                above = False


#        # Blade mid sections
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,3], 'm', label='blade 1 30%')
#        ax1.plot(time, data[:,1], 'k', label='blade 2 30%')
#        ax1.set_title('blade 30% sections')
#        ax1.grid(True)
#        leg = ax1.legend(loc='lower right')
#        leg.get_frame().set_alpha(0.5)
#        ax1.set_xlim(window_blade)
#
#        # dirty hack to get al tower shadow pos for blade1
#        above = False
#        for k in xrange(len(data_azi)):
#            if data_azi[k] >= blade1_ts and not above:
#                above = True
#                ax1.axvline(x=time_ds[k], linewidth=1, color='k',\
#                    linestyle='--', aa=False)
#            elif data_azi[k] < 1.:
#                above = False
#
#        plot_nr += 1

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig()

    def dashb_blade_yaw(self, figpath, grandtitle, **kwargs):
        """
        Plot RPM and yaw on top plot, blade strain on bottom plot
        =========================================================


        """

        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)

        if not self.dspace_is_cal and not self.blade_is_cal:
            rawdata = '_raw'
        else:
            rawdata = '_cal'

        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self._data_window(nr_rev=nr_rev, time=time)

        # -------------------------------------------------
        # setup the figure
        # -------------------------------------------------

        nr_plots = 4

        pa4 = plotting.A4Tuned()
        fig_descr = rawdata + '_dashb_blade_yaw' + zoomtype
        grandtitle = self.resfile.replace('_', '\_')
        pwx = plotting.TexTemplate.pagewidth*2.0
        pwy = plotting.TexTemplate.pagewidth*1.0
        pa4.setup(figpath + self.resfile + fig_descr, nr_plots=nr_plots,
                   grandtitle=grandtitle, wsleft_cm=1.6, wsright_cm=0.4,
                   hspace_cm=2., wspace_cm=5.0, figsize_y=pwy, nr_cols=2,
                   figsize_x=pwx, wstop_cm=1.2, ws_bottom=1.0)

        # -------------------------------------------------
        # data selection from dSPACE
        # -------------------------------------------------
        channels = []

        channels.append(self.dspace.labels_ch['RPM'])
        channels.append(self.dspace.labels_ch['Yaw Laser'])
#        channels.append(self.dspace.labels_ch['Power'])
        channels.append(self.dspace.labels_ch['Tower Strain For-Aft'])
        channels.append(self.dspace.labels_ch['Tower Strain Side-Side'])
#        channels.append(self.dspace.labels_ch['Tower Top acc Y (FA)'])
#        channels.append(self.dspace.labels_ch['Tower Top acc X (SS)'])
#        channels.append(self.dspace.labels_ch['Yaw Laser'])
#        channels.append(self.dspace.labels_ch['Tower Top acc Z'])

        # convert labels for calibrated signals
        ylabels = dict()
        ylabels['Yaw Laser'] = '[deg]'
        ylabels['Tower Strain For-Aft'] = 'Force tower top [N]'
        ylabels['Tower Strain Side-Side'] = 'Force tower top [N]'

        # -------------------------------------------------
        # dSPACE RPM AND YAW
        # -------------------------------------------------
        time = self.dspace.time[slice_dspace]
        data = self.dspace.data[slice_dspace,:]
        plot_nr = 1

        ch_rpm = self.dspace.labels_ch['RPM']
        ch_yaw = self.dspace.labels_ch['Yaw Laser']

        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.set_title('Rotor speed and yaw angle')

        ax1.plot(time, data[:,ch_yaw], 'r--', label='Yaw error')
        ax1.set_ylabel('Yaw angle [deg]')
        ax1.grid(True)
        #leg = ax1.legend(loc='upper left')
        #leg.get_frame().set_alpha(0.5)

        ax2 = ax1.twinx()
        ax2.plot(time, data[:,ch_rpm], 'b', label='RPM')
        ax2.set_ylabel('rotor speed [RPM]')
        #leg = ax2.legend(loc='lower right')
        #leg.get_frame().set_alpha(0.5)

        # one legend
        lines = ax1.lines + ax2.lines
        labels = [l.get_label() for l in lines]
        leg = ax2.legend(lines, labels, loc='best')
        leg.get_frame().set_alpha(0.5)

        ax1, ax2 = plotting.match_axis_ticks(ax1, ax2, ax2_format='%1.0f')

        if time_range:
            ax1.set_xlim(window_dspace)

        plot_nr += 1
        # -------------------------------------------------
        # plotting Blade selection
        # -------------------------------------------------
        # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
        # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)
        timeb = self.blade.time[slice_blade]
        datab = self.blade.data[slice_blade,:]

        # Blade root sections
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(time, data[:,2], 'k', label='blade 1 root')
#        ax1.plot(time, data[:,3], 'm', label='blade 1 30%')
        ax1.plot(timeb, datab[:,0], 'k', label='root')
        ax1.plot(timeb, datab[:,1], 'm', label='30\%')
        ax1.set_title('Blade 2 bending moments [Nm]')
        ax1.grid(True)
        leg = ax1.legend(loc='lower right')
        leg.get_frame().set_alpha(0.5)
        ax1.set_xlim(window_blade)
#        ax1.set_ylabel('Bending moment [Nm]')

        # -------------------------------------------------
        # dSPACE TOWER SS AND FA
        # -------------------------------------------------

        ch_fa = self.dspace.labels_ch['Tower Strain For-Aft']
        ch_ss = self.dspace.labels_ch['Tower Strain Side-Side']

        plot_nr += 1
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax1.set_title('Tower bending FA [Nm]')
        ax1.plot(time, data[:,ch_fa], 'b', label='FA')
        ax1.set_ylabel('Tower bending moment [Nm]')
        ax1.set_xlabel('Time')
        ax1.grid(True)
        if time_range:
            ax1.set_xlim(window_dspace)

        plot_nr += 1
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax1.set_title('Tower bending SS [Nm]')
        ax1.plot(time, data[:,ch_ss], 'b', label='SS')
#        ax1.set_ylabel('Tower bending moment [Nm]')
        ax1.set_xlabel('Time')
        ax1.grid(True)
        if time_range:
            ax1.set_xlim(window_dspace)

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig()

    def _mod_labels_sel_chans(self, forplots=True):
        """
        Consistent labels acrross February/April campaigns, and select
        which channels to take from the dSPACE results.

        For saving to hdf5 pd.DataFrame, use C-compatble variable names
        """

        cnames = {}

        if not self.dspace_is_cal and not self.blade_is_cal:
            rawdata = '_raw'
            ylabels = {}
        else:
            rawdata = '_cal'
            # convert labels for calibrated signals
            ylabels = {}
            fa_key = 'Tower Strain For-Aft'
            ss_key = 'Tower Strain Side-Side'
            if self.dspace_yawcal:
                ylabels['Yaw Laser'] = '[deg]'
            if self.dspace_towercal:
                ylabels[fa_key] = 'Tower base FA moment [Nm]'
                ylabels[ss_key] = 'Tower base SS moment [Nm]'
            if self.dspace_towercal_psicor:
                ylabels[fa_key] = 'Tower base $\psi$ moment [Nm]'
                ylabels[ss_key] = 'Tower base $\psi_{90}$ moment [Nm]'

        # -------------------------------------------------
        # data selection from dSPACE
        # -------------------------------------------------
        channels = []

        try:
            channels.append(self.dspace.labels_ch['RPM'])
            channels.append(self.dspace.labels_ch['Yaw Laser'])
        except KeyError:
            logging.warning('there is no yaw laser channel??')
#        channels.append(self.dspace.labels_ch['Power'])
        try:
            channels.append(self.dspace.labels_ch['Tower Strain For-Aft'])
            channels.append(self.dspace.labels_ch['Tower Strain Side-Side'])
        except KeyError:
            key = 'Tower Strain For-Aft filtered'
            channels.append(self.dspace.labels_ch[key])
            key = 'Tower Strain Side-Side filtered'
            channels.append(self.dspace.labels_ch[key])
        if not forplots:
            channels.append(self.dspace.labels_ch['Tower Top acc Y (FA)'])
            channels.append(self.dspace.labels_ch['Tower Top acc X (SS)'])
            channels.append(self.dspace.labels_ch['Tower Top acc Z'])
            channels.append(self.dspace.labels_ch['Voltage filtered'])
            channels.append(self.dspace.labels_ch['Current Filter'])
            if 'Azimuth' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['Azimuth'])
            if 'RPM Estimator v1' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['RPM Estimator v1'])
            if 'RPM Estimator v2' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['RPM Estimator v2'])
            if 'Duty Cycle' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['Duty Cycle'])
            if 'Power' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['Power'])
            if 'Power2' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['Power2'])
            if 'Sound' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['Sound'])
            if 'Sound_gain' in self.dspace.labels_ch:
                channels.append(self.dspace.labels_ch['Sound_gain'])

        return ylabels, rawdata, channels

    def dashboard_a3(self, figpath, **kwargs):
        """
        Plot OJF test dashboard on A3 paper size
        ========================================

        This includes all data and puts dSPACE, OJF and blade data in just
        one plot. No calibration is performed. If the data has been
        calibrated before, it is tagged as such in the filename.

        """
        eps = kwargs.get('eps', False)
        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)
        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self._data_window(nr_rev=nr_rev, time=time)

        ylabels, rawdata, channels = self._mod_labels_sel_chans(forplots=True)

        # -------------------------------------------------
        # setup the figure
        # -------------------------------------------------
        if self.dspace.campaign == 'February':
            nr_plots = 10
            nr_plots = 7
        else:
            nr_plots = 11
            nr_plots = 8

        pa4 = plotting.A4Tuned()
        fig_descr = rawdata + '_dashboard' + zoomtype
        # escape any underscores in the file name for latex printing
        grandtitle = self.resfile.replace('_', '\_')
        pa4.setup(figpath + self.resfile + fig_descr, nr_plots=nr_plots,
                   grandtitle=grandtitle, wsleft_cm=2., wsright_cm=1.0,
                   hspace_cm=2., wspace_cm=4.2)

        # -------------------------------------------------
        # plotting dSPACE selection
        # -------------------------------------------------
        time = self.dspace.time[slice_dspace]
        data = self.dspace.data[slice_dspace,:]
        plot_nr = 1
        for ch in channels:
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax1.plot(time, data[:,ch], 'b')
            ax1.set_title(self.dspace.labels[ch])
            ax1.grid(True)

            #if plot_nr < len(channels)-1:
                ## clear all x-axis values except the bottom subplots
                #ax1.set_xticklabels([])

            ch_key = self.dspace.labels[ch]
            if ylabels.has_key(ch_key):# and caldict_dspace.has_key(ch_key):
                ax1.set_ylabel(ylabels[self.dspace.labels[ch]])

            ax1.set_xlim(window_dspace)

            plot_nr += 1

        # -------------------------------------------------
        # plotting OJF selection
        # -------------------------------------------------
        if self.isojfdata:
            time = self.ojf.time[slice_ojf]
            data = self.ojf.data[slice_ojf,:]
            ch_t = 1
            ch_w = 4
            ch_ps = 2
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)

            # in the title, inlcude the static pressure
            press = 'static pressure: %1.0f kPa' % data[:,ch_ps].mean()
            title = [self.ojf.labels[ch_t], self.ojf.labels[ch_w], press]
            ax1.set_title(', '.join(title))

#            # --------------------------
#            # plot temperature left
#            ax1.plot(time, data[:,ch_t], 'r', label=self.ojf.labels[ch_t])
#            ax1.set_ylabel(self.ojf.labels[ch_t])
#            ax1.grid(True)
#            # force the scale on y axis to be more readable
#            if (data[:,ch_t].max() - data[:,ch_t].min()) < 1.:
#                ax1.set_ylim([data[:,ch_t].mean()-1, data[:,ch_t].mean()+3])
#            #leg = ax1.legend(loc='upper left')
#            #leg.get_frame().set_alpha(0.5)
#
#            # and wind speed on the right axes
#            ax2 = ax1.twinx()
#            ax2.plot(time, data[:,ch_w], 'b', label=self.ojf.labels[ch_w])
#            ax2.set_ylabel(self.ojf.labels[ch_w])
#            if (data[:,ch_w].max() - data[:,ch_w].min()) < 1.:
#                ax2.set_ylim([data[:,ch_w].mean()-1, data[:,ch_w].mean()+1])
#            #leg = ax2.legend(loc='upper right')
#            #leg.get_frame().set_alpha(0.5)
#
#            # put both plot labels in one legend
#            # plot on the 2nd axis so the legend is always on top!
#            lines = ax1.lines + ax2.lines
#            labels = [l.get_label() for l in lines]
#            leg = ax2.legend(lines, labels, loc='best')
#            leg.get_frame().set_alpha(0.5)
#            # --------------------------

            # or just plot them in one graph! It is too confusing otherwise
            ax1.plot(time, data[:,ch_t], 'r', label=self.ojf.labels[ch_t])
            ax1.set_ylabel('$^{\circ} C$, [m/s]')
            ax1.plot(time, data[:,ch_w], 'b', label=self.ojf.labels[ch_w])
            ax1.grid(True)
            ax1.legend(loc='best')
            # force always the same data range
            ax1.set_ylim(3,25)

            plot_nr += 1
            ax1.set_xlim(window_ojf)

#        # windspeed
#        ch = 4
#        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
#        ax1.plot(self.ojf.time, self.ojf.data[:,ch], 'r')
#        ax1.set_title(self.ojf.labels[ch_w])
#        ax1.grid(True)
#        plot_nr += 1
#        if time_range:
#            ax1.set_xlim(window_ojf)

        # ALSO INCLUDE THE HS camera trigger, put it on negative axis of the
        # overlap pulse
        # overlap pulse only relevant for April
        if self.dspace.campaign == 'April':

            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            # add on the negative axis the high speed camera trigger
            time = self.dspace.time[slice_dspace]
            ch = self.dspace.labels_ch['HS trigger']
            # but make it into the negative space so it doesn't mix with
            # the overlap pulse (normal values are 0 and 1)
            data = -0.5*self.dspace.data[slice_dspace,ch]
            ax1.plot(time, data, 'b', label='HS trigger')
            title = 'HS camera trigger'
            # in April we had two kinds for the trigger, one that shows
            # all the individual triggers, and another marking the start
            # and the end of the trigger session
            try:
                ch2 = self.dspace.labels_ch['HS trigger start-end']
                data = -0.5*self.dspace.data[slice_dspace,ch2]
                ax1.plot(time, data, 'g')
            except KeyError:
                pass

            try:
                time = self.blade.time[slice_blade]
                data = self.blade.data[slice_blade,:]
                ax1.plot(time, data[:,4], 'm', label='pulse strain')

                ch = self.dspace.labels_ch['RPM Pulse']
                time = self.dspace.time[slice_dspace]
                data = self.dspace.data[slice_dspace,ch]
                title += ', overlap pulse'

                ax1.plot(time, data, 'b--', label='pulse dspace')
                if self.dspace_strain_is_synced:
                    title += ', synchronized'
                else:
                    title += ', synchronization failed'
            # which value is possibly not defined??
            # in the other case, just plot the HS Trigger
            except AttributeError:
                pass

            # and finalize the plot
            ax1.set_title(title)
            ax1.grid(True)
            leg = ax1.legend(loc='lower right')
            leg.get_frame().set_alpha(0.5)
            ax1.set_xlim(window_blade)
            plot_nr += 1

        else:
            # For february, only plot the HS trigger, but that channel
            # might not be available for all the DC sweeps
            try:
                ch = self.dspace.labels_ch['HS trigger']
                time = self.dspace.time[slice_dspace]
                data = self.dspace.data[slice_dspace,ch]

                ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
                # the trigger is either 0 or 1
                ax1.plot(time, data, 'b')
                ax1.set_title(self.dspace.labels[ch])
                ax1.grid(True)

                ch_key = self.dspace.labels[ch]
                if ylabels.has_key(ch_key):
                    ax1.set_ylabel(ylabels[self.dspace.labels[ch]])

                ax1.set_xlim(window_dspace)
            except KeyError:
                pass

            plot_nr += 1

        # -------------------------------------------------
        # plotting Blade selection
        # -------------------------------------------------
        # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
        # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)
        try:
            time = self.blade.time[slice_blade]
            data = self.blade.data[slice_blade,:]

            # if blade nr 1 needs to be on uneven plot number, otherwise
            # it is not on the same line as blade nr2 plot
            if math.fmod(plot_nr,2) == 0:
                ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
                plot_nr += 1

            # Blade 1
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax1.plot(time, data[:,2], 'm', label='root')
            ax1.plot(time, data[:,3], 'k', label='30\%')
            ax1.set_title('blade 1 strain sensors')
            ax1.grid(True)
            leg = ax1.legend(loc='lower right')
            leg.get_frame().set_alpha(0.5)
            ax1.set_xlim(window_blade)
            if self.blade_is_cal:
                ax1.set_ylabel('flapwise bending moment [Nm]')
            plot_nr += 1

            # Blade 2
            ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax2.plot(time, data[:,0], 'm', label='root')
            ax2.plot(time, data[:,1], 'k', label='30\%')
            ax2.set_title('blade 2 strain sensors')
            ax2.grid(True)
            leg = ax2.legend(loc='lower right')
            leg.get_frame().set_alpha(0.5)
            ax2.set_xlim(window_blade)
            if self.blade_is_cal:
                ax2.set_ylabel('flapwise bending moment [Nm]')
            plot_nr += 1

            # set the same y range on both axis
            lims1 = ax1.get_ylim()
            lims2 = ax2.get_ylim()
            ymin = min([lims1[0], lims2[0]])
            ymax = max([lims1[1], lims2[1]])
            ax1.set_ylim([ymin, ymax])
            ax2.set_ylim([ymin, ymax])

        except AttributeError:
            pass

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig(eps=eps)


    def plot_combine(self, figpath, ch_dspace, ch_ojf, ch_blade, **kwargs):
        """
        A pre configured dashboard for a given test run.

        Plot a selection of channels of dSPACE, OJF and/or blade

        Parameters
        ----------

        figpath : str
            full path for the to be saved figure

        ch_dspace : list
            List with channel numbers (integers) that will be included in the
            plots with respect to the dSPACE data.

        ch_ojf : list
            List with channel numbers (integers) that will be included in the
            plots with respect to the OJF wind tunnel data.

        ch_blade : list
            List with channel numbers (integers) that will be included in the
            plots with respect to the blade strain data. Ignored if no blade
            strain data file has been found.

        """

        nr_rev = kwargs.get('nr_rev', None)
        time = kwargs.get('time', None)

        if not self.dspace_is_cal and not self.blade_is_cal:
            rawdata = '_raw'
        else:
            rawdata = '_cal'

        # if a data window is specified, get the appropriate slices
        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self._data_window(nr_rev=nr_rev, time=time)

        # -------------------------------------------------
        # setup the plot
        # -------------------------------------------------
        nr_plots = len(ch_dspace) + len(ch_ojf) + len(ch_blade)
        grandtitle = self.resfile
        pa4 = plotting.A4Tuned()
        fig_descr = rawdata + zoomtype
        pa4.setup(figpath + self.resfile + fig_descr, nr_plots=nr_plots,
                   grandtitle=grandtitle, wsleft_cm=2., wsright_cm=1.5,
                   hspace_cm=2., dpi=50)

        # -------------------------------------------------
        # plotting dSPACE selection
        # -------------------------------------------------
        time = self.dspace.time[slice_dspace]
        data = self.dspace.data[slice_dspace,:]
        plot_nr = 1
        for ch in ch_dspace:
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax1.plot(time, data[:,ch], 'b')
            ax1.set_title(self.dspace.labels[ch])
            ax1.grid(True)
            if plot_nr < len(ch_dspace)-1:
                # clear all x-axis values except the bottom subplots
                ax1.set_xticklabels([])

            ax1.set_xlim(window_dspace)

            plot_nr += 1


        # -------------------------------------------------
        # plotting OJF selection
        # -------------------------------------------------
        time = self.ojf.time[slice_ojf]
        data = self.ojf.data[slice_ojf,:]

        for ch in ch_ojf:
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
            ax1.plot(time, data[:,ch], 'r')
            ax1.set_title(self.ojf.labels[ch])
            ax1.grid(True)
            if plot_nr < len(ch_ojf)-1:
                # clear all x-axis values except the bottom subplots
                ax1.set_xticklabels([])

            ax1.set_xlim(window_ojf)

            plot_nr += 1

        # -------------------------------------------------
        # plotting blade strain selection
        # -------------------------------------------------

        # if applicable!
        if self.isbladedata:
            time = self.blade.time[slice_blade]
            data = self.blade.data[slice_blade,:]

            for ch in ch_blade:
                ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
                ax1.plot(time, data[:,ch], 'g')
                ax1.set_title(self.ojf.labels[ch])
                ax1.grid(True)
                if plot_nr < len(ch_blade)-1:
                    # clear all x-axis values except the bottom subplots
                    ax1.set_xticklabels([])

                ax1.set_xlim(window_blade)

                plot_nr += 1

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig()

    def overlap_pulse(self, figpath):
        """
        Plot only the overlap pulse, for synchronisation the dSPACE and
        MicroStrain clocks
        """
        nr_rev = 9
        time_range = nr_rev/(self.rpm_mean/60.)
        start = -0.1

        # -------------------------------------------------
        # setup the figure
        # -------------------------------------------------
        nr_plots = 2
        plot_nr = 1

        pa4 = plotting.A4Tuned(scale=1.5)
        if self.dspace_strain_is_synced:
            extra = '_synced'
        else:
            extra = '_nosync'
        fig_descr = extra + '_overlappulse'
        # escape underscore for LaTeX printing
#        grandtitle = self.resfile.replace('_', '\_')
        figx = plotting.TexTemplate.pagewidth
        figy = plotting.TexTemplate.pagewidth*0.6
        target = figpath + self.resfile + fig_descr
        pa4.setup(target, nr_plots=nr_plots, grandtitle=None, hspace_cm=1.0,
                   figsize_x=figx, figsize_y=figy, wsleft_cm=1.4,
                   wsright_cm=0.4, wstop_cm=0.8, wsbottom_cm=1.0)

        # -------------------------------------------------
        # Plot the overlap pulse
        # -------------------------------------------------

        # -------------------- plot1
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        # dspace
        ch = self.dspace.labels_ch['RPM Pulse']
        time = self.dspace.time[:]
        data = self.dspace.data[:,ch]/self.dspace.data[:,ch].max()
        ax1.plot(time, data, 'r-', label='fixed')
        # blade
        time = self.blade.time[:]
        data = self.blade.data[:,4]/self.blade.data[:,4].max()
        ax1.plot(time, data, 'k--', label='wireless')
        # more plot stuff
        ax1.set_title('start measurement, RPM=%1.0f' % self.rpm_mean)
        ax1.grid(True)
        leg = ax1.legend(loc='upper right')
        leg.get_frame().set_alpha(0.8)
        ax1.set_xlim([start, start+time_range])
        ax1.set_ylim([-0.1, 1.1])

        # -------------------- plot2
        plot_nr += 1
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)

        # dspace, pulse blade ends at
        last_pulse = time[-1]
        ch = self.dspace.labels_ch['RPM Pulse']
        time = self.dspace.time[:]
        data = self.dspace.data[:,ch]/self.dspace.data[:,ch].max()
        ax1.plot(time, data, 'r-', label='fixed')
        # blade
        time = self.blade.time[:]
        data = self.blade.data[:,4]/self.blade.data[:,4].max()
        ax1.plot(time, data, 'k--', label='wireless')
        # more plot stuff
        ax1.set_title('end of measurement')
        ax1.grid(True)
        leg = ax1.legend(loc='upper right')
        leg.get_frame().set_alpha(0.8)
        ax1.set_xlim([last_pulse-(time_range*0.6),last_pulse+(time_range*0.4)])
        ax1.set_ylim([-0.1, 1.1])
        ax1.set_xlabel('time [s]')

        # -------------------------------------------------
        # calculate synchronisation error
        # -------------------------------------------------
        # TODO: finish implementation

        ## minimum wave height
        #h = 0.25
        #
        ## convert the array (first column is assumed to be time) to a wafo
        ## timeseries object. The turning_point method only works with one data
        ## channel
        #signal_dspace = np.empty((len(self.dspace.time),2))
        #signal_dspace[:,0] = self.dspace.time
        #signal_dspace[:,1] = self.dspace.data[:,ch]
        #sig_ts = wafo.objects.mat2timeseries(signal_dspace)
        #
        ## the turning_points method in wafo.objects
        ## set h to appropriate value if only high amplitude cycle count
        #sig_tp = sig_ts.turning_points(h=h, wavetype=None)
        #
        ## cycle pairs, h=0 means no filtering of cycles with certain height
        #self.sig_cp = sig_tp.cycle_pairs(h=h, kind='min2max')

        #print self.sig_cp.data[-10:]
        #print self.sig_cp.args[-10:]

        # -------------------------------------------------
        # save figure
        # -------------------------------------------------
        pa4.save_fig()

    def to_df(self, fpath=None, complevel=0, complib=None):
        """
        Convert the callibrated and combined result file into pandas DataFrame.
        Meta-data is stored in the index name attribute.
        """

        df = pd.DataFrame()

        (slice_dspace, slice_ojf, slice_blade, window_dspace, window_ojf,
                window_blade, zoomtype, time_range) = self._data_window()

        ylabels, rawdata, chans = self._mod_labels_sel_chans(forplots=False)

        # some of the DC runs do not have this one
        try:
            chans.append(self.dspace.labels_ch['HS trigger'])
        except KeyError:
            pass

        if self.dspace.campaign == 'April':
            try:
                chans.append(self.dspace.labels_ch['HS trigger start-end'])
            except KeyError:
                pass
            chans.append(self.dspace.labels_ch['RPM Pulse'])
#            if self.dspace_strain_is_synced:
#                title += ', synchronized'
#            else:
#                title += ', synchronization failed'

        df['time'] = self.dspace.time[slice_dspace]
        for ch in chans:
            try:
                colname = ylabels[self.dspace.labels[ch]]
            except KeyError:
                colname = self.dspace.labels[ch]
            cname = self.dspace.labels_cnames[self.dspace.labels[ch]]
            df[str(cname)] = self.dspace.data[slice_dspace,ch]

        if self.isojfdata:
            for ch in [1, 4, 2]:
                colname = self.ojf.labels[ch].replace(' ', '_')
                df[str(colname)] = self.ojf.data[slice_ojf,ch]

        try:
            # cnames has maximum 5 columns (5th is the pulse), but data can
            # still hold 6 columns (last one is empty)
            for ch in range(min(self.blade.data.shape[1],5)):
                colname = self.blade.cnames[ch]
                df[str(colname)] = self.blade.data[slice_blade,ch]
        except AttributeError:
            pass

        if fpath is not None:
            df.to_hdf(fpath, 'table', complevel=complevel, complib=complib)

        self.df = df

        return df


class BladeContour:
    """
    The array result files are named as follows:
        flex_B1_LE_306
    """

    def __init__(self):
        """
        Load the relevant data from the excel sheet produced by the
        """
        pass

    def _load_blade_cases(self, fpath, search_items, silent=False):
        """
        Load all measurements for given blade
        =====================================


        Parameters
        ----------

        fpath : str
            Path to the blade result file

        search_items : list of strings
            List of strings containing the search criteria.
            For example: ['flex', '0'] will select all flex blades that
            have zero load

        """

        cases = dict()

        if not silent:
            print search_items

        # load all available blade cases
        for f in [f for f in os.walk(fpath)][0][2]:
            # ignore the original long Excel files
            if f.endswith('.xls'): continue

            case_items = f.split('_')

            go = 0
            # all search items have to be in case_items, otherwise do not
            # load the case

            for item in search_items:
                if item in case_items:
                    go += 1
                else: pass
            if len(search_items) == go:
                if not silent:
                    print 'case_items', case_items
                cases[f] = np.loadtxt(fpath+f)

        return cases

    def mean_defl(self, fpath, structure, bladenr, tipload, silent=False,
                  correct=False, plotcorrect=False, istart=0, iend=8):
        """
        Create average of LE, TE, mid deflections curves
        ================================================

        Create for each tip loading case the deflection curve. Also, save an
        average deflection curve by considering all available measurements
        (LE, TE, mid).

        NOTE: there is a very ugley monkey patch applied to find for blade 3
        stiff any missing points somewhere in the middle

        Paramters
        ---------

        fpath : str

        structure : str

        bladenr : str

        tipload : str

        silent : boolean, default=False

        correct : boolean, default=False

        plotcorrect : boolean, default=False

        istart : int, default=0
            Starting point of the correction fitting tangent line index

        iend : int, default=8
            End point for the correction fitting tangent line
        """

        cases = self._load_blade_cases(fpath, [structure, bladenr, tipload],
                                       silent=silent)

        # how many datapoints are there in the deflection curve?
        # they have all been interpollated to the same grid before
        nr_points = len(cases[cases.keys()[0]])
        deltas = np.ndarray( (len(cases), nr_points) )
        # calculate the deltas: cycle through all cases and decide which
        # reference case should be deducted from it
        for ii, cc in enumerate(cases):
            # drop the load indicator and select zero load reference case
            ref = '_'.join(cc.split('_')[0:-1]) + '_0'
            deltas[ii,:] = cases[cc][:,1]
            deltas[ii,:] -= np.loadtxt(fpath+ref)[:,1]

        defl_mean = np.ndarray((2, nr_points))
        # radial positions
        defl_mean[0,:] = cases[cases.keys()[0]][:,0]
        # calcualte the mean values, but specify the axis, otherwise all nan??
        defl_mean[1,:] = deltas.mean(axis=0)

        # DEPRICATED, SEE METHOD BELOW
        ## seems to be that the clamping wasn't perfect for the tests...
        ## set the initial point to zero of the measurement series
        #ifirst = np.isfinite(defl_mean[1,:]).argmax()
        #defl_mean[1,:] -= defl_mean[1,ifirst]

        if correct:
            # more on the assumed bad clamping, assume the initial deflection
            # angle should be zero. So remove a straight line with the rico we
            # find at the root
            nonan = np.isfinite(defl_mean[1,:])
            # and only consider the 5 first points to get angle due to bad
            # clamping
            xcor = defl_mean[0,nonan][istart:iend]
            ycor = defl_mean[1,nonan][istart:iend]
            if not silent:
                print 'r/R used for fit: %1.4f %1.4f' % (xcor[0], xcor[1])
            pol = np.polyfit(xcor,ycor, 1)
            corr = np.polyval(pol, defl_mean[0,:])
            if plotcorrect:
                plt.figure()
                plt.plot(defl_mean[0,:], defl_mean[1,:], 'b',
                         label='no correction')
                plt.plot(defl_mean[0,:], corr, 'k--',
                         label='correction line')
                plt.plot(defl_mean[0,:], defl_mean[1,:]-corr, 'r--',
                         label='corrected')
                plt.grid(True)
                plt.legend(loc='best')
            defl_mean[1,:] -= corr
        else:
            corr = False

        # for one case some intermediate results where missing. Interpolate
        # between the two known points
        # nans at the start and the end of the file is ok, in between not
        # first pass: determine where continiuty is disturbed by nans
        bnans = np.isnan(defl_mean[1,:])
        after1nan = False
        ii = 1
        start = False
        end = False
        if bladenr == 'B3' and structure == 'stiff':
            for kk in bnans:
                # first nans of the series
                if kk and not after1nan and not start:
                    pass
                # first real values of the series
                elif not kk and not after1nan and not start:
                    after1nan = True
                # all next real values after first nans
                elif not kk and not start:
                    pass
                # we reach the second nan bump, the one that should be marked
                elif kk and after1nan and not start:
                    start = ii
                # we are no on the second nan bump
                elif kk and start:
                    pass
                # we reach the end of the second nan bump
                elif not kk and start:
                    end = ii
                    break
                ii += 1

        # interpolate from start to end, make sure they are not too close to
        # the start/end of the global data
        if start and end and start > 10 and end < len(bnans)-10:
            x0 = defl_mean[0,start-2]
            x1 = defl_mean[0,end-1]
            y0 = defl_mean[1,start-2]
            y1 = defl_mean[1,end-1]
            addx = defl_mean[0, start-1:end-1]
            x = np.array([x0,x1])
            y = np.array([y0,y1])
            addy = sp.interpolate.griddata(x, y, addx)
            defl_mean[1, start-1:end-1] = addy

            ## debug plotting
            #print start, end
            #print x0, x1
            #print y0, y1
            #print addx[0], addx[-1]
            #import pylab as plt
            #plt.figure()
            #plt.plot(defl_mean[0,:], defl_mean[1,:], 'yo', alpha=0.5)
            #plt.plot(addx, addy, 'k*')

        # return radial positions and average deflections
        return defl_mean, corr


    def deltas_case(self, fpath, search_items, silent=False):
        """
        Return the delta's for a given blade, load case and location (TE, mid,
        or LE).

        Parameters
        ----------

        fpath : str
            Path to the blade result file

        search_items : list of strings
            List of strings containing the search criteria.
            For example: ['flex', '0'] will select all flex blades that
            have zero load

        """
        cases = self._load_blade_cases(fpath, search_items)

        deltas = dict()
        # calculate the deltas: cycle through all cases and decide which
        # reference case should be deducted from it
        for cc in cases:
            if not silent:
                print cc
            # drop the load indicator and select zero load reference case
            ref = '_'.join(cc.split('_')[0:-1]) + '_0'
            deltas[cc] = cases[cc]
            deltas[cc][:,1] -= np.loadtxt(fpath+ref)[:,1]

        return deltas

    def compare_blade_set(self, fpath, search_items, figpath, correct=False,
                          zoom=False, scale=1.0):
        """
        Compare LE, TE, mid and mean for one blade
        ==========================================

        Optionally, plot the correction

        And also calculate the deltas due to loading

        Case file format: flex_B1_LE_306

        Parameters
        ----------

        fpath : str
            Path to countour result files

        search_items : list of strings
            List of strings containing the search criteria.
            For example: ['flex', '0'] will select all flex blades that
            have zero load

        figpath : str

        correct : Boolean, default=False
            If the data needs to be correct for the assumed bad clamping

        zoom : Boolean, default=False

        scale : float, default=1.0

        """

        deltas = self.deltas_case(fpath, search_items)

        figsize_x = plotting.TexTemplate.pagewidth*0.5
        figsize_y = plotting.TexTemplate.pagewidth*0.4

        # initialize figure helper
        figname = '_'.join(search_items)
        grandtitle = ' '.join(search_items) + ' gr tip load'
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath+figname, nr_plots=1, hspace_cm=2., wsbottom_cm=1.0,
                   grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                   wstop_cm=0.5, figsize_y=figsize_y, figsize_x=figsize_x)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        #plotlines = ['b+-', 'r+-', 'k+-', 'b*-', 'r*-', 'k*-', 'bs-', 'rs-',\
                     #'ks-', 'bo-', 'ro-', 'ko-', 'bd-', 'rd-', 'kd-']
        #plotlines = ['b', 'r', 'k', 'g', 'y', 'm', 'c',\
                    #'b--', 'r--', 'k--', 'g--', 'y--', 'm--', 'c--']
        # cycle through all the cases and plot
        labelb1, labelb2, labelb3 = False, False, False
        for case in sorted(deltas):
            tipload = case.split('_')[-1]
            data = deltas[case]
            if case.find('B1') > -1:
                color = 'r-.'
                bladenr = 'B1'
                if not labelb1:
                    label = 'blade 1'
                    labelb1 = True
                else:
                    label = False
            elif case.find('B2') > -1:
                color = 'b--'
                bladenr = 'B2'
                if not labelb2:
                    label = 'blade 2'
                    labelb2 = True
                else:
                    label = False
            else:
                color = 'k-'
                bladenr = 'B3'
                if not labelb3:
                    label = 'blade 3'
                    labelb3 = True
                else:
                    label = False
            if label:
                ax1.plot(data[:,0], data[:,1], color, label=label)
            else:
                ax1.plot(data[:,0], data[:,1], color)

        # if applicable, include the mean and corrected data
        if case.find('flex') > -1:
            struct = 'flex'
        elif case.find('stiff') > -1:
            struct = 'stiff'

        if correct:
            mean, corr = self.mean_defl(fpath, struct, bladenr, tipload,
                             silent=True, correct=correct, plotcorrect=False)
            if zoom:
                ax1.plot(mean[0,:], mean[1,:], 'c-', label='mean corr')
                ax1.plot(mean[0,:], corr, 'k.', label='corr')
                ax1.set_ylim([-0.2,1.5])
                ax1.set_xlim([0,0.2])
                pa4.figfile += '_zoom'
            else:
                ax1.plot(mean[0,:], mean[1,:], 'c-', label='mean corr')
                ax1.plot(mean[0,:], corr, 'k--', label='corr')
                ax1.set_ylim([-0.2,24])
                ax1.set_xlim([0,1.0])

        else:
            ax1.set_ylim([-0.2,24])

        ax1.set_xlabel('normalised radial position [-]')
        ax1.set_ylabel('flap deflection wrt zero load [mm]')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        pa4.save_fig()

    def compare_blades(self, fpath, struct, tipload, figpath, scale=1.0):
        """
        Compare mean and corrected data for all 3 stiff or flex blades
        ==============================================================

        And also calculate the deltas due to loading

        Case file format: flex_B1_LE_306

        Parameters
        ----------

        fpath : str
            Path to countour result files

        struct : str
            stiff or flex

        tipload : str
            string representation of the tip load in grams

        figpath : str

        """

        figsize_x = plotting.TexTemplate.pagewidth*0.5
        figsize_y = plotting.TexTemplate.pagewidth*0.4

        # initialize figure helper
        figname = '%s_%s_compare_corrected' % (struct, tipload)
        grandtitle = '%s blades, tipload %s gr' % (struct, tipload)
        pa4 = plotting.A4Tuned(scale=2)
        pa4.setup(figpath+figname, nr_plots=1, hspace_cm=2., wsbottom_cm=1.0,
                   grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                   wstop_cm=0.5, figsize_y=figsize_y, figsize_x=figsize_x)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        for bladenr in ['B1', 'B2', 'B3']:
            if bladenr == 'B1':
                col1 = 'r'
                col2 = 'r--'
            elif bladenr == 'B2':
                col1 = 'g'
                col2 = 'g--'
            elif bladenr == 'B3':
                col1 = 'b'
                col2 = 'b--'
            mean, corr = self.mean_defl(fpath, struct, bladenr, tipload,
                             silent=True, correct=True, plotcorrect=False)
            ax1.plot(mean[0,:], mean[1,:], col1, label=bladenr+' corr')
            ax1.plot(mean[0,:], mean[1,:]+corr, col2, label=bladenr)
            ax1.set_ylim([-0.2,24])

        ax1.set_xlabel('normalised radial position [-]')
        ax1.set_ylabel('flap deflection wrt zero load [mm]')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        pa4.save_fig()


    def plot_blade(self, case, figpath, figname, grandtitle):
        """
        Plot selected blade
        ===================

        Simple plot of the given raw result file. No more, no less
        """

        # initialize figure helper
        pa4 = plotting.A4Tuned()
        pa4.setup(figpath+figname, nr_plots=1, hspace_cm=2., wsbottom_cm=1.,
                   grandtitle=grandtitle, wsleft_cm=1.5, wsright_cm=1.0,
                   wstop_cm=1.8)

        # load all the data files
        data = np.loadtxt(case)

        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.plot(data[:,0], data[:,1], 'b+-')
        # set the y limits all the same to the range of the output sensors
        ax1.set_ylim([-12, 12])
        ax1.set_xlim([0,1])
        ax1.grid(True)

        pa4.save_fig()


    def xls_to_txt(self, pathfolder):
        """
        Convert a whole folder of original xls result files to txt arrays
        """

        # for each blade we established what the blade root and tip was
        # blade foam length is 0.549 m
        poscal = dict()
        # [tip voltage, root voltage]
        poscal['flex_B1'] = [3.23, 8.7]
        poscal['flex_B2'] = [3.24, 8.71]
        poscal['flex_B3'] = [3.33, 8.8]
        poscal['stiff_B1'] = [3.35, 8.82]
        poscal['stiff_B2'] = [3.33, 8.81]
        poscal['stiff_B3'] = [3.32, 8.81]
        poscal['flexy_B1'] = [3.27, 8.74]

        # where is the data located in the MS Excel file
        sheetname = 'DMM Scan'
        # make sure the number of rows is larger than the highest occuring
        # number of measurement points
        row_sel = xrange(39,500)
        col_sel = [4,5]

        #files = [f for f in os.walk(pathfolder)][0][2]
        for f in [f for f in os.walk(pathfolder)][0][2]:
            # construct a more compact file name
            # '2012-09-26 Contour meting Verelst Test flex B2 LE lijn 306 gr'
            tmp = f.split(' ')
            if not len(tmp) == 11: continue
            print f
            fname = '%s_%s_%s_%i' % (tmp[5], tmp[6], tmp[7], int(tmp[9]))

            # load the MS Excel file
            rows = misc.read_excel(pathfolder+f, sheetname, row_sel=row_sel,
                                   col_sel=col_sel, data_fmt='ndarray')

            # the data system had only a voltage range of -12+12. Any values
            # outside this interval are noise and should be set to nan
            rows[np.abs(rows[:,1]).__ge__(12),1] = np.nan

            # each measurement serie had ever so slightly different start
            # and end positions
            # calibrate the radial position voltage signal
            kk = '_'.join(tmp[5:7])
            posrange = poscal[kk][1] - poscal[kk][0]
            rows[:,0] = (rows[:,0] + poscal[kk][1]) / posrange
            # create the hd interpolated version of rows
            rows_hd = np.ndarray((200,2))
            rows_hd[:,0] = np.linspace(0, 1, 200)

            # interpolate to a fixed number of points for easy comparison
            # in a later stage
            rows_hd[:,1]=interpolate.griddata(rows[:,0],rows[:,1],rows_hd[:,0])

            # and save the numpy array in simple text format
            np.savetxt(pathfolder+fname, rows_hd, fmt='%0.10f')


class BladeCgMass:
    """
    Measurements originally carried out to determine the mass distribution,
    but after working out the solution I realised all the measurement points
    only gave away one parameter: the center of gravity.
    """

    def __init__(self, **kwargs):
        """
        """
        # set where the files are found
        self.fpath = kwargs.get('fpath', 'data/raw/blade_mass/')
        tmp = 'data/blademassproperties/'
        self.figpath = kwargs.get('figpath',tmp)

        # make an easy entry point, just select blade name and number
        self.bladedict = dict()
        self.bladedict['flex B1'] = 'mass_dist_flex1.csv'
        self.bladedict['flex B2'] = 'mass_dist_flex2.csv'
        self.bladedict['flex B3'] = 'mass_dist_flex3.csv'
        self.bladedict['stiff B1'] = 'mass_dist_stiff1.csv'
        self.bladedict['stiff B2'] = 'mass_dist_stiff2.csv'
        self.bladedict['stiff B3'] = 'mass_dist_stiff3.csv'
        self.bladedict['swept B1'] = 'mass_dist_swept1.csv'
        self.bladedict['swept B2'] = 'mass_dist_swept2.csv'
        self.bladedict['swept B3'] = 'mass_dist_swept3.csv'
        self.bladedict['flexy B1'] = 'mass_dist_flexy1.csv'
        self.bladedict['flexy B2'] = 'mass_dist_flexy2.csv'
        self.bladedict['flexy B3'] = 'mass_dist_flexy3.csv'


    def read(self, structure, bladenr, **kwargs):
        """
        Load a result file
        ==================

        Parameters
        ----------

        structure : str
            Blade structure, either flex, stiff, swept or flexy

        bladenr : str
            B1, B2 or B3


        Returns
        -------

        cg_mean : float
            Measurements based on the whole blade, so including blade clamp
            part. Consequently, cg is zero at the blade root clamp, at the
            blade root arfoil area x=0.078

        M_mean : float

        """

        bladekey = structure + ' ' + bladenr
        fname = self.bladedict[bladekey]

        verbose = kwargs.get('verbose', False)
        if verbose:
            print structure, bladenr, fname

        plot = kwargs.get('plot', False)
        figpath = kwargs.get('figpath', self.figpath)
        figfile = kwargs.get('figfile', fname).replace('.','_') + '_cg_only'
        grandtitle = kwargs.get('grandtitle', figfile).replace('_',' ')

        data = np.genfromtxt(self.fpath+fname,delimiter=',',dtype=np.float64)

        # original headers: overhang, mass (tip on balance), clamp on balance
        # units are in grams and cm
        # 63.3: total blade length in cm, including clamp
        # on the first row, first element we have the total blade mass
        L = 0.633
        #  0.9: offset on the measurement table in cm's
        # convert from cm to meters as well!
        # express as the distance between the balance and the support
        # consequently, this means for:
        #     tip on balance: x_mes=0 -> clamp, root
        #   clamp on balance: x_mes=0 -> tip
        x_mes = L - 0.009 - (data[:,0]/100.)
        # blade total mass from older measurements, already in kg!
        M = data[0,0]
        M_list = [M]

        # create array with all cg values, measured from tip and root
        self.cgarr = np.array([])

        # select non nan values, by chosing column 2, we automatically ignore
        # the mass entry
        #i_tip = np.isnan(data[:,1]).__invert__()
        i_tip = np.isfinite(data[:,1])

        # only when there is actually data present
        if i_tip.any():
            # take also the last measurement as the blade total mass
            M_tip = data[i_tip,1][-1]/1000.
            # the weights, convert to kg
            B_tip = data[i_tip,1]/1000.
            # and see if the theory matches practice
            self.massxpos_tip = x_mes[i_tip][2]*B_tip[2]/x_mes[i_tip]
            # get cg for each measurement and see how much noise there is
            cg_tip = x_mes[i_tip]*B_tip/M_tip
            # add to cg array
            self.cgarr = np.append(self.cgarr, cg_tip)
            # add to the mass list
            M_list.append(M_tip)

        # only do if we have the data for clamp series
        if data.shape[1] > 2:
            i_clamp = np.isfinite(data[:,2])
            # just in case there was no tip measurements
            M_clamp = data[i_clamp,2][-1]/1000.
            B_clamp = data[i_clamp,2]/1000.
            # for the clamp measurements
            self.massxpos_clamp = x_mes[i_clamp][2]*B_clamp[2]/x_mes[i_clamp]
            # express in same x coordinate: measured distance from the clamp!
            cg_clamp = L - (x_mes[i_clamp]*B_clamp/M_clamp)
            # add to cg array
            self.cgarr = np.append(self.cgarr, cg_clamp)
            # add to the mass list
            M_list.append(M_clamp)

        # calculate the average cg point and indicate value in box and line
        if len(self.cgarr) > 0:
            cg_mean = self.cgarr.mean()
        else:
            cg_mean = None

        # convert to array
        self.M_arr = np.array(M_list)
        # and also the average mass
        M_mean = self.M_arr.mean()

        self.x_mes = x_mes

        if plot and (i_tip.any() or data.shape[1] > 2):
            # setup figure
            pa4 = plotting.A4Tuned(scale=1.8)
            figsize_x = plotting.TexTemplate.pagewidth*0.5
            figsize_y = plotting.TexTemplate.pagewidth*0.3
            pa4.setup(figpath+figfile, nr_plots=1, grandtitle=False,
                      wsleft_cm=1.5, wstop_cm=0.7, figsize_y=figsize_y,
                      figsize_x=figsize_x, wsbottom_cm=0.9, wsright_cm=0.5)

            # make some nicer titles
            if figfile.find('flexy') > 0:
                blade_str = 'very flexible'
            elif figfile.find('flex') > 0:
                blade_str = 'flexible'
            elif figfile.find('stiff') > 0:
                blade_str = 'stiff'
            elif figfile.find('swept') > 0:
                blade_str = 'very flexible swept'

            if figfile.find('1') > 0:
                blade_nr = '1'
            elif figfile.find('2') > 0:
                blade_nr = '2'
            elif figfile.find('3') > 0:
                blade_nr = '3'

            title = blade_str + ' blade nr' + blade_nr
            figname = 'cg_' + title.replace(' ', '_')

            #ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
            #if i_tip.any():
                 ## plot the original datapoints
                #ax1.plot(x_mes[i_tip], B_tip, 'r^', label='meas tip')
                ## and see if the theory matches practice
                #ax1.plot(x_mes[i_tip],self.massxpos_tip,'r',label='theory tip')
            #if data.shape[1] > 2:
                #ax1.plot(x_mes[i_clamp], B_clamp, 'bs', label='meas clamp')
                ## for the clamp measurements
                #ax1.plot(x_mes[i_clamp], self.massxpos_clamp, 'b',
                         #label='theory clamp')
            #ax1.grid()
            #ax1.set_title('Comparing measured and theoretical mass*x values')
            #ax1.set_xlim([0.27, 0.65])
            #leg1 = ax1.legend(loc='best')
            #leg1.get_frame().set_alpha(0.5)

            #ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
            ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
            ax2.set_title(title)

            # convert to cm's for better plot label reading
            x_mes *= 100.0

            if i_tip.any():
                # convert to cm's for better plot label reading
                cg_tip *= 100.0
                # get cg for each measurement and see how much noise there is
                ax2.plot(x_mes[i_tip], cg_tip, 'r^', label='cg tip meas')

            if data.shape[1] > 2:
                # convert to cm's for better plot label reading
                cg_clamp *= 100.0
                ax2.plot(x_mes[i_clamp], cg_clamp, 'bs',label='cg clamp meas')

            ax2.grid()
            ax2.set_xlabel('measurement position [cm]')
            ax2.set_ylabel('cg position [cm]')

            # ignore the bottom label, same symbols apply as for above
            #leg2 = ax2.legend(loc='best')
            #leg2.get_frame().set_alpha(0.5)

            ax2.set_ylim([0.28*100, 0.304*100])
            ax2.set_xlim([0.27*100, 0.64*100])

            # indicate cg_mean value in box and plot a line
            ax2.axhline(y=cg_mean*100, linewidth=1, color='k',\
                    linestyle='-', aa=False)
            textbox= '$cg_{mean} = %.3f$' % cg_mean
            ax2.text(0.50*100, 0.302*100, textbox, fontsize=12, va='bottom',
                         bbox = dict(boxstyle="round",
                         ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))


            ## textbox on the how the different mass measurements vary
            #try:
                #tt1 = '$M_{mean} = %.4f$ $M_{v1} = %.4f$' % (M_mean, M)
            #except UnboundLocalError:
                #tt1 = '$M_{v1} = %.4f$' % (M)
            #try:
                #tt2 = '\n$M_{tip} = %.4f$ $M_{clamp} = %.4f$' % (M_tip,M_clamp)
            #except UnboundLocalError:
                #try:
                    #tt2 = '\n$M_{tip} = %.4f$' % (M_tip)
                #except UnboundLocalError:
                    #try:
                        #tt2 = '\n$M_{clamp} = %.4f$' % (M_clamp)
                    #except UnboundLocalError:
                        #tt2=''
            #textbox = tt1 + tt2
            #ax2.text(0.45, 0.302, textbox, fontsize=12, va='bottom',
                         #bbox = dict(boxstyle="round",
                         #ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
            pa4.figfile = figpath + figname
            pa4.save_fig()

        return cg_mean, M_mean


class HighSpeedCamera:
    """
    Get some information from the High Speed camera footage
    """
    class FebruarySwept:
        name = 'FebruarySwept'
        #y = int(img_src.shape[0]/5)
        #h = int(img_src.shape[0]*0.78)
        #x = int(img_src.shape[0]/3)+110
        #w = int(x/4)
        # img_src.shape -> (1136, 1024)

        # for the February results, the ROI is
        x = 420
        y = 227-50
        w = 122
        h = 886+44
        #img_sel = img_src[y:y+h,x:x+w]

        # resolutions are different, so the area thresholds should be different
        # minimum required area
        area_tip_min = 2000
        area_tip_max = 4000

        chord_min = 150
        chord_max = 185

        # when guessing the blade number, we draw the actual contour edges
        # away by giving the contour a certain thickness
        t_tipcnt = 6

        # filtering settings
        # first filtering threshold
        thresh_1 = 150
        # used to be used in find_contour_tip, now replaced with kmean clust
        thresh_2 = 250

        k_find_blade_nr = 10

        # how close can the TE, LE be to the ROI, in pixels
        roi_appr = 20

    class February:
        name = 'February'
        #y = int(img_src.shape[0]/5)
        #h = int(img_src.shape[0]*0.78)
        #x = int(img_src.shape[0]/3)+110
        #w = int(x/4)
        # img_src.shape -> (1136, 1024)

        # for the February results, the ROI is
        x = 488
        y = 227-50
        w = 122
        h = 886+44
        #img_sel = img_src[y:y+h,x:x+w]

        # resolutions are different, so the area thresholds should be different
        # minimum required area
        area_tip_min = 2000
        area_tip_max = 4000

        chord_min = 150
        chord_max = 185

        # when guessing the blade number, we draw the actual contour edges
        # away by giving the contour a certain thickness
        t_tipcnt = 6

        # filtering settings
        # first filtering threshold
        thresh_1 = 150
        # used to be used in find_contour_tip, now replaced with kmean clust
        thresh_2 = 250

        k_find_blade_nr = 10

        # how close can the TE, LE be to the ROI, in pixels
        roi_appr = 20

    class April:
        name = 'April'
        # the ROI
        x = 488 - 110
        y = 227 - 50
        w = 122
        h = 886 + 44

        area_tip_min = 800
        area_tip_max = 1600

        chord_min = 70
        chord_max = 110

        # when guessing the blade number, we draw the actual contour edges
        # away by giving the contour a certain thickness
        t_tipcnt = 2

        # filtering settings
        # first filtering threshold
        thresh_1 = 90
        # used to be used in find_contour_tip, now replaced with kmean clust
        thresh_2 = 250
        # clustering for the
        k_find_blade_nr = 40

        # how close can the TE, LE be to the ROI, in pixels
        roi_appr = 20

    def __init__(self, fpath=False, fpath_out=False, debug=False, save=True,
                 config='feb', verbose=False):
        """

        results array: [boxdeg, deg, chord, blade_nr]

        Parameters
        ----------

        fpath : str, default=False
            Location of the image source file

        fpath_out : str, default=False
            Location of where the final image will be saved. In debug mode,
            all intermediate steps are saved in fpath_out/debug

        debug : boolean, default=False
            Switch debugging on or off

        save : boolean, default=True
            If False, the final result image will not be safed. This does not
            interfere with debug.

        config : str, default='feb'
            feb, febswept, apr

        """
        self.save = save
        self.debug = debug
        self.verbose = verbose
        # for debugging purposes
        self.nr = 0

        # settings dependent on February or April images
        if config == 'feb':
            self.conf = self.February
        elif config == 'apr':
            self.conf = self.April
        elif config == 'febswept':
            self.conf = self.FebruarySwept
        else:
            raise UserWarning, 'Period should be either feb or apr'

        if self.debug:
            self.fpath = fpath
            self.fpath_out = fpath_out
            # check if the paths exists
            # create the folder if it doesn't exist
            if fpath_out:
                try:
                    os.mkdir(fpath_out)
                except OSError:
                    pass
                try:
                    os.mkdir(fpath_out+'debug/')
                except OSError:
                    pass

    def metadata(self, fpath):
        """
        Read the meta data file

        Paramters
        ---------

        fpath : str
            Path to the HS folder (including file name)

        Returns
        -------

        out : dict
            And with following keys
            # FPS : 125
            # Shutter Speed : 1/6000
            # Trigger Mode : Random
            # Trigger Mode Frames: 25
            # Original Total Frame : 775
            # Total Frame : 775
            # Start Frame : 0
            # Correct Trigger Frame : 0
            # Save Step : 1

        """
        #fmeta = fpath.split('/')[-2] + '.cih'

        if not fpath.endswith('.cih'):
            raise ValueError, 'Invalid HS metadata file: no .cih extension'

        # Record Rate(fps) : 125
        # Shutter Speed(s) : 1/6000
        # Trigger Mode : Random 25
        # Original Total Frame : 775
        # Total Frame : 775
        # Start Frame : 0
        # Correct Trigger Frame : 0
        # Save Step : 1

        out = {}

        def read_meta_line(line):
            return int(line.split(':')[-1].replace('\r\n', ''))

        FILE = open(fpath, "r")
        for line in iter(FILE):
            if line.startswith('Record Rate'):
                out['FPS'] = read_meta_line(line)
            elif line.startswith('Shutter Speed'):
                # save shutter time as a string: 1/6000
                tmp = line.split(':')[-1].replace('\r\n', '').strip()
                out['Shutter Speed'] = tmp
            elif line.startswith('Trigger Mode'):
                tmp = line.split(':')[-1].replace('\r\n', '').strip()
                out['Trigger Mode'] = tmp.split(' ')[0].strip()
                out['Trigger Mode Frames'] = int(tmp.split(' ')[-1])
            elif line.startswith('Original Total'):
                out['Original Total Frame'] = read_meta_line(line)
            elif line.startswith('Total Frame'):
                out['Total Frame'] = read_meta_line(line)
            elif line.startswith('Start Frame'):
                out['Start Frame'] = read_meta_line(line)
            elif line.startswith('Correct Trigger'):
                out['Correct Trigger Frame'] = read_meta_line(line)
            elif line.startswith('Save Step'):
                out['Save Step'] = read_meta_line(line)
                break

        FILE.close()

        return out

    def images_folder(self, fpath, fpath_out, **kwargs):
        """
        Analyze all the images in a given folder

        Parameters
        ----------

        fpath : str
            path to folder, or runid

        fpath_out : str

        Returns
        -------

        res : ndarray(11,n)
            where the rows are
            [boxdeg, deg, chord, blade_nr, lex, ley, tex, tey,
             frame nr, trigger count, trig_count_frame, time]
        """

        # two scenario's: runid or fpath
        if not fpath.startswith('/'):
            runid = fpath
            path_db = 'database/'
            database = 'symlinks_all' #kwargs.get('database', 'symlinks_all')
            respath = path_db + database + '/'
            # and load the full file name from the database index
            # load the database index for the dspace-strain-ojf cases
            FILE = open(path_db + 'db_index_%s_runid.pkl' % database)
            db_index_runid = pickle.load(FILE)
            FILE.close()
            resfile = db_index_runid[runid]
            hs_symlinks = kwargs.get('hs_symlinks', 'symlinks_hs_lacie2big')
            fpath = path_db + hs_symlinks + '/' + resfile + '/'
            print 'runid input, resulting target path'
            print respath+resfile

        # create the folder if it doesn't exist
        if fpath_out:
            try:
                os.mkdir(fpath_out)
            except OSError:
                pass
            try:
                os.mkdir(fpath_out+'debug/')
            except OSError:
                pass

        # save the results in an array, anticipate its size
        files = sorted(os.listdir(fpath))
        res = np.ndarray( (12, len(files)), dtype=np.float64 )
        nrfiles = len(files)

        if nrfiles < 1:
            print 'Nothing to do, folder is empty'
            return False

        if self.verbose:
            # print some headers for what self.image will output
            replace = ('case id', 'twist', 'chord', 'Bnr', 'LE_x', 'LE_y')
            print 79*'='
            print '%23s %6s %6s %3s %6s %6s' % replace
            print 79*'='
        else:
            # prepare the progress bar
            progbar.widgets = [progbar.Percentage(), ' ',
                               progbar.Bar(marker='-'), ' ', progbar.ETA()]

            pbar = progbar.ProgressBar(widgets=progbar.widgets, maxval=nrfiles)
            pbar.start()

        # read the data file, how many frames are taken per trigger signal
        meta_dict = self.metadata(fpath+files[0])
        # we want this to keep the trigger count. In dpsace, we have the
        # trigger count, so we need to now how many frames per trigger in order
        # to keep track of the trigger count, but start at zero
        trig_count, trig_count_frame = 0, -1
        frames_trig = meta_dict['Trigger Mode Frames']

        # define the zero time case
        if files[1].endswith('0001.jpg'):
            ctime0 = os.path.getctime(fpath+files[1])
        else:
            print 'file 1 is not the first frame! ctime0 set to 0'
            ctime0 = 0.0

        # and do the analysis for each file in the folder
        # start at frame nr 1, and trigger nr 1
        i,j, fails = 0, 0, 0
        # first file is the metadata file
        for fi in xrange(1,nrfiles):
            f = files[fi]
            j += 1
            trig_count_frame += 1
            # see when we have a new trigger count
            # frame 1-25 : trigger 1, frame 26-50 : trigger 2
            # j=0 already triggers this, so trig_count for j=0 sets it to 1
            if math.fmod(j-1, frames_trig) == 0:
                trig_count += 1
                trig_count_frame = 0
            try:
                if self.verbose:
                    msg = '%4.1f' % (100*j/nrfiles)
                    print msg + '% -',
                else:
                    # progress instead of verbose stuff, zero gives an error
                    pbar.update(j)

                res[0:8,i] = self.image(fpath, f, fpath_out)
                # and also save the current frame nr and the trigger count
                # and count how many frames in the trigger we have
                res[8:11,i] = [j, trig_count, trig_count_frame]
                # and the creation time (hoping that the file has not been
                # changed since created...)
                res[11,i] = os.path.getctime(fpath+f) - ctime0
                # if output contains nans, ignore it
                # sometimes things go to infinity, but I don't know why...
                if np.any(np.isnan(res[:,i])) or res[:,i].max() > 1e10:
                    fails += 1
                else:
                    i += 1
            except cv2.error:
                fails += 1
                fid = f.split('_')[-1]
                if self.verbose:
                    logging.warn('%s failed for unknown reason' % fid)

            # TODO: during one trigger, we always have the same blade nr!
            # are there many different nr's identified during one trigger?
            # if not the case is quite clear

            # TODO: when we have a new trigger, use the first frame as the
            # background, and use background subtraction as Fn = Fn - Fn-1

            # stop trying if there are only 1% hits after 50% of the files
            if i/nrfiles < 0.01 and j/nrfiles > 0.5:
                print 'stop analysis: success rate only at %5i/%5i' % (i,j)
                break

        if not self.verbose:
            pbar.finish()

        if self.verbose:
            print 79*'='
        ratio = fails/nrfiles
        replace = (fails, nrfiles, ratio)
        print 'failed cases: %5i out of %5i (%1.4f)' % replace
        if self.verbose:
            print 79*'='

        # strip the elements not used
        res = res[:,:i]

        # and save the results array
        respath = 'data/HighSpeedCamera/raw/'
        # for the id of the run, take a random file and subtrackt
        resname = '_'.join(files[0].split('_')[:3])
        print 'saving HS image analysis:'
        print respath+resname , '...',
        np.savetxt(respath+resname, res)
        print 'done'

        return res


    def image(self, fpath, fname, fpath_out, **kwargs):
        """
        Analyze a single image file

        Paramters
        ---------



        Returns
        -------

        results : ndarray(8)
            [boxdeg, deg, chord, blade_nr, lex, ley, tex, tey]

        """

        thresh_1 = kwargs.get('thresh_1', self.conf.thresh_1)

        # for debugging purposes
        self.nr = 0

        # load the image
        img_src = cv2.imread(fpath+fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # one in color for plotting the boxes in color
        img_src_col = cv2.imread(fpath+fname, cv2.CV_LOAD_IMAGE_COLOR)

        # manipulate the file name to something shorter
        fid = fname.split('_')[-1]
        self.fid = fid

        #if self.debug:
        self.fpath = fpath
        self.fpath_out = fpath_out

        # raise a warning if the image doesn't exist
        if img_src is None or img_src_col is None:
            if self.verbose:
                logging.warn('%s failed to load =======' % fid)
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        # ===================================================================
        # limit the area of interest (ROI)
        # vertical position and height
        y = self.conf.y
        h = self.conf.h
        # horizontal position and width
        x = self.conf.x
        w = self.conf.w

        img_sel = img_src[y:y+h,x:x+w]
        #offset = (x,y)
        # reduce the area of interest. Since this is just a reference to the
        # big array, we don't need to propagate any drawings to the full size
        # image. They both refer to the same numpy array
        # img_sel in color for plotting
        img_sel_col = img_src_col[y:y+h,x:x+w]

        # for debugging puproses, load a different image for plotting the
        # progress, othwerise we pollute the result
        if self.debug:

            try:
                os.mkdir(fpath_out)
            except OSError:
                pass
            try:
                os.mkdir(fpath_out+'debug/')
            except OSError:
                pass

            print
            print 79*'-'
            print 'fid: %s' % fid
            print 79*'-'
            idbg_col = cv2.imread(fpath+fname, cv2.CV_LOAD_IMAGE_COLOR)
            #idbg_src = cv2.imread(fpath+fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            self.idbg_sel_col = idbg_col[y:y+h,x:x+w]

            # and save the original with the ROI
            self.nr += 1
            tmp = fid.replace('.jpg', '_%02i_original.jpg' % self.nr)
            cv2.imwrite(fpath_out+'debug/'+tmp, img_sel)

#        # ===================================================================
#        # Filtering: k means clustering
#        # if it would be with RGB colors, do not collapse the colors
#        if len(img_sel.shape) == 3:
#            img_sel_z = img_sel.reshape((-1,img_sel.shape[2]))
#        else:
#            img_sel_z = img_sel.ravel()
#            # which is quicker than img_sel.reshape((-1))
#
#        k = 5  # Number of clusters
#        center, dist = vq.kmeans(img_sel_z, k)
#        code, distance = vq.vq(img_sel_z, center)
#        res = center[code]
#        img_sel = res.reshape((img_sel.shape))
#
#        if self.debug:
#            self.nr += 1
#            tmp = fid.replace('.jpg', '_%02i_kmeanclust.jpg' % self.nr)
#            cv2.imwrite(fpath_out+'debug/'+tmp, img_sel)

        # ===================================================================
        # manual treshold filtering so we can easily detect the edges
        # set the threshold relatively low here, because we will have more
        # filtering later
        thresh = thresh_1
        maxval = 255
        ret, img_sel = cv2.threshold(img_sel, thresh,maxval,cv2.THRESH_BINARY)

        if self.debug:
            self.nr += 1
            tmp = fid.replace('.jpg', '_%02i_threshold.jpg' % self.nr)
            cv2.imwrite(fpath_out+'debug/'+tmp, img_sel)

        # ===================================================================
        # Filtering: only leave the blade tip visible in img_sel
        contours = self.find_contour_tip(img_sel)

        if len(contours) == 0:
            # nothing has been found, warning has been printed in
            # find_contour_tip before

            if self.save:
                # print out the original for reference
                self.nr += 1
                tmp = fid.replace('.jpg', '_%02i_original.jpg' % self.nr)
                cv2.imwrite(fpath_out+'debug/'+tmp, img_sel_col)
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        # ===================================================================
        # get the positionion information of the blade tip
        cnt = contours[0]
        img_sel_col, boxdeg, chord, deg, lex, ley, tex, tey \
                    = self.position_tip(img_sel_col, cnt)

        if self.verbose:
            print ''

        # only consider the case if the chordlength is within boundaries
        # if not, we are not looking at the tip or we falsely id'd the tip
        c_size_cr = chord < self.conf.chord_max and chord > self.conf.chord_min
        if not c_size_cr:
            if self.verbose:
                logging.warn('%s chord out of bounds' % fid)
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        # do not allow the TE to be very close to the ROI bottom
        # or to have the LE close to ROI top
        c_pos_cr_te = tey > (y+self.conf.roi_appr)
        c_pos_cr_le = ley < (y+h-self.conf.roi_appr)
        if not c_pos_cr_te or not c_pos_cr_le:
            if self.verbose:
                logging.warn('%s tip position out of bounds' % fid)
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        # whatever is hapenning, it doesn't happen here
        ## warn if we get very big numbres from the positioning
        #if np.array([boxdeg, chord, deg, lex, ley, tex, tey]).max() > 1e10:
            #logging.warn('%s position_tip failed, > 1e10? Why??' % fid)
            #return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        if self.debug:
            self.nr += 1
            tmp = fid.replace('.jpg', '_%02i_pos_tip.jpg' % self.nr)
            cv2.imwrite(fpath_out+'debug/'+tmp, img_sel_col)

        # only draw all contours if something has been found
        if not boxdeg:
            if self.verbose:
                logging.warn('%s position_tip failed' % fid)
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        ## draw the blade tip contour
        #cv2.drawContours(img_src_col, [cnt], 0, color=(0,255,0),
                         #thickness=2, offset=offset)


        # ===================================================================
        # determine the blade number, but start with a fresh new image
        # load the image
        img_src = cv2.imread(fpath+fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img_sel = img_src[y:y+h,x:x+w]
        blade_nr = self.find_blade_nr(img_sel, cnt)

        if self.debug:
            # print some headers for what self.image will output
            replace = ('case id','twist','chord','Bnr','LE_x','LE_y','TE_y')
            print '%23s %6s %6s %3s %6s %6s %6s' % replace

        if self.verbose:
            # print some results for this file
            replace = (fid, deg, chord, blade_nr, lex, ley, tey)
            print '%s %6.2f %6.1f %3i %6.1f %6.1f %6.1f' % replace
            print '%48s %6.1f %6.1f' % ('ROIy', y+h, y)

        # ===================================================================
        # put the positioning of the blade as text in the image
#        text = '  box angle: %5.2f' % boxdeg
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thick = 2
        color = (255, 255, 255)
        org = (x, int(y*0.60))
#        cv2.putText(img_src_col, text, org, fontFace, fontScale, color, thick)
        # put the angle as text too
        text = 'chord angle: %5.2f' % deg
        org = (x, int(y*0.75))
        cv2.putText(img_src_col, text, org, fontFace, fontScale, color, thick)
        # put the angle as text too
        text = '      chord: %5.2f' % chord
        org = (x, int(y*0.90))
        cv2.putText(img_src_col, text, org, fontFace, fontScale, color, thick)

        # put the angle as text too
        text = 'blade_nr: %i' % blade_nr
        org = (int(x+(w*1.1)), int(y*1.05))
        cv2.putText(img_src_col, text, org, fontFace, fontScale, color, thick)

        # ===================================================================
        # manually draw the ROI box
        pt1, pt2 = (x,y), (x+w,y)
        pt3, pt4 = (x+w,y+h), (x, y+h)
        cv2.line(img_src_col, pt1, pt2, color=(0,110,200), thickness=2)
        cv2.line(img_src_col, pt2, pt3, color=(0,110,200), thickness=2)
        cv2.line(img_src_col, pt3, pt4, color=(0,110,200), thickness=2)
        cv2.line(img_src_col, pt4, pt1, color=(0,110,200), thickness=2)

        if self.save:
            # and also save the images
            cv2.imwrite(fpath_out+fid, img_src_col)
            #fid = fid.replace('.jpg', '') + 'sel.jpg'
            cv2.imwrite(fpath_out+'ROI/'+fid, img_sel_col)

        return [boxdeg, deg, chord, blade_nr, lex, ley, tex, tey]

    def find_contour_tip(self, img):
        """
        See if we can reduce the image to only one shape: the blade tip

        Returns
        -------

        img : ndarray
            input img with only the blade tip visible in b/w binary mode

        contours : [ ndarray(n,1,2) ]
            Should only hold n=1 contour of the blade tip. An empty list
            is returned if no contours have been found
        """

        # BE CAREFULL, findContours actually extracts the contour from
        # image passed onto it
        # find all the contours in the filtered region of interest (ROI)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # don't do anything when nothing has been found
        if len(contours) < 1:
            if self.verbose:
                logging.warn('%s no contours found in ROI' % self.fid)
            return []

        # remove all other shapes but the blade tip
        # first pass: fill only the largest area with one color: mark tip only
        for nr, cnt in enumerate(contours):
            # area of the contour
            area = cv2.contourArea(cnt)
            # aspect ratio of the box bounding the contour
            #xx,yy,ww,hh = cv2.boundingRect(cnt)
            (xx,yy), (ww,hh), theta_box = cv2.minAreaRect(cnt)
            try:
                aspect_ratio = float(hh)/ww
            except ZeroDivisionError:
                aspect_ratio = 9.99
            area_box = ww*hh

            # in debug mode we print all the contours we find on img_sel_col
            if self.debug:
                replace = (area, aspect_ratio, area_box, theta_box)
                print 'contour in ROI, nr %2i:' % nr,
                print 'A=%4i, AR=%5.2f, A_box=%8.2f, theta=%6.2f' % replace
                cv2.drawContours(self.idbg_sel_col, [cnt], 0, color=(0,255,0),
                                 thickness=2)
                # and identify which is which contour
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.65
                thick = 2
                color = (10, 10, 255)
                org = ( int(xx+ww/4.0), int(yy+hh/2.0) )
                text = '%i' % nr
                cv2.putText(self.idbg_sel_col, text, org, fontFace, fontScale,
                            color, thick)

            # a large contour with higher aspect ratio is the tip shape we
            # are looking for!
            aspect_crit = aspect_ratio > 3.7 and aspect_ratio < 6.0
            # contour area requirements
            area_gt = area > self.conf.area_tip_min
            area_lt = area < self.conf.area_tip_max
            # bounds on theta of the bounding box
            theta_box_req = abs(theta_box) < 25
            if area_gt and area_lt and aspect_crit and theta_box_req:
                contour = [cnt]
                # for debugging purposes, actually draw the contour
                if self.debug:
                    img = np.ndarray(img.shape, np.uint8)
                    img[:,:] = 255
                    cv2.drawContours(img, [cnt], 0, color=0, thickness=-1)
                # we found the tip! Ignore any other contours
                break

        if self.debug:
            # which contours did we actually found at the tip
            self.nr += 1
            tmp = self.fid.replace('.jpg', '_%02i_tip_contours.jpg' % self.nr)
            cv2.imwrite(self.fpath_out+'debug/'+tmp, self.idbg_sel_col)

            self.nr += 1
            tmp = self.fid.replace('.jpg', '_%02i_tip_only.jpg' % self.nr)
            cv2.imwrite(self.fpath_out+'debug/'+tmp, img)

#        # now we should mainly have the blade tip, filter any other edges out
#        maxval = 255
#        thresh = 250
#        retval, img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
#
#        if self.debug:
#            self.nr += 1
#            tmp = self.fid.replace('.jpg', '_%02i_tip_only_filt.jpg' % self.nr)
#            cv2.imwrite(self.fpath_out+'debug/'+tmp, img)
#
#        # BE CAREFULL, findContours actually extracts the contour from
#        # image passed onto it
#        # Now the ROI should only have the blade tip in black/white BINARY
#        # this a step to add a certain redundancy
#        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
#                                               cv2.CHAIN_APPROX_SIMPLE)
#
#        # See if we realy only found one contour on the tip
#        if len(contours) == 0:
#            # nothing has been found
#            logging.warn('%s no contours found' % self.fid)
#            return img, []
#        elif len(contours) > 1:
#            # there is more than one contour left
#            raise UserWarning, 'found more than one contour for the tip'

        # if variable contour has not been assigned, we did not find the tip
        try:
            contours = contour
        except UnboundLocalError:
            if self.verbose:
                logging.warn('%s no tip contour found in ROI' % self.fid)
            if not self.debug and self.save:
                self.nr += 1
                tmp = self.fid.replace('.jpg', '_%02i_tip_only.jpg' % self.nr)
                cv2.imwrite(self.fpath_out+'debug/'+tmp, img)
            return []

        # make the shape convex so to be sure not to include the blade nr
        # markings in some cases
        ihull = cv2.convexHull(contours[0], returnPoints=False)
        tmp = contours[0][ihull[:,0],0,:]
        contours[0] = np.reshape(tmp, (tmp.shape[0], 1, tmp.shape[1]) )

        if self.debug:
            cv2.drawContours(img, contours, 0, color=0, thickness=-1)
            self.nr += 1
            replace = '_%02i_tip_only_filt_convex.jpg' % self.nr
            tmp = self.fid.replace('.jpg', replace)
            cv2.imwrite(self.fpath_out+'debug/'+tmp, img)

        return contours

    def position_tip(self, img, cnt):
        """
        Evaluate all what has to be known from the given contour

        Returns
        -------

        img, boxdeg, chord, angle, le_x, le_y, te_x, te_y

        """

        # BOX
        # rect is a tuple( (x,y),(w,h),theta )
        rect = cv2.minAreaRect(cnt)
        # convert to box coordinates, old format it appears
        box = cv2.cv.BoxPoints(rect)
        # convert to intigers and rounding off correctly first
        box = np.int0(np.round(box, decimals=0))
        # bring to a correct contour array representation
        boxcnt = np.ndarray((box.shape[0], 1, box.shape[1]))
        boxcnt[:,0,:] = box
        ## draw the rotated contour box
        #cv2.drawContours(img, [boxcnt], 0, color=(0,0,255),
                          #thickness=2)
        # alternatively, manually draw rotated contour box
#        pt1, pt2 = tuple(box[0,:]), tuple(box[1,:])
#        pt3, pt4 = tuple(box[2,:]), tuple(box[3,:])
#        cv2.line(img, pt1, pt2, color=(0,0,255), thickness=2)
#        cv2.line(img, pt2, pt3, color=(0,0,255), thickness=2)
#        cv2.line(img, pt3, pt4, color=(0,0,255), thickness=2)
#        cv2.line(img, pt4, pt1, color=(0,0,255), thickness=2)

        # LEADING AND TRAILING EDGE
        # select the highest and lowest point in the vertical y dimension
        # note that y=0 is the top, so LE is where y is minimum. ile.max is
        # consequently wrong
        ile = cnt[:,0,:].argmax(axis=0)[1]
        ite = cnt[:,0,:].argmin(axis=0)[1]
        # format: le = (x,y)
        le = cnt[ile,0,:]
        te = cnt[ite,0,:]
        # and mark the LE and TE
        cv2.circle(img, tuple(le), 3, (0, 0, 255), -1, 8, 0)
        cv2.circle(img, tuple(te), 3, (0, 0, 255), -1, 8, 0)

        # calculate chord length and twist angle
        le = np.array(le, dtype=np.float32)
        te = np.array(te, dtype=np.float32)
        chord = math.sqrt( (le[1]-te[1])**2 + (le[0]-te[0])**2 )
        # angle = tan( x_te-xle / y_le-y_te )
        angle = 180.0*math.tan( (te[0]-le[0])/(le[1]-le[0]) ) / math.pi

        if self.debug:
            # look at the curvature at the LE,TE
            plt.plot(cnt[:,0,0], cnt[:,0,1], 'g+-')
            plt.plot(cnt[ile-10:ile+10,0,0], cnt[ile-10:ile+10,0,1], 'r*')
            if ite+10 > cnt.shape[0]:
                pass
            # FIXME: this selection does not work. Guessing on the plot
            # we are at the start/end of the array, but also not??
            plt.plot(cnt[ite-10:ite+10,0,0], cnt[ite-10:ite+10,0,1], 'bx')
            plt.plot(cnt[ile,0,0], cnt[ile,0,1], 'ks', label='LE')
            plt.plot(cnt[ite,0,0], cnt[ite,0,1], 'k>', label='TE')

        ## ELLIPSE
        #ellipse = cv2.fitEllipse(cnt)
        ## draw the ellipse
        #cv2.ellipse(img, ellipse, color=(255,0,0), thickness=2)

        return img, rect[2], chord, angle, le[0], le[1], te[0], te[1]

    def find_blade_nr(self, img, cnt):
        """
        When the blade tip contour has been established, derive the blade nr.
        Bound the

        Parameters
        ----------

        img : ndarray(n,m)
            Binary image, ideally the first threshold filtered.

        cnt : ndarray(i,1,j)
            Contour description of the selection
        """

        # how to select only the contour area of the original picture?
        # we draw the contour shape in color 1 on a 0 image
        mask = np.ndarray(img.shape, np.uint8)
        mask[:,:] = 0
        # draw the contour with color 1, the rest is the 0 background
        cv2.drawContours(mask, [cnt], 0, 1, -1)
        # multiply with img, so everything except the contour gets 0=black
        img_tip = mask*img
        # no make all what is black white
        iblack = img_tip.__le__(1)
        img_tip[iblack] = 255
        # and make the tip profile contour itself white
        cv2.drawContours(img_tip, [cnt], 0, 255, self.conf.t_tipcnt)

        if self.debug:
            self.nr += 1
            replace = '_%02i_tip_only_greyscale.jpg' % self.nr
            tmp = self.fid.replace('.jpg', replace)
            cv2.imwrite(self.fpath_out+'debug/'+tmp, img_tip)

        if self.conf.name == 'February':
            # Filtering: k means clustering, remove any small isolated ones,
            # or merge very nearby clusters into a bigger one
            # if it would be with RGB colors, do not collapse the colors
            if len(img_tip.shape) == 3:
                img_sel_z = img_tip.reshape((-1,img_tip.shape[2]))
            else:
                img_sel_z = img_tip.ravel()
                # which is quicker than img_sel.reshape((-1))
            # Number of clusters
            k = self.conf.k_find_blade_nr
            center, dist = vq.kmeans(img_sel_z, k)
            code, distance = vq.vq(img_sel_z, center)
            res = center[code]
            img_tip = res.reshape((img_tip.shape))
            # make a print of this state to see if we got the filtering ok
            if self.debug:
                self.nr += 1
                replace = '_%02i_tip_bladenr_kmeanclust.jpg' % self.nr
                tmp = self.fid.replace('.jpg', replace)
                cv2.imwrite(self.fpath_out+'debug/'+tmp, img_tip)

        elif self.conf.name == 'April':

            pass
#            cv2.HoughCircles(img_tip, cv2.cv.CV_HOUGH_GRADIENT, 1)
#
#            th = self.conf.thresh_1
#            maxth = 255
#            ret, img_tip = cv2.threshold(img_tip, th, maxth, cv2.THRESH_BINARY)

        # agressive threshold filtering: we only have the shapes of the dots
        # indicating the blade nr. Everything else is now white. So something
        # even remotely grey: make it black!
        maxval = 255
        ret, img_tip = cv2.threshold(img_tip, 200, maxval, cv2.THRESH_BINARY)
        # make a print of this state to see if we got the filtering ok
        if self.debug:
            self.nr += 1
            replace = '_%02i_tip_bladenr_thresh.jpg' % self.nr
            tmp = self.fid.replace('.jpg', replace)
            cv2.imwrite(self.fpath_out+'debug/'+tmp, img_tip)

        # discover and subtract the contours
        # BE CAREFULL, findContours actually extracts the contour from
        # image passed onto it
        contours, hierarchy = cv2.findContours(img_tip, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # the hierarchy is as follows
        # For each contour contours[i], the elements hierarchy[i][0],
        # hiearchy[i][1], hiearchy[i][2], and hiearchy[i][3] are set to 0-based
        # indices in contours of the next and previous contours at the same
        # hierarchical level: the first child contour and the parent contour,
        # respectively

#        try:
#            ppi = hierarchy[0,:,3].__eq__(0)
#        except TypeError:
#            # if there was no hierarchy found, ppi will be None
#            return -1
#        blade_nr_a = len(hierarchy[0,ppi,0])

        # add some redundancy, count in another way as well
        blade_nr_b = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # aspect ratio of the box bounding the contour
            xx,yy,ww,hh = cv2.boundingRect(cnt)
            aspect_ratio = float(hh)/ww
            if self.debug:
                replace = (area, aspect_ratio)
                print '   contour in tip: A=%6i, AR=%4.2f' % replace
            if area > 50 and area < 500 and aspect_ratio < 1.7 \
                                        and aspect_ratio > 0.5:
                blade_nr_b += 1
                # fill the dots
                cv2.drawContours(img_tip, [cnt], 0, color=255, thickness=-1)
            else:
                # do not fill the airfoil shape contour
                cv2.drawContours(img_tip, [cnt], 0, color=255, thickness=1)

        # check to see if we have the correct figure
        if self.debug:
            self.nr += 1
            replace = '_%02i_tip_bladenr.jpg' % self.nr
            tmp = self.fid.replace('.jpg', replace)
            cv2.imwrite(self.fpath_out+'debug/'+tmp, img_tip)

#        if not blade_nr_a == blade_nr_b:
#            #raise UserWarning, 'I can not count till 3'
#            logging.warn('%s can not count to 3' % self.fid)
#            blade_nr_a = -1
#
#            if not self.debug and self.save:
#                self.nr += 1
#                replace = '_%02i_tip_bladenr.jpg' % self.nr
#                tmp = self.fid.replace('.jpg', replace)
#                cv2.imwrite(self.fpath_out+'debug/'+tmp, img_tip)

        return blade_nr_b

def check_case():
    """
    Template for quickly checking a single case, including the correct
    calibrations
    """

    # ---------------------------------------------------------------------
    # definition of the calibration files for February
    calpath = 'data/'
    ycp = calpath + 'YawLaserCalibration/runs_050_051.yawcal-pol10'
    caldict_dspace_02 = {}
    caldict_dspace_02['Yaw Laser'] = ycp
    # do not calibrate tower strain in February, results are not reliable
    #tfacp = calpath + 'TowerStrainCal/towercal-pol1_fa'
    #tsscp = calpath + 'TowerStrainCal/towercal-pol1_ss'
    #caldict_dspace_02['Tower Strain For-Aft'] = tfacp
    #caldict_dspace_02['Tower Strain Side-Side'] = tsscp

    caldict_blade_02 = {}
    bcp = 'data/BladeStrainCal/'
    caldict_blade_02[0] = bcp + '0214_run_172_ch1_0214_run_173_ch1.pol1'
    caldict_blade_02[1] = bcp + '0214_run_172_ch2_0214_run_173_ch2.pol1'
    caldict_blade_02[2] = bcp + '0214_run_172_ch3_0214_run_173_ch3.pol1'
    caldict_blade_02[3] = bcp + '0214_run_172_ch4_0214_run_173_ch4.pol1'
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    tw_cal = 'opt'
    # definition of the calibration files for April
    ycp04 = calpath + 'YawLaserCalibration-04/runs_289_295.yawcal-pol10'
    # for the tower calibration, the yaw misalignment is already taken
    # into account in the calibration polynomial, no need to include the
    # yaw angle in the calibration. We always measure in the FA,SS dirs
    # if that needs to be converted, than do sin/cos psi to have the
    # components aligned with the wind
    tfacp  = calpath + 'TowerStrainCal-04/'
    tfacp += 'towercal_249_250_251_yawcorrect_fa-cal_pol1_%s' % tw_cal
    tsscp  = calpath + 'TowerStrainCal-04/'
    tsscp += 'towercal_249_250_251_yawcorrect_ss-cal_pol1_%s' % tw_cal
    caldict_dspace_04 = {}
    caldict_dspace_04['Yaw Laser'] = ycp04
    caldict_dspace_04['Tower Strain For-Aft'] = tfacp
    caldict_dspace_04['Tower Strain Side-Side'] = tsscp
    # and to convert to yaw coordinate frame of reference
    target_fa = calpath + 'TowerStrainCalYaw/psi_fa_max_%s' % tw_cal
    caldict_dspace_04['psi_fa_max'] = target_fa
    target_ss = calpath + 'TowerStrainCalYaw/psi_ss_0_%s' % tw_cal
    caldict_dspace_04['psi_ss_0'] = target_ss

    caldict_blade_04 = {}
    bcp = 'data/BladeStrainCal/'
    caldict_blade_04[0] = bcp + '0412_run_357_ch1_0412_run_358_ch1.pol1'
    caldict_blade_04[1] = bcp + '0412_run_357_ch2_0412_run_358_ch2.pol1'
    caldict_blade_04[2] = bcp + '0412_run_356_ch3_0412_run_358_ch3.pol1'
    caldict_blade_04[3] = bcp + '0412_run_356_ch4_0412_run_358_ch4.pol1'
    # ---------------------------------------------------------------------

    resfile = '0209_run_008_10ms_dc0.5_stiffblades'
    resfile = '0210_run_032_6.0ms_dc0_freeyawsteady0_stiffblades_pwm1000'
    resfile = '0404_run_216_10.0ms_dc0.8_flexies_fixyaw_lowrpm'
    resfile = '0405_run_277_9.0ms_dc1_flexies_freeyaw_highrpm'

    resfile = '0410_run_303_8ms_dc0.5_flexies_freeyawforced_yawerrorrange_'
    resfile += 'fastside_highrpm'
    resfile = '0413_run_415_8ms_dc0_stiffblades_freeyaw_forced'
    resfile = '0412_run_366_9ms_dc1_stiff_highrpm_posb_damper_notunnelheat'

    resfile = '0412_run_363ah_9.5ms_dc0.45_stiff_dcsweep'
    resfile = '0214_run_162_9.0ms_dc1_flexies_fixyaw_shutdown_pwm1000_highrpm'
    resfile = '0405_run_246_9.0ms_dc0_flexies_fixyaw_highrpm_shutdown'
    resfile = '0413_run_419_8ms_dc0.6_stiffblades_freeyaw_forced'

    respath = 'database/symlinks_all/'
    res = ComboResults(respath, resfile, silent=False, sync=True)
    res.dspace.remove_rpm_spike(plot=False)
    # RPM from pulse only returns the pulses, nothing else is done
    #res.dspace.rpm_from_pulse(plot=True, h=0.2)
    res._calibrate_dspace(caldict_dspace_04)
    res._calibrate_blade(caldict_blade_04)
    res.dspace.remove_rpm_spike()

    # and see if the staircase filter can get what we want
    figpath = 'database/steady_rpm_points/'
    irpm = res.dspace.labels_ch['RPM']
    iyaw = res.dspace.labels_ch['Yaw Laser']
    ifa = res.dspace.labels_ch['Tower Strain For-Aft']

    # because we don't get the indices in a reliable manner, do it 3 times and
    # hope we have the same points?? Or use other filtering technique?
    # for the RPM we have
    sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                   figfile=resfile+'_rpm', runid=resfile+'_rpm')
    time_stair, data_stair = sc.setup_filter(res.dspace.time,
                res.dspace.data[:,irpm], smooth_window=1.5,
                cutoff_hz=False, dt=1, dt_treshold=5.0e-4,
                stair_step_tresh=5.0, smoothen='moving',
                points_per_stair=500)
                # dt_treshold=0.00008, start=5000,
    # and for the thrust
    sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                   figfile=resfile+'_fa', runid=resfile+'_fa')
    time_stair, data_stair = sc.setup_filter(res.dspace.time,
                res.dspace.data[:,ifa], smooth_window=1.5,
                cutoff_hz=False, dt=1, dt_treshold=2.0e-4,
                stair_step_tresh=1.0, smoothen='moving',
                points_per_stair=1000)
    # and for the yaw angle
    sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                   figfile=resfile+'_yaw', runid=resfile+'_yaw')
    time_stair, data_stair = sc.setup_filter(res.dspace.time,
                res.dspace.data[:,iyaw], smooth_window=1.5,
                cutoff_hz=False, dt=1, dt_treshold=2.0e-4,
                stair_step_tresh=4.0, smoothen='moving',
                points_per_stair=1000)



if __name__ == '__main__' :

    dummy = None
