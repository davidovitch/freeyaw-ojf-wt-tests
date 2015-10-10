# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:08:12 2013

@author: dave
"""

import numpy as np
import scipy as sp
import pylab as plt
from scipy.interpolate import UnivariateSpline
from matplotlib.ticker import FormatStrFormatter

import plotting
from misc import calc_sample_rate
from filters import Filters

class StairCase:
    
    def __init__(self, **kwargs):
#    def __init__(self, time, data, label, figpath, figfile):
        """
        """
        self.plt_progress = kwargs.get('plt_progress', False)
        self.runid = kwargs.get('runid', 'dummy')
        self.pprpath = kwargs.get('pprpath', None)
        self.figpath = kwargs.get('figpath', None)
        self.figfile = kwargs.get('figfile', None)
#        time_stair, data_stair = self.setup_filter(time, data, label,
#                            start=32500, end=-28001, figpath=figpath, 
#                            figfile=figfile, dt_treshold=2e-6)
    
    def order_staircase(self, time_trim, data_trim, arg_trim):
        """
        Look for a staircase patern in the data and group per stair
        
        Note, there is also a CYTHON version of this function!!
        """
        
        # -------------------------------------------------
        # getting the staircase data out step by step
        # -------------------------------------------------
        
        # cycle through all the data stairs and get the averages
        start, end = 0, 0
        i, imax, j = 1, 0, 0
        nr_stairs = 35
        nr_data_chops = len(data_trim)
        data_ordered = np.ndarray((nr_data_chops, nr_stairs))
        data_ordered[:,:] = np.nan
        # put the first point already in
        data_ordered[0,0] = data_trim[0]
        
        time_ordered = np.ndarray(data_ordered.shape)
        time_ordered[:,:] = np.nan
        # put the first point already in
        time_ordered[0,0] = time_trim[0]
        
        # keep start/stop arguments for each stair of the original dataset
        arg_ordered = sp.zeros((2, nr_stairs), dtype=np.int)
        # start of the first is already known
        arg_ordered[0,0] = arg_trim[0]
        
        print 'start walking through stair case:'
        print 'meanstair-last  previous-last  stairnr  mean',
        print '       istart    istop'
        for kk in xrange(1,len(data_trim)):
            k = data_trim[kk]
            # keep track of the different mean levels, aka stairs
            # is the current number in the same mean category?
            # the nonnan_i creation is computationally heavy!
#            nonnan_i = np.isnan(data_ordered[:,j]).__invert__() 
#            print 'delta: %2.2e' % abs(data_ordered[nonnan_i,j].mean() - k)/k
#            if abs(data_ordered[nonnan_i,j].mean() - k) < stair_step_tresh:
            # more simple: just look at the previous value instead of the mean
            if abs(data_ordered[i-1,j] - k) < self.stair_step_tresh:
                data_ordered[i,j] = k
                time_ordered[i,j] = time_trim[kk]
                # mark the possible end of the stair
                arg_ordered[1,j] = arg_trim[kk]
                i += 1
            # else we have a new stair
            else:
                nonnan_i = np.isnan(data_ordered[:,j]).__invert__()
                
                print '%2.2e' % abs(data_ordered[nonnan_i,j].mean() - k),
                print '       %2.2e' % abs(data_ordered[i-1,j] - k),
                print '%11i' % j,
                print '%12.7f' % data_ordered[nonnan_i,j].mean(),
                print '%8i' % arg_ordered[0,j],
                print '%8i' % arg_ordered[1,j]
                # keep track on maximum points per stair
                if i > imax:
                    imax = i
                # go to a new stair
                j += 1
                try:
                    # current point is the first of a new stair
                    arg_ordered[0,j]  = arg_trim[kk]
                    data_ordered[0,j] = k
                    time_ordered[0,j] = time_trim[kk]
                except IndexError:
                    msg = 'dt_treshold is too high, found more than 35 stairs'
                    raise IndexError, msg
                # and prepare for the next point
                i = 1
        
        print 'found', j, 'stairs'
        # data_ordered array was made too large, cut off empty spaces
        data_ordered = data_ordered[:imax+1,:j+1]
        time_ordered = time_ordered[:imax+1,:j+1]
        arg_ordered = arg_ordered[:,:j+1]
        # select only the values, ignore nans
        nonnan_i = np.isnan(data_ordered).__invert__()
        data_stair = np.ndarray((j+1))
        time_stair = np.ndarray((j+1))
        arg_stair = np.ndarray((2,j+1), dtype=np.int)
        # and save for each found stair the everage in a new array seperately
        # we don't know exactly how many stairs will get selected due to the
        # points_per_stair criterium
        i = 0
        print '\neffectively selected stairs:'
        print 'stair_i'.rjust(7) + 'mean'.rjust(11) + 'time'.rjust(9),
        print 'delta'.rjust(9)
        for kk in xrange(j+1):
            # each stair should have a least a certain amount of points
            points_stair = data_ordered[nonnan_i[:,kk],kk]
            if len(points_stair) >= self.points_per_stair:
                data_stair[i] = data_ordered[nonnan_i[:,kk],kk].mean()
                time_stair[i] = time_ordered[nonnan_i[:,kk],kk].mean()
                arg_stair[:,i] = arg_ordered[:,kk]
                if i >0 : delta = data_stair[i] - data_stair[i-1]
                else: delta = 0
                toprint = (i, data_stair[i], time_stair[i], delta)
                print '%7i %10.3f %8.2f %9.3f' % toprint
                i += 1
        
        # remove empty stair data
        data_stair = data_stair[:i]
        time_stair = time_stair[:i]
        arg_stair = arg_stair[:,:i]
        
        return data_ordered, time_stair, data_stair, arg_stair
    
    def setup_filter(self, time, data, **kwargs):
        """
        Load the callibration runs and convert voltage signal to yaw angles
        
        Parameters
        ----------
        
        time : ndarray(k)
        
        data : ndarray(k)
        
        Returns
        -------
        
        time_stair : ndarray(n)
            Average time stamp over the stair step
        
        data_stair : ndarray(n)
            Average value of the selected stair step
        
        
        """
        # time and data should both be 1D and have the same shape!
        assert time.shape == data.shape
        
        runid = kwargs.get('runid', self.runid)
        
        # smoothen method: spline or moving average
        smoothen = kwargs.get('smoothen', 'spline')
        # what is the window of the moving average in seconds
        smooth_window = kwargs.get('smooth_window', 2)
        
        # specify the window of the staircase
        #start, end = 30100, -30001
        start = kwargs.get('start', 0)
        end = kwargs.get('end', len(time))
        dt = kwargs.get('dt', 1)
        cutoff_hz = kwargs.get('cutoff_hz', None)
        self.points_per_stair = kwargs.get('points_per_stair', 20)
        # at what is the minimum required value on dt or dt2 for a new stair
        self.stair_step_tresh = kwargs.get('stair_step_tresh', 1)
        
#        plot_data = kwargs.get('plot_data', False)
#        respath = kwargs.get('respath', None)
#        run = kwargs.get('run', None)
        
        # sample rate of the signal
        sample_rate = calc_sample_rate(time)
        
        # prepare the data
        time = time[start:end]
        # the actual raw signal
        data = data[start:end]
        
        # -------------------------------------------------
        # Progress plotting
        # ----------------------------------------------
        if self.plt_progress:
            plt.figure()
            Pxx, freqs = plt.psd(data, Fs=sample_rate, label='data')
            plt.show()
            
            plt.figure()
            plt.plot(time, data, label='raw data')
        
        # -------------------------------------------------
        # setup plot
        # -------------------------------------------------
#        labels = np.ndarray(3, dtype='<U100')
#        labels[0] = label
#        labels[1] = 'yawchan derivative'
#        labels[2] = 'psd'
        
        # remove any underscores for latex printing
        grandtitle = self.figfile.replace('_', '\_')
        plot = plotting.A4Tuned(scale=1.5)
        plot.setup(self.figpath+self.figfile+'_filter', nr_plots=3, 
                   grandtitle=grandtitle, wsleft_cm=1.5, wsright_cm=1.8,
                   hspace_cm=1.2, size_x_perfig=10, size_y_perfig=5,
                   wsbottom_cm=1.0, wstop_cm=1.5)
        
        # -------------------------------------------------
        # plotting original and smoothend signal
        # -------------------------------------------------
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 1)
        ax1.plot(time, data, 'b', label='raw data', alpha=0.6)
        data_raw = data.copy()
        
        # -------------------------------------------------
        # signal frequency filtering, if applicable
        # -------------------------------------------------
        # filter the local derivatives if applicable
        if cutoff_hz:
            filt = Filters()
            data_filt, N, delay = filt.fir(time, data, ripple_db=20,
                            freq_trans_width=0.5, cutoff_hz=cutoff_hz, 
                            figpath=self.figpath, 
                            figfile=self.figfile + 'filter_design',
                            sample_rate=sample_rate, plot=False,)
            
            if self.plt_progress:
                # add the results of the filtering technique
                plt.plot(time[N-1:], data_filt[N-1:], 'r', label='freq filt')
            
            data = data_filt
            time = time[N-1:]#-delay
            
        else:
            N = 1
        
        # -------------------------------------------------------
        # smoothen the signal with some splines or moving average
        # -------------------------------------------------------
        # NOTE: the smoothing will make the transitions also smoother. This
        # is not good. The edges of the stair need to be steep!
        # for the binary data this is actually a good thing, since the dt's
        # are almost always the same between time steps. We would otherwise
        # need a dt based on several time steps
        if smoothen == 'splines':
            print 'start applying spline ...',
            uni_spline = UnivariateSpline(time, data)
            data = uni_spline(time)
            print 'done!'
            NN = 0 # no time shift due to filtering?
            if self.plt_progress:
                plt.plot(time, data, label='spline data')
        
        elif smoothen == 'moving':
            print 'start calculating movering average ...',
            filt = Filters()
            # take av2s window, calculate the number of samples per window
            ws = smooth_window*sample_rate
            data = filt.smooth(data, window_len=ws, window='hanning')
            NN = len(data) - len(time)
            data = data[NN:]
            print 'done!'
            
            if self.plt_progress:
                plt.plot(time, data, label='moving average')
        
        else:
            raise ValueError, 'smoothen method should be moving or splines'
        
        # -------------------------------------------------
        # additional smoothening: downsampling
        # -------------------------------------------------
        # and up again in order not to brake the plotting further down
        time_down = np.arange(time[0], time[-1], 0.1)
        data_down = sp.interpolate.griddata(time, data, time_down)
        # and upsampling again
        data = sp.interpolate.griddata(time_down, data_down, time)
        
        # -------------------------------------------------
        # plotting original and smoothend signal
        # -------------------------------------------------
        ax1.plot(time, data, 'r', label='data smooth')
        ax1.grid(True)
        leg1 = ax1.legend(loc='best')
        leg1.get_frame().set_alpha(0.5)
        ax1.set_title('smoothing method: ' + smoothen)
        
        # -------------------------------------------------
        # local derivatives of the signal and filtering
        # -------------------------------------------------
        data_dt = np.ndarray(data.shape)
        data_dt[1:] = data[1:] - data[0:-1]
        data_dt[0] = np.nan
        data_dt = np.abs(data_dt)
        
        # frequency filter was applied here originally
        data_filt_dt = data_dt
        
        # if no threshold is given, just take the 20% of the max value
        dt_max = np.nanmax(np.abs(data_filt_dt))*0.2
        dt_treshold = kwargs.get('dt_treshold', dt_max)
        
        # -------------------------------------------------
        # filter dt or dt2 above certain treshold?
        # -----------------------------------------------
        # only keep values which are steady, meaning dt signal is low!
        
        if dt == 2:
            tmp = np.ndarray(data_filt_dt.shape)
            tmp[1:] = data_filt_dt[1:] - data_filt_dt[0:-1]
            tmp[0] = np.nan
            data_filt_dt = tmp
        # based upon the filtering, only select data points for which the 
        # filtered derivative is between a certain treshold
        staircase_i = np.abs(data_filt_dt).__ge__(dt_treshold)
        # reduce to 1D
        staircase_arg=np.argwhere(np.abs(data_filt_dt)<=dt_treshold).flatten()
        
        # -------------------------------------------------
        # replace values for too high dt with Nan
        # ------------------------------------------------
        
        # ---------------------------------
        # METHOD version2, slower because of staircase_arg computation above
        data_masked = data.copy()
        data_masked[staircase_i] = np.nan
        
        data_masked_dt = data_filt_dt.copy()
        data_masked_dt[staircase_i] = np.nan
        
        data_trim = data[staircase_arg]
        time_trim = time[staircase_arg]
        
        print 'max in data_masked_dt:', np.nanmax(data_masked_dt)
        # ---------------------------------
        # METHOD version2, faster if staircase_arg is not required!
        ## make a copy of the original signal and fill in Nans on the selected 
        ## values
        #data_masked = data.copy()
        #data_masked[staircase_i] = np.nan
        #
        #data_masked_dt = data_filt_dt.copy()
        #data_masked_dt[staircase_i] = np.nan
        #
        ## remove all the nan values
        #data_trim = data_masked[np.isnan(data_masked).__invert__()]
        #time_trim = time[np.isnan(data_masked).__invert__()]
        #
        #dt_noise_treshold = np.nanmax(data_masked_dt)
        #print 'max in data_masked_dt', dt_noise_treshold
        # ---------------------------------
        
#        # figure out which dt's are above the treshold
#        data_trim2 = data_trim.copy()
#        data_trim2.sort()
#        data_trim2.
#        # where the dt of the masked format is above the noise treshold,
#        # we have a stair
#        data_trim_dt = np.abs(data_trim[1:] - data_trim[:-1])
#        argstairs = data_trim_dt.__gt__(dt_noise_treshold)
#        data_trim2 = data_trim_dt.copy()
#        data_trim_dt.sort()
#        data_trim_dt.__gt__(dt_noise_treshold)
        
        # -------------------------------------------------
        # intermediate checking of the signal
        # -------------------------------------------------
        if self.plt_progress:
            # add the results of the filtering technique
            plt.plot(time[N-1:], data_masked[N-1:], 'rs', label='data red')
            plt.legend(loc='best')
            plt.grid(True)
            plt.twinx()
#            plt.plot(time, data_filt_dt, label='data_filt_dt')
            plt.plot(time, data_masked_dt, 'm', label='data\_masked\_dt',
                     alpha=0.4)
            plt.legend(loc='best')
            plt.show()
            print 'saving plt_progress:', 
            print self.figpath+'filter_design_progress.png'
            plt.savefig(self.figpath+'filter_design_progress.png')
        
        # -------------------------------------------------
        # check if we have had sane filtering
        # -------------------------------------------------
        
        print 'data      :', data.shape
        print 'data_trim :', data_trim.shape
        print 'trim ratio:', len(data)/len(data_trim)
        
        # there should be at least one True value
        assert staircase_i.any()
        # they can't all be True, than filtering is too heavy
        if len(data_trim) < len(data)*0.01:
            msg = 'dt_treshold is too low, not enough data left'
            raise ValueError, msg
        # if no data is filtered at all, filtering is too conservative
        elif len(data_trim) > len(data)*0.95:
            msg = 'dt_treshold is too high, too much data left'
            raise ValueError, msg
        # if the data array is too big, abort on memory concerns
        if len(data_trim) > 200000:
            msg = 'too much data points for stair case analysis (cfr memory)'
            raise ValueError, msg
        
        # -------------------------------------------------
        # read the average value over each stair (time and data)
        # ------------------------------------------------
        #try:
            ##np.save('time_trim', time_trim)
            ##np.save('data_trim', data_trim)
            ##np.save('staircase_arg', staircase_arg)
            ##tmp = np.array([self.points_per_stair, self.stair_step_tresh])
            ##np.save('tmp', tmp)
            #data_ordered, time_stair, data_stair, arg_stair \
                #= cython_func.order_staircase(time_trim, data_trim, 
                #staircase_arg, self.points_per_stair, self.stair_step_tresh)
        #except ImportError:
        data_ordered, time_stair, data_stair, arg_stair \
            = self.order_staircase(time_trim, data_trim, staircase_arg)
        
        # convert the arg_stair to a flat set and replace start/stop pairs
        # with all indices in between. Now we can select all stair values
        # in the raw dataset
        arg_st_fl = np.empty(data_raw.shape, dtype=np.int)
        i = 0
        for k in range(arg_stair.shape[1]):
            #print '%6i %6i' % (arg_stair[0,k],arg_stair[1,k])
            tmp = np.arange(arg_stair[0,k], arg_stair[1,k]+1, 1, dtype=np.int)
            #print tmp, '->', i, ':', i+len(tmp)
            arg_st_fl[i:i+len(tmp)] = tmp
            i += len(tmp)
        # remove the unused elements from the array
        arg_st_fl = arg_st_fl[:i]
        
        # -------------------------------------------------
        # plotting of smoothen signal and stairs
        # -------------------------------------------------
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 2)
        ax1.plot(time, data, label='data smooth', alpha=0.6)
        # add the results of the filtering technique
        
        ax1.plot(time[N-1:], data_masked[N-1:], 'r', label='data masked')
#        ax1.plot(time[N-1:], data_filt[N-1:], 'g', label='data_filt')
        # also include the selected chair data
        figlabel = '%i stairs' % data_stair.shape[0]
        ax1.plot(time_stair, data_stair, 'ko', label=figlabel, alpha=0.4)
        ax1.grid(True)
        # the legend, on or off?
        #leg1 = ax1.legend(loc='upper left')
        #leg1.get_frame().set_alpha(0.5)
        # -------------------------------------------------
        # plotting derivatives on right axis
        # -------------------------------------------------
        ax1b = ax1.twinx()
#        ax1b.plot(time[N:]-delay,data_s_dt[N:],alpha=0.2,label='data_s_dt')
        ax1b.plot(time[N:], data_filt_dt[N:], 'r', alpha=0.35,
                  label='data\_filt\_dt')
        majorFormatter = FormatStrFormatter('%8.1e')
        ax1b.yaxis.set_major_formatter(majorFormatter)
#        ax1b.plot(time[N:], data_masked_dt[N:], 'b', alpha=0.2,
#                  label='data_masked_dt')
#        ax1b.plot(time[N-1:]-delay, filtered_x_dt[N-1:], alpha=0.2)
#        leg1b = ax1b.legend(loc='best')
#        leg1b.get_frame().set_alpha(0.5)
#        ax1b.grid(True)
        
        # -------------------------------------------------
        # 3th plot to check if the raw chair signal is ok
        # -------------------------------------------------
        
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 3)
        ax1.plot(time[arg_st_fl], data_raw[arg_st_fl], 'k+', label='rawstair',
                 alpha=0.1)
        ax1.plot(time[N-1:], data_masked[N-1:], 'r', label='data masked')
        ax1.set_xlabel('time [s]')
        
        # -------------------------------------------------
        # the power spectral density
        # -------------------------------------------------
#        ax3 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 3)
#        Pxx, freqs = ax3.psd(data, Fs=sample_rate, label='data smooth')
##        Pxx, freqs = ax3.psd(data_dt, Fs=sample_rate, label='data_dt')
##        Pxx, freqs = ax3.psd(data_filt_dt[N-1:], Fs=sample_rate,
##                             label='data_filt_dt')
#        ax3.legend()
##        print Pxx.shape, freqs.shape

        plot.save_fig()
        
        # -------------------------------------------------
        # get amplitudes of the stair edges
        # -------------------------------------------------
        
#        # max step
#        data_trim_dt_sort = data_trim_dt.sort()[0]
#        # estimate at what kind of a delta we are looking for when changing
#        # stairs
#        data_dt_std = data_trim_dt.std()
#        data_dt_mean = (np.abs(data_trim_dt)).mean()
#        
#        time_data_dt = np.transpose(np.array([time, data_filt_dt]))
#        data_filt_dt_amps = HawcPy.dynprop().amplitudes(time_data_dt, h=1e-3)
#
#        print '=== nr amplitudes'
#        print len(data_filt_dt_amps)
#        print data_filt_dt_amps
        
        # -------------------------------------------------
        # save the data
        # -------------------------------------------------
        
        filename = runid + '-time_stair'
        np.savetxt(self.pprpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(self.pprpath + filename, data_stair)
        
        # in order to maintain backwards compatibility, save the arguments
        # of the stair to self
        self.arg_st_fl = arg_st_fl # flat, contains all indices on the stairs
        # start/stop indeces for stair k = arg_stair[0,k], arg_stair[1,k]
        self.arg_stair = arg_stair
        
        
        return time_stair, data_stair


    def polyfit(self, data_cal, data_stair, **kwargs):
        """
        Fit the staircase data with a polynomial and print the result.
        
        Paramters
        ---------
        
        data_cal : ndarray(n)
            Measurement data in data_stair will be mapped to data_cal. This
            is in principle a yaw angle, blade root bending moment, etc.
            Plotted on the vertical y axis.
        
        data_stair : ndarray(n)
            Raw measurement data: voltage, binary output signal, etc.
            Plotted on the horizontal x axis.
        
        order : int, default=10
            Order of the polynomial fit.
        
        step : float, default=1% of data_stair range
        
        """
        
        order = kwargs.get('order', 10)
#        datarange = data_cal.max() - data_cal.min()
        # default step is 1% of the data range
#        step = kwargs.get('step', datarange*0.001)
        
        # plot settings data_stair
        xlabel = kwargs.get('xlabel', 'data\_cal')
        ylabel = kwargs.get('ylabel', 'data\_stair')
        ylabel_err = kwargs.get('ylabel_err', 'data\_cal error')
#        err_label_abs = kwargs.get('err_label_abs', 'data\_cal err abs')
        err_label_rel = kwargs.get('err_label_rel', 'data\_cal err \%')
        
#        # load the data if not given
#        f = self.runid + '-data_stair'
#        data_stair = kwargs.get('data_stair', np.loadtxt(self.pprpath+f))
        
        # make sure calibration data and stair data have the same shapes
        if not data_stair.shape == data_cal.shape:
            print '\n============================================'
            print data_stair.shape, data_cal.shape
            raise ValueError, 'data_stair, data_cal should have same shapes'
        
        
        # ---------------------------------------------------------
        # Create the transformation function
        # ---------------------------------------------------------
        polx = np.polyfit(data_stair, data_cal, order)
        data_cal_polx = np.polyval(polx, data_stair)
        print self.runid, 'polyfit:', polx
        # and save the data
        filename = self.runid + '-cal-pol' + str(order)
        np.savetxt(self.pprpath + filename, polx)
        filename = self.runid + '-data_cal'
        np.savetxt(self.pprpath + filename, data_cal)      
        
        # ---------------------------------------------------------
        # Errors actual interpolated/polyfitted data
        # ---------------------------------------------------------
        
        err_rel = np.abs((data_cal_polx-data_cal)/data_cal)*100.0
#        err_abs = np.abs(data_cal_polx-data_cal_hd)
        
        # ---------------------------------------------------------
        # plotting the calibrated signal
        # ---------------------------------------------------------
        figfile = self.runid + '_calibration_result' + '_order_' + str(order)
        plot = plotting.A4Tuned(scale=1.5)
        pwx = plotting.TexTemplate.pagewidth
        pwy = plotting.TexTemplate.pagewidth*0.4
        plot.setup(self.figpath+figfile, grandtitle=None, nr_plots=1,
                         wsleft_cm=1.8, wsright_cm=1.8, hspace_cm=2.0,
                         size_x_perfig=pwx, size_y_perfig=pwy, wstop_cm=0.7,
                         wsbottom_cm=1.0)
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 1)
        ax1.plot(data_cal, data_stair, 'rs', label='data', alpha=0.5)
        ax1.plot(data_cal_polx, data_stair, 'k',
                 label='fit, order '+str(order))
        #ax1.plot(data_cal_hd, data_stair_hd, 'g', label='hd')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(figfile.replace('_', ' '))
        ax2 = ax1.twinx()
        ax2.plot(data_cal, err_rel, 'g--', alpha=0.5, label=err_label_rel)
#        ax2.plot(data_cal_hd, err_abs, 'g', alpha=0.5,
#                 label=err_label_abs)
        ax2.set_ylabel(ylabel_err)
        
        # only one legend
        lines = ax1.lines + ax2.lines
        labels = [l.get_label() for l in lines]
        leg = ax2.legend(lines, labels, loc='upper left')
        leg.get_frame().set_alpha(0.7)
        # or legends left and right
#        leg1 = ax1.legend(loc='upper left')
#        leg2 = ax2.legend(loc='lower right')
#        leg2.get_frame().set_alpha(0.5)
#        leg1.get_frame().set_alpha(0.5)
        
        ax1.grid(True)
        plot.save_fig()
