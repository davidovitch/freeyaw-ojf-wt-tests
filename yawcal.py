# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:29:28 2013

@author: dave
"""

import math
import logging
import os

import numpy as np
import scipy as sp
import pylab as plt
import sympy

import plotting
import ojfresult
from ojfresult import calc_sample_rate
from filters import Filters
from staircase import StairCase


RESDATA_02 = 'data/raw/02/calibration/'
RESDATA_04 = 'data/raw/04/calibration/'
CALPATH = 'data/calibration/'


class YawCalibration:

    def __init__(self):
        """
        """
        self.figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        self.pprpath = self.figpath

    def load_cal_dataset(self, run, title, psi_step_deg=0.01):
        """
        Merge, interpolate and fit a calibration data set
        Than plot a nice graph that is also to be used in the thesis
        """
        figpath = self.figpath
        pprpath = self.pprpath

        # ---------------------------------------------------------
        # Load calibration dataset
        # ---------------------------------------------------------

        filename = run + '.yawcal-psiA-stairA'
        savearray = np.loadtxt(pprpath + filename)
        psi_A = savearray[:,0].copy()
        stair_A = savearray[:,1].copy()

        filename = run + '.yawcal-psiB-stairB'
        savearray = np.loadtxt(pprpath + filename)
        psi_B = savearray[:,0].copy()
        stair_B = savearray[:,1].copy()

        # ---------------------------------------------------------
        # Interpolate A and B side to same grid
        # ---------------------------------------------------------
#        pol2_err
        # interpolate to a regular grid, rounded to psi_step_deg?
        psi_hd_A = np.arange(psi_A[0], psi_A[-1], psi_step_deg)
        stair_hd_A = sp.interpolate.griddata(psi_A, stair_A, psi_hd_A)

        psi_hd_B = np.arange(psi_B[0], psi_B[-1], psi_step_deg)
        stair_hd_B = sp.interpolate.griddata(psi_B, stair_B, psi_hd_B)

        # ---------------------------------------------------------
        # Merge A and B into one data series
        # ---------------------------------------------------------

        # how close to zero do we have to look to find index of psi=0
        psi_0_appr = psi_step_deg*0.7

        psi_hd = np.arange(psi_B[0], psi_A[-1], psi_step_deg)
        # find the overlap between psi_A and psi_B
        stair_hd_AB = np.ndarray((len(psi_hd),2))
        stair_hd_AB[:,:] = np.nan

        # starting point of psi_A on the large grid
        A_0i = np.abs(psi_hd-psi_A[0]).__le__(psi_0_appr).argmax()
        # make sure we only have found one maximum!
        nr_found = len(psi_hd[np.abs(psi_hd-psi_A[0]).__le__(psi_0_appr)])
        if not nr_found == 1:
            msg = 'found %i point(s) close to psi=%f1.3' % (psi_A[0], nr_found)
            raise ValueError, msg

        # there is a chance that A_0i is 1 index of
        if len(stair_hd_AB[A_0i:,1]) == len(stair_hd_A)+1:
            stair_hd_AB[A_0i+1:,1] = stair_hd_A
        else:
            stair_hd_AB[A_0i:,1] = stair_hd_A

        stair_hd_AB[0:len(stair_hd_B),0] = stair_hd_B


        # and now put them in one continues series
        stair_hd = stair_hd_AB[:,0]

        # psi=0 zero index
        psi0i = np.abs(psi_hd).__le__(psi_0_appr).argmax()
        nr_found = len(psi_hd[np.abs(psi_hd).__le__(psi_0_appr)])
        if not nr_found == 1:
            raise ValueError, 'found %i point(s) close to psi=0' % nr_found

        # and complete with the A part, start at psi=0
        stair_hd[psi0i:] = stair_hd_AB[psi0i:,1]

        # ---------------------------------------------------------
        # Create the transformation function
        # ---------------------------------------------------------
        # x values are what is given in the measurements: voltage, so stair_hd
        # the transformation function should convert voltages to yaw angle psi

        pol10 = np.polyfit(stair_hd, psi_hd, 10, full=False)
        psi_poly10 = np.polyval(pol10, stair_hd)


        # ---------------------------------------------------------
        # Errors
        # ---------------------------------------------------------

        # then we can also plot the error in the overlap range, in %B
        AB_err = np.abs((stair_hd_AB[:,0]-stair_hd_AB[:,1])/stair_hd_AB[:,0])
        AB_err *= 100.

        # error wrt the fitted dataset
        poly_err = np.abs((psi_poly10-psi_hd)/psi_poly10)*100.
        # ignore range around zero
        poly_err[psi0i-150:psi0i+150] = np.nan

        # ---------------------------------------------------------
        # plotting the calibrated signal with errors
        # ---------------------------------------------------------
        figfile = 'yaw_calibration_' + run + '_err'
        grandtitle = ('yaw_calibration_' + run).replace('_', '-')
        plot = plotting.A4Tuned(scale=1.5)
        figx = plotting.TexTemplate.pagewidth
        figy = plotting.TexTemplate.pagewidth*0.6
        plot.setup(figpath+figfile, nr_plots=1, grandtitle=grandtitle,
                   figsize_x=figx, figsize_y=figy, wsleft_cm=1.4,
                   wsright_cm=1.4, wstop_cm=1.3, wsbottom_cm=1.3)
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 1)
        # note that we haven't read the A ext position!
        ax1.plot(psi_A, stair_A, 'rs', label='A', alpha=0.5)
        ax1.plot(psi_B, stair_B, 'bo', label='B', alpha=0.5)
#        ax1.plot(psi_hd, stair_hd_AB[:,1], 'm', label='A_hd')
#        ax1.plot(psi_hd, stair_hd_AB[:,0], 'k', label='B_hd')
#        ax1.plot(psi_hd, stair_hd, 'k', label='interpolated')
        ax1.plot(psi_poly10, stair_hd, 'k', label='polyfit')
        leg1 = ax1.legend(loc='upper left')
        ax1.set_xlabel('Yaw angle $\psi$ [deg]')
        ax1.set_ylabel('Laser output signal [V]')

        # make an additional plot with the error bars
        ax2 = ax1.twinx()
        ax2.plot(psi_hd, AB_err, 'r', alpha=0.7, label='overlap err A-B [\%]')
        ax2.plot(psi_hd, np.abs(psi_poly10-psi_hd), 'g', alpha=0.5,
                 label='polyfit err $\psi$ [deg]')
#        ax2.plot(psi_hd, poly_err, 'b', alpha=0.3,
#                 label='polyfit err $\psi$ [%]')
        leg2 = ax2.legend(loc='lower right')
        leg2.get_frame().set_alpha(0.5)
        leg1.get_frame().set_alpha(0.5)
        ax2.set_ylabel('error')
        ax1.grid(True)

        plot.save_fig()

        # ---------------------------------------------------------
        # plotting the calibrated signal, no errors for thesis
        # ---------------------------------------------------------
        figfile = 'yaw_calibration_' + run
        plot = plotting.A4Tuned(scale=1.5)
        figx = plotting.TexTemplate.pagewidth*0.5
        figy = plotting.TexTemplate.pagewidth*0.5
        plot.setup(figpath+figfile, nr_plots=1, grandtitle=None,
                   figsize_x=figx, figsize_y=figy, wsleft_cm=1.4,
                   wsright_cm=0.3, wstop_cm=0.6, wsbottom_cm=1.0)
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 1)
        # note that we haven't read the A ext position!
        ax1.plot(psi_A, stair_A, 'rs', label='A', alpha=0.5)
        ax1.plot(psi_B, stair_B, 'bo', label='B', alpha=0.5)
#        ax1.plot(psi_hd, stair_hd_AB[:,1], 'm', label='A_hd')
#        ax1.plot(psi_hd, stair_hd_AB[:,0], 'k', label='B_hd')
#        ax1.plot(psi_hd, stair_hd, 'k', label='interpolated')
        ax1.plot(psi_poly10, stair_hd, 'k', label='polyfit')
        leg1 = ax1.legend(loc='best')
        ax1.set_xlabel('Yaw angle $\psi$ [deg]')
        ax1.set_ylabel('Laser output signal [V]')
        ax1.set_title(title)

        ax1.grid(True)
        plot.save_fig()


        # ---------------------------------------------------------
        # Save the calibration data
        # ---------------------------------------------------------
        savearray = np.ndarray((len(psi_hd),5))
        savearray[:,0] = psi_hd
        savearray[:,1] = psi_poly10
        savearray[:,2] = stair_hd
        savearray[:,3] = stair_hd_AB[:,0] # B
        savearray[:,4] = stair_hd_AB[:,1] # A
        filename = run + '.yawcal-psihd-psipoly10-stairhd-stairhdAB'
        np.savetxt(pprpath + filename, savearray)

        filename = run + '.yawcal-pol10'
        np.savetxt(pprpath + filename, pol10)

    def runs_289_295(self, respath):
        """
        Create the calibration data set for the April session. Use now the
        more robust method of StairCase instead of the first iteration
        as used for the February data sets.

        In April, the fast side are the positive yaw angles, corresponding to
        a positive angle around the tower Z-axis.
        """

        pprpath = self.pprpath
        self.respath = respath
        figpath = self.figpath

        A_range = range(70,184,5)
        A_range[0] = 71
        B_range = range(85,231,5)
        B_range = range(90,231,5)
        psi_A, psi_B = self.psi_lookup_table(plot=False, A_range=A_range,
                                             B_range=B_range)

        # ---------------------------------------------------------
        run_A = '0410_run_289_yawcalibration_a.mat'
        # note that we will now skip the A ext position!
        # note that dt_noise_treshold is heavily tweaked!
        time_A, stair_A = self.setup_filter(respath, run_A, start=18000,
                            end=-3500, figpath=figpath, dt_treshold=2e-6)
        print psi_A.shape, stair_A.shape
#        plt.plot(psi_A[:,1], stair_A)

        # ---------------------------------------------------------

        resfile = '0410_run_295_yawcalibration_b_extended'
        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
        ch = dm.labels_ch['Yaw Laser']

        sc = StairCase(plt_progress=False, pprpath=figpath, runid=resfile)
        sc.figfile = resfile+'_ch'+str(ch)
        sc.figpath = figpath
        time_B, stair_B = sc.setup_filter(dm.time, dm.data[:,ch],
                                  dt_treshold=8e-7, cutoff_hz=False, dt=1,
                                  start=8500, end=-6050, stair_step_tresh=0.03,
                                  smoothen='moving', smooth_window=0.5)

        print psi_B.shape, stair_B.shape


        # ---------------------------------------------------------
        # Save calibration data
        # ---------------------------------------------------------

        # put psi_A and psi_B in increasing order.
        # B starts at -extreme, A ends at +extreme
        psi_A = psi_A[::-1,1]
        psi_B = -psi_B[::-1,1]
        stair_A = stair_A[::-1]
        stair_B = stair_B[::-1]

        savearray = np.ndarray((len(psi_A),2))
        savearray[:,0] = psi_A
        savearray[:,1] = stair_A
        filename = 'runs_289_295.yawcal-psiA-stairA'
        np.savetxt(pprpath + filename, savearray)

        savearray = np.ndarray((len(psi_B),2))
        savearray[:,0] = psi_B
        savearray[:,1] = stair_B
        filename = 'runs_289_295.yawcal-psiB-stairB'
        np.savetxt(pprpath + filename, savearray)

    def runs_050_051(self, respath):
        """
        Create a calirbation dataset: identify all the stair cases
        """
        pprpath = self.pprpath
        self.respath = respath
        figpath = self.figpath

        psi_A, psi_B = self.psi_lookup_table(plot=False)

        # ---------------------------------------------------------
#        run = '0211_run_045_yawlasercallibration_6.4_17.0.mat'
#        run = '0211_run_046_yawlasercallibration_13.0_23.5_b.mat'

        # ---------------------------------------------------------
#        run = '0211_run_048_yawlasercallibration_6.4_19.0_a_better.mat'
        run_A = '0211_run_051_yawlasercallibration_6.4_19.0_a_better2.mat'
        # note that we will now skip the A ext position!
        # note that dt_noise_treshold is heavily tweaked!
        time_A, stair_A = self.setup_filter(respath, run_A, start=32500,
                            end=-28001, figpath=figpath, dt_treshold=2e-6)
        print psi_A.shape, stair_A.shape
#        dt = data_trim[1:] - data_trim[:-1]
#        dt.sort()
#        print dt[:30]
#        plt.plot(psi_A[:,1], stair_A)

        # ---------------------------------------------------------
#        run = '0211_run_049_yawlasercallibration_11.0_23.5_b.mat'
        run_B = '0211_run_050_yawlasercallibration_11.0_23.5_b_better.mat'
        time_B, stair_B = self.setup_filter(respath, run_B, start=32100,
                            end=-30001, figpath=figpath, dt_treshold=2e-6)
        print psi_B.shape, stair_B.shape
#        dt = data_trim[1:] - data_trim[:-1]
#        dt.sort()
#        print dt[:30]
#        plt.plot(-psi_B[:,1], stair_B)

        # ---------------------------------------------------------
        # Save calibration data
        # ---------------------------------------------------------

        # ignore the first entry from A
        psi_A = psi_A[1:,:]

        # put psi_A and psi_B in increasing order.
        # B starts at -extreme, A ends at +extreme
        psi_A = psi_A[::-1,1]
        psi_B = -psi_B[::-1,1]
        stair_A = stair_A[::-1]
        stair_B = stair_B[::-1]

        savearray = np.ndarray((len(psi_A),2))
        savearray[:,0] = psi_A
        savearray[:,1] = stair_A
        filename = 'runs_050_051.yawcal-psiA-stairA'
        np.savetxt(pprpath + filename, savearray)

        savearray = np.ndarray((len(psi_A),2))
        savearray[:,0] = psi_B
        savearray[:,1] = stair_B
        filename = 'runs_050_051.yawcal-psiB-stairB'
        np.savetxt(pprpath + filename, savearray)

    def plotall_feb_raw(self, respath):
        """
        Print all the calibration data raw data from the February series
        """

#        runs = []
#        # files where the cable was poluting the measurements
#        runs.append('0211_run_045_yawlasercallibration_6.4_17.0.mat')
#        runs.append('0211_run_046_yawlasercallibration_13.0_23.5_b.mat')
#        # clean measurements, cable fixed now
#        runs.append('0211_run_047_yawlasercallibration_6.4_19.0_a.mat')
#        runs.append('0211_run_048_yawlasercallibration_6.4_19.0_a_better.mat')
#        runs.append('0211_run_049_yawlasercallibration_11.0_23.5_b.mat')
#        runs.append('0211_run_050_yawlasercallibration_11.0_23.5_b_better.mat')
#        runs.append('0211_run_051_yawlasercallibration_6.4_19.0_a_better2.mat')
#
#        for run in runs:
#            # load the dspace mat file
#            dspace = ojfresult.DspaceMatFile(respath + run)
#            yawchan = 6
#            # plot the yaw signal
#            figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
#            figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
#            plot = plotting.A4Tuned()
#            plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
#                         dspace.labels, channels=[yawchan], grandtitle=figfile,
#                         figsize_y=10)#, ylim=[3, 4])

        # --------------------------------------------------------------------
        # files where the cable was poluting the measurements
        # --------------------------------------------------------------------
        run = '0211_run_045_yawlasercallibration_6.4_17.0.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[2.5, 4])

        run = '0211_run_046_yawlasercallibration_13.0_23.5_b.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[0.8, 3.5])

        # --------------------------------------------------------------------
        # clean measurements, cable fixed now
        # --------------------------------------------------------------------
        run = '0211_run_047_yawlasercallibration_6.4_19.0_a.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[2.5, 4])

        run = '0211_run_048_yawlasercallibration_6.4_19.0_a_better.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[2.8, 4])

        run = '0211_run_049_yawlasercallibration_11.0_23.5_b.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[0.8, 3.5])

        run = '0211_run_050_yawlasercallibration_11.0_23.5_b_better.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[0.8, 3.5])

        run = '0211_run_051_yawlasercallibration_6.4_19.0_a_better2.mat'
        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        yawchan = 6
        # plot the yaw signal
        figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(yawchan)
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, dspace.time, dspace.data,
                         dspace.labels, channels=[yawchan], grandtitle=figfile,
                         figsize_y=10, ylim=[2.5, 4])

    def plotall_apr_raw(self):
        """
        Simply plot all raw data from the april calibration runs.
        Nothing more, nothing less.
        """

        respath = 'data/raw/04/2012.04.10/0410_data/'
        figpath = CALPATH
        figpath += 'YawLaserCalibration-04/'

        # -------------------------------------------------------------------
#        resfile = '0410_run_289_yawcalibration_a'
#        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
#        dm.plot_channel(channel=dm.labels_ch['Yaw Laser'], figpath=figpath)
#        # filter design to see which data set is best
#        A_range = range(70,184,5)
#        A_range[0] = 71
#        B_range = range(120,236,5)
#        psi_A, psi_B = self.psi_lookup_table(plot=False, A_range=A_range,
#                                             B_range=B_range)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=18000,
#                    end=-3500, figpath=figpath, dt_treshold=2e-6)
#
#        # -------------------------------------------------------------------
#        resfile = '0410_run_290_yawcalibration_a'
#        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
#        dm.plot_channel(channel=dm.labels_ch['Yaw Laser'], figpath=figpath)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=19000,
#                    end=-11500, figpath=figpath, dt_treshold=2e-6)
#
#        # -------------------------------------------------------------------
#        resfile = '0410_run_291_yawcalibration_a'
#        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
#        dm.plot_channel(channel=dm.labels_ch['Yaw Laser'], figpath=figpath)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=19000,
#                    end=-12500, figpath=figpath, dt_treshold=2e-6)
#
#        # -------------------------------------------------------------------
#        resfile = '0410_run_292_yawcalibration_b'
#        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
#        dm.plot_channel(channel=dm.labels_ch['Yaw Laser'], figpath=figpath)
#        B_range = range(120,236,5)
#        B_range[-1] = 231
#        psi_A, psi_B = self.psi_lookup_table(plot=False, A_range=A_range,
#                                             B_range=B_range)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=21000,
#                    end=-13000, figpath=figpath, dt_treshold=2e-6)
#
#        # -------------------------------------------------------------------
#        resfile = '0410_run_293_yawcalibration_b'
#        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
#        dm.plot_channel(channel=dm.labels_ch['Yaw Laser'], figpath=figpath)
#        B_range = range(120,236,5)
#        B_range[-1] = 231
#        psi_A, psi_B = self.psi_lookup_table(plot=False, A_range=A_range,
#                                             B_range=B_range)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=19000,
#                    end=-32000, figpath=figpath, dt_treshold=2e-6)
#
#        # -------------------------------------------------------------------
#        resfile = '0410_run_294_yawcalibration_b_extended'
#        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
#        dm.plot_channel(channel=dm.labels_ch['Yaw Laser'], figpath=figpath)
#        B_range = range(85,231,5)
#        B_range[-1] = 231
#        psi_A, psi_B = self.psi_lookup_table(plot=False, A_range=A_range,
#                                             B_range=B_range)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=4000,
#                    end=-22000, figpath=figpath, dt_treshold=2e-6)

        # -------------------------------------------------------------------
        resfile = '0410_run_295_yawcalibration_b_extended'
        dm = ojfresult.DspaceMatFile(matfile=respath+resfile+'.mat' )
        ch = dm.labels_ch['Yaw Laser']
#        dm.plot_channel(channel=ch, figpath=figpath)
#        time_A, stair_A = self.setup_filter(respath, resfile, start=8000,
#                    end=-6050, figpath=figpath, dt_treshold=3.5e-6)

    def _solve_A(self, A, **kwargs):
        """
        d, L are given in mm

        """

        d = kwargs.get('d', 40.)
        L = kwargs.get('L', 150.)
        acc_check = kwargs.get('acc_check', 0.0000001)
        solve_acc = kwargs.get('solve_acc', 20)

        # set the accuracy target of the solver
        sympy.mpmath.mp.dps = solve_acc
        psi = sympy.Symbol('psi')
        f1 = L - (L*sympy.tan(psi)) + (d/(2.*sympy.cos(psi))) - A
        # initial guess: solve system for delta_x = 0
        psi0 = math.atan(1 - (A/L))
        # solve the equation numerically with sympy
        psi_sol = sympy.nsolve(f1, psi, psi0)

        # verify if the solution is valid
        delta_x = d / (2.*math.cos(psi_sol))
        x = L*math.tan(psi_sol)
        Asym = sympy.Symbol('Asym')
        f_check = x - L + Asym - delta_x
        # verify that f_check == 0
        if not sympy.solvers.checksol(f_check, Asym, A):
            # in the event that it does not pass the checksol, see how close
            # the are manually. Seems they are rather close
            check_A = L + delta_x - x
            error = abs(A - check_A) / A
            if error > acc_check:
                msg = 'sympy\'s solution does not passes checksol()'
                msg += '\n A_check=%.12f <=> A=%.12f' % (check_A, A)
                raise ValueError, msg
            else:
                msg = 'sympy.solvers.checksol() failed, manual check is ok. '
                msg += 'A=%.2f, rel error=%2.3e' % (A, error)
                logging.warning(msg)

        return psi_sol*180./math.pi, psi0*180./math.pi

    def _solve_B(self, B, **kwargs):
        """
        """

        d = kwargs.get('d', 40.)
        L = kwargs.get('L', 150.)
        acc_check = kwargs.get('acc_check', 0.0000001)
        solve_acc = kwargs.get('solve_acc', 20)

        # set the accuracy target of the solver
        sympy.mpmath.mp.dps = solve_acc
        psi = sympy.Symbol('psi')
        f1 = L + (L*sympy.tan(psi)) - (d/(2.*sympy.cos(psi))) - B
        # initial guess: solve system for delta_x = 0
        psi0 = math.atan(1 - (B/L))
        # solve the equation numerically with sympy
        psi_sol = sympy.nsolve(f1, psi, psi0)

        # verify if the solution is valid
        delta_x = d / (2.*math.cos(psi_sol))
        x = L*math.tan(psi_sol)
        Bsym = sympy.Symbol('Bsym')
        f_check = x + L - Bsym - delta_x
        # verify that f_check == 0
        if not sympy.solvers.checksol(f_check, Bsym, B):
            # in the event that it does not pass the checksol, see how close
            # the are manually. Seems they are rather close
            check_B = L - delta_x + x
            error = abs(B-check_B)/B
            if error > acc_check:
                msg = 'sympy\'s solution does not passes checksol()'
                msg += '\n B_check=%.12f <=> B=%.12f' % (check_B, B)
                raise ValueError, msg
            else:
                msg = 'sympy.solvers.checksol() failed, manual check is ok. '
                msg += 'B=%.2f, rel error=%2.3e' % (B, error)
                logging.warning(msg)

        return psi_sol*180./math.pi, psi0*180./math.pi

    def psi_lookup_table(self, **kwargs):
        """
        Create the lookup table which relates A and B to the corresponding
        yaw angle psi. Default values for A and B_range are valid for the
        February runs.

        psi(n,2) = [A distance, psi]
        """

        plot = kwargs.get('plot', False)
        verbose = kwargs.get('verbose', False)

        # use A_range_default for February data
        A_range_default = range(60,190,5)
        A_range_default[0] = 64
        A_range_default.append(190)
        A_range = kwargs.get('A_range', A_range_default)

        A_psi = np.ndarray((len(A_range),2))
        A_psi[:,0] = A_range

        n = 0
        for A in A_range:
            A_psi[n,1], psi0 = self._solve_A(A)
            n += 1

        if verbose:
            print A_psi

        # use B_range_default for February data
        B_range_default = range(110,236,5)
        B_range = kwargs.get('B_range', B_range_default)

        B_psi = np.ndarray((len(B_range),2))
        B_psi[:,0] = B_range

        n = 0
        for B in B_range:
            B_psi[n,1], psi0 = self._solve_B(B)
            n += 1

        if verbose:
            print B_psi

        if plot:
            plt.figure()
            plt.plot(A_psi[:,0], A_psi[:,1], 'b', label='A')
            plt.plot(B_psi[:,0], -B_psi[:,1], 'r', label='B')
            plt.legend()
            plt.xlabel('A, B [mm]')
            plt.ylabel('$\psi$ [deg]')
            fig_path = CALPATH
            plt.savefig(fig_path+'yawcal.png', dpi=200)
            plt.savefig(fig_path+'yawcal.eps', dpi=200)
            plt.show()

        return A_psi, B_psi

    def _read_staircase(self, time, data, data_dt):
        """
        For a given staircase data series, substrackt the relevant data,
        i.e. those points whose derivatives are close to zero
        """

    def setup_filter(self, respath, run, **kwargs):
        """
        Load the callibration runs and convert voltage signal to yaw angles
        """

        # specify the window of the staircase
        #start, end = 30100, -30001
        start = kwargs.get('start', None)
        end = kwargs.get('end', None)
        figpath = kwargs.get('figpath', None)
#        figfile = kwargs.get('figfile', None)
        dt_treshold = kwargs.get('dt_treshold', None)
#        plot_data = kwargs.get('plot_data', False)
#        respath = kwargs.get('respath', None)
#        run = kwargs.get('run', None)

        # load the dspace mat file
        dspace = ojfresult.DspaceMatFile(respath + run)
        # the yaw channel
        ch = 6
        # or a more robust way of determining the channel number
        ch = dspace.labels_ch['Yaw Laser']

        # sample rate of the signal
        sample_rate = calc_sample_rate(dspace.time)

        # file name based on the run file
        figfile = dspace.matfile.split('/')[-1] + '_ch' + str(ch)

        # prepare the data
        time = dspace.time[start:end]
        # the actual yaw signal
        data = dspace.data[start:end,ch]

        # -------------------------------------------------
        # smoothen the signal with some splines
        # -------------------------------------------------
        # NOTE: the smoothing will make the transitions also smoother. This
        # is not good. The edges of the stair need to be steep!
#        smoothen = UnivariateSpline(dspace.time, dspace.data[:,ch], s=2)
#        data_s_full = smoothen(dspace.time)
#        # first the derivatices
#        data_s_dt = data_s_full[start+1:end+1]-data_s_full[start:end]
#        # than cut it off
#        data_s = data_s_full[start:end]

        # -------------------------------------------------
        # local derivatives of the yaw signal and filtering
        # -------------------------------------------------
        data_dt = dspace.data[start+1:end+1,ch]-dspace.data[start:end,ch]
        # filter the local derivatives
        filt = Filters()
        data_filt, N, delay = filt.fir(time, data, ripple_db=20,
                        freq_trans_width=0.5, cutoff_hz=0.3, plot=False,
                        figpath=figpath, figfile=figfile + 'filter_design',
                        sample_rate=sample_rate)

        data_filt_dt = np.ndarray(data_filt.shape)
        data_filt_dt[1:] = data_filt[1:] - data_filt[0:-1]
        data_filt_dt[0] = np.nan

        # -------------------------------------------------
        # smoothen the signal with some splines
        # -------------------------------------------------
#        smoothen = UnivariateSpline(time, data_filt, s=2)
#        data_s = smoothen(time)
#        # first the derivatices
#        data_s_dt = np.ndarray(data_s.shape)
#        data_s_dt[1:] = data_s[1:]-data_s[:-1]
#        data_s_dt[0] = np.nan

        # -------------------------------------------------
        # filter values above certain treshold
        # ------------------------------------------------
        # only keep values which are steady, meaning dt signal is low!

        # based upon the filtering, only select data points for which the
        # filtered derivative is between a certain treshold
        staircase_i = np.abs(data_filt_dt).__ge__(dt_treshold)
        # make a copy of the original signal and fill in Nans on the selected
        # values
        data_reduced = data.copy()
        data_reduced[staircase_i] = np.nan
        data_reduced_dt = np.ndarray(data_reduced.shape)
        data_reduced_dt[1:] = np.abs(data_reduced[1:] - data_reduced[:-1])
        data_reduced_dt[0] = np.nan

        nonnan_i = np.isnan(data_reduced_dt).__invert__()
        dt_noise_treshold = data_reduced_dt[nonnan_i].max()
        print ' dt_noise_treshold ', dt_noise_treshold

        # remove all the nan values
        data_trim = data_reduced[np.isnan(data_reduced).__invert__()]
        time_trim = time[np.isnan(data_reduced).__invert__()]
#        # figure out which dt's are above the treshold
#        data_trim2 = data_trim.copy()
#        data_trim2.sort()
#        data_trim2.
#        # where the dt of the reduced format is above the noise treshold,
#        # we have a stair
#        data_trim_dt = np.abs(data_trim[1:] - data_trim[:-1])
#        argstairs = data_trim_dt.__gt__(dt_noise_treshold)
#        data_trim2 = data_trim_dt.copy()
#        data_trim_dt.sort()
#        data_trim_dt.__gt__(dt_noise_treshold)

        # -------------------------------------------------
        # read the average value over each stair (time and data)
        # ------------------------------------------------
        data_ordered, time_stair, data_stair = self.order_staircase(time_trim,
                                        data_trim, dt_noise_treshold*4.)

        # -------------------------------------------------
        # setup plot
        # -------------------------------------------------
        labels = np.ndarray(3, dtype='<U100')
        labels[0] = dspace.labels[ch]
        labels[1] = 'yawchan derivative'
        labels[2] = 'psd'

        plot = plotting.A4Tuned()
        title = figfile.replace('_', ' ')
        plot.setup(figpath+figfile+'_filter', nr_plots=2, grandtitle=title,
                   figsize_y=20, wsleft_cm=2., wsright_cm=2.5)

        # -------------------------------------------------
        # plotting of signal
        # -------------------------------------------------
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 1)
        ax1.plot(time, data, label='data')
        # add the results of the filtering technique
        time_stair, data_stair
        ax1.plot(time[N-1:], data_reduced[N-1:], 'r', label='data red')
#        ax1.plot(time[N-1:], data_filt[N-1:], 'g', label='data_filt')
        # also include the selected chair data
        label = '%i stairs' % data_stair.shape[0]
        ax1.plot(time_stair, data_stair, 'ko', label=label, alpha=0.2)
        ax1.grid(True)
        ax1.legend(loc='lower left')
        # -------------------------------------------------
        # plotting derivatives on right axis
        # -------------------------------------------------
        ax1b = ax1.twinx()
#        ax1b.plot(time[N:]-delay,data_s_dt[N:],alpha=0.2,label='data_s_dt')
        ax1b.plot(time[N:], data_filt_dt[N:], 'r', alpha=0.2,
                  label='data filt dt')
#        ax1b.plot(time[N:], data_reduced_dt[N:], 'b', alpha=0.2,
#                  label='data_reduced_dt')
#        ax1b.plot(time[N-1:]-delay, filtered_x_dt[N-1:], alpha=0.2)
        ax1b.legend()
        ax1b.grid(True)

        # -------------------------------------------------
        # the power spectral density
        # -------------------------------------------------
        ax3 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, 2)
        Pxx, freqs = ax3.psd(data, Fs=sample_rate, label='data')
        Pxx, freqs = ax3.psd(data_dt, Fs=sample_rate, label='data dt')
#        Pxx, freqs = ax3.psd(data_s_dt, Fs=sample_rate, label='data_s_dt')
        Pxx, freqs = ax3.psd(data_filt_dt[N-1:], Fs=sample_rate,
                             label='data filt dt')
        ax3.legend()
#        print Pxx.shape, freqs.shape
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

        return time_stair, data_stair

    def order_staircase(self, time_trim, data_trim, delta_step_tresh):
        """
        Look for a staircase patern in the data and group per stair
        """
        # -------------------------------------------------
        # getting the staircase data out step by step
        # -------------------------------------------------

        # cycle trhough all the data stairs and get the averages
        start, end = 0, 0
        i, imax, j = 1, 0, 0
        data_ordered = np.ndarray((len(data_trim)/2, 100))
        data_ordered[:,:] = np.nan
        # put the first point already in
        data_ordered[0,0] = data_trim[0]

        time_ordered = np.ndarray(data_ordered.shape)
        time_ordered[:,:] = np.nan
        # put the first point already in
        time_ordered[0,0] = time_trim[0]

        for kk in xrange(1,len(data_trim)):
            k = data_trim[kk]
            # keep track of the different mean levels, aka stairs
            # is the current number in the same mean category?
            nonnan_i = np.isnan(data_ordered[:,j]).__invert__()
#            print 'delta: %2.2e' % abs(data_ordered[nonnan_i,j].mean() - k)/k
#            if abs(data_ordered[nonnan_i,j].mean() - k) < delta_step_tresh:
            if abs(data_ordered[i-1,j] - k) < delta_step_tresh:
                data_ordered[i,j] = k
                time_ordered[i,j] = time_trim[kk]
                i += 1
            # else we have a new stair
            else:
                print 'd: %2.2e' % abs(data_ordered[nonnan_i,j].mean() - k),
                print 'd: %2.2e' % abs(data_ordered[i-1,j] - k),
                print j, data_ordered[nonnan_i,j].mean()
                j += 1
                if i > imax:
                    imax = i
                    # first value of the new stair
                data_ordered[0,j] = k
                time_ordered[0,j] = time_trim[kk]
                i = 1

        # data_ordered array was made too large, cut off empty spaces
        data_ordered = data_ordered[:imax+1,:j+1]
        time_ordered = time_ordered[:imax+1,:j+1]
        # select only the values, ignore nans
        nonnan_i = np.isnan(data_ordered).__invert__()
        data_stair = np.ndarray((j+1))
        time_stair = np.ndarray((j+1))
        # and save for each found stair the everage in a new array seperately
        for k in xrange(j+1):
            data_stair[k] = data_ordered[nonnan_i[:,k],k].mean()
            time_stair[k] = time_ordered[nonnan_i[:,k],k].mean()

        return data_ordered, time_stair, data_stair


def feb_yawlaser_calibration():
    # FEBRUARY
    ycal = YawCalibration()
    ycal.figpath = os.path.join(CALPATH, 'YawLaserCalibration/')
    ycal.pprpath = ycal.figpath
    ycal.respath = RESDATA_02
#    ycal.plotall_feb_raw()
    ycal.runs_050_051(ycal.respath)
    ycal.load_cal_dataset('runs_050_051', 'February')


def apr_yawlaser_calibration():
    # APRIL
    ycal = YawCalibration()
    ycal.figpath = os.path.join(CALPATH, 'YawLaserCalibration-04/')
    ycal.pprpath = ycal.figpath
    ycal.respath = RESDATA_04
#    ycal.plotall_apr_raw()
    ycal.runs_289_295(ycal.respath)
    ycal.load_cal_dataset('runs_289_295', 'April', psi_step_deg=0.01)


def all_yawlaser_calibrations():
    """
    Re-run and print complete yaw calibration cycle, including thesis plots.
    """

    feb_yawlaser_calibration()
    apr_yawlaser_calibration()


if __name__ == '__main__':
    dummy=None
