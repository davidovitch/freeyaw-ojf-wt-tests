# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:19:58 2013

@author: dave
"""

import math

import numpy as np
import scipy as sp
import pylab as plt
import scipy.constants as spc

import plotting
import ojfresult
from staircase import StairCase

class BladeCalibration:
    """
    For the stiff blade
    Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
    Blade 2: channels 1 (M1 root) and 2 (M2 mid section)

    For the stiff blade:
    strain gauges M1, root: channels 1 and 3
    strain gauges M2, mids: channels 2 and 4

    500 micro strain = delta 200 on binary output scale
    """

    def __init__(self):
        """
        """

    def february_loads(self):
        """
        Blade calibration cases for february, including what was actually
        done during the measurement series: which moment was applied when.

        Loads here can contain both flex and stiff cases
        """
        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath

        # the load sequence corresponding to the selected stairs
        # maintain same convention as used in beam_only csv files
        # 0      1    2     3
        # w45, w34, w15, wtip

        # 0213_run_096: only loaded at the tip
        loads = sp.zeros((6,4))
        loads[0,3] = 0
        loads[1,3] = 0.1
        loads[2,3] = 0.2
        loads[3,3] = 0.3
        loads[4,3] = 0.2
        loads[5,3] = 0.1
        try:
            self.data_cal_runs['0213_run_096_ch3'] = loads.copy()
            self.data_cal_runs['0213_run_096_ch4'] = loads.copy()
        except AttributeError:
            self.data_cal_runs = dict()

        # 0213_run_097: only loaded at the tip
        loads = sp.zeros((6,4))
        loads[0,3] = 0
        loads[1,3] = 0.1
        loads[2,3] = 0.2
        loads[3,3] = 0.3
        loads[4,3] = 0.4
        loads[5,3] = 0.0
        self.data_cal_runs['0213_run_097_ch3'] = loads.copy()
        self.data_cal_runs['0213_run_097_ch4'] = loads.copy()

        # 0213_run_098: only loaded at the tip, CONE ANGLE
        loads = sp.zeros((6,4))
        loads[0,3] = 0
        loads[1,3] = 0.1
        loads[2,3] = 0.2
        loads[3,3] = 0.3
        loads[4,3] = 0.4
        loads[5,3] = 0.0
        self.data_cal_runs['0213_run_098_ch1'] = loads.copy()
        self.data_cal_runs['0213_run_098_ch2'] = loads.copy()

        # run 0213_run_099 is about the eigenfrequencies
#        self.data_cal_runs['0213_run_099'] = loads

        # 0213_run_100: ignore blade 1
        # blade 1: ch3, ch4
        # 0      1    2     3
        # w45, w34, w15, wtip
        loads = sp.zeros((6,4))
        loads[1,3] = 0.1
        loads[3,3] = 0.2
        loads[4,2] = 0.2
        self.data_cal_runs['0213_run_100_ch1'] = loads.copy()
        self.data_cal_runs['0213_run_100_ch2'] = loads.copy()

        # 0213_run_101
        # TODO: or is it at the tip instead of w15??
        loads = sp.zeros((6,4))
        loads[1,0] = 0.2
        loads[3,1] = 0.5
        loads[4,2] = 0.1
        self.data_cal_runs['0213_run_101_ch1'] = loads.copy()
        loads = sp.zeros((5,4))
        loads[1,1] = 0.5
        loads[3,2] = 0.1
        self.data_cal_runs['0213_run_101_ch2'] = loads.copy()

        # upside down seems to have comparable rico, but they have a slightly
        # different zero crossing point so that will make the fit go bad
#        loads = sp.zeros((7,4))
#        loads[1,0] = -0.2
#        loads[3,1] = -0.5
#        loads[5,2] = -0.1
#        self.data_cal_runs['0213_run_101_ch3'] = loads.copy()
#        loads = sp.zeros((5,4))
#        loads[1,1] = -0.5
#        loads[3,2] = -0.1
#        self.data_cal_runs['0213_run_101_ch4'] = loads.copy()

        # --------------------------------------------------------------------
        # flexible blade calibrations, only tip loading

        # 0214_run_172
        loads = sp.zeros((7,4))
        loads[1,3] = 0.2
        loads[3,3] = 0.3
        loads[5,3] = 0.2
        self.data_cal_runs['0214_run_172_ch1'] = loads.copy()
        self.data_cal_runs['0214_run_172_ch2'] = loads.copy()
        loads = sp.zeros((5,4))
        loads[1,3] = 0.2
        loads[3,3] = 0.3
        self.data_cal_runs['0214_run_172_ch3'] = loads.copy()
        self.data_cal_runs['0214_run_172_ch4'] = loads.copy()

        # 0214_run_173
        loads = sp.zeros((4,4))
        loads[1,3] = 0.1
        loads[2,3] = 0.3
        self.data_cal_runs['0214_run_173_ch1'] = loads.copy()
        self.data_cal_runs['0214_run_173_ch2'] = loads.copy()
        loads = sp.zeros((5,4))
        loads[1,3] = 0.1
        loads[2,3] = 0.3
        loads[3,3] = 0.1
        self.data_cal_runs['0214_run_173_ch3'] = loads.copy()
        self.data_cal_runs['0214_run_173_ch4'] = loads.copy()

        # upside down seems to have comparable rico, but they have a slightly
        # different zero crossing point so that will make the fit go bad
#        # 0214_run_174, upside down, so loads negative
#        loads = sp.zeros((3,4))
#        loads[1,3] = -0.1
#        loads[2,3] = -0.2
#        self.data_cal_runs['0214_run_174_ch1'] = loads.copy()
#        self.data_cal_runs['0214_run_174_ch2'] = loads.copy()
#        loads = sp.zeros((4,4))
#        loads[1,3] = -0.1
#        loads[2,3] = -0.2
#        self.data_cal_runs['0214_run_174_ch3'] = loads.copy()
#        self.data_cal_runs['0214_run_174_ch4'] = loads.copy()
#
#        # 0214_run_175, upside down, so loads negative
#        loads = sp.zeros((3,4))
#        loads[1,3] = -0.3
#        self.data_cal_runs['0214_run_175_ch1'] = loads.copy()
#        self.data_cal_runs['0214_run_175_ch2'] = loads.copy()
#        self.data_cal_runs['0214_run_175_ch3'] = loads.copy()
#        self.data_cal_runs['0214_run_175_ch4'] = loads.copy()


    def feb_stiff_raw(self, respath):
        """
        Strain calibration files for the stiff blades
        """
        # -------------------------------------------------------------------
        # 0213_run_096_straincalibration_blade1.csv
        # -------------------------------------------------------------------

        run = '0213_run_096_straincalibration_blade1.csv'
        blade = ojfresult.BladeStrainFile(respath + run, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_096_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00008, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='splines')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_096_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00005, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='splines')


        # -------------------------------------------------------------------
        # 0213_run_097_straincalibration_blade1.csv
        # -------------------------------------------------------------------

        run = '0213_run_097_straincalibration_blade1.csv'
        blade = ojfresult.BladeStrainFile(respath + run, verbose=True, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_097_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_097_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        # -------------------------------------------------------------------
        # 0213_run_098_straincalibration_blade2.csv
        # with coning angle!
        # -------------------------------------------------------------------

        run = '0213_run_098_straincalibration_blade2.csv'
        blade = ojfresult.BladeStrainFile(respath + run, verbose=True, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_098_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving',points_per_stair=60)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_098_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving',points_per_stair=40)

        # -------------------------------------------------------------------
        # 0213_run_099_straincalibration_blade12.csv
        # -------------------------------------------------------------------

        run = '0213_run_099_straincalibration_blade12'
        run += '_tipdeflectionvibrations.csv'

        # -------------------------------------------------------------------
        # 0213_run_100_straincalibration_blade12.csv
        # -------------------------------------------------------------------

        run = '0213_run_100_straincalibration_blade12.csv'
        blade = ojfresult.BladeStrainFile(respath + run, verbose=True, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_100_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_100_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_100_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_100_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        # -------------------------------------------------------------------
        # 0213_run_101_straincalibration_blade12.csv
        # -------------------------------------------------------------------
        run = '0213_run_101_straincalibration_blade12.csv'
        blade = ojfresult.BladeStrainFile(respath + run, verbose=True, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_101_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_101_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_101_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0213_run_101_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00009, cutoff_hz=False, dt=1, start=3000,
                    stair_step_tresh=2., smoothen='moving')

    def feb_stiff(self, plotv=1):
        """
        Create the transfer function for the stiff blade, february stint
        """

        blade_cases = []
        blade_cases.append('0213_run_096_ch3')
        blade_cases.append('0213_run_096_ch4')
        blade_cases.append('0213_run_097_ch3')
        blade_cases.append('0213_run_097_ch4')
        blade_cases.append('0213_run_100_ch3')
        blade_cases.append('0213_run_100_ch4')
        blade_cases.append('0213_run_101_ch1')
        blade_cases.append('0213_run_101_ch2')
        blade_cases.append('0213_run_101_ch3')
        blade_cases.append('0213_run_101_ch4')

        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath
        figfile = 'bladestraincal-february-stiff'
        title = 'stiff blade, February tests'
        if plotv == 1:
            self._print_make_tranf_func(blade_cases, figfile, grandtitle=title)
        else:
            self._tranf_func_plot_compact(blade_cases,figfile,grandtitle=title)


    def feb_flex_raw(self, respath):
        """
        Strain calibration files for the stiff blades
        """
        # -------------------------------------------------------------------
        # 0214_run_172_flexblades_calibrations.csv
        # -------------------------------------------------------------------

        run = '0214_run_172_flexblades_calibrations.csv'
        blade = ojfresult.BladeStrainFile(respath + run, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_172_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00020, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2.0, smoothen='moving', end=58000,
                    points_per_stair=50)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_172_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00095, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2.0, smoothen='moving', end=58000,
                    points_per_stair=70)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_172_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00022, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2.0, smoothen='moving', end=58000,
                    points_per_stair=40)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_172_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00070, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2.0, smoothen='moving', end=58000,
                    points_per_stair=40)

        # -------------------------------------------------------------------
        # 0214_run_173_flexblades_calibrations_2.csv
        # -------------------------------------------------------------------

        run = '0214_run_173_flexblades_calibrations_2.csv'
        blade = ojfresult.BladeStrainFile(respath + run, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_173_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00020, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=38400)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_173_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00200, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=38400)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_173_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00100, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=38400)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_173_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00200, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=38400)


        # -------------------------------------------------------------------
        # 0214_run_174_flexblades_calibrations_3_upsidedown.csv
        # -------------------------------------------------------------------

        run = '0214_run_174_flexblades_calibrations_3_upsidedown.csv'
        blade = ojfresult.BladeStrainFile(respath + run, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_174_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00025, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=30720)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_174_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00050, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=30720)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_174_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00025, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=2., smoothen='moving', end=30720)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_174_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00100, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=10., smoothen='moving', end=30720)

        # -------------------------------------------------------------------
        # 0214_run_175_flexblades_calibrations_4_upsidedown.csv
        # -------------------------------------------------------------------

        run = '0214_run_175_flexblades_calibrations_4_upsidedown.csv'
        blade = ojfresult.BladeStrainFile(respath + run, Fs=512)
        channels = [0,1,2,3]
        # plot the yaw signal
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run
        title = run.replace('_', '').replace('.csv', '')
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=title)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_175_ch1',
                       figpath=figpath, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00018, cutoff_hz=False, dt=1, start=2000,
                    stair_step_tresh=2., smoothen='moving', end=30720)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_175_ch2',
                       figpath=figpath, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00045, cutoff_hz=False, dt=1, start=2000,
                    stair_step_tresh=2., smoothen='moving', end=30720)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_175_ch3',
                       figpath=figpath, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00018, cutoff_hz=False, dt=1, start=2000,
                    stair_step_tresh=2., smoothen='moving', end=51200)

        sc = StairCase(plt_progress=False, pprpath=figpath,
                       runid='0214_run_175_ch4',
                       figpath=figpath, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00035, cutoff_hz=False, dt=1, start=2000,
                    stair_step_tresh=2., smoothen='moving', end=51200)

    def feb_flex(self, plotv=1):
        """
        Create the transfer function for the flex blade, february stint
        """

        blade_cases = []
        blade_cases.append('0214_run_172_ch1')
        blade_cases.append('0214_run_172_ch2')
        blade_cases.append('0214_run_172_ch3')
        blade_cases.append('0214_run_172_ch4')
        blade_cases.append('0214_run_173_ch1')
        blade_cases.append('0214_run_173_ch2')
        blade_cases.append('0214_run_173_ch3')
        blade_cases.append('0214_run_173_ch4')
        # 174 and 175 are upside down...
#        blade_cases.append('0214_run_174_ch1')
#        blade_cases.append('0214_run_174_ch2')
#        blade_cases.append('0214_run_174_ch3')
#        blade_cases.append('0214_run_174_ch4')
#        blade_cases.append('0214_run_175_ch1')
#        blade_cases.append('0214_run_175_ch2')
#        blade_cases.append('0214_run_175_ch3')
#        blade_cases.append('0214_run_175_ch4')

        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath
        figfile = 'bladestraincal-february-flex'
        title = 'flex blade, February tests'
        if plotv == 1:
            self._print_make_tranf_func(blade_cases, figfile, grandtitle=title)
        else:
            self._tranf_func_plot_compact(blade_cases,figfile,grandtitle=title)

    def april_loads(self):
        """
        Make the calibration load cases that are applicable for the flexible
        april calibration run.

        Loads here can contain both flex and stiff cases
        """
        def correct(loads):
            # correction for the non horizontal load application
            costip = 166. / math.sqrt(166*166 + 9*9)
            cos15 = 166. / math.sqrt(166*166 + 6*6)
            cos34 = 166. / math.sqrt(166*166 + 25*25)
            cos45 = 166. / math.sqrt(166*166 + 36*36)
            loads[:,0] *= cos45
            loads[:,1] *= cos34
            loads[:,2] *= cos15
            loads[:,3] *= costip

            return loads

        mass_holder = 0.57166

        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath

        # the load sequence corresponding to the selected stairs
        # maintain same convention as used in beam_only csv files
        # 0      1    2     3
        # w45, w34, w15, wtip

        # FLEX BLADE
        # the order needs to correspond to the results of apr_flex_raw
        loads = sp.zeros((12,4))
        loads[ 0,2] = 0
        loads[ 1,2] = mass_holder
        loads[ 2,2] = mass_holder + 0.1
        loads[ 3,2] = mass_holder + 0.2
        loads[ 4,2] = mass_holder + 0.3
        loads[ 5,1] = mass_holder + 0.1
        loads[ 6,1] = mass_holder + 0.2
        loads[ 7,1] = mass_holder + 0.3
        loads[ 8,0] = mass_holder
        loads[ 9,0] = mass_holder + 0.1
        loads[10,0] = mass_holder + 0.2
        loads[11,0] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        try:
            self.data_cal_runs['0405_run_252_ch3'] = loads.copy()
        except AttributeError:
            self.data_cal_runs = dict()

        # FLEX BLADE
        # no measuring points at 30% when loading is above that point
        loads = sp.zeros((8,4))
        loads[ 0,2] = 0
        loads[ 1,2] = mass_holder
        loads[ 2,2] = mass_holder + 0.1
        loads[ 3,2] = mass_holder + 0.2
        loads[ 4,2] = mass_holder + 0.3
        loads[ 5,1] = mass_holder + 0.1
        loads[ 6,1] = mass_holder + 0.2
        loads[ 7,1] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0405_run_252_ch4'] = loads.copy()

        # FLEX BLADE
        loads = sp.zeros((13,4))
        loads[ 0,2] = 0
        loads[ 1,2] = mass_holder
        loads[ 2,2] = mass_holder + 0.1
        loads[ 3,2] = mass_holder + 0.2
        loads[ 4,2] = mass_holder + 0.3
        loads[ 5,1] = mass_holder
        loads[ 6,1] = mass_holder + 0.1
        loads[ 7,1] = mass_holder + 0.2
        loads[ 8,1] = mass_holder + 0.3
        loads[ 9,0] = mass_holder
        loads[10,0] = mass_holder + 0.1
        loads[11,0] = mass_holder + 0.2
        loads[12,0] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0405_run_254_ch1'] = loads.copy()

        # FLEX BLADE
        loads = sp.zeros((9,4))
        loads[ 0,2] = 0
        loads[ 1,2] = mass_holder
        loads[ 2,2] = mass_holder + 0.1
        loads[ 3,2] = mass_holder + 0.2
        loads[ 4,2] = mass_holder + 0.3
        loads[ 5,1] = mass_holder
        loads[ 6,1] = mass_holder + 0.1
        loads[ 7,1] = mass_holder + 0.2
        loads[ 8,1] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0405_run_254_ch2'] = loads.copy()

        # STIFF BLADE
        loads = sp.zeros((10,4))
        loads[ 0,0] = 0
        loads[ 1,0] = mass_holder
        loads[ 2,0] = mass_holder + 0.1
        loads[ 3,0] = mass_holder + 0.2
        loads[ 4,0] = mass_holder + 0.3
        loads[ 5,0] = mass_holder + 0.5
        loads[ 6,1] = mass_holder
        loads[ 7,1] = mass_holder + 0.1
        loads[ 8,1] = mass_holder + 0.2
        loads[ 9,1] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0412_run_356_ch3'] = loads.copy()

        loads = sp.zeros((5,4))
        loads[ 0,0] = 0
        loads[ 1,1] = mass_holder
        loads[ 2,1] = mass_holder + 0.1
        loads[ 3,1] = mass_holder + 0.2
        loads[ 4,1] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0412_run_356_ch4'] = loads.copy()

        loads = sp.zeros((10,4))
        loads[ 0,0] = 0
        loads[ 1,0] = mass_holder
        loads[ 2,0] = mass_holder + 0.1
        loads[ 3,0] = mass_holder + 0.2
        loads[ 4,0] = mass_holder + 0.3
        loads[ 5,0] = mass_holder + 0.5
        loads[ 6,1] = mass_holder
        loads[ 7,1] = mass_holder + 0.1
        loads[ 8,1] = mass_holder + 0.2
        loads[ 9,1] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0412_run_357_ch1'] = loads.copy()

        loads = sp.zeros((5,4))
        loads[ 0,0] = 0
        loads[ 1,1] = mass_holder
        loads[ 2,1] = mass_holder + 0.1
        loads[ 3,1] = mass_holder + 0.2
        loads[ 4,1] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0412_run_357_ch2'] = loads.copy()

        # the load sequence corresponding to the selected stairs
        # maintain same convention as used in beam_only csv files
        # 0      1    2     3
        # w45, w34, w15, wtip

        loads = sp.zeros((9,4))
        loads[ 0,2] = 0
        loads[ 1,2] = mass_holder
        loads[ 2,2] = mass_holder + 0.1
        loads[ 3,2] = mass_holder + 0.2
        loads[ 4,2] = mass_holder + 0.3
        loads[ 5,3] = mass_holder
        loads[ 6,3] = mass_holder + 0.1
        loads[ 7,3] = mass_holder + 0.2
        loads[ 8,3] = mass_holder + 0.3
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0412_run_358_ch1'] = loads.copy()

        self.data_cal_runs['0412_run_358_ch2'] = loads.copy()

        loads = sp.zeros((10,4))
        loads[ 0,2] = 0
        loads[ 1,2] = mass_holder
        loads[ 2,2] = mass_holder + 0.1
        loads[ 3,2] = mass_holder + 0.2
        loads[ 4,2] = mass_holder + 0.3
        loads[ 5,3] = mass_holder
        loads[ 6,3] = mass_holder + 0.1
        loads[ 7,3] = mass_holder + 0.2
        loads[ 8,3] = mass_holder + 0.3
        loads[ 9,3] = 0
        # correction for the non horizontal load application
        loads = correct(loads)
        self.data_cal_runs['0412_run_358_ch3'] = loads.copy()

        self.data_cal_runs['0412_run_358_ch4'] = loads.copy()

    def apr_print_all_raw(self):
        """
        """
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/'
        figpath += 'OJF_tests/BladeStrainCal/'

        # =====================================================================
        # BLADE 1
        resfile = '0405_run_252_bladecal_blade1'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)

        # same as 252
        #resfile = '0405_run_254or253_bladecal_virbations_blade2orblade1'
        #channels = [0,1,2,3]
        #blade = ojfresult.BladeStrainFile(respath+resfile)
        #blade.plot_channel(figpath=figpath, channel=channels)

        # BLADE 2
        resfile = '0405_run_254or253_bladecal_virbations_blade2orblade1-2'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)

        # =====================================================================
        # for the vibrations
        respath = '/home/dave/PhD_data/OJF_data_edit/04/vibration/'

        resfile = '257or258'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)

        resfile = '257or258-2'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)

        resfile = '0405_run_255or254a_bladecal_virbations_blade2'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)

        resfile = '0405_run_255or254_virbations_bladecal_blade2'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)

        resfile = '0405_run_257_bladecal_virbations_blad1'
        channels = [0,1,2,3]
        blade = ojfresult.BladeStrainFile(respath+resfile)
        blade.plot_channel(figpath=figpath, channel=channels)


    def apr_flex_raw(self):
        """
        Strain calibration files for the flexible blades, tune the filters
        so we have a good staircase data selection
        """
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'

        # =====================================================================
        # 0405_run_252_bladecal_blade1.csv
        # =====================================================================
        # BLADE 1
        run = '0405_run_252_bladecal_blade1.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=figfile)

        # -------------------------------------------------------------------
        runid = '0405_run_252_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00095, cutoff_hz=False, dt=1, start=1000,
                    stair_step_tresh=8., smoothen='moving')
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(2)
        # carefull with the next indices since the reference to them changed
        # due to each proceeding pop
        ii.pop(9-1)
        ii.pop(14-2)
        ii.pop(15-3)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        runid = '0405_run_252_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00245, cutoff_hz=False, dt=1, start=1000,
                    stair_step_tresh=8., smoothen='moving')
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(1)
        # carefull with the next indices since the reference to them changed
        # due to each proceeding pop
        ii.pop(3-1)
        ii.pop(7-2)
        ii.pop(9-3)
        ii.pop(12-4)
        ii.pop(13-5)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)
        # -------------------------------------------------------------------


        # =====================================================================
        # 0405_run_254or253_bladecal_virbations_blade2orblade1-2.csv
        # =====================================================================
        # BLADE 2
        run = '0405_run_254or253_bladecal_virbations_blade2orblade1-2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=figfile)

        # -------------------------------------------------------------------
        runid = '0405_run_254_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00030, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=8., smoothen='moving')
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(9)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        runid = '0405_run_254_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00070, cutoff_hz=False, dt=1, start=5000,
                    stair_step_tresh=8., smoothen='moving')
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(5)
        ii.pop(10-1)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)
        # -------------------------------------------------------------------




    def apr_flex(self, plotv=1):
        """
        Create the transfer function for the stiff blade, february stint
        """

        blade_cases = []
        blade_cases.append('0405_run_252_ch3')
        blade_cases.append('0405_run_252_ch4')
        blade_cases.append('0405_run_254_ch1')
        blade_cases.append('0405_run_254_ch2')

        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath
        figfile = 'bladestraincal-april-flex'
        title = 'flex blade, April tests'
        if plotv == 1:
            self._print_make_tranf_func(blade_cases, figfile, grandtitle=title)
        else:
            self._tranf_func_plot_compact(blade_cases,figfile,grandtitle=title)


    def apr_stiff_raw(self):
        """
        Strain calibration files for the flexible blades, tune the filters
        so we have a good staircase data selection
        """
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'

        # =====================================================================
        # 0412_run_356_b1_strain_calibration_w45_w34.csv
        # =====================================================================
        # BLADE 1
        run = '0412_run_356_b1_strain_calibration_w45_w34.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=figfile)

        # -------------------------------------------------------------------
        runid = '0412_run_356_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00055, cutoff_hz=False, dt=1, start=500,
                    stair_step_tresh=5., smoothen='moving', end=blade.Fs*199.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(1)
        ii.pop(2-1)
        ii.pop(3-2)
        ii.pop(9-3)
        ii.pop(10-4) # I think this was mark to indicate w45 - w35
        # this assumption is made on the fact that we suddenly see a spike in
        # both the root as 30% channels
        ii.pop(11-5)
        ii.pop(12-6)
        ii.pop(17-7)
        ii.pop(18-8) # marking end of measurement
        ii.pop(19-9)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # -------------------------------------------------------------------
        runid = '0412_run_356_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00055, cutoff_hz=False, dt=1, start=500,
                    stair_step_tresh=2., smoothen='moving', end=blade.Fs*199.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(1)
        ii.pop(2-1)
        ii.pop(3-2)
        ii.pop(4-3)
        ii.pop(9-4)
        ii.pop(10-5)
        ii.pop(11-6)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # =====================================================================
        # 0412_run_357_b2_strain_calibration_w45_w34_or1.csv
        # =====================================================================
        # BLADE 2
        run = '0412_run_357_b2_strain_calibration_w45_w34_or1.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=figfile)

        # -------------------------------------------------------------------
        runid = '0412_run_357_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00055, cutoff_hz=False, dt=1, start=500,
                    stair_step_tresh=5., smoothen='moving', end=blade.Fs*159.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(3)
        ii.pop(7-1)
        ii.pop(8-2)
        ii.pop(9-3)
        ii.pop(10-4)
        ii.pop(15-5)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # -------------------------------------------------------------------
        runid = '0412_run_357_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00055, cutoff_hz=False, dt=1, start=500,
                    stair_step_tresh=2., smoothen='moving', end=blade.Fs*159.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(1)
        ii.pop(2-1)
        ii.pop(3-2)
        ii.pop(4-3)
        ii.pop(9-4)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # =====================================================================
        # 0412_run_358_b1b2_strain_calibration_w15_wtip.csv
        # =====================================================================
        # BLADE 2
        run = '0412_run_358_b1b2_strain_calibration_w15_wtip.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCal/'
        figfile = run

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time, blade.data,
                         blade.labels, channels=channels, grandtitle=figfile)

        # -------------------------------------------------------------------
        runid = '0412_run_358_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                dt_treshold=0.00055, cutoff_hz=False, dt=1, start=blade.Fs*100,
                stair_step_tresh=5., smoothen='moving', end=blade.Fs*219.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(0)
        ii.pop(6-1)
        ii.pop(11-2)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # -------------------------------------------------------------------
        runid = '0412_run_358_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                dt_treshold=0.00150, cutoff_hz=False, dt=1, start=blade.Fs*100,
                stair_step_tresh=5., smoothen='moving', end=blade.Fs*219.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(0)
        ii.pop(6-1)
        ii.pop(11-2)
        ii.pop(12-3)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # -------------------------------------------------------------------
        runid = '0412_run_358_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00055, cutoff_hz=False, dt=1, start=500,
                    stair_step_tresh=2., smoothen='moving', end=blade.Fs*139.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(0)
        ii.pop(6-1)
        ii.pop(7-2)
        ii.pop(12-3)
        ii.pop(13-4)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

        # -------------------------------------------------------------------
        runid = '0412_run_358_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00055, cutoff_hz=False, dt=1, start=500,
                    stair_step_tresh=2., smoothen='moving', end=blade.Fs*139.)
        # and manually remove some of the false positives in the stair case
        plt.figure()
        plt.plot(time_stair, data_stair, 'wo') # to check we did it right
        ii = range(len(time_stair))
        ii.pop(0)
        ii.pop(6-1)
        ii.pop(11-2)
        time_stair = time_stair[ii]
        data_stair = data_stair[ii]
        plt.plot(time_stair, data_stair, 'r*') # to check we did it right
        plt.title(runid)
        # and save the results
        filename = runid + '-time_stair'
        np.savetxt(figpath + filename, time_stair)
        filename = runid + '-data_stair'
        np.savetxt(figpath + filename, data_stair)

    def apr_stiff(self, plotv=1):
        """
        """

        blade_cases = []
        # stff cases w45 w34
        blade_cases.append('0412_run_356_ch3')
        blade_cases.append('0412_run_356_ch4')
        blade_cases.append('0412_run_357_ch1')
        blade_cases.append('0412_run_357_ch2')
        # stiff cases w15 tip
        blade_cases.append('0412_run_358_ch1')
        blade_cases.append('0412_run_358_ch2')
        blade_cases.append('0412_run_358_ch3')
        blade_cases.append('0412_run_358_ch4')

        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath
        figfile = 'bladestraincal-april-stiff'
        title = 'stiff blade, April tests'
        if plotv == 1:
            self._print_make_tranf_func(blade_cases, figfile, grandtitle=title)
        else:
            self._tranf_func_plot_compact(blade_cases,figfile,grandtitle=title)


    def compare_cal_feb_april(self):
        """
        Directly compare april and february calibration data
        """

        self.february_loads()
        self.april_loads()

        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'BladeStrainCal/'
        self.pprpath = self.figpath

        # -------------------------------------------------------------------
        # FEBRUARY AND APRIL STIFF CASES
        blade_cases = []
        blade_cases.append('0213_run_096_ch3')
        blade_cases.append('0213_run_096_ch4')
        blade_cases.append('0213_run_097_ch3')
        blade_cases.append('0213_run_097_ch4')
        blade_cases.append('0213_run_100_ch3')
        blade_cases.append('0213_run_100_ch4')
        blade_cases.append('0213_run_101_ch1')
        blade_cases.append('0213_run_101_ch2')
        blade_cases.append('0213_run_101_ch3')
        blade_cases.append('0213_run_101_ch4')

        blade_cases.append('0412_run_356_ch3')
        blade_cases.append('0412_run_356_ch4')
        blade_cases.append('0412_run_357_ch1')
        blade_cases.append('0412_run_357_ch2')
        blade_cases.append('0412_run_358_ch1')
        blade_cases.append('0412_run_358_ch2')
        blade_cases.append('0412_run_358_ch3')
        blade_cases.append('0412_run_358_ch4')

        figfile = 'bladestraincal-feb-april-stiff'
        self._print_make_tranf_func(blade_cases, figfile)

        # -------------------------------------------------------------------
        # FEBRUARY AND APRIL FLEX CASES
        blade_cases = []
        blade_cases.append('0214_run_172_ch1')
        blade_cases.append('0214_run_172_ch2')
        blade_cases.append('0214_run_172_ch3')
        blade_cases.append('0214_run_172_ch4')
        blade_cases.append('0214_run_173_ch1')
        blade_cases.append('0214_run_173_ch2')
        blade_cases.append('0214_run_173_ch3')
        blade_cases.append('0214_run_173_ch4')
        blade_cases.append('0214_run_174_ch1')
        blade_cases.append('0214_run_174_ch2')
        blade_cases.append('0214_run_174_ch3')
        blade_cases.append('0214_run_174_ch4')
        blade_cases.append('0214_run_175_ch1')
        blade_cases.append('0214_run_175_ch2')
        blade_cases.append('0214_run_175_ch3')
        blade_cases.append('0214_run_175_ch4')

        blade_cases.append('0405_run_252_ch3')
        blade_cases.append('0405_run_252_ch4')
        blade_cases.append('0405_run_254_ch1')
        blade_cases.append('0405_run_254_ch2')

        figfile = 'bladestraincal-feb-april-flex'
        self._print_make_tranf_func(blade_cases, figfile)


    def thesis_plot_blade_strain(self):
        """
        The blade strain plots as used for the thesis
        """
        # define the load cases that correspond the staircase measurements
        self.february_loads()
        self.april_loads()

        # 4 plots above each other on half a page width
        self.feb_stiff(plotv=2)
        self.feb_flex(plotv=2)
        self.apr_stiff(plotv=2)
        self.apr_flex(plotv=2)

        # combined plots: root,feb,apr and 30%,feb,apr


    def _tranf_func_plot_compact(self, blade_cases, figfile, grandtitle=False):
        """
        Same as _print_make_tranf_func, but now combining more data in one
        plot, tuned for thesis inclusion.
        """

        order = 1

        if not grandtitle:
            grandtitle = figfile

        # -------------------------------------------------
        # setup plot
        # -------------------------------------------------
        pa4 = plotting.A4Tuned(scale=1.5)
        pw = plotting.TexTemplate.pagewidth
        pa4.setup(self.figpath+figfile+'_compact', grandtitle=None, nr_plots=1,
                         wsleft_cm=1.7, wsright_cm=0.6, hspace_cm=2.0,
                         size_x_perfig=pw, size_y_perfig=4.5, wstop_cm=0.5,
                         wsbottom_cm=1.0)

        # For the stiff blade
        # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
        # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)

        axch1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
#        axch1.set_title('ch1: blade 2 root')
        axch1.set_title(grandtitle)
        axch1.grid(True)
        axch1.set_ylabel('binary output signal')
        axch1.set_xlabel('bending moment at strain gauge [Nm]')

#        axch2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
#        axch2.set_title('ch2: blade 2 30\%')
#        axch2.grid(True)
#        axch2.set_ylabel('binary output signal')
#
#        axch3 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 3)
#        axch3.set_title('ch3: blade 1 root')
#        axch3.grid(True)
#        axch3.set_ylabel('binary output signal')
#
#        axch4 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 4)
#        axch4.set_title('ch4: blade 1 30\%')
#        axch4.grid(True)
#        # only x-label on the bottom axes
#        axch4.set_xlabel('bending moment at strain gauge [Nm]')
#        axch4.set_ylabel('binary output signal')

        # for convience, put them in a list
#        ax_list = [axch1, axch2, axch3, axch4]

        # if applicable, pull different calibration datasets together
        sigout_ch = [np.array([]), np.array([]), np.array([]), np.array([])]
        M_ch = [np.array([]), np.array([]), np.array([]), np.array([])]

        plotnr1, plotnr2, plotnr3, plotnr4 = 0,0,0,0
        colors = ['yo', 'b^', 'rv', 'gs']
        symb = ['-', '--', '-.', ':']
        case_ch1, case_ch2, case_ch3, case_ch4 = '', '', '', ''
        case_ch = ['','','','']
        tag = ['','','','']

        # for the latex table
        txt = ''

        print
        print '='*79
        print 'data and fits for each measurement serie idepentantly'
        print '='*79

        for case in blade_cases:
            print case

            try:
                # corresponding loads
                load = self.data_cal_runs[case]
            except KeyError:
                print '... ignored'
                continue

            # load the signal output
            sigout = np.loadtxt(self.pprpath + case + '-data_stair')

#            # ONLY USE IN COMBINATION WITH compare_cal_feb_april() UGLY
#            # COMPARING TRICK, make 0405_run_252 and 0405_run_254 positive
#            if case.startswith('0405_run_252'):
#                sigout *= -1.0
#            elif case.startswith('0405_run_254'):
#                sigout *= -1.0

            # convert to moments at the two strain gauges
            M1, M2 = self._m_strain_gauges(load)

            if case[-3:] == 'ch1':
                M = M1
                #ax = axch1
                plotnr1 += 1
                plotnr = plotnr1
                chi = 0
                tag[chi] = 'B1 root:'
            elif case[-3:] == 'ch2':
                M = M2
                #ax = axch2
                plotnr2 += 1
                plotnr = plotnr2
                chi = 1
                tag[chi] = 'B1 30\%:'
            elif case[-3:] == 'ch3':
                M = M1
                #ax = axch3
                plotnr3 += 1
                plotnr = plotnr3
                chi = 2
                tag[chi] = 'B2 root:'
            elif case[-3:] == 'ch4':
                M = M2
                #ax = axch4
                plotnr4 += 1
                plotnr = plotnr4
                chi = 3
                tag[chi] = 'B2 30\%:'
            else:
                raise ValueError, 'don\'t know which moment to take, M1 or M2?'

            # put with possible mates together
            sigout_ch[chi] = np.append(sigout_ch[chi], sigout)
            M_ch[chi] = np.append(M_ch[chi], M)
            case_ch[chi] += case + '_'

#            print 'sigout:', sigout.shape, 'M:', M.shape

            # first, check that the stair case and the loading are inline
            print 'M.shape: %s, data_stair.shape: %s' % (M.shape, sigout.shape)

            polx = np.polyfit(sigout, M, order)
            M_polx = np.polyval(polx, sigout)

            print 'polx coeff:',
            for k in polx: print format(k, '1.8f'),

            # save the polynomial fit for each different measurement case
            np.savetxt(self.pprpath+case+'.pol'+ str(order), polx)

            # make sure the data is sorted according to sigout
            sorti = sigout.argsort()
            M = M[sorti]
            M_polx = M_polx[sorti]
            sigout = sigout[sorti]

            # for the plotting, be consistent and make them all positive
            if polx[0] < 0:
                XX = -1.0
            else:
                XX = 1.0

#            plt.subplot(3,1,plotnr)
            c = colors[chi]
            label = case.replace('_', '\_')
            axch1.plot(M, XX*sigout, c)#, label=label)
#            axch1.plot(M_polx, XX*sigout, c, alpha=0.7)

            plotnr +=1
#            leg = ax_list[chi].legend(loc='best')
#            leg.get_frame().set_alpha(0.5)

        print
        print '-'*79
        print 'fits per blade'
        print '-'*79
        print figfile

        txt += figfile + '\n'
        # for each channel make a transfer function considering all the
        # different cases and save it too.
        polx_ch, M_polx_ch = ['','','',''], ['','','','']
        for chi in range(4):

            # make shure all points are in increasing order
            isort = M_ch[chi].argsort()
            M_ch[chi] = M_ch[chi][isort]
            sigout_ch[chi] = sigout_ch[chi][isort]

            # and now for each plot the global fitted data
            polx_ch[chi] = np.polyfit(sigout_ch[chi], M_ch[chi], order)
            M_polx_ch[chi] = np.polyval(polx_ch[chi], sigout_ch[chi])
            # save the polyfit for the combined measurements
            # for the filename, ditch the last _ from combined case name
            np.savetxt(self.pprpath+case_ch[chi][:-1]+'.pol'+ str(order), polx)

            # calcualte the quality of the fit
            # for the definition of coefficient of determination, denoted R2
            # https://en.wikipedia.org/wiki/Coefficient_of_determination
            SS_tot = np.sum(np.power( (M_ch[chi] - M_ch[chi].mean()), 2 ))
            SS_err = np.sum(np.power( (M_ch[chi] - M_polx_ch[chi]), 2 ))
            R2 = 1 - (SS_err/SS_tot)

            print '%10s %10s %10s %10s' % ('', 'a', 'b', 'error')
            replace = (tag[chi], polx_ch[chi][0], polx_ch[chi][1], R2)
            txt += '%10s & %10.7f & %10.7f & %10.4f \\\\ \n' % replace
            print '%10s %10.7f %10.7f %10.7f' % replace


            # and plot the fitted line: the actual transformation function
            # stuff the make the label for the final fitted line on all results
            if polx_ch[chi][0] < 0:
                XX = -1.0
            else:
                XX = 1.0
            aa = XX*polx_ch[chi][0]
            bb = XX*polx_ch[chi][1]
            if bb < 0:
                operator = '-'
            else:
                operator = '+'
            label = '%s $%1.5f x %s %7.4f$' % (tag[chi], aa, operator, abs(bb))
            colsymb = colors[chi] + symb[chi]
            # only take the first and last point of the straight line
            axch1.plot(M_polx_ch[chi][[0,-1]], XX*sigout_ch[chi][[0,-1]],
                       colsymb, label=label)
            leg = axch1.legend(loc='best')
            leg.get_frame().set_alpha(0.9)

#        # set the maxima the same for each plot for more easy comparison
#        if figfile.find('stiff') > 0:
#            ymax = 600.0
#        else:
#            ymax = 1000.0
##        for ax in ax_list:
#        axch1.set_ylim([-100, ymax])
#        axch1.set_xlim([-0.5, 4.5])

        pa4.save_fig()

        print 79*'*'
        print txt
        print 79*'*'


    def _print_make_tranf_func(self, blade_cases, figfile, grandtitle=False):
        """
        Overplot for the given cases the calibration points

        This is the overview version, each channel/blade has its own plot
        """
        order = 1

        if not grandtitle:
            grandtitle = figfile

        # -------------------------------------------------
        # setup plot
        # -------------------------------------------------
        pa4 = plotting.A4Tuned(scale=1.4)
        pw = plotting.TexTemplate.pagewidth
        pa4.setup(self.figpath+figfile, grandtitle=grandtitle, nr_plots=4,
                         wsleft_cm=1.7, wsright_cm=0.6, hspace_cm=2.0,
                         size_x_perfig=pw, size_y_perfig=5.0, wstop_cm=1.5)

        # For the stiff blade
        # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
        # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)

        axch1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        axch1.set_title('ch1: blade 2 root')
        axch1.grid(True)
        axch1.set_ylabel('binary output signal')

        axch2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
        axch2.set_title('ch2: blade 2 30\%')
        axch2.grid(True)
        axch2.set_ylabel('binary output signal')

        axch3 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 3)
        axch3.set_title('ch3: blade 1 root')
        axch3.grid(True)
        axch3.set_ylabel('binary output signal')

        axch4 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 4)
        axch4.set_title('ch4: blade 1 30\%')
        axch4.grid(True)
        # only x-label on the bottom axes
        axch4.set_xlabel('bending moment at strain gauge [Nm]')
        axch4.set_ylabel('binary output signal')

        # for convience, put them in a list
        ax_list = [axch1, axch2, axch3, axch4]

        # if applicable, pull different calibration datasets together
        sigout_ch = [np.array([]), np.array([]), np.array([]), np.array([])]
        M_ch = [np.array([]), np.array([]), np.array([]), np.array([])]

        plotnr1, plotnr2, plotnr3, plotnr4 = 0,0,0,0
        colors = ['r--', 'b--', 'y--', 'k--','y--', 'g--', 'c--']
        case_ch1, case_ch2, case_ch3, case_ch4 = '', '', '', ''
        case_ch = ['','','','']
        for case in blade_cases:
            print case

            try:
                # corresponding loads
                load = self.data_cal_runs[case]
            except KeyError:
                print '... ignored'
                continue

            # load the signal output
            sigout = np.loadtxt(self.pprpath + case + '-data_stair')

#            # ONLY USE IN COMBINATION WITH compare_cal_feb_april() UGLY
#            # COMPARING TRICK, make 0405_run_252 and 0405_run_254 positive
#            if case.startswith('0405_run_252'):
#                sigout *= -1.0
#            elif case.startswith('0405_run_254'):
#                sigout *= -1.0

            # convert to moments at the two strain gauges
            M1, M2 = self._m_strain_gauges(load)

            if case[-3:] == 'ch1':
                M = M1
                #ax = axch1
                plotnr1 += 1
                plotnr = plotnr1
                chi = 0
            elif case[-3:] == 'ch2':
                M = M2
                #ax = axch2
                plotnr2 += 1
                plotnr = plotnr2
                chi = 1
            elif case[-3:] == 'ch3':
                M = M1
                #ax = axch3
                plotnr3 += 1
                plotnr = plotnr3
                chi = 2
            elif case[-3:] == 'ch4':
                M = M2
                #ax = axch4
                plotnr4 += 1
                plotnr = plotnr4
                chi = 3
            else:
                raise ValueError, 'don\'t know which moment to take, M1 or M2?'

            # put with possible mates together
            sigout_ch[chi] = np.append(sigout_ch[chi], sigout)
            M_ch[chi] = np.append(M_ch[chi], M)
            case_ch[chi] += case + '_'

#            print 'sigout:', sigout.shape, 'M:', M.shape

            # first, check that the stair case and the loading are inline
            print 'M.shape: %s, data_stair.shape: %s' % (M.shape, sigout.shape)

            polx = np.polyfit(sigout, M, order)
            M_polx = np.polyval(polx, sigout)

            print 'polx coeff:',
            for k in polx: print format(k, '1.4f'),
            print

            # save the polynomial fit for each different measurement case
            np.savetxt(self.pprpath+case+'.pol'+ str(order), polx)

            # make sure the data is sorted according to sigout
            sorti = sigout.argsort()
            M = M[sorti]
            M_polx = M_polx[sorti]
            sigout = sigout[sorti]

            # for the plotting, be consistent and make them all positive
            if polx[0] < 0:
                XX = -1.0
            else:
                XX = 1.0

#            plt.subplot(3,1,plotnr)
            c = colors[plotnr-1]
            label = case.replace('_', '\_')
            ax_list[chi].plot(M, XX*sigout, c+'^')#, label=label)
            ax_list[chi].plot(M_polx, XX*sigout, c, alpha=0.7)

            plotnr +=1
#            leg = ax_list[chi].legend(loc='best')
#            leg.get_frame().set_alpha(0.5)

        # for each channel make a transfer function considering all the
        # different cases and save it too.
        polx_ch, M_polx_ch = ['','','',''], ['','','','']
        for chi in range(4):
            # and now for each plot the global fitted data
            polx_ch[chi] = np.polyfit(sigout_ch[chi], M_ch[chi], order)
            M_polx_ch[chi] = np.polyval(polx_ch[chi], sigout_ch[chi])
            # save the polyfit for the combined measurements
            # for the filename, ditch the last _ from combined case name
            np.savetxt(self.pprpath+case_ch[chi][:-1]+'.pol'+ str(order), polx)

            # and plot the fitted line: the actual transformation function
            # stuff the make the label for the final fitted line on all results
            if polx_ch[chi][0] < 0:
                XX = -1.0
            else:
                XX = 1.0
            aa = XX*polx_ch[chi][0]
            bb = XX*polx_ch[chi][1]
            if bb < 0:
                operator = '-'
            else:
                operator = '+'
            label = '$%1.5f x %s %7.4f$' % (aa, operator, abs(bb) )
            ax_list[chi].plot(M_polx_ch[chi], XX*sigout_ch[chi], 'k-',
                                label=label)
            leg = ax_list[chi].legend(loc='best')
            leg.get_frame().set_alpha(0.5)

        # set the maxima the same for each plot for more easy comparison
        if figfile.find('stiff') > 0:
            ymax = 600.0
        else:
            ymax = 1000.0
        for ax in ax_list:
            ax.set_ylim([-100, ymax])
            ax.set_xlim([-0.5, 4.5])

        pa4.save_fig()

    def _m_strain_gauges(self, data):
        """
        Use simple statics to determine the moments at the location of the
        strain gauges when loaded at the clamp in the Vliegtuighal.
        Point "a" is at the clamp.

        Parameters
        ----------

        data : ndarray(n,4)
            weights placed on the blade on following positions
            data[w45, w34, w15, tip]
        """

        # spc: scipy.constants
        w45 = data[:,0]*spc.g
        w34 = data[:,1]*spc.g
        w15 = data[:,2]*spc.g
        wtip= data[:,3]*spc.g

        # method 1: more equitations need to solved
#        Fa = w45 + w34 + w15
#        Ma = -(0.1*w45) - (0.21*w34) - (0.4*w15)
#
#        S1 = Fa
#        M1 = -Ma - (0.02*S1)
#
#        S2 = Fa - w45
#        M2 = -Ma - (0.1*w45) - (0.16*S2)

        # or way more simple, make a cut at the strain gauges and look towards
        # the tip, put moment equilibrium at the strain gauges and solve!
        M1 = - 0.08*w45 - 0.19*w34 - 0.38*w15 - 0.51*wtip
        M2 =            - 0.05*w34 - 0.24*w15 - 0.37*wtip

        return -M1, -M2

    def beam_only(self):
        """
        Calibration data based on Peekel measurements in the Vliegtuighal,
        excluding the foam.

        1V = 1000 micro strain
        """

        # 0      1    2     3  4  5  6  7       8
        # w45, w34, w15, wtip, 1, 2, 3, 4, delta t
        # M1,   M2
        respath = '/home/dave/PhD/Projects/OJF/CalibrationData/'

#        resfile_a1_flex = 'A1_flexblade_nofoam.csv'
#        resfile_b1_stiff = 'B1_stiffblade_nofoam.csv'
#        resfile_b2_flex = 'B2_stiffblade_nofoam.csv'
#        resfile_b1_flex_foam = 'B1_stiffblade_foam.csv'

        M_points = np.linspace(0,1, num=100)

        # -------------------------------------------------
        # setup plot
        # -------------------------------------------------
        figfile = 'bladestraincal-peekel-stiff'
        pa4 = plotting.A4Tuned()
        pa4.setup(self.figpath+figfile, grandtitle=figfile, nr_plots=4,
                         wsleft_cm=2., wsright_cm=1.0, hspace_cm=10.)

        # -------------------------------------------------------------------
        # flexible blade, no foam
        # -------------------------------------------------------------------
        resfile = 'A1_flexblade_nofoam.csv'
        f = respath + resfile
        self.a1_flex = np.loadtxt(f, skiprows=2, delimiter=',')
        M1, M2 = self._m_strain_gauges(self.a1_flex)
        # fit the data with Least Squares, create a polynomial class
        p1 = np.poly1d(np.polyfit(M1, self.a1_flex[:,4], 1))
        p2 = np.poly1d(np.polyfit(M1, self.a1_flex[:,5], 1))
        p3 = np.poly1d(np.polyfit(M2, self.a1_flex[:,6], 1))
        p4 = np.poly1d(np.polyfit(M2, self.a1_flex[:,7], 1))
        print resfile
        print 'p1', p1.coeffs
        print 'p2', p2.coeffs
        print 'p3', p3.coeffs
        print 'p4', p4.coeffs

        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax.set_title(resfile)
        ax.plot(M1, self.a1_flex[:,4], 'r^', label='1')
        ax.plot(M1, self.a1_flex[:,5], 'rs', label='2')
        ax.plot(M2, self.a1_flex[:,6], 'g^', label='3')
        ax.plot(M2, self.a1_flex[:,7], 'gs', label='4')
        leg = ax.legend(loc='upper left')
        leg.get_frame().set_alpha(0.6)
        ax.set_xlabel('Moment')
        ax.set_ylabel('Voltage')
        # also plot the fits
        ax.plot(M_points, p1(M_points), 'r')
        ax.plot(M_points, p2(M_points), 'r')
        ax.plot(M_points, p3(M_points), 'g')
        ax.plot(M_points, p4(M_points), 'g')
        ax.set_xlim(xmax=0.5)
        ax.grid(True)

#        plt.figure()
#        plt.subplot(221)
#        plt.title(resfile)
#        plt.plot(M1, self.a1_flex[:,4], 'r^', label='1')
#        plt.plot(M1, self.a1_flex[:,5], 'rs', label='2')
#        plt.plot(M2, self.a1_flex[:,6], 'g^', label='3')
#        plt.plot(M2, self.a1_flex[:,7], 'gs', label='4')
#        plt.legend(loc='upper left')
#        plt.xlabel('Moment')
#        plt.ylabel('Voltage')
#        # also plot the fits
#        plt.plot(M_points, p1(M_points))
#        plt.xlim(xmax=0.5)

        # -------------------------------------------------------------------
        # stiff blade 1, no foam
        # -------------------------------------------------------------------
        resfile = 'B1_stiffblade_nofoam.csv'
        f = respath + resfile
        self.b1_stiff = np.loadtxt(f, skiprows=2, delimiter=',')
        M1, M2 = self._m_strain_gauges(self.b1_stiff)
        # fit the data with Least Squares, create a polynomial class
        p1 = np.poly1d(np.polyfit(M1, self.b1_stiff[:,4], 1))
        p2 = np.poly1d(np.polyfit(M1, self.b1_stiff[:,5], 1))
        p3 = np.poly1d(np.polyfit(M2, self.b1_stiff[:,6], 1))
        p4 = np.poly1d(np.polyfit(M2, self.b1_stiff[:,7], 1))
        print resfile
        print 'p1', p1.coeffs
        print 'p2', p2.coeffs
        print 'p3', p3.coeffs
        print 'p4', p4.coeffs

        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
        ax.set_title(resfile)
        ax.plot(M1, self.b1_stiff[:,4], 'r^', label='1')
        ax.plot(M1, self.b1_stiff[:,5], 'rs', label='2')
        ax.plot(M2, self.b1_stiff[:,6], 'g^', label='3')
        ax.plot(M2, self.b1_stiff[:,7], 'gs', label='4')
        leg = ax.legend(loc='upper left')
        leg.get_frame().set_alpha(0.6)
        ax.set_xlabel('Moment')
        ax.set_ylabel('Voltage')
        # also plot the fits
        ax.plot(M_points, p1(M_points), 'r')
        ax.plot(M_points, p2(M_points), 'r')
        ax.plot(M_points, p3(M_points), 'g')
        ax.plot(M_points, p4(M_points), 'g')
        ax.grid(True)
#        pa4.xlim(xmax=0.5)

##        plt.figure()
#        plt.subplot(222)
#        plt.title(resfile)
#        plt.plot(M1, self.b1_stiff[:,4], 'r^', label='1')
#        plt.plot(M1, self.b1_stiff[:,5], 'rs', label='2')
#        plt.plot(M2, self.b1_stiff[:,6], 'g^', label='3')
#        plt.plot(M2, self.b1_stiff[:,7], 'gs', label='4')
#        plt.legend(loc='upper left')
#        plt.xlabel('Moment')
#        plt.ylabel('Voltage')
#        # also plot the fits
#        plt.plot(M_points, p1(M_points))

        # -------------------------------------------------------------------
        # stiff blade 2, no foam
        # -------------------------------------------------------------------
        resfile = 'B2_stiffblade_nofoam.csv'
        f = respath + resfile
        self.b2_stiff = np.loadtxt(f, skiprows=2, delimiter=',')
        M1, M2 = self._m_strain_gauges(self.b2_stiff)
        # fit the data with Least Squares, create a polynomial class
        p1 = np.poly1d(np.polyfit(M1, self.b2_stiff[:,4], 1))
        p2 = np.poly1d(np.polyfit(M1, self.b2_stiff[:,5], 1))
        p3 = np.poly1d(np.polyfit(M2, self.b2_stiff[:,6], 1))
        p4 = np.poly1d(np.polyfit(M2, self.b2_stiff[:,7], 1))
        print resfile
        print 'p1', p1.coeffs
        print 'p2', p2.coeffs
        print 'p3', p3.coeffs
        print 'p4', p4.coeffs

        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 3)
        ax.set_title(resfile)
        ax.plot(M1, self.b2_stiff[:,4], 'r^', label='1')
        ax.plot(M1, self.b2_stiff[:,5], 'rs', label='2')
        ax.plot(M2, self.b2_stiff[:,6], 'g^', label='3')
        ax.plot(M2, self.b2_stiff[:,7], 'gs', label='4')
        leg = ax.legend(loc='upper left')
        leg.get_frame().set_alpha(0.6)
        ax.set_xlabel('Moment')
        ax.set_ylabel('Voltage')
        # also plot the fits
        ax.plot(M_points, p1(M_points), 'r')
        ax.plot(M_points, p2(M_points), 'r')
        ax.plot(M_points, p3(M_points), 'g')
        ax.plot(M_points, p4(M_points), 'g')
        ax.grid(True)

##        plt.figure()
#        plt.subplot(223)
#        plt.title(resfile)
#        plt.plot(M1, self.b2_stiff[:,4], 'r^', label='1')
#        plt.plot(M1, self.b2_stiff[:,5], 'rs', label='2')
#        plt.plot(M2, self.b2_stiff[:,6], 'g^', label='3')
#        plt.plot(M2, self.b2_stiff[:,7], 'gs', label='4')
#        plt.legend(loc='upper left')
#        plt.xlabel('Moment')
#        plt.ylabel('Voltage')
#        # also plot the fits
#        plt.plot(M_points, p1(M_points))

        # -------------------------------------------------------------------
        # stiff blade 2, foam
        # -------------------------------------------------------------------
        resfile = 'B2_stiffblade_foam.csv'
        f = respath + resfile
        self.b1_stiff_f = np.loadtxt(f, skiprows=2, delimiter=',')
        M1, M2 = self._m_strain_gauges(self.b1_stiff_f)
        # fit the data with Least Squares, create a polynomial class
        p1 = np.poly1d(np.polyfit(M1, self.b1_stiff_f[:,4], 1))
        p2 = np.poly1d(np.polyfit(M1, self.b1_stiff_f[:,5], 1))
        p3 = np.poly1d(np.polyfit(M2, self.b1_stiff_f[:,6], 1))
        p4 = np.poly1d(np.polyfit(M2, self.b1_stiff_f[:,7], 1))
        print resfile
        print 'p1', p1.coeffs
        print 'p2', p2.coeffs
        print 'p3', p3.coeffs
        print 'p4', p4.coeffs

        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 4)
        ax.set_title(resfile)
        ax.plot(M1, self.b1_stiff_f[:,4], 'r^', label='1')
        ax.plot(M1, self.b1_stiff_f[:,5], 'rs', label='2')
        ax.plot(M2, self.b1_stiff_f[:,6], 'g^', label='3')
        ax.plot(M2, self.b1_stiff_f[:,7], 'gs', label='4')
        leg = ax.legend(loc='upper left')
        leg.get_frame().set_alpha(0.6)
        ax.set_xlabel('Moment')
        ax.set_ylabel('Voltage')
        # also plot the fits
        ax.plot(M_points, p1(M_points), 'r')
        ax.plot(M_points, p2(M_points), 'r')
        ax.plot(M_points, p3(M_points), 'g')
        ax.plot(M_points, p4(M_points), 'g')
        ax.grid(True)

        pa4.save_fig()

##        plt.figure()
#        plt.subplot(224)
#        plt.title(resfile)
#        plt.plot(M1, self.b1_stiff_f[:,4], 'r^', label='1')
#        plt.plot(M1, self.b1_stiff_f[:,5], 'rs', label='2')
#        plt.plot(M2, self.b1_stiff_f[:,6], 'g^', label='3')
#        plt.plot(M2, self.b1_stiff_f[:,7], 'gs', label='4')
#        plt.legend(loc='upper left')
#        plt.xlabel('Moment')
#        plt.ylabel('Voltage')
#        # also plot the fits
#        plt.plot(M_points, p1(M_points))


    def extension(self):
        """
        Verify that there is no influence of the blade axial loading on the
        strain output
        """
        #figpath  = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        #figpath += 'BladeStrainCalExt/'
        #
        #
        #respath = '/home/dave/PhD_data/OJF_data_edit/extension_drag/'
        #cases = []
        #cases.append('01_stiff_B2_ext_trigger01_3641704062.csv')
        #cases.append('02_stiff_B2_ext_trigger01_1582045735.csv')
        #cases.append('03_flex_B2_ext_trigger01_4167243314.csv')
        #cases.append('05_flex_B2_ext_trigger01_3474366086.csv')
        #cases.append('04_flex_B2_drag_trigger01_953281293.csv')
        #cases.append('06_stiff_B2_drag_trigger01_733514185.csv')
        #
        #case = '01_stiff_B2_ext_trigger01_3641704062.csv'
        ## and plot them all, raw
        #blade = ojfresult.BladeStrainFile(respath + case)
        #channels = [0,1,2,3]
        ## plot the raw signal
        #plot = plotting.A4Tuned()
        #title = ' '.join(case.split('_')[1:4])
        #plot.plot_simple(figpath+case, blade.time[200:], blade.data[200:,:],
                         #blade.labels, channels=channels, grandtitle=title)

        respath = '/home/dave/PhD_data/OJF_data_edit/extension_drag/'

        # =====================================================================
        # 01_stiff_B2_ext_trigger01_3641704062.csv
        # =====================================================================
        # BLADE 1
        run = '01_stiff_B2_ext_trigger01_3641704062.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[100:], blade.data[100:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_01_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')
        # -------------------------------------------------------------------
        runid = 'ext_01_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')

        # =====================================================================
        # 02_stiff_B2_ext_trigger01_1582045735.csv
        # =====================================================================
        # BLADE 1
        run = '02_stiff_B2_ext_trigger01_1582045735.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[100:], blade.data[100:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_02_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')
        # -------------------------------------------------------------------
        runid = 'ext_02_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')

        # =====================================================================
        # 03_flex_B2_ext_trigger01_4167243314.csv
        # =====================================================================
        # BLADE 1
        run = '03_flex_B2_ext_trigger01_4167243314.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[100:], blade.data[100:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_03_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')
        # -------------------------------------------------------------------
        runid = 'ext_03_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')

        # =====================================================================
        # 05_flex_B2_ext_trigger01_3474366086.csv
        # =====================================================================
        # BLADE 1
        run = '05_flex_B2_ext_trigger01_3474366086.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[100:], blade.data[100:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_05_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')
        # -------------------------------------------------------------------
        runid = 'ext_05_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')

    def extension2(self):
        """
        Second try of the extension tests
        """

        # also collect only the relevant points for curve fitting
        box = 3.392
        tube = 2.373
        cu = 1.447
        weights = np.array([box, box+tube, box+tube+cu])

        stiff_B2_ch1 = np.ndarray((2,0))
        stiff_B2_ch2 = np.ndarray((2,0))
        flex_B2_ch3 = np.ndarray((2,0))
        flex_B2_ch4 = np.ndarray((2,0))

        delta_ch1 = np.ndarray((2,0))
        delta_ch2 = np.ndarray((2,0))
        delta_ch3 = np.ndarray((2,0))
        #delta_ch4 = np.array([])

        respath = '/home/dave/PhD_data/OJF_data_edit/extension_drag/'

        # =====================================================================
        # 700_stiff_B2.csv
        # =====================================================================
        run = '700_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_700_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([box, box+tube, box+tube+cu, box+tube, box, 0])
        stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        deltas = (t[:-1]-t[-1])
        kgs = w[:-1]
        delta_ch1 = np.append(delta_ch1, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_700_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair[1:]
        w = np.array([box, box+tube, box+tube+cu, box+tube, box, 0])
        stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        deltas = t[:-1]-t[-1]
        kgs = w[:-1]
        delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 701_stiff_B2.csv
        # =====================================================================
        run = '701_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_701_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # -------------------------------------------------------------------
        runid = 'ext_701_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)

        # =====================================================================
        # 702_stiff_B2.csv
        # =====================================================================
        run = '702_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_702_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, 0])
        stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zeromean = (t[-1] + t[0])/2.
        deltas = (t[1:-1]-zeromean)
        kgs = w[1:-1]
        delta_ch1 = np.append(delta_ch1, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_702_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair[[0,1,2,4,5]]
        w = np.array([0, box, box+tube, box+tube+cu, 0])
        stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zeromean = (t[-1] + t[0])/2.
        deltas = t[1:-1]-zeromean
        kgs = w[1:-1]
        delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 703_flex_B2.csv
        # =====================================================================
        run = '703_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_703_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair[[1,3,5,6,7]]
        w = np.array([0, box, box+tube, box+tube+cu, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zeromean = (t[-1] + t[0])/2.
        deltas = (t[1:-1]-zeromean)
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_703_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)

        # =====================================================================
        # 704_flex_B2.csv
        # =====================================================================
        run = '704_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_704_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zeromean = (t[-1] + t[0])/2.
        deltas = (t[1:-1]-zeromean)
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_704_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)

        # =====================================================================
        # 705_flex_B2.csv
        # =====================================================================
        run = '705_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_705_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zeromean = t[-1]
        deltas = (t[:-1]-zeromean)
        kgs = w[:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_705_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)


        # -------------------------------------------------------------------
        # and save the results
        np.savetxt(figpath+'stiff_B2_ch1_allpoints', stiff_B2_ch1)
        np.savetxt(figpath+'stiff_B2_ch2_allpoints', stiff_B2_ch2)
        np.savetxt(figpath+'flex_B2_ch3_allpoints', flex_B2_ch3)
        np.savetxt(figpath+'flex_B2_ch4_allpoints', flex_B2_ch4)

        np.savetxt(figpath+'delta_ch1', delta_ch1)
        np.savetxt(figpath+'delta_ch2', delta_ch2)
        np.savetxt(figpath+'delta_ch3', delta_ch3)

    def extension3_flex(self):
        """
        Third and final attempt of the extension tests
        """

        # also collect only the relevant points for curve fitting
        box = 4.816
        tube = 2.371
        cu = 2.321

        #stiff_B2_ch1 = np.ndarray((2,0))
        #stiff_B2_ch2 = np.ndarray((2,0))
        flex_B2_ch3 = np.ndarray((2,0))
        flex_B2_ch4 = np.ndarray((2,0))

        #delta_ch1 = np.ndarray((2,0))
        #delta_ch2 = np.ndarray((2,0))
        delta_ch3 = np.ndarray((2,0))
        delta_ch4 = np.ndarray((2,0))

        respath = '/home/dave/PhD_data/OJF_data_edit/extension_drag/'

        # =====================================================================
        # 805_flex_B2.csv
        # =====================================================================
        run = '805_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_805_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        # for these measurements, zero is always the first and last point
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_805_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch4 = np.append(flex_B2_ch4, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch4 = np.append(delta_ch4, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 806_flex_B2.csv
        # =====================================================================
        run = '806_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_806_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        # for these measurements, zero is always the first and last point
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_806_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch4 = np.append(flex_B2_ch4, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch4 = np.append(delta_ch4, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 807_flex_B2.csv
        # =====================================================================
        run = '807_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_807_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        # for these measurements, zero is always the first and last point
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_807_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #flex_B2_ch4 = np.append(flex_B2_ch4, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch4 = np.append(delta_ch4, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 808_flex_B2.csv
        # =====================================================================
        run = '808_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_808_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        # for these measurements, zero is always the first and last point
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_808_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch4 = np.append(flex_B2_ch4, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch4 = np.append(delta_ch4, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 810_flex_B2.csv
        # =====================================================================
        run = '810_flex_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_810_flex_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch3 = np.append(flex_B2_ch3, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        # for these measurements, zero is always the first and last point
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch3 = np.append(delta_ch3, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_810_flex_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=0.4, smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        # and add the points to the collection
        t = data_stair[[0,1,2,4,5,6,7]]
        w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        flex_B2_ch4 = np.append(flex_B2_ch4, np.array([w, t]), axis=1)
        # and the strain per kg, remove the zero measurement to get deltas
        zero_mean = (t[0]+t[-1])/2.0
        deltas = t[1:-1] - zero_mean
        kgs = w[1:-1]
        delta_ch4 = np.append(delta_ch4, np.array([kgs, deltas/kgs]), axis=1)
        np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # -------------------------------------------------------------------
        # and save the results
        np.savetxt(figpath+'stiff_B2_ch3_allpoints_8xx', flex_B2_ch3)
        np.savetxt(figpath+'stiff_B2_ch4_allpoints_8xx', flex_B2_ch4)

        np.savetxt(figpath+'delta_ch3_8xx', delta_ch3)
        np.savetxt(figpath+'delta_ch4_8xx', delta_ch4)

    def extension3_stiff(self):
        """
        """

        # also collect only the relevant points for curve fitting
        box = 4.816
        tube = 2.371
        cu = 2.321

        stiff_B2_ch1 = np.ndarray((2,0))
        stiff_B2_ch2 = np.ndarray((2,0))
        #flex_B2_ch3 = np.ndarray((2,0))
        #flex_B2_ch4 = np.ndarray((2,0))

        delta_ch1 = np.ndarray((2,0))
        delta_ch2 = np.ndarray((2,0))
        #delta_ch3 = np.ndarray((2,0))
        #delta_ch4 = np.ndarray((2,0))

        respath = '/home/dave/PhD_data/OJF_data_edit/extension_drag/'

        # =====================================================================
        # 811_flex_B2.csv
        # =====================================================================
        run = '811_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_811_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        ## for these measurements, zero is always the first and last point
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch1 = np.append(delta_ch1, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_811_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=1.4, smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 812_flex_B2.csv
        # =====================================================================
        run = '812_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_812_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        ## for these measurements, zero is always the first and last point
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch1 = np.append(delta_ch1, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_812_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=1.4, smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 813_flex_B2.csv
        # =====================================================================
        run = '813_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_813_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        ## for these measurements, zero is always the first and last point
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch1 = np.append(delta_ch1, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_813_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=1.4, smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 814_flex_B2.csv
        # =====================================================================
        run = '814_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_814_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        ## for these measurements, zero is always the first and last point
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch1 = np.append(delta_ch1, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_814_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=1.4, smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))

        # =====================================================================
        # 815_flex_B2.csv
        # =====================================================================
        run = '815_stiff_B2.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[10:], blade.data[10:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_815_stiff_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch1')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,0],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=2., smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch1 = np.append(stiff_B2_ch1, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        ## for these measurements, zero is always the first and last point
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch1 = np.append(delta_ch1,np.array([kgs,deltas/kgs]),axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))
        # -------------------------------------------------------------------
        runid = 'ext_815_stiff_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch2')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,1],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=1.4, smoothen='moving', smooth_window=3,
                    points_per_stair=40)
        ## and add the points to the collection
        #t = data_stair
        #w = np.array([0, box, box+tube, box+tube+cu, box+tube, box, 0])
        #stiff_B2_ch2 = np.append(stiff_B2_ch2, np.array([w, t]), axis=1)
        ## and the strain per kg, remove the zero measurement to get deltas
        #zero_mean = (t[0]+t[-1])/2.0
        #deltas = t[1:-1] - zero_mean
        #kgs = w[1:-1]
        #delta_ch2 = np.append(delta_ch2, np.array([kgs, deltas/kgs]), axis=1)
        #np.savetxt(figpath+runid+'_deltas', np.array([kgs, deltas/kgs]))


        # -------------------------------------------------------------------
        # and save the results
        np.savetxt(figpath+'stiff_B2_ch1_allpoints_8xx', stiff_B2_ch1)
        np.savetxt(figpath+'stiff_B2_ch2_allpoints_8xx', stiff_B2_ch2)

        np.savetxt(figpath+'delta_ch1_8xx', delta_ch1)
        np.savetxt(figpath+'delta_ch2_8xx', delta_ch2)

    def drag(self):
        """
        See what the influence of an edgewise force is (drag in this case)
        """

        respath = '/home/dave/PhD_data/OJF_data_edit/extension_drag/'

        # =====================================================================
        # 04_flex_B2_drag_trigger01_953281293.csv
        # =====================================================================
        # BLADE 1
        run = '04_flex_B2_drag_trigger01_953281293.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[100:], blade.data[100:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'drag_04_flex_B2_ch1'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')
        # -------------------------------------------------------------------
        runid = 'drag_04_flex_B2_ch2'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')

        # =====================================================================
        # 06_stiff_B2_drag_trigger01_733514185.csv
        # =====================================================================
        # BLADE 1
        run = '06_stiff_B2_drag_trigger01_733514185.csv'
        blade = ojfresult.BladeStrainFile(respath + run)
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'BladeStrainCalExt/'
        figfile = run
        title = ' '.join(run.split('_')[1:4])

        channels = [0,1,2,3]
        # plot the raw signal
        plot = plotting.A4Tuned()
        plot.plot_simple(figpath+figfile, blade.time[100:], blade.data[100:,:],
                         blade.labels, channels=channels, grandtitle=title)

        # -------------------------------------------------------------------
        runid = 'ext_06_stiff_B2_ch3'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch3')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,2],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')
        # -------------------------------------------------------------------
        runid = 'ext_06_stiff_B2_ch4'
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       runid=runid, figfile=figfile+'_ch4')
        time_stair, data_stair = sc.setup_filter(blade.time, blade.data[:,3],
                    dt_treshold=0.00015, cutoff_hz=False, dt=1,
                    stair_step_tresh=5., smoothen='moving')

if __name__ == '__main__':
    dummy=None


