# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:20:54 2013

@author: dave
"""

import math

import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
from scipy import optimize
import pylab as plt
import sympy

import plotting
import ojfresult
from staircase import StairCase
from misc import find0


class TowerCalibration:
    """
    TowerCalibration
    ================
    """
    
    
    def __init__(self):
        """
        1 Volt = 1000 micro strain
        """
        pass
    
    def february(self, plot=False, step=0.00001):
        """
        Tower calibration was done simple in February: put on the weight
        and just read from the Peekel what the strain values are. The noted
        values are included in this method.
        
        This is not a good way to proceed: script and data are in the same 
        place, and that shouldn't be.
        """
        
        
        self.figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        self.figpath += 'TowerStrainCal/'
        self.pprpath = self.figpath
        
        # ---------------------------------------------------------
        # Translate calibration mass to tower top force
        # ---------------------------------------------------------
        
        # data_cal(n,2) = [mass, FA reading, SS reading]
        mass_holder = 0.57166
        
        # readings from 3/2/2012, at 15 deg C, rho=1.253 kg/m3
        self.data_cal = np.ndarray((7,3))
        self.data_cal[0,:] = [0.0,              -0.024, 0.008]
        self.data_cal[1,:] = [mass_holder,       0.001, 0.010]
        self.data_cal[2,:] = [mass_holder+2 ,    0.089, 0.017]
        self.data_cal[3,:] = [mass_holder+3 ,    0.134, 0.021]
        self.data_cal[4,:] = [mass_holder+10.02, 0.449, 0.047]
        self.data_cal[5,:] = [mass_holder,       0.001, 0.009]
        self.data_cal[6,:] = [0,                -0.022, 0.007]
        
        # convert the mass to a bending moment at the strain gauges
        moment_arm_cal = 0.92 - 0.115
        # moment arm rotor is the distance from the rotor centre to the strain
        # gauges at the tower base, just above the first bearing
#        momemt_arm_rotor = 1.64
        
        # convert to a tower top force
        #self.data_cal[:,0] *= 9.81*moment_arm_cal/momemt_arm_rotor
        
        # convert to bending moment at the strain gauges
        self.data_cal *= 9.81*moment_arm_cal
        
        # ---------------------------------------------------------
        # Calibration load not perfectly applied at 0 yaw
        # ---------------------------------------------------------
        delta_ss = self.data_cal[4,2] - self.data_cal[2,2]
        delta_fa = self.data_cal[4,1] - self.data_cal[2,1]
        angle = math.atan(delta_ss/delta_fa)
        print 'off-set angle calibration force:', angle*180/np.pi, 'deg'
        
        # ---------------------------------------------------------
        # FA: fine grid with the different voltage settings
        # ---------------------------------------------------------
        self.volt_hd_fa = np.arange(self.data_cal[0,1],self.data_cal[4,1],step)
        # x is the voltage, that's the value we measure, y is the transposed
        # outcome (bending moment tower at the strain gauges)
        volt_fa = self.data_cal[0:5,1]
        tfa = self.data_cal[0:5,0]*math.cos(angle)
        self.tfa_hd = sp.interpolate.griddata(volt_fa, tfa, self.volt_hd_fa)
        
        # ---------------------------------------------------------
        # SS: fine grid with the different voltage settings
        # ---------------------------------------------------------
        self.volt_hd_ss = np.arange(self.data_cal[0,2],self.data_cal[4,2],step)
        volt_ss = self.data_cal[0:5,2]
        tss = self.data_cal[0:5,0]*math.sin(angle)
        self.tss_hd = sp.interpolate.griddata(volt_ss, tss, self.volt_hd_ss)
        
        # ---------------------------------------------------------
        # Create the transformation function
        # ---------------------------------------------------------
        # x values are what is given in the measurements: voltage, so volt_grid
        # the transformation function should convert voltages to rotor thrust
        
        self.pol1_fa = np.polyfit(self.volt_hd_fa, self.tfa_hd, 1, full=False)
        self.tfa_pol1 = np.polyval(self.pol1_fa, self.volt_hd_fa)
        
        self.pol1_ss = np.polyfit(self.volt_hd_ss, self.tss_hd, 1, full=False)
        self.tss_pol1 = np.polyval(self.pol1_ss, self.volt_hd_ss)
        
        print 'pol1_fa', self.pol1_fa,
        print 'angle:', np.arctan(self.pol1_fa[0])*180/np.pi
#        
        print 'pol1_ss', self.pol1_ss,
        print 'angle:', np.arctan(self.pol1_ss[0])*180/np.pi
        
        # ---------------------------------------------------------
        # Save calibration and polyfit data
        # ---------------------------------------------------------
        np.savetxt(self.pprpath+'towercal-data_cal', self.data_cal)
        np.savetxt(self.pprpath+'towercal-volt_hd_fa', self.volt_hd_fa)
        np.savetxt(self.pprpath+'towercal-volt_hd_ss', self.volt_hd_ss)
        np.savetxt(self.pprpath+'towercal-tfa_hd', self.tfa_hd)
        np.savetxt(self.pprpath+'towercal-tss_hd', self.tss_hd)
        np.savetxt(self.pprpath+'towercal-pol1_fa', self.pol1_fa)
        np.savetxt(self.pprpath+'towercal-pol1_ss', self.pol1_ss)
        
        # ---------------------------------------------------------
        # Errors
        # ---------------------------------------------------------
        pol1_err = np.abs(self.tfa_pol1 - self.tfa_hd)
        pol1_err_p = 100*pol1_err/self.tfa_hd
        # ignore values for which volt_grid too close to zero
        pol1_err_p[0:3000] = np.nan
        
        # ---------------------------------------------------------
        # Extended data range
        # ---------------------------------------------------------
        # to make sure we have cover the entire range of voltages
        # occuring in the measurements
#        volt_ext = np.arange(-1,1,step)
#        tfa_pol2_ext = np.polyval(self.pol2, volt_ext)
        
        # ---------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------
        
        if plot:
            figfile = 'tower-strain-calibration'
            
            pa4 = plotting.A4Tuned()
            pa4.setup(self.figpath+figfile, nr_plots=1, grandtitle=figfile,
                   figsize_y=15, wsleft_cm=2., wsright_cm=2.)
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
            ax2 = ax1.twinx()
            ax2.set_zorder(10)
            
            ax1.plot(self.data_cal[:,0]*math.cos(angle), self.data_cal[:,1],
                      'r-s', label='FA', alpha=0.5)
            ax1.plot(self.data_cal[:,0]*math.sin(angle), self.data_cal[:,2],
                      'b-o', label='SS', alpha=0.5)
#            ax1.plot(self.tfa_hd, self.volt_hd_fa, 'k-', label='FA hd')
            ax1.plot(self.tfa_pol1, self.volt_hd_fa, 'k-', label='FA pol1')
            ax1.plot(self.tss_pol1, self.volt_hd_ss, 'k-', label='SS pol1')
#            ax1.plot(tfa_pol2_ext, volt_ext, 'k-', label='FA pol2 ext')
            
            ax2.plot(self.tfa_hd, pol1_err, label='pol1 err [N]', alpha=0.5,
                     zorder=10)
            ax2.plot(self.tfa_hd, pol1_err_p, label='pol1 err [%]', alpha=0.5,
                     zorder=10)
            
#            ax1.set_xlabel('Tower base bending moment [Nm]')
            ax1.set_xlabel('Tower base bending moment [Nm]')
            ax1.set_ylabel('Tower strain output voltage [V]')
            ax2.set_ylabel('error')
            ax1.grid(True)
#            ax2.grid(True)
            
            leg1 = ax1.legend(loc='upper left')
            leg2 = ax2.legend(loc='center right')
            
            leg1.get_frame().set_alpha(0.8)
            leg2.get_frame().set_alpha(0.8)
            
            pa4.save_fig()
    
    def april_249(self):
        """
        During April the tower calibration was done a bit different: recorded
        in dSPACE instead of manually reading the values from the Peekel.
        """
        
        # ---------------------------------------------------------
        # Translate calibration mass to tower top force
        # ---------------------------------------------------------
        
        # data_cal(n,2) = [mass, FA reading, SS reading]
        # mass holder: the wooden thing where the masses were placed
        mass_holder = 0.57166
        
        # convert the mass to a bending moment at the strain gauges
        # NOTE: this moment arm was different in February and April
        # see drawings
        moment_arm_cal = 0.94 - 0.115
        # moment arm rotor is the distance from the rotor centre to the strain
        # gauges at the tower base, just above the first bearing
#        momemt_arm_rotor = 1.64
        
        # ----------------------------------------------
        # run 249
        # ----------------------------------------------
        
        # calibration dataset: the weights put on the balans
        data_cal = np.array([0, 0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 
                             1.1, 1.2])
        # all but the first zero measurement had the mass holder in place
        data_cal[1:] += mass_holder
        
        # convert to a tower top force
        #data_cal *= 9.81*moment_arm_cal/momemt_arm_rotor
        
        # convert to bending moment at the strain gauges
        data_cal *= 9.81*moment_arm_cal
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCal-04/'
        resfile = '0405_run_249_towercal'
        dspace = ojfresult.DspaceMatFile(respath + resfile)
        # select the channels of interest
        fa = dspace.labels_ch['Tower Strain For-Aft filtered']
        ss = dspace.labels_ch['Tower Strain Side-Side filtered']
        iyaw = dspace.labels_ch['Yaw Laser']
        
        istart = 0*dspace.sample_rate
        istop = 0*dspace.sample_rate
        istop = -1
        
        # and read the yaw angle fot this test
        yaw = dspace.data[istart:istop,iyaw]
        # calibrate the yaw angle signal, load the transformation polynomial
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        ycp = calpath + 'YawLaserCalibration-04/runs_289_295.yawcal-pol10'
        pol = np.loadtxt(ycp)
        yaw = np.polyval(pol, yaw).mean()
        # and save
        np.savetxt(figpath + resfile + '_yawangle', np.array([yaw]))
        
        # FOR AFT ------------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_FA',
                       runid=resfile+'_FA')
        time_stair, data_stair = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,fa], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=1.0e-6,
                    stair_step_tresh=0.0015, smoothen='moving',
                    points_per_stair=5000)
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
                   ylabel='Strain sensor output',
                   ylabel_err='error fitted-measured data [\%]',
                   xlabel='Tower base moment FA [Nm]')
        
        # SIDE-SIDE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_SS',
                       runid=resfile+'_SS')
        time_stair, data_stair = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,ss], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=2.0e-8,
                    stair_step_tresh=0.00015, smoothen='moving',
                    points_per_stair=500)
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
                   ylabel='Strain sensor output',
                   ylabel_err='error fitted-measured data [\%]',
                   xlabel='Tower base moment SS [Nm]')
        
    
    def april_250(self):
        """
        During April the tower calibration was done a bit different: recorded
        in dSPACE instead of manually reading the values from the Peekel.
        """
        
        # ---------------------------------------------------------
        # Translate calibration mass to tower top force
        # ---------------------------------------------------------
        
        # data_cal(n,2) = [mass, FA reading, SS reading]
        # mass holder: the wooden thing where the masses were placed
        mass_holder = 0.57166
        
        # convert the mass to a bending moment at the strain gauges
        # NOTE: this moment arm was different in February and April
        # see drawings
        moment_arm_cal = 0.94 - 0.115
        # moment arm rotor is the distance from the rotor centre to the strain
        # gauges at the tower base, just above the first bearing
        momemt_arm_rotor = 1.64
        
        # ----------------------------------------------
        # run 250
        # ----------------------------------------------
        
        # calibration dataset: the weights put on the balans
        data_cal = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
        # all but the first zero measurement had the mass holder in place
        data_cal[1:] += mass_holder
        
        # convert to a tower top force
        #data_cal *= 9.81*moment_arm_cal/momemt_arm_rotor
        
        # convert to bending moment at the strain gauges
        data_cal *= 9.81*moment_arm_cal
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCal-04/'
        resfile = '0405_run_250_towercal'
        dspace = ojfresult.DspaceMatFile(respath + resfile)
        # select the channels of interest
        fa = dspace.labels_ch['Tower Strain For-Aft filtered']
        ss = dspace.labels_ch['Tower Strain Side-Side filtered']
        iyaw = dspace.labels_ch['Yaw Laser']
        
        istart = 0*dspace.sample_rate
        istop = 0*dspace.sample_rate
        istop = -1
        
        # and read the yaw angle fot this test
        yaw = dspace.data[istart:istop,iyaw]
        # calibrate the yaw angle signal, load the transformation polynomial
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        ycp = calpath + 'YawLaserCalibration-04/runs_289_295.yawcal-pol10'
        pol = np.loadtxt(ycp)
        yaw = np.polyval(pol, yaw).mean()
        # and save
        np.savetxt(figpath + resfile + '_yawangle', np.array([yaw]))
        
        # FOR AFT ------------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_FA',
                       runid=resfile+'_FA')
        time_stair, data_stair = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,fa], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=1.0e-6,
                    stair_step_tresh=0.0015, smoothen='moving',
                    points_per_stair=5000)
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
                   ylabel='Strain sensor output',
                   ylabel_err='error fitted-measured data [\%]',
                   xlabel='Tower base moment FA [Nm]')
        
        # SIDE-SIDE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_SS',
                       runid=resfile+'_SS')
        time_stair, data_stair = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,ss], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=2.0e-8,
                    stair_step_tresh=0.00015, smoothen='moving',
                    points_per_stair=500)
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
                   ylabel='Strain sensor output',
                   ylabel_err='error fitted-measured data [\%]',
                   xlabel='Tower base moment SS [Nm]')
    

    def april_251(self):
        """
        During April the tower calibration was done a bit different: recorded
        in dSPACE instead of manually reading the values from the Peekel.
        """
        
        # ---------------------------------------------------------
        # Translate calibration mass to tower top force
        # ---------------------------------------------------------
        
        # data_cal(n,2) = [mass, FA reading, SS reading]
        # mass holder: the wooden thing where the masses were placed
        mass_holder = 0.57166
        
        # convert the mass to a bending moment at the strain gauges
        # NOTE: this moment arm was different in February and April
        # see drawings
        moment_arm_cal = 0.94 - 0.115
        # moment arm rotor is the distance from the rotor centre to the strain
        # gauges at the tower base, just above the first bearing
#        momemt_arm_rotor = 1.64
        
        # ----------------------------------------------
        # run 250
        # ----------------------------------------------
        
        # calibration dataset: the weights put on the balans
        data_cal = np.array([0.0, 0.0, 10.0, 11.0])
        # all but the first zero measurement had the mass holder in place
        data_cal[1:] += mass_holder
        
        # convert to a tower top force
        #data_cal *= 9.81*moment_arm_cal/momemt_arm_rotor
        
        # convert to bending moment at the strain gauges
        data_cal *= 9.81*moment_arm_cal
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCal-04/'
        resfile = '0405_run_251_towercal'
        dspace = ojfresult.DspaceMatFile(respath + resfile)
        # select the channels of interest
        fa = dspace.labels_ch['Tower Strain For-Aft filtered']
        ss = dspace.labels_ch['Tower Strain Side-Side filtered']
        iyaw = dspace.labels_ch['Yaw Laser']
        
        istart=  6.*dspace.sample_rate
        istop = 65.*dspace.sample_rate
        #istop = -1
        
        # and read the yaw angle fot this test
        yaw = dspace.data[istart:istop,iyaw]
        # calibrate the yaw angle signal, load the transformation polynomial
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        ycp = calpath + 'YawLaserCalibration-04/runs_289_295.yawcal-pol10'
        pol = np.loadtxt(ycp)
        yaw = np.polyval(pol, yaw).mean()
        # and save
        np.savetxt(figpath + resfile + '_yawangle', np.array([yaw]))
        
        # FOR AFT ------------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_FA',
                       runid=resfile+'_FA')
        time_stair, data_stair = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,fa], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=1.0e-6,
                    stair_step_tresh=0.0015, smoothen='moving',
                    points_per_stair=5000)
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
                   ylabel='Strain sensor output',
                   ylabel_err='error fitted-measured data [\%]',
                   xlabel='Tower base moment FA [Nm]')
        
        # SIDE-SIDE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_SS',
                       runid=resfile+'_SS')
        time_stair, data_stair = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,ss], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=2.0e-8,
                    stair_step_tresh=0.00015, smoothen='moving',
                    points_per_stair=500)
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
                   ylabel='Strain sensor output',
                   ylabel_err='error fitted-measured data [\%]',
                   xlabel='Tower base moment SS [Nm]')
    
    def april_combine(self, direction='FA'):
        """
        Combine the 249, 250 and 251 calibration points and make a fit on
        all the available data
        """
        
        dd = direction
        p='/home/dave/PhD/Projects/PostProcessing/OJF_tests/TowerStrainCal-04/'
        
        towercallist = ['0405_run_249_towercal_%s' % direction, 
                        '0405_run_250_towercal_%s' % direction,
                        '0405_run_251_towercal_%s' % direction]
        
        # ---------------------------------------------------------
        # setup the plot
        # ---------------------------------------------------------
        figfile = 'april_tower_calibration_all_%s' % direction
        title = '%s tower strain gauge calibration' % direction
        pa4 = plotting.A4Tuned(scale=1.5)
        pwx = plotting.TexTemplate.pagewidth*0.5
        pwy = plotting.TexTemplate.pagewidth*0.4
        pa4.setup(p+figfile, grandtitle=None, nr_plots=1,
                         wsleft_cm=1.8, wsright_cm=0.5, hspace_cm=2.0,
                         size_x_perfig=pwx, size_y_perfig=pwy, wstop_cm=0.7,
                         wsbottom_cm=1.0)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        
        xlabel = 'Tower base moment [Nm]'
        ylabel = 'Strain sensor output'
        
        # ---------------------------------------------------------
        # combine all the calibration measurement points in one set
        # ---------------------------------------------------------
        data_stair_all = np.array([])
        data_cal_all   = np.array([])
        for item in towercallist:
            # load each dataset
            data_stair = np.loadtxt(p + item + '-data_stair')
            data_cal   = np.loadtxt(p + item + '-data_cal')
            # and put in one array
            data_stair_all = np.append(data_stair_all, data_stair)
            data_cal_all = np.append(data_cal_all, data_cal)
        
        # ---------------------------------------------------------
        # fit the aggregated data
        # ---------------------------------------------------------
        # sort the set first
        isort = np.argsort(data_stair_all)
        data_stair_all = data_stair_all[isort]
        data_cal_all = data_cal_all[isort]
        # and fit
        polx = np.polyfit(data_stair_all, data_cal_all, 1)
        data_cal_polx = np.polyval(polx, data_stair_all)
        # and save
        np.savetxt(p + 'towercal_249_250_251_%s-cal_pol1' % dd, polx)
        np.savetxt(p + 'towercal_249_250_251_%s-data_cal' % dd, data_cal_all)
        np.savetxt(p+'towercal_249_250_251_%s-data_stair' % dd,data_stair_all)
        
        # calcualte the quality of the fit
        # for the definition of coefficient of determination, denoted R2
        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        SS_tot = np.sum(np.power( (data_cal_all - data_cal_all.mean()), 2 ))
        SS_err = np.sum(np.power( (data_cal_all - data_cal_polx), 2 ))
        R2 = 1 - (SS_err/SS_tot)
        
        # ---------------------------------------------------------
        # and start plotting
        # ---------------------------------------------------------
        ax1.plot(data_cal_all, data_stair_all, 'rs', 
                 label='measurements', alpha=0.7)
        # put the transformation function as a label
        aa = polx[0]
        bb = polx[1]
        if bb < 0: 
            op = '-'
        else:
            op = '+'
        label = '$%1.1f x %s %1.2f$\n$R^2=%1.4f$' % (aa, op, abs(bb), R2)
        
        ax1.plot(data_cal_polx, data_stair_all, 'k', label=label)
        ax1.set_title(title)
        leg1 = ax1.legend(loc='upper left')
        leg1.get_frame().set_alpha(0.8)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True)
        pa4.save_fig()
    
    def april_print_all_raw(self):
        """
        To get an overview of the data and quality, print all the raw data
        """
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/'
        figpath += 'OJF_tests/TowerStrainCal-04/'
        
        # using the fish wire, which snapped right at the start
        resfile = '0405_run_248_towercal_destroy_fish_wire'
        channels = ['Tower Strain For-Aft', 'Tower Strain Side-Side']
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
        resfile = '0405_run_249_towercal'
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
        resfile = '0405_run_250_towercal'
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
        resfile = '0405_run_251_towercal'
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
        resfile = '0405_run_253_towervibrations'
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
        resfile = '0405_run_259_towercal_towerstrainwithyawerrors'
        channels.append('Yaw Laser')
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
        resfile = '0405_run_260_towercal_towerstrainwithyawerrors'
        ds = ojfresult.DspaceMatFile(matfile=respath+resfile)
        ds.plot_channel(figpath=figpath, channel=channels)
        
    # TODO: add the wind correction to the correction set
    def windspeed_correction(self):
        """
        Based on the tower only measurements, deduce the tower drag induced
        strains.
        
        CAUTION: we have a different receiver in April and we can't find any
        thrustworty overlapping measurements. This stuff is hence worthless!
        """
        
#        respath = '/home/dave/PhD_data/OJF_data_edit/02/calibration/'
#        # tower only files
#        resfile1 = '0203_tower_norotor_nowire_4_20_0ms'
#        dm1 = ojfresult.DspaceMatFile(matfile=respath+resfile1+'.mat' )
#    
#        resfile2 = '0203_tower_norotor_wire_0_4_20'
#        dm2 = ojfresult.DspaceMatFile(matfile=respath+resfile2+'.mat' )
#        log2 = OJFLogFile(ojffile=respath+resfile2+'.log')
       
        # ----------------------------------------------
        # NO WIRE CASE
        # ----------------------------------------------
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/02/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/TowerWind/'
        resfile = '0203_tower_norotor_nowire_4_20_0ms'
        cr = ojfresult.ComboResults(respath,resfile)
        # select the channels of interest
        fa_fil = cr.dspace.labels_ch['Tower Strain For-Aft filtered']
#        ss_fil = cr.dspace.labels_ch['Tower Strain Side-Side filtered']
        #ch_dspace = [fa_fil, ss_fil]
        #ch_ojf = [cr.ojf.labels_ch['wind speed']]
        #ch_blade = []
        #cr.plot_combine(figpath, ch_dspace, ch_ojf, ch_blade)
        
        # only consider up to 14 m/s wind speeds
        istop = 370*cr.dspace.sample_rate
        istart = 10*cr.dspace.sample_rate
        
        # calirbation dataset: the wind speeds at each stair
        data_cal = np.arange(4,17,1)
        
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_filter_FA',
                       runid='0203_tower_norotor_nowire')
        time_stair, data_stair = sc.setup_filter(cr.dspace.time[istart:istop],
                    cr.dspace.data[istart:istop,fa_fil], smooth_window=3,
                    cutoff_hz=False, dt=1, dt_treshold=0.00000018,
                    stair_step_tresh=0.0015, smoothen='moving')
                    # dt_treshold=0.00008, start=5000,
        # and derive the transformation function
        # note that now it is the other way around, a wind speed (the 
        # calibration data should result in a raw signal output)
        #sc.polyfit(data_cal, data_stair, order=4, xlabel='Wind speed [m/s]',
        sc.polyfit(data_stair, data_cal, order=4, ylabel='Wind speed [m/s]',
                   xlabel='Strain sensor output',
                   ylabel_err='error strain sensor output/fitted',
                   err_label_abs='error fitted data')
        
        # ----------------------------------------------
        # WIRE CASE
        # ----------------------------------------------
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/02/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/TowerWind/'
        resfile = '0203_tower_norotor_wire_0_4_20'
        cr = ojfresult.ComboResults(respath,resfile)
        # select the channels of interest
        fa_fil = cr.dspace.labels_ch['Tower Strain For-Aft filtered']
#        ss_fil = cr.dspace.labels_ch['Tower Strain Side-Side filtered']
        #ch_dspace = [fa_fil, ss_fil]
        #ch_ojf = [cr.ojf.labels_ch['wind speed']]
        #ch_blade = []
        #cr.plot_combine(figpath, ch_dspace, ch_ojf, ch_blade)
        
        # only consider up to 14 m/s wind speeds
        istop = 490*cr.dspace.sample_rate
        istart = 10*cr.dspace.sample_rate
        
        # calirbation dataset: the wind speeds at each stair
        #data_cal = np.arange(0,17,1)
        data_cal = np.arange(4,17,1)
        
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_filter_FA',
                       runid='0203_tower_norotor_wire_0_4_20')
        time_stair, data_stair = sc.setup_filter(cr.dspace.time[istart:istop],
                    cr.dspace.data[istart:istop,fa_fil], smooth_window=3,
                    cutoff_hz=False, dt=1, dt_treshold=0.0000001,
                    stair_step_tresh=0.0015, smoothen='moving')
                    # dt_treshold=0.00008, start=5000,
        # Zero wind condition does not result in zero strain. Take this offset
        # into account
        data_stair = data_stair - data_stair[0]
        # but zero point is irrelevant for the polyfit, take it out
        data_stair = data_stair[1:]
        # and derive the transformation function
        # note that now it is the other way around, a wind speed (the 
        # calibration data should result in a raw signal output)
        #sc.polyfit(data_cal, data_stair, order=9, xlabel='Wind speed [m/s]',
        sc.polyfit(data_stair, data_cal, order=4, ylabel='Wind speed [m/s]',
                   xlabel='Strain sensor output',
                   ylabel_err='error strain sensor output/fitted',
                   err_label_abs='error fitted data')
    
    def yaw_259_260(self):
        """
        See if we can make any conclusions out of the yawed strain calibration
        """
        
        # mass holder: the wooden thing where the masses were placed
        mass_holder = 0.57166
        # convert the mass to a bending moment at the strain gauges
        # NOTE: this moment arm was different in February and April
        # see drawings
        moment_arm_cal = 0.94 - 0.115
        
        # =====================================================================
        # 0405_run_259_towercal_towerstrainwithyawerrors
        # =====================================================================
        # calibration dataset: the weights put on the balans
        data_cal = np.array([2.0]) + mass_holder
        # convert to bending moment at the strain gauges
        data_cal *= 9.81*moment_arm_cal
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCalYaw/'
        resfile = '0405_run_259_towercal_towerstrainwithyawerrors'
        dspace = ojfresult.DspaceMatFile(respath + resfile)
        # select the channels of interest
        fa = dspace.labels_ch['Tower Strain For-Aft filtered']
        ss = dspace.labels_ch['Tower Strain Side-Side filtered']
        yaw = dspace.labels_ch['Yaw Laser']
        
        istart= 17.0*dspace.sample_rate
        istop = 70.0*dspace.sample_rate
        
        # YAW ANGLE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_YAW',
                       runid=resfile+'_YAW')
        time_stair, yaw259 = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,yaw], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=8.0e-6,
                    stair_step_tresh=0.2, smoothen='moving',
                    points_per_stair=1500)
        # FOR AFT ------------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_FA',
                       runid=resfile+'_FA')
        time_stair, FA259 = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,fa], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=7.0e-7,
                    stair_step_tresh=0.0005, smoothen='moving',
                    points_per_stair=1000)
#        # and derive the transformation function
#        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
#                   ylabel='Strain sensor output',
#                   ylabel_err='error fitted-measured data [\%]',
#                   xlabel='Tower base moment FA [Nm]')
        # SIDE-SIDE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_SS',
                       runid=resfile+'_SS')
        time_stair, SS259 = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,ss], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=7.0e-7,
                    stair_step_tresh=0.001, smoothen='moving',
                    points_per_stair=1000)
#        # and derive the transformation function
#        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
#                   ylabel='Strain sensor output',
#                   ylabel_err='error fitted-measured data [\%]',
#                   xlabel='Tower base moment SS [Nm]')
        
        # =====================================================================
        # 0405_run_260_towercal_towerstrainwithyawerrors
        # =====================================================================
        
        # load the dspace and ojf files
        respath = '/home/dave/PhD_data/OJF_data_edit/04/calibration/'
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCalYaw/'
        resfile = '0405_run_260_towercal_towerstrainwithyawerrors'
        dspace = ojfresult.DspaceMatFile(respath + resfile)
        # select the channels of interest
        fa = dspace.labels_ch['Tower Strain For-Aft filtered']
        ss = dspace.labels_ch['Tower Strain Side-Side filtered']
        yaw = dspace.labels_ch['Yaw Laser']
        
        istart=  5.0*dspace.sample_rate
        istop = 80.0*dspace.sample_rate
        
        # YAW ANGLE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_YAW',
                       runid=resfile+'_YAW')
        time_stair, yaw260 = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,yaw], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=9.0e-6,
                    stair_step_tresh=0.2, smoothen='moving',
                    points_per_stair=1000)
        # FOR AFT ------------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_FA',
                       runid=resfile+'_FA')
        time_stair, FA260 = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,fa], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=4.0e-7,
                    stair_step_tresh=0.0002, smoothen='moving',
                    points_per_stair=1000)
#        # and derive the transformation function
#        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
#                   ylabel='Strain sensor output',
#                   ylabel_err='error fitted-measured data [\%]',
#                   xlabel='Tower base moment FA [Nm]')
        # SIDE-SIDE ----------------------------------------------------------
        # setup the filters for each case
        sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                       figfile=resfile+'_SS',
                       runid=resfile+'_SS')
        time_stair, SS260 = sc.setup_filter(dspace.time[istart:istop],
                    dspace.data[istart:istop,ss], smooth_window=1,
                    cutoff_hz=False, dt=1, dt_treshold=7.0e-7,
                    stair_step_tresh=0.001, smoothen='moving',
                    points_per_stair=1500)
#        # and derive the transformation function
#        sc.polyfit(data_cal, data_stair, order=1, err_label_rel='error [\%]',
#                   ylabel='Strain sensor output',
#                   ylabel_err='error fitted-measured data [\%]',
#                   xlabel='Tower base moment SS [Nm]')
        
        # =====================================================================
        # merge, sort, and save all the data
        # =====================================================================
        
        fa = np.append(FA259, FA260)
        ss = np.append(SS259, SS260)
        yaw = np.append(yaw259, yaw260)
        # and sort on yaw angle
        isort = yaw.argsort()
        fa = fa[isort]
        ss = ss[isort]
        yaw = yaw[isort]
        
        # calibrate the yaw signal
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        ycp = calpath + 'YawLaserCalibration-04/runs_289_295.yawcal-pol10'
        pol = np.loadtxt(ycp)
        yaw = np.polyval(pol, yaw)
        
        # and save the data
        cc = calpath + 'TowerStrainCalYaw/'
        np.savetxt(cc + 'towercal_259_260.yawangle', yaw)
        np.savetxt(cc + 'towercal_259_260.FA', fa)
        np.savetxt(cc + 'towercal_259_260.SS', ss)
        
        # =====================================================================
        # Analysis of the yaw dependency
        # =====================================================================
        plt.plot(yaw259, FA259, 'rv-')
        plt.plot(yaw259, SS259, 'b>--')
        plt.plot(yaw260, FA260, 'g^-')
        plt.plot(yaw260, SS260, 'y<--')
        plt.grid()
        
        # is the resultant constant?
        plt.plot(yaw259, np.sqrt( (FA259**2) + (SS259**2) ), 'kd:')
        plt.plot(yaw260, np.sqrt( (FA260**2) + (SS260**2) ), 'kd:')
    
    
    def yaw_influence(self):
        """
        Derive the yaw influence on the strain gauges
        """
        
        # =====================================================================
        # offset for the channels: FA, SS on zero load
        # =====================================================================
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        cc = calpath + 'TowerStrainCal-04/'
        # collect all the measurment data
        Mn = np.loadtxt(cc + 'towercal_249_250_251_FA-data_cal')
        # the ss loads are obviously the same!
        # Mn = np.loadtxt(cc + 'towercal_249_250_251_SS-data_cal')
        FAn = np.loadtxt(cc + 'towercal_249_250_251_FA-data_stair')
        SSn = np.loadtxt(cc + 'towercal_249_250_251_SS-data_stair')
        FA0 = FAn[0:3].mean()
        SS0 = SSn[0:3].mean()
        
        # =====================================================================
        # Load the yawed strain calibration test data
        # =====================================================================
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        cc = calpath + 'TowerStrainCalYaw/'
        PSIy = np.loadtxt(cc + 'towercal_259_260.yawangle')
        FAy = np.loadtxt(cc + 'towercal_259_260.FA')
        SSy = np.loadtxt(cc + 'towercal_259_260.SS')
        # mass holder: the wooden thing where the masses were placed
        mass_holder = 0.57166
        # convert the mass to a bending moment at the strain gauges
        # NOTE: this moment arm was different in February and April
        # see drawings
        moment_arm_cal = 0.94 - 0.115
        My = (2.0 + mass_holder)*moment_arm_cal*9.81
        
        # =====================================================================
        # Analysis of the yaw dependency
        # =====================================================================
        # correct for the non zero offset of the signal
        fay = FAy - FA0
        ssy = SSy - SS0
        fan = FAn - FA0
        ssn = SSn - SS0
        
#        # do the numbers add up?
#        load = np.sqrt( (fa**2) + (ss**2) )
#        load0 = load[6]
#        fa2 = load*np.cos(PSIy*np.pi/180.0)
#        ss2 = load*np.sin(PSIy*np.pi/180.0)
        
        # look at the ratio's excluding the zero point!
        # and it is indeed constant, can't be otherwise, they are both linear!
        print (FAn-FA0) / (SSn-SS0)
        # and look how it goes, does it converge? YES
        plt.plot(Mn, (FAn-FA0) / (SSn-SS0))
        
        # zero yaw angle when FA is max
        psi_fa_max = PSIy[fay.argmax()]
        # zero yaw angle FA: 
        psi_ss_0, iyaw0 = find0(np.array([ssy, PSIy]).transpose())
        # and also save the zero yaw angles!
        np.savetxt(cc + 'psi_fa_max_yawplot', np.array([psi_fa_max]))
        np.savetxt(cc + 'psi_ss_0_yawplot', np.array([psi_ss_0]))
        print
        print 'FA  max at angle:', psi_fa_max
        print 'SS zero at angle:', psi_ss_0
        
        # check that the one measurement of FAn overlaps with FAy
        # find the right moment
        iMy = np.abs(Mn-My).argmin()
        print
        print 'Mn, My', Mn[iMy], My
        print 'FAn under My, zero yaw:', fan[iMy]
        print 'SSn under My, zero yaw:', ssn[iMy]
        
        # now take the yaw angle closest to zero
        iyaw0 = np.abs(PSIy).argmin()
        print 'yaw close to zero:', PSIy[iyaw0]
        print 'FAy under My, zero yaw:', fay[iyaw0]
        print 'SSy under My, zero yaw:', ssy[iyaw0]
        
        print 
        print 'and their relative difference'
        print 'FAn/FAy = %1.5f percent' % (abs(1.0-fan[iMy]/fay[iyaw0])*100.0)
        print 'SSn/SSy = %1.5f percent' % (abs(1.0-ssn[iMy]/ssy[iyaw0])*100.0)
        
        # at which point are FA and SS equal?
        # first, interpolate to the same hd grid
        psi_hd = np.arange(PSIy.min(), PSIy.max(), 0.01)
        fay_hd = interpolate.griddata(PSIy, fay, psi_hd)
        ssy_hd = interpolate.griddata(PSIy, ssy, psi_hd)
        # and find the closest point
        icrossing = np.abs(fay_hd - ssy_hd).argmin()
        intersect = psi_hd[icrossing]
        np.savetxt(cc + 'ss_fa_intersection_yaw_angle', np.array([intersect]))
        print
        print 'FA-SS intersection angle: %1.4f' % intersect
        
        # =====================================================================
        # Plotting
        # =====================================================================
        figfile = calpath + 'TowerStrainCalYaw/strain-yaw-influence'
        
#        allang = np.linspace(-40, 40, 70)
#        
#        plt.plot(PSIy, fa, 'rv-', label='fa')
#        plt.plot(PSIy, ss, 'b>-', label='ss')
#        plt.grid()
#        # is the resultant constant?
#        plt.plot(PSIy, load, 'kd:', label='F')
#        # and what if we only look into the assumed pure FA direction?
#        plt.plot(PSIy, fa2, 'r*-.', label='$F \cos \psi$')
#        plt.plot(PSIy, ss2*-1, 'b*-.', label='$F \sin \psi$')
#        # just plot all the angles
#        plt.plot(allang, -load0*np.sin(allang*np.pi/180.0), 'y+', alpha=0.8)
#        plt.plot(allang, load0*np.cos(allang*np.pi/180.0), 'y+', alpha=0.8)
#        plt.legend(loc='best')
#        plt.savefig(calpath + 'TowerStrainCalYaw/strain-yaw-influence.png')
        
        # save the plotting differently
        pa4 = plotting.A4Tuned(scale=1.5)
        pwx = plotting.TexTemplate.pagewidth*0.6
        pwy = plotting.TexTemplate.pagewidth*0.4
        pa4.setup(figfile, grandtitle=None, nr_plots=1,
                         wsleft_cm=1.8, wsright_cm=0.8, hspace_cm=2.0,
                         size_x_perfig=pwx, size_y_perfig=pwy, wstop_cm=1.0,
                         wsbottom_cm=1.0)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        
        ax1.plot(PSIy, fay, 'r^-', label='for-aft')
        ax1.plot(PSIy, ssy, 'b<--', label='side-side')
        ax1.axvline(x=psi_fa_max, linewidth=1, color='k', aa=False)
        ax1.axvline(x=psi_ss_0, linewidth=1, color='k', aa=False)
        ax1.axvline(x=intersect, linewidth=1, color='k', aa=False)
#        ax1.axhline(y=fa.max(), linewidth=1, color='k', aa=False)
#        ax1.axhline(y=0, linewidth=1, color='k', aa=False)
        
        fatext = '$\psi_{FA_{max}} = %1.2f$' % psi_fa_max
        ax1.text(psi_fa_max, fay.max()*0.7, fatext, va='bottom', 
                 bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5), 
                             fc=(1., 0.8, 0.8), alpha=0.8,))
        
        sstext = '$\psi_{SS_0} = %1.2f$' % psi_ss_0
        ax1.text(psi_ss_0, 0.01, sstext, va='bottom', 
                 bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5), 
                             fc=(1., 0.8, 0.8), alpha=0.8,))
        
        crosstext = '$\psi_{SS=FA} = %1.2f$' % intersect
        ax1.text(intersect*1.12, 0.01, crosstext, va='bottom', 
                 bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5), 
                             fc=(1., 0.8, 0.8), alpha=0.8,))
        
        ax1.legend(loc='best')
        ax1.set_title('Strains at constant loading\nfor different yaw angles')
        ax1.set_xlabel('yaw angle $\psi$')
        ax1.set_ylabel('FA, SS strain signal')
        ax1.grid()
        pa4.save_fig()
    
    def calibrate_zerostrain_angle(self, **kwargs):
        r"""
        Second run over the tower strain calibration. Now include the zero
        strain angles for the FA and SS directions. Now we want to now how
        much load there is on the tower in the direction aligned with the
        yaw angle and perpendicular to that. That would be the true FA/SS 
        directions, and that is not the same as the measured strain directions.
        The measured directions are the \psi_{FA_{max} and \psi_{SS_0}.
        
        Parameters
        ----------
        
        psi_fa_max : float, default=saved angle
            Angle in degrees. Default value is the one saved.
        
        psi_ss_0 : float, default=saved angle
            Angle in degrees. Default value is the one saved.
        
        prefix : str, default='yawplot'
            Specifiy how the transformation needs to be saved. In doing so,
            you can distinguesh bewteen the zero/max strain angles derived
            from the yaw influence plot or the ones through optimisation
        
        """
        
        prefix = kwargs.get('prefix', 'yawplot')
        
        # collect the data points in one array
        fa = np.array([])
        ss = np.array([])
        M_fa = np.array([])
        M_ss = np.array([])
        
        # the psi angles for zero and max strain
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        cc = calpath + 'TowerStrainCalYaw/'
        
        # take the standard saved angles, or some custom angle
        psi_fa_max = kwargs.get('psi_fa_max', 
                                  np.loadtxt(cc + 'psi_fa_max_%s' % prefix))
        psi_ss_0 = kwargs.get('psi_ss_0',
                                  np.loadtxt(cc + 'psi_ss_0_%s' % prefix))
        
        # =====================================================================
        # Load the data again, but now include the yaw angles
        # =====================================================================
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        cc = calpath + 'TowerStrainCal-04/'
        
        # collect all the measurment data
        resfile = '0405_run_249_towercal'
        M_249 = np.loadtxt(cc + resfile + '_FA-data_cal') # same as SS
        FA_249 = np.loadtxt(cc + resfile + '_FA-data_stair')
        SS_249 = np.loadtxt(cc + resfile + '_SS-data_stair')
        yaw_249 = np.loadtxt(cc + resfile + '_yawangle')
        # and convert the applied moment to psi_FA_max, psi_SS_0 directions
        M_249_fa = M_249*np.cos( (psi_fa_max-yaw_249)*np.pi/180.0 )
        M_249_ss = M_249*np.sin( np.abs(psi_ss_0-yaw_249)*np.pi/180.0  )
        # add to the pools
        M_fa = np.append(M_fa, M_249_fa)
        M_ss = np.append(M_ss, M_249_ss)
        fa = np.append(fa, FA_249)
        ss = np.append(ss, SS_249)
        
        # collect all the measurment data
        resfile = '0405_run_250_towercal'
        M_250 = np.loadtxt(cc + resfile + '_FA-data_cal') # same as SS
        FA_250 = np.loadtxt(cc + resfile + '_FA-data_stair')
        SS_250 = np.loadtxt(cc + resfile + '_SS-data_stair')
        yaw_250 = np.loadtxt(cc + resfile + '_yawangle')
        # and convert the applied moment to psi_FA_max, psi_SS_0 directions
        M_250_fa = M_250*np.cos( (psi_fa_max-yaw_250)*np.pi/180.0 )
        M_250_ss = M_250*np.sin( np.abs(psi_ss_0-yaw_250)*np.pi/180.0 )
        # add to the pools
        M_fa = np.append(M_fa, M_250_fa)
        M_ss = np.append(M_ss, M_250_ss)
        fa = np.append(fa, FA_250)
        ss = np.append(ss, SS_250)
        
        # collect all the measurment data
        resfile = '0405_run_251_towercal'
        M_251 = np.loadtxt(cc + resfile + '_FA-data_cal') # same as SS
        FA_251 = np.loadtxt(cc + resfile + '_FA-data_stair')
        SS_251 = np.loadtxt(cc + resfile + '_SS-data_stair')
        yaw_251 = np.loadtxt(cc + resfile + '_yawangle')
        # and convert the applied moment to psi_FA_max, psi_SS_0 directions
        M_251_fa = M_251*np.cos( (psi_fa_max-yaw_251)*np.pi/180.0 )
        M_251_ss = M_251*np.sin( np.abs(psi_ss_0-yaw_251)*np.pi/180.0 )
        # add to the pools
        M_fa = np.append(M_fa, M_251_fa)
        M_ss = np.append(M_ss, M_251_ss)
        fa = np.append(fa, FA_251)
        ss = np.append(ss, SS_251)
        
        
        # =====================================================================
        # fit the aggregated data and save the transormation functions
        # =====================================================================
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        calpath += 'TowerStrainCal-04/towercal_249_250_251_yawcorrect'        
        
        # sort the set first FA
        isort = np.argsort(M_fa)
        M_fa = M_fa[isort]
        fa = fa[isort]
        # and fit
        polx_fa = np.polyfit(fa, M_fa, 1)
        M_fa_polx = np.polyval(polx_fa, fa)
        # and save
        np.savetxt(calpath + '_fa-cal_pol1_%s' % prefix, polx_fa)
        np.savetxt(calpath + '_fa-data_cal_%s' % prefix, M_fa)
        np.savetxt(calpath + '_fa-data_stair_%s' % prefix, fa)
        # calcualte the quality of the fit
        # for the definition of coefficient of determination, denoted R2
        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        fa_tot = np.sum(np.power( (M_fa - M_fa.mean()), 2 ))
        fa_err = np.sum(np.power( (M_fa - M_fa_polx), 2 ))
        R2_fa = 1 - (fa_err/fa_tot)
        
        # sort the set first SS
        isort = np.argsort(M_ss)
        M_ss = M_ss[isort]
        ss = ss[isort]
        # and fit
        polx_ss = np.polyfit(ss, M_ss, 1)
        M_ss_polx = np.polyval(polx_ss, ss)
        # and save
        np.savetxt(calpath + '_ss-cal_pol1_%s' % prefix, polx_ss)
        np.savetxt(calpath + '_ss-data_cal_%s' % prefix, M_ss)
        np.savetxt(calpath + '_ss-data_stair_%s' % prefix, ss)
        # calcualte the quality of the fit
        ss_tot = np.sum(np.power( (M_ss - M_ss.mean()), 2 ))
        ss_err = np.sum(np.power( (M_ss - M_ss_polx), 2 ))
        R2_ss = 1 - (ss_err/ss_tot)
        
        # =====================================================================
        # plot the new calibration functions and compare to calibration data
        # =====================================================================
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCal-04/'
        figfile = 'april_towercal_249_250_251_yawcorrect_fa_%s' % prefix
        # ---------------------------------------------------------
        # setup the plot
        # ---------------------------------------------------------
        title = 'FA April tower strain gauge calibration'
        title += '\nmissalignment corrected'
        pa4 = plotting.A4Tuned(scale=1.5)
        pwx = plotting.TexTemplate.pagewidth*0.5
        pwy = plotting.TexTemplate.pagewidth*0.4
        pa4.setup(figpath+figfile, grandtitle=None, nr_plots=1,
                         wsleft_cm=1.8, wsright_cm=0.5, hspace_cm=2.0,
                         size_x_perfig=pwx, size_y_perfig=pwy, wstop_cm=1.0,
                         wsbottom_cm=1.0)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        
        xlabel = 'Bending moment at strain gauge [Nm]'
        ylabel = 'Strain sensor output'
        
        # ---------------------------------------------------------
        # and start plotting
        # ---------------------------------------------------------
        ax1.plot(M_fa, fa, 'rs', label='measurements', alpha=0.7)
        # put the transformation function as a label
        aa = polx_fa[0]
        bb = polx_fa[1]
        label = '$%1.1f x %+1.2f$\n$R^2=%1.4f$' % (aa, bb, R2_ss)
        
        ax1.plot(M_fa_polx, fa, 'k', label=label)
        ax1.set_title(title)
        leg1 = ax1.legend(loc='upper left')
        leg1.get_frame().set_alpha(0.8)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True)
        pa4.save_fig()
        
        # =====================================================================
        # plot the new calibration functions and compare to calibration data
        # =====================================================================
        figpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        figpath += 'TowerStrainCal-04/'
        figfile = 'april_towercal_249_250_251_yawcorrect_ss_%s' % prefix
        # ---------------------------------------------------------
        # setup the plot
        # ---------------------------------------------------------
        title = 'SS April tower strain gauge calibration'
        title += '\nmissalignment corrected'
        pa4 = plotting.A4Tuned(scale=1.5)
        pa4.setup(figpath+figfile, grandtitle=None, nr_plots=1,
                         wsleft_cm=1.8, wsright_cm=0.5, hspace_cm=2.0,
                         size_x_perfig=pwx, size_y_perfig=pwy, wstop_cm=1.0,
                         wsbottom_cm=1.0)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        
        xlabel = 'Bending moment at strain gauge [Nm]'
        ylabel = 'Strain sensor output'
        
        # ---------------------------------------------------------
        # and start plotting
        # ---------------------------------------------------------
        # CAUTION, SS switched sign for  more easy comparison with FA
        ax1.plot(M_ss, ss, 'rs', label='measurements', alpha=0.7)
        # put the transformation function as a label
        aa = polx_ss[0]
        bb = polx_ss[1]
        label = '$%1.1f x %+1.2f$\n$R^2=%1.4f$' % (aa, bb, R2_fa)
        
        ax1.plot(M_ss_polx, ss, 'k', label=label)
        ax1.set_title(title)
        leg1 = ax1.legend(loc='upper left')
        leg1.get_frame().set_alpha(0.8)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True)
        pa4.save_fig()    
    
    def yaw_influence_calibrate(self):
        """
        Instead of finding the zero crossings, max positions as done in 
        yaw_influence, and than correct the callibration force as done in 
        calibrate_zerostrain_angle, minimize the difference between loading
        and measured loading in the yawed strain calibration experiment
        """
        
        func = find_zerostrain_angle_opt
        x0 = [0, 0]
        
        xopt, cov_x, infodict, mesg, ier = optimize.leastsq(func, x0, 
                              Dfun=None, full_output=1, gtol=0.0,
                              col_deriv=0, ftol=1.0e-07, xtol=1.0e-07, 
                              maxfev=1000, epsfcn=1e-07, diag=None)
        # save the zero/max strain angles
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        cc = calpath + 'TowerStrainCalYaw/'
        np.savetxt(cc + 'psi_fa_max_opt', np.array([xopt[0]]))
        np.savetxt(cc + 'psi_ss_0_opt', np.array([xopt[1]]))
        # now we find the psi_fa_max and psi_ss_0
        # next step: callibrate the loads knowing the right offset angles
        self.calibrate_zerostrain_angle(prefix='opt')
        print 'psi_fa_max', xopt[0]
        print 'psi_ss_0', xopt[1]
        # visually check that we are right
        verify_psi_correction('opt')
        
    
    def try_solve_strain_cal(self):
        r"""
        THIS WHOLE THING IS THE WRONG APPROACH
        we should have looked at the strain formula
        
        Now we assume there are \alpha and \beta offsets for the strain gauges.
        The corresponding calibration process is a bit more complicated
        compared to orthogonally place strain gauges. Since we have plenty
        of measurements for the calibration, we should be able to solve the
        problem.
        """
        
        calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
        cc = calpath + 'TowerStrainCal-04/'
        # collect all the measurment data
        Mn = np.loadtxt(cc + 'towercal_249_250_251_FA-data_cal')
        # the ss loads are obviously the same!
        # Mn = np.loadtxt(cc + 'towercal_249_250_251_SS-data_cal')
        FAn = np.loadtxt(cc + 'towercal_249_250_251_FA-data_stair')
        SSn = np.loadtxt(cc + 'towercal_249_250_251_SS-data_stair')
        cc = calpath + 'TowerStrainCalYaw/'
#        PSIy = np.loadtxt(cc + 'towercal_259_260.yawangle')
        FAy = np.loadtxt(cc + 'towercal_259_260.FA')
        SSy = np.loadtxt(cc + 'towercal_259_260.SS')
        # mass holder: the wooden thing where the masses were placed
#        mass_holder = 0.57166
        # convert the mass to a bending moment at the strain gauges
        # NOTE: this moment arm was different in February and April
        # see drawings
#        moment_arm_cal = 0.94 - 0.115
#        My = (2.0 + mass_holder)*moment_arm_cal
        
        # define all the sympy symbols
        syms = 'M FA SS alpha beta psi a b1 b2'
        M, FA, SS, alpha, beta, psi, a, b1, b2 = sympy.symbols(syms)
        
        # ====================================================================
        # define all parameters
        # ====================================================================
#        psi = PSIy[8]*np.pi/180.0
#        FA = FAy[8]
#        SS = SSy[8]
#        M = My
#        
#        psi = PSIy[9]*np.pi/180.0
#        FA = FAy[9]
#        SS = SSy[9]
#        M = My
#        
#        psi = 0
#        FA = FAn[3]
#        SS = SSn[3]
#        M = Mn[3]
        
        # ====================================================================
        # problem formulation, linear, psi=0
        # ====================================================================
#        alpha = -SS/FA
        
        # ====================================================================
        # problem formulation, linear, psi is not zero
        # ====================================================================
#        # the linearized system, where sin a = a and cos b = 1
#        t2 = ( ( 1.0/(sympy.cos(psi)**2)) - 1.0 - (sympy.tan(psi)**2) )*SS
#        t1 = ( (-1.0/(sympy.cos(psi)**2)) + 1.0 + (sympy.tan(psi)) )*2.0*SS*FA
#        t0 = ( FA*FA/(sympy.cos(psi)**2) - (2*FA*FA) )
#        # the quadratic equation in beta becomes
#        fbeta = t2*beta**2 + t1*beta + t0
#        # and for alpha
#        falpha = (FA - (SS*beta*sympy.tan(psi)) - SS) / FA
#        # and solving will return empty solution. First fill in the values
#        # for FA, SS and psi
        
        # ====================================================================
        # problem formulation, small angles, psi=0
        # ====================================================================
#        # the solution to the linear problem will be a good first guess to
#        # solve numerically the problem?
#        f1 = FAn[3] - SSn[3]*beta - a*Mn[4] - b1
#        f2 = FAn[8] - SSn[8]*beta - a*Mn[8] - b1
#        f3 = FAn[-1] - SSn[-1]*beta - a*Mn[-1] - b1
#        F = [FAn[n] - SSn[n]*beta - a*Mn[n] - b1 for n in range(len(Mn))]
#        F = [ FAn[n] - SSn[n]*beta - a*Mn[n] - b1 for n in [4, 9, 13] ]
#        sympy.solve(F, beta, a, b1)
#        # solve the problem,
#        sympy.solve(F, beta, a, b1)
##        sympy.solve_triangulated([f1,f2,f3], beta, a, b1)
        
        # ====================================================================
        # problem formulation, non linear, psi=0
        # ====================================================================
        
        F = []
        for n in [0, 9, 13, 20]: #[4, 9, 13, 20] range(len(Mn))
            F.append(FAn[n]*sympy.cos(alpha)-SSn[n]*sympy.cos(beta)-a*Mn[n]-b1)
        sympy.solve(F, alpha, beta, a, b1, dict=True)
#        sympy.solve_triangulated([f1,f2,f3], beta, a, b1)
        
        # solve numerically
#        acc_check = 0.0000001
#        solve_acc = 20.0
#        sympy.mpmath.mp.dps = solve_acc
        # initial guess: alpha, beta = 0, b1=0
        x0 = (0, 0, FAn[10]/Mn[10], 0)
        # solve the equation numerically with sympy
        sympy.nsolve(tuple(F), (alpha, beta, a, b1), x0)
        
        # ====================================================================
        # non linear, psi=0, replace FA and SS with linefit using M
        # ====================================================================
        
        F = []
        for n in [0, 4, 9, 13, 18, 20]: #[4, 9, 13, 20] range(len(Mn))
            # this is the inverse of the calibration transformation function
            FA =  (1/366.10)*Mn[n] -  (0.22/366.10)
            SS = (1/1421.09)*Mn[n] + (12.34/1421.09)
            F.append(FA*sympy.cos(alpha)-SS*sympy.sin(beta)-a*Mn[n]-b1)
        print sympy.solve(F, alpha, beta, a, b1)
        x0 = (0, 0, FAn[10]/Mn[10], 0)
        sympy.nsolve(tuple(F), (alpha, beta, a, b1), x0)
        sympy.solve(F, alpha, beta, a, b1)
        
        # ====================================================================
        # X**2 needs to be constant for different yaw angles
        # ====================================================================
        i = 0
        FAtrue = FAy[i]*sympy.cos(alpha)-SSy[i]*sympy.sin(beta)
        SStrue = FAy[i]*sympy.sin(alpha)+SSy[i]*sympy.cos(beta)
        Xi = FAtrue**2 + SStrue**2
        
        i = 5
        FAtrue = FAy[i]*sympy.cos(alpha)-SSy[i]*sympy.sin(beta)
        SStrue = FAy[i]*sympy.sin(alpha)+SSy[i]*sympy.cos(beta)
        Xj = FAtrue**2 + SStrue**2
        
        i = 8
        FAtrue = FAy[i]*sympy.cos(alpha)-SSy[i]*sympy.sin(beta)
        SStrue = FAy[i]*sympy.sin(alpha)+SSy[i]*sympy.cos(beta)
        Xk = FAtrue**2 + SStrue**2
        
        i = 15
        FAtrue = FAy[i]*sympy.cos(alpha)-SSy[i]*sympy.sin(beta)
        SStrue = FAy[i]*sympy.sin(alpha)+SSy[i]*sympy.cos(beta)
        Xl = FAtrue**2 + SStrue**2
        
        # solve algabraicly
        sympy.solve([Xi-Xj, Xk-Xl], alpha, beta)
        
        # solve numerically, dps low is lower accuracy
        sympy.mpmath.mp.dps = 2
        x0 = (0, 0)
        sympy.nsolve((Xi-Xj, Xk-Xl), (alpha, beta), x0)
        
        # see how the values change for different values of alpha, beta
        # create a lambda function to evaluate the shit with numpy
        func_ij = sympy.utilities.lambdify((alpha, beta), Xi-Xj, 'numpy')
        func_kl = sympy.utilities.lambdify((alpha, beta), Xi-Xj, 'numpy')
        
        alphas = np.linspace(-1.5708, 1.5708, 100)
        betas =  np.linspace(-1.5708, 1.5708, 100)
        xx, yy = np.meshgrid(alphas, betas)
        sol_ij = func_ij(xx, yy)
        sol_kl = func_kl(xx, yy)
        
        #CS = plt.contour(xx, yy, sol_ij*100000)
        CS = plt.contour(xx*180/np.pi, yy*180/np.pi, sol_ij*1000)
        plt.clabel(CS, inline=1, fontsize=10)
        
        plt.plot(betas*180/np.pi, sol_ij, 'rs')
        plt.plot(betas*180/np.pi, sol_kl, 'b-')
        
        # ====================================================================
        # the most simple explanation: 
        # ====================================================================
        eq1 = FAn[3]*sympy.sin(alpha)-SSn[3]*sympy.cos(beta)
        eq2 = FAn[8]*sympy.sin(alpha)-SSn[8]*sympy.cos(beta)
        sympy.solve([eq1, eq2], alpha, beta)
        
        # ====================================================================
        # using the ratio of the stress at FA and SS strain locations
        # ====================================================================
        # under different loadings, the ratio between the FA/SS strains should
        # remain the same!
        eq1 = sympy.cos(alpha)/sympy.sin(beta) - (FAn[3]/SSn[3])
        eq2 = sympy.cos(alpha)/sympy.sin(beta) - (FAn[8]/SSn[8])
        sympy.solve([eq1, eq2], alpha, beta)
        # > which of course can't work
        # but look at the ratio's, they are not constant
        print FAn/SSn
        # and look how it goes, does it converge?
        plt.plot(Mn, FAn/SSn)
        
        # ====================================================================
        # problem formulation, non linear, psi is not zero
        # ====================================================================
        
#        # and solve the system for all the measurements we have
#        # initial guess: solve system for delta_x = 0
#        psi0 = math.atan(1 - (A/L))
#        # solve the equation numerically with sympy
#        psi_sol = sympy.nsolve(f1, psi, psi0)

def all_tower_calibrations():
    """
    """
    tc = TowerCalibration()
    # post process the raw measurement data
#    tc.april_249()
#    tc.april_250()
#    tc.april_251()
#    tc.yaw_259_260()
    # plot final results and create transofrmation function
    tc.april_combine(direction='FA')
    tc.april_combine(direction='SS')
    tc.yaw_influence()
    tc.calibrate_zerostrain_angle()


def verify_psi_correction(prefix):
    """
    Is the proposed transformation from FA/SS strain gauge coordinates to
    psi and psi_90 coordinates actually correct? Check with the yawing
    calibration test
    
    Parameters
    ----------
    
    prefix : str
        possible values are opt and yawplot
    """
    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    # definition of the calibration files for April
    ycp04 = calpath + 'YawLaserCalibration-04/runs_289_295.yawcal-pol10'
    # for the tower calibration, the yaw misalignment is already taken
    # into account in the calibration polynomial, no need to include the
    # yaw angle in the calibration. We always measure in the FA,SS dirs
    # if that needs to be converted, than do sin/cos psi to have the
    # components aligned with the wind
    tfacp  = calpath + 'TowerStrainCal-04/'
    tfacp += 'towercal_249_250_251_yawcorrect_fa-cal_pol1_%s' % prefix
    tsscp  = calpath + 'TowerStrainCal-04/'
    tsscp += 'towercal_249_250_251_yawcorrect_ss-cal_pol1_%s' % prefix
    caldict_dspace_04 = {}
    caldict_dspace_04['Yaw Laser'] = ycp04
    caldict_dspace_04['Tower Strain For-Aft'] = tfacp
    caldict_dspace_04['Tower Strain Side-Side'] = tsscp
    # and to convert to yaw coordinate frame of reference
    target_fa = calpath + 'TowerStrainCalYaw/psi_fa_max_%s' % prefix
    caldict_dspace_04['psi_fa_max'] = target_fa
    target_ss = calpath + 'TowerStrainCalYaw/psi_ss_0_%s' % prefix
    caldict_dspace_04['psi_ss_0'] = target_ss
    
    # =====================================================================
    # Load the yawed strain calibration test data
    # =====================================================================
    # mass holder: the wooden thing where the masses were placed
    mass_holder = 0.57166
    # convert the mass to a bending moment at the strain gauges
    # NOTE: this moment arm was different in February and April
    # see drawings
    moment_arm_cal = 0.94 - 0.115
    
    
    # this is uncallibrated data
    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    cc = calpath + 'TowerStrainCalYaw/'
    PSIy = np.loadtxt(cc + 'towercal_259_260.yawangle')
    FAy = np.loadtxt(cc + 'towercal_259_260.FA')
    SSy = np.loadtxt(cc + 'towercal_259_260.SS')
    # calibrate the force, convert to FA,SS coordinate system
    pol = np.loadtxt(caldict_dspace_04['Tower Strain For-Aft'])
    FAy = np.polyval(pol, FAy)
    pol = np.loadtxt(caldict_dspace_04['Tower Strain Side-Side'])
    SSy = np.polyval(pol, SSy)
    # angles between FA/SS axis and psi/psi90
    psi_fa_max = np.loadtxt(caldict_dspace_04['psi_fa_max'])*np.pi/180.0
    psi_ss_0 =   np.loadtxt(caldict_dspace_04['psi_ss_0'])  *np.pi/180.0
    # and transform to psi coordinates
    M_psi = FAy*np.cos(psi_fa_max) + SSy*np.sin(psi_ss_0)
    M_psi90 = -FAy*np.sin(psi_fa_max) + SSy*np.cos(psi_ss_0)
    M_tot = np.sqrt(M_psi**2 + M_psi90**2)
    # the applied force in the psi coordinates
    Mpsi0 = (2.0 + mass_holder)*moment_arm_cal*9.81*sp.ones(len(PSIy))
    M_psi_cal = Mpsi0*np.cos(PSIy*np.pi/180.0)
    M_psi90_cal = -Mpsi0*np.sin(PSIy*np.pi/180.0)
    
    # =====================================================================
    # plot and compare them all
    # =====================================================================
    # save the plotting differently
    figfile=calpath+'TowerStrainCalYaw/strain-yaw-influence-checks_%s' % prefix
    pa4 = plotting.A4Tuned(scale=1.5)
    pwx = plotting.TexTemplate.pagewidth*0.6
    pwy = plotting.TexTemplate.pagewidth*0.6
    pa4.setup(figfile, grandtitle=None, nr_plots=1,
                     wsleft_cm=1.8, wsright_cm=0.8, hspace_cm=2.0,
                     size_x_perfig=pwx, size_y_perfig=pwy, wstop_cm=1.0,
                     wsbottom_cm=1.0)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    
    ax1.plot(PSIy, M_psi_cal, 'bs-', label='$M_{\psi_{{cal}}}$')
    ax1.plot(PSIy, M_psi90_cal, 'bo-', label='$M_{\psi_{90_{cal}}}$')
    ax1.plot(PSIy, Mpsi0, 'bd-', label='$M_{tot_{cal}}$')
    
    ax1.plot(PSIy, M_psi, 'rv--', label='$M_{\psi}$')
    ax1.plot(PSIy, M_psi90, 'r<--', label='$M_{\psi_{90}}$')
    ax1.plot(PSIy, M_tot, 'r>--', label='$M_{tot}$')
    
    #ax1.plot(PSIy, FAy, 'gs--', label='FAy')
    #ax1.plot(PSIy, SSy, 'go-', label='SSy')
    #ax1.plot(PSIy, np.sqrt(FAy**2 + SSy**2), 'gd:', label='$Strain_{tot}$')

#    ax1.plot(PSIy,psi_err*10, 'k--')
#    ax1.plot(PSIy,psi90_err*10, 'k--')
    
    textbox =  '$\psi_{FA_{max}} = %1.2f^{\circ}$' % (psi_fa_max*180.0/np.pi)
    textbox += '\n$\psi_{SS_0} = %1.2f^{\circ}$' % (psi_ss_0*180.0/np.pi)
    ax1.text(30, 10, textbox, va='bottom', horizontalalignment='right',
             bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5), 
                         fc=(1., 0.8, 0.8), alpha=0.8,))
    
    ax1.legend(loc='best')
    title = 'method used $\psi_{FA_{max}}$ and $\psi_{SS_0}$: %s' % prefix
    ax1.set_title(title)
    ax1.set_xlabel('yaw angle $\psi$')
    ax1.set_ylabel('Bending moments in $\psi$ and $\psi_{90}$ directions')
    ax1.grid()
    pa4.save_fig()

def find_zerostrain_angle_opt(x):
    """
    alternative method: find psi_fa_strainmax and psi_ss_strain0 by
    minimizing the difference between measured and actual forces.
    
    The start is the same as calibrate_zerostrain_angle and the finish
    is based on parts of verify_psi_correction.
    
    Parameters
    ----------
    
    x : list
        x[0] = psi_fa_max, x[1] = psi_ss_0. The angles in degrees!
    """
    
    # the psi angles for zero and max strain
    psi_fa_max = x[0]
    psi_ss_0 = x[1]
    
    # collect the data points in one array
    fa = np.array([])
    ss = np.array([])
    M_fa = np.array([])
    M_ss = np.array([])
    
    # =====================================================================
    # Load the data again, but now include the yaw angles
    # =====================================================================
    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    cc = calpath + 'TowerStrainCal-04/'
    
    # collect all the measurment data
    resfile = '0405_run_249_towercal'
    M_249 = np.loadtxt(cc + resfile + '_FA-data_cal') # same as SS
    FA_249 = np.loadtxt(cc + resfile + '_FA-data_stair')
    SS_249 = np.loadtxt(cc + resfile + '_SS-data_stair')
    yaw_249 = np.loadtxt(cc + resfile + '_yawangle')
    # and convert the applied moment to psi_FA_max, psi_SS_0 directions
    M_249_fa = M_249*np.cos( (psi_fa_max-yaw_249)*np.pi/180.0 )
    M_249_ss = M_249*np.sin( np.abs(psi_ss_0-yaw_249)*np.pi/180.0  )
    # add to the pools
    M_fa = np.append(M_fa, M_249_fa)
    M_ss = np.append(M_ss, M_249_ss)
    fa = np.append(fa, FA_249)
    ss = np.append(ss, SS_249)
    
    # collect all the measurment data
    resfile = '0405_run_250_towercal'
    M_250 = np.loadtxt(cc + resfile + '_FA-data_cal') # same as SS
    FA_250 = np.loadtxt(cc + resfile + '_FA-data_stair')
    SS_250 = np.loadtxt(cc + resfile + '_SS-data_stair')
    yaw_250 = np.loadtxt(cc + resfile + '_yawangle')
    # and convert the applied moment to psi_FA_max, psi_SS_0 directions
    M_250_fa = M_250*np.cos( (psi_fa_max-yaw_250)*np.pi/180.0 )
    M_250_ss = M_250*np.sin( np.abs(psi_ss_0-yaw_250)*np.pi/180.0 )
    # add to the pools
    M_fa = np.append(M_fa, M_250_fa)
    M_ss = np.append(M_ss, M_250_ss)
    fa = np.append(fa, FA_250)
    ss = np.append(ss, SS_250)
    
    # collect all the measurment data
    resfile = '0405_run_251_towercal'
    M_251 = np.loadtxt(cc + resfile + '_FA-data_cal') # same as SS
    FA_251 = np.loadtxt(cc + resfile + '_FA-data_stair')
    SS_251 = np.loadtxt(cc + resfile + '_SS-data_stair')
    yaw_251 = np.loadtxt(cc + resfile + '_yawangle')
    # and convert the applied moment to psi_FA_max, psi_SS_0 directions
    M_251_fa = M_251*np.cos( (psi_fa_max-yaw_251)*np.pi/180.0 )
    M_251_ss = M_251*np.sin( np.abs(psi_ss_0-yaw_251)*np.pi/180.0 )
    # add to the pools
    M_fa = np.append(M_fa, M_251_fa)
    M_ss = np.append(M_ss, M_251_ss)
    fa = np.append(fa, FA_251)
    ss = np.append(ss, SS_251)
    
    # sort the set first FA
    isort = np.argsort(M_fa)
    M_fa = M_fa[isort]
    fa = fa[isort]
    # and fit
    polx_fa = np.polyfit(fa, M_fa, 1)
    
    # sort the set first SS
    isort = np.argsort(M_ss)
    M_ss = M_ss[isort]
    ss = ss[isort]
    # and fit
    polx_ss = np.polyfit(ss, M_ss, 1)
    
    # =====================================================================
    # Load the yawed strain calibration test data
    # =====================================================================
    # mass holder: the wooden thing where the masses were placed
    mass_holder = 0.57166
    # convert the mass to a bending moment at the strain gauges
    # NOTE: this moment arm was different in February and April
    # see drawings
    moment_arm_cal = 0.94 - 0.115
    Mpsi0 = (2.0 + mass_holder)*moment_arm_cal*9.81
    
    # this is uncallibrated data
    calpath = '/home/dave/PhD/Projects/PostProcessing/OJF_tests/'
    cc = calpath + 'TowerStrainCalYaw/'
    FAy = np.loadtxt(cc + 'towercal_259_260.FA')
    SSy = np.loadtxt(cc + 'towercal_259_260.SS')
    # calibrate the force, convert to FA,SS coordinate system
    FAy = np.polyval(polx_fa, FAy)
    SSy = np.polyval(polx_ss, SSy)
    # convert to degrees
    psi_fa_max_rad = psi_fa_max * np.pi / 180.0
    psi_ss_0_rad =   psi_ss_0   * np.pi / 180.0
    # and transform to psi coordinates
    M_psi = FAy*np.cos(psi_fa_max_rad) + SSy*np.sin(psi_ss_0_rad)
    M_psi90 = -FAy*np.sin(psi_fa_max_rad) + SSy*np.cos(psi_ss_0_rad)
    
#    # the applied force in the psi coordinates
#    PSIy = np.loadtxt(cc + 'towercal_259_260.yawangle')
#    M_psi_cal = Mpsi0*np.cos(PSIy*np.pi/180.0)
#    M_psi90_cal = -Mpsi0*np.sin(PSIy*np.pi/180.0)
#    # calculate the errors between measured and applied
#    psi_err = M_psi - M_psi_cal
#    psi90_err = M_psi90 - M_psi90_cal
#    return np.abs(np.append(psi_err, psi90_err))
    
    # or only optimize the error between actual and measured load
    return np.abs(np.sqrt(M_psi**2 + M_psi90**2) - Mpsi0)

if __name__ == '__main__':
    dummy = None
    # -------------------------------------------------------------
    # Tower calibration
    # -------------------------------------------------------------
    # create a very first insight into the data: plot raw
    tc = TowerCalibration()
#    tc.april_print_all_raw()
#    tc.windspeed_correction()
#    tc.april_249()
#    tc.april_250()
#    tc.april_251()
#    tc.april_combine(direction='FA')
#    tc.april_combine(direction='SS')
#    tc.yaw_259_260()
#    tc.yaw_influence()
#    tc.calibrate_zerostrain_angle()
    verify_psi_correction('yawplot')
#    tc.yaw_influence_calibrate()
    verify_psi_correction('opt')

