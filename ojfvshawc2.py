# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:06:59 2012

Comparing HAWC2 simulations with OJF results

@author: dave
"""

# built in modules
#import pickle
#import logging
import copy
import os

# 3th party modules
import numpy as np
#import scipy as sp
#from scipy import optimize
#from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interpolate
import pylab as plt

# custom modules
import Simulations as sim
import HawcPy
import plotting
#import ojfpostproc as ojf
import misc
import ojfresult
import ojf_post
import ojfdb

sti = HawcPy.ModelData.st_headers

POSTDIR = 'simulations/hawc2/raw/'
FIGDIR = 'simulations/fig/'
RESDIR = 'simulations/hawc2/'
OJFPATH = 'database/symlinks_all/'
PATH_DB = 'database/'
OJFPATH_RAW = 'data/raw/'

class Chi:
    def __init__(self, ch_dict):
        """
        The HAWC2 channels, based on the given dict
        """
        # global-tower-node-002-forcevec-z
        # local-blade1-node-005-momentvec-z
        # node numbers start with 0 at the root
        # mx is FA, my SS
        self.mx_tower = ch_dict['tower-tower-node-001-momentvec-x']['chi']
        self.my_tower = ch_dict['tower-tower-node-001-momentvec-y']['chi']
        self.mz_tower = ch_dict['tower-tower-node-001-momentvec-z']['chi']

        self.mx_b1_ro = ch_dict['blade1-blade1-node-001-momentvec-x']['chi']
        self.mx_b1_30 = ch_dict['blade1-blade1-node-004-momentvec-x']['chi']
        self.mx_b2_ro = ch_dict['blade2-blade2-node-001-momentvec-x']['chi']
        self.mx_b2_30 = ch_dict['blade2-blade2-node-004-momentvec-x']['chi']

        self.my_b1_ro = ch_dict['blade1-blade1-node-001-momentvec-y']['chi']
        self.my_b1_30 = ch_dict['blade1-blade1-node-004-momentvec-y']['chi']
        self.my_b2_ro = ch_dict['blade2-blade2-node-001-momentvec-y']['chi']
        self.my_b2_30 = ch_dict['blade2-blade2-node-004-momentvec-y']['chi']

        # blade tip deflections, assuming element 11 is at the tip
        blade1 = 'hub1-blade1-elem-011-zrel-1.00-state pos'
        blade2 = 'hub2-blade2-elem-011-zrel-1.00-state pos'
        self.x_b1_tip = ch_dict['%s-x' % blade1]['chi']
        self.x_b2_tip = ch_dict['%s-x' % blade2]['chi']
        self.y_b1_tip = ch_dict['%s-y' % blade1]['chi']
        self.y_b2_tip = ch_dict['%s-y' % blade2]['chi']

        self.fz_shaft = ch_dict['shaft-shaft-node-001-forcevec-z']['chi']
        self.fz_nacelle = ch_dict['nacelle-nacelle-node-002-forcevec-z']['chi']

        self.rpm = ch_dict['bearing-shaft_nacelle-angle_speed-rpm']['chi']
        self.omega = ch_dict['Omega']['chi']
        self.azi = ch_dict['Azi  1']['chi']
        self.mz_shaft = ch_dict['shaft-shaft-node-001-momentvec-z']['chi']

        try:
            self.m_yaw_control = ch_dict['DLL-yaw_control-inpvec-1']['chi']
            self.yaw_ref_angle = ch_dict['DLL-yaw_control-outvec-3']['chi']
            self.yawangle = ch_dict['bearing-yaw_rot-angle-deg']['chi']
        except KeyError:
            # no yaw control channels available
            self.m_yaw_control = None
            self.yaw_ref_angle = None
            self.yawangle = None

        self.aoa16_b1 = ch_dict['Alfa-1-0.16']['chi']
        self.aoa49_b1 = ch_dict['Alfa-1-0.49']['chi']



###############################################################################
### STATIC DEFLECTION BLADE
###############################################################################

class static_blade_deflection:
    """
    Compare the static blade deflection tests (contour tests) with
    equivalent HAWC2 tests
    """

    def __init__(self):
        """
        """
        self.figpath = os.path.join(FIGDIR, 'static_blade_deflection/')

        self.post_dir = POSTDIR
        # measurement data path
        self.testdatapath = os.path.join(OJFPATH_RAW, 'blade_contour/')

    def compare(self, cases, cases0, testcase, figfile, **kwargs):
        """
        Compare a single HAWC2 simulation with the mean deflection values
        from the static tests

        Paramters
        ---------

        cases : sim.Cases object
            Holding only tip loaded HAWC2 simulation

        cases0 : sim.Cases object
            Holding only the zero load HAWC2 simulation

        testcase : dict
            specifying which measurement case: bladenr and structure. Tip mass
            can be taken from the cases object

        figfile : str
            File name for the figure

        Returns
        -------

        defl : ndarray(2,n)

        """

        grandtitle = kwargs.get('grandtitle', False)
        plot = kwargs.get('plot', True)
        scale = kwargs.get('scale', 1.0)
        # ---------------------------------------------------------------------
        # the HAWC2 zero load case
        cc = sim.Cases(cases0)
        rad0, defl0 = cc.blade_deflection(cases0[cases0.keys()[0]])
        # normalize radial position and convert deflection to mm
        rad0, defl0 = rad0/0.555, defl0*1000.

        # ---------------------------------------------------------------------
        # the deflected case
        rad, defl = sim.Cases(cases).blade_deflection(cases[cases.keys()[0]])
        # normalize radial position and convert deflection to mm
        rad, defl = rad/0.555, defl*1000.
        # and the deflection wrt zero case
        defl_result = defl - defl0

        # ---------------------------------------------------------------------
        # the target static blade deflection measurement curve
        fpath = self.testdatapath
        bc = ojfresult.BladeContour()
        bladenr = testcase['bladenr']
        structure = testcase['structure']
        tipmass = str(int(cases[cases.keys()[0]]['[bladetipmass]']*1000))
        mean, corr = bc.mean_defl(fpath, structure, bladenr, tipmass,
                                  silent=False, correct=True)
        deflm = mean[1,:]
        radm = mean[0,:]

        # ---------------------------------------------------------------------
        # dispaly the error, get both deflections to the same grid
        defl_hd = interpolate.griddata(rad, defl_result, radm)
        error = np.abs((defl_hd - deflm)/deflm)*100.

        if plot:
            self.cases = cases
            self.case = cases.keys()[0]
            self._plot(rad, defl_result, radm, deflm, error, figfile,
                       grandtitle=grandtitle, scale=scale)

        return rad, defl_result, radm, deflm, error

    def _plot(self, rad, defl_result, radm, deflm, error, figfile, **kwargs):
        """

        Parameters
        ----------

        optimizer : boolean, default=False
            If True, the x-scaling from an optimized result will be included
            in the plot

        """

        grandtitle = kwargs.get('grandtitle', False)
        optimizer = kwargs.get('optimizer', False)
        scale = kwargs.get('scale', 1.0)
        # ---------------------------------------------------------------------
        # Plotting
        #figfile = file_path + simid
        # we will put two figures beside each other in the document
        figsize_x = plotting.TexTemplate.pagewidth*0.5
        figsize_y = plotting.TexTemplate.pagewidth*0.5
        if grandtitle:
            wstop_cm = 0.6
        else:
            wstop_cm = 0.2
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(self.figpath + figfile, nr_plots=1, hspace_cm=2.,
                   grandtitle=grandtitle, wsleft_cm=1.3, wsright_cm=0.5,
                   wstop_cm=wstop_cm, wsbottom_cm=1., figsize_x=figsize_x,
                   figsize_y=figsize_x)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.plot(rad, defl_result, 'b^--', label='HAWC2 [mm]')
        ax1.plot(radm, deflm, 'r-', label='measured [mm]')
        ax1.plot(radm, error, 'k',label='relative error [\%]')
        if optimizer:
            radopt = self.cases[self.case]['[optimizer results]']['rad']
            xopt = self.cases[self.case]['[optimizer results]']['xopt']
            ax1.plot(radopt, xopt*10., 'go-', alpha=0.5, label='xopt*10')
        ax1.legend(loc='upper left')
        ax1.set_ylim([-1,22])
        ax1.set_xlim([-0.02,1.02])
        ax1.set_xlabel('radial position [\%]')
        ax1.set_ylabel('flap deflection wrt zero load [mm]')
        ax1.grid(True)
        pa4.save_fig()

    def plot_and_compare(self, sim_id, blademap, tiploads, title_add='',
                ploterror=True):
        """

        Plot a range of blades and tip loads

        Parameters
        ----------

        sim_id

        blademap : dict
            Indicate which blade nr and stiffness to inlcude and its
            corresponding st set in the HAWC2 structural file
            {'stiff_B1' : 17}

        tiploads : list
            list ints with the loads in grams

        """
        # load the cases from the dict result database
        cases = sim.Cases(self.post_dir, sim_id)

        # consider all the stiff blade cases
        for m in tiploads:
            # each load case in one plot
            figfile = ''
            grandtitle = 'HAWC2 and measured static deflection\n'
            grandtitle += '%i gr tip load %s' % (m, title_add)
            pa4 = plotting.A4Tuned()
            pa4.setup(figfile, nr_plots=1, hspace_cm=2.,
                       grandtitle=grandtitle, wsleft_cm=1.5, wsright_cm=1.0,
                       wstop_cm=1.8, figsize_y=12., wsbottom_cm=1.)
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

            for blade, st_set in blademap.iteritems():

                # figure blade structure
                if blade.find('flex') > -1:
                    structure = 'flex'
                elif blade.find('stiff') > -1:
                    structure = 'stiff'
                else:
                    structure = ''

                # figure out blade number
                if blade.find('B1') > -1:
                    colorh = 'rs-.'
                    bladenr = 'B1'
                    figfile += blade + '_'
                elif blade.find('B2') > -1:
                    colorh = 'bo-.'
                    bladenr = 'B2'
                    figfile += blade + '_'
                elif blade.find('B3') > -1:
                    colorh = 'g^-.'
                    bladenr = 'B3'
                    figfile += blade + '_'
                # otherwise do not include the case
                else:
                    continue

                # load and plot all OJF measurement cases, LE, TE and mid
                bc = ojfresult.BladeContour()
                search_items = blade.split('_')
                search_items.append(str(m))
                deltas = bc.deltas_case(self.testdatapath, search_items)
                for i, case in enumerate(deltas):
                    data = deltas[case]
                    if   case.find('B1') > -1: color = 'r'
                    elif case.find('B2') > -1: color = 'b'
                    else:                      color = 'g'
                    ax1.plot(data[:,0], data[:,1], color, alpha=0.7)

                    # include mean values in order to calculate error with
                    # HAWC2 simulation
                    if i == 0:
                        mean = data[:,1]
                    else:
                        mean += data[:,1]
                        mean = mean/2.

                # or alternativaly, calculate mean with the new method
                # but than all the files will be loaded from disc again
                # and it gives the same results on the plot
                # the comparison as done below is not working very wel...
                # I am doing something wrong I think
                #mean2=bc.mean_defl(self.testdatapath,structure,bladenr,str(m))
                #mean2 = mean2[1,:]
                #ii = np.isfinite(mean)
                #ii2 = np.isfinite(mean2)
                #print mean2.shape, mean.shape
                #print np.nanmax(np.abs((mean-mean2)/mean))
                #logging.warn(np.allclose(mean[ii], mean2[ii2]))

                # plot the average of LE, TE and mid
                # only give the mean signal a label instead of LE,TE,mid
                # so we avoid cluttering the legend
                label = '%s %s measured' % (bladenr, structure)
                ax1.plot(data[:,0], mean, color, alpha=0.7, label=label)
                #label = '%s %s mean2' % (bladenr, structure)
                #ax1.plot(data[:,0], mean2, 'k', alpha=0.7, label=label)

                # load the HAWC2 ref case
                case0 = '%s_0ms_blade7%i_ab00_tipm0.htc' % (sim_id, st_set)
                rad0, defl0 = self.load_sim_data(cases, case0)
                # load the HAWC2 casefile
                case= '%s_0ms_blade7%i_ab00_tipm0.%03i.htc' % (sim_id,st_set,m)
                rad, defl = self.load_sim_data(cases, case)
                # normalise radial position
                hawc2rad = rad/0.555
                # and the HAWC2 deltas to mm's
                hawc2delta = (defl - defl0)*1000.
                # the actual plotting of the HAWC2 simulation results
                label = '%s %s HAWC2' % (bladenr, structure)
                ax1.plot(hawc2rad, hawc2delta, colorh, label=label)

                # plot the relative errors between HAWC2 and measurements
                # interpolate HAWC2 results to the same grid as measurements
                hawc2hd = interpolate.griddata(hawc2rad,hawc2delta,data[:,0])
                error = np.abs((hawc2hd - mean)/mean)*100.
                if ploterror:
                    ax1.plot(data[:,0], error, 'k', label='rel error \%')

                print case0
                print case
                print

            # finalize the figure name
            pa4.figfile = self.figpath + figfile + str(m)
            ax1.legend(loc='upper left')
            ax1.set_ylim([0,30])
            ax1.set_xlim([-0.02,1.02])
            ax1.set_xlabel('radial position [\%]')
            ax1.set_ylabel('flap deflection wrt zero load [mm]')
            ax1.grid(True)
            pa4.save_fig()

        return

    def load_test_data(self, target):
        """
        """
        data = np.loadtxt(target)
        radpos = data[:,0]
        defl = data[:,1]

        return radpos, defl

    def load_sim_data(self, cases, case):
        """
        """
        hawc2res = cases.load_result_file(cases[case])
        # select all the y deflection channels
        db = misc.DictDB(hawc2res.ch_dict)

        db.search({'sensortype' : 'state pos', 'component' : 'z'})
        # sort the keys and save the mean values to an array/list
        chix, zvals = [], []
        for key in sorted(db.dict_sel.keys()):
            zvals.append(-hawc2res.sig[:,db.dict_sel[key]['chi']].mean())
            chix.append(db.dict_sel[key]['chi'])

        db.search({'sensortype' : 'state pos', 'component' : 'y'})
        # sort the keys and save the mean values to an array/list
        chiy, yvals = [], []
        for key in sorted(db.dict_sel.keys()):
            yvals.append(hawc2res.sig[:,db.dict_sel[key]['chi']].mean())
            chiy.append(db.dict_sel[key]['chi'])

        return np.array(zvals), np.array(yvals)

class blade_aero_only:
    """
    Comparing the blade aero only tests on the OJF balance with the
    corresponding HAWC2 simulations


    """

    def __init__(self):
        """
        """


    def load_sim_data(self, cao):
        """
        Load all mean values of the simulations.
        """

        nrcases = len(cao.cases)
        fm = np.ndarray((6,nrcases))
        aoa = np.ndarray(nrcases)
        i = 0
        for cname, case in cao.cases.iteritems():
            hawc2res = cao.load_result_file(case)

            aoa[i] = case['[pitch_angle]']

            tag = 'balance_arm-balance_arm-node-000-forcevec-x'
            fx = hawc2res.sig[:,hawc2res.ch_dict[tag]['chi']].mean()

            tag = 'balance_arm-balance_arm-node-000-forcevec-y'
            fy = hawc2res.sig[:,hawc2res.ch_dict[tag]['chi']].mean()

            tag = 'balance_arm-balance_arm-node-000-forcevec-z'
            fz = hawc2res.sig[:,hawc2res.ch_dict[tag]['chi']].mean()

            tag = 'balance_arm-balance_arm-node-000-momentvec-x'
            mx = hawc2res.sig[:,hawc2res.ch_dict[tag]['chi']].mean()

            tag = 'balance_arm-balance_arm-node-000-momentvec-y'
            my = hawc2res.sig[:,hawc2res.ch_dict[tag]['chi']].mean()

            tag = 'balance_arm-balance_arm-node-000-momentvec-z'
            mz = hawc2res.sig[:,hawc2res.ch_dict[tag]['chi']].mean()

            fm[0:3,i] = [fx, fy, fz]
            fm[3:6,i] = [mx, my, mz]

            i += 1

        # sort according to aoa
        isort = aoa.argsort()
        aoa = aoa[isort]
        fm = fm[:,isort]

        return aoa, 1000.*fm


    def plot_compare(self, windspeed):
        """
        Indices for blade only aero results OJF
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

        bladepath = os.path.join(OJFPATH_RAW, 'bladeonly/')
        figpath = os.path.join(FIGDIR, 'bladeonly/')

        # settings for the HAWC2 simulations
        postpath = POSTDIR
        resdir = RESDIR
        sim_id = 'aero_02'

        # indices for the OJF result files
        windi=11
        aoai=13
        chis = [1,2,3,4,5,6]
        tt = [r'Drag force, $\frac{F_{x}}{0.5 \rho V^2}$',
              r'Lift force, $\frac{F_{y}}{0.5 \rho V^2}$',
              r'$\frac{F_{z}}{0.5 \rho V^2}$',
              r'Lift moment, $\frac{M_{x}}{0.5 \rho V^2}$',
              r'Drag moment, $\frac{M_{y}}{0.5 \rho V^2}$',
              r'Moment,     $\frac{M_{z}}{0.5 \rho V^2}$']
        names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

        # ====================================================================
        # BLADE 1 FLEX
        #windspeed = 25
        for i, chi in enumerate(chis):
            lowlim = windspeed - 0.5
            uplim  = windspeed + 0.5

            # setup the figure
            figsize_x = plotting.TexTemplate.pagewidth*0.5
            figsize_y = plotting.TexTemplate.pagewidth*0.5
            scale = 1.8
            figname = 'flex-B1-V-%i-ms-%s' % (windspeed, names[i])
            pa4 = plotting.A4Tuned(scale=scale)
            pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                           grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                           wstop_cm=1.0, wsbottom_cm=1.0,
                           figsize_x=figsize_x, figsize_y=figsize_y)
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

            # load the OJF results
            bladefile = '001_flex_b1_t_0.1_n_30.txt'
            res = np.loadtxt(bladepath+bladefile, skiprows=1)
            # only relevant wind speeds
            res = res[res[:,windi].__ge__(lowlim),:]
            res = res[res[:,windi].__le__(uplim),:]

            # hawc2 results
            cao = sim.Cases(postpath, sim_id)
            cao.change_results_dir(resdir)
            # select the relevant cases
            cao.select({'[windspeed]'      : windspeed,
                        '[st_blade_subset]': ojf_post.stsets.b1_flex_opt })
            aoa, fm = self.load_sim_data(cao)

            # select corresponding unit vector with OJF
            if chi == 1:
                # DRAG FORCE
                # x in OJF, y (index=1) in fm
                ii = 1
                fhawc = 1.0
                fojf = 1.0
            elif chi == 2:
                # LIFT FORCE
                # y in ojf -> x in hawc2 fm
                ii = 0
                fhawc = -1.0
                fojf = -1.0 #* 22.0 # calibration issue?
            elif chi == 3:
                # Ignore Fz, nothing interesting to see here
                continue
            elif chi == 4:
                # LIFT MOMENT
                ii = 4
                fhawc = -1.0
                fojf = -1.0
            elif chi == 5:
                # DRAG MOMENT
                ii = 3
                fhawc = -1.0
                fojf = -1.0
            elif chi == 6:
                # aero pitching moment
                ii = 5
                fhawc = -1.0
                fojf = 1.0

            # and normalize
            fm_hawc2 = fhawc * fm[ii,:] / (0.5*1.225*windspeed*windspeed)
            fm_ojf = fojf * res[:,chi] / (0.5*1.225*res[:,windi]*res[:,windi])

            #fm_hawc2 = fhawc * fm[ii,:]
            #fm_ojf = fojf * res[:,chi]

            # and plot
            ax1.plot(res[:,aoai], fm_ojf, 'wo', label='OJF')
            ax1.plot(aoa, fm_hawc2, 'r+-', label='HAWC2')

            ax1.set_title(tt[i])
            ax1.grid(True)
            ax1.legend(loc='best')
            ax1.set_xlabel('blade root pitch angle [deg]')

            pa4.save_fig()

###############################################################################
### OJF vs HAWC2 plots
###############################################################################

def plot_ct_vs_lambda(cao):
    """
    Comparing HAWC2 and OJF: Ct vs lambda plots.

    Parameters
    ----------

    cao : Cases object
        HAWC2 cases that will to be compared to their corresponding OJF cases.
    """

    # -------------------------------------------------------------------------
    # Massage the data in a convienent way
    # -------------------------------------------------------------------------

    # the statistics for the case
    try:
        stats_dict = cao.load_stats()
    except IOError:
        stats_dict = cao.cases_stats
    # extract an array from the stats_dict
    hawc2_tsr = np.zeros(len(stats_dict))
    hawc2_ct = np.zeros(len(stats_dict))
    runs_inc = []
    blade_rad = ojf_post.model.blade_radius
    # keep track of all the pitch angles used
    pitchs = {}
    # and subtrack all what we need to know
    ii = 0
    for cname, case in cao.cases.iteritems():

        # since the February series do not have reliable strains, ignore
        if case['[ojf_case]'].startswith('02'):
            continue

        # ojf case name is saved here
#        runs_inc.append('_'.join(case['[ojf_case]'].split('_')[0:3]))
        runs_inc.append(case['[ojf_case]'].split('_')[2])

        try:
            pitchs[case['[blade_st_group]']][case['[pitch_angle]']] = False
        except KeyError:
            pitchs[case['[blade_st_group]']] = {case['[pitch_angle]'] : False}

        # calculate the tip speed ratio
        wind = float(case['[windspeed]'])
        wr = float(case['[fix_wr]'])
        hawc2_tsr[ii] = blade_rad * wr / wind

        # and the ct
        # convert the tower FA bending moment to rotor thrust
        chis = Chi(stats_dict[cname]['ch_dict'])
        M_base = stats_dict[cname]['sig_stats'][0,2,chis.mx_tower]
        # convert from kN to N
#        thrust = 1000.0 * M_base / ojf_post.model.momemt_arm_rotor
        thrust = 1000.0 * M_base / 1.66 # which is 2cm longer
        tn = stats_dict[cname]['sig_stats'][0,2,chis.fz_nacelle]*1000.0
        ts = stats_dict[cname]['sig_stats'][0,2,chis.fz_shaft]*1000.0
        rho = 1.225
        # and normalize to get the thrust coefficient
        hawc2_ct[ii] = abs(tn) / (0.5*rho*wind*wind*ojf_post.model.A)

        print '%9.4f %9.4f %9.4f' % (thrust, ts, tn)

        ii += 1

    # each blade_st_group should only have one pitch setting
    pitch_text = []
    for key, value in pitchs.iteritems():
        if not len(value) == 1:
            raise UserWarning, 'more than one pitch angle setting selected'
        pitch_text.append( 'pitch_%s_%1.1f' % (key, value.keys()[0]) )
    pitch_text = '_'.join(pitch_text)

    # cut off any unused space
    hawc2_tsr = hawc2_tsr[:ii]
    hawc2_ct = hawc2_ct[:ii]

    # -------------------------------------------------------------------------
    # load the OJF runs that where used for the HAWC2 simulations again
    # -------------------------------------------------------------------------
    prefix = 'symlinks_all_psicor'
    db = ojfdb.ojf_db(prefix, debug=False)
    ojf, ojfcases, ojfheader = db.select(['04'], [], [], runs_inc=runs_inc)

    # -------------------------------------------------------------------------
    # plotting
    # -------------------------------------------------------------------------
    figpath = os.path.join(FIGDIR, 'ct_vs_lambda/')
    scale = 1.5
    figfile = '%s-%s-ct-vs-lambda-%s' % (prefix, cao.sim_id, pitch_text)
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.7, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(db.tsr(ojf), db.ct(ojf), 'wo', label='OJF')
    ax1.plot(hawc2_tsr, hawc2_ct, 'r+', label='HAWC2')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('Thrust coefficients for\nHAWC2 and OJF', size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('thrust coefficient $C_T$')
    ax1.grid(True)
    pa4.save_fig()

def plot_ct_vs_yawerror_vs_lambda(cao):
    """
    Comparing HAWC2 and OJF: Ct vs lambda plots, for the different yaw errors

    Parameters
    ----------

    cao : Cases object
        HAWC2 cases that will to be compared to their corresponding OJF cases.
    """

    # -------------------------------------------------------------------------
    # Massage the data in a convienent way
    # -------------------------------------------------------------------------

    # the statistics for the case
    stats_dict = cao.load_stats()
    # extract an array from the stats_dict
    hawc2_tsr = np.zeros(len(stats_dict))
    hawc2_ct = np.zeros(len(stats_dict))
    hawc2_yaw = np.zeros(len(stats_dict))
    runs_inc = []
    blade_rad = ojf_post.model.blade_radius
    # and subtrack all what we need to know
    ii = 0
    for cname, case in cao.cases.iteritems():

        # since the February series do not have reliable strains, ignore
        if case['[ojf_case]'].startswith('02'):
            continue

        # ojf case name is saved here
#        runs_inc.append('_'.join(case['[ojf_case]'].split('_')[0:3]))
        runs_inc.append(case['[ojf_case]'].split('_')[2])

        # yaw error
        hawc2_yaw[ii] = case['[yaw_angle_misalign]']

        # calculate the tip speed ratio
        wind = float(case['[windspeed]'])
        wr = float(case['[fix_wr]'])
        hawc2_tsr[ii] = blade_rad * wr / wind

        # and the ct
        # convert the tower FA bending moment to rotor thrust
        chis = Chi(stats_dict[cname]['ch_dict'])
        M_base = stats_dict[cname]['sig_stats'][0,2,chis.mx_tower]
        # convert from kN to N
#        thrust = 1000.0 * M_base / ojf_post.model.momemt_arm_rotor
        thrust = 1000.0 * M_base / 1.66 # which is 2cm longer
        tn = stats_dict[cname]['sig_stats'][0,2,chis.fz_nacelle]*1000.0
        ts = stats_dict[cname]['sig_stats'][0,2,chis.fz_shaft]*1000.0
        rho = 1.225
        # and normalize to get the thrust coefficient
        hawc2_ct[ii] = abs(tn) / (0.5*rho*wind*wind*ojf_post.model.A)

        print '%9.4f %9.4f %9.4f' % (thrust, ts, tn)

        ii += 1

    # cut off any unused space
    hawc2_tsr = hawc2_tsr[:ii]
    hawc2_ct = hawc2_ct[:ii]

    # -------------------------------------------------------------------------
    # load the OJF runs that where used for the HAWC2 simulations again
    # -------------------------------------------------------------------------
    prefix = 'symlinks_all_psicor'
    inc = ['force', '_STC_']
    db = ojfdb.ojf_db(prefix, debug=False)
    ojf, ojfcases, ojfheader = db.select(['04'], inc, [])#, runs_inc=runs_inc)
    # OJF TSR's
    ojf_tsr = db.tsr(ojf)
    iyaw = ojfheader['yaw']

    # -------------------------------------------------------------------------
    # plotting
    # -------------------------------------------------------------------------
    figpath = os.path.join(FIGDIR, 'ct_vs_lambda/')
    scale = 1.5
    figfile = '%s-%s-ct-vs-yawerror-vs-lambda' % (prefix,cao.sim_id)
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.7, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    i4 = ojf_tsr.__le__(5.0)
    i5 = ojf_tsr.__ge__(5.0)*ojf_tsr.__lt__(6.0)
    i6 = ojf_tsr.__ge__(6.0)*ojf_tsr.__lt__(7.0)
    i7 = ojf_tsr.__ge__(7.0)*ojf_tsr.__lt__(8.0)
#    i8 = ojf_tsr.__ge__(8.0)*ojf_tsr.__lt__(9.0)
#    i9 = ojf_tsr.__ge__(9.0)*ojf_tsr.__lt__(11.0)
#    ax1.plot(ojf[iyaw,i9], db.ct(ojf[:,i9]),'bo',label='$9<\lambda<10$')
#    ax1.plot(ojf[iyaw,i8], db.ct(ojf[:,i8]),'rs',label='$8<\lambda<9$')
    ax1.plot(ojf[iyaw,i7], db.ct(ojf[:,i7]),'gv',label='$7<\lambda<8$')
    ax1.plot(ojf[iyaw,i6], db.ct(ojf[:,i6]),'m<',label='$6<\lambda<7$')
    ax1.plot(ojf[iyaw,i5], db.ct(ojf[:,i5]),'c^',label='$5<\lambda<6$')
    ax1.plot(ojf[iyaw,i4], db.ct(ojf[:,i4]),'y>',label='$\lambda<5$')

    # for the cos2 fitting, the max/min ct points
    max_up = db.ct(ojf[:,i7]).max()
    max_low = db.ct(ojf[:,i4])
    max_low = np.sort(max_low)[-3]

    i4 = hawc2_tsr.__le__(5.0)
    i5 = hawc2_tsr.__ge__(5.0)*hawc2_tsr.__lt__(6.0)
    i6 = hawc2_tsr.__ge__(6.0)*hawc2_tsr.__lt__(7.0)
    i7 = hawc2_tsr.__ge__(7.0)*hawc2_tsr.__lt__(8.0)
#    i8 = hawc2_tsr.__ge__(8.0)*hawc2_tsr.__lt__(9.0)
#    i9 = hawc2_tsr.__ge__(9.0)*hawc2_tsr.__lt__(11.0)
#    ax1.plot(hawc2_yaw[i9], hawc2_ct[i9],'b+')#,label='$9<\lambda<10$')
#    ax1.plot(hawc2_yaw[i8], hawc2_ct[i8],'r+')#,label='$8<\lambda<9$')
    ax1.plot(hawc2_yaw[i7], hawc2_ct[i7],'g+')#,label='$7<\lambda<8$')
    ax1.plot(hawc2_yaw[i6], hawc2_ct[i6],'m+')#,label='$6<\lambda<7$')
    ax1.plot(hawc2_yaw[i5], hawc2_ct[i5],'c+')#,label='$5<\lambda<6$')
    ax1.plot(hawc2_yaw[i4], hawc2_ct[i4],'y+')#,label='$\lambda<5$')

    # add some cos or cos**2 fits
    angles = np.arange(-40.0, 40.0, 0.1)
    angles_rad = angles.copy() * np.pi/180.0
    max_mid = hawc2_ct[i5].max()
    cos = np.cos(angles_rad)
    ax1.plot(angles, cos*cos*max_up, 'g--')
    ax1.plot(angles, cos*cos*max_mid, 'c--')
    ax1.plot(angles, cos*cos*max_low, 'y--')

    leg = ax1.legend(loc='center', title='TSR')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('Thrust coefficients for\nHAWC2 and OJF', size=14*scale)
#    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_xlabel('yaw error $\psi$')
    ax1.set_ylabel('thrust coefficient $C_T$')
    ax1.grid(True)
    pa4.save_fig()

def plot_blade_vs_azimuth(cao, cname, db, normalize, delay, zeromean,
                          i0=0, i1=-1):
    """

    The loads will be normalized since the different offesets the signals
    both have. Normalisation is done based on the range of each signal.

    For stiff blade 410 and later
    For blade 1: the load signal is + increase direction also +
    For blade 2: the load signal is - increase direction also -

    for flex blade 410 and earlier, blade1/2 are back in line:
    load signal is - increase direction also -

    For HAWC2  : the load signal is - increase direction aslo -

    Parameters
    ----------

    cao : Simulations.Cases object
        cao object that holds at least the case defined in cname

    cname : str
        a valid case name that resides within cao.Cases

    db : ojf_db database dict
        OJF statistics database dictionary
    """

    ojfcase = cao.cases[cname]['[ojf_case]']
    sim_id = cao.cases[cname]['[sim_id]']

    if sim_id.endswith('gen'):
        if delay:
            figpath = FIGDIR + 'bladeloads_vs_azimuth_generator_delayed_BX_X/'
        else:
            if zeromean:
                figpath = FIGDIR + 'bladeloads_vs_azimuth_generator_zeromean_BX_X/'
            else:
                figpath = FIGDIR + 'bladeloads_vs_azimuth_generator_BX_X/'
    else:
        figpath = FIGDIR + 'bladeloads_vs_azimuth_fixrpm_BX_X/'

    # is this a forced yaw error case? than we need to get to the staircases

    # -------------------------------------------------------------------------

    # load the OJF result file, but strip the STC tag if present, because
    # that does not apply to the raw file name
    if ojfcase.find('_STC_') > -1:
        # load the stair case statistics
        [i0, i1] = db.db_stats[ojfcase]['STC index resampled']
        ojfcase = ''.join(ojfcase.split('_STC_')[:-1])
#    else:
#        i0, i1 = 0, -1

    res = ojfresult.ComboResults(OJFPATH, ojfcase, silent=True, cal=True,
                                 sync=True)

    # resample so we can take the correct arguments from the stair case filter
    res._resample()
    # only continue of the dspace-blade was succesfully synced!
    if not res.dspace_strain_is_synced:
        return
    # select the azimuthal channel
    ojf_azi = res.dspace.data[i0:i1,res.dspace.labels_ch['Azimuth']]
    # OJF blade bending moments, and normalize

    if zeromean:
        ojf_b1_r = res.blade.data[i0:i1,2] - res.blade.data[i0:i1,2].mean()
        ojf_b1_30 = res.blade.data[i0:i1,3] - res.blade.data[i0:i1,3].mean()
        ojf_b2_r = res.blade.data[i0:i1,0] - res.blade.data[i0:i1,0].mean()
        ojf_b2_30 = res.blade.data[i0:i1,1] - res.blade.data[i0:i1,1].mean()
    else:
        ojf_b1_r = res.blade.data[i0:i1,2]
        ojf_b1_30 = res.blade.data[i0:i1,3]
        ojf_b2_r = res.blade.data[i0:i1,0]
        ojf_b2_30 = res.blade.data[i0:i1,1]

    ojf_rpm = res.dspace.data[i0:i1,res.dspace.labels_ch['RPM']]

#    # and LATER: NO, NEGATIVE IS STILL INCREASING DIR
#    # the AVERAGE OFFSET WAS NON ZERO
#    # CAUTION: increasing load level is in negative direction.
#    # EXCEPT: for the stiff blade 1 in April, so reverse that!
#    if ojfcase.find('stiff') > -1 and ojfcase.startswith('04'):
#        ojf_b1_r  *= -1.0
#        ojf_b1_30 *= -1.0

    # -------------------------------------------------------------------------
    # load the HAWC2 results
    hawc2res = cao.load_result_file(cao.cases[cname])
    iB1_r  = hawc2res.ch_dict['blade1-blade1-node-001-momentvec-x']['chi']
    iB1_30 = hawc2res.ch_dict['blade1-blade1-node-004-momentvec-x']['chi']
    iB2_r  = hawc2res.ch_dict['blade2-blade2-node-001-momentvec-x']['chi']
    iB2_30 = hawc2res.ch_dict['blade2-blade2-node-004-momentvec-x']['chi']
    iAzi = hawc2res.ch_dict['Azi  1']['chi']
    iRPM = hawc2res.ch_dict['Omega']['chi']
    # have the same azimuth definition as for OJF:
    # first +360 so we are for sure in the 0-360 band
    # than +60 to move B3 to up=0 and have it the same as OJF. Note that we
    # do +60 because the B3 is leading with 60 to the wanted position!
    # the __ge__(360) works here because 180+360+60 = 600 < 2*360 = 720
    hawc2_azi = hawc2res.sig[:,iAzi] + 180.0 + 240.0
    # and move back to 0-360 band
    hawc2_azi[hawc2_azi.__ge__(360.0)] -= 360.0

    # and now to a more easy to read definition: 180 deg blade 1 down
    # now 60 degrees is tower shadow, we want 180 tower shadow, so we are
    # lagging 120 degrees
    hawc2_azi += 120.0
    ojf_azi   += 120.0
    hawc2_azi[hawc2_azi.__ge__(360.0)] -= 360.0
    ojf_azi[ojf_azi.__ge__(360.0)] -= 360.0

    # although the OJF is rotating vector is pointed upwind, and the HAWC2
    # rotation vector is pointed downwind, there is no need to compensate for
    # that. By reversing the direction of the yaw angle, each azimuthal
    # position should be corresponding to similar aerodynamic conditions

    def delay_time(ojf_azi):
        # add a delay to the measurements of a fixed amount of time. This will
        # result in a azimuth phase shift as function of rotor speed.
        # benchmark case: 190 deg at 657 rpm results in a time delay of:

        # ojf_azi_delay > 0: move an OJF peak to the right
        # ojf_azi_delay < 0: move an OJF peak to the left

        # take as callibration case the one with the lowest yaw error,
        # so the wake displacement due to yaw is small, and wake effect length
        # passage is minimal

        # base the offset on the case
        if ojfcase.startswith('0405_run_287'):
            ojf_azi_delay = -70.0
            timeoffset = ojf_azi_delay*np.pi/(626.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0405_run_288'):
            ojf_azi_delay = -50.0
            timeoffset = ojf_azi_delay*np.pi/(649.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0410_run_298'):
            ojf_azi_delay = 8.0
            timeoffset = ojf_azi_delay*np.pi/(70.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0410_run_299'):
            ojf_azi_delay = 10.0
            timeoffset = ojf_azi_delay*np.pi/(77.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0410_run_300'):
            ojf_azi_delay = -75.0
            timeoffset = ojf_azi_delay*np.pi/(734.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0410_run_301'):
            ojf_azi_delay = -75.0
            timeoffset = ojf_azi_delay*np.pi/(739.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0410_run_302'):
            ojf_azi_delay = -35.0
            timeoffset = ojf_azi_delay*np.pi/(639.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0410_run_303'):
            ojf_azi_delay = -35.0
            timeoffset = ojf_azi_delay*np.pi/(642.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0413_run_415'):
            ojf_azi_delay = 100.0
            timeoffset = ojf_azi_delay*np.pi/(730.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0413_run_417'):
            ojf_azi_delay = 70.0
            timeoffset = ojf_azi_delay*np.pi/(657.0*np.pi*180.0/30.0)
        elif ojfcase.startswith('0413_run_419'):
            ojf_azi_delay = 70.0
            timeoffset = ojf_azi_delay*np.pi/(620.0*np.pi*180.0/30.0)
        else:
            raise ValueError, 'which azimuth offset?'
            # the above works, but uses the wrong RPM, more correct would be
            # (but hasn't used now since this last minute noticing)
            #timeoffset = (111*np.pi/(180.0*523.0*np.pi/30.0))
        # which can be calculated with:
        #timeoffset = ojf_azi_delay*np.pi/(180.0*ojf_rpm.mean()*np.pi/30.0)
        # or explicitaly
        #110*np.pi/(180.0*523.0*np.pi/30.0)

        print 'TIME OFFSET:', timeoffset

        # and results in the following azimuth delay
        ojf_azi_delay = (timeoffset*ojf_rpm.mean()*np.pi/30.0)*180.0/np.pi

        # seems all very logical, but what am I overlooking? Something strange
        # goes wrong now here, i
#        ojf_azi += ojf_azi_delay
#        ojf_azi[ojf_azi.__ge__(360.0)] -= 360.0

        ojf_azi_corr = ojf_azi_delay + ojf_azi
        if ojf_azi_delay > 0:
            ojf_azi_corr[ojf_azi_corr.__ge__(360.0)] -= 360.0
        elif ojf_azi_delay < 0:
            ojf_azi_corr[ojf_azi_corr.__le__(0.0)] += 360.0

        return ojf_azi_corr

    if delay:
        ojf_azi_corr = delay_time(ojf_azi)
        ojf_azi = ojf_azi_corr

#        delay_time(ojf_azi)

    # reduce image size, downsample HAWC2 stuff
#    tnew = np.arange(hawc2res.sig[0,0], hawc2res.sig[-1,0], 0.01)
#    hawc2_azi  = np.interp(tnew, hawc2res.sig[:,0], hawc2_azi)
#    hawc2_B1_r  = np.interp(tnew, hawc2res.sig[:,0], hawc2res.sig[:,iB1_r])
#    hawc2_B1_30 = np.interp(tnew, hawc2res.sig[:,0], hawc2res.sig[:,iB1_30])
#    hawc2_B2_r  = np.interp(tnew, hawc2res.sig[:,0], hawc2res.sig[:,iB2_r])
#    hawc2_B2_30 = np.interp(tnew, hawc2res.sig[:,0], hawc2res.sig[:,iB2_30])
#    hawc2_rpm = np.interp(tnew, hawc2res.sig[:,0], hawc2res.sig[:,iRPM])

    # or no interpolation, reduce plot size with markevery
    # and limit the number of revolutions to 5
    sec_per_5rev = (np.pi*2.0*2.0) / cao.cases[cname]['[fix_wr]']
    samp_per_5rev = int(hawc2res.Freq * sec_per_5rev)

    hawc2_azi = hawc2_azi[-samp_per_5rev:]
    hawc2_B1_r  = hawc2res.sig[-samp_per_5rev:,iB1_r]
    hawc2_B1_30 = hawc2res.sig[-samp_per_5rev:,iB1_30]
    hawc2_B2_r  = hawc2res.sig[-samp_per_5rev:,iB2_r]
    hawc2_B2_30 = hawc2res.sig[-samp_per_5rev:,iB2_30]
    hawc2_rpm = hawc2res.sig[-samp_per_5rev:,iRPM]

    hawc2_rpm *= 30.0/np.pi

    # and normalize with the range of the OJF signal. Do not use the range
    # of HAWC2 because than we lose al feeling of proporation and comparison
    if zeromean:
        hawc2_B1_r = (hawc2_B1_r - hawc2_B1_r.mean()) * 1000.0
        hawc2_B1_30 = (hawc2_B1_30 - hawc2_B1_30.mean()) * 1000.0
        hawc2_B2_r = (hawc2_B2_r - hawc2_B2_r.mean()) * 1000.0
        hawc2_B2_30 = (hawc2_B2_30 - hawc2_B2_30.mean()) * 1000.0
    else:
        hawc2_B1_r  = hawc2_B1_r  * 1000.0
        hawc2_B1_30 = hawc2_B1_30 * 1000.0
        hawc2_B2_r  = hawc2_B2_r  * 1000.0
        hawc2_B2_30 = hawc2_B2_30 * 1000.0


    def plot_on_ax(ax, hawc2_azi, hawc2_blade, ojf_azi, ojf_blade):
        """
        These are generic for each blade plot
        """

#        # CAUTION: increasing load level is in negative direction.
#        # EXCEPT: for the stiff blade 1 in April
#        if figfile.find('stiff') > -1 and figfile.find('B1') > -1:
#            ojf_blade   *= -1.0
#        hawc2_blade *= -1.0
#        hawc2_blade += 1.0
#        ojf_blade   += 1.0

        if normalize:
            hawc2_blade /= (hawc2_blade.max() - hawc2_blade.min())
            ojf_blade   /= (ojf_blade.max() - ojf_blade.min())

        # more downscaling for OJF: see how many revolutions we have
        sec_per_rev = (np.pi*2.0) / cao.cases[cname]['[fix_wr]']
        samp_per_rev = int(res.dspace.sample_rate* sec_per_rev)
        revs = (len(ojf_azi) / samp_per_rev )
        print 'number of ojf revolutions:', revs
        # lets say a maximum of 8 revolutions
        if revs > 8:
            ss = samp_per_rev*(revs-8)
        else:
            ss = 0
        # that is too much downsampling, just keep all the data
        revs = 1.0

        # instead of downsampling, just do not mark every point in the plot
        deg_per_sec = cao.cases[cname]['[fix_wr]']*180.0/np.pi
        ojf_mark = max(round(revs*res.dspace.sample_rate / deg_per_sec, 0),1)
        h_mark = max(round(hawc2res.Freq / deg_per_sec, 0),1)

        ax.plot(ojf_azi[ss:], ojf_blade[ss:], 'wo', label='OJF',
                markevery=ojf_mark)
        ax.plot(hawc2_azi, hawc2_blade, 'r+', label='HAWC2', markevery=h_mark)
        ax.axvline(x=170, color='k', linestyle='--')
        ax.axvline(x=180, color='k')
        ax.axvline(x=190, color='k', linestyle='--')
        ax.xaxis.set_ticks(range(0,370,30))
#        ax.axvline(x=200, color='k', linestyle='--')
#        ax.axvline(x=210, color='k', linestyle='--')
#        ax.axvline(x=220, color='k', linestyle='--')

        # rotor speed ontop of it
        divider = plotting.make_axes_locatable(ax)
        height = pwy*0.2/pa4.oneinch
        ax2 = divider.append_axes("top", height, pad=0.2, sharex=ax)
        ax2.plot(ojf_azi[ss:], ojf_rpm[ss:], 'wo', markevery=ojf_mark)
        ax2.plot(hawc2_azi, hawc2_rpm, 'r+', markevery=h_mark)
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
#        ax.set_ylabel('Bending moment [Nm]')
#        ax.set_ylabel('Normalized Bending moment')
        if normalize and not zeromean:
            ax.set_ylabel('Normalized Bending moment')
        elif normalize and zeromean:
            ax.set_ylabel('Normalized Bending moment\nzero base offset')
            # for the range normalized blade load signals, limit to 0.7
            ax.set_ylim([-0.7, 0.7])
        elif not normalize and zeromean:
            ax.set_ylabel('Bending moment [Nm]\nzero base offset')
        elif not normalize and not zeromean:
            ax.set_ylabel('Bending moment [Nm]')

        ax.set_xlim([0, 360])

    def next_blade(azi):
        # change azimuth to next blade
        azi += 120.0
        azi[azi.__ge__(360.0)] -= 360.0
        return azi

    # -------------------------------------------------------------------------
    # plotting properties
    # ------------------------------------------------------------------------
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

    # figure file base name
    fnamebase = sim_id + '_' +'_'.join(ojfcase.split('_')[0:3])
    fnamebase += '_%05.02fms' % cao.cases[cname]['[windspeed]']
    fnamebase += '_%03.0frpm' % (cao.cases[cname]['[fix_wr]']*30.0/np.pi)
    fnamebase += '_y%+05.01fdeg' % (cao.cases[cname]['[yaw_angle_misalign]'])
    fnamebase += '_%s' % cao.cases[cname]['[blade_st_group]']

    # -------------------------------------------------------------------------
    # plotting blade 1 root
    # -------------------------------------------------------------------------
    # file name should relate to the HAWC2 case, since the base cases have
    # a lot of overlap: lots of measurement points in one case file
    #th_05_forcedyaw_7.95ms_s0_y0_yfix_ymis_+17.4_626rpm_sbstiff_p-2.4
    figfile  = fnamebase
    figfile += '_B1_root'
    figpath2 = figpath.replace('_BX_X/', '_B1_root/')
    try:
        os.mkdir(figpath2)
    except OSError:
        pass
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath2+figfile, grandtitle=None, nr_plots=nr, wsbottom_cm=wsb,
              wsleft_cm=wsl, wsright_cm=wsr, hspace_cm=hs, wstop_cm=wst,
              size_x_perfig=pwx, size_y_perfig=pwy)

    title = 'Blade 1 root bending moment vs azimuth position'
    title += '\n%s, %s, %s, %s' % (yawerror, rpm, wind, tsr)
    title += ' '

    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    plot_on_ax(ax1, hawc2_azi, hawc2_B1_r, ojf_azi, ojf_b1_r)

    pa4.save_fig(eps=eps)

#    plt.figure(figfile)
#    plt.plot(ojf_azi, ojf_b1_r, 'wo', label='OJF')
#    plt.plot(hawc2_azi, hawc2_B1_r*1000.0, 'r+', label='HAWC2 B1')
#    plt.axvline(x=180, color='k')
#    plt.axvline(x=190, color='k', linestyle='--')
#    plt.axvline(x=200, color='k', linestyle='--')
#    plt.axvline(x=210, color='k', linestyle='--')
#    plt.axvline(x=220, color='k', linestyle='--')
#    plt.xlim([0, 360])
##    plt.plot(hawc2_azi, hawc2res.sig[:,iB2_r]*1000.0, 'r+', label='HAWC2 B2')
#    plt.grid(True)

    # -------------------------------------------------------------------------
    # plotting blade 1 30%
    # -------------------------------------------------------------------------
    # file name should relate to the HAWC2 case, since the base cases have
    # a lot of overlap: lots of measurement points in one case file
    #th_05_forcedyaw_7.95ms_s0_y0_yfix_ymis_+17.4_626rpm_sbstiff_p-2.4
    figfile  = fnamebase
    figfile += '_B1_30'
    figpath2 = figpath.replace('_BX_X/', '_B1_30/')
    try:
        os.mkdir(figpath2)
    except OSError:
        pass
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath2+figfile, grandtitle=None, nr_plots=nr, wsbottom_cm=wsb,
              wsleft_cm=wsl, wsright_cm=wsr, hspace_cm=hs, wstop_cm=wst,
              size_x_perfig=pwx, size_y_perfig=pwy)

    title = 'Blade 1 30\% bending moment vs azimuth position'
    title += '\n%s, %s, %s, %s' % (yawerror, rpm, wind, tsr)
    title += ' '

    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    plot_on_ax(ax1, hawc2_azi, hawc2_B1_30, ojf_azi, ojf_b1_30)

    pa4.save_fig(eps=eps)

    # -------------------------------------------------------------------------
    # plotting blade 2
    # -------------------------------------------------------------------------
    # move blade 2 down at 180 degrees. blade 2 is leading 120 degrees over 1
    # so make the tower shadow appear 120 degrees ealier
    hawc2_azi = next_blade(hawc2_azi)
    ojf_azi = next_blade(ojf_azi)

    figfile  = fnamebase
    figfile += '_B2_root'
    figpath2 = figpath.replace('_BX_X/', '_B2_root/')
    try:
        os.mkdir(figpath2)
    except OSError:
        pass
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath2+figfile, grandtitle=None, nr_plots=nr, wsbottom_cm=wsb,
              wsleft_cm=wsl, wsright_cm=wsr, hspace_cm=hs, wstop_cm=wst,
              size_x_perfig=pwx, size_y_perfig=pwy)

    title = 'Blade 2 root bending moment vs azimuth position'
    title += '\n%s, %s, %s, %s' % (yawerror, rpm, wind, tsr)
    title += ' '

    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    plot_on_ax(ax1, hawc2_azi, hawc2_B2_r, ojf_azi, ojf_b2_r)

    pa4.save_fig(eps=eps)

#    plt.figure(figfile)
#    plt.plot(ojf_azi, ojf_b1_r, 'wo', label='OJF')
#    plt.plot(hawc2_azi, hawc2res.sig[:,iB1_r]*1000.0, 'r+', label='HAWC2')
#    plt.grid(True)

    # -------------------------------------------------------------------------
    # plotting blade 2 30%
    # -------------------------------------------------------------------------
    # file name should relate to the HAWC2 case, since the base cases have
    # a lot of overlap: lots of measurement points in one case file
    #th_05_forcedyaw_7.95ms_s0_y0_yfix_ymis_+17.4_626rpm_sbstiff_p-2.4
    figfile  = fnamebase
    figfile += '_B2_30'
    figpath2 = figpath.replace('_BX_X/', '_B2_30/')
    try:
        os.mkdir(figpath2)
    except OSError:
        pass
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath2+figfile, grandtitle=None, nr_plots=nr, wsbottom_cm=wsb,
              wsleft_cm=wsl, wsright_cm=wsr, hspace_cm=hs, wstop_cm=wst,
              size_x_perfig=pwx, size_y_perfig=pwy)

    title = 'Blade 2 30\% bending moment vs azimuth position'
    title += '\n%s, %s, %s, %s' % (yawerror, rpm, wind, tsr)
    title += ' '

    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    plot_on_ax(ax1, hawc2_azi, hawc2_B2_30, ojf_azi, ojf_b2_30)

    pa4.save_fig(eps=eps)

def plot_freeyaw_response(cao, cname, db, t0_ojf, t0_hawc, duration):
    """
    Compare the free yawing release response

    Parameters
    ----------

    cao : Simulations.Cases object
        cao object that holds at least the case defined in cname

    cname : str
        a valid case name that resides within cao.Cases

    db : ojf_db database dict
        OJF statistics database dictionary

    """

    ojfcase = cao.cases[cname]['[ojf_case]']
    run_id = '_'.join(ojfcase.split('_')[0:3])
    sim_id = cao.cases[cname]['[sim_id]']

    figpath = os.path.join(FIGDIR, 'freeyaw/')

    # define the plotting range for OJF and HAWC2
    t1_ojf = t0_ojf + duration
    t1_hawc = t0_hawc + duration

    # -------------------------------------------------------------------------
    # OJF results
    # -------------------------------------------------------------------------
    res = ojfresult.ComboResults(OJFPATH, ojfcase, silent=True, cal=True,
                                 sync=True)
    # resample so we can take the correct arguments from the stair case filter
    res._resample()
    # only continue of the dspace-blade was succesfully synced!
    if not res.dspace_strain_is_synced:
        return
    ojf_freq = res.dspace.sample_rate
    i0_ojf, i1_ojf = int(t0_ojf*ojf_freq), int(t1_ojf*ojf_freq)
    ojf_time = res.dspace.time[i0_ojf:i1_ojf] - t0_ojf

    ojf_azi = res.dspace.data[i0_ojf:i1_ojf,res.dspace.labels_ch['Azimuth']]
    ojf_rpm = res.dspace.data[i0_ojf:i1_ojf,res.dspace.labels_ch['RPM']]
    ojf_yaw = res.dspace.data[i0_ojf:i1_ojf,res.dspace.labels_ch['Yaw Laser']]

    # -------------------------------------------------------------------------
    # HAWC2 results
    # -------------------------------------------------------------------------
    hawc2res = cao.load_result_file(cao.cases[cname])
    chi = Chi(hawc2res.ch_dict)
    i0_hawc, i1_hawc = int(t0_hawc*hawc2res.Freq), int(t1_hawc*hawc2res.Freq)
    # in HAWC2 the yaw bearing is kept zero, but the wind yaw angle is set to
    # the yaw error angle. Also, sign on yaw is reversed because of the
    # oposite rotation direction
    yawerror = cao.cases[cname]['[yaw_angle_misalign]']
    hawc_yaw = - hawc2res.sig[i0_hawc:i1_hawc,chi.yawangle] - yawerror
    # additional 5 seconds because they are not logged
    hawc_time = hawc2res.sig[i0_hawc:i1_hawc,0] - t0_hawc - 5
    hawc_rpm = hawc2res.sig[i0_hawc:i1_hawc,chi.omega]*30.0/np.pi
    hawc_azi = hawc2res.sig[i0_hawc:i1_hawc,chi.azi]

    # -------------------------------------------------------------------------
    # SETUP PLOTTING
    # -------------------------------------------------------------------------

    scale = 1.8
    pwx = plotting.TexTemplate.pagewidth*0.99
    pwy = plotting.TexTemplate.pagewidth*0.45
    rpm_ini = cao.cases[cname]['[init_wr]' ]*30.0/np.pi
    replace = (sim_id, run_id, yawerror, rpm_ini)
    figfile = '%s_%s_freeyawyawerror_%+05.1fdeg_%03.0frpm' % replace
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=2, figsize_x=pwx, figsize_y=pwy,
                   grandtitle=False, hspace_cm=0.4, wsright_cm=0.4,
                   wstop_cm=0.6, wsbottom_cm=1.0, wsleft_cm=1.5)

    hawc_mark = hawc2res.Freq/2.0
    ojf_mark = ojf_freq/2.0

    if duration < 15:
        xticks = range(0, duration+1, 1)
    else:
        xticks = range(0, duration+1, 2)

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.set_title('Free yaw response')
    ax1.plot(ojf_time, ojf_yaw, 'ko-', mfc='w', markevery=ojf_mark, label='OJF')
    ax1.plot(hawc_time, hawc_yaw, 'r-+', markevery=hawc_mark, label='HAWC2')
    ax1.set_ylabel('yaw angle [deg]')
    plotting.mpl.artist.setp(ax1.get_xticklabels(), visible=False)
    ax1.legend(loc='best')
    ax1.set_xticks(xticks)
    ax1.set_xlim([0, duration])
    ax1.grid()

    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 2)
    ax1.plot(ojf_time, ojf_rpm, 'ko-', mfc='w', markevery=ojf_mark, label='OJF')
    ax1.plot(hawc_time, hawc_rpm, 'r-+', markevery=hawc_mark, label='HAWC2')
    ax1.set_ylabel('rotor speed [rpm]')
    ax1.set_xlabel('time [sec]')
    ax1.set_xticks(xticks)
    ax1.set_xlim([0, duration])
    ax1.grid()

    pa4.save_fig()




###############################################################################
### OJF to HAWC2
###############################################################################
# create HAWC2 parameters based on OJF cases

def ojf_to_hawc2(**kwargs):
    """
    Translate OJF simulation parameters to HAWC2 settings
    =====================================================

    The OJF selection is made using ojfdb.ojf_db.select(), and all its
    parameters are keyword arguments. Here only the OJF derivable parameters
    are set and nothing else. The master_tags have to be set elsewhere

    Otherwise this function is not general enough.

    Parameters
    ----------

    generator : default=False
        Generator switched on or off. If off, fixed RPM is applied.
        If a path is given, it will load the K value from that path and
        the filename is then the ojf case name

    yawmode : str, default='fix'
        fix, free, or control_ini. If the latter is the case, make sure to
        set the yaw_c_ref_angle.

    yaw_c_ref_angle : float, default=0
        Applicable when yawmode is control_ini.

    rpm_setter : default='all'
        Over which period the RPM should be averaged. Option 'all' (the
        average over the complete OJF case will be taken), or otherwise
        '4sec' (overage over the last 4 seconds, good for free yawing cases)

    path_db

    database

    inc

    exc

    months

    runs_inc

    valuedict

    postpath


    Returns
    -------

    opt_tags : list of dicts
        Each OJF case is defined as a dict in opt_tags.

    """

    def last_4_seconds(ojfcase):
        """
        Return the average RPM, yaw from the last 4 seconds of an OJF case
        """

        dspace = ojfresult.DspaceMatFile(matfile=OJFPATH+ojfcase, silent=True)
        ojf_freq = dspace.sample_rate
        i1 = len(dspace.time)
        # the end minus 4 seconds
        i0 = int(i1 - (4.0*ojf_freq) )
        rpm = dspace.data[i0:i1, dspace.labels_ch['RPM']].mean()
        yaw = -dspace.data[i0:i1, dspace.labels_ch['Yaw Laser']].mean()
        return rpm, yaw


    kgen = 'kgen_09_' # can also be empty string
    kgen = ''

    generator = kwargs.get('generator', False)

    path_db = kwargs.get('path_db', PATH_DB)
    database = kwargs.get('database', 'symlinks_all_psicor')
    inc = kwargs.get('inc', [])
    exc = kwargs.get('exc', [])
    months = kwargs.get('months', ['02','04'])
    runs_inc = kwargs.get('runs_inc', [])
    valuedict = kwargs.get('valuedict', {})
    postpath = kwargs.get('postpath', POSTDIR)
    rpm_setter = kwargs.get('rpm_setter', 'all')

    # load the OJF test database
    db = ojfdb.ojf_db(database, path_db=path_db)
    data, ojf_cases, header = db.select(months, inc, exc, valuedict=valuedict,
                            runs_inc=runs_inc)

    # for each OJF case, create a corresponding HAWC2 simulation with the
    # correct wind speed and fixed rotor speed, also save dc and feb/april
    opt_tags = []

    # cycle through all cases and fill in the HAWC2 parameters
    for i, ojf_case in enumerate(ojf_cases):
        # carefull, case 003 occurs twice
        if ojf_case.find('zerocal') > -1:
            continue
        # set the stiff blades
        elif ojf_case.find('stiff') > -1:
            blade_st_group = 'stiff'
        # flexible blades
        elif ojf_case.find('flexies') > -1:
            blade_st_group = 'flex'
        # TODO: implement samoerai blades?? We don't have a structural model
        else:
            print 'samoerai not implemtented: %s' % ojf_case
            continue

        dc = data[header['dc'],i]

        # TODO: switch to custom kgen for each RPM/TSR case
        # when the generator DLL is to be used
        if generator == True:
            generator_tag = True
            # load the relevant K setting for the current duty cycle dc
            if format(dc, '1.1f') == '0.0':
                K = np.loadtxt(postpath + kgen + 'ojf_gen_K_dc0_0order')
            elif format(dc, '1.1f') == '1.0':
                K = np.loadtxt(postpath + kgen + 'ojf_gen_K_dc1_0order')
            else:
                print 'no K found, ignoring: %s' % ojf_case
                continue
        # each OJF case has a unique generator K setting, and it should
        # carry the name of the OJF case. generator is now the path to where
        # all these custom generator settings are saved
        elif type(generator).__name__ == 'str':
            generator_tag = True
            fpath = generator
            # the file name is the current ojf_case
            fname = ojf_case + '.kgen'
            [windspeed, rpm, K] = np.loadtxt(fpath+fname)
        # for fixed speed cases
        else:
            K = 0
            generator_tag = False

        # in February: stiff has 0.5 degrees less pitch
        # but it also seems that the pitch of stiff is around 2.0-1.8 degrees
        # at the tip, so meaning a pitch setting of -1.2-1.0

        if blade_st_group == 'stiff' and ojf_case.startswith('02'):
            pitch_angle = ojf_post.model.p_stiff_02
        elif blade_st_group == 'flex' and ojf_case.startswith('02'):
            pitch_angle = ojf_post.model.p_flex_02
        elif blade_st_group == 'stiff' and ojf_case.startswith('04'):
            pitch_angle = ojf_post.model.p_stiff_04
        elif blade_st_group == 'flex' and ojf_case.startswith('04'):
            pitch_angle = ojf_post.model.p_flex_04
        else:
            raise ValueError, 'Can not catogarize blade! Which is this?'

        # in case we want the RPM based on the last section of the OJF result
        if rpm_setter == '4sec':
            rpm, yaw = last_4_seconds(ojf_case)
        else:
            rpm =  data[header['RPM'], i]
            yaw = -data[header['yaw'],i]

        # init_wr and fix_rpm do not conflict, generator tag separetes them
        case = {'[windspeed]' : data[header['wind'], i],
                '[generator]' : generator_tag,
                '[init_wr]'   : rpm*np.pi/30.0,
                '[fix_rpm]'   : rpm,
                '[blade_st_group]' : blade_st_group,
                '[ojf_case]'  : ojf_case,
                '[ojf_dc]'    : dc,
                '[gen_K1]'    : K/2.0,
                '[gen_K2]'    : K,
                '[ojf_header]': header,
                '[ojf_data]'  : data[:,i],
                '[yaw_angle_misalign]' : yaw,
                '[coning_angle_b1]'    : 0,
                '[coning_angle_b2]'    : 0,
                '[coning_angle_b3]'    : 0.8,
                '[pitch_angle_imbalance_b1]' :  0.0,
                '[pitch_angle_imbalance_b2]' :  0.0,
                '[pitch_angle_imbalance_b3]' :  0.5,
                '[pitch_angle]'              : pitch_angle }
        opt_tags.append(case)

    return opt_tags

def launch_ojf_to_hawc2():
    """
    th-02 : forced yaw errors, and all the same pitch: -0.8

    th_03 : forced yaw errors, updated pitch, coning imbalence, high conv crit

    th_04 : forced yaw errors, updated pitch, coning imbalence, high conv crit
        > try to reduce the number of failures, increased sampling
        > dt=0.00015 gives smooth blade root torsion moment icw 1e-5 1e-6 7e-7
        convergence criteria series

    th_05 : idem th_04, but reversed sign on yaw error
        > the forced yaw error series was build witouth syncing of blade and
        dspace...wrong RPM's etc consequently

    th_06 : with a rebuild ojf statistics database including indices to
        synced signals

    th_07 : added tower drag, new time constants, generator is on. Used for
        evaluating the near wake time constants

    th_08 : as th_06, but now with the tower drag, generator based on th_06
        results.

    th_09 : corrected tower shadow cd to 0.9

    th_10 : tower shadow to 1.1, final pitch tweaks (taken from ojf_post.model)
    """


    sim_id = 'th_10'
    runmethod = 'thyra'

    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    # from th-02, we could still see too many cases failing. What if we would
    # lower the convergence criteria, so to tricker more iterations?
    master.tags['[epsresq]'] = '1e-5' # internal-external F residual
    master.tags['[epsresd]'] = '1e-6' # increment residual
    master.tags['[epsresg]'] = '7e-7' # constraint equation residual

    master.tags['[walltime]'] = '18:00:00'
    master.tags['[auto_walltime]'] = True

    master.tags['[t0]'] = 12.0
    master.tags['[duration]'] = 8.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[dt_sim]'] = 0.00015
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[windramp]'] = False
    master.tags['[windrampabs]'] = False
    master.tags['[yawmode]'] = 'fix'
    master.tags['[induction_method]'] = 1
    master.tags['[aerocalc_method]'] = 1
    master.tags['[generator]'] = False
    master.tags['[extra_id]'] = 'forcedyaw'
    # aerodrag
    master.tags['[nr_bodies_blade]'] = 11
    master.tags['[nr_nodes_blade]'] = 12
    master.tags['[hub_lenght]'] = 0.245
    master.tags['[hub_drag]'] = True
    master.tags['[nr_nodes_hubaerodrag]'] = 20
    master.tags['[hub_cd]'] = 2.0
    master.tags['[aeset]'] = 1
    master.tags['[strain_root_el]'] = 1
    master.tags['[strain_30_el]'] = 4
    # for comparing the timings of the bem wake
    master.tags['[bemwake_nazi]'] = 32
#    master.tags['[nw_mix]'] = 0.6
#    master.tags['[nw_k3]'] =  0.0
#    master.tags['[nw_k2]'] = -0.4783
#    master.tags['[nw_k1]'] =  0.1025
#    master.tags['[nw_k0]'] =  0.6125

    iter_dict = {}
#    iter_dict['[nw_k1]'] = [0.1025, 0.08]
#    iter_dict['[nw_k2]'] = [-0.55, -0.5, -0.4783]
#    iter_dict['[nw_k3]'] = [-0.02, 0.0]

    # -----------------------------------------------------------------------
    # and the additions/changes to have it run with the OJF generator
#    master.tags['[t0]'] = 0
#    master.tags['[duration]'] = 30.0
#    master.tags['[generator]'] = True
#    master.tags['[ojf_generator_dll]'] = 'ojf_generator.dll'
#    # K0, Torque constant t<=t0
#    master.tags['[gen_K0]'] = 0
#    master.tags['[gen_t0]'] = 6.2
#    #master.tags['[gen_K1]'] = 0.02
#    master.tags['[gen_t1]'] = 8.0
#    # K2, Torque constant t1<t
#    #master.tags['[gen_K2]'] = 0.030 # Nm / (rad/s)
#    # minimum allowable output Torque [Nm]
#    master.tags['[gen_T_min]'] = 0
#    # maximum allowable output Torque [Nm]
#    master.tags['[gen_T_max]'] = 10
#    # 0=no filter, 1=first order filter, 2=2nd
#    master.tags['[gen_filt_type]'] = 0
#    master.tags['[gen_1_tau]'] = 5.0
#    master.tags['[gen_2_f0]'] = 100.0 # cut-off freq
#    # 2 filt: critical damping ratio
#    master.tags['[gen_2_ksi]'] = 0.7
#    master.tags['[init_wr]'] = master.tags['[fix_wr]']
#    master.tags['[gen_K1]'] = K/2.0
#    master.tags['[gen_K2]'] = K
#    master.tags['[case_fix_speed]'] = master.tags['[fix_wr]']
    # -----------------------------------------------------------------------

#    # CONING AND PITCH
#    master.tags['[coning_angle_b1]'] = 0
#    master.tags['[coning_angle_b2]'] = 0
#    master.tags['[coning_angle_b3]'] = 0
#    # blade pitch (positive = pitch to stall, negative = pitch to feather)
#    master.tags['[pitch_angle_imbalance_b1]'] = 0.0
#    master.tags['[pitch_angle_imbalance_b2]'] = 0.0
#    master.tags['[pitch_angle_imbalance_b3]'] = 0.0
#    master.tags['[pitch_angle]'] = -0.8

    # select forced yaw error cases, see ojfdb.plot_yawerr_vs_lambda()
    inc = ['force', '_STC_']
    exc = ['run_413'] # syncing failed for this OJF case
    opt_tags = ojf_to_hawc2(inc=inc, exc=exc, generator=False)

    # more limited selection for timings study
#    runs_inc = ['287', '298']
#    valuedict = {'yaw' : [-40, -26]}
#    opt_tags = ojf_to_hawc2(inc=inc, exc=exc, generator=False,
#                            runs_inc=runs_inc, valuedict=valuedict)

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master, ojf_post.variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)


def launch_dt_conv_crit():
    """
    Compare influence convergence criteria and sample frequency
    Relaunch one failed case of th_03
    """
    # NOTE: THIS CASE DOESN'T EXIST ANYMORE, SELECT ANOTHER ONE
    # see above in launch_ojf_to_hawc2
    ojf_case = '0413_run_413_9ms_dc0_stiffblades_freeyaw_forced_STC_646'

    sim_id = 'test_dt_conv_incr'
#    sim_id = 'test_dt_conv_constr'
    runmethod = 'gorm'

    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    # from th-02, we could still see too many cases failing. What if we would
    # lower the convergence criteria, so to tricker more iterations?
    master.tags['[epsresq]'] = '1e-5' # default=10.0
    # epsresd and epsresg seem not to be critical, they are often way lower
    master.tags['[epsresd]'] = '1e-6' # default= 1.0
    master.tags['[epsresg]'] = '7e-7' # default= 0.7

    master.tags['[t0]'] = 0.0
    master.tags['[duration]'] = 15.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[out_format]'] = 'HAWC_ASCII'
    master.tags['[ascii_buffer]'] = 10
    master.tags['[windramp]'] = False
    master.tags['[windrampabs]'] = False
    master.tags['[yawmode]'] = 'fix'
    master.tags['[induction_method]'] = 1
    master.tags['[aerocalc_method]'] = 1
    master.tags['[generator]'] = False
    master.tags['[extra_id]'] = 'testconv'
    # aerodrag
    master.tags['[nr_bodies_blade]'] = 11
    master.tags['[nr_nodes_blade]'] = 12
    master.tags['[hub_lenght]'] = 0.245
    master.tags['[hub_drag]'] = True
    master.tags['[nr_nodes_hubaerodrag]'] = 20
    master.tags['[hub_cd]'] = 2.0
    master.tags['[aeset]'] = 1
    master.tags['[strain_root_el]'] = 1
    master.tags['[strain_30_el]'] = 4
#    master.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
#    master.tags['[walltime]'] = '00:59:59'
#    master.tags['[auto_walltime]'] = False

    # select forced yaw error cases, see ojfdb.plot_yawerr_vs_lambda()
    inc = [ojf_case]
    months = ['04']
    opt_tags = ojf_to_hawc2(inc=inc, months=months)
    iter_dict = {'[epsresq]' : [1e-3, 5e-4, 1e-4, 1e-5, 5e-5],
                 '[dt_sim]'  : [0.0005, 0.0003, 0.00025, 0.0002, 0.0001]}
    # increment residual
    iter_dict = {'[epsresd]' : [1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
                 '[dt_sim]'  : [0.0005, 0.0003, 0.00025, 0.0002, 0.0001]}
#    # constraint equation residual
#    iter_dict = {'[epsresg]' : [7e-6, 7e-7, 7e-8, 7e-9, 7e-10],
#                 '[dt_sim]'  : [0.0005, 0.0003, 0.00025, 0.0002, 0.0001]}

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master, ojf_post.variable_tag_func,
                    write_htc=False, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)

def launch_benchmark_dt_nazi():
    """
    See if by upping nazi we could lower dt.
    Use the same OJF forced yaw error cases as used in dt_conv_crit
    """


    # select the base OJF case that will be used for the benchmark test
    runs_inc = ['298']
    valuedict = {'yaw' : [-40, -26]}
    opt_tags = ojf_to_hawc2(generator=False, runs_inc=runs_inc,
                            valuedict=valuedict)

    # get the HAWC2 cases up and running
    sim_id = 'test_dt_nazi'
    runmethod = 'gorm'
    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    master.tags['[walltime]'] = '18:00:00'
    master.tags['[auto_walltime]'] = False
    master.tags['[t0]'] = 0.0
    master.tags['[duration]'] = 15.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[out_format]'] = 'HAWC_ASCII'
    master.tags['[ascii_buffer]'] = 10
    master.tags['[yawmode]'] = 'fix'
    master.tags['[generator]'] = False
    master.tags['[extra_id]'] = 'test_dt_nazi'
    iter_dict = {'[bemwake_nazi]' : [16, 32, 48, 64, 80, 96],
                 '[dt_sim]'  : [0.001, 0.00075, 0.0005, 0.0002, 0.00015]}

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master, ojf_post.variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)

def launch_freeyaw_fixrpmyaw():
    """
    Derive the generator K setting for each case based on the average yaw
    error and rotor speed
    """

    sim_id = 'fr_01_fix'
    sim_id = 'fr_02_fix' # use [wyaw] instead of [yaw_angle_misalign]
    sim_id = 'fr_06_fix' # set RPM as in the last 4 seconds of the OJF case
    sim_id = 'fr_10_fix' # ts cd=1.1, final pitch updates
    runmethod = 'thyra'

    # K is than based on average RPM, yaw error of simulation

    # flex free yaw low RPM -yaw: 271, 296
    # flex free yaw low RPM +yaw: 297
    # flex free yaw high RPM: 170 (February), 261-267, 275-282
    # flex -yaw error spindown: 283
    # flex +yaw error spindown: 284
    # flex speedup +yaw error: 165
    runs_inc = ['271','296','297']
    runs_inc.extend(map(str, range(261,268)))
    runs_inc.extend(map(str, range(275,283)))
    # stiff free yaw low RPM -yaw: 400, 401, 404
    # stiff free yaw low RPM +yaw: 399, 402, 403
    # stiff free yaw high RPM: 412, 414, 418
    # stiff -yaw error spindown:
    # stiff +yaw error spindown: 411
    runs_inc.extend(map(str, range(399,405)))
    runs_inc.extend(['412','414','418'])
    opt_tags = ojf_to_hawc2(runs_inc=set(runs_inc), generator=False,
                            rpm_setter='4sec')
    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    master.tags['[walltime]'] = '18:00:00'
    master.tags['[auto_walltime]'] = True
    master.tags['[t0]'] = 12.0
    master.tags['[duration]'] = 8.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[dt_sim]'] = 0.00015
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[yawmode]'] = 'fix'
    master.tags['[generator]'] = False
    master.tags['[extra_id]'] = ''
    master.tags['[bemwake_nazi]'] = 32

    # change yaw_angle_misalign to wyaw
    for case in opt_tags:
        case['[wyaw]'] = -case['[yaw_angle_misalign]']
        case['[yaw_angle_misalign]'] = 0

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch({}, opt_tags, master, ojf_post.variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)

def launch_freeyaw():
    """
    redo the work on the free yawing cases of the Torque 2012 paper
    this is an easy starting point.

    In the first run, we use fixed speed to derive the generator K parameter
    based on the average rotor speed over the simulation.
    """

    sim_id = 'fr_02_free'
    sim_id = 'fr_03_free' # increased time to get to stable yaw error operation
    sim_id = 'fr_04_free' # yaw control study: find better gains
    sim_id = 'fr_05_free' # using improved yaw control gains
    sim_id = 'fr_06_free' # better tuned generator K
    sim_id = 'fr_07_free' # ts cd=0.9
    sim_id = 'fr_10_free' # ts cd=1.1, final pitch updates
    runmethod = 'gorm'

    # flex free yaw low RPM -yaw: 296, (271 is not a good measurement)
    # flex free yaw low RPM +yaw: 297
    # flex free yaw high RPM: 170 (February), 261-267, 275-282
    # flex -yaw error spindown: 283
    # flex +yaw error spindown: 284
    # flex speedup +yaw error: 165
    runs_inc = ['271','296','297']
    runs_inc.extend(map(str, range(261,268)))
    runs_inc.extend(map(str, range(275,283)))
    # stiff free yaw low RPM -yaw: 400, 401, 404
    # stiff free yaw low RPM +yaw: 399, 402, 403
    # stiff free yaw high RPM: 412, 414, 418
    # stiff -yaw error spindown:
    # stiff +yaw error spindown: 411
    runs_inc.extend(map(str, range(399,405)))
    runs_inc.extend(['412','414','418'])
    # and load the generator settings with the OJF case names
    genpath = POSTDIR + 'torque_constant/'
    opt_tags = ojf_to_hawc2(runs_inc=set(runs_inc), generator=genpath,
                            rpm_setter='4sec')
    # -----------------------------------------------------------------------
    # start each case with the correct init_wr and release time
    # -----------------------------------------------------------------------
    # set yawmisalign now to zero, and wyaw to the init yaw error
    # now yaw_c_ref_angle is always zero! But the yaw bearing angle zero
    # is -wyaw
    # however, that is just the same as setting yaw_angle_misalign and keeping
    # wyaw zero

    # in order to get the correct values, do a interactive plot of each case

    #res = ojfresult.ComboResults(OJFPATH, resfile, silent=True, cal=True)

    def plot_yawrpm_dynamic(case):
        """
        do some dynamic plotting to catch the yaw/rpm before release for the
        OJF case
        """
        res = ojfresult.ComboResults(OJFPATH, case, silent=True, cal=True)
        plt.figure(case)
        irpm = res.dspace.labels_ch['RPM']
        iyaw = res.dspace.labels_ch['Yaw Laser']
        plt.plot(res.dspace.time, res.dspace.data[:,irpm])
        plt.grid()
        plt.twinx()
        plt.plot(res.dspace.time, res.dspace.data[:,iyaw], 'red')
        plt.figure('yaw vs rpm: %s' % case )
        plt.plot(res.dspace.data[:,iyaw], res.dspace.data[:,irpm], 'k+')
        plt.grid()

    newtags = []
    yaw_c_tstop = 35

    # cycle through all the OJF cases that got selected, first round: figure
    # out what the yaw error and rotor speed are just before release
    for ojfcase in opt_tags:
        if ojfcase['[ojf_case]'].find('_run_261') > -1:
            # and do some dynamic plotting to catch the yaw/rpm before release
#            plot_yawrpm_dynamic(ojfcase['[ojf_case]'])
            # 545@-22.0, 467@+32.4 (OJF yaw angles)
            update = {'[init_wr]'            : 545*np.pi/30.0,
                      '[yaw_angle_misalign]' :  22.0,
                      '[yaw_c_tstop]'        :  yaw_c_tstop,
                      '[t0]'                 :   5.0,
                      '[duration]'           :  50.0}
            ojfcase.update(update) # update returns a NoneType
            # and store in the updated opt_tags list newcases
            newtags.append(copy.copy(ojfcase))

            # and make a new entry for the case in the other forced yaw
            # error direction
            ojfcase_other = copy.copy(ojfcase)
            update = {'[init_wr]'            : 467*np.pi/30.0,
                      '[yaw_angle_misalign]' : -32.4}
            ojfcase_other.update(update)
            newtags.append(copy.copy(ojfcase_other))

        elif ojfcase['[ojf_case]'].find('_run_262') > -1:
            # and do some dynamic plotting to catch the yaw/rpm before release
#            plot_yawrpm_dynamic(ojfcase['[ojf_case]'])
            # 462.5@-26.5, 397@+33.13 (OJF yaw angles, switch sign for HAWC2)
            update = {'[init_wr]'            : 462.5*np.pi/30.0,
                      '[yaw_angle_misalign]' :  26.5,
                      '[yaw_c_tstop]'        :  yaw_c_tstop,
                      '[t0]'                 :   5.0,
                      '[duration]'           :  50.0}
            ojfcase.update(update) # update returns a NoneType
            # and store in the updated opt_tags list newcases
            newtags.append(copy.copy(ojfcase))

            # and make a new entry for the case in the other forced yaw
            # error direction
            ojfcase_other = copy.copy(ojfcase)
            update = {'[init_wr]'            : 397*np.pi/30.0,
                      '[yaw_angle_misalign]' : -33.13}
            ojfcase_other.update(update)
            newtags.append(copy.copy(ojfcase_other))

        # ---------------------------------------------------------------------
        # LOW RPMs
        elif ojfcase['[ojf_case]'].find('_run_296') > -1:
            # and do some dynamic plotting to catch the yaw/rpm before release
#            plot_yawrpm_dynamic(ojfcase['[ojf_case]'])
            # 53.5@-33.4 (OJF yaw angles, switch sign for HAWC2)
            update = {'[init_wr]'            : 53.5*np.pi/30.0,
                      '[yaw_angle_misalign]' :  33.4,
                      '[yaw_c_tstop]'        :  yaw_c_tstop,
                      '[t0]'                 :   5.0,
                      '[duration]'           :  60.0}
            ojfcase.update(update) # update returns a NoneType
            # and store in the updated opt_tags list newcases
            newtags.append(copy.copy(ojfcase))

        elif ojfcase['[ojf_case]'].find('_run_297') > -1:
            # and do some dynamic plotting to catch the yaw/rpm before release
#            plot_yawrpm_dynamic(ojfcase['[ojf_case]'])
            # 60.5@+33.13 (OJF yaw angles, switch sign for HAWC2)
            update = {'[init_wr]'            :  60.5*np.pi/30.0,
                      '[yaw_angle_misalign]' : -33.13,
                      '[yaw_c_tstop]'        :  yaw_c_tstop,
                      '[t0]'                 :   5.0,
                      '[duration]'           :  60.0}
            ojfcase.update(update) # update returns a NoneType
            # and store in the updated opt_tags list newcases
            newtags.append(copy.copy(ojfcase))


    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    master.tags['[walltime]'] = '23:59:00'
    master.tags['[auto_walltime]'] = False
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[dt_sim]'] = 0.00015
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[generator]'] = False
    master.tags['[extra_id]'] = ''
    master.tags['[bemwake_nazi]'] = 32
    # yaw control target angle is always zero because of yaw_angle_misalign
    # being non zero
    master.tags['[yawmode]'] = 'control_ini'
    master.tags['[yaw_c_ref_angle]'] = 0
    master.tags['[yaw_c_gain_pro_base]'] = 0.0035 # 0.0020 default
    master.tags['[yaw_c_gain_int_base]'] = 0.0016 # 0.0014 default
    master.tags['[yaw_c_gain_dif_base]'] = 0.0025 # 0.0020 default
    iter_dict = {}
#    iter_dict['[yaw_c_gain_pro_base]'] = [0.002, 0.003] # 0.002
#    iter_dict['[yaw_c_gain_int_base]'] = [0.0014, 0.0016, 0.002] # 0.0014
#    iter_dict['[yaw_c_gain_dif_base]'] = [0.002, 0.0025, 0.003] # 0.002
    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, newtags, master, ojf_post.variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)

def relaunch_with_generator():
    """
    Relaunch a fixed RPM sim_id but now with generator on. For this method
    either a Kgen selection strategy needs to exist (dc0 0> K0 etc), or
    each case needs to have corresponding fixed RPM brother/sister

    th_05_gen : based on th_05
    th_06_gen : based on th_06
    th_08_gen : based on th_08
    th_09_gen : based on th_09
    th_10_gen : based on th_10
    """

    sim_id_new = 'th_10_gen'
    # note: the execution dir is set correct again in prepare_launch_cases()
    cao = sim.Cases(POSTDIR, 'th_10', resdir=RESDIR, loadstats=False)

    cases_new = {}

    # for each case, just activate the generator
    for case in cao.cases:

        # copy all the original tags into the new case dict in order to avoid
        # making changes on the original
        cases_new[case] = copy.copy(cao.cases[case])
        casedict = cases_new[case]

        # the master function does propagete the sim_id to some places, we
        # have to manually replace the old sim_id at those spots
        sim_id_old = casedict['[sim_id]']
        casedict['[sim_id]'] = sim_id_new
        # replace everywhere the old sim_id
        for tag, value in cases_new.iteritems():
            # replace only works for strings and since sim_id has to be a str,
            # all the other data types (arrays, floats, ints) are irrelevant
            if type(value).__name__ == 'str':
                value = value.replace(sim_id_old, sim_id_new)
                casedict[tag] = value
        # at the end, add reference to the base case. Do it here otherwise
        # the sim_id might be changed (as done above)
        casedict['[base_case]'] = case

        # load the custom generator K settings from the orignal case
        fpath = cao.cases[case]['[post_dir]'] + 'torque_constant/'
        fname = cao.cases[case]['[case_id]'] + '.kgen'
        [windspeed, rpm, K] = np.loadtxt(fpath+fname)

        # ---------------------------------------------------------------------
        # and the additions/changes
        # ---------------------------------------------------------------------
        # have it run with the OJF generator, all the rest remains
        # RPM below 300, after 15 seconds we have steady state RPM
        if cao.cases[case]['[fix_wr]'] < 300.0*np.pi/30.0:
            casedict['[t0]'] = 30.0
            casedict['[duration]'] = 8.0
        else:
            casedict['[t0]'] = 20.0
            casedict['[duration]'] = 3.0
        casedict['[generator]'] = True
        casedict['[ojf_generator_dll]'] = 'ojf_generator.dll'
        # K0, Torque constant t<=t0
        casedict['[gen_K0]'] = 0
        casedict['[gen_t0]'] = 6.2
        #casedict['[gen_K1]'] = 0.02
        casedict['[gen_t1]'] = 8.0
        # K2, Torque constant t1<t
        #casedict['[gen_K2]'] = 0.030 # Nm / (rad/s)
        # minimum allowable output Torque [Nm]
        casedict['[gen_T_min]'] = 0
        # maximum allowable output Torque [Nm]
        casedict['[gen_T_max]'] = 10
        # 0=no filter, 1=first order filter, 2=2nd
        casedict['[gen_filt_type]'] = 0 # filter zero, SO NO FILTERING
        casedict['[init_wr]'] = casedict['[fix_wr]']
        casedict['[gen_K1]'] = K/2.0
        casedict['[gen_K2]'] = K

    # and launch them all, update the variable tag fields again
    sim.prepare_launch_cases(cases_new, runmethod='gorm', verbose=False,
                             write_htc=True, check_log=False,
                             variable_tag_func=ojf_post.variable_tag_func)


def launch_towershadow_study_fix():
    """
    For a few OJF cases, let the tower shadow CD vary to see what the blade
    load changes are. Take a high RPM and low RPM cases
    """

    sim_id = 'ts_stud_fix'
    runmethod = 'thyra'

    # make a selection from the fixed yaw, flex blade, beginning of April
    runs_inc = ['204', '206', '214', '225', '226', '233']
    opt_tags = ojf_to_hawc2(runs_inc=set(runs_inc), generator=False)
    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    # use the extra ID to identify the version of the sim id
    master.tags['[extra_id]'] = '01'

    master.tags['[walltime]'] = '18:00:00'
    master.tags['[auto_walltime]'] = False
    master.tags['[t0]'] = 12.0
    master.tags['[duration]'] = 8.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[dt_sim]'] = 0.00015
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[yawmode]'] = 'fix'
    master.tags['[generator]'] = False
    master.tags['[bemwake_nazi]'] = 32
    master.tags['[tower_shadow_cd]'] = 0.8
    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch({}, opt_tags, master, ojf_post.variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)

def launch_towershadow_study():
    """
    For a few OJF cases, let the tower shadow CD vary to see what the blade
    load changes are. Take a high RPM and low RPM cases
    """

    sim_id = 'ts_stud'
    runmethod = 'thyra'

    # make a selection from the fixed yaw, flex blade, beginning of April
    runs_inc = ['204', '206', '214', '225', '226', '233']
    # and load the generator settings with the OJF case names
    genpath = POSTDIR + 'torque_constant/'
    opt_tags = ojf_to_hawc2(runs_inc=set(runs_inc), generator=genpath)
    master = ojf_post.master_tags(sim_id, runmethod=runmethod, silent=False,
                                  turbulence=False, verbose=False)
    # use the extra ID to identify the version of the sim id
    master.tags['[extra_id]'] = '01'

    master.tags['[walltime]'] = '23:00:00'
    master.tags['[auto_walltime]'] = True
    master.tags['[t0]'] = 30.0
    master.tags['[duration]'] = 8.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[dt_sim]'] = 0.00015
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[yawmode]'] = 'fix'
    master.tags['[generator]'] = True
    master.tags['[bemwake_nazi]'] = 32
#    master.tags['[tower_shadow_cd]'] = 0.5 # used so far...
    iter_dict = {'[tower_shadow_cd]' : np.arange(0.4, 1.4, 0.1)}
    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master, ojf_post.variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', silent=False, check_log=True)


def post_launch_hawc2():

    sim_id = 'th_04'
    sim_id = 'th_05'
    sim_id = 'th_05_gen'
    sim_id = 'th_06'
    sim_id = 'th_06_gen'
    sim_id = 'th_07'
    sim_id = 'fr_01_fix'
    sim_id = 'fr_02_free'
    sim_id = 'fr_02_fix'
    sim_id = 'fr_03_free'
    sim_id = 'th_08'
    sim_id = 'fr_05_free'
    sim_id = 'th_08_gen'
    sim_id = 'fr_06_fix'
    sim_id = 'fr_06_free'
    sim_id = 'ts_stud_fix'
    sim_id = 'ts_stud'
    sim_id = 'fr_07_free'
    sim_id = 'th_09'
    sim_id = 'th_09_gen'
    sim_id = 'th_10'
    sim_id = 'fr_10_fix'
    sim_id = 'fr_10_free'
    sim_id = 'th_10_gen'

    ojf_post.post_launch(sim_id, POSTDIR)
    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=False)
    # use ojf case naming for the generator
#    stats_dict = cao.calc_stats(calc_torque='ojf')
    # for the th_09, th_09_gen, use hawc2 case naming
    stats_dict = cao.calc_stats(calc_torque='hawc2')


###############################################################################
### PLOT HAWC2
###############################################################################

def plot_hawc2_dashboard():
    """
    Just plot all the dashboards
    """

    sim_id = 'ts_stud_fix'
    figpath = FIGDIR + 'dasbhoard_yawcontrol/'
#    figpath = FIGDIR

    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=False)
#    # a single case
#    cname = cao.cases.keys()[2]
#    grandtitle = cname.replace('_', '\_')
#    res = cao.load_result_file(cao.cases[cname])
#    plot = plotting.Cases(cao, Chi(cao.res.ch_dict))
#    plot.dasbhoard_yawcontrol(cname, figpath, cname, grandtitle=grandtitle)
#    # or all cases
    for cname in cao.cases.iterkeys():
        grandtitle = cname.replace('_', '\_')
        cao.load_result_file(cao.cases[cname])
        plot = plotting.Cases(cao, Chi(cao.res.ch_dict))
        plot.dasbhoard_yawcontrol(cname, figpath, cname, grandtitle=grandtitle)
#        plot.dashboard_rpm(figpath)

def compare_tower_shadow_cd():
    """
    Make a blade load vs azimuth scheme and see how the tower shadow cd changes
    everything, or not
    """

    def convert_azi(sig):
        """
        In place azimuth coversion
        """
        azi = sig[:,chis.azi]
        azi += 180.0 + 240.0
        # and move back to 0-360 band
        azi[azi.__ge__(360.0)] -= 360.0

        # and now to a more easy to read definition: 180 deg blade 1 down
        # now 60 degrees is tower shadow, we want 180 tower shadow, so we are
        # lagging 120 degrees
        azi += 120.0
        azi[azi.__ge__(360.0)] -= 360.0

    figpath = os.path.join(FIGDIR, 'ts/')

    # group per OJF case
    sim_id = 'ts_stud'
    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)
    # make a list of the OJF cases
    ojfcases = {}
    for cname, case in cao.cases.iteritems():
        ojfcases[case['[ojf_case]']] = True

    sortkey = '[tower_shadow_cd]'

    for ojf in ojfcases:
        # load the database again, selection is in place
        cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)
        cao.select(search_keyval={'[ojf_case]' : ojf})
        # sort on cd first
        cases_sorted = {}
        for casekey, caseval in cao.cases.iteritems():
            key = '%0.2f' % (caseval[sortkey])
            cases_sorted[key] = casekey

        # and only cd varies for the considered cases with the same OJF base
        # case, so take data from the first in the dict
        case = cao.cases[cao.cases.keys()[0]]
        rpm = case['[fix_rpm]']
        wind = case['[windspeed]']
        yaw = case['[yaw_angle_misalign]']

        figfile = sim_id + '_' +'_'.join(case['[ojf_case]'].split('_')[0:3])
        figfile += '_ts_cds_%+05.1fdeg_%03.0frpm_%04.1fms' % (yaw, rpm, wind)

        scale = 1.8
        pwx = plotting.TexTemplate.pagewidth*0.99
        pwy = plotting.TexTemplate.pagewidth*0.40
        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath+figfile, nr_plots=1, figsize_x=pwx, figsize_y=pwy,
                       grandtitle=False, hspace_cm=0.4, wsright_cm=0.4,
                       wstop_cm=1.0, wsbottom_cm=1.0, wsleft_cm=1.6)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        # colors for all cd's
#        col = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'k--', 'r--', 'b--', 'g--']
#        col = np.linspace(0.1, 1.0, 13)

        # limit the plot to certain cd's and give better color gradient, RGB
#        cd_sel = set(['%1.02f' % jj for jj in np.arange(0.4, 1.4, 0.3)])
        cd_sel = set(['0.40', '0.70', '1.10', '1.30'])
        xx = np.linspace(0,1,len(cases_sorted))
        col = []
        for ii in xrange(len(cases_sorted)):
            col.append((1*xx[ii],0,1*(1-xx[ii])))

        # and overplot all the signals
        for i, k in enumerate(sorted(cases_sorted)):

            # too many different cd's, only print some of them
            if k not in cd_sel:
                continue

            # indicate what we use
            if k == '1.10':
                mark = 's'
            else:
                mark = '+'

            cname = cases_sorted[k]
            print cname
            # load the HAWC2 results
            res = cao.load_result_file(cao.cases[cname])
            chis = Chi(res.ch_dict)

            # downsample, gives some annoying points outside the normal cloud
#            sigd = sp.signal.decimate(res.sig, 40, n=30, ftype='fir', axis=0)

            # interpolated downsampling: 360 positions per rotation
            # seconds per rotation
            spr = 1.0 / (res.sig[:,chis.rpm].mean()/60.0)

            # limit to two revolutions?
            irange = int( 2.0*spr/np.diff(res.sig[:,0]).mean() )
            res.sig = res.sig[0:irange,:]

            # and hence we have seconds per sample
            sps = round(spr/420.0, 4)
            tnew = np.arange(res.sig[0,0], res.sig[-1,0], sps)
            sigd = np.ndarray( (len(tnew),res.sig.shape[1]), dtype=np.float32)
            ii = chis.azi
            sigd[:,ii]  = np.interp(tnew, res.sig[:,0], res.sig[:,ii])
            ii = chis.mx_b1_30
            sigd[:,ii]  = np.interp(tnew, res.sig[:,0], res.sig[:,ii])

            # reset the azimuth definition
            convert_azi(sigd)

            # and plot
            ax1.plot(sigd[:,chis.azi], sigd[:,chis.mx_b1_30]*1000.0, mark,
                     color=col[i], label=r'$c_d = %s$' % k)

    #    leg = ax1.legend(loc='best')
        ax1.legend(bbox_to_anchor=(0.02, 1.01), ncol=4, loc='lower left')
        ax1.grid()
        ax1.set_xlim([0, 360])
        ax1.set_xlabel('blade azimuth position')
        ax1.set_ylabel('blade root bending moment [Nm]')
        ax1.xaxis.set_ticks(range(0,370,30))

        pa4.save_fig()


###############################################################################
### COMPARING
###############################################################################

def overplot(cao, sortkey, constkey, channel, iychan=0):
    """
    Overplot all cases in cao with pylab for interactive viewing
    'blade1-blade1-node-001-momentvec-x'
    """

    # sort on azim-res first
    cases_sorted = {}
    for casekey, caseval in cao.cases.iteritems():
        key = '%1.1e' % (caseval[sortkey])
        cases_sorted[key] = casekey

    # and plot for each case the dashboard
    plt.figure()
    i = 0
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'k--', 'r--', 'b--', 'g--']
    for key in sorted(cases_sorted):
        print 'start loading: %s' % key
        caseval = cao.cases[cases_sorted[key]]
        # read the HAWC2 file
        res = cao.load_result_file(caseval)
        # plotting
        label = '%1.1e' % (caseval[sortkey])
        chi = res.ch_dict[channel]['chi']
#        chi = res.ch_dict['Alfa-1-0.16']['chi']
#        plt.plot(res.sig[:,0], res.sig[:,chi]*1000.0, colors[i], label=label)
        plt.plot(res.sig[:,iychan], res.sig[:,chi], colors[i], label=label)
        i += 1
    title = '%s=%1.1e, varying %s' % (constkey, caseval[constkey], sortkey)
    plt.title(title)
    plt.grid()
    plt.legend(loc='best', title='%s' % sortkey)
    plt.show()

def overplot_all(cao):
    """
    No sorting, no nothing, just plot all in the cao
    """

    # and plot for each case the dashboard
    plt.figure()
    i = 0
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'k--', 'r--', 'b--', 'g--',
              'k--']
    for key in sorted(cao.cases):
        print 'start loading: %s' % key
        caseval = cao.cases[key]
        # read the HAWC2 file
        res = cao.load_result_file(caseval)
        tnew = np.arange(res.sig[0,0], res.sig[-1,0], 0.01)

        # plotting
#        chi = res.ch_dict['blade1-blade1-node-001-momentvec-x']['chi']
#        plt.plot(res.sig[:,0], res.sig[:,chi]*1000.0, colors[i])
#        chi = res.ch_dict['Alfa-1-0.16']['chi']
#        plt.plot(res.sig[:,0], res.sig[:,chi], colors[i])
        chi = res.ch_dict['bearing-yaw_rot-angle-deg']['chi']
        data  = np.interp(tnew, res.sig[:,0], res.sig[:,chi])
        plt.plot(tnew, data, colors[i])
        i += 1
#    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('blade root flapwise bending')
    plt.grid()
    plt.show()

def plot_wyaw_yawmisalign():
    """
    Compare the fixed rotor speed, fixed yaw error based on the yaw_misalign
    or wyaw methods
    """

    cao = sim.Cases(POSTDIR, ['fr_01_fix', 'fr_02_fix'])
    # take a random ojf case
    cname = cao.cases.keys()[0]
    ojfcase = cao.cases[cname]['[ojf_case]']
    cao.select(search_keyval={'[ojf_case]' : ojfcase})
    overplot_all(cao)

def plot_bem_timeconstants():
    """
    See how the BEM near wake constants influence the loading
    """

    # iter_dict['[nw_k1]'] = [0.1025, 0.08]
    # iter_dict['[nw_k2]'] = [-0.55, -0.5, -0.4783]
    # iter_dict['[nw_k3]'] = [-0.02, 0.0]

    cao = sim.Cases(POSTDIR, 'th_07')
    cao.change_results_dir(RESDIR)
    ojfcase = '0405_run_287_9.0ms_dc0_flexies_freeyawforced_yawerrorrange_'
    ojfcase += 'lowside_highrpm_STC_145'
    cao.select(search_keyval={'[ojf_case]':ojfcase})
    overplot_all(cao)

    # the cases with changing time constants
    cao = sim.Cases(POSTDIR, 'th_07')
    cao.change_results_dir(RESDIR)
    ojfcase = '0405_run_287_9.0ms_dc0_flexies_freeyawforced_yawerrorrange_'
    ojfcase += 'lowside_highrpm_STC_145'
    cao.select(search_keyval={'[nw_k1]' : 0.1025, '[nw_k3]':0.0,
                              '[ojf_case]':ojfcase})

    overplot(cao, '[nw_k2]', '[nw_k1]')
    print cao.cases[cao.cases.keys()[0]]['[case_id]']

    cao = sim.Cases(POSTDIR, 'th_07')
    cao.change_results_dir(RESDIR)
    ojfcase = '0405_run_287_9.0ms_dc0_flexies_freeyawforced_yawerrorrange_'
    ojfcase += 'lowside_highrpm_STC_145'
    cao.select(search_keyval={'[nw_k1]' : 0.1025, '[nw_k3]':-0.02,
                              '[ojf_case]':ojfcase})

    overplot(cao, '[nw_k2]', '[nw_k1]')

    cao = sim.Cases(POSTDIR, 'th_06_gen')
    cao.select(search_keyval={'[ojf_case]':ojfcase})
    overplot(cao, '[fix_rpm]', '[fix_rpm]')


def plot_dt_conv_crit():
    """
    Plotting the benchmarking stuff for different convergence criteria and
    sampling rates
    """

    # ------------------------------------------------------------------------
    ojf_post.post_launch('test_dt_conv_constr', POSTDIR)
    # force residual : [1e-3, 5e-4, 1e-4, 1e-5, 5e-5]
    cao = sim.Cases(POSTDIR, 'test_dt_conv')
    cao.change_results_dir(RESDIR)
    cao.select({'[epsresq]' : 0.0005})
    len(cao.cases)
    overplot(cao, '[dt_sim]', '[epsresq]')

    # for the same convergence criteria, but different sampling
    # dt_sim : [0.0005, 0.0003, 0.00025, 0.0002, 0.0001]
    cao = sim.Cases(POSTDIR, 'test_dt_conv')
    cao.change_results_dir(RESDIR)
    cao.select({'[dt_sim]' : 0.0002})
    len(cao.cases)
    overplot(cao, '[epsresq]', '[dt_sim]')

    # ------------------------------------------------------------------------
    ojf_post.post_launch('test_dt_conv_constr', POSTDIR)
    # constraint eq residual : [7e-6, 7e-7, 7e-8, 7e-9, 7e-10]
    # for the same convergence criteria, but different sampling
    cao = sim.Cases(POSTDIR, 'test_dt_conv_constr')
    cao.change_results_dir(RESDIR)
    cao.select({'[epsresg]' : 7e-9})
    len(cao.cases)
    overplot(cao, '[dt_sim]', '[epsresg]')

    # for the same convergence criteria, but different sampling
    # dt_sim : [0.0005, 0.0003, 0.00025, 0.0002, 0.0001]
    cao = sim.Cases(POSTDIR, 'test_dt_conv_constr', rem_failed=True)
    cao.change_results_dir(RESDIR)
    cao.select({'[dt_sim]' : 0.0001})
    len(cao.cases)
    overplot(cao, '[epsresg]', '[dt_sim]')

    # ------------------------------------------------------------------------
    ojf_post.post_launch('test_dt_conv_incr', POSTDIR)
    # increment residual : [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    cao = sim.Cases(POSTDIR, 'test_dt_conv_incr')
    cao.change_results_dir(RESDIR)
    cao.select({'[epsresd]' : 1e-5})
    len(cao.cases)
    overplot(cao, '[dt_sim]', '[epsresd]')

    # for the same convergence criteria, but different sampling
    # dt_sim : [0.0005, 0.0003, 0.00025, 0.0002, 0.0001]
    cao = sim.Cases(POSTDIR, 'test_dt_conv_incr', rem_failed=True)
    cao.change_results_dir(RESDIR)
    cao.select({'[dt_sim]' : 0.0002})
    len(cao.cases)
    overplot(cao, '[epsresd]', '[dt_sim]')

def plot_yawcontrol_parameters():
    """
    Compare the yaw control performance for different paramters
    """

    # iter_dict['[yaw_c_gain_pro_base]'] = [0.002, 0.003]
    # iter_dict['[yaw_c_gain_int_base]'] = [0.0014, 0.0016, 0.002]
    # iter_dict['[yaw_c_gain_dif_base]'] = [0.002, 0.0025, 0.003]

    # HIGH RPM CASE
    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.002,
                              '[yaw_c_gain_int_base]':0.0014})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_dif_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.003,
                              '[yaw_c_gain_int_base]':0.0014})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_dif_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.003,
                              '[yaw_c_gain_int_base]':0.002})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_dif_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.003,
                              '[yaw_c_gain_dif_base]':0.0025})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_int_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.003,
                              '[yaw_c_gain_dif_base]':0.003})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_int_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.003})
    overplot_all(cao)

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 22.0,
                              '[yaw_c_gain_pro_base]':0.002})
    overplot_all(cao)

    # and the winner seems to be:
    # '[yaw_c_gain_pro_base]':0.003
    # '[yaw_c_gain_dif_base]':0.003
    # '[yaw_c_gain_int_base]':0.002

    # LOW RPM CASE
    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 33.4,
                              '[yaw_c_gain_pro_base]':0.003,
                              '[yaw_c_gain_dif_base]':0.003})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_int_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 33.4,
                              '[yaw_c_gain_pro_base]':0.003,
                              '[yaw_c_gain_int_base]':0.0020})
    # overplot(cao, sortkey, constkey)
    overplot(cao, '[yaw_c_gain_dif_base]', '[yaw_c_gain_pro_base]',
             'bearing-yaw_rot-angle-deg')



    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 33.4,
                              '[yaw_c_gain_pro_base]':0.003})
    overplot_all(cao)

    cao = sim.Cases(POSTDIR, 'fr_04_free', resdir=RESDIR)
    cao.select(search_keyval={'[yaw_angle_misalign]' : 33.4,
                              '[yaw_c_gain_pro_base]':0.002})
    overplot_all(cao)


def blade_deflection():
    """
    Plot blade deflection curves
    """
    pp = os.path.join(RESDIR, 'blosd01/results/')
    ff = 'blosd01_0ms_s0_y0_sbflex_ab00'
    res = HawcPy.LoadResults(pp, ff)
    # select all the y deflection channels
    db = misc.DictDB(res.ch_dict)

    db.search({'sensortype' : 'state pos', 'component' : 'z'})
    # sort the keys and save the mean values to an array/list
    chix, zvals = [], []
    for key in sorted(db.dict_sel.keys()):
        zvals.append(-res.sig[:,db.dict_sel[key]['chi']].mean())
        chix.append(db.dict_sel[key]['chi'])

    db.search({'sensortype' : 'state pos', 'component' : 'y'})
    # sort the keys and save the mean values to an array/list
    chiy, yvals = [], []
    for key in sorted(db.dict_sel.keys()):
        yvals.append(res.sig[:,db.dict_sel[key]['chi']].mean())
        chiy.append(db.dict_sel[key]['chi'])

    # and now plot the magic!
    return zvals, yvals

def blade_deflection2():
    """
    Plot blade deflection curves
    """

    figpath = os.oath.join(FIGDIR, 'static_blade_deflection/')
    sim_id = 'blosd01'
    scenario = 'blade_deflection'

    cc = sim.Cases(POSTDIR, sim_id, resdir=RESDIR)
    cc.printall(scenario, figpath=figpath)

def compare_blade_deflection():

    ss=static_blade_deflection()

    # map the tests to the simulation file names and select what to include
    # in the plot
    blademap = dict()
    blademap['stiff_B1'] = 12
    blademap['stiff_B2'] = 13
    blademap['stiff_B3'] = 14
    ss.plot_and_compare('blosd02',blademap,[106, 606, 706],ploterror=False)

    blademap = dict()
    blademap['stiff_B1'] = 12
    ss.plot_and_compare('blosd02', blademap, [106, 606, 706])

    blademap = dict()
    blademap['stiff_B2'] = 13
    ss.plot_and_compare('blosd02', blademap, [106, 606, 706])

    blademap = dict()
    blademap['stiff_B3'] = 14
    ss.plot_and_compare('blosd02', blademap, [106, 606, 706])

    # map the tests to the simulation file names and select what to include
    # in the plot
    blademap = dict()
    blademap['flex_B1'] = 15
    blademap['flex_B2'] = 16
    blademap['flex_B3'] = 17
    ss.plot_and_compare('blosd01',blademap,[56,156,256,306], ploterror=False)

    blademap = dict()
    blademap['flex_B1'] = 15
    ss.plot_and_compare('blosd01', blademap, [56, 156, 256, 306])

    blademap = dict()
    blademap['flex_B2'] = 16
    ss.plot_and_compare('blosd01', blademap, [56, 156, 256, 306])

    blademap = dict()
    blademap['flex_B3'] = 17
    ss.plot_and_compare('blosd01', blademap, [56, 156, 256, 306])

def steady_fixyaw_comparisons():
    """
    Comparing Ct vs lambda plots, HAWC2 simulations taken from kgen_19, see
    how the pith angle influences everything
    """
    sim_id = 'kgen_19'
    sim_id = 'kgen_20'
    sim_id = ['kgen_20', 'kgen_20b']
    # -------------------------------------------------------------------------
    # Select the HAWC2 cases
    # -------------------------------------------------------------------------

    # load the HAWC2 simulations with OJF wind speed and RPM
    # but do this for each pitch angle, because they are all present in kgen_16

    cao_all = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)
    search = {'[pitch_angle]':ojf_post.model.p_stiff_04,
              '[blade_st_group]':'stiff', '[extra_id]':set(['04_dc0'])}
    cao_all.select(search_keyval=search)

    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=False)
    search = {'[pitch_angle]':ojf_post.model.p_stiff_04,
              '[blade_st_group]':'stiff', '[extra_id]':set(['04_dc1'])}
    cao.select(search_keyval=search)
    cao_all.cases.update(cao.cases)
#    cao_all.cases_stats.update(cao.cases_stats)

    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=False)
    search = {'[pitch_angle]':ojf_post.model.p_flex_04,
              '[blade_st_group]':'flex', '[extra_id]':set(['04_dc0'])}
    cao.select(search_keyval=search)
    cao_all.cases.update(cao.cases)
#    cao_all.cases_stats.update(cao.cases_stats)

    # flex, April, dc1, for given pitch angle
    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=False)
    search = {'[pitch_angle]':ojf_post.model.p_flex_04,
              '[blade_st_group]':'flex', '[extra_id]':set(['04_dc1'])}
    cao.select(search_keyval=search)
    cao_all.cases.update(cao.cases)
#    cao_all.cases_stats.update(cao.cases_stats)

    plot_ct_vs_lambda(cao_all)

def steady_yawerror_comparison():
    """
    Comparing the forced yaw error series
    """

    sim_id = 'th_08'
    sim_id = 'th_09'
    sim_id = 'th_10'

    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)
    plot_ct_vs_yawerror_vs_lambda(cao)

def fixyawerror_bladeload_azimuths():
    """
    For the forced fixed yaw cases, plot the azimuthal dependency of the
    blade flapwise loads. This is on a per case basis.
    """

    # the HAWC2 cases
    sim_id = 'th_06'
    sim_id = 'th_06_gen'
    sim_id = 'th_08_gen'
    sim_id = 'th_09_gen'
#    sim_id = 'th_10_gen'
    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)

    # the OJF statistic database
    db = ojfdb.ojf_db('symlinks_all_psicor')

    # some random cases
#    cname = cao.cases.keys()[0]
#    cname = 'th_04_forcedyaw_6666hz_1e-5_1e-6_7e-7_7.48ms_s0_y0_yfix_'
#    cname += 'ymis_+19.1_84rpm_sbflex_p-2.4.htc'
#    plot_blade_vs_azimuth(cao, cname, db)
#    plt.figure()
#    cname = 'th_04_forcedyaw_6666hz_1e-5_1e-6_7e-7_7.48ms_s0_y0_yfix_'
#    cname =+ 'ymis_-4.6_90rpm_sbflex_p-2.4.htc'
#    plot_blade_vs_azimuth(cao, cname, db)

    # we're on a plotting spree, spit them out, all of them!
    for cname in cao.cases:
        print
        print cao.cases[cname]['[ojf_case]']
        norm = False
        delay = False
        zeromean = False
        plot_blade_vs_azimuth(cao, cname, db, norm, delay, zeromean)

    for cname in cao.cases:
#        target = '0410_run'
#        target = '0405_run_287'
#        target = '0405_run_288'
#        target = '0410_run_299_7.5ms'
#        target = '0410_run_300'
#        target = '0410_run_301'
#        target = '0410_run_302'
#        target = '0410_run_303'
#        target = '0413_run_415'
#        target = '0413_run_417'
#        target = '0413_run_419'

        # low RPM tune case
#        target = 'freeyawforced_yawerrorrange_fastside_lowrpm_STC_1194'
        # high RPM tune case
#        target = 'yawerrorrange_slowside_highrpm_STC_441'
        try:
            if not cao.cases[cname]['[ojf_case]'].find(target) > -1:
                continue
        except NameError:
            pass
        print
        print cao.cases[cname]['[ojf_case]']
        norm = True
        delay = True
        zeromean = True
        plot_blade_vs_azimuth(cao, cname, db, norm, delay, zeromean)

# TODO: select cases and indices
def fixyawerror2_bladeload_azimuths():
    """
    Now do not select from the STC cases (forced free yaw playing), but from
    the free yawing with only a few yaw errors. In those cases we have had
    reached steady conditions longer and we have measured longer.

    However, we will have to select the cases and their start:stop indices
    manually.
    """

    pass

def freeyaw_comparison():
    """
    Compare some free yawing cases between HAWC2 and OJF
    HAWC2s are available under sim_id = 'fr_05_free'
    """

    # the HAWC2 cases
    sim_id = 'fr_06_free'
    sim_id = 'fr_07_free' # ts cd=0.9
    sim_id = 'fr_10_free' # ts cd=1.1
    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)

    # the OJF statistic database
    db = ojfdb.ojf_db('symlinks_all_psicor')

    # mappings for sim_id = 'fr_05_free'
    # because we did not include the OJF case name in the sim_id,
    # we printed it here manually in order to be able to figure out at which
    # point the comparison should overlap
    # map OJF cases with the correct timings
    mp = {'fr_05_free__7.00ms_s0_ojfgen0.056_rpmini060_sbflex.htc' : [29, 20],
          'fr_05_free__7.00ms_s0_ojfgen0.036_rpmini467_sbflex.htc' : [45, 10],
          'fr_05_free__7.00ms_s0_ojfgen0.059_rpmini054_sbflex.htc' : [26, 20],
          'fr_05_free__7.00ms_s0_ojfgen0.036_rpmini545_sbflex.htc' : [21.9,12],
          'fr_05_free__7.01ms_s0_ojfgen0.050_rpmini397_sbflex.htc' : [53, 10],
          'fr_05_free__7.01ms_s0_ojfgen0.050_rpmini462_sbflex.htc' : [27, 13]}

    # mappings for fr_06_free or others with proper OJF runid reference
    mp = {'fr_06_free__0410_run_297_7.00ms_s0_y+00.0_ycon+00.0_ymis_-33.1'+\
          '_ojfgen0.052_rpmini060_sbflex_p-2.4.htc' : [29, 30],
          'fr_06_free__0405_run_261_7.00ms_s0_y+00.0_ycon+00.0_ymis_-32.4'+\
          '_ojfgen0.033_rpmini467_sbflex_p-2.4.htc': [45.2, 17],#
          'fr_06_free__0410_run_296_7.00ms_s0_y+00.0_ycon+00.0_ymis_+33.4'+\
          '_ojfgen0.061_rpmini054_sbflex_p-2.4.htc' : [25.8, 30],
          'fr_06_free__0405_run_261_7.00ms_s0_y+00.0_ycon+00.0_ymis_+22.0'+\
          '_ojfgen0.033_rpmini545_sbflex_p-2.4.htc': [21.9,13],#
          'fr_06_free__0405_run_262_7.01ms_s0_y+00.0_ycon+00.0_ymis_-33.1'+\
          '_ojfgen0.042_rpmini397_sbflex_p-2.4.htc': [53, 10],
          'fr_06_free__0405_run_262_7.01ms_s0_y+00.0_ycon+00.0_ymis_+26.5'+\
          '_ojfgen0.042_rpmini462_sbflex_p-2.4.htc': [27, 15]}

    # mappings for fr_07_free, notice how ts changes the generator K specs!
    mp = {'fr_07_free__0410_run_297_7.00ms_s0_y+00.0_ycon+00.0_ymis_-33.1'+\
          '_ojfgen0.050_rpmini060_sbflex_p-2.4_cd0.9.htc' : [29, 30],
          'fr_07_free__0405_run_261_7.00ms_s0_y+00.0_ycon+00.0_ymis_-32.4'+\
          '_ojfgen0.032_rpmini467_sbflex_p-2.4_cd0.9.htc': [45.2, 17],#
          'fr_07_free__0410_run_296_7.00ms_s0_y+00.0_ycon+00.0_ymis_+33.4'+\
          '_ojfgen0.058_rpmini054_sbflex_p-2.4_cd0.9.htc' : [25.8, 30],
          'fr_07_free__0405_run_261_7.00ms_s0_y+00.0_ycon+00.0_ymis_+22.0'+\
          '_ojfgen0.032_rpmini545_sbflex_p-2.4_cd0.9.htc': [21.9,13],#
          'fr_07_free__0405_run_262_7.01ms_s0_y+00.0_ycon+00.0_ymis_-33.1'+\
          '_ojfgen0.040_rpmini397_sbflex_p-2.4_cd0.9.htc': [53, 10],
          'fr_07_free__0405_run_262_7.01ms_s0_y+00.0_ycon+00.0_ymis_+26.5'+\
          '_ojfgen0.040_rpmini462_sbflex_p-2.4_cd0.9.htc': [27, 15]}

    # mappings for fr_10_free, notice how ts changes the generator K specs!
    mp = {'fr_10_free__0410_run_297_7.00ms_s0_y+00.0_ycon+00.0_ymis_-33.1'+\
          '_ojfgen0.048_rpmini060_sbflex_p-2.0_cd1.1.htc' : [29, 30],
          'fr_10_free__0405_run_261_7.00ms_s0_y+00.0_ycon+00.0_ymis_-32.4'+\
          '_ojfgen0.030_rpmini467_sbflex_p-2.0_cd1.1.htc': [45.2, 17],#
          'fr_10_free__0410_run_296_7.00ms_s0_y+00.0_ycon+00.0_ymis_+33.4'+\
          '_ojfgen0.056_rpmini054_sbflex_p-2.0_cd1.1.htc' : [25.8, 30],
          'fr_10_free__0405_run_261_7.00ms_s0_y+00.0_ycon+00.0_ymis_+22.0'+\
          '_ojfgen0.030_rpmini545_sbflex_p-2.0_cd1.1.htc': [21.9,13],#
          'fr_10_free__0405_run_262_7.01ms_s0_y+00.0_ycon+00.0_ymis_-33.1'+\
          '_ojfgen0.039_rpmini397_sbflex_p-2.0_cd1.1.htc': [53, 10],
          'fr_10_free__0405_run_262_7.01ms_s0_y+00.0_ycon+00.0_ymis_+26.5'+\
          '_ojfgen0.039_rpmini462_sbflex_p-2.0_cd1.1.htc': [27, 15]}

    # and they map to these cases
    # 0410_run_297_7ms_dc0_flexies_freeyaw_lowrpm
    # 0405_run_261_7.0ms_dc0_flexies_freeyaw_highrpm
    # 0410_run_296_7ms_dc0_flexies_freeyaw_lowrpm
    # 0405_run_261_7.0ms_dc0_flexies_freeyaw_highrpm
    # 0405_run_262_7.0ms_dc0.4_flexies_freeyaw_highrpm
    # 0405_run_262_7.0ms_dc0.4_flexies_freeyaw_highrpm

    # we're on a plotting spree, spit them out, all of them!
    for cname in cao.cases:
#        ojfcase = cao.cases[cname]['[ojf_case]']
#        print ojfcase
        print cname
        plot_freeyaw_response(cao, cname, db, mp[cname][0], 28, mp[cname][1])


def waketimecnst():

    k3 =  0.0
    k2 = -0.4783
    k1 =  0.1025
    k0 =  0.6125
    rR = np.linspace(0,1,num=50)
    plt.plot(rR, k3*rR**3 + k2*rR**2 + k1*rR + k0, 'r', label='default')

    k3 =  0.00
    k2 = -0.50
    k1 =  0.1025
    k0 =  0.6125
    plt.plot(rR, k3*rR**3 + k2*rR**2 + k1*rR + k0, 'g', label='mid1')

    k3 = -0.02
    k2 = -0.50
    k1 =  0.1025
    k0 =  0.6125
    plt.plot(rR, k3*rR**3 + k2*rR**2 + k1*rR + k0, 'y', label='mid2')

    k3 = -0.02
    k2 = -0.55
    k1 =  0.08
    k0 =  0.6125
    plt.plot(rR, k3*rR**3 + k2*rR**2 + k1*rR + k0, 'b', label='lowest')

    plt.legend()
    plt.grid()

    # wake induction glaurt, p 103, f 3.108 of WEH
    # psi=0 -> blade upwards
    r = 1.0
    R = 1.0
    psi = np.linspace(0,2*np.pi)
    u0 = 1.0
    K = 1.0
    u = u0*(1.0 + ( K*r/R*np.sin(psi) ) )
    plt.figure()
    plt.plot(psi*180.0/np.pi, u)
    plt.grid()
    plt.xlim([0, 360])

    # what effect has a change in induction on velocity/aoa?

if __name__ == '__main__':

    dummy = False
    db = ojfdb.ojf_db('symlinks_all_psicor')

    # ------------------------------------------------------------------------
    # LAUNCHES
    # ------------------------------------------------------------------------
#    launch_ojf_to_hawc2()
#    relaunch_with_generator()
#    launch_freeyaw_fixrpmyaw()
#    launch_freeyaw()
#    launch_towershadow_study_fix()
#    launch_towershadow_study()
    # ------------------------------------------------------------------------
    # POST LAUNCH
    # ------------------------------------------------------------------------
#    post_launch_hawc2()

#    launch_dt_conv_crit()
#    launch_benchmark_dt_nazi()
#    ojf_post.post_launch('test_dt_nazi', POSTDIR)

    # ------------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------------

#    plot_hawc2_dashboard()

#    compare_blade_deflection()
#    bao = blade_aero_only()
#    bao.plot_compare(15)
#    bao.plot_compare(25)

#    steady_fixyaw_comparisons()
#    steady_yawerror_comparison()

#    plot_yawcontrol_parameters()

#    fixyawerror_bladeload_azimuths()

#    freeyaw_comparison()

#    compare_tower_shadow_cd()

    # ------------------------------------------------------------------------
    # random work, examples
    # ------------------------------------------------------------------------

#    sim_id = 'th_04'
#    cao = sim.Cases(POSTDIR, sim_id, resdir=RESDIR, loadstats=True)
#    db = ojfdb.ojf_db('symlinks_all_psicor')
#    cname = cao.cases.keys()[0]
#    ojfcase = cao.cases[cname]['[ojf_case]']
#    ojfcase = ''.join(ojfcase.split('_STC_')[:-1])
#    res = ojfresult.ComboResults(OJFPATH, ojfcase, silent=True, cal=True,
#                                 sync=True, checkplot=True)
#    res._resample()
#    res.plot_azi_vs_bladeload()
#    plot_blade_vs_azimuth(cao, cname, db)



    # comparing effect of the BEM time constants
#    plot_bem_timeconstants()
#    cao = sim.Cases(POSTDIR, 'th_06_gen', resdir=RESDIR, loadstats=True)
#    for cname in cao.cases:
#        plot_blade_vs_azimuth(cao, cname, db)
