# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:01:56 2013

Select all the OJF free yaw cases, manually or automatically only take those
time ranges so we can actually calculate how fast the yaw angle is recovering

@author: dave
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

# built in modules
import os
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

import ojfdb
import ojfresult
import plotting
import filters
from staircase import StairCase, StairCaseNG
#import misc

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=11)
plt.rc('legend', numpoints=1)
plt.rc('legend', borderaxespad=0)
mpl.rcParams['text.latex.unicode'] = True

PATH_DB = 'database/'
PATH_DATA_CAL_HDF = 'data/calibrated/DataFrame/'

###############################################################################
### YAW ERRORS, OLD
###############################################################################

def results_filtering(cr, figpath, fig_descr, arg_stair):
    """
    Like the ComboResults.dashboard_a3, but now include the filtered signal
    and the subtracted values.

    Saves the datastair data, it is formatted as: ndarray(stair_nr, data_chan)
    [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, windspeed, temperature]
    Note that this is actually not necessary and rendundant, since we properly
    added all the cases as subecases to the MeasureDb database

    Parameters
    ----------

    cr : ComboResults instance
        A ojfresult.ComboResults instance

    figpath : str

    fig_descr : str
        small adition to the file name

    arg_stair : array(2,n)
        Start/stop indices to n stairs
    """

    try:
        bmax = cr.blade.time.max()
    # if there are no blade data measurements, there is no such thing
    except AttributeError:
        bmax = cr.dspace.time.max()
    window_dspace, window_blade = [0, bmax], [0, bmax]

    # save each stair and corresponding data points
    # indices: [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, windspeed, temp]
    datastair = np.ndarray( (arg_stair.shape[1],10) )
    datastair[:,:] = np.nan
    # and fill in the same wind speed for all the stairs
    ch_t = 1
    ch_w = 4
    try:
        datastair[:,8] = cr.dspace.ojf.data[:,ch_w].mean()
        datastair[:,9] = cr.dspace.ojf.data[:,ch_t].mean()
    except AttributeError:
        pass

    if cr.dspace_is_cal:
        # convert labels for calibrated signals
        ylabels = {}
        fa_key = 'Tower Strain For-Aft'
        ss_key = 'Tower Strain Side-Side'
        if cr.dspace_yawcal:
            ylabels['Yaw Laser'] = '[deg]'
        if cr.dspace_towercal:
            ylabels[fa_key] = 'Tower base FA moment [Nm]'
            ylabels[ss_key] = 'Tower base SS moment [Nm]'
        if cr.dspace_towercal_psicor:
            ylabels[fa_key] = 'Tower base $\psi$ moment [Nm]'
            ylabels[ss_key] = 'Tower base $\psi_{90}$ moment [Nm]'

    # -------------------------------------------------
    # setup the figure
    # -------------------------------------------------
    nr_plots = 8

    pa4 = plotting.A4Tuned()
    # escape any underscores in the file name for latex printing
    grandtitle = cr.resfile.replace('_', '\_')
    pa4.setup(figpath + cr.resfile + fig_descr, nr_plots=nr_plots,
               grandtitle=grandtitle, wsleft_cm=2., wsright_cm=1.0,
               hspace_cm=2., wspace_cm=4.2)

    # -------------------------------------------------
    # data selection from dSPACE
    # -------------------------------------------------
    channels = []

    # make sure the order here is the same as for datastair!
    channels.append(cr.dspace.labels_ch['RPM'])
    channels.append(cr.dspace.labels_ch['Yaw Laser'])
    try:
        channels.append(cr.dspace.labels_ch['Tower Strain For-Aft'])
        channels.append(cr.dspace.labels_ch['Tower Strain Side-Side'])
    except KeyError:
        key = 'Tower Strain For-Aft filtered'
        channels.append(cr.dspace.labels_ch[key])
        key = 'Tower Strain Side-Side filtered'
        channels.append(cr.dspace.labels_ch[key])

    # -------------------------------------------------
    # the arguments ready to go as slices and SAVE IN DB
    # -------------------------------------------------

    # and plot each stair! See if RPM and yaw are realy constant!
    ppp = plotting.A4Tuned()
    # escape any underscores in the file name for latex printing
    grandtitle = cr.resfile.replace('_', '\_')
    target = figpath + cr.resfile + fig_descr + '_stairs_rawdata'
    ppp.setup(target, nr_plots=arg_stair.shape[1],
               grandtitle=grandtitle, wsleft_cm=1.6, wsright_cm=1.6,
               hspace_cm=2.5, wspace_cm=7.0)

    isel = []
    for k in range(arg_stair.shape[1]):
        i1 = arg_stair[0,k]
        i2 = arg_stair[1,k]
        isel.append(np.r_[i1:i2])
        # calculate all the means, std, min, max and range for each channel
        cr.statistics(i1=i1, i2=i2)

        # quickly check if the average RPM and actuall measurements are close
        # also assess how messy the yaw angle is bumping around
        axx = ppp.fig.add_subplot(ppp.nr_rows, ppp.nr_cols, k+1)
        irpm = cr.dspace.labels_ch['RPM']
        iyaw = cr.dspace.labels_ch['Yaw Laser']
        # give same title as the key in the OJF db_stats database
        title = '%s_STC_%i' % (cr.resfile, i1/100)
        axx.set_title(title.replace('_','\_'))
        axx.plot(cr.dspace.time[i1:i2], cr.dspace.data[i1:i2,irpm], 'b',
                 label='rpm')
        axx2 = axx.twinx()
        axx2.plot(cr.dspace.time[i1:i2], cr.dspace.data[i1:i2,iyaw], 'r',
                  label='yaw')
        axx.axhline(y=cr.stats['dspace mean'][cr.dspace.labels_ch['RPM']])
        axx.grid()
#        lines = axx.lines + axx2.lines
#        labels = [l.get_label() for l in lines]
#        leg = axx.legend(lines, labels, loc='best')
        axx.set_ylabel('RPM')
        axx2.set_ylabel('Yaw angle')

    ppp.save_fig()

    # -------------------------------------------------
    # plotting dSPACE selection
    # -------------------------------------------------
    filt = filters.Filters()
    # sample rate
    sps = int(cr.dspace.sample_rate)
    plot_nr = 1
    for ii, ch in enumerate(channels):
        time, data = cr.dspace.time, cr.dspace.data[:,ch]
        # filter the signal
        dataf, N, delay = filt.fir(time, data, ripple_db=20, sample_rate=sps,
                                   freq_trans_width=2.5, cutoff_hz=5)

        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax1.plot(time, data, 'b')
        ax1.plot(time[N-1:]-delay, dataf[N-1:], 'r')

        # and mark all the average data
        for k in range(arg_stair.shape[1]):
            ax1.plot(time[isel[k]].mean(), data[isel[k]].mean(), 'ko')
            # save the mean data from the stair
            # indices: [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, V, temp]
            datastair[k,ii] = data[isel[k]].mean()

        ax1.set_title(cr.dspace.labels[ch])
        ax1.grid(True)

        ch_key = cr.dspace.labels[ch]
        if ylabels.has_key(ch_key):# and caldict_dspace.has_key(ch_key):
            ax1.set_ylabel(ylabels[cr.dspace.labels[ch]])

        ax1.set_xlim(window_dspace)

        plot_nr += 1

    # -------------------------------------------------
    # plotting Blade selection
    # -------------------------------------------------
    # Blade 1: channels 3 (M1 root) and 4 (M2 mid section)
    # Blade 2: channels 1 (M1 root) and 2 (M2 mid section)
    try:
        time = cr.blade.time
        data = cr.blade.data

        # -----------------------
        # Blade 1 root
        # -----------------------
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax1.plot(cr.blade.time, cr.blade.data[:,2], 'b', label='root')
        # filtered channel
        dataf, N, delay = filt.fir(time, data[:,2], ripple_db=20, cutoff_hz=5,
                                   freq_trans_width=2.5, sample_rate=sps)
        ax1.plot(time[N-1:]-delay, dataf[N-1:], 'r', label='filtered')
        # and the staircase mean data points
        for k in range(arg_stair.shape[1]):
            ax1.plot(time[isel[k]].mean(), data[isel[k],2].mean(), 'ko')
            # and save the data stair
            # indices: [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, V, temp]
            datastair[k,4] = data[isel[k],1].mean()
        ax1.set_title('blade 1 strain root')
        ax1.grid(True)
        ax1.set_xlim(window_blade)
        if cr.blade_is_cal:
            ax1.set_ylabel('flapwise bending moment [Nm]')
        plot_nr += 1

        # -----------------------
        # blade 2 root
        # -----------------------
        ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax2.plot(cr.blade.time, cr.blade.data[:,0], 'b', label='30\%')
        # filtered channel
        dataf, N, delay = filt.fir(time, data[:,0], ripple_db=20, cutoff_hz=5,
                                   freq_trans_width=2.5, sample_rate=sps)
        ax2.plot(time[N-1:]-delay, dataf[N-1:], 'r', label='filtered')
        # and the staircase mean data points
        for k in range(arg_stair.shape[1]):
            ax2.plot(time[isel[k]].mean(), data[isel[k],0].mean(), 'ko')
            # and save the data stair
            # indices: [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, V, temp]
            datastair[k,6] = data[isel[k],0].mean()
        ax2.set_title('blade 2 strain root')
        ax2.grid(True)
        ax2.set_xlim(window_blade)
        if cr.blade_is_cal:
            ax2.set_ylabel('flapwise bending moment [Nm]')
        plot_nr += 1

        # -----------------------
        # Blade 1 30%
        # -----------------------
        ax3 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax3.plot(cr.blade.time, cr.blade.data[:,3], 'b', label='root')
        # filtered channel
        dataf, N, delay = filt.fir(time, data[:,3], ripple_db=20, cutoff_hz=5,
                                   freq_trans_width=2.5, sample_rate=sps)
        ax3.plot(time[N-1:]-delay, dataf[N-1:], 'r', label='filtered')
        # and the staircase mean data points
        for k in range(arg_stair.shape[1]):
            ax3.plot(time[isel[k]].mean(), data[isel[k],3].mean(), 'ko')
            # and save the data stair
            # indices: [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, V, temp]
            datastair[k,5] = data[isel[k],3].mean()
        ax3.set_title('blade 1 strain 30\%')
        ax3.grid(True)
        ax3.set_xlim(window_blade)
        if cr.blade_is_cal:
            ax3.set_ylabel('flapwise bending moment [Nm]')
        plot_nr += 1

        # -----------------------
        # Blade 2 30%
        # -----------------------
        ax4 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plot_nr)
        ax4.plot(cr.blade.time, cr.blade.data[:,1], 'b', label='root')
        # filtered channel
        dataf, N, delay = filt.fir(time, data[:,1], ripple_db=20, cutoff_hz=5,
                                   freq_trans_width=2.5, sample_rate=sps)
        ax4.plot(time[N-1:]-delay, dataf[N-1:], 'r', label='filtered')
        # and the staircase mean data points
        for k in range(arg_stair.shape[1]):
            ax4.plot(time[isel[k]].mean(), data[isel[k],1].mean(), 'ko')
            # and save the data stair
            # indices: [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, V, temp]
            datastair[k,7] = data[isel[k],1].mean()
        ax4.set_title('blade 2 strain 30\%')
        ax4.grid(True)
        ax4.set_xlim(window_blade)
        if cr.blade_is_cal:
            ax4.set_ylabel('flapwise bending moment [Nm]')
        plot_nr += 1

        # set the same y range on both axis
        lims1 = ax1.get_ylim()
        lims2 = ax2.get_ylim()
        lims3 = ax3.get_ylim()
        lims4 = ax4.get_ylim()
        ymin = min([lims1[0], lims2[0], lims3[0], lims4[0]])
        ymax = max([lims1[1], lims2[1], lims3[1], lims4[1]])
        ax1.set_ylim([ymin, ymax])
        ax2.set_ylim([ymin, ymax])
        ax3.set_ylim([ymin, ymax])
        ax4.set_ylim([ymin, ymax])

    except AttributeError:
        pass

    # -------------------------------------------------
    # save figure and data from the stairs
    # -------------------------------------------------
    pa4.save_fig()
    np.savetxt(figpath + cr.resfile + fig_descr, datastair)


def freeyaw_steady_points(resfile, db, figpath):
    """
    Select the steady points on the forced free yaw OJF cases.

    We only consider data with sane tower strain, so only April!
    """

    respath = os.path.join(PATH_DB, 'symlinks_all/')
    res = ojfresult.ComboResults(respath, resfile, silent=False, sync=True,
                                 cal=True)
    # RPM from pulse only returns the pulses, nothing else is done
#    # we use the standard calcict cases
#    cd = ojfresult.CalibrationData
#    res._calibrate_dspace(cd.caldict_dspace_04)
#    res._calibrate_blade(cd.caldict_blade_04)
    res._resample()
#    res.dashboard_a3(figpath)

    # and see if the staircase filter can get what we want
    irpm = res.dspace.labels_ch['RPM']
    iyaw = res.dspace.labels_ch['Yaw Laser']
    ifa = res.dspace.labels_ch['Tower Strain For-Aft']

    # because we don't get the indices in a reliable manner, do it 3 times and
    # hope we have the same points?? Or use other filtering technique?

#    # Select the stair ranges based on steady, constant RPM's
#    ext = '_RPMargs'
#    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
#               figpath=figpath, figfile=resfile+ext, runid=resfile+ext)
#    time_stair, data_stair = sc.setup_filter(res.dspace.time,
#                res.dspace.data[:,irpm], smooth_window=1.5,
#                cutoff_hz=False, dt=1, dt_treshold=5.0e-4,
#                stair_step_tresh=5.0, smoothen='moving',
#                points_per_stair=500)
#    results_filtering(res, figpath, ext, sc.arg_stair)

    # Select the stair ranges based on steady, constant yaw angles
    # this is a more robust approach when combined with the moving the start
    # index of the stair argument more close to the end index
    sps = int(round(1/np.diff(res.dspace.time).mean(), 0))
    ext = '_YAWargs'
    sc = StairCase(plt_progress=False, pprpath=figpath, figpath=figpath,
                   figfile=resfile+'_yaw', runid=resfile+'_yaw')
    time_stair, data_stair = sc.setup_filter(res.dspace.time,
                res.dspace.data[:,iyaw], smooth_window=1.5,
                cutoff_hz=False, dt=1, dt_treshold=2.0e-4,
                stair_step_tresh=1.5, smoothen='moving',
                points_per_stair=sps*3)

#    # and for the thrust
#    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
#                figpath=figpath, figfile=resfile+'_fa', runid=resfile+'_fa')
#    time_stair, data_stair = sc.setup_filter(res.dspace.time,
#                res.dspace.data[:,ifa], smooth_window=1.5,
#                cutoff_hz=False, dt=1, dt_treshold=2.0e-4,
#                stair_step_tresh=1.0, smoothen='moving',
#                points_per_stair=800)

#    # the blade channel 1
#    pltid = resfile + '_B2_R'
#    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
#                figpath=figpath, figfile=pltid, runid=pltid)
#    time_stair, data_stair = sc.setup_filter(res.blade.time,
#                res.blade.data[:,0], smooth_window=1.5,
#                cutoff_hz=False, dt=1, dt_treshold=5.0e-6,
#                stair_step_tresh=0.005, smoothen='moving',
#                points_per_stair=800)
#    # the blade channel 2
#    pltid = resfile + '_B2_30'
#    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
#                figpath=figpath, figfile=pltid, runid=pltid)
#    time_stair, data_stair = sc.setup_filter(res.blade.time,
#                res.blade.data[:,1], smooth_window=1.5,
#                cutoff_hz=False, dt=1, dt_treshold=5.0e-6,
#                stair_step_tresh=0.015, smoothen='moving',
#                points_per_stair=800)
#    # the blade channel 3
#    pltid = resfile + '_B1_R'
#    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
#                figpath=figpath, figfile=pltid, runid=pltid)
#    time_stair, data_stair = sc.setup_filter(res.blade.time,
#                res.blade.data[:,2], smooth_window=1.5,
#                cutoff_hz=False, dt=1, dt_treshold=5.0e-6,
#                stair_step_tresh=0.005, smoothen='moving',
#                points_per_stair=800)
#    # the blade channel 4
#    pltid = resfile + '_B1_30'
#    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
#                figpath=figpath, figfile=pltid, runid=pltid)
#    time_stair, data_stair = sc.setup_filter(res.blade.time,
#                res.blade.data[:,3], smooth_window=1.5,
#                cutoff_hz=False, dt=1, dt_treshold=5.0e-6,
#                stair_step_tresh=0.015, smoothen='moving',
#                points_per_stair=800)

    # for the yaw angle selection: the yaw angle is steady for the
    # whole duration of the stair, but the other measurements are not!
    # therefore delay the starting stair point 50%
    if ext.find('YAW') > -1:
        delta = (sc.arg_stair[1,:] - sc.arg_stair[0,:]) / 1.25
        sc.arg_stair[0,:] += delta.astype(np.int)

    # make plots to inspect the extraction process, save stair step points in
    # text files
    results_filtering(res, figpath, ext, sc.arg_stair)

    # save statistics of each stair step to database
    df_res = res.to_df()
#    db = ojfdb.MeasureDb(load_index=False)
    db.add_staircase_stats(resfile, df_res, sc.arg_stair)
#    db.save_stats(prefix='yawstairs')

    return db


def add_yawcontrol_stair_steps():
    """
    Use the new DataFrame index to select the cases.

    yaw_mode = freeyawforced
    yaw_mode = freeyaw and the_rest = forced

    Extract the steps from the staircase measurements, and add to index
    and statistics database.

    """
    figpath = 'figures/forced_yaw_error/'
#    fname = os.path.join(PATH_DB, 'db_index_symlinks_all.h5')
#    df_db = pd.read_hdf(fname, 'table')
    db = ojfdb.MeasureDb(prefix='symlinks_all', path_db='database/')
    idf = db.index
    df_sel = idf[(idf.yaw_mode == 'freeyawforced') |
                 ((idf.yaw_mode == 'freeyaw') & (idf.the_rest == 'forced')) |
                 (idf.yaw_mode2 == 'yawcontrol')]

#    df_sel = idf[(idf.yaw_mode == 'freeyawforced') |
#                 (idf.yaw_mode == 'freeyaw') |
#                 (idf.yaw_mode2 == 'yawcontrol')]

    for case in df_sel.index:
        # '0410_run_332'
        if not int(case.split('_')[2]) == 413:
            # plotting of all the cases
            db = freeyaw_steady_points(case, db, figpath)
            #
        else:
            continue

    db.add_df_dict2stat(update=True)
    db.save(complib=None)
    return db


def remove_yawcontrol_db():
    """Remove the yawcontrol cases from the database, they where added using
    the older stair extraction process.
    """
    db = ojfdb.MeasureDb(prefix='symlinks_all', path_db='database/')
    idf = db.index
    idf = idf[(idf.yaw_mode == 'freeyawforced') |
              (idf.yaw_mode2 == 'yawcontrol') |
              ((idf.yaw_mode == 'freeyaw') & (idf.the_rest == 'forced'))]
    # extract the old steps to compare
    idf_old = idf[idf.run_type == 'stair_step'].copy()
    db.load_stats()

#    mean_old = db.mean.loc[idf_old.index]
#    plt.figure('yaw angle vs rpm')
#    plt.plot(mean_old.yaw_angle, mean_old.rpm, 'rs')

    db.remove_from_stats_index(idf_old.index)
    db.save(complib=None)


###############################################################################
### STEADY YAW ERRORS, NG
###############################################################################


def remove_stair_step():
    """Remove all the stair step cases again from index and stats. This is
    helpfull when we want to run the stair extraction process again
    """

    db = ojfdb.MeasureDb(prefix='symlinks_all', path_db='database/')
    db.load_stats()
    id_rem = db.index[db.index.run_type=='stair_step'].index
    db.remove_from_stats_index(id_rem)
    db.save(complib='zlib')


def add_yawcontrol_steps_ng():
    """
    """
    db = ojfdb.MeasureDb(prefix='symlinks_all', path_db='database/')
    idf = db.index#[db.index.month == 2]
    idf = idf[(idf.yaw_mode == 'freeyawforced') |
              (idf.yaw_mode2 == 'yawcontrol') |
              ((idf.yaw_mode == 'freeyaw') & (idf.the_rest == 'forced'))]
#    # remove previous extracted steps from the same cases
#    idf = idf[idf.run_type=='']
#    db.index = idf

    figpath = 'figures/forced_yaw_error_ng/'
    for case in idf.index:
        try:
            fname = os.path.join(PATH_DATA_CAL_HDF, case + '.h5')
            res = pd.read_hdf(fname, 'table')
            time = res.time.values
            figname = os.path.join(figpath, case + '_filter_stairs.png')

            sc = StairCaseNG(time, freq_ds=10)
            sigs = [res.yaw_angle.values, res.rpm.values]
            weights = [1.5, 1.0]

            # low RPM: much bigger window compared to high RPM?
            if res.rpm.mean() < 200:
                w_lregr = 3.5
                x_threshold = 0.3
                min_step_window = 2.5
            else:
                w_lregr = 1.5
                x_threshold = 0.6
                min_step_window = 1.0
            steps = sc.get_steps(sigs, weights, cutoff_hz=1.0, window=w_lregr,
                                 x_threshold=x_threshold, figname=figname,
                                 min_step_window=min_step_window, order=2)
            db.add_staircase_stats(case, res, steps)

        except Exception as e:
            print(e)
            print('*'*80)

#    # evaluate the performance
#    db._dict2df()
#    # compare with old results
#    plt.figure('yaw angle vs rpm')
#    plt.plot(mean_old.yaw_angle, mean_old.rpm, 'rs')
#    plt.plot(db.df_mean.yaw_angle, db.df_mean.rpm, 'b>', alpha=0.7)

    db.add_df_dict2stat(update=True)
    db.save(complib='zlib')
    return db


def add_freeyaw_steady_steps():
    """
    """
    db = ojfdb.MeasureDb(prefix='symlinks_all', path_db='database/')
    idf = db.index
    idf = idf[(idf.rpm_change=='') & (idf.sweepid=='') &
              (idf.yaw_mode.str.startswith('free'))]
    # remove cases that have been used for add_yawcontrol_stair_steps()
    idf = idf[(idf.yaw_mode != 'freeyawforced') & (idf.the_rest != 'forced') &
              (idf.yaw_mode2 != 'yawcontrol') ]

    figpath = 'figures/steps_freeyaw_play/'
    for case in idf.index:
        try:
            fname = os.path.join(PATH_DATA_CAL_HDF, case + '.h5')
            res = pd.read_hdf(fname, 'table')
            time = res.time.values
            figname = os.path.join(figpath, case + '_filter_stairs.png')

            sc = StairCaseNG(time, freq_ds=10)
            sigs = [res.yaw_angle.values, res.rpm.values]

            # low RPM: much bigger window compared to high RPM?
            if res.rpm.mean() < 200:
                w_lregr = 3.5
                x_threshold = 0.25
                min_step_window = 3.0
                weights = [1.5, 1.0]
            else:
                w_lregr = 1.5
                x_threshold = 0.6
                min_step_window = 1.0
                weights = [1.5, 1.0]
            steps = sc.get_steps(sigs, weights, cutoff_hz=1.0, window=w_lregr,
                                 x_threshold=x_threshold, figname=figname,
                                 min_step_window=min_step_window, order=2)
            db.add_staircase_stats(case, res, steps)

        except Exception as e:
            print(e)
            print('*'*80)

    db.add_df_dict2stat(update=True)
    db.save(complib='zlib')

    return db


def test_add():

    db = ojfdb.MeasureDb(prefix='symlinks_all', path_db='database/')
    idf = db.index
    df_sel = idf[(idf.yaw_mode == 'freeyawforced') |
                 ((idf.yaw_mode == 'freeyaw') & (idf.the_rest == 'forced')) |
                 (idf.yaw_mode2 == 'yawcontrol')]

    case = df_sel.index.values[0]
    fname = os.path.join('data', 'calibrated', 'DataFrame', case+'.h5')
    df_res = pd.read_hdf(fname, 'table')

    arg_stair = np.array([[1,200, 500], [100, 300,1000]])
    i1 = arg_stair[0,1]
    i2 = arg_stair[1,1]
    resfile_step = '%s_i0_%i_i1_%i' % (case, i1, i2)

    db.add_stats(resfile_step, df_res[i1:i2])
    mean = db.df_dict_mean

    db._init_df_dict_stats()
    db._add_stats_df_dict(case, df_res[i1:i2].mean(), db.df_dict_mean)
    mean = db.df_dict_mean
    maxx = db.df_dict_max

###############################################################################
### DYNAMIC YAW ERR PERF
###############################################################################
# from the OJF data and HAWC2 results: subtract the yaw response parameters:
# how quickly is it falling back to steady yaw error
# Much of the work is done manually: many freeyaw cases have been performed
# in one case, so we have to manually specify the time ranges

class plot_freeyaw_response(object):
    """Plot response of single free yaw case in one plot
    """

    def __init__(self, df, t0, duration, ncols=1):

        if ncols == 1:
            self.fig, self.axes = plotting.subplots(nrows=1, ncols=ncols,
                                                    figsize=(4,2), dpi=120)
            self.axes = self.axes.flatten()
            self.ax = self.axes[0]
            self.first(df, t0, duration)
        else:
            self.fig, self.axes = plotting.subplots(nrows=1, ncols=2,
                                                    figsize=(7,2), dpi=120)
            self.axes = self.axes.flatten()
            self.ax = self.axes[0]
            self.ax2 = self.axes[1]
            self.first(df, t0, duration)

    def first(self, df, t0, duration):
        nre = len(df.time)
        te = df.time.values[-1]
        i0 = int(t0*nre/te)
        i1 = i0 + int(duration*nre/te)

        self.ax.plot(df.time[i0:i1]-t0, df.rpm[i0:i1], 'b-', label='RPM')
        self.axr = self.ax.twinx()
        self.axr.plot(df.time[i0:i1]-t0, df.yaw_angle[i0:i1], 'r-',
                      label='yaw [deg]', alpha=0.7)

        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel('rotor speed [RPM]')
        self.axr.set_ylabel('yaw angle [deg]')
        self.ax.set_xlim([0, duration])
        self.ax.grid(True)

    def second(self, df, t0, duration):

        nre = len(df.time)
        te = df.time.values[-1]
        i0 = int(t0*nre/te)
        i1 = i0 + int(duration*nre/te)

        self.ax2.plot(df.time[i0:i1]-t0, df.rpm[i0:i1], 'b-', label='RPM')
        self.ax2r = self.ax2.twinx()
        self.ax2r.plot(df.time[i0:i1]-t0, df.yaw_angle[i0:i1], 'r-',
                       label='yaw [deg]', alpha=0.7)

        self.ax2.set_xlabel('time [s]')
        self.ax2r.set_ylabel('yaw angle [deg]')
        self.ax2.set_xlim([0, duration])
        self.ax2.grid(True)


    def set_ylim_ticks(self, ylim, nrticks=9):
        self.ax.set_ylim([ylim[0], ylim[1]])
        self.ax.yaxis.set_ticks(np.linspace(ylim[0], ylim[1], nrticks).tolist())

    def legend_axmatch(self):
        self.ax, self.axr = plotting.match_yticks(self.ax, self.axr)
        self.leg = plotting.one_legend(self.ax, self.axr, loc='best',
                                       borderaxespad=0)
        self.leg.get_frame().set_alpha(0.6)

    def save(self, fname):
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.92, bottom=0.25, right=0.85)
        print('saving: %s' % fname)
        self.fig.savefig(fname + '.png')
        self.fig.savefig(fname + '.eps')


class FreeyawRespons(ojfdb.MeasureDb):

    def __init__(self, interactive=True):
        # initialize the MeasureDb first
        super(FreeyawRespons, self).__init__()
        self.load_stats()
        self.save_stats_index = False

        self.fpath = 'data/calibrated/DataFrame'
        self.fdir = 'figures/freeyaw'
        self.interactive = interactive
        cols = ['runid', 'index', 't0', 'i0', 't1', 'i1', 'duration']
        cols_df = ['time', 'rpm', 'yaw_angle', 'tower_strain_fa',
                   'tower_strain_ss', 'towertop_acc_fa', 'towertop_acc_ss',
                   'towertop_acc_z', 'voltage_filt', 'current_filt',
                   'rotor_azimuth', 'power', 'power2', 'hs_trigger',
                   'hs_trigger_start_end', 'rpm_pulse', 'temperature',
                   'wind_speed', 'static_p', 'blade2_root', 'blade2_30pc',
                   'blade1_root', 'blade1_30pc', 'blade_rpm_pulse']
        self.cols_df = set(cols_df)
        self.df_dict = {col:[] for col in cols}
        for col in cols_df:
            self.df_dict['%s_init' % col] = []
            self.df_dict['%s_end' % col] = []

    def add_run(self, df, t0, duration, resfile, t_s_init=1.9, t_s_end=1.9):
        nre = len(df.time)
        te = df.time.values[-1]
        i0 = int(t0*nre/te)
        i1 = i0 + int(duration*nre/te)
        i_s_init = int(t_s_init*nre/te)
        i_s_end = int(t_s_end*nre/te)

        if df.yaw_angle.values[i0:i0+i_s_init].mean() >= 0.0:
            app = 'fastside'
        else:
            app = 'slowside'

        res_init = resfile + '_%s_t0_%05.02fs_init' % (app, t0)
        index_init = self.index.loc[resfile].to_dict()
        index_init['sweepid'] = '%i_%i' % (i0, i0+i_s_init)
        index_init['run_type'] = 'fyr_%s_init' % app
        self.add_stats(res_init, df[i0:i0+i_s_init], index_row=index_init)

        res_end = resfile + '_%s_t0_%05.02fs_end' % (app, t0)
        index_end = self.index.loc[resfile].to_dict()
        index_end['sweepid'] = '%i_%i' % (i1-i_s_end, i1)
        index_end['run_type'] = 'fyr_%s_end' % app
        self.add_stats(res_end, df[i1-i_s_end:i1], index_row=index_end)

    def do_all_runs(self, save_stats_index=True, complib='blosc'):
        self.interactive = False
        self.save_stats_index = save_stats_index
        for item in dir(self):
            if item[:4] == 'run_':
                run = getattr(self, item)
                run()
        if save_stats_index:
            self.save_all_runs(complib=complib)

    def save_all_runs(self, complib='blosc'):
        self.add_df_dict2stat(update=True)
        self.save(complib=complib)

    def run_263(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_263_7.0ms_dc0.4_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_263_7.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

    #    plt.figure('yaw, 0405_run_263_7.0ms')
    #    plt.plot(df.time, df.yaw_angle, label='flex')
    #    plt.legend(loc='best')
    #    plt.grid()

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 25.7
        duration = 12.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([380, 540])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        t0 = 52.6
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([380, 540])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_264(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_264_7.0ms_dc0.6_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_264_7.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 15.82
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([340, 500])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 45.56
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([340, 500])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_265(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_265_8.0ms_dc0.0_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_265_8.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 21.16 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([550, 750])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 52.34 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([550, 750])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_266(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_266_8.0ms_dc0.4_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_266_8.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 18.12 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([500, 660])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 48.7 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([500, 660])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_267(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_267_8.0ms_dc0.6_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_267_8.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 14.48 - 2
        duration = 13.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([460, 620])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 39.01 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.ax.set_ylim([460, 620])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_269(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_269_9.0ms_dc0_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_269_9.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 25.05 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([675, 875])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 52.36 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([675, 875])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_275(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0405_run_275_9.0ms_dc0.4_flexies_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0405_run_275_9.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 21.32 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([600, 800])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 49.71 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([600, 800])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_325(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0410_run_325_8ms_dc0_samoerai_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0410_run_325_8ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 20.62 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([550, 750])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 46.73 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([550, 750])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_326(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0410_run_326_8ms_dc0.4_samoerai_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0410_run_326_8ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 23.29 - 2
        duration = 12.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([550, 750])
        q.axr.set_ylim([-35, 5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 47.51 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([500, 700])
        q.axr.set_ylim([35, -5])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def run_330(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0410_run_330_9ms_dc1_samoerai_freeyaw_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0410_run_330_9ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 20.85 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([390, 660], nrticks=10)
        q.axr.set_ylim([-35, 10])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 51.47 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([390, 660], nrticks=10)
        q.axr.set_ylim([35, -10])
        q.legend_axmatch()
        q.save(figname)
        if self.save_stats_index:
            self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

    def badrun_131(self):
        """
        """
        fpath = 'data/calibrated/DataFrame'
        fdir = 'figures/freeyaw'

        fname = '0213_run_131_7.0ms_dc1_samoerai_freeyawplaying_pwm1000_highrpm'
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')

        # overview plot
        if self.interactive:
            plt.figure('rpm, 0213_run_131_7.0ms')
            plt.plot(df.time, df.rpm, 'b', label='RPM')
            plt.grid()
            plt.twinx()
            plt.plot(df.time, df.yaw_angle, 'r', label='yaw')
            plt.legend(loc='best')

        # ---------------------------------------------------------------------
        # detailed report plots
        t0 = 20.85 - 2
        duration = 14.0
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([390, 660], nrticks=10)
        q.axr.set_ylim([-35, 10])
        q.legend_axmatch()
        q.save(figname)
        self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)

        # detailed report plots
        t0 = 51.47 - 2
        figname = os.path.join(fdir, fname + '_t0_%03.02f' % t0)
        q = plot_freeyaw_response(df, t0, duration)
        q.set_ylim_ticks([390, 660], nrticks=10)
        q.axr.set_ylim([35, -10])
        q.legend_axmatch()
        q.save(figname)
        self.add_run(df, t0, duration, fname, t_s_init=1.9, t_s_end=1.9)


def _get_fyr_range(gr, df_res, case):
    sel = gr[gr['run_type'].str.startswith(case)]
    iis = chain.from_iterable([k.split('_') for k in sel['sweepid'].values])
    iis = np.array(list(iis), dtype=np.int)
    i0, i1 = iis.min(), iis.max()
    df_range = df_res[i0:i1].copy()
    df_range['time'] -= df_range['time'].values[0]
    return df_range


def plot_fyr_all_normalized():
    """Normalize RPM and yaw angle changes, compare the response
    """

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    db = ojfdb.MeasureDb()
    db.load_stats()
    # select all fyr cases
    isel = db.index[db.index['run_type'].str.startswith('fyr')]

    fig, axes = plotting.subplots(nrows=1, ncols=1, figsize=(5,2), dpi=120)
    axes = axes.flatten()
    ax = axes[0]

#    fig, ax = plt.subplots(nrows=1, ncols=1)

    axs = ax.twinx() # inner
    axf = ax.twinx() # outer
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    axf.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(axf)
    # Second, show the right spine.
    axf.spines["right"].set_visible(True)

    for runid, gr in isel.groupby(isel['runid']):
        # just take the first entry, it is the same result file
        df_base = db.load_measurement_fromindex(gr.iloc[0])

        # slow side
        # fast side case, inner right axis
        df = _get_fyr_range(gr, df_base, 'fyr_slow')
        rpm_norm = df.rpm - df.rpm.values[0]
        rpm_norm /= rpm_norm.values[-1]
        ax.plot(df.time, rpm_norm, 'k-', label='RPM')

        yaw_norm = df.yaw_angle - df.yaw_angle.values[0]
        yaw_norm /= yaw_norm.values[-1]
        axs.plot(df.time, yaw_norm, 'y-', label='yaw [deg]', alpha=0.7)

        # fast side case, outer right axis
        df = _get_fyr_range(gr, df_base, 'fyr_fast')
        rpm_norm = df.rpm - df.rpm.values[0]
        rpm_norm /= rpm_norm.values[-1]
        ax.plot(df.time, rpm_norm, 'b-', label='RPM')

        yaw_norm = df.yaw_angle - df.yaw_angle.values[0]
        yaw_norm /= yaw_norm.values[-1]
        axf.plot(df.time, yaw_norm, 'r-', label='yaw [deg]', alpha=0.7)

#    axf.set_ylim([35, -10])
#    axs.set_ylim([-35, 10])
    ax.grid()
    ax.set_xlabel('time [s]')
#    ax.set_title('time [s]')

    for tl in axf.get_yticklabels():
        tl.set_color('r')
    for tl in axs.get_yticklabels():
        tl.set_color('y')

    fig.tight_layout()

    pfig = 'figures/freeyaw/'
    fig.savefig(os.path.join(pfig, 'allfreeyaw'))


def plot_fyr_respons(col):
    """col is the channel name to be plotted of all the selected free yaw
    respons intervals in the index database.
    """

    db = ojfdb.MeasureDb()
    db.load_stats()
    # select all fyr cases
    isel = db.index[db.index['run_type'].str.startswith('fyr')]

    # the normalized plots
    fig, axes = plotting.subplots(nrows=1, ncols=1, figsize=(6,3), dpi=120)
    axes = axes.flatten()
    ax = axes[0]

    # original values
    fig2, axes2 = plotting.subplots(nrows=1, ncols=1, figsize=(6,3), dpi=120)
    axes2 = axes2.flatten()
    ax2 = axes2[0]
    if col == 'yaw_angle':
        ax3 = ax2.twinx()
    else:
        ax3 = ax2

    for runid, gr in isel.groupby(isel['runid']):
        # just take the first entry, it is the same result file
        df_base = db.load_measurement_fromindex(gr.iloc[0])
        dff = _get_fyr_range(gr, df_base, 'fyr_fast')
        dfs = _get_fyr_range(gr, df_base, 'fyr_slow')

        if gr['blades'].iloc[0][:3] == 'sam':
            c1, c2 = 'r-^', 'b-v'
            kw = {'markevery':300}
        else:
            c1, c2 = 'r-', 'b-'
            kw = {}

        norm = dff[col] - dff[col].values[0]
        norm /= norm.values[-1]
        ax.plot(dff.time, norm, c1, alpha=0.7, **kw)

        norm = dfs[col] - dfs[col].values[0]
        norm /= norm.values[-1]
        ax.plot(dfs.time, norm, c2, alpha=0.7, **kw)

        ax2.plot(dff.time, dff[col], c1, alpha=0.7, **kw)
        ax3.plot(dfs.time, dfs[col], c2, alpha=0.7, **kw)

    ax.grid()
    ax.set_xlabel('time [s]')
    ax2.set_xlabel('time [s]')
    if col == 'rpm':
        ax.set_title('Normalized rotor speed')
        ax.set_ylabel('[-]')
        ax2.set_title('Rotor speed [rpm]')
        ax2.set_ylabel('[rpm]')
    else:
        ax.set_title('Normalized yaw angle')
        ax.set_ylabel('[-]')
        ax2.set_title('Yaw angle [deg]')
        ax2.set_ylabel('[deg]')
    fig.tight_layout()

    ax2.grid()
    fig2.tight_layout()

    if col == 'yaw_angle':
        ax2.set_ylim([35, -10])
        ax3.set_ylim([-35, 10])
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        for tl in ax3.get_yticklabels():
            tl.set_color('b')

    pfig = 'figures/freeyaw/'
    fig.savefig(os.path.join(pfig, 'allfreeyaw_%s_norm.png' % col))
    fig.savefig(os.path.join(pfig, 'allfreeyaw_%s_norm.eps' % col))

    pfig = 'figures/freeyaw/'
    fig2.savefig(os.path.join(pfig, 'allfreeyaw_%s.png' % col))
    fig2.savefig(os.path.join(pfig, 'allfreeyaw_%s.eps' % col))


def deltas_fyr_yaw_rpm():
    """Visualize the yaw/rpm differences as function of delta TSR.
    """
    db = ojfdb.MeasureDb()
    db.load_stats()
    # select all fyr cases
    isel = db.index[db.index['run_type'].str.startswith('fyr')]
    msel = db.mean[db.mean.index.isin(isel.index.tolist())]

#    dt_table = pd.DataFrame(columns=['d_rpm', 'rpm_0', 'rpm_1', 'd_yaw',
#                                     'yaw_0', 'yaw_1', 'tsr_0', 'tsr_1', 'ws'])
    cols = ['d_rpm', 'rpm_0', 'rpm_1', 'd_yaw', 'yaw_angle_0', 'yaw_angle_1',
            'd_tsr', 'tsr_0', 'tsr_1', 'wind_speed', 'blades', 'side']
    df_dict = {col:[] for col in cols}
    index = []
    for runid, gr in isel.groupby(isel['runid']):
#        # corresponding means
#        grm = msel[msel.index.isin(gr.index.tolist())]
        # slow/fast
        fs0 = gr[gr['run_type']=='fyr_fastside_init']
        fs1 = gr[gr['run_type']=='fyr_fastside_end']
        ss0 = gr[gr['run_type']=='fyr_slowside_init']
        ss1 = gr[gr['run_type']=='fyr_slowside_end']
        blades = gr['blades'].iloc[0]

        # delta's
        for xs0, xs1 in zip([fs0, ss0], [fs1, ss1]):
            xs0_m = msel.loc[xs0.iloc[0].name]
            xs1_m = msel.loc[xs1.iloc[0].name]
            fs_index = xs0.iloc[0].name.replace('_init', '')
            index.append(fs_index)
            df_dict['side'].append(fs_index.split('_')[-3])
            df_dict['blades'].append(blades)
            df_dict['wind_speed'].append(xs1_m['wind_speed'])
            df_dict['d_rpm'].append(xs1_m['rpm'] - xs0_m['rpm'])
            df_dict['d_yaw'].append(xs1_m['yaw_angle'] - xs0_m['yaw_angle'])
            df_dict['rpm_0'].append(xs0_m['rpm'])
            df_dict['rpm_1'].append(xs1_m['rpm'])
            df_dict['yaw_angle_0'].append(xs0_m['yaw_angle'])
            df_dict['yaw_angle_1'].append(xs1_m['yaw_angle'])
            df_dict['tsr_0'].append(ojfdb.tsr(xs0_m))
            df_dict['tsr_1'].append(ojfdb.tsr(xs1_m))
            df_dict['d_tsr'].append(df_dict['tsr_1'][-1] - df_dict['tsr_0'][-1])
    df = pd.DataFrame(df_dict, index=index)

#    # They show similar trends, but:
#    plt.figure('d rpm, d yaw')
#    plt.plot(df.d_rpm, np.abs(df.d_yaw), 'bo')
#    plt.figure('d tsr, d yaw')
#    plt.plot(df.d_tsr, np.abs(df.d_yaw), 'rs')
#
#    # this is the best, nice and linear
#    plt.figure('d tsr, d rpm')
#    plt.plot(df.d_tsr, df.d_rpm, 'k>')

    fig1, axes1 = plotting.subplots(nrows=1, ncols=1, figsize=(4.3,2.3), dpi=120)
    ax1 = axes1.flatten()[0]
    fig2, axes2 = plotting.subplots(nrows=1, ncols=1, figsize=(4.3,2.3), dpi=120)
    ax2 = axes2.flatten()[0]
    colors = {'fastside':'r', 'slowside':'b'}
    symbols = {'samoerai':'x', 'flexies':'v'}
    mps_blade = {'samoerai':'swept', 'flexies':'straight'}
    mps_side = {'fastside':'A', 'slowside':'B'}
    units = {'yaw':'[deg]', 'rpm':''}

    for side, gr in df.groupby('side'):
        col = colors[side]
        for blade, sel in gr.groupby('blades'):
            symb = symbols[blade]
            label = '%s, %s' % (mps_side[side], mps_blade[blade])
            ax1.plot(sel.d_rpm, sel.d_tsr, col+symb, label=label)
            ax2.plot(sel.d_yaw.abs(), sel.d_tsr, col+symb, label=label)

    for chan, ax, fig in zip(['rpm', 'yaw'], [ax1, ax2], [fig1, fig2]):
        ax.grid()
        ax.set_ylabel('$\\Delta$ TSR [-]')
        ax.set_xlabel('$\\Delta$ %s %s' % (chan, units[chan]))
#        leg = ax.legend(loc='lower right', ncol=2, labelspacing=0, columnspacing=0)
        leg = ax.legend(loc='best', labelspacing=0, columnspacing=0)
        leg.get_frame().set_alpha(0.6)
        fig.tight_layout()
        pfig = 'figures/freeyaw/'
        fig.savefig(os.path.join(pfig, 'freeyaw_deltas_tsr_%s.png' % chan))
        fig.savefig(os.path.join(pfig, 'freeyaw_deltas_tsr_%s.eps' % chan))


def freeyaw_april_timings():
    """
    Isolate the free yawing cases, from fixed to release and if it reaches
    steady state
    """

    def select(t0, t1):
        i0 = t0*res.dspace.sample_rate
        i1 = t1*res.dspace.sample_rate
        return res.dspace.time[i0:i1], res.dspace.data[i0:i1,irpm], \
                            res.dspace.data[i0:i1,iyaw]

    # ------------------------------------------------------------------------
    # dashboard plot of all the april freeyaw cases
#    db = ojfdb.MeasureDb('symlinks_all_psicor')
#    inc = ['free']
#    exc = ['force']
#    data, cases, head = db.select(['04'], inc,  exc)
#    # and plot all those cases
#    db.plot(cases, figfolder='figures_freeyaw_psicor/', calibrate=True,
#            caldict_dspace_04=ojfresult.CalibrationData.caldict_dspace_04,
#            caldict_blade_04=ojfresult.CalibrationData.caldict_blade_04)
    # ------------------------------------------------------------------------

    # and manually define for each case a subdomain of interest
    respath = os.path.join(PATH_DB, 'symlinks_all/')

    # ------------------------------------------------------------------------
    # 0413_run_414_8ms_dc0_stiffblades_freeyaw
    resfile = '0413_run_414_8ms_dc0_stiffblades_freeyaw'
    res = ojfresult.ComboResults(respath, resfile, silent=True)
    res._calibrate_dspace(ojfresult.CalibrationData.caldict_dspace_04)
    res._calibrate_blade(ojfresult.CalibrationData.caldict_blade_04)
    filt = filters.Filters()

    # first fixed-release cyce
    irpm = res.dspace.labels_ch['RPM']
    iyaw = res.dspace.labels_ch['Yaw Laser']
    time, rpm, yaw = select(18, 31)

    plt.plot(time, rpm)
    plt.grid()
    # we need to filter the signals to extract the properties
    rpm_filt, N, delay = filt.fir(time, rpm, freq_trans_width=1.0,
                                  sample_rate=res.dspace.sample_rate,
                                  ripple_db=40.0, cutoff_hz=0.5)
    # and plotting of the filtered signal
    plt.plot(time[N-1:]-delay, rpm_filt[N-1:], 'r--')

    plt.plot(time, yaw)
    yaw_filt, N, delay = filt.fir(time, yaw, freq_trans_width=1.0,
                                  sample_rate=res.dspace.sample_rate,
                                  ripple_db=10.0, cutoff_hz=0.1)
    plt.plot(time[N-1:]-delay, yaw_filt[N-1:], 'r--')
    yawdiff = np.diff(yaw)
    plt.plot(time[N+1:], yawdiff[N:])

    yaw_filt2 = filt.butter(time, yaw, cutoff_hz=1.0,
                         sample_rate=res.dspace.sample_rate)
    plt.plot(time, yaw_filt2, 'g--')
    # ------------------------------------------------------------------------

###############################################################################
### COMPARE YAW RESPOSNE
###############################################################################
# compare the free yaw response between flex, stiff, coning and swept blades
# also include

if __name__ == '__main__':

    dummy = False

    # extract steady states from forced yaw control runs, save to index/stats
#    db = add_yawcontrol_steps_ng()

    # extract steady states from free yaw cases, save to index/stats
#    db = add_freeyaw_steady_steps()

    # plot free yaw response unified way, save init/end to index/stats
#    fyr = FreeyawRespons(interactive=False)
#    fyr.do_all_runs(save_stats_index=False)
#    fyr.run_325()
#    fyr.run_326()
#    fyr.run_330()
#    fyr.do_all_runs(save_stats_index=False)
#    fyr.save_all_runs(complib='zlib')
    plot_fyr_respons('rpm')
    plot_fyr_respons('yaw_angle')
