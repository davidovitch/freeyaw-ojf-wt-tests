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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ojfdb
import ojfresult
import plotting
import filters
from staircase import StairCase, StairCaseNG

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
#    db = add_yawcontrol_steps_ng()
#    db = add_freeyaw_steady_steps()
#    freeyaw_april_timings()
