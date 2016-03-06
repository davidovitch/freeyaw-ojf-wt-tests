# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:01:56 2013

Select all the OJF free yaw cases, manually or automatically only take those
time ranges so we can actually calculate how fast the yaw angle is recovering

@author: dave
"""

# built in modules
import pickle
import os
from copy import copy

import numpy as np
import pylab as plt

# 3th party modules
#import numpy as np
# custom modules
#import Simulations as sim
#import plotting
#import ojfvshawc2
#from ojf_post import master_tags, variable_tag_func, post_launch
import ojfdb
import ojfresult
import plotting
import filters

PATH_DB = 'database/'

###############################################################################
### STEADY YAW ERR PERF
###############################################################################

def add_to_database(res, istart, istop):
    """
    add the carufelly selected steady states from the forced yaw error
    to the database

    Al stair cases are marked with _STC_%i
    """
    db_stats = {}
    # dividing by 100 is safe, since we set points_per_stair=800, so we do not
    # risque of having a non unique key
    case = '%s_STC_%i' % (res.resfile, istart/100)
    # stats is already a dictionary
    db_stats[case] = res.stats.copy()
    # save the indices into the statsdict for later reference
    db_stats[case]['STC index resampled'] = [istart, istop]
    # add the channel discriptions
    db_stats[case]['dspace labels_ch'] = copy(res.dspace.labels_ch)
    # incase there is no OJF data
    try:
        db_stats[case]['ojf labels'] = copy(res.ojf.labels)
    except AttributeError:
        pass

    return db_stats

def results_filtering(cr, figpath, fig_descr, arg_stair):
    """
    Like the ComboResults.dashboard_a3, but now include the filtered signal
    and the subtracted values.

    Saves the datastair data, it is formatted as: ndarray(stair_nr, data_chan)
    [RPM, yaw, FA, SS, B1_r, B1_30, B2_r, B2_30, windspeed, temperature]
    Note that this is actually not necessary and rendundant, since we properly
    added all the cases as subecases to the ojf_db database

    Parameters
    ----------

    cr : ComboResults instance
        A ojfresult.ComboResults instance

    figpath : str

    fig_descr : str
        small adition to the file name

    arg_stair : array(2,n)
        Start/stop indices to n stairs

    arg_st_fl : array(n) (not used)
        array with the indices of the stairs, invalid points are masked


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

    isel, db_stats = [], {}
    for k in range(arg_stair.shape[1]):
        # for the yaw angle selection: the yaw angle is steady for the
        # whole duration of the stair, but the other measurements are not!
        # therefore delay the starting stair point 50%
        if fig_descr.find('YAW') > -1:
            delta = int( (arg_stair[1,k]-arg_stair[0,k])/1.25 )
            i1 = arg_stair[0,k]+delta
            isel.append(np.r_[i1:arg_stair[1,k]])
        else:
            i1 = arg_stair[0,k]
            isel.append(np.r_[i1:arg_stair[1,k]])
        i2 = arg_stair[1,k]
        # calculate all the means, std, min, max and range for each channel
        cr.statistics(i1=i1, i2=i2)
        # and save into the database
        db_stats.update(add_to_database(cr, i1, i2))

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

    # and update the database
    prefix = 'symlinks_all'
    try:
        # if it exists, update the file first before saving
        FILE = open(os.path.join(PATH_DB, 'db_stats_%s.pkl' % prefix))
        db_stats_update = pickle.load(FILE)
        # overwrite the old entries with new ones! not the other way around
        db_stats_update.update(db_stats)
        FILE.close()
    except IOError:
        # no need to update an existing database file
        db_stats_update = db_stats
    # and save the database stats
    FILE = open(os.path.join(PATH_DB, 'db_stats_%s.pkl' % prefix), 'wb')
    pickle.dump(db_stats_update, FILE, protocol=2)
    FILE.close()
    print 'updated db: %sdb_stats_%s.pkl' % (PATH_DB, prefix)

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


def freeyaw_steady_points(resfile):
    """
    Select the steady points on the forced free yaw OJF cases.

    We only consider data with sane tower strain, so only April!
    """

    figpath = 'figures/forced_yaw_error/'

    # we use the standard calcict cases
    cd = ojfresult.CalibrationData

    respath = os.path.join(PATH_DB, 'symlinks_all/')
    res = ojfresult.ComboResults(respath, resfile, silent=False, sync=True)
    # RPM from pulse only returns the pulses, nothing else is done
    res._calibrate_dspace(cd.caldict_dspace_04)
    res._calibrate_blade(cd.caldict_blade_04)
    res._resample()
    res.dashboard_a3(figpath)

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
    ext = '_YAWargs'
    sc = ojfresult.StairCase(plt_progress=False, pprpath=figpath,
                figpath=figpath, figfile=resfile+'_yaw', runid=resfile+'_yaw')
    time_stair, data_stair = sc.setup_filter(res.dspace.time,
                res.dspace.data[:,iyaw], smooth_window=1.5,
                cutoff_hz=False, dt=1, dt_treshold=2.0e-4,
                stair_step_tresh=1.5, smoothen='moving',
                points_per_stair=800)
    results_filtering(res, figpath, ext, sc.arg_stair)

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

def select_freeyaw():
    """
    Select all the forced free yawing cases and derive properties for
    each stair case. Stair cases are selected based on the yaw angle.

    Select the stair ranges based on steady, constant yaw angles
    this is a more robust approach when combined with the moving the start
    index of the stair argument more close to the end index
    """

    # load the OJF test database
    db = ojfdb.ojf_db('symlinks_all')
    exc = []
    inc = ['forced']
    valuedict = {}
    std = {'yaw':[1.0, 100.0]}
    data, cases, head = db.select(['04'], inc,  exc, valuedict=valuedict,
                                  values_std=std)

    # and apply the StairCase filtering to find the steady values of RPM,
    # yaw angle, and tower for aft bending moment
    for case in cases:
        # '0410_run_332'
        if not int(case.split('_')[2]) == 413:
            freeyaw_steady_points(case)
        else:
            continue

    # now will we manually select all the time ranges so we have well defined
    # cases as: forced yaw error -> release, instead of the mixed measurements
    # or can we automate this?

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
#    db = ojfdb.ojf_db('symlinks_all_psicor')
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

#    select_freeyaw()
#    yawerror_performance()

    freeyaw_april_timings()
