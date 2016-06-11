# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:03:39 2016

@author: dave
"""

# For Python compatibility
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import os
#import timeit

# COMMON 3TH PARTY
import numpy as np
#import scipy.io
#import scipy.integrate as integrate
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# NOT SO COMMON 3TH PARTY
#from scikits.audiolab import play
#import wafo

# CUSTOM
import plotting
import ojf_post
#import HawcPy
#import cython_func
#import ojfdesign
#import bladeprop
#import materials
#import Simulations as sim
#import misc
#import ojfresult
#import towercal
#import bladecal
#import yawcal
#from ojfdb import MeasureDb

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=11)
plt.rc('legend', numpoints=1)
plt.rc('legend', borderaxespad=0)
mpl.rcParams['text.latex.unicode'] = True


RESDATA_CAL_02 = 'data/raw/02/calibration/'
RESDATA_CAL_04 = 'data/raw/04/calibration/'
PROCESSING = 'processing/'

df_index = pd.read_hdf('database/db_index_symlinks_all.h5', 'table')
#df_index.set_index('basename', inplace=True)
df_mean = pd.read_hdf('database/db_stats_symlinks_all_mean.h5', 'table')
#df_mean.set_index('index', inplace=True)
df_std = pd.read_hdf('database/db_stats_symlinks_all_std.h5', 'table')
#df_std.set_index('index', inplace=True)


def ct(df):

    thrust = df.tower_strain_fa / ojf_post.model.momemt_arm_rotor
    # TODO: calculate rho from wind tunnel temperature and static pressure
    # rho = R*T / P   R_dryair = 287.058
    # but this results in rho=0.825, do we need an R value for more humid air?
#    R_dryair = 287.058
#    kelvin = 273.15
#    rho = (sel_mean.temperature + kelvin) * R_dryair / sel_mean.static_p
    rho = 1.225
    V = df.wind_speed
    # and normalize to get the thrust coefficient
    ct = thrust / (0.5*rho*V*V*ojf_post.model.A)

    return ct


def cp(df):
    rho = 1.225
    V = df.wind_speed
    return df.power / (0.5*rho*V*V*V*ojf_post.model.A)


def tsr(df):
    R = ojf_post.model.blade_radius
    return R*df.rpm*np.pi/(df.wind_speed*30.0)


def power_torque_rpm(df):
    return df.tower_strain_ss * (df.rpm*np.pi/30.0) * -1.0


def ct_vs_yawangle():
    """
    """

    # april only
    sel_ind = df_index[df_index.month==4]
    sel_std = df_std[df_std.index.isin(sel_ind.index.tolist())]
    # std should be low enough
    sel_std = df_std[(df_std.yaw_angle > -1.2) & (df_std.yaw_angle < 1.2)]
    # corresponding avarages
    sel_mean = df_mean[df_mean.index.isin(sel_std.index.tolist())]

    istiff = sel_ind[(sel_ind.blades=='stiff') | (sel_ind.blades=='stiffblades')]
    stiff = df_mean[df_mean.index.isin(istiff.index.tolist())]

    iflex = sel_ind[sel_ind.blades=='flex']
    flex = df_mean[df_mean.index.isin(iflex.index.tolist())]

    iflexies = sel_ind[sel_ind.blades=='flexies']
    flexies = df_mean[df_mean.index.isin(iflexies.index.tolist())]

    isam = sel_ind[sel_ind.blades=='samoerai']
    sam = df_mean[df_mean.index.isin(isam.index.tolist())]

    plt.figure('$C_T$ vs $\\psi$ (yaw angle)')
    plt.plot(stiff.yaw_angle, ct(stiff), 'rs', label='stiff')
    plt.plot(flex.yaw_angle, ct(flex), 'b<', label='flex')
    plt.plot(flexies.yaw_angle, ct(flexies), 'g>', label='flexies')
    plt.plot(sam.yaw_angle, ct(sam), 'ko', label='samoerai')
    plt.ylim([-0.1, 0.8])
    plt.legend(loc='best')
    plt.grid()


def power():
    """
    Can de tower bottom SS moment be correlated to rotor torque?

    TODO:
        * proper statistical corrolation plots
    """

    sel_ind = df_index[df_index.month==4].copy()
    sel_std = df_std[df_std.index.isin(sel_ind.index.tolist())].copy()
    sel_std = df_std[(df_std.yaw_angle > -1) & (df_std.yaw_angle < 1)]
    sel_mean = df_mean[df_mean.index.isin(sel_std.index.tolist())].copy()
    sel_mean = sel_mean[(sel_mean.yaw_angle > -1) & (sel_mean.yaw_angle < 1)]

    plt.figure(0)
    plt.plot(tsr(sel_mean), sel_mean.power, 'rs')
    plt.plot(tsr(sel_mean), power_torque_rpm(sel_mean), 'b>', alpha=0.7)
    plt.xlim([0,10])

    plt.figure(1)
    plt.plot(sel_mean.rpm, sel_mean.power, 'rs')
    plt.plot(sel_mean.rpm, power_torque_rpm(sel_mean), 'b>', alpha=0.7)
#    plt.xlim([0,10])

    plt.figure(2)
    plt.plot(sel_mean.duty_cycle, sel_mean.power, 'rx')
    plt.plot(sel_mean.duty_cycle, power_torque_rpm(sel_mean), 'bx')

    plt.figure(3)
    plt.plot(sel_mean.duty_cycle, sel_mean.tower_strain_ss, 'rx')


def rpm_vs_tower_ss():

    # only select cases with low std on yaw angle
    sel_std = df_std[(df_std.yaw_angle > -1) & (df_std.yaw_angle < 1)].copy()
    sel_mean = df_mean[df_mean.index.isin(sel_std.index.tolist())].copy()
    # only cases with yaw error below 1 degrees
    sel = sel_mean[(sel_mean.yaw_angle > -1) & (sel_mean.yaw_angle < 1)].copy()
    sel_mean = sel_mean[sel.index.isin(sel.index.tolist())].copy()

    plt.figure('rpm vs tower SS, steady yaw')
    plt.plot(sel_mean.rpm, sel_mean.tower_strain_ss, 'r>')
    plt.ylim([-5, 1])


def select_samoerai():

    sel_def = df_index[df_index.blades=='samoerai'].copy()
    sel_def = sel_def[sel_def.month==4]

    sel_std = df_std[df_std.index.isin(sel_def.index.tolist())].copy()
    sel_std = sel_std[(sel_std.yaw_angle > -1) & (sel_std.yaw_angle < 1)]

    sel_mean = df_mean[df_mean.index.isin(sel_std.index.tolist())].copy()


    # and only if std on yaw error is low

    plt.plot(sel_mean.rpm, sel_mean.power, 'rs')
    plt.plot(sel_mean.rpm, sel_mean.tower_strain_ss, 'b>')
    plt.plot(sel_mean.power, sel_mean.tower_strain_ss, 'b>')
    plt.plot(sel_mean.power, power_torque_rpm(sel_mean), 'b>')
    plt.plot(sel_mean.rpm, sel_mean.yaw_angle, 'b>')

    figpath = 'figures/overview/'
    figfile = 'test.png'
    scale = 1.5

    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
              grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
              wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(apr10[irpm,:],apr10[iblade,:],'bo', label='10 m/s')
    ax1.plot(apr9[irpm,:], apr9[iblade,:], 'rs', label='9 m/s')
    ax1.plot(apr8[irpm,:], apr8[iblade,:], 'gv', label='8 m/s')
    ax1.plot(apr7[irpm,:], apr7[iblade,:], 'm<', label='7 m/s')
    ax1.plot(apr6[irpm,:], apr6[iblade,:], 'c^', label='6 m/s')
    ax1.plot(apr5[irpm,:], apr5[iblade,:], 'y>', label='5 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title(title, size=14*scale)
    ax1.set_xlabel('Rotor speed [RPM]')
    ax1.set_ylabel(ylabel)
    ax1.grid(True)
    pa4.save_fig()


def freeyaw():
    """
    Selected time series of free yaw response cases
    """

    fpath = 'data/calibrated/DataFrame'
    fdir = 'figures/freeyaw'

    fname1 = '0405_run_277_9.0ms_dc1_flexies_freeyaw_highrpm.h5'
    fname2 = '0410_run_330_9ms_dc1_samoerai_freeyaw_highrpm.h5'

    flex = pd.read_hdf(os.path.join(fpath, fname1), 'table')
    samo = pd.read_hdf(os.path.join(fpath, fname2), 'table')

    fname = '277-vs-330-9ms-rpm'
    fig, axes = plotting.subplots(nrows=1, ncols=1, figsize=(5,2), dpi=120)
    axes = axes.flatten()

    duration = 15

    ax = axes[0]
    t0 = 48
    i0 = int(t0*129024/63.0)
    i1 = i0 + int(duration*129024/63.0)
    ax.plot(flex.time[i0:i1]-t0, flex.rpm[i0:i1], 'b-', label='straight blade')
    t0 = 49.6
    i0 = int(t0*69000/69.0)
    i1 = i0 + int(duration*69000/69.)
    ax.plot(samo.time[i0:i1]-t0, samo.rpm[i0:i1], 'r-', label='swept blade', alpha=0.7)

    for ax in axes:
        leg = ax.legend(loc='best', borderaxespad=0)
        leg.get_frame().set_alpha(0.6)
        ax.grid(True)

    ax.set_xlabel('time [s]')
    ax.set_ylabel('RPM')
    ax.set_xlim([0, 15])

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
#    fig.suptitle('windspeed 9 m/s')
    print('saving: %s' % os.path.join(fdir, fname))
    fig.savefig(os.path.join(fdir, fname + '.png'))
    fig.savefig(os.path.join(fdir, fname + '.eps'))


    # =========================================================================
    fname = '277-vs-330-9ms-yaw'
    fig, axes = plotting.subplots(nrows=1, ncols=1, figsize=(5,2), dpi=120)
    axes = axes.flatten()

    duration = 15

    ax = axes[0]
    t0 = 48
    i0 = int(t0*129024/63.0)
    i1 = i0 + int(duration*129024/63.0)
    ax.plot(flex.time[i0:i1]-t0, flex.yaw_angle[i0:i1], 'b-',
            label='straight blade')
    t0 = 49.6
    i0 = int(t0*69000/69.0)
    i1 = i0 + int(duration*69000/69.)
    ax.plot(samo.time[i0:i1]-t0, samo.yaw_angle[i0:i1], 'r-',
            label='swept blade', alpha=0.7)

    for ax in axes:
        leg = ax.legend(loc='best', borderaxespad=0)
        leg.get_frame().set_alpha(0.6)
        ax.grid(True)

    ax.set_xlabel('time [s]')
    ax.set_ylabel('yaw angle [deg]')
    ax.set_xlim([0, 15])

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
#    fig.suptitle('windspeed 9 m/s')
    print('saving: %s' % os.path.join(fdir, fname))
    fig.savefig(os.path.join(fdir, fname + '.png'))
    fig.savefig(os.path.join(fdir, fname + '.eps'))


def freeyaw2():
    """
    """
    # first, just overplot them all!

    fpath = 'data/calibrated/DataFrame'
    fdir = 'figures/freeyaw'

    flex = ['0405_run_263_7.0ms_dc0.4_flexies_freeyaw_highrpm',
            '0405_run_264_7.0ms_dc0.6_flexies_freeyaw_highrpm',
            '0405_run_265_8.0ms_dc0.0_flexies_freeyaw_highrpm',
            '0405_run_266_8.0ms_dc0.4_flexies_freeyaw_highrpm',
            '0405_run_267_8.0ms_dc0.6_flexies_freeyaw_highrpm',
            '0405_run_269_9.0ms_dc0_flexies_freeyaw_highrpm',
            '0405_run_275_9.0ms_dc0.4_flexies_freeyaw_highrpm']

    swep = ['0410_run_325_8ms_dc0_samoerai_freeyaw_highrpm',
            '0410_run_326_8ms_dc0.4_samoerai_freeyaw_highrpm',
            '0410_run_330_9ms_dc1_samoerai_freeyaw_highrpm',
            '0213_run_131_7.0ms_dc1_samoerai_freeyawplaying_pwm1000_highrpm',
            ]

    stif = ['0413_run_414_8ms_dc0_stiffblades_freeyaw',
            '0413_run_418_8ms_dc0.6_stiffblades_freeyaw']

    plt.figure('rpm, swept blade')
    for i, fname in enumerate(swep):
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')
        plt.plot(df.time, df.rpm, label=i)
    plt.legend(loc='best')
    plt.grid()

    plt.figure('yaw, swept blade')
    for i, fname in enumerate(swep):
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')
        plt.plot(df.time, df.yaw_angle, label=i)
    plt.legend(loc='best')
    plt.grid()

    plt.figure('rpm, flex blade')
    for i, fname in enumerate(flex):
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')
        plt.plot(df.time, df.rpm, label=i)
    plt.legend(loc='best')
    plt.grid()

    plt.figure('yaw, flex blade')
    for i, fname in enumerate(flex):
        df = pd.read_hdf(os.path.join(fpath, fname+'.h5'), 'table')
        plt.plot(df.time, df.yaw_angle, label=i)
    plt.legend(loc='best')
    plt.grid()

    # both have some overshoot in response
    f = '0405_run_264_7.0ms_dc0.6_flexies_freeyaw_highrpm'
    dff = pd.read_hdf(os.path.join(fpath, f+'.h5'), 'table')
    s = '0410_run_330_9ms_dc1_samoerai_freeyaw_highrpm'
    dfs = pd.read_hdf(os.path.join(fpath, s+'.h5'), 'table')
    plt.figure('rpm, flex vs swept')
    plt.plot(dff.time, dff.rpm, label='flex')
    plt.plot(dfs.time, dfs.rpm, label='swept')
    plt.legend(loc='best')
    plt.grid()

    plt.figure('yaw, flex vs swept')
    plt.plot(dff.time, dff.yaw_angle, label='flex')
    plt.plot(dfs.time, dfs.yaw_angle, label='swept')
    plt.legend(loc='best')
    plt.grid()



# good speedup case, also speedup in high rpm modus when in yaw error!
# 0405_run_270_9.0ms_dc0_flexies_freeyaw_spinupyawerror

if __name__ == '__main__':

    dummy = None

#    df_index = pd.read_hdf('database/db_index_symlinks_all.h5', 'table')
#    df_mean = pd.read_hdf('database/db_stats_symlinks_all_mean.h5', 'table')
#    df_std = pd.read_hdf('database/db_stats_symlinks_all_mean.h5', 'table')
