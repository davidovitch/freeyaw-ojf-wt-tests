# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:38:18 2016

@author: dave
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import scipy.interpolate as interpolate
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd


def torque_power_at_ohm(Rset):
    qlist, rpmlist, plist, efflist = [], [], [], []
    fname = 'data/model/generator-windbluepower-st-540.csv'
    df = pd.read_csv(fname, sep=',')
    for k in [49, 117, 209, 275, 365, 490, 650, 870]:
        sel = df[df['W [rpm]']==k]
        qs = sel['tau_input [N-m]'].values
        rs = sel['Rsetting [ohm]'].values
        ps = sel['P_output [W]'].values
        effs = sel['Eff [-]'].values
        q = float(interpolate.griddata(rs, qs, Rset))
        p = float(interpolate.griddata(rs, ps, Rset))
        eff = float(interpolate.griddata(rs, effs, Rset))
        rpmlist.append(k)
        qlist.append(q)
        plist.append(p)
        efflist.append(eff)
    return np.array(qlist), np.array(plist), np.array(rpmlist), np.array(efflist)


def rpm2torque(rpms, ohms):
    """Given the RPM and resistance setting, return the torque and electrical
    power output. Based on windbluepower.com provided data. Re-used with their
    permission.

    Return
    ------

    q : shaft torque

    p : output electrical power

    e : effeciency
    """
    qlist, plist, efflist = [], [], []
    for rpm, ohm in zip(rpms, ohms):
        qs, ps, rpms, effs = torque_power_at_ohm(ohm)
        qlist.append(interpolate.griddata(rpms, qs, rpm))
        plist.append(interpolate.griddata(rpms, ps, rpm))
        efflist.append(interpolate.griddata(rpms, effs, rpm))
    return np.array(qlist), np.array(plist), np.array(efflist)


def plot_windbluepower_st_540():
    """Plot the data and add some other constant resistance values to it.
    """

    q_rmax, p_rmax, rpm_rmax, emax = torque_power_at_ohm(28)
    q_rmin, p_rmin, rpm_rmin, emin = torque_power_at_ohm(11)

    # headers:
    # Run #,W [rpm],Rsetting [ohm],Voltage [V],Current [A],tau_input [N-m]
    # P_output [W],Eff [-]
    fname = 'data/model/generator-windbluepower-st-540.csv'
    df = pd.read_csv(fname, sep=',')
    rad = df['W [rpm]'].values*np.pi/30.0
    rpm = df['W [rpm]'].values
    Qc = df['tau_input [N-m]']
    R = df['Rsetting [ohm]']

    xi = np.linspace(rpm.min(), rpm.max(), 200)
    yi = np.linspace(Qc.min(), Qc.max(), 200)
    # grid the data
    zi = mpl.mlab.griddata(rpm, Qc, R, xi, yi, interp='linear')
    # consider switching to: matplotlib.tri.Triangulation or
    # matplotlib.tri.TriInterpolator, see: matplotlib.org/api/tri_api.html

    plt.figure()
#    plt.contour(xi, yi, zi,6, colors='k') #, vmax=35, vmin=0)

    for r in range(5, 30, 1):
        q_, p_, rpm_ = torque_power_at_ohm(r)
        plt.plot(rpm_, q_, 'k--+', alpha=0.6)

    plt.plot(rpm_rmax, q_rmax, 'r-', label='R=28 (dc0)')
    plt.plot(rpm_rmin, q_rmin, 'b-', label='R=11 (dc1)')
    plt.legend(loc='best')

    # slope, intercept, r_value, p_value, std_err
    # regress : ndarray(len(x)-samples, 5)
    # 2nd dimension holds: slope, intercept, r_value, p_value, std_err
#    regress[i,:] = sp.stats.linregress(x[i:i1], y=y[i:i1])


def plot_windblue():
    """
    Make some Torque, efficiency plots from the wind blue power data
    """
    # data originally obtained from:
    # http://www.windbluepower.com/
    # Permanent_Magnet_Alternator_Wind_Blue_Low_Wind_p/dc-540.htm
    # TODO: can I trace back the original download link?
    fname = 'data/model/generator-windbluepower-st-540.csv'
    # headers:
    # Run #,W [rpm],Rsetting [ohm],Voltage [V],Current [A],tau_input [N-m]
    # P_output [W],Eff [-]
    iff = 5
    iP = 4
    iQ = 3
    iR = 0
    data = np.genfromtxt(fname, delimiter=',', dtype=np.float64,
                         skip_header=1, usecols=range(2,8))
    rpm = np.genfromtxt(fname, delimiter=',', dtype=np.int16,
                         skip_header=1, usecols=[1])
    rad = rpm*np.pi/30.0
    Qc = data[:,iQ]/rad

    # selection based on R setting? No, the whole range isn't tested at all
    # rotor speeds

    # plot RPM vs torque constant
#    plt.figure()
#    plt.plot(rpm, Qc, 'r+')
#    reds = plt.get_cmap("Reds")
#    plt.scatter(rpm, Qc, c=data[:,iR], s=20.0*data[:,iR], cmap=reds,
#                vmin=0, vmax=35)
#    plt.xlabel('RPM')
#    plt.ylabel('Torque Constant')
#    plt.grid()
#
#    plt.figure()
#    plt.plot(rpm, data[:,iQ], 'bo')
#    plt.xlabel('RPM')
#    plt.ylabel('Torque')
#    plt.grid()
#
#    plt.figure()
#    plt.plot(data[:,iR], data[:,iQ], 'ks')
#    plt.xlabel('Resistance')
#    plt.ylabel('Torque')
#    plt.grid()

    # ------------------------------------------------------------------------
    # torque vs RPM at constant resistance setting
    # ------------------------------------------------------------------------
    # because not always the same resistance settings are used, interpolate
    # the data to a uniform grid. Higher RPMs are only tested for higher
    # resistance settings. So take the lowest at 490 RPM and interpolate
    # to that value for the other lower RPMs
    rpmlist, Qlist, rpmlist21, Qlist21 = windblue_measurements()

    xi = np.linspace(rpm.min(), rpm.max(), 200)
    yi = np.linspace(data[:,iQ].min(), data[:,iQ].max(), 200)
    # grid the data
    zi = mpl.mlab.griddata(rpm,data[:,iQ],data[:,iR],xi,yi,interp='nn')

    figpath = '/home/dave/PhD/Projects/OJF/SmallTurbine/'
    figname = 'sta-540-torque-rpm'
    figsize_x = plotting.TexTemplate.pagewidth*0.49
    figsize_y = plotting.TexTemplate.pagewidth*0.55
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                   wstop_cm=1.5, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    # the manually interpollated torque curve at constant R setting
    ax1.plot(rpmlist, Qlist, 'r--', label=r'$R=10.5 \Omega$')
    ax1.plot(rpmlist21, Qlist21, 'b--', label=r'$R=21.0 \Omega$')
    ax1.legend(loc='lower right')
    # the actual measurements
#    ax1.plot(rpm, data[:,iQ], 'r+')
#    reds = plotting.mpl.cm.get_cmap("Reds")
#    ax1.scatter(rpm, data[:,iQ], c=data[:,iR], s=10.0*data[:,iR])#, cmap=reds)

    # contour the gridded data, plotting dots at the nonuniform data points
    # draw the contour lines
    CT = ax1.contour(xi,yi,zi,6, colors='k') #, vmax=35, vmin=0)
    ax1.clabel(CT, fontsize=7*scale, inline=1, fmt='%1.0f')
    title = 'Generator torque for different\nload settings (contours in Ohm)'
    title += '\nWindbluepower measurements'
    ax1.set_title(title)
    ax1.set_ylabel('torque [Nm]')
    ax1.set_xlabel('rotor speed [rpm]')
    ax1.set_xlim([0, 1000])
    ax1.grid(True)
    pa4.save_fig()

    # ------------------------------------------------------------------------
    # torque constant plot
    # ------------------------------------------------------------------------

    figpath = '/home/dave/PhD/Projects/OJF/SmallTurbine/'
    figname = 'sta-540-torque-constant'
    figsize_x = plotting.TexTemplate.pagewidth*0.49
    figsize_y = plotting.TexTemplate.pagewidth*0.6
    scale = 1.8
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.7, wsright_cm=1.0,
                   wstop_cm=1.0, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    # the interpolated result didn't look right: the offgrid points had a range
    # that by far exceeded the R settings
#    xi = np.linspace(rpm.min(), rpm.max(), 20)
#    yi = np.linspace(Qc.min(), Qc.max(), 20)
#    # grid the data
#    zi = plotting.mpl.mlab.griddata(rpm,Qc,data[:,iR],xi,yi,interp='nn')
#    # contour the gridded data, plotting dots at the nonuniform data points
#    # draw the contour lines
#    ct = ax1.contourf(xi,yi,zi,10, cmap=plotting.mpl.cm.rainbow)#,
#                      #vmax=35, vmin=0)
#    pa4.fig.colorbar(plotting.mpl.cm.rainbow) # draw colorbar
    # plot data points
#    ax1.scatter(rpm, Qc, s=data[:,iR], marker='o')

    reds = plotting.mpl.cm.get_cmap("Reds")
    ax1.scatter(rpm, Qc, c=data[:,iR], s=10.0*data[:,iR], cmap=reds)
    ax1.grid()
    ax1.set_xlim([0, 1200])
    ax1.set_xlabel('RPM')
    ax1.set_ylabel(r'torque constant $\frac{Nm}{rad/s}$')
    title = 'Torque constants for different\ngenerator loads '
    title += '(circle radius)'
    ax1.set_title(title)

    pa4.save_fig()

#    ax1.plot(rpm, data[:,iQ]/rad, 'r+')


def windblue_measurements():
    """
    just return the torque/rpms for R=10.5 and R=21 of the windbluepower.com
    measurements
    """

    fname = 'data/model/generator-windbluepower-st-540.csv'
    # headers:
    # Run #,W [rpm],Rsetting [ohm],Voltage [V],Current [A],tau_input [N-m]
    # P_output [W],Eff [-]
    iff = 5
    iP = 4
    iQ = 3
    iR = 0
    data = np.genfromtxt(fname, delimiter=',', dtype=np.float64,
                         skip_header=1, usecols=range(2,8))
    rpm = np.genfromtxt(fname, delimiter=',', dtype=np.int16,
                         skip_header=1, usecols=[1])
    rad = rpm*np.pi/30.0
    Qc = data[:,iQ]/rad

    iRmin = data[rpm.__eq__(490), iR].argmin()
    Rset = data[rpm.__eq__(490), iR][iRmin]
    Qlist = []
    rpmlist = []
    # and for all other RPMs, find the (interpolated) performance for that R
    # setting
    for k in [49, 117, 209, 275, 365]:
        qs = data[rpm.__eq__(k), iQ]
        rs = data[rpm.__eq__(k), iR]
        qnew = float(interpolate.griddata(rs, qs, Rset))
        Qlist.append(qnew)
        rpmlist.append(k)
    Qlist.append(data[rpm.__eq__(490), iQ][iRmin])
    rpmlist.append(490)

    Qlist21, rpmlist21 = [], []
    # make another list at 21 ohm
    Rset = 21.0
    for k in [49, 117, 209, 275, 365, 490, 650, 870]:
        qs = data[rpm.__eq__(k), iQ]
        rs = data[rpm.__eq__(k), iR]
        qnew = float(interpolate.griddata(rs, qs, Rset))
        Qlist21.append(qnew)
        rpmlist21.append(k)

    return rpmlist, Qlist, rpmlist21, Qlist21

if __name__ == '__main__':
    dummy = None
