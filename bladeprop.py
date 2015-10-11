# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:56:21 2011

@author: dave
"""

import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate as integrate
#from scipy import optimize
import math
import warnings

import pylab as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure
import matplotlib as mpl

import HawcPy
from crosssection import properties as crosprop
import materials
import Simulations as sim
from misc import find0
import plotting

sti = HawcPy.ModelData.st_headers

# numpy print options:
np.set_printoptions(precision=6, suppress=True)


def coord_continuous(coord, res=100000, method=None):
    """
    Continuous series of coordinate points
    ======================================

    The original format of the coordinates is not increasing and
    continious. Convert to a two series (upper and lower serface) with
    continious properties. If method is properly defined, the coordinates
    will also be interpolated onto a uni-spaced high resolution grid.

    Main assumption on the data is that the coordinates are ordered. The
    points are described following the circular motion: starting from the
    LE, going over either pressure or suction side to the TE and back over
    the other surface. This method will fail if the data has a more
    random character.

    Parameters
    ----------

    coord : ndarray(n,2)
        Array holding the x/c and y/c coordinates respectivaly.

    res : int, default=100000
        Integer specifying how many x/c points should be maintained for
        the higher resultion coord array. Value ignored when method=None.
        Note that a leading edge usually already has a high resolution
        hence the relative high default resultion.

    method : str, default=None
        String specifying the interpolation method to go from low to high
        resoltion. See scipy.interpolate.griddata for more info. Possible
        values are 'cubic', 'linear' or 'nearest'. If None no interpolation
        to a higher resolution grid will be carried out

    Returns
    -------

    coord_up : ndarray(n_new1, 2)
        Array holding the x/c and y/c coordinates of the upper surface,
        and where n = n_new1 + n_new2.

    coord_low : ndarray(n_new2, 2)
        Array holding the x/c and y/c coordinates of the upper surface,
        and where n = n_new1 + n_new2

    """

    # TODO: add check on continuity: smoothness of the function
    # see if the first, second, third derivatives are within a certain
    # range

    xval = coord[:,0]
    yval = coord[:,1]

    # split up in two increasing sets
    xval_delta = xval[1::] - xval[0:-1]
    # values are >0 when x is increasingxval_delta = xval[1::] - xval[0:-1]
    # the last value falls of now, but it should be the same as the
    # previous value (n-1)
    xval_sel = xval_delta.__ge__(0)
    p1_x = xval[np.append(xval_sel, xval_sel[-1])]
    p1_y = yval[np.append(xval_sel, xval_sel[-1])]

    # values are <=0 when decreasing, and switch the array order to get
    # increasing steps on the x axis
    xval_sel = xval_delta.__lt__(0)
    p2_x = xval[np.append(xval_sel, xval_sel[-1])][::-1]
    p2_y = yval[np.append(xval_sel, xval_sel[-1])][::-1]

    # split in upper and lower surface
    if p1_y.mean() < 0:
        # now close the circle by adding the first point of the lower
        # in front of the upper surface. Note that continuity has to
        # be preserved!
        # By doing so, both upper and lower surface start at the same
        # point at the leading edge!
        p2_x = np.append(p1_x[0], p2_x)
        p2_y = np.append(p1_y[0], p2_y)
        coord_low = np.array([p1_x, p1_y]).transpose()
        coord_up  = np.array([p2_x, p2_y]).transpose()

    else:
        # now close the circle by adding the first point of the lower
        # in front of the upper surface. Note that continuity has to
        # be preserved!
        p1_x = np.append(p2_x[0], p1_x)
        p1_y = np.append(p2_y[0], p1_y)
        coord_low = np.array([p2_x, p2_y]).transpose()
        coord_up  = np.array([p1_x, p1_y]).transpose()

    # alternative, where all is in one coord array
    # xval_mod = np.append(p1_x, p2_x)
    # yval_mod = np.append(p1_y, p2_y)
    # coord_mod = np.array([xval_mod, yval_mod]).transpose()

    # if applicable, increase resolution.

    if res < 2*len(coord_up) and type(method).__name__ == 'str':
        raise ValueError, 'res needs to be at least 2x len(coord1_up)'

    if method in ['cubic', 'linear', 'nearest']:

        # convert to float128, to be sure not to lose anything when
        # interpolating to a very high resolution grid
        coord_up = np.array(coord_up, dtype=np.float128)
        coord_low = np.array(coord_low, dtype=np.float128)

        # the first point is close to but not exactly 0
        # both up and lower surface share the beginning point
        xval_hr = np.linspace(coord_up[0,0],np.float128(1),res)
        # upper surface
        xval = coord_up[:,0].__copy__()
        yval = coord_up[:,1].__copy__()
        coord_up = scipy.zeros((len(xval_hr),2), dtype=np.float128)
        # make first xval a tiny bit smaller than xval_hr so it will
        # include the first point

        coord_up[:,1] = scipy.interpolate.griddata(xval, yval, xval_hr,
                                              method=method)
        coord_up[:,0] = xval_hr

        # lower surface
        xval = coord_low[:,0].__copy__()
        yval = coord_low[:,1].__copy__()
        coord_low = scipy.zeros((len(xval_hr),2), dtype=np.float128)
        coord_low[:,1] = scipy.interpolate.griddata(xval, yval, xval_hr,
                                              method=method)
        coord_low[:,0] = xval_hr

    # check if the two series realy are continuous data sets:
    # delta = n - n_-1: all should be possitive for increasing x
    x_delta_up = coord_up[1::,0] - coord_up[0:-1,0]
    if not x_delta_up.__gt__(0).all():
        raise UserWarning, 'upper coord: only increasing x allowed'

    x_delta_low = coord_low[1::,0] - coord_low[0:-1,0]
    if not x_delta_low.__gt__(0).all():
        raise UserWarning, 'lower coord: only increasing x allowed'

    # make sure the interpolation didn't created any nan's
    if not np.all(np.isfinite(coord_up)):
        raise UserWarning, 'found nan\'s in coord_up, check interpolation'
    if not np.all(np.isfinite(coord_low)):
        raise UserWarning, 'found nan\'s in coord_low, check interpolation'

    return coord_up, coord_low


# TODO: implement the list of float on t_new
def interp_airfoils(coord1, t1, coord2, t2, t_new, verplot=False, verbose=False):
    """
    Interpolate two airfoils with respect to thickness
    ==================================================

    Weight two different airfoils with respect to their max thickness.

    Parameters
    ----------

    coord1 : ndarray(n,2)
        Array holding the x/c and y/c coordinates respectivaly.

    t1 : float

    coord2 : ndarray(n,2)
        Array holding the x/c and y/c coordinates respectivaly.

    t2 : float

    t_new : list of floats
        Desired thickness for the new airfoil. This value should always
        lay in between the thickness of airfoil1 and airfoil2.
        Necessary condition for t_new: t1 > t_new > t2 (root-mid-tip).
        For each list entry a new weighted airfoil will be created.

    """
    # TODO: also allow array input, so you could interpollate any airfoil
    # not in the database as well

    # data check: t_new has to be inbetween t1 and t2!
    if not t_new <= t1 or not t_new >= t2:
        raise ValueError, 'wrong size for t_new: t1 > t_new > t2'

    # coord_continuous will split up in 2 continuous functions, one
    # for lower and one for upper surface. Next it will interpolate
    # the grid to a uni-spaced high resolution grid
    coord1_up, coord1_low = coord_continuous(coord1, method='linear')
    coord2_up, coord2_low = coord_continuous(coord2, method='linear')

    # and now interpolate with respect to thickness, meaning for each
    # x/c point we have a pair of (t1,y1) and (t2,y2) -> (t_new,y_new)
    # t is the xvalue, y/c is the yvalue for a given x/c position

    # TODO:
    # for tt in t_new:
        # save coordinates as an array in root.coord and make a note
        # in root.coord.coord_tbl
        # airfoil name: airfoil1_frac1_airfoil2_frac2

    # or simply do that by weighting
    frac1 = 1.-((t1-t_new) / (t1-t2))
    frac2 = 1.-((t_new-t2) / (t1-t2))

    if verbose:
        print 't1   :', t1
        print 't_new:', t_new
        print 't2   :', t2
        print 'frac1:', frac1, 'frac1:', frac2

    # both airfoil1 and airfoil2 have the same number of points,
    # since coord_continuous will take create an equal grid for both
    # x/c grid remains the same (non dimensional chord!)
    coord_new_up = coord2_up.__copy__()
    coord_new_low = coord2_low.__copy__()
    # and weight y/c accorindg to relative thickness
    coord_new_up[:,1] = frac1*coord1_up[:,1]  + frac2*coord2_up[:,1]
    coord_new_low[:,1]= frac1*coord1_low[:,1] + frac2*coord2_low[:,1]

    # check that coord2 and coord2_int are similar
    if verplot:
        plt.figure(1)
        # airfoil 1
        plt.plot(coord1[:,0], coord1[:,1], 'co', label='c1')
        plt.plot(coord1_up[:,0], coord1_up[:,1], 'c--', label='c1 up')
        plt.plot(coord1_low[:,0], coord1_low[:,1], 'b--', label='c1 low')
        # arfoil 2
        plt.plot(coord2[:,0], coord2[:,1], 'ro', label='c2')
        plt.plot(coord2_up[:,0], coord2_up[:,1], 'r--', label='c2 up')
        plt.plot(coord2_low[:,0], coord2_low[:,1], 'k--', label='c2 low')
        # interpolated airfoil
        plt.plot(coord_new_up[:,0], coord_new_up[:,1],'g-',label='c3 up')
        plt.plot(coord_new_low[:,0],coord_new_low[:,1],'m-',label='c3 low')

        plt.legend()
        plt.show()

    return coord_new_up, coord_new_low

def cross_prop(coord_up, coord_low, tests=False, verplot=False):
    """
    Cross sectional airfoil properties
    ==================================

    Calculate the properties of the 2D airfoil cross section. Assume it
    is a full enclosed 2D shape. Following properties are supported:
        * Ixx and Iyy wrt neutral bending axis
        * location of centroid (neutral axis coordinates) wrt to LE
        * cross sectional area

    The upper and lower coordinates should have the same x coordinates,
    as given by coord_continuous() and interp_airfoils().

    This method is a piecewise linear integration method that also
    calculates the second moment of inertia wrt the neutral x and y axis.

    Note that for the mass moment of inertia we have: Im_xx = Ixx*rho

    Parameters
    ----------

    coord_up : ndarray(n,2)
        Upper surface coordinates of the airfoil, moving from LE to TE

    coord_low : ndarray(n,2)
        Lower surface coordinates of the airfoil, moving from LE to TE

    tests : boolean, default=False

    Returns
    -------

    A

    x_na

    y_na

    Ixx

    Iyy


    """

    # TODO: data checks
    #   continuity: does it? I think it can deal with any curvature
    #   better and seperate tests and result verification

    # make sure the grids for both upper and lower surface are the same
    if not np.allclose(coord_up[:,0], coord_low[:,0]):
        msg = 'coord_up and low need to be defined on the same grid'
        raise ValueError, msg
   # TODO: if not, interpolate and fix it on equal x grids

    # NUMERICAL APPROACH: split up into blocks. Since we already
    # interpolated the coordinates to a high res grid, this approach
    # should be sound. The upper block is a triangle, lower is just a
    # rectangle (see drawings)

    # even though they are identical, put in one array to have the
    # correct summation over all the elements of upper and lower curve
    x = np.ndarray((len(coord_up),2), dtype=np.float128)
    x[:,0] = coord_up[:,0]
    x[:,1] = coord_low[:,0]
    x1 = x[:-1,:]

    # for convience, put both y's in one array: [x, up, low]
    y = np.ndarray((len(coord_up),2), dtype=np.float128)
    y[:,0] = coord_up[:,1]
    y[:,1] = coord_low[:,1]
    # y1, first y value of each element
    y1 = y[:-1,:]

    # delta's define each element
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)

    # we have the top triangle (A) and the bottom rectangle (B)
    # This approach goes right automatically. When dy<0, B is too big
    # (because y1 > y2) and dA_a gets negative. Other way around for lower
    # surface. Note that we need to be consistent in this approach all
    # the way through
    dA_a = dx * dy * 0.5
    dA_b = y1 * dx
    # reverse sign for lower surface area's, they are negative if under
    # the y axis=0
    dA_a[:,1] *= -1.
    dA_b[:,1] *= -1.
    # total area is then
    A = np.sum(dA_a + dA_b)

    # cg positions for each block with respect to (x,y)=(0,0)
    # upper and lower share the same x_cg positions
    x_cg_b = x1 + (dx*0.5)
    y_cg_b = y1*0.5

    # inclined side is directed towards y axis, so 2x/3 instead of x/3.
    x_cg_a = (2.*dx/3.) + x1
    # when on the lower side this reverses, but since then y1 > y2,
    # and y1 + dy/3 becomes y2 + 2*dy/3. Check the figures for better
    # and more correct understanding/description of why it goes right.
    y_cg_a = (dy/3.) + y1

    # find the neutral axis wrt (0,0)
    x_na = np.sum( (dA_a*x_cg_a) + (dA_b*x_cg_b) ) /A
    y_na = np.sum( (dA_a*y_cg_a) + (dA_b*y_cg_b) ) /A

    # Moments of inertia for each piece around local cg
    # since dy can switch sign, only consider absolute value
    Ixx_cg_a = np.abs(dx*dy*dy*dy/36.)
    Iyy_cg_a = np.abs(dx*dx*dx*dy/36.)
    Ixx_cg_b = np.abs(dx*y1*y1*y1/12.)
    Iyy_cg_b = np.abs(dx*dx*dx*y1/12.)

    # the Steiner terms: distance local cg to neutral axis
    Ixx_steiner_a = dA_a*(y_na-y_cg_a)*(y_na-y_cg_a)
    Ixx_steiner_b = dA_b*(y_na-y_cg_b)*(y_na-y_cg_b)
    Iyy_steiner_a = dA_a*(x_na-x_cg_a)*(x_na-x_cg_a)
    Iyy_steiner_b = dA_b*(x_na-x_cg_b)*(x_na-x_cg_b)

    # and finally the moments of inertia with respect to the neutral axis
    Ixx = np.sum(Ixx_cg_a + Ixx_steiner_a + Ixx_cg_b + Ixx_steiner_b)
    Iyy = np.sum(Iyy_cg_a + Iyy_steiner_a + Iyy_cg_b + Iyy_steiner_b)

    if tests:
        # ANALYTICAL APPROACH
        # find enclosed airfoil area, integrate two curves
        # Since we have a closed curve, negative y/c values for the upper
        # side will result be corrected by the exact same amount of area
        # surplus on the lower side
        area_up  = integrate.trapz(coord_up[:,1], x=x[:,0])
        area_low = integrate.trapz(coord_low[:,1], x=x[:,0])
        area = abs(area_up) + abs(area_low)

        # area moments, moment of inertia
        # wrt to neutral line, see also:
        # Area and Bending Inertia of Airfoil Sections in Library
        cc_up = np.array(coord_up[:,1])
        cc_low = np.array(coord_low[:,1])
        # for convenience, make shorter notation for the squared coord.
        yyu = cc_up*cc_up
        yyl = cc_low*cc_low
        # neutral bending line
        neutral = (0.5/area)*integrate.trapz(yyu-yyl, x=x[:,0])
        # moment of inertia with respect to neutral line
        term = np.power(cc_up-neutral,3) - np.power(cc_low-neutral,3)
        Ixx_anal  = (1./3.)*integrate.trapz(term, x=x[:,0])
        # Ixx upper analytical
        term = np.power(cc_up-neutral,3)
        Ixx_up_anal  = (1./3.)*integrate.trapz(term, x=x[:,0])
        # Ixx numerical upper only
        Ixx_up = np.sum( Ixx_cg_a[:,0] + Ixx_steiner_a[:,0] \
                    + Ixx_cg_b[:,0] + Ixx_steiner_b[:,0])

        # only upper side
        A_up = np.sum(dA_a[:,0] + dA_b[:,0])
        y_na_up = np.sum( (dA_a[:,0]*y_cg_a[:,0]) \
                        + (dA_b[:,0]*y_cg_b[:,0]) )/A_up
        y_na_up_anal = (0.5/area_up)*integrate.trapz(yyu, x=x[:,0])

        print 'Comparing analytical and numerical approaches'
        print '   area:', np.allclose(area, A)
        print '   y_na:', np.allclose(neutral, y_na)
        print '   A_up:', np.allclose(area_up, A_up)
        print 'y_na_up:', np.allclose(y_na_up, y_na_up_anal)
        print '    Ixx:', np.allclose(Ixx, Ixx_anal)
        print Ixx
        print Ixx_anal
        print (1-(Ixx/Ixx_anal))
        print ' Ixx_up:', np.allclose(Ixx_up, Ixx_up_anal)
        print format(Ixx_up, '2.15f')
        print format(Ixx_up_anal, '2.15f')
        print (1-(Ixx_up/Ixx_up_anal))

    # check that coord2 and coord2_int are similar
    if verplot:
        plt.figure(1)
        # airfoil
        plt.plot(x[:,0], y[:,0], 'c--', label='up')
        plt.plot(x[:,0], y[:,1], 'b--', label='low')
        # x_na
        plt.hlines(y_na,0,1, label='y_na', colors='r')
        plt.vlines(x_na,y.min(),y.max(), label='x_na', colors='k')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.plot(y_cg_a[:,0], label='y_cg_a up')
        plt.plot(y_cg_a[:,1], label='y_cg_a low')
        plt.plot(y_cg_b[:,0], label='y_cg_b up')
        plt.plot(y_cg_b[:,1], label='y_cg_b low')
        plt.hlines(y_na,0,len(y_cg_a))
        plt.legend()
        plt.show()

        plt.figure(3)
        plt.plot(x_cg_a, label='x_cg_a')
        plt.plot(x_cg_b, label='x_cg_b')
        plt.legend()
        plt.show()

    return A, x_na, y_na, Ixx, Iyy

def S823_coordinates():
    """return coordinates and max t/c
    """
    return np.loadtxt('data/S823.dat', skiprows=1), 21.0

def S822_coordinates():
    """return coordinates and max t/c
    """
    return np.loadtxt('data/S822.dat', skiprows=1), 16.0

class Blade:
    """
    Create a Blade based on airfoil coordinates
    ===========================================
    """

    def __init__(self, **kwargs):
        """
        """

        self.figpath = kwargs.get('figpath', '')
        self.plot = kwargs.get('plot', False)

        # one inch expressed in centimeters
        self.oneinch = 2.54
        self.units = 'SI'

    def scale_figure(self, xlim, ylim):
        """
        Set figure size so that when printing 1cm = 1cm

        This should be a very generic plotting method/class.

        By default it is assumed that the units for xlim, ylim are in cm!
        """
        if self.units is not 'SI':
            raise UserWarning, 'unsupported units for scaling plot'

        # unit: inches. 1 inch =  2.54 cm

        # define the length based on the limits
        xlength = xlim[1] - xlim[0]
        ylength = ylim[1] - ylim[0]
        # scale the figure size accordingly
        if self.units == 'SI':
            figsize_x = xlength/self.oneinch
            figsize_y = ylength/self.oneinch

        return figsize_x, figsize_y



    def _plot_cross_section(self, section_nr, **kwargs):
        """
        Plot Airfoil cross section
        ==========================

        Generate a plot of the given airofil coordinates.

        """

        points = kwargs.get('points', False)
        if points:
            pointsx = points[0]
            pointsy = points[1]

        fontsize = kwargs.get('fontsize', 'medium')
        figsize_x = kwargs.get('figsize_x', 6)
        figsize_y = kwargs.get('figsize_y', 4)
        title = kwargs.get('title', '')
        dpi = kwargs.get('dpi', 200)

#        wsleft = kwargs.get('wsleft', 0.15)
#        wsbottom = kwargs.get('wsbottom', 0.15)
#        wsright = kwargs.get('wsright', 0.95)
#        wstop = kwargs.get('wstop', 0.90)
#        wspace = kwargs.get('wspace', 0.2)
#        hspace = kwargs.get('hspace', 0.2)

        wsleft = kwargs.get('wsleft', 0.0)
        wsbottom = kwargs.get('wsbottom', 0.0)
        wsright = kwargs.get('wsright', 1.0)
        wstop = kwargs.get('wstop', 1.0)
        wspace = kwargs.get('wspace', 0.0)
        hspace = kwargs.get('hspace', 0.0)

        textbox = kwargs.get('textbox', None)
        xlabel = kwargs.get('xlabel', 'x [m]')
        ylabel = kwargs.get('ylabel', 'y [m]')

        # for half chord coordinates
        xlim = kwargs.get('xlim', [-0.07,  0.06])
        ylim = kwargs.get('ylim', [-0.025, 0.020])
        # for the LE coordinates
        xlim = kwargs.get('xlim', [-0.01,  0.12])
        ylim = kwargs.get('ylim', [-0.025, 0.020])

        # scale the image
        xlength = (xlim[1] - xlim[0])*100.
        ylength = (ylim[1] - ylim[0])*100.
        figsize_x = xlength/self.oneinch
        figsize_y = ylength/self.oneinch

        figpath = kwargs.get('figpath', self.figpath)

        fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)

        mpl.rc('font',**{'family':'lmodern','serif':['Computer Modern'], \
                         'monospace': ['Computer Modern Typewriter']})
        mpl.rc('text', usetex=True)
        mpl.rcParams['font.size'] = 11

        # add_subplot(nr rows nr cols plot_number)
        ax1 = fig.add_subplot(111)

        # original linewidth was set to 0.3
        ax1.plot(self.coord_up[:,0], self.coord_up[:,1],
                 'k-', label='upper', linewidth=1.1)
        ax1.plot(self.coord_low[:,0],  self.coord_low[:,1],
                 'k-', label='lower', linewidth=1.1)

        # insert the plate
        # plate[point on plate, coordinate(x,y)]
        ax1.plot(self.plate[:,0], self.plate[:,1], 'k-', linewidth=1.1)

        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                            top=wstop, wspace=wspace, hspace=hspace)

        # set the textboxres=None
        textbox = 't/c=' + format(self.tc, '1.02f') + '\%'
        textbox += ' r=' + format(self.r*1000, '1.0f') +'mm'
        textbox += ' c=' + format(self.chord*1000, '1.0f') +'mm'
        if textbox:
            xpos = xlim[0] + ((xlim[1]-xlim[0])/3.)
            ypos = ylim[0]
            ax1.text(xpos, ypos, textbox, fontsize=12, va='bottom',
                     bbox = dict(boxstyle="round",
                     ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
            # textbox with the number of stations
            numberbox = str(section_nr+1)
            ax1.text(xlim[0], ylim[0], numberbox, fontsize=12, va='bottom',
                         bbox = dict(boxstyle="round",
                         ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))

        # plot the neutral axes
        # this should be plotting line, otherwise it not rotated as the
        # rest of the coordinates
        #if x_na:
            #ax1.axvline(x=x_na,linewidth=1,color='k',linestyle='-',aa=False)
        #if y_na:
            #ax1.axhline(y=y_na,linewidth=1,color='k',linestyle='-',aa=False)

        # plot the indicated points as well
        if points:
            ax1.plot(pointsx[0], pointsy[0], 'ro') # strain gauge pos
            ax1.plot(pointsx[1], pointsy[1], 'rs') # cg
            ax1.plot(pointsx[2], pointsy[2], 'r^') # na
            ax1.plot(0, 0, 'kd') # half chord point

        ax1.set_xlabel(xlabel, size=fontsize)
        ax1.set_ylabel(ylabel, size=fontsize)
        title = self.airfoil1 + ' - ' + self.airfoil2
        title += ' t/c=' + format(self.tc, '1.02f') + '\%'
        ax1.set_title(title)

        ax1.xaxis.set_ticks( np.arange(xlim[0], xlim[1], 0.005).tolist() )
        ax1.yaxis.set_ticks( np.arange(ylim[0], ylim[1], 0.005).tolist() )

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        ax1.grid(True)

        figpath += self.airfoil1 + '-' + self.airfoil2
        figpath += '_' + format(section_nr+1, '02.0f')
        figpath += '_tc_'  + format(self.tc, '1.03f')
        figpath += '_c_' + format(self.chord, '1.03f')
        figpath += '_r_' + format(self.r, '1.03f')
        # filter out the dots in the file name, LaTeX doesn't like them
        figpath = figpath.replace('.', '')

        print 'saving: ' + figpath
        fig.savefig(figpath + '.png')
        fig.savefig(figpath + '.eps')
#        fig.savefig(figpath + '.jpg')
#        fig.savefig(figpath + '.svg')
        canvas.close()
        fig.clear()

    def build(self, blade, airfoils, res=None, step=None, plate=False,
              tw1_t=0, tw2_t=0, st_arr_tw=None):
        """
        Create a blade
        ==============

        Create a blade based on radius, chord, t/c and twist distributions,
        and combined with a list of airfoil names (from the database).

        Additionally, cross-sectional paramters are calculated for the full
        2D airfoil sections.

        Parameters
        ----------

        blade : array
            blade[radius, chord, t/c, twist]

        airfoils : list
            [airfoil1 name, coord1, t1, airfoil2 name, coord2, t2]

        res : int, default=None
            If set, the blade input data will be interpolated to a linspace
            with res number of steps. For accurate volume integration results,
            the OJF example with 18 stations showed to be sufficient.
            Higher resolutions are not required.

        step : float, default=None
            Optionaly interpolate blade input data with np.arange using
            step.

        plate : boolean, default=False
            If set to True, the glass fibre sandwich will be included

        tw1_t : float, default=0
            Thickness of the thin walled structural part

        tw1_t : float, default=0
            Thickness of the thin walled structural part

        Returns
        -------

        blade_hr : ndarray(n, 4)
            radial stations, [radius, chord, t/c, twist]

        volume : float

        area : ndarray(n)
            cross sectional area per section

        x_na : ndarray(n)
            Location of the neutral axis for the full airfoil wrt half chord
            point.

        y_na : ndarray(n)
            Location of the neutral axis for the full airfoil wrt half chord
            point.

        Ixx : ndarray(n)
            Moment of inertia wrt neutral axis for the full airfoil section

        Iyy : ndarray(n)
            Moment of inertia wrt neutral axis for the full airfoil section

        st_arr : ndarray(n,19)

        strainpos : ndarray(n,3)
            Position of the strain gauges (r,x,y) expressed in half chord
            coordinates and where r refers to the radial position along the
            blade

        """
        plate_twist = 0

        # should the blade have a high res for better volume estimate?
        # for res=1000 : 0.000838272318841
        # for res= 100 : 0.000838306414751
        # for res=None : 0.000838681278719
        # NO NEED TO INTERPOLATE TO HIGH RES
        if res is None and step is None:
            blade_hr = blade
            hr = blade_hr[:,0]
        elif step is None:
            hr = np.linspace(blade[0,0],blade[-1,0],res)
            blade_hr = scipy.zeros((len(hr),4))
            blade_hr[:,0]= hr
            blade_hr[:,1]= scipy.interpolate.griddata(blade[:,0],blade[:,1],hr)
            blade_hr[:,2]= scipy.interpolate.griddata(blade[:,0],blade[:,2],hr)
            blade_hr[:,3]= scipy.interpolate.griddata(blade[:,0],blade[:,3],hr)
        else:
            # step distribution
            hr = np.arange(blade[0,0],blade[-1,0],step)
            # and add the tip position
            hr = np.append(hr, blade[-1,0])
            blade_hr = scipy.zeros((len(hr),4))
            blade_hr[:,0]= hr
            blade_hr[:,1]= scipy.interpolate.griddata(blade[:,0],blade[:,1],hr)
            blade_hr[:,2]= scipy.interpolate.griddata(blade[:,0],blade[:,2],hr)
            blade_hr[:,3]= scipy.interpolate.griddata(blade[:,0],blade[:,3],hr)

        # TODO: move how the plate is made to another method!!
        # The plate is expressed in LE (0,0) coordinates
        if plate:
            # and corresponding plate:
            plate_chord =    np.linspace(3.5e-2, 1.5e-2, len(hr))
            plate_t_height = np.linspace(1.0e-2, 0.2e-2, len(hr))
            plate_t_width =  np.linspace(2.5e-2, 0.5e-2, len(hr))

            # define the plates position
            plate_twist = -10*np.pi/180.
            plate_x0 = 0.0059
            plate_y0 = 0.0015

            # 6 points on the plate are
            #       2----3
            #       |    |
            # 0-----1    4-----5
            # 6----------------5
            # plate[radial station, point on plate, coordinate(x,y)]
            plate_blade = scipy.zeros((len(hr),7,2))
            # x positions of first points
            plate_blade[:,0,0] = plate_x0
            plate_blade[:,0,1] = plate_y0

            plate_blade[:,1,0] = plate_x0 + plate_chord/2. - plate_t_width/2.
            plate_blade[:,1,1] = plate_y0

            plate_blade[:,2,0] = plate_x0 + plate_chord/2. - plate_t_width/2.
            plate_blade[:,2,1] = plate_y0 - plate_t_height

            plate_blade[:,3,0] = plate_x0 + plate_chord/2. + plate_t_width/2.
            plate_blade[:,3,1] = plate_y0 - plate_t_height

            plate_blade[:,4,0] = plate_x0 + plate_chord/2. + plate_t_width/2.
            plate_blade[:,4,1] = plate_y0

            plate_blade[:,5,0] = plate_x0 + plate_chord
            plate_blade[:,5,1] = plate_y0
            # and back to zero, close the section
            plate_blade[:,6,0] = plate_x0
            plate_blade[:,6,1] = plate_y0

        # structural data for the thin walled section interpollated to hr
        if type(st_arr_tw).__name__ is 'ndarray':
            st_hr = scipy.interpolate.griddata(st_arr_tw[:,0], st_arr_tw, hr)
        else:
            st_hr = False

        area = scipy.zeros(len(blade_hr), dtype=np.float128)
        x_na = scipy.zeros(len(blade_hr), dtype=np.float128)
        y_na = scipy.zeros(len(blade_hr), dtype=np.float128)
        Ixx = scipy.zeros(len(blade_hr), dtype=np.float128)
        Iyy = scipy.zeros(len(blade_hr), dtype=np.float128)
        st_arr = scipy.zeros((len(blade_hr),19), dtype=np.float128)
        strainpos = scipy.zeros((len(blade_hr),3), dtype=np.float128)
        # calculate the volume at each blade radial station
        for k in range(len(blade_hr)):

            self.r = blade_hr[k,0]
            self.tc = blade_hr[k,2]
            twist = blade_hr[k,3]*np.pi/180. + plate_twist
            chord = blade_hr[k,1]
            coord1, t1, = airfoils[1], airfoils[2]
            coord2, t2, = airfoils[4], airfoils[5]
            coord_up, coord_low = interp_airfoils(coord1, t1, coord2, t2, self.tc)
            self.airfoil1 = airfoils[0]
            self.airfoil2 = airfoils[3]

            # dimensionalize: multiply with chord length
            self.coord_up, self.coord_low = coord_up*chord, coord_low*chord
            self.chord = chord

            # we need a different grid for the thin walled concept of
            # crosssection.properties: tw[x,y,t]
            #       2----3
            #       |    |
            # 0-----1    4-----5
            # 6----------------5
            # plate for plotting format, set to zero for no plotting
            self.plate = scipy.zeros((7,2))
            if plate:
                tw1 = np.ndarray((6,3))
                # beam x coordinates of the section
                tw1[:,0] = plate_blade[k,:6,0]
                # beam y coordinates of the section
                tw1[:,1] = plate_blade[k,:6,1]
                # and the thickness for each element is
                tw1[:,2] = tw1_t
                # and number two is from point 0 to point 5
                # scipy.interpolate.griddata(points, values, high_res_points)
                tw2 = np.ndarray(tw1.shape)
                # beam x coordinates of the section
                xtmp = np.linspace(plate_blade[k,0,0], plate_blade[k,5,0], 6)
                tw2[:,0] = xtmp
                # beam y coordinates of the section
                tw2[:,1] = scipy.interp(xtmp, plate_blade[k,[0,5],0],
                                         plate_blade[k,[0,5],1])
                # and the thickness for each element is
                tw2[:,2] = tw2_t
                # put in the plotting format
                self.plate[:,0] = plate_blade[k,:,0]
                self.plate[:,1] = plate_blade[k,:,1]
            else:
                tw1 = scipy.zeros((4,3))
                tw2 = scipy.zeros((4,3))

            # cross sectional area properties
            area[k], x_na[k], y_na[k], Ixx[k], Iyy[k] \
                = cross_prop(self.coord_up, self.coord_low)

            # the other cross sectional analysis
            E   = materials.Materials().cores['PVC_H200'][1]
            rho = materials.Materials().cores['PVC_H200'][15]
            st_arr[k,:],EA,EIxx, EIyy = crosprop(self.coord_up, self.coord_low,
                              tw1=tw1, tw2=tw2, rho=rho, E=E, st_arr_tw=st_hr)
            # put the correct radial position in the st_arr because that is
            # not set in the crosssection.properties method
            st_arr[k,sti.r] = self.r

            # express x_na and y_na wrt half chord point
            # chord length is determined by knowing the TE and LE coordinates
            # LE should be (0,0), but take into account just to be sure
            x = self.coord_up[-1,0] - self.coord_up[0,0]
            y = self.coord_up[-1,1] - self.coord_up[0,1]
            chordlen = math.sqrt(x*x + y*y)
            c2 = chordlen/2.
            cos_a = x/chordlen
            sin_a = y/chordlen
            x_c2 = cos_a*c2
            y_c2 = sin_a*c2
            # and convert to half chord point coordinates
            # in the original format, the chord line is already horizontal
            # no need to apply any transformation
            #theta = math.acos(cos_a)
            #x_na[k] = x_c2*math.cos(theta) - y_c2*math.sin(theta)
            #y_na[k] = x_c2*math.sin(theta) + y_c2*math.cos(theta)
            # the translation of the coordinate system to half chord
            x_na[k] = x_c2 - x_na[k]
            y_na[k] = y_c2 + y_na[k]

            # -----------------------------------------------------------------
            # determine coordinates of the strain gauges, wrt c/2 point
            if plate:
                strain_x = x_c2 - plate_blade[k,2:4,0].mean()
                strain_y = y_c2 + plate_blade[k,2,1]
            else:
                strain_x, strain_y = 0, 0

            # remember, HAWC2 is half chord points as reference,
            # the plot uses LE as reference point
            xcgi = sim.ModelData.st_headers.x_cg
            xnai = sim.ModelData.st_headers.x_e
            ycgi = sim.ModelData.st_headers.y_cg
            ynai = sim.ModelData.st_headers.y_e
            # -----------------------------------------------------------------

            # rotate with the twist angle
            # plot the current cross section
            if self.plot:
                # -------------------------------------------------------------
                # plot in half chord coordinates to have better check

#                # put the three points as follows: [strain, CG, NA]
#                # now all points are in half chord points
#                pointsx = np.array([strain_x, st_arr[k,xcgi], st_arr[k,xnai]])
#                pointsy = np.array([strain_y, st_arr[k,ycgi], st_arr[k,ynai]])
#                # the airfoil coordinates
#                self.coord_up[:,0]   = x_c2 - self.coord_up[:,0]
#                self.coord_low[:,0]  = x_c2 - self.coord_low[:,0]
#                self.coord_up[:,1]  += y_c2
#                self.coord_low[:,1] += y_c2
#                # now we rotate the plate instead of the airfoil
#                x, y = self.plate[:,0], self.plate[:,1]
#                self.plate[:,0], self.plate[:,1] \
#                        = self._tohalfchord(x, y, -twist, x_c2, y_c2)
#                # the strain gauge position now also needs to be rotated
#                pointsx[0], pointsy[0] = self._rotate(pointsx[0], pointsy[0],
#                                                      -twist)
#                # and plot
#                self._plot_cross_section(k, points=[pointsx, pointsy],
#                                         xlim=[-0.07,  0.06],
#                                         ylim=[-0.025, 0.020])
#
#                # for later reference, save in half chord points
#                strainpos[k,:] = [self.r, pointsx[0], pointsy[0]]

                # -------------------------------------------------------------
                # plotting in LE coordinates was done for easy drafting the

                # put the three points as follows: [strain, CG, NA]
                # now already all is in half chord points
                pointsx = np.array([strain_x, st_arr[k,xcgi], st_arr[k,xnai]])
                pointsy = np.array([strain_y, st_arr[k,ycgi], st_arr[k,ynai]])
                # for later reference, save in half chord points
                strainpos[k,:] = [self.r, strain_x, strain_y]

                # cross sections for production
                # rotate the coordinates with the given twist angles
                self.coord_up[:,0], self.coord_up[:,1] = \
                  self._rotate(self.coord_up[:,0], self.coord_up[:,1], twist)
                self.coord_low[:,0], self.coord_low[:,1] = \
                  self._rotate(self.coord_low[:,0], self.coord_low[:,1],twist)
                # for printing, put back to LE coordinates
                px_le = x_c2 - pointsx
                py_le = pointsy - y_c2
                # rotate for the cross sectional plot, only for cg, na and not
                # for the strain gauges, since they are on the beam and that
                # one is not rotating
                px_le[1:],py_le[1:] = self._rotate(px_le[1:],py_le[1:],twist)
                # and finally plot the cross section, including cg, na and
                # strain positions
#                self._plot_cross_section(k, points=[px_le, py_le])
                self._plot_cross_section(k, points=False, xlim=[-0.01,  0.12],
                                         ylim=[-0.025, 0.020])

        # and integrate the cross sections to get the total volume
        volume = integrate.trapz(area, x=blade_hr[:,0])

        return blade_hr, volume, area, x_na, y_na, Ixx, Iyy, st_arr, strainpos

    def _tohalfchord(self, x, y, theta, x_c2, y_c2):
        """
        Transform from LE coordinates to half chord coordinates. In half chord
        coordinates, there is no twist presence, that happens one level up
        """

        # first rotate so x-axis is paralelel with the chord
        x, y = self._rotate(x, y, theta)
        # and move to the half chord point
        x = x_c2 - x
        y = y_c2 + y
        return x, y

    def _rotate(self, x, y, twist):
        """
        like a coordinate transformation, rotate the cross section

        """

        # transform to polar coordinates
        r = np.sqrt( x*x + y*y)
        theta = np.arctan(y/x)

        # rotate the whole setup
        theta += twist

        # and back to cartesian coordinates
        xnew = r*np.cos(theta)
        ynew = r*np.sin(theta)

        return xnew, ynew


class AirfoilProperties:
    """
    Extend Airfoil Coefficients over +-180 deg Range
    ================================================

    Extend a limited set of airfoil coefficients as function of angle of attack
    from either wind tunnel tests or computations (like XFOIL) over the full
    -180 +180 deg AoA range required for aeroelastic computations

    Input is a airfoil coefficient table: an array with cols AoA, Cl, Cd, Cm

    Parameters
    ----------
    airf_coeff : ndarray
        2D array with the airfoil coefficients, AoA, Cl, Cd, Cm

    verbose : optional, default=False
        If True, additional debugging information is printed

    verplot : optional, default=False

    aoa_res : optional, default=0.01

    cambereffect : boolean, default=False
        Correct the flat lift coefficient for camber. Method described in
        ref[1] ch2.2.4, p14-15.

    roundednose : boolean, default=True
        Correction for the flat plate lift coefficient due to the rounded
        nose of the leading edge, see ref[1] ch2.2.4, p14-15

    Members
    -------
    C_l_0 : float
    C_l_0i : int
    aoa_0 : float
    aoa_0i : int
    C_l_max : float
    C_l_maxi : int
    C_l_alpha : float
    C_l_pot : ndarray
    C_l_plate : ndarray
    aoafull : ndarray

    cambereffect
    roundednose
    airf_coeff

    """

    # TODO: this entire class needs to be refactored!
    # TODO: complete montgomerie and add other methods as well
    def __init__(self, airf_coeff, **kwargs):
        """
        """

        self.verbose = kwargs.get('verbose', False)
        self.verplot = kwargs.get('verplot', False)
        self.aoa_res = kwargs.get('aoa_res', 0.01)
        # options for _clplate()
        self.cambereffect = kwargs.get('cambereffect', False)
        self.roundednose = kwargs.get('roundednose', True)

        self.airf_coeff = airf_coeff

        # a simple shape(n, 4) ndarray is required: aoa, cl, cd, cm
        if not type(self.airf_coeff).__name__ == 'ndarray':
            raise TypeError, 'Input variable airf_coeff has to be a ndarray'

        # and it has to have 4 columns
        if not self.airf_coeff.shape[1] == 4:
            raise UserWarning, 'Input variable airf_coeff has to have 4 columns'

        # the full range of angles of attack
        self.aoafull = np.arange(-180,180, self.aoa_res)

        # calculate the actual properties
        # C_l for AoA=0
        self.C_l_0, self.C_l_0i = find0(airf_coeff, xi=0, yi=1)
        # c_m for AoA=0
        self.C_m_0, self.C_m_0i = find0(airf_coeff, xi=0, yi=3)
        # AoA for C_l=0
        self.aoa_0, self.aoa_0i = find0(airf_coeff, xi=1, yi=0)
        # Maximum lift coefficient
        self.C_l_maxi = airf_coeff[:,1].argmax()
        self.C_l_max = airf_coeff[self.C_l_maxi,1]
        self.aoa_clmax = airf_coeff[self.C_l_maxi,0]
        # lift gradient
        self.C_l_alpha = self._clalpha()
        # C_d0, minimum drag coefficient
        # note that airf_coeff has nan values!
        self.C_d0 = np.nanmin(airf_coeff[:,2])
        # lift over drag ratios
        self.ld_max = np.nanmax(airf_coeff[:,1] / airf_coeff[:,2])
        self.aoa_ld_max_i = np.nanargmax(airf_coeff[:,1] / airf_coeff[:,2])

        # prepare data for _cd90
        self.data_path = ''
        self.ae_file = ''
        self.ae_set = 0

        # potential flow line
        # don't forget to convert C_l_pot to rad since C_l_alpha is in rad
        self.C_l_pot = \
            (self.C_l_alpha*self.airf_coeff[:,0]*np.pi/180.) + self.C_l_0

        self.C_l_potfull = \
            (self.C_l_alpha*self.aoafull*np.pi/180.) + self.C_l_0


    def _print(self, prec=' 2.04f'):
        """
        """
        print '============================'
        print 'AirfoilProperties'
        print 'aoa_0      :', format(self.aoa_0, prec),
        print 'i:', format(self.aoa_0i, '4.0f')
        print 'C_l_0      :', format(self.C_l_0, prec),
        print 'i:', format(self.C_l_0i, '4.0f')
        print 'C_l_max    :', format(self.C_l_max, prec),
        print 'i:', format(self.C_l_maxi, '4.0f')
        print 'aoa_clmax  :', format(self.aoa_clmax, prec)
        print 'C_l_alpha  :', format(self.C_l_alpha, prec)
        print '2pi        :', format(np.pi*2., prec)
        print 'C_d0       :', format(self.C_d0, prec)
        print 'C_m_0      :', format(self.C_m_0, prec),
        print 'i:', format(self.C_m_0i, '4.0f')
        print 'L/D max    :', format(self.ld_max, '4.0f'),
        print 'i:', self.aoa_ld_max_i
#        tmp = self.airf_coeff[self.aoa_ld_max_i,0]
        print 'AoA L/D max:',format(self.airf_coeff[self.aoa_ld_max_i,0],'2.0f')
        # phase 2 variables
#        print 'C_d_90_2d :', format(self.C_d_90_2d, prec)
#        print 'alpha_am  :', format(self.alpha_am, prec)
#        print 'C_l_min   :', format(self.C_l_min, prec)
#        print 'aoa_min   :', format(self.aoa_min, prec)
        print '============================'

    def _clplate(self, cd_90):
        """
        Lift Coefficient for Fully separated Flow on Cambared Thin Plates
        =================================================================

        As described in [1], chapter 2.2, page 11-17.

        Parameters
        ----------

        cd_90 : float
            Drag coefficient at 90 degrees AoA

        Members
        -------

        cambereffect : boolean, default=False
            Correct the flat lift coefficient for camber. Method described in
            ref[1] ch2.2.4, p14-15.

        roundednose : boolean, default=True
            Correction for the flat plate lift coefficient due to the rounded
            nose of the leading edge, see ref[1] ch2.2.4, p14-15

        Returns
        -------

        C_l_plate : array(n)

        References
        ----------
        [1] B. Montgomerie, Methods for root effects, tip effects and
        extending the angle of attack range to +-180 deg, with application to
        aerodynamics for blades on wind turbines and propellers.
        FOI - Swedish Defence Research Agency, 2004,
        URL: http://www2.foi.se/rapp/foir1305.pdf
        """

        # the full range of angles of attack, convert to radians
        aoafull_r = self.aoafull*np.pi/180.

        # imperical constant, see page 14 ref[1]
        C_l_90 = 0.08
        alpha0 = self.aoa_0

        if self.roundednose:
            # equation 15, p14, ref[1], rounded nose effect in degrees
            delta1 = 57.6 * C_l_90 * np.sin(aoafull_r)
            # equation 16, p14, ref[1], camber effect in degrees
            delta2 = alpha0 * np.cos(aoafull_r)
            # equation 14, p14, ref[1], beta is alpha's replacement
            beta = self.aoafull - delta1 - delta2
            beta_r = beta*np.pi/180.
        else:
            beta_r = aoafull_r

        if self.cambereffect:
            # max lift and min lift are not symmetric due to airfoil camber
            # zero lift angle (or C_l at AoA=0) is a measure of camber, hence
            # equation 19, p15, ref[1]
            A = 1.+((self.C_l_0/math.sin(45.*np.pi/180.)) * np.sin(aoafull_r))
            # equation 24, p16, ref[1]
            # and finally the flat plate basic curve
            C_l_plate = A * cd_90 * np.sin(beta_r) * np.cos(beta_r)
        else:
            # the original expression from Horner
            # equation 13, p14, ref[1]
            C_l_plate = cd_90 * np.sin(beta_r) * np.cos(beta_r)

        return C_l_plate

    def _cdplate(self, cd_90):
        """
        Drag Coefficient for Fully Separated Flow on Cambared Thin Plates
        =================================================================

        As described in [1], chapter 2.3.2, page 19.

        """
        # the full range of angles of attack
        aoa_r = self.aoafull*np.pi/180.
        # and the thin plate C_d curve goes as follows
        # equation 34, p19, ref[1]
        C_d_plate = cd_90 * np.power(np.sin(aoa_r), 2)

        return C_d_plate


    def _clalpha(self, aoa1=0, aoa2=5):
        """
        Determine arfoil lift gradient C_l_alpha
        ========================================

        Calculate the lift gradient, usually based on the gradient
        between the points C_l=0 and alpha=0.

        Maybe better to base it on the range: AoA=0 - AoA=5

        Parameters
        ----------
        aoa1 : float, default=0
            starting point AoA (in [deg]) for gradient calculation

        aoa2 : float, default=5
            end point AoA (in [deg]) for gradient calculation

        """

        # search for the indeces marking the defined aoa range:
        aoa1i = self.airf_coeff[:,0].__ge__(aoa1).argmax()
        aoa2i = self.airf_coeff[:,0].__le__(aoa2).argmin()

        if self.verbose:
            print 'aoa1, aoa1i', self.airf_coeff[aoa1i,0], aoa1i
            print 'aoa2, aoa2i', self.airf_coeff[aoa2i,0], aoa2i

        y1 = self.airf_coeff[aoa1i,1]
        y2 = self.airf_coeff[aoa2i,1]
        # convert the AoA to radians first!
        x1 = self.airf_coeff[aoa1i,0]*np.pi/180.
        x2 = self.airf_coeff[aoa2i,0]*np.pi/180.

        # and thus the lift gradient is
        return (y2-y1)/(x2-x1)

    def _cd90(self, *args, **kwargs):
        """
        Drag Coefficient for Fully Separated Flow at 90 degrees AoA
        ===========================================================

        As described in [1], chapter 2.3.1, page 17-19.

        The drag coefficient can only be determined when the full blade layout
        is known (blade length, chord length distribution). Three different
        C_d90 values are calculated: on for the tip, root and mid section.
        The root and tip section length are stored in s_root and s_tip resp.


        Parameters, case 1
        ------------------
        chord_distr : ndarray
            2D array with radial position and chord length

        blade_length : float
            blade length excluding hub and circular root section
            see figure 9, p17, ref[1]


        Parameters, case 2
        ------------------

        data_path : str
            Path to the file defining the blade aerodynamic layout (.ae file)

        ae_file : str
            .ae file name

        ae_set : int
            set number in the .ae file


        Keyword Arguments
        -----------------

        radial_res : float, default = 0.001
            specify the radial resolution of the iterpolated chord distribution


        Members
        -------

        s_tip : float
            lenght for which C_d_90_3dtip applies, starting from the tip

        s_root : float
            lenght for which C_d_90_3droot applies, starting from the root

        C_d_90_3dtip : float
            average 90deg drag coefficient for the tip region

        C_d_90_3droot : float
            average 90deg drag coefficient for the root region

        C_d_90_2d : float
            standard 2D 90deg drag coefficient for a flat plate and sharp edges


        References
        ----------

        [1] B. Montgomerie, Methods for root effects, tip effects and
        extending the angle of attack range to +-180 deg, with application to
        aerodynamics for blades on wind turbines and propellers.
        FOI - Swedish Defence Research Agency, 2004,
        URL: http://www2.foi.se/rapp/foir1305.pdf
        """

        # from ref[1], page 17: imperical constants
        # round edges
        C_d_2dround = 2.06
        C_d_3dround = 1.45
        # sharp edges
        C_d_2dsharp = 1.98
        C_d_3dsharp = 1.17

        # ----------------------------------------------------------------------
        # input case1: data
        if len(args) == 2:
            chord_distr = args[0]
            # blade length excluding hub and circular root section
            # see figure 9, p17, ref[1]
            blade_length = args[1]
            # check data types
            if not type(chord_distr).__name__ == 'ndarray':
                raise TypeError, 'chord_distr should be ndarray'
            if not type(blade_length).__name__ in ('int', 'float'):
                raise TypeError, 'blade_lenth should be float or int'
        # input case2: path and set number
        elif len(args) == 3:
            data_path = args[0]
            ae_file = args[1]
            ae_set = args[2]

            # check data types
            if not type(data_path).__name__ == 'str':
                raise TypeError, 'chord_distr should be str'
            if not type(ae_file).__name__ == 'str':
                raise TypeError, 'ae_file should be str'
            if not type(ae_set).__name__ == 'int':
                raise TypeError, 'ae_set should be int'

            # load the required data
            model_obj = HawcPy.ModelData()
            model_obj.data_path = data_path
            model_obj.ae_file = ae_file
            model_obj.load_ae()

            # and substract required blade data
            # ae_dict[ae_set] = [label, data_array]
            # data_array = [Radius [m], Chord[m], T/C[%], Set no. of pc file]
            # only get the two first columns (radial position and chord)
            chord_distr = model_obj.ae_dict[ae_set][1][:,0:2]
            # blade length excluding hub and circular root section
            # see figure 9, p17, ref[1]
            # the total blade length is than the last radius value
            blade_length = chord_distr[-1,0]

        # ----------------------------------------------------------------------
        # force a higher interpolated radial resolution on chord_distr
        radial_res = kwargs.get('radial_res', 0.001)
        radial_grid = np.arange(0, blade_length, radial_res)
        chord_distr_highres = scipy.zeros((len(radial_grid),2))
        # scipy.interpolate.griddata(points, values, high_res_points)
        chord_distr_highres[:,0] = radial_grid
        chord_distr_highres[:,1] = scipy.interpolate.griddata(\
                    chord_distr[:,0], chord_distr[:,1], radial_grid)

        if self.verbose:
            print ''
            print 'chord_distr:'
            print chord_distr

        b = blade_length

        # ----------------------------------------------------------------------
        # FOR THE TIP PART
        # find s at the tip, impose maximum number of iterations
        # select all y,c values for the tip region (0.6R - R)
        ys_tip = b - chord_distr_highres[len(chord_distr_highres)/2::,0]
        cs_tip = chord_distr_highres[len(chord_distr_highres)/2::,1]
        # calculate kappa for all y values at the tip
        # equation 27, p17, ref[1], kappa is a help variable
        kappa_tip = (1. - np.exp(-(20.*cs_tip/b))) / (2.*cs_tip/b)
        # equation 28, p17, ref[1]
        ss_tip = kappa_tip * cs_tip
        # now the effective s value is where ys_tip == s_tip
        # iteration scheme to find s, as defined in figure 9, p17, ref[1]
        # and epxlained in p18, 4th paragraph, ref[1]
        s_tip = ys_tip[(ss_tip - ys_tip).__abs__().argmin()]

        if self.verbose:
            print '  blade_length:', blade_length
            print '         s_tip:', s_tip
            print ' s_tip r/R pos:', (blade_length-s_tip)/blade_length
            print ' redidual on s:', (ss_tip - ys_tip).__abs__().min()

        # ----------------------------------------------------------------------
        # FOR THE ROOT PART
        # find s at the tip, impose maximum number of iterations
        # select all y,c values for the root region (0 - 0.4R)
        ys_root = chord_distr_highres[0:len(chord_distr_highres)/2,0]
        cs_root = chord_distr_highres[0:len(chord_distr_highres)/2,1]
        # calculate kappa for all y values at the root
        # equation 27, p17, ref[1], kappa is a help variable
        kappa_root = (1. - np.exp(-(20.*cs_root/b))) / (2.*cs_root/b)
        # equation 28, p17, ref[1]
        ss_root = kappa_root * cs_root
        # now the effective s value is where ys_root == s_root
        # iteration scheme to find s, as defined in figure 9, p17, ref[1]
        # and epxlained in p18, 4th paragraph, ref[1]
        s_root = ys_root[(ss_root - ys_root).__abs__().argmin()]
        d = blade_length - s_root - s_tip

        if self.verbose:
            print '        s_root:', s_root
            print 's_root r/R pos:', s_root/blade_length
            print ' redidual on s:', (ss_root - ys_root).__abs__().min()
            print '             d:', d

        # check if s values make physical sense!
        # just a wild guess, d should still be at least 5% of blade_length
        if d < 0.05*blade_length:
            raise UserWarning, 'either s_tip and/or s_root are/is too large'

        # ----------------------------------------------------------------------
        # C_d_90 distribution is than
        # equation 29, p17, ref[1]
        n_r = C_d_3dround / (C_d_2dround - C_d_3dround)
        n_s = C_d_3dsharp / (C_d_2dsharp - C_d_3dsharp)

        # select only y < s points
        ys_tip  = np.arange(0,s_tip, radial_res)
        ys_root = np.arange(0,s_root,radial_res)
        # equation 30, p17, ref[1]
        # eq 30 is only valid when y < s. That means either at the tip or root
        C_d_90_tip = C_d_2dsharp* ( 1-np.power( ((s_tip-ys_tip )/s_tip) ,n_s) )
        C_d_90_root = C_d_2dround*( 1-np.power( ((s_root-ys_root)/s_root),n_r))

        # ----------------------------------------------------------------------
        # check the chord wise distribution of C_d_90 and compare with fig9 [1]
        if self.verplot:
            # plot the Cd90 line to check

            # the full chordwise cd90 distribution: pase root, d and tip in one
            # continues array. ys_d are the radial position in between tip
            # and root
            ys_d = np.arange(s_root, blade_length-s_tip, radial_res)
            nn = len(C_d_90_tip) + len(ys_d) + len(C_d_90_root)
            C_d_90 = scipy.zeros(nn)
            C_d_90[0:len(C_d_90_root)] = C_d_90_root
            C_d_90[len(C_d_90_root):len(ys_d)+len(C_d_90_root)] = C_d_2dround
            # C_d_90 at tip in reverse order, see definition of y in fig9 [1]
            C_d_90[len(ys_d)+len(C_d_90_root)::] = C_d_90_tip[::-1]

            # plot settings
            labelsize = 'x-large'
            figdir = '/home/dave/PhD/Projects/DataSets/'
            figname = 'cd90-chordwise'
            # initialize the plot object
            fig = Figure(figsize=(16, 9), dpi=200)
            canvas = FigCanvas(fig)
            fig.set_canvas(canvas)
            ax1 = fig.add_subplot(111)
            ax1.plot(chord_distr_highres[:,0], C_d_90,label='curved flat plate')
            ax1.grid(True)
            ax1.set_xlabel('radial position b', size=labelsize)
            ax1.set_ylabel('$C_{D90}$', size=labelsize)
            # save and clear the figure
            fig.savefig(figdir + figname +  '.eps')
            canvas.close()
            fig.clear()
            print 'saved C_d90 radial distribution in a figure:'
            print figdir + figname +  '.eps'

        # ----------------------------------------------------------------------
        # in [1] there is an additional dependency of C_d_90 on AoA. This is
        # ignored here since it depends on geometric airfoil data. That data
        # is not yet implemented in AirfoilDB. See [1], p18-19, equations 31-33
        # see implemented example in _delta_cd90_round(), it shows for the
        # E387 airfoil only a difference of 6% due to this dependency

        # ----------------------------------------------------------------------
        # the C_d3D value is the average over the y < s range, see fig9 [1]
        self.C_d_90_3dtip = C_d_90_tip.mean()
        self.C_d_90_3droot = C_d_90_root.mean()

        # TODO: implement a configurable y position. Maybe you need for several
        # radial positions a 3D C_d in the y < s area. In that case, take the
        # average over each bin.

        # and for the midsection we have
        self.C_d_90_2d = C_d_2dsharp

        self.s_tip, self.s_root = s_tip, s_root


    def _rearendfirst():
        """
        Rear-end flying first
        =====================

        Around the 180 degrees point, the trailing edge becomes the leading
        edge. For this part, the curved plate basic curve is not a good
        approximation. It should be based on C_l_max and C_l_min +- a
        rear-end-first correction.
        """
        # TODO: finish this implementation

    def _am(self, grid, points, gradient):
        """
        Determine A_M
        =============

        Based on figure 2 of [1]: point where C_l curves diverges from
        the potential flow lift curve

        Parameters
        ----------

        grid : ndarray

        points: ndarray

        gradient : float

        Returns
        -------

        alpha_am : float
        alpha_ami : int
        alpha2 : float
        point2i : int

        """
        # determine over which interval the gradient should be calculated
        # base it on 0.5 degrees
        rangei = int(0.4 / self.aoa_res)

        # local gradient, in each point (looking forward)
        # grad_loc = (y2-y1)/(x2-x1)
        grad_loc = (points[1::]-points[0:-1])/(grid[1::]-grid[0:-1])

        grad_mean = scipy.zeros(len(grid))

        print
        print 'grad_loc max, min', grad_loc.max(), grad_loc.min()

        # and take the average gradient over each rangei points
        for ii in range(rangei/2, len(grid)-rangei/2):
            grad_mean[ii] = grad_loc[ii-rangei/2:ii+rangei/2].mean()

#        ii = rangei*3
#        print 'example mean:'
#        print grad_loc[ii-rangei/2:ii+rangei/2]

        # fill in the blanks
        grad_mean[0:rangei/2] = grad_mean[0]
        grad_mean[len(grid)-rangei/2::] = grad_mean[-1]

        # and find where it diverges, only consider AoA>5
        istart = grid.__ge__(5).argmax()
        # and calculate the percentage deviation
        grad_delta = (grad_mean[istart::] - gradient) / gradient
        # we are looking for lower gradients, so grad_delta negative
        # call for the max, since False is the min
        # take a value of -0.5, meaning 50% lower (=less steep) than gradient
        # remember that 1/gradient would be perpendicular
        alpha_ami = grad_delta.__le__(-.50).argmax() + istart
        alpha_am = grid[alpha_ami]

        # ---------------------------------------------------------------
        # point2 in the middle of alpha_am and last point on cl (point1)
        # determine last point on the cl curve
        endi = np.isnan(points[::-1]).argmin()
        # endi starts counting in the reversed order
        point1i = len(points) - endi -1
        point2i = alpha_ami + ((point1i - alpha_ami)/2)
        alpha2 = grid[point2i]
        # ---------------------------------------------------------------

        if self.verplot:
            plt.plot(grid[istart::], grad_delta, label='grad_delta')
            plt.plot(grid, grad_mean, label='grad_mean')
            plt.xlabel('Angle of Attack')
            plt.ylabel('gradient')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()
#            plt.close()

        if self.verbose:
            print 'starting aoa position:', grid[istart]
            print 'gradient', gradient
            print 'alpha_AM:', alpha_am
#            print 'grad_mean:', grad_mean.min(), grad_mean.max()
#            print 'grad_delta:', grad_delta.min(), grad_delta.max()
#            print 'AoA and grad_delta slice'
#            tmp=np.array([grid[istart::][0:-1:rangei],grad_delta[0:-1:rangei]])
#            print tmp.transpose()

        return alpha_am, alpha_ami, alpha2, point2i

    def _merge_func_f(self):
        """
        A merging function between C_l data and C_l flat plate separated
        ================================================================

        As described in [1], chapter 2.3.1, page 17-19.

        References
        ----------
        [1] B. Montgomerie, Methods for root effects, tip effects and
        extending the angle of attack range to +-180 deg, with application to
        aerodynamics for blades on wind turbines and propellers.
        FOI - Swedish Defence Research Agency, 2004,
        URL: http://www2.foi.se/rapp/foir1305.pdf
        """

        # calculate for all aoa's from the original curve the value of f
        # only take the part of C_l_plate overlapping with airf_coeff
        # find the overlapping AoA range between C_l_plate and airf_coeff
        # the starint AoA
        aoastart_i = np.nanargmin(abs(self.aoafull - self.airf_coeff[0,0]))
        aoastop_i = aoastart_i + self.airf_coeff.shape[0]

        # equation 5, p12, ref[1]
        f = (self.airf_coeff[:,1] - self.C_l_plate[aoastart_i:aoastop_i]) \
            / (self.C_l_pot - self.C_l_plate[aoastart_i:aoastop_i])

#        # ---------------------------------------------------------------
#        # point2 in the middle of alpha_am and last point on cl (point1)
#        # determine last point on the cl curve
#        endi = np.isnan(self.airf_coeff[:,1][::-1]).argmin()
#        # endi starts counting in the reversed order
#        point1i = self.airf_coeff.shape[0] - endi -1
#        point2i = self.alpha_ami + ((point1i - self.alpha_ami)/2)
#        alpha2 = self.airf_coeff[point2i,0]
#        # ---------------------------------------------------------------

#        # verify there is a value at that angle of attack
#        point1i = np.nanargmax(self.airf_coeff[:,0])
#        if np.isnan(self.airf_coeff[point2i,1]):
#            raise UserWarning, 'last AoA does not have a Cl value...'

        # calculate k
        # equation 9, p13, ref[1]
        f2 = f[self.point2i]
        k = ( (1./f2) -1 ) * (1. / np.power(self.alpha2 - self.alpha_am, 4.) )

        if self.verbose:
            print ''
#            print 'point last', self.airf_coeff[self.point1i,0]
            print 'point2', self.alpha2
            print 'k', k

        # delta alpha
        # equation 4, p12, ref[1]
        d_aoa = self.alpha_am - self.aoafull

        # Montgomery's emperical obtained fit for the transfer fuction f
        # equation 6, p12, ref[1]
        self.f_mont = 1. / ( 1 + (k*d_aoa**4.) )

        if self.verplot:
            plt.plot(d_aoa, self.f_mont, label='grad_delta')
            plt.xlabel('$\\Delta \\alpha = \\alpha - \\alpha_M$')
            plt.ylabel('merge funciton f')
            plt.xlim([-10, 20])
            plt.title('merge function between flat plate and lift gradient')
#            plt.legend(loc='best')
            plt.grid(True)
            plt.show()

    def _mont_opt(self):
        """
        Force C_l_max to be part of the merged function
        ===============================================

        As discussed in [1], chapter 2.2.2, page 13-14.

        References
        ----------
        [1] B. Montgomerie, Methods for root effects, tip effects and
        extending the angle of attack range to +-180 deg, with application to
        aerodynamics for blades on wind turbines and propellers.
        FOI - Swedish Defence Research Agency, 2004,
        URL: http://www2.foi.se/rapp/foir1305.pdf
        """

        # do some optimization, create range around initial guess of alpha_am
        start= self.alpha_ami - int(10/self.aoa_res)
        stop = self.alpha_ami + int(3/self.aoa_res)
        alpha_ami_range = range(start, stop)

        if self.verbose:
            print ''
            print 'start, stop opti range:', self.airf_coeff[start,0],
            print self.airf_coeff[stop,0]

        # -----------------------------------------------------------------
        # remember plotting and verbose settings, don't use them in them here
        # it will produce too much prints
        verbose_orig, verplot_orig = self.verbose, self.verplot
        self.verbose, self.verplot = False, False
        # -----------------------------------------------------------------

        # keep track of the results in a 2D array
        self.am_results = scipy.zeros((len(alpha_ami_range),3))

        kk = 0
        for ami in alpha_ami_range:
#            print 'alpha_am, ami:', self.airf_coeff[ami,0], ami
            # alpha_am is regarded with respect to the cl-aoa data, not aoafull
            self.alpha_am, self.alpha_ami = self.airf_coeff[ami,0], ami

            # create the merge function
            # equation 6, p12, ref[1]
            self._merge_func_f()

            # now merge cl curves
            # equation 10, p13, ref[1]
            potential = self.C_l_potfull
            flatplate = self.C_l_plate
            self.cl_montg = (self.f_mont*potential)+((1-self.f_mont)*flatplate)

            # check if C_l_max is on the merged function, however, make sure
            # no to select the cl max at around aoa = 45 deg
            starti, stopi = int(181/self.aoa_res), int(205/self.aoa_res)
            # determine the max position, cl and aoa values
            cl_montg_maxi = self.cl_montg[starti:stopi].argmax()
            cl_montg_max = self.cl_montg[cl_montg_maxi+starti]
#            cl_montg_max_aoa = self.aoafull[cl_montg_maxi+starti]

            # and the delta, to be used for the optimizer
            self.cl_max_delta =  cl_montg_max-self.C_l_max

            # and keep track of the results for differen alpha_am values
            self.am_results[kk,:] = [self.cl_max_delta, self.alpha_am, ami]
            kk += 1

        # determine the best position
        self.winner = self.am_results[:,0].argmin()
        self.alpha_am = self.am_results[self.winner,1]
        self.alpha_ami = self.am_results[self.winner,2]

        # restore original plotting settings
        self.verbose, self.verplot = verbose_orig, verplot_orig

        # plot the function
        if self.verplot:
            plt.plot(self.am_results[:,1], self.am_results[:,0])


    def plot180(self, dbrow, **kwargs):
        """
        Plot AoA range -180 +180
        ========================

        This is a temporary function untill we figured out how to properly
        integrate this whole shit

        Make a plot over the full -180 +180 angle of attack range
        """

        # plot settings
        figdir = kwargs.get('figdir', '/home/dave/PhD/Projects/DataSets/')
        labelsize = kwargs.get('labelsize', 'x-large')
        # a row from the airfoil database holding runs or runs_merge
#        dbrow = kwargs.get('dbrow', None)

        # set figname, title if data provided
        if type(dbrow).__name__ != 'NoneType':
            airf_name = dbrow['airfoilname']
            reynolds = format(dbrow['reynolds'], '2.0f')
            figname = airf_name + '_' + reynolds + '_fullAoArange'
            figtitle = airf_name +  ' Re=' +format(dbrow['reynolds'], '2.02e')
            figtitle = figtitle.replace('(B)', '')

        pa4 = plotting.A4Tuned(scale=1.5)
        height = plotting.TexTemplate.pagewidth/2.0
        width = plotting.TexTemplate.pageheight
        pa4.setup(figdir+figname , nr_plots=1, hspace_cm=2., figsize_x=width,
                  figsize_y=height, grandtitle=False, wsleft_cm=1.8,
                  wsright_cm=0.4, wstop_cm=0.6, wsbottom_cm=1.)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
        ax1.set_title(figtitle, horizontalalignment='center')

#        # define the whitespaces in percentages of the total width and height
#        pa4.fig.subplots_adjust( left=0.04,  bottom=0.08, wspace=0.2,
#                            right=0.96,     top=0.96, hspace=0.2)

        # x axis labels
        ax1.set_xlabel(r'Angle of attack $\alpha$')
        # y axis labels
        ax1.set_ylabel('$C_l$ $C_d$ $C_m$')

        # the data is structured as follows
        aoapart = self.airf_coeff[:,0]
        clpart = self.airf_coeff[:,1]
        cdpart = self.airf_coeff[:,2]
        cmpart = self.airf_coeff[:,3]

        # the full range
        ax1.plot(self.aoafull, self.C_l_plate,'b--', label='$C_l$ flat plate')
        ax1.plot(self.aoafull, self.C_d_plate,'b-.', label='$C_d$ flat plate')
        ax1.plot(self.aoafull, self.C_m_montg,'b--', label='$C_m$ Montgomerie')

        # only plot merged if they exists
        try:
            ax1.plot(self.aoafull, self.cl_extended, 'r',label='$C_l$ ext')
        except:
            print 'no merged extended cl data'
        try:
            ax1.plot(self.aoafull, self.cd_extended, 'g',label='$C_d$ ext')
        except:
            print 'no merged extended cd data'
        try:
            ax1.plot(self.aoafull, self.cm_extended, 'y',label='$C_m$ ext')
        except:
            print 'no merged extended cm data'

        # only plot if at least one value is not zero for the original cl curve
        if clpart.any():
            # also plot the calculated lift gradient (potential flow line)
            # but limit to the interval -5 20 AoA range
            # lower bounds of the interval
            aoapart_redi = aoapart.__ge__(-10)
            aoapart_red = aoapart[aoapart_redi]
            C_l_pot_red = self.C_l_pot[aoapart_redi]
            # and the upper bounds of the interval
            aoapart_redi = aoapart_red.__le__(20)
            aoapart_red = aoapart_red[aoapart_redi]
            C_l_pot_red = C_l_pot_red[aoapart_redi]

            ax1.plot(aoapart_red, C_l_pot_red, 'r--',
                     label='$c_{l_{\\alpha}} WT data$')

            ax1.plot(aoapart, clpart, 'k',label='$c_{l}$ WT data')

        if cdpart.any():
            ax1.plot(aoapart, cdpart, 'k', label='$c_{d}$ WT data')

        if cmpart.any():
            ax1.plot(aoapart, cmpart, 'k', label='$c_{m}$ WT data')

        # add the legends
        ax1.legend(loc='best')
        # set xlimits to -180, 180 deg
        ax1.set_xlim([-120, 120])
        ax1.set_ylim([-1, 2])
        # change the ticker locators
        ymajorLocator = mpl.ticker.MultipleLocator(base=0.2)
        ax1.yaxis.set_major_locator( ymajorLocator )

#        matplotlib.ticker.Formatter.format_data('2.02f')
#        ymajorFormatter = matplotlib.ticker.ScalarFormatter('2.02f')
#        ax1.yaxis.set_major_formatter( ymajorFormatter )

        # set the major ticker on the x axis (AoA) to 10
        xmajorLocator = mpl.ticker.MultipleLocator(base=10)
        ax1.xaxis.set_major_locator( xmajorLocator )
        # activate the grid
        ax1.grid(True)
        # save the figure
        pa4.save_fig()


    def extrapolate(self, method):
        """
        Extrapolation Method
        ====================

        Parameters
        ----------
        method : string
            Define which airfoil extrapolation method to be used

        """

        self.extr_method = method

        if method == 'simple':
            self.extrapolate_simple()

        elif method == 'montgomerie':
            self.montgomerie(optimize=False)

        else:
            raise UserWarning, 'Unknown airfoil extrapolation method: '+method

    def extrapolate_simple(self):
        """
        Extend AoA Range from -180 to 180 deg
        =====================================

        Based on flat plate cl and cd data, extend the measured lift, drag and
        moment curves over the -180+180 angle of attack range. Do not apply
        the fancy Montgomerie merging curves and corrections for airfoil
        camber etc, but just apply a smooth linear merging between a line
        from the last measured point and the +-40 deg flat plate, and the
        actual flat plate values.

        Parameters
        ----------

        Members
        -------

        """

        # ae file as specified in __init__
        self._cd90(self.data_path, self.ae_file, self.ae_set)

        # the thin curved plate separated flow
        self.C_l_plate = self._clplate(self.C_d_90_2d)

        # cd for the stalled plate, depending on radial position
        self.C_d_plate = self._cdplate(self.C_d_90_2d)

        self.cl_extended = self._merge_180('cl', self.C_l_plate, aoa_merge=40.)
        self.cd_extended = self._merge_180('cd', self.C_d_plate, aoa_merge=25.)
        self.cm_extended = self._cm_montgomrie()


    def _cm_montgomrie(self):
        """
        Extrapolation Function for Cm, based on Montgomerie
        ===================================================

        The flat plate approach where the moment arm (see ref[1] ch2.4 for
        the definitions) is assumed as a straight line. The measured data
        is than merged with the flat plate one using _merge_180()

        Parameters
        ----------

        Members
        -------

        Returns
        -------

        References
        ----------
        [1] B. Montgomerie, Methods for root effects, tip effects and
        extending the angle of attack range to +-180 deg, with application to
        aerodynamics for blades on wind turbines and propellers.
        FOI - Swedish Defence Research Agency, 2004,
        URL: http://www2.foi.se/rapp/foir1305.pdf

        """

        # TODO: again, this should be an object that can only be called
        # when some previous steps have been performed: calcualtion of cl and
        # cd extended for instance... this needs some refactoring.

#        # as explained in ref[1] chapter 2.4, pages 20-27
#        # formula 39, ref[1] p22
#        arm0 = 0.25 - (self.C_m_0/self.C_l_0)
#        # formula 48, ref[1] p25
#        arm_calc = (( 0.25*(self.cl_extended-self.C_l_0) + (arm0*self.C_l_0) )\
#                   / self.cl_extended) \
#                   * ( 1 - (0.2*np.power(self.aoafull/18.,2)) )

        # moment arm for a flat plate based theory, positive angles of attack
        # formulas 46,47 in 45, ref[1], p24
        offset = 0.5111 - (0.001337*self.aoa_0)
        slope  = 0.001653 + (0.00016*self.aoa_0)
        arm_line =  offset + (slope * (self.aoafull - 90.))
        # moment arm for flat plate based theroy, negative AoA
        # formulas 52-57, ref[1] p27, see also fig23, p26, ref[1]
        xa = abs(self.aoa_0)
        ya = offset + (slope * (self.aoa_0-90.) )
        xb = -180. - xa
        yb = offset + (slope * 90.)
        k = (yb-ya) / (xb-xa)
        arm_neg = ya + (k*(self.aoafull-xa))

        # and merge the positive sided arm_line with the negative side
        aoa_0i = int( (self.aoa_0+180) / self.aoa_res)
        arm_line[0:aoa_0i] = arm_neg[0:aoa_0i]

        # the full range of angles of attack, convert to radians
        aoafull_r = self.aoafull*np.pi/180.
        # formula 37, ref[1] p21
        self.C_m_montg =((-self.cl_extended*np.cos(aoafull_r)) \
                       - ( self.cd_extended*np.sin(aoafull_r)))*(arm_line-0.25)

        if self.verbose:
            print '\n------------------------'
            print '_cm_montgomrie()'
            print 'aoa_0i', aoa_0i
            print 'min,max arm_calc', arm_line.min(), arm_line.max()

        if self.verplot:
            plt.figure(3)
            plt.title('arm_line should cover arm_neg')
            plt.plot(self.aoafull, arm_line, label='arm_line')
            plt.plot(self.aoafull, arm_neg, label='arm_neg')
            plt.plot(self.aoafull, self.C_m_montg, label='arm_neg')
            plt.legend(loc='best')
            plt.xlim([-180,180])
            plt.grid(True)
            plt.show()

        return self._merge_180('cm', self.C_m_montg, aoa_merge=40.)


    def _merge_180(self, index, plate, aoa_merge=40.):
        """
        Merge different coefficient data into one big curve
        ===================================================

        Based on measured data and flat plate. Merging is done according to
        a linear line drawn from the last measured point to the flat plate
        at aoa_merge. A linear weighting factor is used between the linear
        curve and the flat plate.

        Parameters
        ----------
        index : string
            Specify the type of data: cl, cd or cm

        plate : 1-D array
            Holding the coefficients for a flat plate, in correspondence with
            index

        aoa_merge : float, default=40
            Angle of attack in degrees. Point where the airfoil coefficient
            should be merged into the flat plate data again.

        Members
        -------

        Returns
        -------

        References
        ----------

        """

        if index == 'cl':
            ii = 1
        elif index == 'cd':
            ii = 2
        elif index == 'cm':
            ii = 3
        else:
            raise UserWarning, 'wrong input for AirfoilProperties._merge_180()'

        # note: b stands for begin, e for end, i for index on the array

        # flat plate cl/cd at plate_merge_point deg AoA
        aoa_mp1_i = int((180. - aoa_merge)/self.aoa_res)
        plate_mp1 = plate[aoa_mp1_i]

        aoa_mp2_i = int((180. + aoa_merge)/self.aoa_res)
        plate_mp2 = plate[aoa_mp2_i]

        # ---------------------------------------------------------------------
        # make a non nan version for cl,cd,cm based on airf_coeff
        # select last cl, cd, cm values
        # note that it might include nans...
        # first select the non nan cl values (AoA and Cl columns are selected)
        nonan_i = np.isnan(self.airf_coeff)[:,ii].__invert__()
        # since we are not taking a slice on the last axis, we need to make a
        # detour to select column 0 and 2. Apparantly, if you cherry pick on
        # one axis (=list of indices), you can only slice on the other axis
        nonan = self.airf_coeff[nonan_i,:]
        nonan = nonan[:,[0,ii]]

        # if there is no data for cm, fall back to plate only
        if index == 'cm' and len(nonan) < 1:
            print 'falling back to flat plate cm coefficients'
            warnings.warn('falling back to flat plate cm coefficients')
            return plate

        # ---------------------------------------------------------------------
        # extract the first and last values of the original data series
        # first values
        begin = nonan[0,1]
        # if we know the corresponding aoa, we can get the correct index pos
        # with respect to the full -180+180 aoa series
        beg_i = int( (nonan[0,0] + 180.) /self.aoa_res)
        # last value can now safely be assumed as
        end = nonan[-1,1]
        # corresponding AoA on the full series
        end_i = int( (nonan[-1,0] + 180.) /self.aoa_res)
        # begin/end indices with respect to airf_coeff
        ac_beg_i = nonan_i.argmax()
        ac_end_i = len(nonan_i) - nonan_i[::-1].argmax() -1

        # determine the amount of steps it will take to go from AoA@cl_last
        # and AoA@cl_ini until AoA=40, AoA=-40
        steps_b = beg_i - aoa_mp1_i
        steps_e = aoa_mp2_i - end_i
        # draw line from last measured point to 40deg on flat plate
        line_b = np.linspace( plate_mp1, begin, num=steps_b )
        line_e = np.linspace( end,  plate_mp2, num=steps_e )
        # merge function between line and flat plate
        mf_b = np.linspace( 0, 1, num=steps_b )
        mf_e = np.linspace( 0, 1, num=steps_e )

        if self.verbose:
            print '\n============> _merge_180(): index:', index
            print 'len line_b, plate:', len(line_b), len(plate[aoa_mp1_i:beg_i])
            print 'len line_e, plate:', len(line_e), len(plate[end_i:aoa_mp2_i])

#            print 'plate_mp1, begin, end, plate_mp2',
#            print  plate_mp1, begin, end, plate_mp2

            # printing for checking if the overlap went correct
#            print '-----------------'
#            if not np.allclose(nonan[0,0], self.aoafull[beg_i]):
#                print '########### =>',
#                print 'aoa beg nonan, aoafull:',nonan[0,0], self.aoafull[beg_i]
#                warnings.warn('aoa are not the same')
#            if not np.allclose(nonan[-1,0],self.aoafull[end_i]):
#                print '########### =>',
#                print 'aoa end nonan, aoafull:',nonan[-1,0],self.aoafull[end_i]
#                warnings.warn('aoa are not the same')
#            if not np.allclose(self.airf_coeff[ac_beg_i,0],nonan[0,0]):
#                print '########### =>',
#                print 'aoa ac beg:', self.airf_coeff[ac_beg_i,0]
#                warnings.warn('aoa are not the same')
#            if not np.allclose(self.airf_coeff[ac_end_i,0],nonan[-1,0]):
#                print '########### =>',
#                print 'aoa ac beg:', self.airf_coeff[ac_end_i,0]
#                warnings.warn('aoa are not the same')
#            print '-----------------'

#            print 'are AoAs creating a continues increasing series?'
#            print self.aoafull[aoa_mp1_i], self.aoafull[beg_i]
#            print self.aoafull[end_i], self.aoafull[aoa_mp2_i]

        # the merged data is then
        scaled_b = (1-mf_b)*plate[aoa_mp1_i:beg_i] + mf_b*line_b
        scaled_e = mf_e*plate[end_i:aoa_mp2_i] + (1-mf_e)*line_e
        # and compose the extended curve: original cl data,
        # merge original with flat plate, and flat plate
        merged = np.append(plate[0:aoa_mp1_i],scaled_b)
        merged = np.append(merged, self.airf_coeff[ac_beg_i:ac_end_i,ii])

        # check if the lengths are still ok
        if self.verbose:
            print '**',end_i, len(merged), ac_end_i, len(self.airf_coeff)

        # TODO: a possible check: aoa=0 needs to be exactly in the middle of
        # the -180 +180 range!
        if end_i+1 == len(merged):
            # sometime the merged has one point too much for some very strange
            # reason
            # TODO: figure out why sometimes the merged version is in excess
            # of one point
            merged = merged[0:-1]

        merged = np.append(merged, scaled_e)
        merged = np.append(merged, plate[aoa_mp2_i::])

        if self.verbose:
            print 'len  merged:', len(merged)
            print 'len aoafull:', len(self.aoafull)

            print 'are the merged points in line with airf_coeff?'
            print 'beg_i-1, beg_i, beg_i+1:',
            print np.allclose(merged[beg_i-1],self.airf_coeff[ac_beg_i,ii]),
            print np.allclose(merged[beg_i],self.airf_coeff[ac_beg_i,ii]),
            print np.allclose(merged[beg_i+1],self.airf_coeff[ac_beg_i,ii])
            print 'end_i-1, end_i, end_i+1:',
            print np.allclose(merged[end_i-1],self.airf_coeff[ac_end_i,ii]),
            print np.allclose(merged[end_i],self.airf_coeff[ac_end_i,ii]),
            print np.allclose(merged[end_i+1],self.airf_coeff[ac_end_i,ii])

            print np.allclose(plate[aoa_mp1_i], scaled_b[0]),
            print np.allclose(scaled_b[-1], self.airf_coeff[ac_beg_i,ii]),
            print np.allclose(self.airf_coeff[ac_end_i-1,ii], scaled_e[0]),
            print np.allclose(scaled_e[-1], plate[aoa_mp2_i])

            print self.aoafull[aoa_mp1_i], self.aoafull[beg_i]
            print self.aoafull[end_i], self.aoafull[aoa_mp2_i]

        return merged

    def montgomerie(self, optimize=False):
        """
        Montgomerie's Method for Airfoil Coefficient Extension
        ======================================================

        Same name conventions used as in [1]

        Also used in Qblade.

        Parameters
        ----------
        optimize : boolean, default = False
            Toggle optimisation on/off on the merge function between tabulated
            data and flat plate


        References
        ----------
        [1] B. Montgomerie, Methods for root effects, tip effects and
        extending the angle of attack range to +-180 deg, with application to
        aerodynamics for blades on wind turbines and propellers.
        FOI - Swedish Defence Research Agency, 2004,
        URL: http://www2.foi.se/rapp/foir1305.pdf
        """

        # ---------------------------------------------------------------------
        # equation 1, p11, ref[1]
        k_star = 0
        k = ( 1 - (k_star*self.C_l_0) )
        a = (self.C_l_max - self.C_l_0)
        b = k*a

        # equation 2, p11, ref[1]
        # Cl min is usually not determined with the wind tunnel data
        # due to camber it will not equal to clmax, so hence the b parameter
        self.C_l_min = self.C_l_0 - b

        # equation 3, p12, ref[1]
        # the difference between alpha_min and alpha_min_potential, in radians
        d_aoa_min = 3.*np.pi/180.
        # equation 3 gives 2pi, which would hold for a thin plate. The current
        # gradient of the airfoil, however, is known: C_l_alpha
        aoa_min_r = ( (self.C_l_min - self.C_l_0)/self.C_l_alpha ) - d_aoa_min
        self.aoa_min = aoa_min_r*180./np.pi
        # ---------------------------------------------------------------------

        # ae file as specified in __init__
        self._cd90(self.data_path, self.ae_file, self.ae_set)

        # the thin curved plate separated flow
        self.C_l_plate = self._clplate(self.C_d_90_2d)

        # cd for the stalled plate, depending on radial position
        self.C_d_plate = self._cdplate(self.C_d_90_2d)
#        cd_plate_root = self._cdplate(self.C_d_90_3droot)
#        cd_plate_tip  = self._cdplate(self.C_d_90_3dtip)

        # find alpha_am
        # equations 7,8,9, p13, ref[1]
        # convert gradient from [/rad] to [/deg]
        gradient = self.C_l_alpha*np.pi/180.
        self.alpha_am, self.alpha_ami, self.alpha2, self.point2i= \
            self._am(self.airf_coeff[:,0], self.airf_coeff[:,1], gradient)

        # do we need to optimize for alpha_am?
        # if not the initial value is assumed
#        self.verplot = True
        if optimize:
            self._mont_opt()

        # create the merge function
        # equation 6, p12, ref[1]
        self._merge_func_f()

        # now merge cl curves
        # equation 10, p13, ref[1]
        potential = self.C_l_potfull
        flatplate = self.C_l_plate
        self.cl_montg = (self.f_mont*potential)+((1-self.f_mont)*flatplate)

#        # replace the original cl data in the merged curve
#        # find the overlapping AoA range between aoafull and airf_coeff
#        aoastart_i = np.nanargmin(abs(self.aoafull - self.airf_coeff[0,0]))
#        aoastop_i = aoastart_i + self.airf_coeff.shape[0]
#        self.cl_montg[aoastart_i:aoastop_i] = self.airf_coeff[:,1]

        # check if C_l_max is on the merged function, however, make sure
        # no to select the cl max at around aoa = 45 deg
        starti, stopi = int(181/self.aoa_res), int(205/self.aoa_res)
        # determine the max position, cl and aoa values
        cl_montg_maxi = self.cl_montg[starti:stopi].argmax()
        cl_montg_max = self.cl_montg[cl_montg_maxi+starti]
        cl_montg_max_aoa = self.aoafull[cl_montg_maxi+starti]

        if self.verbose:
            print '           alpha_am:',self.alpha_am
            print '            C_l_min:', format(self.C_l_min)
            print '            aoa_min:', format(self.aoa_min)
#            print '   range for cl max:', self.aoafull[starti],
#            print self.aoafull[stopi]
            print '       delta cl max:', cl_montg_max-self.C_l_max
            print 'delta aoa at cl max:', cl_montg_max_aoa-self.aoa_clmax


    def __delta_cd90_round__(self):
        """
        Example of C_d_90 dependency of alpha as explained in [1],
        p18-19, equations 31-33

        Conclusion, there is only a 6% difference between max/min values
        generated as result of the AoA dependency (for the E387 that is)
        """

        C_d_90_tp = 2.0
        # data from http://www.worldofkrauss.com/foils/292
        # leading edge radius (normalized by chord length c)
        r_le_c = 0.018
        # thickness ratio
        t_c = 0.091
        # camber
        h_c = 0.038

        # the full range of angles of attack
        aoa = np.arange(-180,180, self.aoa_res)
        aoa_r = aoa*np.pi/180.

        C_d_90 = C_d_90_tp - 0.83*r_le_c - 1.46*t_c/2. + 1.46*h_c*np.sin(aoa_r)

        # it only has a very modest influence
        print C_d_90.max(), C_d_90.min()
        # = 1.97411, 1.8631500000000001
        print C_d_90.max()/C_d_90.min()
        # = 1.0595550546118133

    def print_extrapolate(self, dbrow, **kwargs):
        """
        Print the Extrapolated Profiles to Text File
        ============================================

        Depending on the required format, print to a text file.
        For instance, HAWTOPT requires that each airfoil has the same amount
        of airfoil coefficient data for the same angles of attack

        Parameters
        ----------

        Members
        -------

        Returns
        -------

        References
        ----------

        """
        resolution = kwargs.get('resolution', 'high')
        if resolution == 'high':
            hrss = 20
            lrss = 180
            step_hr = 0.5
            step_lr = 10.
        elif resolution == 'mid':
            hrss = 20
            lrss = 110
            step_hr = 1
            step_lr = 10.

        # TODO: still need to implement printing when selecting from database
        # directly

        filepath = kwargs.get('filepath', '/home/dave/PhD/Projects/DataSets/')
        re = format(dbrow['reynolds'], '1.0f')
        tmp = '_' + resolution + 'res'
        filename = dbrow['airfoilname'] +'_Re_'+ re + tmp + '.dat'

        # first slice, only every 0.5 degrees
        step = int(step_hr/self.aoa_res)
        dataset = self.coeff_full[::step,:]

        # in the -20+20 range, only every 10deg
        start = int( (-lrss+180)/step_hr )
        stop =  int( (-hrss+180)/step_hr )
        step = int(step_lr/step_hr)
        dataset_beg = dataset[start:stop:step,:]

        start = int( (hrss+180)/step_hr )
        stop =  int( (lrss+180)/step_hr  )
        step = int(step_lr/step_hr)
        dataset_end = dataset[start:stop:step,:]

        start = int( (-hrss+180)/step_hr )
        stop =  int( ( hrss+180)/step_hr )
        step = int(1)
        dataset_mid = dataset[start:stop:step,:]

        if self.verbose:
            print 'dataset.shape', dataset.shape
            print 'dataset_beg.shape', dataset_beg.shape
            print 'dataset_mid.shape', dataset_mid.shape
            print 'dataset_end.shape', dataset_end.shape

        # and the result is a reduced dataset
        dataset_reduced = np.append(dataset_beg, dataset_mid, axis=0)
        dataset_reduced = np.append(dataset_reduced, dataset_end, axis=0)

        if self.verbose:
            print 'dataset_reduced.shape', dataset_reduced.shape

        np.savetxt(filepath+filename, dataset_reduced, fmt="% 10.04f")
        print 'extrapolated set saved:', filepath+filename

        if self.verplot:
            plt.figure(5)
            plt.plot(dataset_reduced[:,0], dataset_reduced[:,1], label='C_L')
            plt.plot(dataset_reduced[:,0], dataset_reduced[:,2], label='C_D')
            plt.plot(dataset_reduced[:,0], dataset_reduced[:,3], label='C_M')
            plt.legend(loc='best')
            plt.xlim([-180,180])
            plt.grid(True)
            plt.show()

if __name__ == '__main__':

    dummy = 0

#    builtfirstdb()

#    # ---------------------------------------------------------------------
#    # load the db and make selection
#    # selection criteria for merging the E387
#    db = AirfoilDB(verbose=False, verplot=False,
#                   Re=1e5, airfoilname='E387', run='', table='runs')
#    db.plot()
#    db.merge()
#    db.properties()
#    db.h5f.close()
#    # ---------------------------------------------------------------------


    # ---------------------------------------------------------------------
    # search and list airfoils who match
    # ---------------------------------------------------------------------
#    db = AirfoilDB(Re=3e5, airfoilname='S822', run='', table='runs',
#                   verplot=False, verbose=False, re_acc=1e4)
#    db.plot(cl_pot=False)
#    db.merge(overwrite=False)
#    db.h5f.close()
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # PLOTTING DATA
    # ---------------------------------------------------------------------
#    db = AirfoilDB(Re=1e5, airfoilname='S822', run='', table='runs_merge',
#                   verplot=False, verbose=False, re_acc=1e4)
#    db.plot180()
#    db.h5f.close()

    # ---------------------------------------------------------------------
    # add an airfoil to the coordinate database
    # ---------------------------------------------------------------------
#    coord = AirfoilCoord()

#    target = '/home/dave/PhD/Projects/DataSets/coordinates/S822.dat'
#    coord.txt2tbl(target, 'S822')
#
#    target = '/home/dave/PhD/Projects/DataSets/coordinates/S823.dat'
#    coord.txt2tbl(target, 'S823', overwrite=False)

#    coord._init_db_rplus()
#    coord.update_coord_tbl('S822', 16, '', '')
#    coord.h5f.close()

#    coord._init_db_rplus()
#    coord.update_coord_tbl('S823', 21, '', '')
#    coord.h5f.close()

#    coord._init_db_rplus()
#    coord.h5f.root.coord.coord_tbl.cols.t_c_max[0] = 16.
#    coord.h5f.root.coord.coord_tbl.flush()
#    coord.h5f.close()

#    coord_up, coord_low = coord.interp_airfoils('S823', 'S822', 20.)
#    coord.verbose = True
#    coord.verplot = True
#    coord.cross_prop(coord_up, coord_low, tests=True)

    # ---------------------------------------------------------------------
#    pass
