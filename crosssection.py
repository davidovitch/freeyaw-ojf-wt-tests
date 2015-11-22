# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 20:24:38 2012

@author: dave
"""

import numpy as np
#import scipy.integrate as integrate
#import pylab as plt
import math
import scipy as sp
#import timeit

import HawcPy


class test:

    def __init__(self):
        """
        """
        pass

    def prop_ring(self):
        """
        Test properties for a ring, modelled as a thin walled something
        """

        radius = 1.
        # make sure the simple test cases go well
        x = np.linspace(0,radius,100000)
        y = np.sqrt(radius*radius - x*x)
        x = np.append(-x[::-1], x)
        y_up = np.append(y[::-1], y)
        tw1 = np.ndarray((len(x),3), order='F')
        tw1[:,0] = x
        tw1[:,1] = y_up
        tw1[:,2] = 0.01

        tw2 = np.ndarray((len(x),3), order='F')
        y_low = np.append(-y[::-1], -y)
        tw2[:,0] = x
        tw2[:,1] = y_low
        tw2[:,2] = 0.01

        # tw1 and tw2 need to be of the same size, give all zeros
        upper_bound = sp.zeros((4,2), order='F')
        lower_bound = sp.zeros((4,2), order='F')

        st_arr, EA, EIxx, EIyy = properties(upper_bound, lower_bound,
                    tw1=tw1, tw2=tw2, rho=1., rho_tw=1., E=1., E_tw=1.)

        headers = HawcPy.ModelData().st_column_header_list
        print '\nRING PROPERTIES'
        for index, item in enumerate(headers):
            tmp = item + ' :'
            print tmp.rjust(8), st_arr[index]

    def prop_solidcircle(self):
        """
        Test properties for a solid circle and compare with theory
        """

        radius = 1.
        r = radius
        # make sure the simple test cases go well
        x = np.linspace(0,radius,100000)
        y = np.sqrt(radius*radius - x*x)
        x = np.append(-x[::-1], x)
        y_up = np.append(y[::-1], y)
        upper_bound = np.ndarray((len(x),2), order='F')
        upper_bound[:,0] = x
        upper_bound[:,1] = y_up

        lower_bound = np.ndarray((len(x),2), order='F')
        y_low = np.append(-y[::-1], -y)
        lower_bound[:,0] = x
        lower_bound[:,1] = y_low

        # tw1 and tw2 need to be of the same size, give all zeros
        tw1 = sp.zeros((len(x),3), order='F')
        tw2 = sp.zeros((len(x),3), order='F')

        # and return as a 1D array, with st ordering of the elements
        #0  1   2     3     4     5     6     7    8  9   10    11  12
        #r  m  x_cg  y_cg  ri_x  ri_y  x_sh  y_sh  E  G  I_x   I_y   J
        #13   14  15   16    17   18
        #k_x  k_y  A  pitch  x_e  y_e
        st_arr, EA, EIxx, EIyy = properties(upper_bound, lower_bound,
                    tw1=tw1, tw2=tw2, rho=1., rho_tw=1., E=1., E_tw=1.)

        headers = HawcPy.ModelData().st_column_header_list
        print '\nSOLID CIRCLE PROPERTIES'
        for index, item in enumerate(headers):
            tmp = item + ' :'
            print tmp.rjust(8), st_arr[index]

        if not np.allclose(st_arr[1], np.pi*r*r): print 'A WRONG !!'
        if not np.allclose(st_arr[2], 0): print 'x_cg WRONG !!'
        if not np.allclose(st_arr[3], 0): print 'y_cg WRONG !!'
        if not np.allclose(st_arr[4], r/2.): print 'ri_x WRONG !!'
        if not np.allclose(st_arr[5], r/2.): print 'ri_y WRONG !!'
        if not np.allclose(st_arr[6], 0): print 'x_sh WRONG !!'
        if not np.allclose(st_arr[7], 0): print 'y_sh WRONG !!'
        if not np.allclose(st_arr[10], np.pi*r*r*r*r/4): print 'Ixx WRONG !!'
        if not np.allclose(st_arr[11], np.pi*r*r*r*r/4): print 'Iyy WRONG !!'

        # check speed
        #%timeit properties(upper_bound, lower_bound, tw1=tw1, tw2=tw2,
                            #rho=1., rho_tw=1., E=1., E_tw=1.)


def properties(upper_bound, lower_bound, tw1=[], tw2=[], rho=1.,
                rho_tw=1., E=1., E_tw=1., tests=False, verplot=False,
                order='F', st_arr_tw=False):
    """
    Cross sectional airfoil properties
    ==================================

    Calculate the cross sectional properties of a general shape defined by
    an upper and lower bound function, or/and by a thin walled curve.

    The upper and lower coordinates should have the same x coordinates,
    as given by coord_continuous() and interp_airfoils().

    This method is a piecewise linear integration method that also
    calculates the second moment of inertia wrt the neutral x and y axis.

    Note that for the mass moment of inertia we have: Im_xx = Ixx*rho

    Parameters
    ----------

    upper_bound : ndarray(m,2)
        Upper bounded curve (x,y)

    lower_bound : ndarray(m,2)
        Lower bounded curve (x,y)

    tw1 : ndarray(n,3), default=0
        Thin walled curve and corresponding thickness (x,y,t). x grid can be
        different than upper and lower_bound, but tw1 and tw2 need to have the
        same number of points (not necessarily identical though)

    tw2 : ndarray(n,3), default=0
        Thin walled curve and corresponding thickness (x,y,t). x grid can be
        different than upper and lower_bound, but tw1 and tw2 need to have the
        same number of points (not necessarily identical though)

    tests : boolean, default=False

    rho : float or ndarray(m), default=1

    rho_tw : float or ndarray(n), default=1

    E=1 : float or ndarray(m), default=1

    E_tw=1 : float, default=1

    verplot=False,

    order='F'

    st_arr_tw=False
        overwrites E_tw and rho_tw

    Returns
    -------

    A

    x_na

    y_na

    Ixx

    Iyy


    """

    # increase speed: convert to fortran memory layout because we are using
    # the numpy C layout WRONG: using the first index with a C array is SLOW
    # however, if the second dimension is small (like here it is 2 or 3)
    # there isn't much difference
    #if asfortran:
        #print 'convert to F'
        #upper_bound = np.asfortranarray(upper_bound)
        #lower_bound = np.asfortranarray(lower_bound)

    # TODO: data checks
    #   continuity: does it? I think it can deal with any curvature
    #   better and seperate tests and result verification

    # make sure the grids for both upper and lower surface are the same
    if not np.allclose(upper_bound[:,0], lower_bound[:,0]):
        msg = 'coord_up and low need to be defined on the same grid'
        raise ValueError, msg
    # TODO: if not, interpolate and fix it on equal x grids

    # get data out of the structural array for the thin walled piece
    if type(st_arr_tw).__name__ is 'ndarray':
        sti = HawcPy.ModelData().st_headers
        E_tw = st_arr_tw[sti.E]
        rho_tw = st_arr_tw[sti.m] / st_arr_tw[sti.A]

    # NUMERICAL APPROACH: split up into blocks. Sinc6e we already
    # interpolated the coordinates to a high res grid, this approach
    # should be sound. The upper block is a triangle, lower is just a
    # rectangle (see drawings)

    # even though they are identical, put in one array to have the
    # correct summation over all the elements of upper and lower curve
    x = np.ndarray((len(upper_bound),2), dtype=np.float128, order=order)
    x[:,0] = upper_bound[:,0]
    x[:,1] = lower_bound[:,0]
    x1 = x[:-1,:]

    # for convience, put both y's in one array: [x, up, low]
    y = np.ndarray((len(upper_bound),2), dtype=np.float128, order=order)
    y[:,0] = upper_bound[:,1]
    y[:,1] = lower_bound[:,1]
    # y1, first y value of each element
    y1 = y[:-1,:]

    # for the thin walled curve
    x_tw = np.ndarray((len(tw1),2), dtype=np.float128, order=order)
    y_tw = np.ndarray((len(tw1),2), dtype=np.float128, order=order)
    t_tw = np.ndarray((len(tw1),2), dtype=np.float128, order=order)
    x_tw[:,0] = tw1[:,0]
    x_tw[:,1] = tw2[:,0]
    y_tw[:,0] = tw1[:,1]
    y_tw[:,1] = tw2[:,1]
    t_tw[:,0] = tw1[:,2]
    t_tw[:,1] = tw2[:,2]

    # delta's define each element
    x1_tw = x_tw[:-1,:]
    y1_tw = y_tw[:-1,:]
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    dx_tw = np.diff(x_tw, axis=0)
    dy_tw = np.diff(y_tw, axis=0)

    # ===== AREA =====
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

    # area of elements for the thin walled section is much simpler
    dl_tw = np.sqrt( (dx_tw*dx_tw) + (dy_tw*dy_tw) )
    dA_tw = dl_tw*t_tw[:-1,:]

    # in case we have a zero dy_tw
    if np.allclose(dy_tw, sp.zeros(dy_tw.shape)):
        t_star = sp.zeros(dy_tw.shape)
    else:
        t_star = dA_tw / dy_tw

    # total area is then
    A = np.sum(dA_a + dA_b) + np.nansum(dA_tw)

    # ===== CENTROID =====
    # cg positions for each block with respect to (x,y)=(0,0)
    # upper and lower share the same x_cg positions
    x_ct_b = x1 + (dx*0.5)
    y_ct_b = y1*0.5

    # inclined side is directed towards y axis, so 2x/3 instead of x/3.
    x_ct_a = (2.*dx/3.) + x1
    # when on the lower side this reverses, but since then y1 > y2,
    # and y1 + dy/3 becomes y2 + 2*dy/3. Check the figures for better
    # and more correct understanding/description of why it goes right.
    y_ct_a = (dy/3.) + y1

    # fot the thin walled pieces
    x_ct_tw = x1_tw + (dx_tw*0.5)
    y_ct_tw = y1_tw + (dy_tw*0.5)

    # find the centroid wrt (0,0)
    #x_ct = np.sum( (dA_a*x_ct_a) + (dA_b*x_ct_b) + (dA_tw*x_ct_tw) ) /A
    #y_ct = np.sum( (dA_a*y_ct_a) + (dA_b*y_ct_b) + (dA_tw*y_ct_tw) ) /A

    # ===== MASS =====
    # total section mass
    m = np.sum(rho*dA_a + rho*dA_b) + np.nansum(rho_tw*dA_tw)

    # ===== CENTER OF GRAVITY =====
    # and hence
    x_cg = ( np.sum( (rho*dA_a*x_ct_a) + (rho*dA_b*x_ct_b) )\
           + np.nansum(rho_tw*dA_tw*x_ct_tw) ) /m
    y_cg = ( np.sum( (rho*dA_a*y_ct_a) + (rho*dA_b*y_ct_b) )\
           + np.nansum(rho_tw*dA_tw*y_ct_tw) ) /m

    # ===== NEUTRAL AXIS =====
    ea_s = np.sum(E*dA_a + E*dA_b) + np.nansum(E_tw*dA_tw)
    x_na = ( np.sum( (E*dA_a*x_ct_a) + (E*dA_b*x_ct_b) )\
           + np.nansum(E_tw*dA_tw*x_ct_tw) ) / (ea_s)
    y_na = ( np.sum( (E*dA_a*y_ct_a) + (E*dA_b*y_ct_b) )\
           + np.nansum(E_tw*dA_tw*y_ct_tw) ) / (ea_s)

    # ===== MOMENTS OF INERTIA =====
    # Moments of inertia for each piece around local cg
    # since dy can switch sign, only consider absolute value
    Ixx_cg_a = np.abs(dx*dy*dy*dy/36.)
    Iyy_cg_a = np.abs(dx*dx*dx*dy/36.)
    Ixx_cg_b = np.abs(dx*y1*y1*y1/12.)
    Iyy_cg_b = np.abs(dx*dx*dx*y1/12.)
    Ixx_cg_tw = np.abs(dA_tw*dy_tw*dy_tw/12.)
    Iyy_cg_tw = np.abs(dA_tw*t_star*t_star/12.)

    # the Steiner terms: distance local cg to neutral axis
    Ixx_steiner_a = dA_a*(y_na-y_ct_a)*(y_na-y_ct_a)
    Ixx_steiner_b = dA_b*(y_na-y_ct_b)*(y_na-y_ct_b)
    Iyy_steiner_a = dA_a*(x_na-x_ct_a)*(x_na-x_ct_a)
    Iyy_steiner_b = dA_b*(x_na-x_ct_b)*(x_na-x_ct_b)
    Ixx_steiner_tw = dA_tw*(x_na-x_ct_tw)*(x_na-x_ct_tw)
    Iyy_steiner_tw = dA_tw*(y_na-y_ct_tw)*(y_na-y_ct_tw)

    # and finally the moments of inertia with respect to the neutral axis
    Ixx = np.sum(Ixx_cg_a  + Ixx_steiner_a   + Ixx_cg_b + Ixx_steiner_b) \
        + np.nansum(Ixx_cg_tw + Ixx_steiner_tw)
    Iyy = np.sum(Iyy_cg_a  + Iyy_steiner_a + Iyy_cg_b + Iyy_steiner_b) \
        + np.nansum(Iyy_cg_tw + Iyy_steiner_tw)

    # ===== MASS MOMENTS OF INERTIA =====
    Im_xx_cg_a = np.abs(rho*dx*dy*dy*dy/36.)
    Im_yy_cg_a = np.abs(rho*dx*dx*dx*dy/36.)
    Im_xx_cg_b = np.abs(rho*dx*y1*y1*y1/12.)
    Im_yy_cg_b = np.abs(rho*dx*dx*dx*y1/12.)
    Im_xx_cg_tw = np.abs(rho_tw*dA_tw*dy_tw*dy_tw/12.)
    Im_yy_cg_tw = np.abs(rho_tw*dA_tw*t_star*t_star/12.)

    # the Steiner terms: distance local cg to neutral axis
    Im_xx_steiner_a = rho*dA_a*(y_na-y_ct_a)*(y_na-y_ct_a)
    Im_xx_steiner_b = rho*dA_b*(y_na-y_ct_b)*(y_na-y_ct_b)
    Im_yy_steiner_a = rho*dA_a*(x_na-x_ct_a)*(x_na-x_ct_a)
    Im_yy_steiner_b = rho*dA_b*(x_na-x_ct_b)*(x_na-x_ct_b)
    Im_xx_steiner_tw = rho_tw*dA_tw*(x_na-x_ct_tw)*(x_na-x_ct_tw)
    Im_yy_steiner_tw = rho_tw*dA_tw*(y_na-y_ct_tw)*(y_na-y_ct_tw)

    # and finally the moments of inertia with respect to the neutral axis
    Im_xx = np.sum(Im_xx_cg_a  + Im_xx_steiner_a   + Im_xx_cg_b  \
          + Im_xx_steiner_b) + np.nansum(Im_xx_cg_tw + Im_xx_steiner_tw)
    Im_yy = np.sum(Im_yy_cg_a  + Im_yy_steiner_a + Im_yy_cg_b  \
          + Im_yy_steiner_b) + np.nansum(Im_yy_cg_tw + Im_yy_steiner_tw)
    # convert to radius of gyration
    ri_x = math.sqrt(Im_xx/m)
    ri_y = math.sqrt(Im_yy/m)

    # ===== BEAM STIFFNESS =====
    EA = np.sum(E*(dA_a + dA_b)) + np.nansum(E_tw*dA_tw)

    EIxx = np.sum( E*(Ixx_cg_a  + Ixx_steiner_a + Ixx_cg_b + Ixx_steiner_b) )\
         + np.nansum(E_tw*(Ixx_cg_tw + Ixx_steiner_tw))
    EIyy = np.sum( E*(Iyy_cg_a  + Iyy_steiner_a + Iyy_cg_b + Iyy_steiner_b) )\
         + np.nansum(E_tw*(Iyy_cg_tw + Iyy_steiner_tw))

    # and return as a 1D array, with st ordering of the elements
    #0  1   2     3     4     5     6     7    8  9   10    11  12
    #r  m  x_cg  y_cg  ri_x  ri_y  x_sh  y_sh  E  G  I_x   I_y   J
    #13   14  15   16    17   18
    #k_x  k_y  A  pitch  x_e  y_e

    # expres x_cg, y_cg, x_na and y_na wrt half chord point instead of LE
    # express x_na and y_na wrt half chord point
    # chord length is determined by knowing the TE and LE coordinates
    # LE should be (0,0), but take into account just to be sure
    x = upper_bound[-1,0] - upper_bound[0,0]
    y = upper_bound[-1,1] - upper_bound[0,1]
    chordlen = math.sqrt(x*x + y*y)
    c2 = chordlen/2.
    cos_a = x/chordlen
    sin_a = y/chordlen
    x_c2 = cos_a*c2
    y_c2 = sin_a*c2
    # and convert to half chord point coordinates
    # first the translation of the coordinate system
    xna_norot = x_c2 - x_na
    yna_norot = y_c2 + y_na
    xcg_norot = x_c2 - x_cg
    ycg_norot = y_c2 + y_cg

    # and than the rotation due to camber, if any
    #theta = math.acos(cos_a)
    #x_na = xna_norot*math.cos(theta) - yna_norot*math.sin(theta)
    #y_na = xna_norot*math.sin(theta) + yna_norot*math.cos(theta)
    #x_cg = xcg_norot*math.cos(theta) - ycg_norot*math.sin(theta)
    #y_cg = xcg_norot*math.sin(theta) + ycg_norot*math.cos(theta)

    # since we ingored camber line angle already with the moments of inertia
    # ignore it here as well
    x_na = xna_norot
    y_na = yna_norot
    x_cg = xcg_norot
    y_cg = ycg_norot

    st_arr = np.array([np.nan, m, x_cg, y_cg, ri_x, ri_y, x_na, y_na, E,np.nan,
                      Ixx,Iyy, Ixx+Iyy, np.nan, np.nan, A, np.nan, x_na, y_na])

    return st_arr, EA, EIxx, EIyy

if __name__ == '__main__':

    tests = test()
    tests.prop_solidcircle()
    tests.prop_ring()
