# -*- coding: utf-8 -*-
"""
Author: David Verelst


"""
from __future__ import division # always devide as floats

__author__ = 'David Verelst'
__license__ = 'GPL'
__version__ = '0.5'

from time import time
import scipy
import scipy.interpolate as interpolate
import array
import numpy as np
import math
import os
import logging
#import sys
import copy
import pickle
from datetime import datetime
# decimal has a higher precision as float
# import decimal as D

# import matplotlib.pyplot as plt

# backend is already changed in to Qt in Spyder
## first switch to backend, than import pylab
#import matplotlib
## GTKCairo can produce EPS files (vector rendered images)
###matplotlib.use('Qt4Agg', warn=True) # no errors
#matplotlib.use('GTKCairo', warn=True) # depricated warning gtk.Tooltips()

#matplotlib.use('QtAgg', warn=True) # requires to install pyqt, dependency error
#matplotlib.use('GTKAgg', warn=True) # depricated warning gtk.Tooltips()
#matplotlib.use('TkAgg', warn=True) # ImportError: No module named _tkagg
#import pylab as plt
# working directly with the matplotlib API instead
import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

# for 3D plotting
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#import matplotlib.ticker

import wafo

import misc

#import HawcPyParsing as hawcpar
#import pprint

PRECISION = np.float64

############################################################################
### PLATFOFM DEPENDENT VARIABLES
############################################################################


############################################################################
### GENERAL functions and classes
############################################################################

#def printarray():
#    """
#    Print all values of an array nicely in the same format, in human readable
#    output, fixed with text output. Make a fast check of the values inside an
#    array, withouth the need to output it to a csv file to evaluate it in a
#    spreadsheet program
#    """
#
#    # TODO: make printing somehow general or make a selection of columns
#    for k in range(array.shape[2]):
#        for m in range(4):
#            print format(diff[0,m,k], ' 1.5f').ljust(10),
#    print ''


def linspace_around(start, stop, points=None, num=50, verbose=False):
    """
    Create a linspace between an interval while considering some points

    start < points < stop
    """
    # data checks
    if start > stop:
        raise ValueError, 'start should be smaller than stop!'

    # ordinary linspace of points is not defined
    if type(points).__name__ == 'NoneType':
        return np.linspace(start, stop, num=num)

    res = np.array([start])
    res = np.append(res, points)
    res = np.append(res, stop)

    # define the fractions and fill the linspace
    # num-2 because we exclude start and stop points
    num_missing = num - 2 - len(points)
    if num_missing < 1:
        raise ValueError, 'num has to be at least +3 longer than nr of points'

    fractions = num_missing*(res[1:]-res[:-1])/res[-1]
    # and see how that would translete to nr of points
    distribution = np.around(fractions, decimals=0)
    # missing points
    deltas = distribution - fractions
    # if we miss points, missing will be negative, too much points it will be
    # possitive. This number is always either 1 or -1!!
    missing = np.sum(deltas)
    # add/remove missing point to largest delta interval
    if missing < 0:
        # we are missing a point, add one to the largest deficit
        fractions[np.argmin(deltas)] -= missing
    else:
        # we have too many points, remove on the largest surplus
        fractions[np.argmax(deltas)] -= missing
    # convert to int
    distribution = np.int32(np.around(fractions, decimals=0))

    # fill up the gaps, start with the starting point, because we will allways
    # ignore the first element of the linspace, to avoid double entries
    res_fill= res[0]
    for k in range(len(res)-1):
        fill = np.linspace(res[k], res[k+1], num=(2+distribution[k]))
        res_fill = np.append(res_fill, fill[1:])

    if verbose:
        print 'fractions', fractions
        print 'distribution', distribution
        print 'missing', missing
        print 'res_fill', res_fill
        print 'points', points
        print len(res_fill), num

    # and make sure we have the same number of elements as num and
    if not len(res_fill) == num:
        msg = 'result does not have the specified number of items'
        raise ValueError, msg
    # make sure we have an increasing set of numbers
    if (res_fill[1:]-res_fill[:-1]).min() < 0:
        raise ValueError, 'result is not strictly increasing'

    return res_fill

def frange(*args):
    """A float range generator.
    SOURCE: http://code.activestate.com/recipes/
    66472-frange-a-range-function-with-float-increments/#c13

    Usage:
        for i in frange(42.):
            ...
        l = list(frange(42.)

    *args: [[start], [[stop], [step]]]
    """
    start = 0.0
    step = 1.0

    l = len(args)
    if l == 1:
        end = args[0]
    elif l == 2:
        start, end = args
    elif l == 3:
        start, end, step = args
        if step == 0.0:
            raise ValueError, "step must not be zero"
    else:
        raise TypeError, "frange expects 1-3 arguments, got %d" % l

    v = start
    while True:
        if (step > 0 and v >= end) or (step < 0 and v <= end):
            raise StopIteration
        yield v
        v += step

def unique(s):
    """
    SOURCE: http://code.activestate.com/recipes/52560/
    AUTHOR: Tim Peters

    Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        del u  # move on to the next method
    else:
        return u.keys()

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.
    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t  # move on to the next method
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.
    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u

def write_file(file_path, file_contents, mode):
    """
    INPUT:
        file_path: path/to/file/name.csv
        string   : file contents is a string
        mode     : reading (r), writing (w), append (a),...
    """

    FILE = open(file_path, mode)
    FILE.write(file_contents)
    FILE.close()


def write_csv_from_array(file_name, array, mode):
    """
    INPUT:
        file_name: path/to/file/name.csv
        array    : a numpy array
        mode     : reading (r), writing (w), append (a),...
    """

    # columns = array.shape[1]
    contents = ''
    for k in array:
        for n in k:
            contents = contents + str(n) + ';'
        # at the end of each line, new line symbol
        contents = contents + '\n'

    write_file(file_name, contents, mode)


def load_pickled_file(source):
    FILE = open(source, 'rb')
    result = pickle.load(FILE)
    FILE.close()
    return result

def save_pickle(source, variable):
    FILE = open(source, 'wb')
    pickle.dump(variable, FILE, protocol=2)
    FILE.close()

def create_multiloop_list(iter_dict, debug=False):
    """
    Create a list based on multiple nested loops
    ============================================

    Considerd the following example:
    for v in range(V_start, V_end, V_delta):
        for y in range(y_start, y_end, y_delta):
            for c in range(c_start, c_end, c_delta):

    Could be replaced by a list with all these combinations.

    Parameters
    ----------
    iter_dict : dictionary
        as value, give a list of the range to be considered

    Output
    ------
    combi_list : list
        List containing dictionaries. Each entry is a combination of the
        given iter_dict

    Example
    -------
    >>> iter_dict = {'[wind]':[5,6,7], '[coning]':['aa','bb'], \
                        '[yaw]':['zz0', 'zz1']}
    >>> combi_list = create_multiloop_list(iter_dict)
    """

    combi_list = []

    # fix the order of the keys
    key_order = iter_dict.keys()
    nr_keys = len(key_order)
    nr_values,indices = [],[]
    # determine how many items on each key
    for key in key_order:
        # count how many values there are for each key
        nr_values.append(len(iter_dict[key]))
        # create an initial indices list
        indices.append(0)

    if debug: print nr_values, indices

    go_on = True
    # keep track on which index you are counting, start at the back
    loopkey = nr_keys -1
    cc = 0
    while go_on:
        if debug: print indices

        # Each entry on the list is a dictionary with the parameter combination
        combi_list.append(dict())

        # save all the different combination into one list
        for keyi in range(len(key_order)):
            key = key_order[keyi]
            # add the current combination of values as one dictionary
            combi_list[cc][key] = iter_dict[key][indices[keyi]]

        # +1 on the indices of the last entry, the overflow principle
        indices[loopkey] += 1

        # cycle backwards thourgh all dimensions and propagate the +1 if the
        # current dimension is full. Hence overflow.
        for k in range(loopkey,-1,-1):
            # if the current dimension is over its max, set to zero and change
            # the dimension of the next. Remember we are going backwards
            if not indices[k] < nr_values[k] and k > 0:
                # +1 on the index of the previous dimension
                indices[k-1] += 1
                # set current loopkey index back to zero
                indices[k] = 0
                # if the previous dimension is not on max, break out
                if indices[k-1] < nr_values[k-1]:
                    break
            # if we are on the last dimension, break out if that is also on max
            elif k == 0 and not indices[k] < nr_values[k]:
                if debug: print cc
                go_on = False

        # fail safe exit mechanism...
        if cc > 20000:
            raise UserWarning, 'multiloop_list has already '+str(cc)+' items..'
            go_on = False

        cc += 1

    return combi_list


class remove_from_list:
    def __init__(self):
        self.array = []
        self.value = 0

    def remove(self):
        # create a list first
        temp = []
        for k in self.array:
            if k == self.value:
                pass
            else:
                temp.append(k)
        # # now convert to array again
        # array = scipy.zeros((1, len(temp)), float)
        # for k in range(len(array)):
            # array[0, k] = float(temp[k])
        return temp


def remove_items(list, value):
    """Remove items from list
    The given list wil be returned withouth the items equal to value.
    Empty ('') is allowed. So this is een extension on list.remove()
    """
    # remove list entries who are equal to value
    ind_del = []
    for i in xrange(len(list)):
        if list[i] == value:
            # add item at the beginning of the list
            ind_del.insert(0, i)

    # remove only when there is something to remove
    if len(ind_del) > 0:
        for k in ind_del:
            del list[k]

    return list

def remove_items2(listt, value):
    """
    THIS METHOD IS SLOWER

    Tested with a very short list
    %timeit remove_items2(pp, '')
    100000 loops, best of 3: 3.12 us per loop

    %timeit remove_items(pp, '')
    1000000 loops, best of 3: 1.55 us per loop
    """
    try:
        while True:
            listt.pop(listt.index(value))
    except ValueError:
        return listt

class IOFiles:
    def __init__(self):
        self.content = ''
        self.f_name = ''
        self.f_path = ''

    def LoadFile(self):
        FILE = open(self.f_path + self.f_name, "r")
        template = FILE.readlines()
        FILE.close()

    def WriteFile(self):
        FILE = open(self.f_path + self.f_name, "w")
        FILE.write(self.content)
        FILE.close()

class Log:
    """
    Class for convinient logging. Create an instance and add lines to the
    logfile as a list with the function add.
    The added items will be printed if
        self.print_logging = True. Default value is False

    Create the instance, add with .add('lines') (lines=list), save with
    .save(target), print current log to screen with .printLog()
    """
    def __init__(self):
        self.log = []
        # option, should the lines added to the log be printed as well?
        self.print_logging = False
        self.file_mode = 'a'

    def add(self, lines):
        # the input is a list, where each entry is considered as a new line
        for k in lines:
            self.log.append(k)
            if self.print_logging:
                print k

    def save(self, target):
        # tread every item in the log list as a new line
        FILE = open(target, self.file_mode)
        for k in self.log:
            FILE.write(k + '\n')
        FILE.close()
        # and empty the log again
        self.log = []

    def printscreen(self):
        for k in self.log:
            print k


def SignalStatistics(sig, start=0, stop=-1):
    """WILL BECOME DEPRICATED, include in v2 the indices of the maxima!!!

    Statistical properties of a HAWC2 result file.

    Input:
    sig as outputted by the class LoadResults: sig[timeStep,channel]
    start, stop: indices (not the time!!) of starting and stopping point
    in the signal matrix. This option allows you to skip a period at the
    beginning or the end of the time series

    Default start and stop are for the entire signal range. Note that
    np.array[0:-1] will exclude the last element! You should use
    np.array[0:len(array)]

    Output:
    np array:
        sig_stat = [statistic parameter, channel]
        statistic parameter = max, min, mean, std, range, abs max

    Or just input a time series.
    """

    # convert index -1 to whole range:
    if stop == -1:
        stop = len(sig)

    nr_statistics = 6

    sig_stats = scipy.zeros([nr_statistics, sig.shape[1]])
    # calculate the statistics:
    sig_stats[0,:] = sig[start:stop,:].max(axis=0)
    sig_stats[1,:] = sig[start:stop,:].min(axis=0)
    sig_stats[2,:] = sig[start:stop,:].mean(axis=0)
    sig_stats[3,:] = sig[start:stop,:].std(axis=0)
    # range:
    sig_stats[4,:] = sig_stats[0,:] - sig_stats[1,:]
    sig_stats[5,:] = np.absolute(sig[start:stop,:]).max(axis=0)

    return sig_stats

def SignalStatisticsNew(sig, start=0, stop=-1, dtype='Float64'):
    """
    Statistical properties of a HAWC2 result file
    =============================================

    Input:
    sig as outputted by the class LoadResults: sig[timeStep,channel]
    start, stop: indices (not the time!!) of starting and stopping point
    in the signal matrix. This option allows you to skip a period at the
    beginning or the end of the time series

    Default start and stop are for the entire signal range. Note that
    np.array[0:-1] will exclude the last element! You should use
    np.array[0:len(array)]

    Output:
    np array:
        sig_stat = [(0=value,1=index),statistic parameter, channel]
        statistic parameter = 0 max, 1 min, 2 mean, 3 std, 4 range, 5 abs max

    Or just input a time series.
    """

    # convert index -1 to also inlcude the last element:
    if stop == -1:
        stop = len(sig)

    nr_statistics = 6

    sig_stats = scipy.zeros([2,nr_statistics, sig.shape[1]], dtype=dtype)
    # calculate the statistics values:
    sig_stats[0,0,:] = sig[start:stop,:].max(axis=0)
    sig_stats[0,1,:] = sig[start:stop,:].min(axis=0)
    sig_stats[0,2,:] = sig[start:stop,:].mean(axis=0)
    sig_stats[0,3,:] = sig[start:stop,:].std(axis=0)
    sig_stats[0,4,:] = sig_stats[0,0,:] - sig_stats[0,1,:] # range
    sig_stats[0,5,:] = np.absolute(sig[start:stop,:]).max(axis=0)

    # and the corresponding indices:
    sig_stats[1,0,:] = sig[start:stop,:].argmax(axis=0)
    sig_stats[1,1,:] = sig[start:stop,:].argmin(axis=0)
    # not relevant for mean, std and range
    sig_stats[1,2,:] = 0
    sig_stats[1,3,:] = 0
    sig_stats[1,4,:] = 0 # range
    sig_stats[1,5,:] = np.absolute(sig[start:stop,:]).argmax(axis=0)

    return sig_stats


def CDF(series, sort=True):
    """
    Cumulative distribution function
    ================================

    Cumulative distribution function of the form:

    .. math::
        CDF(i) = \\frac{i-0.3}{N - 0.9}

    Paramters
    ---------
        series: 1D array
        sort  : boolean (optional, True=default) - to sort or not to sort

    Returns
    -------
    cdf : ndarray (N,2)
        Array with the sorted input series on the first column
        and the cumulative distribution function on the second.

    .. math::
       (a + b)^2 = a^2 + 2ab + b^2

       (a - b)^2 = a^2 - 2ab + b^2

    where
        i : the index of the sorted item in the series
        N : total number of elements in the serie
    Series will be sorted first.
    """

    N = len(series)
    # column array
    i_range = np.arange(N)
    # convert to row array
    x, i_range = np.meshgrid([1], i_range)
    # to sort or not to sort the input series
    if sort:
        series.sort(axis=0)
    # convert to row array. Do after sort, otherwise again 1D column array
    x, series = np.meshgrid([1], series)
    # cdf array
    cdf = scipy.zeros((N,2))
    # calculate the actual cdf values
    cdf[:,1] = (i_range[:,0]-0.3)/(float(N)-0.9)
    # make sure it is sorted from small to large
    if abs(series[0,0]) > abs(series[series.shape[0]-1,0]) and series[0,0] < 0:
        # save in new variable, otherwise things go wrong!!
        # if we do series[:,0] = series[::-1,0], we get somekind of mirrord
        # array
        series2 = series[::-1,0]
    # x-channel should be on zero for plotting
    cdf[:,0] = series2[:]

    return cdf


def error(msg):
    raise UserWarning, msg

################################################################################
### Generic HAWC2 classes
############################################################################

def SurfacePlot3D(x, y, Z, save=False, figpath='', figname='test.png', \
                    figsize_x=16, figsize_y=12, dpi=100):
    """Generate a 3D surface plot.
    INPUT:
        x: 1D array of the x-coordinates (can be constructed with range())
        y: 1D array of the y-coordinates
        Z: 2D array (has to be of shape(x,y))

    """
    fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=dpi)
    ax = Axes3D(fig)
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(x, y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)

    # this will cause the external console to hang until you close the window
    # in that case it will not be saved correctly
    # plt.show()

    if save:
        plt.savefig(figpath+figname, orientation='landscape')

def plottest():
    fig = plt.figure(111)

    data0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    data1 = np.array([1,2,3,4,5,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    data2 = np.array([3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    data3 = np.array([9,6,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    data4 = np.array([3,7,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    data = np.array([data0,data1,data2,data3,data4]).transpose()

#    print data
#    # first subplot
#    ax = fig.add_subplot(111)
#    im = ax.imshow(data, cmap=plt.get_cmap('binary'), interpolation="nearest",
#               origin="lower")
#    ax.set_xlim([0.5,5])
#    ax.set_ylim([0,10])
#    fig.colorbar(im)
#    ax.plot([1,2,3,4],[3,4,4.5,5])

    fig = plt.figure(111)
    ax = fig.add_subplot(111)
    ax.plot([1,2,3],[1,2,3])
    plt.title('something')
    plt.grid(True)

    plt.close()

class PlotSignal:
    """Plot HAWC2 result file

    Usage:
    p = PlotSignal(sig, channel, ch_details, cases)
    p.plot2D()
    Note that first index (=0) of sig is the x-axis channel (time in most cases)

    -> INPUT:
    nparray :: ch_details[channel number,(0=ID; 1=units; 2=description)]
    nparray :: sig[timestep, channel] -> sig[time (ch1), ch2, ch3, ch4, ...]
    list    :: cases[filename1, filename2,...] defines the title or legend,
                    depending on the type of plot

    -> ch_details will be used for the legend and axis labels.
        units will be placed as axis labels (ch_details[channel,1])
        -> for multiple channels, it will take the units of the last channel
        the ID will be used for the legend (ch_details[channel,0])

    -> object can be re-used! Available parameters and their defaults:
    p.chan = update channel
    p.sig =  update signal
    p.title = update title
    p.ch_details = update channel details
    p.colers_left_axis = ['b', 'r', 'g', 'k']
    p.colers_right_axis = ['r', 'b', 'g', 'k']
    p.plotsymbol = ''

    Used variables:
    ch_details[channel, (0=ID; 1=units; 2=description)]
    """

    def __init__(self, signals, channels, ch_details, cases):
        """Title of the docstring
        Doc string here
        """
        self.debug = False

        # color lists, ordered for the overplots
        self.collors_left_axis = ['k', 'b', 'm', 'r','y', 'g', 'c']
        self.collors_right_axis = ['c', 'g', 'y', 'r','m', 'b', 'k']

        self.channels = channels
        self.signals = signals
        self.cases = cases
        self.ch_details = ch_details
        self.fig_path = ''

        self.plotsymbol = ''
        self.plotsymbolright =['','','','','','','','','','','','','','','','']
        self.plotsymbolleft = ['','','','','','','','','','','','','','','','']

        # these come in group, use all correctly if markersize is set
        self.markersizeleft = []
        self.markerleft = []
        self.markerfacecolorleft = []
        self.markeredgecolorleft = []
        self.linestyelleft= []
        self.linewidthleft = []
        self.markeveryleft = []

        self.markersizeright = []
        self.markerright = []
        self.markerfacecolorright = []
        self.markeredgecolorright = []
        self.linestyelright = []
        self.linewidthright = []
        self.markeveryright = []

        # the horizontal and vertical line variables,
        # change them to a float to activate
        self.hor_line_right = 'none'
        self.hor_line_left = 'none'
        self.ver_line = 'none'

    def data_check(self):
        """
        """

        # checks concerning cases
        if not type(self.cases).__name__ == 'list':
        # if str(type(self.cases)).find('list') == -1:
            print type(self.cases).__name__
            raise UserWarning, \
            'E001 - cases should be a list of strings (=case ID\'s)'
        # see if all items are strings
        for k in self.cases:
            if not type(k).__name__ == 'str':
            # if str(type(k)).find('str') == -1:
                raise UserWarning, \
                'E002 - cases should be a list of strings (=case ID\'s)' + \
                ' current type is: ' + str(type(k))

        try:
            nr_cases = len(self.cases)
        except:
            raise UserWarning, \
            'E003 - cases should be a list of strings (=case ID\'s)'

        # checks concerning signals:
        if str(type(self.signals)).find('list') == -1:
            raise UserWarning, \
            'E004 - signals should be a list of numpy.ndarray'
        # check if all items are numpy arrays
        for k in self.signals:
            if str(type(k)).find('numpy.ndarray') == -1:
                raise UserWarning, \
                'E005 - signals should be a list of numpy.ndarray'
        try:
            nr_signals = len(self.signals)
        except:
            raise UserWarning, \
            'E006 - signals should be a list of numpy.ndarray'

        # number of cases and signals should be the same!
        if nr_signals != nr_cases:
            msg = 'E007 - for each signal, a case entry should exist! ' +\
                    '(nr_signals==nr_cases)'
            raise UserWarning, msg


        # nr_channels = len(self.channels[0])

        # ch_details should also be an numpy.ndarray
        if str(type(self.ch_details)).find('numpy.ndarray') == -1:
            raise UserWarning, \
            'E008 - signals should be a list of numpy.ndarray'
        # should have th ID, units and description parts!
        if self.ch_details.shape[1] != 3:
            msg = 'E009 - ch_details should have the following shape: ' +\
                '(nr_channels,3) \n (0=ID, 1=units, 2=description'
            raise UserWarning, msg
        if str(type(self.ch_details[0,0])).find('numpy.unicode_') == -1:
            msg = 'E010 - ch_details should be a numpy.ndarray of data ' +\
                    'type numpy.unicode_'
            raise UserWarning, msg

#        try:
#            if nr_signals == nr_cases:
#                pass
#            else:
#                raise UserWarning, \
#                'case list should match the number of signals'
#        except:
#            pass

        try:
            len(self.channels)
            len(self.channels[0])
            # if only left channels, add an empty list for right channels
            try:
                len(self.channels[1])
            except:
                self.channels.append([])
        except:
            raise UserWarning, \
            'E011 - channels should be a list containing a list of channels'

        # checks on the x-y axis limits:
        # xlen = len(xlim), len(ylim)


    def overplot2D(self, **kwargs):
#                   ibeg=0, iend=-1, xchan=0, figsize_x=16, figsize_y=9, \
#            dpi=100,save=False,figpath='',figname='',legend_loc='upper center',\
#            legend_left='upper left', legend_right='upper right', \
#            xlim=[], ylim=[], xlim_left=[], ylim_left=[], \
#            xlim_right=[], ylim_right=[], x_0=0.0, labelsize='x-large',\
#            legendsize='x-large', titlesize='x-large',
#            ylabel_man_left=None, ylabel_man_right=None,
#            wsleft=0.075, wsbottom=0.1, wsright=0.95, wstop=0.9, wspace=0.2,\
#            hspace=0.2):
        """
        Create overplots for channels or cases
        ======================================

        Uses input as follows (text between brackets () is optional input):
        channels: list: [[channels left axis] (, [channels right axis]) ]
        signals : list: [signal1, signal2]
        cases   : list: plot title for channel overplots of the same signal,
                        legends for signal overplots of the same channel

        Usage:
        p = HawcPy.PlotSignal([sig], [[chan left], [chan right]],
            sig.ch_details, plot_title or legend)
        p.overplot2D(**kwargs)
            **kwargs: ibeg, iend, xchan (all integers, refering to sig indices)
                      figsize_x,figsize_y (inches), dpi
                      save (boolean), figpath, figname
            all **kwargs are optional
        for example: p.overplot2D(ibeg=500, iend=-100, xchan=6)

        SIGNAL OVERPLOT
        If multiple signals are combined with multiple channels, a signal
        overplot will be created for the first left channel only
        The title for the signal overplot is the channel name and a legend is
        added to identify the different signals (which is the cases list).

        The optional input arguments ibeg, iend and xchan allows to control
        the range of the plot (ibeg and iend indicating the beginning and
        ending indices of the HAWC2 signal) and to specificy on which channel
        (or better index, ichan) the x-axis is located.

        CHANNEL OVERLPLOT
        Title for the channel overplot the first entry of the cases list

        The colors of the left and right axis are matched witht the color of
        the plot itself, if only one channel is used for that axis
        """

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        # default values for kwargs
        figsize_x = kwargs.get('figsize_x',16)
        figsize_y = kwargs.get('figsize_y',9)
        dpi = kwargs.get('dpi',100)
        save = kwargs.get('save',False)
        xchan = kwargs.get('xchan', 0)
        iend = kwargs.get('iend', -1)
        ibeg = kwargs.get('ibeg', 0)
        x_0 = kwargs.get('x_0', 0.0)
        # figdir = kwargs.get('figdir','')
        figname = kwargs.get('figname','')
        figpath = kwargs.get('figpath','')
        legend_loc = kwargs.get('legend_loc','upper center')
        legend_left = kwargs.get('legend_left','upper left')
        legend_right = kwargs.get('legend_right', 'upper right')
        ylabel_man_left = kwargs.get('ylabel_man_left', None)
        ylabel_man_right = kwargs.get('ylabel_man_right', None)
        xlim = kwargs.get('xlim',[])
        ylim = kwargs.get('ylim',[])
        xlim_left = kwargs.get('xlim_left',[])
        ylim_left = kwargs.get('ylim_left',[])
        xlim_right = kwargs.get('xlim_right',[])
        ylim_right = kwargs.get('ylim_right',[])
        labelsize = kwargs.get('labelsize', 'medium')
        legendsize = kwargs.get('legendsize', 'medium')
        titlesize = kwargs.get('titlesize', 'medium')
        wsleft = kwargs.get('wsleft', 0.08)
        wsbottom = kwargs.get('wsbottom', 0.1)
        wsright = kwargs.get('wsright', 0.95)
        wstop = kwargs.get('wstop', 0.95)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)
        legendoutside = kwargs.get('legendoutside',False)
        # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        # Becase this whole plotting procedure is crap, save the data to
        # a csv file and plot in better layout
        datasave = {}

        if self.debug: print '======= start of overplot2D ======= '

        # data check, aborts execution if exception is thrown
        self.data_check()

        fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        canvas = FigureCanvas(fig)
        fig.set_canvas(canvas)

        # define the whitespaces in percentages of the total width and height
        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                            top=wstop, wspace=wspace, hspace=hspace)

        # number of signals input:
        nr_sig = len(self.signals)
        # number of channels input:
        nr_chan_left = len(self.channels[0])
        nr_chan_right = len(self.channels[1])

        for k in range(len(self.signals)):
            # order data on x-channel channel
            sort_args = np.argsort(self.signals[k][:,xchan])
            # only sort the wind speed channel
            self.signals[k] = self.signals[k][sort_args,:]

        #-----------------------------------------------------------------------
        # overplotting signals scenario
        if nr_sig > 1:

            # add a subplot
            ax = fig.add_subplot(111)

            # convert index -1 to length to capture correct range!
            if iend == -1:
                iend = len(self.signals[0])

            # multiple channels are ignored, take only the first left one
            chan = self.channels[0][0]
            # corresponding title: the channel
            title = self.ch_details[chan,0] + ' // ' + self.ch_details[chan,2]

            # the legend is stored in a tuple
            legend = tuple()
            # overplot the actual signals
            i = 0
            for sig in self.signals:
                # force datatype on the legend label entry
                label = str(self.cases[i])

                # select coler and symbol
                c = self.collors_left_axis[i] + self.plotsymbol
                if len(self.markersizeleft) > 0:
                    m = self.markerleft[i]
                    ms = self.markersizeleft[i]
                    mfc = self.markerfacecolorleft[i]
                    mec = self.markeredgecolorleft[i]
                    # actual and factual plot command is finally issued here!
                    ax.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],
                             marker=m, ms=ms, mfc=mfc, mec=mec,
                             label=label)
                else:
                    ax.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],c,
                             label=label)
                # legend contains the cases
                legend += (label,)
                i += 1

                # save the data for replotting
                key = self.cases[i]
                datasave[key] = [sig[ibeg:iend,xchan]-x_0, sig[ibeg:iend,chan]]

            # further configuration of the plot

            ax.grid(True)
            # hard press the axis limits if given
            if len(xlim) == 2:
                ax.set_xlim(xlim)
            if len(ylim) == 2:
                ax.set_ylim(ylim)

            # x and y labels
            # ch_details[channel,(0=ID; 1=units; 2=description)]
            xlabel = self.ch_details[xchan,0]+' ['+self.ch_details[xchan,1]+']'
            ax.set_xlabel(xlabel, size=labelsize)
            ylabel = self.ch_details[chan,0]+ ' ['+self.ch_details[chan,1]+']'
            ax.set_ylabel(ylabel, size=labelsize)
            # set the legend
            leg = ax.legend(legend, legend_loc, shadow=True)
            # and alter the legend font size
            for t in leg.get_texts():
                t.set_fontsize(legendsize)

            ax.set_title(title, size=titlesize)
            ax.grid(True)

            datasave['ylabel_left'] = ylabel
            datasave['xlabel'] = xlabel
        #-----------------------------------------------------------------------

        #-----------------------------------------------------------------------
        # overplotting channels scenario:
        # also valid for a simple one single/channel plot
        elif nr_chan_left > 0:
            # multiple signals are ignored, take only the first one
            sig = self.signals[0]
            # convert index -1 to length to capture correct range (i.e. all)!
            if iend == -1:
                iend = len(sig)
            # corresponding title: first case
            title = self.cases[0]
            # the legend is stored in a tuple
            legend = tuple()
            # the y-labels are also stored in a tuple
            ylabels = tuple()
            if self.debug: print 'sig.shape:', sig.shape
            #-------------------------------------------------------------------
            # overplot the actual signals on the left axis
            ax1 = fig.add_subplot(111)
            ylabels = tuple()
            i=0
            for chan in self.channels[0]:
                # force datatype on the legend label entry
                label = str(self.ch_details[chan,0])

                # select coler
                c = self.collors_left_axis[i] + self.plotsymbolleft[i]

                if len(self.markersizeleft) > 0:
                    m = self.markerleft[i]
                    ms = self.markersizeleft[i]
                    mfc = self.markerfacecolorleft[i]
                    mec = self.markeredgecolorleft[i]
                    ls = self.linestyelleft[i]
                    lw = self.linewidthleft[i]
                    me = self.markeveryleft[i]
                    # actual and factual plot command is finally issued here!
                    ax1.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],c,
                             marker=m, ms=ms, mfc=mfc, mec=mec, ls=ls, lw=lw,
                             markevery=me, label=label)
                else:
                    ax1.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],c,
                             label=label)

                if self.debug: print 'plot: sig[' + str(ibeg) + \
                    ':' + str(iend) + ',' + str(xchan) + '], ' + \
                    'sig[' + str(ibeg) + ':' +str(iend) +','+ str(chan) + ']'


                i+=1
                # channeld id in legend
                legend += (label,)
                # save all the units, to be used for setting y-axis label later
                ylabels += (self.ch_details[chan,1],)

                # save the data for replotting
                key = self.ch_details[chan,0]
                datasave[key] = [sig[ibeg:iend,xchan]-x_0, sig[ibeg:iend,chan]]


            # set legend
            if legendoutside:
                    # top right of figure. BboxBase, tuple of 4 floats
                    # (x, y, width, height of the bbox)
#                    leg = fig.legend(bbox_to_anchor=(0, 0, 1, 0.9),
#                             bbox_transform=plt.gcf().transFigure, title=ylabel)
                    leg = ax1.legend(bbox_to_anchor=(0, 0, 1, 0.9),
                             bbox_transform=fig.transFigure, title=ylabel)
            else:
                leg = ax1.legend(legend, legend_left, shadow=True)

            # and alter the legend font size
            for t in leg.get_texts():
                t.set_fontsize(legendsize)

            # set the x-axis label
            xlabel = self.ch_details[xchan,0]+' ['+self.ch_details[xchan,1]+ ']'
            ax1.set_xlabel(xlabel, size=labelsize)

            # y-label only units (combination of all units plotted)
            # create a string first based on the tuple and make sure each
            # unit appears only once! > select unique labels:
            if type(ylabel_man_left).__name__ == 'str':
                    ylabel = ylabel_man_left
            else:
                ylabel_unique = unique(ylabels)
                ylabel = ''
                for k in ylabel_unique:
                    ylabel += k + ', '
                # remove last ', '
                ylabel = ylabel[0:len(ylabel)-2]

#            # Make the y-axis label and tick labels match the line color.
#            # only applicable for one channel. Take only the first character of
#            # the string color, it can contain line propertie information
#            c = self.collors_left_axis[0][0]
#            if len(self.channels[0]) == 1:
#                ax1.set_ylabel(ylabel, color=c, size=labelsize)
#                for tl in ax1.get_yticklabels():
#                    tl.set_color(c)
#            # for more channels, just give the label and axis standard black
#            else:
#                ax1.set_ylabel(ylabel, color='k', size=labelsize)
            # y-axis always black!
            ax1.set_ylabel(ylabel, color='k', size=labelsize)

            # hard press the axis limits if given
            # if limits:
            if len(xlim_left) == 2:
                ax1.set_xlim(xlim_left)
            if len(ylim_left) == 2:
                ax1.set_ylim(ylim_left)

            # set horizontal line on left axis:
            if type(self.hor_line_left).__name__== 'float':
                ax1.axhline(y=self.hor_line_left, linewidth=1, color='k',\
                linestyle='-', aa=False)

            # save the number of ticks on the left axis
            nr_yticks_left = len(ax1.get_yticks())
            # isn't there an alternative method?
            # ax1y = ax1.get_axes()
            # the grid can also be accessed somehow there
            ax1.grid(True)

            datasave['ylabel_left'] = ylabel
            datasave['xlabel'] = xlabel
            #-------------------------------------------------------------------

            # new legend for the right axis
            legend = tuple()
            #-------------------------------------------------------------------
            # overplot the actual signals on the right axis
            if nr_chan_right > 0:
                # switch to the right axis
                ax2 = ax1.twinx()
                ylabels = tuple()
                i=0
                for chan in self.channels[1]:
                    # force datatype on the legend label entry
                    label = str(self.ch_details[chan,0])

                    # select coler
                    c = self.collors_right_axis[i] + self.plotsymbolright[i]

                    if len(self.markersizeright) > 0:
                        m = self.markerright[i]
                        ms = self.markersizeright[i]
                        mfc = self.markerfacecolorright[i]
                        mec = self.markeredgecolorright[i]
                        ls = self.linestyelright[i]
                        lw = self.linewidthright[i]
                        me = self.markeveryright[i]
                        # actual and factual plot command finally issued here!
                        ax2.plot((sig[ibeg:iend,xchan]-x_0),sig[ibeg:iend,chan],
                                 c,marker=m, ms=ms, mfc=mfc,mec=mec,ls=ls,
                                 lw=lw,markevery=me, label=label)
                    else:
                        ax2.plot((sig[ibeg:iend,xchan]-x_0),sig[ibeg:iend,chan],
                                c, label=label)
                    i += 1
                    # legend: channeld id
                    legend += (label,)
                    # save all units, to be used for setting y-axis label later
                    ylabels += (self.ch_details[chan,1],)

                    # save the data for replotting
                    key = self.ch_details[chan,0]
                    datasave[key] = [ sig[ibeg:iend,xchan]-x_0, \
                                      sig[ibeg:iend,chan] ]

                # y-label only units (combination of all units plotted)
                # create a string first based on the tuple and make sure each
                # unit appears only once! > select unique labels:
                if type(ylabel_man_right).__name__ == 'str':
                    ylabel = ylabel_man_right
                else:
                    ylabel_unique = unique(ylabels)
                    ylabel = ''
                    for k in ylabel_unique:
                        # ylabel += '[' + k + '] '
                        ylabel += k
                    # remove last ', '
                    ylabel = ylabel[0:len(ylabel)-2]

                # set the legend
                if legendoutside:
                    # BboxBase, tuple of 4 floats (x, y, width, height of the
                    # bbox). bottom right of figure
                    leg = ax2.legend(bbox_to_anchor=(0, 0, 1, 0.45),
                             bbox_transform=fig.transFigure, title=ylabel)
                else:
                    leg = ax2.legend(legend, legend_right, shadow=True)

                # and alter the legend font size
                for t in leg.get_texts():
                    t.set_fontsize(legendsize)


#                # Make the y-axis label and tick labels match the line color.
#                # only applicable for one channel. Take only the first
#                # character of the string color, it can contain line property
#                # information
#                c = self.collors_right_axis[0][0]
#                if len(self.channels[1]) == 1:
#                    ax2.set_ylabel(ylabel, color=c, size=labelsize)
#                    for tl in ax2.get_yticklabels():
#                        tl.set_color(c)
#                # for more channels, give the label and axis standard black
#                else:
#                    ax2.set_ylabel(ylabel, color='k', size=labelsize)

                # all is black on the y-axis!
                ax2.set_ylabel(ylabel, color='k', size=labelsize)

                # hard press the axis limits if given
                if len(xlim_right) == 2:
                    ax2.set_xlim(xlim_right)
                if len(ylim_right) == 2:
                    ax2.set_ylim(ylim_right)

                # get the current ticks of the right axis
                yticks_right = ax2.get_yticks()
                tickmin, tickmax = yticks_right[0], yticks_right[-1]
                tickloc_yleft = np.linspace(tickmin,tickmax, num=nr_yticks_left)
                ax2.yaxis.set_ticks(tickloc_yleft)
                # and set the precision nicely
                ax2.yaxis.set_ticklabels(["%.0f"% val for val in tickloc_yleft])

                # set horizontal line on right axis:
                if type(self.hor_line_right).__name__== 'float':
                    ax2.axhline(y=self.hor_line_right, linewidth=1, color='k',\
                    linestyle='-', aa=False)

                ax2.grid(True)

                datasave['ylabel_right'] = ylabel
            #-------------------------------------------------------------------
            # plt.show()

            # set a vertical line
            if type(self.ver_line).__name__== 'float':
                # print 'the vertical line: ' + str(self.ver_line)
                ax1.axvline(x=self.ver_line, linewidth=1, color='k',\
                linestyle='--', aa=False)


            # how to control the grid ticks? This does not work
            # mpl.axes.set_xticks([range(500,520,1)])

            # set title
            ax1.set_title(title, size=titlesize, horizontalalignment='center')

        #-----------------------------------------------------------------------

        #-----------------------------------------------------------------------
        # TODO: implement annotation boxes:
#        fig = figure(1,figsize=(8,5))
#        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-4,3))
#
#        t = np.arange(0.0, 5.0, 0.01)
#        s = np.cos(2*np.pi*t)
#        line, = ax.plot(t, s, lw=3, color='purple')
#
#        ax.annotate('arrowstyle', xy=(0, 1),  xycoords='data',
#                    xytext=(-50, 30), textcoords='offset points',
#                    arrowprops=dict(arrowstyle="->")
#                    )
        #-----------------------------------------------------------------------

        datasave['title'] = title

        # save the figure if required
        if save:
            # if no extension specified, save in both eps and png
            if figname.endswith('.png') or figname.endswith('.jpg') \
                or figname.endswith('.jpeg') or figname.endswith('.eps'):
                fig.savefig(figpath+figname, orientation='landscape')
            else:
                fig.savefig(figpath + figname + '.png', orientation='landscape')
                fig.savefig(figpath + figname + '.eps', orientation='landscape')


        # and now close to the figure, otherwise the figure will keep
        # existing in the memory and this will look like a memory leak in a loop
        canvas.close()
        fig.clear()

        # save the data
        figname = figname.replace('.png', '').replace('.eps', '')
        figname = figname.replace('.jpg', '').replace('.jpeg', '')
        fname = figpath + figname + '.pkl'
        file = open(fname, 'w')
        pickle.dump(datasave, file)
        file.close()

    def plot_flex(self, **kwargs):
#        figsize_x=16, figsize_y=9, \
#            dpi=100,save=False,figdir='',figname='',figtitle='',\
#            legend_left='upper left', legend_right='upper right', \
#            xlim_left=[], ylim_left=[], xlim_right=[], ylim_right=[],\
#            labelsize='medium', legendsize='medium', titlesize='medium',\
#            wsleft=0.08, wsbottom=0.1, wsright=0.95, wstop=0.95, wspace=0.2,\
#            hspace=0.2,legendoutside=False,
        """Create overplots for channels or cases, 2nd iteration (based on
        overplot2D)

        allow flexible input, which is not uniform and has different
        x-values etc. Work thourgh dictionaries instead of directly with
        the numpy arrays. This approach allows better integration with the
        updated htc_dict methods and classes.

        Input is list of plot_dict's. Each plot_dict corresponds to a different
        overplot. Each plot_dict holds an array containing all channels and data
        points. Also other plot settings are included, like what channels on the
        left and right, index range, x_0, sig_details.

        > plot_list
        is a list of dictionaries, holding the following key/value pairs
        (these dictionaries are cold plot_dict):
            sig : signal array (as used before, channels on axis 1,
                                different data points on axis 0)
            ibeg : start index, default = 0
            iend : end index,   default = -1 (last entry, converts to len(sig) )
            xchan: default = 0
            ychan_left: list holding the channels to plot
            ychan_right: list holding the channels to plot
            sig_details: indicate the legend text, should match the included
                        channels as given in sig. Format it like sig_details
            x_0: value to subtracted from the x-channel

        ** kwargs:
        > figdir: location of the figure (was figpath before!)
        > figname: used for saving
        > figtitle: default is the same as figname
        > legend_left: default = upper left
        > legend_right: legend location, default = upper right
        > xlim_left
        > xlim_right
        > figsize_x
        > figsize_y
        > dpi
        > save

        Usage:
        p = HawcPy.PlotSignal(plot_dict, '', '', '', '')
        p.plot_flex(**kwargs)
            **kwargs: figsize_x,figsize_y (inches), dpi
                      save (boolean), figpdir, figname, figtitle,
                      legend_left, legend_right, (location of the legends)
                      xlim_left, xlim_right, ylim_left, ylim_right
            all **kwargs are optional
        for example: p.plot_flex(legend_left='lower left', xlim_left=[0 12])

        The optional input arguments ibeg, iend and xchan allows to control
        the range of the plot (ibeg and iend indicating the beginning and
        ending indices of the HAWC2 signal) and to specificy on which channel
        (or better index, ichan) the x-axis is located.
        """

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        # default values for kwargs
        figsize_x = kwargs.get('figsize_x',16)
        figsize_y = kwargs.get('figsize_y',9)
        dpi = kwargs.get('dpi',100)
        save = kwargs.get('save',False)
        figdir = kwargs.get('figdir','')
        figname = kwargs.get('figname','')
        figtitle = kwargs.get('figtitle','')
        legend_left = kwargs.get('legend_left','upper left')
        legend_right = kwargs.get('legend_right', 'upper right')
        xlim_left = kwargs.get('xlim_left',[])
        ylim_left = kwargs.get('ylim_left',[])
        xlim_right = kwargs.get('xlim_right',[])
        ylim_right = kwargs.get('ylim_right',[])
        labelsize = kwargs.get('labelsize', 'medium')
        legendsize = kwargs.get('legendsize', 'medium')
        titlesize = kwargs.get('titlesize', 'medium')
        wsleft = kwargs.get('wsleft', 0.08)
        wsbottom = kwargs.get('wsbottom', 0.1)
        wsright = kwargs.get('wsright', 0.95)
        wstop = kwargs.get('wstop', 0.95)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)
        legendoutside = kwargs.get('legendoutside',False)
        plotmap_amp = kwargs.get('plotmap_amp', None)
        plotmap_damp = kwargs.get('plotmap_damp', None)
        # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        # Becase this whole plotting procedure is crap, save the data to
        # a csv file and plot in better layout
        datasave = {}

        # data check, aborts execution if exception is thrown
        # self.data_check()

        # define the keys that has to be present in the channels
        # note that only the left channel is required for plotting
        chan_keys = ['sig','ibeg','iend','xchan','ychan','sig_details','x_0',\
                     'ychan_left']
        # default values for the different keys:
        default_dict = dict()
        default_dict['ibeg'] = 0
        default_dict['iend'] = -1
        default_dict['xchan'] = 0
        default_dict['ychan'] = [1]
        default_dict['sig_details'] = ''
        default_dict['x_0'] = 0
        # we could also add the plot color and symbol in this way, but than we
        # would have to define it in each dictionary, which might not be that
        # convienent?

        plot_list = self.signals

#        # create plot_list for left and right plotting:
#        channel_left, channel_right = [], []
#        for plot_dict in self.signals:
#            if 'ychan_left' in plot_dict:
#                channel_left.append(plot_dict)
#            elif 'ychan_right' in plot_dict:
#                channel_right.append(plot_dict)

        # DATA SANITY CHECK
        # empty plot_list contains an empty dictionary
        if len(plot_list) < 2 and len(plot_list[0]) < 1:
            raise UserWarning, 'E100: the suplied plot_list is empty!'
        # Check if all required keys are present in the channel lists.
        # If not, add the default values
        for plot_dict in plot_list:
            # go through the required keys and see if they are present in the
            # given channel dictionary
            for key in chan_keys:
                # key is present, which is fine!
                if key in plot_dict:
                    # TODO: do a data check on the corresponding value!
                    pass
                # if the sig key is not present, raise an error, we can't have
                # default value here!
                elif key == 'sig':
                    raise UserWarning, 'E101: sig key in plot_dict is missing!'
                # key is not present, give default value
                else:
                    try:
                        plot_dict[key] = default_dict[key]
                    except:
                        raise UserWarning, 'E101: missing key in plot_dict: '+\
                                key + ' ...'

        fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        canvas = FigureCanvas(fig)
        fig.set_canvas(canvas)

        # -------------------------------------------------------------------
        # overplot the actual signals on the LEFT axis
        # add_subplot(nr rows nr cols plot_number)
        if legendoutside:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(111)

        # define the whitespaces in percentages of the total width and height
        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                            top=wstop, wspace=wspace, hspace=hspace)

#        # l,b,w,h
#        rec = [wsleft, wsbottom, wsright, wstop]
#        ax1 = fig.add_axes(rec)

        # -------------------------------------------------------------------
        # is there a color map as background?
        # only consider the color map of the first overplot, the rest is not
        # relevant. Overplotting maps does not provide a clear plot
        if 'sig_amp' in plot_list[0] and plotmap_amp == True:
            data = plot_list[0]['sig_amp']
            # sort on the given x channel of the first plot dict
            xchan = plot_list[0]['xchan']
            data = data[:,np.argsort(plot_list[0]['sig'][:,xchan])]

            cmapbin = mpl.cm.get_cmap('binary')
            ax1.imshow(data, cmap=cmapbin,
                           interpolation="nearest", origin="lower")

        elif 'sig_damp' in plot_list[0] and plotmap_damp == True:
            data = plot_list[0]['sig_damp']
            # sort on the given x channel of the first plot dict
            xchan = plot_list[0]['xchan']
            data = data[:,np.argsort(plot_list[0]['sig'][:,xchan])]

            ax1.imshow(data, cmap=cmapbin,
                       interpolation="nearest", origin="lower")
        # -------------------------------------------------------------------
        # START LEFT
        # -------------------------------------------------------------------
        # new legend for the left axis
        legend = tuple()

        ylabels = tuple()
        i=0 # tracking the number of overplots on the left
        # cycle through the different data sets. Each set could hold more
        # channels to plot!
        if self.debug: print 'about to start left plotting...'
        for plot_dict in plot_list:
            # for each dict, we load the values:
            xchan = plot_dict['xchan']
            sig = plot_dict['sig']
            x_0 = plot_dict['x_0']
            ibeg = plot_dict['ibeg']
            iend = plot_dict['iend']
            if iend == -1:
                iend = len(sig)

            # sort the signal on the first channel
            sort_args = np.argsort(sig[:,xchan])
            sig = sig[sort_args,:]

            # now cycle through all the channels for this signal
            for chan in plot_dict['ychan_left']:
                # force datatype on the legend label entry
                label = str(plot_dict['sig_details'][chan,0])

                # select color
                if self.debug: print 'index for color: '+str(i),
                if self.debug: print 'len color:', len(self.collors_left_axis)
                c = self.collors_left_axis[i] + self.plotsymbolleft[i]
                # TODO: make different plot types available!!
                # possible options: error bars, contour, ...
                ax1.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],c,
                         label=label)
                # channeld id in legend
                legend += (label,)
                # save all the units, to be used for setting y-axis label later
                ylabels += (plot_dict['sig_details'][chan,1],)
                i+=1

                # save the data for replotting
                key = plot_dict['sig_details'][chan,0]
                datasave[key] = [sig[ibeg:iend,xchan]-x_0, sig[ibeg:iend,chan]]

        # set the x-axis label: take it from the last plot_dict (they should all
        # hold the same x-channel and type of x-values!!) treated above
        xlabel = plot_dict['sig_details'][xchan,0] + ' [' \
               + plot_dict['sig_details'][xchan,1] + ']'

        ax1.set_xlabel(xlabel, size=labelsize)
        datasave['xlabel'] = xlabel

        # y-label only units (combination of all units plotted)
        # create a string first based on the tuple and make sure each
        # unit appears only once! > select unique labels:
        ylabel_unique = unique(ylabels)
        ylabel = ''
        for k in ylabel_unique:
#            ylabel += '[' + k + '] '
            ylabel += k + ', '
        # remove last ', '
        ylabel = ylabel[0:len(ylabel)-2]

        # for more channels, just give the label and axis standard black
        ax1.set_ylabel(ylabel, color='k', size=labelsize)
        datasave['ylabel_left'] = ylabel

#        # Make the y-axis label and tick labels match the line color.
#        # only applicable for one channel. Take only the first character of
#        # the string color, it can contain line propertie information
#        c = self.collors_left_axis[0][0]
#        if len(self.channels[0]) == 1:
#            ax1.set_ylabel(ylabel, color=c)
#            for tl in ax1.get_yticklabels():
#                tl.set_color(c)
#        # for more channels, just give the label and axis standard black
#        else:
#            ax1.set_ylabel(ylabel, color='k')

        # hard press the axis limits if given
        # if limits:
        if len(xlim_left) == 2:
            ax1.set_xlim(xlim_left)
        if len(ylim_left) == 2:
            ax1.set_ylim(ylim_left)

        ax1.grid(True)

        # set legend
        if legendoutside:
#                plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
                # top right of figure
                # BboxBase, tuple of 4 floats (x, y, width, height of the bbox)
                leg = ax1.legend(bbox_to_anchor=(0, 0, 1, 0.9),
                           bbox_transform=fig.transFigure, title=ylabel)
        else:
            leg = ax1.legend(legend, legend_left, shadow=True)

        # and alter the legend font size
        for t in leg.get_texts():
            t.set_fontsize(legendsize)

        # set horizontal line on left axis:
        if type(self.hor_line_left).__name__== 'float':
            ax1.axhline(y=self.hor_line_left, linewidth=1, color='k',\
            linestyle='-', aa=False)

        # save the number of ticks on the left axis
        nr_yticks_left = len(ax1.get_yticks())
        # -------------------------------------------------------------------
        # END LEFT
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # START RIGHT
        # -------------------------------------------------------------------
        # new legend for the right axis
        legend = tuple()
        # is there anything to be plot on the right?
        plot_right = False
        for plot_dict in self.signals:
            if 'ychan_right' in plot_dict:
                if len(plot_dict['ychan_right']) > 0:
                    plot_right = True
                    break
        # -------------------------------------------------------------------
        # overplot the actual signals on the RIGHT axis
        # only true if there is anything to be plot on the right
        if plot_right:
            if self.debug: print 'entering right channel plotting...'
            # switch to the right axis
            ax2 = ax1.twinx()
            ylabels = tuple()
            i=0 # tracking the number of overplots on the right
            # cycle through the different data sets. Each set could hold more
            # channels to plot!
            for plot_dict in plot_list:
                # for each dict, we load the values:
                xchan = plot_dict['xchan']
                sig = plot_dict['sig']
                x_0 = plot_dict['x_0']
                ibeg = plot_dict['ibeg']
                iend = plot_dict['iend']

                # sort the signal on the first channel
                sort_args = np.argsort(sig[:,xchan])
                sig = sig[sort_args,:]

                if iend == -1:
                    iend = len(sig)
                # now cycle through all the channels for this signal
                for chan in plot_dict['ychan_right']:
                    # force datatype on the legend label entry
                    label = str(plot_dict['sig_details'][chan,0])

                    # select color
                    if self.debug: print 'index for color: '+str(i)
                    if self.debug:
                        print 'len color:', len(self.collors_left_axis)
                    c = self.collors_right_axis[i] + self.plotsymbolright[i]
                    # TODO: make different plot types available!!
                    # possible options: error bars, contour, ...
                    ax2.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],c,
                             label=label)
                    # channeld id in legend
                    legend += (label,)
                    # save all the units, for setting y-axis label later
                    ylabels += (plot_dict['sig_details'][chan,1],)
                    i+=1

                    # save the data for replotting
                    key = plot_dict['sig_details'][chan,0]
                    # if the labels on the right are the same as on the left
                    # axes, they will be overwritten. Prevend if the case
                    if key in datasave:
                        key = 'right_' + key
                    datasave[key] = [ sig[ibeg:iend,xchan]-x_0, \
                                      sig[ibeg:iend,chan] ]

            # y-label only units (combination of all units plotted)
            # create a string first based on the tuple and make sure each
            # unit appears only once! > select unique labels:
            ylabel_unique = unique(ylabels)
            ylabel = ''
            for k in ylabel_unique:
#                ylabel += '[' + k + '] '
                ylabel += k + ', '
            # remove last ', '
            ylabel = ylabel[0:len(ylabel)-2]

            # set the legend
            if legendoutside:
#                plt.legend(bbox_to_anchor=(1.05, 0.1), loc=4, borderaxespad=0.)
                # BboxBase, tuple of 4 floats (x, y, width, height of the bbox)
                # bottom right of figure
                leg = ax2.legend(bbox_to_anchor=(0, 0, 1, 0.45),
                           bbox_transform=fig.transFigure, title=ylabel)
            else:
                leg = ax2.legend(legend, legend_right, shadow=True)

            # and alter the legend font size
            for t in leg.get_texts():
                t.set_fontsize(legendsize)

            # for more channels, just give the label and axis standard black
            ax2.set_ylabel(ylabel, color='k', size=labelsize)
            datasave['ylabel_right'] = ylabel

#            # Make the y-axis label and tick labels match the line color.
#            # only applicable for one channel. Take only the first
#            # character of the string color, it can contain line property
#            # information
#            c = self.collors_right_axis[0][0]
#            if len(self.channels[1]) == 1:
#                ax2.set_ylabel(ylabel, color=c)
#                for tl in ax2.get_yticklabels():
#                    tl.set_color(c)
#            # for more channels, just give the label and axis standard black
#            else:
#                ax2.set_ylabel(ylabel, color='k')

            # hard press the axis limits if given
            if len(xlim_right) == 2:
                ax2.set_xlim(xlim_right)
            if len(ylim_right) == 2:
                ax2.set_ylim(ylim_right)

            # match the number of yticks with the left axis
            # get the current ticks from the right axis

            # this will only offset the ticks by the provided value (int)
#            ax2.yaxis.get_major_locator().pan(5)

            # get the current ticks of the right axis
            yticks_right = ax2.get_yticks()
            tickmin, tickmax = yticks_right[0], yticks_right[-1]
            tickloc_yleft = np.linspace(tickmin,tickmax, num=nr_yticks_left)
            ax2.yaxis.set_ticks(tickloc_yleft)
            # and set the precision nicely
            ax2.yaxis.set_ticklabels(["%.0f" % val for val in tickloc_yleft])

#            ymajorLocator = mpl.ticker.MultipleLocator(base=0.1)
#            ax.yaxis.set_major_locator( ymajorLocator )
#            xmajorFormatter = mpl.ticker...
#            ax.xaxis.set_major_formatter( xmajorFormatter )

#            ax2.yaxis.get_major_formatter().format_data('1.02f')

#            tickloc_yleft = ax2.yaxis.get_major_locator()
#            mpl.ticker.FixedLocator(tickloc_yleft,nbins=nr_yticks_left-1)

#            tickloc_yleft = mpl.ticker.pan(nr_yticks_left)
#            ax2.yaxis.set_major_locator(tickloc_yleft)

#            tickloc_yleft = mpl.ticker.FixedLocator(nr_yticks_left)
#            ax2.yaxis.set_major_locator(\
#                mpl.ticker.MaxNLocator(\
#                tickloc_yleft, nrbins=nr_yticks_left))

            # set horizontal line on right axis:
            if type(self.hor_line_right).__name__== 'float':
                ax2.axhline(y=self.hor_line_right, linewidth=1, color='k',\
                linestyle='-', aa=False)

            ax2.grid(True)
        # -------------------------------------------------------------------

        # set a vertical line
        if type(self.ver_line).__name__== 'float':
            # print 'the vertical line: ' + str(self.ver_line)
            # or ax2.axvline, ax1.axvline
            ax1.axvline(x=self.ver_line, linewidth=1, color='k',\
            linestyle='--', aa=False)

        # -------------------------------------------------------------------
        # END RIGHT
        # -------------------------------------------------------------------

        # how to control the grid ticks? This does not work
        # mpl.axes.set_xticks([range(500,520,1)])

        # set title
        ax1.set_title(figtitle, size=titlesize, horizontalalignment='center')
        datasave['title'] = figtitle
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # TODO: implement annotation boxes:
#        fig = figure(1,figsize=(8,5))
#        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-4,3))
#
#        t = np.arange(0.0, 5.0, 0.01)
#        s = np.cos(2*np.pi*t)
#        line, = ax.plot(t, s, lw=3, color='purple')
#
#        ax.annotate('arrowstyle', xy=(0, 1),  xycoords='data',
#                    xytext=(-50, 30), textcoords='offset points',
#                    arrowprops=dict(arrowstyle="->")
#                    )
        # ----------------------------------------------------------------------

        # save the figure if required
        if save:
            # if no extension specified, save in both eps and png
            if figname.endswith('.png') or figname.endswith('.jpg') \
                or figname.endswith('.jpeg') or figname.endswith('.eps'):
                fig.savefig(figdir+figname, orientation='landscape')
            else:
                fig.savefig(figdir + figname + '.png', orientation='landscape')
                fig.savefig(figdir + figname + '.eps', orientation='landscape')

        # and now close to the figure, otherwise the figure will keep
        # existing in the memory and this will look like a memory leak in a loop
        canvas.close()
        fig.clear()

        # save the data
        figname = figname.replace('.png', '').replace('.eps', '')
        figname = figname.replace('.jpg', '').replace('.jpeg', '')
        fname = figdir + figname + '.pkl'
        file = open(fname, 'w')
        pickle.dump(datasave, file)
        file.close()

    def plot_dyn(self, **kwargs):
        """
        Plot Dynamic properties
        =======================

        Based on plot_flex, input is the same plot_dict, but with additional
        keys: str(chi) + '_sig_damp_list' and str(chi) + '_sig_amp_list'
        Where they hold the 2D arrays to plot

        Important difference: only one plot list is allowed. Overplots are not
        realy relevant in this context, unless

        Parameters
        ----------

        Returns
        -------

        """

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        # default values for kwargs
        figsize_x = kwargs.get('figsize_x',16)
        figsize_y = kwargs.get('figsize_y',9)
        dpi = kwargs.get('dpi',100)
        save = kwargs.get('save',False)
        figdir = kwargs.get('figdir','')
        figname = kwargs.get('figname','')
        figtitle = kwargs.get('figtitle','')
#        legend_left = kwargs.get('legend_left','upper left')
#        legend_right = kwargs.get('legend_right', 'upper right')
        xlim_left = kwargs.get('xlim_left',[])
        ylim_left = kwargs.get('ylim_left',[])
#        xlim_right = kwargs.get('xlim_right',[])
#        ylim_right = kwargs.get('ylim_right',[])
        labelsize = kwargs.get('labelsize', 'large')
#        legendsize = kwargs.get('legendsize', 'medium')
        titlesize = kwargs.get('titlesize', 'medium')
        wsleft = kwargs.get('wsleft', 0.08)
        wsbottom = kwargs.get('wsbottom', 0.1)
        wsright = kwargs.get('wsright', 0.95)
        wstop = kwargs.get('wstop', 0.95)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)
#        legendoutside = kwargs.get('legendoutside',False)
        plotmap_amp = kwargs.get('plotmap_amp', None)
        plotmap_damp = kwargs.get('plotmap_damp', None)
        # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        # data check, aborts execution if exception is thrown
        # self.data_check()

        # define the keys that has to be present in the channels
        # note that only the left channel is required for plotting
        chan_keys = ['sig','ibeg','iend','xchan','ychan','sig_details','x_0',\
                     'ychan_left']
        # default values for the different keys:
        default_dict = dict()
        default_dict['ibeg'] = 0
        default_dict['iend'] = -1
        default_dict['xchan'] = 0
        default_dict['ychan'] = [1]
        default_dict['sig_details'] = ''
        default_dict['x_0'] = 0
        # we could also add the plot color and symbol in this way, but than we
        # would have to define it in each dictionary, which might not be that
        # convienent?

        plot_list = self.signals

        # DATA SANITY CHECK
        # empty plot_list contains an empty dictionary
        if len(plot_list) < 2 and len(plot_list[0]) < 1:
            raise UserWarning, 'E100: the suplied plot_list is empty!'
        # Check if all required keys are present in the channel lists.
        # If not, add the default values
        for plot_dict in plot_list:
            # go through the required keys and see if they are present in the
            # given channel dictionary
            for key in chan_keys:
                # key is present, which is fine!
                if key in plot_dict:
                    # TODO: do a data check on the corresponding value!
                    pass
                # if the sig key is not present, raise an error, we can't have
                # default value here!
                elif key == 'sig':
                    raise UserWarning, 'E101: sig key in plot_dict is missing!'
                # key is not present, give default value
                else:
                    try:
                        plot_dict[key] = default_dict[key]
                    except:
                        raise UserWarning, 'E101: missing key in plot_dict: '+\
                                key + ' ...'

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        # add_subplot(nr rows nr cols plot_number)
        ax1 = fig.add_subplot(111)

        # define the whitespaces in percentages of the total width and height
#        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
#                            top=wstop, wspace=wspace, hspace=hspace)

        # -------------------------------------------------------------------
        # is there a color map as background?
        # only consider the color map of the first overplot, the rest is not
        # relevant. Overplotting maps does not provide a clear plot
        if 'sig_amp' in plot_list[0] and plotmap_amp == True:
            data = plot_list[0]['sig_amp']
            # sort on the given x channel of the first plot dict
            xchan = plot_list[0]['xchan']
            # data can be sorted on sig, not sig_amp but they have been
            # constructed in the same manner. Sorting sig wil sort sig_amp
            data = data[:,np.argsort(plot_list[0]['sig'][:,xchan])]

            im = ax1.imshow(data, cmap=plt.get_cmap('binary'),
                           interpolation="nearest", origin="lower")
            plt.colorbar(im)

        elif 'sig_damp' in plot_list[0] and plotmap_damp == True:
            data = plot_list[0]['sig_damp']
            # sort on the given x channel of the first plot dict
            xchan = plot_list[0]['xchan']
            data = data[:,np.argsort(plot_list[0]['sig'][:,xchan])]

            im = ax1.imshow(data, cmap=plt.get_cmap('binary'),
                           interpolation="nearest", origin="lower")
            plt.colorbar(im)

        # set the x-labels correct. They have to come from sig
        xlabels = np.sort(plot_list[0]['sig'][:,xchan])

        # get the current ticks of the right axis
#        xticks_left, xlabels_left = ax1.xticks()
#        tickmin, tickmax = xticks_left[0], xticks_left[-1]
#        tickloc_xleft = np.linspace(tickmin,tickmax, num=len(xticks_left))
#        ax1.xaxis.set_ticks(xlabels)

        # change xticks according to wind channel
        xticks_left, xlabels_left = plt.xticks()
        # for the wind speeds, this would be the xticks
        # [ -5.   0.   5.  10.  15.  20.]
        # where 0 is the first value of the wind speeds
        xticks_left_new = xticks_left + xlabels[0]
        if self.debug:
            print 'xticks proposed2', xticks_left_new
        ax1.xaxis.set_ticklabels(["%.0f" % val for val in xticks_left_new])

        # on the yaxis, we have the cycle number, which should start at 1
        yticks_left, ylabels_left = plt.yticks()
        yticks_left_new = yticks_left + 1
        ax1.yaxis.set_ticklabels(["%.0f" % val for val in yticks_left_new])

        if len(ylim_left) > 1:
            ax1.set_ylim(ylim_left)

        if len(xlim_left) > 1:
            ax1.set_xlim(xlim_left)

        # xaxis label
        ax1.set_xlabel('Windspeed', size=labelsize)
        ax1.set_ylabel('nr of cycles', size=labelsize)


        # define the whitespaces in percentages of the total width and height
        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                            top=wstop, wspace=wspace, hspace=hspace)
        plt.grid(True)
        # set title
        plt.title(figtitle, size=titlesize, horizontalalignment='center')

#        plt.draw()
#        plt.show()

        # save the figure if required
        if save:
            # if no extension specified, save in both eps and png
            if figname.endswith('.png') or figname.endswith('.jpg') \
                or figname.endswith('.jpeg') or figname.endswith('.eps'):
                fig.savefig(figdir+figname, orientation='landscape')
            else:
                fig.savefig(figdir + figname + '.png', orientation='landscape')
                fig.savefig(figdir + figname + '.eps', orientation='landscape')

        # and now close to the figure, otherwise the figure will keep
        # existing in the memory and this will look like a memory leak in a loop
        plt.close('all')


    def plot_start():
        """Start plotting:
            give the signal array, labels, channels, make loop for overplotting
            return the figure object
        """
        pass

    def plot_end(figure):
        """End plotting:
            give back the figure object, where now all kind of nice things
            have happened to: setting correct size of plot markers, white spaces
            etc
        """

#    def savefig(self, fig_path, fig_name):
#        """Save the current figure
#        """
#
#        mpl.pyplot.savefig(fig_path+fig_name, facecolor='w', \
#            edgecolor='w', orientation='landscape', papertype=None, \
#            format=None, transparent=False)

class ModelData:
    # DEPRICATED, use Simulations.ModelData instead!!
    """
    DEPRICATED, use Simulations.ModelData instead!!

    Make plots of defined HAWC2 model:
        * aerodynamic coeficients in the .pc file
        * chord and twist distributions in the .ae file
        * structural properties in the .st file
    """

    class st_headers:
        """
        Indices to the respective parameters in the HAWC2 st data file
        """
        r     = 0
        m     = 1
        x_cg  = 2
        y_cg  = 3
        ri_x  = 4
        ri_y  = 5
        x_sh  = 6
        y_sh  = 7
        E     = 8
        G     = 9
        Ixx   = 10
        Iyy   = 11
        I_p   = 12
        k_x   = 13
        k_y   = 14
        A     = 15
        pitch = 16
        x_e   = 17
        y_e   = 18

    def __init__(self):
        """
        """

        self.data_path = None
        # saving plots in the same directory
        self.save_path = None
        self.ae_file = 'somefile.ae'
        self.pc_file = 'somefile.pc'
        self.st_file = 'somefile.st'
        self.debug = False

        # relevant for write_st2
        self.st_file2 = 'somefile2.st'
        self.st2_filemode = 'w'
        # define the column width for printing
        self.col_width = 13
        # formatting and precision
        self.float_hi = 999.9999
        self.float_lo =  0.01
        self.prec_float = ' 8.04f'
        self.prec_exp =   ' 8.04e'

        #0 1  2    3    4    5    6    7   8 9 10   11
        #r m x_cg y_cg ri_x ri_y x_sh y_sh E G I_x  I_y
        #12    13  14  15  16  17  18
        #I_p/K k_x k_y A pitch x_e y_e
        # 19 cols
        self.st_column_header_list = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', \
            'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'I_p/K', 'k_x', \
            'k_y', 'A', 'pitch', 'x_e', 'y_e']

        self.st_column_header_list_latex = ['r','m','x_{cg}','y_{cg}','ri_x',\
            'ri_y', 'x_{sh}','y_{sh}','E', 'G', 'I_x', 'I_y', 'J', 'k_x', \
            'k_y', 'A', 'pitch', 'x_e', 'y_e']

        self.column_header_line = ''
        for k in self.st_column_header_list:
            self.column_header_line += k.rjust(self.col_width)

    def _data_checks(self):
        """
        Data Checks on self
        ===================
        """
        if self.data_path is None:
            raise UserWarning, 'specify data_path first'


    def fromline(self, line, separator=' '):
        # TODO: move this to the global function space (dav-general-module)
        """
        split a line, but ignore any blank spaces and return a list with only
        the values, not empty places
        """
        # remove all tabs, new lines, etc? (\t, \r, \n)
        line = line.replace('\t',' ').replace('\n','').replace('\r','')
        line = line.split(separator)
        values = []
        for k in range(len(line)):
            if len(line[k]) > 0: #and k == item_nr:
                values.append(line[k])
                # break

        return values

    def load_pc(self):
        """
        Load the Profile coeficients file (pc)
        ======================================

        DEPRICATED, use Simulations.ModelData instead!!

        Members
        -------
        pc_dict : dict
            pc_dict[pc_set] = [label,tc_ratio,data]
            data = array[AoA [deg], C_L, C_D, C_M]
        """

        self._data_checks()

        FILE = open(self.data_path + self.pc_file)
        lines = FILE.readlines()
        FILE.close()

        self.pc_dict = dict()

        # dummy values for
        nr_points, start, point = -10, -10, -10

        # go through all the lines
        n = 0
        for line in lines:
            # create a list with all words on the line
            line_list = self.fromline(line)

            if self.debug:
#                print n, ' -- ', line
                print n, start, start+nr_points, \
                        line_list[0:4],

            # ignore empty lines
            if len(line_list) < 2:
                if self.debug:
                    print 'empty line ignored'

            # first two lines are not relevant in this context
            elif n == 0 or n == 1:
                # number of data sets
                # nr_sets = int(line_list[0])
                pass


            # the first set, label line
            elif n == 2:

                set_nr = int(line_list[0])
                # nr of points in set
                nr_points = int(line_list[1])
                tc_ratio = line_list[2]
                start = n
                # back to a string and skip the first 2 elements
                label = ''
                for m in range(len(line_list)-2):
                    label += line_list[m+2]
                    label += ' '
                label = label.rstrip()
                data = scipy.zeros((nr_points,4),dtype=PRECISION)

                if self.debug:
                    print ' -> first set'

            # the other sets, label line
            elif (n - nr_points-1) == start:
                # save the previous data set
                self.pc_dict[set_nr] = [label,tc_ratio,data]

                set_nr = int(line_list[0])
                # number of data points in this set
                nr_points = int(line_list[1])
                tc_ratio = line_list[2]
                # mark start line of data set
                start = n
                # new data array for the dataset
                data = scipy.zeros((nr_points,4),dtype=PRECISION)
                # back to a string and skip the first element ()
                label = ''
                for m in range(len(line_list)-2):
                    label += line_list[m+2]
                    label += ' '
                label = label.rstrip()

                if self.debug:
                    print ' -> other sets, label line'

            # if in between, save the data
            elif n > start and n <= (start + nr_points):
                # only take the first 4 elements, rest of line can be comments
                data[n-start-1,:] = line_list[0:4]
                if self.debug:
                    print ' -> data line',
                    print n-start-1, '/', nr_points

            else:
                msg = 'ERROR in ae file, the contents is not correctly defined'
                raise UserWarning, msg

            n += 1

        # save the last label and data array to the dictionary
        self.pc_dict[set_nr] = [label,tc_ratio,data]

    # TODO: implement aspect ratio determination
    def load_ae(self):
        """
        Load the aerodynamic layout file (.ae)
        ======================================

        DEPRICATED, use Simulations.ModelData instead!!

        Members
        -------
        ae_dict : dict
            ae_dict[ae_set] = [label, data_array]
            data_array = [Radius [m], Chord[m], T/C[%], Set no. of pc file]
        """

        self._data_checks()

        FILE = open(self.data_path + self.ae_file)
        lines = FILE.readlines()
        FILE.close()

        self.ae_dict = dict()

        # dummy values for
        nr_points, start, point = -10, -10, -10

        # go through all the lines
        n = 0
        for line in lines:
            # create a list with all words on the line
            line_list = self.fromline(line)

            if self.debug:
#                print n, ' -- ', line
                print n, start, start+nr_points, \
                        line_list[0:4],

            # ignore empty lines
            if len(line_list) < 2:
                if self.debug:
                    print 'empty line ignored'

            # first line is the header stating how many sets
            elif n == 0:
                # number of data sets
                # nr_sets = int(line_list[0])
                pass


            # the first set, label line
            elif n == 1:
                # nr of points in set
                set_nr = int(line_list[0])
                nr_points = int(line_list[1])
                start = n
                # back to a string and skip the first 2 elements
                label = ''
                for m in range(len(line_list)-2):
                    label += line_list[m+2]
                    label += ' '
                label = label.rstrip()
                data = scipy.zeros((nr_points,4),dtype=PRECISION)

                if self.debug:
                    print ' -> first set'

            # the other sets, label line
            elif (n - nr_points-1) == start:
                # save the previous data set
                self.ae_dict[set_nr] = [label,data]

                set_nr = int(line_list[0])
                # number of data points in this set
                nr_points = int(line_list[1])
                # mark start line of data set
                start = n
                # new data array for the dataset
                data = scipy.zeros((nr_points,4),dtype=PRECISION)
                # back to a string and skip the first element ()
                label = ''
                for m in range(len(line_list)-2):
                    label += line_list[m+2]
                    label += ' '
                label = label.rstrip()

                if self.debug:
                    print ' -> other sets, label line'

            # if in between, save the data
            elif n > start and n <= (start + nr_points):
                # only take the first 4 elements, rest of line can be comments
                data[n-start-1,:] = line_list[0:4]
                if self.debug:
                    print ' -> data line',
                    print n-start-1, '/', nr_points

            else:
                msg = 'ERROR in ae file, the contents is not correctly defined'
                raise UserWarning, msg

            n += 1

        # save the last label and data array to the dictionary
        self.ae_dict[set_nr] = [label,data]


    def load_st(self):
        """
        Load the structural data file (.st)
        ===================================

        DEPRICATED, use Simulations.ModelData instead!!

        Load a given st file into st_dict
        This could actually be taken care of by pyparsing....

        Members
        -------
        st_dict : dict
            st_dict[tag] = [data]
            possible tag/data combinations:
            st_dict[tag + set_nr + '-' + sub_set_nr + '-data'] = sub_set_arr
            sub_set_arr has following columns:
                0  1   2     3     4     5     6     7    8  9   10    11  12
                r  m  x_cg  y_cg  ri_x  ri_y  x_sh  y_sh  E  G  I_x   I_y   J

                 13   14  15   16    17   18
                k_x  k_y  A  pitch  x_e  y_e

            each row is a new data point

            The sorted st_dict.keys() key/value pairs looks as follows
                00-00-header_comments : list
                00-00-nset            : list
                01-00-setcomments     : list
                01-01-comments        : list
                01-01-data            : ndarray
                02-00-setcomments     : list
                02-01-comments        : list
                02-01-data            : ndarray
            A new set is created if a setcomments tags is present

            Comments are placed in an iterable (list, tuple,...). Each element
            gets a space placed in between.

        """


        self._data_checks()

        # TODO: store this in an HDF5 format! This is perfect for that.
        # the structural file saved in a structured way
        # keep track of how many subsets there are in each set
        self.subsets_per_set = []

        # also remember how many sets and how many subsets in each set.
        # this to faciliate to add more sets later

        if self.debug:
            print 'loading st file from:'
            print self.data_path + self.st_file
        # read all the lines of the file into memory
        FILE = open(self.data_path + self.st_file)
        lines = FILE.readlines()
        FILE.close()

        # all flags to false
        start_sets, set_comments, sub_set = False, False, False

        self.st_dict = dict()
        for nr in range(len(lines)):
            # first, read all the items in the line, loose all spaces, which
            # can vary over the document
            items = self.fromline(lines[nr])
#            nr_items = len(items)

            # if we have an empty line, items will be an empty list.
            # Because we investigate the first item of items, append an empty
            # string, otherwise we get an index error
            if len(items) == 0:
                items.append('')

            # for sorting the keys back in the right order, based on line number
            tag = format(nr+1, '04.0f') + '-'
            # or just emtpy
            tag = ''

            if self.debug == True:
                if nr == 0:
                    print 'start_sets,  startwith #, set_comments, ',
                    print 'startswith $, sub_set, len(items)'
                print nr+1, start_sets, items[0].startswith('#'),
                print set_comments, items[0].startswith('$'), len(items)
                print items

            # first line: number of sets is the first item
            if nr == 0:
#                nset = int(items[0])
                # and save to dict
                self.st_dict[tag + '00-00-nset'] = items
                # prepare the header comments
                header_comments = []

            # after the first line, you can have all kind of crap lines, holding
            # any comments you like
            # the "and not..." will trigger to go the next type of line
            elif not start_sets and not items[0].startswith('#'):
                header_comments.append(items)

            # each set starts with #1, followed by comments/name/whatever
            elif items[0].startswith('#'):

                # when we just leave the header comments, save it first
                if not start_sets:
                    self.st_dict[tag +'00-00-header_comments'] = header_comments

                start_sets = True
                # read the set number, attached to # withouth space
                set_nr = int(items[0][1:len(items[0])])
                # make a big comments list, where each line is a new list
                comments = []
                comments.append(items)
                set_comments = True

                self.subsets_per_set.append([])

            # more comments are allowed to follow until the start of a
            # subset, marked with $
            elif set_comments and not items[0].startswith('$'):
                comments.append(items)

            # if we have a subset (starting with $)
            elif items[0].startswith('$') and not sub_set:
                # next time we can start with the subset
                sub_set = True
                # sub set number
                sub_set_nr = int(items[0][1:len(items[0])])

                # attach the sub_set_nr to the current set
                self.subsets_per_set[-1].append(sub_set_nr)

                # also store the set comments
                if set_comments:
                    tmp = format(set_nr, '02.0f')
                    self.st_dict[tag + str(tmp) + '-00-setcomments']=comments
                # stop the set comments
                set_comments = False

                # read the number of data points
                nr_points = int(items[1])
                point = 0

                # store the sub set comments included on this line
                tmp1 = format(set_nr, '02.0f')
                tmp2 = format(sub_set_nr, '02.0f')
                self.st_dict[tag+str(tmp1)+'-'+str(tmp2)+'-comments'] \
                    = items

                # create array to store all the data points:
                sub_set_arr = scipy.zeros((nr_points,19), dtype = np.float128)

            # if the we have the data points
            elif len(items) == 19 and sub_set:
                if point < nr_points-1:
                    # we can store it in the array
                    sub_set_arr[point,:] = items
                    point += 1
                # on the last entry:
                elif point == nr_points-1:
                    sub_set_arr[point,:] = items
                    tmp1 = format(set_nr, '02.0f')
                    tmp2 = format(sub_set_nr, '02.0f')
                    # save to the dict:
                    self.st_dict[tag + str(tmp1)+'-'+str(tmp2)+'-data']\
                        = sub_set_arr
                    # and prepare for the next loop
                    sub_set = False

        # save the last data points to st_dict
        tmp1 = format(set_nr, '02.0f')
        tmp2 = format(sub_set_nr, '02.0f')
        self.st_dict[tag + str(tmp1)+'-'+str(tmp2)+'-data'] = sub_set_arr

    def plot_pc(self, ylim=None, xlim=None):
        """
        Plot given pc set
        =================

        DEPRICATED, use Simulations.ModelData instead!!

        pc_dict[pc_set] = [label,tc_ratio,data]
        data = array[AoA [deg], C_L, C_D, C_M]

        WARNING: only one aero set is supported for the moment!
        """
        # load the pc file
        self.load_pc()

        if self.debug:
            print 'saving pc plots in:',self.data_path


        # plot all aero data
#        pc_sets = self.pc_dict.keys()
#        nr_sets = len(pc_sets)

        for k in self.pc_dict.iteritems():

            # k is now a key, value pair of the dictionary

            pc_set = k[0]
            label = k[1][0]
            tc_ratio = k[1][1]
            data = k[1][2]

            x = data[:,0]
            # add_subplot(nr rows nr cols plot_number)

            # initialize the plot object
            fig = Figure(figsize=(16, 9), dpi=200)
            canvas = FigureCanvas(fig)
            fig.set_canvas(canvas)

            ax = fig.add_subplot(1,2,1)
            ax.plot(x,data[:,1], 'bo-', label=r'$C_L$')
            ax.plot(x,data[:,2], 'rx-', label=r'$C_D$')
            ax.plot(x,data[:,3], 'g*-', label=r'$C_M$')
            ax.set_xlim([-180,180])

            if type(ylim).__name__ == 'list':
                ax.set_ylim(ylim)

            if type(xlim).__name__ == 'list':
                ax.set_xlim(xlim)

            ax.set_xlabel('Angle of Attack [deg]')
            ax.legend()
            ax.grid(True)

            ax = fig.add_subplot(1,2,2)
            ax.plot(data[:,2],data[:,1], 'bo-')
            ax.set_xlabel(r'$C_D$')
            ax.set_ylabel(r'$C_L$')
            ax.grid(True)

            fig.suptitle(label)
            figpath = self.save_path + self.pc_file + '_set' + str(pc_set) + \
                        '_tc_' + str(tc_ratio)

            fig.savefig(figpath +  '.eps')
            print 'saved:', figpath + '.eps'
            fig.savefig(figpath +  '.png')
            print 'saved:', figpath + '.png'
            canvas.close()
            fig.clear()


    def plot_ae(self, ae_set):
        """
        Plot given ae set
        =================

        """
        # load the ae file
        self.load_ae()

        if self.debug:
            print 'saving ae plots in:',self.data_path
#            print 'ae_dict.keys', self.ae_dict.keys()

        label = self.ae_dict[ae_set][0]
        data  = self.ae_dict[ae_set][1]
        x = data[:,0]
        x_label = 'blade radius [m]'

#        self.numStepsY = 10
#        self.numStepsX = 10

        # add_subplot(nr rows nr cols plot_number)
        fig = Figure(figsize=(8, 4), dpi=200)
        canvas = FigureCanvas(fig)
        fig.set_canvas(canvas)

        ax = fig.add_subplot(1,1,1)
        ax.plot(x,data[:,1], 'bo-', label='chord')
        ax.set_xlabel(x_label)
#        ax.set_yticks(plt.linspace(plt.ylim()[0],plt.ylim()[1],self.numStepsY))
#        ax.set_xticks(plt.linspace(plt.xlim()[0],plt.xlim()[1],self.numStepsX))

        ax.plot(x,data[:,2]/100.0, 'rx-', label='T/C')
        # set the ticks the same way as on the left axes
#        ax2.set_yticks(plt.linspace(plt.ylim()[0],plt.ylim()[1],self.numStepsY))
#        ax2.set_xticks(plt.linspace(plt.xlim()[0],plt.xlim()[1],self.numStepsX))
        ax.legend(loc='upper right')
        ax.grid(True)

        fig.suptitle(label)

        figpath = self.save_path+self.ae_file +'_set' + str(ae_set)
        fig.savefig(figpath +  '.png')
        canvas.close()
        fig.clear()

        if self.debug:
            print 'saved;',figpath+'.png'


    def plot_st(self, sets):
        """
        Plot a list of set-subset pairs
        ===============================

        Parameters
        ----------

        sets : list
            [ [set, subset], [set, subset], ... ]
        """

        # load the structure file
        self.load_st()

        # number of sets in the file
#        nr_sets = int(st_dict['0-0-nset'][0])

        if self.debug:
            print 'saving st plots in:',self.save_path
            print self.st_dict.keys()

#        # plot all subsets
#        for k in range(nr_sets):

        # define the label precision
        majorFormatter = FormatStrFormatter('%1.1e')
        set_labels = self.st_column_header_list_latex
        # plot given set-subsets
        for x in sets:
            # define the set-subset numbers
            k = x[0] -1
            i = x[1] -1
            # get the set name, the set_comments object looks like
            # [['#1', 'Blade', 'data'], ['r', 'm', 'x_cg', 'y_cg', 'ri_x',
            # 'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'I_p/K', 'k_x',
            # 'k_y', 'A', 'pitch', 'x_e', 'y_e']]
            set_comment=self.st_dict[format(k+1,'02.0f')+'-00-setcomments'][0]
#            set_labels =self.st_dict[format(k+1,'02.0f')+'-00-setcomments'][1]
            set_comment_str = ''
            # back to a string and skip the first element ()
            for m in range(len(set_comment)-1):
                set_comment_str += set_comment[m+1]
                set_comment_str += ' '

            set_comment_str = set_comment_str.rstrip()

            # color-label for left and right axis plots
            left = 'bo-'
            right = 'rx-'
            ax2_left = 'ko-'
            ax2_right = 'gx-'
            label_size = 'large'
            legend_ax1 = 'upper right'
            legend_ax2 = ''

#            # cycle through all subsets
#            # assume there are maximum 20 subsets
#            for i in range(20):

            # see if the subset exists
            try:
                data = self.st_dict[format(k+1,'02.0f') \
                                + '-' + format(i+1,'02.0f') + '-data']
            # if the subset does not exist, go to the next set
            except KeyError:
                print 'Can\'t find set in st_dict:'
                print format(k+1,'02.0f')+'-'+format(i+1,'02.0f')+'-data'
                continue

            if self.debug:
                print 'set ' + format(k+1, '02.0f'),
                print 'subset ' + format(i+1, '02.0f')

            # and plot some items of the structural data
            fig = Figure(figsize=(16, 9), dpi=200)
            canvas = FigureCanvas(fig)
            fig.set_canvas(canvas)

            ax = fig.add_subplot(2, 3, 1)
            fig.subplots_adjust(left= 0.1, bottom=0.1, right=0.9,
                            top=0.95, wspace=0.35, hspace=0.2)

            # x-axis is always the radius
            x = data[:,0]
            # mass

            ax.plot(x,data[:,1], left, label=r'$'+set_labels[1]+'$')
            ax.legend()
            ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)

            # x_cg and y_cg and pitch
            ax = fig.add_subplot(2, 3, 2)
            ax.plot(x,data[:,2], left, label=r'$'+set_labels[2]+'$')
            ax.plot(x,data[:,3], right, label=r'$'+set_labels[3]+'$')
            ax.plot(x,data[:,16], ax2_left, label=r'$'+set_labels[16]+'$')
            ax.legend(loc=legend_ax1)
#                plt.grid(True)
#                ax2 = ax.twinx()
#                ax2.legend(loc='upper right')
            ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)

            # x_sh and y_sh, x_e and y_e
            ax = fig.add_subplot(2, 3, 3)
            ax.plot(x,data[:,6], left,label=r'$'+set_labels[6]+'$')
            ax.plot(x,data[:,7], right, label=r'$'+set_labels[7]+'$')
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)
            ax2 = ax.twinx()
            ax2.plot(x,data[:,17], ax2_left, label=r'$'+set_labels[17]+'$')
            ax2.plot(x,data[:,18], ax2_right, label=r'$'+set_labels[18]+'$')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax2.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)

            # second row of plots
            # EI_x and EI_y
            ax = fig.add_subplot(2, 3, 4)
            label = set_labels[8] + '*' + set_labels[10]
            ax.plot(x,data[:,8]*data[:,10], left, label=r'$'+label+'$')
#                ax2 = ax.twinx()
            label = set_labels[8] + '*' + set_labels[11]
            ax.plot(x,data[:,8]*data[:,11], right, label=r'$'+label+'$')
            ax.legend(loc=legend_ax1)
#                ax2.legend(loc='upper right')
            ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)

            # m*ri_x and m*ri_y
            ax = fig.add_subplot(2, 3, 5)
            label = set_labels[1] + '*' + set_labels[4] + '^2'
            ax.plot(x,data[:,1]*np.power(data[:,4],2),
                    left, label=r'$'+label+'$')
#                ax2 = ax.twinx()
            label = set_labels[1] + '*' + set_labels[5] + '^2'
            ax.plot(x,data[:,1]*np.power(data[:,5],2),
                    right, label=r'$'+label+'$')
            ax.legend(loc=legend_ax1)
#                ax2.legend(loc='upper right')
            ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)

            # GI_p/K and EA
            ax = fig.add_subplot(2, 3, 6)
            label = set_labels[9] + '*' + set_labels[12]
            ax.plot(x,data[:,9]*data[:,12], left, label=r'$'+label+'$')
            ax.legend(loc='upper left')
            ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax.yaxis.set_major_formatter(majorFormatter)
            ax.grid(True)
            ax2 = ax.twinx()
            label = set_labels[8] + '*' + set_labels[15]
            ax2.plot(x,data[:,8]*data[:,15], right, label=r'$'+label+'$')
            ax2.legend(loc='upper right')
            ax2.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
            ax2.yaxis.set_major_formatter(majorFormatter)
            ax2.grid(True)

            fig_title = set_comment_str+' set '+str(k+1)+' subset '+str(i+1)
            fig.suptitle(fig_title)

            # and save
#                title = 'Sandwich Beam Stresses, layout=' + str(self.layout)
#                plt.title(title, size='large')
            tmp = 'set_'+format(k+1,'02.0f')+'_subset_'+format(i+1,'02.0f')
            tmp = tmp + '_' + set_comment_str.replace(' ','_')
            tmp = tmp.replace(',','')

            figpath = self.save_path+self.st_file +'_plot_' + tmp
            fig.savefig(figpath +  '.png', orientation='landscape')
            canvas.close()
            fig.clear()

            if self.debug:
                print 'saved:', figpath + '.png'

    def scale_st(self, sc=1000.):
        """
        Scale the st file, for instance m to mm, with sc
        """

        # get all the keys
        keys = self.st_dict.keys()
        # and scale all data keys
        for k in keys:
            # only select the keys containing the data key
            if k.find('data') > 0:
                # radius: m
                self.st_dict[k][:,0] = self.st_dict[k][:,0]*sc
                # mass: kg/m -> g / mm
                #self.st_dict[k][:,1] = self.st_dict[k][:,1]
                # x_cg: m
                self.st_dict[k][:,2] = self.st_dict[k][:,2]*sc
                # y_cg: m
                self.st_dict[k][:,3] = self.st_dict[k][:,3]*sc
                # ri_x: m
                self.st_dict[k][:,4] = self.st_dict[k][:,4]*sc
                # ri_y: m
                self.st_dict[k][:,5] = self.st_dict[k][:,5]*sc
                # x_sh: m
                self.st_dict[k][:,6] = self.st_dict[k][:,6]*sc
                # y_sh: m
                self.st_dict[k][:,7] = self.st_dict[k][:,7]*sc
                # E: N/m2 : kg / (s2 m) -> g / (mm s2)
                #self.st_dict[k][:,8] = self.st_dict[k][:,8]
                # G: N/m2 : kg / (s2 m) -> g / (mm s2)
                #self.st_dict[k][:,9] = self.st_dict[k][:,9]
                # I_x: m4
                self.st_dict[k][:,10] = self.st_dict[k][:,10]*(sc*sc*sc*sc)
                # I_y: m4
                self.st_dict[k][:,11] = self.st_dict[k][:,11]*(sc*sc*sc*sc)
                # I_p/K: m4/rad
                self.st_dict[k][:,12] = self.st_dict[k][:,12]*(sc*sc*sc*sc)
                # A: m2
                self.st_dict[k][:,15] = self.st_dict[k][:,15]*sc*sc
                # x_e: m
                self.st_dict[k][:,17] = self.st_dict[k][:,16]*sc
                # y_e: m
                self.st_dict[k][:,18] = self.st_dict[k][:,16]*sc




    def write_st2(self, st_file2 = 'somefile2.st'):
        """
        Write the st_dict back to a text file
        """
        self._data_checks()

        # if the column width has been changed, recreate the column line
        self.column_header_line = ''
        for k in self.st_column_header_list:
            self.column_header_line += k.rjust(self.col_width)

        self.st_file2 = st_file2

        file = ''
#        self.load_st()
        # get all the keys
        keys = self.st_dict.keys()
        # sort them
        keys.sort()
        # and recreate file
        for k in keys:

            # the first line
            if k.find('nset') > 0:
                for m in self.st_dict[k]:
                    file += str(m) + ' '
                file += '\n'

            # for the start of a new set (#), multiple lines are allowed:
            # or the header_comments
            elif k.find('setcomments') > 0 or k.find('header_comments') > 0:
                # spaces are replaced by list entries
                header = False
                for line in self.st_dict[k]:
                    # could contain multiple lines
                    for m in line:
                        # is the first line m or M? than we have the column head
                        if (m == 'r' or m == 'R') or header == True:
                            file += str(m).rjust(self.col_width)
                            header = True
                        else:
                            file += str(m) + ' '
                    file += '\n'
                    header = False

            # for the start of new subset ($)
            # currently, additional comment lines are ignored, so by default
            # print the column header again
            elif k.find('comments') > 0:
                # first the column header (this extra line in between the last
                # data point and the new subset start is ignored by load_st())
                if int(k.split('-')[1]) != 1:
                    # do not add the header for the first subset, it will
                    # grow together with the set comments
                    file += 19 * self.col_width * '=' + '\n'
                    file += self.column_header_line + '\n'
                    file += 19 * self.col_width * '-' + '\n'
                for m in self.st_dict[k]:
                    file += str(m) + ' '
                file += '\n'

            # keys containing data, are arrays
            elif k.find('data') > 0:
                # current subset numbers is
                subset = int(k.split('-')[1])
                # print set numbet etc
                # file +=
                # first line is the set number an number of data points
                file += '$%i %i\n' % (subset, self.st_dict[k].shape[0])
                # cycle through data points
                for m in range(self.st_dict[k].shape[0]):
                    for n in range(self.st_dict[k].shape[1]):
                        # TODO: check what do we lose here?
                        # we are coming from a np.float128, as set in the array
                        # but than it will not work with the format()
                        number = float(self.st_dict[k][m,n])
                        if self.debug: print type(number)
                        # the formatting of the number
                        numabs = abs(number)
                        # just a float precision defined in self.prec_float
                        if (numabs < self.float_hi and numabs > self.float_lo):
                            numfor = format(number, self.prec_float)
                        # if it is zero, just simply print as 0.0
                        elif number == 0.0:
                            numfor = format(number, ' 1.1f')
                        # exponentional, precision defined in self.prec_exp
                        else:
                            numfor = format(number, self.prec_exp)
                        file += numfor.rjust(self.col_width)
                    file += '\n'
                # TODO: option to create tab delimited file!!

        # and write file to disk again
        FILE = open(self.data_path + self.st_file2, self.st2_filemode)
        FILE.write(file)
        FILE.close()
        print 'st file written:', self.data_path + self.st_file2

    def check_st(self, sets):
        """
        Check data sanity of a given structure subset
        =============================================

        Verify if the relations between mass, radius of inertia, inertia, etc
        are all within physical limits. This should to assess if the
        structural data can actual relate to a real structure

        Structure of the data array of a subset:

        0  1   2     3     4     5     6     7    8  9   10    11  12
        r  m  x_cg  y_cg  ri_x  ri_y  x_sh  y_sh  E  G  I_x   I_y   J

         13   14  15   16    17   18
        k_x  k_y  A  pitch  x_e  y_e

        Parameters
        ----------

        sets : list
            [ [set, subset], [set, subset], ... ]


        """

        # load the structure file
        self.load_st()
        for x in sets:
            # define the set-subset numbers
            k = x[0] -1
            i = x[1] -1
            # get the set name, the set_comments object looks like
            # [['#1', 'Blade', 'data'], ['r', 'm', 'x_cg', 'y_cg', 'ri_x',
            # 'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'I_p/K', 'k_x',
            # 'k_y', 'A', 'pitch', 'x_e', 'y_e']]
            set_comment=self.st_dict[format(k+1,'02.0f')+'-00-setcomments'][0]
            set_comment_str = ''
            # back to a string and skip the first element ()
            for m in range(len(set_comment)-1):
                set_comment_str += set_comment[m+1]
                set_comment_str += ' '

            set_comment_str = set_comment_str.rstrip()
            datakey = format(k+1,'02.0f') +'-'+ format(i+1,'02.0f') + '-data'
            # see if the subset exists
            try:
                data = self.st_dict[datakey]
            # if the subset does not exist, go to the next set
            except KeyError:
                print 'Can\'t find set in st_dict:', datakey
                continue

            #0 1  2    3    4    5    6    7   8 9 10   11
            #r m x_cg y_cg ri_x ri_y x_sh y_sh E G I_x  I_y
            #12    13  14  15  16  17  18
            #I_p/K k_x k_y A pitch x_e y_e

            # density rho = m/A
            rho = data[:,1]/data[:,15]
            # radius of gyratoin
            r_x = np.sqrt(data[:,10]/data[:,15])
            r_y = np.sqrt(data[:,11]/data[:,15])

            print datakey
            print 'rho', rho
            print 'r_x', r_x
            print 'r_y', r_y
            print ''


# DEPRICATED: use Simulations.py insted!!
class ErrorLogs:
    """
    Analyse all HAWC2 log files in any given directory
    ==================================================

    DEPRICATED: use Simulations.py insted!!

    Usage:
    logs = ErrorLogs()
    logs.MsgList    : list with the to be checked messages. Add more if required
    logs.ResultFile : name of the result file (default is ErrorLog.csv)
    logs.PathToLogs : specify the directory where the logsfile reside,
                        the ResultFile will be saved in the same directory.
                        It is also possible to give the path of a specific
                        file, the logfile will not be saved in this case. Save
                        when all required messages are analysed with save()
    logs.check() to analyse all the logfiles and create the ResultFile
    logs.save() to save after single file analysis

    logs.MsgListLog : [ [case, line nr, error1, line nr, error2, ....], [], ...]
    holding the error messages, empty if no err msg found
    will survive as long as the logs object exists. Keep in
    mind that when processing many messages with many error types (as defined)
    in MsgList might lead to an increase in memory usage.

    logs.MsgListLog2 : dict(key=case, value=[found_error, exit_correct]
        where found_error and exit_correct are booleans. Found error will just
        indicate whether or not any error message has been found

    All files in the speficied folder (PathToLogs) will be evaluated.
    When Any item present in MsgList occurs, the line number of the first
    occurance will be displayed in the ResultFile.
    If more messages are required, add them to the MsgList
    """

    def __init__(self):
        """
        DEPRICATED: use Simulations.py insted!!
        """

        # specify folder which contains the log files
        self.PathToLogs = ''
        self.ResultFile = 'ErrorLog.csv'

        # FIXME: MsgListLog needs to be dict with case as key!
        # the total message list log:
        self.MsgListLog = []
        # a smaller version, just indication if there are errors:
        self.MsgListLog2 = dict()

        # specify which message to look for. The number track's the order.
        # this makes it easier to view afterwards in spreadsheet:
        # every error will have its own column
        self.MsgList = list()
        self.MsgList.append(['*** ERROR ***  in command line',
                             len(self.MsgList)+1])
        self.MsgList.append(['*** WARNING *** A comma', len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** Not correct number of parameters',\
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** Wind speed requested inside', \
                        len(self.MsgList)+1])
        self.MsgList.append(['Maximum iterations exceeded at time step:', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** Out of x bounds:', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** In body actions', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** Command unknown', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** opening', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** No line termination', \
                        len(self.MsgList)+1])
        self.MsgList.append(['Solver seems not to conv', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** MATRIX IS NOT DEFINITE', \
                        len(self.MsgList)+1])
        self.MsgList.append(['*** ERROR *** There are unused relative', \
                        len(self.MsgList)+1])

        # TODO: error message from a non existing channel output/input
        # add more messages if required...

        # to keep this message in the last column of the csv file,
        # this should remain the last entry
        self.MsgList.append(['Elapsed time :', len(self.MsgList)+1,'dummy'])

    def check(self):

        # MsgListLog = []

        # load all the files in the given path
        FileList = []
        for files in os.walk(self.PathToLogs):
            FileList.append(files)

        # if the instead of a directory, a file path is given
        # the generated FileList will be empty!
        try:
            NrFiles = len(FileList[0][2])
        # input was a single file:
        except:
            NrFiles = 1
            # simulate one entry on FileList[0][2], give it the file name
            # and save the directory on in self.PathToLogs
            tmp = self.PathToLogs.split('/')[-1]
            # cut out the file name from the directory
            self.PathToLogs = self.PathToLogs.replace(tmp, '')
            FileList.append([ [],[],[tmp] ])
            single_file = True
        i=1

        # walk trough the files present in the folder path
        for file in FileList[0][2]:
            # progress indicator
            if NrFiles > 1:
                print 'progress: ' + str(i) + '/' + str(NrFiles)

            # open the current log file
            FILE = open(self.PathToLogs+str(file), 'r')
            lines = FILE.readlines()
            FILE.close()

            # create a copy of the Messagelist, have to do it this way,
            # otherwise you just create a shortcut to the original one.
            # In doing so, changing the copy would change the original as well
            MsgList2 = copy.copy(self.MsgList)
#            for l in range(len(self.MsgList)):
#                MsgList2.append(self.MsgList[l])

            # keep track of the messages allready found in this file
            tempLog = []
            tempLog.append(file)
            exit_correct, found_error = False, False
            # create empty list item for the different messages and line number:
            for j in range(len(MsgList2)):
                tempLog.append('')
                tempLog.append('')

            # check for messages in the current line
            # for speed: delete from message watch list if message is found
            j=0
            for line in lines:
                j += 1
                for k in range(len(MsgList2)):
                    if line.find(MsgList2[k][0]) >= 0:
                        # 2nd item is the column position of the message
                        tempLog[2*MsgList2[k][1]] = MsgList2[k][0]
                        # line number of the message
                        tempLog[2*MsgList2[k][1]-1] = j

                        # if the entry contains 3 elements, read value (time)
                        if len(MsgList2[k]) == 3:
                            elapsed_time = line[18:30]
                            tempLog.append('')
                            elapsed_time = elapsed_time.replace('\n','')
                            tempLog[2*MsgList2[k][1]+1] = elapsed_time
                            exit_correct = True
                        else:
                            # flag we have an error only when the found message
                            # was not the exit_correct one.
                            found_error = True

                        del MsgList2[k]
                        break

            # append the messages found in the current file to the overview log
            self.MsgListLog.append(tempLog)
            self.MsgListLog2[file] = [found_error, exit_correct]
            i += 1

            # if no messages are found for the current file, than say so:
            if len(MsgList2) == len(self.MsgList):
                tempLog[-1] = 'NO MESSAGES FOUND'

        # if we have only one file, don't save the log file to disk. It is
        # expected that if we analyse many different single files, this will
        # cause a slower script
        if single_file:
            # now we make it available over the object to save and let it grow
            # over many analysis
            # self.MsgListLog = copy.copy(MsgListLog)
            pass
        else:
            self.save()

    def save(self):

        # write the results in a file, start with a header
        contents = 'file name;' + 'lnr;msg;'*(len(self.MsgList)) + '\n'
        for k in self.MsgListLog:
            for n in k:
                contents = contents + str(n) + ';'
            # at the end of each line, new line symbol
            contents = contents + '\n'

        # write csv file to disk, append to facilitate more logfile analysis
        print 'Error log analysis saved at:'
        print self.PathToLogs+str(self.ResultFile)
        FILE = open(self.PathToLogs+str(self.ResultFile), 'a')
        FILE.write(contents)
        FILE.close()

def scale_structure():
    """
    Scale the structure blade
    """
    # load the structural file
    m = ModelData()
    m.data_path = '/home/dave/PhD/Projects/Hawc2Models/'+MODEL+'/data/'
    m.st_file = '3bdown.st'
    m.float_lo = 0.001
    m.prec_float = ' 8.06f'
    m.prec_exp = ' 8.4e'
    m.load_st()

    # first, determine how many blade sets there are
    # set 1 has index 0 etc
    nr_blade_sets = len(m.subsets_per_set[0])

    # scale set 16 and 17 (=16 with smaller A)
    base_set = m.st_dict['01-17-data'].copy()
    # newset = scipy.zeros(base_set.shape, dtype=base_set.dtype)

    # nr of data points will remain the same as the base set
    num_points = str(base_set.shape[0])

    # write down new version and compare with original
    m.write_st2(st_file2 = '3bdown_scaled.st')
    # loading
    ms = HawcPy.ModelData()
    ms.data_path = '/home/dave/PhD/Projects/Hawc2Models/'+MODEL+'/data/'
    ms.st_file = '3bdown_scaled.st'
    ms.float_lo = 0.001
    ms.prec_float = ' 8.06f'
    ms.prec_exp = ' 8.4e'
    ms.load_st()

    # cycle trough all data arrays and asses if any precision has been lost
    lost_precision = False
    for k in m.st_dict:
        if k.find('data') > 0:
            test = False
            test = np.allclose(m.st_dict[k],ms.st_dict[k],
                               rtol=1.0e-05, atol=1e-06)
            if not test:
                print 'precision lost in:', k
                lost_precision = True

    if not lost_precision:
        print 'precision is preserved'

    scaling_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    k = 1
    for scaling in scaling_list:
        # mass is index 1, E=8, G=9
        newset = base_set.copy()
        newset[:,1] = base_set[:,1]*scaling
        newset[:,8] = base_set[:,8]*scaling
        newset[:,9] = base_set[:,9]*scaling
        # save the new scaled set
        newset_nr = nr_blade_sets + k
        newlabel = '01-'+ format(newset_nr, '02.0f') + '-comments'

        # also add a comment line, indicate subset number and number of points
        m.st_dict[newlabel] = ['$' + str(newset_nr) + ' ' + num_points \
            +' ----> set17, m,E,G scaled with: '+str(scaling) ]
        newset_nr = format(newset_nr, '02.0f')
        # the data with correct formatted label for sorting
        m.st_dict['01-'+str(newset_nr)+'-data'] = newset.copy()

        k += 1

    m.write_st2(st_file2 = '3bdown_scaled.st')

    return m

class TowerClearance:
    """
    INPUTs:
        sig          : HawcPy.sig.sig object
                       (signal object as returned by HawcPy.LoadResults.sig)
        azi_ind :      int, index of rotor azimuth angle channel
                       (allowed range: 0-359 deg)
        azi_ang      : tuple of 1 to 3 doubles, azimuth angles for which a blade
                       passes the tower
        bladetip_ind : tuple with same length as azi_ang, indices of blade tip
                       coordinate in the relevant direction and coordinate syst.
        azi_sector_size: double, half of the azimuth sector range (to both
                         sides of the tower)
    OUTPUT:
        tower_clear    : list [ np.array[tower clearance], blade2, blade3 ]
        tower_clear_min: max of tower_clear, list[blade1 max, blade2, blade3]
        tower_cross    : list [ np.array[crossing indices], blade2, blade3 ]
        tow_clr_matrix : [blade, (0=crossing indices, 1=clearance coord),values]
            np.array

    Define azimuthal sector for the tower clearence area.
    Select azimuthal and blade tip deflection (in tower body coordinate)
    channels.

    Shaft rotor angle: channel 2 (deg)
    blade tip deflection channels: 123, 126, 129 (coordinate: Tower_y)

    For the upwind_LoadExt:
    blade1 down, azimuth=0
    blade2 down, azimuth=120
    blade3 down, azimuth=240
    """

    def __init__(self, sig, azi_ind=1, azi_ang=(0.0,120.0,240.0), \
                    bladetip_ind=(122,125,128),azi_sector_size=15.0):
        """

        """

        # channel settings and azimuth offsets
        self.azi_ind = azi_ind
        self.bladetip_ind = bladetip_ind
        self.azi_ang = azi_ang
        self.sig = sig
        self.azi_sector_size = azi_sector_size
        # estimate the number of tower passages for matrix size:
        # 10 RPM / 10 min simulation / 3 blades: 10*10*3 = 300
        # azimuthal range: 60 deg, considere time steps, speed and you can
        # determine on how many time steps you got a tower passage coverage
        self.nr_cros = 6000

    def calc(self):
        """

        """
        # get the azimuthal and blade tip data
        azi_sig = self.sig[:,self.azi_ind]

        # has to be valid for 1, 2 or 3 given blade series
        nr_blades = len(self.bladetip_ind)
        tower_cross, tower_clear = [[],[],[]], [[],[],[]]
        tower_clear_min = scipy.zeros(3)

        # in compact matrix form:
        # columns: crossing indices, corresponding clearance
        # tow_clr_matrix = [blades, (0=crossing indices, 1=clearance), values ]
        tow_clr_matrix = scipy.zeros((nr_blades, 2, self.nr_cros))

        for k in range(nr_blades):
            # if we have 0 as azimuthal tower passing angle,
            # set the tower passing angle to 0 in two steps as follows:
            if self.azi_ang[k] == 0.0:
                azi_tower_cross = abs(azi_sig - 180.0)
                azi_tower_cross = abs(azi_tower_cross - 180.0)
            else:
                # the azimuth angles of a blade crossing the tower will be zero
                # when the tower passing value is deducted, absolute value for
                # easy location of tower crossing sector range
                azi_tower_cross = abs(azi_sig - self.azi_ang[k])
            # indices of the tower crossing range, returns a 1 column array!
            tower_cross[k] = np.argwhere(azi_tower_cross < self.azi_sector_size)
            items = len(tower_cross[k])
            tow_clr_matrix[k,0,0:items] = tower_cross[k][:,0]
            # create an array for each blade in which the blade tip, expressed
            # in tower coordinates are saved
            tower_clear[k] = scipy.zeros(len(tower_cross[k]))
            # we might have a different number of crossing indices
            # the corresponding deflections in that azimutal sector:
            for i in range(len(tower_cross[k])):
                # tower crossing index, answer is array, so get the index value!
                index = tower_cross[k][i,0]
                # get the corresponding tower clearance
                tower_clear[k][i] = self.sig[index,self.bladetip_ind[k]]
                tow_clr_matrix[k,1,i] = tower_clear[k][i]

            # the minimum tower clearance distance:
            tower_clear_min[k] = tower_clear[k].max()

        return tower_clear, tower_clear_min, tower_cross, tow_clr_matrix


class Fatigue_RFC_EXE:
    """Fatigue calculations for a set of channels and result files

    Usage:
    f = Fatigue(channels, cases, hours, windchan)
    f.bin_path = 'c:/HAWC2/development/fatigue/'
    f.results_path = 'c:/HAWC2/development/fatigue/results/'
    f.execute()

    Fatigue calculations for a series of channels considereing
    different signals, by using the external windows executable
    rfc_j.exe.  Input file is rfc_j.inp

    Inputs:
    channels  -> list with channel numbers corresponding to HAWC2 result file
    cases     -> list with paths of cases contributing to the fatigue loading
    hours     -> list with number of hours for each corresponding case
    windchan  -> indicate wind channel (should be the same for all cases!)

    Optional inputs:
    starttime -> if different than zero, specify how many initial sec. to skip

    The results are stored in:
    .chan_eqloads = []          -> list of equivalent load arrays
    .chan_eqloads_header = []   -> list of headers for eqload arrays
    .error_msg                  -> list containing the error msg's
    """

    def __init__(self, channels, cases, hours, windchan):

        self.debug = False
        # constant: input template file
        self.template_path = 'c:/HAWC2/development/fatigue/TEMPLATE_rfc_j.txt'

        # the bin path should be one level up with respect to results_path
        # in this setup the results are not saved in the bin folder!
#        self.bin_path = 'c:/HAWC2/development/fatigue/'
#        self.results_path = 'c:/HAWC2/development/fatigue/results/'

        # path's of exe, input and define output dir
        self.input_path = 'rfc_j.inp' # parsed to cmd
        self.rfc_j_path = 'rfc_j.exe' # parsed to cmd

        # create inheritance of the input variables
        self.channels = channels
        self.cases = cases
        self.hours = hours
        self.windchan = windchan
        self.starttime = 0

        # Output: the results
        self.chan_eqloads = []
        self.chan_eqloads_header = []

        # create a list to store the possible error messages
        self.error_msg = []

    def execute(self):
        # data check
        if len(self.cases) != len(self.hours):
            print 50*'='
            print 'number of cases and hours should be the same!'
            print 50*'='
        else:
            # set the correct working directory
            os.chdir(self.bin_path)
            template = self.read_template()
            if template == 'template error':
                print 50*'='
                print 'template error!'
                print 50*'='
            else:
                self.create_input(template)
                # now switch to results path
                os.chdir(self.results_path)
                self.cmd_out = self.calc_fatigue_RFC_j()
#                # if there was an error during running RFC, don't load results
#                if self.error_msg[-1].startswith('RFC'):
#                    pass
#                else:
                self.load_fatigue_results()

    def read_template(self):
        try:
            # load the template
            FILE = open(self.template_path, "r")
            template = FILE.readlines()
            FILE.close()
            return template
        except:
            print 50*'='
            print 'input TEMPLATE could not be found'
            print 50*'='
            self.error_msg.append('rfc_j input template error: read_template()')
            return 'template error'

    def create_input(self, template):
        # create a valid input file
        input_file = []
        # line_nr_t = 1
        # line_nr_input = 1

        # copy the first 2 lines
        for line in range(0,2):
            input_file.append(template[line])
        # set start time and wind channel
        input_file.append(str(self.starttime) + template[2])
        input_file.append(str(self.windchan) + template[3])
        # header for channels
        input_file.append(template[4])
        # fill in all channels
        for chan in self.channels:
            input_file.append(str(chan) + '    1.0   1.0 \n')
        # copy lines in between
        for line in range(6,11):
            input_file.append(template[line])
        # fill in all cases and corresponding hours
        for k in range(len(self.cases)):
            input_file.append(self.cases[k]+'     '+str(self.hours[k]) +' \n')

        # convert list to one string for writing the file
        file_contents = ''
        for k in input_file:
            file_contents = file_contents + k

        # save the file:
        FILE = open(self.input_path, "w")
        FILE.write(file_contents)
        FILE.close()

    def calc_fatigue_RFC_j(self):

        # specifics for rfc_j.exe
        error_list = []
        error_list.append('rfc_jd.inp mangler')

        # navigate to correct directory, so output will be stored properly
        os.chdir(self.results_path)

        # execution command
        command = '..\\' + self.rfc_j_path + ' ..\\' + self.input_path
        output = os.popen(command)
        # read cmd output
        cmd_out = ''
        for line in output.readlines():
            if len(line) > 1:
                for err in error_list:
                    if line.find(err) > 0:
                        self.error_msg.append\
                            ('RFC_j.exe error: input file not found')
                        print 50*'='
                        print 'RFC_j.exe could not find the input file: ' \
                            + self.input_path
                        print 50*'='
                    cmd_out = cmd_out + line
        return cmd_out

    def load_fatigue_results(self):
        # for each channel there will be a results file:
        result_files = []

        # markers in file
        start = 'Combined fatique, all time-series, sensor'

        ## read for each channel the fatigue results
        for chan in self.channels:
            # results file name
            temp = self.results_path + 'RFCs.%03i' % chan
            result_files.append(temp)
            # read the file
            FILE = open(temp, "r")
            lines = FILE.readlines()
            FILE.close()

            ## look for the starting point of the equivalent loads results part
            line_nr = 0
            for line in lines:
                # determine line nr of start of results:
                if line.startswith(start):
                    line_start = line_nr
                    break
                line_nr += 1

            ## value of N-ref
            N_ref = lines[line_start + 2].replace('\n','')
            # N_ref_i = N_ref.find('N-ref = ')
            N_ref = N_ref[57:len(N_ref)]

            ## header:"     m    N-ref   N=10E7   N=10E6"
            N_header = lines[line_start + 4].replace('\n','')
            N_header = N_header.split('   ')
            # remove the empty list entries
            N_header = remove_items(N_header, '')
            # put value for N-ref in place
            N_header[1] = 'N=' + N_ref

            ## read the results table:
            eqloads = scipy.zeros([6,4], dtype=float)
            for k in range(line_start + 6, line_start + 12):
                # list containing the row's columns
                line = lines[k].replace('\n','')
                line = line.split(' ')
                line = remove_items(line, '')
                # save the columns in numpy array
                eqloads[k-line_start-6] = line

            self.chan_eqloads.append(eqloads)
            self.chan_eqloads_header.append(N_header)

class LoadResults:
    """Read a HAWC2 result data file

    Usage:
    obj = LoadResults(file_path, file_name)

    This class is called like a function:
    HawcResultData() will read the specified file upon object initialization.

    Available output:
    obj.sig[timeStep,channel]   : complete result file in a numpy array
    obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
    obj.error_msg: is 'none' if everything went OK, otherwise it holds the
    error

    The ch_dict key/values pairs are structured differently for different
        type of channels. Currently supported channels are:

        For forcevec, momentvec, state commands:
            key:
                coord-bodyname-pos-sensortype-component
                global-tower-node-002-forcevec-z
                local-blade1-node-005-momentvec-z
                hub1-blade1-elem-011-zrel-1.00-state pos-z
            value:
                ch_dict[tag]['coord']
                ch_dict[tag]['bodyname']
                ch_dict[tag]['pos'] = pos
                ch_dict[tag]['sensortype']
                ch_dict[tag]['component']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the DLL's this is:
            key:
                DLL-dll_name-io-io_nr
                DLL-yaw_control-outvec-3
                DLL-yaw_control-inpvec-1
            value:
                ch_dict[tag]['dll_name']
                ch_dict[tag]['io']
                ch_dict[tag]['io_nr']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the bearings this is:
            key:
                bearing-bearing_name-output_type-units
                bearing-shaft_nacelle-angle_speed-rpm
            value:
                ch_dict[tag]['bearing_name']
                ch_dict[tag]['output_type']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']

    """

    # start with reading the .sel file, containing the info regarding
    # how to read the binary file and the channel information
    def __init__(self, file_path, file_name, debug=False):

        self.debug = debug

        # timer in debug mode
        if self.debug:
            start = time()

        self.file_path = file_path
        # remove .log, .dat, .sel extensions who might be accedental left
        if file_name[-4:] in ['.htc','.sel','.dat','.log']:
            file_name = file_name[:-4]
        self.file_name = file_name
        self.read_sel()
        # create for any supported channel the
        # continue if the file has been succesfully read
        if self.error_msg == 'none':
            # load the channel id's and scale factors
            scale_factors = self.data_sel()
            # with the sel file loaded, we have all the channel names to
            # squeeze into a more consistant naming scheme
            self._unified_channel_names()
            # if there is sel file but it is empty or whatever else
            # FilType will not exists
            try:
                # read the binary file
                if self.FileType == 'BINARY':
                    self.read_bin(scale_factors)
                # read the ASCII file
                elif self.FileType == 'ASCII':
                    self.read_ascii()
                else:
                    print '='*79
                    print 'unknown file type: ' + self.FileType
                    print '='*79
                    self.error_msg = 'error: unknown file type'
                    self.sig = []
            except:
                print '='*79
                print 'couldn\'t determine FileType'
                print '='*79
                self.error_msg = 'error: no file type'
                self.sig = []

        if self.debug:
            stop = time() - start
            print 'time to load HAWC2 file:', stop, 's'

    def read_sel(self):
        # anticipate error on file reading
        try:
            # open file, read and close
            go_sel = self.file_path + self.file_name + '.sel'
            FILE = open(go_sel, "r")
            self.lines = FILE.readlines()
            FILE.close()
            self.error_msg = 'none'

        # error message if the file does not exists
        except:
            # print 26*' ' + 'ERROR'
            print 50*'='
            print self.file_path
            print self.file_name + '.sel could not be found'
            print 50*'='
            self.error_msg = 'error: file not found'

    def data_sel(self):
        # increase precision
        # D.getcontext().prec = 50

        # scan through all the lines in the file
        line_nr = 1
        # channel counter for ch_details
        ch = 0
        for line in self.lines:
            # on line 9 we can read following paramaters:
            if line_nr == 9:
                # remove the end of line character
                line = line.replace('\n','')

                settings = line.split(' ')
                # delete all empty string values
                for k in range(settings.count('')):
                    settings.remove('')

                # and assign proper values with correct data type
                self.N = int(settings[0])
                self.Nch = int(settings[1])
                self.Time = float(settings[2])
                # there are HEX values at the end of this line...
                # On Linux they will show up in the last variable, so don't inc
                if os.name == 'posix':
                    nrchars = len(settings[3])-1
                elif os.name == 'nt':
                    nrchars = len(settings[3])
                else:
                    raise UserWarning, \
                    'Untested platform:', os.name
                settings[3] = settings[3][0:nrchars]
                self.FileType = settings[3]
                self.Freq = self.N/self.Time

                # prepare list variables
                self.ch_details = np.ndarray(shape=(self.Nch,3),dtype='<U100')
                # it seems that float64 reeds the data correctly from the file
                scale_factors = scipy.zeros(self.Nch, dtype='Float64')
                #self.scale_factors_dec = scipy.zeros(self.Nch, dtype='f8')
                i = 0

            # starting from line 13, we have the channels info
            if line_nr > 12:
                # read the signal details
                if line_nr < 13 + self.Nch:
                    # remove leading and trailing whitespaces from line parts
                    self.ch_details[ch,0] = str(line[12:43]).strip() # chID
                    self.ch_details[ch,1] = str(line[43:54]).strip() # chUnits
                    self.ch_details[ch,2] = str(line[54:-1]).strip() # chDescr
                    ch += 1
                # read the signal scale parameters for binary format
                elif line_nr > 14 + self.Nch:
                    scale_factors[i] = line
                    # print scale_factors[i]
                    #self.scale_factors_dec[i] = D.Decimal(line)
                    i = i + 1
                # stop going through the lines if at the end of the file
                if line_nr == 2*self.Nch + 14:
                    self.scale_factors = scale_factors

                    if self.debug:
                        print 'N       ', self.N
                        print 'Nch     ', self.Nch
                        print 'Time    ', self.Time
                        print 'FileType', self.FileType
                        print 'Freq    ', self.Freq
                        print 'scale_factors', scale_factors.shape

                    return scale_factors
                    break

            # counting the line numbers
            line_nr = line_nr + 1

    def read_bin(self, scale_factors, ChVec=[]):
        if not ChVec:
            ChVec = range(0, self.Nch)
        fid = open(self.file_path + self.file_name + '.dat', 'rb')
        self.sig = np.zeros( (self.N, len(ChVec)) )
        for j, i in enumerate(ChVec):
            fid.seek(i*self.N*2,0)
            self.sig[:,j] = np.fromfile(fid, 'int16', self.N)*scale_factors[i]

    def read_bin_old(self, scale_factors):
        # if there is an error reading the binary file (for instance if empty)
        try:
            # read the binary file
            go_binary = self.file_path + self.file_name + '.dat'
            FILE = open(go_binary, mode='rb')

            # create array, put all the binary elements as one long chain in it
            binvalues = array.array('h')
            binvalues.fromfile(FILE, self.N * self.Nch)
            FILE.close()
            # convert now to a structured numpy array
            # sig = np.array(binvalues, np.float)
#            sig = np.array(binvalues)
            # this is faster! the saved bin values are only of type int16
            sig = np.array(binvalues, dtype='int16')

            if self.debug: print self.N, self.Nch, sig.shape

#            sig = np.reshape(sig, (self.Nch, self.N))
#            # apperently Nch and N had to be reversed to read it correctly
#            # is this because we are reading a Fortran array with Python C
#            # code? so now transpose again so we have sig(time, channel)
#            sig = np.transpose(sig)

            # reshape the array to 2D and transpose (Fortran to C array)
            sig = sig.reshape((self.Nch, self.N)).T

            # create diagonal vector of size (Nch,Nch)
            dig = np.diag(scale_factors)
            # now all rows of column 1 are multiplied with dig(1,1)
            sig = np.dot(sig,dig)
            self.sig = sig
            # 'file name;' + 'lnr;msg;'*(len(MsgList)) + '\n'
        except:
            self.sig = []
            self.error_msg = 'error: reading binary file failed'
            print '========================================================'
            print self.error_msg
            print self.file_path
            print self.file_name
            print '========================================================'

    def read_ascii(self):

        try:
            go_ascii = self.file_path + self.file_name + '.dat'
            self.sig = np.fromfile(go_ascii, dtype=np.float32, sep='  ')
            self.sig = self.sig.reshape((self.N, self.Nch))
        except:
            self.sig = []
            self.error_msg = 'error: reading ascii file failed'
            print '========================================================'
            print self.error_msg
            print self.file_path
            print self.file_name
            print '========================================================'

#        print '========================================================'
#        print 'ASCII reading not implemented yet'
#        print '========================================================'
#        self.sig = []
#        self.error_msg = 'error: ASCII reading not implemented yet'

    def reformat_sig_details(self):
        """Change HAWC2 output description of the channels short descriptive
        strings, usable in plots

        obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
        """

        # CONFIGURATION: mappings between HAWC2 and short good output:
        change_list = []
        change_list.append( ['original','new improved'] )

#        change_list.append( ['Mx coo: hub1','blade1 root bending: flap'] )
#        change_list.append( ['My coo: hub1','blade1 root bending: edge'] )
#        change_list.append( ['Mz coo: hub1','blade1 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub2','blade2 root bending: flap'] )
#        change_list.append( ['My coo: hub2','blade2 root bending: edge'] )
#        change_list.append( ['Mz coo: hub2','blade2 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub3','blade3 root bending: flap'] )
#        change_list.append( ['My coo: hub3','blade3 root bending: edge'] )
#        change_list.append( ['Mz coo: hub3','blade3 root bending: torsion'] )

        change_list.append( ['Mx coo: blade1','blade1 flap'] )
        change_list.append( ['My coo: blade1','blade1 edge'] )
        change_list.append( ['Mz coo: blade1','blade1 torsion'] )

        change_list.append( ['Mx coo: blade2','blade2 flap'] )
        change_list.append( ['My coo: blade2','blade2 edge'] )
        change_list.append( ['Mz coo: blade2','blade2 torsion'] )

        change_list.append( ['Mx coo: blade3','blade3 flap'] )
        change_list.append( ['My coo: blade3','blade3 edeg'] )
        change_list.append( ['Mz coo: blade3','blade3 torsion'] )

        change_list.append( ['Mx coo: hub1','blade1 out-of-plane'] )
        change_list.append( ['My coo: hub1','blade1 in-plane'] )
        change_list.append( ['Mz coo: hub1','blade1 torsion'] )

        change_list.append( ['Mx coo: hub2','blade2 out-of-plane'] )
        change_list.append( ['My coo: hub2','blade2 in-plane'] )
        change_list.append( ['Mz coo: hub2','blade2 torsion'] )

        change_list.append( ['Mx coo: hub3','blade3 out-of-plane'] )
        change_list.append( ['My coo: hub3','blade3 in-plane'] )
        change_list.append( ['Mz coo: hub3','blade3 torsion'] )
        # this one will create a false positive for tower node nr1
        change_list.append( ['Mx coo: tower','tower top momemt FA'] )
        change_list.append( ['My coo: tower','tower top momemt SS'] )
        change_list.append( ['Mz coo: tower','yaw-moment'] )

        change_list.append( ['Mx coo: chasis','chasis momemt FA'] )
        change_list.append( ['My coo: chasis','yaw-moment chasis'] )
        change_list.append( ['Mz coo: chasis','chasis moment SS'] )

        change_list.append( ['DLL inp  2:  2','tower clearance'] )

        self.ch_details_new = np.ndarray(shape=(self.Nch,3),dtype='<U100')

        # approach: look for a specific description and change it.
        # This approach is slow, but will not fail if the channel numbers change
        # over different simulations
        for ch in range(self.Nch):
            # the change_list will always be slower, so this loop will be
            # inside the bigger loop of all channels
            self.ch_details_new[ch,:] = self.ch_details[ch,:]
            for k in range(len(change_list)):
                if change_list[k][0] == self.ch_details[ch,0]:
                    self.ch_details_new[ch,0] =  change_list[k][1]
                    # channel description should be unique, so delete current
                    # entry and stop looking in the change list
                    del change_list[k]
                    break

#        self.ch_details_new = ch_details_new

    def _unified_channel_names(self):
        """
        Make certain channels independent from their index.

        The unified channel dictionary ch_dict holds consequently named
        channels as the key, and the all information is stored in the value
        as another dictionary.

        The ch_dict key/values pairs are structured differently for different
        type of channels. Currently supported channels are:

        For forcevec, momentvec, state commands:
            node numbers start with 0 at the root
            element numbers start with 1 at the root
            key:
                coord-bodyname-pos-sensortype-component
                global-tower-node-002-forcevec-z
                local-blade1-node-005-momentvec-z
                hub1-blade1-elem-011-zrel-1.00-state pos-z
            value:
                ch_dict[tag]['coord']
                ch_dict[tag]['bodyname']
                ch_dict[tag]['pos']
                ch_dict[tag]['sensortype']
                ch_dict[tag]['component']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the DLL's this is:
            key:
                DLL-dll_name-io-io_nr
                DLL-yaw_control-outvec-3
                DLL-yaw_control-inpvec-1
            value:
                ch_dict[tag]['dll_name']
                ch_dict[tag]['io']
                ch_dict[tag]['io_nr']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the bearings this is:
            key:
                bearing-bearing_name-output_type-units
                bearing-shaft_nacelle-angle_speed-rpm
            value:
                ch_dict[tag]['bearing_name']
                ch_dict[tag]['output_type']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']

        For many of the aero sensors:
            'Cl', 'Cd', 'Alfa', 'Vrel'
            key:
                sensortype-blade_nr-pos
                Cl-1-0.01
            value:
                ch_dict[tag]['sensortype']
                ch_dict[tag]['blade_nr']
                ch_dict[tag]['pos']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']


        """
        # save them in a dictionary, use the new coherent naming structure
        # as the key, and as value again a dict that hols all the different
        # classifications: (chi, channel nr), (coord, coord), ...
        self.ch_dict = dict()

        # some channel ID's are unique, use them
        ch_unique = set(['Omega', 'Ae rot. torque', 'Ae rot. power',
                     'Ae rot. thrust', 'Time', 'Azi  1'])
        ch_aero = set(['Cl', 'Cd', 'Alfa', 'Vrel'])

        # scan through all channels and see which can be converted
        # to sensible unified name
        for ch in range(self.Nch):
            items = self.ch_details[ch,2].split(' ')
            # remove empty values in the list
            items = remove_items(items, '')

            dll = False

            # be carefull, identify only on the starting characters, because
            # the signal tag can hold random text that in some cases might
            # trigger a false positive

            # -----------------------------------------------------------------
            # check for all the unique channel descriptions
            if self.ch_details[ch,0].strip() in ch_unique:
                tag = self.ch_details[ch,0].strip()
                channelinfo = {}
                channelinfo['units'] = self.ch_details[ch,1]
                channelinfo['sensortag'] = self.ch_details[ch,2]
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # or in the long description:
            #    0          1        2      3  4    5     6 and up
            # MomentMz Mbdy:blade nodenr:   5 coo: blade  TAG TEXT
            elif self.ch_details[ch,2].startswith('MomentM'):
                coord = items[5]
                bodyname = items[1].replace('Mbdy:', '')
                # set nodenr to sortable way, include leading zeros
                # node numbers start with 0 at the root
                nodenr = '%03i' % int(items[3])
                # skip the attached the component
                #sensortype = items[0][:-2]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'momentvec'
                component = items[0][-1:len(items[0])]
                # the tag only exists if defined
                if len(items) > 6:
                    sensortag = ' '.join(items[6:])

                # and tag it
                pos = 'node-%s' % nodenr
                tagitems = (coord,bodyname,pos,sensortype,component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            #   0    1      2        3       4  5     6     7 and up
            # Force  Fx Mbdy:blade nodenr:   2 coo: blade  TAG TEXT
            elif self.ch_details[ch,2].startswith('Force'):
                coord = items[6]
                bodyname = items[2].replace('Mbdy:', '')
                nodenr = '%03i' % int(items[4])
                # skipe the attached the component
                #sensortype = items[0]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'forcevec'
                component = items[1][1]
                if len(items) > 7:
                    sensortag = ' '.join(items[7:])

                # and tag it
                pos = 'node-%s' % nodenr
                tagitems = (coord,bodyname,pos,sensortype,component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            #   0    1  2      3       4      5   6         7    8
            # State pos x  Mbdy:blade E-nr:   1 Z-rel:0.00 coo: blade
            elif self.ch_details[ch,2].startswith('State pos'):
                coord = items[8]
                bodyname = items[3].replace('Mbdy:', '')
                # element numbers start with 1 at the root
                elementnr = '%03i' % int(items[5])
                zrel = '%04.2f' % float(items[6].replace('Z-rel:', ''))
                # skip the attached the component
                #sensortype = ''.join(items[0:2])
                # or give the sensor type the same name as in HAWC2
                sensortype = 'state pos'
                component = items[2]
                if len(items) > 7:
                    sensortag = ' '.join(items[7:])

                # and tag it
                pos = 'elem-%s-zrel-%s' % (elementnr, zrel)
                tagitems = (coord,bodyname,pos,sensortype,component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            # DLL CONTROL I/O
            # the actual output:
            # DLL out 1: 2   -  ojf_generator outvec  2  rads rotor speed
            #
            #                    0         1    2   3     4+
            # channel desc: yaw_control outvec  3  yaw_c input reference angle
            elif self.ch_details[ch,0].startswith('DLL'):
                dll_name = items[0]
                io = items[1]
                io_nr = items[2]
                sensortag = ' '.join(items[3:])

                # and tag it
                tag = 'DLL-%s-%s-%s' % (dll_name,io,io_nr)
                # save all info in the dict
                channelinfo = {}
                channelinfo['dll'] = dll
                channelinfo['io'] = io
                channelinfo['io_nr'] = io_nr
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            # BEARING OUTPUS
            # bea1 angle_speed       rpm      shaft_nacelle angle speed
            elif self.ch_details[ch,0].startswith('bea'):
                output_type = self.ch_details[ch,0].split(' ')[1]
                bearing_name = items[0]
                units = self.ch_details[ch,1]
                # there is no label option for the bearing output

                # and tag it
                tag = 'bearing-%s-%s-%s' % (bearing_name,output_type,units)
                # save all info in the dict
                channelinfo = {}
                channelinfo['bearing_name'] = bearing_name
                channelinfo['output_type'] = output_type
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # AERO CL, CD, CM, VREL, ALFA, LIFT, DRAG, etc
            # Cl, R=  0.5     deg      Cl of blade  1 at radius   0.49
            # Azi  1          deg      Azimuth of blade  1
            elif self.ch_details[ch,0].split(',')[0] in ch_aero:
                dscr_list = self.ch_details[ch,2].split(' ')
                dscr_list = remove_items(dscr_list, '')

                sensortype = self.ch_details[ch,0].split(',')[0]
                radius = dscr_list[-1]
                # is this always valid?
                blade_nr = self.ch_details[ch,2].split('blade  ')[1][0]
                # sometimes the units for aero sensors are wrong!
                units = self.ch_details[ch,1]
                # there is no label option

                # and tag it
                tag = '%s-%s-%s' % (sensortype,blade_nr,radius)
                # save all info in the dict
                channelinfo = {}
                channelinfo['sensortype'] = sensortype
                channelinfo['radius'] = float(radius)
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # TODO: wind speed
            # some spaces have been trimmed here
            # WSP gl. coo.,Vy          m/s
            # // Free wind speed Vy, gl. coo, of gl. pos   0.00,  0.00,  -2.31
            # WSP gl. coo.,Vdir_hor          deg
            # Free wind speed Vdir_hor, gl. coo, of gl. pos  0.00,  0.00, -2.31

            # -----------------------------------------------------------------
            # ignore all the other cases we don't know how to deal with
            else:
                # if we get here, we don't have support yet for that sensor
                # and hence we can't save it. Continue with next channel
                continue

            # -----------------------------------------------------------------
            # ignore if we have a non unique tag
            if self.ch_dict.has_key(tag):
                msg = 'non unique tag for HAWC2 results, ignoring: %s' % tag
                logging.warn(msg)
            else:
                self.ch_dict[tag] = copy.copy(channelinfo)



    def _data_window(self, nr_rev=None, time=None):
        """
        Based on a time interval, create a proper slice object
        ======================================================

        The window will start at zero and ends with the covered time range
        of the time input.

        Paramters
        ---------

        nr_rev : int, default=None
            NOT IMPLEMENTED YET

        time : list, default=None
            time = [time start, time stop]

        Returns
        -------

        slice_

        window

        zoomtype

        time_range
            time_range = [0, time[1]]

        """

        # -------------------------------------------------
        # determine zome range if necesary
        # -------------------------------------------------
        time_range = None
        if nr_rev:
            raise NotImplementedError
            # input is a number of revolutions, get RPM and sample rate to
            # calculate the required range
            # TODO: automatich detection of RPM channel!
            time_range = nr_rev/(self.rpm_mean/60.)
            # convert to indices instead of seconds
            i_range = int(self.Freq*time_range)
            window = [0, time_range]
            # in case the first datapoint is not at 0 seconds
            i_zero = int(self.sig[0,0]*self.Freq)
            slice_ = np.r_[i_zero:i_range+i_zero]

            zoomtype = '_nrrev_' + format(nr_rev, '1.0f') + 'rev'

        elif time.any():
            time_range = time[1] - time[0]

            i_start = int(time[0]*self.Freq)
            i_end = int(time[1]*self.Freq)
            slice_ = np.r_[i_start:i_end]
            window = [time[0], time[1]]

            zoomtype = '_zoom_%1.1f-%1.1fsec' %  (time[0], time[1])

        return slice_, window, zoomtype, time_range

    def blade_deflection(self):
        """
        """

        # select all the y deflection channels
        db = misc.DictDB(self.ch_dict)

        db.search({'sensortype' : 'state pos', 'component' : 'z'})
        # sort the keys and save the mean values to an array/list
        chiz, zvals = [], []
        for key in sorted(db.dict_sel.keys()):
            zvals.append(-self.sig[:,db.dict_sel[key]['chi']].mean())
            chiz.append(db.dict_sel[key]['chi'])

        db.search({'sensortype' : 'state pos', 'component' : 'y'})
        # sort the keys and save the mean values to an array/list
        chiy, yvals = [], []
        for key in sorted(db.dict_sel.keys()):
            yvals.append(self.sig[:,db.dict_sel[key]['chi']].mean())
            chiy.append(db.dict_sel[key]['chi'])

        return np.array(zvals), np.array(yvals)

class LoadResults_h:
    """Read a HAWC2 result data file, exactly the same, but with float128
    precesision. However, it seems to give the same results as the normal one

    Usage:
    obj = LoadResults(file_path, file_name)

    This class is called like a function:
    HawcResultData() will read the specified file upon object initialization.

    Available output:
    obj.sig[timeStep,channel]   : complete result file in a numpy array
    obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
    obj.error_msg: is 'none' if everything went OK, otherwise it holds the
    error
    """

    # start with reading the .sel file, containing the info regarding
    # how to read the binary file and the channel information
    def __init__(self, file_path, file_name):

        self.debug = False

        self.file_path = file_path
        # remove .log, .dat, .sel extensions who might be accedental left
        file_name = file_name.replace('.htc', '')
        file_name = file_name.replace('.sel', '')
        file_name = file_name.replace('.dat', '')
        file_name = file_name.replace('.log', '')
        self.file_name = file_name
        self.read_sel()
        # continue if the file has been succesfully read
        if self.error_msg == 'none':
            # load the channel id's and scale factors
            scale_factors = self.data_sel()
            # read the binary file
            if self.FileType == 'BINARY':
                self.read_bin(scale_factors)
                # reformat output
                self.reformat_sig_details()
            # read the ASCII file
            elif self.FileType == 'ASCII':
                self.read_ascii()
                # reformat output
                self.reformat_sig_details()
            else:
                print '========================================================'
                print 'unknown file type: ' + self.FileType
                print '========================================================'
                self.error_msg = 'error: unknown file type'
                self.sig = []

    def read_sel(self):
        # anticipate error on file reading
        try:
            # open file, read and close
            go_sel = self.file_path + self.file_name + '.sel'
            FILE = open(go_sel, "r")
            self.lines = FILE.readlines()
            FILE.close()
            self.error_msg = 'none'

        # error message if the file does not exists
        except:
            # print 26*' ' + 'ERROR'
            print 50*'='
            print self.file_name + '.sel could not be found'
            print 50*'='
            self.error_msg = 'error: file not found'

    def data_sel(self):
        # increase precision
        # D.getcontext().prec = 50

        # scan through all the lines in the file
        line_nr = 1
        # channel counter for ch_details
        ch = 0
        for line in self.lines:
            # on line 9 we can read following paramaters:
            if line_nr == 9:
                settings = line.split(' ')
                # delete all empty string values
                for k in range(settings.count('')):
                    settings.remove('')

                # and assign proper values with correct data type
                self.N = np.float128(settings[0])
                self.Nch = np.float128(settings[1])
                self.Time = np.float128(settings[2])
                # ditch the last 3 characters, the \n does not work under linux
                nrchars = len(settings[3])-2
                settings[3] = settings[3][0:nrchars]
                self.FileType = settings[3].replace('\n','')
                self.Freq = self.N/self.Time

                # prepare list variables
                self.ch_details = np.ndarray(shape=(self.Nch,3),dtype='<U100')
                # it seems that float64 reeds the data correctly from the file
                scale_factors = scipy.zeros(self.Nch, dtype='Float128')
                #self.scale_factors_dec = scipy.zeros(self.Nch, dtype='f8')
                i = 0

            # starting from line 13, we have the channels info
            if line_nr > 12:
                # read the signal details
                if line_nr < 13 + self.Nch:
                    # remove leading and trailing whitespaces from line parts
                    self.ch_details[ch,0] = line[12:43].strip() # chID
                    self.ch_details[ch,1] = line[43:54].strip() # chUnits
                    self.ch_details[ch,2] = line[54:-1].strip() # chDescr
                    ch += 1
                # read the signal scale parameters for binary format
                elif line_nr > 14 + self.Nch:
                    scale_factors[i] = np.float128(line)
                    # print scale_factors[i]
                    #self.scale_factors_dec[i] = D.Decimal(line)
                    i = i + 1
                # stop going through the lines if at the end of the file
                if line_nr == 2*self.Nch + 14:
                    return scale_factors
                    break

            # counting the line numbers
            line_nr = line_nr + 1

    def read_bin(self, scale_factors):
        # if there is an error reading the binary file (for instance if empty)
        try:
            if self.debug: print self.N, self.Nch

            # read the binary file
            go_binary = self.file_path + self.file_name + '.dat'
            FILE = open(go_binary, mode='rb')

            # create array, put all the binary elements as one long chain in it

            # binvalues = array.array('h')
            # old method
            # binvalues.fromfile(FILE, self.N * self.Nch)

            # new method, more accurate? same results?
            binvalues = np.fromfile(FILE, dtype='h', count=-1, sep='')
            FILE.close()

            # convert now to a structured numpy array
            # sig = np.array(binvalues, np.float)
#            sig = np.array(binvalues, dtype='Float128')
            sig = np.array(binvalues, dtype='int16')
            sig = np.reshape(sig, (self.Nch, self.N))
            # sig = np.array(binvalues)
            # sig = np.reshape(sig, (self.Nch, self.N))

            # apperently Nch and N had to be reversed to read it correctly
            # is this because we are reading a Fortran array with Python C code?
            # so now transpose again so we have sig(time, channel)
            sig = np.transpose(sig)
            # create diagonal vector of size (Nch,Nch)
            dig = np.diag(scale_factors)
            # now all rows of column 1 are multiplied with dig(1,1)
            sig = np.dot(sig,dig)
            self.sig = sig
            # 'file name;' + 'lnr;msg;'*(len(MsgList)) + '\n'
        except:
            self.sig = []
            self.error_msg = 'error: reading binary file failed'

    def read_ascii(self):
        try:
            go_ascii = self.file_path + self.file_name + '.dat'
            self.sig = np.fromfile(go_ascii, dtype=np.float128, sep='  ')
            self.sig = self.sig.reshape((self.N, self.Nch))
        except:
            self.sig = []
            self.error_msg = 'error: reading ascii file failed'
            print '========================================================'
            print self.error_msg
            print '========================================================'


    def reformat_sig_details(self):
        """Change HAWC2 output description of the channels short descriptive
        strings, usable in plots

        obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
        """

        # CONFIGURATION: mappings between HAWC2 and short good output:
        change_list = []
        change_list.append( ['original','new improved'] )

#        change_list.append( ['Mx coo: hub1','blade1 root bending: flap'] )
#        change_list.append( ['My coo: hub1','blade1 root bending: edge'] )
#        change_list.append( ['Mz coo: hub1','blade1 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub2','blade2 root bending: flap'] )
#        change_list.append( ['My coo: hub2','blade2 root bending: edge'] )
#        change_list.append( ['Mz coo: hub2','blade2 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub3','blade3 root bending: flap'] )
#        change_list.append( ['My coo: hub3','blade3 root bending: edge'] )
#        change_list.append( ['Mz coo: hub3','blade3 root bending: torsion'] )

        change_list.append( ['Mx coo: blade1','blade1 flap'] )
        change_list.append( ['My coo: blade1','blade1 edge'] )
        change_list.append( ['Mz coo: blade1','blade1 torsion'] )

        change_list.append( ['Mx coo: blade2','blade2 flap'] )
        change_list.append( ['My coo: blade2','blade2 edge'] )
        change_list.append( ['Mz coo: blade2','blade2 torsion'] )

        change_list.append( ['Mx coo: blade3','blade3 flap'] )
        change_list.append( ['My coo: blade3','blade3 edeg'] )
        change_list.append( ['Mz coo: blade3','blade3 torsion'] )

        change_list.append( ['Mx coo: hub1','blade1 out-of-plane'] )
        change_list.append( ['My coo: hub1','blade1 in-plane'] )
        change_list.append( ['Mz coo: hub1','blade1 torsion'] )

        change_list.append( ['Mx coo: hub2','blade2 out-of-plane'] )
        change_list.append( ['My coo: hub2','blade2 in-plane'] )
        change_list.append( ['Mz coo: hub2','blade2 torsion'] )

        change_list.append( ['Mx coo: hub3','blade3 out-of-plane'] )
        change_list.append( ['My coo: hub3','blade3 in-plane'] )
        change_list.append( ['Mz coo: hub3','blade3 torsion'] )
        # this one will create a false positive for tower node nr1
        change_list.append( ['Mx coo: tower','tower top momemt FA'] )
        change_list.append( ['My coo: tower','tower top momemt SS'] )
        change_list.append( ['Mz coo: tower','yaw-moment'] )

        change_list.append( ['Mx coo: chasis','chasis momemt FA'] )
        change_list.append( ['My coo: chasis','yaw-moment chasis'] )
        change_list.append( ['Mz coo: chasis','chasis moment SS'] )

        change_list.append( ['DLL inp  2:  2','tower clearance'] )

        self.ch_details_new = np.ndarray(shape=(self.Nch,3),dtype='<U100')

        # approach: look for a specific description and change it.
        # This approach is slow, but will not fail if the channel numbers change
        # over different simulations
        for ch in range(self.Nch):
            # the change_list will always be slower, so this loop will be
            # inside the bigger loop of all channels
            self.ch_details_new[ch,:] = self.ch_details[ch,:]
            for k in range(len(change_list)):
                if change_list[k][0] == self.ch_details[ch,0]:
                    self.ch_details_new[ch,0] =  change_list[k][1]
                    # channel description should be unique, so delete current
                    # entry and stop looking in the change list
                    del change_list[k]
                    break

#        self.ch_details_new = ch_details_new

#class htc:
#
#    def __init__(self, targetpath, htcfile):
#        """
#        Read a htc file and parse it to a list object
#        """
#        # set the HawcPars model
#        hm = hawcpar.HawcGrammer()
#        hm.model_name = htcfile
#        hm.model_path = targetpath
#        self.result = hm.parse()
##        pprint.pprint(self.result.asList())
#
#    def htc2file(self):
#        """
#        Print the list htc object back to a file
#        """
#
#        for k in range(len(self.result)):
#            for m in range(len(self.result[k])):
#                if str(type(self.result)).find('list') > -1:
#                    for n in range(len(self.result[k][m])):
#                        pass

class St:
    # TODO: Merge this class with PlotModelData
    """
    Dealing with the HAWC2 Structure file (st). It contains all the structural
    data of different bodies

    LOAD a specific file and convert to st_dict:
        st_dict:
            key ID: st filename
            key   : set-subset-some comments?
            value : array holding the data points
    HOW TO DEAL WITH THE COMMENTS? add a key comments?

    SAVE a st_dict to a file
    """

def rayleigh_damping(alpha, beta, omega_range=[0.1, 10]):
    """
    psi: damping ratio
    omega: range of natural frequencies
    alpha: mass proportional damping
    beta: stiffness proportional damping
    """

    # based on:
    # Computation of Rayleigh Damping Coefficients for Large Systems
    omega = np.arange(omega_range[0], omega_range[1], 0.01)
    psi = (alpha/(2.*omega)) + (beta*omega/2.)
    return psi, omega



class ConvertPC:
    # constructor
    def __init__(self):
        self.rows = 0
        self.columns = 0
        self.readPath = ''
        self.writePath = ''
        self.listArray = []
        self.spacing = 0

    # read the file
    def readFile(self):
        FILE = open(self.readPath)
        lines = FILE.readlines()
        FILE.close()
        return lines

    def read_pc_file(self):
        FILE = open(self.readPath)
        lines = FILE.readlines()
        FILE.close()

        # total number of lines

        # total number of columns

        # create array
        for line in lines:
            # do nothing if it is an empty line
            go = 'False'
            try:
                float(line[0:2].strip())
                go = 'True'
            except:
                if line[0:1] != 'r':
                    line = line.replace('\n', '')
                    temp = []
                    temp.append('ADDNOSPACING')
                    temp.append(line)
                    self.listArray.append(temp)
            if (line == '\n'):
                pass
                # temp.append('ADDNOSPACING')
                # temp.append(line)
                # self.listArray.append(temp)
            elif (go == 'True') or line[0:1] == 'r':
                # lose all characters originating from other formats
                line = line.replace('\n', '')
                line = line.replace('\t', ' ')
                lineArray = line.split(' ')
                # lose all the empty places and convert to array
                rem = remove_from_list()
                rem.value = ''
                rem.array = lineArray
                lineArray = rem.remove()
                self.listArray.append(lineArray)
        self.rows = len(self.listArray)

    def write_pc_file(self):
        """
        Starting point is a list which contains 1D arrays.
        Create an equally spaced text file for which each list item corresponds
        to a row,the 1D array items corresponds to the columns
        """
        # go through all the data sets
        lines = ''
        noSpacing = 'False'
        for row in self.listArray:
            for column in row:
                if column == 'ADDNOSPACING':
                    noSpacing = 'True'
                elif (noSpacing == 'True'):
                    lines = lines + column
                else:
                    # if self.spacing is not a number, than use tabs
                    try:
                        lines = lines + column.ljust(self.spacing)
                    except:
                        lines = lines + column + '    '
            lines = lines + '\n'
            noSpacing = 'False'
        # write the file
        FILE = open(self.writePath, 'w')
        FILE.write(lines)
        FILE.close()
# DEPRICATED: use Simulations.py insted!!
class HtcMaster:
    """
    DEPRICATED: use Simulations.py insted!!
    """

    def __init__(self):
        """
        DEPRICATED: use Simulations.py insted!!
        """
        self.Ti_ref = 0.14
        # 13.0 is the rotor diameter + 5%
        self.rotor_diameter = 13.
        self.tip_speed_ratio_star = 7.8
        self.blade_radius = 0.82

        # create a dictionary with the tag name as key as the default value
        self.tags_def = dict()

        # is it an automatic generated case_id? True for yes
        self.set_case_id = True

        # should we print where the file is written?
        self.debug_print = True

        # the master file directory is not allowed to change in one htc_dict
        self.target = None
        self.tags_def['[master_htc_dir]'] = self.target
        # neither is the master file
        self.master = None
        # switch to True if the blade_hawtopt key is present in the tags_def
#        self.blade_hawtopt = None
        self.tags_def['[master_htc_file]'] = self.master

        #-----------------------------------------------------------------------
        # PBS script settings
        self.tags_def['[model_dir_server]'] = None
        #self.tags_def['[model_dir_server]'] = '/home/dave/tmp/'
        # following dirs are relative to the model_dir_server!!
        # they indicate the location of the SAVED (!!) results, they can be
        # different from the execution dirs on the node
        self.tags_def['[res_dir]'] = 'results/'
        self.tags_def['[log_dir]'] = 'logfiles/'
        self.tags_def['[turb_dir]'] = 'turb/'
        self.tags_def['[animation_dir]'] = 'animation/'
        self.tags_def['[wake_dir]'] = 'none/'
        self.tags_def['[meander_dir]'] = 'none/'
        self.tags_def['[model_zip]'] = None
        self.tags_def['[htc_dir]'] = 'htc/'

        self.tags_def['[out_format]'] = 'HAWC_BINARY'
        self.tags_def['[turb_seed]'] = 10
        self.tags_def['[turb_grid_x]'] = 8192
        self.tags_def['[turb_grid_yz]'] = 8

    def _sweep_tags(self):
        """
        The original way with all tags in the htc file for each blade node
        """
        # set the correct sweep cruve, these values are used
        a = self.tags_def['[sweep_amp]']
        b = self.tags_def['[sweep_exp]']
        z0 = self.tags_def['[sweep_curve_z0]']
        ze = self.tags_def['[sweep_curve_ze]']
        nr = self.tags_def['[nr-of-nodes-per-blade]']
        # format for the x values in the htc file
        ff = ' 1.03f'
        for zz in range(nr):
            it_nosweep = '[x'+str(zz+1)+'-nosweep]'
            item = '[x'+str(zz+1)+']'
            z = self.tags_def['[z'+str(zz+1)+']']
            if z >= z0:
                curve = eval(self.tags_def['[sweep_curve_def]'])
                # new swept position = original + sweep curve
                self.tags_def[item] = format(self.tags_def[it_nosweep]+curve,ff)
            else:
                self.tags_def[item] = format(self.tags_def[it_nosweep], ff)

    def _all_in_one_blade_tag(self):
        """
        Automatically get the number of nodes correct in master.tags_def based
        on the number of blade nodes

        WARNING: initial x position of the half chord point is assumed to be
        zero
        """
        # and save under tag [blade_htc_node_input] in htc input format

        nr_nodes = self.tags_def['[nr-of-nodes-per-blade]']

        blade = np.loadtxt(self.tags_def['[blade_hawtopt]'])
        # in the htc file, blade root =0 and not blade hub radius
        blade[:,0] = blade[:,0] - blade[0,0]
        # interpolate to the specified number of nodes
        radius_new = np.linspace(blade[0,0], blade[-1,0], nr_nodes)
        # make sure that radius_hr is just slightly smaller than radius low res
        radius_new[-1] = blade[-1,0]-0.00000001
        twist_new = interpolate.griddata(blade[:,0], blade[:,2], radius_new)
        # blade_new is the htc node input part:
        # sec 1   x     y     z   twist;
        blade_new = scipy.zeros((len(radius_new),4))
        blade_new[:,2] = radius_new
        blade_new[:,3] = twist_new*-1

        # set the correct sweep cruve, these values are used
        a = self.tags_def['[sweep_amp]']
        b = self.tags_def['[sweep_exp]']
        z0 = self.tags_def['[sweep_curve_z0]']
        ze = self.tags_def['[sweep_curve_ze]']
        tmp = 'nsec ' + str(nr_nodes) + ';'
        for k in range(nr_nodes):
            tmp += '\n'
            i = k+1
            z = blade_new[k,2]
            y = blade_new[k,1]
            twist = blade_new[k,3]
            # x position, sweeping?
            if z >= z0:
                x = eval(self.tags_def['[sweep_curve_def]'])
            else:
                x = 0.0

            # the node number
            tmp += '        sec ' + format(i, '2.0f')
            tmp += format(x, ' 11.04f')
            tmp += format(y, ' 11.04f')
            tmp += format(z, ' 11.04f')
            tmp += format(twist, ' 11.04f')
            tmp += ' ;'

        self.tags_def['[blade_htc_node_input]'] = tmp

    def variable_tags(self):
        """
        case_id can be set outside this function, for instance for benchmarking
        """
        # TODO: tip_speed_ratio_start, blade_radius should be determined
        # outside HawcPy, they change for each turbine!!

        #-----------------------------------------------------------------------
        # these parameters are functions of others!! reload when changing!
        V = self.tags_def['[windspeed]']
        t = self.tags_def['[duration]']
        self.tags_def['[TI]'] = self.Ti_ref * ((0.75*V)+5.6) / V
        self.tags_def['[turb_dx]'] = V*t/self.tags_def['[turb_grid_x]']
        self.tags_def['[turb_dy]'] = \
                self.rotor_diameter / self.tags_def['[turb_grid_yz]']
        self.tags_def['[turb_dz]'] = \
                self.rotor_diameter / self.tags_def['[turb_grid_yz]']
        self.tags_def['[Turb_base_name]'] = 'turb_s' + \
            str(self.tags_def['[turb_seed]']) + '_' + str(V)
        # total simulation time
        self.tags_def['[time_stop]'] = self.tags_def['[t0]'] + t
        # controller tag
        # self.tags_def['[stop_t0]'] = 0
        # for the wind speed factor, make sure that each simulation has the same
        # starting tip speed ratio. Given is the initial rotor speed
        tsr_start = self.tip_speed_ratio_star
        blade_radius = self.blade_radius
        V_start = self.tags_def['[ini_rotvec]']*blade_radius/tsr_start
        # wsp_factor * V = V_start, in order to maintain same tip speed ratio
        self.tags_def['[wsp_factor]'] = V_start / V

        # which tower shadow model to activate?
        # for the tower_shadow_jet
        if self.tags_def['[tower_shadow]'] == 2:
            self.tags_def['[tsj1]'] = ''
            self.tags_def['[tsj2]'] = ';'
        # tower_shadow_jet2
        elif self.tags_def['[tower_shadow]'] == 4:
            self.tags_def['[tsj1]'] = ';'
            self.tags_def['[tsj2]'] = ''
        # others are currently not supported by the master file, switch off all
        else:
            self.tags_def['[tsj1]'] = ';'
            self.tags_def['[tsj2]'] = ';'

        # set the sweep tags [x1], [tw1], based on [x1-nosweep]
        if self.tags_def.has_key('[x1]'):
            self._sweep_tags()
        if self.tags_def.has_key('[blade_htc_node_input]') \
            and self.tags_def.has_key('[blade_hawtopt]'):
            self._all_in_one_blade_tag()

        # directory settings:
        self.tags_def['[htc_dir_server]'] = \
            self.tags_def['[model_dir_server]'] + self.tags_def['[htc_dir]']

        if self.set_case_id:
            # last entry, it is a function of the parameters defined above
            self.tags_def['[case_id]'] = 's' + str(self.tags_def['[tu_seed]'])+\
                '_' + self.tags_def['[extra sim ID]'] + \
                '_c'  + str(self.tags_def['[coning]']) + \
                '_y' + str(self.tags_def['[wyaw]']) + \
                '_ab' + str(self.tags_def['[sweep_amp]']) + \
                str(self.tags_def['[sweep_exp]']) + '_' + str(V) + 'ms'
        #-----------------------------------------------------------------------
        # return tags

    def loadmaster(self):
        """
        Load the master file, path to master file is defined in
        __init__(): target, master
        """

        # what is faster, load the file in one string and do replace()?
        # or the check error log approach?

        # load the file:
        print 'loading master: ' + self.target + self.master
        FILE = open(self.target + self.master, 'r')
        lines = FILE.readlines()
        FILE.close()

        # convert to string:
        self.master_str = ''
        for line in lines:
            self.master_str += line


    def createcase_check(self, htc_dict_repo, \
                            tags, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        Check if a certain case name already exists in a specified htc_dict.
        If true, return a message and do not create the case. It can be that
        either the case name is a duplicate and should be named differently,
        or that the simulation is a duplicate and it shouldn't be repeated.
        """

        # TODO: redesign tags and tags_def approach. It behaves not safe!

        # replace all tags which are specific for this case, ie different than
        # the default values in tags_def
        for k in tags:
            # this will give a copy of the value, so no reference
            self.tags_def[k] = tags[k]
        # since some tags are functions of others, re-calculate them
        # TODO: WARNING !! make sure you NOT to overwrite the freshly set tags
        # with the variable_tags() command!!
        self.variable_tags()

        # is the [case_id] tag unique, given the htc_dict_repo?
        if self.debug_print:
            print 'checking if following case is in htc_dict_repo: '
            print self.tags_def['[case_id]'] + '.htc'

        if htc_dict_repo.has_key(self.tags_def['[case_id]'] + '.htc'):
            # if the new case_id already exists in the htc_dict_repo
            # do not add it again!
            # print 'case_id key is not unique in the given htc_dict_repo!'
            raise UserWarning, \
                'case_id key is not unique in the given htc_dict_repo!'
        else:
            htc = self.createcase(tags, tmp_dir=tmp_dir, write_htc=write_htc)
            return htc


    def createcase(self, tags, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        replace all the tags from the master file and save the new htc file
        """

        htc = self.master_str
        # tags_def = copy.copy(self.tags_def)

        # this part is moved up to the createcase_check!!
#        # replace all tags which are specific for this case, ie different than
#        # the default values in tags_def
#        for k in tags:
#            # this will give a copy of the value, so no reference
#            self.tags_def[k] = tags[k]
#        # since some tags are functions of others, re-calculate them
#        # TODO: WARNING !! make sure you NOT to overwrite the freshly set tags
#        # with the variable_tags() command!!
#        self.variable_tags()

        # and now replace all the tags in the htc master file
        # when iterating over a dict, it will give the key, given in the
        # corresponding format (string keys as strings, int keys as ints...)
        for k in self.tags_def:
            # TODO: give error if a default is not defined, like null
            # if string is not found, it will do nothing
            htc = htc.replace(k, str(self.tags_def[k]))

        # and save the the case htc file:
#        try:

        case = self.tags_def['[case_id]'] + '.htc'
        htc_target = self.tags_def['[htc_dir_server]']
        if self.debug_print:
            print 'htc will be written to: ' + htc_target + case

        # if the htc directory does not exists yet, create first
        if not os.path.exists(htc_target):
            os.makedirs(htc_target)

        # and write the htc file to the temp dir first
        if write_htc:
            write_file(htc_target + case, htc, 'w')
            # write_file(tmp_dir + case, htc, 'w')

        # if the results dir does not exists, create it!
        dir=self.tags_def['[model_dir_server]'] +self.tags_def['[res_dir]']
        if not os.path.exists(dir):
            os.makedirs(dir)
        # if the logfile dir does not exists, create it!
        dir=self.tags_def['[model_dir_server]'] +self.tags_def['[log_dir]']
        if not os.path.exists(dir):
            os.makedirs(dir)
        # if the eigenfreq dir does not exists, create it!
        dir = self.tags_def['[model_dir_server]'] \
            + self.tags_def['[eigenfreq_dir]']
        if not os.path.exists(dir):
            os.makedirs(dir)
        # if the animation dir does not exists, create it!
        dir = self.tags_def['[model_dir_server]'] \
            + self.tags_def['[animation_dir]']
        if not os.path.exists(dir):
            os.makedirs(dir)

#        except:
#            raise UserWarning, \
#            'MasterFile001 - dict tags should have a key \'case_id\''

        # return the used tags, some parameters can be used later, such as the
        # turbulence name in the pbs script
        # return as a dictionary, to be used in htc_dict
        tmp = dict()
        # return a copy of the tags_def, otherwise you will not catch changes
        # made to the different tags in your sim series
        tmp[case] = copy.copy(self.tags_def)
        return tmp
#        return [case, self.tags_def]


class MasterFile: # depricated, use HtcMaster instead
    """
    Like the spreadsheet for creating a lot of htc files, based on a master.
    First a dictionary is created with all available tags and corresponding
    default values.
    Afterwards, for each case, a new dictionary is created wich defines the
    non default values for certain parameters.

    Usage:
        master = MasterFile()

        modify following to change the default values
            master.tags_def['[key_name]'] = value
            ...
        or create a list which will overwrite the default values:
            new_tags['[existing key]'] = new value
            ...

        The htc files are generated as follows:
            htc_dict = createcase(new_tags)

        where htc_dict is a dictionary with
            [key=case name, value=used_tags_dict]

    For the moment, some configuration parameters are also stored in tags_def:
        tags_def['[walltime]'] = '01:30:00'
    other general configuration paramters in the __init__()
        target = '/home/dave/PhD/Projects/Hawc2Models/3e_yaw/htc/'
        master = 'master2_0_v57_1_dve.htc'
    """

    def __init__(self):
        """
        All tags, like for example [time_step], are defined here in a list
        This should be made as a simple input file, with either a space,
        new line or wathever as seperator
        """
        # is it an automatic generated case_id? True for yes
        self.set_case_id = True

        # should we print where the file is written?
        self.debug_print = True

        self.target = ''
        self.Ti_ref = 0.14

        # create a dictionary with the tag name as key as the default value
        self.tags_def = dict()

        # these are the options, they are not the tags...
        # quite general one, this is used for the pbs script
        # format: HOURS:MINUTES:SECONDS
        self.tags_def['[walltime]'] = '01:30:00'

        # the master file directory is not allowed to change in one htc_dict
        self.target = '/home/dave/PhD/Projects/Hawc2Models/3e_yaw/htc/'
        self.tags_def['[master_htc_dir]'] = self.target
        # neither is the master file
        self.master = 'master2_0_v57_1_dve.htc'
        self.tags_def['[master_htc_file]'] = self.master

        #-----------------------------------------------------------------------
        # PBS script settings
        self.tags_def['[model_dir_server]'] = '/mnt/thyra/HAWC2/3e_yaw/'
        #self.tags_def['[model_dir_server]'] = '/home/dave/tmp/'
        # following dirs are relative to the model_dir_server!!
        # they indicate the location of the SAVED (!!) results, they can be
        # different from the execution dirs on the node
        self.tags_def['[res_dir]'] = 'results/'
        self.tags_def['[log_dir]'] = 'logfiles/'
        self.tags_def['[turb_dir]'] = 'turb/'
        self.tags_def['[wake_dir]'] = 'none/'
        self.tags_def['[meander_dir]'] = 'none/'
        self.tags_def['[model_zip]'] = '3e_yaw.zip'
        self.tags_def['[htc_dir]'] = 'htc/'
        #-----------------------------------------------------------------------
        self.tags_def['[extra sim ID]'] = ''
        self.tags_def['[Windspeed]'] = 10.0
        self.tags_def['[duration]'] = 50.0
        # start outputting data from (in order to ignore initial transients)
        self.tags_def['[t0]'] = 20.0

        # turbulence parameters
        self.tags_def['[dt_sim]'] = 0.005
        self.tags_def['[out_format]'] = 'HAWC_BINARY'
        self.tags_def['[tu_seed]'] = 10
        self.tags_def['[turb_grid_x]'] = 8192
        self.tags_def['[turb_grid_yz]'] = 8
        self.tags_def['[tu_model]'] = 1

        self.tags_def['[wyaw]'] = 0
        self.tags_def['[wtilt]'] = 0
        self.tags_def['[coning]'] = -2.5
        self.tags_def['[tilt]'] = 0
        self.tags_def['[aefile]'] = ''
        self.tags_def['[pcfile]'] = ''

        # wind conditions
        self.tags_def['[shear_type]'] = 3
        self.tags_def['[shear_exp]'] = 0.2
        self.tags_def['[gust]'] = ';' # switch
        self.tags_def['[gust_type]'] = ''
        self.tags_def['[G_A]'] = ''
        self.tags_def['[G_phi0]'] = ''
        self.tags_def['[G_T]'] = ''
        self.tags_def['[G_t0]'] = ''
        self.tags_def['[e1]'] = ';' # switch

#        self.tags_def['[stop_pitvel]'] = 8
#        self.tags_def['[cutin_t0]'] = -1
#        self.tags_def['[nshutd]'] = -1
#        self.tags_def['[eshutd]'] = -1

        self.tags_def['[rotor start]'] = 0.0
        self.tags_def['[init_wr]'] = 0.2
        self.tags_def['[fix_wr]'] = 0.86
        self.tags_def['[wind_ramp]'] = ';' # switch
        self.tags_def['[induction]'] = 1
        self.tags_def['dyn_stall_method'] = 2
        self.tags_def['[x1]'] = 0
        self.tags_def['[x2]'] = 0
        self.tags_def['[x3]'] = 0
        self.tags_def['[x4]'] = 0
        self.tags_def['[x5]'] = 0
        self.tags_def['[x6]'] = 0
        self.tags_def['[x7]'] = 0
        self.tags_def['[x8]'] = 0
        self.tags_def['[x9]'] = 0
        self.tags_def['[x10]'] = 0
        self.tags_def['[x11]'] = 0
        self.tags_def['[x12]'] = 0
        self.tags_def['[x13]'] = 0
        self.tags_def['[x14]'] = 0
        self.tags_def['[x15]'] = 0
        self.tags_def['[x16]'] = 0
        self.tags_def['[x17]'] = 0
        self.tags_def['[x18]'] = 0
        self.tags_def['[x19]'] = 0
        self.tags_def['[sweep_amp]'] = 1
        self.tags_def['[sweep_exp]'] = 3

        self.tags_def['[tower_shadow]'] = 2
        self.tags_def['[animation]'] = ';' # switch
        self.tags_def['[tow_damp]'] = ';' # switch
        self.tags_def['[td_start]'] = 0
        self.tags_def['[td_stop]'] = 0
        self.tags_def['[aeset]'] = 8

        # the variable tags are set when createcase_check is called. No need
        # to call that here, since we are not ready yet to make a htc file
#        # set tags who are actually a function of the others (as defined above)
#        self.variable_tags()

        # and load the master file into the memory--do only after __init__(),
        # otherwise you lose changes in the default location parameters
        # this should be done only once!
        # self.loadmaster()

    def variable_tags(self):
        """
        case_id can be set outside this function, for instance for benchmarking
        """
        #-----------------------------------------------------------------------
        # these parameters are functions of others!! reload when changing!
        V = self.tags_def['[Windspeed]']
        t = self.tags_def['[duration]']
        self.tags_def['[TI]'] = self.Ti_ref * ((0.75*V)+5.6) / V
        self.tags_def['[turb_dx]'] = V*t/self.tags_def['[turb_grid_x]']
        # 13.0 is the rotor diameter + 5%
        self.tags_def['[turb_dy]'] = 13.0 / self.tags_def['[turb_grid_yz]']
        self.tags_def['[turb_dz]'] = 13.0 / self.tags_def['[turb_grid_yz]']
        self.tags_def['[Turb_base_name]'] = 'turb_s' + \
            str(self.tags_def['[tu_seed]']) + '_' + str(V)
        # total simulation time
        self.tags_def['[time_stop]'] = self.tags_def['[t0]'] + t
        # controller tag
        # self.tags_def['[stop_t0]'] = 0
        # for the wind speed factor, make sure that each simulation has the same
        # starting tip speed ratio. Given is the initial rotor speed
        tip_speed_ratio_start = 7.8
        blade_radius = 10.5
        V_start = self.tags_def['[init_wr]']*blade_radius/tip_speed_ratio_start
        # wsp_factor * V = V_start, in order to maintain same tip speed ratio
        self.tags_def['[wsp_factor]'] = V_start / V

        # which tower shadow model to activate?
        # for the tower_shadow_jet
        if self.tags_def['[tower_shadow]'] == 2:
            self.tags_def['[tsj1]'] = ''
            self.tags_def['[tsj2]'] = ';'
        # tower_shadow_jet2
        elif self.tags_def['[tower_shadow]'] == 4:
            self.tags_def['[tsj1]'] = ';'
            self.tags_def['[tsj2]'] = ''
        # others are currently not supported by the master file, switch off all
        else:
            self.tags_def['[tsj1]'] = ';'
            self.tags_def['[tsj2]'] = ';'

        # set the correct sweep cruve
        a = self.tags_def['[sweep_amp]']
        b = self.tags_def['[sweep_exp]']
        z0 = self.tags_def['[sweep_curve_z0]']
        ze = self.tags_def['[sweep_curve_ze]']
        nr = self.tags_def['[nr-of-nodes-per-blade]']
        # format for the x values in the htc file
        ff = ' 1.05f'
        for zz in range(nr):
            it_nosweep = '[x'+str(zz+1)+'-nosweep]'
            item = '[x'+str(zz+1)+']'
            z = self.tags_def['[z'+str(zz+1)+']']
            if z >= z0:
                curve = eval(self.tags_def['[sweep_curve_def]'])
                # new swept position = original + sweep curve
                self.tags_def[item] = format(self.tags_def[it_nosweep]+curve,ff)
            else:
                self.tags_def[item] = format(self.tags_def[it_nosweep], ff)

        # directory settings:
        self.tags_def['[htc_dir_server]'] = \
            self.tags_def['[model_dir_server]'] + self.tags_def['[htc_dir]']

        if self.set_case_id:
            # last entry, it is a function of the parameters defined above
            self.tags_def['[case_id]'] = 's' + str(self.tags_def['[tu_seed]'])+\
                '_' + self.tags_def['[extra sim ID]'] + \
                '_c'  + str(self.tags_def['[coning]']) + \
                '_y' + str(self.tags_def['[wyaw]']) + \
                '_ab' + str(self.tags_def['[sweep_amp]']) + \
                str(self.tags_def['[sweep_exp]']) + '_' + str(V) + 'ms'
        #-----------------------------------------------------------------------
        # return tags

    def loadmaster(self):
        """
        Load the master file, path to master file is defined in
        __init__(): target, master
        """

        # what is faster, load the file in one string and do replace()?
        # or the check error log approach?

        # load the file:
        print 'loading master: ' + self.target + self.master
        FILE = open(self.target + self.master, 'r')
        lines = FILE.readlines()
        FILE.close()

        # convert to string:
        self.master_str = ''
        for line in lines:
            self.master_str += line


    def createcase_check(self, htc_dict_repo, \
                            tags, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        Check if a certain case name already exists in a specified htc_dict.
        If true, return a message and do not create the case. It can be that
        either the case name is a duplicate and should be named differently,
        or that the simulation is a duplicate and it shouldn't be repeated.
        """

        # replace all tags which are specific for this case, ie different than
        # the default values in tags_def
        for k in tags:
            # this will give a copy of the value, so no reference
            self.tags_def[k] = tags[k]
        # since some tags are functions of others, re-calculate them
        # TODO: WARNING !! make sure you NOT to overwrite the freshly set tags
        # with the variable_tags() command!!
        self.variable_tags()

        # is the [case_id] tag unique, given the htc_dict_repo?
        if self.debug_print:
            print 'checking if following case is in htc_dict_repo: '
            print self.tags_def['[case_id]'] + '.htc'

        if htc_dict_repo.has_key(self.tags_def['[case_id]'] + '.htc'):
            # if the new case_id already exists in the htc_dict_repo
            # do not add it again!
            # print 'case_id key is not unique in the given htc_dict_repo!'
            raise UserWarning, \
                'case_id key is not unique in the given htc_dict_repo!'
        else:
            htc = self.createcase(tags, tmp_dir=tmp_dir, write_htc=write_htc)
            return htc


    def createcase(self, tags, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        replace all the tags from the master file and save the new htc file
        """

        htc = self.master_str
        # tags_def = copy.copy(self.tags_def)

        # this part is moved up to the createcase_check!!
#        # replace all tags which are specific for this case, ie different than
#        # the default values in tags_def
#        for k in tags:
#            # this will give a copy of the value, so no reference
#            self.tags_def[k] = tags[k]
#        # since some tags are functions of others, re-calculate them
#        # TODO: WARNING !! make sure you NOT to overwrite the freshly set tags
#        # with the variable_tags() command!!
#        self.variable_tags()

        # and now replace all the tags in the htc master file
        # when iterating over a dict, it will give the key, given in the
        # corresponding format (string keys as strings, int keys as ints...)
        for k in self.tags_def:
            # TODO: give error if a default is not defined, like null
            # if string is not found, it will do nothing
            htc = htc.replace(k, str(self.tags_def[k]))

        # and save the the case htc file:
        try:
            case = self.tags_def['[case_id]'] + '.htc'
            htc_target = self.tags_def['[htc_dir_server]']
            if self.debug_print:
                print 'htc will be written to: ' + htc_target + case

            # if the htc directory does not exists yet, create first
            if not os.path.exists(htc_target):
                os.makedirs(htc_target)

            # and write the htc file to the temp dir first
            if write_htc:
                write_file(htc_target + case, htc, 'w')
                # write_file(tmp_dir + case, htc, 'w')

            # if the results dir does not exists, create it!
            dir=self.tags_def['[model_dir_server]'] + self.tags_def['[res_dir]']
            if not os.path.exists(dir):
                os.makedirs(dir)
            # if the logfile dir does not exists, create it!
            dir=self.tags_def['[model_dir_server]'] + self.tags_def['[log_dir]']
            if not os.path.exists(dir):
                os.makedirs(dir)

        except:
            raise UserWarning, \
            'MasterFile001 - dict tags should have a key \'case_id\''

        # return the used tags, some parameters can be used later, such as the
        # turbulence name in the pbs script
        # return as a dictionary, to be used in htc_dict
        tmp = dict()
        # return a copy of the tags_def, otherwise you will not catch changes
        # made to the different tags in your sim series
        tmp[case] = copy.copy(self.tags_def)
        return tmp
#        return [case, self.tags_def]

def runlocal(htc_dict, sim_id):
    shellscript = ''
    for case in htc_dict:
        # get a shorter version for the current cases tag_dict:
        scriptpath = htc_dict[case]['[model_dir_server]'] +sim_id+ '_runall.sh'
        shellscript += "wine hawc2mb.exe htc/" + case + "\n"

    write_file(scriptpath, shellscript, 'w')

# DEPRICATED: use Simulations.py insted!!
class PBS:
    """
    DEPRICATED: use Simulations.py insted!!

    The part where the actual pbs script is writtin in this class (functions
    create(), starting() and ending() ) is based on the MS Excel macro
    written by Torben J. Larsen

    input a list with htc file names, and a dict with the other paths,
    such as the turbulence file and folder, htc folder and others
    """

    def __init__(self, htc_dict, server='thyra'):
        """
        DEPRICATED: use Simulations.py insted!!

        Define the settings here. This should be done outside, but how?
        In a text file, paramters list or first create the object and than set
        the non standard values??

        where htc_dict is a dictionary with
            [key=case name, value=used_tags_dict]

        where tags as outputted by MasterFile (dict with the chosen options)

        For gorm, maxcpu is set to 1, do not change otherwise you might need to
        change the scratch dir handling.
        """
        self.server = server
        self.debug_print = True

        if server == 'thyra':
            self.maxcpu = 4
        elif server == 'gorm':
            self.maxcpu = 1
        else:
            raise UserWarning, 'server support only for thyra or gorm'

        # pbs script prefix, this name will show up in the qstat
        self.pref = 'HAWC2_'
        self.pbs_dir = ''
        # the actual script starts empty
        self.pbs = ''

        self.htc_dict = htc_dict
        # all directory settings are determined in the htc_dict and are allowed
        # to vary over different cases in on htc_dict!!

        # these directories are also used in check_results()
        # now they are replaced by tags in the htc_dict
#        self.MasterPC = '/mnt/thyra/HAWC2/'
#        self.MasterDir = '3e_yaw/'
#        self.model_path = self.MasterPC + self.MasterDir
#        self.ModelZipFile = '3e_yaw.zip'

        # location of the output messages .err and .out created by the node
        self.pbs_out_dir = 'pbs_out/'

        # NODE DIRs, should match the zip file!! for running on the node
        # all is relative to the zip file structure
        self.results_dir_node = 'results/'
        self.logs_dir_node = 'logfiles/'
        self.htc_dir_node = 'htc/'
        self.animation_dir_node = 'animation/'
        self.eigenfreq_dir_node = 'eigenfreq/'
        self.TurbDirName_node = 'turb/' # turbulence
        self.Turb2DirName_node = 'none/' # wake
        self.Turb3DirName_node = 'none/' # meander
        # these dirs are also specified in the htc file itself, since that
        # will run local on the server node. Therefore they do not need to be
        # tag and changed

        # for the start number, take hour/minute combo
        d = datetime.today()
        tmp = int( str(d.hour)+format(d.minute, '02.0f') )*100
        self.pbs_start_number = tmp
        self.copyback_turb = True


    def create(self):
        """
        Main loop for creating the pbs scripts, based on the htc_dict, which
        contains the case name as key and tag dictionairy as value
        """

        # REMARK: this i not realy consistent with how the result and log file
        # dirs are allowed to change for each individual case...
        # first check if the pbs_out_dir exists, this dir is considered to be
        # the same for all cases present in the htc_dict
        # self.tags_def['[model_dir_server]']
        case0 = self.htc_dict.keys()[0]
        dir = self.htc_dict[case0]['[model_dir_server]'] + self.pbs_out_dir
        if not os.path.exists(dir):
            os.makedirs(dir)

        # number the pbs jobs:
        count2 = self.pbs_start_number
        # initial cpu count is zero
        count1 = 0
        # scan through all the cases
        i, i_tot = 1, len(self.htc_dict)

        for case in self.htc_dict:

            # get a shorter version for the current cases tag_dict:
            tag_dict = self.htc_dict[case]

            # the directories to SAVE the results/logs/turb files
            # load all relevant dir settings: the result/logfile/turbulence/zip
            # they are now also available for starting() and ending() parts
            self.results_dir = tag_dict['[res_dir]']
            self.eigenfreq_dir = tag_dict['[eigenfreq_dir]']
            self.logs_dir = tag_dict['[log_dir]']
            self.animation_dir = tag_dict['[animation_dir]']
            self.TurbDirName = tag_dict['[turb_dir]']
            self.Turb2DirName = tag_dict['[wake_dir]']
            self.Turb3DirName = tag_dict['[meander_dir]']
            self.ModelZipFile = tag_dict['[model_zip]']
            self.htc_dir = tag_dict['[htc_dir]']
            self.model_path = tag_dict['[model_dir_server]']

            if self.debug_print:
                print 'htc_dir in pbs.create:'
                print self.htc_dir
                print self.model_path

            # CAUTION: for copying to the node, you should have the same
            # directory structure as in the zip file!!!
            # only when COPY BACK from the node, place in the custom dir

            # for the very first iteration:
            if count1 == 0:
                # define the path for the new pbs script
                jobid = self.pref + str(count2)
                pbs_path = self.model_path + self.pbs_dir + jobid + ".p"
                # Start a new pbs script, we only need the tag_dict here
                self.starting(tag_dict, jobid)
                count1 = 1

            # if we have used all the cpu's on the node, wrap it up and
            # go to the next node with a new pbs script
            if count1 > self.maxcpu:
                # write the end part of the previous pbs script
                self.ending(pbs_path)
                # print progress:
                print 'pbs progress, script ' + format(i/self.maxcpu, '2.0f')\
                        + '/' + format(i_tot/self.maxcpu, '2.0f')

                # keep track of the number of cpu's (count1) and
                count1 = 1
                # pbs numbering (count2), for the total amount of jobs
                count2 = count2 + 1
                # define the path for the new pbs script
                jobid = self.pref + str(count2)
                pbs_path = self.model_path + self.pbs_dir + jobid + ".p"

                # Start a new pbs script, we only need the tag_dict here
                self.starting(self.htc_dict[case], jobid)

                # The batch system on Gorm allows more than one job per node.
                # Because of this the scratch directory name includes both the
                # user name and the job ID, that is /scratch/$USER/$PBS_JOBID
                # NB! This is different from Thyra!
                if self.server == 'thyra':
                    # navigate to the current cpu on the node
                    self.pbs += "cd /scratch/$USER/CPU_" + str(count1) + '\n'
                elif self.server == 'gorm':
                    self.pbs += 'cd /scratch/$USER/$PBS_JOBID\n'

                # output the current scratch directory
                self.pbs += "pwd\n"
                # zip file has been copied to the node before (in start_pbs())
                # unzip now in the node
                self.pbs += "/usr/bin/unzip " + self.ModelZipFile + '\n'
                # and copy the htc file to the node
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.htc_dir \
                    + case +" ./" + self.htc_dir_node + '\n'

                # turbulence files basenames are defined for the case
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.TurbDirName + \
                    tag_dict['[Turb_base_name]'] + "?.bin" + \
                    " ./"+self.TurbDirName_node + '\n'

                # TODO: the tags_def should first include thise info!!
#                pbs_script += "cp -R $PBS_O_WORKDIR/" + Turb2DirName + "/" \
#                tag_dict['[Turb2_base_name]'] +"?.bin"+" ./"+Turb2DirName +'\n'
#
#                pbs_script += "cp -R $PBS_O_WORKDIR/" + Turb3DirName + "/" + \
#                tag_dict['[Turb3_base_name]'] +"?.bin"+" ./"+Turb3DirName +'\n'

                # the hawc2 execution commands via wine
                self.pbs += "wine HAWC2MB ./" + self.htc_dir_node + case +" &\n"
                self.pbs += "wine get_mac_adresses" + '\n'
                # self.pbs += "cp -R ./*.mac  $PBS_O_WORKDIR/." + '\n'
            else:
                # go the next cpu on the node, no need to start new pbs script
                # writing jobscriptpart in #10
                if self.server == 'thyra':
                    self.pbs += "cd /scratch/$USER/CPU_" + str(count1) + '\n'
                elif self.server == 'gorm':
                    self.pbs += 'cd /scratch/$USER/$PBS_JOBID\n'

                # output the current scratch directory
                self.pbs += "pwd" + ' \n'
                # zip files has been copied to the node before (in start_pbs())
                # unzip now in the node
                self.pbs += "/usr/bin/unzip " + self.ModelZipFile + '\n'
                # and copy the htc file to the node
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.htc_dir \
                    + case +" ./" + self.htc_dir_node + '\n'
                # turbulence files basenames are defined for the case
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.TurbDirName + \
                    tag_dict['[Turb_base_name]'] + "?.bin" + \
                    " ./" + self.TurbDirName_node + '\n'

                # TODO: the tags_def should first include these info!!
#                pbs_script += "cp -R $PBS_O_WORKDIR/" + Turb2DirName + "/" \
#                tag_dict['[Turb2_base_name]']+"?.bin"+" ./"+Turb2DirName +' \n'
#
#                pbs_script += "cp -R $PBS_O_WORKDIR/" + Turb3DirName + "/" + \
#                tag_dict['[Turb3_base_name]']+"?.bin"+" ./"+Turb3DirName +' \n'
                # the hawc2 execution commands via wine
                self.pbs += "wine HAWC2MB ./" + self.htc_dir_node +case+" &\n"
                self.pbs += "wine get_mac_adresses" + '\n'
                # self.pbs += "cp -R ./*.mac  $PBS_O_WORKDIR/." + '\n'

            # if the last job has a partially loaded node or is fully loaded,
            # we still need to finish the pbs script
            if i == i_tot:
                # in case of a partially loaded last node
                if count1 < self.maxcpu:
                    print 'pbs progress, script '+format(i/self.maxcpu,'2.0f')\
                        + '/' + format(i_tot/self.maxcpu, '2.0f') \
                        + ' partially loaded...'
                    # write the end part of the previous pbs script
                    self.ending(pbs_path)

                # or fully loaded last node
                elif count1 == self.maxcpu:
                    print 'pbs progress, script '+format(i/self.maxcpu,'2.0f')\
                        + '/' + format(i_tot/self.maxcpu, '2.0f')
                    # write the end part of the previous pbs script
                    self.ending(pbs_path)
            i += 1

            # the next cpu
            count1 += 1

#        # if we have only one node loaded with 4 cpu's we still need to
#        # finish the pbs script
#        if i_tot == 4:
#            self.ending(pbs_path)


    def starting(self, tag_dict, jobid):
        """
        First part of the pbs script
        """

        # a new clean pbs script!
        self.pbs = ''
        self.pbs += "### Standard Output" + ' \n'

        self.pbs += "#PBS -o ./" + self.pbs_out_dir + jobid + ".out" + '\n'
        # self.pbs += "#PBS -o ./pbs_out/" + jobid + ".out" + '\n'
        self.pbs += "### Standard Error" + ' \n'
        self.pbs += "#PBS -e ./" + self.pbs_out_dir + jobid + ".err" + '\n'
        # self.pbs += "#PBS -e ./pbs_out/" + jobid + ".err" + '\n'
        self.pbs += "### Maximum wallclock time format HOURS:MINUTES:SECONDS\n"
        self.pbs += "#PBS -l walltime=" + tag_dict['[walltime]'] + '\n'
        self.pbs += "#PBS -a [start_time]" + '\n'
        self.pbs += "### Queue name" + '\n'
        # queue names for Thyra are as follows:
        # short walltime queue (shorter than an hour): '#PBS -q xpresq'
        # or otherwise for longer jobs: '#PBS -q workq'
        self.pbs += tag_dict['[pbs_queue_command]'] + '\n'
        self.pbs += "cd $PBS_O_WORKDIR" + '\n'
        # output the current scratch directory
        self.pbs += "pwd \n"
        self.pbs += "### Copy to scratch directory \n"

        for i in range(1,self.maxcpu+1,1):
            if self.server == 'thyra':
                # create for each cpu a different directory on the node
                self.pbs += "mkdir /scratch/$USER/CPU_" + str(i) + '\n'
            # not necesary for gorm, each pbs file is for one hawc2 simulation
            # only so each job is one simulation having one scratch dir

            # output the current scratch directory
            self.pbs += "pwd \n"
            # self.pbs += "cp -R hawc2_model/ /scratch/$USER/CPU_" + i
            # copy the zip files to the cpu dir on the node
            if self.server == 'thyra':
                self.pbs += "cp -R ./" + self.ModelZipFile + \
                    " /scratch/$USER/CPU_" + str(i) + '\n'
            elif self.server == 'gorm':
                self.pbs += "cp -R ./" + self.ModelZipFile + \
                    ' /scratch/$USER/$PBS_JOBID\n'

        self.pbs += "### Execute commands on scratch nodes \n"

    def ending(self, pbs_path):
        """
        Last part of the pbs script, including command to write script to disc
        COPY BACK: from node to
        """

        self.pbs += "### wait for jobs to finish \n"
        self.pbs += "wait\n"
        self.pbs += "### Copy back from scratch directory \n"
        for i in range(1,self.maxcpu+1,1):

            # navigate to the cpu dir on the node
            if self.server == 'thyra':
                self.pbs += "cd /scratch/$USER/CPU_" + str(i) + '\n'
            # The batch system on Gorm allows more than one job per node.
            # Because of this the scratch directory name includes both the
            # user name and the job ID, that is /scratch/$USER/$PBS_JOBID
            # NB! This is different from Thyra!
            elif self.server == 'gorm':
                self.pbs += "cd /scratch/$USER/$PBS_JOBID\n"

            # and copy the results and log files frome the node to the
            # thyra home dir
            self.pbs += "cp -R " + self.results_dir_node + \
                ". $PBS_O_WORKDIR/" + self.results_dir + ".\n"
            self.pbs += "cp -R " + self.logs_dir_node + \
                ". $PBS_O_WORKDIR/" + self.logs_dir + ".\n"
            self.pbs += "cp -R " + self.animation_dir_node + \
                ". $PBS_O_WORKDIR/" + self.animation_dir + ".\n"
            self.pbs += "cp -R " + self.eigenfreq_dir_node + \
                ". $PBS_O_WORKDIR/" + self.eigenfreq_dir + ".\n"

            # copy back turbulence file?
            if self.copyback_turb:
                self.pbs += "cp -R " + self.TurbDirName_node + \
                    ". $PBS_O_WORKDIR/" + self.TurbDirName + ".\n"
#                self.pbs += "cp -R " + self.Turb2DirName + \
#                    ". $PBS_O_WORKDIR/" + self.Turb2DirName + ".\n"
#                self.pbs += "cp -R " + self.Turb3DirName + \
#                    ". $PBS_O_WORKDIR/" + self.Turb3DirName + ".\n"
            # Delete the batch file at the end. However, is this possible since
            # the batch file is still open at this point????
            # self.pbs += "rm "

        self.pbs += "exit"

        if self.debug_print:
            print 'writing pbs script to path: ' + pbs_path

        # and write the script to a file:
        write_file(pbs_path,self.pbs, 'w')
        # make the string empty again, for memory
        self.pbs = ''


    def check_dirs(self):
        """
        Check if all directories exist
        """



    def launch(self, delta_min, detla_sec, delay_min, delay_hours):
        # TODO: finish this function
        """
        For the moment this is just a wrapper for the launch shell script.
        Should be converted to Python at some point.

        However, requires some ssh commands, including login, how to deal
        with that?

        input: sec, min (between jobs), min, hours (delay of first job)
        maximum: 59, 59, 59, 23
        """
        # navigate to correct directory, so output will be stored properly
        os.chdir(self.results_path)

        # execution command
        command = 'ssh dave@thyra launch 1 2 3 4'
        output = os.popen(command)
        # read cmd output
        cmd_out = ''
        for line in output.readlines():
            if len(line) > 1:
                cmd_out += line

    def check_results(self, htc_dict):
        """
        Cross-check if all simulations on the list have returned a simulation.
        Combine with ErrorLogs to identify which errors occur where.

        All directory settings in the given htc_dict should be the same.

        It will look into the directories defined in:
            htc_dict[case]['[model_dir_server]'] = '/mnt/thyra/HAWC2/3e_yaw/'
            following dirs are relative to the model_dir_server!!
            htc_dict[case]['[res_dir]'] = 'results/'
            htc_dict[case]['[log_dir]'] = 'logfiles/'
            htc_dict[case]['[turb_dir]'] = 'turb/'
            htc_dict[case]['[wake_dir]'] = 'none/'
            htc_dict[case]['[meander_dir]'] = 'none/'
            htc_dict[case]['[model_zip]'] = '3e_yaw.zip'
            htc_dict[case]['[htc_dir]'] = 'htc/'
        which are defined for each case in htc_dict

        htc_dict : dict(key=case names, value=corresponding tag_dict's)
        This dictionary can be used to launch the same identical jobs again,
        via PBS.create(). htc files are still in place and do not need to be
        created again.
        """

        # the directories where results and logfiles resides:
        # so just take the dir settings from the first case in the list
        case = htc_dict.keys()[0]
        model_path = htc_dict[case]['[model_dir_server]']
        results_dir = htc_dict[case]['[res_dir]']
        logs_dir = htc_dict[case]['[log_dir]']

        # first create a list with all the present result and log files:
        # load all the files in the given path
        tmp, log_files, res_files = [], [], []

        # be aware, it wil also list the files in the directories who resides
        # in the given directory!
        # files[directory number, 0=path 1=residing dirs 2=filelist]

        # logfiles
        for files in os.walk(model_path + logs_dir):
            tmp.append(files)

#        print '***LOGFILES'
#        print tmp
        print 'path to logfiles:', model_path + logs_dir

        for file in tmp[0][2]:
            # only select the .log files as a positive
            if file.endswith('.log'):
                # and remove the extension
                log_files.append(file.replace('.log', ''))

        # result files
        tmp = []
        for files in os.walk(model_path + results_dir):
            tmp.append(files)

#        print '***RESULTFILES'
#        print tmp

        print 'path to result files:', model_path + results_dir

        datok, selok = dict(), dict()

        for file in tmp[0][2]:

            if file.endswith('.dat'):
                # it can be that the .dat file is very small, or close to zero
                # in that case we had an error as well!
                size = os.stat(model_path + results_dir + file).st_size
                if size > 5:
                    # add the case name only
                    datok[file.replace('.dat','')] = True

            elif file.endswith('.sel'):
                size = os.stat(model_path + results_dir + file).st_size
                if size > 5:
                    # add the case name only
                    selok[file.replace('.sel','')] = True

        # --------
#        # add only the cases which have both sel and dat ok
#        for datkey in datok.keys():
#            # add to temporary dictionary
#            if datkey in selok:
#                res_files.append( datkey )

        # or the fast way: create a list populated by keys occuring in both
        # datok and selok. The value of corresponding key is ignored
        res_files = set(datok) & set(selok)
        # --------

        # now also make a list with only the htc file names, drop the extension
        htc_list = []
        for k in htc_dict:
            htc_list.append(k.replace('.htc',''))

#        # FIRST APPROACH
#        # sort all lists, will this make the searching faster?
#        res_files.sort()
#        log_files.sort()
#        htc_list2.sort()
#
#        # now we can me make a very quick first check. If the 3 lists are not
#        # the same, we need to look for the missing files
#        check_res, check_log = False, False
#        if htc_list2 == res_files:
#            check_res = True
#        if htc_list2 == log_files:
#            check_log = True

        # SECOND APPROACH:
        # determine with cases have both a log and res file: this is the
        # intersection of log and res files! htc_list2 is included to avoid
        # that older log/res files still in the directory polute the results

        htc_set = set(htc_list)
        successes = set(res_files) & set(log_files) & htc_set
        # failures are than the difference between all htc's and the successes
        failures = list(htc_set.difference(successes))

        # now we have list with failure cases, create a htc_dict_fail
        htc_dict_fail = dict()
        for k in failures:
            key = k + '.htc'
            htc_dict_fail[key] = copy.copy(htc_dict[key])

        # length will be zero if there are no failures
        return htc_dict_fail


class dynprop:
    """
    Dynamic Properties
    ==================

    Determine the dynamic properties, such as amplitude and damping, of a
    certain time domain signal.

    For this end, the Wafo toolbox is used.
    """

    def __init__(self, debug=False):
        self.debug = debug

    def amplitudes(self, signal, h=1.):
        """
        Amplitudes
        ==========

        Determine the amplitudes of the given signal.

        Parameters
        ----------
        signal : ndarray
            first column time, second column data
        h : number, optional
            wave height treshold, values below h to be ignored. For free yaw,
            set to h=1. (in degrees).

        Returns
        -------
        sig_amp : ndarray
            amplitudes, equivalent to sig_cp.amplitudes()

        Available Objects
        -----------------
        sig_cp : object
            wafo cycle pairs object

        sig_cp.data : ndarray
            wave maxima

        sig_cp.args : ndarray
            wave minima

        sig_cp.amplitudes() : ndarray
            amplitudes

        """
        # convert the array (first column is assumed to be time) to a wafo
        # timeseries object. The turning_point method only works with one data
        # channel
        sig_ts = wafo.objects.mat2timeseries(signal)

        # only continue if there are any tunring points found
        if len(sig_ts.data) > 2:

            # the turning_points method in wafo.objects
            # set h to appropriate value if only high amplitude cycle count
            sig_tp = sig_ts.turning_points(h=h, wavetype=None)

            # ignore waves which are smaller than 1/20 of the channels range
            # h = abs(signal[:,1].max() - signal[:,1].min())/50.

            if len(sig_tp.data) > 2:
                # cycle pairs, h=0 means no filtering of cycles with a certain
                # height
                self.sig_cp = sig_tp.cycle_pairs(h=h, kind='min2max')

                # amplitudes of the rainflow cycles
                return self.sig_cp.amplitudes()
            else:
                return np.array([])
        else:
            return np.array([])


    def damping(self, signal, h=1.):
        """
        Damping
        =======

        Determine input signal damping, based on the rainflow counted
        amplitudes Using the

        Parameters
        ----------

        signal : ndarray
            first column time, second column data

        h : number, optional
            wave height treshold, values below h to be ignored. For free yaw,
            set to h=1. (in degrees).

        Returns
        -------

        sig_damp_ln : ndarray
            logarithmic damping

        Available Objects
        -----------------

        sig_cp : object
            wafo cycle pairs object

        sig_cp.data : ndarray
            wave maxima

        sig_cp.args : ndarray
            wave minima

        sig_cp.amplitudes() : ndarray
            amplitudes
        """

        # convert the array (first column is assumed to be time) to a wafo
        # timeseries object. The turning_point method only works with one data
        # channel
        sig_ts = wafo.objects.mat2timeseries(signal)

        # the turning_points method in wafo.objects
        # set h to appropriate value if only high amplitude cycle count
        sig_tp = sig_ts.turning_points(h=h, wavetype=None)

        # if there too few turning points, we can't do anything sensible
        if len(sig_tp.data) > 3:
            # ignore waves which are smaller than 1/20 of the channels range
#            h = abs(signal[:,1].max() - signal[:,1].min())/50.

            # cycle pairs, h=0 means no filtering of cycles with certain height
            self.sig_cp = sig_tp.cycle_pairs(h=h, kind='min2max')

            # amplitudes of the rainflow cycles
            sig_amp = self.sig_cp.amplitudes()

            # take a range of peaks and determine the damping ratio
            # definitions of logarithmic damping and damping ratio:
            # http://en.wikipedia.org/wiki/Logarithmic_decrement
            # all the local maxima: self.sig_cp.data (using kind='min2max')
            # all the local minima: self.sig_cp.args (using kind='min2max')

            # for the damping, convert all peaks to a positive numbers by adding
            # the minimum value to the entire chain. In this way, all peaks will
            # be lifted above the zero axes.
    #        tmp_damp =

            # if the peaks are all < 0, make them positive, otherwise the
            # damping will be negative
            if self.sig_cp.data.__lt__(0.).all():
                maxs = self.sig_cp.data.__abs__()
                # now the order in which the peaks occure have to be reversed,
                # in order not to have negative damping when damping is pos
                sig_damp_ln = np.log( maxs[1:len(maxs)] / maxs[0:-1])
            # all other cases, should be limited to all positive though
            else:
                maxs = self.sig_cp.data
                sig_damp_ln = np.log(maxs[0:-1] / maxs[1:len(maxs)])
             # TODO: implement scenario where the peaks change sign

            # Other approach, just take the amplitudes! However, this does not
            # match the definition of the logarithmic damping!
            # sig_dampa_lna = np.log(sig_rf[0:-1] / sig_rf[1:len(sig_rf)])

            if self.debug:
                print 'peaks0', maxs[0:-1]
                print 'peaks1', maxs[1:len(maxs)]

            # the damping ratio
            # sig_damp_ratio=1./np.sqrt(1. + np.power(np.pi*2./sig_damp_ln,2.))
        else:
            sig_damp_ln = np.array([0])
            sig_amp = np.array([0])

        return sig_damp_ln, sig_amp

if __name__ == '__main__':

#    # check the log files
#    logs = ErrorLogs()
#    logs.PathToLogs = 'path/to/logs/'
#    logs.check()

    ## convert tabspaced document to fixed spaced document
    #r = ConvertPC()
    #r.readPath = 'C:\HAWC2\models\upwind_sweep\data\hawc_st_new3.NRL'
    #r.writePath = 'C:\HAWC2\models\upwind_sweep\data\NREL_new3_dave_v01.st'
    #r.spacing = 13
    #r.read_pc_file()
    #r.write_pc_file()

#    # reading a HAWC2 binary result file
#
#    # yaw angle is channeli 9
#    chani_yaw = 9
#
#    # this file has no damping, constant amplitude
#    print '====================undamped signal'
#    file_path = '/home/dave/HAWC2_results/3e_ewec/results/steady_free_z/'
#    file_name='s1_steady_free_z_yawfreenoda_c0_y-10_ab00_12.5ms_sh_0.2tow_sh_0'
#    sig = LoadResults(file_path, file_name)
#    sig2 = sig.sig[:,[0,chani_yaw]]
#    dp = dynprop(debug=True)
#    rf, damp = dp.damping(sig2)
#    print 'rainflow spotted cycles:', len(rf)
#    print 'amplitudes', rf
#    print 'log damping', damp
#    plt.plot(sig.sig[:,0],sig.sig[:,chani_yaw])
#    plt.show()
#
#    # plot signal
#    #sig, ch_details, chan, title
#    p = PlotSignal(q.sig, q.ch_details, 1, file_name)
#    p.plot2D()
#
#    # statistics
#    sig_stats = SignalStatistics(q.sig)
#    print sig_stats[:,0]

    ## Fatigue calculations
    #channels = [20, 21, 22]
    #p = 'c:\\Temp\\res_sweep\\'
    #cases = [p+'18ms_ti0.18_s11_tshad_a2b2.dat', \
    #p+'18ms_ti0.18_s11_tshad_a2b1.dat']
    #hours = [1111, 2222]
    #windchan = 11
    #f = Fatigue(channels, cases, hours, windchan)
    #f.bin_path = 'c:/HAWC2/development/fatigue/'
    #f.results_path = 'c:/HAWC2/development/fatigue/results/'
    #f.execute()

    ## parsing
#    htcfile = 'bench_V_12_noSweep_downwind.htc'
#    target = '/home/dave/PhD/Projects/Hawc2Models/upwind_sweep/htc_sweep/'

#    htcfile = 'master2_0_v57_1.htc'
#    target = '/home/dave/PhD/Projects/Hawc2Models/3E_20100708/'
#    htc(target, htcfile)

    ## Master file + pbs script
#    master = MasterFile()
#    # empty input means the default configuration
#    htc_dict = master.createcase([])
#    pbs = PBS([htc_dict])

    # ====================================================================
    ## linspace_around TESTS
    #import pylab as plt
    #num = 15
    #start = 0
    #stop = 0.55
    #points = np.array([0.02, 0.18])
    #tt = 0
    #for k in range(5, 30):
        ##plt.figure()
        #f1 = linspace_around(start, stop, points, num=k, verbose=False)
        #f2 = np.linspace(start, stop,num=k)
        ##plt.plot(f1)
        ##plt.plot(f2)
        #print (f1-f2).sum()
        #tt += (f1-f2).sum()
    #print '--->', tt
    # ====================================================================

    pass


