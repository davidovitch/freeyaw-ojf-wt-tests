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
import array
import numpy as np
import os
import logging
import copy

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

import wafo

import misc

PRECISION = np.float64

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
#            canvas.close()
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
#        canvas.close()
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
#            canvas.close()
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


def ReadEigenBody(file_path, file_name, debug=False, nrmodes=1000):
    """
    Read HAWC2 body eigenalysis result file
    =======================================

    Parameters
    ----------

    file_path : str

    file_name : str



    Returns
    -------

    results : dict{body : ndarray(3,1)}
        Dictionary with body name as key and an ndarray(3,1) holding Fd, Fn
        [Hz] and the logarithmic damping decrement [%]

    results2 : dict{body : dict{Fn : [Fd, damp]}  }
        Dictionary with the body name as keys and another dictionary holding
        the eigenfrequency and damping information. The latter has the
        natural eigenfrequncy as key (hence all duplicates are ignored) with
        the damped eigenfrequency and logarithmic damping decrement as values.

    """

    #Body data for body number : 3 with the name :nacelle
    #Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #Mode nr:  1:   1.45388E-21    1.74896E-03    6.28319E+02
    FILE = open(file_path + file_name)
    lines = FILE.readlines()
    FILE.close()

    results = dict()
    results2 = dict()
    for line in lines:
        # identify for which body we will read the data
        if line.startswith('Body data for body number'):
            body = line.split(':')[2].rstrip().lstrip()
            # remove any annoying characters
            body = body.replace('\n','').replace('\r','')
        # identify mode number and read the eigenfrequencies
        elif line.startswith('Mode nr:'):
            # stop if we have found a certain amount of
            if results.has_key(body) and len(results[body]) > nrmodes:
                continue

            linelist = line.replace('\n','').replace('\r','').split(':')
            #modenr = linelist[1].rstrip().lstrip()
            eigenmodes = linelist[2].rstrip().lstrip().split('   ')
            if debug: print eigenmodes
            # in case we have more than 3, remove all the empty ones
            # this can happen when there are NaN values
            if not len(eigenmodes) == 3:
                eigenmodes = linelist[2].rstrip().lstrip().split(' ')
                eigmod = []
                for k in eigenmodes:
                    if len(k) > 1:
                        eigmod.append(k)
                #eigenmodes = eigmod
            else:
                eigmod = eigenmodes
            # remove any trailing spaces for each element
            for k in range(len(eigmod)):
                eigmod[k] = eigmod[k].lstrip().rstrip()
            eigmod_arr = np.array(eigmod,dtype=np.float64).reshape(3,1)
            if debug: print eigmod_arr
            if results.has_key(body):
                results[body] = np.append(results[body],eigmod_arr,axis=1)
            else:
                results[body] = eigmod_arr

            # or alternatively, save in a dict first so we ignore all the
            # duplicates
            #if results2.has_key(body):
                #results2[body][eigmod[1]] = [eigmod[0], eigmod[2]]
            #else:
                #results2[body] = {eigmod[1] : [eigmod[0], eigmod[2]]}

    return results, results2


def ReadEigenStructure(file_path, file_name, debug=False, max_modes=500):
    """
    Read HAWC2 structure eigenalysis result file
    ============================================

    The file looks as follows:
    #0 Version ID : HAWC2MB 11.3
    #1 ___________________________________________________________________
    #2 Structure eigenanalysis output
    #3 ___________________________________________________________________
    #4 Time : 13:46:59
    #5 Date : 28:11.2012
    #6 ___________________________________________________________________
    #7 Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #8 Mode nr:  1:   3.58673E+00    3.58688E+00    5.81231E+00
    #...
    #302  Mode nr:294:   0.00000E+00    6.72419E+09    6.28319E+02

    Parameters
    ----------

    file_path : str

    file_name : str

    debug : boolean, default=False

    max_modes : int
        Stop evaluating the result after max_modes number of modes have been
        identified

    Returns
    -------

    modes_arr : ndarray(3,n)
        An ndarray(3,n) holding Fd, Fn [Hz] and the logarithmic damping
        decrement [%] for n different structural eigenmodes

    """

    #0 Version ID : HAWC2MB 11.3
    #1 ___________________________________________________________________
    #2 Structure eigenanalysis output
    #3 ___________________________________________________________________
    #4 Time : 13:46:59
    #5 Date : 28:11.2012
    #6 ___________________________________________________________________
    #7 Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #8 Mode nr:  1:   3.58673E+00    3.58688E+00    5.81231E+00
    #  Mode nr:294:   0.00000E+00    6.72419E+09    6.28319E+02

    FILE = open(file_path + file_name)
    lines = FILE.readlines()
    FILE.close()

    header_lines = 8

    # we now the number of modes by having the number of lines
    nrofmodes = len(lines) - header_lines

    modes_arr = np.ndarray((3,nrofmodes))

    for i, line in enumerate(lines):
        if i > max_modes:
            # cut off the unused rest
            modes_arr = modes_arr[:,:i]
            break

        # ignore the header
        if i < header_lines:
            continue

        # split up mode nr from the rest
        parts = line.split(':')
        #modenr = int(parts[1])
        # get fd, fn and damping, but remove all empty items on the list
        modes_arr[:,i-header_lines]=misc.remove_items(parts[2].split(' '),'')

    return modes_arr


if __name__ == '__main__':

    pass
