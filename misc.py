# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:09:04 2012

Library for general stuff

@author: dave
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from xlrd import open_workbook
import numpy as np
import scipy as sp
from scipy.interpolate import griddata as interp
import wafo
import pylab as plt
import matplotlib as mpl


def CoeffDeter(obs, model):
    """
    Coefficient of determination
    ============================

    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Parameters
    ----------

    obs : ndarray(n) or list
        The observed dataset

    model : ndarray(n), list or scalar
        The fitted dataset

    Returns
    -------

    R2 : float
        The coefficient of determination, varies between 1 for a perfect fit,
        and 0 for the worst possible fit ever

    """

    if type(obs).__name__ == 'list':
        obs = np.array(obs)

    SS_tot = np.sum(np.power( (obs - obs.mean()), 2 ))
    SS_err = np.sum(np.power( (obs - model), 2 ))
    R2 = 1 - (SS_err/SS_tot)

    return R2


def calc_sample_rate(time, rel_error=1e-4):
    """
    the sample rate should be constant throughout the measurement serie
    define the maximum allowable relative error on the local sample rate

    rel_error = 1e-4 # 0.0001 = 0.01%
    """
    deltas = np.diff(time)
    # the sample rate should be constant throughout the measurement serie
    # define the maximum allowable relative error on the local sample rate
    if not (deltas.max() - deltas.min())/deltas.max() <  rel_error:
        print 'Sample rate not constant, max, min values:',
        print '%1.6f, %1.6f' % (1/deltas.max(), 1/deltas.min())
#        raise AssertionError
    return 1/deltas.mean()


def findIntersection(fun1, fun2, x0):
    """
    Find Intersection points of two functions
    =========================================

    Find the intersection between two random callable functions.
    The other alternative is that they are not callable, but are just numpy
    arrays describing the functions.

    Parameters
    ----------

    fun1 : calable
        Function 1, should return a scalar and have one argument

    fun2 : calable
        Function 2, should return a scalar and have one argument

    x0 : float
        Initial guess for sp.optimize.fsolve

    Returns
    -------



    """
    return sp.optimize.fsolve(lambda x : fun1(x) - fun2(x), x0)


# TODO: replace this with some of the pyrain functions
def find0(array, xi=0, yi=1, verbose=False, zerovalue=0.0):
    """
    Find single zero crossing
    =========================

    Find the point where a x-y dataset crosses zero. This method can only
    handle one zero crossing point.

    Parameters
    ----------
    array : ndarray
        should be 2D, with a least 2 columns and 2 rows

    xi : int, default=0
        index of the x values on array[:,xi]

    yi : int, default=1
        index of the y values on array[:,yi]

    zerovalue : float, default=0
        Set tot non zero to find the corresponding crossing.

    verbose : boolean, default=False
        if True intermediate results are printed. Usefull for debugging

    Returns
    -------
    y0 : float
        if no x0=0 exists, the result will be an interpolation between
        the two points around 0.

    y0i : int
        index leading to y0 in the input array. In case y0 was the
        result of an interpolation, the result is the one closest to x0=0

    """

    # Determine the two points where aoa=0 lies in between
    # take all the negative values, the maximum is the one closest to 0
    try:
        neg0i = np.abs(array[array[:,xi].__le__(zerovalue),xi]).argmax()
    # This method will fail if there is no zero crossing (not enough data)
    # in other words: does the given data range span from negative, to zero to
    # positive?
    except ValueError:
        print 'Given data range does not include zero crossing.'
        return 0,0

    # find the points closest to zero, sort on absolute values
    isort = np.argsort(np.abs(array[:,xi]-zerovalue))
    if verbose:
        print array[isort,:]
    # find the points closest to zero on both ends of the axis
    neg0i = isort[0]
    sign = int(np.sign(array[neg0i,xi]))
    # only search for ten points
    for i in xrange(1,20):
        # first time we switch sign, we have it
        if int(np.sign(array[isort[i],xi])) is not sign:
            pos0i = isort[i]
            break

    try:
        pos0i
    except NameError:
        print 'Given data range does not include zero crossing.'
        return 0,0

    # find the value closest to zero on the positive side
#    pos0i = neg0i +1

    if verbose:
        print '0_negi, 0_posi', neg0i, pos0i
        print 'x[neg0i], x[pos0i]', array[neg0i,xi], array[pos0i,xi]

    # check if x=0 is an actual point of the series
    if np.allclose(array[neg0i,xi], 0):
        y0 = array[neg0i,yi]
        if verbose:
            prec = ' 01.08f'
            print 'y0:', format(y0, prec)
            print 'x0:', format(array[neg0i,xi], prec)
    # check if x=0 is an actual point of the series
    elif np.allclose(array[pos0i,xi], 0):
        y0 = array[pos0i,yi]
        if verbose:
            prec = ' 01.08f'
            print 'y0:', format(y0, prec)
            print 'x0:', format(array[pos0i,xi], prec)
    # if not very close to zero, interpollate to find the zero point
    else:
        y1 = array[neg0i,yi]
        y2 = array[pos0i,yi]
        x1 = array[neg0i,xi]
        x2 = array[pos0i,xi]
        y0 = (-x1*(y2-y1)/(x2-x1)) + y1

        if verbose:
            prec = ' 01.08f'
            print 'y0:', format(y0, prec)
            print 'y1, y2', format(y1, prec), format(y2, prec)
            print 'x1, x2', format(x1, prec), format(x2, prec)

    # return the index closest to the value of AoA zero
    if abs(array[neg0i,0]) > abs(array[pos0i,0]):
        y0i = pos0i
    else:
        y0i = neg0i

    return y0, y0i


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


class DictDB(object):
    """
    A dictionary based database class
    =================================

    Each tag corresponds to a row and each value holds another tag holding
    the tables values, or for the current row the column values.

    Each tag should hold a dictionary for which the subtags are the same for
    each row entry. Otherwise you have columns appearing and dissapearing.
    That is not how a database is expected to behave.
    """

    def __init__(self, dict_db):
        """
        """
        # TODO: data checks to see if the dict can qualify as a database
        # in this context

        self.dict_db = dict_db

    def search(self, dict_search):
        """
        Search a dictionary based database
        ==================================

        Searching on based keys having a certain value.

        Parameters
        ----------

        search_dict : dictionary
            Keys are the column names. If the values match the ones in the
            database, the respective row gets selected. Each tag is hence
            a unique row identifier. In case the value is a list (or it will
            be faster if it is a set), all the list entries are considered as
            a go.
        """
        self.dict_sel = dict()

        # browse through all the rows
        for row in self.dict_db:
            # and for each search value, check if the row holds the requested
            # column value
            init = True
            alltrue = True
            for col_search, val_search in dict_search.items():
                # for backwards compatibility, convert val_search to list
                if not type(val_search).__name__ in ['set', 'list']:
                    # conversion to set is more costly than what you gain
                    # by target in set([]) compared to target in []
                    # conclusion: keep it as a list
                    val_search = [val_search]

                # all items should be true
                # if the key doesn't exists, it is not to be considered
                try:
                    if self.dict_db[row][col_search] in val_search:
                        if init or alltrue:
                            alltrue = True
                    else:
                        alltrue = False
                except KeyError:
                    alltrue = False
                init = False
            # all search criteria match, save the row
            if alltrue:
                self.dict_sel[row] = self.dict_db[row]

    # TODO: merge with search into a more general search/select method?
    # shouldn't I be moving to a proper database with queries?
    def search_key(self, dict_search):
        """
        Search for a string in dictionary keys
        ======================================

        Searching based on the key of the dictionaries, not the values

        Parameters
        ----------

        searchdict : dict
            As key the search string, as value the operator: True for inclusive
            and False for exclusive. Operator is AND.

        """

        self.dict_sel = dict()

        # browse through all the rows
        for row in self.dict_db:
            # and see for each row if its name contains the search strings
            init = True
            alltrue = True
            for col_search, inc_exc in dict_search.iteritems():
                # is it inclusive the search string or exclusive?
                if (row.find(col_search) > -1) == inc_exc:
                    if init:
                        alltrue = True
                else:
                    alltrue = False
                    break
                init = False
            # all search criteria matched, save the row
            if alltrue:
                self.dict_sel[row] = self.dict_db[row]


class DictDiff(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values

    Source
    ------

    Basic idea of the magic is based on following stackoverflow question
    http://stackoverflow.com/questions/1165352/
    fast-comparison-between-two-python-dictionary
    """
    def __init__(self, current_dict, past_dict):
        self.current_d = current_dict
        self.past_d    = past_dict
        self.set_current  = set(current_dict.keys())
        self.set_past     = set(past_dict.keys())
        self.intersect    = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect
    def removed(self):
        return self.set_past - self.intersect
    def changed(self):
        #set(o for o in self.intersect if self.past_d[o] != self.current_d[o])
        # which is the  similar (exept for the extension) as below
        olist = []
        for o in self.intersect:
            # if we have a numpy array
            if type(self.past_d[o]).__name__ == 'ndarray':
                if not np.allclose(self.past_d[o], self.current_d[o]):
                    olist.append(o)
            elif self.past_d[o] != self.current_d[o]:
                olist.append(o)
        return set(olist)

    def unchanged(self):
        t=set(o for o in self.intersect if self.past_d[o] == self.current_d[o])
        return t


def check_df_dict(df_dict):
    """
    Verify if the dictionary that needs to be transferred to a Pandas DataFrame
    makes sense

    Returns
    -------

    collens : dict
        Dictionary with df_dict keys as keys, len(df_dict[key]) as column.
        In other words: the length of each column (=rows) of the soon to be df.
    """
    collens = {}
    for col, values in df_dict.iteritems():
        print('%6i : %30s' % (len(values), col), type(values))
        collens[col] = len(values)
    return collens


def df_dict_check_datatypes(df_dict):
    """
    there might be a mix of strings and numbers now, see if we can have
    the same data type throughout a column
    nasty hack: because of the unicode -> string conversion we might not
    overwrite the same key in the dict.
    """
    # FIXME: this approach will result in twice the memory useage though...
    # we can not pop/delete items from a dict while iterating over it
    df_dict2 = {}
    for colkey, col in df_dict.items():
        # if we have a list, convert to string
        if type(col[0]).__name__ == 'list':
            for ii, item in enumerate(col):
                col[ii] = '**'.join(item)
        # if we already have an array (statistics) or a list of numbers
        # do not try to cast into another data type, because downcasting
        # in that case will not raise any exception
        elif type(col[0]).__name__[:3] in ['flo', 'int', 'nda']:
            df_dict2[str(colkey)] = np.array(col)
            continue
        # in case we have unicodes instead of strings, we need to convert
        # to strings otherwise the saved .h5 file will have pickled elements
        try:
            df_dict2[str(colkey)] = np.array(col, dtype=np.int32)
        except OverflowError:
            try:
                df_dict2[str(colkey)] = np.array(col, dtype=np.int64)
            except OverflowError:
                df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
        except ValueError:
            try:
                df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
            except ValueError:
                df_dict2[str(colkey)] = np.array(col, dtype=np.str)
        except TypeError:
            # in all other cases, make sure we have converted them to
            # strings and NOT unicode
            df_dict2[str(colkey)] = np.array(col, dtype=np.str)
        except Exception as e:
            print('failed to convert column %s to single data type' % colkey)
            raise(e)
    return df_dict2


def read_excel(ftarget, sheetname, row_sel=[], col_sel=[], data_fmt='list'):
    """
    Read a MS Excel spreadsheet

    Parameters
    ----------

    ftarget

    sheentame

    row_sel : list, default=[]

    col_sel : list, default=[]

    data_fmt : 'list', 'ndarray'

    Source based on:
    http://stackoverflow.com/questions/3239207/
    how-can-i-open-an-excel-file-in-python
    http://stackoverflow.com/questions/3241039/
    how-do-i-extract-specific-lines-of-data-from-a-huge-excel-sheet-using-python
    """

    book = open_workbook(ftarget, on_demand=True)
    sheet = book.sheet_by_name(sheetname)

    # load the whole worksheet
    if len(row_sel) < 1:
        # load each row as a list of the columns
        if data_fmt == 'list':
            for i in xrange(sheet.nrows):
                rows = [cell.value for cell in sheet.row(i)]
        # load as a numpy array, what if there are text values?
        elif data_fmt == 'ndarray':
            msg = 'Loading complete worksheet only works as a list'
            raise UserWarning, msg

    # load selection of the worksheet
    else:
        # load each row as a list of the columns
        if data_fmt == 'list':
            rows = []
            for rowi in row_sel:
                rows.append([sheet.row(rowi)[coli].value for coli in col_sel])
        # load as a numpy array, what if there are text values?
        elif data_fmt == 'ndarray':
            # initialize the array
            rows = np.ndarray( (len(row_sel),len(col_sel)), order='F')
            ii,jj = 0,0
            # IndeError is thrown if we reach the end
            # note that if we have an index error because of wrong selection
            # you are on your own in finding out what goes wrong
            try:
                for rowi in row_sel:
                    for coli in col_sel:
                        rows[jj,ii] = sheet.row(rowi)[coli].value
                        ii += 1
                    jj += 1
                    ii = 0
            except IndexError:
                # crop array correspondingly
                rows = rows[:jj,:]

    book.unload_sheet(sheetname)
    return rows


def fit_exp(time, data, checkplot=True, method='linear', func=None, C0=0.0):
    """
    Note that all values in data have to be possitive for this method to work!
    """

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def fit_exp_nonlinear(t, y):
        # The model function, f(x, ...). It must take the independent variable
        # as the first argument and the parameters to fit as separate remaining
        # arguments.
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func,t,y)
        A, K, C = opt_parms
        return A, K, C

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    # Linear fit
    if method == 'linear':
#        if data.min() < 0.0:
#            msg = 'Linear exponential fitting only works for positive values'
#            raise ValueError, msg
        A, K = fit_exp_linear(time, data, C=C0)
        fit = model_func(time, A, K, C0)
        C = C0

    # Non-linear Fit
    elif method == 'nonlinear':
        A, K, C = fit_exp_nonlinear(time, data)
        fit = model_func(time, A, K, C)

    if checkplot:
        plt.figure()
        plt.plot(time, data, 'ro', label='data')
        plt.plot(time, fit, 'b', label=method)
        plt.legend(bbox_to_anchor=(0.9, 1.1), ncol=2)
        plt.grid()

    return fit, A, K, C


def curve_fit_exp(time, data, checkplot=True, weights=None):
    """
    This code is based on a StackOverflow question/answer:
    http://stackoverflow.com/questions/3938042/
    fitting-exponential-decay-with-no-initial-guessing

    A*e**(K*t) + C
    """

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def fit_exp_nonlinear(t, y):
        # The model function, f(x, ...). It must take the independent variable
        # as the first argument and the parameters to fit as separate remaining
        # arguments.
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func,t,y)
        A, K, C = opt_parms
        return A, K, C

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    C0 = 0

    ## Actual parameters
    #A0, K0, C0 = 2.5, -4.0, 0.0
    ## Generate some data based on these
    #tmin, tmax = 0, 0.5
    #num = 20
    #t = np.linspace(tmin, tmax, num)
    #y = model_func(t, A0, K0, C0)
    ## Add noise
    #noisy_y = y + 0.5 * (np.random.random(num) - 0.5)

    # Linear fit
    A_lin, K_lin = fit_exp_linear(time, data, C=C0)
    fit_lin = model_func(time, A_lin, K_lin, C0)

    # Non-linear Fit
    A_nonlin, K_nonlin, C = fit_exp_nonlinear(time, data)
    fit_nonlin = model_func(time, A_nonlin, K_nonlin, C)

    # and plot
    if checkplot:
        plt.figure()
        plt.plot(time, data, 'ro', label='data')
        plt.plot(time, fit_lin, 'b', label='linear')
        plt.plot(time[::-1], fit_nonlin, 'g', label='nonlinear')
        plt.legend(bbox_to_anchor=(0.9, 1.0), ncol=3)
        plt.grid()

    return


# TODO: this should be a class, so you can more easily accesss all the
# data inbetween, switch on off block detection, use other exponential fit
# techniques etc...
def damping(time, data, checkplot=False, offset_start=100, offset_end=200,
            verbose=False, NFFT=2048):
    """
    Calculate the damping of a vibration signal
    ===========================================

    A nice vibration curve serves as input. Multiple cycles are supported.

    This should be a class, to many things are done here: breaking up in blocs,
    finding all the peaks with different thresholds over each block,
    PSD of each block and derive natural frequency from it (fn)

    Parameters
    ----------

    time : ndarray(n)

    data : ndarray(n)

    checkplot : boolean, default=True

    verbose : boolean, default=False

    offset : int, default=100
        How much data points before/after the peak should be excluded. So it
        refers to the index of the time, not time in itself.

    Returns
    -------

    i_peaks : list of 1D-ndarrays
        Each ndarray holds the indices to the peaks of the block that
        defines one decay test

    i_blocks : list of slices
        Each list entry holds the slice (np.s_) that defines the block

    damp_blocks : list
        Damping per block

    """
    # center data around a zero mean for more convinient processing
    data_mean = data - data.mean()
    # also normalize the data so it is more general applicable
    data_range = data_mean.max() - data_mean.min()
    data_norm = data_mean / data_range

    # Find biggest peaks first. Use this to seperate the repititions in
    # the experiment
    h = 0.3 # 0.42 for the blades
    ip = wafo.misc.findpeaks(data_norm,  n=len(time), min_h=h)
    ip.sort()
    # difference between the peaks. Seems to be that normally the difference
    # between peaks is in the range 100-200 (relates to eigenfrequency).
    # so everytime there is a bigger jump, that marks a new blok
    diff = np.diff(ip)
    imarks = ip[1:][diff.__gt__(300)]
    # also add the first item to it
    imarks = np.append(np.array([ip[0]]), imarks)
    if verbose:
        print 'found %i blocks' % len(imarks)

    # power spectral density, assume constant time step
    sps = int(1.0/(np.diff(time).mean()))

    if checkplot:
        plt.figure()
        plt.subplot(211)
        plt.plot(time, data_norm, 'b')
        plt.plot(time[ip], data_norm[ip], 'go')
        # and mark the regions
        for k in imarks:
            plt.vlines(time[k], -0.7, 0.7, color='y')
        plt.grid()

        # also add the PSD for the whole signal
        plt.subplot(212)
        Pxx, freqs = mpl.mlab.psd(data_mean, NFFT=NFFT, Fs=sps)
        # derive eigenfrequency from this block only, but convert to log scale,
        # that will make it easier to identify the first peak
        Pxx_log = 10.*np.log10(Pxx)
        # find all the peaks on the PSD
        ifns = wafo.misc.findpeaks(Pxx_log, n=len(data_mean), min_h=0)
        # already sorted according to significant wave height
        fn5 = freqs[ifns[0:5]]
        ifn = ifns[0]
        plt.title('frequency spectrum')
        plt.plot(freqs, Pxx, 'r')
        plt.plot(fn5[0], Pxx[ifn], 'bs')
        plt.annotate(r'$%1.1f Hz$' % fn5[0],
            xy=(fn5[0], Pxx[ifn]), xycoords='data',
            xytext=(+5, -15), textcoords='offset points', fontsize=11,
            arrowprops=dict(arrowstyle="->"))
        plt.yscale('log')
        plt.xlim([0, 100])
        plt.grid(True)

    # keep track of all peaks in a list
    i_peaks, i_bloks, damp_blocks, fit_blocks, psd_blocks = [], [], [], [], []

    # for each block, grab all the peaks
    for k in xrange(len(imarks)):
        # on the last block, we need to guess where it ends...
        try:
            istart = imarks[k]
            iend = imarks[k+1]
        except IndexError:
            istart = imarks[k]
            iend = len(time)

        # if istart is lower than the offset, don't make it rewind
        # this way is much faster compared to min(istart, offset_start)
        if istart > offset_start:
            slice_block = np.s_[istart-offset_start:iend-offset_end]
        else:
            slice_block = np.s_[istart:iend-offset_end]
        time_block = time[slice_block]
        data_block = data[slice_block]

        # only consider this block if it is long enough. Some ending blocks
        # are short and those could crash wafo.misc.findpeaks(Pxx_log)
        if len(time_block) < 3000 or time_block[-1]-time_block[0]<2.5:
            if verbose:
                print "too short, block %i ignored" % k
            continue

        # and once more, check that data is centered around 0. The mean level
        # might vary between different blocks
        data_block -= data_block.mean()
        normalize = 1.0 / (data_block.max() - data_block.min())
        # and normalize again, but this time only the block
        data_block *= normalize

        # all the high peaks for the block
        h = 0.52
        ip1 = wafo.misc.findpeaks(data_block, n=len(time_block), min_h=h)
        ip1.sort()
        # and all the intermediate peaks
        h = 0.24
        ip2 = wafo.misc.findpeaks(data_block, n=len(time_block), min_h=h)
        ip2.sort()
        # merge the two series
        overlap2 = (ip2-ip1[-1]).__eq__(0).argmax()
        # if there is an overlapping peak, exclude it
        if ip1[-1] == ip2[overlap2]:
            ip = np.append(ip1, ip2[overlap2+1:])
        else:
            ip = np.append(ip1, ip2[overlap2:])
        # and now all the peaks
        h = 0.02
        ip3 = wafo.misc.findpeaks(data_block, n=len(time_block), min_h=h)
        ip3.sort()
        # merge the two series
        overlap3 = (ip3-ip[-1]).__eq__(0).argmax()
        # do not include the overlapping peak again
        ip = np.append(ip, ip3[overlap3+1:])

        # undo all the normalizing stuff, we selected the peaks so its ok
        data_block *= 1.0/normalize

        # FIXME: sometimes there is a missing peak at overlap2, why??

        if checkplot:
            plt.figure()
            plt.subplot(311)
            plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95,
                                 top=0.95, wspace=0.3, hspace=0.35)
            plt.title('time plot block %i' % k)
            plt.plot(time_block, data_block, 'r')
            plt.plot(time_block[ip], data_block[ip], 'go')
            plt.axvline(x=time_block[ip2[overlap2]])
            plt.axvline(x=time_block[ip3[overlap3]])
            plt.grid(True)

        # keep the indices for the peaks, but remember we only looked for
        # them within a sliced block, so offset for that
        i_peaks.append(ip + istart - offset_start)
        i_bloks.append(slice_block)

        # get the damping, use exponential fitting and alternative method
        fit,A,K,C = fit_exp(time_block[ip], data_block[ip], checkplot=False,
                            method='linear', func=None, C0=0.0)
        fit_blocks.append(fit)

        # add the fitted exponential damping model
        if checkplot:
            plt.plot(time_block[ip], fit, 'b-')


        # don't use the normalized values, energy will be very low
        # nnfts = kwargs.get('nnfts', [16384, 8192, 4096, 2048])
        Pxx, freqs = mpl.mlab.psd(data[slice_block], NFFT=NFFT, Fs=sps)
        psd_blocks.append([Pxx, freqs])

        if checkplot:
            plt.subplot(312)
            plt.title('frequency spectrum block %i' % k)
            plt.plot(freqs, Pxx, 'r')

        # derive eigenfrequency from this block only, but convert to log scale,
        # that will make it easier to identify the first peak
        Pxx_log = 10.*np.log10(Pxx)
        # find all the peaks on the PSD
        ifns = wafo.misc.findpeaks(Pxx_log, n=len(Pxx_log), min_h=0)
        # already sorted according to significant wave height
        fn5 = freqs[ifns[0:5]]
        ifn = ifns[0]

        if checkplot:
            plt.plot(fn5[0], Pxx[ifn], 'bs')
            plt.annotate(r'$%1.1f Hz$' % fn5[0],
                xy=(fn5[0], Pxx[ifn]), xycoords='data',
                xytext=(+5, -15), textcoords='offset points', fontsize=11,
                arrowprops=dict(arrowstyle="->"))
            plt.yscale('log')
            plt.xlim([0, 100])
            plt.grid(True)

        # calculate damping based on exponential fit
        zeta_fit = -K/fn5[0]
        # local damping per cycle
        zeta_loc = np.log(data_block[ip][:-1] / data_block[ip][1:])
        damp_blocks.append(zeta_fit)

        # optionally plot the damping per vibration cycle
        if checkplot:
            plt.subplot(313)
            plt.title('damping per peak')
            plt.plot(zeta_loc, 'gs--')
            plt.grid()


        if verbose:
            replace = (k, fn5[0], Pxx[ifn], zeta_fit)
            print 'block; %2i; fn; %7.3f; Pxx; %6.2e; damp; %6.2e' % replace

        # sort the Pxx peaks on Freq. of occurence instead of wave height
        ifns.sort()
        # and that should be in either of the first 5 peaks. Note that since
        # h=0 was used on peak detection, small peaks are not ingored...
        if ifn not in ifns[:5].tolist():
            #print ifn, ifns[:5]
            msg = 'Eigenfreq: PSD max is not one of the first 5 freq. peaks'
            raise ValueError, msg

    return i_peaks, i_bloks, damp_blocks, fit_blocks, fn5, psd_blocks


def _linear_distr_blade(blade, nr_points=None):
    """
    Interpolate the blade.dat data onto linearly distributed radial
    positions
    """
    if nr_points is None:
        nr_points = blade.shape[0]
    # make a linear distribution of radial positions
    radius = np.linspace(blade[0,0], blade[-1,0], num=nr_points)
    blade_new = sp.zeros((nr_points, blade.shape[1]))
    blade_new[:,0] = radius
    # and interpolate all points from the hawtopt result on a linear grid
    for k in range(1,blade.shape[1]):
        blade_new[:,k] = interp(blade[:,0], blade[:,k], radius)

    return blade_new


if __name__ == '__main__':

    fpath = 'data/raw/blade_contour/'
    fname = '2012-09-26 Contour meting Verelst Test flex B1 LE lijn 0 gr.xls'
    sheetname = 'DMM Scan'
    #row_sel = range(39,200)
    #col_sel = [4,5]
    #rows = read_excel(fpath+fname, sheetname, row_sel=row_sel,
               #col_sel=col_sel, data_fmt='ndarray')
