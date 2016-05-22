# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:22:50 2011

@author: dave
"""

# standard library
import math
import pickle

# external libraries
import numpy as np
import scipy

import matplotlib as mpl
# set the backend
#mpl.use('GTKCairo')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import matplotlib.font_manager as mpl_font
from matplotlib.ticker import FormatStrFormatter
#from matplotlib import tight_layout as tight_layout
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as plt

import wafo

# custom libraries
import HawcPy
import Simulations as sim

# TODO: implement these tricks?

"""
How to remove the frame legend? This tip makes it transparant. Will not work
with EPS though... so set frame colour to something else? or line width to 0?
    leg = ax.legend(...)
    leg.get_frame().set_alpha(0.5)
"""

class TexTemplate:
    """
    Since psfrag is not working with matplotlib due to a bug for the moment,
    make the figures in the correct size for the thesis right away.
    Measures are in cm
    """
    pagewidth = 12.968715 # cm, or more accurately: \textwidth
    size_x_perfig = pagewidth
    size_y_perfig = 12.
    pageheight = 20.939738 # cm, ore more accurately: \textheight

# convert strings to raw format
def raw_string(s):
    """
    Source taken from
    http://code.activestate.com/recipes/
    65211-convert-a-string-into-a-raw-string/
    """
    if isinstance(s, str):
        s = s.encode('string-escape')
    elif isinstance(s, unicode):
        s = s.encode('unicode-escape')
    return s

class template:
    def __init__(**kwargs):

        # load a HAWC2 result file
        file_path = '/some/path/'
        file_name = 'aero_test_stbl40'
        sig = HawcPy.LoadResults(file_path, file_name)
        stats = HawcPy.SignalStatisticsNew(sig.sig, start=0)

        fontsize = kwargs.get('fontsize', 'medium')
        figsize_x = kwargs.get('figsize_x', 6)
        figsize_y = kwargs.get('figsize_y', 4)
        title = kwargs.get('title', '')
        dpi = kwargs.get('dpi', 200)

        wsleft = kwargs.get('wsleft', 0.15)
        wsbottom = kwargs.get('wsbottom', 0.1)
        wsright = kwargs.get('wsright', 0.90)
        wstop = kwargs.get('wstop', 0.90)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)

        textbox = kwargs.get('textbox', None)
        xlabel = kwargs.get('xlabel', 'x [m]')
        ylabel = kwargs.get('ylabel', 'y [m]')

        xlim = kwargs.get('xlim', [0.0,  0.55])
        ylim = kwargs.get('ylim', [-0.060, 0.010])

        figpath = file_path


        # ====================================================================
        # configure in a more practical way?
        # ====================================================================
        mpl.rcParams['figure.figsize'] = (11.0,7.0)
        mpl.rcParams['font.size'] = 12.0
        mpl.rcParams['figure.subplot.left'] = 0.15
        mpl.rcParams['figure.subplot.right'] = 0.85
        mpl.rcParams['figure.subplot.bottom'] = 0.15
        mpl.rcParams['figure.facecolor'] = 'white'
        mpl.rcParams['lines.linewidth'] = 1
#        mpl.rcParams['axes.color_cycle'] = ['blue','magenta','green','cyan']
#        mpl.rcParams['axes.color_cycle'] = ['#0000CC']
        mpl.rcParams['axes.color_cycle'] = ['#0000AA','#AA0000','#00AA00']
        cycle = mpl.rcParams['axes.color_cycle']


        # ====================================================================
        # FONT PROPERTIES
        # ====================================================================
        font_medium = mpl_font.FontProperties()
        font_medium.set_size('medium')


        # ====================================================================
        # scale the figure to real size
        # ====================================================================
        xlength = (xlim[1] - xlim[0])*100.
        ylength = (ylim[1] - ylim[0])*100.
        figsize_x = xlength/oneinch
        figsize_y = ylength/oneinch


        # ====================================================================
        # global figure: initialize and configure
        # ====================================================================
        fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)
        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                                top=wstop, wspace=wspace, hspace=hspace)


        # ====================================================================
        # configure a subplot
        # ====================================================================
        # add_subplot(nr rows nr cols plot_number)
        ax1 = fig.add_subplot(111)


        # ====================================================================
        # PLOTTING
        # ====================================================================
        # title for the subplot
        title = 'sweep blade planform\n'
        ax1.set_title(title, fontproperties=font_medium)
        ax1.set_title(title, size=12)
        # plotting on the left y-axis
        ax1.plot(radius, stats[0,2,aoas], 'k-', label='extremely stiff')
        ax1.plot(radius, stats[0,2,aoas], 'k-', label='flexible')
        # plot on the right y-axis
        ax2 = ax1.twinx()

        # share axis
        ax3 = fig.add_subplot(2, 3, 5, sharex=ax2, sharey=ax2)

        # do not plot the marker at every data point
        ax1.plot(x, y, 'ko-', markevery=5, markerfacecolor='w')

        # ====================================================================
        # CONTOUR PLOTS, COLOR BARS
        # ====================================================================
        plt.figure()
        # Black contour lines, dashed lines negative by defaults
        CS = plt.contour(X, Y, Z, nr_contours, colors='k')
        CS = plt.contour(X, Y, Z, contour_sequence, colors='k')
        # if you want to draw color fields instead of contour lines
        cmap = mpl.cm.get_cmap("Reds")
        cmap = mpl.cm.rainbow
        CS = plt.contourf(X, Y, Z, nr_contours, colors=cmap)
        # if you want to place the labels yourself. It will find the closest
        # contour line, and place a label there.
        lablocs = [(10.5,1100), (12,1150), (12,1000), (12,850), (12, 700),
                   (  12, 580), (12, 420), (12, 300), (16,200)]
        ax1.clabel(CS, fontsize=7*scale, inline=1, fmt='%1.0f', manual=lablocs)
        # set the labels
        plt.clabel(CS, fontsize=9, inline=1)
        # color bar with the scales
        cbar = fig.colorbar(mpl.cm.rainbow)
        cbar.set_label('color bar label')
        plt.title('Single color - negative contours dashed')

        # ====================================================================
        # COLOR GRADIENTS/MAPS
        # ====================================================================
        # overview of some of the predifined color maps
        # http://dept.astro.lsa.umich.edu/~msshin/science/code/matplotlib_cm/
        # http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps

        # manually define some gradients based on RGB tuples
        xx = np.linspace(0,1,len(cases_sorted))
        col = []
        for ii in xrange(len(cases_sorted)):
            col.append((1*xx[ii],0,1*(1-xx[ii])))

        # color can also be a grey value between 0 (white) and 1 (black)
        col = np.linspace(0.1, 1.0, 13)

        # define the number of positions you want to have the color for
        N = 10
        # select a color map
        cmap = mpl.cm.get_cmap('jet', 10)
        # convert to array
        cmap_arr = cmap(np.arange(N))
        # and now you have each color as an RGB tuple as
        for i in cmap_arr:
            coltuple = tuple(i[0:3])

        # ====================================================================
        # CREATE COLORED AREA
        # ====================================================================
        verts = [(a,0)] + list(zip(ix,iy)) + [(b,0)]
        # for each patch, recreate the poly, it might fuck up when you create
        # the patch during one subplot and reuse for other subplots
        # it seems it is bound to the
        # http://matplotlib.org/api/artist_api.html#matplotlib.patches.Polygon
        poly = mpl.patches.Polygon(verts, facecolor='0.8', edgecolor='k',
                                   hatch='/')
        # where hatch can be:
        # [ ‘/’ | ‘\’ | ‘|’ | ‘-‘ | ‘+’ | ‘x’ | ‘o’ | ‘O’ | ‘.’ | ‘*’ ]
        ax.add_patch(poly)

        # ====================================================================
        # PLOT MARKERS
        # ====================================================================
        #  7 , 4 , 5 , 6 , 'o' , 'D' , 'h' , 'H' , '_' , '' , 'None' , ' '
        # None , '8' , 'p' , ',' , '+' , '.' , 's' , '*' , 'd' , 3 , 0 , 1
        # 2 , '1' , '3' , '4' , '2' , 'v' , '<' , '>' , '^' , ',' , 'x'
        # '$...$' , tuple , Nx2 array
        ax1.plot(radius, stats[0,2,aoas], maker='$tex$', markersize=10.2)

        # ====================================================================
        # placing a plot inside a plot (like zooming in on a part)
        # ====================================================================
        # get the drawing box of a given axes
        # -----------------------------------
#        renderer = tight_layout.get_renderer(plot.fig)
#        print ax2.get_tightbbox(renderer)
        # in order to place the nex axes inside following figure, first
        # determine the ax2 bounding box
        # points: a 2x2 numpy array of the form [[x0, y0], [x1, y1]]
        ax2box = ax2.get_window_extent().get_points()
        # seems to be expressed in pixels so convert to relative coordinates
#        print ax2box
        # figure size in pixels
        figsize_x_pix = figsize_x*dpi
        figsize_y_pix = figsize_y*dpi
        # ax2 box in relative coordinates
        ax2box[:,0] = ax2box[:,0] / figsize_x_pix
        ax2box[:,1] = ax2box[:,1] / figsize_y_pix
#        print ax2box[0,0], ax2box[1,0], ax2box[0,1], ax2box[1,1]
        # left position new box at 10% of x1
        left   = ax2box[0,0] + ((ax2box[1,0] - ax2box[0,0]) * 0.15)
        bottom = ax2box[0,1] + ((ax2box[1,1] - ax2box[0,1]) * 0.30)  # x2
        width  = (ax2box[1,0] - ax2box[0,0]) * 0.35
        height = (ax2box[1,1] - ax2box[0,1]) * 0.6
#        print [left, bottom, width, height]
        # inset plot.
        # [left, bottom, width, height]
#        ax2a = plot.fig.add_axes([0.42, 0.6, .45, .25])
        ax2a = fig.add_axes([left, bottom, width, height])

        # ====================================================================
        # ADD MORE PLOTS USING SAME AXIS
        # ====================================================================
        # taken from example:
        # http://matplotlib.org/examples/axes_grid/scatter_hist.html
        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(ax1)
        axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax1)
        axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax1)


        # ====================================================================
        # LEGENDS
        # ====================================================================
        leg = ax1.legend(loc='lower left', markerscale=0.75, prop=font_medium,
                         title='SOME TITLE')
        # set legend transparency
        leg.get_frame().set_alpha(0.5)
        # move legend to the forefround, higher numbers are on top
        # Artists with lower zorder values are drawn first.
        leg.set_zorder(1)

        # bbox coordinates are here in percentages
        leg = ax1.legend(bbox_to_anchor=(0.1, 1.2), ncol=3, loc='upper right')
        # when used with the loc='upper right' keyword, you specify which
        # corner should be pinned at the bbx_to_anchor position. In this case
        # 0.1, 1.2 is the location of the upper right legend corner

        # legend outside the plot area
        # http://stackoverflow.com/questions/4700614/
        # how-to-put-the-legend-out-of-the-plot
        # http://matplotlib.org/users/legend_guide.html#plotting-guide-legend

        # ====================================================================
        # LEGENDS ALIGNMENT, or TEXT ALIGNMENT IN GENERAL
        # ====================================================================
        # http://stackoverflow.com/questions/7936034/
        # text-alignment-in-a-matplotlib-legend

        # get the width of your widest label, since every label will need
        # to shift by this amount after we align to the right
        shift = max([t.get_window_extent().width for t in legend.get_texts()])
        for t in legend.get_texts():
            t.set_ha('right') # ha is alias for horizontalalignment
            t.set_position((shift,0))

        # ====================================================================
        # put both plot labels in one legend
        # ====================================================================
        # plot on the 2nd axis so the legend is always on top!
        lines = ax1.lines + ax2.lines
        labels = [l.get_label() for l in lines]
        leg = ax2.legend(lines, labels, loc='best')
        # or alternatively ask for the plotted objects and their labels
        #lines, labels = ax1.get_legend_handles_labels()
        #lines2, labels2 = ax2.get_legend_handles_labels()


        # ====================================================================
        # AXIS LABELS
        # ====================================================================
        # formatting of the axis scale
        majorFormatter = FormatStrFormatter('%1.1e')
        ax1.yaxis.set_major_formatter(majorFormatter)
        # set a label
        ax1.set_xlabel(xlabel, size=fontsize) # or: fontproperties=font_medium
        ax1.set_ylabel(ylabel, size=fontsize) # or: fontproperties=font_medium
        # control the ticks
        ax1.xaxis.set_ticks( np.arange(xlim[0], xlim[1], 0.005).tolist() )
        ax1.yaxis.set_ticks( np.arange(ylim[0], ylim[1], 0.005).tolist() )
        # limits
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        # clear all the labels
        ax1.set_xticklabels([])
        # alternatively, make the labels invisible
        mpl.artist.setp(ax1.get_xticklabels() + ax1.get_yticklabels(),
                        visible=False)

        # ====================================================================
        # REMOVE AXIS AND FRAME
        # ====================================================================
        ax1.set_frame_on(False)
        ax1.axes.get_yaxis().set_visible(False)

        # ====================================================================
        # CONTROL AXIS TICKS
        # ====================================================================
        # http://matplotlib.org/api/ticker_api.html
        # a locator
        locator = matplotlib.ticker.LinearLocator(numticks=None, presets=None)
        ax.xaxis.set_major_locator( xmajorLocator )
        # xminorLocator defaults to None
        ax.xaxis.set_minor_locator( xminorLocator )

        # which is completely the same as doing
        xticks = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], num=11)
        ax1.xaxis.set_ticks(xticks.tolist())

        # ====================================================================
        # MATCH TICKS LEFT AND RIGHT
        # ====================================================================

        yticks1 = len(ax1.get_yticks())
        ylim2 = ax2.get_ylim()
        yticks2 = np.linspace(ylim2[0], ylim2[1], num=yticks1).tolist()
        ax2.yaxis.set_ticks(yticks2)

        # ====================================================================
        # HORIZONTAL AND VERTICAL LINES
        # ====================================================================
        ax1.axvline(x=ver_line, linewidth=1, color='k',\
                linestyle='--', aa=False)
        ax1.axhline(y=hor_line_left, linewidth=1, color='k',\
                linestyle='-', aa=False)
        ax1.hlines(y, xmin, xmax, colors='', linestyle='', label='')

        # ====================================================================
        # TEXT BOX
        # ====================================================================
        #  bbox is a dictionary of matplotlib.patches.Rectangle properties:
        # matplotlib.org/1.2.0/api/artist_api.html#matplotlib.patches.Rectangle
        bbox = dict(boxstyle="round", alpha=0.5, edgecolor=(1., 0.5, 0.5),
                    facecolor=(1., 0.8, 0.8),)
        ax1.text(0, 0, textbox, fontsize=12, verticalalignment='bottom',
                 horizontalalignment='center', bbox=bbox)


        # http://www.loria.fr/~rougier/teaching/matplotlib/#annotate-some-points
        ax1.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(fn, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        # ====================================================================
        # auto formatting labels
        # ====================================================================
        # formatting of the axis labels, puts it under an angle if they are
        # too long
        fig.autofmt_xdate()


        # ====================================================================
        # global title of the figure
        # ====================================================================
        fig.suptitle(fig_title, size='x-large') # or: fontproperties=font


        # ====================================================================
        # save figure
        # ====================================================================
        ax1.grid(True)
        print 'saving: ' + figpath
        fig.savefig(figpath + '.png')
        fig.savefig(figpath + '_tip.eps')
        fig.clear()

def match_axis_ticks(ax1, ax2, ax1_format=None, ax2_format=None):
    """
    Match ticks of ax2 to ax1
    =========================

    ax1_format: '%1.1f'

    """
    # match the ticks of ax2 to ax1
    yticks1 = len(ax1.get_yticks())
    ylim2 = ax2.get_ylim()
    yticks2 = np.linspace(ylim2[0], ylim2[1], num=yticks1).tolist()
    ax2.yaxis.set_ticks(yticks2)

    # give the tick labels a given precision
    if ax1_format:
        majorFormatter = FormatStrFormatter(ax1_format)
        ax1.yaxis.set_major_formatter(majorFormatter)

    if ax2_format:
        majorFormatter = FormatStrFormatter(ax2_format)
        ax2.yaxis.set_major_formatter(majorFormatter)

    return ax1, ax2

class StatResults:
    """
    Plot Statistics for a given set of Simulations
    ==============================================

    Input is an htc_dict with carefully selected simulations. X and y axis
    correspond to the channels of the results

    """

    def __init__(self, htc_dict, htc_dict_stats, **kwargs):

        self.verbose = kwargs.get('verbose', False)

        self.htc_grid, self.stats_grid = \
            self._grid_1d(htc_dict, htc_dict_stats)

        self._init_plot(**kwargs)


    def _grid_1d(self, htc_dict, htc_dict_stats):
        """
        Organize a collection of htc_dicts on a 1D grid
        ===============================================

        One dimensional because on one axis you can either have a tag or
        a channeli, and the other axis you have a channeli
        """

        # TODO: this should be general and generic function: selecting
        # cases from a database and make them ready to plot.

        # make a 1D array with all the simulations on it, so we can relate
        # the datapoints with an exact simulation
        htc_grid = np.ndarray(shape=(len(htc_dict.keys()),), dtype='<S100')
        # and make it all empty
        htc_grid[:] = ''

        # the stat grid holds all stat parameters and channels. We do this
        # to facilitate plotting several stuff while reusing a certain sort
        # order
        statshape = htc_dict_stats[htc_dict_stats.keys()[0]].shape
        shape = (len(htc_dict.keys()), statshape[1], statshape[2])
        stats_grid = scipy.zeros(shape)

        i = 0
        for k in htc_dict.keys():
            htc_grid[i] = k
            stats_grid[i,:,:] = htc_dict_stats[k][0,:,:]
            i += 1

        if self.verbose:
            print 'htc_grid shape:', htc_grid.shape
            print 'stats_grid shape:', stats_grid.shape

        return htc_grid, stats_grid


    def sort_grid(self, chi, stat_par):
        """
        """

        # sort first on the given axis and channel
        # and ignore all chi and stat_par dimensions
        i_sort = self.stats_grid[:,stat_par,chi].argsort(axis=0)
        self.htc_grid = self.htc_grid[i_sort]
        self.stats_grid = self.stats_grid[i_sort,:,:]


    def _init_plot(self, **kwargs):
        """
        Initialise the plot
        ===================

        Setup the plot, all pre configurations
        """

        fontsize = kwargs.get('fontsize', 'medium')
        figsize_x = kwargs.get('figsize_x', 6)
        figsize_y = kwargs.get('figsize_y', 4)
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
#        cblabel = kwargs.get('cblabel', '')
        title = kwargs.get('title', '')
        dpi = kwargs.get('dpi', 200)
        wsleft = kwargs.get('wsleft', 0.15)
        wsbottom = kwargs.get('wsbottom', 0.1)
        wsright = kwargs.get('wsright', 0.90)
        wstop = kwargs.get('wstop', 0.90)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)

        textbox = kwargs.get('textbox', None)

        self.fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        self.canvas = FigCanvas(self.fig)
        self.fig.set_canvas(self.canvas)

        self.ax1 = self.fig.add_subplot(111)
#        self.ax2 = self.ax1.twinx()

        self.fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                            top=wstop, wspace=wspace, hspace=hspace)

#        ax2.plot((sig[ibeg:iend,xchan]-x_0), sig[ibeg:iend,chan],c,
#                             label=label)



    def plot(self, **kwargs):
        """
        Finish the plot
        ===============

        """

        figpath = kwargs.get('figpath', None)

        xlim = kwargs.get('xlim', None)
        ylim_right = kwargs.get('ylim_right', None)
        ylim_left = kwargs.get('ylim_left', None)

        if xlim:
            self.ax2.set_xlim(xlim)
        if ylim_left:
            self.ax1.set_ylim(ylim_left)
        if ylim_right:
            self.ax2.set_ylim(ylim_right)

        self.ax1.grid(True)
#        self.ax2.grid(True)
#        self.ax1.legend()
#        self.ax2.legend()

        print 'saving: ' + figpath
        self.fig.savefig(figpath + '.png')
        self.fig.savefig(figpath + '.eps')
        self.fig.clear()


class StatDesings:
    """
    Plot Statistics for Different Designs
    =====================================

    Plotting selection is based on the tags in the htc file. The tags are
    recovered from the htc_dict. The values on the x and y axsis are thus
    corresponding to the tags. Use this function to compare statistics of
    different designs.

    Members
    -------
    htc_grid : ndarray

    stats_grid : ndarray

    xkey_values : list

    ykey_values : list


    Parameters
    ---------
    In case the optional 4 arguments are used, __init__ will create the
    grid members.

    htc_dict : dict
        dictionary where each key is the case name and the value the
        a simulation dictionary

    htc_dict_stats : dict
        dictionary where each key is the case name and the value a sig_stats
        array (see HawcPy.SignalStatistics)

    xkey : str
        string specifying a keyword from the simulation dictionary,
        for example [windspeed], which will be the x axis argument

    xkey : list
        List of strings specifying keywords from the simulation dictionary.
        Allows to combine different results from keywords on one axis

    ykey : str
        string specifying a keyword from the simulation dictionary,
        for example [windspeed], which will be the y axis argument

    ykey : list
        List of strings specifying keywords from the simulation dictionary.
        Allows to combine different results from keywords on one axis

    debug : boolean, default=False

    """

    # TODO: x and y axis are switched around, fix this!

    def __init__(self, *args, **kwargs):
        """


        """

        self.debug = kwargs.get('debug', False)
        self.stat_par = kwargs.get('stat_par', 2)
        self.xprec = kwargs.get('xprec', '1.1f')
        self.yprec = kwargs.get('yprec', '1.1f')

        if len(args) == 4:
            self.htc_dict = args[0]
            self.htc_dict_stats = args[1]
            self.xkey = args[2]
            self.ykey = args[3]

            self.htc_grid, self.xkey_values, self.ykey_values \
                = self.case_grid_2d(self.htc_dict, self.xkey, self.ykey)
            self.stats_grid \
                = self.stats_grid_2d(self.htc_grid, self.htc_dict_stats)

#        self.htc_grid, self.stats_grid, self.ykey_values = self.sort_grid('y')

#    def prep_data(self)



    def case_grid_2d(self, htc_dict, xkey, ykey):
        """
        Organize a collection of htc_dicts on a 2D grid
        ===============================================

        Create a 2D map where at each point x,y corresponds to xkey=x and
        ykey=y. The results is an x axis with the x labels (xkey values) and
        y axis with y labels (ykey values). Both the x and y axis labels are
        sorted alphabetically.

        Only supports a unique case for a given (xkey_val,ykey_val) value pair.
        Raises an error if otherwise.

        """

        # TODO: support for multi placing on a multi demnsional grid. Input as
        # a list of keys. The dimension is than the number of keys in the list.

        xkey_values, ykey_values = [], []
        for k in htc_dict.keys():
            # in the first round, determine the data range of xkey and ykey
            # they will form the data grid for 2D color map plotting
            if type(xkey).__name__ == 'str':
                xkey_values.append(htc_dict[k][xkey])
            # for a list of keys
            elif type(xkey).__name__ == 'list':
                sub_xkey_val = ''
                for kk in xkey:
                    value = htc_dict[k][kk]
                    # add the first letter of the tag to the label
                    # formatting of the tag
                    sub_xkey_val += kk[1] + format(value, self.xprec) + '_'
                # remove the last underscore from the current label
                xkey_values.append(sub_xkey_val[:-1])

            if type(ykey).__name__ == 'str':
                ykey_values.append(htc_dict[k][ykey])
            # for a list of keys
            elif type(ykey).__name__ == 'list':
                sub_ykey_val = ''
                for kk in ykey:
                    value = htc_dict[k][kk]
                    # add the first letter of the tag to the label
                    sub_ykey_val += kk[1] + format(value, self.yprec) + '_'
                # and strip the last dash
                ykey_values.append(sub_ykey_val[:-1])

            # TODO: what if xkey/ykey doesn't exist?

        # remove all double entries: each xkey has unique(ykey) occurences
        # this will determine the entries on x and y axis: the x and y labels
        xkey_values = HawcPy.unique(xkey_values)
        ykey_values = HawcPy.unique(ykey_values)

        # sort the keys
        xkey_values.sort()
        ykey_values.sort()

        if self.debug:
            print 'len xkey_values, ykey_values and their product',
            print len(xkey_values), len(ykey_values),
            print len(xkey_values)*len(ykey_values)
            print 'nr of cases in htc_dict', len(htc_dict.keys())
            print 'xkey_values', xkey_values
            print 'ykey_values', ykey_values

        # this means we have a total of xkey_values x ykey_values data points
        try:
            assert len(xkey_values)*len(ykey_values) == len(htc_dict.keys())
        except AssertionError:
            # if this fails, we already have an indication that we still have
            # more than one case per point on htc_grid
            if self.debug:
                print 'htc_dict keys:'
                htc_dict_sorted = htc_dict.keys()
                htc_dict_sorted.sort()
                for k in htc_dict_sorted:
                    print k
            raise UserWarning, 'multiple cases per grid point, refine htc_dict'

        # if using arrays
        #xval_arr = np.array(xkey_values)
        #yval_arr = np.array(ykey_values)

        # now determine for each case the index on the grid.
        # Create a 2D array containing the data keys of htc_dict
        shape = (len(ykey_values),len(xkey_values))
        htc_grid = np.ndarray(shape=shape, dtype='<S100')
        # and make it all empty
        htc_grid[:,:] = ''

        for k in htc_dict.keys():
            # determine for case k the x and y coordinates on the 2D grid
            # when we only have one key
            if type(xkey).__name__ == 'str':
                xval = htc_dict[k][xkey]
            # in case of multiple keys,
            elif type(xkey).__name__ == 'list':
                xval = ''
                for kk in xkey:
                    value = htc_dict[k][kk]
                    # for sorting purposes, if it is zero, format to
                    # the precision of the other values
                    xval += kk[1] + format(value, self.xprec) + '_'
                xval = xval[:-1]
            # when we only have one key
            if type(ykey).__name__ == 'str':
                yval = htc_dict[k][ykey]
            # in case of multiple keys
            elif type(ykey).__name__ == 'list':
                yval = ''
                for kk in ykey:
                    value = htc_dict[k][kk]
                    # for sorting purposes, if it is zero, format to
                    # the precision of the other values
                    yval += kk[1] + format(value, self.yprec) + '_'
                yval = yval[:-1]

            # determine the index of the current case in *keys_arr
            # using numpy array
            #xi = xval_arr.__eq__(xval).argmax()
            #yi = yval_arr.__eq__(yval).argmax()
            # using a list
            xi = xkey_values.index(xval)
            yi = ykey_values.index(yval)

            # make sure case k is unique, raise an error when it is not.
            # if there is an error, make a better htc_dict selection.
            assert len(htc_grid[yi,xi]) == 0
            # and save in the big 2D case array
            htc_grid[yi,xi] = k

        return htc_grid, xkey_values, ykey_values

    def stats_grid_2d(self, htc_grid, htc_dict_stats):
        """
        Data grid for a given statistical parameter and data channel
        ============================================================

        Create a data array corresponding to htc_grid where each point on
        the grid corresponds the defined stat_par and channel index chi.
        """

        # the stat grid holds all stat parameters and channels. We do this
        # to facilitate plotting several stuff while reusing a certain sort
        # order
        statshape = htc_dict_stats[htc_dict_stats.keys()[0]].shape
        shape = (htc_grid.shape[0],htc_grid.shape[1],statshape[1],statshape[2])
        stats_grid = scipy.zeros(shape)

        # we need to cyclye through because the statistics are first contained
        # in a dictionary and not an array or something.
        for x in range(htc_grid.shape[1]):
            for y in range(htc_grid.shape[0]):
                stats_grid[y,x,:,:] = htc_dict_stats[htc_grid[y,x]][0,:,:]

        return stats_grid

    def sort_grid(self, sort_axis, chi, stat_par, col=0):
        """
        Sort stats_grid
        ===============

        In case the htc_grid needs to sorted based on the plotted values
        corresponding to chi. Note that the htc_grid is already sorted
        alphabetically with respect to the x and y axis labels.

        Parameters
        ==========

        sort_axis : str
            indicate on which axis to sort, x or y.

        chi : int
            Sort channel index of the data set

        stat_par : int
            Refers to the statistical parameter of the sig_stats array
            sig_stat = [(0=value,1=index),statistic parameter, channel]
            statistic parameter = max, min, mean, std, range, abs max

        col : int, default=0
            specify which data channel will be used for sorting on the given
            sort_axis

        """

        if sort_axis == 'y':
            # sort first on the given axis and channel
            # and ignore all chi and stat_par dimensions
            i_sort = self.stats_grid[:,col,stat_par,chi].argsort(axis=0)
            self.htc_grid = self.htc_grid[i_sort,:]
            self.stats_grid = self.stats_grid[i_sort,:,:,:]

            # and sort the list accordingly
            ykey_values_sort = []
            for k in i_sort:
                ykey_values_sort.append(self.ykey_values[k])
            self.ykey_values = ykey_values_sort

        elif sort_axis == 'x':
            # sort first on the given axis and channel
            i_sort = self.stats_grid[col,:,stat_par,chi].argsort(axis=0)
            self.htc_grid = self.htc_grid[i_sort,:].__copy__()
            self.stats_grid = self.stats_grid[i_sort,:,:,:].__copy__()

            # and sort the list accordingly
            xkey_values_sort = []
            for k in i_sort:
                xkey_values_sort.append(self.xkey_values[k])
            self.xkey_values = xkey_values_sort

    def plot(self, figpath, chi, stat_par, **kwargs):
        """
        Plot the 2d grid on a colour map
        ================================

        Arguments
        ---------

        figpath : str
            full path for saving the figure

        chi : int
            channel index of the data set

        stat_par : int
            Refers to the statistical parameter of the sig_stats array
            sig_stat = [(0=value,1=index),statistic parameter, channel]
            statistic parameter = max, min, mean, std, range, abs max

        fontsize : str, default=large

        """

        fontsize = kwargs.get('fontsize', 'medium')
        figsize_x = kwargs.get('figsize_x', 6)
        figsize_y = kwargs.get('figsize_y', 4)
        xlabel = kwargs.get('xlabel', self.xkey)
        ylabel = kwargs.get('ylabel', self.ykey)
#        cblabel = kwargs.get('cblabel', '')
        title = kwargs.get('title', '')
        dpi = kwargs.get('dpi', 200)
        wsleft = kwargs.get('wsleft', 0.15)
        wsbottom = kwargs.get('wsbottom', 0.25)
        wsright = kwargs.get('wsright', 0.95)
        wstop = kwargs.get('wstop', 0.90)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)

        textbox = kwargs.get('textbox', None)

        fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)
        # add_subplot(nr rows nr cols plot_number)
        ax1 = fig.add_subplot(111)
        # or add axes by giving coordinates: [left, bottom, width, height]
#        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # countour plot
        CS = ax1.contour(self.stats_grid[:,:,stat_par, chi])

        # TODO: determine what the range of the dataset is, adapt formating
        # label accordingly
        # you need two criteria: range and max or min. The range to estimate
        # how many places behind the comma, the max or min to decide if you
        # need to switch to exponential or not. Raise a warning that when
        # the range is too high and the max or min is also too high, the labels
        # will not tell you anything. For instance when one data value is
        # so low/high with respect to the rest that there is actually nothing
        # to be made from the labels
        data_max = self.stats_grid[:,:,stat_par, chi].max()
        data_min = self.stats_grid[:,:,stat_par, chi].min()
        data_range = data_max - data_min
        # the data accuracy estimator
#        acc_est = data_max / data_range

#        # base it on the max/ran
#        if acc_est

        # very simple criterium
        if data_min > 0.01 and data_max < 999.:
            if data_max < 10:
                format_str = '%1.4f'
            else:
                format_str = '%1.2f'
        else:
            format_str = '%1.2e'

        ax1.clabel(CS, inline=1, fontsize=10, fmt=format_str)

        # color map
#        cmap = matplotlib.cm.get_cmap('jet', 10)
#        image = ax1.imshow(data, cmap=cmap)
#        #interpolation="nearest", origin="lower"
#        # give the location of where the colorbar should be placed
##        cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
#        cax, kw = matplotlib.colorbar.make_axes(ax1, pad=0.01)
#        colorbar = fig.colorbar(image, cax=cax)
#        colorbar.set_label(cblabel)
#        ax1.autoscale(enable=True, axis='both', tight=True)
##        fig.colorbar(image, ax=ax1.twinx(),use_gridspec=True)
##        ax1.cax.colorbar(image)

        ax1.xaxis.set_ticks(range(len(self.xkey_values)))
        ax1.xaxis.set_ticklabels(self.xkey_values)
        ax1.yaxis.set_ticks(range(len(self.ykey_values)))
        ax1.yaxis.set_ticklabels(self.ykey_values)
        ax1.set_xlabel(xlabel, size=fontsize)
        ax1.set_ylabel(ylabel, size=fontsize)
        ax1.set_title(title)

        # set the textbox
        if textbox:
            ax1.text(0, 0, textbox, fontsize=12, va='bottom',
                     bbox = dict(boxstyle="round",
                     ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))

        fig.autofmt_xdate()
        fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                            top=wstop, wspace=wspace, hspace=hspace)

        ax1.grid(True)
        print 'saving: ' + figpath
        fig.savefig(figpath + '.png')
        fig.savefig(figpath + '.eps')
        fig.clear()


    def check_grid(self):

        for x in range(self.htc_grid.shape[0]):
            for y in range(self.htc_grid.shape[1]):
                print self.htc_grid[x,y].rjust(70), self.stats_grid[x,y],
                print self.xkey, '=', self.xkey_values[x],
                print self.ykey, '=', self.ykey_values[y]


class A4Tuned:
    """
    plotting for A4 paper size tuned plots: appropriate margins and font sizes
    and sufficient resolution (publication grade)
    """

    def __init__(self, scale=1):
        """
        Set the configuration options

        http://matplotlib.org/users/customizing.html
        """

        self.scale = scale

        self.oneinch = 2.54

        ## using the dictionary-like variable called matplotlib.rcParams,
        ## which is global to the matplotlib package
        #mpl.rcParams['lines.linewidth'] = 2
        #mpl.rcParams['lines.color'] = 'r'
        #
        ## or convenience function for modifying rc settings
        #mpl.rc('lines', linewidth=2, color='r')
        ## reset to default values
        #mpl.rcdefaults()

        #mpl.rc('font',**{'family':'lmodern'})
        ## for Palatino and other serif fonts use:
#        font = {'family'   : 'lmodern',
#                'serif'    : 'Computer Modern',
#                'monospace': 'Computer Modern Typewriter'}
#        mpl.rc('font', **font)
        mpl.rc('text', usetex=True)
        mpl.rcParams['font.family'] = 'lmodern'
        mpl.rcParams['font.serif'] = 'Computer Modern'
        mpl.rcParams['font.monospace'] = 'Computer Modern Typewriter'

        # note that font.size controls default text sizes.  To configure
        # special text sizes tick labels, axes, labels, title, etc, see the rc
        # settings for axes and ticks. Special text sizes can be defined
        # relative to font.size, using the following values: xx-small, x-small,
        # small, medium, large, x-large, xx-large, larger, or smaller
        mpl.rcParams['font.size'] = 9*scale

        # see also
        # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
        # http://stackoverflow.com/questions/7125009/
        # how-to-change-legend-size-with-matplotlib-pyplot

        # -----------------------------
        # font settings for the legend
        # -----------------------------
        mpl.rcParams['legend.fontsize']  = 9*scale
        #mpl.rcParams['legend.linewidth'] = 2
        mpl.rcParams['legend.labelspacing'] = 0.25
        # only take one point for the label instead of two
        mpl.rcParams['legend.numpoints'] = 1
        #mpl.rcParams['legend.markerscale'] = 2
        #mpl.rcParams['legend.frameon'] = 2
        #mpl.rcParams['legend.ncol'] = 1

        # -----------------------------
        # Axes settings
        # -----------------------------
        #axes.titlesize      : large   # fontsize of the axes title
        # not sure which this one affects
        mpl.rcParams['axes.titlesize'] = 9*scale
        # axes titles
        mpl.rcParams['axes.labelsize'] = 9*scale
        mpl.rcParams['axes.labelweight'] = 'bold'

        # the tick lables
        #mpl.rcParams['xtick.color'] = 'k'
        # fontsize of the tick labels
        mpl.rcParams['xtick.labelsize'] = 8*scale

        #mpl.rcParams['ytick.color'] = 'k'
        # fontsize of the tick labels
        mpl.rcParams['ytick.labelsize'] = 8*scale

        # Padding and spacing between various elements use following keywords
        # parameters. These values are measure in font-size units. E.g., a
        # fontsize of 10 points and a handlelength=5 implies a handlelength of
        # 50 points. Values from rcParams will be used if None.

        # the fractional whitespace inside the legend border
        #mpl.rcParams['legend.borderpad'] = 2
        # the vertical space between the legend entries
        #mpl.rcParams['legend.labelspacing'] = 2
        # the length of the legend handles
        #mpl.rcParams['legend.handlelength'] = 2
        # the pad between the legend handle and text
        #mpl.rcParams['legend.handletextpad'] = 2
        # the pad between the axes and legend border
        mpl.rcParams['legend.borderaxespad'] = 0
        # the spacing between columns
        #mpl.rcParams['legend.columnspacing'] = 2

    def setup(self, figfile, **kwargs):
        """
        Just setup the basics, leave plotting for elsewhere

        TODO: migrate from subplots to gridspec
        http://matplotlib.org/users/gridspec.html

        Parameters
        ----------

        nr_plots : int, default=-1
            Specify the number of subplots. Default -1 means there is no
            dynamic sizing of the figure size.

        figsize_x

        figsize_y

        wsleft_cm

        wsright_cm

        wsbottom_cm

        wstop_cm

        wspace

        hspace

        grandtitle

        dpi

        interactive

        """

        self.figfile = figfile
        # TODO: data checks, finish documentation!

        self.dpi = kwargs.get('dpi', 200)
        wsleft_cm = kwargs.get('wsleft_cm', 1.5)
        wsright_cm = kwargs.get('wsright_cm', 1.5)
        wsbottom_cm = kwargs.get('wsbottom_cm', 1.5)
        wstop_cm = kwargs.get('wstop_cm', 2.0)
        #wspace = kwargs.get('wspace', 0.2)
        #hspace = kwargs.get('hspace', 0.2)
        hspace_cm = kwargs.get('hspace_cm', 2.)
        wspace_cm = kwargs.get('wspace_cm', 4.)
        grandtitle = kwargs.get('grandtitle', False)
        self.interactive = kwargs.get('interactive', False)
        # size per figure
        size_y_perfig = kwargs.get('size_y_perfig', TexTemplate.size_y_perfig)
        size_x_perfig = kwargs.get('size_x_perfig', TexTemplate.size_x_perfig)


        # TODO: move to gridspec instead of subplots, its more powerfull
        # http://matplotlib.org/users/gridspec.html
        # determine the number of subplots and corresponding A4 or A3 format
        # 6 subplots per A4
        self.nr_plots = kwargs.get('nr_plots', -1)
        self.nr_cols = kwargs.get('nr_cols', -1)

        # if nr_cols is > 0, we go to the custom way
        if self.nr_plots < 5 and self.nr_plots > 0 and self.nr_cols < 0:
            self.nr_rows = self.nr_plots
            self.nr_cols = 1
            # dynamic sizing of figsize_y based on nr_plots
            if grandtitle:
                header_y = 2.0
            else:
                header_y = 0.1
            fixsize_y_dyn = header_y + (size_y_perfig*self.nr_plots)
            # input is now in cm, standard A4 with 1cm margins
            # was 27.7 before
            # figure size x based on latex pagewidth for thesis
            figsize_x = kwargs.get('figsize_x', 38/2.0)
            figsize_y = kwargs.get('figsize_y', fixsize_y_dyn)
        elif self.nr_plots < 13 and self.nr_plots > 4 and self.nr_cols < 0:
            self.nr_rows = math.ceil(self.nr_plots/2.)
            self.nr_cols = 2
            # A3 paper size
            figsize_x = kwargs.get('figsize_x', 38)
            figsize_y = kwargs.get('figsize_y', 27.7)
        elif self.nr_plots == -1 and self.nr_cols < 0:
            self.nr_rows = 1
            self.nr_cols = 1
            # figure size x based on latex pagewidth for thesis
            figsize_x = kwargs.get('figsize_x', TexTemplate.pagewidth)
            figsize_y = kwargs.get('figsize_y', 10)
            print 'nr_rows and nr_cols set both to 1'
        # the custom way
        else:
            # nr_plots has to be a multiple of nr_rows
            if math.fmod(self.nr_plots, self.nr_plots) > 0:
                raise ValueError, 'nr_plots has to be a multiple of nr_rows'

            self.nr_rows = self.nr_plots/self.nr_cols
            # dynamic sizing of figsize_y based on nr_plots
            header_x = 2.
            singe_plot = 5.
            fixsize_x_dyn = header_x + (12.968*self.nr_cols)
            fixsize_y_dyn = header_x + (singe_plot*self.nr_plots)
            # input is now in cm, standard A4 with 1cm margins
            # was 27.7 before
            # figure size x based on latex pagewidth for thesis
            figsize_x = kwargs.get('figsize_x', fixsize_x_dyn)
            figsize_y = kwargs.get('figsize_y', fixsize_y_dyn)

#        nr_rows = kwargs.get('nr_rows', 6)
#        nr_cols = kwargs.get('nr_cols', 2)

#        fontsize = kwargs.get('fontsize', 'medium')

#        textbox = kwargs.get('textbox', None)
#        xlabel = kwargs.get('xlabel', 'x [m]')
#        ylabel = kwargs.get('ylabel', 'y [m]')
#
#        xlim = kwargs.get('xlim', False)
#        ylim = kwargs.get('ylim', False)
#
#        # if False, plot_simple returns the ax1 and fig instances!
#        # remember to finish and close afterwards manually
#        save = kwargs.get('save', True)

        # and is there any scaling we need to apply?
        if self.scale > 1:
            scaleb = self.scale*0.8
        else:
            scaleb = self.scale
        scale = self.scale

        # and convert to relative terms
        wsleft = (wsleft_cm/figsize_x) * (scaleb/scale)
        wsright = ( 1.0 - ((wsright_cm/figsize_x)*(scaleb/scale)) )
        # seems that hspace and wspace are expressed not per subplot, but
        # the total available space for spacing
        hspace = (self.nr_rows)*hspace_cm/figsize_y
        wspace = (self.nr_cols-1)*wspace_cm/figsize_x
        # top margin to allow grand title and subplot title
        wsbottom = (wsbottom_cm/figsize_y) * (scaleb/scale)
        wstop = ( 1.0 - ((wstop_cm/figsize_y)*(scaleb/scale)) )

        # automatically convert the cm input to matplotlib inches
        self.figsize_x = figsize_x/self.oneinch
        self.figsize_y = figsize_y/self.oneinch

        # global figure setup
        figsize = (self.figsize_x*scale, self.figsize_y*scale)
        if self.interactive:
            self.fig = plt.figure(figsize=figsize, dpi=self.dpi)
        else:
            self.fig = Figure(figsize=figsize, dpi=self.dpi)
            self.canvas = FigCanvas(self.fig)
            self.fig.set_canvas(self.canvas)

        self.fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                                 top=wstop, wspace=wspace, hspace=hspace)

        #font_medium = mpl_font.FontProperties()
        #font_medium.set_size(11)

        # the grand title, only set when given.
        # no titles when making thesis figures, use captions instead!
        if grandtitle:
            self.fig.suptitle(grandtitle, fontsize=11*scale)

        # TODO: which is the best approach?
        # How to deal with backwards compatibility?
        #self.ax = []
        #for k in range(self.nr_plots):
            #self.ax.append(self.fig.add_subplot(self.nr_rows,self.nr_cols,k+1))

        #nr = 1
        #if self.nr_plots > 0:
            #self.ax1 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 1:
            #self.ax2 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 2:
            #self.ax3 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 3:
            #self.ax4 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 4:
            #self.ax5 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 5:
            #self.ax6 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 6:
            #self.ax7 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 7:
            #self.ax8 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 8:
            #self.ax9 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 9:
            #self.ax10 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 10:
            #self.ax11 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1
        #if self.nr_plots > 11:
            #self.ax12 = self.fig.add_subplot(self.nr_rows, self.nr_cols, nr)
            #nr += 1

    def match_yticks(self, ax1, ax2):
        """
        """
        yticks1 = len(ax1.get_yticks())
        ylim2 = ax2.get_ylim()
        yticks2 = np.linspace(ylim2[0], ylim2[1], num=yticks1).tolist()
        ax2.yaxis.set_ticks(yticks2)

        return ax1, ax2

    def plot_simple(self, figfile, time, data, labels, **kwargs):
        """
        Plot Simple: generic plotting
        =============================

        a default plotting theme tuned for A4 paper size (210 mm × 297 mm)
        Each channel a different subplot. Either define a list of channels
        or plot them all.

        Figure size in centimeters, it is converted to inches here
        """
        self.figfile = figfile
        # TODO: data checks, finish documentation!

#        print 'time.shape  :', time.shape
#        print 'data.shape  :', data.shape
#        print 'labels.shape:', labels.shape

        # determine the number of subplots and corresponding A4 or A3 format
        # 6 subplots per A4
        channels = kwargs.get('channels', None)
        self.interactive = kwargs.get('interactive', False)

        if channels:
            nr_channels = len(channels)
        else:
            nr_channels = data.shape[1]
            channels = range(data.shape[1])

        # base size is 2cm, add 5cm per plot
        figsize_y = 2. + nr_channels*7.

        if nr_channels < 7:
            nr_rows = nr_channels
            nr_cols = 1
            nr_plots = nr_rows
            # input is now in cm, standard A4 with 1cm margins
            figsize_x = kwargs.get('figsize_x', 19)
#            figsize_y = kwargs.get('figsize_y', 27.7)
        elif nr_channels < 13:
            nr_rows = 6
            nr_cols = 2
            nr_plots = nr_channels
            # A3 paper size
            figsize_x = kwargs.get('figsize_x', 38)
#            figsize_y = kwargs.get('figsize_y', 27.7)
        else:
            raise UserWarning, 'too many subplots for plot_simple'

#        nr_rows = kwargs.get('nr_rows', 6)
#        nr_cols = kwargs.get('nr_cols', 2)

#        fontsize = kwargs.get('fontsize', 'medium')

        dpi = kwargs.get('dpi', 200)

        wsleft_cm = kwargs.get('wsleft', 1.5)
        wsright_cm = kwargs.get('wsright', 1.5)
        wsbottom_cm = kwargs.get('wsbottom', 1.5)
        wstop_cm = kwargs.get('wstop_cm', 2.0)
        wspace = kwargs.get('wspace', 0.2)
        hspace = kwargs.get('hspace', 0.2)

        grandtitle = kwargs.get('grandtitle', '')
#        textbox = kwargs.get('textbox', None)
#        xlabel = kwargs.get('xlabel', 'x [m]')
#        ylabel = kwargs.get('ylabel', 'y [m]')

        xlim = kwargs.get('xlim', False)
        ylim = kwargs.get('ylim', False)

        # if False, plot_simple returns the ax1 and fig instances!
        # remember to finish and close afterwards manually
        save = kwargs.get('save', True)

        # and convert to relative terms
        wsleft, wsright = wsleft_cm/figsize_x, (1.-(wsright_cm/figsize_x))
        # top margin to allow grand title and subplot title
        wsbottom, wstop = wsbottom_cm/figsize_y, (1.-(wstop_cm/figsize_y))

        # automatically convert the cm input to matplotlib inches
        figsize_x = figsize_x/self.oneinch
        figsize_y = figsize_y/self.oneinch

        # global figure setup
        self.fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
        self.canvas = FigCanvas(self.fig)
        self.fig.set_canvas(self.canvas)
        self.fig.subplots_adjust(left=wsleft, bottom=wsbottom, right=wsright,
                                top=wstop, wspace=wspace, hspace=hspace)

        font_medium = mpl_font.FontProperties()
        font_medium.set_size(12)

        plot_nr = 0
        for channel in channels:
            plot_nr += 1
            title = labels[channel]
            self.ax1 = self.fig.add_subplot(nr_rows, nr_cols, plot_nr)
            self.ax1.plot(time, data[:,channel])
            self.ax1.set_title(title)
            self.ax1.grid(True)
            # in case we have two cols, leave x-axis intact on bottom 2 fig
            if nr_cols == 2:
                if plot_nr < nr_plots-1:
                    # clear all x-axis values but the bottom subplots
                    self.ax1.set_xticklabels([])
            # only leave the single bottom intact
            elif nr_cols == 1:
                if plot_nr < nr_plots:
                    # clear all x-axis values but the bottom subplots
                    self.ax1.set_xticklabels([])

            # set axis limits if applicable
            if xlim:
                self.ax1.set_xlim(xlim)
            if ylim:
                self.ax1.set_ylim(ylim)

        # the grand title
        self.fig.suptitle(grandtitle, size='x-large')

        if save:
            self.save_fig()


    def save_fig(self, eps=True, png=True, figfile=False):
        # first remove all points, latex doesn't like them
        if figfile:
            self.figfile = figfile
        self.figfile = self.figfile.replace('.', '')
        print 'saving: ' + self.figfile + '  ... ',
        if png:
            # for some figures, the size is quite small. Good for eps, but
            # not very for png
            #w = 1.5
            #size = self.fig.get_size_inches()
            #self.fig.set_size_inches(size[0]*w, size[1]*w)
            self.fig.savefig(self.figfile + '.png')
        # but I can't figure out what to pass one??
        # http://matplotlib.sourceforge.net/api/backend_bases_api.html
#        FigCanvasBase.switch_bakend()
        if eps: self.fig.savefig(self.figfile + '.eps')
#        self.fig.savefig(self.figfile + '.svg')
        # FigCanvasBase does not have close argument
        if not self.interactive:
            self.fig.clear()
        print 'done'


    def psd_eigenfreq(self, time, data, channels,
                      sample_rate, **kwargs):
        """
        Power Spectral Density Analysis
        ===============================

        Compare the signal and PSD analysis for all given channels.

        Parameters
        ----------

        time : ndarray(n)

        data : ndarray(n,k)

        channels : list of int

        sample_rate : float

        figfile : str, default=self.resfile

        channel_names : list of str, default=None
            Subplot titles

        nnfts  :list of int, default=[16384, 8192, 4096, 2048]

        fn_max : float, default=100
            Set the maximum value on the eigenfrequency (x-axis). Higher
            values are not retunred in eigenfreqs nor plotted

        saveresults : boolean, default=False
            Save the eigenfreqs output in the same place as the figure

        Returns
        -------

        eigenfreqs : dict with ndarray
            key is channeli, value is ndarray([freq peaks, Pxx_log values])

        """

        # take a few different number of samples for the NFFT
        nnfts = kwargs.get('nnfts', [16384, 8192, 4096, 2048])
        channel_names = kwargs.get('channel_names', None)
        fn_max = kwargs.get('fn_max', 100)
        saveresults = kwargs.get('saveresults', False)

        # save for each channel the peak frequencies, but only for nnft=2048
        eigenfreqs = dict()

        for chi, ch in enumerate(channels):
            # axis for the raw data
            ax = self.fig.add_subplot(self.nr_rows, self.nr_cols, (2*chi)+1)
            ax.plot(time, data[:,ch])

            # and for the PSD
            ax = self.fig.add_subplot(self.nr_rows, self.nr_cols, (2*chi)+2)
            for index, nn in enumerate(nnfts):
                #Pxx,freqs=ax.psd(data[:,ch],NFFT=nn,Fs=sample_rate)
                Pxx, freqs = mpl.mlab.psd(data[:,ch], NFFT=nn, Fs=sample_rate)
                # and plot on a 10*log10() scale
                ax.plot(freqs, 10*np.log10(Pxx))
                # but now the values are in logs, but the scaling indications
                # (the ticks) are not! It is much more simple to use:
                # ax.plot(freqs, Pxx)
                # ax1.set_yscale('log')

                # find the frequency holding the most energy
                # ignore everything below 2hz
                #sel = freqs.__ge__(2.)
                #freqs_sel = freqs[sel]
                #Pxx_sel = Pxx[sel]
                ## and keep track of them
                #eigenfreqs[chi, index] = freqs_sel[Pxx_sel.argmax()]

                # second approach: use wafo to get all the peaks
                # but ignore everything below 3.5dB
                # convert to logarithmic scale for better peak detection
                Pxx_log = 10.*np.log10(Pxx)
                # just make n as large as the array
                pi = wafo.misc.findpeaks(Pxx_log, n=len(Pxx), min_h=3.5)
                # sort, otherwise it is ordered wrt significant wave height
                pi.sort()
                # and indicate the peaks
                ax.plot(freqs[pi], Pxx_log[pi], 'o')
                ax.grid(True)
                print 'peaks found at:', freqs[pi]

                # save the data
                key = '%i-%i' % (ch, nn)
                eigenfreqs[key] = np.array([freqs[pi], Pxx_log[pi]])

                switch = True
                # and mark all peaks
                for peak in freqs[pi]:
                    # ignore everything above the xlimit
                    if peak > fn_max:
                        break
                    # take the average frequency and plot vertical line
                    ax.axvline(x=peak, linewidth=1, color='k')
                    # and the value in a text box
                    textbox = '%2.2f' % peak
                    yrange_plot = ( 10.*math.log10(Pxx.max()) \
                                   -10.*math.log10(Pxx.min()))
                    if switch:
                        # locate at the min value (down the plot), but a little
                        # lower so it does not interfere with the plot itself
                        text_ypos = 10.*math.log10(Pxx.min())-yrange_plot*0.1
                        switch = False
                    else:
                        # put it a little lower than the max value so it does
                        # not mess with the title (up the plot)
                        text_ypos = 10.*math.log10(Pxx.max())-yrange_plot*0.1
                        switch = True
                    ax.text(peak, text_ypos, textbox, fontsize=12,
                             va='bottom', bbox = dict(boxstyle="round",
                             ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),alpha=0.8,))

            # give the channel name as a title
            if channel_names:
                ax.set_title(channel_names[chi])

            #print 'channel', ch, ':', eigenfreqs[ch]
            ax.set_xlim([0,fn_max])

            ## take the average frequency and plot vertical line
            #ax.axvline(x=eigenfreqs[chi,:].mean(), linewidth=1, color='k')
            ## and the value in a text box
            #textbox = '%2.2f' % eigenfreqs[chi,:].mean()
            #text_ypos = 10.*math.log10(Pxx.min())
            #ax.text(eigenfreqs[chi,:].mean(), text_ypos, textbox, fontsize=12,
                             #va='bottom', bbox = dict(boxstyle="round",
                             #ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))

        if saveresults:
            resfile = self.figfile.replace('.', '') + '_eigenfreqs.pkl'
            print 'saving: ' + resfile + '  ... ',
            # and safe the eigenfrequency data for later reference
            FILE = open(resfile,'wb')
            pickle.dump(eigenfreqs, FILE, protocol=2)
            FILE.close()

        return eigenfreqs

class PlotStData:
    """
    Class for plotting a HAWC2 structural input file (*.st)
    """

    def __init__(self, **kwargs):
        """
        """
        self.md = sim.ModelData(silent=True)
        #st_icol = md.st_headers.E

        case = kwargs.get('case', False)
        if case:
            st_path = case['[model_dir_local]'] + 'data/'
            st_file = case['[st_file]']
        else:
            st_path = kwargs.get('st_path', False)
            st_file = kwargs.get('st_file', False)

        if st_path and st_file:
            self.md.load_st(st_path, st_file)

    def compare(self, figpath, sets, sti_col, **kwargs):
        """
        Compare different subsets
        =========================

        From either different st files or one and the same file.

        Standard formatting is for figure size of width=12cm (thesis pagewidth)

        Parameters
        ----------

        sets : list
            List of Simulation.ModelData object, set and subset number
            [ [md1, setnr, subsetnr, label], [md2,setnr,subsetnr,label], ...]

        sti_col :
            Index to column from the st_arr that has to be compared

        figfile : str, default=''

        scale : float, default=1

        figsize_x : float, default=TexTemplate.pagewidth*0.5

        figsize_y : float, default=TexTemplate.pagewidth*0.5

        ylim : list, default=False

        """

        figfile = kwargs.get('figfile', '')
        scale = kwargs.get('scale', 1)
        figsize_x = kwargs.get('figsize_x', TexTemplate.pagewidth*0.5)
        figsize_y = kwargs.get('figsize_y', TexTemplate.pagewidth*0.5)
        ylim = kwargs.get('ylim', False)

        # setup the figure
        pa4 = A4Tuned(scale=scale)
        pa4.setup(figpath + figfile, nr_plots=1, hspace_cm=2.,
                   grandtitle=False, wsleft_cm=1.3, wsright_cm=0.5,
                   wstop_cm=0.5, wsbottom_cm=1.0,
                   figsize_x=figsize_x, figsize_y=figsize_y)

        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        # determine the ylabel for the plot
        headers_tex = sets[0][0].st_column_header_list_latex
        headers = sets[0][0].st_column_header_list
        if type(sti_col).__name__ == 'int':
            ylabel = r'$%s$' % headers_tex[sti_col]
            sti_name = headers[sti_col]
        elif len(sti_col) == 2:
            tmp = [headers_tex[k] for k in sti_col]
            ylabel = r'$%s$' % ''.join(tmp)
            tmp = [headers[k] for k in sti_col]
            sti_name = ''.join(tmp)

        colors = ['ks-','r^--', 'bo-', 'yd-', 'g*--']
        file_ref = []
        index = 0
        for md, i, j, label in sets:
            set_comment = md.st_comments['%03i-000-0' % (i)]
            subset_comment = md.st_comments['%03i-%03i-b' % (i,j)]
            st_arr = md.st_dict['%03i-%03i-d' % (i,j)]
            file_ref.append('%s_%i_%i' % (md.st_file, i, j))

            x = st_arr[:,0]
            if type(sti_col).__name__ == 'int':
                y = st_arr[:,sti_col]
            elif len(sti_col) == 2:
                y = st_arr[:,sti_col[0]]*st_arr[:,sti_col[1]]

            ax1.plot(x, y, colors[index], label=label)

            index += 1

        ax1.legend(loc='upper right')
        #ax1.set_ylim([-1,22])
        #ax1.set_xlim([-0.02,1.02])
        ax1.set_xlabel('radial position [m]')
        ax1.set_ylabel(ylabel)
        ax1.grid(True)

        if ylim:
            ax1.set_ylim(ylim)

        if figfile == '':
            # add the column info to the plot name
            file_ref.append(sti_name)
            figfile = '_'.join(file_ref)

        pa4.save_fig(figfile=figpath+figfile)


    def plot_all(self, figpath, selection=[]):
        """
        Plot all set/subsets
        ====================

        Plot all set/subsets from the st file or the given selection

        Parameters
        ----------

        figpath : str

        selection : list
            [ [setnr, subsetnr], [setnr, subsetnr], ... ]

        """

        for i,j in selection:
            self.dashboard(figpath, set_subset=[i,j])


    def dashboard(self, figpath, figname=None, st_arr=None, md=None,
                  set_subset=None):
        """
        Plot all parameters in one overview plot
        ========================================

        Based on the old st plot from HawcPy. Give either an st_arr or an
        Simulation.Modeldata object and set_subset=[setnr, subsetnr] list

        st_dict has following key/value pairs
            'nset'    : total number of sets in the file (int).
                        This should be autocalculated every time when writing
                        a new file.
            '007-000-0' : set number line in one peace
            '007-001-a' : comments for set-subset nr 07-01 (str)
            '007-001-b' : subset nr and number of data points, should be
                        autocalculate every time you generate a file
            '007-001-d' : data for set-subset nr 07-01 (ndarray(n,19))
        for the comments stripped from their set nr indication and stuff:
            st_comments and the same keys as st_dict
        """

        # the different input scenario's: md is already loaded with init
        if not md and not st_arr:
            md = self.md
            set_labels = md.st_column_header_list_latex

        if md and not set_subset:
            raise TypeError, 'set_subset is not defined correctly'
        elif set_subset:
            # define the set-subset numbers
            i = set_subset[0]
            j = set_subset[1]
            # get the comments
            try:
                # set comment should be the name of the body
                set_comment = md.st_comments['%03i-000-0' % (i)]
                subset_comment = md.st_comments['%03i-%03i-b' % (i,j)]
                st_arr = md.st_dict['%03i-%03i-d' % (i,j)]
            except AttributeError:
                msg = 'ModelData object md is not loaded properly'
                raise AttributeError, msg

        if not figname:
            figname = '%s_%s_%i_%i' % (md.st_file, set_comment, i, j)

        # define the label precision
        majorFormatter = FormatStrFormatter('%1.1e')

        # color-label for left and right axis plots
        left = 'bo-'
        right = 'rx-'
        ax2_left = 'ko-'
        ax2_right = 'gx-'
        label_size = 'large'
        legend_ax1 = 'upper right'

        # and plot some items of the structural data
        fig = Figure(figsize=(16, 9), dpi=200)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)

        ax = fig.add_subplot(2, 3, 1)
        fig.subplots_adjust(left= 0.1, bottom=0.1, right=0.9,
                        top=0.95, wspace=0.35, hspace=0.2)

        # x-axis is always the radius
        x = st_arr[:,0]
        # mass

        ax.plot(x,st_arr[:,1], left, label=r'$'+set_labels[1]+'$')
        ax.legend()
        ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)

        # x_cg and y_cg and pitch
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(x,st_arr[:,2], left, label=r'$'+set_labels[2]+'$')
        ax.plot(x,st_arr[:,3], right, label=r'$'+set_labels[3]+'$')
        ax.plot(x,st_arr[:,16], ax2_left, label=r'$'+set_labels[16]+'$')
        ax.legend(loc=legend_ax1)
#                plt.grid(True)
#                ax2 = ax.twinx()
#                ax2.legend(loc='upper right')
        ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)

        # x_sh and y_sh, x_e and y_e
        ax = fig.add_subplot(2, 3, 3)
        ax.plot(x,st_arr[:,6], left,label=r'$'+set_labels[6]+'$')
        ax.plot(x,st_arr[:,7], right, label=r'$'+set_labels[7]+'$')
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)
        ax2 = ax.twinx()
        ax2.plot(x,st_arr[:,17], ax2_left, label=r'$'+set_labels[17]+'$')
        ax2.plot(x,st_arr[:,18], ax2_right, label=r'$'+set_labels[18]+'$')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax2.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)

        # second row of plots
        # EI_x and EI_y
        ax = fig.add_subplot(2, 3, 4)
        label = set_labels[8] + ' ' + set_labels[10]
        ax.plot(x,st_arr[:,8]*st_arr[:,10], left, label=r'$'+label+'$')
#                ax2 = ax.twinx()
        label = set_labels[8] + ' ' + set_labels[11]
        ax.plot(x,st_arr[:,8]*st_arr[:,11], right, label=r'$'+label+'$')
        ax.legend(loc=legend_ax1)
#                ax2.legend(loc='upper right')
        ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)

        # m*ri_x and m*ri_y
        ax = fig.add_subplot(2, 3, 5)
        label = set_labels[1] + ' ' + set_labels[4] + '^2'
        ax.plot(x,st_arr[:,1]*np.power(st_arr[:,4],2),
                left, label=r'$'+label+'$')
#                ax2 = ax.twinx()
        label = set_labels[1] + ' ' + set_labels[5] + '^2'
        ax.plot(x,st_arr[:,1]*np.power(st_arr[:,5],2),
                right, label=r'$'+label+'$')
        ax.legend(loc=legend_ax1)
#                ax2.legend(loc='upper right')
        ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)

        # GI_p/K and EA
        ax = fig.add_subplot(2, 3, 6)
        label = set_labels[9] + ' ' + set_labels[12]
        ax.plot(x,st_arr[:,9]*st_arr[:,12], left, label=r'$'+label+'$')
        ax.legend(loc='upper left')
        ax.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.grid(True)
        ax2 = ax.twinx()
        label = set_labels[8] + ' ' + set_labels[15]
        ax2.plot(x,st_arr[:,8]*st_arr[:,15], right, label=r'$'+label+'$')
        ax2.legend(loc='upper right')
        ax2.set_xlabel(r'$' + set_labels[0] + '$', size=label_size)
        ax2.yaxis.set_major_formatter(majorFormatter)
        ax2.grid(True)

        title = '%s set %i subset %i' % (set_comment, i, j)
        title += '\n%s' % subset_comment
        fig.suptitle(title)

        # and save
        fig.savefig(figpath + figname + '.png', orientation='landscape')
        fig.savefig(figpath + figname + '.eps', orientation='landscape')
        fig.clear()

def hawc2dashboard():
    """
    """
    pass

class Cases:
    """
    Plotting for with support for HAWC2 Cases dictionary
    """

    def __init__(self, cao, Chi):
        """

        Parameters
        ----------

        cao : cases object
            a Simulations.Cases object

        """
        self.cao = cao
        self.Chi = Chi

    def dasbhoard_yawcontrol(self, cname, figpath, figfile, grandtitle=False):

        Chi = self.Chi

        # load the HAWC2 result file
        res = self.cao.load_result_file(self.cao.cases[cname])
        time = res.sig[:,0]
        casedict = self.cao.cases[cname]

        # --------------------------------------------------------------------
        rho = 1.225
        A = ((0.245+0.555)*(0.245+0.555)*np.pi) - (0.245*0.245*np.pi)

        # mean windspeed can only be used if the ramping up in HAWC2 init
        # is ignored
        #V = np.mean(self.sig[:,Chi.wind])
        V = float(casedict['[windspeed]'])

        factor_p = 0.5*A*rho*V*V*V
        factor_t = 0.5*A*rho*V*V
        # load units are in kNm
        mechpower = -res.sig[:,Chi.mz_shaft]*1000*res.sig[:,Chi.omega]
        cp = mechpower/factor_p
        ct = 1000.0*res.sig[:,Chi.fz_shaft]/factor_t

        ## moving average for the mechanical power
        #filt = ojf.Filters()
        ## take av2s window, calculate the number of samples per window
        #ws = 0.1/self.casedict['[dt_sim]']
        #mechpower_avg = filt.smooth(mechpower, window_len=ws, window='hanning')
        #N = len(mechpower_avg) - len(self.sig[:,0])
        #mechpower_avg = mechpower_avg[N:]

        # --------------------------------------------------------------------
        pa4 = A4Tuned()
        pa4.setup(figpath+figfile, grandtitle=grandtitle, nr_plots=5,
                  wsleft_cm=2., wsright_cm=1.5, hspace_cm=1., wspace_cm=2.1,
                  interactive=False)

        plotnr = 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Omega [RPM]')
        ax.plot(time[:], res.sig[:,Chi.omega]*30./np.pi, 'b')
        #leg = ax.legend(loc='best')
        #leg.get_frame().set_alpha(0.5)
        ax.grid(True)
        ax.set_xlim([time[:].min(), time[:].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Yaw Angle (blue) [deg], normalised yaw control (red)')
        ax.plot(time[:], res.sig[:,Chi.yawangle], 'b')
        if casedict['[yawmode]'] == 'control_ini':

            # normalise wrt set max/min control output
            qq = res.sig[:,Chi.m_yaw_control]/casedict['[yaw_c_max]']
            # and centre around the yaw ref angle +-20 degrees
            qq = (qq*20.) + res.sig[:,Chi.yaw_ref_angle]
            # print the boundaries of the yaw control box
            bound_up = casedict['[yaw_c_ref_angle]'] + 20
            ax.axhline(y=bound_up, linewidth=1, color='k',\
                linestyle='-', aa=False)
            bound_low = casedict['[yaw_c_ref_angle]'] - 20
            ax.axhline(y=bound_low, linewidth=1, color='k',\
                linestyle='-', aa=False)

            ## for the yaw control: normalise wrt max value and center around
            ## the reference yaw angle
            #ci = Chi.yawangle
            #maxrange = self.sig[:,ci].max() - self.sig[:,ci].min()
            #qq = self.sig[:,Chi.dll_control]*1000.
            #qq = (qq/(qq.max()-qq.min()))*maxrange
            ## centre around ref angle instead of zero
            #qq += self.sig[:,Chi.yaw_ref_angle]

            ax.plot(time[:], qq, 'r')
            ax.plot(time[:], res.sig[:,Chi.yaw_ref_angle], 'g')

        else:
            pass
        ax.set_xlim([time[:].min(), time[:].max()])
        ax.grid(True)

#        plotnr += 1
#        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
#        ax.set_title('Tower base FA [Nm]')
#        ax.plot(time[:], res.sig[:,Chi.mx_tower]*1000., 'b')
#        ax.grid(True)
#        ax.set_xlim([time[:].min(), time[:].max()])
#
#        #plotnr += 1
#        #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
#        #ax.set_title('Tower base SS (blue yawing, red non yawing) [Nm]')
#        #ax.plot(self.time[:], self.sig[:,Chi.mytower]*1000., 'b')
#        #ax.plot(self.time[:], self.sig[:,Chi.mytower_glo]*1000., 'r')
#        #ax.grid(True)
#
#        plotnr += 1
#        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
#        title = 'Thrust [N] (blue), Thrust coefficient [\%] (red)'
#        #title += 'and Mech Thrust shaft [N] (black)'
#        ax.set_title(title)
#        ax.plot(time[:], res.sig[:,Chi.fz_shaft]*1000., 'b')
#        ax.plot(time[:], ct*100, 'r')
#        #ax.plot(self.time[:], self.sig[:,Chi.fzshaft]*1000., 'k')
#        ax.grid(True)
#        ax.set_xlim([time[:].min(), time[:].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Blade 1 flapwise loads [Nm]')
        ax.plot(time[:], res.sig[:,Chi.mx_b1_ro]*1000., 'b')
        ax.grid(True)
        ax.set_xlim([time[:].min(), time[:].max()])

#        plotnr += 1
#        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
#        #ax.set_title('Aero Power [W] (blue) and Power Coefficient [%] (red)')
#        ax.set_title('Mech Power shaft [W] (red)')
#        ax.plot(time[:], mechpower, 'r')
#        #ax.plot(self.time[:], mechpower_avg, 'r')
#        # betz limit
#        #ax.axhline(y=100*16./27.,linewidth=1,color='k',linestyle='-',aa=False)
#        ax.grid(True)
#        ax.set_xlim([time[:].min(), time[:].max()])

#        plotnr += 1
#        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
#        ax.set_title('Blade 1 tip deflection flapwise [m]')
#        ax.plot(time[:], res.sig[:,Chi.y_b1_tip], 'b')
#        ax.grid(True)
#        ax.set_xlim([time[:].min(), time[:].max()])
#
#        plotnr += 1
#        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
#        ax.set_title('Blade 1 tip deflection edgewise [m]')
#        ax.plot(time[:], res.sig[:,Chi.x_b1_tip], 'b')
#        ax.grid(True)
#        ax.set_xlim([time[:].min(), time[:].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Angle of attack near the root (50\%) [deg]')
        ax.plot(time[:], res.sig[:,Chi.aoa16_b1], 'b')
        ax.grid(True)
        ax.set_xlim([time[:].min(), time[:].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Angle of attack near the tip (92\%) [deg]')
        ax.plot(time[:], res.sig[:,Chi.aoa49_b1], 'b')
        ax.grid(True)
        ax.set_xlim([time[:].min(), time[:].max()])

        pa4.save_fig()

    def dashboard_blade(self, figpath):
        """
        cases : dict
            A cases dictionary
        """

        # select the channels
        tag_tip = 'bearing-shaft_nacelle-angle_speed-rpm'
        #tag_tshaft = 'shaft-shaft-node-002-momentvec-z'
        tag_power = 'DLL-ojf_generator-inpvec-1'

        # for each case in the cases dictionary
        for case, casedict in self.cao.cases.iteritems():

            # load the HAWC2 results
            res = self.cao.load_result_file(casedict)
            #for k in sorted(res.ch_dict):
                #print k
            itgen = res.ch_dict[tag_tgen]['chi']
            irpm = res.ch_dict[tag_rpm]['chi']


            # setup the figure
            figname = case
            figsize_x = TexTemplate.pagewidth*0.5
            figsize_y = TexTemplate.pagewidth*0.5
            scale = 1.8
            pa4 = A4Tuned(scale=scale)
            pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                           grandtitle=False, wsleft_cm=1.7, wsright_cm=0.5,
                           wstop_cm=0.8, wsbottom_cm=1.0,
                           figsize_x=figsize_x, figsize_y=figsize_y)
            # actual plotting
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
            ax1.plot(res.sig[:,0], res.sig[:,irpm], 'r', label='RPM')
            ax1.plot(res.sig[:,0], res.sig[:,itgen]*100, 'b',
                     label='$100T_{gen}$')

            # finishing the plot
            #ax1.set_title('')
            ax1.grid(True)
            #ax1.legend(loc='upper right', bbox_to_anchor=(1.04, 1.36), ncol=2)
            ax1.legend(loc='best')
            ax1.set_xlabel('time [s]')
            ax1.set_ylabel(r'RPM, $10e^2 T_{gen} [N]$')
            pa4.save_fig()

    def dashboard_rpm(self, figpath):
        """
        cases : dict
            A cases dictionary
        """

        # select the channels
        tag_rpm = 'bearing-shaft_nacelle-angle_speed-rpm'
        #tag_tshaft = 'shaft-shaft-node-002-momentvec-z'
        tag_tgen = 'DLL-ojf_generator-inpvec-1'

        # for each case in the cases dictionary
        for case, casedict in self.cao.cases.iteritems():

            # load the HAWC2 results
            res = self.cao.load_result_file(casedict)
            #for k in sorted(res.ch_dict):
                #print k
            irpm = res.ch_dict[tag_rpm]['chi']

            # setup the figure
            figname = case
            figsize_x = TexTemplate.pagewidth*0.5
            figsize_y = TexTemplate.pagewidth*0.5
            scale = 1.8
            pa4 = A4Tuned(scale=scale)
            pa4.setup(figpath + figname, nr_plots=1, hspace_cm=2.,
                           grandtitle=False, wsleft_cm=1.7, wsright_cm=0.5,
                           wstop_cm=0.8, wsbottom_cm=1.0,
                           figsize_x=figsize_x, figsize_y=figsize_y)
            # actual plotting
            ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
            ax1.plot(res.sig[:,0], res.sig[:,irpm], 'r', label='RPM')

            # the generator might not be activated
            try:
                itgen = res.ch_dict[tag_tgen]['chi']
                ax1.plot(res.sig[:,0], res.sig[:,itgen]*100, 'b',
                         label='$100T_{gen}$')
            except KeyError:
                pass

            # finishing the plot
            #ax1.set_title('')
            ax1.grid(True)
            #ax1.legend(loc='upper right', bbox_to_anchor=(1.04, 1.36), ncol=2)
            ax1.legend(loc='best')
            ax1.set_xlabel('time [s]')
            ax1.set_ylabel(r'RPM, $10e^2 T_{gen} [N]$')
            pa4.save_fig()


    def compare(self, chi, xchannel, ychannel, sortkey=False):
        """
        Compare (overplot) all the simulations in cases for channel index chi.
        """

        if sortkey:
            cases_sorted = {}
            for casekey, caseval in self.cao.cases.iteritems():
                key = '%1.1e' % (caseval[sortkey])
                cases_sorted[key] = casekey

        # and plot for each case the dashboard
        plt.figure()
        i = 0
        colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'k--', 'r--', 'b--', 'g--']
        for key in sorted(cases_sorted):
            print 'start loading: %s' % key
            caseval = self.cao.cases[cases_sorted[key]]
            # read the HAWC2 file
            res = self.cao.load_result_file(caseval)
            # plotting
            label = '%1.1e' % (caseval[sortkey])
            chi = res.ch_dict[xchannel]['chi']
            plt.plot(res.sig[:,0], res.sig[:,chi], colors[i], label=label)
            plt.plot(hawc2res.sig[:,chis.azi], hawc2res.sig[:,chis.mx_b1_30])
            i += 1
        title = '%s=%1.1e, varying %s' % (constkey, caseval[constkey], sortkey)
        plt.title(title)
        plt.grid()
        plt.legend(loc='best', title='%s' % sortkey)
        plt.show()

    def case_interactive(self, cname):
        """
        Interactive plot with pylab, for a single case
        First load a case with load_result_file, than plot
        """

        res = self.cao.load_result_file(cname)
        fig = plt.figure(self.case['[case_id]'])
        nr_rows, nr_cols, plot_nr = 1, 1, 1
        ax = fig.add_subplot(nr_rows, nr_cols, plot_nr)


def subplots(nrows=1, ncols=1, figsize=(12,8), dpi=120, num=0):
    """

    Equivalent function of pyplot.subplots(). The difference is that this one
    is not interactive and is used with backend plotting only.

    Parameters
    ----------
    nrows=1, ncols=1, figsize=(12,8), dpi=120

    num : dummy variable for compatibility

    Returns
    -------
    fig, axes


    """

    fig = mpl.figure.Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)
    axes = np.ndarray((nrows, ncols), dtype=np.object)
    plt_nr = 1
    for row in range(nrows):
        for col in range(ncols):
            axes[row,col] = fig.add_subplot(nrows, ncols, plt_nr)
            plt_nr += 1
    return fig, axes


def one_legend(*args, **kwargs):
    # or more general: not only simple line plots (bars, hist, ...)
    objs = []
    for ax in args:
        objs += ax.get_legend_handles_labels()[0]
#    objs = [ax.get_legend_handles_labels()[0] for ax in args]
    labels = [obj.get_label() for obj in objs]
    # place the legend on the last axes
    leg = ax.legend(objs, labels, **kwargs)
    return leg


def match_yticks(ax1, ax2, nr_ticks_forced=None, extend=False):
    """
    """

    if nr_ticks_forced is None:
        nr_yticks1 = len(ax1.get_yticks())
    else:
        nr_yticks1 = nr_ticks_forced
        ylim1 = ax1.get_ylim()
        yticks1 = np.linspace(ylim1[0], ylim1[1], num=nr_yticks1).tolist()
        ax1.yaxis.set_ticks(yticks1)

    ylim2 = ax2.get_ylim()
    yticks2 = np.linspace(ylim2[0], ylim2[1], num=nr_yticks1).tolist()
    ax2.yaxis.set_ticks(yticks2)

    if extend:
        offset1 = (ylim1[1] - ylim1[0])*0.1
        ax1.set_ylim(ylim1[0]-offset1, ylim1[1]+offset1)
        offset2 = (ylim2[1] - ylim2[0])*0.1
        ax2.set_ylim(ylim2[0]-offset2, ylim2[1]+offset2)

    return ax1, ax2


if __name__ == '__main__':

    pass

