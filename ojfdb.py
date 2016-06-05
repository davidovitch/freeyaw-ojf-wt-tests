# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:37:00 2012

Make a database of all the test and their results

@author: dave
"""

#from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import pickle
import string
import shutil

import numpy as np
import matplotlib as mpl
import pandas as pd

import ojfresult
import plotting
import ojf_post
import misc
from ojfdb_dict import ojf_db

PATH_DB = 'database/'
OJFPATH_RAW = 'data/raw/'

def symlink_to_hs_folder(source_folder, path_db, symf='symlinks_hs_mimer/'):
    """
    Create simlinks to the HS camera footage folders on Mimer

    source_folder should be the path to to where all the days are saved:
    source_folder = '/x/y/z/02/'
    source_folder = '/x/y/z/04/'

    on Mimer, path's to HS footage:
    02/2012-02-12/HS 0212/
    04/2012-04-05/0405_HScamera/

    see also make_symlinks_hs further down
    """

    # load the database index for the dspace-strain-ojf cases
    FILE = open(path_db + 'db_index_symlinks_all_runid.pkl')
    db_index_runid = pickle.load(FILE)
    FILE.close()

    path_db += symf

    # create the folder if it doesn't exist
    try:
        os.mkdir(path_db)
    except OSError:
        pass

    # -------------------------------------------------------------------------
    # FEBRUARY, LaCie2Big, Lacie
    # -------------------------------------------------------------------------
    # for Lacie February, we just pass on the folder directly, they are
    # already grouped in one folder
    if source_folder.endswith('HighSpeedCamera'):
        for result in os.listdir(source_folder):
            # give the same name as the dspace-strain-ojf case
            runid = '_'.join(result.split('_')[0:3])
            print
            print runid
            # if we can not find the case, keep the original name
            try:
                resshort = db_index_runid[runid]
                print '   ', '_'.join(result.split('_')[0:-1])
                print '   ', resshort
            except KeyError:
                resshort = '_'.join(result.split('_')[0:-1])
                print resshort

            source = os.path.join(source_folder, result)
            target = os.path.join(path_db, resshort)
            try:
                os.symlink(source, target)
            except OSError:
                print '   **** file exists ****'
                print '   source:', result
                print '   target:', resshort
        # and we are done, do not get to the next level
        return

    # -------------------------------------------------------------------------
    # ALL OTHER CASES
    # -------------------------------------------------------------------------
    # all the days are listed in here
    for day in os.listdir(source_folder):
        # for each day, check if there is HS camera footage available
        subfolders = os.listdir(os.path.join(source_folder, day))
        for folder in subfolders:
            if folder.find('HS ') > -1 or folder.find('_HScamera') > -1:
                # now we have the folder containing all the results folders
                results = os.listdir(os.path.join(source_folder, day, folder))
                for result in results:
                    # give the same name as the dspace-strain-ojf case
                    runid = '_'.join(result.split('_')[0:3])
                    print
                    print runid

                    # if we can not find the case, ignore it and just give
                    # maintain its name
                    try:
                        resshort = db_index_runid[runid]
                        print '   ', '_'.join(result.split('_')[0:-1])
                        print '   ', resshort
                    except KeyError:
                        resshort = '_'.join(result.split('_')[0:-1])
                        print resshort

                    source = os.path.join(source_folder, day, folder, result)
                    target = os.path.join(path_db, resshort)
                    try:
                        os.symlink(source, target)
                    except OSError:
                        print '   **** file exists ****'
                        print '   source:', result
                        print '   target:', resshort
#                    print source
#                    print target
#                    print


def symlink_to_folder(source_folder, path_db, **kwargs):
    """
    Create symlinks in one database folder and use consitent naming for the
    OJF, DSpace and strain result files. That makes combining all three
    result files later a breeze.

    This is by far the most safe way since the files are not actually renamed.
    """

    df_index = {'source':[], 'runid':[], 'caseid':[]}

    db_id = kwargs.get('db_id', 'symlinks')
    path_db += db_id + '/'
    # create the folder if it doesn't exist
    try:
        os.mkdir(path_db)
    except OSError:
        pass

    # TODO: move filtering to the build_db stuff
    # file ignore list, looks if the keyword occurs in the file name
    fileignore = kwargs.get('fileignore',
        ['zerorun', 'calibration', 'slowdown', 'spindown', 'bladecal', \
         'towercal', 'virbations', 'eigen', 'sweep',  'vibration', 'speedup',\
         'spinup', 'shutdown', 'startup'])

    # folder ignore operates on the first 3 characters of the folder name
    folderignore = kwargs.get('folderignore', \
        ['mea', 'dsp', 'tri', 'cal', 'hs ', 'dc_', 'ojf', 'hs_'])

    # save a pickled dictionary that holds all the unique base names
    db_index = {}
    # and a short version where only the runid is taken
    db_index_runid = {}

    ignore_root = 'NA'
    # cycle through everything we can reach from the target path
    for root, dirs, files in os.walk(source_folder, topdown=True):
        file_dict = {}

        # and also ingore any subfolders, only works with topdown approach
        if root.startswith(ignore_root):
            #print 'ignore:', root
            continue

        # do not consider content of folders: triggers, Measurement, dSPACE
        folder = root.split('/')[-1]
        # cut them all to the same length
        #if len(folder) > 5 and folder[0:6] in folderignore:
        if len(folder) > 5 and folder[0:3].lower() in folderignore:
            ignore_root = root
            #print 'ignore:', root
            continue
        else:
            ignore_root = 'NA'

        # for each folder, keep all the filenames in a dictionary. On the key
        # keep what we have the same for sure: data_runid
        for name in files:
            # ignore movie files
            ext = name[-4:len(name)]
            if ext in ['.asf', '.avi', '.rar', '.bmp']:
                continue

            # the current file is not allowed to have any item occuring in the
            # ignore list
            nextfile = False
            for k in fileignore:
                nextfile = False
                if name.find(k) > -1:
                    nextfile = True
                    break

            if nextfile: continue

            # prepare the dictionary key
            key = name.replace('.log','').replace('.mat','').replace('.csv','')
            # ignore if the first item of the key is not the date
            try:
                int(key.split('_')[0])
            except ValueError:
                continue
            key = '_'.join(key.split('_')[0:3])
            # each key can have up to 3 files (dspace, ojf, strain)
            if file_dict.has_key(key):
                file_dict[key][name] = root
            else:
                # you can't have the same filename in one dir, so no risk
                # of a previously created key
                file_dict[key] = {name : root}

        # and cycle through al the files in the directory that have to be
        # renamed consistantly. Each key is the case id, value are the
        # files and their full path
        print root
        for key, values in file_dict.iteritems():
            print '   '+ key
            # only consider for renaming if we have more than one file
            # but also not more than 3 (than we don't now exactly what is
            # going on)
            if not len(values) > 1 or not len(values) < 4: continue

            # first pass over the files with same id
            # always use the mat file as a basis for renaming
            basisname = False
            for i in file_dict[key]:
                print '        ' + i
                if i.endswith('.mat'):
                    basisname = i.replace('.mat', '')
                    # and the short index
                    runid = '_'.join(basisname.split('_')[0:3])
                    db_index[basisname] = runid
            # if there is no mat file, take the name of the first we get
            if not basisname:
                i = file_dict[key].keys()[0]
                basisname = i.replace('.csv', '').replace('.log', '')
                # and the short index
                runid = '_'.join(basisname.split('_')[0:3])
                db_index[basisname] = runid
            # and also have the inverse index file, probably redundant....
            db_index_runid[runid] = basisname

            print
            # second pass for the actual renamed symlink
            for name, rootn in values.iteritems():
                ext = name[-4:len(name)]
                # extension can be blank for some log files
                if ext not in ['.log', '.csv', '.mat']:
                    newname = basisname + '.log'
                else:
                    newname = basisname + ext
                print '        ' + newname

                path_source = os.path.join('../../', rootn, name)
                path_target = os.path.join(path_db, newname)

                # collect all cases as simlinks in the database folder
                # this holds the lowest risk of destroying the actual data!
                os.symlink(path_source, path_target)

                # save in the df database
                # remove the root folder from the source path
                source_rel = os.path.commonprefix([path_source, source_folder])
                df_index['source'].append(source_rel)
                df_index['runid'].append(runid)
                df_index['caseid'].append(newname)

                ## do not rename a file if the target already exists
                ## based on stackoverflow answer
                #try:
                   #with open(root+'/'+newname) as f:
                       #print '        ' + name
                #except IOError:
                   ## it will raise an exception if it does not exist
                   ##os.rename(root+'/'+name, root+'/'+newname)
                   ## or just create a simlink in the big database folder
                   #print '        ' + newname

    # save in the root folder
    path_db = path_db.replace(db_id+'/', '')

    # first, update the existing file, so results of february and april merge
    # based on stackoverflow answer, check if the index file exists
    try:
        # if it exists, update the file first before saving
        FILE = open(path_db + 'db_index_%s.pkl' % db_id)
        db_index.update(pickle.load(FILE))
        FILE.close()
    except IOError:
        # no need to update an existing database file
        pass

    try:
        # if it exists, update the file first before saving
        FILE = open(path_db + 'db_index_%s_runid.pkl' % db_id)
        db_index_runid.update(pickle.load(FILE))
        FILE.close()
    except IOError:
        # no need to update an existing database file
        pass

    # and save the database index
    FILE = open(path_db + 'db_index_%s.pkl' % db_id, 'wb')
    pickle.dump(db_index, FILE, protocol=2)
    FILE.close()

    # and save the database index
    FILE = open(path_db + 'db_index_%s_runid.pkl' % db_id, 'wb')
    pickle.dump(db_index_runid, FILE, protocol=2)
    FILE.close()

def symlinks_to_dcsweep(source_folder, path_db, db_id):
    """
    The dc-sweep cases are already grouped in a different folder, now put them
    on the same pile as all the rest
    """
    path_db += db_id + '/'
    # create the folder if it doesn't exist
    try:
        os.mkdir(path_db)
    except OSError:
        pass

    # save a pickled dictionary that holds all the unique base names
    db_index = {}
    # and a short version where only the runid is taken
    db_index_runid = {}

    # becauase each case needs a unique run id
    alphabet = []
    # and make it go from aa, ab, ac, ... yz, zz
    for i in string.ascii_lowercase:
        for j in string.ascii_lowercase:
            alphabet.append('%s%s' % (i,j))

    # ignore the ones from February, they are with the alu blades
    folderignore = 'alublades'

    # fname  'Measurement_12-Apr-2012_DCycle_0.1_V_8_run_365.mat'
    # folder '2012-02-06_06_alublades'

    iis = {}
    for root, dirs, files in os.walk(source_folder, topdown=True):

        folder = root.split('/')[-1]
        if folder.find(folderignore) > -1 or len(folder) < 1:
            continue

        date = ''.join(folder.split('_')[0].split('-')[1:3])
        case = ('_'.join(folder.split('_')[1:])) + '_dcsweep'

        for fname in sorted(files):

            if fname.endswith('.log'):
                continue

            print fname, '  ->',
            fname_parts = fname.split('_')
            dc = fname_parts[3]
            wind = fname_parts[5]
            run = fname_parts[7].replace('.mat', '')
            run = format(int(run), '03.0f')

            try:
                iis[run]
            except KeyError:
                iis[run] = 0

            runa = run + alphabet[iis[run]]
            runid = '_'.join([date,'run',runa])
            iis[run] += 1

            # new '0209_run_020_15ms_dc10_stiffblades_pwm1000_cal_dashboard'
            new = '_'.join([date,'run',runa,wind+'ms','dc'+dc,case+'.mat'])
            print new

            # and make a OJF log file with only the wind speed in it
            try:
                # for some cases we actually have the source
                logsrc = root+'/'+'_'.join([date, 'run', run])+'.log'
                logdstname = new.replace('.mat', '.log')
                shutil.copy(logsrc, root+'/'+logdstname)
            except IOError:
                ojfline = '0.0	0.0	0.0	0.0	%s\n' % wind
                FILE = open(root+'/'+logdstname, 'w')
                FILE.writelines([ojfline]*30)

            # and make the symlinks
            # relative symbolic links: first two levels up
            root_ = os.path.join('../../', root)
            os.symlink(os.path.join(root_, fname), path_db+new)
            os.symlink(os.path.join(root_, logdstname), path_db+logdstname)

            # save in the index file
            db_index[new.replace('.mat', '')] = runid
            # and also have the inverse index file, probably redundant....
            db_index_runid[runid] = new.replace('.mat', '')

    # save in the root folder
    path_db = path_db.replace(db_id+'/', '')

    # first, update the existing file, so results of february and april merge
    try:
        # if it exists, update the file first before saving
        FILE = open(path_db + 'db_index_%s.pkl' % db_id)
        db_index_update = pickle.load(FILE)
        # overwrite the old entries with new ones! not the other way around
        db_index_update.update(db_index)
        FILE.close()
    except IOError:
        # no need to update an existing database file
        db_index_update = db_index

    try:
        # if it exists, update the file first before saving
        FILE = open(path_db + 'db_index_%s_runid.pkl' % db_id)
        db_index_runid_up = pickle.load(FILE)
        # overwrite the old entries with new ones! not the other way around
        db_index_runid_up.update(db_index_runid)
        FILE.close()
    except IOError:
        # no need to update an existing database file
        db_index_runid_up = db_index_runid

    # and save the database index
    FILE = open(path_db + 'db_index_%s.pkl' % db_id, 'wb')
    pickle.dump(db_index_update, FILE, protocol=2)
    FILE.close()

    # and save the database index
    FILE = open(path_db + 'db_index_%s_runid.pkl' % db_id, 'wb')
    pickle.dump(db_index_runid_up, FILE, protocol=2)
    FILE.close()


def convert_pkl_index_df(path_db, db_id='symlinks'):
    """
    Convert the pickled database index db_index and db_index_runid to a
    DataFrame. The database is a dictionary holding (basename,runid)
    key/value pairs (runid has it the other way around).
    Additionally, the file name is scanned for other known patterns such as
    dc, wind speeds, type of blades, type of run, etc. All these values
    are then placed in respective columns so you can more easily select only
    those cases you are interested in.
    """

#    path_db += db_id + '/'
    fname = os.path.join(path_db, 'db_index_%s' % db_id)

    with open(fname + '.pkl') as f:
        db_index = pickle.load(f)

    df_dict = {'basename':[], 'runid':[], 'dc':[], 'blades':[], 'yaw_mode':[],
               'run_type':[], 'rpm_change':[], 'coning':[], 'yaw_mode2':[],
               'the_rest':[], 'windspeed':[], 'sweepid':[], 'month':[],
               'day':[], 'runnr':[]}
    blades = set(['flexies', 'flex', 'stiffblades', 'stiff', 'samoerai',
                  'stffblades'])
    onoff = set(['spinup', 'spinupfast', 'spinuppartial', 'slowdown',
                 'speedup', 'shutdown', 'spinningdown', 'startup', 'speedup',
                 'speedingup'])
    allitems = set([])
    ignore = set(['basename', 'runid', 'month', 'day', 'runnr'])

    for basename, runid in db_index.iteritems():
        df_dict['basename'].append(basename)
        df_dict['runid'].append(runid)
        df_dict['runnr'].append(int(runid[9:12]))
        df_dict['month'].append(int(runid[:2]))
        df_dict['day'].append(int(runid[2:4]))

        # get as much out of the file name as possible
        items = basename.split('_')
        allitems = allitems | set(items)
        found = {k:False for k in df_dict.keys()}
        therest = []
        for k in items:
            if k == 'dcsweep':
                df_dict['run_type'].append(k)
                found['run_type'] = True
                if len(runid) > 13:
                    df_dict['runid'][-1] = runid[:-2]
                    df_dict['sweepid'].append(runid[-2:])
                    found['sweepid'] = True
            elif k.startswith('dc'):
                # in case that fails, we don't know: like when dc is
                # something like 0.65-0.70
                try:
                    df_dict['dc'].append(float(k.replace('dc', '')))
                except ValueError:
                    df_dict['dc'].append(-1.0)
                found['dc'] = True
            elif k in blades:
                if k == 'stffblades':
                    k = 'stiffblades'
                df_dict['blades'].append(k)
                found['blades'] = True
            elif k.find('yaw') > -1:
                if not found['yaw_mode']:
                    df_dict['yaw_mode'].append(k)
                    found['yaw_mode'] = True
                else:
                    df_dict['yaw_mode2'].append(k)
                    found['yaw_mode2'] = True
            elif k.find('coning') > -1:
                df_dict['coning'].append(k)
                found['coning'] = True
            elif k in onoff and not found['rpm_change']:
                df_dict['rpm_change'].append(k)
                found['rpm_change'] = True
            elif k[-2:] == 'ms':
                try:
                    df_dict['windspeed'].append(float(k[:-2]))
                except:
                    df_dict['windspeed'].append(-1.0)
                found['windspeed'] = True
            elif basename.find(k) < 0 or runid.find(k) < 0:
                therest.append(k)

        df_dict['the_rest'].append('_'.join(therest))
        found['the_rest'] = True

        for key, value in found.iteritems():
            if not value and key not in ignore:
                # to make sure dc items are floats (no mixing of datatypes)
                if key == 'dc' or key == 'windspeed':
                    df_dict[key].append(-1.0)
                else:
                    df_dict[key].append('')

    for k in sorted(allitems):
        print(k)

    misc.check_df_dict(df_dict)

    df = pd.DataFrame(df_dict)
    df.sort_values('basename', inplace=True)
    df.to_hdf(fname + '.h5', 'table', complevel=9)#, complib='blosc')
    df.to_csv(fname + '.csv', index=False)
    df.to_excel(fname + '.xlsx', index=True)


def dc_from_casename(case):
        # try to read the dc from the case file name
        items = case.split('_')
        for k in items:
            if k.startswith('dc'):
                # in case that fails, we don't know: like when dc is
                # something like 0.65-0.70
                try:
                    return float(k.replace('dc', ''))
                except ValueError:
                    return -1.0
        return np.nan


def build_stats_db(path_db, prefix, **kwargs):
    """
    Create the statistics for each OJF case in the index database
    =============================================================

    Scan through all cases in the db_index (each case should have symlinks to
    the results files in the symlink folder) and evaluate the mean values and
    the standard deviations of key parameters.

    Yaw laser and tower strain sensors are calibrated.

    Parameters
    ----------

    path_db : str
        Full path to the to be build database

    prefix : str
        Identifier for the database index

    output : str, default=prefix
        Identifier for the figures output path, and the db stats file

    calibrate : boolean, default=True
        Should the data be calibrated? Set to False if not.

    dashplot : boolean, default=False
        If True, a dashboard plot will be made for each case

    key_inc : list
        Keywords that should occur in the database, operator is AND

    resample : boolean, default=False

    """
    folder_df = kwargs.get('folder_df', 'data/calibrated/DataFrame/')
    folder_csv = kwargs.get('folder_csv', 'data/calibrated/CSV/')
    output = kwargs.get('output', prefix)
    dashplot = kwargs.get('dashplot', False)
    calibrate = kwargs.get('calibrate', True)
    key_inc = kwargs.get('key_inc', [])
    resample = kwargs.get('resample', False)
    save_df = kwargs.get('save_df', False)
    save_df_csv = kwargs.get('save_df_csv', False)
    continue_build = kwargs.get('continue_build', True)

    # initialize the database
    db = MeasureDb(prefix=prefix, path_db=path_db, load_index=True)
    db_index = db.index.index.tolist()

    # remove the files we've already done
    if continue_build:
        source_folder = os.path.join(folder_df)
        for root, dirs, files in os.walk(source_folder, topdown=True):
            for fname in files:
                db_index.pop(fname[:-3])
    # respath is where all the symlinks are
    respath = os.path.join(path_db, prefix + '/')

    # create the figure folder if it doesn't exist
    try:
        os.mkdir(os.path.join(path_db, 'figures_%s/' % output))
    except OSError:
        pass

    try:
        os.mkdir(folder_df)
    except OSError:
        pass

    try:
        os.mkdir(folder_csv)
    except OSError:
        pass

    nr, nrfiles = 0, len(db_index)

    # and cycle through all the files present
    for resfile in db_index:

        # only continue if all the keywords are present in the file name
        ignore = False
        for key in key_inc:
            if not resfile.find(key) > -1:
                ignore = True
        if ignore:
            continue

        # if we catch any error, ignore that file for now and go on
        nr += 1
        print
        print '=== %4i/%4i ' % (nr, nrfiles) + 67*'='
        print resfile
        res = ojfresult.ComboResults(respath, resfile, silent=True, sync=True)
        # just in case there is no dspace file, ignore it
        if res.nodspacefile:
            continue
        if calibrate:
            res.calibrate()

        # for the dc-sweep cases, ditch the first 4 seconds where the rotor
        # speed is still changing too much
        if res.dspace.campaign == 'dc-sweep':
            # 4 seconds of the ones lasting 12 seconds
            if res.dspace.time[-1] < 13.0:
                cutoff = 4.0
            else:
                cutoff = 12.0
            istart = res.dspace.sample_rate*cutoff
            res.dspace.data = res.dspace.data[istart:,:]
            res.dspace.time = res.dspace.time[istart:]-res.dspace.time[istart]

        if resample:
            res._resample()

        # make a dashboard plot
        if dashplot:
            res.dashboard_a3(os.path.join(path_db, 'figures_%s/' % output))

        df_res = res.to_df()
        # statistics
        db.add_stats(resfile, df_res)

        if save_df:
            ftarget = os.path.join(folder_df, resfile + '.h5')
            df_res.to_hdf(ftarget, 'table', complevel=9, complib='blosc')
        if save_df_csv:
            df_res.to_csv(os.path.join(folder_csv, resfile + '.csv'))

#        if nr > 100:
#            break

    # and save all the statistics
    db.save_stats(path_db=path_db, prefix=prefix, update=continue_build)


class MeasureDb(object):
    """Class to conviently select and load all measurements and their
    statistics.

    Members
    -------

    df_dict_mean, df_dict_min, df_dict_max, df_dict_std, df_dict_range

    df_mean, df_min, df_max, df_std, df_range
    """

    def __init__(self, prefix='symlinks_all', path_db='database/',
                 load_index=True):
        self.path_db = path_db
        if load_index:
            self.index_fname = os.path.join(path_db, 'db_index_%s.h5' % prefix)
            self.load_index()
            self.prefix = prefix
        self.stat_types = ['mean', 'min', 'max', 'std', 'range']

    def load_index(self, fname=None):
        """Load the index DataFrame. When fname is None, use self.index_fname.
        """
        if fname is None:
            fname = self.index_fname
        self.index = pd.read_hdf(fname, 'table')
        if not self.index.index.name == 'basename' and 'basename' in self.index:
            self.index.set_index('basename', inplace=True)
        self.index_cols = set(self.index.columns)
        # this is expensive, only worth when doing a LOT of lookups (>100)
#        self.index_names = set(self.index.tolist())

    def load_stats(self):
        for stat in self.stat_types:
            rpl = (self.prefix, stat)
            fname = os.path.join(self.path_db, 'db_stats_%s_%s.h5' % rpl)
            setattr(self, '%s_fname' % stat, fname)
            setattr(self, stat, pd.read_hdf(fname, 'table'))

    def load_measurement(self, fname):
        """Load a measurement data file.
        """
        if fname[-2:] == 'h5':
            return pd.read_hdf(fname, 'table')
        elif fname[-3:] == 'csv':
            return pd.read_csv(fname)
        elif fname[-4:] in ['.xls', 'xlsx']:
            return pd.read_excel(fname)
        else:
            raise ValueError('Provide either h5, csv, xls, or xlsx file.')

    def _init_df_dict_stats(self):
        """Initialize all columns for the stats DataFrames
        """

        # only take the unique entries, cnames contains all possible mappings
        dspace = ojfresult.DspaceMatFile(matfile=None)
        self.all_c_columns = list(set(dspace.cnames.values()))
        self.all_c_columns.append('time')
        # also add cnames for OJF
        ojf = ojfresult.OJFLogFile(ojffile=None)
        self.all_c_columns.extend(ojf.cnames)
        # and the blade strains
        blade = ojfresult.BladeStrainFile(None)
        self.all_c_columns.extend(blade.cnames)
        # create a DataFrame formatted dictionary for each of the stat types
        for stat in self.stat_types:
            setattr(self, 'df_dict_' + stat,
                    {col:[] for col in self.all_c_columns + ['index']})

    def _init_df_dict_index(self):
        """Initialize all columns for the index DataFrame
        """
        cols = self.index.columns.tolist() + ['index']
        self.df_dict_index = {k:[] for k in cols}

    def _add_stats_df_dict(self, resfile, df_stats, df_dict) :
        """Add df_stats to df_dict with resfile on the index column.

        Parameters
        ----------

        resfile : str
            name of the result file from the index

        df_stats : DataFrame
            Statistics of a measurement DataFrame (df_res)

        df_dict : dictionary
            DataFrame formatted dictionary of the to be added statistics.
            Should be df_dict_mean, df_dict_min, etc.
        """

        df_dict['index'].append(resfile)
        for col in df_stats.index:
            df_dict[col].append(df_stats[col])

        # and empty items for those for which there is no data
        for col in (set(self.all_c_columns) - set(df_stats.index.tolist())):
            if col == 'duty_cycle':
                dc = dc_from_casename(resfile)
                df_dict[col].append(dc)
            else:
                df_dict[col].append(np.nan)
        return df_dict

    def _add_index_df_dict(self, resfile, index_row):
        """Add a new entry to the index.
        """
        self.df_dict_index['index'].append(resfile)
        for col, value in index_row.items():
            self.df_dict_index[col].append(value)
        # add empty values for any missing keys
        for col in (set(self.index_cols) - set(index_row.keys())):
            self.df_dict_index[col].append(np.nan)

    def add_stats(self, resfile, df_res, index_row={}):
        """Add stats from DataFrame results to df statistics dictionaries,
        and save as attributes: df_dict_%s, using mean, min, max, std, range.
        Also add to the index database.

        Parameters
        ----------

        resfile : str
            Name used for the index entry (should be unique).

        df_res : DataFrame
            Measurement result DataFrame

        index_row : dict, default={}
            Entries for the index columns (except for the index of the index)
        """
        if not hasattr(self, 'df_dict_mean'):
            self._init_df_dict_stats()
        if not hasattr(self, 'df_dict_index'):
            self._init_df_dict_index()

        # complain if already exists in the index
        if (resfile in self.df_dict_mean) or (hasattr(self, 'mean') and \
            resfile in self.mean.index):
            raise IndexError('Index already exists in db: %s' % resfile)
        # add to the index
        self._add_index_df_dict(resfile, index_row)

        rf = resfile
        self._add_stats_df_dict(rf, df_res.mean(), self.df_dict_mean)
        self._add_stats_df_dict(rf, df_res.min(), self.df_dict_min)
        self._add_stats_df_dict(rf, df_res.max(), self.df_dict_max)
        self._add_stats_df_dict(rf, df_res.std(), self.df_dict_std)
        df_range = df_res.max() - df_res.min()
        self._add_stats_df_dict(rf, df_range, self.df_dict_range)

    def add_staircase_stats(self, resfile, df_res, arg_stair):
        """Add statistics for each of the given intervals. Also creates an
        index entry for each step.
        """

        index_row = self.index.loc[resfile].to_dict()

        for k in range(arg_stair.shape[1]):
            i1 = arg_stair[0,k]
            i2 = arg_stair[1,k]
            # add stair step number to resfile name to create unique index
            resfile_step = '%s_step_%02i' % (resfile, k)
            index_row['sweepid'] = '%i_%i' % (i1, i2)
            index_row['run_type'] = 'stair_step'
            # and add to the database dict
            self.add_stats(resfile_step, df_res[i1:i2], index_row=index_row)

    def _dict2df(self):
        """Convert the DataFrame formatted dictionaries to DataFrames. If
        the conversion failes the dictionary has wrongly composed. Perform
        post-mortem checks for debugging purposes.

        Attributes df_mean are set based on df_dict_mean.

        New indices are in the index_up attribute.

        Besides mean, same is done for min, max, std and range.
        """

        for stat in self.stat_types:
            if not hasattr(self, 'df_dict_%s' % stat):
                continue
            df_dict = getattr(self, 'df_dict_%s' % stat)
            try:
                df = pd.DataFrame(df_dict)
                df.set_index('index', inplace=True)
                setattr(self, 'df_%s' % stat, df)
            except ValueError as e:
                print('df_dict_%s' % stat)
                misc.check_df_dict(df_dict)
                raise e

        try:
            self.index_up = pd.DataFrame(self.df_dict_index)
            self.index_up.set_index('index', inplace=True)
        except ValueError as e:
            print('df_dict_index')
            misc.check_df_dict(self.df_dict_index)
            raise e

    def add_df_dict2stat(self, update=True):
        """Convert the df_dict_mean stat (created with add_stats or
        add_staircase_stats) to df_mean. When update is True, append df_mean
        to mean and index. Copy of added indices are in the index_up attribute.

        Besides mean, same is done for min, max, std and range.
        """
        self._dict2df()

        for stat in self.stat_types:
            df_add = getattr(self, 'df_%s' % stat)
            if update:
                if not hasattr(self, stat):
                    self.load_stats()
                df_stat = getattr(self, stat)
                df_stat = pd.concat([df_stat, df_add])
                setattr(self, stat, df_stat)

        # also add the new cases to the index
        if update:
            if not hasattr(self, 'index'):
                self.load_index()
            self.index = pd.concat([self.index, self.index_up])

    def save(self, complib='blosc'):
        """Save curren index and stats DataFrames.
        """
        print('saving: %s ...' % self.index_fname)
        self.index.to_hdf(self.index_fname, 'table', compression=9,
                          complib=complib)
        for stat in self.stat_types:
            df_stat = getattr(self, stat)
            stat_fname = getattr(self, '%s_fname' % stat)
            print('saving: %s ...' % stat_fname)
            df_stat.to_hdf(stat_fname, 'table', compression=9, complib=complib)

    def remove_from_stats_index(self, indices):
        """Remove given indices from the index and the different stats df.
        """
        ind_remains = set(self.index.index.tolist()) - set(indices.tolist())
        self.index = self.index.loc[ind_remains]
        for stat in self.stat_types:
            df_stat = getattr(self, stat)
            setattr(self, stat, df_stat.loc[ind_remains])


def tsr(df):
    R = ojf_post.model.blade_radius
    return R*df.rpm*np.pi/(df.wind_speed*30.0)


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
    # and nodfrmalize to get the thrust coefficient
    ct = thrust / (0.5*rho*V*V*ojf_post.model.A)

    return ct


###############################################################################
### PLOTS
###############################################################################

def plot_voltage_current():
    """
    Establish connection between measured current and rotor speed.
    Consider all the cases, aero side is completely irrelevant here.

    Also at linear fits to the data
    """

    def fit(x, y, deg, res=50):
        pol = np.polyfit(x,y, deg)
        # but generate polyval on equi spaced x grid
        x_grid = np.linspace(x[0], x[-1], res)
        return  x_grid, np.polyval(pol, x_grid)

    figpath = 'figures/overview/'
    scale = 1.5

    prefix = 'symlinks_all'
    db = MeasureDb(prefix='symlinks_all', path_db='database/')
    db.load_stats()
    #df_mean.set_index('index', inplace=True)
#    df_std = pd.read_hdf('database/db_stats_symlinks_all_std.h5', 'table')
#    sel_std = df_std[(df_std.yaw_angle > -1.2) & (df_std.yaw_angle < 1.2)]

    sel02_dc0 = db.index[(db.index.month==2) & (db.index.dc==0)]
    sel02_dc1 = db.index[(db.index.month==2) & (db.index.dc==1)]
    sel04_dc0 = db.index[(db.index.month==4) & (db.index.dc==0)]
    sel04_dc1 = db.index[(db.index.month==4) & (db.index.dc==1)]

    m2_dc0 = db.mean[db.mean.index.isin(sel02_dc0.index.tolist())]
    m2_dc1 = db.mean[db.mean.index.isin(sel02_dc1.index.tolist())]
    m4_dc0 = db.mean[db.mean.index.isin(sel04_dc0.index.tolist())]
    m4_dc1 = db.mean[db.mean.index.isin(sel04_dc1.index.tolist())]

    # --------------------------------------------------------------------

    figfile = '%s-volt-vs-current' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
              grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
              wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(m2_dc0.voltage_filt, m2_dc0.current_filt, 'rs', label='dc0 02')
    ax1.plot(m2_dc1.voltage_filt, m2_dc1.current_filt, 'gs', label='dc1 02')
    ax1.plot(m4_dc0.voltage_filt, m4_dc0.current_filt, 'bd', label='dc0 04')
    ax1.plot(m4_dc1.voltage_filt, m4_dc1.current_filt, 'yd', label='dc1 04')
    #ax1.plot(stiff_dc0[0,:], stiff_dc0[1,:], 'ro', label='dc0 stiff')
    #ax1.plot(flex_dc1[0,:],  flex_dc1[1,:],  'g*', label='dc1 flex')
    #ax1.plot(stiff_dc1[0,:], stiff_dc1[1,:], 'g^', label='dc1 stiff')
    #ax1.plot(dc5[0,:], dc5[1,:], 'bd', label='dc0.5')
    #ax1.plot(wind, rpm, 'b*', label='unknown')

    # fitting the data
#    ax1.plot(dc0_02[3,:], fit(dc0_02[3,:], dc0_02[4,:], 1), 'r--')
#    ax1.plot(dc1_02[3,:], fit(dc1_02[3,:], dc1_02[4,:], 1), 'g--')
#    ax1.plot(dc0_04[3,:], fit(dc0_04[3,:], dc0_04[4,:], 1), 'b--')
#    ax1.plot(dc1_04[3,:], fit(dc1_04[3,:], dc1_04[4,:], 1), 'y--')

    ax1.legend(loc='upper right')
    ax1.set_title('Feb and Apr, all', size=14*scale)
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Current [A]')
    ax1.set_xlim([-1, 45])
    ax1.set_ylim([-0.5, 14])
    ax1.grid(True)
    pa4.save_fig()


    # --------------------------------------------------------------------
    figfile = '%s-rpm-vs-current' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(m2_dc0.rpm, m2_dc0.current_filt, 'rs', label='dc0 02')
    ax1.plot(m2_dc1.rpm, m2_dc1.current_filt, 'gs', label='dc1 02')
    ax1.plot(m4_dc0.rpm, m4_dc0.current_filt, 'bd', label='dc0 04')
    ax1.plot(m4_dc1.rpm, m4_dc1.current_filt, 'yd', label='dc1 04')
    #ax1.plot(stiff_dc0[0,:], stiff_dc0[1,:], 'ro', label='dc0 stiff')
    #ax1.plot(flex_dc1[0,:],  flex_dc1[1,:],  'g*', label='dc1 flex')
    #ax1.plot(stiff_dc1[0,:], stiff_dc1[1,:], 'g^', label='dc1 stiff')
    #ax1.plot(dc5[0,:], dc5[1,:], 'bd', label='dc0.5')
    #ax1.plot(wind, rpm, 'b*', label='unknown')

    # fitting the data
#    ax1.plot(dc0_02[1,:], fit(dc0_02[1,:], dc0_02[4,:], 1), 'r--')
#    ax1.plot(dc1_02[1,:], fit(dc1_02[1,:], dc1_02[4,:], 1), 'g--')
#    ax1.plot(dc0_04[1,:], fit(dc0_04[1,:], dc0_04[4,:], 1), 'b--')
#    ax1.plot(dc1_04[1,:], fit(dc1_04[1,:], dc1_04[4,:], 1), 'y--')

    ax1.legend(loc='upper left')
    ax1.set_title('Feb and Apr, all', size=14*scale)
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Current [A]')
    #ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------
    figfile = '%s-rpm-vs-volt' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(m2_dc0.rpm, m2_dc0.voltage_filt, 'rs', label='dc0 02')
    ax1.plot(m2_dc1.rpm, m2_dc1.voltage_filt, 'gs', label='dc1 02')
    ax1.plot(m4_dc0.rpm, m4_dc0.voltage_filt, 'bd', label='dc0 04')
    ax1.plot(m4_dc1.rpm, m4_dc1.voltage_filt, 'yd', label='dc1 04')
    #ax1.plot(stiff_dc0[0,:], stiff_dc0[1,:], 'ro', label='dc0 stiff')
    #ax1.plot(flex_dc1[0,:],  flex_dc1[1,:],  'g*', label='dc1 flex')
    #ax1.plot(stiff_dc1[0,:], stiff_dc1[1,:], 'g^', label='dc1 stiff')
    #ax1.plot(dc5[0,:], dc5[1,:], 'bd', label='dc0.5')
    #ax1.plot(wind, rpm, 'b*', label='unknown')

    # fitting the data
#    ax1.plot(dc0_02[1,:], fit(dc0_02[1,:], dc0_02[3,:], 1), 'r--')
#    ax1.plot(dc1_02[1,:], fit(dc1_02[1,:], dc1_02[3,:], 1), 'g--')
#    ax1.plot(dc0_04[1,:], fit(dc0_04[1,:], dc0_04[3,:], 1), 'b--')
#    ax1.plot(dc1_04[1,:], fit(dc1_04[1,:], dc1_04[3,:], 1), 'y--')

    ax1.legend(loc='upper left')
    ax1.set_title('Feb and Apr, all', size=14*scale)
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Voltage [V]')
    ax1.set_xlim([0, 1000])
    ax1.set_ylim([-1, 50])
    ax1.grid(True)
    pa4.save_fig()


def plot_rpm_wind():
    """
    """

    def fit(x, y, deg, res=50):
        pol = np.polyfit(x,y, deg)
        # but generate polyval on equi spaced x grid
        x_grid = np.linspace(x[0], x[-1], res)
        return  x_grid, np.polyval(pol, x_grid)

    prefix = 'symlinks_all'
    db = MeasureDb(prefix='symlinks_all', path_db='database/')
    db.load_stats()

    figpath = 'figures/overview/'
    scale = 1.5

    # --------------------------------------------------------------------
    # only limited yaw angles
    myaw0 = db.mean[(db.std.yaw_angle>-1.0) & (db.std.yaw_angle<0.5)]
    iyaw0 = db.index[db.index.index.isin(myaw0.index.tolist())]
    # no coning, and no free yawing, and no big rpm changes
    iyaw0 = iyaw0[(iyaw0.coning=='') & (iyaw0.rpm_change=='') &
                  (iyaw0.yaw_mode.str.find('free')<0)]

    iflex_dc0 = iyaw0[(iyaw0.blades=='flex') & (iyaw0.dc==0)]
    istiff_dc0 = iyaw0[(iyaw0.blades=='stiff') & (iyaw0.dc==0)]
    iflex_dc1 = iyaw0[(iyaw0.blades=='flex') & (iyaw0.dc==1)]
    istiff_dc1 = iyaw0[(iyaw0.blades=='stiff') & (iyaw0.dc==1)]
    idc5 = iyaw0[(iyaw0.dc==0.5)]

    flex_dc0 = myaw0[myaw0.index.isin(iflex_dc0.index.tolist())]
    stiff_dc0 = myaw0[myaw0.index.isin(istiff_dc0.index.tolist())]
    flex_dc1 = myaw0[myaw0.index.isin(iflex_dc1.index.tolist())]
    stiff_dc1 = myaw0[myaw0.index.isin(istiff_dc1.index.tolist())]
    dc5 = myaw0[myaw0.index.isin(idc5.index.tolist())]

#    ex = ['coning', 'free']
#    flex_dc0, ca, hd = db.select(['02','04'], ['flex'], ex,
#                                 valuedict={'dc':0, 'yaw':[-1.0,0.5]})
#    stiff_dc0,ca, hd = db.select(['02','04'], ['stiff'],ex,
#                                 valuedict={'dc':0, 'yaw':[-1.0,0.5]})
#    flex_dc1, ca, hd = db.select(['02','04'], ['flex'], ex,
#                                 valuedict={'dc':1, 'yaw':[-1.0,0.5]})
#    stiff_dc1,ca, hd = db.select(['02','04'], ['stiff'],ex,
#                                 valuedict={'dc':1, 'yaw':[-1.0,0.5]})
#    dc5, ca, hd = db.select(['02','04'], [], ex,
#                            valuedict={'dc':0.5, 'yaw':[-1.0,0.5]})

    figfile = '%s-rpm-vs-wind-dc0-dc1' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(flex_dc0.wind_speed,  flex_dc0.rpm,  'rs', label='dc0 flex')
    ax1.plot(stiff_dc0.wind_speed, stiff_dc0.rpm, 'ro', label='dc0 stiff')
    ax1.plot(flex_dc1.wind_speed,  flex_dc1.rpm,  'g>', label='dc1 flex')
    ax1.plot(stiff_dc1.wind_speed, stiff_dc1.rpm, 'g<', label='dc1 stiff')
    ax1.plot(dc5.wind_speed, dc5.rpm, 'bd', label='dc0.5')
    #ax1.plot(wind, rpm, 'b*', label='unknown')

#    # fitting the data
#    x, y = fit(flex_dc0[0,:], flex_dc0[1,:], 1)
#    ax1.plot(x, y, 'r--', label='dc0 flex')
#    x, y = fit(stiff_dc0[0,:], stiff_dc0[1,:], 1)
#    ax1.plot(x, y, 'r--', label='dc0 stiff')
#    x, y = fit(flex_dc1[0,:],  flex_dc1[1,:], 1)
#    ax1.plot(x, y, 'g--', label='dc1 flex')
#    x, y = fit(stiff_dc1[0,:], stiff_dc1[1,:], 1)
#    ax1.plot(x, y, 'g--', label='dc1 stiff')
#    x, y = fit(dc5[0,:], dc5[1,:], 1)
#    ax1.plot(x, y, 'b--', label='dc0.5')

    # plot the tip speed ratio's as contour lines on the background
    # iso TSR lines
    RPMs = np.arange(0, 1300, 100)
    Vs = np.arange(4,20,1)
    R = ojf_post.model.blade_radius
    RPM_grid, V_grid = np.meshgrid(RPMs, Vs)
    TSR = R*RPM_grid*np.pi/(V_grid*30.0)
    contours = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    cs = ax1.contour(V_grid, RPM_grid, TSR, contours[::-1], colors='grey',
                     linewidth=0.5, linestyles='dashdot', label='TSR')
    # set the labels
    lablocs = [(10.5,1100), (12,1150), (12,1000), (12,850), (12, 700),
               (  12, 580), (12, 420), (12, 300), (16,200)]
    ax1.clabel(cs, fontsize=9*scale, inline=1, fmt='%1.0f', manual=lablocs,
               colors='k')

    ax1.legend(loc='upper right')
    ax1.set_title('Feb and Apr, fixed zero yaw', size=14*scale)
    ax1.set_xlabel('Wind speed [m/s]')
    ax1.set_ylabel('Rotor speed [RPM]')
    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------

    idc_p0 = iyaw0[(iyaw0.dc>-0.1) & (iyaw0.dc<0.25)]
    idc_p1 = iyaw0[(iyaw0.dc>0.25) & (iyaw0.dc<0.50)]
    idc_p2 = iyaw0[(iyaw0.dc>0.50) & (iyaw0.dc<0.75)]
    idc_p3 = iyaw0[(iyaw0.dc>0.75) & (iyaw0.dc<1.1)]

    dc_p0 = myaw0[myaw0.index.isin(idc_p0.index.tolist())]
    dc_p1 = myaw0[myaw0.index.isin(idc_p1.index.tolist())]
    dc_p2 = myaw0[myaw0.index.isin(idc_p2.index.tolist())]
    dc_p3 = myaw0[myaw0.index.isin(idc_p3.index.tolist())]

    figfile = '%s-rpm-vs-wind-dc-all' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=1.0,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(dc_p0.wind_speed,  dc_p0.rpm, 'rs', label='$0.00 \leq dc < 0.25$')
    ax1.plot(dc_p1.wind_speed,  dc_p1.rpm, 'g*', label='$0.25 \leq dc < 0.50$')
    ax1.plot(dc_p2.wind_speed,  dc_p2.rpm, 'bd', label='$0.50 \leq dc < 0.75$')
    ax1.plot(dc_p3.wind_speed,  dc_p3.rpm,'y^',label='$0.75 \leq dc \leq 1.00$')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.14,1.0))
    ax1.set_xlabel('Wind speed [m/s]')
    ax1.set_ylabel('Rotor speed [RPM]')
    ax1.set_title('Feb and Apr, fixed zero yaw', size=14*scale)
    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------
    # and now for each wind speed, see the dc-rpm plot

    figfile = '%s-rpm-vs-dc-all' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    colors = ['rs', 'bo', 'g*', 'y^', 'gd', 'm*', 'ys', '', '']

    for k in range(5,12):
        low = k - 0.2
        up = k + 0.2
        idc = iyaw0[(iyaw0.dc>low) & (iyaw0.dc<up)]
        dc = myaw0[myaw0.index.isin(idc.index.tolist())]
        label = '$%i m/s$' % k
        ax1.plot(dc.duty_cycle, dc.rpm, colors[k-5], label=label)

    ax1.legend(bbox_to_anchor=(1.05,1.1), ncol=3)
    ax1.set_xlabel('duty cycle [-]]')
    ax1.set_ylabel('Rotor speed [RPM]')
    ax1.set_xlim([-0.1, 1.1])
    ax1.grid(True)
    pa4.save_fig()


def plot_rpm_vs_towerstrain(prefix):
    """
    Plot for several different wind speeds the tower strain SS and FA as
    function or rotor speed. Is the tower load the same in February and
    April? However, ignore February all together because we don't have
    the data recorded for some very strange reason...after the analogue
    filter was installed the recording stopped.
    """

    cal = True
    path_db = PATH_DB
    db = ojf_db(prefix, debug=True, path_db=path_db)

    figpath = 'figures/overview/'
    scale = 1.5

    ex = ['coning', 'free', 'samoerai']
    apr10,ca,hd = db.select(['04'], [], ex,
                            valuedict={'wind':[9.82, 10.18],'yaw':[-1.0,0.5]})
    apr9,ca,hd = db.select(['04'], [], ex,
                           valuedict={'wind':[8.82, 9.18],'yaw':[-1.0,0.5]})
    apr8,ca,hd = db.select(['04'], [], ex,
                           valuedict={'wind':[7.82, 8.18],'yaw':[-1.0,0.5]})
    apr7,ca,hd = db.select(['04'], [], ex,
                           valuedict={'wind':[6.82, 7.18],'yaw':[-1.0,0.5]})
    apr6,ca,hd = db.select(['04'], [], ex,
                           valuedict={'wind':[5.82, 6.18],'yaw':[-1.0,0.5]})
    apr5,ca,hd = db.select(['04'], [], ex,
                           valuedict={'wind':[4.82, 5.18],'yaw':[-1.0,0.5]})
#    apr4,ca,hd = db.select(['04'], [], ex, valuedict={'wind':[3.82, 4.18]})
    ifa = hd['FA']
    iss = hd['SS']
    irpm = hd['RPM']
    iyaw = hd['yaw']
    # --------------------------------------------------------------------
    # RPM vs FA
    # --------------------------------------------------------------------
    figfile = '%s-rpm-vs-towerstrain-FA-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(apr10[irpm,:], apr10[ifa,:], 'bo', label='10 m/s')
    ax1.plot(apr9[irpm,:], apr9[ifa,:], 'rs', label='9 m/s')
    ax1.plot(apr8[irpm,:], apr8[ifa,:], 'gv', label='8 m/s')
    ax1.plot(apr7[irpm,:], apr7[ifa,:], 'm<', label='7 m/s')
    ax1.plot(apr6[irpm,:], apr6[ifa,:], 'c^', label='6 m/s')
    ax1.plot(apr5[irpm,:], apr5[ifa,:], 'y>', label='5 m/s')
#    ax1.plot(apr4[irpm,:], apr4[ifa,:], 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed zero yaw', size=14*scale)
    ax1.set_xlabel('rotor speed [RPM]')
    if cal:
        ax1.set_ylabel('tower base bending FA [Nm]')
    else:
        ax1.set_ylabel('tower base strain FA [raw]')
#    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()
    # --------------------------------------------------------------------
    # RPM vs SS
    # --------------------------------------------------------------------
    figfile = '%s-rpm-vs-towerstrain-SS-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(apr10[irpm,:], apr10[iss,:], 'bo', label='10 m/s')
    ax1.plot(apr9[irpm,:], apr9[iss,:], 'rs', label='9 m/s')
    ax1.plot(apr8[irpm,:], apr8[iss,:], 'gv', label='8 m/s')
    ax1.plot(apr7[irpm,:], apr7[iss,:], 'm<', label='7 m/s')
    ax1.plot(apr6[irpm,:], apr6[iss,:], 'c^', label='6 m/s')
    ax1.plot(apr5[irpm,:], apr5[iss,:], 'y>', label='5 m/s')
#    ax1.plot(apr4[irpm,:], apr4[iss,:], 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed yaw', size=14*scale)
    ax1.set_xlabel('rotor speed [RPM]')
    if cal:
        ax1.set_ylabel('tower base bending SS [Nm]')
    else:
        ax1.set_ylabel('tower base strain SS [raw]')
#    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()
    # --------------------------------------------------------------------
    # FA vs SS
    # --------------------------------------------------------------------
    figfile = '%s-FA-vs-SS-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(apr10[ifa,:], apr10[iss,:], 'bo', label='10 m/s')
    ax1.plot(apr9[ifa,:], apr9[iss,:], 'rs', label='9 m/s')
    ax1.plot(apr8[ifa,:], apr8[iss,:], 'gv', label='8 m/s')
    ax1.plot(apr7[ifa,:], apr7[iss,:], 'm<', label='7 m/s')
    ax1.plot(apr6[ifa,:], apr6[iss,:], 'c^', label='6 m/s')
    ax1.plot(apr5[ifa,:], apr5[iss,:], 'y>', label='5 m/s')
#    ax1.plot(apr4[irpm,:], apr4[iss,:], 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed zero yaw', size=14*scale)
    if cal:
        ax1.set_xlabel('tower base bending FA [Nm]')
        ax1.set_ylabel('tower base bending SS [Nm]')
    else:
        ax1.set_xlabel('tower base strain FA [raw]')
        ax1.set_ylabel('tower base strain SS [raw]')
#    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()
    # --------------------------------------------------------------------
    # yaw angle vs SS
    # --------------------------------------------------------------------
    figfile = '%s-yaw-vs-SS-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(apr10[iyaw,:], apr10[iss,:], 'bo', label='10 m/s')
    ax1.plot(apr9[iyaw,:], apr9[iss,:], 'rs', label='9 m/s')
    ax1.plot(apr8[iyaw,:], apr8[iss,:], 'gv', label='8 m/s')
    ax1.plot(apr7[iyaw,:], apr7[iss,:], 'm<', label='7 m/s')
    ax1.plot(apr6[iyaw,:], apr6[iss,:], 'c^', label='6 m/s')
    ax1.plot(apr5[iyaw,:], apr5[iss,:], 'y>', label='5 m/s')
#    ax1.plot(apr4[irpm,:], apr4[iss,:], 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed yaw', size=14*scale)
    if cal:
        ax1.set_xlabel('yaw angle [deg]')
        ax1.set_ylabel('tower base bending SS [Nm]')
    else:
        ax1.set_xlabel('tower base strain FA [raw]')
        ax1.set_ylabel('tower base strain SS [raw]')
#    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()
    # ========================================================================


def plot_rpm_vs_tower_allfeb(prefix):
    """
    It looks like something went wrong with the tower strain in February??
    After the installation of the analogue filters, no more strain signal...
    """

    path_db = PATH_DB
    db = ojf_db(prefix, debug=True, path_db=path_db)

    figpath = 'figures/overview/'
    scale = 1.5

    ex = []
    inc = []
    feb, ca, headers = db.select(['02'], inc, ex, valuedict={})
    ifa = headers['FA']
    iss = headers['SS']
    irpm = headers['RPM']
    # --------------------------------------------------------------------
    figfile = '%s-rpm-vs-towerstrain-cal-feb-all' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(feb[irpm,:], feb[ifa,:], 'ks', label='fa')
    ax1.plot(feb[irpm,:], feb[iss,:], 'bo', label='ss')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('all Feb, calibrated', size=14*scale)
    ax1.set_xlabel('rotor speed [RPM]')
    ax1.set_ylabel('tower base bending moment [Nm]')
    ax1.grid(True)
    pa4.save_fig()

def plot_rpm_vs_blade(prefix, blade):
    """
    """

    figpath = 'figures/overview/'
    scale = 1.5

    def doplot(figfile, iblade, title, ylabel):

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

    # =======================================================================
    # FEB AND APRIL
    db = ojf_db(prefix, debug=True, path_db=path_db)
    inc = [blade]
    exc = ['coning', 'samoerai']
    std = {'RPM':[0,10], 'yaw':[0, 0.5], 'wind':[0, 0.1]}
    apr10,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                            valuedict={'wind':[9.82, 10.18],'yaw':[-1.0,0.5]})
    apr9,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[8.82, 9.18],'yaw':[-1.0,0.5]})
    apr8,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[7.82, 8.18],'yaw':[-1.0,0.5]})
    apr7,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[6.82, 7.18],'yaw':[-1.0,0.5]})
    apr6,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[5.82, 6.18],'yaw':[-1.0,0.5]})
    apr5,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[4.82, 5.18],'yaw':[-1.0,0.5]})
    irpm = hd['RPM']

    # --------------------------------------------------------------------
    # BLADE1_root-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B1_root_%s-vs-RPM-feb-april' % (prefix, blade)
    title = '%s Feb, Apr, 0 deg fixed yaw' % blade
    ylabel = 'Blade 1 root bending [Nm]'
    iblade = hd['B1 root']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE1_30-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B1_30_%s-vs-RPM-feb-april' % (prefix, blade)
    ylabel = 'Blade 1 30\% bending [Nm]'
    iblade = hd['B1 30']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE2_root-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B2_root_%s-vs-RPM-feb-april' % (prefix, blade)
    ylabel = 'Blade 2 root bending [Nm]'
    iblade = hd['B2 root']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE2_30-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B2_30_%s-vs-RPM-feb-april' % (prefix, blade)
    ylabel = 'Blade 2 30\% bending [Nm]'
    iblade = hd['B2 30']
    doplot(figfile, iblade, title, ylabel)


def plot_ct_vs_lambda(blades='straight'):
    """
    Have all the relevant stuff plotted versus the tip speed ratio lambda!
    That will gave a better understanding of the stuff
    """

    figpath = 'figures/overview/'
    scale = 1.5

    prefix = 'symlinks_all'
    db = MeasureDb(prefix=prefix, path_db='database/')
    db.load_stats()
    mbase = db.mean[(db.std.rpm>=0.0) & (db.std.rpm<10.0) &
                    (db.std.yaw_angle>=0.0) & (db.std.yaw_angle<0.5) &
                    (db.std.wind_speed>=0.0) & (db.std.wind_speed<0.1) &
                    (db.mean.yaw_angle>-1.0) & (db.mean.yaw_angle<1.0)]
    if blades == 'straight':
        ibase = db.index[(db.index.coning=='') & (db.index.blades!='samoerai') &
                         (db.index.month==4)]
    elif blades == 'swept':
        ibase = db.index[(db.index.coning=='') & (db.index.blades=='samoerai') &
                         (db.index.month==4)]
    else:
        raise(ValueError, 'blades should be straight or swept')
    mbase = mbase[mbase.index.isin(ibase.index.tolist())]

    apr10 = mbase[(mbase.wind_speed>9.82) & (mbase.wind_speed<10.18)]
    apr9 = mbase[(mbase.wind_speed>8.82) & (mbase.wind_speed<9.18)]
    apr8 = mbase[(mbase.wind_speed>7.82) & (mbase.wind_speed<8.18)]
    apr7 = mbase[(mbase.wind_speed>6.82) & (mbase.wind_speed<7.18)]
    apr6 = mbase[(mbase.wind_speed>5.82) & (mbase.wind_speed<6.18)]
    apr5 = mbase[(mbase.wind_speed>4.82) & (mbase.wind_speed<5.18)]

    # --------------------------------------------------------------------
    # CT-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-ct-vs-lambda-april-blades-%s' % (prefix, blades)
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(tsr(apr10), ct(apr10), 'bo', label='10 m/s')
    ax1.plot(tsr(apr9), ct(apr9), 'rs', label='9 m/s')
    ax1.plot(tsr(apr8), ct(apr8), 'gv', label='8 m/s')
    ax1.plot(tsr(apr7), ct(apr7), 'm<', label='7 m/s')
    ax1.plot(tsr(apr6), ct(apr6), 'c^', label='6 m/s')
    ax1.plot(tsr(apr5), ct(apr5), 'y>', label='5 m/s')
#    ax1.plot(apr4), apr4), 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, zero yaw, %s blades' % blades, size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('thrust coefficient $C_T$')
    ax1.set_xlim([0, 9])
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------
    # THRUST-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-fa-vs-lambda-april-blades-%s' % (prefix, blades)
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(tsr(apr10), apr10.tower_strain_fa, 'bo', label='10 m/s')
    ax1.plot(tsr(apr9), apr9.tower_strain_fa, 'rs', label='9 m/s')
    ax1.plot(tsr(apr8), apr8.tower_strain_fa, 'gv', label='8 m/s')
    ax1.plot(tsr(apr7), apr7.tower_strain_fa, 'm<', label='7 m/s')
    ax1.plot(tsr(apr6), apr6.tower_strain_fa, 'c^', label='6 m/s')
    ax1.plot(tsr(apr5), apr5.tower_strain_fa, 'y>', label='5 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, zero yaw, %s blades' % blades, size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('Tower FA bending [Nm]')
    ax1.set_xlim([0, 9])
    ax1.grid(True)
    pa4.save_fig()


def plot_blade_vs_lambda(prefix, blade):
    """
    Plot blade loads as function of tip speed ratio lambda
    """

    figpath = 'figures/overview/'
    scale = 1.5

    def doplot(figfile, iblade, title, ylabel):

        pa4 = plotting.A4Tuned(scale=scale)
        pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                       grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                       wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
        ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

        ax1.plot(db.tsr(apr10),apr10[iblade,:],'bo', label='10 m/s')
        ax1.plot(db.tsr(apr9), apr9[iblade,:], 'rs', label='9 m/s')
        ax1.plot(db.tsr(apr8), apr8[iblade,:], 'gv', label='8 m/s')
        ax1.plot(db.tsr(apr7), apr7[iblade,:], 'm<', label='7 m/s')
        ax1.plot(db.tsr(apr6), apr6[iblade,:], 'c^', label='6 m/s')
        ax1.plot(db.tsr(apr5), apr5[iblade,:], 'y>', label='5 m/s')

        leg = ax1.legend(loc='best')
        leg.get_frame().set_alpha(0.5)
        ax1.set_title(title, size=14*scale)
        ax1.set_xlabel('tip speed ratio $\lambda$')
        ax1.set_ylabel(ylabel)
        ax1.grid(True)
        pa4.save_fig()

    path_db = PATH_DB
    db = ojf_db(prefix, debug=True, path_db=path_db)
    inc = [blade]
    exc = ['coning', 'samoerai']
    std = {'RPM':[0,10], 'yaw':[0, 0.5], 'wind':[0, 0.1]}
    apr10,ca,hd = db.select(['04'], inc, exc, values_std=std,
                            valuedict={'wind':[9.82, 10.18],'yaw':[-1.0,0.5]})
    apr9,ca,hd = db.select(['04'], inc, exc, values_std=std,
                           valuedict={'wind':[8.82, 9.18],'yaw':[-1.0,0.5]})
    apr8,ca,hd = db.select(['04'], inc, exc, values_std=std,
                           valuedict={'wind':[7.82, 8.18],'yaw':[-1.0,0.5]})
    apr7,ca,hd = db.select(['04'], inc, exc, values_std=std,
                           valuedict={'wind':[6.82, 7.18],'yaw':[-1.0,0.5]})
    apr6,ca,hd = db.select(['04'], inc, exc, values_std=std,
                           valuedict={'wind':[5.82, 6.18],'yaw':[-1.0,0.5]})
    apr5,ca,hd = db.select(['04'], inc, exc, values_std=std,
                           valuedict={'wind':[4.82, 5.18],'yaw':[-1.0,0.5]})
#    apr4,ca,hd = db.select(['04'], [], ex, valuedict={'wind':[3.82, 4.18]})

    # --------------------------------------------------------------------
    # BLADE1_root-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B1_root_%s-vs-lambda-april' % (prefix, blade)
    title = '%s Apr, 0 deg fixed yaw' % blade
    ylabel = 'Blade 1 root bending [Nm]'
    iblade = hd['B1 root']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE1_30-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B1_30_%s-vs-lambda-april' % (prefix, blade)
    ylabel = 'Blade 1 30\% bending [Nm]'
    iblade = hd['B1 30']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE2_root-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B2_root_%s-vs-lambda-april' % (prefix, blade)
    ylabel = 'Blade 2 root bending [Nm]'
    iblade = hd['B2 root']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE2_30-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B2_30_%s-vs-lambda-april' % (prefix, blade)
    ylabel = 'Blade 2 30\% bending [Nm]'
    iblade = hd['B2 30']
    doplot(figfile, iblade, title, ylabel)

    # =======================================================================
    # FEB AND APRIL
    db = ojf_db(prefix, debug=True, path_db=path_db)
    inc = [blade]
    exc = ['coning', 'samoerai']
    std = {'RPM':[0,10], 'yaw':[0, 0.5], 'wind':[0, 0.1]}
    apr10,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                            valuedict={'wind':[9.82, 10.18],'yaw':[-1.0,0.5]})
    apr9,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[8.82, 9.18],'yaw':[-1.0,0.5]})
    apr8,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[7.82, 8.18],'yaw':[-1.0,0.5]})
    apr7,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[6.82, 7.18],'yaw':[-1.0,0.5]})
    apr6,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[5.82, 6.18],'yaw':[-1.0,0.5]})
    apr5,ca,hd = db.select(['04', '02'], inc, exc, values_std=std,
                           valuedict={'wind':[4.82, 5.18],'yaw':[-1.0,0.5]})
    # --------------------------------------------------------------------
    # BLADE1_root-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B1_root_%s-vs-lambda-feb-april' % (prefix, blade)
    title = '%s Feb, Apr, 0 deg fixed yaw' % blade
    ylabel = 'Blade 1 root bending [Nm]'
    iblade = hd['B1 root']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE1_30-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B1_30_%s-vs-lambda-feb-april' % (prefix, blade)
    ylabel = 'Blade 1 30\% bending [Nm]'
    iblade = hd['B1 30']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE2_root-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B2_root_%s-vs-lambda-feb-april' % (prefix, blade)
    ylabel = 'Blade 2 root bending [Nm]'
    iblade = hd['B2 root']
    doplot(figfile, iblade, title, ylabel)
    # --------------------------------------------------------------------
    # BLADE2_30-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-B2_30_%s-vs-lambda-feb-april' % (prefix, blade)
    ylabel = 'Blade 2 30\% bending [Nm]'
    iblade = hd['B2 30']
    doplot(figfile, iblade, title, ylabel)

def plot_yawerr_vs_lambda(prefix):

    figpath = 'figures/overview/'

    # the forced series is when different yaw errors are applied in one session
    db = ojf_db(prefix, debug=True)

    inc = ['force', '_STC_']
    exc = []

    data, ca, hd = db.select(['04'], inc, exc, values_std={}, valuedict={})

#    mean = {'wind':[7.40, 7.60]}
#    v75, ca, hd = db.select(['04'], inc, exc, values_std={}, valuedict=mean)
#    mean = {'wind':[7.89, 8.11]}
#    v80, ca, hd = db.select(['04'], inc, exc, values_std={}, valuedict=mean)
#    mean = {'wind':[8.89, 9.11]}
#    v90, ca, hd = db.select(['04'], inc, exc, values_std={}, valuedict=mean)

    # --------------------------------------------------------------------
    # CT-LAMBDA as function of yaw error
    # --------------------------------------------------------------------

    scale = 1.5
    figfile = '%s-yawerror-vs-ct-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

#    ax1.plot(db.tsr(v90, hd), db.ct(v90, hd), 'rs', label='9 m/s')
#    ax1.plot(db.tsr(v80, hd), db.ct(v80, hd), 'gv', label='8 m/s')
#    ax1.plot(db.tsr(v75, hd), db.ct(v75, hd), 'm<', label='7.5 m/s')
#    ax1.set_xlabel('tip speed ratio $\lambda$')
#    ax1.set_ylabel('thrust coefficient $C_T$')
#    leg = ax1.legend(loc='best')
#    leg.get_frame().set_alpha(0.5)

#    ax1.plot(v90[hd['yaw'],:], db.ct(v90, hd), 'rs', label='9 m/s')
#    ax1.plot(v80[hd['yaw'],:], db.ct(v80, hd), 'gv', label='8 m/s')
#    ax1.plot(v75[hd['yaw'],:], db.ct(v75, hd), 'm<', label='7.5 m/s')
#    ax1.set_xlabel('yaw angle $\psi$')
#    ax1.set_ylabel('thrust coefficient $C_T$')
#    leg = ax1.legend(loc='best')
#    leg.get_frame().set_alpha(0.5)

    # or bin on TSR instead of wind speed
    tsr = db.tsr(data)
    i4 = tsr.__le__(5.0)
    i5 = tsr.__ge__(5.0)*tsr.__lt__(6.0)
    i6 = tsr.__ge__(6.0)*tsr.__lt__(7.0)
    i7 = tsr.__ge__(7.0)*tsr.__lt__(8.0)
    i8 = tsr.__ge__(8.0)*tsr.__lt__(9.0)
#    i9 = tsr.__ge__(9.0)*tsr.__lt__(11.0)
    iyaw = hd['yaw']
#    ax1.plot(data[iyaw,i9], db.ct(data[:,i9]),'bo',label='$9<\lambda<10$')
    ax1.plot(data[iyaw,i8], db.ct(data[:,i8]),'rs',label='$8<\lambda<9$')
    ax1.plot(data[iyaw,i7], db.ct(data[:,i7]),'gv',label='$7<\lambda<8$')
    ax1.plot(data[iyaw,i6], db.ct(data[:,i6]),'m<',label='$6<\lambda<7$')
    ax1.plot(data[iyaw,i5], db.ct(data[:,i5]),'c^',label='$5<\lambda<6$')
    ax1.plot(data[iyaw,i4], db.ct(data[:,i4]),'y>',label='$\lambda<5$')

    # add some cos or cos**2 fits
    angles = np.arange(-40.0, 40.0, 0.1)
    angles_rad = angles.copy() * np.pi/180.0
    max_up = db.ct(data[:,i8]).max()
    max_mid = db.ct(data[:,i5]).max()
    max_low = db.ct(data[:,i4])
    max_low = np.sort(max_low)[-3]
    cos = np.cos(angles_rad)
    ax1.plot(angles, cos*cos*max_up, 'r-')
    ax1.plot(angles, cos*cos*max_mid, 'c-')
    ax1.plot(angles, cos*cos*max_low, 'y-')

    ax1.set_xlabel('yaw angle $\psi$')
    ax1.set_ylabel('thrust coefficient $C_T$')
    leg = ax1.legend(loc='center')
    leg.get_frame().set_alpha(0.5)

    ax1.set_title('April, forced yaw error', size=14*scale)
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------
    # YawError-CT-LAMBDA as function of yaw error (CONTOUR PLOT)
    # --------------------------------------------------------------------
    # ignore the low RPM's
    vals = {'RPM':[200,1100]}
    data, ca, hd = db.select(['04'], inc, exc, values_std={}, valuedict=vals)

    scale = 1.5
    figfile = '%s-yawerror-vs-ct-vs-lambda-contour-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    # prep the data for contour plotting
    z = db.tsr(data) # tsr
    x = data[hd['yaw'],:] # yawerror
    y = db.ct(data) # ct
    # define the grid
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    # grid the data
    zi = mpl.mlab.griddata(x,y,z,xi,yi,interp='nn')
    # contour the gridded data, plotting dots at the nonuniform data points
    # draw the contour lines
#    ct = ax1.contour(xi,yi,zi,10,linewidths=0.5,colors='k')
    # draw the colors
    ct = ax1.contourf(xi,yi,zi,20, cmap=mpl.cm.rainbow)
    pa4.fig.colorbar(ct) # draw colorbar
    # plot data points
    ax1.scatter(x,y, marker='+', c='k')

    ax1.set_title('April, forced yaw error', size=14*scale)
    ax1.grid(True)
    pa4.save_fig()


###############################################################################
### TASKS
###############################################################################
# instead of having to comment and un-comment stuff in main, give each task a
# seperate method.

def make_symlinks_hs():
    """
    First create make_symlinks_all(). This one uses its index file
    """

    # =========================================================================
    # CREATE SYMLINKS FOR THE HIGH SPEED CAMERA FOLDERS
    path_db = PATH_DB

#    sf = '/mnt/mimer/backup_dave/PhD_archive/OJF_data_orig/02/'
#    symlink_to_hs_folder(sf, path_db, symf='symlinks_hs_mimer/')
#    sf = '/mnt/mimer/backup_dave/PhD_archive/OJF_data_orig/04/'
#    symlink_to_hs_folder(sf, path_db, symf='symlinks_hs_mimer/')

#    sf = '/run/media/dave/LaCie2big/backup_dave/PhD_archive/OJF_data_orig/02'
#    symlink_to_hs_folder(sf, path_db, symf='symlinks_hs_lacie2big/')
#    sf = '/run/media/dave/LaCie2big/backup_dave/PhD_archive/OJF_data_orig/04'
#    symlink_to_hs_folder(sf, path_db, symf='symlinks_hs_lacie2big/')

    sf = '/run/media/dave/LaCie/DATA/OJF_data_orig/04'
    symlink_to_hs_folder(sf, path_db, symf='symlinks_hs_lacie/')
    sf = '/run/media/dave/LaCie/DATA/OJF_data_orig/02/HighSpeedCamera'
    symlink_to_hs_folder(sf, path_db, symf='symlinks_hs_lacie/')
    # =========================================================================

def make_symlinks_filtered():
    """
    """

    # -------------------------------------------------------------------------
    # SYMLINKS, DATABASE FOR THE DSPACE, BLADE, AND WIND TUNNEL RESULTS
    source_folder = os.path.join(OJFPATH_RAW, '02/')
    symlink_to_folder(source_folder, PATH_DB)
    source_folder = os.path.join(OJFPATH_RAW, '04/')
    symlink_to_folder(source_folder, PATH_DB)
    build_db(PATH_DB, calibrate=False, dashplot=False)
    # -------------------------------------------------------------------------

def make_symlinks_all(path_db, data_source_root):
    """
    """
    # -------------------------------------------------------------------------
    # SYMLINKS for all results files, no filtering
    db_id = 'symlinks_all'

    source_folder = os.path.join(data_source_root, 'dc_sweep/')
    symlinks_to_dcsweep(source_folder, path_db, db_id)

    source_folder = os.path.join(data_source_root, '02/')
    symlink_to_folder(source_folder, path_db, db_id=db_id, fileignore=[])
    source_folder = os.path.join(data_source_root, '04/')
    symlink_to_folder(source_folder, path_db, db_id=db_id, fileignore=[])
    # -------------------------------------------------------------------------

###############################################################################
### ANALYSIS
###############################################################################

def steady_rpms():
    """
    For all steady RPM's, find the corresponding other steady parematers: yaw,
    FA, SS, blade load. Save those time series in a seperate datafile
    """
    # TODO: finish this
    pass


if __name__ == '__main__':

    dummy = None

#    make_symlinks_all()
#    make_symlinks_hs()

#    build_db(path_db, 'symlinks_all', calibrate=True, dashplot=True)
#    build_db(path_db, 'symlinks_all', calibrate=True, dashplot=True,
#             output='symlinks_all', key_inc=['dcsweep'])

#    prefix = 'symlinks_all'
#    plot_rpm_wind(prefix)
#    plot_voltage_current(prefix)
#    plot_rpm_vs_towerstrain(prefix)
#    plot_rpm_vs_tower_allfeb(prefix)
#    plot_ct_vs_lambda(prefix)
#    plot_yawerr_vs_lambda(prefix)
#    plot_blade_vs_lambda(prefix, 'flex')
#    plot_blade_vs_lambda(prefix, 'stiff')
#    plot_rpm_vs_blade(prefix, 'flex')
#    plot_rpm_vs_blade(prefix, 'stiff')

    # new df db format
#    plot_voltage_current()
#    plot_rpm_wind()
#    plot_ct_vs_lambda(blades='straight')
#    plot_ct_vs_lambda(blades='swept')

    # read a single file for debugging/checking
#    case = '0213_run_108_8.0ms_dc1_samoerai_fixyaw_pwm1000_highrpm'
#    res = ojfresult.ComboResults(PATH_DB+'symlinks/', case)
#    stats = res.statistics()
