# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:37:00 2012

Make a database of all the test and their results

@author: dave
"""

#import sys
import os
import pickle
#import logging
from copy import copy
import string
import shutil

import numpy as np
import matplotlib as mpl
import pandas as pd

import ojfresult
import plotting
import ojf_post
import misc

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
                 'speedup', 'shutdown', 'spinningdown', 'startup'])
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
    df.to_hdf(fname + '.h5', 'table', complevel=9, complib='blosc')
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


def build_db(path_db, prefix, **kwargs):
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

    dataframe : boolean, default=False
        From a single case combine dSPACE, OJF and blade strain into a single
        Pandas DataFrame.

    """
    folder_df = kwargs.get('folder_df', 'data/calibrated/DataFrame/')
    folder_csv = kwargs.get('folder_csv', 'data/calibrated/CSV/')
    output = kwargs.get('output', prefix)
    dashplot = kwargs.get('dashplot', False)
    calibrate = kwargs.get('calibrate', True)
    key_inc = kwargs.get('key_inc', [])
    resample = kwargs.get('resample', False)
    dataframe = kwargs.get('dataframe', False)
    save_df = kwargs.get('save_df', False)
    save_df_csv = kwargs.get('save_df_csv', False)
    continue_build = kwargs.get('continue_build', True)
    db_index_file = kwargs.get('db_index_file', 'db_index_%s.pkl' % prefix)

    # read the database
    FILE = open(path_db + db_index_file)
    db_index = pickle.load(FILE)
    FILE.close()

    # remove the files we've already done
    if continue_build:
        source_folder = os.path.join(folder_df)
        for root, dirs, files in os.walk(source_folder, topdown=True):
            for fname in files:
                db_index.pop(fname[:-3])
    # respath is where all the symlinks are
    respath = path_db + prefix + '/'

    # create the figure folder if it doesn't exist
    try:
        os.mkdir(path_db+'figures_%s/' % output)
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

    # save the statistics in a dict
    db_stats = {}
    df_stats = None

    nr, nrfiles = 0, len(db_index)

    # initialize the DataFrame formatted dictionary
    CR2stats_df = ComboResults2stats_df()

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

        #except:
            #logging.warn('ignored: %s' % resfile)
            #logging.warn(sys.exc_info()[0])
            #continue

        # make a dashboard plot
        if dashplot:
            res.dashboard_a3(path_db+'figures_%s/' % output)

        # calculate all the means, std, min, max and range for each channel
        res.statistics()
        # stats is already a dictionary
        db_stats[resfile] = res.stats
        # add the channel discriptions
        db_stats[resfile]['dspace labels_ch'] = res.dspace.labels_ch
        # incase there is no OJF data
        try:
            db_stats[resfile]['ojf labels'] = res.ojf.labels
        except AttributeError:
            pass

        if dataframe:
            if save_df:
                ftarget = os.path.join(folder_df, resfile + '.h5')
            else:
                ftarget = None
            df = res.to_df(ftarget, complevel=9, complib='blosc')
            if save_df_csv:
                df.to_csv(os.path.join(folder_csv, resfile + '.csv'))

            CR2stats_df.add_all_stats(res)

#        if nr > 100:
#            break

    if df_stats:
        CR2stats_df.dict2df()
        CR2stats_df.save_all_stats(path_db, output)

    # load an existing database first, update
    try:
        # if it exists, update the file first before saving
        FILE = open(path_db + 'db_stats_%s.pkl' % output)
        db_stats_update = pickle.load(FILE)
        # overwrite the old entries with new ones! not the other way around
        db_stats_update.update(db_stats)
        FILE.close()
    except IOError:
        # no need to update an existing database file
        db_stats_update = db_stats

    # and save the database stats
    FILE = open(path_db + 'db_stats_%s.pkl' % output, 'wb')
    pickle.dump(db_stats_update, FILE, protocol=2)
    FILE.close()


class ComboResults2stats_df(object):
    """add the statistics of a given case to the statistics DataFrame
    """

    def __init__(self):
        """Initialize all columns
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
        # create a DataFrame formatted dictionary
        self.stats_mean = {col:[] for col in self.all_c_columns + ['index']}
        self.stats_min = {col:[] for col in self.all_c_columns + ['index']}
        self.stats_max = {col:[] for col in self.all_c_columns + ['index']}
        self.stats_std = {col:[] for col in self.all_c_columns + ['index']}
        self.stats_range = {col:[] for col in self.all_c_columns + ['index']}

    def add_all_stats(self, res):
        """add all stats from ComboResults to DataFrame formatted dictionaries
        """
        # TODO: should also work with df if not ComboResults is given

        def add_stats(stats, df_dict):
            df_dict['index'].append(res.resfile)
            for col in stats.index:
                if col == 'time':
                    df_dict[col].append(res.dspace.time[-1])
                else:
                    df_dict[col].append(stats[col])

            # and empty items for those for which there is no data
            for col in (set(self.all_c_columns) - {str(k) for k in stats.index}):
                if col == 'duty_cycle':
                    dc = dc_from_casename(self.res.resfile)
                    df_dict[col].append(dc)
                else:
                    df_dict[col].append(np.nan)
            return df_dict

        if not hasattr(res, 'df'):
            res.to_df()
        self.stats_mean = add_stats(res.df.mean(), self.stats_mean)
        self.stats_min = add_stats(res.df.min(), self.stats_min)
        self.stats_max = add_stats(res.df.max(), self.stats_max)
        self.stats_std = add_stats(res.df.std(), self.stats_std)
        self.stats_range = add_stats(res.df.max()-res.df.min(), self.stats_range)

    def dict2df(self):
        """Convert the DataFrame formatted dictionaries to DataFrames. If
        the conversion failes, perform post-mortem checks for debugging
        purposes.
        """
        try:
            self.df_mean = pd.DataFrame(self.stats_mean)
        except ValueError:
            print('stats_mean')
            misc.check_df_dict(self.stats_mean)

        try:
            self.df_min = pd.DataFrame(self.stats_min)
        except ValueError:
            print('stats_min')
            misc.check_df_dict(self.stats_min)

        try:
            self.df_max = pd.DataFrame(self.stats_max)
        except ValueError:
            print('stats_max')
            misc.check_df_dict(self.stats_max)

        try:
            self.df_std = pd.DataFrame(self.stats_std)
        except ValueError:
            print('stats_std')
            misc.check_df_dict(self.stats_std)

        try:
            self.df_range = pd.DataFrame(self.stats_range)
        except ValueError:
            print('stats_range')
            misc.check_df_dict(self.stats_range)

    def save_all_stats(self, path_db, prefix='symlinks_all', update=False):
        """Save to h5 and xlsx
        """

        fname = os.path.join(path_db, 'db_stats_%s_mean.h5' % prefix)
        self.df_mean.to_hdf(fname, 'table', compression=9, complib='blosc')
        fname = os.path.join(path_db, 'db_stats_%s_mean.xlsx' % prefix)
        self.df_mean.to_excel(fname)

        fname = os.path.join(path_db, 'db_stats_%s_min.h5' % prefix)
        self.df_min.to_hdf(fname, 'table', compression=9, complib='blosc')
        fname = os.path.join(path_db, 'db_stats_%s_min.xlsx' % prefix)
        self.df_min.to_excel(fname)

        fname = os.path.join(path_db, 'db_stats_%s_max.h5' % prefix)
        self.df_max.to_hdf(fname, 'table', compression=9, complib='blosc')
        fname = os.path.join(path_db, 'db_stats_%s_max.xlsx' % prefix)
        self.df_max.to_excel(fname)

        fname = os.path.join(path_db, 'db_stats_%s_std.h5' % prefix)
        self.df_std.to_hdf(fname, 'table', compression=9, complib='blosc')
        fname = os.path.join(path_db, 'db_stats_%s_std.xlsx' % prefix)
        self.df_std.to_excel(fname)

        fname = os.path.join(path_db, 'db_stats_%s_range.h5' % prefix)
        self.df_range.to_hdf(fname, 'table', compression=9, complib='blosc')
        fname = os.path.join(path_db, 'db_stats_%s_range.xlsx' % prefix)
        self.df_range.to_excel(fname)


class ojf_db:
    """
    OJF database class
    ==================

    The OJF statistics database has following structure

    db_stats = {ojf_resfile : stats_dict}

    stats_dict has the following keys:

    'blade max', 'blade mean', 'blade min', 'blade range', 'blade std',
    'dspace labels_ch', 'dspace max', 'dspace mean', 'dspace min',
    'dspace range', 'dspace std', 'ojf labels', 'ojf max', 'ojf mean',
    'ojf min', 'ojf range', 'ojf std'

    The corresponding values are the statistical values for the channels
    described in the lables keys.

    """

    def __init__(self, prefix, **kwargs):
        """
        """

        debug = kwargs.get('debug', False)
        path_db = kwargs.get('path_db', 'database/')

        FILE = open(path_db + 'db_stats_%s.pkl' % prefix)
        self.db_stats = pickle.load(FILE)
        FILE.close()

        self.path_db = path_db
        self.debug = debug
        self.prefix = prefix

    def ct(self, data):
        """

        Parameters
        ----------

        data : ndarray
            ojf_db.select output
        """
        data_headers = {'wind':0, 'RPM':1, 'dc':2, 'volt':3, 'amp':4, 'FA':5,
                   'SS':6, 'yaw':7, 'power':8, 'temp':9, 'B2 root':10,
                   'B2 30':11, 'B1 root':12, 'B1 30':13, 'static_p':14}

        ifa = data_headers['FA']
        iwind = data_headers['wind']

        # convert the tower FA bending moment to rotor thrust
        thrust = data[ifa,:] / ojf_post.model.momemt_arm_rotor
        # TODO: calculate rho from wind tunnel temperature and static pressure
        # rho = R*T / P   R_dryair = 287.058
        rho = 1.225
        V = data[iwind,:]
        # and normalize to get the thrust coefficient
        return thrust / (0.5*rho*V*V*ojf_post.model.A)

    def tsr(self, data):
        r"""
        Tip Speed Ratio lambda :math:`\lambda=\frac{V_{tip}}{V}`, or we can
        also write it as :math:`\lambda=\frac{R\Omega_{RPM}\pi/30}{V}`

        Parameters
        ----------

        data : ndarray
            ojf_db.select output

        """
        data_headers = {'wind':0, 'RPM':1, 'dc':2, 'volt':3, 'amp':4, 'FA':5,
                   'SS':6, 'yaw':7, 'power':8, 'temp':9, 'B2 root':10,
                   'B2 30':11, 'B1 root':12, 'B1 30':13, 'static_p':14}

        irpm = data_headers['RPM']
        iwind = data_headers['wind']

        R = ojf_post.model.blade_radius
        return R*data[irpm,:]*np.pi/(data[iwind,:]*30.0)

    def select(self, months, include, exclude, valuedict={}, verbose=True,
               runs_inc=[], values_std={}):
        """
        Make an array holding wind, rpm and dc values for each entry in the
        database, filtered with the search terms occuring in case name

        Note that the verbose plotting does not show the final merged dc
        column. It shows data available from the dspace field and the dc
        obtained from the case name.

        This method allows to search and select the database based on only
        a few values of the results: wind, RPM, dc, volt and amp.

        The operator among the search criteria then the following logic
        needs to be evaluate to True:
            months and include and runs_inc and not exclude

        Parameters
        ----------

        months : list
            Allowable items are '02' and/or '04'

        include : list
            list of strings with keywords that have to be included in the
            case name. Operator is AND

        exclude : list
            list of strings with keywords that have to be excluded in the
            case name. Operator is OR

        valuedict : dict, default={}
            In- or exclude any mean values in the statistics file. Allowable
            entries on the keys are wind, RPM, dc, volt, amp, FA, SS, yaw,
            power, temp, B2 root, B2 30, B1 root, or B1 30. If the value is a
            list, it indicates the upper and lower bounds of the allowed
            interval. Lower bound is inclusive, upper bound exclusive.

        runs_inc : list or set, default=[]
            Run number id's of that need to be included. Operator is OR. The
            list should be populated with strings, and not integers. Note that
            some run id's contain characters, such as 358b for instance.
            Note that sets is faster than a list.

        values_std : dict, default={}
            Same as valuedict, but now selection based on the standard
            deviation. Both valuedict and values_std have to evaluate True
            if the case needs to be accepted.

        Returns
        -------

        data : ndarray(14,n)
            Holding wind, RPM, dc, volt, amp, FA, SS, yaw, power, temp,
            B2 root, B2 30, B1 root, and B1 30. DC is set to -1 if no data is
            available. The Duty Cycle has been constructed from the dc dspace
            field or the one mentioned in the case name if the former was not
            available.

        case_arr : ndarray(n)
            Case names corresponding to the data in the data array

        data_headers : dict
            Column headers for the data array.
            {'wind':0, 'RPM':1, 'dc':2, 'volt':3, 'amp':4, 'FA':5,
             'SS':6, 'yaw':7, 'power':8, 'temp':9, 'B2 root':10,
             'B2 30':11, 'B1 root':12, 'B1 30':13}

        """

        def get_data(statval, statpar='mean'):
            """
            Get that statistcal data from one single OJF measurements



            Parameters
            ----------

            statval : dict
                A dictionary holding the statistics for that case

            statpar : str, default='mean'
                Valid entries are max, mean, min, range, std

            Returns
            -------

            data : list
                [windspeed, RPM, dc, volt, amp, FA, SS, yaw, power, temp,
                 static_p, blade strain ch1, ch2, ch3, ch4]
            """

            if not statpar in ['max', 'mean', 'min', 'range', 'std']:
                msg = 'statpar can only be either: max, mean, min, range, std'
                raise ValueError, msg

            iwind = 4
            itemp = 1
            ipstatic = 2
            # make sure we are selecting the wind speed every time again
            # doesn't seem necesary, wind speed was always on the same index
            try:
                assert statval['ojf labels'][iwind] == 'wind speed'
                assert statval['ojf labels'][itemp] == 'temperature'
                assert statval['ojf labels'][ipstatic] == 'static_p'
            except KeyError:
                # there was no ojf data
                pass

            dspace_labels = statval['dspace labels_ch']
            ivolt = dspace_labels['Voltage filtered']
            iamp = dspace_labels['Current Filter']
            irpm = statval['dspace labels_ch']['RPM']
            # in april the channel was called filtered?
            try:
                itfa = dspace_labels['Tower Strain For-Aft']
            except KeyError:
                itfa = dspace_labels['Tower Strain For-Aft filtered']
            try:
                itss = dspace_labels['Tower Strain Side-Side']
            except KeyError:
                itss = dspace_labels['Tower Strain Side-Side filtered']
            ipow = dspace_labels['Power']

            data = []
            try:
                data.append(statval['ojf %s' % statpar][iwind])
            except:
                data.append(np.nan)
            data.append(statval['dspace %s' % statpar][irpm])
            # not all cases have a logged duty cycle value, anticpate for that
            try:
                idc = statval['dspace labels_ch']['Duty Cycle']
                data.append(statval['dspace %s' % statpar][idc])
            # fill in -1 if no recorded dc value is present
            except KeyError:
                data.append(-1)

            # and all other data
            data.append(statval['dspace %s' % statpar][ivolt])
            data.append(statval['dspace %s' % statpar][iamp])
            data.append(statval['dspace %s' % statpar][itfa])
            data.append(statval['dspace %s' % statpar][itss])
            try:
                iyaw = dspace_labels['Yaw Laser']
                data.append(statval['dspace %s' % statpar][iyaw])
            except KeyError:
                # some rare case doesn't have yaw measurements saved...
                data.append(np.nan)
            data.append(statval['dspace %s' % statpar][ipow])
            # addditional wind tunnel data
            try:
                data.append(statval['ojf %s' % statpar][itemp])
            except:
                data.append(np.nan)
            # FIXME: what if there is no blade strain data?
            # and also include all the blade strains
            try:
                data.append(statval['blade %s' % statpar][0])
                data.append(statval['blade %s' % statpar][1])
                data.append(statval['blade %s' % statpar][2])
                data.append(statval['blade %s' % statpar][3])
            except KeyError:
                data.append(np.nan)
                data.append(np.nan)
                data.append(np.nan)
                data.append(np.nan)

            # and the static pressure, added later
            try:
                data.append(statval['ojf %s' % statpar][ipstatic])
            except:
                data.append(np.nan)

            return data

        # convert the valuedict key from text to indices
        # [windspeed, RPM, dc, volt, amp, FA, SS, yaw, power, temp,
        #          blade strain ch1, ch2, ch3, ch4]
        data_headers = {'wind':0, 'RPM':1, 'dc':2, 'volt':3, 'amp':4, 'FA':5,
                   'SS':6, 'yaw':7, 'power':8, 'temp':9, 'B2 root':10,
                   'B2 30':11, 'B1 root':12, 'B1 30':13, 'static_p':14}

        # guess the size of the obtained statistics, add one more item for
        # the dc from file name
        statsrand = self.db_stats[self.db_stats.keys()[0]]
        size = len(get_data(statsrand, statpar='mean'))
        data_arr = np.ndarray((size+1,len(self.db_stats)))

        # now we do not have an extra value for the dc of the file name
        size = len(get_data(statsrand, statpar='std'))
        data_std_arr = np.ndarray((size,len(self.db_stats)))
        caselist = []

        # convert from a ch dict with channel names to channel indices
        valuedict_chi = {}
        for key, value in valuedict.iteritems():
            indexkey = data_headers[key]
            valuedict_chi[indexkey] = value

        value_std_chi = {}
        for key, value in values_std.iteritems():
            indexkey = data_headers[key]
            value_std_chi[indexkey] = value

        #dtypes = [('wind', float), ('RPM', float), ('dc_dspace', float), \
                  #('dc_name', float), ('case', str) ]
        #data_arr = np.recarray((len(self.db_stats),), dtype=dtypes)

        i = 0
        for case, statval in self.db_stats.iteritems():

            # -----------------------------------------------------------------
            # case name based selection
            # -----------------------------------------------------------------
            # ingore if not the right month
            if not case[:2] in months:
                continue

            # select on run number id
            if len(runs_inc) > 0:
                runid = case.split('_')[2]
                # ignore the current case if it is not in the runs_inc list
                if not runid in runs_inc:
                    continue

            # only select the current case if all searchitems are satisfied
            find = True
            for k in include:
                if case.find(k) > -1:
                    # multiply because all arguments need to be True. One
                    # false argument should still have result=False
                    find *= True
                # if we can't find switch to False if not already set to False
                elif find:
                    find = False
            if not find:
                continue

            # if we find any element from exclude, ignore the case
            find = False
            for k in exclude:
                if case.find(k) > -1:
                    find = True
                    break
            if find:
                continue

            # -----------------------------------------------------------------
            # load the statistics
            # -----------------------------------------------------------------
            data = get_data(statval, statpar='mean')
            data_std = get_data(statval, statpar='std')
            # try to read the dc from the case file name
            items = case.split('_')
            for k in items:
                if k.startswith('dc'):
                    # in case that fails, we don't know: like when dc is
                    # something like 0.65-0.70
                    try:
                        data.append(float(k.replace('dc', '')))
                    except ValueError:
                        pass
            # in case we didn't find any dc in the name, add -1
            if not len(data) == size+1:
                data.append(-1)
            # remember the index number of the manually added index for dc
            # obtained from the case name
            idc = len(data)-1

            # if there is no recorded dc, save it to the dc field now, other
            # wise we can not make any selection based on it!
            if data[2] < -0.5:
                data[2] = data[idc]

            # -----------------------------------------------------------------
            # selections based on mean values obtained with get_data
            # -----------------------------------------------------------------
            find = True
            for chi, value in valuedict_chi.iteritems():
                # if it is a list, we have an allowable min/max range
                if type(value).__name__ == 'list':
                    if not data[chi] >= value[0] or not data[chi] < value[1]:
                        find = False
                        break
                else:
                    # for the comparison, make string with finite precision
                    #p1, p2 = '%1.3f' % data[chi], '%1.3f' % value
                    #if p1 == p2:
                    if not format(data[chi],'1.3f') == format(value, '1.3f'):
                        find = False
                        break
            if not find:
                continue

            # -----------------------------------------------------------------
            # selections based on std values obtained with get_data
            # -----------------------------------------------------------------
            find = True
            for chi, value in value_std_chi.iteritems():
                # if it is a list, we have an allowable min/max range
                if type(value).__name__ == 'list':
                    low = value[0]
                    up = value[1]
                    if not data_std[chi] >= low or not data_std[chi] < up:
                        find = False
                        break
                else:
                    # for the comparison, make string with finite precision
                    #p1, p2 = '%1.3f' % data[chi], '%1.3f' % value
                    #if p1 == p2:
                    if not format(data_std[chi],'1.3f')==format(value,'1.3f'):
                        find = False
                        break
            if not find:
                continue

            # -----------------------------------------------------------------
            # and finally, all conditions are met, we can save the case
            # -----------------------------------------------------------------
            data_arr[:,i] = data
            data_std_arr[:,i] = data_std
            caselist.append(case)

            i += 1

        # data_arr was created too big, remove any unused elements
        data_arr = data_arr[:,:i]
        data_std_arr = data_std_arr[:,:i]
        case_arr = np.char.array(caselist)
        if not data_arr.shape[1] == len(case_arr):
            print data_arr.shape
            print case_arr.shape
            msg = 'case_arr and data_arr do not have the same nr of items'
            raise IndexError, msg

        if len(caselist) > 0:
            # organize the array and do some printing
            isort = case_arr.argsort()
            data_arr = data_arr[:,isort]
            data_std_arr = data_std_arr[:,isort]
            case_arr = case_arr[isort]
        elif not verbose:
            print 'nothing found in database'

        if verbose:
            # print some headers
            #headers = ['wind', 'RPM', 'dc', 'volt', 'amp', 'fa', 'ss', 'case']
            headers = ['   wind', '    RPM', '    yaw', '     dc', '     FA',
                       '     SS', '  TSR',
                       '   Vstd', '    RPM', '    YAW', '  case']

            isel = [data_headers['wind'], data_headers['RPM'],
                    data_headers['yaw'],  data_headers['dc'],
                    data_headers['FA'],   data_headers['SS']]
            print
            print '='*80
            print '    months:', months
            print '   include:', include
            print '   exclude:', exclude
            print ' valuedict:', valuedict
            print 'values_std:', values_std
            print '  runs_inc:', runs_inc
            print '='*80
            #print ''.join(['%7s' % k for k in headers])
            print ''.join(headers)
            for i in xrange(len(case_arr)):
                # the mean values from
                print ''.join(['%7.2f' % k for k in data_arr[isel,i]]),

#                # dc from either dspace or name and if they are the same
#                p = '1.3f'
#                if data_arr[idc,i] == -1 or data_arr[2,i] == -1:
#                    print '%3s' % '',
#                elif not format(data_arr[idc,i],p) == format(data_arr[2,i],p):
#                    print '%3s' % 'x',
#                else:
#                    print '%3s' % '',

                # or instead of the dc mismatch name-dspace, print TSR
                wind = data_arr[data_headers['wind'],i]
                rpm = data_arr[data_headers['RPM'],i]
                print '%3.1f' % (np.pi*rpm*0.8/(30.0*wind)),

                # standard deviations for wind (0), RPM (1), yaw (7)
                print ''.join(['%7.2f' % k for k in data_std_arr[[0,1,7],i]]),

                # and last entry the case name, but limit to 50 characters
                if len(case_arr[i]) > 48:
                    print ' %s ...' % case_arr[i][:48]
                else:
                    print ' %s' % case_arr[i]

        # any known data is merged into the dc field of idc=2
#        sel = data_arr[2,:].__le__(-0.5)
#        data_arr[2,sel] = data_arr[idc,sel]

        # and ditch the case name based dc column. We only care about the
        # final merged data. Printing stuff is done for debugging purposes
        self.data_arr = data_arr[:len(data)-1,:]
        self.case_arr = case_arr
        self.data_headers = data_headers
        return data_arr[:len(data)-1,:], case_arr, data_headers


    def load_case(self, resfile, **kwargs):
        """
        Load the result file from a given case name

        Returns
        -------

        res : ojfresult.ComboResults object
        """

        cal = kwargs.get('calibrate', False)

        respath = self.path_db + 'symlinks_all/'

        return ojfresult.ComboResults(respath, resfile, silent=True, cal=cal)


    def plot(self, case_arr, **kwargs):
        """
        Plot a selection of the database.

        Paremeters
        ----------

        case_arr : iterable
            Iterable holding all the case names to be plotted

        calibrate : boolean, default=False
        """

#        data_arr = kwargs.get('data_arr', False)
#        data_headers = kwargs.get('data_headers', False)
        calibrate = kwargs.get('calibrate', False)

        respath = self.path_db + 'symlinks_all/'
        figfolder = kwargs.get('figfolder', 'figures_%s/' % self.prefix)

        for resfile in case_arr:
            # if we catch any error, ignore that file for now and go on
            print
            print 80*'='
            print resfile
            res = ojfresult.ComboResults(respath, resfile, silent=True,
                                         cal=calibrate)
            res.dashboard_a3(self.path_db + figfolder)

    def add_staircase_steps(self, res, arg_stair):
        """Add the statistics of a stair case analysis

        Parameters
        ----------

        res : ojfresult.ComboResults

        arg_stair : ndarray()
            as given by staircase.StairCase.arg_stair
        """

        def add_to_database(res, istart, istop):
            """
            add the carufelly selected steady states from a stair case
            to the database

            Al stair cases are marked with _STC_%i
            """
            db_stats = {}
            # dividing by 100 is safe, since we set points_per_stair=800,
            # so we do not risque of having a non unique key
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

        # and save into the database
        for k in range(arg_stair.shape[1]):
            i1 = arg_stair[0,k]
            i2 = arg_stair[1,k]
            # calculate all the means, std, min, max and range for each channel
            res.statistics(i1=i1, i2=i2)
            # and add to the database dict
            self.db_stats.update(add_to_database(res, i1, i2))

    def save_updates(self):
        """Save any updates made to the statistics database
        """
        # and update the database
        try:
            # if it exists, update the file first before saving
            FILE = open(os.path.join(PATH_DB, 'db_stats_%s.pkl' % self.prefix))
            db_stats_update = pickle.load(FILE)
            # overwrite the old entries with new ones! not the other way around
            db_stats_update.update(self.db_stats)
            FILE.close()
        except IOError:
            # no need to update an existing database file
            db_stats_update = self.db_stats
        # and save the database stats
        FILE = open(os.path.join(PATH_DB, 'db_stats_%s.pkl' % self.prefix), 'wb')
        pickle.dump(db_stats_update, FILE, protocol=2)
        FILE.close()
        print 'updated db: %sdb_stats_%s.pkl' % (PATH_DB, self.prefix)


###############################################################################
### PLOTS
###############################################################################

def plot_voltage_current(prefix):
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


    path_db = PATH_DB
    db = ojf_db(prefix, path_db=path_db)

    figpath = 'figures/overview/'
    scale = 1.5

    # --------------------------------------------------------------------
    #ex = ['coning', 'free']
    ex = []
    dc0_02, ca, headers = db.select(['02'], [], ex, valuedict={'dc':0})
    dc1_02, ca, headers = db.select(['02'], [], ex, valuedict={'dc':1})
    dc0_04, ca, headers = db.select(['04'], [], ex, valuedict={'dc':0})
    dc1_04, ca, headers = db.select(['04'], [], ex, valuedict={'dc':1})

    figfile = '%s-volt-vs-current' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(dc0_02[3,:], dc0_02[4,:], 'rs', label='dc0 02')
    ax1.plot(dc1_02[3,:], dc1_02[4,:], 'gs', label='dc1 02')
    ax1.plot(dc0_04[3,:], dc0_04[4,:], 'bd', label='dc0 04')
    ax1.plot(dc1_04[3,:], dc1_04[4,:], 'yd', label='dc1 04')
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
    ax1.plot(dc0_02[1,:], dc0_02[4,:],  'rs', label='dc0 02')
    ax1.plot(dc1_02[1,:], dc1_02[4,:],  'gs', label='dc1 02')
    ax1.plot(dc0_04[1,:], dc0_04[4,:],  'bd', label='dc0 04')
    ax1.plot(dc1_04[1,:], dc1_04[4,:],  'yd', label='dc1 04')
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
    ax1.plot(dc0_02[1,:], dc0_02[3,:],  'rs', label='dc0 02')
    ax1.plot(dc1_02[1,:], dc1_02[3,:],  'gs', label='dc1 02')
    ax1.plot(dc0_04[1,:], dc0_04[3,:],  'bd', label='dc0 04')
    ax1.plot(dc1_04[1,:], dc1_04[3,:],  'yd', label='dc1 04')
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


def plot_rpm_wind(prefix):
    """
    """

    def fit(x, y, deg, res=50):
        pol = np.polyfit(x,y, deg)
        # but generate polyval on equi spaced x grid
        x_grid = np.linspace(x[0], x[-1], res)
        return  x_grid, np.polyval(pol, x_grid)

    path_db = PATH_DB
    db = ojf_db(prefix, path_db=path_db)

    figpath = 'figures/overview/'
    scale = 1.5

    # --------------------------------------------------------------------
    ex = ['coning', 'free']
    flex_dc0, ca, hd = db.select(['02','04'], ['flex'], ex,
                                 valuedict={'dc':0, 'yaw':[-1.0,0.5]})
    stiff_dc0,ca, hd = db.select(['02','04'], ['stiff'],ex,
                                 valuedict={'dc':0, 'yaw':[-1.0,0.5]})
    flex_dc1, ca, hd = db.select(['02','04'], ['flex'], ex,
                                 valuedict={'dc':1, 'yaw':[-1.0,0.5]})
    stiff_dc1,ca, hd = db.select(['02','04'], ['stiff'],ex,
                                 valuedict={'dc':1, 'yaw':[-1.0,0.5]})
    dc5, ca, hd = db.select(['02','04'], [], ex,
                            valuedict={'dc':0.5, 'yaw':[-1.0,0.5]})

    figfile = '%s-rpm-vs-wind-dc0-dc1' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(flex_dc0[0,:],  flex_dc0[1,:],  'rs', label='dc0 flex')
    ax1.plot(stiff_dc0[0,:], stiff_dc0[1,:], 'ro', label='dc0 stiff')
    ax1.plot(flex_dc1[0,:],  flex_dc1[1,:],  'g>', label='dc1 flex')
    ax1.plot(stiff_dc1[0,:], stiff_dc1[1,:], 'g<', label='dc1 stiff')
    ax1.plot(dc5[0,:], dc5[1,:], 'bd', label='dc0.5')
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
    cs = ax1.contour(V_grid, RPM_grid, TSR, contours, colors='grey',
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
    ex = ['coning', 'free']
    dc_p0,ca,hd = db.select(['02','04'], [], ex,
                            valuedict={'dc':[-0.1, 0.25], 'yaw':[-1.0,0.5]})
    dc_p1,ca,hd = db.select(['02','04'], [], ex,
                            valuedict={'dc':[0.25, 0.5], 'yaw':[-1.0,0.5]})
    dc_p2,ca,hd = db.select(['02','04'], [], ex,
                            valuedict={'dc':[0.5 , 0.75], 'yaw':[-1.0,0.5]})
    dc_p3,ca,hd = db.select(['02','04'], [], ex,
                            valuedict={'dc':[0.75, 1.1], 'yaw':[-1.0,0.5]})

    figfile = '%s-rpm-vs-wind-dc-all' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=1.0,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    ax1.plot(dc_p0[0,:],  dc_p0[1,:], 'rs', label='$0.00 \leq dc < 0.25$')
    ax1.plot(dc_p1[0,:],  dc_p1[1,:], 'g*', label='$0.25 \leq dc < 0.50$')
    ax1.plot(dc_p2[0,:],  dc_p2[1,:], 'bd', label='$0.50 \leq dc < 0.75$')
    ax1.plot(dc_p3[0,:],  dc_p3[1,:],'y^',label='$0.75 \leq dc \leq 1.00$')
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
        data,ca,hd = db.select(['02','04'], [], ex,
                               valuedict={'wind':[low,up], 'yaw':[-1.0,0.5]})

        label = '$%i m/s$' % k
        ax1.plot(data[2,:], data[1,:], colors[k-5], label=label)

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

def plot_ct_vs_lambda_rotors(prefix):
    """
    Compare the CT of the swept, and non swept blades
    """

    path_db = PATH_DB
    db = ojf_db(prefix, debug=True, path_db=path_db)

    figpath = 'figures/overview/'
    scale = 1.5

    exc = ['coning']
    std = {'RPM':[0,10], 'yaw':[0, 0.5], 'wind':[0, 0.1]}
    apr10,ca,hd = db.select(['04'], [], exc, values_std=std,
                            valuedict={'wind':[9.82, 10.18],'yaw':[-1.0,0.5]})
    apr9,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[8.82, 9.18],'yaw':[-1.0,0.5]})
    apr8,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[7.82, 8.18],'yaw':[-1.0,0.5]})
    apr7,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[6.82, 7.18],'yaw':[-1.0,0.5]})
    apr6,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[5.82, 6.18],'yaw':[-1.0,0.5]})
    apr5,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[4.82, 5.18],'yaw':[-1.0,0.5]})
#    apr4,ca,hd = db.select(['04'], [], ex, valuedict={'wind':[3.82, 4.18]})

    # --------------------------------------------------------------------
    # CT-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-ct-vs-lambda-april-swept' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(db.tsr(apr10), db.ct(apr10), 'bo', label='10 m/s')
    ax1.plot(db.tsr(apr9), db.ct(apr9), 'rs', label='9 m/s')
    ax1.plot(db.tsr(apr8), db.ct(apr8), 'gv', label='8 m/s')
    ax1.plot(db.tsr(apr7), db.ct(apr7), 'm<', label='7 m/s')
    ax1.plot(db.tsr(apr6), db.ct(apr6), 'c^', label='6 m/s')
    ax1.plot(db.tsr(apr5), db.ct(apr5), 'y>', label='5 m/s')
#    ax1.plot(apr4), apr4), 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed yaw', size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('thrust coefficient $C_T$')
#    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------
    # THRUST-LAMBDA
    # --------------------------------------------------------------------
    ifa = hd['FA']
    figfile = '%s-fa-vs-lambda-april-swept' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(db.tsr(apr10),apr10[ifa,:],'bo', label='10 m/s')
    ax1.plot(db.tsr(apr9), apr9[ifa,:], 'rs', label='9 m/s')
    ax1.plot(db.tsr(apr8), apr8[ifa,:], 'gv', label='8 m/s')
    ax1.plot(db.tsr(apr7), apr7[ifa,:], 'm<', label='7 m/s')
    ax1.plot(db.tsr(apr6), apr6[ifa,:], 'c^', label='6 m/s')
    ax1.plot(db.tsr(apr5), apr5[ifa,:], 'y>', label='5 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed zero yaw', size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('Tower FA bending [Nm]')
    ax1.grid(True)
    pa4.save_fig()

def plot_ct_vs_lambda(prefix):
    """
    Have all the relevant stuff plotted versus the tip speed ratio lambda!
    That will gave a better understanding of the stuff
    """

    path_db = PATH_DB
    db = ojf_db(prefix, debug=True, path_db=path_db)

    figpath = 'figures/overview/'
    scale = 1.5

    exc = ['coning', 'samoerai']
    std = {'RPM':[0,10], 'yaw':[0, 0.5], 'wind':[0, 0.1]}
    apr10,ca,hd = db.select(['04'], [], exc, values_std=std,
                            valuedict={'wind':[9.82, 10.18],'yaw':[-1.0,0.5]})
    apr9,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[8.82, 9.18],'yaw':[-1.0,0.5]})
    apr8,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[7.82, 8.18],'yaw':[-1.0,0.5]})
    apr7,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[6.82, 7.18],'yaw':[-1.0,0.5]})
    apr6,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[5.82, 6.18],'yaw':[-1.0,0.5]})
    apr5,ca,hd = db.select(['04'], [], exc, values_std=std,
                           valuedict={'wind':[4.82, 5.18],'yaw':[-1.0,0.5]})
#    apr4,ca,hd = db.select(['04'], [], ex, valuedict={'wind':[3.82, 4.18]})

    # --------------------------------------------------------------------
    # CT-LAMBDA
    # --------------------------------------------------------------------
    figfile = '%s-ct-vs-lambda-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(db.tsr(apr10), db.ct(apr10), 'bo', label='10 m/s')
    ax1.plot(db.tsr(apr9), db.ct(apr9), 'rs', label='9 m/s')
    ax1.plot(db.tsr(apr8), db.ct(apr8), 'gv', label='8 m/s')
    ax1.plot(db.tsr(apr7), db.ct(apr7), 'm<', label='7 m/s')
    ax1.plot(db.tsr(apr6), db.ct(apr6), 'c^', label='6 m/s')
    ax1.plot(db.tsr(apr5), db.ct(apr5), 'y>', label='5 m/s')
#    ax1.plot(apr4), apr4), 'bo', label='4 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed yaw', size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('thrust coefficient $C_T$')
#    ax1.set_xlim([4, 19])
    ax1.grid(True)
    pa4.save_fig()

    # --------------------------------------------------------------------
    # THRUST-LAMBDA
    # --------------------------------------------------------------------
    ifa = hd['FA']
    figfile = '%s-fa-vs-lambda-april' % prefix
    pa4 = plotting.A4Tuned(scale=scale)
    pa4.setup(figpath+figfile, nr_plots=1, hspace_cm=2., figsize_x=8,
                   grandtitle=False, wsleft_cm=1.5, wsright_cm=0.4,
                   wstop_cm=1.0, figsize_y=8., wsbottom_cm=1.)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    ax1.plot(db.tsr(apr10),apr10[ifa,:],'bo', label='10 m/s')
    ax1.plot(db.tsr(apr9), apr9[ifa,:], 'rs', label='9 m/s')
    ax1.plot(db.tsr(apr8), apr8[ifa,:], 'gv', label='8 m/s')
    ax1.plot(db.tsr(apr7), apr7[ifa,:], 'm<', label='7 m/s')
    ax1.plot(db.tsr(apr6), apr6[ifa,:], 'c^', label='6 m/s')
    ax1.plot(db.tsr(apr5), apr5[ifa,:], 'y>', label='5 m/s')

    leg = ax1.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax1.set_title('April, fixed zero yaw', size=14*scale)
    ax1.set_xlabel('tip speed ratio $\lambda$')
    ax1.set_ylabel('Tower FA bending [Nm]')
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

    # read a single file for debugging/checking
#    case = '0213_run_108_8.0ms_dc1_samoerai_fixyaw_pwm1000_highrpm'
#    res = ojfresult.ComboResults(PATH_DB+'symlinks/', case)
#    stats = res.statistics()
