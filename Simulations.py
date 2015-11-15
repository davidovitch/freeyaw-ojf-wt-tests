# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:16:34 2011

@author: dave
"""

# NOTE TO SELF: PLOTTING IN DIFFERENT MODULE. KEEP THIS AS SIMPLE AS POSSIBLE
# for the the dependencies, so you can run it easily on the cluster

from __future__ import division

# standard python library
import os
import subprocess as sproc
import copy
import zipfile
import shutil
import datetime
import math
import pickle
# what is actually the difference between warnings and logging.warn?
# for which context is which better?
#import warnings
import logging
from operator import itemgetter
from time import time

# numpy and scipy only used in HtcMaster._all_in_one_blade_tag
import numpy as np
import scipy
import scipy.interpolate as interpolate
import pylab as plt

# custom libraries
import misc
import HawcPy

def load_pickled_file(source):
    FILE = open(source, 'rb')
    result = pickle.load(FILE)
    FILE.close()
    return result

def save_pickle(source, variable):
    FILE = open(source, 'wb')
    pickle.dump(variable, FILE, protocol=2)
    FILE.close()

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

def create_multiloop_list(iter_dict, debug=False):
    """
    Create a list based on multiple nested loops
    ============================================

    Considerd the following example

    >>> for v in range(V_start, V_end, V_delta):
    >>>     for y in range(y_start, y_end, y_delta):
    >>>         for c in range(c_start, c_end, c_delta):
    >>>             print v, y, c

    Could be replaced by a list with all these combinations. In order to
    replicate this with create_multiloop_list, iter_dict should have
    the following structure

    >>> iter_dict = dict()
    >>> iter_dict['v'] = range(V_start, V_end, V_delta)
    >>> iter_dict['y'] = range(y_start, y_end, y_delta)
    >>> iter_dict['c'] = range(c_start, c_end, c_delta)
    >>> iter_list = create_multiloop_list(iter_dict)
    >>> for case in iter_list:
    >>>     print case['v'], case['y'], case['c']

    Parameters
    ----------

    iter_dict : dictionary
        Key holds a valid tag as used in HtcMaster.tags. The corresponding
        value shouuld be a list of values to be considered.

    Output
    ------

    iter_list : list
        List containing dictionaries. Each entry is a combination of the
        given iter_dict keys.

    Example
    -------

    >>> iter_dict={'[wind]':[5,6,7],'[coning]':[0,-5,-10]}
    >>> create_multiloop_list(iter_dict)
    [{'[wind]': 5, '[coning]': 0},
     {'[wind]': 5, '[coning]': -5},
     {'[wind]': 5, '[coning]': -10},
     {'[wind]': 6, '[coning]': 0},
     {'[wind]': 6, '[coning]': -5},
     {'[wind]': 6, '[coning]': -10},
     {'[wind]': 7, '[coning]': 0},
     {'[wind]': 7, '[coning]': -5},
     {'[wind]': 7, '[coning]': -10}]
    """

    iter_list = []

    # fix the order of the keys
    key_order = iter_dict.keys()
    nr_keys = len(key_order)
    nr_values,indices = [],[]
    # determine how many items on each key
    for key in key_order:
        # each value needs to be an iterable! len() will fail if it isn't
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
        iter_list.append(dict())

        # save all the different combination into one list
        for keyi in range(len(key_order)):
            key = key_order[keyi]
            # add the current combination of values as one dictionary
            iter_list[cc][key] = iter_dict[key][indices[keyi]]

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

    return iter_list

def local_shell_script(htc_dict, sim_id):
    """
    """
    shellscript = ''
    breakline = '"' + '*'*80 + '"'
    nr_cases = len(htc_dict)
    nr = 1
    for case in htc_dict:
        shellscript += 'echo ""' + '\n'
        shellscript += 'echo ' + breakline + '\n' + 'echo '
        shellscript += '" ===> Progress:'+str(nr)+'/'+str(nr_cases)+'"\n'
        # get a shorter version for the current cases tag_dict:
        scriptpath = htc_dict[case]['[run_dir]'] + 'runall.sh'
        #shellscript += 'cd /home/dave/Projects/0_Risø_NDA/HAWC2/run'
        wine = 'WINEDEBUG=-all WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        shellscript += wine + " hawc2mb.exe htc/" + case + "\n"
        shellscript += 'echo ' + breakline + '\n'
        nr+=1

    write_file(scriptpath, shellscript, 'w')
    print '\nrun local shell script written to:'
    print scriptpath

def run_local(cases, silent=False, check_log=True):
    """
    Run all HAWC2 simulations locally from cases
    ===============================================

    Run all case present in a cases dict locally and wait until HAWC2 is ready.

    In verbose mode, each HAWC2 simulation is also timed

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary holding
        all the tags/value pairs as used for that case

    check_log : boolean, default=False
        Check the log file emmidiately after execution of the HAWC2 case

    silent : boolean, default=False
        When False, usefull information will be printed and the HAWC2
        simulation time will be calculated from the Python perspective. The
        silent variable is also passed on to logcheck_case

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Update cases with the STDOUT of the respective HAWC2 simulation

    """

    # remember the current working directory
    cwd = os.getcwd()
    nr = len(cases)
    if not silent:
        print ''
        print '='*79
        print 'Be advised, launching %i HAWC2 simulation(s) sequentially' % nr
        print 'run dir: %s' % cases[cases.keys()[0]]['[run_dir]']
        print ''

    if check_log:
        errorlogs = ErrorLogs(silent=silent)

    for ii, case in enumerate(cases):
        # all tags for the current case
        tags = cases[case]
        # the launch command
        cmd  = 'WINEDEBUG=-all WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        cmd += " hawc2mb.exe " + tags['[htc_dir]'] + case
        # remove any escaping in tags and case for security reasons
        cmd = cmd.replace('\\','')
        # browse to the correct launch path for the HAWC2 simulation
        os.chdir(tags['[run_dir]'])

        if not silent:
            start = time()
            progress = '%4i/%i  : %s%s' % (ii+1, nr, tags['[htc_dir]'], case)
            print '*'*75
            print progress

        # and launch the HAWC2 simulation
        p = sproc.Popen(cmd,stdout=sproc.PIPE,stderr=sproc.STDOUT,shell=True)
        # save the output that HAWC2 sends to the shell to the cases
        # note that this is a list, each item holding a line
        cases[case]['sim_STDOUT'] = p.stdout.readlines()
        # wait until HAWC2 finished doing its magic
        p.wait()

        if not silent:
            # print the simulation command line output
            print ' ' + '-'*75
            print ''.join(cases[case]['sim_STDOUT'])
            print ' ' + '-'*75
            # caclulation time
            stp = time() - start
            stpmin = stp/60.
            print 'HAWC2 execution time: %8.2f sec (%8.2f min)' % (stp,stpmin)

        # where there any errors in the output? If yes, abort
        for k in cases[case]['sim_STDOUT']:
            kstart = k[:14]
            if kstart in [' *** ERROR ***', 'forrtl: severe']:
                cases[case]['[hawc2_sim_ok]'] = False
                #raise UserWarning, 'Found error in HAWC2 STDOUT'
            else:
                cases[case]['[hawc2_sim_ok]'] = True

        # check the log file strait away if required
        if check_log:
            start = time()
            errorlogs = logcheck_case(errorlogs, cases, case, silent=silent)
            stop = time() - start
            if case.endswith('.htc'):
                kk = case[:-4] + '.log'
            else:
                kk = case + '.log'
            errors = errorlogs.MsgListLog2[kk][0]
            exitok = errorlogs.MsgListLog2[kk][1]
            if not silent:
                print 'log checks took %5.2f sec' % stop
                print '    found error: ', errors
                print ' exit correctly: ', exitok
                print '*'*75
                print
            # also save in cases
            if not errors and exitok:
                cases[case]['[hawc2_sim_ok]'] = True
            else:
                cases[case]['[hawc2_sim_ok]'] = False

    if check_log:
        # take the last case to determine sim_id, run_dir and log_dir
        sim_id = cases[case]['[sim_id]']
        run_dir = cases[case]['[run_dir]']
        log_dir = cases[case]['[log_dir]']
        # save the extended (.csv format) errorlog list?
        # but put in one level up, so in the logfiles folder directly
        errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
        # use the model path of the last encoutered case in cases
        errorlogs.PathToLogs = run_dir + log_dir
        errorlogs.save()

    # just in case, browse back the working path relevant for the python magic
    os.chdir(cwd)
    if not silent:
        print '\nHAWC2 has done all of its sequential magic!'
        print '='*79
        print ''

    return cases


def prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                write_htc=True, runmethod='local', verbose=False,
                copyback_turb=True, msg='', silent=False, check_log=True):
    """
    Create the htc files, pbs scripts and replace the tags in master file
    =====================================================================

    Do not use any uppercase letters in the filenames, since HAWC2 will
    convert all of them to lower case results file names (.sel, .dat, .log)

    create sub folders according to sim_id, in order to not create one
    folder for the htc, results, logfiles which grows very large in due
    time!!

    opt_tags is a list of dictionaries of tags:
        [ {tag1=12,tag2=23,..},{tag1=11, tag2=33, tag9=5,...},...]
    for each wind, yaw and coning combi, each tag dictionary in the list
    will be set.

    Make sure to always define all dictionary keys in each list, otherwise
    the value of the first appareance will remain set for the remaining
    simulations in the list.
    For instance, in the example above, if tag9=5 is not set for subsequent
    lists, tag9 will remain having value 5 for these subsequent sets

    The tags for each case are consequently set in following order (or
    presedence):
        * master
        * opt_tags
        * iter_dict
        * variable_tag_func

    Parameters
    ----------

    iter_dict : dict

    opt_tags : dict

    master : HtcMaster object

    variable_tag_func : function object

    write_htc : boolean, default=True

    verbose : boolean, default=False

    runmethod : {'local' (default),'thyra','gorm','local-script','none'}
        Specify how/what to run where. For local, each case in cases is
        run locally via python directly. If set to 'local-script' a shell
        script is written to run all cases locally sequential. If set to
        'thyra' or 'gorm', PBS scripts are written to the respective server.

    msg : str, default=''
        A descriptive message of the simulation series is saved at
        "post_dir + master.tags['[sim_id]'] + '_tags.txt'". Additionally, this
         tagfile also holds the opt_tags and iter_dict values.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary holding
        all the tags/value pairs as used for that case

    """

    cases = dict()

    # if empty, just create a dummy item so we get into the loops
    if len(iter_dict) == 0:
        iter_dict = {'__dummy__': [0]}
    combi_list = create_multiloop_list(iter_dict)

    # load the master htc file as a string under the master.tags
    master.loadmaster()

    # create the execution folder structure and copy all data to it
    master.copy_model_data()

    # create the zip file
    master.create_model_zip()

    # ignore if the opt_tags is empty, will result in zero
    if len(opt_tags) > 0:
        sim_total = len(combi_list)*len(opt_tags)
    else:
        sim_total = len(combi_list)
        # if no opt_tags specified, create an empty dummy tag
        opt_tags = [dict({'__DUMMY_TAG__' : 0})]
    sim_nr = 0

    # cycle thourgh all the combinations
    for it in combi_list:
        for ot in opt_tags:
            sim_nr += 1
            # update the tags from the opt_tags list
            if not '__DUMMY_TAG__' in ot:
                master.tags.update(ot)
            # update the tags set in the combi_list
            master.tags.update(it)
            # -----------------------------------------------------------
            # start variable tags update
            master = variable_tag_func(master)
            # end variable tags
            # -----------------------------------------------------------
            if not silent:
                print 'htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                       format(sim_total, '3.0f')

            if verbose:
                print '===master.tags===\n', master.tags

            # returns a dictionary with all the tags used for this
            # specific case
            htc = master.createcase(write_htc=write_htc)
            #htc=master.createcase_check(cases_repo,write_htc=write_htc)

            # make sure the current cases is unique!
            if htc.keys()[0] in cases:
                msg = 'non unique case in cases: %s' % htc.keys()[0]
                raise KeyError, msg

            # save in the big cases. Note that values() gives a copy!
            cases[htc.keys()[0]] = htc.values()[0]

            if verbose:
                print 'created cases for: ' + \
                    master.tags['[case_id]'] + '.htc\n'

    post_dir = master.tags['[post_dir]']

    # create directory if post_dir does not exists
    try:
        os.mkdir(post_dir)
    except OSError:
        pass
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'wb')
    pickle.dump(cases, FILE, protocol=2)
    FILE.close()

    if not silent:
        print '\ncases saved at:'
        print post_dir + master.tags['[sim_id]'] + '.pkl'

    # also save the iter_dict and opt_tags in a text file for easy reference
    # or quick checks on what each sim_id actually contains
    # sort the taglist for convienent reading/comparing
    tagfile = msg + '\n\n'
    tagfile += '='*79 + '\n'
    tagfile += 'iter_dict\n'.rjust(30)
    tagfile += '='*79 + '\n'
    iter_dict_list = sorted(iter_dict.iteritems(), key=itemgetter(0))
    for k in iter_dict_list:
        tagfile += str(k[0]).rjust(30) + ' : ' + str(k[1]).ljust(20) + '\n'

    tagfile += '\n'
    tagfile += '='*79 + '\n'
    tagfile += 'opt_tags\n'.rjust(30)
    tagfile += '='*79 + '\n'
    for k in opt_tags:
        tagfile += '\n'
        tagfile += '-'*79 + '\n'
        tagfile += 'opt_tags set\n'.rjust(30)
        tagfile += '-'*79 + '\n'
        opt_dict = sorted(k.iteritems(), key=itemgetter(0), reverse=False)
        for kk in opt_dict:
            tagfile += str(kk[0]).rjust(30)+' : '+str(kk[1]).ljust(20) + '\n'
    write_file(post_dir + master.tags['[sim_id]'] + '_tags.txt', tagfile, 'w')

    launch(cases, runmethod=runmethod, verbose=verbose,
           copyback_turb=copyback_turb, check_log=check_log)

    return cases

def prepare_relaunch(cases, runmethod='gorm', verbose=False, write_htc=True,
                     copyback_turb=True, silent=False, check_log=True):
    """
    Instead of redoing everything, we know recreate the HTC file for those
    in the given cases dict. Nothing else changes. The data and zip files
    are not updated, the convience tagfile is not recreated. However, the
    saved (pickled) cases dict corresponding to the sim_id is updated!

    This method is usefull to correct mistakes made for some cases.

    It is adviced to not change the case_id, sim_id, from the cases.
    """

    # initiate the HtcMaster object, load the master file
    master = HtcMaster()
    # for invariant tags, load random case. Necessary before we can load
    # the master file, otherwise we don't know which master to load
    master.tags = cases[cases.keys()[0]]
    master.loadmaster()

    # load the original cases dict
    post_dir = master.tags['[post_dir]']
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'rb')
    cases_orig = pickle.load(FILE)
    FILE.close()

    sim_nr = 0
    sim_total = len(cases)
    for case, casedict in cases.iteritems():
        sim_nr += 1

        # set all the tags in the HtcMaster file
        master.tags = casedict
        # returns a dictionary with all the tags used for this
        # specific case
        htc = master.createcase(write_htc=write_htc)
        #htc=master.createcase_check(cases_repo,write_htc=write_htc)

        if not silent:
            print 'htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                   format(sim_total, '3.0f')

        if verbose:
            print '===master.tags===\n', master.tags

        # make sure the current cases already exists, otherwise we are not
        # relaunching!
        if case not in cases_orig:
            msg = 'relaunch only works for existing cases: %s' % case
            raise KeyError, msg

        # save in the big cases. Note that values() gives a copy!
        # remark, what about the copying done at the end of master.createcase?
        # is that redundant then?
        cases[htc.keys()[0]] = htc.values()[0]

        if verbose:
            print 'created cases for: ' + \
                master.tags['[case_id]'] + '.htc\n'

    launch(cases, runmethod=runmethod, verbose=verbose, check_log=check_log,
           copyback_turb=copyback_turb, silent=silent)

    # update the original file: overwrite the newly set cases
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'wb')
    cases_orig.update(cases)
    pickle.dump(cases_orig, FILE, protocol=2)
    FILE.close()

def prepare_launch_cases(cases, runmethod='gorm', verbose=False,write_htc=True,
                         copyback_turb=True, silent=False, check_log=True,
                         variable_tag_func=None, sim_id_new=None):
    """
    Same as prepare_launch, but now the input is just a cases object (cao).
    If relaunching some earlier defined simulations, make sure to at least
    rename the sim_id, otherwise it could become messy: things end up in the
    same folder, sim_id post file get overwritten, ...

    In case you do not use a variable_tag_fuc, make sure all your tags are
    defined in cases. First and foremost, this means that the case_id does not
    get updated to have a new sim_id, the path's are not updated, etc

    When given a variable_tag_func, make sure it is properly
    defined: do not base a variable tag's value on itself to avoid value chains

    The master htc file will be loaded and alls tags defined in the cases dict
    will be applied to it as is.
    """

    # initiate the HtcMaster object, load the master file
    master = HtcMaster()
    # for invariant tags, load random case. Necessary before we can load
    # the master file, otherwise we don't know which master to load
    master.tags = cases[cases.keys()[0]]
    # load the master htc file as a string under the master.tags
    master.loadmaster()
    # create the execution folder structure and copy all data to it
    # but reset to the correct launch dirs first
    sim_id = master.tags['[sim_id]']
    if runmethod in ['local', 'local-script', 'none']:
        path = '/home/dave/PhD_data/HAWC2_results/ojf_post/%s/' % sim_id
        master.tags['[run_dir]'] = path
    elif runmethod == 'thyra':
        master.tags['[run_dir]'] = '/mnt/thyra/HAWC2/ojf_post/%s/' % sim_id
    elif runmethod == 'gorm':
        master.tags['[run_dir]'] = '/mnt/gorm/HAWC2/ojf_post/%s/' % sim_id
    else:
        msg='unsupported runmethod, options: none, local, thyra, gorm, opt'
        raise ValueError, msg
    master.copy_model_data()
    # create the zip file
    master.create_model_zip()

    sim_nr = 0
    sim_total = len(cases)

    # for safety, create a new cases dict. At the end of the ride both cases
    # and cases_new should be identical!
    cases_new = {}

    # cycle thourgh all the combinations
    for case, casedict in cases.iteritems():
        sim_nr += 1

        sim_id = casedict['[sim_id]']
        # reset the launch dirs
        if runmethod in ['local', 'local-script', 'none']:
            path = '/home/dave/PhD_data/HAWC2_results/ojf_post/%s/' % sim_id
            casedict['[run_dir]'] = path
        elif runmethod == 'thyra':
            casedict['[run_dir]'] = '/mnt/thyra/HAWC2/ojf_post/%s/' % sim_id
        elif runmethod == 'gorm':
            casedict['[run_dir]'] = '/mnt/gorm/HAWC2/ojf_post/%s/' % sim_id
        else:
            msg='unsupported runmethod, options: none, local, thyra, gorm, opt'
            raise ValueError, msg

        # -----------------------------------------------------------
        # set all the tags in the HtcMaster file
        master.tags = casedict
        # apply the variable tags if applicable
        if variable_tag_func:
            master = variable_tag_func(master)
        elif sim_id_new:
            # TODO: finish this
            # replace all the sim_id occurences with the updated one
            # this means also the case_id tag changes!
            pass
        # -----------------------------------------------------------

        # returns a dictionary with all the tags used for this specific case
        htc = master.createcase(write_htc=write_htc)

        if not silent:
            print 'htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                   format(sim_total, '3.0f')

        if verbose:
            print '===master.tags===\n', master.tags

        # make sure the current cases is unique!
        if htc.keys()[0] in cases_new:
            msg = 'non unique case in cases: %s' % htc.keys()[0]
            raise KeyError, msg
        # save in the big cases. Note that values() gives a copy!
        # remark, what about the copying done at the end of master.createcase?
        # is that redundant then?
        cases_new[htc.keys()[0]] = htc.values()[0]

        if verbose:
            print 'created cases for: ' + \
                master.tags['[case_id]'] + '.htc\n'

    post_dir = master.tags['[post_dir]']

    # create directory if post_dir does not exists
    try:
        os.mkdir(post_dir)
    except OSError:
        pass
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'wb')
    pickle.dump(cases_new, FILE, protocol=2)
    FILE.close()

    if not silent:
        print '\ncases saved at:'
        print post_dir + master.tags['[sim_id]'] + '.pkl'

    launch(cases_new, runmethod=runmethod, verbose=verbose,
           copyback_turb=copyback_turb, check_log=check_log)

    return cases_new



def launch(cases, runmethod='local', verbose=False, copyback_turb=True,
           silent=False, check_log=True):
    """
    The actual launching of all cases in the Cases dictionary. Note that here
    only the PBS files are written and not the actuall htc files.

    Parameters
    ----------

    cases : dict
        Dictionary with the case name as key and another dictionary as value.
        The latter holds all the tag/value pairs used in the respective
        simulation.

    verbose : boolean, default=False

    runmethod : {'local' (default),'thyra','gorm','local-script','none'}
        Specify how/what to run where. For local, each case in cases is
        run locally via python directly. If set to 'local-script' a shell
        script is written to run all cases locally sequential. If set to
        'thyra' or 'gorm', PBS scripts are written to the respective server.
    """

    random_case = cases.keys()[0]
    sim_id = cases[random_case]['[sim_id]']
    pbs_out_dir = cases[random_case]['[pbs_out_dir]']

    if runmethod == 'local-script':
        local_shell_script(cases, sim_id)
    elif runmethod in ['thyra','gorm']:
        # create the pbs object
        pbs = PBS(cases, server=runmethod)
        pbs.copyback_turb = copyback_turb
        pbs.verbose = verbose
        pbs.pbs_out_dir = pbs_out_dir
        pbs.create()
    elif runmethod == 'local':
        cases = run_local(cases, silent=silent, check_log=check_log)
    elif runmethod == 'none':
        pass
    else:
        msg = 'unsupported runmethod, valid options: local, thyra, gorm or opt'
        raise ValueError, msg

def post_launch(cases):
    """
    Do some basics checks: do all launched cases have a result and LOG file
    and are there any errors in the LOG files?

    Parameters
    ----------

    cases : either a string (path to file) or the cases itself
    """

    # TODO: finish support for default location of the cases and file name
    # two scenario's: either pass on an cases and get from their the
    # post processing path or pass on the simid and load from the cases
    # from the default location
    # in case run_local, do not check PBS!

    # in case it is a path, load the cases
    if type(cases).__name__ == 'str':
        cases = load_pickled_file(cases)

    # saving output to textfile and print at the same time
    LOG = Log()
    LOG.print_logging = True

    # load one case dictionary from the cases to get data that is the same
    # over all simulations in the cases
    master = cases.keys()[0]
    post_dir = cases[master]['[post_dir]']
    sim_id = cases[master]['[sim_id]']
    run_dir = cases[master]['[run_dir]']
    log_dir = cases[master]['[log_dir]']

    # for how many of the created cases are there actually result, log files
    pbs = PBS(cases)
    pbs.cases = cases
    cases_fail = pbs.check_results(cases)

    # add the failed cases to the LOG:
    LOG.add(['number of failed cases: ' + str(len(cases_fail))])
    LOG.add(list(cases_fail))
    # for k in cases_fail:
    #    print k

    # initiate the object to check the log files
    errorlogs = ErrorLogs()
    LOG.add(['checking ' + str(len(cases)) + ' LOG files...'])
    nr = 1
    nr_tot = len(cases)

    tmp = cases.keys()[0]
    print 'checking logs, path (from a random item in cases):'
    print run_dir + log_dir

    for k in cases:
        # if it did not fail, we can read the logfile, otherwise not
        if k not in cases_fail:
            # see if there is an htc extension still standing
            if k.endswith('.htc'):
                kk = k[:-4] + '.log'
            else:
                kk = k + '.log'
            errorlogs.PathToLogs = run_dir + log_dir + kk
            errorlogs.check()
            print 'checking logfile progress: ' + str(nr) + '/' + str(nr_tot)
            nr += 1

            # if simulation did not ended correctly, put it on the fail list
            if not errorlogs.MsgListLog2[kk][1]:
                cases_fail[k] = cases[k]

    # now see how many cases resulted in an error and add to the general LOG
    # determine how long the first case name is
    spacing = len(errorlogs.MsgListLog2.keys()[0]) + 9
    LOG.add(['display log check'.ljust(spacing) + 'found_error?'.ljust(15) + \
            'exit_correctly?'])
    for k in errorlogs.MsgListLog2:
        LOG.add([k.ljust(spacing)+str(errorlogs.MsgListLog2[k][0]).ljust(15)+\
            str(errorlogs.MsgListLog2[k][1]) ])
    # save the extended (.csv format) errorlog list?
    # but put in one level up, so in the logfiles folder directly
    errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
    # use the model path of the last encoutered case in cases
    errorlogs.PathToLogs = run_dir + log_dir
    errorlogs.save()

    # save the error LOG list, this is redundant, since it already exists in
    # the general LOG file (but only as a print, not the python variable)
    tmp = post_dir + sim_id + '_MsgListLog2'
    save_pickle(tmp, errorlogs.MsgListLog2)

    # save the list of failed cases
    save_pickle(post_dir + sim_id + '_fail.pkl', cases_fail)

    return cases_fail

def logcheck_case(errorlogs, cases, case, silent=False):
    """
    Check logfile of a single case
    ==============================

    Given the cases and a case, check that single case on errors in the
    logfile.

    """

    #post_dir = cases[case]['[post_dir]']
    #sim_id = cases[case]['[sim_id]']
    run_dir = cases[case]['[run_dir]']
    log_dir = cases[case]['[log_dir]']
    if case.endswith('.htc'):
        caselog = case[:-4] + '.log'
    else:
        caselog = case + '.log'
    errorlogs.PathToLogs = run_dir + log_dir + caselog
    errorlogs.check()

    # in case we find an error, abort or not?
    errors = errorlogs.MsgListLog2[caselog][0]
    exitcorrect = errorlogs.MsgListLog2[caselog][1]
    if errors:
        # print all error messages
        #logs.MsgListLog : [ [case, line nr, error1, line nr, error2, ....], ]
        # difficult: MsgListLog is not a dict!!
        #raise UserWarning, 'HAWC2 simulation has errors in logfile, abort!'
        #warnings.warn('HAWC2 simulation has errors in logfile!')
        logging.warn('HAWC2 simulation has errors in logfile!')
    elif not exitcorrect:
        #raise UserWarning, 'HAWC2 simulation did not ended correctly, abort!'
        #warnings.warn('HAWC2 simulation did not ended correctly!')
        logging.warn('HAWC2 simulation did not ended correctly!')

    # no need to do that, aborts on failure anyway and OK log check will be
    # printed in run_local when also printing how long it took to check
    #if not silent:
        #print 'log checks ok'
        #print '   found error: %s' % errorlogs.MsgListLog2[caselog][0]
        #print 'exit correctly: %s' % errorlogs.MsgListLog2[caselog][1]

    return errorlogs

    ## save the extended (.csv format) errorlog list?
    ## but put in one level up, so in the logfiles folder directly
    #errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
    ## use the model path of the last encoutered case in cases
    #errorlogs.PathToLogs = run_dir + log_dir
    #errorlogs.save()

def get_htc_dict(post_dir, simid):
    """
    Load the htc_dict, remove failed cases
    """
    htc_dict = load_pickled_file(post_dir + simid + '.pkl')

    # if the post processing is done on simulations done by thyra/gorm, and is
    # downloaded locally, change path to results
    for case in htc_dict:
        if htc_dict[case]['[run_dir]'][:4] == '/mnt':
            path = '/home/dave/PhD_data/HAWC2_results/ojf_post/' +simid +'/'
            htc_dict[case]['[run_dir]'] = path

    try:
        htc_dict_fail = load_pickled_file(post_dir + simid + '_fail.pkl')
    except IOError:
        return htc_dict

    # ditch all the failed cases out of the htc_dict
    # otherwise we will have fails when reading the results data files
    for k in htc_dict_fail:
        del htc_dict[k]
        print 'removed from htc_dict due to error: ' + k

    return htc_dict

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

class HtcMaster:
    """
    """

    def __init__(self, verbose=False, silent=False):
        """
        """

        # TODO: make HtcMaster callable, so that when called you actually
        # set a value for a certain tag or add a new one. In doing so,
        # you can actually warn when you are overwriting a tag, or when
        # a different tag has the same name, etc

        # create a dictionary with the tag name as key as the default value
        self.tags = dict()

        # should we print where the file is written?
        self.verbose = verbose
        self.silent = silent

        # following tags are required
        #---------------------------------------------------------------------
        self.tags['[case_id]'] = None

        self.tags['[master_htc_file]'] = None
        self.tags['[master_htc_dir]'] = None
        # path to model zip file, needs to accessible from the server
        # relative from the directory where the pbs files are launched on the
        # server. Suggestions is to always place the zip file in the model
        # folder, so only the zip file name has to be defined
        self.tags['[model_zip]'] = None

        # path to HAWTOPT blade result file: quasi/res/blade.dat
        self.tags['[blade_hawtopt_dir]'] = None
        self.tags['[blade_hawtopt]'] = None
        self.tags['[zaxis_fact]'] = 1.0
        # TODO: rename to execution dir, that description fits much better!
        self.tags['[run_dir]'] = None
        #self.tags['[run_dir]'] = '/home/dave/tmp/'

        # following dirs are relative to the run_dir!!
        # they indicate the location of the SAVED (!!) results, they can be
        # different from the execution dirs on the node which are set in PBS
        self.tags['[res_dir]'] = 'results/'
        self.tags['[log_dir]'] = 'logfiles/'
        self.tags['[turb_dir]'] = 'turb/'
        self.tags['[animation_dir]'] = 'animation/'
        self.tags['[eigenfreq_dir]'] = 'eigenfreq/'
        self.tags['[wake_dir]'] = 'wake/'
        self.tags['[meander_dir]'] = 'meander/'
        self.tags['[htc_dir]'] = 'htc/'
        self.tags['[pbs_out_dir]'] = 'pbs_out/'
        self.tags['[turb_base_name]'] = 'turb_'
        self.tags['[wake_base_name]'] = 'turb_'
        self.tags['[meand_base_name]'] = 'turb_'

        self.tags['[pbs_queue_command]'] = '#PBS -q workq'
        # the express que has 2 thyra nodes with max walltime of 1h
#        self.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
        # walltime should have following format: hh:mm:ss
        self.tags['[walltime]'] = '04:00:00'

    def copy_model_data(self):
        """

        Copy the model data to the execution folder

        """

        # create the remote folder structure
        if not os.path.exists(self.tags['[run_dir]']):
            os.makedirs(self.tags['[run_dir]'])
        # the data folder
        data_run = self.tags['[run_dir]'] + self.tags['[data_dir]']
        if not os.path.exists(data_run):
            os.makedirs(data_run)
        # the htc folder
        path = self.tags['[run_dir]'] + self.tags['[htc_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the results dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[res_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the logfile dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[log_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the eigenfreq dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[eigenfreq_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the animation dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[animation_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[turb_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[wake_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[meander_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[opt_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[control_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        # data files in data folder
        st = self.tags['[st_file]']
        ae = self.tags['[ae_file]']
        pc = self.tags['[pc_file]']
        ct_target = self.tags['[run_dir]'] + self.tags['[control_dir]']

        data_local = self.tags['[model_dir_local]']+self.tags['[data_dir]']
        cont_local = self.tags['[model_dir_local]']+self.tags['[control_dir]']

        # in case we are running local and the model dir is the server dir
        # we do not need to copy the data files, they are already on location
        if not data_local == data_run:
            shutil.copy2(data_local +st, data_run +st)
            shutil.copy2(data_local +ae, data_run +ae)
            shutil.copy2(data_local +pc, data_run +pc)
            # copy all files present in the control folder
            for root, dirs, files in os.walk(cont_local):
                for file_name in files:
                    src = os.path.join(root,file_name)
                    dst = ct_target + file_name
                    shutil.copy2(src, dst)


    def create_model_zip(self):
        """

        Create the model zip file based on the master tags file settings.

        Paremeters
        ----------

        master : HtcMaster object


        """

        # FIXME: all directories should be called trough their appropriate tag!

        #model_dir = HOME_DIR + 'PhD/Projects/Hawc2Models/'+MODEL+'/'
        model_dir_server = self.tags['[run_dir]']

        model_dir_local = self.tags['[model_dir_local]']
        data_dir_local = model_dir_local + self.tags['[data_dir]']

        # ---------------------------------------------------------------------
        # create the zipfile object locally
        zf = zipfile.ZipFile(model_dir_local + self.tags['[model_zip]'],'w')

        # empty folders, the'll hold the outputs
        # zf.write(source, target in zip, )
        zf.write('.', 'animation/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'control/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'eigenfreq/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'htc/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'logfiles/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'results/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'turb/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'wake/.', zipfile.ZIP_DEFLATED)
        zf.write('.', 'meander/.', zipfile.ZIP_DEFLATED)

        # data files in data folder
        st = self.tags['[st_file]']
        ae = self.tags['[ae_file]']
        pc = self.tags['[pc_file]']
        zf.write(data_dir_local + st, 'data/'+st, zipfile.ZIP_DEFLATED )
        zf.write(data_dir_local + ae, 'data/'+ae, zipfile.ZIP_DEFLATED )
        zf.write(data_dir_local + pc, 'data/'+pc, zipfile.ZIP_DEFLATED )

        # manually add all that resides in control
        target_path = model_dir_local + self.tags['[control_dir]']
        for root, dirs, files in os.walk(target_path):
            for file_name in files:
                #print 'adding', file_name
                zf.write(os.path.join(root,file_name), 'control/'+file_name,
                         zipfile.ZIP_DEFLATED)

        # and close again
        zf.close()

        # ---------------------------------------------------------------------
        # copy zip file to the server, this will be used on the nodes
        src = model_dir_local  + self.tags['[model_zip]']
        dst = model_dir_server + self.tags['[model_zip]']

        # in case we are running local and the model dir is the server dir
        # we do not need to copy the zip file, it is already on location
        if not src == dst:
            shutil.copy2(src, dst)

        ## copy to zip data file to sim_id htc folder on the server dir
        ## so we now have exactly all data to relaunch any htc file later
        #dst  = model_dir_server + self.tags['[htc_dir]']
        #dst += self.tags['[model_zip]']
        #shutil.copy2(src, dst)

    def _sweep_tags(self):
        """
        The original way with all tags in the htc file for each blade node
        """
        # set the correct sweep cruve, these values are used
        a = self.tags['[sweep_amp]']
        b = self.tags['[sweep_exp]']
        z0 = self.tags['[sweep_curve_z0]']
        ze = self.tags['[sweep_curve_ze]']
        nr = self.tags['[nr_nodes_blade]']
        # format for the x values in the htc file
        ff = ' 1.03f'
        for zz in range(nr):
            it_nosweep = '[x'+str(zz+1)+'-nosweep]'
            item = '[x'+str(zz+1)+']'
            z = self.tags['[z'+str(zz+1)+']']
            if z >= z0:
                curve = eval(self.tags['[sweep_curve_def]'])
                # new swept position = original + sweep curve
                self.tags[item]=format(self.tags[it_nosweep]+curve,ff)
            else:
                self.tags[item]=format(self.tags[it_nosweep], ff)

    def _all_in_one_blade_tag(self, radius_new=None):
        """
        Create htc input based on a HAWTOPT blade result file

        Automatically get the number of nodes correct in master.tags based
        on the number of blade nodes

        WARNING: initial x position of the half chord point is assumed to be
        zero

        zaxis_fact : int, default=1.0 --> is member of default tags
            Factor for the htc z-axis coordinates. The htc z axis is mapped to
            the HAWTOPT radius. If the blade radius develops in negative z
            direction, set to -1

        Parameters
        ----------

        radius_new : ndarray(n), default=False
            z coordinates of the nodes. If False, a linear distribution is
            used and the tag [nr--of-nodes-per-blade] sets the number of nodes


        """
        # TODO: implement support for x position to be other than zero

        # TODO: This is not a good place, should live somewhere else. Or
        # reconsider inputs etc so there is more freedom in changing the
        # location of the nodes, set initial x position of the blade etc

        # and save under tag [blade_htc_node_input] in htc input format

        nr_nodes = self.tags['[nr_nodes_blade]']

        blade = self.tags['[blade_hawtopt]']
        # in the htc file, blade root =0 and not blade hub radius
        blade[:,0] = blade[:,0] - blade[0,0]

        if type(radius_new).__name__ == 'NoneType':
            # interpolate to the specified number of nodes
            radius_new = np.linspace(blade[0,0], blade[-1,0], nr_nodes)

        # Data checks on radius_new
        elif not type(radius_new).__name__ == 'ndarray':
            raise ValueError, 'radius_new has to be either NoneType or ndarray'
        else:
            if not len(radius_new.shape) == 1:
                raise ValueError, 'radius_new has to be 1D'
            elif not len(radius_new) == nr_nodes:
                msg = 'radius_new has to have ' + str(nr_nodes) + ' elements'
                raise ValueError, msg

        # save the nodal positions in the tag cloud
        self.tags['[blade_nodes_z_positions]'] = radius_new

        # make sure that radius_hr is just slightly smaller than radius low res
        radius_new[-1] = blade[-1,0]-0.00000001
        twist_new = interpolate.griddata(blade[:,0], blade[:,2], radius_new)
        # blade_new is the htc node input part:
        # sec 1   x     y     z   twist;
        blade_new = scipy.zeros((len(radius_new),4))
        blade_new[:,2] = radius_new*self.tags['[zaxis_fact]']
        # twist angle remains the same in either case (standard/ojf rotation)
        blade_new[:,3] = twist_new*-1.

        # set the correct sweep cruve, these values are used
        a = self.tags['[sweep_amp]']
        b = self.tags['[sweep_exp]']
        z0 = self.tags['[sweep_curve_z0]']
        ze = self.tags['[sweep_curve_ze]']
        tmp = 'nsec ' + str(nr_nodes) + ';'
        for k in range(nr_nodes):
            tmp += '\n'
            i = k+1
            z = blade_new[k,2]
            y = blade_new[k,1]
            twist = blade_new[k,3]
            # x position, sweeping?
            if z >= z0:
                x = eval(self.tags['[sweep_curve_def]'])
            else:
                x = 0.0

            # the node number
            tmp += '        sec ' + format(i, '2.0f')
            tmp += format(x, ' 11.03f')
            tmp += format(y, ' 11.03f')
            tmp += format(z, ' 11.03f')
            tmp += format(twist, ' 11.03f')
            tmp += ' ;'

        self.tags['[blade_htc_node_input]'] = tmp

        # and create the ae file
        #5	Blade Radius [m] 	Chord[m]  T/C[%]  Set no. of pc file
        #1 25 some comments
        #0.000     0.100    21.000   1
        nr_points = blade.shape[0]
        tmp2 = '1  Blade Radius [m] Chord [m] T/C [%] pc file set nr\n'
        tmp2 += '1  %i auto generated by _all_in_one_blade_tag()' % nr_points

        for k in range(nr_points):
            tmp2 += '\n'
            tmp2 += '%9.3f %9.3f %9.3f' % (blade[k,0], blade[k,1], blade[k,3])
            tmp2 += ' %4i' % (k+1)
        # end with newline
        tmp2 += '\n'

        # TODO: finish writing file, implement proper handling of hawtopt path
        # and save the file
        #if self.tags['aefile']
        #write_file(file_path, tmp2, 'w')


    def loadmaster(self):
        """
        Load the master file, path to master file is defined in
        __init__(): target, master
        """

        # what is faster, load the file in one string and do replace()?
        # or the check error log approach?

        path_to_master  = self.tags['[master_htc_dir]']
        path_to_master += self.tags['[master_htc_file]']

        # load the file:
        if not self.silent:
            print 'loading master: ' + path_to_master
        FILE = open(path_to_master, 'r')
        lines = FILE.readlines()
        FILE.close()

        # convert to string:
        self.master_str = ''
        for line in lines:
            self.master_str += line

    def createcase_check(self, htc_dict_repo, \
                            tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        Check if a certain case name already exists in a specified htc_dict.
        If true, return a message and do not create the case. It can be that
        either the case name is a duplicate and should be named differently,
        or that the simulation is a duplicate and it shouldn't be repeated.
        """

        # is the [case_id] tag unique, given the htc_dict_repo?
        if self.verbose:
            print 'checking if following case is in htc_dict_repo: '
            print self.tags['[case_id]'] + '.htc'

        if htc_dict_repo.has_key(self.tags['[case_id]'] + '.htc'):
            # if the new case_id already exists in the htc_dict_repo
            # do not add it again!
            # print 'case_id key is not unique in the given htc_dict_repo!'
            raise UserWarning, \
                'case_id key is not unique in the given htc_dict_repo!'
        else:
            htc = self.createcase(tmp_dir=tmp_dir, write_htc=write_htc)
            return htc

    def createcase(self, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        replace all the tags from the master file and save the new htc file
        """

        htc = self.master_str

        # and now replace all the tags in the htc master file
        # when iterating over a dict, it will give the key, given in the
        # corresponding format (string keys as strings, int keys as ints...)
        for k in self.tags:
            value = self.tags[k]
            # TODO: give error if a default is not defined, like null
            # if it is a boolean, replace with ; or blank
            if type(self.tags[k]).__name__ == 'bool' and self.tags[k]:
                # we have a boolean that is True, switch it on
                value = ''
            elif type(self.tags[k]).__name__ == 'bool' and not self.tags[k]:
                value = ';'
            # if string is not found, it will do nothing
            htc = htc.replace(k, str(value))

        # and save the the case htc file:
        case = self.tags['[case_id]'] + '.htc'

        htc_target=self.tags['[run_dir]']+self.tags['[htc_dir]']
        if not self.silent:
            print 'htc will be written to: '
            print '  ' + htc_target
            print '  ' + case

        # and write the htc file to the temp dir first
        if write_htc:
            write_file(htc_target + case, htc, 'w')
            # write_file(tmp_dir + case, htc, 'w')

        # return the used tags, some parameters can be used later, such as the
        # turbulence name in the pbs script
        # return as a dictionary, to be used in htc_dict
        tmp = dict()
        # return a copy of the tags, otherwise you will not catch changes
        # made to the different tags in your sim series
        tmp[case] = copy.copy(self.tags)
        return tmp

class PBS:
    """
    The part where the actual pbs script is writtin in this class (functions
    create(), starting() and ending() ) is based on the MS Excel macro
    written by Torben J. Larsen

    input a list with htc file names, and a dict with the other paths,
    such as the turbulence file and folder, htc folder and others
    """

    def __init__(self, htc_dict, server='thyra'):
        """
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
        self.verbose = True

        if server == 'thyra':
            self.maxcpu = 4
            self.secperiter = 0.020
        elif server == 'gorm':
            self.maxcpu = 1
            self.secperiter = 0.012
        else:
            raise UserWarning, 'server support only for thyra or gorm'

        # the output channels comes with a price tag. Each time step
        # will have a penelty depending on the number of output channels

        self.iterperstep = 8.0 # average nr of iterations per time step
        # lead time: account for time losses when starting a simulation,
        # copying the turbulence data, generating the turbulence
        self.tlead = 5.0*60.0

        # pbs script prefix, this name will show up in the qstat listings
        self.pref = 'HAWC2_'
        self.pbs_dir = ''
        # the actual script starts empty
        self.pbs = ''

        self.htc_dict = htc_dict

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
        self.WakeDirName_node = 'wake/' # wake
        self.MeanderDirName_node = 'meander/' # meander
        # these dirs are also specified in the htc file itself, since that
        # will run local on the server node. Therefore they do not need to be
        # tag and changed

        # for the start number, take hour/minute combo
        d = datetime.datetime.today()
        tmp = int( str(d.hour)+format(d.minute, '02.0f') )*100
        self.pbs_start_number = tmp
        self.copyback_turb = True


    def create(self):
        """
        Main loop for creating the pbs scripts, based on the htc_dict, which
        contains the case name as key and tag dictionairy as value
        """

        # dynamically set walltime based on the number of time steps
        # for thyra, make a list so we base the walltime on the slowest case
        self.nr_time_steps = []
        self.duration = []
        self.t0 = []
        # '[time_stop]' '[dt_sim]'

        # REMARK: this i not realy consistent with how the result and log file
        # dirs are allowed to change for each individual case...
        # first check if the pbs_out_dir exists, this dir is considered to be
        # the same for all cases present in the htc_dict
        # self.tags['[run_dir]']
        case0 = self.htc_dict.keys()[0]
        path = self.htc_dict[case0]['[run_dir]'] + self.pbs_out_dir
        if not os.path.exists(path):
            os.makedirs(path)

        # number the pbs jobs:
        count2 = self.pbs_start_number
        # initial cpu count is zero
        count1 = 1
        # scan through all the cases
        i, i_tot = 1, len(self.htc_dict)
        ended = True

        for case in self.htc_dict:

            # get a shorter version for the current cases tag_dict:
            tag_dict = self.htc_dict[case]

            # group all values loaded from the tag_dict here, to keep overview
            # the directories to SAVE the results/logs/turb files
            # load all relevant dir settings: the result/logfile/turbulence/zip
            # they are now also available for starting() and ending() parts
            self.results_dir = tag_dict['[res_dir]']
            self.eigenfreq_dir = tag_dict['[eigenfreq_dir]']
            self.logs_dir = tag_dict['[log_dir]']
            self.animation_dir = tag_dict['[animation_dir]']
            self.TurbDirName = tag_dict['[turb_dir]']
            self.WakeDirName = tag_dict['[wake_dir]']
            self.MeanderDirName = tag_dict['[meander_dir]']
            self.ModelZipFile = tag_dict['[model_zip]']
            self.htc_dir = tag_dict['[htc_dir]']
            self.model_path = tag_dict['[run_dir]']
            self.turb_base_name = tag_dict['[turb_base_name]']
            self.wake_base_name = tag_dict['[wake_base_name]']
            self.meand_base_name = tag_dict['[meand_base_name]']
            self.pbs_queue_command = tag_dict['[pbs_queue_command]']
            self.walltime = tag_dict['[walltime]']
            self.dyn_walltime = tag_dict['[auto_walltime]']

            # related to the dynamically setting the walltime
            duration = float(tag_dict['[time_stop]'])
            dt = float(tag_dict['[dt_sim]'])
            self.nr_time_steps.append(duration/dt)
            self.duration.append(float(tag_dict['[duration]']))
            self.t0.append(float(tag_dict['[t0]']))

            if self.verbose:
                print 'htc_dir in pbs.create:'
                print self.htc_dir
                print self.model_path

            # CAUTION: for copying to the node, you should have the same
            # directory structure as in the zip file!!!
            # only when COPY BACK from the node, place in the custom dir

            # we only start a new case, if we have something that ended before
            # the very first case has to start with starting
            if ended:
                count1 = 1
                # define the path for the new pbs script
                jobid = self.pref + str(count2)
                pbs_path = self.model_path + self.pbs_dir + jobid + ".p"
                # Start a new pbs script, we only need the tag_dict here
                self.starting(tag_dict, jobid)
                ended = False

            # -----------------------------------------------------------------
            # WRITING THE ACTUAL JOB PARAMETERS
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
                self.turb_base_name + "?.bin" + \
                " ./"+self.TurbDirName_node + '\n'

            self.pbs += "cp -R $PBS_O_WORKDIR/" + self.WakeDirName + \
                self.wake_base_name + "?.bin" + \
                " ./"+self.WakeDirName_node + '\n'

            self.pbs += "cp -R $PBS_O_WORKDIR/" + self.MeanderDirName + \
                self.meand_base_name + "?.bin" + \
                " ./"+self.MeanderDirName_node + '\n'

            # the hawc2 execution commands via wine
            self.pbs += "wine HAWC2MB ./" +self.htc_dir_node + case +" &\n"
            #self.pbs += "wine get_mac_adresses" + '\n'
            # self.pbs += "cp -R ./*.mac  $PBS_O_WORKDIR/." + '\n'
            # -----------------------------------------------------------------

            # and we end when the cpu's per node are full
            if int(count1/self.maxcpu) == 1:
                # write the end part of the pbs script
                self.ending(pbs_path)
                ended = True
                # print progress:
                replace = ((i/self.maxcpu), (i_tot/self.maxcpu), self.walltime)
                print 'pbs script %3i/%i walltime=%s' % replace

            count2 += 1
            i += 1
            # the next cpu
            count1 += 1

        # it could be that the last node was not fully loaded. In that case
        # we do not have had a succesfull ending, and we still need to finish
        if not ended:
            # write the end part of the pbs script
            self.ending(pbs_path)
            # progress printing
            replace = ( (i/self.maxcpu), (i_tot/self.maxcpu), self.walltime )
            print 'pbs script %3i/%i walltime=%s, partially loaded' % replace
#            print 'pbs progress, script '+format(i/self.maxcpu,'2.0f')\
#                + '/' + format(i_tot/self.maxcpu, '2.0f') \
#                + ' partially loaded...'


    def starting(self, tag_dict, jobid):
        """
        First part of the pbs script
        """

        # a new clean pbs script!
        self.pbs = ''
        self.pbs += "### Standard Output" + ' \n'

        # PBS job name
        self.pbs += "#PBS -N %s \n" % (jobid)
        self.pbs += "#PBS -o ./" + self.pbs_out_dir + jobid + ".out" + '\n'
        # self.pbs += "#PBS -o ./pbs_out/" + jobid + ".out" + '\n'
        self.pbs += "### Standard Error" + ' \n'
        self.pbs += "#PBS -e ./" + self.pbs_out_dir + jobid + ".err" + '\n'
        # self.pbs += "#PBS -e ./pbs_out/" + jobid + ".err" + '\n'
        self.pbs += "### Maximum wallclock time format HOURS:MINUTES:SECONDS\n"
#        self.pbs += "#PBS -l walltime=" + self.walltime + '\n'
        self.pbs += "#PBS -l walltime=[walltime]\n"
        self.pbs += "#PBS -a [start_time]" + '\n'
        # in case of gorm, we need to make it work correctly. Now each job
        # has a different scratch dir. If we set maxcpu to 12 they all have
        # the same scratch dir. In that case there should be done something
        # differently

        # only do for thyra, not sure how to deal with on gorm
        if self.server == 'thyra':
            # Number of nodes and cpus per node (ppn)
            lnodes = int(math.ceil(len(self.htc_dict)/float(self.maxcpu)))
            lnodes = 1
            self.pbs += "#PBS -lnodes=%i:ppn=%i\n" % (lnodes, self.maxcpu)
        # specify the number of nodes and cpu's per node required
        #PBS -lnodes=4:ppn=1
        # Number of nodes and cpus per node (ppn)

        self.pbs += "### Queue name" + '\n'
        # queue names for Thyra are as follows:
        # short walltime queue (shorter than an hour): '#PBS -q xpresq'
        # or otherwise for longer jobs: '#PBS -q workq'
        self.pbs += self.pbs_queue_command + '\n'
        self.pbs += "cd $PBS_O_WORKDIR" + '\n'
        # output the current scratch directory
        self.pbs += "pwd \n"
        self.pbs += "### Copy to scratch directory \n"

        if self.server == 'thyra':
            for i in range(1,self.maxcpu+1,1):
                # create for each cpu a different directory on the node
                self.pbs += "mkdir /scratch/$USER/CPU_" + str(i) + '\n'
                # output the current scratch directory
                self.pbs += "pwd \n"
                # self.pbs += "cp -R hawc2_model/ /scratch/$USER/CPU_" + i
                # copy the zip files to the cpu dir on the node
                self.pbs += "cp -R ./" + self.ModelZipFile + \
                    " /scratch/$USER/CPU_" + str(i) + '\n'

        elif self.server == 'gorm':
            # output the current scratch directory
            self.pbs += "pwd \n"
            # copy the zip files to the cpu dir on the node
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
                self.pbs += "cp -R " + self.WakeDirName_node + \
                    ". $PBS_O_WORKDIR/" + self.WakeDirName + ".\n"
                self.pbs += "cp -R " + self.MeanderDirName_node + \
                    ". $PBS_O_WORKDIR/" + self.MeanderDirName + ".\n"
            # Delete the batch file at the end. However, is this possible since
            # the batch file is still open at this point????
            # self.pbs += "rm "

        # base walltime on the longest simulation in the batch
        nr_time_steps = max(self.nr_time_steps)
        # TODO: take into acccount the difference between time steps with
        # and without output. This penelaty also depends on the number of
        # channels outputted. So from 0 until t0 we have no penalty,
        # from t0 until t0+duration we have the output penalty.

        # always a predifined lead time to account for startup losses
        tmax = int(nr_time_steps*self.secperiter*self.iterperstep + self.tlead)
        if self.dyn_walltime:
            dt_seconds = datetime.datetime.fromtimestamp(tmax)
            self.walltime = dt_seconds.strftime('%H:%M:%S')
            self.pbs = self.pbs.replace('[walltime]', self.walltime)
        else:
            self.pbs = self.pbs.replace('[walltime]', self.walltime)
        # and reset the nr_time_steps list for the next pbs job file
        self.nr_time_steps = []
        self.t0 = []
        self.duration = []

        # TODO: add logfile checking support directly here. In that way each
        # node will do the logfile checking and statistics calculations right
        # after the simulation. Figure out a way how to merge the data from
        # all the different cases afterwards

        self.pbs += "exit"

        if self.verbose:
            print 'writing pbs script to path: ' + pbs_path

        # and write the script to a file:
        write_file(pbs_path,self.pbs, 'w')
        # make the string empty again, for memory
        self.pbs = ''

    def check_results(self, htc_dict):
        """
        Cross-check if all simulations on the list have returned a simulation.
        Combine with ErrorLogs to identify which errors occur where.

        All directory settings in the given htc_dict should be the same.

        It will look into the directories defined in:
            htc_dict[case]['[run_dir]'] = '/mnt/thyra/HAWC2/3e_yaw/'
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
        model_path = htc_dict[case]['[run_dir]']
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
        print 'path   log  files:', model_path + logs_dir

        for file in tmp[0][2]:
            # only select the .log files as a positive
            if file.endswith('.log'):
                # and remove the extension
                log_files.append(file[:-4])

        # result files
        tmp = []
        for files in os.walk(model_path + results_dir):
            tmp.append(files)

#        print '***RESULTFILES'
#        print tmp

        print 'path result files:', model_path + results_dir

        datok, selok = dict(), dict()

        for file in tmp[0][2]:

            if file.endswith('.dat'):
                # it can be that the .dat file is very small, or close to zero
                # in that case we had an error as well!
                size = os.stat(model_path + results_dir + file).st_size
                if size > 5:
                    # add the case name only, ditch the extension
                    datok[file[:-4]] = True

            elif file.endswith('.sel'):
                size = os.stat(model_path + results_dir + file).st_size
                if size > 5:
                    # add the case name only, ditch the extension
                    selok[file[:-4]] = True

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
            htc_list.append(k[:-4])

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

# TODO: rewrite the error log analysis to something better. Take different
# approach: start from the case and see if the results are present. Than we
# also have the tags_dict available when log-checking a certain case
class ErrorLogs:
    """
    Analyse all HAWC2 log files in any given directory
    ==================================================

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

    def __init__(self, silent=False):

        self.silent = silent
        # specify folder which contains the log files
        self.PathToLogs = ''
        self.ResultFile = 'ErrorLog.csv'

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
        self.MsgList.append(['*** ERROR *** Error finding body based', \
                        len(self.MsgList)+1])
        # in case an undefinied error slips under the radar
        self.MsgList.append(['*** ERROR ***', len(self.MsgList)+1])

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
                if not self.silent:
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
            iterations = np.ndarray( (len(lines)) )
            iterations[:] = np.nan
            dt = False
            for line in lines:
                j += 1

                # keep track of the number of iterations
                if line.startswith(' Global time'):
                    iterations[j-1] = int(line[-4:])
                    # time step is the first time stamp
                    if not dt:
                        dt = float(line[15:40])

                for k in range(len(MsgList2)):
                    # TODO: change approach, make each message a fixed number
                    # of characters and do
                    #if line[:xx] in MsgList2:
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

            # as last element, add the total number of iterations
            iterations = iterations[np.isfinite(iterations)]
            itertotal = float(iterations.sum())
            if exit_correct:
                tempLog.append('%i' % itertotal)
            else:
                # if we didn't found an elapsed time message, enter blank one
                tempLog.append('')
                tempLog.append('%i' % itertotal)

            # the delta t used for the simulation
            if dt:
                tempLog.append('%1.7f' % dt)
            else:
                tempLog.append('failed to find dt')

            # number of time steps
            tempLog.append('%i' % len(iterations) )

            # if the simulation didn't end correctly, the elapsed_time doesn't
            # exist. Add the average and maximum nr of iterations per step
            try:
                ratio = float(elapsed_time)/float(itertotal)
                tempLog.append('%1.6f' % ratio)
            except UnboundLocalError:
                tempLog.append('')
            tempLog.append('%1.2f' % iterations.mean() )
            tempLog.append('%1.2f' % iterations.max() )

            # also add total simulation time (we need tag_dict for that)

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
        contents = 'file name;' + 'lnr;msg;'*(len(self.MsgList))
        # and add headers for elapsed time, nr of iterations, and sec/iteration
        contents += 'Elapsted time;total iterations;dt;nr time steps;'
        contents += 'seconds/iteration;average iterations/time step;'
        contents += 'maximum iterations/time step;\n'
        for k in self.MsgListLog:
            for n in k:
                contents = contents + str(n) + ';'
            # at the end of each line, new line symbol
            contents = contents + '\n'

        # write csv file to disk, append to facilitate more logfile analysis
        if not self.silent:
            print 'Error log analysis saved at:'
            print self.PathToLogs+str(self.ResultFile)
        FILE = open(self.PathToLogs+str(self.ResultFile), 'a')
        FILE.write(contents)
        FILE.close()


class ModelData:
    """
    Second generation ModelData function. The HawcPy version is crappy, buggy
    and not mutch of use in the optimisation context.
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

    def __init__(self, verbose=False, silent=False):
        self.verbose = verbose
        self.silent = silent
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

        # make the column header
        self.column_header_line = 19 * self.col_width * '=' + '\n'
        for k in self.st_column_header_list:
            self.column_header_line += k.rjust(self.col_width)
        self.column_header_line += '\n' + (19 * self.col_width * '=') + '\n'

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

    def load_st(self, file_path, file_name):
        """
        Now a better format: st_dict has following key/value pairs
            'nset'    : total number of sets in the file (int).
                        This should be autocalculated every time when writing
                        a new file.
            '007-000-0' : set number line in one peace
            '007-001-a' : comments for set-subset nr 07-01 (str)
            '007-001-b' : subset nr and number of data points, should be
                        autocalculate every time you generate a file
            '007-001-d' : data for set-subset nr 07-01 (ndarray(n,19))

        NOW WE ONLY CONSIDER SUBSET COMMENTS, SET COMMENTS, HOW ARE THEY
        TREADED NOW??

        st_dict is for easy remaking the same file. We need a different format
        for easy reading the comments as well. For that we have the st_comments
        """

        # TODO: store this in an HDF5 format! This is perfect for that.

        # read all the lines of the file into memory
        self.st_path, self.st_file = file_path, file_name
        FILE = open(file_path + file_name)
        lines = FILE.readlines()
        FILE.close()

        subset = False
        st_dict = dict()
        st_comments = dict()
        for i, line in enumerate(lines):

            # convert line to list space seperated list
            line_list = self.fromline(line)

            # see if the first character is marking something
            if i == 0:
                # first item is the number of sets enclosed in the file
                #nset = line_list[0]
                set_nr = 0
                subset_nr = 0
                st_dict['000-000-0'] = line

            # marks the start of a set
            elif line[0] == '#':
                #sett = True
                # first character is the #, the rest is the number
                set_nr = int(line_list[0][1:])
                st_dict['%03i-000-0' % set_nr] = line
                # and reset subset nr to zero now
                subset_nr = 0
                subset_nr_track = 0
                # and comments only format, back to one string
                st_comments['%03i-000-0' % set_nr] = ' '.join(line_list[1:])

            # marks the start of a subset
            elif line[0] == '$':
                subset_nr_track += 1
                subset = True
                subset_nr = int(line_list[0][1:])
                # and comments only format, back to one string
                setid = '%03i-%03i-b' % (set_nr, subset_nr)
                st_comments[setid] = ' '.join(line_list[1:])

                # check if the number read corresponds to tracking
                if subset_nr is not subset_nr_track:
                    msg = 'subset_nr and subset_nr_track do not match'
                    raise UserWarning, msg

                nr_points = int(line_list[1])
                st_dict[setid] = line
                # prepare read data points
                sub_set_arr = scipy.zeros((nr_points,19), dtype=np.float64)
                # keep track of where we are on the data array, initialize
                # to 0 for starters
                point = 0

            # in case we are not in subset mode, we only have comments left
            elif not subset:
                # FIXME: how are we dealing with set comments now?
                # subset comments are coming before the actual subset
                # so we account them to one set later than we are now
                #if subset_nr > 0 :
                key = '%03i-%03i-a' % (set_nr, subset_nr+1)
                # in case it is not the first comment line
                if st_dict.has_key(key): st_dict[key] += line
                else: st_dict[key]  = line
                ## otherwise we have the set comments
                #else:
                    #key = '%03i-%03i-a' % (set_nr, subset_nr)
                    ## in case it is not the first comment line
                    #if st_dict.has_key(key): st_dict[key] += line
                    #else: st_dict[key]  = line

            # in case we have the data points, make sure there are enough
            # data poinst present, raise an error if it doesn't
            elif len(line_list)==19 and subset:
                # we can store it in the array
                sub_set_arr[point,:] = line_list
                # on the last entry:
                if point == nr_points-1:
                    # save to the dict:
                    st_dict['%03i-%03i-d' % (set_nr, subset_nr)]= sub_set_arr
                    # and indicate we're done subsetting, next we can have
                    # either set or subset comments
                    subset = False
                point += 1

            #else:
                #msg='error in st format: don't know where to put current line'
                #raise UserWarning, msg

        self.st_dict = st_dict
        self.st_comments = st_comments

    def write_st(self, file_path, file_name):
        """
        """
        # TODO: implement all the tests when writing on nset, number of data
        # points, subsetnumber sequence etc

        content = ''

        # sort the key list
        keysort = self.st_dict.keys()
        keysort.sort()

        for key in keysort:

            # in case we are just printing what was recorded before
            if not key.endswith('d'):
                content += self.st_dict[key]
            # else we have an array
            else:
                # cycle through data points and print them orderly: control
                # precision depending on the number, keep spacing constant
                # so it is easy to read the textfile
                for m in range(self.st_dict[key].shape[0]):
                    for n in range(self.st_dict[key].shape[1]):
                        # TODO: check what do we lose here?
                        # we are coming from a np.float64, as set in the array
                        # but than it will not work with the format()
                        number = float(self.st_dict[key][m,n])
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
                        content += numfor.rjust(self.col_width)
                    content += '\n'

        # and write file to disk again
        FILE = open(file_path + file_name, 'w')
        FILE.write(content)
        FILE.close()
        if not self.silent:
            print 'st file written:', file_path + file_name

# TODO: create a class for the htc_dict. Like a True Simulation object.
# could that be valuable Python module by itself? Talk with Roel about it

class Cases:
    """
    Class for the old htc_dict
    ==========================

    Formerly known as htc_dict: a dictionary with on the key a case identifier
    (case name) and the value is a dictionary holding all the different tags
    and value pairs which define the case
    """

    # TODO: add a method that can reload a certain case_dict, you change
    # some parameters for each case (or some) and than launch again

    #def __init__(self, post_dir, sim_id, resdir=False):
    def __init__(self, *args, **kwargs):
        """
        Either load the cases dictionary if post_dir and sim_id is given,
        otherwise the input is a cases dictionary

        Paramters
        ---------

        cases : dict
            The cases dictionary in case there is only one argument

        post_dir : str
            When using two arguments

        sim_id : str or list
            When using two arguments

        resdir : str, default=False

        loadstats : boolean, default=False

        rem_failed : boolean, default=True

        """

        resdir = kwargs.get('resdir', False)
        self.loadstats = kwargs.get('loadstats', False)
        self.rem_failed = kwargs.get('rem_failed', True)

        # determine the input argument scenario
        if len(args) == 1:
            if type(args[0]).__name__ == 'dict':
                self.cases = args[0]
                sim_id = False
            else:
                raise ValueError, 'One argument input should be a cases dict'
        elif len(args) == 2:
            self.post_dir = args[0]
            sim_id = args[1]
        else:
            raise ValueError, 'Only one or two arguments are allowed.'

        # if sim_id is a list, than merge all sim_id's of that list
        if type(sim_id).__name__ == 'list':
            # stats, dynprop and fail are empty dictionaries if they do not
            # exist
            self.cases, self.cases_stats, self.cases_dynprop, self.cases_fail \
                 = self.merge_sim_ids(sim_id)
            # and define a new sim_id based on all items from the list
            self.sim_id = '_'.join(sim_id)
        # in case we still need to load the cases dict
        elif type(sim_id).__name__ == 'str':
            self.sim_id = sim_id
            self.cases = self.get_cases_dict(self.post_dir, sim_id,
                                             rem_failed=self.rem_failed)
            # load the statistics if applicable
            if self.loadstats:
                self.stats_dict = self.load_stats()

        # change the results directory if applicable
        if resdir:
            self.change_results_dir(resdir)

        #return self.cases

    def select(self, search_keyval=False, search_key=False):
        """
        Select only a sub set of the cases

        Select either search_keyval or search_key. Using both is not supported
        yet. Run select twice to achieve the same effect. If both are False,
        cases will be emptied!

        Parameters
        ----------

        search_keyval : dictionary, default=False
            Keys are the column names. If the values match the ones in the
            database, the respective row gets selected. Each tag is hence
            a unique row identifier

        search_key : dict, default=False
            The key is the string that should either be inclusive (value TRUE)
            or exclusive (value FALSE) in the case key
        """

        db = misc.DictDB(self.cases)
        if search_keyval:
            db.search(search_keyval)
        elif search_key:
            db.search_key(search_keyval)
        else:
            db.dict_sel = {}
        # and remove all keys that are not in the list
        remove = set(self.cases) - set(db.dict_sel)
        for k in remove:
            self.cases.pop(k)


    def launch(self, runmethod='local', verbose=False, copyback_turb=True,
           silent=False, check_log=True):
        """
        Launch all cases
        """

        launch(self.cases, runmethod=runmethod, verbose=verbose, silent=silent,
               check_log=check_log, copyback_turb=copyback_turb)

    # TODO: HAWC2 result file reading should be moved to Simulations
    # and we should also switch to faster HAWC2 reading!
    def load_result_file(self, case, _slice=False):
        """
        Set the correct HAWC2 channels

        Parameters
        ----------

        case : dict
            a case dictionary holding all the tags set for this specific
            HAWC2 simulation

        """

        respath = case['[run_dir]'] + case['[res_dir]']
        resfile = case['[case_id]']
        self.res = HawcPy.LoadResults(respath, resfile)
        if not _slice:
            _slice = np.r_[0:len(self.res.sig)]
        self.time = self.res.sig[_slice,0]
        self.sig = self.res.sig[_slice,:]
        self.case = case

        return self.res

    def change_results_dir(self, resdir):
        """
        if the post processing concerns simulations done by thyra/gorm, and
        is downloaded locally, change path to results accordingly

        NOTE: THIS IS ALSO DONE IN get_htc_dict()
        """
        for case in self.cases:
            if self.cases[case]['[run_dir]'][:4] == '/mnt':
                sim_id = self.cases[case]['[sim_id]']
                newpath = resdir + sim_id + '/'
                self.cases[case]['[run_dir]'] = newpath

        #return cases

    def get_cases_dict(self, post_dir, sim_id, **kwargs):
        """
        Load the pickled dictionary containing all the cases and their
        respective tags.
        """
        self.rem_failed = kwargs.get('rem_failed', self.rem_failed)

        cases = load_pickled_file(post_dir + sim_id + '.pkl')
        self.cases_fail = {}

        if self.rem_failed:
            try:
                cases_fail = load_pickled_file(post_dir + sim_id +'_fail.pkl')
            except IOError:
                return cases
        else:
            return cases

        # ditch all the failed cases out of the htc_dict
        # otherwise we will have fails when reading the results data files
        for k in cases_fail:
            try:
                self.cases_fail[k] = copy.copy(cases[k])
                del cases[k]
                print 'removed from htc_dict due to error: ' + k
            except:
                print 'WARNING: failed case does not occur in cases'
                print '   ', k

        return cases

    def merge_sim_ids(self, sim_id_list, silent=False):
        """
        Load and merge for a list of sim_id's cases, fail, dynprop and stats
        ====================================================================

        For all sim_id's in the sim_id_list the cases, stats, fail and dynprop
        dictionaries are loaded. If one of them doesn't exists, an empty
        dictionary is returned.

        Currently, there is no warning given when a certain case will be
        overwritten upon merging.

        """

        cases_merged, cases_stats_merged, cases_dynprop_merged,\
            cases_fail_merged = dict(), dict(), dict(), dict()

        for sim_id in sim_id_list:

            # TODO: give a warning if we have double entries or not?

            # load the cases of the current serie:
            sim_id_file = sim_id + '.pkl'
            cases = load_pickled_file(self.post_dir + sim_id_file)
            # and copy to htc_dict_merged. Note that non unique keys will be
            # overwritten: each case has to have a unique name!
            cases_merged.update(cases)

            # load the statistics file
            if self.loadstats:
                try:
                    sim_id_file = sim_id + '_statistics.pkl'
                    cases_stats = load_pickled_file(self.post_dir+sim_id_file)
                    cases_stats_merged.update(cases_stats)
                except IOError:
                    if not silent:
                        print 'NO STATS FOUND FOR', sim_id

                # are there dynprop post processing files available?
                try:
                    sim_id_file = sim_id + '_dynprop.pkl'
                    cases_dynprop=load_pickled_file(self.post_dir+sim_id_file)
                    cases_dynprop_merged.update(cases_dynprop)
                except IOError:
                    if not silent:
                        print 'NO DYNPROPS FOUND FOR', sim_id

            # and the failed ones, if available
            try:
                sim_id_file = sim_id + '_fail.pkl'
                cases_fail = load_pickled_file(self.post_dir + sim_id_file)
                cases_fail_merged.update(cases_fail)
                # ditch all the failed cases out of the cases dict. Otherwise
                # we will have fails when reading the results data files
                for k in cases_fail_merged:
                    del cases_merged[k]
                    if not silent:
                        print 'removed from htc_dict due to error: ' + k
            except IOError:
                if not silent:
                    print 'NO FAILED htc_dict FOUND FOR', sim_id

        return cases_merged, cases_stats_merged, cases_dynprop_merged,\
                cases_fail_merged

    def printall(self, scenario, figpath=''):
        """
        For all the cases, get the average value of a certain channel
        """
        self.figpath = figpath

        # plot for each case the dashboard
        for k in self.cases:

            if scenario == 'blade_deflection':
                self.blade_deflection(self.cases[k], self.figpath)

    def diff(self, refcase_dict, cases):
        """
        See wich tags change over the given cases of the simulation object
        """

        # there is only one case allowed in refcase dict
        if not len(refcase_dict) == 1:
            return ValueError, 'Only one case allowed in refcase dict'

        # take an arbritrary case as baseline for comparison
        refcase = refcase_dict[refcase_dict.keys()[0]]
        #reftags = sim_dict[refcase]

        diffdict = dict()
        adddict = dict()
        remdict = dict()
        print
        print '*'*80
        print 'comparing %i cases' % len(cases)
        print '*'*80
        print
        # compare each case with the refcase and see if there are any diffs
        for case in sorted(cases.keys()):
            dd = misc.DictDiff(refcase, cases[case])
            diffdict[case] = dd.changed()
            adddict[case] = dd.added()
            remdict[case] = dd.removed()
            print ''
            print '='*80
            print case
            print '='*80
            for tag in sorted(diffdict[case]):
                print tag.rjust(20),':', cases[case][tag]

        return diffdict, adddict, remdict

    def blade_deflection(self, case, **kwargs):
        """
        """

        # read the HAWC2 result file
        self.load_result_file(case)

        # select all the y deflection channels
        db = misc.DictDB(self.res.ch_dict)

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

    def load_stats(self, **kwargs):
        """
        Load an existing statistcs file
        """
        post_dir = kwargs.get('post_dir', self.post_dir)
        sim_id = kwargs.get('sim_id', self.sim_id)

        FILE = open(post_dir + sim_id + '_statistics.pkl', 'rb')
        stats_dict = pickle.load(FILE)
        FILE.close()

        return stats_dict


    def calc_stats(self, new_sim_id=False, silent=False, calc_torque=False):
        """
        Calculate and save all the statistics present in cases

        stats_dict is a dictionary with following key/value pairs
        'sig_stats' : ndarray(2,6,nr_channels)
            holding HawcPy.SignalStatisticsNew(res.sig)
            sig_stat = [(0=value,1=index),statistic parameter, channel]
            stat params = 0 max, 1 min, 2 mean, 3 std, 4 range, 5 abs max
            note that min, mean, std, and range are not relevant for index
            values. Set to zero there.
        'ch_dict' : dict
            holding the res.ch_dict, relating channel description to chi

        If new_sim_id is false, we will assume all sim_id's in cases are the
        same. Otherwise we save the results with the provided sim_id

        Parameters
        ----------

        calc_torque : default=False
            Set to either 'hawc2' or 'ojf' if the K file needs to be named
            either after the hawc2 case_id or the ojf ojf_case name. Only
            used when the generator was not set.
        """

        # save them in a big dictionary, key is the case name
        stats_dict = {}

        # get some basic parameters required to calculate statistics
        case = self.cases.keys()[0]
        post_dir = self.cases[case]['[post_dir]']
        if not new_sim_id:
            # select the sim_id from a random case
            sim_id = self.cases[case]['[sim_id]']
        else:
            sim_id = new_sim_id

        if not silent:
            nrcases = len(self.cases)
            print '='*79
            print 'statistics for %s, nr cases: %i' % (sim_id, nrcases)

        for ii, case in enumerate(self.cases):
            if not silent:
                print 'stats progress: %4i/%i' % (ii, nrcases)

            self.load_result_file(self.cases[case])
            sig_stats = HawcPy.SignalStatisticsNew(self.sig)
            stats_dict[case] = {'sig_stats' : sig_stats.copy()}
            # als save the channel information for each case
            stats_dict[case]['ch_dict'] = self.res.ch_dict

            # if applicable, calculate the torque constant for gen_K
            # only for fixed rpm cases!
            if calc_torque and not self.cases[case]['[generator]']:
                self.calc_torque_const(save=True, name=calc_torque)

        # is there a stats file to be updated?
        try:
            FILE = open(post_dir + sim_id + '_statistics.pkl', 'rb')
            stats_orig = pickle.load(FILE)
            FILE.close()
        except IOError:
            stats_orig = {}

        # and save/update the statistics database
        FILE = open(post_dir + sim_id + '_statistics.pkl', 'wb')
        stats_orig.update(stats_dict)
        pickle.dump(stats_orig, FILE, protocol=2)
        FILE.close()

        return stats_dict

    def calc_torque_const(self, save=False, name='ojf'):
        """
        If we have constant RPM over the simulation, calculate the torque
        constant. The current loaded HAWC2 case is considered. Consequently,
        first load a result file with load_result_file

        Parameters
        ----------

        save : boolean, default=False

        name : str, default='ojf'
            File name of the torque constant result. Default to using the
            ojf case name. If set to hawc2, it will the case_id. In both
            cases the file name will be extended with '.kgen'

        Returns
        -------

        [windspeed, rpm, K] : list

        """
        # make sure the results have been loaded previously
        try:
            # get the relevant index to the wanted channels
            # tag: coord-bodyname-pos-sensortype-component
            tag = 'bearing-shaft_nacelle-angle_speed-rpm'
            irpm = self.res.ch_dict[tag]['chi']
            chi_rads = self.res.ch_dict['Omega']['chi']
            tag = 'shaft-shaft-node-001-momentvec-z'
            chi_q = self.res.ch_dict[tag]['chi']
        except AttributeError:
            msg = 'load results first with Cases.load_result_file()'
            raise ValueError, msg

#        if not self.case['[fix_rpm]']:
#            print
#            return

        windspeed = self.case['[windspeed]']
        rpm = self.res.sig[:,irpm].mean()
        # and get the average rotor torque applied to maintain
        # constant rotor speed
        K = -np.mean(self.res.sig[:,chi_q]*1000./self.res.sig[:,chi_rads])

        result = np.array([windspeed, rpm, K])

        # optionally, save the values and give the case name as file name
        if save:
            fpath = self.case['[post_dir]'] + 'torque_constant/'
            if name == 'hawc2':
                fname = self.case['[case_id]'] + '.kgen'
            elif name == 'ojf':
                fname = self.case['[ojf_case]'] + '.kgen'
            else:
                raise ValueError, 'name should be either ojf or hawc2'
            # create the torque_constant dir if it doesn't exists
            try:
                os.mkdir(fpath)
            except OSError:
                pass

#            print 'gen K saving at:', fpath+fname
            np.savetxt(fpath+fname, result, header='windspeed, rpm, K')

        return result

# TODO: implement this
class Results():
    """
    There should be a bare metal module/class for those who only want basic
    python support for HAWC2 result files and/or launching simulations.

    How to properly design this module? Change each class into a module? Or
    leave like this?
    """

    # OK, for now use this to do operations on HAWC2 results files

    def __init___(self):
        """
        """
        pass

    def m_equiv(self, st_arr, load, pos):
        r"""
        Centrifugal corrected equivalent moment
        =======================================

        Convert beam loading into a single equivalent bending moment. Note that
        this is dependent on the location in the cross section. Due to the
        way we measure the strain on the blade and how we did the calibration
        of those sensors.

        .. math::

            \epsilon = \frac{M_{x_{equiv}}y}{EI_{xx}} = \frac{M_x y}{EI_{xx}}
            + \frac{M_y x}{EI_{yy}} + \frac{F_z}{EA}

            M_{x_{equiv}} = M_x + \frac{I_{xx}}{I_{yy}} M_y \frac{x}{y}
            + \frac{I_{xx}}{Ay} F_z

        Parameters
        ----------

        st_arr : np.ndarray(19)
            Only one line of the st_arr is allowed and it should correspond
            to the correct radial position of the strain gauge.

        load : list(6)
            list containing the load time series of following components
            .. math:: load = F_x, F_y, F_z, M_x, M_y, M_z
            and where each component is an ndarray(m)

        pos : np.ndarray(2)
            x,y position wrt neutral axis in the cross section for which the
            equivalent load should be calculated

        Returns
        -------

        m_eq : ndarray(m)
            Equivalent load, see main title

        """

        F_z = load[2]
        M_x = load[3]
        M_y = load[4]

        x, y = pos[0], pos[1]

        A = st_arr[ModelData.st_headers.A]
        I_xx = st_arr[ModelData.st_headers.Ixx]
        I_yy = st_arr[ModelData.st_headers.Iyy]

        M_x_equiv = M_x + ( (I_xx/I_yy)*M_y*(x/y) ) + ( F_z*I_xx/(A*y) )
        # or ignore edgewise moment
        #M_x_equiv = M_x + ( F_z*I_xx/(A*y) )

        return M_x_equiv


def eigenbody(cases, debug=False):
    """
    Read HAWC2 body eigenalysis result file
    =======================================

    This is basically a cases convience wrapper around HawcPy.ReadEigenBody

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case. For each
        case, it is updated with the results, results2 of the eigenvalue
        analysis performed for each body using the following respective
        tags: [eigen_body_results] and [eigen_body_results2].

    """

    #Body data for body number : 3 with the name :nacelle
    #Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #Mode nr:  1:   1.45388E-21    1.74896E-03    6.28319E+02

    for case in cases:
        # tags for the current case
        tags = cases[case]
        file_path = tags['[run_dir]'] + tags['[eigenfreq_dir]']
        file_name = tags['[case_id]'] + '_eigen_body.dat'
        # and load the eigenfrequency body results
        results, results2 = HawcPy.ReadEigenBody(file_path, file_name,
                                                 nrmodes=10)
        # add them to the htc_dict
        cases[case]['[eigen_body_results]'] = results
        cases[case]['[eigen_body_results2]'] = results2

    return cases

def eigenstructure(cases, debug=False):
    """
    Read HAWC2 structure eigenalysis result file
    ============================================

    This is basically a cases convience wrapper around
    Hawc2io.ReadEigenStructure

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case. For each
        case, it is updated with the modes_arr of the eigenvalue
        analysis performed for the structure.
        The modes array (ndarray(3,n)) holds fd, fn and damping.
    """

    for case in cases:
        # tags for the current case
        tags = cases[case]
        file_path = tags['[run_dir]'] + tags['[eigenfreq_dir]']
        file_name = tags['[case_id]'] + '_eigen_strc.dat'
        # and load the eigenfrequency structure results
        modes = HawcPy.ReadEigenStructure(file_path, file_name, max_modes=10)
        # add them to the htc_dict
        cases[case]['[eigen_structure]'] = modes

    return cases


if __name__ == '__main__':
    pass

