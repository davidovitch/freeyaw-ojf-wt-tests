# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

#import os
import sys

# cleanup path
blacklist = [ '/home/dave/Repositories/public/MMPE',
              '/home/dave/Repositories/DTU/prepost',
              '/home/dave/Repositories/DTU/pythontoolbox/fatigue_tools',
              '/home/dave/Repositories/DTU/pythontoolbox']
rm = []
for path_rm in blacklist:
    for i, path in enumerate(sys.path):
        if path == path_rm:
            print('removed from path: %s' % path)
            sys.path.pop(i)
            break

import ojfdb
#import ojfresult
#import ojfpostproc
import towercal

def rebuild_symlink_database():
    # first, rebuild the symlink list: all files in one folder
    # this includes the sweep cases as well
    path_db = 'database/'
    data_source_root = 'data/raw/'
#    ojfdb.make_symlinks_all(path_db, data_source_root)

    # convert the database index to csv/DataFrame_hdf5/xls
#    ojfdb.convert_pkl_index_df(path_db, db_id='symlinks_all')

    # create all the calibration files
    towercal.all_tower_calibrations()

    # create a stastistics database
    # create a dashboard plot for all the cases
#    ojfdb.build_db(path_db, 'symlinks_all', calibrate=True, dashplot=True,
#                   dataframe=True, resample=True, continue_build=True)


def rebuild_calibration_data():
    """
    Rebuild the calibration data based on the raw calibration measurements.
    """

    towercal.all_tower_calibrations()


if __name__ == '__main__':
    dummy=None

#    rebuild_symlink_database()

#    respath = 'database/symlinks_all/'
#    f = '0410_run_299_7.5ms_dc0_flexies_freeyawforced_yawerrorrange_fastside_lowrpm'
#    f = '0412_run_363ah_9.5ms_dc0.45_stiff_dcsweep'
#    res = ojfresult.ComboResults(respath, f, silent=True, sync=True, cal=True)
#    res._resample()
#    ftarget = os.path.join('database/DataFrames/',f + '.h5')

#    libs = ['zlib', 'bzip2', 'lzo', 'blosc']
#    for lib in libs:
#        rpl = (f, lib)
#        ftarget = fpath=os.path.join('database/DataFrames/','%s-%s.h5' % rpl)
#        print(ftarget)
#        df = res.to_df(None, complevel=9, complib=lib)
#        break

#    ftarget = os.path.join('database/DataFrames/','%s-%s.h5' % (f, 'blosc'))
#    print(ftarget)
#    df = res.to_df(None, complevel=9, complib='blosc')
