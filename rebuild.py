# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import ojfdb
#import ojfresult
#import ojfpostproc
import towercal
import yawcal
import bladecal
import ojf_freeyaw


def rebuild_symlink_database():
    # first, rebuild the symlink list: all files in one folder
    # this includes the sweep cases as well
    path_db = 'database/'
    data_source_root = 'data/raw/'
    ojfdb.make_symlinks_all(path_db, data_source_root)

    # convert the database index to csv/DataFrame_hdf5/xls. Based on the file
    # name other usefull selection columns are created
    ojfdb.convert_pkl_index_df(path_db, db_id='symlinks_all')

    # create a stastistics database
    # create a dashboard plot for all the cases
    ojfdb.build_stats_db(path_db, 'symlinks_all', calibrate=True, dashplot=True,
                         dataframe=True, resample=True, continue_build=True,
                         save_df=True, save_df_csv=True)

    # only rebuild the statistics database
    ojfdb.build_stats_db(path_db, 'symlinks_all', calibrate=True, dashplot=False,
                         dataframe=True, resample=True, continue_build=False,
                         save_df=False, save_df_csv=False)

    # add the freeyaw control stair cases to the stats collection
    ojf_freeyaw.add_yawcontrol_stair_steps()


def rebuild_calibration_data():
    """
    Rebuild the calibration data based on the raw calibration measurements.
    """

    towercal.all_tower_calibrations()
    yawcal.all_yawlaser_calibrations()

    # bladecall fails currently for the stair detection in April
    bladecal.all_blade_calibrations()


if __name__ == '__main__':

    dummy=None

    path_db = 'database/'
    data_source_root = 'data/raw/'
