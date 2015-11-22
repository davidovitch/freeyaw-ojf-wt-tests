# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:42:33 2011

@author: dave
"""

# built in modules
import pickle
import logging
import math

# 3th party modules
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
#from scipy.interpolate import UnivariateSpline
import pylab as plt # for debug plotting

# custom modules
import Simulations as sim
import HawcPy
import plotting
#import ojfpostproc as ojf
#import misc
import ojfresult
#import ojfvshawc2

sti = HawcPy.ModelData.st_headers

# TODO: create a HAWC2 model class/object. It has the variable_tag_func,
# a case_name_func, the master.tags, and any other stuff for launching and
# working with a HAWC2 model. So in effect, unifica all HAWC2 model methods
# under one supper powerfull class
# also consider how we can easily launch the simulation, check its progress
# by periodically looking into the logfile, and later easily plot some
# results, all from the hawc2model class. Integration into Simulation module?
# or how could this be done into openMDAO?

def variable_tag_func(master, case_id_short=False):

    """
    Function which updates HtcMaster.tags and returns an HtcMaster object

    Only use lower case characters for case_id since a hawc2 result and
    logfile are always in lower case characters.

    BE CAREFULL: if you change a master tag that is used to dynamically
    calculate an other tag, that change will be propageted over all cases,
    for example:
    master.tags['tag1'] *= master.tags[tag2]*master.tags[tag3']
    it will accumlate over each new case. After 20 cases
    master.tags['tag1'] = (master.tags[tag2]*master.tags[tag3'])^20
    which is not wanted, you should do
    master.tags['tag1'] = tag1_base*master.tags[tag2]*master.tags[tag3']
    """

    # in order to verify that non of the variable tags is dependent on its
    # own value (see the BE CAREFULL note) none of the variable tags are
    # allowed to present as a tag in master
    vt = {}
    # this approach will fail when we are changing an existing Cases object:
    # all the variable tags have been set before, and are reset again. The
    # question than comes back to the first BE CAREFULL note. If that complies
    # there will be no problem. But how to verify that? Are we bumping into
    # the limits of doing all the tag/simulation parameters in dictionaries?


    mt = master.tags

    # TODO: write a lot of logical tests for the tags!!

    # -------------------------------------------------------------------------
    V = mt['[windspeed]']
    t = mt['[duration]']
    if not V < 1e-15 and not V > -1e-15:
        mt['[TI]'] = mt['[TI_ref]'] * ((0.75*V)+5.6) / V
    else:
        mt['[TI]'] = 0

    mt['[turb_dx]'] = V*t/mt['[turb_grid_x]']

    mt['[turb_dy]'] = mt['[rotor_diameter]'] / mt['[turb_grid_yz]']

    mt['[turb_dz]'] = mt['[rotor_diameter]'] / mt['[turb_grid_yz]']

    #mt['[turb_base_name]'] = 'turb_s' + str(mt['[turb_seed]']) + '_' + str(V)
    mt['[turb_base_name]'] = 'turb_s%i_%1.2f' % (mt['[turb_seed]'], V)

    # total simulation time
    mt['[time_stop]'] = mt['[t0]'] + t

    # rotation type: standard or OJF
    if mt['[rotation_type]'] == 'ojf':
        mt['[zhub_sign]'] = '-'
        mt['[std_rotation]'] = ';'
        mt['[zaxis_fact]'] = -1.
        # switch rotation direction, given in -Z_shaft direction
        #mt['[fix_wr]'] *=  -1.
        #rot_type = 'rot_ojf'

    elif mt['[rotation_type]'] == 'std':
        mt['[zhub_sign]'] = ''
        mt['[std_rotation]'] = ''
        mt['[zaxis_fact]'] = 1.
        #rot_type = 'rot_std'
    #else:
        #rot_type = ''

    # include support structure for hub height
    if mt['[tower_support]']:
        mt['[hub_height]'] = mt['[tower_length]'] + 0.55

    # -----------------------------------------------------------------------
    # BLADE RELATED STUFF
    # -----------------------------------------------------------------------
    mt['[tip_element_blade]'] = mt['[nr_nodes_blade]']-1

    # set the sweep tags [x1], [tw1], based on [x1-nosweep]
    if mt.has_key('[x1]'):
        master._sweep_tags()
    # the blade c2_def is only represented by one tag
    elif mt.has_key('[blade_htc_node_input]') \
        and mt['[blade_hawtopt_dir]'] is not None:

        # load the  hawtopt file
        tmp = mt['[blade_hawtopt_dir]'] + mt['[blade_hawtopt_file]']

        blade = np.loadtxt(tmp)
        # remove the hub contribution
        # in the htc file, blade root =0 and not blade hub radius
        blade[:,0] = blade[:,0] - blade[0,0]

        nr_nodes = mt['[nr_nodes_blade]']
        # make node positions match the strain gauges
        # strain gauge locations: 2cm and 18cm from root=end profile
        # required intermediate positions for the strains
        points = np.array([0.02, 0.18])
        radius_new = HawcPy.linspace_around(0, blade[-1,0],
                        points=points, num=nr_nodes, verbose=False)

        # the traditional version: hub is all in the hub and we will add
        # aerodrag to those elements
        if mt['[hub_lenght]'] == 0.245:
            # create the HTC input
            master._all_in_one_blade_tag(radius_new=radius_new)

            # set the position of the aero root and tip sensors correct
            mt['[aero_tip_pos]'] = 0.555
            mt['[aero_tip_region]']  = 0.51
            mt['[aero_50_region]']   = '%1.3f' % (mt['[aero_tip_pos]']*0.5)
            mt['[aero_30_region]']   = '%1.3f' % (mt['[aero_tip_pos]']*0.3)
            mt['[aero_20_region]']   = '%1.3f' % (mt['[aero_tip_pos]']*0.2)
            mt['[aero_root_region]'] = 0.04

        # when the hub length only consists for the hub disc. The cylinder
        # root parts are now integrated in the blade model.
        elif mt['[hub_lenght]'] == 0.083:
            # add the cylinder points manually to the blade model. Do not use
            # mt['[nr_nodes_blade]'], because those changes will propagete
            # to the next case created. In doing so, nr_nodes will be increased
            # by two for every case created
            nr_nodes += 2
            add = sp.zeros( (2,blade.shape[1]) )
            add[1,0] = 0.1619
            # shift the normal blade points outwards
            radius_new += 0.162
            blade[:,0] += 0.162
            radius_new = np.append(add[:,0], radius_new, axis=0)
            # also add those points to blade_hawtopt
            blade = np.append(add, blade, axis=0)
            mt['[blade_hawtopt]'] = blade
            # create the HTC input
            mt = _all_in_one_blade_tag(mt, radius_new=radius_new,
                                       nr_nodes=nr_nodes)

            # set the position of the aero root and tip sensors correct
            mt['[aero_tip_pos]'] = 0.555 + 0.162
            mt['[aero_tip_region]'] = 0.51 + 0.162
            mt['[aero_root_region]'] = 0.04 + 0.162

        else:
            raise ValueError, 'unsupported hub lenght'

    pitch0 = mt['[pitch_angle]']
    mt['[pitch_angle_b1]'] = mt['[pitch_angle_imbalance_b1]'] + pitch0
    mt['[pitch_angle_b2]'] = mt['[pitch_angle_imbalance_b2]'] + pitch0
    mt['[pitch_angle_b3]'] = mt['[pitch_angle_imbalance_b3]'] + pitch0

    # ==============================================================
    # yawmode
    if not mt.has_key('[yawmode]'):
        yawMode = ''
        msg = 'tag [yawmode] should be in [free, fix, control_ini], ignored'
        logging.warn(msg)
        #raise ValueError,'tag [yawmode] should be in [free, fix, control_ini]'
    elif mt['[yawmode]'] == 'free':
        yawMode = 'yfree'
        mt['[yawfix]'] = False
        mt['[yawfree]'] = True
        mt['[yaw_c]'] = False
        # only non zero when in fixed yawing
        mt['[yaw_angle_misalign]'] = 0

    elif mt['[yawmode]'] == 'fix':
        yawMode = 'yfix'
        mt['[yawfix]'] = True
        mt['[yawfree]'] = False
        mt['[yaw_c]'] = False

    elif mt['[yawmode]'] == 'control_ini':
        yawMode = 'ycon%+05.1f' % (mt['[yaw_c_ref_angle]'])
        mt['[yawfix]'] = False
        mt['[yawfree]'] = False
        mt['[yaw_c]'] = True

    #if mt['[tower_simple]']:
        #towermode = 'twsimple'
    #elif mt['[tower_support]']:
        #towermode = 'twsupport'

    # ==============================================================
    if not mt['[generator]']:
        mt['[nogenerator]'] = True
        # check the units on the rpm setting
        # for when it is given in RPM
        if mt.has_key('[fix_rads]') \
            and not mt.has_key('[fix_rpm]'):
            rpm = mt['[fix_rads]']*30./np.pi
            # in both cases, convert to the generic tag in rad/s
            mt['[fix_wr]'] = mt['[fix_rads]']
            # and also set the initial rotation speed!
            mt['[init_wr]'] = mt['[fix_rads]']
        # when given in rad/s
        elif not mt.has_key('[fix_rads]') \
            and mt.has_key('[fix_rpm]'):
            rpm = mt['[fix_rpm]']
            mt['[fix_wr]'] = rpm*np.pi/30.
            # and also set the initial rotation speed!
            mt['[init_wr]'] = mt['[fix_wr]']
        else:
            msg = 'set either [fix_rpm] or [fix_rads], but not both!'
            raise ValueError, msg
        rpm_mode = format(rpm, '03.0f') + 'rpm'
    # For the generator cases we have
    else:
        if mt['[ojf_generator_dll]'].find('borland') > -1:
            dll_tag = '_borland'
        elif mt['[ojf_generator_dll]'].find('lazarus') > -1:
            dll_tag = '_lazarus'
        else:
            dll_tag = ''
        # using all K's
#        replace = (mt['[gen_K0]'], mt['[gen_K1]'], mt['[gen_K2]'], dll_tag)
#        rpm_mode = 'ojfgen%1.3f_%1.3f_%1.3f%s' % replace
        # or just the on that matters: K2
        rpm_mode = 'ojfgen%1.03f%s' % (mt['[gen_K2]'], dll_tag)
        rpm_mode += '_rpmini%03.0f' % (mt['[init_wr]']*30./np.pi)
        # set rpm from the init_wr
        rpm = mt['[init_wr]']*30./np.pi
        mt['[nogenerator]'] = False
        #mt['[init_wr]'] = 0

    # ==============================================================
    # Tune gains of yaw control as function of rotor speed
    if rpm and rpm > 100:
        gmp = 1 + ((rpm - 100)/150.)
        tmp = '[yaw_c_gain_'
        mt['[yaw_c_gain_multiplyer]'] = gmp
        mt['[yaw_c_gain_pro]'] = mt[tmp + 'pro_base]']*gmp
        mt['[yaw_c_gain_int]'] = mt[tmp + 'int_base]']*gmp
        mt['[yaw_c_gain_dif]'] = mt[tmp + 'dif_base]']*gmp
    else:
        mt['[yaw_c_gain_multiplyer]'] = 1
        mt['[yaw_c_gain_pro]'] = mt['[yaw_c_gain_pro_base]']
        mt['[yaw_c_gain_int]'] = mt['[yaw_c_gain_int_base]']
        mt['[yaw_c_gain_dif]'] = mt['[yaw_c_gain_dif_base]']
    pro  = 10000.0*mt['[yaw_c_gain_pro]']
    intg = 10000.0*mt['[yaw_c_gain_int]']
    diff = 10000.0*mt['[yaw_c_gain_dif]']
    yawcontrol = '%02i-%02i-%02i' % (pro, intg, diff)
    # ==============================================================
    # keep ratio between rpm and simulation sampling time constant
    # use the mantra: 30 azimuthal positions per revolution
    # but that is not good for our case, especially when under yaw error
    # for free yawing cases, rpm is based on [init_wr]
    if rpm and mt['[auto_set_sim_time]']:

#        dt = 1. / (mt['[azim-res]'] * rpm / 60.)
#        #dt = 1. / (azim_res * rpm / 60.)
#        # and do not go above certain treshold
#        mt['[dt_sim]'] = '%1.4f' % min([0.01, dt])

        # azimuthal positions per revolation is not the limiting factor here
        # for the OJF model it is the structural response
        pol2 = np.array([4.40811315, 900.0])
        #pol4 = np.array([3.46641503, 1176.67924838])
        freqmin = 2400
        turnpoint = ( (pol2[0]*400) + pol2[1]) # = 2663
        # pol2 might have not been conservative enough for the +yaw mis align
        # cases, pol4 is an attempt to increase the freq up to 200 rpm
        pol4 = np.polyfit([50, 400], [freqmin, turnpoint], 1)
        rico3 = 0.5
        pol3 = np.array([rico3, ( pol2[0]*400 + pol2[1] - rico3*400) ])
        if rpm <= 400: # RPM at turningpoint
            freq = np.polyval(pol4, rpm)
        else:
            freq = np.polyval(pol3, rpm)
        dt = 1.0/freq
        # but do not drop below freqmin Hz !
        mt['[dt_sim]'] = '%1.6f' % min([1.0/freqmin, dt])
        # and consequently, the azimuthal resolution is
        mt['[azim-res]'] = freq*60.0/rpm

    elif '[dt_sim]' not in mt:
        raise ValueError, 'dt_sim is not set'

    # ==============================================================
    # blade structural settings
    if not mt.has_key('[blade_st_group]'):
        blade_struct = str(mt['[st_blade_set]'])
        blade_struct += str(mt['[st_blade1_subset]'])
        blade_struct += str(mt['[st_blade2_subset]'])
        blade_struct += str(mt['[st_blade3_subset]'])
    elif mt['[blade_st_group]'] == 'flex':
        mt['[st_blade_set]'] = 7
        mt['[st_blade1_subset]'] = stsets.b1_flex_opt2
        mt['[st_blade2_subset]'] = stsets.b2_flex_opt2
        mt['[st_blade3_subset]'] = stsets.b3_flex_opt2
        mt['[damp_blade1]'] = model.b1_flex_damp
        mt['[damp_blade2]'] = model.b2_flex_damp
        mt['[damp_blade3]'] = model.b3_flex_damp
        blade_struct = 'flex'
    elif mt['[blade_st_group]'] == 'stiff':
        mt['[st_blade_set]'] = 7
        mt['[st_blade1_subset]'] = stsets.b1_stiff_opt2
        mt['[st_blade2_subset]'] = stsets.b2_stiff_opt2
        mt['[st_blade3_subset]'] = stsets.b3_stiff_opt2
        mt['[damp_blade1]'] = model.b1_stiff_damp
        mt['[damp_blade2]'] = model.b2_stiff_damp
        mt['[damp_blade3]'] = model.b3_stiff_damp
        blade_struct = 'stiff'
    elif mt['[blade_st_group]'] == 'verystiff':
        mt['[st_blade_set]'] = 7
        mt['[st_blade1_subset]'] = 1
        mt['[st_blade2_subset]'] = 1
        mt['[st_blade3_subset]'] = 1
        blade_struct = 'verystiff'
    elif mt['[blade_st_group]'] == 'flex-cyl':
        mt['[st_blade_set]'] = 7
        mt['[st_blade1_subset]'] = stsets.b1_flex_opt2_cyl
        mt['[st_blade2_subset]'] = stsets.b2_flex_opt2_cyl
        mt['[st_blade3_subset]'] = stsets.b3_flex_opt2_cyl
        mt['[damp_blade1]'] = model.b1_flex_damp
        mt['[damp_blade2]'] = model.b2_flex_damp
        mt['[damp_blade3]'] = model.b3_flex_damp
        blade_struct = 'flex-cyl'
    elif mt['[blade_st_group]'] == 'stiff-cyl':
        mt['[st_blade_set]'] = 7
        mt['[st_blade1_subset]'] = stsets.b1_stiff_opt2_cyl
        mt['[st_blade2_subset]'] = stsets.b2_stiff_opt2_cyl
        mt['[st_blade3_subset]'] = stsets.b3_stiff_opt2_cyl
        mt['[damp_blade1]'] = model.b1_stiff_damp
        mt['[damp_blade2]'] = model.b2_stiff_damp
        mt['[damp_blade3]'] = model.b3_stiff_damp
        blade_struct = 'stiff-cyl'
    else:
        raise ValueError, 'tag [blade_st_group] options: flex, (very)stiff'

    # add an optional other tag
    if mt.has_key('[extra_id]'):
        extra_id = mt['[extra_id]']
    else:
        extra_id = ''

    # =========================================================================
    #mt['[yawmode_casetag]'] = yawMode
    #mt['[rpm_mode_casetag]'] = rpm_mode
    #mt['[blade_struct_casetag]'] = blade_struct
    ii = '[case_id]'
    # set case_id here instead of HawcPy.master.variable_tags
    mt[ii]  =  mt['[sim_id]']
    mt[ii] += '_%s' % extra_id
    # includate OJF data and run_id
    mt[ii] +=  '_' + '_'.join(mt['[ojf_case]'].split('_')[0:3])

#    mt[ii] += '_%ihz' % (1.0/float(mt['[dt_sim]']))
#    mt[ii] += '_' + str(mt['[epsresq]'])
#    mt[ii] += '_' + str(mt['[epsresd]'])
#    mt[ii] += '_' + str(mt['[epsresg]'])

    mt[ii] += '_%1.2fms' % mt['[windspeed]']
    mt[ii] += '_s'    + str(mt['[turb_seed]'])
    mt[ii] += '_y%+05.1f' % mt['[wyaw]']
    #mt[ii] += '_ini'  + str(mt['[yaw_angle_init]'])
    mt[ii] += '_'     + yawMode
#    mt[ii] += '_'     + yawcontrol
    mt[ii] += '_ymis_%+05.1f' % mt['[yaw_angle_misalign]']
    #mt[ii] += '_'     + rot_type
    mt[ii] += '_'     + rpm_mode
    #mt[ii] += '_'     + mt['[blade_hawtopt_file]']
    mt[ii] += '_sb'   + str(blade_struct)
    #mt[ii] += '_ab'   + str(mt['[sweep_amp]'])
    #mt[ii] +=           str(mt['[sweep_exp]'])
    #mt[ii] += '_c'    + str(mt['[coning_angle]'])
    mt[ii] += '_p'    + str(mt['[pitch_angle_b1]'])
#    mt[ii] += '_nazi' + format(mt['[bemwake_nazi]'], '1.0f')
#    mt[ii] += '_nw'   + format(mt['[nw_k3]'], '1.2f')
#    mt[ii] += '_'    + format(mt['[nw_k2]'], '1.2f')
#    mt[ii] += '_'    + format(mt['[nw_k1]'], '1.2f')
    #mt[ii] += '_ts'   + str(mt['[tower_shadow]'])
    mt[ii] += '_cd'   + str(mt['[tower_shadow_cd]'])
    #mt[ii] += '_'     + towermode
        #+ '_st'     + str(mt['[st_tower_subset]'])
        #+ '_tlen'   + format(mt['[tower_length]'], '2.1f')
        #+ '_nrnb'   + str(mt['[nr_nodes_blade]']) \
    # ==============================================================

    return master

def variable_tag_func_bladeonly(master):
    """
    just a little different for the blade only simulations
    """

    mt = master.tags

    # -------------------------------------------------------------------------
    V = mt['[windspeed]']
    t = mt['[duration]']
    if not V < 1e-15 and not V > -1e-15:
        mt['[TI]'] = mt['[TI_ref]'] * ((0.75*V)+5.6) / V
    else:
        mt['[TI]'] = 0

    mt['[turb_dx]'] = V*t/mt['[turb_grid_x]']
    mt['[turb_dy]'] = mt['[rotor_diameter]'] / mt['[turb_grid_yz]']
    mt['[turb_dz]'] = mt['[rotor_diameter]'] / mt['[turb_grid_yz]']
    mt['[turb_base_name]'] = 'turb_s' + str(mt['[turb_seed]']) + '_' + str(V)

    # total simulation time
    mt['[time_stop]'] = mt['[t0]'] + t

    # -----------------------------------------------------------------------
    mt['[tip_element_blade]'] = mt['[nr_nodes_blade]']-1

    # set the sweep tags [x1], [tw1], based on [x1-nosweep]
    if mt.has_key('[x1]'):
        master._sweep_tags()
    # the blade c2_def is only represented by one tag
    elif mt.has_key('[blade_htc_node_input]') \
        and mt['[blade_hawtopt_dir]'] is not None:

        # load the  hawtopt file
        tmp = mt['[blade_hawtopt_dir]'] + mt['[blade_hawtopt_file]']

        blade = np.loadtxt(tmp)
        mt['[blade_hawtopt]'] = blade
        # remove the hub contribution
        # in the htc file, blade root =0 and not blade hub radius
        blade[:,0] = blade[:,0] - blade[0,0]
        # tweak the blade tip angle wrt to blade a little bit
        blade[:,2] -= 5.

        # make node positions match the strain gauges
        # strain gauge locations: 2cm and 18cm from root=end profile
        nr_nodes = mt['[nr_nodes_blade]']
        # only do this for more nodes per blade
        if nr_nodes > 4:
            # required intermediate positions
            points = np.array([0.02, 0.18])
            radius_new = HawcPy.linspace_around(0, blade[-1,0],
                            points=points, num=nr_nodes, verbose=False)
            master._all_in_one_blade_tag(radius_new=radius_new)
        else:
            master._all_in_one_blade_tag()

    # ==============================================================
    # blade structural settings
    blade_struct = str(mt['[st_blade_set]'])
    blade_struct += str(mt['[st_blade_subset]'])

    # =========================================================================
    #mt['[yawmode_casetag]'] = yawMode
    #mt['[rpm_mode_casetag]'] = rpm_mode
    #mt['[blade_struct_casetag]'] = blade_struct
    ii = '[case_id]'
    # set case_id here instead of HawcPy.master.variable_tags
    mt[ii]  =  mt['[sim_id]']
    mt[ii] += '_'     + str(mt['[windspeed]']) + 'ms'
    mt[ii] += '_blade'+ blade_struct
    mt[ii] += '_ab'   + str(mt['[sweep_amp]'])
    mt[ii] +=           str(mt['[sweep_exp]'])
    #mt[ii] += '_tipm' + str(mt['[bladetipmass]'])
    mt[ii] += '_p' + str(mt['[pitch_angle]'])
    #mt[ii] += '_sb'   + str(blade_struct)
    #mt[ii] += '_'     + str(mt['[blade_st_group]'])
    #mt[ii] += '_'     + towermode
        #+ '_st'     + str(mt['[st_tower_subset]'])
        #+ '_tlen'   + format(mt['[tower_length]'], '2.1f')
        #+ '_nrnb'   + str(mt['[nr_nodes_blade]']) \
    # ==============================================================

    return master

def var_case_name(master):
    """
    Method that generates the case name
    This one should be more dependent on the project. The variable tag could
    than remain very general and keep covering lots of cases for minimal
    required changes. However, on the downside, if you need to change the
    variable_tag_func it gets more difficult because of increased complexity
    """
    yawMode = master.tags['[yawMode_casetag]']
    rpm_mode = master.tags['[rpm_mode_casetag]']
    blade_struct = master.tags['[blade_struct_casetag]']
    ii = '[case_id]'
    # set case_id here instead of HawcPy.master.variable_tags
    master.tags[ii]  =  master.tags['[sim_id]']
    master.tags[ii] += '_'     + str(master.tags['[windspeed]']) + 'ms'
    master.tags[ii] += '_s'    + str(master.tags['[turb_seed]'])
    master.tags[ii] += '_y'    + str(master.tags['[wyaw]'])
    #master.tags[ii] += '_ini'  + str(master.tags['[yaw_angle_init]'])
    master.tags[ii] += '_'     + yawMode
    #master.tags[ii] += '_'     + rot_type
    master.tags[ii] += '_'     + rpm_mode
    #master.tags[ii] += '_'     + master.tags['[blade_hawtopt_file]']
    master.tags[ii] += '_sb'   + str(blade_struct)
    master.tags[ii] += '_ab'   + str(master.tags['[sweep_amp]'])
    master.tags[ii] +=           str(master.tags['[sweep_exp]'])
    master.tags[ii] += '_c'    + str(master.tags['[coning_angle]'])
    master.tags[ii] += '_p'    + str(master.tags['[pitch_angle]'])
    master.tags[ii] += '_ts'   + str(master.tags['[tower_shadow]'])
    master.tags[ii] += '_cd'   + str(master.tags['[tower_shadow_cd]'])
    #master.tags[ii] += '_'     + towermode
        #+ '_st'     + str(master.tags['[st_tower_subset]'])
        #+ '_tlen'   + format(master.tags['[tower_length]'], '2.1f')
        #+ '_nrnb'   + str(master.tags['[nr_nodes_blade]']) \
    return master

def _all_in_one_blade_tag(mt, radius_new=None, nr_nodes=False):
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

    nr_nodes : int, default=False
        If set to false, nr_nodes is taken from mt['[nr_nodes_blade]']

    """

    # TODO: implement support for x position to be other than zero

    # TODO: This is not a good place, should live somewhere else. Or
    # reconsider inputs etc so there is more freedom in changing the
    # location of the nodes, set initial x position of the blade etc

    # and save under tag [blade_htc_node_input] in htc input format

    if not nr_nodes:
        nr_nodes = mt['[nr_nodes_blade]']

    blade = mt['[blade_hawtopt]']

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
    mt['[blade_nodes_z_positions]'] = radius_new

    # make sure that radius_hr is just slightly smaller than radius low res
    radius_new[-1] = blade[-1,0]-0.00000001
    twist_new = interpolate.griddata(blade[:,0], blade[:,2], radius_new)
    # blade_new is the htc node input part:
    # sec 1   x     y     z   twist;
    blade_new = sp.zeros((len(radius_new),4))
    blade_new[:,2] = radius_new*mt['[zaxis_fact]']
    # twist angle remains the same in either case (standard/ojf rotation)
    blade_new[:,3] = twist_new*-1.

    # add one node for the cilinder root sections

    # set the correct sweep cruve, these values are used with the eval
    # statement
    a = mt['[sweep_amp]']
    b = mt['[sweep_exp]']
    z0 = mt['[sweep_curve_z0]']
    ze = mt['[sweep_curve_ze]']
    tmp = 'nsec ' + str(nr_nodes) + ';'
    for k in range(nr_nodes):
        tmp += '\n'
        i = k+1
        z = blade_new[k,2]
        y = blade_new[k,1]
        twist = blade_new[k,3]
        # x position, sweeping?
        if z >= z0:
            x = eval(mt['[sweep_curve_def]'])
        else:
            x = 0.0

        # the node number
        tmp += '        sec ' + format(i, '2.0f')
        tmp += format(x, ' 11.03f')
        tmp += format(y, ' 11.03f')
        tmp += format(z, ' 11.04f')
        tmp += format(twist, ' 11.03f')
        tmp += ' ;'

    mt['[blade_htc_node_input]'] = tmp

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

    return mt

def master_tags(sim_id, runmethod=True, turbulence=False, silent=False,
                verbose=False):
    """
    Create HtcMaster() object
    =========================

    the HtcMaster contains all the settings to start creating htc files.
    It holds the master file, server paths and more.

    The master.tags dictionary holds those tags who do not vary for different
    cases. Variable tags, i.e. tags who are a function of other variables
    or other tags, are defined in the function variable_tag_func().

    It is considered as good practice to define the default values for all
    the variable tags in the master_tags

    Members
    -------

    Returns
    -------

    """

    # TODO: write a lot of logical tests for the tags!!

    # FIXME: some tags are still variable! Only static tags here that do
    # not depent on any other variable that can change

    master = sim.HtcMaster(verbose=verbose, silent=silent)

    # =========================================================================
    # SOURCE FILES
    # =========================================================================
    # TODO: move to variable_tag
    if runmethod in ['local', 'local-script', 'none']:
        path = 'simulations/hawc2/' + sim_id + '/'
        master.tags['[run_dir]'] = path
    elif runmethod == 'thyra':
        master.tags['[run_dir]'] = '/mnt/thyra/HAWC2/ojf_post/' + sim_id + '/'
    elif runmethod == 'gorm':
        master.tags['[run_dir]'] = '/mnt/gorm/HAWC2/ojf_post/' + sim_id + '/'
    else:
        msg='unsupported runmethod, options: none, local, thyra, gorm or opt'
        raise ValueError, msg

    # blade layout is taken form an HAWTOPT blade.dat result file
    # make sure that aeset and blade.dat file name are refering to the same!
    master.tags['[blade_hawtopt_file]'] = 'blade.dat'
    master.tags['[blade_hawtopt_dir]'] = 'data/model/blade_hawtopt/'
    master.tags['[master_htc_file]'] = 'ojf_post_master.htc'
    master.tags['[master_htc_dir]'] = 'data/model/hawc2/htc/'
    master.tags['[model_dir_local]'] = 'data/model/hawc2/'
    master.tags['[post_dir]'] = 'simulations/hawc2/raw/'

    # -------------------------------------------------------------------------
    # semi variable tags that only change per simulation series
    # TODO: create a stand alone function for this? As in variable_tag_func?
    master.tags['[sim_id]'] = sim_id
    # folder names for the saved results, htc, data, zip files
    # Following dirs are relative to the model_dir_server and they specify
    # the location of where the results, logfiles, animation files that where
    # run on the server should be copied to after the simulation has finished.
    # on the node, it will try to copy the turbulence files from these dirs
    master.tags['[animation_dir]'] = 'animation/'
    master.tags['[control_dir]']   = 'control/'
    master.tags['[data_dir]']      = 'data/'
    master.tags['[eigenfreq_dir]'] = 'eigenfreq/'
    master.tags['[htc_dir]']       = 'htc/'
    master.tags['[log_dir]']       = 'logfiles/'
    master.tags['[meander_dir]']   = 'meander/'
    master.tags['[opt_dir]']       = 'opt/'
    master.tags['[pbs_out_dir]']   = 'pbs_out/'
    master.tags['[res_dir]']       = 'results/'
    master.tags['[turb_dir]']      = 'turb/'
    master.tags['[wake_dir]']      = 'wake/'

    # set the model_zip tag to include the sim_id
    master.tags['[model_zip]'] = 'ojf_post'
    master.tags['[model_zip]'] += '_' + master.tags['[sim_id]'] + '.zip'
    # -------------------------------------------------------------------------

    # =========================================================================
    # basic required tags by HtcMaster and PBS in order to function properly
    # =========================================================================
    # case_id will be set with variable_tag_func
    master.tags['[case_id]'] = None

    master.tags['[pbs_queue_command]'] = '#PBS -q workq'
    # the express queue has 2 thyra nodes with max walltime of 1h
    #master.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
    # walltime should have following format: hh:mm:ss
    #master.tags['[walltime]'] = '00:20:00'
    master.tags['[walltime]'] = '03:00:00'
    master.tags['[auto_walltime]'] = True

    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================
    # NO UPPERCASES IN FILENAMES, HAWC2 WILL MAKE THEM LOWER CASE!!!!
    master.tags['[sim]'] = True
    master.tags['[out_format]'] = 'hawc_ascii'
    master.tags['[ascii_buffer]'] = 1
    #master.tags['[out_format]'] = 'HAWC_BINARY'
    #master.tags['[dt_sim]'] = 0.00015
    master.tags['[auto_set_sim_time]'] = True
    master.tags['[azim-res]'] = 120

    master.tags['[beam_output]'] = False
    master.tags['[body_output]'] = False
    master.tags['[body_eigena]'] = False
    master.tags['[stru_eigena]'] = False
    master.tags['[animation]'] = False
    master.tags['[logfile]'] = True
    # convergence_limits  0.001  0.005  0.005 ;
    # critical one, risidual on the forces: 0.0001 = 1e-4
    master.tags['[epsresq]'] = '1e-5' # default=10.0
    # increment residual
    master.tags['[epsresd]'] = '1e-6' # default= 1.0
    # constraint equation residual
    master.tags['[epsresg]'] = '7e-7' # default= 0.7

    # =========================================================================
    # ========= DLL and CONTROL parametrs switches ==========
    # =========================================================================
    # if generator is off, than fix_rpm needs to be defined
    # switch: "" for ON or ";" for DLL control
    master.tags['[generator]'] = False
    master.tags['[ojf_generator_dll]'] = 'ojf_generator.dll'
    # 300 RPM = 31.41 rad/s, 400 RPM = 41.89 rad/s
    # 600 RPM = 62.83 rad/s, 800 RPM = 83.78 rad/s
    # you can chose which format, but don't inlcude both of them
    #master.tags['[fix_rads]'] = 62.83
    master.tags['[fix_rpm]'] = 60.
    # K0, Torque constant t<=t0
    master.tags['[gen_K0]'] = 0
    master.tags['[gen_t0]'] = 6.2
    # K1, Torque constant t0<t<=t1
    master.tags['[gen_K1]'] = 0
    master.tags['[gen_t1]'] = 8.0
    # K2, Torque constant t1<t
    master.tags['[gen_K2]'] = 0.06 # Nm / (rad/s)
    # minimum allowable output Torque
    master.tags['[gen_T_min]'] = 0
    # maximum allowable output Torque
    master.tags['[gen_T_max]'] = 10
    # lowpass filter applied on RPM signal: 1=first order filter, 2=2nd
    master.tags['[gen_filt_type]'] = 0
    # first order filter setting
    master.tags['[gen_1_tau]'] = 5.0
    # second order filter settings
    master.tags['[gen_2_f0]'] = 100.0 # cut-off freq
    # 2 filt: critical damping ratio
    master.tags['[gen_2_ksi]'] = 0.7

    # yaw controller, switch off by default
    master.tags['[yaw_c]'] = False
    master.tags['[yaw_control_dll]'] = 'yaw_control.dll'
    # time when the controller should stop working
    master.tags['[yaw_c_tstop]'] = 10
    # angle the controller will try to maintain
    master.tags['[yaw_c_ref_angle]'] = 0
    #master.tags['[yawmode]'] = 'control_ini'
    #master.tags['[yaw_c_tstop]'] = 30
    #master.tags['[yaw_c_ref_angle]'] = 10
    # 35, 16, 25 after tweaking done in sim_id = 'fr_04_free'
    master.tags['[yaw_c_gain_pro_base]'] = 0.0035 # 0.002
    master.tags['[yaw_c_gain_int_base]'] = 0.0016 # 0.0014 # 0.0003
    master.tags['[yaw_c_gain_dif_base]'] = 0.0025 # 0.002
    master.tags['[yaw_c_damp]'] = 0.7
    # maximum allowable yaw moment applied by the controller
    master.tags['[yaw_c_min]'] = -2.0
    master.tags['[yaw_c_max]'] = 2.0
    # the yaw misalignment is only relevant in fixed yaw cases!
    master.tags['[yaw_angle_misalign]'] = 0.0

    # =========================================================================
    # ========= GENERAL DESIGN OPTIONS =========
    # =========================================================================
    master.tags['[shaft]'] = True
    master.tags['[noshaft]'] = False
    master.tags['[rotation_type]'] = 'std' # ojf
    # switches for yaw bearing and controller
    master.tags['[yawfree]'] = True
    master.tags['[yawfix]'] = False
    # tower simple or with support structure
    master.tags['[tower_simple]'] = False
    master.tags['[tower_support]'] = True

    # =========================================================================
    # ========= WIND CONDITIONS =========
    # =========================================================================
    # tags with respect to turbulence conditions
    master.tags['[TI_ref]'] = 0.14
    # number of points on the turbulence grid point have to be powers of 2
    master.tags['[turb_grid_x]'] = 8192
    master.tags['[turb_grid_yz]'] = 8
    master.tags['[turb_seed]'] = 0
    master.tags['[rotor_diameter]'] = 1.6

    master.tags['[windspeed]'] = 8
    master.tags['[wyaw]'] = 0
    master.tags['[wtilt]'] = 0
    # shear
    master.tags['[shear_type]'] = 1 # no shear=1, power law=3
    master.tags['[shear_exp]'] = 0 # 0.2
    # gust
    master.tags['[gust]'] = ';' # switch
    master.tags['[gust_type]'] = ''
    master.tags['[G_A]'] = ''
    master.tags['[G_phi0]'] = ''
    master.tags['[G_T]'] = ''
    master.tags['[G_t0]'] = ''
    master.tags['[e1]'] = ';' # switch
    # wind ramping: going from factor 0 at t0=0 to 1 at t1
    master.tags['[windramp]'] = False
    master.tags['[t1_windramp]'] = 8
    # absolute ramping: adding up wind_ramp over t0,t1 interval
    master.tags['[windrampabs]'] = False
    master.tags['[wind_ramp]'] = 8 # wind speed at the end of the ramp
    master.tags['[t0_rampabs]'] = 5
    master.tags['[t1_rampabs]'] = 80
    # turbulence
    master.tags['[turb_format]'] = 1
    master.tags['[turb_grid_x]'] = 4096 # number of turb points x-wise
    # number of grid points in y and z direction: in the rotor plane
    master.tags['[turb_grid_yz]'] = 8
    master.tags['[wsp_factor]'] = 1 # resided under Masterfile.variable_tags

    # =========================================================================
    # ========= GENERIC structural data =========
    # =========================================================================
    master.tags['[st_file]'] = 'ojf.st'
    # stiff_set's currently used for nothing
    master.tags['[st_stiff_set]'] = 5
    master.tags['[st_stiff_subset]'] = 4
    # insane high damping gives trash on the eigen analysis Fn values
    #master.tags['[damp_normal]'] = \
    #'0.0e-00   0.0e-00   0.0e-00   1.0e-08   1.0e-08   1.0e-08'

    master.tags['[st_hub_set]'] = 5
    master.tags['[st_hub_subset]'] = 4

    master.tags['[st_nose_cone_set]'] = 5
    master.tags['[st_nose_cone_subset]'] = 5

    # the nacelle subset includes the generator stator properties at the end!
    master.tags['[st_nacelle_set]'] = 4
    master.tags['[st_nacelle_subset]'] = 5

    master.tags['[st_nacelle_aft_set]'] = 4
    master.tags['[st_nacelle_aft_subset]'] = 3

    master.tags['[st_shaft_set]'] = 6
    master.tags['[st_shaft_subset]'] = 4

    master.tags['[damp_nacelle_aft]'] = \
    '0.0e-00   0.0e-00   0.0e-00   2.0e-08   2.0e-08   2.0e-08'
    master.tags['[damp_nacelle]'] = \
    '0.0e-00   0.0e-00   0.0e-00   2.0e-08   2.0e-08   2.0e-08'

    master.tags['[damp_nose_cone]'] = \
    '0.0e-00   0.0e-00   0.0e-00   5.5e-07   5.5e-07   5.5e-07'

    # set the torsional damping higher, is this where the instability is
    # coming from?
    master.tags['[damp_hub]'] = \
    '0.0e-00   0.0e-00   0.0e-00   5.0e-07   5.0e-07   5.0e-07'
    master.tags['[damp_hub]'] = \
    '0.0e-00   0.0e-00   0.0e-00   7.0e-07   7.0e-07   7.0e-07'

    master.tags['[damp_shaft]'] = \
    '0.0e-00   0.0e-00   0.0e-00   2.5e-05   2.5e-05   2.5e-05'

    # =========================================================================
    # ========= BLADE SETTINGS =========
    # =========================================================================
    master.tags['[pc_file]'] = 's822_s823_2e5.pc'
    master.tags['[hub_vec]'] = 'nose_cone -3'
    # rotation direction: standard when rotation vector points same direction
    # as wind, OJF mode when pointing agains the wind
    # OJF mode: rotation vector points into the wind
    master.tags['[rot_dir_rotor]'] = '-' # negative for OJF mode
    master.tags['[std_rotation]'] = ';' # switch off for OJF mode
    master.tags['[zaxis_fact]'] = -1.   # blade span develops along negative z
    # standard mode: rotation vector pointing down wind (same dir as wind)
    #master.tags['[rot_dir_rotor]'] = '' # negative for OJF mode
    #master.tags['[std_rotation]'] = '' # switch off for OJF mode
    #master.tags['[zaxis_fact]'] = 1.   # blade span develops along pos z

    # blade set: 7
    # very stiff=1, OJF FINAL inc pitch: flexible=39, stiff=40, foamonly=42
    master.tags['[st_blade_set]'] = 7
    master.tags['[st_blade1_subset]'] = stsets.b1_flex_opt2
    master.tags['[st_blade2_subset]'] = stsets.b2_flex_opt2
    master.tags['[st_blade3_subset]'] = stsets.b3_flex_opt2

    master.tags['[damp_blade1]'] = model.b1_flex_damp
    master.tags['[damp_blade2]'] = model.b2_flex_damp
    master.tags['[damp_blade3]'] = model.b3_flex_damp
    #master.tags['[damp_blade]'] = master.tags['[damp_insane]']

    # bodies and nodes
    master.tags['[nr_bodies_blade]'] = 11
    # do not change nr_nodes_blade anymore to in order not to break the
    # node id on the strain gauges
    master.tags['[nr_nodes_blade]'] = 12

    # =========================================================================
    # ========= GEOMETRICAL BLADE LAYOUTR =========
    # =========================================================================
    # blade layout is taken form an HAWTOPT blade.dat result file
    # make sure that aeset and blade.dat file name are refering to the same!
    tmp=master.tags['[blade_hawtopt_dir]']+master.tags['[blade_hawtopt_file]']
    master.tags['[blade_hawtopt]'] = np.loadtxt(tmp)
    # create empty to switch on, blade will be based on [blade_hawtopt_dir]
    master.tags['[blade_htc_node_input]'] = ''
    # TODO: implement aefile creation instead of refering to set numbers
    # and force consitancy with the blade_hawtopt tag
    master.tags['[ae_file]'] = 'ojf.ae'
    master.tags['[aeset]'] = 1
    # blade.dat: aeset=1, normal case
    # chord_L.blade.dat: aeset=2
    # chord_XL.blade.dat: aeset=3
    # chord_XXL.blade.dat: aeset=4

    # REMARK: z0 should be the starting point of the sweep! Which is not equal
    # to zero when starting else than the blade root!!
    # x = a [(z-z0)/(ze-z0)]^b
    # with
    # a : sweep amplitude
    # b : sweep exponent
    # z0 : z-axis coordinate of blade root
    # ze : z-axis coordinate of blade tip
    master.tags['[sweep_amp]'] =  0 # -0.04
    master.tags['[sweep_exp]'] =  0 #  4.0
    master.tags['[sweep_curve_z0]'] = 0.2  # sweep starting point radius
    master.tags['[sweep_curve_ze]'] = 0.55 # blade length
    master.tags['[sweep_curve_def]'] = 'a*math.pow((z-z0)/(ze-z0),b)'

    master.tags['[hub_lenght]'] = 0.245
    # HUB AERODRAG ON BY DEFAULT
    master.tags['[hub_drag]'] = True
    master.tags['[nr_nodes_hubaerodrag]'] = 20
    master.tags['[hub_cd]'] = 2.0
    master.tags['[strain_root_el]'] = 1
    master.tags['[strain_30_el]'] = 4

    # CONING AND PITCH
    master.tags['[coning_angle_b1]'] = 0
    master.tags['[coning_angle_b2]'] = 0
    master.tags['[coning_angle_b3]'] = 0
    # blade pitch (positive = pitch to stall, negative = pitch to feather)
    master.tags['[pitch_angle_imbalance_b1]'] = 0.0
    master.tags['[pitch_angle_imbalance_b2]'] = 0.0
    master.tags['[pitch_angle_imbalance_b3]'] = 0.0
    master.tags['[pitch_angle]'] = -0.8

    # the following is than defined in the variable_taf_func
    # pitch_angle_b1 = pitch_angle + pitch_angle_imbalance_b1

    # =========================================================================
    # ========= TOWER structural data =========
    # =========================================================================
    master.tags['[nr_bodies_tower]'] = 3#2
    # due to strain gauges position required, quick fix to do it manually
    # in the htc file, no more support for nr_nodes_tower as a consequence
    #master.tags['[nr_nodes_tower]'] = 3#2

    # tower length refers to distance hub-yaw bearing upper
    master.tags['[tower_length]'] = 1.755
    # will be overwritten in case of tower_support version, than it is .55 more
    master.tags['[hub_height]'] = 1.755

    # the realistic set with steel tower tube: 8 (2mm)
    # the realistic set with steel tower tube: 9 (3mm) *> is the real one?
    # optimized tower: 12
    master.tags['[st_tower_set]'] = 3
    master.tags['[st_tower_subset]'] = 12

    # tower damping, tuned to match 7% damping on the tower body alone
    #master.tags['[damp_tower]'] = \
        #'0.0e-00   0.0e-00   0.0e-00   4.2e-04   4.2e-04   4.2e-04'

    # tower damping, tuned to match 7% damping on the structural modes
    master.tags['[damp_tower]'] = \
        '0.0e-00   0.0e-00   0.0e-00   0.85e-03   0.85e-03   0.85e-03'

    # =========================================================================
    # ========= TOWER SUPPORT structural data =========
    # =========================================================================
    master.tags['[st_towersupport_set]'] = 5
    master.tags['[st_towersupport_subset]'] = 3

    master.tags['[damp_towersupport_p1]'] = \
    '0.0e-00   0.0e-00   0.0e-00   1.0e-08   1.0e-08   1.0e-08'
    master.tags['[damp_towersupport_p12]'] = \
    '0.0e-00   0.0e-00   0.0e-00   3.0e-08   3.0e-08   3.0e-08'
    master.tags['[damp_towersupport_p2]'] = \
    '0.0e-00   0.0e-00   0.0e-00   5.0e-09   5.0e-09   5.0e-09'
    master.tags['[damp_towersupport_addemdum]'] = \
    '0.0e-00   0.0e-00   0.0e-00   5.0e-09   5.0e-09   5.0e-09'
    master.tags['[damp_tower_addendum]'] = \
    '0.0e-00   0.0e-00   0.0e-00   4.5e-05   4.5e-05   4.5e-05'

    master.tags['[damp_towerbase]'] = \
    '0.0e-00   0.0e-00   0.0e-00   1.0e-01   1.0e-01   1.0e-01'

    # =========================================================================
    # ========= AERODYNAMICS =========
    # =========================================================================
    master.tags['[induction_method]'] = 1
    master.tags['[aerocalc_method]'] = 1
    master.tags['[dynstall]'] = 2
    master.tags['[tiploss]'] = 1
    # 0=none, 1=potential flow, 2=jet, 4=jet_2
    master.tags['[tower_shadow]'] = 4
    # originally 0.5, than I set it to 0.9, but Helge suggest going to 1.1
    master.tags['[tower_shadow_cd]'] = 1.1
    master.tags['[aerosections]'] = 15
    # bemwake settings, HAWC2 default values
    master.tags['[bemwake_nazi]'] = 16
    master.tags['[nw_mix]'] = 0.6
    master.tags['[nw_k3]'] =  0.0
    master.tags['[nw_k2]'] = -0.4783
    master.tags['[nw_k1]'] =  0.1025
    master.tags['[nw_k0]'] =  0.6125

    master.tags['[ini_rotvec]'] = 4.0 # rad/s

    # =========================================================================
    # tags falling under easy scenario switching
    # =========================================================================
    if turbulence:
        master.tags['[turb_format]'] = 1 # 0=none, 1=mann, 2=flex
        # use seed=0 in combination with no turbulence, if seed is given as a
        # string, HAWC2 will give an error on that one!
        master.tags['[turb_seed]'] = 5
        master.tags['[walltime]'] = '04:00:00'
        master.tags['[pbs_queue_command]'] = '#PBS -q workq'
        #master.tags['[duration]'] = 10.0
        ## start outputting data from (in order to ignore initial transients)
        #master.tags['[t0]'] = 0.0
        ## wind ramping ends a little bit before t0
        #master.tags['[t1_windramp]'] = 0.0
        #master.tags['[windramp]'] = False
    else:
        # for the STEADY FREE CASES
        master.tags['[turb_format]'] = 0 # 0=none, 1=mann, 2=flex
        # use seed=0 in combination with no turbulence, if seed is given as a
        # string, HAWC2 will give an error on that one!
        master.tags['[turb_seed]'] = 0
        master.tags['[walltime]'] = '04:00:00'
        master.tags['[pbs_queue_command]'] = '#PBS -q workq'
        master.tags['[duration]'] = 10.0
        # TODO: walltime as function of duration and dt
        # start outputting data from (in order to ignore initial transients)
        #master.tags['[t0]'] = 30.0
        ## master file implements t0=0, f0=0, f1=1
        #master.tags['[t1_windramp]'] = 5
        #master.tags['[windramp]'] = True

    return master

###############################################################################
### LAUNCH JOBS
###############################################################################

def launch(simid, msg):
    # =========================================================================
    # CREATE ALL HTC FILES AND PBS FILES TO LAUNCH ON THYRA/GORM
    # =========================================================================
    # see create_multiloop_list() docstring in Simulations.py
    iter_dict = dict()
    #iter_dict['[coning_angle]'] = [0,-10]
    #iter_dict['[st_blade_subset]'] = np.arange(22, 27, 1).tolist()
    #iter_dict['[st_tower_subset]'] = [11]
    #iter_dict['[tower_length]'] = [-2.0]
    iter_dict['[windspeed]'] = [1]
    #iter_dict['[st_blade_subset]'] = [1]#,39,40]
    #iter_dict['[tower_shadow]'] = [0]
    #iter_dict['[tower_shadow_cd]'] = [0.5]
    #iter_dict['[body_eigena]'] = ''
    # 300 RPM = 31.41 rad/s, 400 RPM = 41.89 rad/s
    # 600 RPM = 62.83 rad/s, 800 RPM = 83.78 rad/s
    #iter_dict['[fix_wr]'] = [10.47, 31.41, 62.83]
    #iter_dict['[fix_wr]'] = [31.41]


    normal = dict({'[blade_hawtopt_file]' : 'blade.dat',
                   '[tower_length]'  : 1.8,
                   '[aeset]'         : 5})

    medium = dict({'[blade_hawtopt_file]' : 'blade_m.dat',
                   '[tower_length]'  : 3.,
                   '[aeset]'         : 5})

    xl = dict({    '[blade_hawtopt_file]' : 'blade_x10.dat',
                   '[tower_length]'  : 10.,
                   '[aeset]' : 5})

    std_rot = dict({'[rotation_type]' : 'std'})
    ojf_rot = dict({'[rotation_type]' : 'ojf'})

    tower_simple_fix = dict({'[tower_simple]'  : True,
                             '[tower_support]' : False,
                             '[yawfix]'        : True,
                             '[yawfree]'       : False})

    tower_support_fix = dict({'[tower_simple]'  : False,
                              '[tower_support]' : True,
                              '[yawfix]'        : True,
                              '[yawfree]'       : False})

    tower_simple_free = dict({'[tower_simple]'  : True,
                             '[tower_support]' : False,
                             '[yawfix]'        : False,
                             '[yawfree]'       : True})

    tower_support_free = dict({'[tower_simple]'  : False,
                              '[tower_support]' : True,
                              '[yawfix]'        : False,
                              '[yawfree]'       : True})

    # for the tuning of the stiffness/mass distribution
    eigenanalysis = dict({'[body_eigena]'     : '',
                          '[stru_eigena]'     : '',
                          '[sim]'             : ';',
                          '[nr_bodies_tower]' : 1,
                          '[nr_bodies_blade]' : 1,
                          '[tower_simple]'    : True,
                          '[tower_support]'   : False,
                          '[yawfix]'          : True,
                          '[yawfree]'         : False})

    opt_tags = [tower_support_fix, tower_simple_fix]
    opt_tags = [eigenanalysis]
    #, tower_support_free,tower_simple_free]

    ## tip speed ratios for small blade
    #wr = np.array([10.47, 31.41, 62.83])
    #tsr = wr*(0.555+0.245)/10.
    ## rotation speed required for big blade, maintaining tip speed ratios
    #wr_xl = tsr*10./(5.55+0.245)
    ## wr_xl
    ## Out[24]: array([ 1.44538395,  4.33615186,  8.67368421])

    # for each case in iter_dict, use each dictionary in opt_tags
    #nosweep = dict({'[sweep_amp]' : 0, '[sweep_exp]' : 0})
    #sweep01 = dict({'[sweep_amp]' : -0.02, '[sweep_exp]' : 4.})
    #sweep02 = dict({'[sweep_amp]' : -0.04, '[sweep_exp]' : 4.})
    #sweep03 = dict({'[sweep_amp]' : -0.06, '[sweep_exp]' : 4.})
    #sweep04 = dict({'[sweep_amp]' : -0.08, '[sweep_exp]' : 4.})
    #opt_tags = [nosweep, sweep01, sweep02, sweep03, sweep04]

    # do not change tags after this point, use iter_dict or opt_tags instead
    # that is much safer, you wil not mix up for instance the directory
    # structure that might have sim_id component
    runmethod = 'local'
    master = master_tags(simid, runmethod=runmethod, turbulence=False)

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg=msg)

    # overview of launched sims
    # b2: 4ms 300rpm ts(on off) rotation(std ojf) nosweep stiff
    # b3: 4ms 300rpm ts(off) rotation(std ojf) nosweep stiff
    # b4: idem b3, aero rotation in rotor polar coordinates
    # b5: idems as b3, now corrected imposed rotor speed for ojf case
    # before it was always rotating in the same direction
    # b6: idem b5, changed hub_vec for ojf rotation type to 3 instead of -3
    # b7: idem b3, simple aero: no induction, tip loss or dyn stall
    # c1: major change: first tests of the support structure
    # c2: support(on off) freeyaw(on off) std rotation
    # c3: support(on no addendum, off), freeyaw(on, off)
    # still support not working, still mistakes in fix
    # c4: support(on, off) freeyaw(on, off) higher residual
    # -> simple tower works, still crash after 8.55 seconds for support
    # c5: support(on, each part different body) -> same problem
    # c6: support(on, off), freeyaw(on, off) high damp -> finally working!
    # c7: support(on, off), freeyaw(on, off) high damp, good time window, BIN
    # c8: idem c7, ascii
    # c9: tower addendum wasn't put correctly
    # d1: some more testing
    # d2: compare if nr nodes has computational penalty, BENCHMARK
    # d3: testing on dashboard, final adjustments nr nodes
    # d4: eigenfrequency for body and structure
    # d5: testing/debugging the renewed sim.run_local and optimizer construct

def eigenvalue_analysis():
    """
    Run the HAWC2 model for only the structural analysis, no time simluation
    """

    simid = 'eigenanal_00'
    msg = 'Checking structrural sanity of the model'

    # see create_multiloop_list() docstring in Simulations.py
    iter_dict = dict()
    iter_dict['[windspeed]'] = [1]

    opt_tags = []
    # do not change tags after this point, use iter_dict or opt_tags instead
    # that is much safer, you wil not mix up for instance the directory
    # structure that might have sim_id component
    runmethod = 'local'
    master=master_tags(simid,runmethod=runmethod,turbulence=False,silent=True)
    # just to be sure, set the blade st set correctly
    master.tags['[fix_rpm]'] = 100
    master.tags['[generator]'] = False
    #master.tags['[beam_output]'] = True
    #master.tags['[body_output]'] = True
    master.tags['[body_eigena]'] = True
    #master.tags['[stru_eigena]'] = True
    master.tags['[animation]'] = False
    master.tags['[sim]'] = False
    master.tags['[nr_bodies_tower]'] = 1
    master.tags['[nr_bodies_blade]'] = 1
    master.tags['[yawmode]'] = 'fix'
    master.tags['[tower_simple]'] = False
    master.tags['[tower_support]'] = True

    master.tags['[damp_blade1]'] = model.b1_flex_damp
    master.tags['[damp_blade2]'] = model.b2_flex_damp
    master.tags['[damp_blade3]'] = model.b3_flex_damp

    htc_dict=sim.prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg=msg, silent=False, check_log=False)

    #htc_dict = sim.run_local(htc_dict, silent=False, check_log=False)

def post_launch(sim_id, post_dir):
    # THIS FUNCTION SHOULD BE MOVED TO SIMULATIONS
    # TODO: finish support for default location of the htc_dict and file name
    # two scenario's: either pass on an htc_dict and get from their the
    # post processing path or pass on the simid and load from the htc_dict
    # from the default location

    # =========================================================================
    # check logfiles, results files, pbs output files
    # logfile analysis is written to a csv file in logfiles directory
    # =========================================================================
    # load the file saved in post_dir
    htc_dict = sim.load_pickled_file(post_dir + sim_id + '.pkl')

    # if the post processing is done on simulations done by thyra/gorm, and is
    # downloaded locally, change path to results
    for case in htc_dict:
        if htc_dict[case]['[run_dir]'][:4] == '/mnt':
            htc_dict[case]['[run_dir]'] = 'simulations/hawc2/%s/' % sim_id

    # check if all results are ok. An overview of which errors where found in
    # the logfiles is written in the post_dir
    htc_dict_fail = sim.post_launch(htc_dict)
    # htc_dict_fail is saved in sim.post_launch() as simid_fail.pkl

    # ditch all the failed cases out of the htc_dict
    # otherwise we will have fails when reading the results data files
    for k in htc_dict_fail:
        del htc_dict[k]
        print 'removed from htc_dict due to error: ' + k

###############################################################################
### DATA FILES
###############################################################################

def reformat_st():
    """
    After some changes, you might wanne have the layout of the file nice again
    """
    file_path = 'data/model/hawc2/data/'
    file_name = 'ojf.st'

    md = sim.ModelData(silent=True)
    md.prec_float = ' 10.06f'
    md.prec_exp = ' 10.5e'
    md.load_st(file_path, file_name)
    md.write_st(file_path, file_name)

def blade_st_increase_t_stiffness():
    """
    There shouldn't be any torsional deformation of the blade
    Increase stiffness with factor 2, is that ok?

    CAUTION: set the stsets correctly, define all B1,B2,B3, do not ignore
    the wrongly measured blades
    """

    file_path = 'data/model/hawc2/data/'
    file_name = 'ojf.st'

    md = sim.ModelData(silent=True)
    md.prec_float = ' 10.06f'
    md.prec_exp = ' 10.5e'
    md.load_st(file_path, file_name)

    todo = [ [stsets.b1_stiff_opt2, stsets.b1_stiff_opt2_G2 ],
             [stsets.b2_stiff_opt2, stsets.b2_stiff_opt2_G2 ],
             [stsets.b3_stiff_opt2, stsets.b3_stiff_opt2_G2 ],
             [stsets.b1_flex_opt2, stsets.b1_flex_opt2_G2 ],
             [stsets.b2_flex_opt2, stsets.b2_flex_opt2_G2 ],
             [stsets.b3_flex_opt2, stsets.b3_flex_opt2_G2 ],
             [stsets.b1_stiff_opt2_cyl, stsets.b1_stiff_opt2_cyl_G2 ],
             [stsets.b2_stiff_opt2_cyl, stsets.b2_stiff_opt2_cyl_G2 ],
             [stsets.b3_stiff_opt2_cyl, stsets.b3_stiff_opt2_cyl_G2 ],
             [stsets.b1_flex_opt2_cyl, stsets.b1_flex_opt2_cyl_G2 ],
             [stsets.b2_flex_opt2_cyl, stsets.b2_flex_opt2_cyl_G2 ],
             [stsets.b3_flex_opt2_cyl, stsets.b3_flex_opt2_cyl_G2 ]
             ]

    iG = sim.ModelData.st_headers.G

    for st_source, sttarget in todo:
        # create new sets for the sections with cylinder
        msg = 'same as %i, but with increased G\n' % st_source
        st_arr_ref = md.st_dict['007-%03i-d' % st_source].copy()
        header_org = md.st_dict['007-%03i-a' % st_source]
        st_arr_new = st_arr_ref.copy()
        # and scale the torsional stiffness
        st_arr_new[:,iG] *= 2.0
        md.st_dict['007-%03i-d' % sttarget] = st_arr_new
        md.st_dict['007-%03i-a' % sttarget] = header_org + msg
        replace = (sttarget, st_arr_new.shape[0])
        md.st_dict['007-%03i-b' % sttarget] = '$%i %i\n' % replace

    md.write_st(file_path, file_name)


def blade_st_with_cylinder_root():
    """
    Now change the st files so the first 162mm is actually the cylinder
    root sections
    """

    file_path = 'data/model/hawc2/data/'
    file_name = 'ojf.st'

    md = sim.ModelData(silent=True)
    md.prec_float = ' 10.06f'
    md.prec_exp = ' 10.5e'
    md.load_st(file_path, file_name)

    todo = [ [stsets.b1_stiff_opt2, stsets.b1_stiff_opt2_cyl],
             [stsets.b2_stiff_opt2, stsets.b2_stiff_opt2_cyl],
             [stsets.b3_stiff_opt2, stsets.b3_stiff_opt2_cyl],
             [stsets.b1_flex_opt2, stsets.b1_flex_opt2_cyl],
             [stsets.b2_flex_opt2, stsets.b2_flex_opt2_cyl],
             [stsets.b3_flex_opt2, stsets.b3_flex_opt2_cyl] ]

    for st_source, sttarget in todo:
        # create new sets for the sections with cylinder
        msg = 'INCLUDING ROOT CYLINDER WITH L=162mm\n'
        #st_source = 18
        #sttarget = 24
        cyl_length = 0.162
        st_arr_ref = md.st_dict['007-%03i-d' % st_source].copy()
        header_org = md.st_dict['007-%03i-a' % st_source]
        st_arr_new = st_arr_ref.copy()
        # blade model now includes the cylinder root part
        st_arr_new[:,0] += cyl_length
        # add two cylinder nodal points
        st_cylinder = md.st_dict['005-001-d'].copy()
        st_cylinder[1,0] = cyl_length-0.0001
        st_arr_cyl = np.append(st_cylinder, st_arr_new, axis=0)
        md.st_dict['007-%03i-d' % sttarget] = st_arr_cyl
        md.st_dict['007-%03i-a' % sttarget] = header_org + msg
        replace = (sttarget, st_arr_cyl.shape[0])
        md.st_dict['007-%03i-b' % sttarget] = '$%i %i\n' % replace

    md.write_st(file_path, file_name)


def check_opt_results(file_target):

    FILE = open(file_target, 'rb')
    htc_dict = pickle.load(FILE)
    FILE.close()

    case = htc_dict.keys()[0]
    # the objectives are the blades first 3 eigenfrequencies
    blade1_freq = htc_dict[case]['[eigen_body_results]']['blade1']
    fn = blade1_freq[1,:]
    # several modes are repeated, so we need to find the ones that differ
    # more than just a few percent
    df = np.diff(fn)
    sel1 = (df/fn[1:]).__ge__(0.01)
    sel = np.append(sel1, sel1[-1])
    print fn[sel][:3]


###############################################################################
### CHANNELS
###############################################################################

class model:
    # final optimized values!
    b1_flex_damp  = '  0.0    0.0    0.0    2.880e-04   2.880e-04   2.880e-04'
    b2_flex_damp  = '  0.0    0.0    0.0    2.976e-04   2.976e-04   2.976e-04'
    b3_flex_damp  = '  0.0    0.0    0.0    2.751e-04   2.751e-04   2.751e-04'
    b1_stiff_damp = '  0.0    0.0    0.0    1.305e-04   1.305e-04   1.305e-04'
    b2_stiff_damp = '  0.0    0.0    0.0    1.362e-04   1.362e-04   1.362e-04'
    b3_stiff_damp = '  0.0    0.0    0.0    1.339e-04   1.339e-04   1.339e-04'

    # increasing damping on the torsion mode to avoid instabilities
    b1_flex_damp  = '  0.0    0.0    0.0    2.880e-04   2.880e-04   2.880e-03'
    b2_flex_damp  = '  0.0    0.0    0.0    2.976e-04   2.976e-04   2.976e-03'
    b3_flex_damp  = '  0.0    0.0    0.0    2.751e-04   2.751e-04   2.751e-03'
    b1_stiff_damp = '  0.0    0.0    0.0    1.305e-04   1.305e-04   1.305e-03'
    b2_stiff_damp = '  0.0    0.0    0.0    1.362e-04   1.362e-04   1.362e-03'
    b3_stiff_damp = '  0.0    0.0    0.0    1.339e-04   1.339e-04   1.339e-03'

    blade_length = 0.555
    hub_radius = 0.245
    blade_radius = blade_length + hub_radius
    A_tot = (blade_length + hub_radius)**2*np.pi
    A = A_tot - (hub_radius**2*np.pi)

    # moment arm rotor is the distance from the rotor centre to the strain
    # gauges at the tower base, just above the first bearing
    momemt_arm_rotor = 1.64

    # blade pitch settings during the experiment
    # the blade has a tip pitch of -0.8 deg in blade coordinates
    # seems that for February, stiff was 0.5-1.0 degrees less pitch (aka closer
    # to the having no pitch at the tip at all, zero pitch at tip)
    # Values uses, among others, at ojfvshawc2.ojf_to_hawc2(), ojf_gen
    p_stiff_02 = -0.4     # HS tip: -1.4
    # from HS cam: at max chord tip pitch is for B1,2 = -1.7, B3 = -1.4 deg
    p_flex_02  = -1.0     # HS tip: -1.8
    # from the pictures taken of the stiff blade manually, we have a pitch
    # angle of around 2.8-3.8 degrees
    p_stiff_04 = -2.0     # pictures manual: -3.8 -2.8
    p_flex_04  = -2.0     # HS tip: -3.4 -2.0

    # 0209_run_016--chord-pitch: 1 blade has a pitch imbalance of 0.5 degrees
    # less pitching than the other 2

    # coning imbalance
    # >>> math.atan(0.7/55)*180/math.pi
    # 0.7291796420020735
    coning_imbalance = 0.8 # NOT USED ANYWHERE, just for illustration
    pith_imbalance = 0.5 # NOT USED ANYWHERE

class stsets:
    """
    st subset numbers for the blades
    """

    tower_opt = 12
    tower_3mm = 9

    stiff = 1
    fanzhong = 2
    foamonly = 3
    # original etanna output for the stiff beam (only glass fibre)
    beam_stiff = 8
    # mass corrected for measured mass
    b1_stiff_cor = 5
    b2_stiff_cor = 6
    b3_stiff_cor = 7
    # original etanna output for the flexible beam (only glass fibre)
    beam_flex = 4
    # mass corrected for measured mass
    b1_flex_cor = 9
    b2_flex_cor = 10
    b3_flex_cor = 11
    # E optimized to match eigenfrequencies
    b1_stiff_opt = 12
    b2_stiff_opt = 13
    b3_stiff_opt = 14
    # E optimized to match eigenfrequencies
    b1_flex_opt = 15
    b2_flex_opt = 16
    b3_flex_opt = 17

    # ------------------------------------------------------------------------
    # for the deflection curve based (changing E), and eigenfrequency based
    # (changing mass) optimisiations we use the same sets
    # for the time being, ignore the bogus blades: blade 3 stiff, blade 1 flex
    # ------------------------------------------------------------------------
    b1_stiff_opt2 = 18
    b2_stiff_opt2 = 19
    b3_stiff_opt2 = 20
    b3_stiff_opt2 = b1_stiff_opt2

    b1_flex_opt2 = 21
    b2_flex_opt2 = 22
    b3_flex_opt2 = 23
    b1_flex_opt2 = b2_flex_opt2

    # now blade sets including the cylinder root sections
    b1_stiff_opt2_cyl = 24
    b2_stiff_opt2_cyl = 25
    b3_stiff_opt2_cyl = 26
    b3_stiff_opt2_cyl = b1_stiff_opt2_cyl

    b1_flex_opt2_cyl = 27
    b2_flex_opt2_cyl = 28
    b3_flex_opt2_cyl = 29
    b1_flex_opt2_cyl = b2_flex_opt2_cyl

    # ------------------------------------------------------------------------
    # and scaling the torsion of the blade to higher values so we have no
    # torsional deformation as in the measurements
    # ------------------------------------------------------------------------
    b1_stiff_opt2_G2 = 30
    b2_stiff_opt2_G2 = 31
    b3_stiff_opt2_G2 = 32
    b3_stiff_opt2_G2 = b1_stiff_opt2_G2

    b1_flex_opt2_G2 = 33
    b2_flex_opt2_G2 = 34
    b3_flex_opt2_G2 = 35
    b1_flex_opt2_G2 = b2_flex_opt2_G2

    # now blade sets including the cylinder root sections
    b1_stiff_opt2_cyl_G2 = 36
    b2_stiff_opt2_cyl_G2 = 37
    b3_stiff_opt2_cyl_G2 = 38
    b3_stiff_opt2_cyl_G2 = b1_stiff_opt2_cyl_G2

    b1_flex_opt2_cyl_G2 = 39
    b2_flex_opt2_cyl_G2 = 40
    b3_flex_opt2_cyl_G2 = 41
    b1_flex_opt2_cyl_G2 = b2_flex_opt2_cyl_G2

class Chi:
    """
    Some model specific data for OJF
    """

    # general
    rotorspd = 1
    aetorq = 2
    aepow = 3
    aethrust = 4
    azi_b1 = 5
    windy = 7
    yawwind = 9

    #azi_b1 = 21

    # wind y at blade
    windy_r55 = 10
    #windy_r49 = 10
    #windy_r45 = 11
    #windy_r40 = 12
    #windy_r34 = 13
    #windy_r28 = 14
    #windy_r22 = 16
    #windy_r16 = 17
    #windy_r10 = 18
    #windy_r06 = 19
    #windy_r00 = 20

    chi_details = dict()

    # CAUTION: no overlap between amp_chilist and damp_chilist!
    # calculate amplitudes for channels
    amp_chilist = []

    # damping is only relevant for a certain set of channels,
    # otherwise too many warnings for divide by zero or invaled log value
    damp_chilist = []

class ChiT:
    """
    channel indeces used for Torque 2012
    """
    # --------------------------------------------------------------------
    # channels
    # NOTE: we have chi = HAWC2 channel - 1
    omega = 1
    aepower = 3
    aethrust = 4
    azi1 = 5
    wind = 6
    winddir = 7
    vrel28 = 8
    vrel49 = 9
    aoa28 = 10
    aoa49 = 11
    cl28 = 12
    cl49 = 13

    # all for blade 1
    # root means root strain gauges
    mxroot_loc = 14 # Flapwise
    myroot_loc = 15
    mzroot_loc = 16
    mx30_loc = 17 # Flapwise
    my30_loc = 18
    mz30_loc = 19
    # blade 1 bending moments at strain gauges
    mxroot = 20 # Flapwise
    myroot = 21
    mzroot = 22
    mx30 = 23 # Flapwise
    my30 = 24
    mz30 = 25
    # blade 1 forces at strain gauges
    fxroot = 26 # Flapwise
    fyroot = 27
    fzroot = 28
    fx30 = 29 # Flapwise
    fy30 = 30
    fz30 = 31
    # blade 1 tip deflections
    tipx = 32
    tipy = 33
    tipz = 34

    tip_rotaero = 38

    yawangle=40
    tiprot1_ae_tz = 38

    mxtower = 41 # FA
    mytower = 42 # SS
    mxtower_glo = 44 # FA
    mytower_glo = 45 # SS

    dll_control = 56
    yaw_ref_angle = 59

    mxshaft = 60
    myshaft = 61
    mzshaft = 62
    fxshaft = 63
    fyshaft = 64
    fzshaft = 65
    fxnacelle = 66
    fynacelle = 67
    fznacelle = 68

###############################################################################
### PLOTTING
###############################################################################

def tower_shadow_profile():
    """
    compare different RPM's for the small blade

    Print tower shadow profile using the aero windspeed output command.
    This is not ideal, since we only have the vertical wind speed component
    as seen from the blade. Only the blade positions can be mapped. By
    overlaying with azimuth angle, the wind profile can be mapped as function
    of lateral position.
    """
    # setup the plot
    figpath = 'simulations/fig/ts/'
    figfile = 'tower-shadow-comparison'
    pa4 = plotting.A4Tuned()
    title = 'Tower shadow at 10 m/s'
    pa4.setup(figpath+figfile, grandtitle=title, figsize_x=16, figsize_y=16,
              wsleft_cm=2., wsright_cm=1.5, hspace=0.3)
    ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    # case 100 rpm
    respath = 'simulations/hawc2/'
    resfile = 's0_a0_yfix_10ms_y0_ts4_100rpm_c0_p0_sb71_ab00_st2'
    res100 = HawcPy.LoadResults(respath, resfile)
    # find the lowest tower shadow passage event
    azi_b1 = res100.sig[:,Chi.azi_b1]
    wind = res100.sig[:,Chi.windy_r55]
    starti = np.argmin(wind)
    end = 50
    slice_100 = np.r_[starti-end:starti+end]
    # convert azimuth angle to lateral positions
    xlat = np.tan(azi_b1*np.pi/180.)*0.8/0.05
    # plot
    ax1.plot(xlat[slice_100], wind[slice_100], label='100 RPM')


    # case 300 rpm
    resfile = 's0_a0_yfix_10ms_y0_ts4_300rpm_c0_p0_sb71_ab00_st2'
    res300 = HawcPy.LoadResults(respath, resfile)
    # find the lowest tower shadow passage event
    azi_b1 = res300.sig[:,Chi.azi_b1]
    wind = res300.sig[:,Chi.windy_r55]
    starti = np.argmin(wind)
    end = 50
    slice_300 = np.r_[starti-end:starti+end]
    # convert azimuth angle to lateral positions
    xlat = np.tan(azi_b1*np.pi/180.)*0.8/0.05
    # plot
    ax1.plot(xlat[slice_300], wind[slice_300], label='300 RPM')


    # case 600 rpm
    resfile = 's0_a0_yfix_10ms_y0_ts4_600rpm_c0_p0_sb71_ab00_st2'
    res600 = HawcPy.LoadResults(respath, resfile)
    # find the lowest tower shadow passage event
    azi_b1 = res600.sig[:,Chi.azi_b1]
    wind = res600.sig[:,Chi.windy_r55]
    starti = np.argmin(wind)
    end = 50
    slice_600 = np.r_[starti-end:starti+end]
    # convert azimuth angle to lateral positions
    xlat = np.tan(azi_b1*np.pi/180.)*0.8/0.05
    # plot
    ax1.plot(xlat[slice_600], wind[slice_600], label='600 RPM')


    # wrapping up the figure
    ax1.set_xlabel('Lateral distance $X/R_{tow}$ from tower centre [-]')
    ax1.set_ylabel('Axial Wind Speed [m/s]')
    ax1.legend(loc='best')
    ax1.set_xlim([-6, 6])
    ax1.grid(True)
    pa4.save_fig()

def tower_shadow_profile_2():
    """
    Compare small blade with different tower shadow cd's and rpm's
    """

    def plotcase(resfile, ax, label, bladeradius, mark=False):
        respath = 'simulations/hawc2/'
        res = HawcPy.LoadResults(respath, resfile)
        # find the lowest tower shadow passage event
        azi_b1 = res.sig[:,Chi.azi_b1]
        wind = res.sig[:,Chi.windy_r55]
        starti = np.argmin(wind)
        end = 50
        s_ = np.r_[starti-end:starti+end]
        # convert azimuth angle to lateral positions
        xlat = np.tan(azi_b1*np.pi/180.)*bladeradius/0.05
        # plot
        if mark:
            ax.plot(xlat[s_], wind[s_], mark, label=label)
        else:
            ax.plot(xlat[s_], wind[s_], label=label)

        return ax

    radiusxl = 8.-0.245
    radiusm = 1.3+0.245

    # -----------------------------------------------------------------------
    # setup the plot
    figpath = 'simulations/fig/ts/'
    figfile = 'tower-shadow-comparison-a2-a3-cd0.3'
    pa4 = plotting.A4Tuned()
    title = 'Wind speed 10 m/s, tower radius 5cm, cd 0.3'
    pa4.setup(figpath+figfile, grandtitle=title, figsize_x=16, figsize_y=16,
              wsleft_cm=2., wsright_cm=1.5, hspace=0.3)
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    # TSR                      0.8         2.5          5.0
    #iter_dict['[fix_wr]'] = [10.47,      31.41,       62.83]
    #iter_dict['[fix_wr]'] = [ 1.44538395, 4.33615186,  8.67368421]

    # case TSR=0.8
    rf = 's0_a2_yfix_10ms_y0_blade_x10.dat_ts4_cd0.3_14rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'XL, TSR=0.8', radiusxl)
    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.3_100rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'TSR=0.8', 0.8)

    # case TSR=2.5
    rf = 's0_a2_yfix_10ms_y0_blade_x10.dat_ts4_cd0.3_41rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'XL, TSR=2.5', radiusxl)
    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.3_300rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'TSR=2.5', 0.8)

    # case TSR 5
    rf = 's0_a2_yfix_10ms_y0_blade_x10.dat_ts4_cd0.3_83rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'XL, TSR=5.0', radiusxl)
    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.3_600rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'TSR=5.0', 0.8)

    # wrapping up the figure
    ax.set_xlabel('Lateral distance $X/R_{tow}$ from tower centre [-]')
    ax.set_ylabel('Axial Wind Speed [m/s]')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.set_xlim([-3, 3])
    ax.grid(True)
    pa4.save_fig()
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # setup the plot
    figpath = 'simulations/fig/ts/'
    figfile = 'tower-shadow-comparison-a2-a3-cd0.5'
    pa4 = plotting.A4Tuned()
    title = 'Wind speed 10 m/s, tower radius 5cm, cd 0.5'
    pa4.setup(figpath+figfile, grandtitle=title, figsize_x=16, figsize_y=16,
              wsleft_cm=2., wsright_cm=1.5, hspace=0.3)
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)
    rf = 's0_a2_yfix_10ms_y0_blade_x10.dat_ts4_cd0.5_14rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'XL, TSR=0.8', radiusxl)
    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.5_100rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'TSR=0.8', 0.8)

    rf = 's0_a2_yfix_10ms_y0_blade_x10.dat_ts4_cd0.5_41rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'XL, TSR=2.5', radiusxl)
    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.5_300rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'TSR=2.5', 0.8)

    rf = 's0_a2_yfix_10ms_y0_blade_x10.dat_ts4_cd0.5_83rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax,  'XL, TSR=5.0', radiusxl)
    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.5_600rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, 'TSR=5.0', 0.8)

    ax.set_xlabel('Lateral distance $X/R_{tow}$ from tower centre [-]')
    ax.set_ylabel('Axial Wind Speed [m/s]')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.set_xlim([-3, 3])
    ax.grid(True)
    pa4.save_fig()
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # setup the plot
    figpath = 'simulations/fig/ts/'
    figfile = 'tower-shadow-comparison-a2-a3-a4-cd0.5'
    pa4 = plotting.A4Tuned()
    title = 'Wind speed 10 m/s, tower radius 5cm, cd 0.5'
    pa4.setup(figpath+figfile, grandtitle=title, figsize_x=16, figsize_y=16,
              wsleft_cm=2., wsright_cm=1.5, hspace=0.3)
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, 1)

    rf = 's0_a3_yfix_10ms_y0_blade.dat_ts4_cd0.5_100rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, '0.55m', 0.8, mark='r^-')
    #rf = 's0_a6_yfix_10ms_y0_blade.dat_ts4_cd0.5_100rpm_c0_p0_sb51_ab00_st1'
    #ax = plotcase(rf, ax, 'normal2, rpm=100')
    #rf = 's0_a4_yfix_10ms_y0_blade_m.dat_ts4_cd0.5_100rpm_c0_p0_sb51_ab00_st1'
    #ax = plotcase(rf, ax, 'ext1, rpm=100', mark='r-')
    rf = 's0_a5_yfix_10ms_y0_blade_m.dat_ts4_cd0.5_100rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, '1.3m', radiusm, mark='ks--')
    rf='s0_a6_yfix_10ms_y0_blade_x10.dat_ts4_cd0.5_100rpm_c0_p0_sb51_ab00_st1'
    ax = plotcase(rf, ax, '5.5m', radiusxl, mark='g-*')

    ax.set_xlabel('Lateral distance $X/R_{tow}$ from tower centre [-]')
    ax.set_ylabel('Axial Wind Speed [m/s]')
    ax.set_title('all at 100 RPM')
    leg = ax.legend(loc='best', title='blade length')
    leg.get_frame().set_alpha(0.5)
    ax.set_xlim([-3, 3])
    ax.grid(True)
    pa4.save_fig()
    # -----------------------------------------------------------------------

def compare_rot_dirs(simid):
    """
    Compare standard and OJF rotation directions on the OJF model
    """

    respath = 'simulations/hawc2/' + simid + '/results/'

    resfile = simid+'_4.0ms_s0_y0_yfix_rot_ojf_blade.dat_ts0_cd0.5_'
    resfile += '300rpm_c0_p0_sb51_ab00_st1'
    res_ojf = HawcPy.LoadResults(respath, resfile)

    resfile = simid+'_4.0ms_s0_y0_yfix_rot_std_blade.dat_ts0_cd0.5_'
    resfile += '300rpm_c0_p0_sb51_ab00_st1'
    res_std = HawcPy.LoadResults(respath, resfile)

    # -----------------------------------------------------------------------
    # channels
    omega = 1
    aepower = 3
    azi1 = 5
    aoa22 = 16
    aoa49 = 19
    cl22 = 22
    cl49 = 25
    tipx = 28
    tipy = 29
    tipz = 30
    tiprotx = 33
    tiproty = 34
    tiprotz = 35
    mxroot = 36
    myroot = 37
    mzroot = 38

    # -------------------------------------------rs('b3')
    #compare_rot_dirs('b4')
    #compare_rot_dir----------------------------
    # setup the plot
    figpath = 'simulations/fig/rot_direction/'
    figfile = 'rotation-comparison-'+simid+'-nots-nosweep-stiff'
    pa4 = plotting.A4Tuned()
    title = 'Wind speed 4 m/s @ 300 rpm'
    pa4.setup(figpath+figfile, grandtitle=title, nr_plots=10,
              wsleft_cm=2., wsright_cm=1.5, hspace=0.3)

    tt = len(res_ojf.sig)/2

    plotnr = 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Aerodynamic power')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,aepower], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,aepower], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Blade root flapwise bending moment')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,mxroot], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,mxroot], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Blade tip deflections')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,tipx], 'b', label='ojf x')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,tipx], 'g', label='std x')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,tipy], 'b--', label='ojf y')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,tipy], 'g--', label='std y')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Blade tip rotations')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,tiprotx], 'b', label='ojf x')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,tiprotx], 'g', label='std x')
    #ax.plot(res_ojf.sig[:,0], res_ojf.sig[:,tiproty], 'b--', label='ojf y')
    #ax.plot(res_std.sig[:,0], res_std.sig[:,tiproty], 'g--', label='std y')
    #ax.plot(res_ojf.sig[:,0], res_ojf.sig[:,tiprotz], 'b-.', label='ojf z')
    #ax.plot(res_std.sig[:,0], res_std.sig[:,tiprotz], 'g-.', label='std z')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Blade tip rotations')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,tiproty], 'b--', label='ojf y')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,tiproty], 'g--', label='std y')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    #plotnr += 1
    #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    #ax.set_title('Blade tip rotations')
    #ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,tiprotz], 'b-.',label='ojf z')
    #ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,tiprotz], 'g-.',label='std z')
    #leg = ax.legend(loc='best')
    #leg.get_frame().set_alpha(0.5)
    #ax.grid(True)

    #plotnr += 1
    #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    #ax.set_title('Azimuth blade 1')
    #ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,azi1], 'b', label='ojf')
    #ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,azi1], 'g', label='std')
    #leg = ax.legend(loc='best')
    #leg.get_frame().set_alpha(0.5)
    #ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Rotor speed')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,omega], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,omega], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    #ax.set_ylim([31, 32])
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Angle of attack at r=0.22m')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,aoa22], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,aoa22], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('Angle of attack at r=0.49m')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,aoa49], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,aoa49], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('lift coefficient at r=0.22m')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,cl22], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,cl22], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    plotnr += 1
    ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
    ax.set_title('lift coefficient at r=0.49m')
    ax.plot(res_ojf.sig[tt:,0], res_ojf.sig[tt:,cl49], 'b', label='ojf')
    ax.plot(res_std.sig[tt:,0], res_std.sig[tt:,cl49], 'g', label='std')
    leg = ax.legend(loc='best')
    leg.get_frame().set_alpha(0.5)
    ax.grid(True)

    pa4.save_fig()
    # -----------------------------------------------------------------------


class DashBoard:
    """
    Print standard HAWC2 output parameters
    ======================================

    For easy comparison with OJF test results
    """


    def __init__(self, sim_id, post_dir, figpath):
        """
        """
        htc_dict = sim.get_htc_dict(post_dir, sim_id)
        # and plot for each case the dashboard
        for k in htc_dict:
            if htc_dict[k]['[run_dir]'][:4] == '/mnt':
                htc_dict[k]['[run_dir]'] = 'simulations/hawc2/%s/' + sim_id
            basepath = htc_dict[k]['[run_dir]']
            respath = htc_dict[k]['[res_dir]']
            resfile = htc_dict[k]['[case_id]']
            self.htc_dict_case = htc_dict[k]
            self.read(basepath+respath, resfile)

            self.casedict = htc_dict[k]

            # ignore first second
            jj = int(1./self.casedict['[dt_sim]'])
            sl = np.r_[jj:len(self.time)]
            self.plot(figpath, resfile, resfile, sl)

            # take 4 revolutions at the end
            rpm = self.sig[:,ChiT.omega].mean()*30./np.pi
            Fs = self.res.Freq
            data_window = 4*int(Fs/(rpm/60.))
            sl = np.r_[len(self.time)-data_window:len(self.time)]
            self.plot(figpath, resfile+'_zoom', resfile, sl)

    def read(self, respath, resfile, _slice=False):
        """
        Set the correct HAWC2 channels
        """
        self.respath = respath
        self.resfile = resfile
        self.res = HawcPy.LoadResults(respath, resfile)
        if not _slice:
            _slice = np.r_[0:len(self.res.sig)]
        self.time = self.res.sig[_slice,0]
        self.sig = self.res.sig[_slice,:]

    def plot(self, figpath, figfile, title, sl):
        """
        RPM        yaw angle
        FA moment  SS moment
        blade1     blade2
        Ae power/cp   AoA
        """

        # --------------------------------------------------------------------
        rho = 1.225
        A = ((0.245+0.555)*(0.245+0.555)*np.pi) - (0.245*0.245*np.pi)

        # mean windspeed can only be used if the ramping up in HAWC2 init
        # is ignored
        #V = np.mean(self.sig[sl,ChiT.wind])
        V = float(self.casedict['[windspeed]'])

        factor_p = 0.5*A*rho*V*V*V
        factor_t = 0.5*A*rho*V*V
        # aero power units are in kW
        cp = 1000.*self.sig[sl,ChiT.aepower]/factor_p
        ct = 1000.*self.sig[sl,ChiT.aethrust]/factor_t
        mechpower = -self.sig[sl,ChiT.mzshaft]*1000*self.sig[sl,ChiT.omega]

        ## moving average for the mechanical power
        #filt = ojf.Filters()
        ## take av2s window, calculate the number of samples per window
        #ws = 0.1/self.casedict['[dt_sim]']
        #mechpower_avg = filt.smooth(mechpower, window_len=ws, window='hanning')
        #N = len(mechpower_avg) - len(self.sig[sl,0])
        #mechpower_avg = mechpower_avg[N:]


        # --------------------------------------------------------------------
        pa4 = plotting.A4Tuned()
        pa4.setup(figpath+figfile, grandtitle=title, nr_plots=10,
                  wsleft_cm=2., wsright_cm=1.5, hspace_cm=1., wspace_cm=2.1)

        plotnr = 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Omega [RPM]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.omega]*30./np.pi, 'b')
        #leg = ax.legend(loc='best')
        #leg.get_frame().set_alpha(0.5)
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Yaw Angle (blue), normalised yaw control (red) [deg]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.yawangle], 'b')
        if self.htc_dict_case['[yawmode]'] == 'control_ini':

            # normalise wrt set max/min control output
            qq=self.sig[sl,ChiT.dll_control]/self.htc_dict_case['[yaw_c_max]']
            # and centre around the yaw ref angle +-20 degrees
            qq = (qq*20.) + self.sig[sl,ChiT.yaw_ref_angle]
            # print the boundaries of the yaw control box
            bound_up = self.htc_dict_case['[yaw_c_ref_angle]'] + 20
            ax.axhline(y=bound_up, linewidth=1, color='k',\
                linestyle='-', aa=False)
            bound_low = self.htc_dict_case['[yaw_c_ref_angle]'] - 20
            ax.axhline(y=bound_low, linewidth=1, color='k',\
                linestyle='-', aa=False)

            ## for the yaw control: normalise wrt max value and center around
            ## the reference yaw angle
            #ci = ChiT.yawangle
            #maxrange = self.sig[sl,ci].max() - self.sig[sl,ci].min()
            #qq = self.sig[sl,ChiT.dll_control]*1000.
            #qq = (qq/(qq.max()-qq.min()))*maxrange
            ## centre around ref angle instead of zero
            #qq += self.sig[sl,ChiT.yaw_ref_angle]

            ax.plot(self.time[sl], qq, 'r')
            ax.plot(self.time[sl], self.sig[sl,ChiT.yaw_ref_angle], 'g')

        else:
            pass
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])
        ax.grid(True)

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Tower base FA (blue yawing, red non yawing) [Nm]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.mxtower]*1000., 'b')
        ax.plot(self.time[sl], self.sig[sl,ChiT.mxtower_glo]*1000., 'r')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        #plotnr += 1
        #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #ax.set_title('Tower base SS (blue yawing, red non yawing) [Nm]')
        #ax.plot(self.time[sl], self.sig[sl,ChiT.mytower]*1000., 'b')
        #ax.plot(self.time[sl], self.sig[sl,ChiT.mytower_glo]*1000., 'r')
        #ax.grid(True)

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        title = 'Aero thrust [N] (blue), Thrust coefficient [%] (red)'
        #title += 'and Mech Thrust shaft [N] (black)'
        ax.set_title(title)
        ax.plot(self.time[sl], self.sig[sl,ChiT.aethrust]*1000., 'b')
        ax.plot(self.time[sl], ct*100, 'r')
        #ax.plot(self.time[sl], self.sig[sl,ChiT.fzshaft]*1000., 'k')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Blade 1 flapwise loads [Nm]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.mxroot]*1000., 'b')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #ax.set_title('Aero Power [W] (blue) and Power Coefficient [%] (red)')
        ax.set_title('Aero Power [W] (blue) Mech Power shaft [W] (red)')
        ax.plot(self.time[sl], self.sig[sl,ChiT.aepower]*1000., 'b')
        #ax.plot(self.time[sl], cp*100, 'r')
        ax.plot(self.time[sl], mechpower, 'r')
        #ax.plot(self.time[sl], mechpower_avg, 'r')
        # betz limit
        #ax.axhline(y=100*16./27.,linewidth=1,color='k',linestyle='-',aa=False)
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Blade tip deflection flapwise [m]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.tipy], 'b')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Blade tip deflection edgewise [m]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.tipx], 'b')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Angle of attack near the root [deg]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.aoa28], 'b')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Angle of attack near the tip [deg]')
        ax.plot(self.time[sl], self.sig[sl,ChiT.aoa49], 'b')
        ax.grid(True)
        ax.set_xlim([self.time[sl].min(), self.time[sl].max()])

        pa4.save_fig()

# TODO: move this to ojf_vs_hawc2
class HAWC2andOJF:
    """
    Combine OJF and HAWC2 datasets in one plot
    """

    def __init__(self, respath, resfile, caldict_dspace, caldict_blade,
                 simid, case):
        """
        Parameters
        ----------

        respath : str
            path holding the OJF result files

        resfile : str
            OJF result file name (excluding extension)

        caldict_dspace : dict
            Dictionary holding the paths to the calibration files. Valid keys
            are the channel names as defined in dspace.labels_ch

        caldict_blade : dict
            Dictionary holding the paths to the calibration files. Keys are
            the channel numbers: 0-1 for blade 2, 2-3 for blade 1.

        simid : str
            string identifying the simid name

        case : str
            case name string (includes the .htc extension currently)

        """

        self.load_ojf(respath, resfile, caldict_dspace, caldict_blade)
        self.load_hawc2(simid, case)
        self.Mxb1_equiv_r, self.Mxb1_equiv_30 = self._m_equiv_blades()


    def load_ojf(self, respath, resfile, caldict_dspace, caldict_blade):
        # load the OJF data files
        self.ojfres = ojfresult.ComboResults(respath, resfile)
        self.ojfres._sync_strain_dspace(min_h=0.3, checkplot=False)
        self.ojfres._calibrate_dspace(caldict_dspace, rem_wind=True)
        self.ojfres._calibrate_blade(caldict_blade)

    def load_hawc2(self, simid, case):
        # htc_dict
        htc_dict = sim.load_pickled_file(post_dir + simid + '.pkl')

        htc_dict_sel = dict()
        htc_dict_sel[case] = htc_dict[case]
        if htc_dict_sel[case]['[run_dir]'][:4] == '/mnt':
            htc_dict_sel[case]['[run_dir]'] = 'simulations/hawc2/%s/' + simid
        self.casedict = htc_dict_sel[case]

        basepath = htc_dict[case]['[run_dir]']
        respath = htc_dict[case]['[res_dir]']
        resfile = htc_dict[case]['[case_id]']
        print '\n' + '*'*80
        print 'LOADING HAWC2 FILE'
        self.hawcres = HawcPy.LoadResults(basepath+respath, resfile)
        # and keep a class reference to the case dictionary directly
        self.cased = htc_dict[case]

    def _m_equiv_blades(self):
        """
        Convert the HAWC2 blade root bending moment the equivalent bending
        moment as measured by the OJF strain gauges

        Note that Mxequiv is now for blade1
        """

        # load the st file
        md = sim.ModelData(silent=True)
        stpath = self.cased['[model_dir_local]'] + self.cased['[data_dir]']
        stfile = self.cased['[st_file]']
        md.load_st(stpath, stfile)
        setnr = self.cased['[st_blade_set]']
        subsetnr = self.cased['[st_blade1_subset]']
        st_arr = md.st_dict['%03i-%03i-d' % (setnr, subsetnr)].copy()

        if self.cased['[blade_st_group]'] == 'stiff':
            filename = 'strainpos-41-beam.txt'
        elif self.cased['[blade_st_group]'] == 'flex':
            filename = 'strainpos-42-beam.txt'
        # load the positions of the strain gauges (c/2 coordinates)
        strainpos = np.loadtxt(self.cased['[blade_hawtopt_dir]'] + filename)

        y = st_arr
        # we have only 2 strain gauges: at 0.02 and 0.18 meter
        # interpolate cross sectional data to strain gauge position
        # and flattend with ravel to 1D, otherwise shape will be (1,19)
        st_arr_r  = sp.interpolate.griddata(st_arr[:,sti.r],y,[0.02]).ravel()
        st_arr_30 = sp.interpolate.griddata(st_arr[:,sti.r],y,[0.18]).ravel()
        # and also the strain gauge positions
        pos_r  = sp.interpolate.griddata(strainpos[:,0],strainpos,[0.02])[0,1:]
        pos_30 = sp.interpolate.griddata(strainpos[:,0],strainpos,[0.18])[0,1:]

        # and the required loads from the HAWC2 simulation
        mxroot = self.hawcres.sig[:,ChiT.mxroot]
        myroot = self.hawcres.sig[:,ChiT.myroot]
        fzroot = self.hawcres.sig[:,ChiT.fzroot]
        # load = [Fx, Fy, Fz, Mx, My, Mz]
        load_r = [0, 0, fzroot, mxroot, myroot, 0]

        # and the required loads from the HAWC2 simulation
        mxroot = self.hawcres.sig[:,ChiT.mx30]
        myroot = self.hawcres.sig[:,ChiT.my30]
        fzroot = self.hawcres.sig[:,ChiT.fz30]
        # load = [Fx, Fy, Fz, Mx, My, Mz]
        load_30 = [0, 0, fzroot, mxroot, myroot, 0]

        Mxb1_equiv_r = sim.Results().m_equiv(st_arr_r, load_r, pos_r)
        Mxb1_equiv_30 = sim.Results().m_equiv(st_arr_30, load_30, pos_30)

        return Mxb1_equiv_r, Mxb1_equiv_30


    def data_selection(self):
        """
        Select that correct data ranges to compare. Has to be based on input.
        Events from the OJF experiment need to overlap with the HAWC2
        simulations
        """
        self.sl = False

    def plot_yawrpm(self, figpath, figfile, title, ds0, hawc0, interval):
        """
        Only plot and compare yaw and rpm's

        Parameters
        ----------

        figpath : str

        figfile : str

        title : str
            Grand title of the plot

        ds0 : float
            starting point in seconds of the dspace result set

        hawc0 : float
            starting point in seconds of the HAWC2 simulations

        interval : ndarray(2)
            array holding the range that will be plotten, starting from ds0 or
            hawc0
        """
        print '\n' + '*'*80
        print 'START PLOTTING YAWRPM'

        # =====================================================================
        # TIME SLICING
        # =====================================================================
        # for the OJF results
        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self.ojfres._data_window(time=ds0+interval)
        # for the HAWC2 results
        slice_, window, zt, tr = self.hawcres._data_window(time=hawc0+interval)
        # time range of dspace and hawc2 need to be the same
        if not np.allclose(tr, time_range):
            raise ValueError, 'time range of OJF and HAWC2 have to be the same'

        # and for convience, shorter notations for the sliced data selection
        dspace_t = self.ojfres.dspace.time[slice_dspace] - ds0
        dspace_d = self.ojfres.dspace.data[slice_dspace,:]
        #blade_t = self.ojfres.blade.time[slice_blade] - ds0
        #blade_d = self.ojfres.blade.data[slice_blade,:]
        self.hawct = self.hawcres.sig[slice_,0] - hawc0 - self.casedict['[t0]']
        self.hawcd = self.hawcres.sig[slice_,:]

        time_range = interval

        # ---------------------------------------------------------------------
        #rho = 1.225
        #A = ((0.245+0.555)*(0.245+0.555)*np.pi) - (0.245*0.245*np.pi)
        #
        ##V = np.mean(self.hawcd[:,ChiT.wind])
        #V = float(self.casedict['[windspeed]'])
        #
        #factor = 0.5*A*rho*V*V*V
        ## aero power units are in kW
        #cp = 1000.*self.hawcd[:,ChiT.aepower]/factor
        #mechpower = -self.hawcd[:,ChiT.mzshaft]*1000*self.hawcd[:,ChiT.omega]

        ## moving average for the mechanical power
        #filt = ojf.Filters()
        ## take av2s window, calculate the number of samples per window
        #ws = 0.1/self.casedict['[dt_sim]']
        #mechpower_avg = filt.smooth(mechpower, window_len=ws, window='hanning')
        #N = len(mechpower_avg) - len(self.hawcd[:,0])
        #mechpower_avg = mechpower_avg[N:]

        # ---------------------------------------------------------------------

        # =====================================================================
        # Channel index selection
        # =====================================================================
        ds_rpm = self.ojfres.dspace.labels_ch['RPM']
        ds_yaw = self.ojfres.dspace.labels_ch['Yaw Laser']

        # =====================================================================
        # SETTING UP FIGURE
        # =====================================================================
        # add the dspace time stamp too, remember that LaTeX doesn't like
        # dots in the file name, except for the extension
        addendum = '_yawrpm'
        addendum += '_%s-%ss' % (interval[0], interval[1])
        pa4 = plotting.A4Tuned()
        pa4.setup(figpath+figfile+addendum, grandtitle=title, nr_plots=2,
                  wsleft_cm=1.5, wsright_cm=1.5, hspace_cm=2.0, wspace_cm=0.0,
                  wstop_cm=2.5)

        #sl = self.sl

        oj = 'r-'
        ha = 'b--'

        # =====================================================================
        # COMPARE RPM's
        # =====================================================================
        plotnr = 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Rotor speed [RPM]')
        rpm = self.hawcd[:,ChiT.omega]*30./np.pi
        ax.plot(self.hawct, rpm, ha,label='HAWC2')
        ax.plot(dspace_t, dspace_d[:,ds_rpm], oj, label='OJF')
        ax.set_xlim(time_range)
        leg = ax.legend(loc='best')
        leg.get_frame().set_alpha(0.5)
        ax.grid(True)
        # =====================================================================
        # YAW ANGLES
        # =====================================================================
        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Yaw angle [deg]')
        ax.plot(self.hawct, self.hawcd[:,ChiT.yawangle], ha, label='HAWC2')
        ax.plot(dspace_t, dspace_d[:,ds_yaw], oj, label='OJF')
        ax.set_xlim(time_range)
        leg = ax.legend(loc='best')
        leg.get_frame().set_alpha(0.5)
        ax.set_xlabel('Time')
        ax.grid(True)


        pa4.save_fig()



    def dashboard(self, figpath, figfile, title, ds0, hawc0, interval):
        """
        Parameters
        ----------

        figpath : str

        figfile : str

        title : str
            Grand title of the plot

        ds0 : float
            starting point in seconds of the dspace result set

        hawc0 : float
            starting point in seconds of the HAWC2 simulations

        interval : ndarray(2)
            array holding the range that will be plotten, starting from ds0 or
            hawc0

        """

        print '\n' + '*'*80
        print 'START PLOTTING DASHBOARD'

        # convert title to raw string
        #title = plotting.raw_string(title)

        # =====================================================================
        # TIME SLICING
        # =====================================================================
        # for the OJF results
        slice_dspace, slice_ojf, slice_blade, window_dspace, \
                window_ojf, window_blade, zoomtype, time_range \
                = self.ojfres._data_window(time=ds0+interval)
        # for the HAWC2 results
        slice_, window, zt, tr = self.hawcres._data_window(time=hawc0+interval)
        # time range of dspace and hawc2 need to be the same
        if not np.allclose(tr, time_range):
            raise ValueError, 'time range of OJF and HAWC2 have to be the same'

        # and for convience, shorter notations for the sliced data selection
        dspace_t = self.ojfres.dspace.time[slice_dspace] - ds0
        dspace_d = self.ojfres.dspace.data[slice_dspace,:]
        blade_t = self.ojfres.blade.time[slice_blade] - ds0
        blade_d = self.ojfres.blade.data[slice_blade,:]
        self.hawct = self.hawcres.sig[slice_,0] - hawc0 - self.casedict['[t0]']
        self.hawcd = self.hawcres.sig[slice_,:]
        Mxb1_equiv_r_sl = self.Mxb1_equiv_r[slice_]
        Mxb1_equiv_30_sl = self.Mxb1_equiv_30[slice_]

        time_range = interval

        # ---------------------------------------------------------------------
        #rho = 1.225
        #A = ((0.245+0.555)*(0.245+0.555)*np.pi) - (0.245*0.245*np.pi)
        ##V = np.mean(self.hawcd[:,ChiT.wind])
        #V = float(self.casedict['[windspeed]'])
        #factor = 0.5*A*rho*V*V*V
        ## aero power units are in kW
        #cp = 1000.*self.hawcd[:,ChiT.aepower]/factor

        mechpower = -self.hawcd[:,ChiT.mzshaft]*1000*self.hawcd[:,ChiT.omega]

        ## moving average for the mechanical power
        #filt = ojf.Filters()
        ## take av2s window, calculate the number of samples per window
        #ws = 0.1/self.casedict['[dt_sim]']
        #mechpower_avg = filt.smooth(mechpower, window_len=ws,window='hanning')
        #N = len(mechpower_avg) - len(self.hawcd[:,0])
        #mechpower_avg = mechpower_avg[N:]

        # ---------------------------------------------------------------------

        # =====================================================================
        # Channel index selection
        # =====================================================================
        ds_rpm = self.ojfres.dspace.labels_ch['RPM']
        ds_yaw = self.ojfres.dspace.labels_ch['Yaw Laser']
        #self.res.dspace.labels_ch['Power']
        ds_tw_fa=self.ojfres.dspace.labels_ch['Tower Strain For-Aft']
        #=self.ojfres.dspace.labels_ch['Tower Strain Side-Side filtered']
        ds_power = self.ojfres.dspace.labels_ch['Power']
        #ds_power2 = self.ojfres.dspace.labels_ch['Power2']
        #for qq in self.ojfres.dspace.labels:
            #print qq

        bl1_ro = 2
        bl1_30 = 3
        bl2_ro = 0
        bl2_30 = 1

        # =====================================================================
        # SETTING UP FIGURE
        # =====================================================================
        # add the dspace time stamp too, remember that LaTeX doesn't like
        # dots in the file name, except for the extension
        addendum = '_%s-%ss' % (interval[0], interval[1])
        pa4 = plotting.A4Tuned()
        pa4.setup(figpath+figfile+addendum, grandtitle=title, nr_plots=6,
                  wsleft_cm=2., wsright_cm=1.5, hspace_cm=2.5, wspace_cm=4.0)
        #sl = self.sl

        oj = 'r'
        ha = 'b'

        # =====================================================================
        # COMPARE RPM's
        # =====================================================================
        plotnr = 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Rotor speed [RPM] (blue HAWC2, red OJF)')
        ax.plot(self.hawct, self.hawcd[:,ChiT.omega]*30./np.pi, ha)
        ax.plot(dspace_t, dspace_d[:,ds_rpm], oj, label='Yaw error')
        ax.set_xlim(time_range)
        #leg = ax.legend(loc='best')
        #leg.get_frame().set_alpha(0.5)
        ax.grid(True)
        # =====================================================================
        # YAW ANGLES
        # =====================================================================
        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Yaw angle [deg] (blue HAWC2, red OJF)')
        ax.plot(self.hawct, self.hawcd[:,ChiT.yawangle], ha)
        ax.plot(dspace_t, dspace_d[:,ds_yaw], oj, label='Yaw error')
        ax.set_xlim(time_range)
        #leg = ax.legend(loc='best')
        #leg.get_frame().set_alpha(0.5)
        ax.grid(True)
        # =====================================================================
        # THERE IS SOMETHING GOING ON WITH BLADE 1
        # strains at 30% are higher than at the root. WHY??
        # =====================================================================
        # =====================================================================
        # BLADE 1 ROOT
        # =====================================================================
        #plotnr += 1
        #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #ax.set_title('Blade 1 root flap bending [Nm] (blue HAWC2, red OJF)')
        ##ax.plot(self.hawct, self.hawcd[:,ChiT.mxroot]*1000., ha)
        ## normalise wrt max signal, just as is done with blade strain
        ##norm=self.hawcd[:,ChiT.mxroot].max()-self.hawcd[:,ChiT.mxroot].min()
        #ax.plot(self.hawct, self.hawcd[:,ChiT.mxroot]*1000., 'k')
        #ax.plot(self.hawct, Mxb1_equiv_r_sl*1000., ha)
        ##norm = blade_d[:,bl1_ro].max() - blade_d[:,bl1_ro].min()
        #ax.plot(blade_t, blade_d[:,bl1_ro], oj, label='blade1 root')
        #ax.set_xlim(time_range)
        #ax.grid(True)
        ## =====================================================================
        ## BLADE 1 30%
        ## =====================================================================
        #plotnr += 1
        #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #ax.set_title('Blade 1 30\% flap bending [Nm] (blue HAWC2, red OJF)')
        ##ax.plot(self.hawct, self.hawcd[:,ChiT.mx30]*1000., ha)
        ## normalise wrt max signal, just as is done with blade strain
        ##norm = self.hawcd[:,ChiT.mx30].max() - self.hawcd[:,ChiT.mx30].min()
        #ax.plot(self.hawct, self.hawcd[:,ChiT.mx30]*1000., 'k')
        #ax.plot(self.hawct, Mxb1_equiv_30_sl*1000., ha)
        ## blade strain is already centered around 0 instead of 2048
        ##norm = blade_d[:,bl1_30].max() - blade_d[:,bl1_30].min()
        #ax.plot(blade_t, blade_d[:,bl1_30], oj, label='blade1 30\%')
        #ax.set_xlim(time_range)
        #ax.grid(True)
        # =====================================================================
        # BLADE 2 ROOT
        # =====================================================================
        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Blade 2 root flap bending [Nm] (blue HAWC2, red OJF)')
        #ax.plot(self.hawct, self.hawcd[:,ChiT.mxroot]*1000., ha)
        # normalise wrt max signal, just as is done with blade strain
        #norm = self.hawcd[:,ChiT.mxroot].max()-self.hawcd[:,ChiT.mxroot].min()
        ax.plot(self.hawct, self.hawcd[:,ChiT.mxroot]*1000., 'k')
        ax.plot(self.hawct, Mxb1_equiv_r_sl*1000., ha)
        #norm = blade_d[:,bl1_ro].max() - blade_d[:,bl1_ro].min()
        ax.plot(blade_t, blade_d[:,bl2_ro], oj, label='blade2 root')
        ax.set_xlim(time_range)
        ax.grid(True)
        # remember the ylimits for the next plot
        ylims = ax.get_ylim()
        # =====================================================================
        # BLADE 2 30%
        # =====================================================================
        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Blade 2 30\% flap bending [Nm] (blue HAWC2, red OJF)')
        #ax.plot(self.hawct, self.hawcd[:,ChiT.mx30]*1000., ha)
        # normalise wrt max signal, just as is done with blade strain
        #norm = self.hawcd[:,ChiT.mx30].max() - self.hawcd[:,ChiT.mx30].min()
        ax.plot(self.hawct, self.hawcd[:,ChiT.mx30]*1000., 'k')
        ax.plot(self.hawct, Mxb1_equiv_30_sl*1000., ha)
        # blade strain is already centered around 0 instead of 2048
        #norm = blade_d[:,bl1_30].max() - blade_d[:,bl1_30].min()
        ax.plot(blade_t, blade_d[:,bl2_30], oj, label='blade2 30\%')
        ax.set_xlim(time_range)
        # use the same ylimits as the root strain plot
        ax.set_ylim(ylims)
        ax.grid(True)
        # =====================================================================
        # TOWER BASE FA
        # =====================================================================
        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #ax.set_title('Tower base FA (blue HAWC2, red OJF) [Nm]')
        subtitle = 'Tower base FA bending moment [Nm] (blue HAWC2, red OJF)'
        subtitle += '\nyawing frame of reference'
        ax.set_title(subtitle)
        ax.plot(self.hawct, self.hawcd[:,ChiT.mxtower]*1000., ha)
        #ax.plot(self.hawct, self.hawcd[:,ChiT.aethrust]*1000., ha)
        ax.plot(dspace_t, dspace_d[:,ds_tw_fa], oj, label='FA tower')
        ax.set_xlim(time_range)
        ax.grid(True)
        # =====================================================================
        # TOWER BASE SS
        # =====================================================================
        #plotnr += 1
        #ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #ax.set_title('Tower base SS (blue HAWC2, red OJF) [Nm]')
        #ax.plot(self.hawct, self.hawcd[:,ChiT.mytower]*1000., ha)
        #ax.plot(dspace_t, dspace_d[:,ds_tw_ss], oj, label='SS tower')
        #ax.grid(True)
        # =====================================================================
        # CP, AERO AND ELECTRICAL POWER
        # =====================================================================
        plotnr += 1
        ax = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        ax.set_title('Power Coefficient, based on aero power [-]')
        #tt = 'Cp [-] (green), Aero Power [W] (blue), Elec P OJF [W] (red)'
        tt = 'Mech Power shaft [W] (blue), Elec P OJF [W] (red)'
        ax.set_title(tt)
        #ax.plot(self.hawct, cp, 'g')
        # plot Betz limit
        #ax.axhline(y=16./27., linewidth=1, color='k',\
                #linestyle='-', aa=False)
        ax.grid(True)
        ax.set_xlim(time_range)

        #ax = ax.twinx()
        ax.plot(self.hawct, mechpower, ha)
        #ax.plot(self.hawct, mechpower_avg, ha)
        ax.plot(dspace_t, dspace_d[:,ds_power], oj, label='P elec')
        #ax.plot(dspace_t, dspace_d[:,ds_power2], 'k', label='P elec')
        ax.set_xlim(time_range)
        ax.grid(True)

        pa4.save_fig()

class compare_HAWC2:
    """
    Compare a set of HAWC2 simulations in the same plot
    """

    def __init__(self, htc_dict, post_dir, figpath, figfile, title):
        """
        """

        colors = ['b', 'r', 'g']

        # init the plot
        self.init_plot(figpath, figfile, title)
        # and plot for each case the dashboard
        for i, k in enumerate(htc_dict):
            self.casedict = htc_dict[k]
            basepath = htc_dict[k]['[run_dir]']
            respath = htc_dict[k]['[res_dir]']
            resfile = htc_dict[k]['[case_id]']
            res = self.read(basepath+respath, resfile)
            # ignore first second
            jj = int(1./self.casedict['[dt_sim]'])
            sl = np.r_[jj:len(res.sig)]

            # labels and colors in the plots
            ll = htc_dict[k]['[plotlabel]']
            cc = colors[i]

            self.plot(res.sig[sl,0], res.sig[sl,:], cc, ll)
        self.finalize()

        # init the plot
        self.init_plot(figpath, figfile+'_zoom', title)
        # and plot for each case the dashboard
        for i, k in enumerate(htc_dict):
            basepath = htc_dict[k]['[run_dir]']
            respath = htc_dict[k]['[res_dir]']
            resfile = htc_dict[k]['[case_id]']
            res = self.read(basepath+respath, resfile)
            # take 4 revolutions at the end
            rpm = res.sig[:,ChiT.omega].mean()*30./np.pi
            Fs = res.Freq
            data_window = 4*int(Fs/(rpm/60.))
            sl = np.r_[len(res.sig)-data_window:len(res.sig)]

            # labels and colors in the plots
            ll = htc_dict[k]['[plotlabel]']
            cc = colors[i]
            self.casedict = htc_dict[k]

            self.plot(res.sig[sl,0], res.sig[sl,:], cc, ll)
        self.finalize()

    def read(self, respath, resfile, _slice=False):
        """
        Set the correct HAWC2 channels
        """
        respath = respath
        resfile = resfile
        res = HawcPy.LoadResults(respath, resfile)
        if not _slice:
            _slice = np.r_[0:len(res.sig)]
        res.sig = res.sig[_slice,:]
        return res

    def init_plot(self, figpath, figfile, title):
        """
        Create all the plot axis
        """
        # tlist = [[title, channeli], [title, channeli]]
        # nr plots 0 len(tlist)
        tlist = []
        tlist.append([])

        # --------------------------------------------------------------------
        pa4 = plotting.A4Tuned()
        pa4.setup(figpath+figfile, grandtitle=title, nr_plots=12,
                  wsleft_cm=2., wsright_cm=1.0, hspace_cm=1.7, wspace_cm=4.5)

        plotnr = 1
        self.ax1 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax1.set_title('Omega [RPM]')

        plotnr += 1
        self.ax2 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax2.set_title('Yaw Angle [deg]')

        plotnr += 1
        self.ax3 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax3.set_title('Tower base FA (yawing frame of reference) [Nm]')

        plotnr += 1
        self.ax4 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        #self.ax4.set_title('Tower base SS (yawing frame of reference) [Nm]')
        #self.ax4.set_title('Blade tip deflection z-dir (extending) [m]')
        self.ax4.set_title('Thrust coefficient [-] (based on aero Thrust)')

        plotnr += 1
        self.ax5 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax5.set_title('Blade 1 flapwise loads [Nm]')

        plotnr += 1
        self.ax6 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax6.set_title('Power [W] (shaft mechanical)')
        # plot Betz limit
        #self.ax6.axhline(y=16./27., linewidth=1, color='k',\
                #linestyle='-', aa=False)

        plotnr += 1
        self.ax7 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax7.set_title('Blade tip deflection flapwise [m]')

        plotnr += 1
        self.ax8 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax8.set_title('Blade tip deflection edgewise [m]')

        plotnr += 1
        self.ax9 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax9.set_title('Angle of attack near the root [deg]')

        plotnr += 1
        self.ax10 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax10.set_title('Angle of attack near the tip [deg]')

        plotnr += 1
        self.ax11 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax11.set_title('Relative velocity near the root [deg]')

        plotnr += 1
        self.ax12 = pa4.fig.add_subplot(pa4.nr_rows, pa4.nr_cols, plotnr)
        self.ax12.set_title('Relative velocity near the tip [deg]')

        self.pa4 = pa4

    def plot(self, time, sig, cc, ll):
        """
        RPM        yaw angle
        FA moment  SS moment
        blade1     blade2
        Ae power/cp   AoA
        """

        # --------------------------------------------------------------------
        rho = 1.225
        A = ((0.245+0.555)*(0.245+0.555)*np.pi) - (0.245*0.245*np.pi)

        # cautious with the wind speed! It can not include the wind ramp!
        #V = np.mean(sig[:,ChiT.wind])
        V = float(self.casedict['[windspeed]'])

        factor_p = 0.5*A*rho*V*V*V
        factor_t = 0.5*A*rho*V*V
        # aero power units are in kW
        cp = 1000.*sig[:,ChiT.aepower]/factor_p
        ct = 1000.*sig[:,ChiT.aethrust]/factor_t

        mechpower = -sig[:,ChiT.mzshaft]*1000*sig[:,ChiT.omega]

        ## moving average for the mechanical power
        #filt = ojf.Filters()
        ## take av2s window, calculate the number of samples per window
        #ws = 0.1/self.casedict['[dt_sim]']
        #mechpower_avg = filt.smooth(mechpower, window_len=ws, window='hanning')
        #N = len(mechpower_avg) - len(sig[:,0])
        #mechpower_avg = mechpower_avg[N:]

        # =====================================================================
        # Rotor speed
        # =====================================================================
        self.ax1.plot(time[:], sig[:,ChiT.omega]*30./np.pi, cc, label=ll)
        #leg = ax.legend(loc='best')
        #leg.get_frame().set_alpha(0.5)
        self.ax1.grid(True)
        self.ax1.set_xlim([time.min(), time.max()])

        # =====================================================================
        # yaw angle
        # =====================================================================
        self.ax2.plot(time[:], sig[:,ChiT.yawangle], cc, label=ll)
        #try:
            ## for the yaw control: normalise wrt max value and center around
            ## the reference yaw angle
            #ci = ChiT.yawangle
            #maxrange = sig[:,ci].max() - sig[:,ci].min()
            ##print maxrange
            #qq = sig[:,ChiT.dll_control]*1000.
            #qq = (qq/(qq.max()-qq.min()))*maxrange
            ## centre around ref angle instead of zero
            #qq += sig[:,ChiT.yaw_ref_angle]
            ## multiply with 10
            ##qq *= 10.
            #self.ax2.plot(time[:], qq, 'r')
            #self.ax2.plot(time[:], sig[:,ChiT.yaw_ref_angle], 'g')
        #except IndexError:
            #pass
        self.ax2.grid(True)
        self.ax2.set_xlim([time.min(), time.max()])

        # =====================================================================
        # Tower base FA
        # =====================================================================
        self.ax3.plot(time[:], sig[:,ChiT.mxtower]*1000., cc, label=ll)
        #self.ax3.plot(time[:], sig[:,ChiT.mxtower_glo]*1000., 'r')
        self.ax3.grid(True)
        self.ax3.set_xlim([time.min(), time.max()])

        # =====================================================================
        # TOWER BASE SS
        # =====================================================================
        #self.ax4.plot(time[:], sig[:,ChiT.tipz], cc, label=ll)
        ##self.ax4.plot(time[:], sig[:,ChiT.mytower]*1000., cc, label=ll)
        ##self.ax4.plot(time[:], sig[:,ChiT.mytower_glo]*1000., 'r')
        #self.ax4.grid(True)

        # =====================================================================
        # AERO THRUST COEFFICIENT
        # =====================================================================
        #ax4.plot(time, sig[:,ChiT.aethrust]*1000., cc, label=ll)
        self.ax4.plot(time, ct, cc, label=ll)
        self.ax4.grid(True)
        self.ax4.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE ROOT BENDING flapwise
        # =====================================================================
        self.ax5.plot(time[:], sig[:,ChiT.mxroot]*1000., cc, label=ll)
        self.ax5.grid(True)
        self.ax5.set_xlim([time.min(), time.max()])

        # =====================================================================
        # MECHANICAL POWER
        # =====================================================================
        self.ax6.plot(time[:], mechpower, cc, label=ll)
        self.ax6.grid(True)
        self.ax6.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE TIP flap deflection
        # =====================================================================
        self.ax7.plot(time[:], sig[:,ChiT.tipy], cc, label=ll)
        self.ax7.grid(True)
        self.ax7.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE TIP EDGE deflection
        # =====================================================================
        self.ax8.plot(time[:], sig[:,ChiT.tipx], cc, label=ll)
        self.ax8.grid(True)
        self.ax8.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE ROOT AOA
        # =====================================================================
        self.ax9.plot(time[:], sig[:,ChiT.aoa28], cc, label=ll)
        self.ax9.grid(True)
        self.ax9.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE TIP AOA
        # =====================================================================
        self.ax10.plot(time[:], sig[:,ChiT.aoa49], cc, label=ll)
        self.ax10.grid(True)
        self.ax10.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE root V_eff
        # =====================================================================
        self.ax11.plot(time[:], sig[:,ChiT.vrel28], cc, label=ll)
        self.ax11.grid(True)
        self.ax11.set_xlim([time.min(), time.max()])

        # =====================================================================
        # BLADE TIP V_eff
        # =====================================================================
        self.ax12.plot(time[:], sig[:,ChiT.vrel49], cc, label=ll)
        self.ax12.grid(True)
        self.ax12.set_xlim([time.min(), time.max()])

    def finalize(self):
        """
        """
        leg = self.ax1.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax2.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax3.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax4.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax5.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax6.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax7.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax8.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax9.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax10.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax11.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        leg = self.ax12.legend(loc='best')
        leg.get_frame().set_alpha(0.5)

        self.pa4.save_fig()

###############################################################################
### BLADE ONLY
###############################################################################

def launch_static_blade_deflection():
    """
    Seperate method to keep together all jobs launched for blade only
    """

    # BLade Only Static Deflection

    simid = 'blosd01' # static deflection for flex
    #simid = 'blosd02' # static deflection for stiff

    # =========================================================================
    # CREATE ALL HTC FILES AND PBS FILES TO LAUNCH ON THYRA/GORM
    # =========================================================================
    # see create_multiloop_list() docstring in Simulations.py
    iter_dict = dict()
    iter_dict['[st_blade_subset]'] = [stsets.b1_flex_opt,\
                                     stsets.b2_flex_opt, stsets.b3_flex_opt]
    iter_dict['[bladetipmass]'] = [0, 0.056, 0.156, 0.256, 0.306 ]
    #iter_dict['[st_blade_subset]'] = [stsets.b1_stiff_opt,\
                                     #stsets.b2_stiff_opt,stsets.b3_stiff_opt]
    #iter_dict['[bladetipmass]'] = [0, 0.106, 0.606, 0.706]

    opt_tags = []
    runmethod = 'local'
    #runmethod = 'gorm'
    #runmethod = 'none'

    master = master_tags(simid, runmethod=runmethod, turbulence=False,
                         verbose=False, silent=False)
    master.tags['[duration]'] = 12.0
    master.tags['[t0]'] = 10.0
    master.tags['[auto_set_sim_time]'] = False
    # master file implements t0=0, f0=0, f1=1
    master.tags['[t1_windramp]'] = 5
    master.tags['[windramp]'] = True
    master.tags['[dt_sim]'] = 0.01
    master.tags['[windspeed]'] = 0
    master.tags['[horizontal]'] = True
    master.tags['[vertical]'] = False
    master.tags['[induction_method]'] = 0
    master.tags['[aerocalc_method]'] = 0
    master.tags['[dynstall]'] = 2
    master.tags['[tiploss]'] = 1
    # 0=none, 1=potential flow, 2=jet, 4=jet_2
    master.tags['[tower_shadow]'] = 0
    master.tags['[master_htc_file]'] = 'ojf_post_master_bladeonly.htc'
    master.tags['[nr_bodies_blade]'] = 11
    master.tags['[conc_tipmass]'] = True
    #master.tags['[bladetipmass]'] = 0.306
    damp_x = float(model.b1_flex_damp.split('  ')[-3].strip())
    damp_y = float(model.b1_flex_damp.split('  ')[-2].strip())
    damp_z = float(model.b1_flex_damp.split('  ')[-1].strip())
    master.tags['[damp_blade_mx]'] = '%5.3e' % 0
    master.tags['[damp_blade_my]'] = '%5.3e' % 0
    master.tags['[damp_blade_mz]'] = '%5.3e' % 0
    master.tags['[damp_blade_kx]'] = '%5.3e' % damp_x
    master.tags['[damp_blade_ky]'] = '%5.3e' % damp_y
    master.tags['[damp_blade_kz]'] = '%5.3e' % damp_z

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master,variable_tag_func_bladeonly,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg=msg, silent=False, check_log=True)

def launch_blade_only_aero():
    """
    Simulate the blade only aero experiments

    aero_01: with tip loss, no drag on base
    aero_02: no tip loss, with drag on base
    """

    simid = 'aero_02'

    iter_dict = dict()
    iter_dict['[windspeed]'] = [15, 25]
    iter_dict['[pitch_angle]'] = range(-20,40,1)
    iter_dict['[st_blade_subset]'] = [stsets.b1_flex_opt, stsets.b2_flex_opt,
                                        stsets.b3_stiff_opt]

    opt_tags = []
    runmethod = 'local'
    #runmethod = 'gorm'
    #runmethod = 'none'

    master = master_tags(simid, runmethod=runmethod, turbulence=False,
                         verbose=False, silent=False)
    master.tags['[animation]'] = False
    master.tags['[duration]'] = 2.0
    master.tags['[t0]'] = 11.0
    master.tags['[auto_set_sim_time]'] = False
    master.tags['[dt_sim]'] = 0.01
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[windspeed]'] = 10
    master.tags['[horizontal]'] = False
    master.tags['[vertical]'] = True
    master.tags['[induction_method]'] = 0
    master.tags['[aerocalc_method]'] = 1
    master.tags['[dynstall]'] = 0
    master.tags['[tiploss]'] = 0
    # 0=none, 1=potential flow, 2=jet, 4=jet_2
    master.tags['[tower_shadow]'] = 0
    master.tags['[master_htc_file]'] = 'ojf_post_master_bladeonly.htc'
    #master.tags['[st_blade_subset]'] = stsets.b1_flex_opt
    master.tags['[nr_bodies_blade]'] = 11
    master.tags['[conc_tipmass]'] = False
    master.tags['[bladetipmass]'] = 0
    damp_x = float(model.b1_flex_damp.split('  ')[-3].strip())
    damp_y = float(model.b1_flex_damp.split('  ')[-2].strip())
    damp_z = float(model.b1_flex_damp.split('  ')[-1].strip())
    master.tags['[damp_blade_mx]'] = '%5.3e' % 0
    master.tags['[damp_blade_my]'] = '%5.3e' % 0
    master.tags['[damp_blade_mz]'] = '%5.3e' % 0
    master.tags['[damp_blade_kx]'] = '%5.3e' % damp_x
    master.tags['[damp_blade_ky]'] = '%5.3e' % damp_y
    master.tags['[damp_blade_kz]'] = '%5.3e' % damp_z

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    sim.prepare_launch(iter_dict, opt_tags, master,variable_tag_func_bladeonly,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg=msg, silent=False, check_log=True)

def blade_only_aero_plot():
    """
    """
    # quick and dirty visualisation
    sim_id = 'aero_00'
    figpath = 'simulations/fig/aero_blade_only/'
    postpath = 'simulations/hawc2/raw/'
    cao = sim.Cases(postpath, sim_id)

    # global-tower-node-002-forcevec-z
    # coord-bodyname-pos-sensortype-component
    ch0_tag = 'balance_arm-balance_arm-node-000-momentvec-x'
    ch1_tag = 'balance_arm-balance_arm-node-001-momentvec-x'

    for case, casedict in cao.cases.iteritems():
        res = cao.load_result_file(casedict)
        plt.plot(res.sig[:,0], res.sig[:,res.ch_dict[ch0_tag]['chi']], 'rs')
        plt.plot(res.sig[:,0], res.sig[:,res.ch_dict[ch1_tag]['chi']], 'go')

###############################################################################
### TEST BLADE CYLINDER
###############################################################################

def launch_test_cylinder():
    """
    Launching the first tests with the cylinder blade root section
    """
    sim_id = 'cyl_root_02'

    runmethod = 'local'
    #runmethod = 'none'
    #runmethod = 'gorm'
    msg = ''
    master = master_tags(sim_id, runmethod=runmethod, turbulence=False,
                         verbose=False, silent=False)
    master.tags['[t0]'] = 0.0
    master.tags['[duration]'] = 0.1
    master.tags['[auto_set_sim_time]'] = True
    master.tags['[azim-res]'] = 70 # azimuth positions per rotation
    #master.tags['[dt_sim]'] = 0.001
    master.tags['[out_format]'] = 'HAWC_BINARY'
    master.tags['[out_format]'] = 'HAWC_ASCII'
    master.tags['[windramp]'] = False
    master.tags['[windrampabs]'] = False
    master.tags['[yawmode]'] = 'fix'
    master.tags['[induction_method]'] = 1
    master.tags['[aerocalc_method]'] = 1
    master.tags['[generator]'] = False
    master.tags['[fix_rpm]'] = 700
    master.tags['[windspeed]'] = 8

    #master.tags['[damp_blade]'] = master.tags['[damp_insane]']

    # blade with cylinder at the root
    master.tags['[nr_bodies_blade]'] = 13
    master.tags['[nr_nodes_blade]'] = 14
    master.tags['[hub_lenght]'] = 0.083
    master.tags['[hub_drag]'] = False
    master.tags['[aeset]'] = 6
    master.tags['[strain_root_el]'] = 3
    master.tags['[strain_30_el]'] = 7
    master.tags['[st_blade1_subset]'] = stsets.b1_flex_opt2_cyl
    master.tags['[st_blade2_subset]'] = stsets.b2_flex_opt2_cyl
    master.tags['[st_blade3_subset]'] = stsets.b3_flex_opt2_cyl
    master.tags['[damp_blade1]'] = model.b1_flex_damp
    master.tags['[damp_blade2]'] = model.b2_flex_damp
    master.tags['[damp_blade3]'] = model.b3_flex_damp

    ## no cylinder in blade root model, use hub drag instead
    #master.tags['[nr_bodies_blade]'] = 11
    #master.tags['[nr_nodes_blade]'] = 12
    #master.tags['[hub_lenght]'] = 0.245
    #master.tags['[hub_drag]'] = False
    #master.tags['[aeset]'] = 1
    #master.tags['[strain_root_el]'] = 1
    #master.tags['[strain_30_el]'] = 4
    #master.tags['[st_blade1_subset]'] = stsets.b1_flex_opt2
    #master.tags['[st_blade2_subset]'] = stsets.b2_flex_opt2
    #master.tags['[st_blade3_subset]'] = stsets.b3_flex_opt2
    #master.tags['[damp_blade1]'] = model.b1_flex_damp
    #master.tags['[damp_blade2]'] = model.b2_flex_damp
    #master.tags['[damp_blade3]'] = model.b3_flex_damp

    #master.tags['[beam_output]'] = True
    #master.tags['[body_output]'] = True
    master.tags['[body_eigena]'] = True
    master.tags['[stru_eigena]'] = True
    master.tags['[sim]'] = False
    master.tags['[nr_bodies_blade]'] = 1
    master.tags['[nr_bodies_tower]'] = 1

    opt_tags = []
    iter_dict = {'[[qwerty]]' : ['']}

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    cases = sim.prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg=msg, silent=False, check_log=True)

    ## load the test result file
    #cao = sim.Cases(cases)
    #casedict = cases[cases.keys()[0]]
    #res = cao.load_result_file(casedict)

if __name__ == '__main__':

    figpath = 'simulations/fig/'
    post_dir = 'simulations/hawc2/raw/'

    #blade_st_with_cylinder_root()
#    launch_test_cylinder()

    # either launch jobs or check the results
    #msg = 'eigenfrequency analysis with optimize loop tests and debugging'
    #msg+='nr nodes now fixed at 4 for the tower and 12 for the blades \n'
    #msg+='in order to facilitate the nodes output at the strain gauges'
#    launch('d5', msg)
#    check_opt_results(post_dir + 'd5.pkl')
#    reformat_st()
#    blade_st_increase_t_stiffness()

#    launch_static_blade_deflection()
#    launch_blade_only_aero()
#    post_launch('aero_02', post_dir)
#    blade_only_aero_plot()

#    eigenvalue_analysis()

#    tower_shadow_profile()
#    tower_shadow_profile_2()
#    compare_rot_dirs('b3') # omega equal
#    compare_rot_dirs('b4') # omega equal
#    compare_rot_dirs('b5') # omega switched
#    compare_rot_dirs('b7') # omega equal
