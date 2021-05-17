
# freeyaw-ojf-wt-tests

```
This repository is still a work-in-progress (WIP). The documentation is
incomplete at this point and not all of the scripts to reproduce the results are
in the repository yet. Please contact the author for more information.
```

This repository holds the post-processing utilities that have been used to
process wind tunnel measurements performed at the TU Delft Open Jet Facility.
A three bladed, downwind, free yawing wind turbine was tested during 2 campaigns:
one in February 2012, and a second one in April 2012. These tests took place
within the framework of a PhD project at DTU Wind Energy in Denmark (some
additional information can be found
[here](http://orbit.dtu.dk/en/projects/highly-flexible-wind-turbine-rotor-design%28498c5d0c-69b8-4fbd-96b0-a37ea9013789%29.html)).

These wind tunnel test results have been funded by:

* EU FP7 Marie-Curie IAPP grant [#230698](http://cordis.europa.eu/project/rcn/90495_en.html):
Windflower - Aeroelastic tailoring of a passive wind turbine rotor and
experiment-based validation of design code
* Dutch NWO Veni Grant [#11930](http://www.nwo.nl/onderzoek-en-resultaten/onderzoeksprojecten/45/2300167745.html):
Reconfigurable Floating Wind Farms

The wind tunnel tests have been organized by (job description listed is at the
time of the experiment):

* [David Robert Verelst](http://orcid.org/0000-0002-3687-0636), PhD student at DTU Wind Energy
* [Jan-Willem van Wingerden](http://www.dcsc.tudelft.nl/~jwvanwingerden/index.shtml), Associate Professor at the TU Delft

The results have been discussed in much more detail in the following publications:

* [Numerical and Experimental Results of a Passive Free Yawing Downwind Wind Turbine](http://orbit.dtu.dk/en/publications/numerical-and-experimental-results-of-a-passive-free-yawing-downwind-wind-turbine%28b4d534ad-b3c1-42e7-87b6-d72d1c21ea6c%29.html)
* [Open Access Wind Tunnel Measurements of a Downwind Free Yawing Wind Turbine](http://dx.doi.org/10.1088/1742-6596/753/7/072013)

When referring to the this data-set, please cite the above mentioned references.

This Github repository does not include the raw and calibrated result files,
but has to be downloaded separately due to its size. The current version of the
data set is [v2015-11-26](https://data.deic.dk/shared/62ffdf2d57c8a0133a7f3a43671d0e23)
and contains the following:

* [data/model/cross-sections](https://data.deic.dk/shared/34d18938b18e204f72bf182b0d913ef2) (67.3 MB):
blade cross section profile coordinates from root to tip (tabulated and plots)
* [data/raw](https://data.deic.dk/shared/2d9ae456b8cbefd0b399f9f1403f4497) (5.5GB):
unedited, raw result files
* [calibrated/DataFrame](https://data.deic.dk/shared/98ff753fd65e9ee589a5e11d837a20a1) (5.4GB):
calibrated, unified result files, in pandas.DataFrame HDF5 format
* [calibrated/CSV](https://data.deic.dk/shared/bcccf37b2adf03cd56652974603c541b) (3.8GB):
calibrated, unified result files, in CSV format (plain text)
* [database](https://data.deic.dk/shared/2bc207f4173783e95878d868e780c2fb)
Index of all the measurements, including various labels to distinguish between the
different measurement runs (such as blade type, wind speed, etc). Also includes
mean, max, min and standard deviations for each run and all channels.
* [database/figures](https://data.deic.dk/shared/38fbad00d00057c3834c17bdbabf7b66) (117.2 MB):
plots of the calibrated result files, in PNG format
* [media/pictures](https://data.deic.dk/shared/32990785caddb7f704bd4384cf03429c) (720.4 MB):
pictures of various phases of the experiment

These data files are compressed with [7zip](http://www.7-zip.org/) and the
archives are split in parts of maximum 1000MB.


## Wind turbine aeroelastic description

The aeroelastic description of the wind turbine that was used for this campaign 
is included under [data/model](data/model):

* The blade mass properties for all 6 blades are given under [data/model/blademassproperties](data/model/blademassproperties)
* The airfoil shapes are given under [data/model/S822.dat](data/model/S822.dat)
and [data/model/S823.dat](data/model/S823.dat)
* The full HAWC2 model description is filed under [data/model/hawc2](data/model/hawc2),
users looking for a single example to run can use
[data/model/hawc2/htc/ojf_post_example.htc](data/model/hawc2/htc/ojf_post_example.htc)
* Generator torque source code:
[data/model/hawc2/control/ojf_generator](data/model/hawc2/control/ojf_generator)
* Yaw control source code (for simulating a fixed-yaw-free-yaw transition):
[data/model/hawc2/control/yaw_control](data/model/hawc2/control/yaw_control)
* The control DLLs are under [data/model/hawc2/control](data/model/hawc2/control).
Note that the control DLL source can be compiled with the open source compiler
[Lazarus](https://www.lazarus-ide.org/).


## Description of the measurement channels

Channel names as used in the calibrated and unified result files as used in:

* [calibrated/DataFrame](https://data.deic.dk/shared/98ff753fd65e9ee589a5e11d837a20a1)
* [calibrated/CSV](https://data.deic.dk/shared/bcccf37b2adf03cd56652974603c541b)

Channel header names:

* ```time``` : time stamp [s]
* ```rpm``` : rotor speed [rpm]
* ```yaw_angle``` : yaw inflow angle [deg]
* ```tower_strain_fa``` : tower base for-aft (FA) bending moment in [Nm], based on tower
base straing gauges. Calibrated results only available for April campaign.
* ```tower_strain_ss``` : tower base side-side (SS) bending moment in [Nm], based on tower
base straing gauges. Calibrated results only available for April campaign.
* ```towertop_acc_fa``` : acceleration in for-aft direction, measured from an
accelerometer placed in the nacelle
* ```towertop_acc_ss``` : acceleration in side-side direction, measured from an
accelerometer placed in the nacelle
* ```towertop_acc_z``` : acceleration in vertical direction, measured from an
accelerometer placed in the nacelle
* ```voltage_filt``` : voltage measurement
* ```current_filt``` : current measurement
* ```rotor_azimuth``` : rotor azimuth position, where 180 degrees refers to
blade 3 pointing down (in tower shadow)
* ```duty_cycle``` : generator dump load setting, where 0 refers to lower generator
torque, and 1 refers to a higher generator torque setting.
* ```power``` : amount of power being dissipated in the resistance. Note that
this excludes the losses in the generator, the PWM module, cabling, and the mechanical
losses (bearings). This sensor can not be used for reliable rotor power measurements.
* ```power2``` : not reliable
* ```sound``` : not reliable
* ```sound_gain``` : not reliable
* ```hs_trigger``` :
* ```hs_trigger_start_end``` :
* ```rpm_pulse``` : rotor speed [rpm] based on the one pulse per revolution
measurement
* ```temperature``` : wind tunnel temperature [deg C]
* ```wind_speed``` : wind speed in wind tunnel [m/s]
* ```static_p``` : static pressure in wind tunnel [kPa]
* ```blade2_root``` :
* ```blade2_30pc``` :
* ```blade1_root``` :
* ```blade1_30pc``` :
* ```blade_rpm_pulse``` : rotor speed [rpm] based on the one pulse per
revolution measurement, based on pulse sensor on the wireless blade strain DAQ

