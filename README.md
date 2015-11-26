
freeyaw-ojf-wt-tests
====================

This repository holds the post-processing utilities that have been used to process wind tunnel measurements performed at the TU Delft Open Jet Facility. A 3 bladed, downwind, free yawing wind turbine was tested during 2 campaigns: one in February 2012, and a second one in April 2012. These tests took place within the framework of a PhD project that took place at DTU Wind Energy in Denmark (some additional information can be found [here](http://orbit.dtu.dk/en/projects/highly-flexible-wind-turbine-rotor-design%28498c5d0c-69b8-4fbd-96b0-a37ea9013789%29.html)).

These wind tunnel test results have been funded by:

* EU FP7 Marie-Curie IAPP grant [#230698](http://cordis.europa.eu/project/rcn/90495_en.html): Windflower - Aeroelastic tailoring of a passive wind turbine
rotor and experiment-based validation of design code
* Dutch NWO Veni Grant [#11930](http://www.nwo.nl/onderzoek-en-resultaten/onderzoeksprojecten/45/2300167745.html): Reconfigurable Floating Wind Farms

The wind tunnel tests have been organized by (job description listed is at the time of the experiment):

* [David Robert Verelst](http://orcid.org/0000-0002-3687-0636), PhD student at DTU Wind Energy
* [Jan-Willem van Wingerden](http://www.dcsc.tudelft.nl/~jwvanwingerden/index.shtml), Associate Professor at the TU Delft

The results have been discussed in much more detail in the PhD thesis of David Verelst:

* [Numerical and Experimental Results of a Passive Free Yawing Downwind Wind Turbine](http://orbit.dtu.dk/en/publications/numerical-and-experimental-results-of-a-passive-free-yawing-downwind-wind-turbine%28b4d534ad-b3c1-42e7-87b6-d72d1c21ea6c%29.html)

When referring to the this data-set, please cite the above mentioned PhD thesis.

This Github repository does not include the raw and calibrated result files, but has to be downloaded separately due to its size. The current version of the data set is [v2015-11-26](https://data.deic.dk/shared/62ffdf2d57c8a0133a7f3a43671d0e23) and contains the following:

* unedited, raw result files: [data/raw](https://data.deic.dk/shared/2d9ae456b8cbefd0b399f9f1403f4497) (5.5GB)
* calibrated, unified result files, in pandas.DataFrame HDF5 format: [calibrated/DataFrame](https://data.deic.dk/shared/98ff753fd65e9ee589a5e11d837a20a1) (5.4GB)
* calibrated, unified result files, in CSV format (plain text): [calibrated/CSV](https://data.deic.dk/shared/bcccf37b2adf03cd56652974603c541b) (3.8GB)

These data files are compressed with [7zip](http://www.7-zip.org/) and the archives are split in parts of maximum 1000MB.
