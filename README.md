# Py4MT
This repository contains simple scripts useful for EM imaging, modelling , and inversion, partly using mtpy (https://github.com/MTgeophysics/mtpy). In particular  there are helper scripts for manipulating edi files, ModEM input data, and model files. There are utilities to manipulate and write data, model, sensitivity and Jacobian in different formats as netcdf or npy/z. It also includes 3-D image processing on models, body insertion (general ellipsoids, boxes), calculation of quantities derived from the Jacobian, nullspace shuttles, and (soon) expansion of 3-D models to low-rank parametrizations (DCT, legendre, PCA). 

Please keep in mind that this is an experimental software, and may contain errors. Use at your own risk! However, we will frequently update the repository correcting bugs, and adding additional functionality.                 
 
This repository contains the following subdirectories:


 -	**py4mt/info**
 	Doumentation for the toolbox, and some useful documentation for python, 
 	including the most important extensions, numpy, scipy, and matplotlib 
 	
 -	**py4mt/modules**
 	Contains the modules aesys.py, prep.py, inv.py, algs.py, util.py, and 
	modem.py which are called from the Python scripts run for different tasks of AEM inversion.
 	
 - 	**py4mt/scripts**
 	Contains the scripts  for preprocessing, visualization, and one-dimensional inversion of 
 	AEM data, explaining the typical work flow using the toolbox. Also included is a workflow 
	for static shift correction of MT observations (work in progress).     	 

- 	**rcfiles**
	Contains some useful helper files for working with the conda EM environment.


Get your working copy via git from the command line:

_git clone https://github.com/volkerrath/Py4MT/_

This version will run under Python 3.6 - 3.8 (3.9 not yet fully tested, but seems to work correctly). To install it in an Linux environment (e.g. Ubuntu, SuSE), you need to do the following:

(1) Download the latest Anaconda or Miniconda version (https://www.anaconda.com/distribution/), and install by running the downloaded bash script. In order to make updates secure and avoid inconsistencies, copy _.condarc_ to your home directory. 

(2) Create an appropriate conda environment (including the necessary prerequisites) from the files EM.yml or EM.txt found in the Py4MT base directory by:

_conda env create -f EM.yml_

or:

_conda create --name EM --file EM.txt_

This will set up a Python 3.8 environment with all dependencies for aempy.

(3) Activate this environment by:

_conda activate EM_

(4) Now the open source toolboxes used need to be installed (currently only mtpy, _https://github.com/MTgeophysics/mtpy_). For mtpy, download the current development version via the git system. Then, enter the download directory (usually called _mtpy_) and execute:

_python setup.py install_ or _pip install [-e] ._

(5) In order to reproduce the identical behavior of matplotlib, you should copy the included  _matplotlibrc_ file to the appropriate directory. Under Linux (Ubuntu), this should be : _$HOME/.config/matplotlib/matplotlibrc_. Pertinent changes should be made there, or have to be made within the scripts/modules using the _mpl.rcParams[name]=value_ mechanism. 


Easiest way to run scripts is using spyder. Enjoy!

