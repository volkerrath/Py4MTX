# Py4MT (version 2)

# This repo is in a state of reorganization and adaption to Python 3.9+ and mtpy-v2. Not ready for production use!

This repository currently contains simple scripts useful for EM imaging, modelling, and inversion, partly using mtpy (https://github.com/MTgeophysics/mtpy). In particular there are helper scripts for manipulating edi files, ModEM input data, and model files.

Please keep in mind that this is experimental software, and will contain errors. Use at your own risk! However, we will frequently update the repository correcting bugs, and (re)adding additional functionality. Thee toolbox related to working on the Jacobian,
(https://github.com/volkerrath/JacoPyAn) will be merged in the near future.       
 
This repository contains the following subdirectories:


 -	**py4mt/info**
 	Doumentation for the toolbox, and some useful documentation for python, 
 	including the most important extensions, numpy, scipy, and matplotlib 
 	
 -	**py4mt/modules**
 	Contains the modules distortion.py, jacproc.py, mimdas.py, modem.py  mtplots.py,  plot.py,  
	plotrjmcmc.py, and util.py, called from the Python scripts run for different tasks of MT
	interprretation.
 	
 - 	**py4mt/scripts** and **py4mt/scripts2**
 	Contains the scripts  for preprocessing, visualization, and preparing the inversion of 
 	MT data.  The scripts in the former directory will move to the latter in the process of adaption.
	The workflow for reading and processing Jacobians from ModEM outputs (original & new)
	has been moved to https://github.com/volkerrath/JacoPyAN.     	 
 
- 	**py4mt/notebooks**
 	Contains jupyter notebooks for the most important scripts using the toolbox. 
	  	
- 	**modem**
	Modified and original ModEM source code files including corresponding Makefiles, useful for 
	sensitivity output.  
	
- 	**environment**
	Contains conda environment description files, and some useful helper files for working 
	within the conda environment. The current Py4MT environments contain a lot of packages
	which are not strictly necessary for running aempy, but useful for related geoscientific work.
	


Get your working copy via git from the command line:

_git clone https://github.com/volkerrath/Py4MT2/_

This version will run under Python 3.9+ (3.11 being the current development platform). To install it in an Linux environment (e.g. Ubuntu, SuSE), you need to do the following:

(1) Download the latest Anaconda or Miniconda version  (https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html), and install by running the downloaded bash script.  In order to make updates secure and avoid inconsistencies, copy .condarc to your home directory. As the Miniconda installer is not updated very frequently, it is useful to run the following within the Miniconda base environment:

_conda update conda_

_conda update --all_

Do this regularly to keep everything consistent!

(2) Create an appropriate conda environment (including the necessary prerequisites) from the files Py4MT.yml or Py4MT.txt found in the Py4MT base directory by:

_conda env create -f Py4MT.yml_

or:

_conda create --name Py4MT --file Py4MT.txt_

This will set up a Python 3.11 environment with all dependencies for aempy. Don't forget to update also Py4MT regularly, using _conda update --name Py4MT --all_! 


(3) Activate this environment by:
conda activate Py4MT_
(4) In order to reproduce the identical behavior of matplotlib, you should copy the included  _matplotlibrc_ file to the appropriate directory. Under Linux (Ubuntu), this should be : _$HOME/.config/matplotlib/matplotlibrc_. Pertinent changes should be made there, or have to be made within the scripts/modules using the _mpl.rcParams[name]=value_ mechanism. 

(6) Currently we have defined two environmental variable, _PY4MT_ROOT_ and _PY4MT_DATA_. These need to be set in your .bashrc file pointing to the place where Py4MT is installed, and where you keep your MT data, respectively. Keeping to this scheme makes life much easier when different persons work on the tools.

Example: 

_export PY4MT_ROOT='${HOME}/Py4MT/'_
	
_export PY4MT_DATA='${HOME}/Py4MT/data/'_


Easiest way to run scripts is using spyder. Enjoy!



