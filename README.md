# Py4MT
This repository contains simple scripts useful for EM imaging, modelling , and inversion, artly using mtpy (https://github.com/MTgeophysics/mtpy). In particular  there are helper scripts for maipulatting edi files, ModEM input data and model files. There are utilities to manipulate and write data, model, sensitivity and Jacobian in different formats as netcdf or npy/z. This includes 3-D image processing on models, quantities derived from the Jacobian, and (soon) expansion of 3-D models to low-rank paramtrizations, and 3-D nullspace shuttles. 


Get your working copy via git from the command line:

_git clone https://github.com/volkerrath/Py4MT/_

This version will run under Python 3.6 - 3.8 (3.9 not yet fully tested, but seems to work correctly). To install it in an Linux environment (e.g. Ubuntu, SuSE), you need to do the following:

(1) Download the latest Anaconda or Miniconda version (https://www.anaconda.com/distribution/), and install by running the downloaded bash script. 

(2) Create an appropriate conda environment (including the necessary prerequisites) from the files EM.yml or EM.txt found in the Py4MT base directory by:

_conda env create -f EM.yml_

or:

_conda create --name EM --file EM.txt_

This will set up a Python 3.8 environment with all dependencies for aempy.

(3) Activate this environment by:

_conda activate EM_

Easiest way to run scripts is using spyder.


Enjoy!
This version will run under Python up to 3.7 (3.8 not yet tested). To install it in an Linux environment (e.g. Ubuntu, SuSE), you need to the following:

(1) Download the latest Anaconda or Miniconda version (https://www.anaconda.com/distribution/), and install by runing the downloaded bash script.

(2) Create an appropriate conda environment (including the necessary prerequisites) by:

_conda env create -f Py4MT.yml_

(3) Activate this environment by:

_conda activate Py4MT_

(4) Now the open souce toolboxes used need to be installed (currently only mtpy, _https://github.com/MTgeophysics/mtpy_). for mtpy, download the current development version via the git system. Then, enter the download directory 9usually callen _mtpy_) and execute:

_python setup.py install_
