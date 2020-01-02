# Py4MT
This repository contains simple scripts for EM imaging, allowing easy access to mtpy, empymod,,simpeg, fenics and other electromagnetic toolboxes.


Get your working copy via git from the command line:

_git clone https://github.com/volkerrath/Py4MT/_

The scripts and jupyter notebooks are available in the subdirectory py4mt. 

This version will run under Python up to 3.7 (3.8 not yet tested). To install it in an Linux environment (e.g. Ubuntu, SuSE), you need to the following:

(1) Download the latest Anaconda or Miniconda version (https://www.anaconda.com/distribution/), and install by runing the downloaded bash script.

(2) Create an appropriate conda environment (including the necessary prerequisites) by:

_conda env create -f Py4MT.yml_

(3) Activate this environment by:

_conda activate Py4MT_

(4) Now the open souce toolboxes used need to be installed (currently only mtpy,_https://github.com/MTgeophysics/mtpy_)'. for mtpy, download the current development version via the git system. Then, enter the download directory 9usually callen _mtpy_) and execute:

_python setup.py install_
