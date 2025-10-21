#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Plot slices through 3D model (modem)

Created on Mon Oct 20 10:09:45 2025

@author: vrath
'''


# Import python modules
# edit according to your needs


import os
import sys
import time
from datetime import datetime
import warnings
import csv
import inspect

# Import numerical or other specialised modules
import numpy as np
# from mtpy.core.mt import MT

# add py4mt modules to pythonpath
mypath = ['/home/vrath/Py4MT/py4mt/modules/',
          '/home/vrath/Py4MT/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)
        
from version import versionstrg
import inverse as inv
import util as utl
import viz
import modem as mod

# Import required py4mt modules for your script

# import jacproc as jac
# import mtproc as proc
# import plotrjmcmc as plmc
# import femtic as fem
# import cluster as fcm

rhoair = 1.e17
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

Anisotropy = True


WorkDir = '/home/vrath/FEMTIC_work/test/'  # PY4MTX_DATA+'Misti/MISTI_test/'
ModFile = [WorkDir+'/Peru/1_feb_ell/TAC_100']

rho = []
if Anisotropy:
    dx, dy, dz, rhotmp, refmod, _ = mod.read_mod_aniso(
        ModFile, '.rho', trans='log10')
    print(' read model from %s ' % (ModFile + '.rho'))
    # rhotmp = mod.prepare_model(rhotmp, rhoair=rhoair)

else:
    dx, dy, dz, rhotmp, refmod, _ = mod.read_mod(
        ModFile, '.rho', trans='log10')
    print(' read model from %s ' % (ModFile + '.rho'))
    # rhotmp = mod.prepare_model(rhotmp, rhoair=rhoair)
    rho.append(rhotmp)

aircells = np.where(rho > rhoair/10)
PlotFile = WorkDir+'XXX'


# function [nx, ny, nz] = write_WS3d_model_P3(fname,x,y,z,rho,nzAir,type,origin,rotation)
# % writes a 3D resistivity model in Weerachai Siripunvaraporn's format;
# % allows for natural log resistivity by setting type = 'LOGE'
# %  (c) Anna Kelbert, 2009
# %  open file for output
# fid = fopen(fname,'w');
# [nx, ny, nz, na] = size(rho); na = 3;
# if nargin <= 8
#     type = 'LOGE';
# end
# % output file
# comment = 'Written by Matlab write_WS3d_model script';
# fprintf(fid, '# %s\n', comment);
# fprintf(fid, '%d %d %d %d %s\n', nx, ny, nz, 0, type);
# for j = 1:nx
#     status = fprintf(fid,'%G ',x(j));
# end
# fprintf(fid, '\n');
# for j = 1:ny
#     status = fprintf(fid,'%G ',y(j));
# end
# fprintf(fid, '\n');
# for j = 1:nz
#     status = fprintf(fid,'%G ',z(j));
# end
# fprintf(fid, '\n');
# for ia = 1:na
#     for k = 1:nz
#         fprintf(fid,'\n');
#         for j = 1:ny
#             % x index incremented fastest
#             for i = nx:-1:1
#                 fprintf(fid,'%15.5E',rho(i,j,k,ia));
#             end
#             fprintf(fid, '\n');
#         end
#     end
# end
# %  add origin, rotation angle to end
# if nargin <= 8
#     origin = [0 0 0];
#     rotation = 0;
# end
# fprintf(fid, '%d %d %d\n', origin);
# fprintf(fid, '%d\n', rotation);
# status = fclose(fid);
