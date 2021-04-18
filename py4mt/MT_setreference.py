# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
# ---

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
# import gdal
# import scipy as sc
# import vtk
# import pyvista as pv
# import pyvistaqt as pvqt
# import discretize
# import tarfile
# import pylab as pl
# from time import sleep

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Set new reference"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)

rhoair = 1.e17

total = 0

# MFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/run3_NLCG_039"
# PFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/run3_NLCG_039"
# OFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/fogo_modem_phaset_tip_100s_data"
OFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/fogo_modem_data_zzt_3pc_003_100s_edited"
PFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/run7_NLCG_035"
MFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/run7_NLCG_035"
"""

"""
EPSG = 5015
ReferenceType = "Site"

if ReferenceType.lower()[0:3] == "sit":

    SiteReference = "FOG933A"
    # values from edi file
    longitude = -25.46609
    latitude  =  37.76242
    elevation = 566.


    NewReferenceMod = [409426.000, 412426.000, 350.0+elevation]
    NewReferenceGeo = [37.76242, -25.46609, 350.+elevation]

    utm_x, utm_y = utl.proj_latlon_to_utm(longitude, latitude, utm_zone=EPSG)

    OFile_out = OFile+"_Refsite_"+SiteReference+".dat"
    PFile_out = PFile+"_Refsite_"+SiteReference+".dat"
    Mfile_out = MFile+"_Refsite_"+SiteReference+".rho"

elif ReferenceType.lower()[0:3] == "cen":
    error("Center reference not yet implemented! Exit.")
    OFile_out = OFile+"_RefCenter.dat"
    PFile_out = PFile+"_RefCenter.dat"
    Mfile_out = MFile+"_RefCenter.rho"


else:
    error("Reference type "+ReferenceType+" does not exist! Exit")


start = time.time()
dx, dy, dz, rho, refer = mod.read_model(MFile+".rho")

print("ModEM reference is "+str(refer))
print("Min/max rho = "+str(np.min(rho))+"/"+str(np.max(rho)))
print('New reference will be set.')

refer = [-NewReferenceMod[0],-NewReferenceMod[1], NewReferenceMod[2]]
mod.write_model(ModFile=Mfile_out,
                dx=dx, dy=dy, dz=dz, rho=rho, reference=refer,
                out=True)
elapsed = time.time() - start
print("Used %7.4f s for processing model from %s " % (elapsed, MFile))

start = time.time()
for FF in [OFile, PFile]:

    Site, Comp, Data, Head = mod.read_data(FF+".dat")
    in_lat = Data[:,1]
    in_lon = Data[:,2]
    Data[:,3] = Data[:,3] - NewReferenceMod[0]
    Data[:,4] = Data[:,4] - NewReferenceMod[1]

    print("New reference will be set in %s" % (FF))
    hlin = 0
    nhead = len(Head)
    nblck = int(nhead/8)
    print(str(nblck)+" blocks will be written.")


    for ib in np.arange(nblck):
        blockheader = Head[hlin:hlin+8]
        print("Original: %s" % blockheader[6].replace("\n",""))
        blockheader[6] = "> "+str(NewReferenceGeo[0])+"  "+str(NewReferenceGeo[1])+"\n"
        print("New     : %s" % blockheader[6].replace("\n",""))
        Head[hlin:hlin+8] = blockheader
        hlin = hlin+8

    mod.write_data(DatFile=OFile_out,
               Dat=Data, Site=Site, Comp=Comp, Head=Head, out=True)

elapsed = time.time() - start
print("Used %7.4f s for processing data files" % (elapsed))
