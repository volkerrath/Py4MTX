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
print("Set reference based on site"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)

rhoair = 1.e17

total = 0

DFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/fogo_modem_data_zzt_3pc_003_100s_edited"
MFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/run7_NLCG_035"
"""

"""

ReferenceType = "Site"

if ReferenceType.lower()[0:3] == "sit":

    EPSG = 5015
    # values from edi file
    SiteReference = "FOG933A"
    longitude = -25.46609
    latitude  =  37.76242
    elevation = 566.


    NewReferenceMod = [409426.000, 412426.000, 350.0+elevation]
    NewReferenceGeo = [37.76242, -25.46609, 350.+elevation]

    utm_x, utm_y = utl.proj_latlon_to_utm(longitude, latitude, utm_zone=EPSG)

    Dfile_out = DFile+"_reference"+SiteReference+".dat"
    Mfile_out = MFile+"_reference"+SiteReference+".rho"

elif ReferenceType.lower()[0:3] == "cen":
    EPSG = 5015
    Dfile_out = DFile+"_referenceCenter.dat"
    Mfile_out = MFile+"_referenceCenter.rho"


else:
    error("Reference type "+ReferenceType+" does not exist! Exit")


start = time.time()
dx, dy, dz, rho, refer = mod.read_model(MFile+".rho")
elapsed = time.time() - start
total = total + elapsed
print("Used %7.4f s for reading model from %s " % (elapsed, MFile))
print("ModEM reference is "+str(refer))
print("Min/max rho = "+str(np.min(rho))+"/"+str(np.max(rho)))

start = time.time()
Site, Comp, Data, Head = mod.read_data(DFile+".dat")
elapsed = time.time() - start
total = total + elapsed
print("Used %7.4f s for reading data from %s " % (elapsed, DFile))

if all(refer) == 0:
    print('Reference values are zero. New reference will be set.')
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


    x, y, z = mod.cells3d(dx, dy, dz, reference=refer)

    in_lat = Data[:,1]
    in_lon = Data[:,2]
    Data[:,3] = Data[:,3] - NewReferenceMod[0]
    Data[:,4] = Data[:,4] - NewReferenceMod[1]

    refer = [-NewReferenceMod[0],-NewReferenceMod[1], 0.]

else:
    error("reference exits and is nonzero! Exit.)")


mod.write_data(DatFile=Dfile_out,
               Dat=Data, Site=Site, Comp=Comp, Head=Head, out=True)


mod.write_model(ModFile=Mfile_out,
                dx=dx, dy=dy, dz=dz, rho=rho, reference=refer,
                out=True)
