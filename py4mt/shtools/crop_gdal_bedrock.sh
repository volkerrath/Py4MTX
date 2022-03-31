#! /bin/bash 
rm *crop.tif

gdalwarp -te_srs EPSG:4326 -te 351.35 52.45 351.85 52.65 BedRock.tif BedRock_crop.tif
gdalinfo BedRock_crop.tif> BedRock_crop.tif.txt



