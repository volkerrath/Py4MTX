#! /bin/bash 
rm *crop.tif

gdalwarp -te_srs EPSG:4326 -te 351.35 52.45 351.85 52.65 eire_fem_PLM_nT_MC_CET-L16.tif PLM_crop.tif
gdalinfo PLM_crop.tif> PLM_crop.tif.txt



