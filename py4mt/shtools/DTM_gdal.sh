#! /bin/bash 

## ITM
gdal_translate -a_srs EPSG:2157 -of GTiff ./dtm_epa_20m/w001001.adf Ireland20m_EPA_EPSG2157_itm.tif
gdalinfo Ireland20m_EPA_EPSG2157_itm.tif > Ireland20m_EPA_EPSG2157_itm.txt
## latlon
gdal_translate -a_srs EPSG:4326 -of GTiff ./dtm_epa_20m/w001001.adf Ireland20m_EPA_EPSG4326_latlon.tif
gdalinfo Ireland20m_EPA_EPSG4326_latlon.tif > Ireland20m_EPA_EPSG4326_latlon.txt
## UTM - geotiff
gdal_translate -a_srs EPSG:32629 -of GTiff ./dtm_epa_20m/w001001.adf ./Ireland20m_EPA_EPSG32629_utm.tif
gdalinfo Ireland20m_EPA_EPSG32629_utm.tif > Ireland20m_EPA_EPSG32629_utm.txt
## UTM - netcdf (e.g.GMT)
gdal_translate -a_srs EPSG:32629 -of GMT ./dtm_epa_20m/w001001.adf Ireland20m_EPA_EPSG32629_utm.nc
gdalinfo Ireland20m_EPA_EPSG32629_utm.nc > Ireland20m_EPA_EPSG32629_utm.txt
