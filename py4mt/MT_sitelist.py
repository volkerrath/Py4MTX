# -*- coding: utf-8 -*-
'''
@author: sb & vr oct 2019
'''

# import required modules
import os
import csv
from mtpy.core.mt import MT
import numpy as np



dialect = 'unix'

# Define the path to your edi files
edi_dir = './edifiles_bbmt_roi/'
print(' Edifiles read from: %s' % edi_dir)
csv_file ='Timor_bbmt.csv'
print('Writing data to CSV file: '+csv_file)



edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)
ns =  np.size(edi_files)

with open(csv_file, 'w') as f:
    sitelist = csv.writer(f, delimiter=' ')
    sitelist.writerow(['Sitename', 'Latitude', 'Longitude'])
    sitelist.writerow([ns, ' ', ' '])

## loop over edifiles
#
    for filename in edi_files :
        print('reading data from: '+filename)
        name, ext = os.path.splitext(filename)
        file_i = edi_dir+filename
        # Create MT object
        mt_obj = MT(file_i)
        lon = mt_obj.lon
        lat = mt_obj.lat
        elev = mt_obj.elev
        east = mt_obj.east
        north = mt_obj.north

        sitename = mt_obj.station
        sitelist.writerow([sitename, lat, lon])
