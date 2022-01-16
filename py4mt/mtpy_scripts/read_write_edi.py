#!/usr/bin/env python3
#!/bin/env python
"""
Description:
    Read and write and edi file
References:

CreationDate:   4/12/18
Developer:      Alison.Kirkby@ga.gov.au

Revision History:
    LastUpdate:     5/15/18   Alison Kirkby
"""


from mtpy.core.mt import MT
import os
os.chdir(r'C:/mtpywin/mtpy')  # change to path where mtpy is installed


# directory format for windows users
edi_path = r'C:\mtpywin\mtpy\examples\data\edi_files_2'
savepath = r'C:\tmp'


edi_file = os.path.join(edi_path, 'Synth00.edi')

mtObj = MT(edi_file)
mtObj.write_mt_file(save_dir=savepath,
                    fn_basename='Synth00_new',
                    file_type='edi',  # edi or xml format
                    new_Z_obj=None,  # provide a z object to update the data
                    new_Tipper_obj=None,  # provide a tipper object to update the data
                    longitude_format='LONG',  # write longitudes as 'LON' or 'LONG'
                    # write as decimal degrees (any other input
                    latlon_format='dd'
                    # will write as degrees minutes seconds
                    )
