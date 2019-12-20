# -*- coding: utf-8 -*-
"""

@author: sb & vr oct 2019

"""

# import required modules
import os
from mtpy.core.mt import MT

plot_pdf=True
plot_png=False
plot_eps=False
dpi = 400

# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal  + determinant
plot_z = 1
# plot tipper 'y' or 'n'
plot_t = 'yri'
# plot phase tensor 'y' or 'n'
plot_p  = 'y'

# Define the path to your edi files
edi_dir = './edifiles_bbmt_roi/'
print(' Edifiles read from: %s' % edi_dir)

# Define the path for saving  plots
plots_dir = './plots_bbmt_roi/' #edi_dir #'./plots_synth/'

edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)

## loop
for filename in edi_files :
    print('\n \n \n reading data from '+filename)
    name, ext = os.path.splitext(filename)
    
    # Create an MT object 
    file_i = edi_dir+filename
    mt_obj = MT(file_i)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))
    pt_obj = mt_obj.plot_mt_response(plot_num=plot_z,
                                     plot_tipper = plot_t,
                                     plot_pt = plot_p,
                                     x_limits = (0.00001,10000.),
                                     res_limits=(0.1 ,1000.), # log resistivity limits
                                     phase_limits=(0,90), # log phase limits
#                                     shift_yx_phase = True, # True or False
    )
    # now save figure
    if plot_png:
        pt_obj.save_plot(os.path.join(plots_dir,name+".png"),file_format='png',fig_dpi=dpi)
    if plot_pdf:
        pt_obj.save_plot(os.path.join(plots_dir,name+".pdf"),file_format='pdf',fig_dpi=dpi)
    if plot_eps:
        pt_obj.save_plot(os.path.join(plots_dir,name+".eps"),file_format='eps',fig_dpi=dpi)
        
        
          
  
