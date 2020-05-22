# -*- coding: utf-8 -*-


"""
This script constructs a list of edifiles in a given directory, and produces 
plots for all of them. 

@author: sb & vr oct 2019

"""

# Import required modules

import os
#from mtpy.core.mt import MT
from modules.phase_tensor_pseudosection import PlotPhaseTensorPseudoSection


# Graphical paramter. Determine the plot formats produced, 
# and the required resolution: 

plot_pdf=True
plot_png=True
plot_eps=False
dpi = 600       
fsiz=6
lwid=0.1
stretch=(2500, 50)
prefix_remove = 'XXX'
plot_name = 'RRV_PhaseTensorSection'

# colorby:          - colour by phimin, phimax, skew, skew_seg
# ellipse_range     - 3 numbers, the 3rd indicates interval, e.g. [-12,12,3]
# set color limits  - default 0,90 for phimin or max, [-12,12] for skew. 
#                     
# If plotting skew_seg need to provide ellipse_dic, e.g: 
#                   ellipse_dict={'ellipse_colorby':'skew_seg',
#                                 'ellipse_range':[-12, 12, 3]} 


edict =  {'ellipse_colorby':'skew_seg',
          'ellipse_range':[-12, 12, 3]} 

# Plot tipper?
# plot tipper       - 'n'/'y' + 'ri/r/i' means real+imag
plot_t = 'n'




# Define the path to your EDI-files:
edi_in_dir =  r'/home/vrath/RRV_work/edifiles_in/'
print(' Edifiles read from: %s' % edi_in_dir)

# Define the path for saving  plots:
plots_dir = edi_in_dir 
# plots_dir = r'/home/vrath/RRV_work/edifiles_in/' 
print(' Plots written to: %s' % plots_dir)
if not os.path.isdir(plots_dir):
    print(' File: %s does not exist, but will be created' % plots_dir)
    os.mkdir(plots_dir)

# No changes required after this line!

# Construct list of EDI-files:


edi_files=[]
files=os.listdir(edi_in_dir)            
for entry in files:
    if entry.endswith('.edi') and not entry.startswith(prefix_remove):
        full = os.path.join(edi_in_dir,entry)
        edi_files.append(full)



#  print(edi_files)
#  create a plot object

plot_obj = PlotPhaseTensorPseudoSection(fn_list = edi_files,
                                 linedir='ns',          
                                 stretch=stretch,       
                                 station_id=(0,34),     # 'ns' if the line is closer to north-south, 'ew' if line is closer to east-west
                                 plot_tipper = plot_t,                           
                                 ellipse_dict = {'ellipse_colorby':'skew_seg',# option to colour by phimin, phimax, skew, skew_seg
                                                 'ellipse_range':[-12, 12, 3]} # set color limits - default 0,90 for phimin or max,
                                                                         # [-12,12] for skew. If plotting skew_seg need to provide
                                                                         # 3 numbers, the 3rd indicates interval, e.g. [-12,12,3]
                                 )

# update parameters (tweak for your dataset)
plot_obj.ellipse_size   = 12.
plot_obj.ylim           = (.0001,1000)
plot_obj.lw             = lwid
plot_obj.font_size      = fsiz
plot_obj.title          = 'Rainy River Transect' 

plot_obj.plot()


# Finally save figure

if plot_png:
    plot_obj.save_figure(os.path.join(plots_dir,plot_name+".png"),file_format='png',fig_dpi=dpi)
if plot_pdf:
    plot_obj.save_figure(os.path.join(plots_dir,plot_name+".pdf"),file_format='pdf',fig_dpi=dpi)
if plot_eps:
    plot_obj.save_figure(os.path.join(plots_dir,plot_name+".eps"),file_format='eps',fig_dpi=dpi)

        

  
