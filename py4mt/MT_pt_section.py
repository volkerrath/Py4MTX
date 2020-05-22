# -*- coding: utf-8 -*-


"""
This script constructs a list of edifiles in a given directory, and produces 
plots for all of them. 

@author: sb & vr oct 2019

"""

# Import required modules

import os
#from mtpy.core.mt import MT
from mtpy.imaging.phase_tensor_pseudosection import PlotPhaseTensorPseudoSection


# Graphical paramter. Determine the plot formats produced, 
# and the required resolution: 

plot_pdf=True
plot_png=True
plot_eps=False
dpi = 400       
fsiz=5
lwid=0.,

plot_name = 'PhaseTensorSection'

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
files= os.listdir(edi_in_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)

print(edi_files)
#  create a plot object

plot_obj = PlotPhaseTensorPseudoSection(fn_list = edi_files,
                                 linedir='ns',          # 'ns' if the line is closer to north-south, 'ew' if line is closer to east-west
                                 stretch=(17,8),        # determines (x,y) aspect ratio of plot
                                 station_id=(0,10),     # indices for showing station names
                                 plot_tipper = plot_t,   # plot tipper ('y') + 'ri' means real+imag
                                 font_size=fsiz,
                                 lw=lwid,
                                 ellipse_dict = {'ellipse_colorby':'skew_seg',# option to colour by phimin, phimax, skew, skew_seg
                                                 'ellipse_range':[-12, 12, 3]} # set color limits - default 0,90 for phimin or max,
                                                                         # [-12,12] for skew. If plotting skew_seg need to provide
                                                                         # 3 numbers, the 3rd indicates interval, e.g. [-12,12,3]
                                 )

# update ellipse size (tweak for your dataset)
plot_obj.ellipse_size = 2.5





# Finally save figure

if plot_png:
    plot_obj.save_plot(os.path.join(plots_dir,plot_name+".png"),file_format='png',fig_dpi=dpi)
if plot_pdf:
    plot_obj.save_plot(os.path.join(plots_dir,plot_name+".pdf"),file_format='pdf',fig_dpi=dpi)
if plot_eps:
    plot_obj.save_plot(os.path.join(plots_dir,plot_name+".eps"),file_format='eps',fig_dpi=dpi)

        

  
