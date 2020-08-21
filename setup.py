#!/usr/bin/env python

from  setuptools import setup, find_packages

LONG_DESC = """
Py4MT is an open source Python package to assist MT Modelling, 
Inversion, Visualization, and Interpretation. It is based on 
mtpy (https://github.com/MTgeophysics/mtpy). 
"""


# # Add names of scripts here. You can also specify the function to call
# # by adding :func_name after the module name, and the name of the script
# # can be customized before the equals sign.




setup(
	name="aempy3",
	version=".99.dev",
	author="Volker Rath, Duygu Kiyan",
    author_email="volker.rath.montblanc@gmail.com",
	description="Python toolkit for Airborne ElectroMagnetics.",
	long_description=LONG_DESC,
    url="https://github.com/volkerrath/AEMPY3",
	license="GNU GENERAL PUBLIC LICENSE v3",
	classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ],
    
    install_requires = ['numpy',
                        'scipy',
                        'matplotlib',
                        'pyproj ',
                        'cython',
                        'gfortran_linux-64',
                        'gcc_linux-64',
                        'gxx_linux-64',
                        'geopandas',
                        'jupyter',
                        'spyder=4'
                        'nbconvert',
                        'xarray',
                        'shapely',
                        'scikit-learn'
                        'mtpy']
                        
    packages=find_packages(),
#   packages = [ 
#				]
	entry_points = {'console_scripts': 
                    ['ws2vtk = mtpy.utils.ws2vtk:main',
                      'modem_pyqt = mtpy.gui.modem_pyqt:main',
                      'modem_plot_response = mtpy.gui.modem_plot_response:main',
                      'modem_plot_pt_maps = mtpy.gui.modem_plot_pt_maps:main',
                      'modem_mesh_builder = mtpy.gui.modem_mesh_builder:main',
                      'modem2vtk = mtpy.utils.modem2vtk:main',
                      'occam1d_gui = mtpy.gui.occam1d_gui:main',
                      'edi_editor = mtpy.gui.edi_editor:main'
                      ]}
	
    )