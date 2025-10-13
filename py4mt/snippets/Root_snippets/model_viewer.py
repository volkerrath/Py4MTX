# TODO
# Menu for point options (marker shape, size, annotations)
# Inclusion of data plotting (induction arrows, phase tensors, etc.)
# Will eventually want to be able to save images.
# File entry for new model files
#   - Which also means a mechanism to switch between models
# Single rotated slices?
# Single rotated slices? Path slice?
# Bring in data and start plotting sites, RMS, tipper, etc...
# 2-D views. Can each tab be run separately? e.g., when in images. 3-D tab, halt updates
# Colour map LUT (currently just uses default)
# 2-D views. Can each tab be run separately? e.g., when in 3-D tab, halt updates
# to 2-D views and vice-versa?
# Still have to do above
# Can also think about separate tabs for X, Y, Z slices to be plotted on their own
# Can these plots just be copies (linked) to the original ones? That way they auto-update?
from os import path as ospath
import sys
from copy import deepcopy
import numpy as np
import pyvista as pv
pv.set_plot_theme('dark')
pv.global_theme.font.size = 10
pv.global_theme.font.family = 'arial'
from pyvistaqt import QtInteractor
import pyMT.data_structures as WSDS
from pyMT.utils import sort_files, check_file
import pyMT.e_colours.colourmaps as cm
from pyMT.GUI_common import classes
from pyMT import gplot
from pyMT import resources
from PyQt5.uic import loadUiType
from pyMT.IO import debug_print
from scipy.interpolate import RegularGridInterpolator as RGI
# from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5 import QtWidgets, QtGui
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


path = ospath.dirname(ospath.realpath(__file__))
with pkg_resources.path(resources, 'model_viewer.jpg') as p:
    model_viewer_jpg = str(p)
# model_viewer_jpg = str(next(pkg_resources.path(resources, 'model_viewer.jpg').func(resources, 'model_viewer.jpg')))
# model_viewer_jpg = path + '/../resources/images/data_plot.jpg'
# Ui_MainWindow, QMainWindow = loadUiType(ospath.join(path, 'data_plot.ui'))
# UiPopupMain, QPopupWindow = loadUiType(ospath.join(path, 'saveFile.ui'))
UI_ModelWindow, QModelWindow = loadUiType(ospath.join(path, 'model_viewer.ui'))


def model_to_rectgrid(model, resolution=None, rho_axis='rho_x'):
    if '/' in rho_axis:
        rho_axis = rho_axis.split('/')
        v1 = getattr(model, rho_axis[0].strip())
        v2 = getattr(model, rho_axis[1].strip())
        vals = v1 / v2
    else:
        vals = getattr(model, rho_axis, 'vals')
    grid = pv.RectilinearGrid(np.array(model.dy),
                              np.array(model.dx),
                              -np.array(model.dz[::-1]))
    vals = np.log10(np.swapaxes(np.flip(vals, 2), 0, 1)).flatten(order='F')
    grid.cell_data['Resistivity'] = vals
    if resolution:
        X, Y, Z = np.meshgrid(resolution.yCS, resolution.xCS, resolution.zCS)
        volume = np.swapaxes(np.flip(X * Y * Z, 2), 0, 1).flatten(order='F')
        resvals = np.swapaxes(np.flip(resolution.vals, 2), 0, 1).flatten(order='F')
        grid.cell_data['RawResolution'] = np.log10(resvals)
        resvals = resvals / (volume.ravel())# ** (2/3))
        resvals[resvals > np.median(resvals.ravel())] = np.median(resvals.ravel())
        resvals = resvals - np.mean(resvals)
        resvals = resvals / np.std(resvals)
        resvals = 0.5 + resvals * np.sqrt(0.5)
        resvals[resvals > 1] = 1
        grid.cell_data['Resolution'] = resvals
    else:
        grid.cell_data['Resolution'] = np.ones(vals.shape)
        grid.cell_data['Resolution'][0] = .99
    return grid


class ModelWindow(QModelWindow, UI_ModelWindow):
    def __init__(self, files, parent=None):
        super(ModelWindow, self).__init__()
        self.setupUi(self)

        # Add the model
        self.setWindowTitle('Model Viewer - {}'.format(ospath.abspath(files['model'])))
        if files.get('data', None):
            self.dataset = WSDS.Dataset(modelfile=files['model'],
                                        datafile=files['data'])
        elif files.get('response', None):
            self.dataset = WSDS.Dataset(modelfile=files['model'],
                                        datafile=files['response'])
        else:
            self.dataset = WSDS.Dataset(modelfile=files['model'])
            print('No data file given. Site locations will not be available.')
        # try:
            # self.dataset = WSDS.Dataset(modelfile=files['model'],
                                        # datafile=files['data'])
        # except KeyError:
            # self.dataset = WSDS.Dataset(modelfile=files['model'])
        if self.dataset.model.dimensionality == '2d':
            # idx = np.argmin(abs(np.array(self.dataset.model.dy)))
            self.dataset.data.locations[:, 1] += self.dataset.model.dy[0] #sum(self.dataset.model.dy[:idx])
        if files.get('resolution', None):
            try:
                self.resolution = WSDS.Model(files['resolution'])
                self.use_resolution = 'Resolution'
            except ValueError:
                with open(files['resolution'], 'r') as f:
                    r = [float(x.strip()) for x in f.readlines()]
                    self.resolution = deepcopy(self.dataset.model)
                    self.resolution.vals = np.transpose(np.reshape(r, [self.resolution.nz,
                                                                       self.resolution.ny,
                                                                       self.resolution.nx]), [2, 1, 0])
                    self.use_resolution = 'Resolution'
            except KeyError:
                self.resolution = None            
                # self.resolution.vals = np.ones(self.dataset.model.vals.shape)
        else:
            self.resolution = None
        self.use_resolution = 1
        self.use_scalar = 'Resistivity'
        self.rho_axis = 'rho_x'
        self.model = self.dataset.model
        self.clip_model = deepcopy(self.model)
        self.clip_resolution = deepcopy(self.resolution)
        # self.dataset.data.locations /= 1000
        self.dataset.spatial_units = 'km'
        self.model.spatial_units = 'km'
        self.clip_model.spatial_units = 'km'
        # Make sure the frame fills the tab
        vlayout3D = QtWidgets.QVBoxLayout()
        # add the pyvista interactor object
        self.vtk_widget = QtInteractor(self.frame3D)
        self.vtk_widget.background_color = 'paraview'
        # self.vtk_widget.auto_update = 0
        vlayout3D.addWidget(self.vtk_widget)
        self.frame3D.setLayout(vlayout3D)
        self.pv_default_background = 'paraview'
        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        self.colourMenu = classes.ColourMenu(self)
        mainMenu.addMenu(self.colourMenu)
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)
        self.mesh_group = QtWidgets.QActionGroup(self)
        self.mesh_group.addAction(self.meshOn)
        self.mesh_group.addAction(self.meshOff)
        self.mesh_group.setExclusive(True)

        self.plot_data = {'stations': []}
        has_data = files.get('data', None)
        has_response = files.get('response', None)
        if has_data or has_response:
            self.plot_locations = True
            self.locs_3D = np.zeros((len(self.dataset.data.locations), 3))
            self.plot_data['stations'] = pv.PolyData(self.locs_3D)
            self.locs_3D[:, 0] = self.dataset.data.locations[:, 1]
            self.locs_3D[:, 1] = self.dataset.data.locations[:, 0]
        else:
            self.locs_3D = np.zeros((1, 3))
            self.plot_locations = False
        self.rect_grid = model_to_rectgrid(self.clip_model, self.clip_resolution, rho_axis=self.rho_axis)
        self.lut = 32
        # self.colourmap = 'turbo_r'
        self.colourmap = self.colourMenu.action_group.checkedAction().text()
        self.cmap = cm.get_cmap(self.colourmap, self.lut, self.colourMenu.map.invert_cmap.isChecked())
        self.cax, self.rho_cax, self.resolution_cax, self.aniso_cax = [1, 5], [1, 5], [-6, -2], [-2, 2]
        self.actors = {'X': [], 'Y': [], 'Z': [], 'transect': [], 'isosurface': []}
        self.contours = []
        self.slices = {'X': [], 'Y': [], 'Z': [], 'transect': []}
        self.transect_plot = {'easting': [], 'northing': []}
        self.transect_picking = False
        self.n_interp = 200
        self.show_transect_markers = True
        self.show_transect_line = True
        self.transect_artists = {'line': [], 'markers': []}
        self.faces = []
        self.init_widgets()
        # self.clip_volume()
        self.connect_widgets()
        self.render_3D()
        self.view_xy()
        # self.vtk_widget.view_isometric()
        # self.vtk_widget.view_xy()

        # Set up 2-D views
        self.canvas = {'2D': [], 'transect': []}
        self.toolbar = {'2D': [], 'transect': []}
        self.fig_2D = Figure()
        self.spec = {'X': [], 'Y': [], 'Z': []}
        # self.spec['Y'] = GridSpec(3, 3).new_subplotspec((0, 2), colspan=1, rowspan=2)
        # self.spec['X'] = GridSpec(3, 3).new_subplotspec((2, 0), colspan=2, rowspan=1)
        # self.spec['Z'] = GridSpec(3, 3).new_subplotspec((0, 0), colspan=2, rowspan=2)
        # self.spec['colorbar'] = GridSpec(3, 3).new_subplotspec((2, 2), colspan=1, rowspan=1)
        self.spec['Y'] = GridSpec(12, 12).new_subplotspec((0, 8), colspan=4, rowspan=8)
        self.spec['X'] = GridSpec(12, 12).new_subplotspec((8, 0), colspan=8, rowspan=4)
        self.spec['Z'] = GridSpec(12, 12).new_subplotspec((0, 0), colspan=8, rowspan=8)
        self.spec['colorbar'] = GridSpec(12, 12).new_subplotspec((11, 9), colspan=3, rowspan=1)
        self.fig_2D.add_subplot(self.spec['Z'])
        self.fig_2D.add_subplot(self.spec['Y'])
        self.fig_2D.add_subplot(self.spec['X'])
        self.fig_2D.add_subplot(self.spec['colorbar'])
        self.fig_2D.subplots_adjust(top=0.95, bottom=0.1,
                                    left=0.06, right=0.94,
                                    hspace=0.08, wspace=0.02)
        self.map = gplot.MapView(figure=self.fig_2D)
        # self.map.window['colorbar'] = self.
        self.init_map(self.dataset)
        self.add_mpl(self.map.window['figure'], '2D')
        self.fig_transect = Figure()
        self.add_mpl(self.fig_transect, 'transect')
        self.fig_transect.add_subplot(111)
        # Click event hooks
        self.cid = {'transect_2D': []}

        self.update_plan_view()
        self.update_2D_X()
        self.update_2D_Y()
        self.update_2D_colorbar()

    @property
    def x_slice(self):
        return self._x_slice

    @x_slice.setter
    def x_slice(self, value):
        self._x_slice = value
        self.xSliceLabel.setText('{:<4.4g} {}'.format(self.model.dx[self.x_slice],
                                                      self.dataset.spatial_units))

    @property
    def y_slice(self):
        return self._y_slice

    @y_slice.setter
    def y_slice(self, value):
        self._y_slice = value
        self.ySliceLabel.setText('{:<4.4g} {}'.format(self.model.dy[self.y_slice],
                                                      self.dataset.spatial_units))

    @property
    def z_slice(self):
        return self._z_slice

    @z_slice.setter
    def z_slice(self, value):
        self._z_slice = value
        self.zSliceLabel.setText('{:<4.4g} {}'.format(self.model.dz[self.z_slice],
                                                      self.dataset.spatial_units))

    def add_mpl(self, fig, tab='2D'):
        self.canvas[tab] = FigureCanvas(fig)
        if tab == '2D':
            frame = self.widget2D
        else:
            frame = self.widgetTransect
        frame.addWidget(self.canvas[tab])
        self.toolbar[tab] = NavigationToolbar(canvas=self.canvas[tab],
                                              parent=self.canvas[tab], coordinates=True)
        self.canvas[tab].draw()
        frame.addWidget(self.toolbar[tab])

    def init_map(self, dataset):
        self.map.dataset = dataset
        self.map.data = dataset.data
        self.map.raw_data = dataset.raw_data
        self.map.response = dataset.response
        self.map.site_names = dataset.data.site_names
        self.map.model = self.model
        self.map.colourmap = self.colourmap
        # self.map.model_cax = self.cax
        self.map._active_sites = []
        self.map.site_locations['generic'] = self.map.get_locations(
            sites=self.map.generic_sites)
        self.map.site_locations['active'] = self.map.get_locations(
            sites=self.map.active_sites)
        self.map.site_locations['all'] = self.map.get_locations(self.map.site_names)

    def update_2D_X(self):
        self.map.window['axes'][2].clear()
        self.map.plot_x_slice(x_slice=self.x_slice,
                              ax=self.map.window['axes'][2],
                              rho_axis=self.rho_axis)
        bounds = np.array([self.clip_model.dy[0], self.clip_model.dy[-1],
                           self.clip_model.dz[0], self.clip_model.dz[-1]])
        self.map.set_axis_limits(ax=self.map.window['axes'][2], bounds=bounds)
        if not self.map.window['axes'][2].yaxis_inverted():
            self.map.window['axes'][2].invert_yaxis()
        self.canvas['2D'].draw()

    def update_2D_Y(self):
        self.map.window['axes'][1].clear()
        self.map.plot_y_slice(y_slice=self.y_slice,
                              ax=self.map.window['axes'][1],
                              orientation='zx',
                              rho_axis=self.rho_axis)
        bounds = np.array([self.clip_model.dz[0], self.clip_model.dz[-1],
                           self.clip_model.dx[0], self.clip_model.dx[-1]])
        self.map.set_axis_limits(ax=self.map.window['axes'][1], bounds=bounds)
        self.map.window['axes'][1].yaxis.tick_right()
        self.map.window['axes'][1].yaxis.set_label_position('right')
        # if not self.map.window['axes'][1].yaxis_inverted():
        #     self.map.window['axes'][1].invert_yaxis()
        self.canvas['2D'].draw()

    def update_plan_view(self):
        # Currently redraws the whole map every time
        # This should be changed to just destroy and redraw whatever features are needed
        # print(self.map.site_locations['generic'])
        # Also there should be a mechanism that makes sure this is only redrawn if something changes
        self.map.window['axes'][0].clear()
        if self.map.window['colorbar']:
            self.map.window['colorbar'].remove()
            self.map.window['colorbar'] = None
        # DEBUG
        # print('I am updating the map')
        # pseudosection_toggles = self.get_pseudosection_toggles()
        # if pseudosection_toggles['data'] and pseudosection_toggles['fill']:
        #     fill_param = ''.join([pseudosection_toggles['fill'],
        #                           pseudosection_toggles['component']])
        #     self.map.plan_pseudosection(data_type=pseudosection_toggles['data'],
        #                                 fill_param=fill_param,
        #                                 period_idx=self.active_period,
        #                                 n_interp=self.nInterp.value())
        # self.map.plot_rms = self.plotRMS.checkState()
        # self.map.plot_locations()
        # PT_toggles = self.get_PT_toggles()
        # if 'None' not in PT_toggles['data']:
        #     self.map.plot_phase_tensor(data_type=PT_toggles['data'],
        #                                fill_param=PT_toggles['fill'],
        #                                period_idx=self.active_period)
        # induction_toggles = self.get_induction_toggles()
        # if induction_toggles['data']:
        #     self.map.plot_induction_arrows(data_type=induction_toggles['data'],
        #                                    normalize=induction_toggles['normalize'],
        #                                    period_idx=self.active_period)
        # self.toolbar.update()
        # self.toolbar.push_current()
        bounds = np.array([self.clip_model.dy[0], self.clip_model.dy[-1],
                           self.clip_model.dx[0], self.clip_model.dx[-1]])
        self.map.plot_plan_view(z_slice=self.z_slice, ax=self.map.window['axes'][0], rho_axis=self.rho_axis)
        self.update_slice_indicators('xy')
        if self.plot_locations:
            self.map.plot_locations()
            # print(self.map.data.locations)
        self.map.set_axis_limits(bounds=bounds)
        self.map.window['axes'][0].get_xaxis().set_visible(False)
        self.plot_transect_markers(redraw=True)
        self.plot_transect_line(redraw=True)
        self.update_plan_title()
        # self.map.window['axes'][0].set_aspect('equal')
        self.canvas['2D'].draw()

    def update_2D_colorbar(self):
        self.map.window['axes'][3].clear()
        # fake_vals = np.linspace(self.cax[0], self.cax[1], self.lut)
        # fake_im = self.map.window['axes'][3].scatter(range(self.lut), range(self.lut),
                                                     # c=fake_vals, cmap=self.cmap)
        fake_vals = np.linspace(self.cax[0], self.cax[1], self.lut)
        fake_im = self.map.window['axes'][3].scatter(range(self.lut), range(self.lut), s=10, marker='.',
                                                     c=fake_vals, cmap=self.cmap)
        fake_im.set_visible(False)
        self.map.window['figure'].colorbar(mappable=fake_im, cax=self.map.window['axes'][3],
                                           orientation='horizontal')
        self.map.window['axes'][3].set_xlabel(r'$Log_{10}$ Resistivity ($\Omega$m)', fontsize=self.map.label_fontsize)
        self.canvas['2D'].draw()

    def update_plan_title(self):
        if self.depthLabelRange.isChecked():
            self.map.window['axes'][0].set_title('Layer ' + str(self.z_slice) +
                                                 ', Depth = '+'{:.3g}'.format((self.model.dz[self.z_slice]+self.model.dz[self.z_slice-1])/2)+' km', fontsize=self.map.label_fontsize)
        elif self.depthLabelCenter.isChecked():
            self.map.window['axes'][0].set_title('Layer ' + str(self.z_slice) +
                                                 ', Depth = '+'{:.3g}'.format(self.model.dz[self.z_slice-1])+u'\u2013'+'{:.3g}'.format(self.model.dz[self.z_slice])+' km', fontsize=self.map.label_fontsize)
        self.canvas['2D'].draw()

    def update_slice_indicators(self, direction='xy', redraw=True):
        x_slice_loc = self.model.dx[self.x_slice]
        y_slice_loc = self.model.dy[self.y_slice]
        x_slice_limits = (self.clip_model.dy[0], self.clip_model.dy[-1])
        y_slice_limits = (self.clip_model.dx[0], self.clip_model.dx[-1])
        if 'y' in direction.lower() and redraw:
            self.plot_data['Y_indicator'], = self.map.window['axes'][0].plot((y_slice_loc, y_slice_loc),
                                                                             y_slice_limits,
                                                                             'w--')
        if 'x' in direction.lower() and redraw:
            self.plot_data['X_indicator'], = self.map.window['axes'][0].plot(x_slice_limits,
                                                                             (x_slice_loc, x_slice_loc),
                                                                             'w--')
        if 'x' in direction.lower() and not redraw:
            self.plot_data['X_indicator'].set_ydata((x_slice_loc, x_slice_loc))
        if 'y' in direction.lower() and not redraw:
            self.plot_data['Y_indicator'].set_xdata((y_slice_loc, y_slice_loc))

    def init_widgets(self):
        # Set up all the slicing widgets
        self.x_slice = 1
        self.y_slice = 1
        self.z_slice = 1
        self.xSliceEdit.setText('1')
        self.ySliceEdit.setText('1')
        self.zSliceEdit.setText('1')
        self.xSliceLabel.setText('1')
        self.ySliceLabel.setText('1')
        self.zSliceLabel.setText('1')
        self.xSliceSlider.setValue(1)
        self.ySliceSlider.setValue(1)
        self.zSliceSlider.setValue(1)
        self.xSliceSlider.setSingleStep(1)
        self.ySliceSlider.setSingleStep(1)
        self.zSliceSlider.setSingleStep(1)
        self.xSliceSlider.setPageStep(5)
        self.ySliceSlider.setPageStep(5)
        self.zSliceSlider.setPageStep(5)
        self.xSliceSlider.setMinimum(1)
        self.ySliceSlider.setMinimum(1)
        self.zSliceSlider.setMinimum(1)
        self.xSliceSlider.setMaximum(self.model.nx - 1)
        self.ySliceSlider.setMaximum(self.model.ny - 1)
        self.zSliceSlider.setMaximum(self.model.nz - 1)
        self.xSliceCheckbox.setCheckState(2)
        self.ySliceCheckbox.setCheckState(2)
        self.zSliceCheckbox.setCheckState(2)
        # Init clip boxes
        self.x_clip = [0, 0]
        self.y_clip = [0, 0]
        self.z_clip = [0, 0]
        self.x0ClipEdit.setText('0')
        self.y0ClipEdit.setText('0')
        self.z0ClipEdit.setText('0')
        self.x1ClipEdit.setText('0')
        self.y1ClipEdit.setText('0')
        self.z1ClipEdit.setText('0')
        self.nInterp.setText(str(self.n_interp))
        if not self.resolution:
            self.resolutionCheckBox.setEnabled(False)
        if self.plot_locations:
            if np.any([site.locations['elev'] for site in self.dataset.data.sites.values()]):
                self.elevationCheckBox.setEnabled(True)


    def connect_widgets(self):
        # Slicer widgets
        self.xSliceEdit.editingFinished.connect(self.x_text_change)
        self.ySliceEdit.editingFinished.connect(self.y_text_change)
        self.zSliceEdit.editingFinished.connect(self.z_text_change)
        self.xSliceSlider.valueChanged.connect(self.x_slider_change)
        self.ySliceSlider.valueChanged.connect(self.y_slider_change)
        self.zSliceSlider.valueChanged.connect(self.z_slider_change)
        # Clipping widgets
        self.x0ClipEdit.editingFinished.connect(self.clip_x)
        self.x1ClipEdit.editingFinished.connect(self.clip_x)
        self.y0ClipEdit.editingFinished.connect(self.clip_y)
        self.y1ClipEdit.editingFinished.connect(self.clip_y)
        self.z0ClipEdit.editingFinished.connect(self.clip_z)
        self.z1ClipEdit.editingFinished.connect(self.clip_z)
        self.clipVolume.pressed.connect(self.clip_volume)
        # Slice Checkboxes
        self.xSliceCheckbox.stateChanged.connect(self.x_slice_change)
        self.ySliceCheckbox.stateChanged.connect(self.y_slice_change)
        self.zSliceCheckbox.stateChanged.connect(self.z_slice_change)
        # Colour options
        self.colourMenu.action_group.triggered.connect(self.change_cmap)
        self.colourMenu.limits.triggered.connect(self.set_clim)
        self.colourMenu.lut.triggered.connect(self.set_lut)
        self.colourMenu.map.invert_cmap.triggered.connect(self.change_cmap)
        # self.colourMenu.all_maps[self.colourmap].setChecked(True)
        # 2-D Plot options
        self.mesh_group.triggered.connect(self.show_mesh)
        self.depthTitles = QtWidgets.QActionGroup(self)
        self.depthTitles.addAction(self.depthLabelRange)
        self.depthTitles.addAction(self.depthLabelCenter)
        self.depthTitles.triggered.connect(self.update_plan_title)
        # View buttons
        self.viewXY.triggered.connect(self.view_xy)
        self.viewXZ.triggered.connect(self.view_xz)
        self.viewYZ.triggered.connect(self.view_yz)
        # Interpolation widgets
        self.selectPoints.clicked.connect(self.transect_pick)
        self.nInterp.editingFinished.connect(self.set_nInterp)
        self.interpCheckbox.clicked.connect(self.toggle_transect3D)
        # Isosurface widgets
        self.isoPlot.stateChanged.connect(self.update_isosurface)
        self.isoRecalculate.clicked.connect(self.generate_isosurface)
        self.isoOpacity.valueChanged.connect(self.update_isosurface)
        # Resolution toggles
        self.resolutionCheckBox.clicked.connect(self.set_resolution_plot)
        # Station location toggles
        self.elevationCheckBox.clicked.connect(self.set_station_elevation)
        # PyVista backgrounds
        self.pv_background_group = QtWidgets.QActionGroup(self)
        self.pv_background_group.addAction(self.actionDefault)
        self.pv_background_group.addAction(self.actionGrey)
        self.pv_background_group.addAction(self.actionBlack)
        self.pv_background_group.addAction(self.actionBlue)
        self.pv_background_group.addAction(self.actionWhite)
        self.pv_background_group.setExclusive(True)
        self.pv_background_group.triggered.connect(self.change_background)
        # Rho axis selection actions
        self.rho_axis_group = QtWidgets.QActionGroup(self)
        self.rho_axis_group.addAction(self.action_plot_rho_x)
        self.rho_axis_group.addAction(self.action_plot_rho_y)
        self.rho_axis_group.addAction(self.action_plot_rho_z)
        self.rho_axis_group.addAction(self.action_plot_rho_xy)
        self.rho_axis_group.addAction(self.action_plot_rho_xz)
        self.rho_axis_group.addAction(self.action_plot_rho_yz)
        self.rho_axis_group.setExclusive(True)

        if self.dataset.model.isotropic:
            self.rho_axis_group.setEnabled(False)
        else:
            self.rho_axis_group.triggered.connect(self.change_rho_axis)

    def change_rho_axis(self):
        self.rho_axis = self.rho_axis_group.checkedAction().text().lower()
        if '/' in self.rho_axis:
            self.cax = self.aniso_cax
        else:
            self.cax = self.rho_cax
        self.rect_grid = model_to_rectgrid(self.clip_model, self.clip_resolution, rho_axis=self.rho_axis)
        self.update_all(reset_camera=False)


    def change_background(self):
        color = self.pv_background_group.checkedAction().text().lower()
        if color != 'default':
            self.vtk_widget.background_color = color
        else:
            self.vtk_widget.background_color = deepcopy(self.pv_default_background)
        # self.render_3D(reset_camera=False)

    def set_station_elevation(self):
        if self.elevationCheckBox.checkState():
            # self.locs_3D = np.zeros((len(self.dataset.data.locations), 3))
            # self.plot_data['stations'] = pv.PolyData(self.locs_3D)
            # self.locs_3D[:, 0] = self.dataset.data.locations[:, 1]
            # self.locs_3D[:, 1] = self.dataset.data.locations[:, 0]
            self.locs_3D[:, 2] = np.array([-self.dataset.data.sites[site].locations['elev'] / 1000 for site in self.dataset.data.site_names])
        else:
            self.locs_3D[:, 2] = 0
        self.render_3D(reset_camera=False)

    def set_resolution_plot(self):
        if self.resolutionCheckBox.checkState() == 0:
            self.use_scalar = 'Resistivity'
            self.use_resolution = 1
            self.cax = self.rho_cax
        elif self.resolutionCheckBox.checkState() == 1:
            self.use_scalar = 'Resistivity'
            self.use_resolution = 'Resolution'
            self.cax = self.rho_cax
        elif self.resolutionCheckBox.checkState() == 2:
            self.use_scalar = 'RawResolution'
            self.use_resolution = 1
            self.cax = self.resolution_cax
        self.render_3D(reset_camera=False)

    def generate_isosurface(self):
        # if self.actors['isosurface']:
            # self.vtk_widget.remove_actor(self.actors['isosurface'])
        low_val = self.isoLow.value()
        high_val = self.isoHigh.value()
        if not (low_val <= high_val):
            low_val = high_val
            self.isoLow.setValue(high_val)
        if low_val == high_val:
            self.isoNum.setValue(1)
        self.contours = pv.DataSetFilters.cell_data_to_point_data(self.rect_grid).contour(isosurfaces=self.isoNum.value(),
                                                                  rng=(np.log10(low_val), np.log10(high_val)))
        # self.actors['isosurface'] = self.vtk_widget.add_points(self.contours,
        #                                                      cmap=self.cmap,
        #                                                      clim=self.cax)
        self.isoPlot.setCheckState(2)
        self.update_isosurface()

    def update_isosurface(self):
        # print(self.isoPlot.checkState())
        # print(self.actors['isosurface'])
        # print(self.contours)
        # self.vtk_widget.remove_actor(self.actors['isosurface'])
        if self.isoPlot.checkState():
            if self.contours:
                opacity = self.isoOpacity.value()
                try:
                    self.actors['isosurface'] = self.vtk_widget.add_mesh(self.contours,
                                                                         cmap=self.cmap,
                                                                         clim=self.cax,
                                                                         opacity=opacity,
                                                                         name='isosurface')
                except RuntimeError:
                    self.isoPlot.setCheckState(0)
                    # QtWidgets.QMessageBox.warning(self, 'Unable to change coordinate system.')
                    QtWidgets.QMessageBox.warning(self, '...', 'No model values fit the requested contour.')
                    return
            else:
                self.isoPlot.setCheckState(0)

        # self.vtk_widget.update()
        self.show_bounds()

    def show_mesh(self):
        if self.meshOn.isChecked():
            self.map.mesh = True
        else:
            self.map.mesh = False
        self.update_all()

    def set_nInterp(self):
        try:
            N = int(self.nInterp.text())
            self.n_interp = N
            if len(self.transect_plot['easting']) > 1:
                self.generate_transect2D()
                self.generate_transect3D()
                # self.plot_transect()
        except ValueError:
            self.nInterp.setText(str(self.n_interp))
            self.debugLabel.setText('# of interp points must be an integer')

    def transect_pick(self):
        if not self.cid['transect_2D']:
            self.cid['transect_2D'] = self.canvas['2D'].mpl_connect('button_release_event', self.click_2D)
            self.tabWidget_2.setCurrentIndex(1)
            self.selectPoints.setStyleSheet('QPushButton {background-color: lightgreen; color: black;}')
            self.transect_plot['easting'] = []
            self.transect_plot['northing'] = []
            self.debugLabel.setText('Picking turned on.')
        else:
            self.canvas['2D'].mpl_disconnect(self.cid['transect_2D'])
            self.selectPoints.setStyleSheet('QPushButton {background-color: None; color: black;}')
            if len(self.transect_plot['easting']) > 1:
                self.generate_transect2D()
                self.generate_transect3D()
            self.cid['transect_2D'] = 0
            self.interpCheckbox.setCheckState(2)
            self.toggle_transect3D()
            self.tabWidget_2.setCurrentIndex(0)
            # self.plot_transect()
            # print(self.len(transect_plot['easting']) > 1)
            # print(self.transect_plot['northing'])
        # if self.transect_picking:
        #     self.vtk_widget.enable_cell_picking()
        #     self.transect_picking = False
        #     print('Done picking')
        #     print(self.picks[0].points)
        # else:
        #     self.transect_picking = True
        #     self.picks = []
        #     print('Start picking')
        #     self.vtk_widget.enable_cell_picking(mesh=self.slices['Z'], callback=self.save_picks)

    def save_picks(self, picks):
        self.picks.append(picks)
        print('Saving picks')

    def change_cmap(self):
        self.colourmap = self.colourMenu.action_group.checkedAction().text()
        self.cmap = cm.get_cmap(self.colourMenu.action_group.checkedAction().text(),
                                self.lut,
                                self.colourMenu.map.invert_cmap.isChecked())
        self.map.colourmap = self.colourmap
        self.map.lut = self.lut
        self.update_all(reset_camera=False)

    def set_lut(self):
        inputs, ret = self.colourMenu.set_lut(initial=self.lut)
        if ret:
            self.lut = inputs
            self.change_cmap()
            # self.update_all()

    def set_clim(self):
        inputs, ret = self.colourMenu.set_clim(initial_1=str(self.cax[0]),
                                               initial_2=str(self.cax[1]))
        lower, upper = inputs
        if ret and [lower, upper] != self.cax:
            try:
                self.cax[0] = float(lower)
                self.cax[1] = float(upper)
                self.map.model_cax = self.cax
                self.update_all()
            except ValueError:
                print('Bad inputs to clim')

    def clip_x(self):
        try:
            x0 = int(self.x0ClipEdit.text())
            x1 = int(self.x1ClipEdit.text())
        except ValueError:
            self.x0ClipEdit.setText(str(self.x_clip[0]))
            self.x1ClipEdit.setText(str(self.x_clip[1]))
            x0 = int(self.x0ClipEdit.text())
            x1 = int(self.x1ClipEdit.text())
        if x0 + x1 > self.model.nx:
            self.x0ClipEdit.setText(str(self.x_clip[0]))
            self.x1ClipEdit.setText(str(self.x_clip[1]))
        else:
            self.x_clip[0] = x0
            self.x_clip[1] = x1
        # if x0 > np.ceil(self.model.nx):
        #     self.x0ClipEdit.setText(str(self.x_clip[0]))
        # else:
        #     self.x_clip[0] = x0
        # if x1 > np.ceil(self.model.nx):
        #     self.x1ClipEdit.setText(str(self.x_clip[1]))
        # else:
        #     self.x_clip[1] = x1
        # self.clip_volume()
        # self.render_3D()

    def clip_y(self):
        try:
            y0 = int(self.y0ClipEdit.text())
            y1 = int(self.y1ClipEdit.text())
        except ValueError:
            self.y0ClipEdit.setText(str(self.y_clip[0]))
            self.y1ClipEdit.setText(str(self.y_clip[1]))
            y0 = int(self.y0ClipEdit.text())
            y1 = int(self.y1ClipEdit.text())
        if y0 + y1 > self.model.ny:
            self.y0ClipEdit.setText(str(self.y_clip[0]))
            self.y1ClipEdit.setText(str(self.y_clip[1]))
        # if y0 > np.ceil(self.model.ny):
        #     self.y0ClipEdit.setText(str(self.y_clip[0]))
        # else:
        #     self.y_clip[0] = y0
        # if y1 > np.ceil(self.model.ny):
        #     self.y1ClipEdit.setText(str(self.y_clip[1]))
        else:
            self.y_clip[0] = y0
            self.y_clip[1] = y1
        # self.clip_volume()
        # self.render_3D()

    def clip_z(self):
        try:
            z0 = int(self.z0ClipEdit.text())
            z1 = int(self.z1ClipEdit.text())
        except ValueError:
            self.z0ClipEdit.setText(str(self.z_clip[0]))
            self.z1ClipEdit.setText(str(self.z_clip[1]))
            z0 = int(self.z0ClipEdit.text())
            z1 = int(self.z1ClipEdit.text())
        if z0 + z1 > self.model.nz:
            self.z0ClipEdit.setText(str(self.z_clip[0]))
            self.z1ClipEdit.setText(str(self.z_clip[1]))
        else:
            self.z_clip[0] = z0
            self.z_clip[1] = z1
        # self.clip_volume()
        # self.render_3D()

    def x_text_change(self):
        try:
            x = int(self.xSliceEdit.text())
        except ValueError:
            self.xSliceEdit.setText(str(self.x_slice))
            return
        if x < 1 or x > self.model.nx:
            self.xSliceEdit.setText(str(self.x_slice))
        else:
            self.x_slice = x
            self.xSliceSlider.setValue(x)
            self.x_slice_change()

    def y_text_change(self):
        try:
            y = int(self.ySliceEdit.text())
        except ValueError:
            self.ySliceEdit.setText(str(self.y_slice))
            return
        if y < 1 or y > self.model.ny:
            self.ySliceEdit.setText(str(self.y_slice))
        else:
            self.y_slice = y
            self.ySliceSlider.setValue(y)
            self.y_slice_change()

    def z_text_change(self):
        try:
            z = int(self.zSliceEdit.text())
        except ValueError:
            self.zSliceEdit.setText(str(self.z_slice))
            return
        if z < 1 or z > self.model.nz:
            self.zSliceEdit.setText(str(self.z_slice))
        else:
            self.z_slice = z
            self.zSliceSlider.setValue(z)
            self.z_slice_change()

    def x_slider_change(self, value):
        self.xSliceEdit.setText(str(value))
        self.x_slice = value
        self.x_slice_change()

    def y_slider_change(self, value):
        self.ySliceEdit.setText(str(value))
        self.y_slice = value
        self.y_slice_change()

    def z_slider_change(self, value):
        self.zSliceEdit.setText(str(value))
        self.z_slice = value
        self.z_slice_change()

    def x_slice_change(self):
        # self.vtk_widget.remove_actor(self.actors['X'])
        if self.xSliceCheckbox.checkState():
            # x_slice = self.clip_model.dx[self.x_slice] 
            # self.slices['X'] = self.rect_grid.slice(normal='y',
            #                                         origin=(0, x_slice, 0))
            self.slices['X'] = self.generate_slice('x')
            self.actors['X'] = self.vtk_widget.add_mesh(self.slices['X'],
                                                        cmap=self.cmap,
                                                        clim=self.cax,
                                                        scalars=self.use_scalar,
                                                        opacity=self.use_resolution,
                                                        use_transparency=False,
                                                        name='x_slice')
        else:
            self.vtk_widget.remove_actor(self.actors['X'])
        self.update_2D_X()
        self.update_slice_indicators(direction='x', redraw=False)
        self.fig_2D.canvas.draw()
        # self.vtk_widget.update()
        # self.show_bounds()

    def y_slice_change(self):
        # self.vtk_widget.remove_actor(self.actors['Y'])
        if self.ySliceCheckbox.checkState():
            # y_slice = self.clip_model.dy[self.y_slice] 
            # self.actors['Y'] = self.rect_grid.slice(normal='x',
            #                                         origin=(y_slice, 0, 0))
            self.slices['Y'] = self.generate_slice('y')
            self.actors['Y'] = self.vtk_widget.add_mesh(self.slices['Y'],
                                                        cmap=self.cmap,
                                                        clim=self.cax,
                                                        scalars=self.use_scalar,
                                                        opacity=self.use_resolution,
                                                        use_transparency=False,
                                                        name='y_slice')
        else:
            self.vtk_widget.remove_actor(self.actors['Y'])
        self.update_2D_Y()
        self.update_slice_indicators(direction='y', redraw=False)
        self.fig_2D.canvas.draw()
        self.vtk_widget.update()
        # self.show_bounds()

    def z_slice_change(self):
        # self.vtk_widget.remove_actor(self.actors['Z'])
        if self.zSliceCheckbox.checkState():
            # z_slice = self.clip_model.dz[self.z_slice] 
            # self.actors['Z'] = self.rect_grid.slice(normal='z', origin=(0, 0, z_slice))
            self.slices['Z'] = self.generate_slice('z')
            self.actors['Z'] = self.vtk_widget.add_mesh(self.slices['Z'],
                                                        cmap=self.cmap,
                                                        clim=self.cax,
                                                        scalars=self.use_scalar,
                                                        opacity=self.use_resolution,
                                                        use_transparency=False,
                                                        name='z_slice')
        else:
            self.vtk_widget.remove_actor(self.actors['Z'])
        self.update_plan_view()
        # self.vtk_widget.update()
        # self.show_bounds()

    def generate_slice(self, normal='X'):
        ox = max(0, self.model.dx[self.x_clip[0]])
        oy = max(0, self.model.dy[self.y_clip[0]])
        if normal == 'x':
            slice_loc = self.model.dx[self.x_slice]
            gen_slice = self.rect_grid.slice(normal='y',
                                             origin=(oy, slice_loc, -self.model.dz[self.z_clip[0] + 1]))
        elif normal == 'y':
            slice_loc = self.model.dy[self.y_slice]
            gen_slice = self.rect_grid.slice(normal='x',
                                             origin=(slice_loc, ox, -self.model.dz[self.z_clip[0] + 1]))
        elif normal == 'z':
            slice_loc = -self.model.dz[self.z_slice]
            gen_slice = self.rect_grid.slice(normal=normal,
                                             origin=(oy, ox, slice_loc))
        return gen_slice

    def clip_volume(self):
        # x = [self.model.dx[self.x_clip[0]], self.model.dx[self.model.nx - self.x_clip[1]]]
        # y = [self.model.dy[self.y_clip[0]], self.model.dy[self.model.ny - self.y_clip[1]]]
        # z = [self.model.dz[self.z_clip[0]], self.model.dz[self.model.nz - self.z_clip[1]]]
        # clip = [x[0], x[1], y[0], y[1], z[0], z[1]]
        # self.clipped_volume = self.rect_grid.clip_box(clip)
        self.clip_model = deepcopy(self.model)
        self.clip_resolution = deepcopy(self.resolution)
        for ix in range(self.x_clip[0]):
            self.clip_model.dx_delete(0)
            if self.resolution:
                self.clip_resolution.dx_delete(0)
        for ix in range(self.x_clip[1]):
            self.clip_model.dx_delete(self.clip_model.nx)
            if self.resolution:
                self.clip_resolution.dx_delete(self.clip_model.nx)
        for iy in range(self.y_clip[0]):
            self.clip_model.dy_delete(0)
            if self.resolution:
                self.clip_resolution.dy_delete(0)
        for iy in range(self.y_clip[1]):
            self.clip_model.dy_delete(self.clip_model.ny)
            if self.resolution:
                self.clip_resolution.dy_delete(self.clip_model.ny)
        for iz in range(self.z_clip[0]):
            self.clip_model.dz_delete(0)
            if self.resolution:
                self.clip_resolution.dz_delete(0)
        for iz in range(self.z_clip[1]):
            self.clip_model.dz_delete(self.clip_model.nz)
            if self.resolution:
                self.clip_resolution.dz_delete(self.clip_model.nz)
        # self.map.model = self.clip_model
        self.rect_grid = model_to_rectgrid(self.clip_model, self.clip_resolution, rho_axis=self.rho_axis)
        self.validate_slice_locations()
        # debug_print([self.model.dy[self.y_slice], self.model.dy[self.y_clip[0]], self.model.dy[self.model.ny]], 'C:/Users/eric/Desktop/debug.log')
        self.xSliceSlider.setMinimum(self.x_clip[0] + 1)
        self.ySliceSlider.setMinimum(self.y_clip[0] + 1)
        self.zSliceSlider.setMinimum(self.z_clip[0] + 1)
        self.xSliceSlider.setMaximum(self.model.nx - self.x_clip[1] - 1)
        self.ySliceSlider.setMaximum(self.model.ny - self.y_clip[1] - 1)
        self.zSliceSlider.setMaximum(self.model.nz - self.z_clip[1] - 1)
        self.update_all()

    def validate_slice_locations(self):
        self.xSliceSlider.valueChanged.disconnect(self.x_slider_change)
        self.ySliceSlider.valueChanged.disconnect(self.y_slider_change)
        self.zSliceSlider.valueChanged.disconnect(self.z_slider_change)
        self.xSliceEdit.editingFinished.disconnect(self.x_text_change)
        self.ySliceEdit.editingFinished.disconnect(self.y_text_change)
        self.zSliceEdit.editingFinished.disconnect(self.z_text_change)
        if self.x_slice <= self.x_clip[0]:
            self.x_slice = self.x_clip[0] + 1
            self.xSliceSlider.setValue(self.x_slice)
            self.xSliceEdit.setText(str(self.x_slice))
        if self.y_slice <= self.y_clip[0]:
            self.y_slice = self.y_clip[0] + 1
            self.ySliceSlider.setValue(self.y_slice)
            self.ySliceEdit.setText(str(self.y_slice))
        if self.z_slice <= self.z_clip[0]:
            self.z_slice = self.z_clip[0] + 1
            self.zSliceSlider.setValue(self.z_slice)
            self.zSliceEdit.setText(str(self.z_slice))
        if self.x_slice > self.model.nx - self.x_clip[1]:
            self.x_slice = self.model.nx - self.x_clip[1] - 1
            self.xSliceSlider.setValue(self.x_slice)
            self.xSliceEdit.setText(str(self.x_slice))
        if self.y_slice > self.model.ny - self.y_clip[1]:
            self.y_slice = self.model.ny - self.y_clip[1] - 1
            self.ySliceSlider.setValue(self.y_slice)
            self.ySliceEdit.setText(str(self.y_slice))
        if self.z_slice > self.model.nz - self.z_clip[1]:
            self.z_slice = self.model.nz - self.z_clip[1] - 1
            self.zSliceSlider.setValue(self.z_slice)
            self.zSliceEdit.setText(str(self.z_slice))
        self.xSliceSlider.valueChanged.connect(self.x_slider_change)
        self.ySliceSlider.valueChanged.connect(self.y_slider_change)
        self.zSliceSlider.valueChanged.connect(self.z_slider_change)
        self.xSliceEdit.editingFinished.connect(self.x_text_change)
        self.ySliceEdit.editingFinished.connect(self.y_text_change)
        self.zSliceEdit.editingFinished.connect(self.z_text_change)

    def render_3D(self, reset_camera=True):
        self.vtk_widget.clear()
        # This method is a full redraw - but slices should still be taken again even
        # if checks are off otherwise they will not get trimmed
        if self.xSliceCheckbox.checkState():
            self.slices['X'] = self.generate_slice('x')
            self.actors['X'] = self.vtk_widget.add_mesh(self.slices['X'],
                                                        scalars=self.use_scalar,
                                                        cmap=self.cmap,
                                                        clim=self.cax,
                                                        opacity=self.use_resolution,
                                                        use_transparency=False,
                                                        name='x_slice')
        if self.ySliceCheckbox.checkState():
            self.slices['Y'] = self.generate_slice('y')
            self.actors['Y'] = self.vtk_widget.add_mesh(self.slices['Y'],
                                                        scalars=self.use_scalar,
                                                        cmap=self.cmap,
                                                        clim=self.cax,
                                                        opacity=self.use_resolution,
                                                        use_transparency=False,
                                                        name='y_slice')
        if self.zSliceCheckbox.checkState():
            self.slices['Z'] = self.generate_slice('z')
            self.actors['Z'] = self.vtk_widget.add_mesh(self.slices['Z'],
                                                        scalars=self.use_scalar,
                                                        cmap=self.cmap,
                                                        clim=self.cax,
                                                        opacity=self.use_resolution,
                                                        use_transparency=False,
                                                        name='z_slice')
        if len(self.transect_plot['easting']) > 1:
            self.generate_transect3D(redraw=False)
        # Checkbox for sites?
        if self.plot_locations:
            self.actors['stations'] = self.vtk_widget.add_mesh(self.plot_data['stations'],
                                                               style='points',
                                                               color='white',
                                                               name='stations')
        # self.vtk_widget.update()
        self.show_bounds()
        if reset_camera:
            self.vtk_widget.reset_camera()

    def show_bounds(self):
        # pass
        # print('Show bounds called')
        # print(self.clip_model.spatial_units)
        # print(round(self.clip_model.dy[0], -1))
        self.vtk_widget.show_bounds(bounds=[self.clip_model.dy[0], self.clip_model.dy[-1],
                                            self.clip_model.dx[0], self.clip_model.dx[-1],
                                           -self.clip_model.dz[-1], self.clip_model.dz[0]],
                                    axes_ranges=[round(self.clip_model.dy[0], -1), round(self.clip_model.dy[-1],-1),
                                                 round(self.clip_model.dx[0], -1), round(self.clip_model.dx[-1],-1),
                                                -round(self.clip_model.dz[-1], -1), round(self.clip_model.dz[0],-1)],
                                    xtitle='Easting (km)',
                                    ytitle='Northing (km)',
                                    ztitle='Depth (km)',
                                    font_family='arial',
                                    grid=True,
                                    location='outer',
                                    use_3d_text=False)
                                  # font_size=10)
                                  # padding=0.1)

    def view_xy(self):
        self.vtk_widget.view_xy()
        # self.vtk_widget.view_vector([0, 0, 1])
        # self.vtk_widget.view_xy()
        # self.vtk_widget.update()

    def view_xz(self):
        self.vtk_widget.view_xz()
        # self.vtk_widget.view_vector([0, 0, 1])
        # self.vtk_widget.view_xy()
        # self.vtk_widget.update()

    def view_yz(self):
        self.vtk_widget.view_yz()
        # self.vtk_widget.view_vector([0, 0, 1])
        # self.vtk_widget.view_xy()
        # self.vtk_widget.update()

    def update_all(self, reset_camera=True):
        self.update_2D_X()
        self.update_2D_Y()
        self.update_plan_view()
        self.update_2D_colorbar()
        self.render_3D(reset_camera)

    def click_2D(self, event):
        if self.map.window['axes'][0] == event.inaxes:
            self.transect_plot['easting'].append(event.xdata)
            self.transect_plot['northing'].append(event.ydata)
            self.debugLabel.setText('{}'.format(self.transect_plot))
        else:
            print('Outside valid axis')
            return

    def interpolate_between_points(self, E, N):
        qx = []
        qy = []
        distance = []
        for ii in range(len(E) - 1):
            distance.append(np.sqrt((E[ii] - E[ii + 1]) ** 2 +
                                    (N[ii] - N[ii + 1]) ** 2))
        distance = np.array(distance)
        total_distance = np.sum(distance)
        N_per_segment = np.int64(np.ceil(self.n_interp * distance / total_distance))
        for ii in range(len(E) - 1):
            # E1, E2 = (min(E[ii], E[ii + 1]), max(E[ii], E[ii + 1]))
            if E[ii] > E[ii + 1]:
                E0, E1 = E[ii + 1], E[ii]
                N0, N1 = N[ii + 1], N[ii]
            else:
                E0, E1 = E[ii], E[ii + 1]
                N0, N1 = N[ii], N[ii + 1]
            X = np.linspace(E0, E1, N_per_segment[ii])
            Y = np.interp(X, (E0, E1), (N0, N1))
            if E[ii] > E[ii + 1]:
                X = np.flip(X, axis=0)
                Y = np.flip(Y, axis=0)
            qx.append(X)
            qy.append(Y)
        qx = np.concatenate(qx).ravel()
        qy = np.concatenate(qy).ravel()
        # debug_print((qx, qy), 'debugT.log')
        return qx, qy

    def generate_transect2D(self):
        if not (self.transect_plot['easting'] and self.transect_plot['northing']):
            return
        qx, qy = self.interpolate_between_points(self.transect_plot['easting'],
                                                 self.transect_plot['northing'])
        qz = np.array(self.clip_model.dz)
        query_points = np.zeros((len(qx) * len(qz), 3))
        faces = np.zeros(((len(qx) - 1) * (len(qz) - 1), 5))
        cc = 0
        cc2 = 0
        for ix in range(len(qx)):
            for ii, iz in enumerate(qz):
                query_points[cc, :] = np.array((qx[ix], qy[ix], iz))
                if ii < len(qz) - 1 and ix < len(qx) - 1:
                    faces[cc2, 0] = 4
                    faces[cc2, 1] = cc
                    faces[cc2, 2] = cc + 1
                    faces[cc2, 3] = len(qz) + cc + 1
                    faces[cc2, 4] = len(qz) + cc
                    cc2 += 1
                cc += 1

        cell_N, cell_E, cell_z = self.clip_model.cell_centers()
        rho = getattr(self.clip_model, self.rho_axis)
        vals = np.transpose(np.log10(rho), [1, 0, 2])
        self.interpolator = RGI((cell_E, cell_N, cell_z),
                                vals, bounds_error=False, fill_value=None)
        interp_vals = self.interpolator(query_points)
        # self.map.window['axes'][0].plot(qx, qy, 'k--')
        # self.canvas['2D'].draw()
        self.interp_vals = np.reshape(interp_vals, [len(qx), len(qz)])
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.query_points = query_points
        self.faces = faces
        self.plot_transect2D(redraw=True)

    def generate_transect3D(self, redraw=True):
        query_points = deepcopy(self.query_points)
        query_points[:, 2] *= -1
        qp = np.vstack([self.transect_plot['easting'],
                        self.transect_plot['northing'],
                        np.zeros((len(self.transect_plot['easting'])))]).T
        # line = pv.Line(pointa=qp[0, :], pointb=qp[1, :], resolution=10)
        # for ii in range(len(qp[1:-1])):
            # line = line.merge(pv.Line(pointa=qp[ii+1, :], pointb=qp[ii+2, :], resolution=10))
        for ii in range(len(qp) - 1):
            line = pv.Line(pointa=qp[ii, :], pointb=qp[ii+1, :], resolution=10)
            if ii == 0:
                self.slices['transect'] = self.rect_grid.slice_along_line(line).clip_box(invert=False,
                                                                                         bounds=line.bounds[:4] +
                                                                                         (-self.clip_model.dz[-1],
                                                                                           self.clip_model.dz[0]))
            else:
                self.slices['transect'].merge(self.rect_grid.slice_along_line(line).clip_box(invert=False,
                                                                                             bounds=line.bounds[:4] +
                                                                                           (-self.clip_model.dz[-1],
                                                                                             self.clip_model.dz[0])),
                                              merge_points=False,
                                              inplace=True)
        # spline = pv.Spline(qp, 50)
        # self.slices['transect'] = self.rect_grid.slice_along_line(line).clip_box(invert=False, 
                                                                                 # bounds=line.bounds[:4] +
                                                                                        # (-self.clip_model.dz[-1],
                                                                                          # self.clip_model.dz[0]))
        # self.slices['transect'] = pv.PolyData(query_points, self.faces)
        self.plot_transect3D(redraw=redraw)

    def plot_transect2D(self, redraw=True):
        self.fig_transect.axes[0].clear()
        linear_x = np.zeros((len(self.qx)) + 1)
        linear_x[1:-1] = np.sqrt((self.qx[1:] - self.qx[:-1]) ** 2 + (self.qy[1:] - self.qy[:-1]) ** 2)
        linear_x = np.cumsum(linear_x)
        # Hack to get dimesniosn right
        linear_x[-1] += (linear_x[-1] - linear_x[-1])
        # qz = list(qz)
        # qz = list(self.qz)
        # qz.append(qz[-1] + 1)
        # qz = np.array(qz)
        # sub the hack into call to pcolormesh below, replacing self.qz 
        
        self.fig_transect.axes[0].clear()
        self.fig_transect.axes[0].pcolormesh(linear_x, np.array(list(self.qz) + [self.qz[-1] + 1]),
                                             self.interp_vals.T,
                                             vmin=self.cax[0], vmax=self.cax[1],
                                             cmap=self.cmap)
        self.fig_transect.axes[0].invert_yaxis()
        self.fig_transect.axes[0].set_xlabel('Linear Distance (km)')
        self.fig_transect.axes[0].set_ylabel('Depth (km)')
        self.canvas['transect'].draw()
        self.plot_transect_line()
        self.plot_transect_markers()

    def plot_transect3D(self, redraw=False):
        # if redraw and self.actors['transect']:
            
        if self.interpCheckbox.checkState():
            # self.vtk_widget.remove_actor(self.actors['transect'])
            # self.slices['transect'] = self.generate_slice('z')
            self.actors['transect'] = self.vtk_widget.add_mesh(self.slices['transect'],
                                                               style='surface',
                                                               # scalars=self.interp_vals,
                                                               # axis=0,
                                                               cmap=self.cmap,
                                                               clim=self.cax,
                                                               name='transect')
        else:
            self.vtk_widget.remove_actor(self.actors['transect'])
        if redraw:
            # self.vtk_widget.update()
            self.show_bounds()

    def plot_transect_line(self, redraw=False):
        if self.show_transect_line and self.transect_plot['easting']:
            if not self.transect_artists['line'] or redraw:
                self.transect_artists['line'], = self.map.window['axes'][0].plot(self.qx, self.qy, 'k--')
            else:
                self.transect_artists['line'].set_xdata(self.qx)
                self.transect_artists['line'].set_ydata(self.qy)
                # self.debugLabel.setText('Redrawing lines')
            self.canvas['2D'].draw()

    def plot_transect_markers(self, redraw=False):
        if self.show_transect_markers and self.transect_plot['easting']:
            if not self.transect_artists['markers'] or redraw:
                self.transect_artists['markers'], = self.map.window['axes'][0].plot(self.transect_plot['easting'],
                                                                                    self.transect_plot['northing'],
                                                                                    'yv')
            else:
                self.transect_artists['markers'].set_xdata(self.transect_plot['easting'])
                self.transect_artists['markers'].set_ydata(self.transect_plot['northing'])
            #     self.debugLabel.setText('Redrawing markers')
            self.canvas['2D'].draw()
        # self.map.plot_y_slice(y_slice=self.y_slice,
        #                       ax=self.map.window['axes'][1],
        #                       orientation='zx')
        # bounds = np.array([self.clip_model.dz[0], self.clip_model.dz[-1],
        #                    self.clip_model.dx[0], self.clip_model.dx[-1]])
        # self.map.set_axis_limits(ax=self.map.window['axes'][1], bounds=bounds)
        # self.map.window['axes'][1].yaxis.tick_right()
        # self.map.window['axes'][1].yaxis.set_label_position('right')

    def toggle_transect3D(self):
        if self.slices['transect'] != []:
            if self.interpCheckbox.checkState():
                self.plot_transect3D(redraw=True)
            else:
                self.vtk_widget.remove_actor(self.actors['transect'])
                # self.vtk_widget.update()
                self.show_bounds()
        else:
            self.interpCheckbox.setCheckState(False)


def main():
    files = sys.argv[1:]
    for file in files:
        if not check_file(file):
            print('File {} not found.'.format(file))
            return
    files = sort_files(files=files)
    # try:
    #     data = WSDS.Data(datafile=files['dat'])
    # except KeyError:
    #     print('No data file given. Site locations will not be available.')
    #     data = None
    try:
        model = WSDS.Model(files['model'])
    except KeyError:
        print('Model must be specified')
        return
    app = QtWidgets.QApplication(sys.argv)
    viewer = ModelWindow(files)
    viewer.setWindowIcon(QtGui.QIcon(model_viewer_jpg))
    # viewer.setWindowTitle('Model Viewer - {}'.format(ospath.abspath(files['model'])))

    viewer.show()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == '__main__':
    main()
    
    # window.disconnect_mpl_events()
