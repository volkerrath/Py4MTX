# This script needs to be run in the 'rasterio' environment.
# Rasterio has dependency issues with other packages used in my main environment.
import pyMT.data_structures as DS
import pyMT.IO as IO
import pyMT.utils as utils
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.interpolate import RegularGridInterpolator


# These functions are defined as methods within the Model object, but that requires an up-to-date pyMT installment
def conductance(model, idx_0, idx_1):
    conductance = np.zeros((model.nx, model.ny))
    for ii in range(idx_0, idx_1):
        conductance += model.zCS[ii] * (1 / model.vals[:, :, ii])
    return conductance


def depth_weighted_rho(model, idx_0, idx_1):
    cond = conductance(model, idx_0, idx_1)
    return (1 / cond) * np.sum(model.dz[idx_0:idx_1])



# model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/smooth2/wstZK-s2_lastIter.rho')
outfile = 'E:/phd/NextCloud/data/ArcMap/WST/models/wstZK_s1_depthWeightedRho_0-5km.tiff'
model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.rho')
data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst')
data.locations = data.get_locs(mode='lambert')

# Remove the pads so you don't have a million cells
# for ii in range(5):
# 	model.dx_delete(0)
# 	model.dx_delete(-1)
# 	model.dy_delete(0)
# 	model.dy_delete(-1)
# 	model.dz_delete(-1) # If you want

model.origin = data.origin
model.to_lambert()

x, y, z = (utils.edge2center(model.dy),
		   utils.edge2center(model.dx),
		   [0])
X, Y, Z =  np.meshgrid(x, y, z, indexing='ij', sparse=True)
# points = np.array((X.flatten(), Y.flatten())).T
# Z = np.array([[iz] * X.size for iz in model.dz])

xCS = np.min(model.xCS)
yCS = np.min(model.yCS)

x_reg, y_reg = (np.linspace(x[0], x[-1], int((x[-1] - x[0]) / yCS)),
				np.linspace(y[0], y[-1], int((y[-1] - y[0]) / xCS)))
qx, qy = np.meshgrid(x_reg, y_reg)
RGI = RegularGridInterpolator((x, y),
							  depth_weighted_rho(model, idx_0=0, idx_1=30).transpose([1, 0]), method='nearest')

transform = Affine.translation(x[0], y[0]) * Affine.scale(np.min(model.yCS), np.min(model.xCS))

with rasterio.open(outfile,
					'w', driver='GTiff',
					height=y_reg.size, width=x_reg.size, count=1,
					dtype=np.float64,
					crs={'init': 'EPSG:3979'},
					transform=transform) as dst:
	dst.update_tags(0, depth=0)
	query_points = np.zeros((qx.size, 2))
	query_points[:, 0] = qx.flatten()
	query_points[:, 1] = qy.flatten()
	# query_points[:, 2] = 0
	vals = RGI(query_points)
	dst.write(np.log10(vals.reshape([qx.shape[0], qx.shape[1]])), 1)

	
