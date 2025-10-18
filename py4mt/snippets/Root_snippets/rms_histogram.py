import numpy as np
import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import pyMT.utils as utils


# listfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.dat'
save_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/rms_plots'
dataset = DS.Dataset(datafile=data_file, responsefile=response_file)

# for ii, p in enumerate(dataset.data.periods):
# p_rms = {site: {x for x in range(dataset.data.NP)} for site in dataset.data.site_names}
all_rms = np.zeros((dataset.data.NP, dataset.data.NS, dataset.data.NR))
for ii, site in enumerate(dataset.data.site_names):
    for jj, comp in enumerate(dataset.data.components):
        for kk, p in enumerate(dataset.data.periods):
            all_rms[kk][ii][jj] = (np.abs(dataset.response.sites[site].data[comp][kk] - dataset.data.sites[site].data[comp][kk]) / 
                                   dataset.data.sites[site].used_error[comp][kk])

# Plot a histogram of rms per period
p_rms = np.sqrt(np.mean(all_rms**2, axis=2))
bins = np.linspace(0.5, 4, 15)
for ii, period in enumerate(dataset.data.periods[::4]):
    if ii < 5:
        end_point = (ii+1)*4
    else:
        end_point = dataset.data.NP + 1
    # Average them?
    # to_plot = np.sqrt(np.mean(p_rms[ii*4:(ii+1)*multiple, :], axis=0))
    # Or stack them?
    to_plot = p_rms[ii*4:end_point,:].flatten()
    # plt.hist(to_plot, bins, density=True, color='orange', edgecolor='black', linewidth='1.5')
    plt.hist(to_plot, bins, weights=np.ones(len(to_plot))/len(to_plot), color='orange', edgecolor='black', linewidth='1.5')
    plt.gca().set_ylim(0, 0.3)
    plt.savefig(save_path + '/rms_histogram_period_{}-{}s'.format(int(period), int(dataset.data.periods[min(22,end_point-1)])))
    plt.close()

# xy_rms = np.zeros((dataset.data.NP, dataset.data.NS))
# yx_rms = np.zeros((dataset.data.NP, dataset.data.NS))
# xy_diff = np.zeros((dataset.data.NP, dataset.data.NS))
# yx_diff = np.zeros((dataset.data.NP, dataset.data.NS))
# # Histogram of rms per period for phase (xy and yx)
# for ii, site in enumerate(dataset.data.site_names):
#     dphase_xy, dphase_xy_err = utils.compute_phase(dataset.data.sites[site], calc_comp='xy', wrap=1, errtype='used_error')
#     dphase_yx, dphase_yx_err = utils.compute_phase(dataset.data.sites[site], calc_comp='yx', wrap=1, errtype='used_error')

#     rphase_xy = utils.compute_phase(dataset.response.sites[site], calc_comp='xy', wrap=1)[0]
#     rphase_yx = utils.compute_phase(dataset.response.sites[site], calc_comp='yx', wrap=1)[0]

#     xy_rms[:,ii] = (np.abs(rphase_xy - dphase_xy) / dphase_xy_err)
#     yx_rms[:,ii] = (np.abs(rphase_yx - dphase_yx) / dphase_yx_err)
#     xy_diff[:,ii] = rphase_xy - dphase_xy
#     yx_diff[:,ii] = rphase_yx - dphase_yx

# bins = np.linspace(0.5, 5, 10)
# for ii, period in enumerate(dataset.data.periods[::4]):
#     if ii < 5:
#         end_point = (ii+1)*4
#     else:
#         end_point = dataset.data.NP + 1
#     # to_plot = np.sqrt(np.mean(p_rms[ii*4:(ii+1)*multiple, :], axis=0))
#     to_plot_xy = xy_rms[ii*4:end_point:].flatten()
#     to_plot_yx = yx_rms[ii*4:end_point:].flatten()
#     plt.hist([to_plot_xy, to_plot_yx], bins, density=True, color=['orange', 'blue'], edgecolor='black', linewidth='1.5')
#     plt.savefig(save_path + '/phase-rms_histogram_period_{}-{}s'.format(int(period), int(dataset.data.periods[min(22,end_point-1)])))
#     plt.close()

# # Plot differences in phase (not normalized by the error)
# bins = np.linspace(-25, 25, 26)
# for ii, period in enumerate(dataset.data.periods[::4]):
#     if ii < 5:
#         end_point = (ii+1)*4
#     else:
#         end_point = dataset.data.NP + 1
#     to_plot_xy = xy_diff[ii*4:end_point,:].flatten()
#     to_plot_yx = yx_diff[ii*4:end_point,:].flatten()
#     plt.hist([to_plot_xy, to_plot_yx], bins, density=True, color=['orange', 'blue'], edgecolor='black', linewidth='1.5')
#     plt.savefig(save_path + '/phase-diff_histogram_period_{}-{}s'.format(int(period), int(dataset.data.periods[min(22,end_point-1)])))
#     plt.close()