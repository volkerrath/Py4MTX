FEMTIC Clean Rho Project

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking)

Overview
--------
Complete, pure-Python workflow for FEMTIC MT resistivity (rho) files:
- femtic_pipeline.py: skeleton pipeline showing how to wire edits/injection/slice.
- femtic_rho_inject.py: injects resistivity/limits/n/flags into rho-files (from NPZ or vectors).
- femtic_profiles.py: vertical profiles via RBF/IDW; CSV/NPZ/HDF5 export; Matplotlib plotting.
- femtic_polyline_slice.py: vertical 'curtain' slice along XY polyline; PyVista viewer; VTK/PNG/CSV/NPZ export.
- femtic_borehole_viz.py: plotting helpers for single/multi borehole profiles.

Dependencies
------------
pip install numpy matplotlib pyvista h5py

Example 1 – Borehole Profile Plotting
-------------------------------------
python -m femtic_profiles \      --npz elements_arrays.npz \      --values-key log10_resistivity \      --x 1000 --y 500 \      --zmin -2000 --zmax 0 --n 201 \      --method rbf --k 50 \      --in-space log10 --out-space linear \      --logx --z-positive-down \      --out-png borehole_profile.png \      --out-csv borehole_profile.csv

Example 2 – Curtain Slice from NPZ
----------------------------------
python -m femtic_polyline_slice \      --npz elements_arrays.npz \      --values-key log10_resistivity \      --polyline-csv path_points.csv \      --zmin -2500 --zmax 0 --nz 201 --ns 301 \      --method rbf --k 60 --kernel gaussian \      --in-space log10 --out-space linear \      --vtk curtain_slice.vts \      --screenshot curtain_slice.png

Notes
-----
- Fixed elements (flag==1) are ignored automatically where available.
- Values default to log10 space unless specified otherwise.
- PyVista viewer supports screenshots and VTK export for ParaView.
