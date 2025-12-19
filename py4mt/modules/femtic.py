#!/usr/bin/env python3
"""
femtic_core.py

Unified FEMTIC utilities for:

- Generating data and model ensembles (perturbed observe.dat and resistivity
  blocks) driven by Gaussian covariance / precision constructions.
- Building and analysing roughness matrices and prior covariance proxies based
  on FEMTIC's user-defined roughening_matrix files.
- Gaussian sampling with precision matrix Q = R.T @ R (+ λI) using iterative
  solvers and low-rank eigendecompositions.
- Converting between FEMTIC mesh/dat formats and compact NPZ model files:
    * mesh.dat + resistivity_block_iterX.dat → NPZ
    * NPZ → mesh.dat + resistivity_block_iterX.dat
    * NPZ → VTK / VTU unstructured grids (PyVista).

The goal is to have a single importable (and CLI-callable) module that
collects:

- The ensemble / precision tools from femtic.py.
- The mesh/NPZ conversion tools from femtic_femtic_to_npz.py.
- The NPZ → VTK exporter from femtic_npz_to_vtk.py.
- The NPZ → FEMTIC reconstructor from femtic_npz_to_mesh.py.

Command-line interface
----------------------
The module provides a simple subcommand-style CLI:

    python femtic_core.py femtic-to-npz \\
        --mesh mesh.dat \\
        --rho-block resistivity_block_iter0.dat \\
        --out-npz femtic_model.npz

    python femtic_core.py npz-to-vtk \\
        --npz femtic_model.npz \\
        --out-vtu model.vtu \\
        --out-legacy model.vtk

    python femtic_core.py npz-to-femtic \\
        --npz femtic_model.npz \\
        --mesh-out mesh_reconstructed.dat \\
        --rho-block-out resistivity_block_iter0_reconstructed.dat

All functionality is also available as regular Python functions.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-09
"""
from __future__ import annotations

import os
import sys
import shutil
import time
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Literal,
)

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from numpy.random import Generator, default_rng
from scipy.sparse import isspmatrix, issparse
from scipy.sparse.linalg import LinearOperator, cg, eigsh, bicgstab, spilu

# Optional but kept for compatibility with earlier versions
import joblib  # noqa: F401


# ============================================================================
# SECTION 1: directory & data / model ensemble utilities (from femtic.py)
# ============================================================================


def generate_directories(
    dir_base: str = "./ens_",
    templates: str = "",
    file_list: Sequence[str] = (
        "control.dat",
        "observe.dat",
        "mesh.dat",
        "resistivity_block_iter0.dat",
        "distortion_iter0.dat",
        "run_dub.sh",
        "run_oar_sh",
    ),
    n_samples: int = 1,
    fromto: Optional[Tuple[int, int]] = None,
    out: bool = True,
) -> list[str]:
    """
    Create ensemble directories and copy template files into each of them.

    Parameters
    ----------
    dir_base : str, optional
        Base name for ensemble directories (e.g. "./ens_").
    templates : str, optional
        Directory prefix from which the template files are copied.
    file_list : sequence of str
        Names of files to copy into each ensemble directory.
    n_samples : int
        Number of ensemble members if ``fromto`` is None.
    fromto : (int, int), optional
        Explicit start/stop indices (Python-style: start, stop) for ensemble
        member numbering. If None, indices range from 0..n_samples-1.
    out : bool, optional
        If True, print status messages.

    Returns
    -------
    dir_list : list of str
        List of created ensemble directory paths.
    """
    if fromto is None:
        from_to = np.arange(n_samples)
    else:
        from_to = np.arange(fromto[0], fromto[1])

    dir_list: list[str] = []
    for iens in from_to:
        directory = f"{dir_base}{iens}/"
        os.makedirs(directory, exist_ok=True)
        copy_files(file_list, directory, templates)
        dir_list.append(directory)

    if out:
        print("list of directories:")
        print(dir_list)

    return dir_list


def copy_files(filelist: Sequence[str], directory: str, templates: str) -> None:
    """
    Copy a list of files from `templates` prefix into `directory`.

    Parameters
    ----------
    filelist : sequence of str
        Files to copy.
    directory : str
        Target directory.
    templates : str
        Template directory (prefix) for input files.
    """
    for fname in filelist:
        src = templates + fname
        shutil.copy2(src, directory)


def generate_data_ensemble(
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[Tuple[int, int]] = None,
    file_in: str = "observe.dat",
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    method: str = "add",
    errors: Sequence[Sequence[float]] | Sequence[list] = (),
    out: bool = True,
) -> list[str]:
    """
    Generate an ensemble of perturbed observation files in ensemble directories.

    For each ensemble member i, reads
        dir_base + f"{i}/{file_in}"
    makes a backup <file>_orig.dat,
    and calls :func:`modify_data` to inject Gaussian noise.

    Parameters
    ----------
    dir_base : str
        Base directory for ensemble members (e.g. "./ens_").
    n_samples : int
        Number of ensemble members if ``fromto`` is None.
    fromto : (int, int), optional
        Explicit start/end indices (Python-style).
    file_in : str
        Name of the observation file in each ensemble directory.
    draw_from : sequence
        Currently kept for future extension; noise is driven by errors.
    method : str
        Placeholder for different perturbation strategies (currently unused).
    errors : sequence
        Error specifications forwarded to :func:`modify_data`.
    out : bool
        If True, print status messages.

    Returns
    -------
    obs_list : list of str
        Paths to the perturbed observation files.
    """
    if fromto is None:
        fromto_arr = np.arange(n_samples)
    else:
        fromto_arr = np.arange(fromto[0], fromto[1])

    obs_list: list[str] = []
    for iens in fromto_arr:
        file = f"{dir_base}{iens}/{file_in}"
        shutil.copy(file, file.replace(".dat", "_orig.dat"))
        modify_data(
            template_file=file,
            draw_from=draw_from,
            method=method,
            errors=errors,
            out=out,
        )
        obs_list.append(file)

    if out:
        print("list of perturbed observation files:")
        print(obs_list)

    return obs_list


def modify_data(
    template_file: str = "observe.dat",
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    method: str = "add",
    errors: Sequence[Sequence[float]] | Sequence[list] = ([], [], []),
    out: bool = True,
) -> None:
    """
    Modify an MT/VTF/PT FEMTIC-style observe.dat in-place by drawing perturbed data.

    The function parses the FEMTIC data layout:

    - header line with obs_type and number of sites
    - data blocks (MT, VTF, or PT)
    - site blocks with frequencies and data + error columns

    For each datum d with associated error σ:

        d_new ~ N(d, σ)

    Optionally, relative errors can be (re)set from the ``errors`` parameter.

    Parameters
    ----------
    template_file : str
        Path to observe.dat.
    draw_from : sequence
        Noise distribution specification (currently fixed to normal).
    method : str
        Placeholder for different perturbation strategies (unused).
    errors : sequence of length 3
        [errors_MT, errors_VTF, errors_PT], each itself a list/array specifying
        relative errors that will be used to overwrite existing error columns
        before drawing perturbed values.
    out : bool
        If True, print status messages.

    Notes
    -----
    The implementation mirrors the original femtic.py logic and is verbose
    in its printing when ``out=True``. It assumes the usual FEMTIC text
    layout for MT/VTF/PT observation files.
    """
    if template_file is None:
        template_file = "observe.dat"

    with open(template_file, "r") as file:
        content = file.readlines()

    line0 = content[0].split()
    obs_type = line0[0]

    # locate data blocks
    start_lines_datablock: list[int] = []
    for number, line in enumerate(content, 0):
        l = line.split()
        if len(l) == 2:
            start_lines_datablock.append(number)
            if out:
                print(" data block", l[0], "with", l[1], "sites begins at line", number)
        if "END" in l:
            start_lines_datablock.append(number - 1)
            if out:
                print(" no further data block in file")

    num_datablock = len(start_lines_datablock) - 1
    for block in np.arange(num_datablock):
        start_block = start_lines_datablock[block]
        end_block = start_lines_datablock[block + 1]
        data_block = content[start_block:end_block]

        if out:
            print(np.shape(data_block))
        start_lines_site: list[int] = []
        num_freqs: list[int] = []
        for number, line in enumerate(data_block, 0):
            l = line.split()
            if len(l) == 4:
                if out:
                    print(l)
                start_lines_site.append(number)
                num_freqs.append(int(data_block[number + 1].split()[0]))
                if out:
                    print("  site", l[0], "begins at line", number)
            if "END" in l:
                start_lines_datablock.append(number - 1)
                if out:
                    print(" no further site block in file")
        if out:
            print("\n")

        num_sites = len(start_lines_site)
        for site in np.arange(num_sites):
            start_site = start_lines_site[site]
            end_site = start_site + num_freqs[site] + 2
            site_block = data_block[start_site:end_site]

            if "MT" in obs_type:
                set_errors_mt = len(errors[0]) != 0
                dat_length = 8
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs: list[list[float]] = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                if set_errors_mt:
                    new_errors = np.asarray(errors[0], dtype=float)
                    if out:
                        print("MT errors will be replaced with relative errors:")
                        print(new_errors)
                    for comp in obs:
                        for ii in range(1, dat_length + 1):
                            val = comp[ii]
                            err = val * new_errors
                            comp[ii + dat_length] = err

                for comp in obs:
                    for ii in range(1, dat_length + 1):
                        val = comp[ii]
                        err = comp[ii + dat_length]
                        comp[ii] = np.random.normal(loc=val, scale=err)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"

            elif "VTF" in obs_type:
                set_errors_vtf = len(errors[1]) != 0
                dat_length = 4
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                if set_errors_vtf:
                    new_errors = np.asarray(errors[1], dtype=float)
                    if out:
                        print("VTF errors will be replaced with relative errors:")
                        print(new_errors)
                    for line in obs:
                        for ii in range(1, dat_length + 1):
                            val = line[ii]
                            err = new_errors
                            line[ii + dat_length] = err

                for comp in obs:
                    for ii in range(1, dat_length + 1):
                        val = comp[ii]
                        err = comp[ii + dat_length]
                        comp[ii] = np.random.normal(loc=val, scale=err)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"

            elif "PT" in obs_type:
                set_errors_pt = len(errors[2]) != 0
                dat_length = 4
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                if set_errors_pt:
                    new_errors = np.asarray(errors[2], dtype=float)
                    if out:
                        print("PT errors will be replaced with relative errors:")
                        print(new_errors)
                    for comp in obs:
                        for ii in range(1, dat_length + 1):
                            val = comp[ii]
                            err = new_errors
                            comp[ii + dat_length] = err

                for comp in obs:
                    for ii in range(1, dat_length + 1):
                        val = comp[ii]
                        err = comp[ii + dat_length]
                        comp[ii] = np.random.normal(loc=val, scale=err)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"
            else:
                sys.exit(f"modify_data: {obs_type} not yet implemented! Exit.")

            data_block[start_site:end_site] = site_block

        content[start_block:end_block] = data_block

    with open(template_file, "w") as f:
        f.writelines(content)

    if out:
        print(f"File {template_file} successfully written.")


def generate_model_ensemble(
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[Tuple[int, int]] = None,
    refmod: str = "resistivity_block_iter0.dat",
    q: Optional[scipy.sparse.spmatrix | np.ndarray] = None,
    method: str = "add",
    out: bool = True,
) -> list[str]:
    """
    Generate a resistivity model ensemble based on a precision matrix R (roughness).

    Two sampling strategies are supported:

    - low_rank = True:
        Use :func:`sample_rtr_low_rank` with eigenpairs of Q = R.T @ R
        (currently estimating them internally).
    - low_rank = False:
        Use :func:`sample_rtr_full_rank`.

    The resulting log10-resistivity samples are inserted into
    FEMTIC resistivity_block_iter0.dat files in each ensemble directory.

    Parameters
    ----------
    dir_base : str
        Ensemble base directory (e.g. "./ens_").
    n_samples : int
        Number of samples to generate.
    fromto : (int, int), optional
        Ensemble index range. If None, use 0..n_samples-1.
    refmod : str
        Name of the reference block file inside each ensemble directory.
    q : ndarray or sparse matrix, optional
        Roughness matrix R to define Q = R.T @ R. If None, the low-rank
        branch is currently a placeholder.
    method : {"add", "replace"}
        How to combine samples with existing log10 resistivity:
        - "add": add perturbation to log10(rho_ref).
        - "replace": ignore original and directly use the samples.
    out : bool
        If True, print status messages.

    Returns
    -------
    mod_list : list of str
        Paths to the perturbed resistivity block files.
    """
    low_rank = True
    if low_rank:
        # Placeholder: currently estimates eigpairs from R.T @ R internally.
        # For large problems, pre-compute eigpairs and pass them instead.
        samples = sample_rtr_low_rank(
            q if q is not None else np.eye(4),  # dummy fallback
            n_samples=n_samples,
            n_eig=32,
            sigma2_residual=0.0,
        )
    else:
        if q is None:
            raise ValueError("generate_model_ensemble: q must be provided for full-rank.")
        samples = sample_rtr_full_rank(
            R=q,
            n_samples=n_samples,
            lam=0.0,
        )

    if fromto is None:
        fromto_arr = np.arange(n_samples)
    else:
        fromto_arr = np.arange(fromto[0], fromto[1])

    mod_list: list[str] = []
    for iens, sample in zip(fromto_arr, samples):
        file = f"{dir_base}{iens}/{refmod}"
        shutil.copy(file, file.replace(".dat", "_orig.dat"))
        insert_model(
            template=refmod,
            data=sample,
            data_file=file,
            data_name=f"sample{iens}",
        )
        mod_list.append(file)

    if out:
        print("\nlist of perturbed model files:")
        print(mod_list)

    return mod_list


def read_model(
    model_file: str | Path,
    model_trans: str = "log10",
    out: bool = True,
) -> np.ndarray:
    """
    Read a FEMTIC resistivity_block_iterX.dat and return log10-resistivity vector.

    Parameters
    ----------
    model_file : str or Path
        Path to FEMTIC resistivity block file.
    model_trans : {"log10", "none"}
        If "log10", return log10(rho); otherwise return rho itself.
    out : bool
        If True, print small info.

    Returns
    -------
    model : ndarray, shape (n_cells - 2,)
        Model values in requested space. The first two region indices (air,
        ocean) are skipped, following the convention of the other routines.
    """
    if model_file is None:
        sys.exit("read_model: No model file given! Exit.")

    with open(model_file, "r") as file:
        content = file.readlines()

    nn = [int(tmp) for tmp in content[0].split()]

    values: list[float] = []
    for elem in range(nn[0] + 1, nn[0] + nn[1] + 1):
        x = float(content[elem].split()[1])
        values.append(x)

    model = np.asarray(values, dtype=float)

    if "log10" in model_trans.lower():
        if out:
            print("model is log10 resistivity!")
        model = np.log10(model)

    return model


def modify_model(
    template_file: str = "resistivity_block_iter0.dat",
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    method: str = "add",
    q: Optional[scipy.sparse.spmatrix | np.ndarray] = None,
    decomposed: bool = False,
    regeps: float = 1.0e-8,
    out: bool = True,
) -> None:
    """
    Modify a FEMTIC resistivity_block_iter0.dat in-place by adding random
    perturbations in log10-resistivity.

    Parameters
    ----------
    template_file : str
        Path to block file.
    draw_from : sequence
        Either ["normal", mu, sigma] or ["uniform", a, b].
    method : {"add", "replace"}
        If "add", add the perturbation in log10 domain. If "replace", ignore
        original rho and use the random samples as log10(rho).
    q : sparse or dense matrix, optional
        Placeholder for covariance-based perturbations (not used in this
        simplified version; noise is i.i.d.).
    decomposed : bool
        Placeholder flag for pre-decomposed covariance (unused).
    regeps : float
        Placeholder diagonal regularisation (unused).
    out : bool
        If True, print statistics about perturbations.

    Notes
    -----
    The first two region entries (0 = air, 1 = ocean) are kept fixed,
    matching FEMTIC conventions.
    """
    if template_file is None:
        template_file = "resistivity_block_iter0.dat"

    with open(template_file, "r") as file:
        content = file.readlines()

    nn = [int(tmp) for tmp in content[0].split()]
    n_cells = nn[1]

    if "normal" in str(draw_from[0]).lower():
        samples = np.random.normal(
            loc=float(draw_from[1]),
            scale=float(draw_from[2]),
            size=n_cells - 2,
        )
    else:
        samples = np.random.uniform(
            low=float(draw_from[1]),
            high=float(draw_from[2]),
            size=n_cells - 2,
        )

    new_lines: list[str] = [
        "         0        1.000000e+09   1.000000e-20   1.000000e+20   1.000000e+00         1",
        "         1        2.500000e-01   1.000000e-20   1.000000e+20   1.000000e+00         1",
    ]

    if out:
        print(nn[0], nn[0] + nn[1] - 1, nn[1] - 1, np.shape(samples))

    e_num = 1
    for elem in range(nn[0] + 3, nn[0] + nn[1] + 1):
        e_num += 1
        line_parts = content[elem].split()
        x = float(line_parts[1])

        if "add" in method.lower():
            x_log = np.log10(x) + samples[e_num - 2]
        else:
            x_log = samples[e_num - 2]

        x_new = 10.0 ** x_log
        line = (
            f" {e_num:9d}        {x_new:.6e}   1.000000e-20   1.000000e+20   "
            f"1.000000e+00         0"
        )
        new_lines.append(line)

    new_blob = "\n".join(new_lines) + "\n"

    with open(template_file, "w") as f:
        f.writelines(content[0 : nn[0] + 1])
        f.write(new_blob)

    if out:
        print(f"File {template_file} successfully written.")
        print("Number of perturbations", len(samples))
        print("Average perturbation", np.mean(samples))
        print("StdDev perturbation", np.std(samples))


def insert_model(
    template: str = "resistivity_block_iter0.dat",
    data: np.ndarray | Sequence[float] | None = None,
    data_file: Optional[str] = None,
    data_name: str = "",
    out: bool = True,
) -> None:
    """
    Insert a log10-resistivity vector into a FEMTIC resistivity block file.

    The input `data` is interpreted as log10(ρ) for all regions except the
    first two (air, ocean), which are kept fixed with standard values:

        region 0 → 1e9 Ωm, fixed
        region 1 → 0.25 Ωm, fixed (ocean)

    Parameters
    ----------
    template : str
        Template block file to read header and region mapping from.
    data : array-like, shape (nreg - 2,)
        Log10-resistivity values for regions 2..nreg-1.
    data_file : str, optional
        Output file. If None, overwrite `template`.
    data_name : str
        Optional label (unused, for bookkeeping).
    out : bool
        If True, print summary.
    """
    if data is None:
        sys.exit("insert_model: No data given! Exit.")

    if template is None:
        template = "resistivity_block_iter0.dat"

    if data_file is None:
        data_file = template

    with open(template, "r") as file:
        content = file.readlines()

    nn = [int(tmp) for tmp in content[0].split()]
    n_cells = nn[1]

    size = n_cells - 2
    data_arr = np.asarray(data, dtype=float)
    if data_arr.size != size:
        raise ValueError(
            f"insert_model: expected {size} entries, got {data_arr.size}."
        )

    new_lines: list[str] = [
        "         0        1.000000e+09   1.000000e-20   1.000000e+20   1.000000e+00         1",
        "         1        2.500000e-01   1.000000e-20   1.000000e+20   1.000000e+00         1",
    ]

    if out:
        print(nn[0], nn[0] + nn[1] - 1, nn[1] - 1, np.shape(data_arr))

    rho = np.power(10.0, data_arr)

    e_num = 1
    s_num = -1
    for elem in range(nn[0] + 3, nn[0] + nn[1] + 1):
        e_num += 1
        s_num += 1
        x = rho[s_num]
        line = (
            f" {e_num:9d}        {x:.6e}   1.000000e-20   1.000000e+20   "
            f"1.000000e+00         0"
        )
        new_lines.append(line)

    new_blob = "\n".join(new_lines) + "\n"

    with open(data_file, "w") as f:
        f.writelines(content[0 : nn[0] + 1])
        f.write(new_blob)

    if out:
        print(f"File {data_file} successfully written.")
        print("Number of data replaced", len(data_arr))


def modify_data_fcn(
    template_file: str = "observe.dat",
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    scalfac: float = 1.0,
    out: bool = True,
) -> None:
    """
    Simpler variant of :func:`modify_data` using existing error columns scaled
    by a factor `scalfac`.

    Parameters
    ----------
    template_file : str
        FEMTIC observe.dat path.
    draw_from : sequence
        Noise distribution spec (currently normal only).
    scalfac : float
        Scale factor applied to existing error columns before sampling.
    out : bool
        If True, print status messages.

    Notes
    -----
    This function preserves existing relative error structure, only scales it
    and draws noise accordingly.
    """
    if template_file is None:
        template_file = "observe.dat"

    with open(template_file, "r") as file:
        content = file.readlines()

    line0 = content[0].split()
    obs_type = line0[0]

    start_lines_datablock: list[int] = []
    for number, line in enumerate(content, 0):
        l = line.split()
        if len(l) == 2:
            start_lines_datablock.append(number)
            if out:
                print(" data block", l[0], "with", l[1], "sites begins at line", number)
        if "END" in l:
            start_lines_datablock.append(number - 1)
            if out:
                print(" no further data block in file")

    num_datablock = len(start_lines_datablock) - 1
    for block in np.arange(num_datablock):
        start_block = start_lines_datablock[block]
        end_block = start_lines_datablock[block + 1]
        data_block = content[start_block:end_block]

        if out:
            print(np.shape(data_block))
        start_lines_site: list[int] = []
        num_freqs: list[int] = []
        for number, line in enumerate(data_block, 0):
            l = line.split()
            if len(l) == 4:
                if out:
                    print(l)
                start_lines_site.append(number)
                num_freqs.append(int(data_block[number + 1].split()[0]))
                if out:
                    print("  site", l[0], "begins at line", number)
            if "END" in l:
                start_lines_datablock.append(number - 1)
                if out:
                    print(" no further site block in file")
        if out:
            print("\n")

        num_sites = len(start_lines_site)
        for site in np.arange(num_sites):
            start_site = start_lines_site[site]
            end_site = start_site + num_freqs[site] + 2
            site_block = data_block[start_site:end_site]

            if "MT" in obs_type:
                dat_length = 8
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs: list[list[float]] = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                for line in obs:
                    for ii in range(1, dat_length + 1):
                        val = line[ii]
                        err = line[ii + dat_length] * scalfac
                        line[ii] = np.random.normal(loc=val, scale=err)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"

            elif "VTF" in obs_type:
                dat_length = 4
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                for line in obs:
                    for ii in range(1, dat_length + 1):
                        val = line[ii]
                        err = line[ii + dat_length] * scalfac
                        line[ii] = np.random.normal(loc=val, scale=err)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"
            else:
                sys.exit(f"modify_data_fcn: {obs_type} not yet implemented! Exit.")

            data_block[start_site:end_site] = site_block

        content[start_block:end_block] = data_block

    with open(template_file, "w") as f:
        f.writelines(content)

    if out:
        print(f"File {template_file} successfully written.")


def get_femtic_sorted(files: Sequence[str], out: bool = True) -> list[str]:
    """
    Sort FEMTIC sensitivity matrix files of the form 'sensMatFreqXXXX'
    by their integer suffix and return a sorted list.

    Parameters
    ----------
    files : sequence of str
        File names.
    out : bool
        If True, print the sorted list.

    Returns
    -------
    listfiles : list of str
        Sorted file names.
    """
    numbers = [int(f[11:]) for f in files]
    numbers = sorted(numbers)

    listfiles = [f"sensMatFreq{ii}" for ii in numbers]
    if out:
        print(listfiles)

    return listfiles


def get_femtic_sites(
    imp_file: str = "result_MT.txt",
    vtf_file: str = "result_VTF.txt",
    pt_file: str = "results_PT.txt",
) -> None:
    """
    Generate sites_XXX.txt files from FEMTIC results files.

    Parameters
    ----------
    imp_file : str
        MT result file -> sites_MT.txt
    vtf_file : str
        VTF result file -> sites_VTF.txt
    pt_file : str
        PT result file -> sites_PT.txt

    Notes
    -----
    The function mirrors the original femtic.py behaviour but fixes some
    minor robustness issues. It writes files where each line is

        sitename sitename

    for each unique site encountered in the corresponding results file.
    """
    # MT
    if len(imp_file) > 0 and os.path.exists(imp_file):
        with open(imp_file, "r") as filein_imp:
            site = ""
            fileout_imp = open(imp_file.replace("results", "sites"), "w")
            filein_imp.readline()
            for line in filein_imp:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_imp.write(nextsite + " " + nextsite + "\n")
                    site = nextsite
            fileout_imp.close()
    else:
        if len(imp_file) > 0:
            print(imp_file, "does not exist!")
        else:
            print("imp_file not defined!")

    # VTF
    if len(vtf_file) > 0 and os.path.exists(vtf_file):
        with open(vtf_file, "r") as filein_vtf:
            site = ""
            fileout_vtf = open(vtf_file.replace("results", "sites"), "w")
            filein_vtf.readline()
            for line in filein_vtf:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_vtf.write(nextsite + " " + nextsite + "\n")
                    site = nextsite
            fileout_vtf.close()
    else:
        if len(vtf_file) > 0:
            print(vtf_file, "does not exist!")
        else:
            print("vtf_file not defined!")

    # PT
    if len(pt_file) > 0 and os.path.exists(pt_file):
        with open(pt_file, "r") as filein_pt:
            site = ""
            fileout_pt = open(pt_file.replace("results", "sites"), "w")
            filein_pt.readline()
            for line in filein_pt:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_pt.write(nextsite + " " + nextsite + "\n")
                    site = nextsite
            fileout_pt.close()
    else:
        if len(pt_file) > 0:
            print(pt_file, "does not exist!")
        else:
            print("pt_file not defined!")


def get_femtic_data(
    data_file: str,
    site_file: str,
    data_type: str = "rhophas",
    out: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Read FEMTIC results (MT, VTF or PT) and site information.

    Parameters
    ----------
    data_file : str
        results_XXX.txt file from FEMTIC.
    site_file : str
        site metadata CSV-like file with columns:
            name, lat, lon, elev, num
    data_type : {"rhophas", "imp", "vtf", "pt"}
        Type of FEMTIC results file.
    out : bool
        If True, print debugging info.

    Returns
    -------
    data_dict : dict
        Dictionary containing:
            - "sites": site indices (0-based)
            - "frq": frequencies
            - "per": periods
            - "lat", "lon", "elv"
            - "num": site index (0-based)
            - "nam": site name
            - "cal", "obs", "err" (arrays of appropriate shape)
    """
    data: list[list[float]] = []
    with open(data_file, "r") as f:
        iline = -1
        for line in f:
            iline += 1
            if iline == 0:
                continue
            l = [float(x) for x in line.split()]
            l[0] = int(l[0])
            data.append(l)
    data_arr = np.asarray(data, dtype=float)

    info: list[list[float | str]] = []
    with open(site_file, "r") as f:
        for line in f:
            l = line.split(",")
            l[1] = float(l[1])
            l[2] = float(l[2])
            l[3] = float(l[3])
            l[4] = int(l[4])
            info.append(l)
    info_arr = np.asarray(info, dtype=object)

    sites = np.unique(data_arr[:, 0]).astype(int) - 1

    head_dict: Dict[str, np.ndarray] = {
        "sites": sites,
        "frq": data_arr[:, 1],
        "per": 1.0 / data_arr[:, 1],
        "lat": np.asarray(info_arr[:, 1], dtype=float),
        "lon": np.asarray(info_arr[:, 2], dtype=float),
        "elv": np.asarray(info_arr[:, 3], dtype=float),
        "num": data_arr[:, 0].astype(int) - 1,
        "nam": np.asarray(info_arr[:, 0])[sites.astype(int)],
    }

    dtyp = data_type.lower()
    if "rhophas" in dtyp:
        type_dict = {
            "cal": data_arr[:, 2:10],
            "obs": data_arr[:, 10:18],
            "err": data_arr[:, 18:26],
        }
    elif "imp" in dtyp:
        ufact = 1.0e-4 / (4.0 * np.pi)
        type_dict = {
            "cal": ufact * data_arr[:, 2:10],
            "obs": ufact * data_arr[:, 10:18],
            "err": ufact * data_arr[:, 18:26],
        }
    elif "vtf" in dtyp:
        type_dict = {
            "cal": data_arr[:, 2:6],
            "obs": data_arr[:, 6:10],
            "err": data_arr[:, 10:14],
        }
    elif "pt" in dtyp:
        type_dict = {
            "cal": data_arr[:, 2:6],
            "obs": data_arr[:, 6:10],
            "err": data_arr[:, 10:14],
        }
    else:
        sys.exit(f"get_femtic_data: data type {dtyp} not implemented! Exit.")

    data_dict = {**head_dict, **type_dict}
    if out:
        print("get_femtic_data: loaded", data_type, "with shape", data_arr.shape)
    return data_dict


def centroid_tetrahedron(nodes: np.ndarray) -> np.ndarray:
    """
    Compute centroid of a tetrahedron given node coordinates.

    Parameters
    ----------
    nodes : ndarray, shape (4, 3) or (3, 4)
        Node coordinates of the tetrahedron. Both shapes are accepted;
        the function will transpose if needed.

    Returns
    -------
    centre : ndarray, shape (3,)
        Centroid coordinates.
    """
    arr = np.asarray(nodes, dtype=float)
    if arr.shape == (3, 4):
        arr = arr.T
    if arr.shape != (4, 3):
        sys.exit("centroid_tetrahedron: Nodes shape must be (4,3) or (3,4)! Exit.")
    centre = np.mean(arr, axis=0)
    return centre


# ============================================================================
# SECTION 2: roughness / prior covariance / matrix tools (from femtic.py)
# ============================================================================


def get_roughness(
    filerough: str = "roughening_matrix.out",
    regeps: Optional[float] = None,
    spformat: str = "csc",
    out: bool = True,
) -> scipy.sparse.spmatrix:
    """
    Read FEMTIC roughening_matrix.dat and build sparse roughness matrix R.

    Parameters
    ----------
    filerough : str
        Path to roughening_matrix.dat file.
    regeps : float or None
        If not None, add regeps * I to R.
    spformat : {"csr", "csc", "coo"}
        Sparse format of the returned matrix.
    out : bool
        If True, print timings and matrix statistics.

    Returns
    -------
    R : sparse matrix
        Roughness matrix in chosen sparse format.

    Notes
    -----
    The function implements the logic implied by FEMTIC's C++ code for
    user-defined roughening matrices. It first counts blocks with zero
    non-zeros, then builds the triplet arrays (row, col, val).
    """
    from scipy.sparse import csr_array, csc_array, coo_array, eye as eye_array

    start = time.perf_counter()
    if out:
        print("get_roughness: Reading from", filerough)
    irow: list[int] = []
    icol: list[int] = []
    vals: list[float] = []

    with open(filerough, "r") as file:
        content = file.readlines()

    num_elem = int(content[0].split()[0])
    if out:
        print("get_roughness: File read:", time.perf_counter() - start, "s")
        print("get_roughness: Number of elements:", num_elem)

    iline = 0
    zeros = 0
    # first pass: count zero blocks
    while iline < len(content) - 1:
        iline += 1
        ele = int(content[iline].split()[0])
        nel = int(content[iline + 1].split()[0])
        if nel == 0:
            iline += 1
            zeros += 1
            if out:
                print("passed", ele, nel, iline)
        else:
            iline += 2
    if out:
        print("Zero elements:", zeros)

    # second pass: collect non-zero triplets
    start = time.perf_counter()
    iline = 0
    while iline < len(content) - 1:
        iline += 1
        ele = int(content[iline].split()[0])
        nel = int(content[iline + 1].split()[0])
        if nel == 0:
            iline += 1
            continue
        else:
            irow += [ele - zeros] * nel
            col = [int(x) - zeros for x in content[iline + 1].split()[1:]]
            icol += col
            val = [float(x) for x in content[iline + 2].split()]
            vals += val
            iline += 2

    irow_arr = np.asarray(irow, dtype=int)
    icol_arr = np.asarray(icol, dtype=int)
    vals_arr = np.asarray(vals, dtype=float)

    R = coo_array((vals_arr, (irow_arr, icol_arr)))
    if out:
        print(R.shape)
        print("get_roughness: R sparse format is", R.format)

    if regeps is not None:
        R = R + regeps * eye_array(R.shape[0], format=R.format)
        if out:
            print(regeps, "added to diag(R)")

    if out:
        print("get_roughness: R generated:", time.perf_counter() - start, "s")
        print("get_roughness: R sparse format is", R.format)
        print(R.shape, R.nnz)

    if "csc" in spformat.lower():
        R = csc_array((vals_arr, (irow_arr, icol_arr)))
    elif "csr" in spformat.lower():
        R = csr_array((vals_arr, (irow_arr, icol_arr)))
    else:
        R = coo_array((vals_arr, (irow_arr, icol_arr)))

    if out:
        print("get_roughness: Output sparse format:", spformat)
        print("get_roughness: R sparse format is", R.format)
        print(R.shape, R.nnz)
        print(R.nnz, "nonzeros, ", 100 * R.nnz / R.shape[0] ** 2, "%")
        print("get_roughness: Done!\n\n")

    return R


def make_prior_cov(
    rough: scipy.sparse.spmatrix,
    regeps: float = 1.0e-5,
    spformat: str = "csr",
    spthresh: float = 1.0e-4,
    spfill: float = 10.0,
    spsolver: Optional[str] = "ilu",
    spmeth: str = "basic,area",
    outmatrix: str = "invRTR",
    nthreads: int = 16,
    out: bool = True,
) -> scipy.sparse.spmatrix | np.ndarray:
    """
    Generate a prior covariance proxy M from a roughness matrix R.

    Strategy
    --------
    1. Stabilise by adding a small diagonal (regeps).
    2. Invert R (approximately) using either:
       - sparse LU ("slu"), or
       - incomplete LU ("ilu").
    3. Optionally sparsify the inverse via :func:`matrix_reduce`.
    4. Build

           M = invR @ invR.T        if outmatrix contains "rtr"
           M = invR                 otherwise

    Parameters
    ----------
    rough : sparse matrix
        Roughness matrix R.
    regeps : float
        Small diagonal added to R for stability.
    spformat : {"csr", "csc", "coo"}
        Sparse format used in some steps.
    spthresh : float
        Drop tolerance for ILU and for matrix_reduce.
    spfill : float
        Fill-factor for ILU.
    spsolver : {"slu", "ilu"}
        Choice of sparse direct solve / factorization.
    spmeth : str
        Drop-rule string for ILU; currently not used explicitly.
    outmatrix : {"invR", "invRTR", "invRTR_deco"}
        Output matrix type.
    nthreads : int
        Thread limit for underlying BLAS/LAPACK where supported.
    out : bool
        If True, print info.

    Returns
    -------
    M : sparse or dense array
        Prior covariance proxy.
    """
    from scipy.sparse import (
        csr_array,
        csc_array,
        coo_array,
        eye as eye_array,
        issparse,
    )
    from threadpoolctl import threadpool_limits

    if rough is None:
        sys.exit("make_prior_cov: No roughness matrix given! Exit.")

    if not issparse(rough):
        sys.exit("make_prior_cov: Roughness matrix is not sparse! Exit.")

    start = time.perf_counter()

    if out:
        print("make_prior_cov: Shape of input roughness is", rough.shape)
        print("make_prior_cov: Format of input roughness is", rough.format)

    if regeps is not None:
        rough = rough + regeps * eye_array(rough.shape[0], format=spformat.lower())
        if out:
            print(regeps, "added to diag(R)")

    if spsolver is None:
        sys.exit("make_prior_cov: spsolver must be 'slu' or 'ilu'.")

    if "slu" in spsolver.lower():
        from scipy.sparse.linalg import spsolve

        R = csc_array(rough)
        RHS = eye_array(R.shape[0], format=R.format)
        with threadpool_limits(limits=nthreads):
            invR = spsolve(R, RHS)

    elif "ilu" in spsolver.lower():
        R = csc_array(rough)
        RHS = eye_array(R.shape[0], format=R.format)
        beg = time.perf_counter()
        with threadpool_limits(limits=nthreads):
            iluR = spilu(R, drop_tol=spthresh, fill_factor=spfill)
            if out:
                print("spilu decomposed:", time.perf_counter() - beg, "s")
            beg = time.perf_counter()
            invR = iluR.solve(RHS.toarray())
            if out:
                print("spilu solved:", time.perf_counter() - beg, "s")
    else:
        sys.exit(f"make_prior_cov: solver {spsolver} not available! Exit")

    if out:
        print("make_prior_cov: invR generated:", time.perf_counter() - start, "s")
        print("make_prior_cov: invR type", type(invR))

    if spthresh is not None:
        invR = matrix_reduce(
            M=invR,
            howto="relative",
            spthresh=spthresh,
            spformat=spformat,
            out=out,
        )

    M = invR
    if "rtr" in outmatrix.lower():
        M = invR @ invR.T

    if out:
        print("make_prior_cov: M generated:", time.perf_counter() - start, "s")
        print("make_prior_cov: M is", outmatrix)
        print("make_prior_cov: M type", type(M))
        print("make_prior_cov:  Done!\n\n")
    return M


def prune_inplace(M: scipy.sparse.spmatrix, threshold: float) -> scipy.sparse.spmatrix:
    """
    In-place pruning of small entries in a CSR sparse matrix.

    Parameters
    ----------
    M : sparse matrix
        Sparse matrix to prune.
    threshold : float
        Entries with |M_ij| < threshold are set to zero.

    Returns
    -------
    M : sparse matrix
        The pruned matrix (same object, modified in-place).
    """
    from scipy.sparse import csr_array, issparse

    if issparse(M):
        if M.format != "csr":
            M = M.tocsr()
    else:
        M = csr_array(M)

    mask = np.abs(M.data) < threshold
    if mask.any():
        M.data[mask] = 0.0
        M.eliminate_zeros()
    return M


def prune_rebuild(M: scipy.sparse.spmatrix, threshold: float) -> scipy.sparse.spmatrix:
    """
    Rebuild a CSR sparse matrix from COO representation after pruning.

    Parameters
    ----------
    M : sparse matrix
        Matrix to prune.
    threshold : float
        Entries with |M_ij| < threshold are dropped.

    Returns
    -------
    M_csr : sparse matrix
        New CSR matrix with pruned entries.
    """
    from scipy.sparse import csr_array

    coo = M.tocoo()
    absdata = np.abs(coo.data)
    keep = absdata >= threshold
    if not keep.all():
        return csr_array(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=M.shape,
        )
    else:
        return M.tocsr()


def dense_to_csr(
    M: np.ndarray,
    threshold: float = 0.0,
    chunk_rows: int = 1000,
    dtype: Optional[np.dtype] = None,
) -> scipy.sparse.spmatrix:
    """
    Convert a dense matrix to CSR, dropping entries below a threshold.

    Parameters
    ----------
    M : ndarray
        Dense matrix.
    threshold : float
        Entries with |M_ij| <= threshold are dropped.
    chunk_rows : int
        Process rows in chunks of this size to reduce memory peaks.
    dtype : numpy dtype, optional
        Output dtype; if None, use M.dtype.

    Returns
    -------
    M_csr : sparse matrix
        CSR matrix.
    """
    from scipy.sparse import csr_array

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []
    nrows = M.shape[0]
    for r0 in range(0, nrows, chunk_rows):
        r1 = min(nrows, r0 + chunk_rows)
        block = M[r0:r1]
        mask = np.abs(block) > threshold
        rr, cc = np.nonzero(mask)
        rows_list.append((rr + r0).astype(np.int64))
        cols_list.append(cc.astype(np.int64))
        data_list.append(
            block[rr, cc].astype(dtype if dtype is not None else M.dtype)
        )

    if not rows_list:
        return csr_array(M.shape, dtype=dtype if dtype is not None else M.dtype)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    return csr_array((data, (rows, cols)), shape=M.shape)


def save_spilu(filename: str = "ILU.npz", ILU=None) -> None:
    """
    Save a SciPy ILU decomposition object to a single .npz file.

    Parameters
    ----------
    filename : str
        Output npz file.
    ILU : object
        ILU object returned by scipy.sparse.linalg.spilu.

    Notes
    -----
    To restore, use :func:`load_spilu`.
    """
    if ILU is None:
        sys.exit("save_spilu: No ILU object given! Exit.")

    np.savez(
        filename,
        L_data=ILU.L.data,
        L_indices=ILU.L.indices,
        L_indptr=ILU.L.indptr,
        L_shape=ILU.L.shape,
        U_data=ILU.U.data,
        U_indices=ILU.U.indices,
        U_indptr=ILU.U.indptr,
        U_shape=ILU.U.shape,
        perm_r=ILU.perm_r,
        perm_c=ILU.perm_c,
    )


def load_spilu(filename: str = "ILU.npz"):
    """
    Load ILU decomposition components from npz file.

    Parameters
    ----------
    filename : str
        npz file created by :func:`save_spilu`.

    Returns
    -------
    L, U, perm_r, perm_c
        Components of the ILU decomposition.
    """
    from scipy.sparse import csc_array

    data = np.load(filename)
    L = csc_array(
        (data["L_data"], data["L_indices"], data["L_indptr"]),
        shape=tuple(data["L_shape"]),
    )
    U = csc_array(
        (data["U_data"], data["U_indices"], data["U_indptr"]),
        shape=tuple(data["U_shape"]),
    )
    perm_r = data["perm_r"]
    perm_c = data["perm_c"]
    return L, U, perm_r, perm_c


def matrix_reduce(
    M: np.ndarray | scipy.sparse.spmatrix,
    howto: str = "relative",
    spformat: str = "csr",
    spthresh: float = 1.0e-6,
    prune: str = "rebuild",
    out: bool = True,
) -> scipy.sparse.spmatrix:
    """
    Reduce a (possibly dense) matrix to sparse form by dropping small entries.

    Parameters
    ----------
    M : array_like or sparse matrix
        Matrix to sparsify.
    howto : {"relative", "absolute"}
        If "relative", use spthresh * max(|M|) as threshold;
        if "absolute", use spthresh directly.
    spformat : {"csr", "csc", "coo"}
        Output sparse format.
    spthresh : float
        Threshold parameter.
    prune : {"inplace", "rebuild"}
        Strategy for pruning; currently "rebuild" is more robust.
    out : bool
        If True, print info.

    Returns
    -------
    M_sp : sparse matrix
        Sparsified matrix.
    """
    from scipy.sparse import csr_array, csc_array, coo_array, issparse

    if M is None:
        sys.exit("matrix_reduce: no matrix given! Exit.")

    if issparse(M):
        M_sp = M.tocsr()
        if out:
            print("matrix_reduce: Matrix is sparse.")
            print("matrix_reduce: Type:", type(M_sp))
            print("matrix_reduce: Format:", M_sp.format)
            print("matrix_reduce: Shape:", M_sp.shape)
    else:
        M_sp = csr_array(M)
        if out:
            print("matrix_reduce: Matrix is dense.")
            print("matrix_reduce: Type:", type(M_sp))
            print("matrix_reduce: Shape:", M_sp.shape)

    n = M_sp.shape[0]
    if out:
        print(
            "matrix_reduce:",
            M_sp.nnz,
            "nonzeros, ",
            100 * M_sp.nnz / n ** 2,
            "%",
        )

    if "abs" in howto.lower():
        threshold = spthresh
    else:
        maxM = np.max(np.abs(M_sp.data))
        threshold = spthresh * maxM

    if issparse(M_sp):
        if "in" in prune:
            M_sp = prune_inplace(M_sp, threshold)
        else:
            M_sp = prune_rebuild(M_sp, threshold)
    else:
        M_sp = dense_to_csr(M_sp, threshold=threshold, chunk_rows=10000)

    if "csr" in spformat.lower():
        M_sp = M_sp.tocsr()
    elif "csc" in spformat.lower():
        M_sp = M_sp.tocsc()
    else:
        M_sp = M_sp.tocoo()

    if out:
        print("matrix_reduce: New Format:", M_sp.format)
        print("matrix_reduce: Shape:", M_sp.shape)
        print(
            "matrix_reduce:",
            M_sp.nnz,
            "nonzeros, ",
            100 * M_sp.nnz / n ** 2,
            "%",
        )
        print("matrix_reduce: Done!\n\n")
    return M_sp


def check_sparse_matrix(M: scipy.sparse.spmatrix, condition: bool = True) -> None:
    """
    Print diagnostic information about a sparse matrix.

    Parameters
    ----------
    M : sparse matrix
        Matrix to be tested.
    condition : bool
        Placeholder (not used).

    Returns
    -------
    None
    """
    from scipy.sparse import issparse

    if M is None:
        sys.exit("check_sparse_matrix: No matrix given! Exit.")

    if not issparse(M):
        sys.exit("check_sparse_matrix: Matrix is not sparse! Exit.")

    print("check_sparse_matrix: Type:", type(M))
    print("check_sparse_matrix: Format:", M.format)
    print("check_sparse_matrix: Shape:", M.shape)
    print(
        "check_sparse_matrix:",
        M.nnz,
        "nonzeros, ",
        100 * M.nnz / M.shape[0] ** 2,
        "%",
    )

    if M.shape[0] == M.shape[1]:
        print("check_sparse_matrix: Matrix is square!")
        test = M - M.T
        print("   R-R^T max/min:", test.max(), test.min())
        if test.max() + test.min() == 0.0:
            print("check_sparse_matrix: Matrix is symmetric!")
        else:
            print("check_sparse_matrix: Matrix is not symmetric!")

    maxaM = np.amax(np.abs(M))
    minaM = np.amin(np.abs(M))
    print("check_sparse_matrix: M max/min:", M.max(), M.min())
    print("check_sparse_matrix: M abs max/min:", maxaM, minaM)

    if np.any(np.abs(M.diagonal()) == 0):
        print("check_sparse_matrix: M diagonal element is 0!")
        print(np.where(np.abs(M.diagonal()) == 0))

    print("check_sparse_matrix: Done!\n\n")


# ============================================================================
# ============================================================================
# SECTION 3: Gaussian sampling with Q = R.T @ R (+ λI)
# ============================================================================

import warnings


def build_rtr_operator(
    R: np.ndarray | scipy.sparse.spmatrix,
    lam: float = 0.0,
) -> LinearOperator:
    """Create a LinearOperator representing Q = R.T @ R + lam * I.

    This is the **matrix-free** representation used by iterative solvers.

    Parameters
    ----------
    R : array_like or scipy.sparse.spmatrix, shape (m, n)
        Matrix defining the precision via Q = R.T @ R. In FEMTIC workflows,
        this is typically a roughness / regularisation operator.
    lam : float, optional
        Diagonal Tikhonov shift. If ``lam > 0`` then
        Q = R.T @ R + lam * I is strictly positive definite.

    Returns
    -------
    Q_op : scipy.sparse.linalg.LinearOperator, shape (n, n)
        Linear operator that applies Q to a vector.

    Notes
    -----
    The matvec evaluates::

        y = R @ x
        z = R.T @ y
        z += lam * x

    without explicitly forming Q.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    _, n = R.shape

    def matvec(x: np.ndarray) -> np.ndarray:
        """Compute z = (R.T @ R + lam * I) @ x."""
        y = R @ x
        z = R.T @ y
        if lam != 0.0:
            z = z + lam * x
        return np.asarray(z, dtype=np.float64)

    return LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def _diag_rtr(R: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    """Compute diag(R.T @ R) without forming R.T @ R.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)

    Returns
    -------
    d : ndarray, shape (n,)
        Column-wise sum of squares of R, i.e. diag(R.T R).

    Notes
    -----
    For sparse R, this uses element-wise square and column sum. For dense
    R, it uses ``(R**2).sum(axis=0)``.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    if scipy.sparse.issparse(R):
        # (R.multiply(R)) keeps sparsity pattern, column sum returns (1, n).
        d = np.asarray(R.multiply(R).sum(axis=0)).ravel()
        return d.astype(np.float64, copy=False)
    arr = np.asarray(R, dtype=np.float64)
    return np.sum(arr * arr, axis=0)


def make_rtr_preconditioner(
    R: np.ndarray | scipy.sparse.spmatrix,
    lam: float = 0.0,
    *,
    precond: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
) -> Optional[LinearOperator]:
    """Build a CG-compatible preconditioner M ≈ Q^{-1} for Q = R.T@R + lam*I.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Operator defining Q.
    lam : float, optional
        Diagonal shift used in Q.
    precond : {None, 'jacobi', 'ilu', 'amg', 'identity'}, optional
        Preconditioner type:

        - None: no preconditioning (M = None).
        - 'jacobi': diagonal (inverse) preconditioner using diag(R.T R) + lam.
          **Does not form Q**.
        - 'ilu': incomplete LU on sparse Q (forms Q as sparse).
        - 'amg': algebraic multigrid on sparse Q (requires `pyamg`, forms Q).
        - 'identity': explicit identity operator.

    precond_kwargs : dict, optional
        Options for the chosen preconditioner.

        For 'jacobi':
            - min_diagonal (float): floor on the diagonal before inversion
              (default 1e-30).

        For 'ilu' (scipy.sparse.linalg.spilu):
            - drop_tol (float)
            - fill_factor (float)
            - diag_pivot_thresh (float)
            - permc_spec (str)

        For 'amg' (pyamg.smoothed_aggregation_solver):
            - any kwargs accepted by pyamg, plus:
            - tol (float): solve tolerance used per application (default 1e-8)
            - maxiter (int): max iterations per application (default 1)

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator or None
        Preconditioner suitable for SciPy iterative solvers.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    if precond is None:
        return None

    key = str(precond).strip().lower()
    opts = {} if precond_kwargs is None else dict(precond_kwargs)

    if key in {"none"}:
        return None

    if key in {"identity", "i"}:
        n = R.shape[1]
        return LinearOperator((n, n), matvec=lambda x: x, dtype=np.float64)

    if key in {"jacobi", "diag"}:
        d = _diag_rtr(R)
        if lam != 0.0:
            d = d + float(lam)
        floor = float(opts.pop("min_diagonal", 1.0e-30))
        d = np.maximum(d, floor)
        invd = 1.0 / d

        def mvec(x: np.ndarray) -> np.ndarray:
            """Apply Jacobi Mx = diag(Q)^{-1} x."""
            return invd * x

        return LinearOperator((invd.size, invd.size), matvec=mvec, dtype=np.float64)

    # The remaining options require forming sparse Q.
    if not scipy.sparse.issparse(R):
        raise ValueError(
            f"Preconditioner {precond!r} requires sparse R (to form sparse Q)."
        )

    Q = (R.T @ R).tocsc()
    if lam != 0.0:
        Q = Q + float(lam) * scipy.sparse.identity(Q.shape[0], format="csc")

    if key in {"ilu"}:
        ilu = scipy.sparse.linalg.spilu(
            Q,
            drop_tol=float(opts.pop("drop_tol", 1.0e-4)),
            fill_factor=float(opts.pop("fill_factor", 10.0)),
            diag_pivot_thresh=float(opts.pop("diag_pivot_thresh", 0.0)),
            permc_spec=str(opts.pop("permc_spec", "COLAMD")),
        )

        def mvec(x: np.ndarray) -> np.ndarray:
            """Apply ILU preconditioner via one triangular solve."""
            return ilu.solve(x)

        return LinearOperator(Q.shape, matvec=mvec, dtype=np.float64)

    if key in {"amg"}:
        try:
            import pyamg  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Preconditioner 'amg' requires pyamg.") from exc

        tol = float(opts.pop("tol", 1.0e-8))
        maxiter = int(opts.pop("maxiter", 1))
        ml = pyamg.smoothed_aggregation_solver(Q, **opts)

        def mvec(x: np.ndarray) -> np.ndarray:
            """Apply one (or few) AMG V-cycles as approximate inverse."""
            return ml.solve(x, tol=tol, maxiter=maxiter)

        return LinearOperator(Q.shape, matvec=mvec, dtype=np.float64)

    raise ValueError(
        "Unknown preconditioner. Use None|'jacobi'|'ilu'|'amg'|'identity' "
        f"(got {precond!r})."
    )


def make_sparse_cholesky_precision_solver(
    R: np.ndarray | scipy.sparse.spmatrix,
    lam: float = 0.0,
    *,
    use_cholmod: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a direct solver for Qx=b using sparse Cholesky (or fallback LU).

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining Q = R.T @ R (+ lam * I).
    lam : float, optional
        Diagonal shift added to Q.
    use_cholmod : bool, optional
        If True (default), try SuiteSparse CHOLMOD via ``scikit-sparse``.
        If unavailable, fall back to SciPy sparse LU with a RuntimeWarning.

    Returns
    -------
    solve_Q : callable
        A function solving Qx=b.

    Notes
    -----
    For large 3-D problems, explicit factorisation can be memory intensive
    due to fill-in. It is most useful when Q is moderate in size or when many
    solves with the same Q are needed.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    if scipy.sparse.issparse(R):
        Q = (R.T @ R).tocsc()
        if lam != 0.0:
            Q = Q + float(lam) * scipy.sparse.identity(Q.shape[0], format="csc")

        if use_cholmod:
            try:
                from sksparse.cholmod import cholesky  # type: ignore
            except Exception:  # pragma: no cover
                warnings.warn(
                    "CHOLMOD (scikit-sparse) not available; falling back to SciPy splu.",
                    RuntimeWarning,
                )
                lu = scipy.sparse.linalg.splu(Q)

                def solve_Q(b: np.ndarray) -> np.ndarray:
                    """Solve using SciPy sparse LU."""
                    return lu.solve(b)

                return solve_Q

            factor = cholesky(Q)

            def solve_Q(b: np.ndarray) -> np.ndarray:
                """Solve using CHOLMOD sparse Cholesky."""
                return factor(b)

            return solve_Q

        lu = scipy.sparse.linalg.splu(Q)

        def solve_Q(b: np.ndarray) -> np.ndarray:
            """Solve using SciPy sparse LU."""
            return lu.solve(b)

        return solve_Q

    # Dense fallback (small problems)
    Rm = np.asarray(R, dtype=np.float64)
    Q = Rm.T @ Rm
    if lam != 0.0:
        Q = Q + float(lam) * np.identity(Q.shape[0], dtype=np.float64)

    c, lower = scipy.linalg.cho_factor(Q, overwrite_a=False, check_finite=True)

    def solve_Q(b: np.ndarray) -> np.ndarray:
        """Solve using dense Cholesky."""
        return scipy.linalg.cho_solve((c, lower), b, check_finite=True)

    return solve_Q


def make_precision_solver(
    R: np.ndarray | scipy.sparse.spmatrix,
    lam: float = 0.0,
    rtol: float = 1.0e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    M: Optional[LinearOperator] = None,
    msolver: Optional[str] = "cg",
    mprec: Optional[str] = "jacobi",
    *,
    solver_method: Optional[str] = None,
    precond: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
    use_cholmod: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a solver for Qx=b with Q = R.T@R + lam*I.

    This is a compatibility-preserving upgrade of the earlier FEMTIC helper:

    - **Iterative solvers**: CG (recommended for SPD Q) and BiCGStab.
    - **Direct solver**: sparse Cholesky via CHOLMOD (or SciPy LU fallback).

    Parameters
    ----------
    R : array_like or sparse matrix
        Operator defining the precision Q.
    lam : float, optional
        Diagonal shift.
    rtol, atol : float, optional
        Iterative solver tolerances (SciPy style).
    maxiter : int, optional
        Maximum number of iterations for iterative solvers.
    M : LinearOperator, optional
        Explicit preconditioner. If provided, it overrides ``mprec/precond``.
    msolver : {"cg", "bicg", "bicgstab"}, optional
        Iterative solver choice (legacy argument name).
    mprec : {"jacobi", "ilu", "amg", "identity", None}, optional
        Preconditioner choice (legacy argument name). See ``precond``.
    solver_method : {"cg", "cholesky", "bicgstab"}, optional
        New preferred name for the solver. If provided, overrides ``msolver``.
    precond : {None, "jacobi", "ilu", "amg", "identity"}, optional
        New preferred name for the preconditioner. If provided, overrides
        ``mprec``.
    precond_kwargs : dict, optional
        Extra options for the chosen preconditioner.
    use_cholmod : bool, optional
        If True (default), try CHOLMOD for the 'cholesky' method.

    Returns
    -------
    solve_Q : callable
        Function that solves Qx=b.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    meth = (solver_method or msolver or "cg").strip().lower()

    if meth in {"chol", "cholesky", "sparse_cholesky", "direct"}:
        return make_sparse_cholesky_precision_solver(
            R=R,
            lam=lam,
            use_cholmod=use_cholmod,
        )

    # Iterative branch
    Q_op = build_rtr_operator(R, lam=lam)

    if M is None:
        M = make_rtr_preconditioner(
            R=R,
            lam=lam,
            precond=precond if precond is not None else mprec,
            precond_kwargs=precond_kwargs,
        )

    def solve_Q(b: np.ndarray) -> np.ndarray:
        """Solve Qx=b using the selected iterative method."""
        if meth in {"cg", "pcg"}:
            x, info = cg(Q_op, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        else:
            # BiCGStab can handle non-SPD, but is not needed if Q is SPD.
            x, info = bicgstab(Q_op, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)

        if info > 0:
            raise RuntimeError(
                f"Iterative solver did not converge within {info} iterations."
            )
        if info < 0:
            raise RuntimeError(f"Iterative solver failed (info={info}).")
        return np.asarray(x, dtype=np.float64)

    return solve_Q


def sample_rtr_full_rank(
    R: np.ndarray | scipy.sparse.spmatrix,
    n_samples: int = 1,
    lam: float = 1.0e-4,
    solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rng: Optional[Generator] = None,
    *,
    solver_method: str = "cg",
    solver_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Sample from N(0, (R.T@R + lam I)^{-1}) using full-rank solves.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision.
    n_samples : int, optional
        Number of samples.
    lam : float, optional
        Diagonal shift added to the precision.
    solver : callable, optional
        Pre-built solver for Qx=b. If None, it is created from ``solver_method``.
    rng : numpy.random.Generator, optional
        Random generator.
    solver_method : {"cg", "cholesky", "bicgstab"}, optional
        Solver used when ``solver`` is None.
    solver_kwargs : dict, optional
        Extra args passed to :func:`make_precision_solver`.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)

    Notes
    -----
    Uses the identity::

        x = (R.T R + lam I)^{-1} R.T ξ,   ξ ~ N(0, I)

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    rng = default_rng() if rng is None else rng
    m, n = R.shape

    if solver is None:
        solver = make_precision_solver(
            R=R,
            lam=lam,
            solver_method=solver_method,
            **(solver_kwargs or {}),
        )

    samples = np.empty((n_samples, n), dtype=np.float64)
    for ix in range(n_samples):
        xi = rng.standard_normal(size=m)
        b = R.T @ xi
        samples[ix, :] = solver(b)

    return samples


def estimate_low_rank_eigpairs(
    Q: scipy.sparse.spmatrix | LinearOperator,
    k: int,
    which: str = "SM",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k extremal eigenpairs of a symmetric precision operator.

    Parameters
    ----------
    Q : sparse matrix or LinearOperator
        Symmetric positive (semi-)definite operator.
    k : int
        Number of eigenpairs.
    which : {"SM", "LM"}, optional
        Smallest / largest magnitude eigenvalues.

    Returns
    -------
    eigvals, eigvecs

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    eigvals, eigvecs = eigsh(Q, k=k, which=which)
    return eigvals, eigvecs


def sample_rtr_low_rank(
    R: np.ndarray | scipy.sparse.spmatrix,
    n_samples: int = 1,
    n_eig: int = 32,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
    *,
    which: str = "SM",
) -> np.ndarray:
    """Approximate sampling from N(0, (R.T R)^{-1}) using k eigenpairs.

    Parameters
    ----------
    R : array_like or sparse matrix
        Matrix defining Q = R.T @ R.
    n_samples : int
        Number of samples.
    n_eig : int
        Number of eigenpairs.
    sigma2_residual : float
        Optional isotropic residual variance.
    rng : numpy.random.Generator, optional
    which : {"SM", "LM"}
        Which eigenvalues to request from eigsh.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)

    Notes
    -----
    This uses ``eigsh`` on a **LinearOperator** for Q, avoiding explicit
    formation of Q. For very large systems, computing smallest eigenpairs
    can still be expensive.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    rng = default_rng() if rng is None else rng
    Q_op = build_rtr_operator(R, lam=0.0)

    eigvals, eigvecs = estimate_low_rank_eigpairs(Q_op, k=n_eig, which=which)

    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)

    n, k = eigvecs.shape
    samples = np.empty((n_samples, n), dtype=np.float64)
    inv_sqrt = 1.0 / np.sqrt(eigvals)

    for ix in range(n_samples):
        z = rng.standard_normal(size=k)
        x = eigvecs @ (inv_sqrt * z)
        if sigma2_residual > 0.0:
            x = x + np.sqrt(sigma2_residual) * rng.standard_normal(size=n)
        samples[ix, :] = x

    return samples


def sample_prior_from_roughness(
    R: np.ndarray | scipy.sparse.spmatrix,
    n_samples: int = 1,
    lam: float = 1.0e-4,
    mode: Literal["full", "low-rank"] = "full",
    n_eig: int = 32,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
    msolver: Optional[str] = "cg",
    mprec: Optional[str] = "jacobi",
    rtol: float = 1.0e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    *,
    solver_method: Optional[str] = None,
    precond: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
    use_cholmod: bool = True,
) -> np.ndarray:
    """Sample from the Gaussian prior implied by a FEMTIC roughness matrix.

    The implied precision is::

        Q = R.T @ R + lam * I

    Parameters
    ----------
    R : array_like or sparse matrix
        Roughness matrix.
    n_samples : int
        Number of samples.
    lam : float
        Diagonal shift in Q.
    mode : {"full", "low-rank"}
        Full-rank sampling uses iterative/direct solves. Low-rank uses eigenpairs.
    n_eig : int
        Number of eigenpairs in low-rank mode.
    sigma2_residual : float
        Residual isotropic variance in low-rank mode.
    rng : numpy.random.Generator, optional
    msolver, mprec, rtol, atol, maxiter
        Legacy iterative-solver settings (kept for compatibility).
    solver_method, precond, precond_kwargs, use_cholmod
        Preferred new names; see :func:`make_precision_solver`.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Samples from N(0, Q^{-1}).

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    mode_l = str(mode).lower()
    if mode_l.startswith("low"):
        return sample_rtr_low_rank(
            R=R,
            n_samples=n_samples,
            n_eig=n_eig,
            sigma2_residual=sigma2_residual,
            rng=rng,
        )

    solver = make_precision_solver(
        R=R,
        lam=lam,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        msolver=msolver,
        mprec=mprec,
        solver_method=solver_method,
        precond=precond,
        precond_kwargs=precond_kwargs,
        use_cholmod=use_cholmod,
    )
    return sample_rtr_full_rank(
        R=R,
        n_samples=n_samples,
        lam=lam,
        solver=solver,
        rng=rng,
        solver_method=solver_method or (msolver or "cg"),
        solver_kwargs={},  # solver already built above
    )


# SECTION 4: mesh.dat + resistivity_block_iterX.dat → NPZ
# ============================================================================


def read_femtic_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a FEMTIC TETRA mesh file and return node coordinates and connectivity.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh file (usually "mesh.dat").

    Returns
    -------
    nodes : ndarray, shape (nn, 3)
        Node coordinates [x, y, z].
    conn : ndarray, shape (nelem, 4)
        Tetrahedral connectivity (0-based node indices).
    """
    with open(mesh_path, "r", errors="ignore") as f:
        header = f.readline().strip()
        if header.upper() != "TETRA":
            raise ValueError(f"Unsupported mesh type '{header}', expected 'TETRA'.")

        nn_line = f.readline().split()
        if not nn_line:
            raise ValueError("Missing node count after 'TETRA' header.")
        nn = int(nn_line[0])

        nodes = np.empty((nn, 3), dtype=float)
        for _ in range(nn):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading node coordinates.")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Node line has too few columns: {line!r}")
            idx = int(parts[0])
            x, y, z = map(float, parts[1:4])
            if not (0 <= idx < nn):
                raise ValueError(f"Node index {idx} out of range 0..{nn-1}.")
            nodes[idx] = (x, y, z)

        nelem_line = f.readline().split()
        if not nelem_line:
            raise ValueError("Missing element count line after node block.")
        nelem = int(nelem_line[0])

        conn = np.empty((nelem, 4), dtype=int)
        for _ in range(nelem):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading element block.")
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Element line has too few columns: {line!r}")
            ie = int(parts[0])
            n1, n2, n3, n4 = map(int, parts[-4:])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            conn[ie] = (n1, n2, n3, n4)

    return nodes, conn


def read_resistivity_block(block_path: str) -> Dict[str, np.ndarray]:
    """
    Read a FEMTIC resistivity_block_iterX.dat and return region-based data.

    Parameters
    ----------
    block_path : str
        Path to resistivity_block_iterX.dat file.

    Returns
    -------
    data : dict
        Keys:
            "nelem", "nreg",
            "region_of_elem",
            "region_rho", "region_rho_lower", "region_rho_upper",
            "region_n", "region_flag"
    """
    with open(block_path, "r", errors="ignore") as f:
        first = f.readline().split()
        if len(first) < 2:
            raise ValueError("First line must contain 'nelem nreg'.")
        nelem = int(first[0])
        nreg = int(first[1])

        region_of_elem = np.empty(nelem, dtype=int)
        for _ in range(nelem):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading element-region map.")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Element-region line has too few columns: {line!r}")
            ie = int(parts[0])
            ireg = int(parts[1])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            region_of_elem[ie] = ireg

        region_rho = np.empty(nreg, dtype=float)
        region_rho_lower = np.empty(nreg, dtype=float)
        region_rho_upper = np.empty(nreg, dtype=float)
        region_n = np.empty(nreg, dtype=float)
        region_flag = np.empty(nreg, dtype=int)

        for _ in range(nreg):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading region lines.")
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Region line has too few columns: {line!r}")
            ireg = int(parts[0])
            rho = float(parts[1])
            rho_min = float(parts[2])
            rho_max = float(parts[3])
            n = float(parts[4])
            flag = int(parts[5])
            if not (0 <= ireg < nreg):
                raise ValueError(f"Region index {ireg} out of range 0..{nreg-1}.")
            region_rho[ireg] = rho
            region_rho_lower[ireg] = rho_min
            region_rho_upper[ireg] = rho_max
            region_n[ireg] = n
            region_flag[ireg] = flag

    return {
        "nelem": np.array(nelem, dtype=int),
        "nreg": np.array(nreg, dtype=int),
        "region_of_elem": region_of_elem,
        "region_rho": region_rho,
        "region_rho_lower": region_rho_lower,
        "region_rho_upper": region_rho_upper,
        "region_n": region_n,
        "region_flag": region_flag,
    }


def build_element_arrays(
    nodes: np.ndarray,
    conn: np.ndarray,
    region_of_elem: np.ndarray,
    region_rho: np.ndarray,
    region_rho_lower: np.ndarray,
    region_rho_upper: np.ndarray,
    region_n: np.ndarray,
    region_flag: np.ndarray,
    clip_eps: float = 1.0e-30,
) -> Dict[str, np.ndarray]:
    """
    Build per-element arrays (centroids, log10 resistivity, bounds, flags, n).

    Parameters
    ----------
    nodes : ndarray, shape (nn, 3)
    conn : ndarray, shape (nelem, 4)
    region_of_elem : ndarray, shape (nelem,)
    region_rho, region_rho_lower, region_rho_upper : ndarray, shape (nreg,)
    region_n : ndarray, shape (nreg,)
    region_flag : ndarray, shape (nreg,)
    clip_eps : float
        Values smaller than this are clipped before log10.

    Returns
    -------
    arrays : dict
        Element-based arrays, including:
            centroid, region, log10_resistivity,
            rho_lower, rho_upper, flag, n
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)
    region_of_elem = np.asarray(region_of_elem, dtype=int)

    nelem = conn.shape[0]
    if region_of_elem.shape[0] != nelem:
        raise ValueError("region_of_elem length does not match number of elements.")

    coords = nodes[conn]
    centroid = coords.mean(axis=1)

    rho = np.clip(region_rho, clip_eps, np.inf)
    rho_min = np.clip(region_rho_lower, clip_eps, np.inf)
    rho_max = np.clip(region_rho_upper, clip_eps, np.inf)

    rid = region_of_elem
    rho_elem = rho[rid]
    rho_min_elem = rho_min[rid]
    rho_max_elem = rho_max[rid]
    n_elem = region_n[rid]
    flag_elem = region_flag[rid]

    log10_rho = np.log10(rho_elem)
    log10_rho_min = np.log10(rho_min_elem)
    log10_rho_max = np.log10(rho_max_elem)

    return {
        "centroid": centroid,
        "region": rid,
        "log10_resistivity": log10_rho,
        "rho_lower": log10_rho_min,
        "rho_upper": log10_rho_max,
        "flag": flag_elem,
        "n": n_elem,
    }


def save_element_npz_with_mesh_and_regions(
    out_path: str,
    nodes: np.ndarray,
    conn: np.ndarray,
    block: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
) -> None:
    """
    Save mesh, region data, and element arrays into a compressed NPZ file.

    Parameters
    ----------
    out_path : str
        Output npz path.
    nodes : ndarray, shape (nn, 3)
    conn : ndarray, shape (nelem, 4)
    block : dict
        Region-level data from :func:`read_resistivity_block`.
    arrays : dict
        Element arrays from :func:`build_element_arrays`.
    """
    np.savez_compressed(
        out_path,
        nodes=nodes,
        conn=conn,
        nelem=block["nelem"],
        nreg=block["nreg"],
        region_of_elem=block["region_of_elem"],
        region_rho=block["region_rho"],
        region_rho_lower=block["region_rho_lower"],
        region_rho_upper=block["region_rho_upper"],
        region_n=block["region_n"],
        region_flag=block["region_flag"],
        **arrays,
    )


def mesh_and_block_to_npz(
    mesh_path: str,
    rho_block_path: str,
    out_npz: str,
) -> None:
    """
    High-level helper: mesh.dat + resistivity_block_iterX.dat → element NPZ.

    Parameters
    ----------
    mesh_path : str
        FEMTIC mesh.dat.
    rho_block_path : str
        FEMTIC resistivity_block_iterX.dat.
    out_npz : str
        Output npz file.
    """
    nodes, conn = read_femtic_mesh(mesh_path)
    block = read_resistivity_block(rho_block_path)
    arrays = build_element_arrays(
        nodes=nodes,
        conn=conn,
        region_of_elem=block["region_of_elem"],
        region_rho=block["region_rho"],
        region_rho_lower=block["region_rho_lower"],
        region_rho_upper=block["region_rho_upper"],
        region_n=block["region_n"],
        region_flag=block["region_flag"],
    )
    save_element_npz_with_mesh_and_regions(out_npz, nodes, conn, block, arrays)


# ============================================================================
# SECTION 5: NPZ → VTK/VTU
# ============================================================================


def npz_to_unstructured_grid(npz_path: str):
    """
    Build a PyVista UnstructuredGrid from a FEMTIC NPZ file.

    Parameters
    ----------
    npz_path : str
        NPZ created by :func:`mesh_and_block_to_npz`.

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        Unstructured grid with cell_data fields.
    """
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover
        raise ImportError("pyvista is required for VTK export.") from exc

    data = np.load(npz_path)
    if "nodes" not in data or "conn" not in data:
        raise KeyError("NPZ must at least contain 'nodes' and 'conn'.")

    nodes = np.asarray(data["nodes"], dtype=float)
    conn = np.asarray(data["conn"], dtype=int)

    nelem = conn.shape[0]
    if conn.shape[1] != 4:
        raise ValueError("Connectivity must have 4 nodes per element (tetra).")

    n_per_cell = 4
    cells = np.hstack(
        [np.full((nelem, 1), n_per_cell, dtype=np.int64), conn.astype(np.int64)]
    ).ravel()

    try:
        celltypes = np.full(nelem, pv.CellType.TETRA, dtype=np.uint8)
    except Exception:  # old pyvista
        celltypes = np.full(nelem, 10, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    preferred_keys = [
        "log10_resistivity",
        "rho_lower",
        "rho_upper",
        "flag",
        "n",
        "region",
        "region_of_elem",
    ]
    for key in preferred_keys:
        if key in data:
            arr = np.asarray(data[key])
            if arr.shape[0] == nelem:
                name = "region" if key == "region_of_elem" else key
                grid.cell_data[name] = arr

    return grid


def save_vtk_from_npz(
    npz_path: str,
    out_vtu: str,
    out_legacy: Optional[str] = None,
    scalar_name: str = "log10_resistivity",
) -> None:
    """
    Export a FEMTIC NPZ model to VTK/VTU unstructured grid file(s).

    Parameters
    ----------
    npz_path : str
    out_vtu : str
        VTU output file.
    out_legacy : str, optional
        Optional legacy .vtk file.
    scalar_name : str
        Name of scalar to use for plotting.
    """
    grid = npz_to_unstructured_grid(npz_path)
    grid.save(out_vtu)
    print("Wrote VTU:", out_vtu)

    if out_legacy is not None:
        grid.save(out_legacy)
        print("Wrote legacy VTK:", out_legacy)

    if scalar_name in grid.cell_data:
        print(f"Scalar '{scalar_name}' available for plotting.")
    else:
        print(f"Scalar '{scalar_name}' not present in grid.cell_data.")


# ============================================================================
# ============================================================================
# SECTION 6: NPZ ↔ HDF5 / NetCDF
# ============================================================================


def npz_to_hdf5(
    npz_path: str,
    hdf5_path: str,
    group: str = "femtic_model",
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write all arrays from a FEMTIC NPZ file to an HDF5 file.

    Parameters
    ----------
    npz_path : str
        Input NPZ file created by :func:`mesh_and_block_to_npz`.
    hdf5_path : str
        Output HDF5 file.
    group : str, optional
        Name of the top-level group into which all datasets are written.
    compression : str, optional
        HDF5 compression method for datasets (default "gzip").
    compression_opts : int, optional
        Compression level (only used for some compressors such as "gzip").

    Notes
    -----
    - Each key in the NPZ is written as one dataset under the given group.
    - Scalar values (0-D arrays) are written as scalar datasets.
    - Basic attributes "source_npz" and "created_by" are attached to the group.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required for npz_to_hdf5().") from exc

    data = np.load(npz_path)
    with h5py.File(hdf5_path, "w") as h5:
        g = h5.create_group(group)
        g.attrs["source_npz"] = str(npz_path)
        g.attrs["created_by"] = "femtic.npz_to_hdf5"

        for key in data.files:
            arr = data[key]
            if arr.shape == ():
                dset = g.create_dataset(key, data=arr[()])
            else:
                dset = g.create_dataset(
                    key,
                    data=arr,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            dset.attrs["dtype"] = str(arr.dtype)


def hdf5_to_npz(
    hdf5_path: str,
    npz_path: str,
    group: str = "femtic_model",
) -> None:
    """
    Convert a FEMTIC-style HDF5 file back to NPZ.

    Parameters
    ----------
    hdf5_path : str
        Input HDF5 file previously written by :func:`npz_to_hdf5`.
    npz_path : str
        Output NPZ file path.
    group : str, optional
        Name of the group that contains the FEMTIC datasets. If the group
        is not found, the HDF5 root is used instead.

    Notes
    -----
    The function simply restores one NPZ field per dataset contained in
    the group.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("hdf5_to_npz requires the 'h5py' package.") from exc

    import numpy as _np

    with h5py.File(hdf5_path, "r") as h5:
        if group in h5:
            g = h5[group]
        else:
            g = h5
        arrays: dict[str, _np.ndarray] = {}
        for key, item in g.items():
            if isinstance(item, h5py.Dataset):
                arrays[key] = _np.array(item[()])

    _np.savez(npz_path, **arrays)


def npz_to_netcdf(
    npz_path: str,
    netcdf_path: str,
) -> None:
    """
    Convert a FEMTIC NPZ file to a NetCDF file with simple, named dimensions.

    Parameters
    ----------
    npz_path : str
        Input NPZ file created by :func:`mesh_and_block_to_npz`.
    netcdf_path : str
        Output NetCDF file (NetCDF4 format).

    Notes
    -----
    The function tries to create meaningful dimensions when the standard FEMTIC
    NPZ layout is present:

    - "node", "xyz" for ``nodes`` (nn, 3)
    - "cell", "nne" for ``conn`` (nelem, 4)
    - "cell" for cell-based 1-D arrays (e.g. ``log10_resistivity``)
    - "region" for region-based arrays (e.g. ``region_rho``)

    Any remaining arrays are stored using automatically generated dimensions.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required for npz_to_netcdf().") from exc

    import numpy as _np

    data = np.load(npz_path)

    ds = Dataset(netcdf_path, "w", format="NETCDF4")
    ds.setncattr("source_npz", str(npz_path))
    ds.setncattr("created_by", "femtic.npz_to_netcdf")

    def ensure_dim(name: str, size: int) -> None:
        """Create a dimension if it does not yet exist."""
        if name not in ds.dimensions:
            ds.createDimension(name, size)

    # Nodes: (nn, 3) → dims ("node", "xyz")
    if "nodes" in data:
        nodes = data["nodes"]
        if nodes.ndim == 2:
            nn, nxyz = nodes.shape
            ensure_dim("node", nn)
            ensure_dim("xyz", nxyz)
            var = ds.createVariable("nodes", "f8", ("node", "xyz"))
            var[:, :] = nodes

    # Connectivity: (nelem, 4) → dims ("cell", "nne")
    if "conn" in data:
        conn = data["conn"]
        if conn.ndim == 2:
            nelem, nne = conn.shape
            ensure_dim("cell", nelem)
            ensure_dim("nne", nne)
            var = ds.createVariable("conn", "i4", ("cell", "nne"))
            var[:, :] = conn

    # Element centroids: (nelem, 3) → dims ("cell", "xyz")
    if "centroid" in data and "cell" in ds.dimensions and "xyz" in ds.dimensions:
        cent = data["centroid"]
        if cent.shape == (len(ds.dimensions["cell"]), len(ds.dimensions["xyz"])):
            var = ds.createVariable("centroid", "f8", ("cell", "xyz"))
            var[:, :] = cent

    # Scalars nelem / nreg
    if "nelem" in data:
        var = ds.createVariable("nelem", "i4")
        var[...] = int(data["nelem"])
    if "nreg" in data:
        var = ds.createVariable("nreg", "i4")
        var[...] = int(data["nreg"])

    # Region-based arrays
    if "nreg" in data:
        nreg = int(data["nreg"])
        ensure_dim("region", nreg)
        region_keys = [
            "region_rho",
            "region_rho_lower",
            "region_rho_upper",
            "region_n",
            "region_flag",
        ]
        for key in region_keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 1 and arr.shape[0] == nreg:
                    dtype = "f8" if arr.dtype.kind in "fc" else "i4"
                    var = ds.createVariable(key, dtype, ("region",))
                    var[:] = arr

    # Cell-based arrays
    if "cell" in ds.dimensions:
        nelem = len(ds.dimensions["cell"])
        cell_keys = [
            "region_of_elem",
            "log10_resistivity",
            "rho_lower",
            "rho_upper",
            "flag",
            "n",
        ]
        for key in cell_keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 1 and arr.shape[0] == nelem:
                    dtype = "f8" if arr.dtype.kind in "fc" else "i4"
                    var = ds.createVariable(key, dtype, ("cell",))
                    var[:] = arr

    # Any remaining arrays that have not yet been written
    written = set(ds.variables.keys())
    for key in data.files:
        if key in written:
            continue

        arr = data[key]
        if arr.shape == ():
            var = ds.createVariable(key, "f8")
            var[...] = arr[()]
            continue

        dims = []
        for iax, size in enumerate(arr.shape):
            dname = f"{key}_dim{iax}"
            ensure_dim(dname, size)
            dims.append(dname)
        dtype = "f8" if arr.dtype.kind in "fc" else "i4"
        var = ds.createVariable(key, dtype, tuple(dims))
        var[...] = arr

    ds.close()


def netcdf_to_npz(
    netcdf_path: str,
    npz_path: str,
) -> None:
    """
    Convert a NetCDF file written by :func:`npz_to_netcdf` back to NPZ.

    Parameters
    ----------
    netcdf_path : str
        Input NetCDF file (NetCDF4 format).
    npz_path : str
        Output NPZ file path.

    Notes
    -----
    This helper is deliberately simple: it converts each NetCDF variable to
    one NPZ field with the same name. Masked values (if any) are filled with
    ``NaN`` before being written.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("netcdf_to_npz requires the 'netCDF4' package.") from exc

    import numpy as _np
    import numpy.ma as _ma

    ds = Dataset(netcdf_path, "r")
    arrays: dict[str, _np.ndarray] = {}
    try:
        for name, var in ds.variables.items():
            data = var[...]
            if isinstance(data, _ma.MaskedArray):
                data = data.filled(_np.nan)
            arrays[name] = _np.array(data)
    finally:
        ds.close()

    _np.savez(npz_path, **arrays)


# SECTION 7: NPZ → FEMTIC mesh.dat + resistivity_block
# ============================================================================


def write_femtic_mesh(
    out_path: str,
    nodes: np.ndarray,
    conn: np.ndarray,
) -> None:
    """
    Write a FEMTIC-style mesh.dat from nodes and connectivity.

    Parameters
    ----------
    out_path : str
    nodes : ndarray, shape (nn, 3)
    conn : ndarray, shape (nelem, 4)
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)
    nn = nodes.shape[0]
    nelem = conn.shape[0]

    with open(out_path, "w") as f:
        f.write("TETRA\n")
        f.write(f"{nn:d}\n")
        for i in range(nn):
            x, y, z = nodes[i]
            f.write(f"{i:d} {x:.8e} {y:.8e} {z:.8e}\n")

        f.write(f"{nelem:d}\n")
        for ie in range(nelem):
            n1, n2, n3, n4 = conn[ie]
            f.write(f"{ie:d} -1 -1 -1 -1 {n1:d} {n2:d} {n3:d} {n4:d}\n")


def write_resistivity_block(
    out_path: str,
    nelem: int,
    nreg: int,
    region_of_elem: np.ndarray,
    region_rho: np.ndarray,
    region_rho_lower: np.ndarray,
    region_rho_upper: np.ndarray,
    region_n: np.ndarray,
    region_flag: np.ndarray,
    fmt: str = "{:.6g}",
) -> None:
    """
    Write a FEMTIC resistivity_block_iterX.dat from region arrays.

    Parameters
    ----------
    out_path : str
    nelem : int
    nreg : int
    region_of_elem : ndarray, shape (nelem,)
    region_rho, region_rho_lower, region_rho_upper : ndarray, shape (nreg,)
    region_n : ndarray, shape (nreg,)
    region_flag : ndarray, shape (nreg,)
    fmt : str
        Float format.
    """
    region_of_elem = np.asarray(region_of_elem, dtype=int)
    region_rho = np.asarray(region_rho, dtype=float)
    region_rho_lower = np.asarray(region_rho_lower, dtype=float)
    region_rho_upper = np.asarray(region_rho_upper, dtype=float)
    region_n = np.asarray(region_n, dtype=float)
    region_flag = np.asarray(region_flag, dtype=int)

    if region_of_elem.shape[0] != nelem:
        raise ValueError("region_of_elem length does not match nelem.")
    if any(
        arr.shape[0] != nreg
        for arr in [region_rho, region_rho_lower, region_rho_upper, region_n, region_flag]
    ):
        raise ValueError("Region arrays must all have length nreg.")

    with open(out_path, "w") as f:
        f.write(f"{nelem:d} {nreg:d}\n")
        for ie in range(nelem):
            f.write(f"{ie:d} {int(region_of_elem[ie]):d}\n")

        for ireg in range(nreg):
            rho = fmt.format(region_rho[ireg])
            rmin = fmt.format(region_rho_lower[ireg])
            rmax = fmt.format(region_rho_upper[ireg])
            nval = int(region_n[ireg])
            flag = int(region_flag[ireg])
            f.write(f"{ireg:d} {rho} {rmin} {rmax} {nval:d} {flag:d}\n")


def npz_to_femtic(
    npz_path: str,
    mesh_out: str,
    rho_block_out: str,
    fmt: str = "{:.6g}",
) -> None:
    """
    Reconstruct FEMTIC mesh.dat and resistivity_block_iterX.dat from NPZ.

    Parameters
    ----------
    npz_path : str
    mesh_out : str
    rho_block_out : str
    fmt : str
        Float format for resistivities and bounds.
    """
    data = np.load(npz_path)

    required = [
        "nodes",
        "conn",
        "nelem",
        "nreg",
        "region_of_elem",
        "region_rho",
        "region_rho_lower",
        "region_rho_upper",
        "region_n",
        "region_flag",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"NPZ is missing required arrays: {missing}")

    nodes = data["nodes"]
    conn = data["conn"]
    nelem = int(data["nelem"])
    nreg = int(data["nreg"])
    region_of_elem = data["region_of_elem"]
    region_rho = data["region_rho"]
    region_rho_lower = data["region_rho_lower"]
    region_rho_upper = data["region_rho_upper"]
    region_n = data["region_n"]
    region_flag = data["region_flag"]

    write_femtic_mesh(mesh_out, nodes, conn)
    write_resistivity_block(
        rho_block_out,
        nelem,
        nreg,
        region_of_elem,
        region_rho,
        region_rho_lower,
        region_rho_upper,
        region_n,
        region_flag,
        fmt=fmt,
    )


# ============================================================================
# SECTION 7: CLI wrapper for unified module
# ============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for femtic_core.py with subcommands:

    - femtic-to-npz
    - npz-to-vtk
    - npz-to-femtic
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Unified FEMTIC utilities: ensembles, mesh↔NPZ↔VTK/FEMTIC."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # femtic-to-npz
    p_m2n = sub.add_parser(
        "femtic-to-npz",
        help="Build FEMTIC mesh + region + element NPZ from mesh.dat and block.",
    )
    p_m2n.add_argument("--mesh", required=True, help="Path to FEMTIC mesh.dat.")
    p_m2n.add_argument(
        "--rho-block",
        required=True,
        help="Path to resistivity_block_iterX.dat.",
    )
    p_m2n.add_argument("--out-npz", required=True, help="Output NPZ file.")

    # npz-to-vtk
    p_n2v = sub.add_parser(
        "npz-to-vtk",
        help="Export FEMTIC NPZ model to VTK/VTU unstructured grid.",
    )
    p_n2v.add_argument("--npz", required=True, help="Input NPZ file.")
    p_n2v.add_argument("--out-vtu", required=True, help="Output VTU file.")
    p_n2v.add_argument("--out-legacy", default=None, help="Optional legacy .vtk file.")
    p_n2v.add_argument(
        "--scalar",
        default="log10_resistivity",
        help="Cell scalar name for plotting (default 'log10_resistivity').",
    )

    # hdf5-to-npz
    p_h2n = sub.add_parser(
        "hdf5-to-npz",
        help="Convert FEMTIC-style HDF5 (written by npz-to-hdf5) back to NPZ.",
    )
    p_h2n.add_argument("--hdf5", required=True, help="Input HDF5 file.")
    p_h2n.add_argument("--out-npz", required=True, help="Output NPZ file.")
    p_h2n.add_argument(
        "--group",
        default="femtic_model",
        help="HDF5 group containing the FEMTIC datasets (default 'femtic_model').",
    )

    # netcdf-to-npz
    p_nc2n = sub.add_parser(
        "netcdf-to-npz",
        help="Convert FEMTIC-style NetCDF (written by npz-to-netcdf) back to NPZ.",
    )
    p_nc2n.add_argument("--netcdf", required=True, help="Input NetCDF file.")
    p_nc2n.add_argument("--out-npz", required=True, help="Output NPZ file.")

    # npz-to-femtic
    p_n2f = sub.add_parser(
        "npz-to-femtic",
        help="Recreate FEMTIC mesh.dat and resistivity_block from NPZ.",
    )
    p_n2f.add_argument("--npz", required=True, help="Input NPZ file.")
    p_n2f.add_argument("--mesh-out", required=True, help="Output mesh.dat.")
    p_n2f.add_argument(
        "--rho-block-out",
        required=True,
        help="Output resistivity_block_iterX.dat.",
    )
    p_n2f.add_argument(
        "--format",
        dest="fmt",
        default="{:.6g}",
        help='Float format for resistivity and bounds (default "{:.6g}").',
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "femtic-to-npz":
        mesh_and_block_to_npz(args.mesh, args.rho_block, args.out_npz)
        print("Saved mesh + region + element NPZ:", args.out_npz)
        return 0

    if args.cmd == "npz-to-vtk":
        save_vtk_from_npz(
            npz_path=args.npz,
            out_vtu=args.out_vtu,
            out_legacy=args.out_legacy,
            scalar_name=args.scalar,
        )
        return 0

    if args.cmd == "hdf5-to-npz":
        hdf5_to_npz(
            hdf5_path=args.hdf5,
            npz_path=args.out_npz,
            group=args.group,
        )
        print("Wrote NPZ:", args.out_npz)
        return 0

    if args.cmd == "netcdf-to-npz":
        netcdf_to_npz(
            netcdf_path=args.netcdf,
            npz_path=args.out_npz,
        )
        print("Wrote NPZ:", args.out_npz)
        return 0

    if args.cmd == "npz-to-femtic":
        npz_to_femtic(
            npz_path=args.npz,
            mesh_out=args.mesh_out,
            rho_block_out=args.rho_block_out,
            fmt=args.fmt,
        )
        print("Wrote mesh to:", args.mesh_out)
        print("Wrote resistivity block to:", args.rho_block_out)
        return 0

    ap.error(f"Unknown command {args.cmd!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
