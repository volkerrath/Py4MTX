#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_post.py — Ensemble postprocessing for FEMTIC

Collects all converged members of a FEMTIC ensemble (RTO, GST, or any
directory-based ensemble), computes summary statistics (mean, variance,
median, MAD, percentiles, inter-quantile spreads, bootstrap variance),
optional ensemble sensitivity measures (per-member cumulative sensitivity,
SimRC) and the empirical covariance, and writes

  - the results ``.npz`` (``ENSEMBLE_RESULTS``): everything the plotting
    scripts need -- ensemble matrix, statistics, member list, best member,
    perturbed models (``ENS_MODEL_SOURCE = "both"``), and the nRMS table of
    all scanned members;
  - the nRMS table as CSV (``NRMS_CSV``);
  - the parameter summary ``femtic_ens_post_summary.md`` with the directory
    scan report.

Optionally (ENS_SUBSAMPLE_N) only N members drawn at random from the
accepted ones are used, e.g. to see how the statistics depend on the ensemble
size or to get a quick preview; everything (statistics, sensitivity, SimRC,
covariance, bootstrap, "best" member, .npz) then refers to the N drawn members
and the output names get a suffix, so the full results are not overwritten.

This script only COMPUTES.  It draws nothing.  The figures are made from the
results ``.npz`` by two plot scripts (each has its own user section) and by
one method of the quality-control script:

  femtic_ens_plot_slices.py  best-member and statistics slice panels
  femtic_ens_plot_bh.py      borehole profiles and heat maps
  femtic_data_misfit.py      method "nrms": nRMS histogram of all members
                             (the same script does the data-level checks)

so that styling, slices, vertical exaggeration, the reference model etc. can
be iterated in seconds without rescanning the member directories.  (The
former ``PLOT_ONLY`` mode of this script is gone: it is what these scripts
do.)  These are the "ens" scripts (quality control and ensemble summary
plots); plots of single models are the "mod" scripts, femtic_mod_plot_slice.py
and femtic_mod_plot_bh.py, which also take lists / globs of member directories.

References
----------
Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
    Randomize-Then-Optimize: a Method for Sampling from Posterior
    Distributions in Nonlinear Inverse Problems.
    SIAM J. Sci. Comp., 2014, 36, A1895-A1910.

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data. Part I: Motivation and Theory.
    Geophysical Journal International, doi:10.1093/gji/ggac241, 2022.

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data – Part II: application in 1-D and 2-D problems.
    Geophysical Journal International, doi:10.1093/gji/ggac242, 2022.

Suzuki, K.; Assessing inversion uncertainty from initial-model variability in
    3-D magnetotelluric inversion: Application to a geothermal field
    Journal of Applied Geophysics, 251, 106320
    doi:10.1016/j.jappgeo.2026.106320, 2026

Christiansen, A. V. & Auken, E.
    A global measure for depth of investigation.
    Geophysics, 2012, 77, WB171-WB177, doi:10.1190/geo2011-0393.1

Bobe, C.; Keller, J. & Van De Vijver, E.
    Sensitivity and depth of investigation from Monte Carlo ensemble
    statistics.
    Geophysical Prospecting, 2021, doi:10.1111/1365-2478.13068


@author: vrath

Provenance
----------
2025-04-30  vrath
            Created as femtic_rto_post.py.
2026-03-03  Claude (Anthropic)
            Renamed user-set parameters to UPPERCASE; generated README.
2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
            Added femtic_viz import; MESH_FILE / MOD_QC / MOD_QC_FILE /
            MOD_QC_SLICES / MOD_QC_* config vars; QC slice plot of
            best-nRMS member at end of main block (calls
            fviz.plot_model_slices).
2026-06-11  vrath / Claude Sonnet 4.6 (Anthropic)
            Renamed femtic_rto_post.py → femtic_ens_post.py for
            algorithm-agnostic use.  Fixed axis bug: mean/var/median/MAD
            were computed over axis=1 (free parameters) instead of axis=0
            (members) — statistics are now correct.  MOD_QC block
            replaced by full MOD_SLICES framework matching
            femtic_mod_edit.py / femtic_mod_math.py: UTM origin
            resolution, CRS-aware fem.resolve_slice_positions, site
            overlay via _resolve_origin_and_sites(); new MOD_STATS block
            plots mean/variance/median/MAD as individual slice figures.
            ENSEMBLE_PREFIX added for generic naming of output keys.
2026-07-07  vrath / Claude Sonnet 5 (Anthropic)
            Aligned the entire plotting config surface with
            femtic_gst_prep.py / femtic_rto_prep.py: MESH_FILE →
            MOD_MESH; UTM_ORIGIN_* → MOD_UTM_ORIGIN_*; UTM_ZONE_OVERRIDE →
            MOD_UTM_ZONE_OVERRIDE; ORIGIN_METHOD → MOD_ORIGIN_METHOD;
            DISPLAY_COORDS → MOD_DISPLAY_COORDS; SITE_DAT/SITE_NAMES →
            MOD_SITE_DAT/MOD_SITE_NAMES; MOD_SITES_MAPS/SLICES →
            MOD_PLOT_SITES_MAPS/SLICES; PROJECTION_DIST →
            MOD_PROJECTION_DIST; SITE_MARKER(_SLICES) →
            MOD_SITE_MARKER(_SLICES); MAP_MARKERS → MOD_MAP_MARKERS;
            DEPTH_KM/HORIZ_KM → MOD_DEPTH_KM/MOD_HORIZ_KM.  Added
            MOD_OCEAN/MOD_AIR_RHO, MOD_SITE_NUMBER (observe.dat fallback,
            same as femtic_gst_prep.py), MOD_AIR_COLOR, MOD_ALPHA_FILE/
            MODE/BLANK_THRESH, MOD_PANEL_WIDTH, MOD_FIGSIZE.  Removed a
            latent duplicate MOD_XLIM/YLIM/ZLIM assignment that silently
            discarded the first (non-None) values.  _resolve_origin_and_
            sites() and _plot_slice() now match femtic_gst_prep.py's
            origin-resolution and plot_model_slices() call byte-for-byte
            in option coverage, so QC and statistics figures render
            identically to the ensemble-generation scripts given the
            same MOD_* settings.
2026-07-09  vrath / Claude Sonnet 5 (Anthropic)
            Merged MOD_QC_DPI / MOD_STATS_DPI into a single MOD_DPI knob
            (matching femtic_gst_prep.py / femtic_nss.py — one figure-DPI
            setting per script, not one per plot type).  _plot_slice() no
            longer takes a dpi argument; it reads MOD_DPI directly.
2026-07-17  Claude Sonnet 5 (Anthropic)
            scipy.sparse: migrated from legacy matrix to array-equivalent
            API — scs.csr_matrix(tmp) → scs.csr_array(tmp) when building
            the sparsified empirical covariance (ens_covs). No functional
            change; ens_covs is only used for its .nnz count.
2026-08-24  Claude Sonnet 5 (Anthropic)
            Best-iteration selection (step (1), main block) fixed: (a)
            dir_list is now filtered to os.path.isdir() entries immediately
            after get_filelist(), since that function matches on name only
            and can return non-directory hits; (b) the "best" iteration for
            each ensemble member is now the one with the smallest nRMS
            across the *entire* femtic.cnv convergence history, not simply
            the last row — FEMTIC does not guarantee monotonic nRMS
            reduction, so the last iteration is not necessarily the best.
            iter0 (prior/starting model) is excluded from eligibility; a
            WARNING is printed if iter0's nRMS is <= the best eligible
            iter>0 nRMS, since that indicates the inversion did not
            improve on the starting model. Ties (identical minimum nRMS at
            two-plus iterations) resolve to the first (lowest-iteration)
            occurrence.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Covariance estimation made optional (COMPUTE_COV; skips step
            (3) entirely, omitting the *_cov* keys from the .npz output).
            Added COV_METHOD="low_rank": thin SVD of the centred ensemble
            matrix (n_members, n_free) instead of the dense empirical
            covariance. Since the sample covariance of n_members draws
            has rank <= n_members-1, this is an *exact* factorisation
            (not an approximation) whenever n_members << n_free — the
            usual case — cutting cost from O(n_free^2 * n_members) time /
            O(n_free^2) memory to O(n_members^2 * n_free) time /
            O(n_members * n_free) memory. Stores f"{P}_cov_eigval" /
            f"{P}_cov_eigvec" in place of the dense f"{P}_cov"; full
            covariance reconstructs exactly as
            eigvec @ diag(eigval) @ eigvec.T. Also saved f"{P}_prc_levels"
            (the PERCENTILES list itself) alongside f"{P}_prc" for
            self-describing output. MOD_STATS now writes a block file and
            slice figure for each PERCENTILES level in addition to
            avg/var/med/mad, keyed as "p2_3", "p50", "p97_7", etc.
            (default MOD_STATS_WHAT includes all of them).
2026-09-24  Claude Sonnet 5 (Anthropic)
            Added a third sensitivity panel, "sens_mean_raw": the plain
            ensemble mean of sens_matrix with no normalisation at all, in
            SENS_KIND's native physical units ("raw" = sum|J|, or
            "volume_normalised" = sum|J|/block volume) -- alongside the
            existing dimensionless, max-normalised sens_mean_na/_an.
            Plotted on the same MOD_STATS slice geometry (same template,
            mesh, ROI, MOD_XLIM/YLIM/ZLIM) as every other panel, including
            the model itself. Auto-scaled MOD_STATS_CLIM by default (its
            scale is run- and SENS_KIND-dependent). New .npz key:
            <P>_sens_mean_raw. Not wired into MOD_STATS_BLANK_SOURCES
            (its un-normalised scale makes a fixed threshold meaningless,
            same reasoning as sens_mean_na/_an before normalisation).
2026-09-23c Claude Sonnet 5 (Anthropic)
            Changed MOD_STATS_BLANK_SOURCES' default from three separately-
            ANDed sources (sens_mean_na/_an, err) to flag_null_space alone.
            Rationale: low spread and low sensitivity only need blanking
            when they COINCIDE (regularisation parked a cell near the
            prior and the ensemble never got pushed away) -- low spread
            with high sensitivity is a genuine result, and low sensitivity
            with high spread is already visible as spread in VAR/ERR/MAD,
            so neither alone should hide a cell. flag_null_space already
            encodes exactly that AND, and does so using
            FLAG_SPREAD_PERCENTILE (a percentile of the spread
            distribution) rather than an absolute fraction-of-max
            threshold, so it can't silently collapse to zero blanked
            cells the way a miscalibrated SPREAD_LOW_THRESH_FRAC ANDed
            directly against the sensitivity sources just did in
            practice. The three individual sources remain available,
            commented out, for per-criterion diagnostic tuning.
2026-09-23b Claude Sonnet 5 (Anthropic)
            Per user confirmation against an actual results_iter10.h5:
            per-member sensitivity is NOT normalised to that member's own
            max as previously assumed. Replaced the single SENS_RENORMALISE
            switch with two side-by-side aggregation pipelines, always both
            computed when COMPUTE_SENS succeeds (suffixes _na/_an
            throughout -- sens_mean_na/_median_na/_min_na/_max_na/_std_na/
            _cv_na/_low_mask_na and the _an equivalents, plus matching
            MOD_STATS_WHAT/MOD_STATS_CLIM entries, .npz keys, and
            MOD_STATS_BLANK_SOURCES source names):
              "_na" (normalise, then average) -- each member's raw vector
              is divided by its OWN max right after reading, so every
              member contributes equally regardless of its absolute
              sensitivity scale, THEN mean/median/min/max/std/cv are taken
              across the (now per-member-normalised) ensemble.
              "_an" (average, then normalise) -- mean/median/min/max/std/cv
              are taken across the RAW ensemble first (a larger-magnitude
              member dominates the average, as a plain ensemble mean
              would), THEN the resulting aggregate is divided by ITS OWN
              max so the final numbers sit on a [0, ~1] scale.
            flag_null_space's low-sensitivity combination now considers
            sens_low_mask_na, sens_low_mask_an, and simrc_low_mask (was:
            a single sens_low_mask + simrc_low_mask). Old singular
            sens_mean/_median/_min/_max/_std/_cv/_low_mask names and
            SENS_RENORMALISE are gone; every consumer (blanking, plotting,
            .npz output) now works from the two suffixed sets.
2026-09-23  Claude Sonnet 5 (Anthropic)
            Fixed MOD_STATS_BLANK_SOURCES not blanking anything regardless
            of threshold: "sens_mean" and "err" are now compared on a
            consistently max-normalised scale wherever the blank mask is
            built, so a plain fraction (SENS_LOW_THRESH_FRAC/
            SPREAD_LOW_THRESH_FRAC, e.g. 0.05 = "5% of max") is always a
            valid threshold, regardless of SENS_KIND, physical units, or
            ensemble size -- previously an absolute number like 0.01 was
            compared against sens_mean's raw (SENS_KIND- and run-
            dependent) scale, which for "volume_normalised" sensitivity
            in particular could sit nowhere near typical sens_mean values,
            silently blanking nothing no matter what threshold was tried.
            Added SENS_RENORMALISE (default True): since FEMTIC normalises
            each member's own per-block sensitivity to that member's own
            max=1 before writing it, sens_mean/_median/_min/_max/_std are
            now renormalised to sens_mean's own max=1 immediately after
            ensemble aggregation, matching FEMTIC's convention and making
            MOD_STATS_CLIM["sens_mean"] default to the fixed [0, 1] range
            (previously auto-scaled). MOD_STATS_BLANK_SOURCES' default
            list now references SENS_LOW_THRESH_FRAC/SPREAD_LOW_THRESH_FRAC
            directly instead of a guessed absolute number. Blanking now
            applies to every MOD_STATS panel, including a listed source's
            own plot (previously excluded); MOD_QC is still unaffected.
            Added two new plottable diagnostic panels, analogous to
            flag_null_space: "spread_low_mask" (always available) and
            "sens_low_mask" (when COMPUTE_SENS succeeds) -- boolean 0/1
            maps of exactly which cells each criterion would blank, auto-
            added to MOD_STATS_WHAT, useful for tuning thresholds visually
            before combining them via MOD_STATS_BLANK_SOURCES.
2026-09-22  Claude Sonnet 5 (Anthropic)
            Added a simple, max-normalised "small spread" blanking option
            alongside the existing sensitivity one: SPREAD_LOW_THRESH_FRAC
            (default 0.05, mirroring SENS_LOW_THRESH_FRAC) + spread_low_mask
            (ens_err < SPREAD_LOW_THRESH_FRAC * ens_err.max(), always
            computed, saved as f"{P}_spread_low_mask"). Registered as
            source_name "err" in MOD_STATS_BLANK_SOURCES (keyed to match
            _stat_map's "err" entry so it is correctly excluded from its
            own plot, same as the other sources). Does not change
            flag_null_space, which still uses the percentile-based
            FLAG_SPREAD_PERCENTILE.
2026-08-11  Claude Sonnet 5 (Anthropic)
            Configuration block now explicitly labelled as the USER
            SECTION. Added MOD_PLOT_FORMAT, documenting the Matplotlib
            savefig extensions supported by fviz.plot_model_slices
            (pdf/svg/eps vector; png/jpg/tif/webp raster). MOD_QC_FILE
            and the MOD_STATS per-key figure path are now built without
            an extension; _plot_slice() saves once per MOD_PLOT_FORMAT
            entry (e.g. ["pdf", "jpg"] -> both "..._qc.pdf" and
            "..._qc.jpg" from the same slice geometry). A bare string
            ("pdf") still works, normalised internally to a single-entry
            list.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Added MOD_STATS_CLIM: per-statistic colour-scale override for
            MOD_STATS plots. AVG/MED/percentile panels default to the
            shared MOD_CLIM (same log10(Ω·m) space as the model); VAR/MAD/
            QDIFF panels — a different scale entirely — default to a fixed
            [-2, 2] range instead of silently reusing MOD_CLIM (which
            previously made them blank or meaningless); set any of those
            keys to None in MOD_STATS_CLIM to fall back to auto per-panel
            scaling instead. _plot_slice() gained a clim= parameter to
            carry this through.
            Added QDIFF_PAIRS (default [(15.9, 84.1)]): computes
            |P_hi - P_lo| per free parameter as an outlier-robust spread
            statistic, saved to the .npz as f"{P}_qdiff_<lo>_<hi>" and
            available in MOD_STATS under the same key. Added
            MOD_ROI_AUTO/MOD_ROI_PAD_XY/MOD_ROI_ZLIM: MOD_XLIM/MOD_YLIM
            are now derived automatically from the site bounding box (+
            padding) and MOD_ZLIM from MOD_ROI_ZLIM, applied once after
            _resolve_origin_and_sites() and picked up by every subsequent
            _plot_slice() call. This also feeds femtic_viz.py's existing
            aspect-ratio panel-width logic (MOD_PANEL_WIDTH=None +
            MOD_EQUAL_ASPECT=True), so map vs. ns vs. ew panels now size
            themselves differently instead of coming out uniformly square
            whenever xlim/ylim/zlim were previously left at their
            full-mesh-auto None default. Changed MOD_NROWS/MOD_NCOLS
            defaults from None/None (1 row) to 2/2, matching the 4 default
            MOD_SLICES panels. Also fixed a matching sign bug in
            femtic_viz.py's plot_model_slices (ns/ew curtain panels came
            out blank/upside down whenever MOD_ZLIM was actually set,
            which is why the bug was invisible until this ROI change made
            MOD_ZLIM non-None by default) — see femtic_viz.py provenance.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Added MOD_TICK_FONTSIZE / MOD_LABEL_FONTSIZE (defaults 7 / 8,
            matching fviz.plot_model_slices' own defaults), passed through
            to every _plot_slice() call (MOD_QC and MOD_STATS). Axis tick
            labels, axis labels, panel titles, and colourbar text were
            previously fixed at plot_model_slices' internal defaults with
            no way to override them from this script.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Added the (2.3, 97.7) pair to QDIFF_PAIRS (default now
            [(15.9, 84.1), (2.3, 97.7)]), giving both a 1-sigma- and
            2-sigma-equivalent interquantile spread statistic.
            Added ens_err = sqrt(ens_var) ("err"): VAR is in
            (log10 Ω·m)² and was never on the same scale as MAD/QDIFF
            (log10 Ω·m), so MOD_STATS_WHAT's default now plots "err"
            instead of "var" (var is still computed, saved, and available
            as a MOD_STATS_WHAT entry on request). Added BOOTSTRAP_VAR /
            BOOTSTRAP_N / BOOTSTRAP_SEED: an alternative bootstrap
            estimate of the ensemble variance (_bootstrap_variance() new
            helper), resampling the N_members members with replacement
            BOOTSTRAP_N times and averaging the plug-in variance of each
            replicate -- generally more stable than the single plug-in
            estimate when N_members is small. Reports both var_boot and
            its own bootstrap standard error var_boot_se (diagnostic of
            how noisy the variance estimate itself is). err_boot =
            sqrt(var_boot) is added to MOD_STATS_WHAT automatically when
            BOOTSTRAP_VAR=True. All new arrays (err, and when enabled
            var_boot/err_boot/var_boot_se) are saved to the .npz.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Added MOD_SHOW_IN_SPYDER (default True): when this script is
            running inside Spyder (detected once via utl.runtime_env() ==
            "spyder"), every saved figure is also displayed inline via
            plt.show() (fviz.plot_model_slices' new show= parameter),
            without changing what gets saved to disk. No effect outside
            Spyder. Set MOD_SHOW_IN_SPYDER=False to disable even under
            Spyder.
2026-07-26  Claude Sonnet 5 (Anthropic)
            Fixed MOD_ROI_ZLIM default: was [0.0, 20000.0], which — now
            that the z-increases-downward sign convention is correct —
            clipped the ns/ew/plane panels exactly at the z=0 datum and
            hid any topography (mesh cells with z < 0) sitting above it.
            Changed to [-1000.0, 20000.0], giving 1 km of headroom above
            the datum so topography is included in the plotted range.
2026-07-27  Claude Sonnet 5 (Anthropic)
            Homogenized spatial config parameters to km: MOD_XLIM /
            MOD_YLIM / MOD_ZLIM, MOD_PROJECTION_DIST, MOD_ROI_PAD_XY,
            and MOD_ROI_ZLIM are now specified in km instead of metres
            (values and comments updated, e.g. MOD_ROI_ZLIM =
            [-1.0, 20.0] instead of [-1000.0, 20000.0]). The
            ROI-from-site-bbox block now divides the (still-metres)
            site coordinates by 1000 before combining them with the
            km-valued MOD_ROI_PAD_XY. femtic_viz.py itself is
            unchanged and still expects metres (matching mesh.dat);
            new module-level helpers _km_to_m() / _lim_km_to_m()
            convert these config values to metres at the
            fviz.plot_model_slices call site in _plot_slice().
            MOD_UTM_ORIGIN_E/N left in metres (absolute UTM geodetic
            convention, not a model-local span).
2026-07-27  Claude Sonnet 5 (Anthropic)
            MOD_PLOT_SITES_SLICES: default changed False -> True, so
            site markers (projected onto the plane, within
            MOD_PROJECTION_DIST) now also appear on the ns/ew/plane
            curtain panels by default, matching femtic_rto_prep.py /
            femtic_gst_prep.py. No femtic_viz.py change needed -- the
            sites_in_slices / site_marker_slices plumbing already
            existed; only this script's default was off.
2026-07-27  Claude Sonnet 5 (Anthropic)
            Extended the km homogenization to the MOD_SLICES slice-spec
            dicts, which were missed in the earlier pass: numeric
            x0/y0/z0 values (e.g. dict(kind="map", z0=5.0)) are now km
            instead of metres. New helper _slices_km_to_m() walks the
            list of dicts, scaling any plain-numeric x0/y0/z0 by 1000
            and leaving (value, "latlon") tuples / other keys
            untouched; applied right before
            fem.resolve_slice_positions(MOD_SLICES, ...) in
            _plot_slice().
2026-08-10  Claude Sonnet 5 (Anthropic)
            femtic_viz.py fix: N-S curtain panels ("ns" kind, e.g. the
            "N-S easting = ... km" panel in MOD_SLICES) were plotting
            mirrored left-right relative to true geography -- verified
            against a real ensemble QC figure where a resistive body at
            positive y/north appeared on the "S" side of the panel. Root
            cause and fix are in fviz.plot_model_slices' internal
            _axis_slice_params() helper; no changes needed here beyond
            picking up the updated femtic_viz.py. Also added
            MOD_TICK_DECIMALS (default None = unchanged formatting):
            controls the number of decimal digits shown on depth,
            easting/northing, and lat/lon axis tick labels, threaded
            through to fviz.plot_model_slices' new tick_decimals
            parameter in _plot_slice().
2026-08-12  Claude Sonnet 5 (Anthropic)
            Step (1)'s scan loop now checks os.path.isdir(d) before
            looking for femtic.cnv/the model file inside it. dir_list
            comes from utl.get_filelist(searchstr=[ENSEMBLE_NAME+"*"]),
            a glob-style match against ENSEMBLE_DIR that can in
            principle return non-directory matches (e.g. a stray file
            sharing the ENSEMBLE_NAME prefix); previously such an entry
            would fall through to os.path.join(d, "femtic.cnv") and get
            printed/counted as a skipped ensemble member as if it were
            a failed inversion run. Non-directory matches are now
            skipped immediately with their own message. No change for
            genuine run directories.
2026-08-12  Claude Sonnet 5 (Anthropic)
            A convergence diagnostic (nRMS bar chart / histogram over
            all scanned directories, not just accepted members) and a
            REPAIR procedure (rebuild non-converged directories with a
            log10-mean-of-2-random-converged-members starting model,
            for a restart) were prototyped in this script across several
            iterations today, then moved out into a new standalone
            script, femtic_ens_repair.py, once the design settled --
            keeping this script focused on its original scope (summary
            statistics, covariance, QC/statistics slice plots) rather
            than growing an unrelated directory-repair responsibility.
            No MOD_CONV*/MOD_REPAIR* config, conv_list, or step (1b)/(1c)
            remain here; see femtic_ens_repair.py and
            femtic_ens_repair_readme.md. fviz.plot_convergence_bar() and
            fviz.plot_convergence_histogram() (added to femtic_viz.py
            during the same work) are unaffected and still exported from
            femtic_viz.py for that script to use.
2026-08-13  Claude Sonnet 5 (Anthropic)
            Added femtic_ens_post_summary.md output at end of run: writes
            user-set (UPPERCASE) parameters, script path, and run
            date/time via utl.write_param_summary().
2026-08-21  Claude Sonnet 5 (Anthropic)
            Added COMPUTE_VAR_REDUX (default True): alongside each
            member's converged (posterior) model, the scan loop now also
            reads that member's iter0 (prior) model
            (resistivity_block_iter0.dat) into a second matrix,
            ens_matrix_prior. From it, ens_var_prior = np.var(
            ens_matrix_prior, axis=0) is computed (same log10(Ω·m)²
            space as ens_var) and var_redux = 1 - ens_var / ens_var_prior
            -- the fractional variance reduction achieved by the
            inversion, per free parameter. var_redux is only computed if
            every accepted member's iter0 file was found; otherwise it is
            skipped with a warning and the run proceeds unaffected (no
            change to the existing posterior-only statistics). Division
            by a zero prior variance is guarded (result set to nan for
            that parameter). f"{P}_var_prior" and f"{P}_var_redux" are
            saved to the .npz whenever computed. "var_redux" is appended
            to MOD_STATS_WHAT automatically when COMPUTE_VAR_REDUX=True
            (same pattern as "err_boot" for BOOTSTRAP_VAR), with
            MOD_STATS_CLIM entries for "var_prior" ([-.0, .5], matching
            "var") and "var_redux" ([0.0, 1.0], since it is a bounded
            fraction in typical use -- override either to None for
            auto-scaling, or to a custom range, as usual.
2026-08-21  Claude Sonnet 5 (Anthropic)
            Added MOD_STATS_BLANK_BY_REDUX (default False) + REDUX_EPS
            (default 0.1): when enabled (and var_redux was computed),
            free parameters with var_redux < REDUX_EPS -- i.e. cells the
            inversion barely moved away from the prior -- are blanked
            (MOD_STATS_BLANK_MODE, default "blank") in every MOD_STATS
            plot except var_redux's own. Implemented by writing the
            var_redux block file once up front (same path the "var_redux"
            MOD_STATS_WHAT entry would write anyway) and passing it as a
            per-call alpha_file/alpha_mode/alpha_blank_thresh override to
            _plot_slice(), which now accepts these as optional
            parameters (None = fall back to the existing module-level
            MOD_ALPHA_FILE/MODE/BLANK_THRESH, so MOD_QC and any run with
            MOD_STATS_BLANK_BY_REDUX=False are completely unaffected).
2026-09-06  Claude Sonnet 5 (Anthropic)
            Added MOD_SHOW_MODEL_CENTRE (default True): marks the model
            origin on "map" panels whenever MOD_DISPLAY_COORDS is
            "utm"/"latlon" via fviz.plot_model_slices'
            show_model_centre parameter; override style with a
            MOD_MAP_MARKERS entry carrying "is_model_centre": True.
2026-09-08  Claude Sonnet 5 (Anthropic)
            Added a "Sensitivity statistics" section mitigating the
            ambiguity that low ensemble spread (VAR/ERR/MAD) can mean
            either "well-resolved by the data" or "collapsed by the
            regularisation regardless of the data" (null space). Two
            independent, ensemble-level diagnostics, either individually
            switchable:
            (1) COMPUTE_SENS: per-member linearised cumulative
            sensitivity (Christiansen & Auken, 2012, Geophysics 77,
            WB171, doi:10.1190/geo2011-0393.1), read from each accepted
            member's Jacobian via the new fem.read_h5_sensitivity()
            (SENS_H5_PATTERN="model_iter{numit}.h5", schema-provisional
            -- see femtic_readme.md), then aggregated across the
            ensemble (mean/median/min/max/std/CV) instead of trusting a
            single member's linearisation; large CV flags cells where
            the sensitivity structure is strongly model-dependent
            (nonlinear regime), where SimRC below should be trusted
            instead.
            (2) COMPUTE_SIMRC: ensemble-native "SimRC" sensitivity (Bobe,
            Keller & Van De Vijver, 2021, Geophysical Prospecting,
            doi:10.1111/1365-2478.13068): per-member forward ("cal")
            responses are collected via the new
            fem.collect_forward_response() (SIMRC_RESULT_FILES,
            SIMRC_SITE_FILE) into a data ensemble matrix, and the
            cumulative |regression coefficient| and |correlation|
            between that data ensemble and the already-collected model
            ensemble (ens_matrix) are computed in data-column chunks
            (SIMRC_CHUNK) to bound memory -- no Jacobian needed. The
            model-side rows used are exactly the members that also
            produced a forward-response row (simrc_keep_idx), since
            COMPUTE_SENS/COMPUTE_VAR_REDUX can drop different members
            than COMPUTE_SIMRC does.
            Both (1) and (2) degrade gracefully -- printed warning,
            corresponding MOD_STATS_WHAT entries and .npz keys omitted --
            when their required per-member files are missing for some or
            all members, following the existing COMPUTE_VAR_REDUX
            pattern.
            Added COMPUTE_NULL_SPACE_FLAG (default True): flags free
            parameters with ERR below its own FLAG_SPREAD_PERCENTILE
            *and* low sensitivity by every available measure above
            (AND, not OR -- conservative, minimises false positives) as
            likely null-space rather than genuinely data-constrained;
            saved as boolean f"{P}_flag_null_space" and plottable/
            usable as an alpha-blank source the same way var_redux is.
            New .npz keys: f"{P}_sens_mean/_median/_min/_max/_std/_cv/
            _low_mask", f"{P}_simrc_coef/_corr/_low_mask",
            f"{P}_flag_null_space". New MOD_STATS_WHAT entries
            "sens_mean", "sens_cv", "simrc_corr", "flag_null_space"
            (auto-added when the corresponding COMPUTE_* flag is True,
            auto-removed if that diagnostic could not actually be
            computed for this ensemble).
2026-09-16  Claude Sonnet 5 (Anthropic)
            Adapted COMPUTE_SENS to the settled (2026-09-13) FEMTIC HDF5
            schema: model_iterX.h5/data_iterX.h5 were merged into one
            results_iterX.h5 per iteration (sibling /model, /data,
            /distortion groups), with the per-block cumulative
            sensitivity now precomputed by FEMTIC itself at
            /model/sensitivity/{raw,volume_normalised} -- no Jacobian
            assembly needed. SENS_H5_PATTERN changed from
            "model_iter{numit}.h5" to "results_iter{numit}.h5" (older,
            pre-2026-09-13 ensembles are no longer supported by this
            script). Replaced the hardcoded SENS_CUMSENS_KEY with
            SENS_KIND ("raw" | "volume_normalised"), matching
            fem.read_h5_sensitivity()'s updated cumsens_key default and
            automatic model/sensitivity group fallback (fem.py, same
            date). No fallback to jacobian.h5 if a member's
            results_iter{numit}.h5 lacks sensitivity for that iteration
            (by design -- jacobian.h5 is a separate, fixed-name file
            that may hold a different iteration entirely): that member
            is skipped for sensitivity with a printed warning, same as
            before. Updated stale "model_iterX.h5" wording in the
            COMPUTE_SENS section's comments, warnings, and variable
            comments to match.
2026-09-17  Claude Sonnet 5 (Anthropic)
            Generalised the MOD_STATS blanking/fading scheme from a single
            hardcoded var_redux switch to MOD_STATS_BLANK_SOURCES: a list
            of (source_name, direction, thresh) triples, source_name one
            of "var_redux"/"sens_mean"/"sens_cv"/"simrc_corr"/
            "flag_null_space" and direction one of "below" (blank where
            value < thresh -- var_redux/sens_mean/simrc_corr: low means
            poorly resolved), "above" (blank where value > thresh --
            sens_cv: high means the linearised sensitivity itself is
            untrustworthy there, a different question from resolution),
            or "flag" (blank where value != 0 -- flag_null_space is
            already boolean). Multiple entries combine via the new
            MOD_STATS_BLANK_COMBINE ("and"/"or", default "and"), with a
            printed warning whenever more than one entry is listed (e.g.
            flag_null_space alongside its own sens_mean/simrc_corr inputs
            is redundant, though harmless). Replaces the old
            MOD_STATS_BLANK_BY_REDUX boolean; default is an empty list
            (blanking off), matching the previous default. Internally,
            every source is turned into a plain boolean "blank this cell"
            mask in Python (per its own direction) before being combined
            and written out as a single 0/1 alpha-block file, so
            femtic_viz.py's existing alpha_file/alpha_mode/
            alpha_blank_thresh mechanism (unchanged) only ever sees the
            "value < 0.5" case it already handled for var_redux. Also
            added SENS_CV_HIGH_THRESH (default 1.0, i.e. std >= mean) for
            the "sens_cv"/"above" case, and NULL_SPACE_COMBINE ("and"/"or",
            default "and", matching prior hardcoded behaviour) generalising
            flag_null_space's own sens_low_mask/simrc_low_mask combination
            the same way.
2026-09-19  Claude Sonnet 5 (Anthropic)
            Plot-extent consistency with femtic_rto_prep.py /
            femtic_gst_prep.py: no functional change here -- this
            script's MOD_ROI_AUTO/MOD_ROI_PAD_XY/MOD_ROI_ZLIM block is
            the reference, and the two prep scripts now carry the same
            three variables with the same defaults and semantics.
            Comment/whitespace tidy only.

2026-09-25  Claude Opus 5.5 (Anthropic)
            Simplification and blanking fix.
            Root cause of "no blanked areas": fem.insert_model() writes
            10**v, while plot_model_slices() reads an alpha file's RAW
            region values -- the 0/1 blank mask arrived as 1/10 and never
            fell below alpha_blank_thresh=0.5, so nothing was ever blanked.
            The alpha block is now written with raw values == alpha
            (_write_alpha_block) and plotted with femtic_viz's new
            alpha_mode="direct".
            Blanking/fading reduced to ONE expression in the shared slice
            block: MOD_ALPHA_SOURCE = "<name> <op> <value>" (op in > >= <
            <=, name any computed statistic, regex-parsed, never eval'ed),
            with MOD_ALPHA_MODE ("blank"|"fade"), MOD_ALPHA_FADE_WIDTH and
            MOD_ALPHA_QC (apply to the best-member figure too). Removed
            MOD_ALPHA_FILE/_BLANK_THRESH and MOD_STATS_BLANK_SOURCES/
            _COMBINE/_MODE.
            Sensitivity aggregates (mean/median/min/max/std, raw/na/an) and
            SimRC sums (simrc_coef/_corr) are now log10 in memory, .npz and
            plots (values <= 0 -> NaN); sens_cv_* stay linear.
            Removed var_redux entirely (COMPUTE_VAR_REDUX, REDUX_EPS, iter0
            reads, var_prior). Removed the threshold/mask/flag machinery
            that only fed the old blanking: SENS_LOW_THRESH_FRAC,
            SENS_CV_HIGH_THRESH, SIMRC_LOW_THRESH_FRAC,
            SPREAD_LOW_THRESH_FRAC, COMPUTE_NULL_SPACE_FLAG,
            FLAG_SPREAD_PERCENTILE, NULL_SPACE_COMBINE and the
            *_low_mask / flag_null_space arrays, panels and .npz keys.
            MOD_STATS_CLIM replaced by MOD_STATS_STYLE (per-panel cmap/
            clim/label), moved to the shared slice block. Fixed clim=None
            silently falling back to MOD_CLIM in _plot_slice (auto-scaling
            never happened). MOD_STATS block files now use sentinel air
            (1e30) / ocean (1e-30) resistivities, plotted with matching
            air_log10_thresh/ocean_value, so a statistic value can no
            longer be painted as air or ocean. Requires the 2026-09-25
            femtic_viz.py (alpha_mode="direct", cbar_label,
            air_log10_thresh).
            Same day: per-member COMPUTE_SENS/COMPUTE_SIMRC skip reasons
            are collected and summarised after the scan loop, and the
            MOD_ALPHA_SOURCE "not available" message now states why (flag
            off, or first per-member failure reason).
            Same day: FIX sensitivity length mismatch (e.g. 54588 vs 54587
            -> every member skipped, no sens_* keys). FEMTIC writes one
            sensitivity value per resistivity block including fixed ones;
            new _sens_to_free() maps it onto the free parameters using the
            region table of the first member (len == n_free / nreg /
            nreg-1). Block order is verified once against
            /model/blocks["blockID"] (permutation undone if needed).
            Same day: COMPUTE_SIMRC now reads the calculated responses
            (cal_re/cal_im) and errors from /data/data of the same
            results_iter{numit}.h5 (fem.read_results_data) instead of
            per-member result_MT.txt / result_VTF.txt, which the current
            FEMTIC runs no longer write. Removed SIMRC_RESULT_FILES,
            SIMRC_SITE_FILE, SIMRC_DATA_KIND; new SIMRC_ERR_WEIGHT (default
            True: columns divided by the first member's errors, so |coef|
            sums over mixed datatypes are dimensionless). Imaginary parts
            of real-valued datatypes and columns with error <= 0 are
            dropped; members with NaN calculated values or a different
            data composition are skipped.
            Same day: MOD_TITLE_FONTSIZE (default None = MOD_LABEL_FONTSIZE
            + 6) for the file name shown as figure title
            (plot_model_slices figure_title_fontsize).
2026-09-26  Claude Sonnet 5 (Anthropic)
            Model-centre marker: see femtic_viz.py provenance (2026-09-26)
            -- default style changed from a thin black "+" (effectively
            invisible against the axes frame and other black markers) to
            a large yellow star with a black edge. No change here beyond
            picking up the updated femtic_viz.py; MOD_SHOW_MODEL_CENTRE /
            MOD_MAP_MARKERS wiring is unchanged.
            Added a region-of-interest diagnostic for the log10 sensitivity
            panels: every sens_*/simrc_* entry in MOD_STATS_WHAT that gets
            plotted now has its original (whole-mesh-normalised, as
            already stored/plotted) min/max printed to the console,
            restricted to the free parameters whose volume-weighted
            centroid (fem.build_region_geometry(), same free_idx mapping
            as _sens_to_free()) falls inside the resolved ROI
            (MOD_XLIM/MOD_YLIM/MOD_ZLIM, i.e. the same box already driving
            the MOD_STATS slice plots, including any MOD_ROI_AUTO
            override) -- meant to make picking a MOD_ALPHA_SOURCE
            threshold easier without guessing at the value range in view.
            New SENS_ROI_NORMALIZE (default False): when True, additionally
            re-centres each sens_* panel (not sens_cv_*, a linear ratio) so
            the ROI's own max becomes log10 = 0 instead of the whole-mesh
            max, right before that panel's block file is written --
            display-only, the .npz output and MOD_ALPHA_SOURCE evaluation
            are unaffected. simrc_coef/simrc_corr are diagnosed (min/max
            printed) but never renormalised by this switch, per explicit
            request -- no experience yet with tuning SimRC thresholds.
            New helper _roi_free_mask() (module scope, cached): resolves
            the ROI membership mask once per run and is reused across all
            sens_*/simrc_* panels; degrades gracefully (printed warning,
            ROI diagnostic skipped) if mesh/region geometry can't be read.
            Simplified the panel-selection interface per user request ("set
            panels to plot by a list of sources and one blanking option;
            determine the other settings from these"): MOD_ALPHA_SOURCE was
            already the one blanking option (2026-09-25). MOD_QC was a
            second, separate on/off switch that had to be kept in sync with
            MOD_STATS_WHAT by hand; it is now the special "best" entry in
            MOD_STATS_WHAT itself (default MOD_STATS_WHAT gained "best" as
            its first entry, preserving the previous MOD_QC=True default),
            handled by the existing QC code path (section 7) and skipped in
            the per-statistic loop (section 8). MOD_QC and MOD_STATS are
            still set (derived: MOD_QC = "best" in MOD_STATS_WHAT;
            MOD_STATS = bool(set(MOD_STATS_WHAT) - {"best"})) purely so
            every downstream use of those two names is unchanged. Turning
            either off is now just editing the one list (drop "best", or
            set MOD_STATS_WHAT = [] for neither). Scope note: MOD_NROWS/
            MOD_NCOLS (governs the MOD_SLICES grid per panel, an unrelated
            axis to "which panels") and the COMPUTE_SENS/COMPUTE_SIMRC/
            COMPUTE_COV/BOOTSTRAP_VAR .npz-computation switches were left
            untouched by design -- both control things beyond what a single
            "what to plot" list can safely imply (grid size has no relation
            to len(MOD_STATS_WHAT); the COMPUTE_* flags also gate what is
            saved to the .npz regardless of whether it is ever plotted).
2026-10-07  Claude Sonnet 5.5 (Anthropic)
            Vertical sections are now configured in MOD_SLICES itself, as
            dict(kind="plane", point=[x, y] km, strike=azimuth deg,
            dip=90); the default list ends with a SW-NE (strike 45) section
            through the model origin. New helper _plane_spec_km_to_m()
            converts such entries (km -> m) in _plot_slice(); other kinds
            still go through fem.resolve_slice_positions and the original
            order is kept. The section uses the same limits as the other
            slices: along-strike range = MOD_XLIM x MOD_YLIM projected onto
            the strike (whole mesh if either is None; optional per-entry
            xlim in km), depth range = MOD_ZLIM; horizontal axis in model
            km. Requires the updated femtic_viz.py (vertical "plane"
            panels, 2026-10-07). Also fixed a missing comma in the default
            MOD_STATS_WHAT that silently merged "sens_median_an" and
            "sens_cv_an" into the non-existent key
            "sens_median_ansens_cv_an" (neither panel was plotted).
2026-10-11  Claude Sonnet 5.5 (Anthropic)
            New optional block MOD_BH (step (9)): 1-D resistivity-depth
            profiles at borehole positions, plotted from the ensemble
            statistics (avg, med, percentiles), the individual members
            ("ensemble", drawn with opacity alpha) and the best member
            ("best"). MOD_BH is a list of dicts, analogous to MOD_SLICES
            (horizontal position as model-local km or CRS-tagged
            (value, "latlon"/"utm"), depths in km, plus a "what" list);
            an empty list switches the block off. The borehole element
            search is done once per borehole and reused for every
            statistic and every member, so sampling the whole ensemble
            costs one array lookup per member. Shares MOD_PLOT_FORMAT,
            MOD_DPI, MOD_CLIM (x-range), the font sizes, MOD_DEPTH_KM,
            MOD_OCEAN/MOD_OCEAN_RHO and the UTM-origin resolution of the
            slice figures; step (5) now also runs when only MOD_BH is
            set. Sampled profiles are exported to an .npz. AI-generated
            code -- review before production use.
2026-10-11c Claude Sonnet 5.5 (Anthropic)
            MOD_BH: "n_random" / "seed" keys draw N sites at random from a
            "sites" template; "sigma1" / "sigma2" in "what" draw the
            1-/2-sigma quantile pairs (P15.87/P84.13, P2.28/P97.72 of the
            members, computed directly, independent of PERCENTILES). The
            default MOD_BH entry now draws the ensemble at alpha=0.05
            underneath the median and the 1-sigma quantiles. AI-generated
            code -- review before production use.
2026-10-11b Claude Sonnet 5.5 (Anthropic)
            MOD_BH: (1) new "sites" key -- a template entry expanded to one
            borehole per site of MOD_SITE_DAT ("all" or a list of names);
            MOD_BH_PER_FIG pages many panels over several figures and
            MOD_BH_NCOLS = None now means up to 4 columns. (2) Depth is
            z positive downward (axis increases downward, values above the
            datum are negative), same as the curtain panels. (3) The mesh
            is read by fem.read_femtic_mesh only (the femtic_viz copy was
            removed); the old x/y swap is no longer applied by default and
            is an option, MOD_MESH_SWAP_XY (None = femtic default). This
            also makes _roi_free_mask() read the mesh in the same axis
            convention as the plots. AI-generated code -- review before
            production use.
2026-10-20  Claude Sonnet 5.5 (Anthropic)
            "best" figure title now has a second line naming the member
            (directory, iteration, nRMS), via a new figure_title argument of
            _plot_slice() passed on to fviz.plot_model_slices. AI-generated
            code -- review before production use.
2026-10-20d Claude Sonnet 5.5 (Anthropic)
            Per-member best nRMS written to NRMS_CSV (sorted, with status)
            and plotted as a histogram (MOD_NRMS_HIST, MOD_NRMS_HIST_FILE,
            MOD_NRMS_BINS). AI-generated code -- review before use.
2026-10-20c Claude Sonnet 5.5 (Anthropic)
            New user setting RESULTS_DIR (default <ENSEMBLE_DIR>/results/);
            results .npz, "best" figure, stats_plots/ and borehole outputs
            now default to it. AI-generated code -- review before use.
2026-10-20b Claude Sonnet 5.5 (Anthropic)
            Best-nRMS member also saved in the results .npz: <P>_best (vector),
            <P>_best_member, <P>_best_file, <P>_best_iter, <P>_best_nrms.
            AI-generated code -- review before production use.
2026-10-27  Claude Sonnet 5.5 (Anthropic)
            Optional reference model. New settings MOD_REF_FILE (block file
            on the MOD_MESH mesh; None = off), MOD_REF_SLICES, MOD_REF_BH,
            MOD_REF_LEVELS, MOD_REF_STYLE, MOD_REF_LABEL. Slices: the
            reference is drawn as iso-resistivity lines on the best panel and
            on every MOD_STATS panel (femtic_viz.plot_model_slices,
            reference_*). Boreholes: dashed "ref" line (MOD_BH_STYLE["ref"]),
            added to every MOD_BH panel unless the entry has ref=False, or
            listed in "what"; also saved as ref_<name> in the borehole .npz
            (header: reference_file). A missing or wrong-mesh reference is
            reported and switched off, the run continues. New MOD_VERT_EXAG
            (default None = unchanged behaviour): vertical exaggeration of the
            ns / ew / plane panels from real axis units, also for
            MOD_DISPLAY_COORDS="latlon", where MOD_EQUAL_ASPECT alone does not
            act. Needs the 2026-10-27 femtic_viz.py. AI-generated code --
            review before production use.
2026-10-27b Claude Sonnet 5.5 (Anthropic)
            Plot-only mode. New settings PLOT_ONLY (default False),
            PLOT_ONLY_NPZ (None = ENSEMBLE_RESULTS) and PLOT_ONLY_TEMPLATE.
            With PLOT_ONLY=True, sections (1)-(4b) (member scan, statistics,
            sensitivity / SimRC, covariance, writing the .npz and the nRMS
            table) are replaced by reading the existing results .npz; the
            variables the plotting sections use are rebuilt from it (members
            with their nRMS, ensemble matrix, avg / var / err / med / mad /
            percentiles, qdiff_*, *_boot, sens_*, simrc_*; the percentile
            levels come from the file; missing avg ... mad / prc are
            recomputed from the members) and sections (5)-(9) run unchanged:
            "best" panel, MOD_STATS_WHAT panels, MOD_ALPHA_SOURCE blanking,
            boreholes (with the reference model, MOD_VERT_EXAG etc.), and the
            nRMS histogram from NRMS_CSV (also searched next to the .npz).
            The template block file is the best member's own file as stored
            in the .npz, or PLOT_ONLY_TEMPLATE when that is gone (results
            moved to another machine); the "best" panel is then drawn from the
            stored best vector. The .npz is never rewritten. In this mode the
            default MOD_STATS_WHAT lists the sens / simrc / bootstrap panels
            regardless of the COMPUTE_* / BOOTSTRAP_VAR flags (keys not in the
            file are skipped with a message) and their MOD_STATS_STYLE entries
            are always defined. The full-run code path is unchanged (figures
            and .npz identical to the previous version, tested). The block
            "(1) ... (4b)" is indented under "else:" of "if PLOT_ONLY:".
            AI-generated code -- review before production use.
2026-10-27c Claude Sonnet 5.5 (Anthropic)
            Fallback for a missing block file. New setting ITER_FALLBACK
            (default True). If resistivity_block_iter<N>.dat of a member's
            best-nRMS iteration N does not exist, the member's model is taken
            from the best iteration that has a block file (iteration > 0,
            lowest nRMS among those with a file, ties -> lower iteration;
            usually N-1) instead of dropping the member; NRMS_MAX is applied
            to that iteration's nRMS, and the member enters the ensemble (and
            its sensitivity file, SENS_H5_PATTERN) with that iteration. Logged
            per member; status "accepted_fallback" in NRMS_CSV, counted in
            the nRMS histogram legend (the histogram treats every status
            starting with "accepted" as accepted). Without a usable
            alternative the member is still "no_model_file". ITER_FALLBACK=
            False restores the previous behaviour. With nothing missing the
            results are identical to the previous version (tested).
            AI-generated code -- review before production use.
2026-10-28  Claude Sonnet 5.5 (Anthropic)
            The MOD_ALPHA_SOURCE mask (blank / fade) is no longer applied to
            the sensitivity panels themselves (sens_*, simrc_*): shading a
            sensitivity map by sensitivity is circular, and the unmasked map
            shows where the threshold cuts. New MOD_ALPHA_EXCLUDE (tuple of
            key prefixes, default ("sens_", "simrc_"); () restores the old
            behaviour of masking every statistic panel). All other panels and
            the best panel (MOD_ALPHA_QC) are masked as before.
            AI-generated code -- review before production use.
2026-10-29  Claude Sonnet 5.5 (Anthropic)
            MOD_BH: new "heatmap" item for the "what" list -- a density
            heat map of the ensemble behind the other borehole items. At
            every depth level the members' log10(rho) are histogrammed
            (_bh_heatmap) and drawn with pcolormesh (resistivity on the
            log x axis, depth down); one shared colour bar per figure.
            New settings MOD_BH_HEAT_BINS (int, or list of log10(rho)
            edges; per-entry "heat_bins"), MOD_BH_HEAT_NORM ("fraction":
            fraction of the members per bin, "row_max": each depth scaled
            to its maximum), MOD_BH_HEAT_CMAP, MOD_BH_HEAT_SCALE ("linear" |
            "sqrt" | "log" colour scale), MOD_BH_HEAT_VMIN, MOD_BH_HEAT_VMAX
            (explicit limits show as arrows on the colour bar; a log scale
            leaves empty bins blank), MOD_BH_HEAT_CBAR, and
            MOD_BH_STYLE["heatmap"] (cmap and pcolormesh keywords,
            overriding MOD_BH_HEAT_CMAP, default Greys). The bins span the panel's xlim (entry "xlim",
            MOD_BH_XLIM, 10**MOD_CLIM) or, when none is set, the range of
            the members; members outside it are not shown (fractions
            still refer to all valid members). Auto colour maximum
            (MOD_BH_HEAT_VMAX = None) = 99th percentile of the non-zero
            fractions of the figure, so that levels with a single value in
            every member (ocean, fixed regions) do not wash out the rest.
            "heatmap" can be combined with "ensemble", "med", "sigma1", ...
            (drawn on top of it) or used alone. The borehole .npz gets
            heat_<name> (n_levels x n_bins fractions) and
            heat_edges_<name> (log10 edges), and the header the number of
            bins ("heat_bins") and the keys' unit text ("heat_unit").
            Nothing changes unless "heatmap" is listed in a MOD_BH entry
            (default entries unchanged). AI-generated code -- review
            before production use.
2026-10-31  Claude Sonnet 5.5 (Anthropic)
            New ENS_MODEL_SOURCE ("converged" default | "perturbed"): the
            ensemble can be built from each member's PERTURBED REFERENCE
            (ENS_PERTURBED_FILE, default "referencemodel.dat" -- the RTO
            prior draw femtic_rto_prep.py writes per member) instead of its
            converged model resistivity_block_iter<N>.dat. Everything
            downstream (statistics, covariance, slices, "best" panel,
            boreholes, heat maps) then describes the prior ensemble. The
            member set is unchanged (still chosen from the converged
            inversion: NRMS_MAX, ITER_FALLBACK), so prior and converged
            ensembles compare like for like. The base, unperturbed reference
            (MOD_REF_FILE) is drawn as before. ENSEMBLE_PREFIX gets
            ENS_PERTURBED_SUFFIX ("_perturbed") so the results .npz, nRMS
            table, figures and .npz keys do not overwrite those of the
            converged run; the .npz stores <P>_model_source and
            <P>_model_file_name, PLOT_ONLY reads them (older files: converged),
            and the borehole .npz header gets "ensemble_source". COMPUTE_SIMRC
            is switched off in this mode (models vs converged responses);
            COMPUTE_SENS (sensitivities of the converged inversion) is not.
            A missing file skips the member ("no_perturbed_file" in
            NRMS_CSV); a symbolic link (shared, unperturbed -- e.g.
            resistivity_block_iter0.dat without MOD_START_FROM_PERTURBED) or
            zero spread is reported as a warning. With the default the
            script behaves as before. AI-generated code -- review before
            production use.
2026-11-02  Claude Sonnet 5.5 (Anthropic)
            Heat maps of the converged AND the perturbed models with a
            common colour bar (same options as femtic_ens_plot.py).
            (1) ENS_MODEL_SOURCE = "both": the run (prefix, statistics,
            slices, results .npz) is the converged run, and each member's
            perturbed reference (ENS_PERTURBED_FILE) is loaded in addition
            (a member lacking it is skipped, "no_perturbed_file"); MOD_BH
            then draws every borehole for both ensembles (heat map, "ensemble"
            lines, avg / med / percentiles / sigma, "best"), converged and
            perturbed side by side in one figure with ONE colour bar
            (MOD_BH_HEAT_FIGURE = "combined"), or as two figures
            ("separate"); the perturbed profiles go to
            <MOD_BH_FILE>_perturbed.npz. The perturbed matrix is stored in
            the results .npz as <P>_ens_perturbed, so PLOT_ONLY can draw
            "both" too. (2) New MOD_BH_HEAT_SHARE_SCALE (default False = one
            colour scale per figure, as before; True = ONE scale -- bin edges,
            colour limits, colormap -- for all figures of the run).
            (3) New MOD_BH_HEAT_SCALE_FILE / MOD_BH_HEAT_SCALE_IO ("auto" |
            "read" | "write"): the resolved colour scale is saved as JSON and
            read back by other runs (a converged and a perturbed run, other
            ensembles, femtic_ens_plot.py -- same file format), so their heat
            maps share the colour bar; every key of the file is optional.
            With the defaults the output is unchanged (the borehole .npz
            header gains "colour_scale"). AI-generated code -- review before
            production use.
2026-11-06  Claude Sonnet 5.5 (Anthropic)
            "Directory scan report" appended to femtic_ens_post_summary.md
            (the file utl.write_param_summary writes; its return value gives
            the path): every scanned directory that is not in the ensemble,
            with a reason key (no_cnv_file, cnv_unreadable,
            cnv_no_iterations, rejected_nrms, no_model_file,
            no_perturbed_file) and a detail text, counts per reason, the
            members accepted via ITER_FALLBACK, and glob matches that are not
            directories. Unlike NRMS_CSV (which has no row for directories
            without a usable convergence file) it accounts for every
            scanned directory. Not written in PLOT_ONLY mode (nothing is
            scanned). If the summary cannot be appended to, the report goes to
            femtic_ens_post_scan_report.md instead. The selection of members
            itself is unchanged. Follow-up the same day: the report said
            "255 directories scanned" for an ensemble numbered 0..255
            without saying which one was missing. It now always shows the
            number of glob matches (directories scanned + matches that are
            not directories, e.g. a broken symbolic link) and has a
            "Directory numbering" part: it reads the number after
            ENSEMBLE_NAME in every matching name and lists numbers in the
            expected range with no entry at all, numbers matched only by a
            non-directory, numbers matched by several names (e.g. X and
            X_restart) and names without a number. New optional
            ENSEMBLE_INDEX_RANGE = (first, last), default None = the range
            found in the names (which cannot show a missing first or last
            member). AI-generated code -- review before production use.

2026-11-08  Claude Sonnet 5.5 (Anthropic)
            Split into a computing script and plotting scripts. This
            script keeps the member scan, the statistics, the sensitivity /
            SimRC / covariance / bootstrap calculations, the results .npz,
            the nRMS CSV and the parameter summary with the directory scan
            report. Moved out: the "best" and statistics slice panels (and
            MOD_ALPHA_SOURCE blanking, the reference-model lines, ROI and
            origin/site handling) -> femtic_ens_plot_slices.py; the borehole
            profiles and heat maps (MOD_BH*) -> femtic_ens_plot_bh.py; the
            nRMS histogram -> method "nrms" of femtic_data_misfit.py; the
            PLOT_ONLY mode -> the results-file reader
            femtic.read_ensemble_results() that these scripts use. Reusable code went to femtic.py (Section 6c) and
            femtic_viz.py (build_scan_report / append_scan_report are now
            femtic.py functions, shared with femtic_ens_repair.py). All MOD_*
            plotting settings, MOD_STATS_WHAT, PLOT_ONLY* and MOD_NRMS_* are
            removed from this script; MOD_OCEAN stays (block-file reader).
            The .npz additionally holds <P>_nrms_table (member, best iter,
            nRMS, status of EVERY scanned member that has an nRMS, i.e. also
            the rejected ones) and <P>_nrms_max, so that the misfit
            histogram needs no CSV. The statistics and keys written are
            otherwise unchanged.
            New option ENS_SUBSAMPLE_N (+ ENS_SUBSAMPLE_SEED,
            ENS_SUBSAMPLE_SUFFIX): draw N members at random, without
            replacement, from the accepted members (after NRMS_MAX /
            ITER_FALLBACK) and compute everything from them; the drawn rows
            are stored in the .npz (<P>_subsample_idx, _subsample_seed,
            _subsample_n_accepted), the nRMS table keeps every scanned member
            (accepted members that were not drawn get the status
            "accepted_not_drawn"). The parameter summary is now written to
            ENSEMBLE_DIR (out_dir=ENSEMBLE_DIR, as in femtic_ens_repair.py)
            instead of next to the script. AI-generated code -- review
            before production use.

    This script targets FEMTIC's current HDF5 output layout only.
    AI-generated code -- review before production use.
"""


from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

import sklearn.covariance
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg


rng = np.random.default_rng()
nan = np.nan

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# ===========================================================================
# ---------------------------------------------------------------------------
# Ensemble input
# ---------------------------------------------------------------------------
# ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_0_reset-err/"
# ENSEMBLE_NAME = "annecy_gst_"
# #: Prefix used for .npz output keys and default file/figure names.
# #: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
# ENSEMBLE_PREFIX = "annecy_gst_0"

ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"
ENSEMBLE_NAME = "ann_rto_orig-err_"
#: Prefix used for .npz output keys and default file/figure names.
#: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
ENSEMBLE_PREFIX = "annecy_rto_orig-err"

#: Expected member numbers as (first, last), inclusive, e.g. (0, 255) for 256
#: members named <ENSEMBLE_NAME>0 ... <ENSEMBLE_NAME>255. Only used by the
#: "Directory numbering" part of the scan report (parameter summary): numbers
#: in this range for which no directory was found are listed. None = the
#: range found in the names themselves (cannot show a missing first or last
#: member).
ENSEMBLE_INDEX_RANGE = None

# ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/misti2026/ensemble_gst_0/"
# ENSEMBLE_NAME = "misti_rnd_0_"
# #: Prefix used for .npz output keys and default file/figure names.
# #: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
# ENSEMBLE_PREFIX = "misti_rnd_0"

# ---------------------------------------------------------------------------
# Which model of each member enters the ensemble
# ---------------------------------------------------------------------------
#: "converged" (default) -- the member's best-fit model,
#:     resistivity_block_iter<N>.dat (N = best-nRMS iteration from femtic.cnv).
#: "perturbed" -- the member's PERTURBED REFERENCE (the RTO prior draw)
#:     instead: the file ENS_PERTURBED_FILE inside the member directory. The
#:     whole run (mean, median, percentiles, covariance, slices, boreholes,
#:     heat maps, "best" panel) then describes the prior ensemble rather than
#:     the posterior one. Which members take part is unchanged: they are still
#:     chosen from the converged inversion (NRMS_MAX, ITER_FALLBACK, a block
#:     file must exist for the best iteration), so the prior and the converged
#:     ensemble are made of the SAME members and compare like for like; raise
#:     NRMS_MAX to include every member. "best" is still the best-nRMS member
#:     of the converged inversion. The perturbed models scatter around the base
#:     (unperturbed) reference. COMPUTE_SIMRC is switched off in this mode (it correlates the
#:     models with the converged responses); the sensitivities (COMPUTE_SENS)
#:     are those of the converged inversion and stay on.
#: "both" -- the run is the converged run (ENSEMBLE_PREFIX, statistics, slices,
#:     results .npz exactly as for "converged"), and each member's perturbed
#:     reference (ENS_PERTURBED_FILE) is loaded IN ADDITION. A member lacking
#:     ENS_PERTURBED_FILE is skipped. The perturbed models are stored in the
#:     results .npz (<P>_ens_perturbed); femtic_ens_plot_bh.py then draws every
#:     borehole for both ensembles on one common colour scale. Nothing else
#:     changes (statistics, slices, covariance are those of the converged run).
ENS_MODEL_SOURCE = "both"  # "converged" | "perturbed" | "both"

#: File of the perturbed reference inside each member directory.
#: femtic_rto_prep.py writes the RTO draw to "referencemodel.dat" (a real
#: file per member). "resistivity_block_iter0.dat" holds the perturbed model
#: only if the ensemble was prepared with MOD_START_FROM_PERTURBED = True; else
#: it is a symbolic link to the shared, unperturbed template. A link, or a
#: set of identical members, is reported as a warning.
ENS_PERTURBED_FILE = "referencemodel.dat"

#: Appended to ENSEMBLE_PREFIX when ENS_MODEL_SOURCE = "perturbed", so that the
#: results .npz, nRMS table, figures and .npz keys of the two ensembles never
#: overwrite each other. Set the same ENS_MODEL_SOURCE in the plotting
#: scripts (they read it from the .npz and need no setting, but the file name
#: ENSEMBLE_RESULTS must match).
ENS_PERTURBED_SUFFIX = "_perturbed"

if ENS_MODEL_SOURCE not in ("converged", "perturbed", "both"):
    raise ValueError(
        f"ENS_MODEL_SOURCE={ENS_MODEL_SOURCE!r}: use 'converged', 'perturbed' "
        f"or 'both'."
    )
if (
    not ENS_PERTURBED_FILE
    or os.path.basename(ENS_PERTURBED_FILE) != ENS_PERTURBED_FILE
):
    raise ValueError(
        f"ENS_PERTURBED_FILE={ENS_PERTURBED_FILE!r}: give a file name inside the "
        f"member directory, without a path."
    )
#: Which model the rows of the run's ensemble are ("both": the converged ones).
#: The plotting scripts read the same information from the .npz
#: (<P>_model_source).
_ENS_SRC = "converged" if ENS_MODEL_SOURCE == "both" else ENS_MODEL_SOURCE
if ENS_MODEL_SOURCE == "perturbed":
    ENSEMBLE_PREFIX = ENSEMBLE_PREFIX + ENS_PERTURBED_SUFFIX

#: Name of the FEMTIC convergence file inside each ensemble member
#: directory. A single name or a list of candidates (first existing wins),
#: e.g. CNV_FILE = ["femtic.cnv", "femtic_run2.cnv"].
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

#: Maximum normalised RMS accepted from the convergence file (CNV_FILE).
NRMS_MAX = 1.5

#: A member's model is taken from its best-nRMS iteration N
#: (resistivity_block_iter<N>.dat). If that file does not exist (FEMTIC wrote
#: no block file for that iteration), True = use the best iteration that does
#: have a block file instead -- in practice usually N-1: the iteration > 0
#: with the lowest nRMS among those with a file (ties: the lower iteration),
#: and NRMS_MAX is then applied to ITS nRMS. The member enters with that
#: iteration's nRMS and model (also for its sensitivity file), is reported in
#: the log, and has status "accepted_fallback" in NRMS_CSV. If no such
#: iteration exists or its nRMS exceeds NRMS_MAX, the member is skipped as
#: "no_model_file". False = skip it without looking for another iteration.
ITER_FALLBACK = True


# ---------------------------------------------------------------------------
# Random subsample of the ensemble
# ---------------------------------------------------------------------------
#: None (default) = use all accepted members. An integer N = draw N members at
#: random, without replacement, from the ACCEPTED members (those that passed
#: NRMS_MAX / ITER_FALLBACK and have a model file, see above) and compute
#: everything from these N only: statistics, sensitivity, SimRC, covariance,
#: bootstrap, the "best" member (= best nRMS within the draw) and the results
#: .npz. The scan still reads all accepted members once; switch off
#: COMPUTE_SENS / COMPUTE_SIMRC for a fast run. N >= number of accepted
#: members: all are used. The rows drawn are stored in the .npz
#: (<P>_subsample_idx, indices into the accepted members in scan order), so a
#: draw with ENS_SUBSAMPLE_SEED = None can be reproduced afterwards.
ENS_SUBSAMPLE_N = None

# if ENS_SUBSAMPLE_N is None:
#     ENS_SUBSAMPLE_N = 999
    
#: Appended to ENSEMBLE_PREFIX when a draw is made (ENS_SUBSAMPLE_N not None),
#: so that the .npz, nRMS table and keys of the subsample never overwrite
#: those of the full ensemble. May contain {n} (= ENS_SUBSAMPLE_N) and {seed}
#: (= ENS_SUBSAMPLE_SEED), e.g. "_sub{n}_s{seed}".
ENS_SUBSAMPLE_SUFFIX = "_sub{n}"
#: Seed of the draw. None = fresh OS entropy (a different draw each run); an
#: int = reproducible draw.
ENS_SUBSAMPLE_SEED = None



if ENS_SUBSAMPLE_N is not None:
    if int(ENS_SUBSAMPLE_N) != ENS_SUBSAMPLE_N or ENS_SUBSAMPLE_N < 1:
        raise ValueError(
            f"ENS_SUBSAMPLE_N={ENS_SUBSAMPLE_N!r}: use None or an integer >= 1."
        )
    ENS_SUBSAMPLE_N = int(ENS_SUBSAMPLE_N)
    ENSEMBLE_PREFIX = ENSEMBLE_PREFIX + ENS_SUBSAMPLE_SUFFIX.format(
        n=ENS_SUBSAMPLE_N, seed=ENS_SUBSAMPLE_SEED
    )

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
#: Percentile levels. Default: 2-σ / 1-σ normal-equivalent.
PERCENTILES = [2.3, 15.9, 50.0, 84.1, 97.7]

#: Percentile-pair differences to compute as extra spread statistics, e.g.
#: (15.9, 84.1) -> the 1-sigma-equivalent interquantile range
#: |P84.1 - P15.9|ᵢ per free parameter (an outlier-robust alternative to
#: VAR/MAD). (2.3, 97.7) similarly gives the 2-sigma-equivalent range.
#: Each value in a pair must also appear in PERCENTILES. Stored in the
#: .npz as f"{P}_qdiff_<lo>_<hi>" and available for the statistics panels of
#: femtic_ens_plot_slices.py (MOD_STATS_WHAT) under the same key.
QDIFF_PAIRS = [(15.9, 84.1), (2.3, 97.7)]

# ---------------------------------------------------------------------------
# Sensitivity statistics
# ---------------------------------------------------------------------------
# Low ensemble spread (small VAR/ERR/MAD above) can mean either that the
# data genuinely constrain a cell, or that regularisation collapsed a
# null-space cell regardless of the data -- the two are indistinguishable
# from spread alone. This section adds two independent, ensemble-level
# sensitivity diagnostics to disambiguate them:
#
#   (1) SENS_*   -- per-member linearised cumulative sensitivity
#                   (Christiansen & Auken, 2012, Geophysics 77, WB171,
#                   doi:10.1190/geo2011-0393.1), read from each accepted
#                   member's results_iter<numit>.h5 via
#                   fem.read_h5_sensitivity() -- FEMTIC writes this
#                   precomputed, at /model/sensitivity/{raw,volume_normalised},
#                   whenever sensitivity was computed for that iteration, so
#                   no Jacobian assembly is needed here -- then aggregated
#                   (mean, median, min, max, std) across the ensemble instead
#                   of trusting a single member's linearisation.
#   (2) SIMRC_*  -- ensemble-native "SimRC" sensitivity (Bobe, Keller &
#                   Van De Vijver, 2021, Geophysical Prospecting,
#                   doi:10.1111/1365-2478.13068): regression-coefficient
#                   and correlation between the model ensemble
#                   (ens_matrix, already collected above) and a forward-
#                   response ensemble read from /data/data (cal_re/cal_im)
#                   of the same results_iter<numit>.h5. Needs no Jacobian, and
#                   -- being built from the finite ensemble spread rather
#                   than an infinitesimal derivative -- remains valid
#                   even where the problem is locally nonlinear.
#
# Either block can be disabled independently; each degrades gracefully
# (printed warning, keys omitted from the .npz) if its required files are
# missing for some or all members.

# --- (1) Per-member Jacobian-based cumulative sensitivity -------------------
#: Set False to skip entirely (no results_iterX.h5 files available yet).
COMPUTE_SENS = True

#: Per-member HDF5 file, relative to each ensemble-member run directory,
#: with "{numit}" substituted by that member's best (accepted) iteration
#: number -- i.e. the same iteration whose resistivity block was used to
#: build ens_matrix above.
#:
#: This is FEMTIC's current (settled 2026-09-13) merged output file --
#: model + data + distortion as sibling /model, /data, /distortion groups
#: in one HDF5 file per iteration, replacing the older separate
#: model_iterX.h5/data_iterX.h5 pair. Older ensemble runs predating this
#: change are not supported by this script; rerun those inversions with a
#: current FEMTIC build if their sensitivity is needed here.
SENS_H5_PATTERN = "results_iter{numit}.h5"

#: See fem.read_h5_sensitivity() for the full lookup rules. group=None
#: tries the file root, then falls back to "model/sensitivity"
#: automatically -- FEMTIC's real location for these datasets in
#: results_iterX.h5 -- so no group override is normally needed.
SENS_H5_GROUP = None

#: Which of FEMTIC's two precomputed per-block sensitivity datasets to use:
#:   "raw"                -- sum|J[:,block]| across all data rows (Ohm.m
#:                            units follow the model parameterisation)
#:   "volume_normalised"  -- the above divided by block volume (m^-3),
#:                            i.e. sensitivity density
#: Passed through as fem.read_h5_sensitivity()'s cumsens_key.
SENS_KIND = "volume_normalised"
assert SENS_KIND in (
    "raw",
    "volume_normalised",
), f"SENS_KIND must be 'raw' or 'volume_normalised', got {SENS_KIND!r}"

#: FEMTIC's per-block sensitivity in results_iterX.h5 (whichever SENS_KIND
#: dataset is used) is NOT normalised to each member's own max -- confirmed
#: against an actual results_iterX.h5 (raw min/max did not peak at 1). So
#: there is no single "correct" way to bring per-member sensitivities onto
#: a common, threshold-friendly [0, 1]-ish scale before aggregating across
#: the ensemble; two different, both defensible, orders of operations are
#: computed side by side below instead of picking one:
#:
#:   "_na" ("normalise, then average") -- each member's own raw vector is
#:         divided by ITS OWN max right after reading (so every member
#:         contributes equally regardless of its absolute sensitivity
#:         scale), THEN mean/median/min/max/std/cv are taken across the
#:         (now per-member-normalised) ensemble. A member with uniformly
#:         weak sensitivity everywhere still gets to say "relatively
#:         speaking, my most-sensitive cell is here".
#:   "_an" ("average, then normalise") -- mean/median/min/max/std/cv are
#:         taken across the RAW ensemble first (so a member with larger
#:         absolute sensitivity dominates the average, same as a plain
#:         ensemble mean would), THEN the resulting aggregate is divided
#:         by ITS OWN max, purely so the final numbers/threshold sit on a
#:         [0, ~1] scale (log10 <= 0) for plotting and the alpha mask
#:         (MOD_ALPHA_SOURCE in femtic_ens_plot_slices.py).
#:
#: Both give a max-normalised statistic (_na's own max may be < 1, since
#: different members peak at different cells; _an's is exactly 1 by
#: construction), i.e. log10 values <= 0 after the log10 conversion. Both
#: are always computed when COMPUTE_SENS succeeds; use whichever (or both)
#: in MOD_STATS_WHAT / MOD_ALPHA_SOURCE of femtic_ens_plot_slices.py.

#: Jacobian/error dataset names, used only if SENS_KIND's dataset is not
#: found in results_iter{numit}.h5 (e.g. sensitivity was disabled for that
#: iteration in control.dat). These never resolve against results_iterX.h5
#: in practice (the raw Jacobian only exists in the separate, fixed-name
#: jacobian.h5 -- which holds an arbitrary, possibly different iteration,
#: and is deliberately NOT consulted here): a missing SENS_KIND dataset
#: therefore falls straight through to "not found", and that member is
#: skipped for sensitivity with a printed warning -- see the COMPUTE_SENS
#: loop below. Left in place only because fem.read_h5_sensitivity() always
#: takes these arguments; do not point them at jacobian.h5.
SENS_JACOBIAN_KEY = "jacobian"
SENS_ERROR_KEY = "data_errors"  # None to disable error-weighting

#: All sensitivity aggregates (sens_mean/_median/_min/_max/_std, in their
#: _raw/_na/_an versions) and the SimRC sums (simrc_coef/_corr) are strictly
#: positive and span many decades, so they are converted to log10 right
#: after aggregation -- in memory, in the .npz, in the plotted block files,
#: and in MOD_ALPHA_SOURCE expressions of femtic_ens_plot_slices.py
#: (e.g. "sens_mean_an > -3." means
#: "an-normalised mean sensitivity above 1e-3 of its maximum"). Values <= 0
#: (e.g. a cell with exactly zero sensitivity) become NaN. sens_cv_na/_an
#: are ratios and stay linear.

# --- (2) Ensemble-native SimRC sensitivity ----------------------------------
#: Set False to skip entirely.
#: The forward responses are read from the same per-member
#: results_iter{numit}.h5 as the sensitivity (SENS_H5_PATTERN, best-nRMS
#: iteration), group /data/data: calculated values cal_re/cal_im, with
#: re_err/im_err for weighting. One data column per real part, plus one per
#: imaginary part for complex datatypes (MT, HTF, VTF, NMT, NMT2); the
#: imaginary part of real-valued datatypes (APP_RES_AND_PHS, PT,
#: NMT2_APP_RES_AND_PHS) is dropped, as are columns with error <= 0. The
#: data composition (datatype, site, frequency, component) must match the
#: first member's; members with missing calculated values (NaN) are skipped.
COMPUTE_SIMRC = True

#: Divide every data column by its error (from the first usable member),
#: making the regression coefficients dimensionless so that |coef| sums
#: over mixed datatypes (impedance, tipper, rho/phase) are meaningful.
#: The correlation measure (simrc_corr) is unaffected by this scaling.
SIMRC_ERR_WEIGHT = True

#: Number of data columns processed per chunk when accumulating the
#: (n_data, n_free) cross-covariance between the data and model
#: ensembles -- bounds peak memory to O(SIMRC_CHUNK * n_free) instead of
#: forming the full cross-covariance matrix at once.
SIMRC_CHUNK = 200

if ENS_MODEL_SOURCE == "perturbed" and COMPUTE_SIMRC:
    print(
        "ENS_MODEL_SOURCE = 'perturbed': COMPUTE_SIMRC switched off (it would "
        "correlate the perturbed models with the converged responses)."
    )
    COMPUTE_SIMRC = False

# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
#: Set False to skip covariance estimation entirely.  Mean/var/median/MAD/
#: percentiles are unaffected — only the *_cov* /
#: *_cov_eigval* / *_cov_eigvec* keys are omitted from the .npz output.
#: Skipping is worthwhile whenever the covariance itself isn't needed
#: downstream (e.g. plain ensemble statistics runs), since it is by far
#: the most expensive step for large meshes.
COMPUTE_COV = True

#: "full"     — dense empirical covariance (n_free x n_free) via
#:              sklearn.covariance.empirical_covariance.  Cost
#:              O(n_free^2 * n_members) time, O(n_free^2) memory — fine
#:              for a few thousand free parameters, prohibitive beyond
#:              that (e.g. n_free=1e5 -> 80 GB just to store it).
#: "low_rank" — thin SVD of the centred ensemble matrix instead.  Since
#:              the empirical covariance of n_members samples has rank
#:              <= n_members-1, this is an *exact* factorisation (not an
#:              approximation) whenever n_members << n_free, which is the
#:              usual case here.  Cost drops to
#:              O(n_members^2 * n_free) time and O(n_members * n_free)
#:              memory — orders of magnitude cheaper.  Stores
#:              f"{P}_cov_eigval" (r,) and f"{P}_cov_eigvec" (n_free, r)
#:              with r = n_members instead of a dense f"{P}_cov"; the
#:              full covariance can be reconstructed exactly as
#:              eigvec @ diag(eigval) @ eigvec.T.  Same spirit as the
#:              randomized-SVD-on-R approach used for low-rank prior
#:              sampling in ensembles.py.
COV_METHOD = "low_rank"

#: Sparsify the dense covariance (COV_METHOD="full" only). Ignored for
#: "low_rank", which is already a compact factorisation.
SPARSIFY = True
SPARSE_THRESH = 1.0e-8  # relative threshold for zeroing small entries

# ---------------------------------------------------------------------------
# Bootstrap variance estimation (optional, alternative to the plug-in VAR)
# ---------------------------------------------------------------------------
#: Direct/plug-in variance (np.var(ens_matrix, axis=0, ddof=1)) uses each
#: member exactly once. When N_members is small (order 30-100, typical for
#: RTO/GST ensembles), that single estimate can be noisy. Setting
#: BOOTSTRAP_VAR=True adds an alternative: resample the N_members members
#: with replacement BOOTSTRAP_N times, compute the plug-in variance of each
#: resample, and report the mean across resamples (a smoothed, generally
#: more stable estimate) together with its own bootstrap standard error --
#: i.e. how noisy the variance estimate itself is. This does not replace
#: VAR; both are computed and saved side by side.
BOOTSTRAP_VAR = True
BOOTSTRAP_N = 500  # number of bootstrap resamples
BOOTSTRAP_SEED = None  # None = fresh OS entropy; int = reproducible

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#: Directory for ALL outputs: the results .npz and the nRMS table. The plotting
#: scripts (femtic_ens_plot_*.py) read the .npz from here and write their
#: figures here by default. Created if missing. Absolute path, or build it from
#: ENSEMBLE_DIR (default: <ENSEMBLE_DIR>/results/).

RESULTS_DIR = os.path.join(ENSEMBLE_DIR, "results_test") + os.sep

ENSEMBLE_RESULTS = RESULTS_DIR + (ENSEMBLE_PREFIX  #.upper() 
                                  + "_results" 
                                  + ".npz")

#: Per-member best nRMS table (CSV): member, best iteration, nRMS, status
#: ("accepted", "accepted_fallback" (see ITER_FALLBACK), "accepted_not_drawn"
#: (accepted but not in the random draw, see ENS_SUBSAMPLE_N), "rejected_nrms_max",
#: "no_model_file", and with ENS_MODEL_SOURCE = "perturbed" or "both" also
#: "no_perturbed_file"). Always written; the same table is stored in the
#: results .npz (<P>_nrms_table) for femtic_data_misfit.py (method "nrms").
NRMS_CSV = RESULTS_DIR + ENSEMBLE_PREFIX + "_nrms.csv"

# ---------------------------------------------------------------------------
# Block-file reading
# ---------------------------------------------------------------------------
#: Ocean flag handed to the resistivity-block reader (must match the inversion
#: setup); None = automatic.
MOD_OCEAN = None

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True


# ===========================================================================
# Helpers
# ===========================================================================


def _bootstrap_variance(
    ens_matrix: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    out: bool = True,
):
    """Bootstrap estimate of per-free-parameter variance across members.

    Resamples the N_members ensemble members with replacement ``n_boot``
    times; each replicate's variance is computed the same way as the
    plug-in estimator (``np.var(replicate, axis=0)``, ddof=0, matching
    ``ens_var`` elsewhere in this script). The bootstrap variance estimate
    is the mean of these ``n_boot`` replicate variances -- generally a
    smoother, more stable estimate than the single plug-in value when
    N_members is small (order 30-100, typical for RTO/GST ensembles).

    Memory use is O(n_free), independent of n_boot: running sums are
    accumulated one bootstrap replicate at a time rather than storing the
    full (n_boot, n_free) array of replicate variances (which would be
    prohibitively large for big meshes, e.g. n_boot=500 x n_free=1e5 x
    8 bytes = 400 MB, worse for larger meshes).

    Parameters
    ----------
    ens_matrix : ndarray, shape (n_members, n_free)
        Stacked ensemble in log10(rho).
    n_boot : int
        Number of bootstrap resamples.
    rng : numpy.random.Generator
        Random generator driving the resampling (pass a seeded one via
        BOOTSTRAP_SEED for reproducibility).
    out : bool
        Print progress every ~10% of resamples.

    Returns
    -------
    var_boot : ndarray, shape (n_free,)
        Mean plug-in variance across the n_boot bootstrap replicates.
    var_boot_se : ndarray, shape (n_free,)
        Bootstrap standard error of var_boot itself (spread of the
        replicate variances around their mean) -- a diagnostic of how
        noisy the variance estimate is, not a spread statistic of the
        model directly.
    """
    n_members, n_free = ens_matrix.shape
    sum_v = np.zeros(n_free)
    sum_v2 = np.zeros(n_free)
    _report_every = max(n_boot // 10, 1)
    for _b in range(n_boot):
        _idx = rng.integers(0, n_members, size=n_members)
        _v = np.var(ens_matrix[_idx], axis=0)  # ddof=0, matches ens_var
        sum_v += _v
        sum_v2 += _v * _v
        if out and ((_b + 1) % _report_every == 0 or _b == n_boot - 1):
            print(f"  bootstrap {_b + 1}/{n_boot}")
    var_boot = sum_v / n_boot
    var_boot_se = np.sqrt(np.maximum(sum_v2 / n_boot - var_boot**2, 0.0))
    return var_boot, var_boot_se


def _log10_pos(x):
    """log10 of a strictly positive array; values <= 0 or non-finite -> NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0.0)
    out[ok] = np.log10(x[ok])
    return out


def _sens_block_order(sens_h5, n):
    """Region index of each sensitivity entry, from /model/blocks["blockID"].

    Returns None when blockID is simply 0..n-1 (or cannot be read), else
    the blockID array to scatter by. Read once (same mesh for all members).
    """
    try:
        import h5py

        with h5py.File(sens_h5, "r") as h5:
            ids = np.asarray(h5["model/blocks"]["blockID"][()], dtype=int)
    except Exception as e:  # schema/h5py issues: warn only
        print(
            f"    COMPUTE_SENS: could not read model/blocks/blockID "
            f"({type(e).__name__}: {e}) — assuming block order == region order."
        )
        return None
    if ids.shape[0] != n:
        print(
            f"    COMPUTE_SENS: blockID length {ids.shape[0]} != {n} — "
            f"assuming block order == region order."
        )
        return None
    if np.array_equal(ids, np.arange(n)):
        print(
            f"    COMPUTE_SENS: blockID == 0..{n-1} — block order == region order (verified)."
        )
        return None
    if np.array_equal(np.sort(ids), np.arange(n)):
        print(
            f"    COMPUTE_SENS: blockID is a permutation of 0..{n-1} — reordering."
        )
        return ids
    print(
        f"    COMPUTE_SENS: blockID not a permutation of 0..{n-1} "
        f"(min {ids.min()}, max {ids.max()}) — assuming block order == region order."
    )
    return None


_SENS_MAP = {}  # cached once: nreg, free_idx, chosen mapping (see below)


def _sens_to_free(sens_vec, block_file, sens_h5=None):
    """Reduce a FEMTIC sensitivity vector to the free-parameter vector.

    FEMTIC writes /model/sensitivity/* with one entry per resistivity
    block (region), including fixed ones (air, ocean, flagged blocks),
    while read_model() returns free regions only. Region layout and flags
    are identical across the ensemble, so they are read once (from the
    first member's block file) and the mapping is chosen by length:
        len == n_free   -> used as is
        len == nreg     -> sens[free_idx]        (all regions, incl. air)
        len == nreg - 1 -> sens[free_idx - 1]    (all regions except air)
    If ``sens_h5`` is given, /model/blocks["blockID"] is checked once; a
    permuted order is undone before mapping (identity: nothing to do).
    Returns None if no mapping fits.
    """
    if not _SENS_MAP:
        st = fem._read_resistivity_block_struct(
            block_file, ocean=MOD_OCEAN, out=False
        )
        _SENS_MAP["nreg"] = int(st["nreg"])
        _SENS_MAP["free_idx"] = np.asarray(st["free_idx"], dtype=int)
    nreg, free_idx = _SENS_MAP["nreg"], _SENS_MAP["free_idx"]
    n = sens_vec.shape[0]
    if sens_h5 is not None and "order" not in _SENS_MAP:
        _SENS_MAP["order"] = _sens_block_order(sens_h5, n)
    _order = _SENS_MAP.get("order")
    if _order is not None and _order.shape[0] == n:
        _tmp = np.empty_like(sens_vec)
        _tmp[_order] = sens_vec  # entry k belongs to region blockID[k]
        sens_vec = _tmp
    if n == free_idx.size:
        how, out_vec = "free only", sens_vec
    elif n == nreg:
        how, out_vec = "all regions -> free_idx", sens_vec[free_idx]
    elif n == nreg - 1:
        how, out_vec = "all but air -> free_idx-1", sens_vec[free_idx - 1]
    else:
        return None
    if _SENS_MAP.get("how") != how:
        _SENS_MAP["how"] = how
        print(
            f"    COMPUTE_SENS: sensitivity length {n}, nreg={nreg}, "
            f"n_free={free_idx.size} -> mapping '{how}'."
        )
    return out_vec


#: FEMTIC datatype codes whose values are real (cal_im == 0 by design).
_REAL_DATATYPES = {1, 4, 7}  # APP_RES_AND_PHS, PT, NMT2_APP_RES_AND_PHS


def _simrc_columns(rows):
    """Calculated-data vector, composition key and errors from /data/data.

    Returns (d_cal, manifest, d_err): real parts for all rows, then
    imaginary parts for complex datatypes; columns with error <= 0 or
    non-finite error are dropped. ``manifest`` is a bytes key over
    (datatype, site_id, freq, component) plus the column selection, used
    to check that all members share the same data layout.
    """
    if rows.size == 0:
        raise ValueError("no data rows in /data/data")
    if "cal_re" not in rows.dtype.names:
        raise KeyError("no cal_re/cal_im in /data/data (pre-2026-09-13 file)")
    dt = np.asarray(rows["datatype"], dtype=int)
    re_err = np.asarray(rows["re_err"], dtype=float)
    im_err = np.asarray(rows["im_err"], dtype=float)
    keep_re = np.isfinite(re_err) & (re_err > 0.0)
    keep_im = (
        ~np.isin(dt, list(_REAL_DATATYPES))
        & np.isfinite(im_err)
        & (im_err > 0.0)
    )
    d_cal = np.concatenate(
        [
            np.asarray(rows["cal_re"], dtype=float)[keep_re],
            np.asarray(rows["cal_im"], dtype=float)[keep_im],
        ]
    )
    d_err = np.concatenate([re_err[keep_re], im_err[keep_im]])
    key = np.rec.fromarrays(
        [
            dt,
            np.asarray(rows["site_id"], dtype=int),
            np.round(np.asarray(rows["freq"], dtype=float), 10),
            np.asarray(rows["component"], dtype=int),
        ]
    )
    manifest = key.tobytes() + keep_re.tobytes() + keep_im.tobytes()
    return d_cal, manifest, d_err



# ===========================================================================
# Main
# ===========================================================================

#: ENS_MODEL_SOURCE = "both": the members' perturbed models, one row per row of
#: ens_matrix (None otherwise).
pert_matrix = None

dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME + "*"],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
)
#: EVERY glob match as (name, is_directory), for the match totals and the
#: numbering check of the directory scan report (the filter below drops the
#: non-directories).
scan_matches = [
    (os.path.basename(os.path.normpath(d)), os.path.isdir(d)) for d in dir_list
]
dir_list = [d for d in dir_list if os.path.isdir(d)]
print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

model_list = []  # list of [block_file, n_iter, nRMS]
nrms_table = []  # (member, best_iter, nRMS, status) for NRMS_CSV / histogram
#: For the "Directory scan report" appended to the parameter summary at the
#: end of the run: why each non-accepted directory is not in the ensemble
#: (dict(label, reason, detail)), and which accepted members needed
#: ITER_FALLBACK. Unlike nrms_table (-> NRMS_CSV, histogram), this also
#: covers directories without a usable convergence file.
scan_not_accepted = []
scan_fallback = []
pert_links = []  # ENS_MODEL_SOURCE="perturbed": members whose file is a symlink
model_count = 0
ens_matrix = None  # will become (n_members, n_free) float64

sens_matrix = None  # will become (n_sens_members, n_free) float64
sens_count = 0  # accepted members whose results_iterX.h5 was found
#: Row indices into ens_matrix of the members that contributed a row to
#: sens_matrix (needed by the random subsample, like simrc_keep_idx).
sens_keep_idx = []
sens_missing_any = False
sens_fail = []  # (member dir, reason) for skipped members

ens_data_matrix = None  # will become (n_simrc_members, n_data) float64
simrc_count = 0  # accepted members whose result files were found
simrc_missing_any = False
simrc_fail = []  # (member dir, reason) for skipped members
simrc_manifest = None  # bytes key of the first member's data composition
simrc_err = None  # first member's per-column errors (SIMRC_ERR_WEIGHT)
#: Row indices into ens_matrix (i.e. which accepted members, in append
#: order) that also contributed a row to ens_data_matrix -- needed so the
#: model/data cross-covariance below pairs up the *same* members, since
#: COMPUTE_SIMRC can drop members that COMPUTE_SENS did not (and vice
#: versa).
simrc_keep_idx = []

for d in dir_list:
    print(f"\n  Inversion run: {d}")
    _lbl = os.path.basename(os.path.normpath(d))  # scan report label
    _fb_rec = None  # set if accepted via ITER_FALLBACK (scan report)
    cnv_file = str(fem.resolve_cnv_path(d, CNV_FILE))
    if not os.path.isfile(cnv_file):
        print(f"    {os.path.basename(cnv_file)} not found — skipped.")
        _names = [CNV_FILE] if isinstance(CNV_FILE, str) else list(CNV_FILE)
        scan_not_accepted.append(dict(
            label=_lbl, reason="no_cnv_file",
            detail="none of " + ", ".join(map(str, _names)) + " found"))
        continue

    # Column positions are read from this file's own header row via
    # fem.read_cnv() -- robust to FEMTIC version and to whether distortion
    # parameters (the Beta/Distortion columns) are present, which shift
    # RMS's column position independently of version (e.g. a 4.3 run WITH
    # distortion has RMS at the same column as a typical 5.x run, not at
    # the plain-4.3 position). A version-string switch conflated the two
    # and silently misread nRMS for 4.3-with-distortion runs. Fixed
    # 2026-09-02 -- see femtic_readme.md's read_cnv() entry.
    try:
        _cnv = fem.read_cnv(cnv_file)
    except ValueError as e:
        print(f"    {e} — skipped.")
        scan_not_accepted.append(dict(
            label=_lbl, reason="cnv_unreadable",
            detail=f"{os.path.basename(cnv_file)}: no header row naming an "
                   f"Iter and an RMS column"))
        continue

    # Scan *every* convergence-history row — not just the last — to find
    # the iteration with the smallest nRMS. The last iteration is not
    # necessarily the best: FEMTIC does not guarantee monotonic nRMS
    # reduction, and a run may drift upward again after its actual best
    # iteration.
    numit = None
    nrms = None
    iter0_nrms = None  # tracked separately only to power the warning below
    _cands = []  # (iteration, nRMS) of every eligible (iter > 0) row
    for _row in _cnv["rows"]:
        _it, _nrms = int(_row["Iter"]), _row["RMS"]
        if _it == 0:
            # iter0 (prior/starting model) is not eligible as "best
            # iteration" — it precedes any inversion step.
            iter0_nrms = _nrms
            continue
        _cands.append((_it, _nrms))
        if nrms is None or _nrms < nrms:
            numit, nrms = _it, _nrms
            # ties: keep the first (lowest-iteration) occurrence of the
            # minimum, so a strict "<" (not "<=") comparison is correct
            # here — later equal values are simply not adopted.

    if numit is None:
        print(
            f"    no eligible (iter>0) convergence rows found in "
            f"{cnv_file} — skipped."
        )
        scan_not_accepted.append(dict(
            label=_lbl, reason="cnv_no_iterations",
            detail=f"{os.path.basename(cnv_file)}: "
                   + ("no iteration > 0" if _cnv["rows"] else "no data rows")))
        continue

    if iter0_nrms is not None and iter0_nrms <= nrms:
        print(
            f"    WARNING: iter0 nRMS={iter0_nrms:.4f} is <= the best "
            f"eligible iter>0 nRMS={nrms:.4f} (iter={numit}) — the "
            f"inversion did not improve on the starting model, but "
            f"iter0 is excluded from selection by design."
        )

    if nrms > NRMS_MAX:
        print(
            f"    best nRMS={nrms:.4f} (iter={numit}) > "
            f"NRMS_MAX={NRMS_MAX} — skipped."
        )
        nrms_table.append(
            (os.path.basename(os.path.normpath(d)), numit, nrms, "rejected_nrms_max")
        )
        scan_not_accepted.append(dict(
            label=_lbl, reason="rejected_nrms",
            detail=f"best nRMS {nrms:.4f} (iter {numit}) > NRMS_MAX {NRMS_MAX}"))
        continue

    mod_file = os.path.join(d, f"resistivity_block_iter{numit}.dat")
    _status = "accepted"
    #: why no fallback model was found (scan report); only used if the
    #: member ends up as "no_model_file"
    _fb_note = ("ITER_FALLBACK is False" if not ITER_FALLBACK
                else "no other iteration > 0 has a block file")
    if not os.path.isfile(mod_file) and ITER_FALLBACK:
        # best iteration has no block file: take the best iteration that
        # has one (usually N-1); NRMS_MAX applies to ITS nRMS.
        _avail = [
            (_n2, _i2)
            for _i2, _n2 in _cands
            if os.path.isfile(
                os.path.join(d, f"resistivity_block_iter{_i2}.dat")
            )
        ]
        if _avail:
            _n2, _i2 = min(_avail)  # lowest nRMS, ties -> lowest iteration
            if _n2 <= NRMS_MAX:
                print(
                    f"    resistivity_block_iter{numit}.dat not found - "
                    f"using iter{_i2} (nRMS={_n2:.4f}, best available; "
                    f"iter{numit} had {nrms:.4f}) instead."
                )
                _fb_rec = dict(label=_lbl, best_iter=numit, best_nrms=nrms,
                               used_iter=_i2, used_nrms=_n2)
                numit, nrms, _status = _i2, _n2, "accepted_fallback"
                mod_file = os.path.join(
                    d, f"resistivity_block_iter{numit}.dat"
                )
            else:
                print(
                    f"    best available iteration iter{_i2} has "
                    f"nRMS={_n2:.4f} > NRMS_MAX={NRMS_MAX}."
                )
                _fb_note = (f"best available iteration iter{_i2} has "
                            f"nRMS={_n2:.4f} > NRMS_MAX={NRMS_MAX}")
    if not os.path.isfile(mod_file):
        print(f"    {mod_file} not found — skipped.")
        nrms_table.append(
            (os.path.basename(os.path.normpath(d)), numit, nrms, "no_model_file")
        )
        scan_not_accepted.append(dict(
            label=_lbl, reason="no_model_file",
            detail=f"{os.path.basename(mod_file)} not found "
                   f"(best nRMS {nrms:.4f}); {_fb_note}"))
        continue

    # Model that enters the ensemble: the converged one (default) or the
    # member's perturbed reference (ENS_MODEL_SOURCE = "perturbed").
    # mod_file (the converged block file) keeps selecting the member and
    # serves the sensitivity mapping below in both cases.
    ens_file = mod_file
    pert_file = None
    if ENS_MODEL_SOURCE in ("perturbed", "both"):
        pert_file = os.path.join(d, ENS_PERTURBED_FILE)
        if not os.path.isfile(pert_file):
            print(f"    {pert_file} not found — skipped.")
            nrms_table.append(
                (os.path.basename(os.path.normpath(d)), numit, nrms,
                 "no_perturbed_file")
            )
            scan_not_accepted.append(dict(
                label=_lbl, reason="no_perturbed_file",
                detail=f"{ENS_PERTURBED_FILE} not found (member otherwise "
                       f"accepted: iter {numit}, nRMS {nrms:.4f})"))
            continue
        if os.path.islink(pert_file):
            pert_links.append(d)
        if ENS_MODEL_SOURCE == "perturbed":
            ens_file = pert_file

    nrms_table.append(
        (os.path.basename(os.path.normpath(d)), numit, nrms, _status)
    )
    print(f"    iter={numit}  nRMS={nrms:.4f}  {ens_file}")
    model_list.append([ens_file, numit, nrms])
    if _fb_rec is not None:
        scan_fallback.append(_fb_rec)

    log_m = fem.read_model(model_file=ens_file, model_trans="log10", out=OUT)

    if ens_matrix is None:
        ens_matrix = log_m[np.newaxis, :]  # (1, n_free)
    else:
        ens_matrix = np.vstack((ens_matrix, log_m))  # (k, n_free)

    if ENS_MODEL_SOURCE == "both":
        # the perturbed model of the same member, row for row
        log_p = fem.read_model(
            model_file=pert_file, model_trans="log10", out=OUT
        )
        if log_p.shape != log_m.shape:
            sys.exit(
                f"ENS_MODEL_SOURCE = 'both': {pert_file} has "
                f"{log_p.shape[0]} free parameters but {mod_file} has "
                f"{log_m.shape[0]} - not the same model structure."
            )
        if pert_matrix is None:
            pert_matrix = log_p[np.newaxis, :]
        else:
            pert_matrix = np.vstack((pert_matrix, log_p))

    model_count += 1

    # --- (1) Per-member cumulative sensitivity (optional) -------------
    if COMPUTE_SENS:
        sens_h5 = os.path.join(d, SENS_H5_PATTERN.format(numit=numit))
        try:
            sens_vec = fem.read_h5_sensitivity(
                sens_h5,
                group=SENS_H5_GROUP,
                cumsens_key=SENS_KIND,
                jacobian_key=SENS_JACOBIAN_KEY,
                error_key=SENS_ERROR_KEY,
                normalize=False,
                out=OUT,
            )
        except (FileNotFoundError, KeyError) as e:
            # No fallback to jacobian.h5 by design (see SENS_JACOBIAN_KEY
            # comment above) -- a missing SENS_KIND dataset at this
            # member's numit just means sensitivity wasn't computed for
            # that iteration; skip this member for sensitivity and move on.
            print(f"    COMPUTE_SENS: {e} — member skipped for sensitivity.")
            sens_missing_any = True
            sens_fail.append((d, f"{type(e).__name__}: {e}"))
        else:
            _n_raw = sens_vec.shape[0]
            sens_vec = _sens_to_free(sens_vec, mod_file, sens_h5)
            if sens_vec is None or sens_vec.shape[0] != log_m.shape[0]:
                print(
                    f"    COMPUTE_SENS: {sens_h5} gives {_n_raw} "
                    f"values; no mapping to {log_m.shape[0]} free "
                    f"parameters (nreg={_SENS_MAP.get('nreg')}) — member "
                    f"skipped for sensitivity."
                )
                sens_missing_any = True
                sens_fail.append(
                    (
                        d,
                        f"size {_n_raw} vs n_free "
                        f"{log_m.shape[0]}, nreg "
                        f"{_SENS_MAP.get('nreg')}",
                    )
                )
            else:
                if sens_matrix is None:
                    sens_matrix = sens_vec[np.newaxis, :]
                else:
                    sens_matrix = np.vstack((sens_matrix, sens_vec))
                sens_count += 1
                sens_keep_idx.append(
                    model_count - 1
                )  # row just appended to ens_matrix

    # --- (2) Per-member forward response for SimRC (optional) ---------
    if COMPUTE_SIMRC:
        _res_h5 = os.path.join(d, SENS_H5_PATTERN.format(numit=numit))
        try:
            _rd = fem.read_results_data(_res_h5, out=OUT)
            d_cal, manifest, d_err = _simrc_columns(_rd["rows"])
        except (FileNotFoundError, KeyError, ValueError, OSError) as e:
            print(f"    COMPUTE_SIMRC: {e} — member skipped for SimRC.")
            simrc_missing_any = True
            simrc_fail.append((d, f"{type(e).__name__}: {e}"))
        else:
            if simrc_manifest is None:
                simrc_manifest = manifest
                simrc_err = d_err
            if manifest != simrc_manifest:
                print(
                    f"    COMPUTE_SIMRC: {_res_h5} data composition differs "
                    f"from the first member's — member skipped for SimRC."
                )
                simrc_missing_any = True
                simrc_fail.append(
                    (d, "data composition differs from first member")
                )
            elif not np.all(np.isfinite(d_cal)):
                _nbad = int(np.sum(~np.isfinite(d_cal)))
                print(
                    f"    COMPUTE_SIMRC: {_nbad} missing calculated value(s) "
                    f"in {_res_h5} — member skipped for SimRC."
                )
                simrc_missing_any = True
                simrc_fail.append((d, f"{_nbad} missing calculated values"))
            else:
                if SIMRC_ERR_WEIGHT:
                    d_cal = d_cal / simrc_err
                if ens_data_matrix is None:
                    ens_data_matrix = d_cal[np.newaxis, :]
                else:
                    ens_data_matrix = np.vstack((ens_data_matrix, d_cal))
                simrc_keep_idx.append(
                    model_count - 1
                )  # row just appended to ens_matrix
                simrc_count += 1

print(f"\nConverged members: {model_count}")

# --- (1b) Random subsample of the accepted members (optional) -------------
#: Number of accepted members before the draw (scan report, .npz).
n_accepted_all = model_count
subsample_idx = None  # rows of the accepted members that were drawn (sorted)
if ENS_SUBSAMPLE_N is not None and model_count > 0:
    if ENS_SUBSAMPLE_N >= model_count:
        print()
        print(
            f"ENS_SUBSAMPLE_N = {ENS_SUBSAMPLE_N} >= {model_count} accepted "
            f"members: nothing to draw, all members are used."
        )
    else:
        _sub_rng = np.random.default_rng(ENS_SUBSAMPLE_SEED)
        subsample_idx = np.sort(
            _sub_rng.choice(model_count, size=ENS_SUBSAMPLE_N, replace=False)
        )
        _pos = {int(r): k for k, r in enumerate(subsample_idx)}  # old -> new row
        model_list = [model_list[i] for i in subsample_idx]
        ens_matrix = ens_matrix[subsample_idx]
        if pert_matrix is not None:
            pert_matrix = pert_matrix[subsample_idx]
        # sensitivity / SimRC rows belong to accepted members; keep those drawn
        if sens_matrix is not None:
            _k = [j for j, r in enumerate(sens_keep_idx) if r in _pos]
            sens_matrix = sens_matrix[_k] if _k else None
            sens_keep_idx = [_pos[sens_keep_idx[j]] for j in _k]
            sens_count = len(_k)
        if ens_data_matrix is not None:
            _k = [j for j, r in enumerate(simrc_keep_idx) if r in _pos]
            ens_data_matrix = ens_data_matrix[_k] if _k else None
            simrc_keep_idx = [_pos[simrc_keep_idx[j]] for j in _k]
            simrc_count = len(_k)
        model_count = len(subsample_idx)
        # failure lists and the nRMS table: only the drawn members stay
        # "accepted"; the others are marked, the rejected ones are untouched
        _drawn = {os.path.basename(os.path.dirname(m[0])) for m in model_list}
        _drawn_dirs = {os.path.normpath(os.path.dirname(m[0])) for m in model_list}
        sens_fail = [f for f in sens_fail if os.path.normpath(f[0]) in _drawn_dirs]
        simrc_fail = [f for f in simrc_fail if os.path.normpath(f[0]) in _drawn_dirs]
        sens_missing_any = bool(sens_fail)
        simrc_missing_any = bool(simrc_fail)
        nrms_table = [
            (r[0], r[1], r[2], "accepted_not_drawn")
            if (r[3].startswith("accepted") and r[0] not in _drawn)
            else r
            for r in nrms_table
        ]
        _seed_txt = "fresh entropy" if ENS_SUBSAMPLE_SEED is None else ENS_SUBSAMPLE_SEED
        print()
        print(
            f"ENS_SUBSAMPLE_N = {ENS_SUBSAMPLE_N}: drew {model_count} of "
            f"{n_accepted_all} accepted members at random (seed={_seed_txt})."
        )

n_members = model_count


def _report_skips(tag, flag, count, fails, nmax=5):
    """One-block summary of why per-member sensitivity/SimRC reads failed."""
    if not flag:
        print(f"  {tag}: disabled.")
        return
    print(f"  {tag}: usable for {count}/{n_members} accepted members.")
    for _d, _why in fails[:nmax]:
        print(f"    skipped {os.path.basename(os.path.normpath(_d))}: {_why}")
    if len(fails) > nmax:
        print(f"    ... and {len(fails) - nmax} more.")


_report_skips("COMPUTE_SENS", COMPUTE_SENS, sens_count, sens_fail)
_report_skips("COMPUTE_SIMRC", COMPUTE_SIMRC, simrc_count, simrc_fail)

if n_members == 0:
    sys.exit("No converged members found. Nothing to do.")

if ENS_MODEL_SOURCE in ("perturbed", "both"):
    # Is the "perturbed" ensemble really perturbed?
    if ENS_MODEL_SOURCE == "perturbed":
        print(
            f"\n  ENS_MODEL_SOURCE = 'perturbed': {n_members} member(s) from "
            f"{ENS_PERTURBED_FILE} (the converged models are not used)."
        )
    else:
        print(
            f"\n  ENS_MODEL_SOURCE = 'both': the {n_members} converged "
            f"models are the ensemble; the {ENS_PERTURBED_FILE} of the same "
            f"members is loaded for the borehole figure (MOD_BH) only."
        )
    _pm = ens_matrix if ENS_MODEL_SOURCE == "perturbed" else pert_matrix
    if pert_links:
        print(
            f"  WARNING: {ENS_PERTURBED_FILE} is a symbolic link in "
            f"{len(pert_links)} of {n_members} member(s) (e.g. "
            f"{os.path.basename(os.path.normpath(pert_links[0]))}) - a "
            f"link points to the shared template, i.e. an UNPERTURBED model. "
            f"femtic_rto_prep.py writes the draw to referencemodel.dat; "
            f"resistivity_block_iter0.dat holds it only with "
            f"MOD_START_FROM_PERTURBED = True."
        )
    if n_members > 1 and not np.any(np.ptp(_pm, axis=0) > 0.0):
        print(
            f"  WARNING: all {n_members} members of {ENS_PERTURBED_FILE} are "
            f"identical - the ensemble is not perturbed (zero spread)."
        )

# ens_matrix shape: (n_members, n_free)
# axis=0 → reduce over members  (correct for all aggregate statistics)
# axis=1 → reduce over free parameters (was the bug in the original script)

# --- (2) Summary statistics -----------------------------------------------
P = ENSEMBLE_PREFIX
ne = ens_matrix.shape

ens_avg = np.mean(ens_matrix, axis=0)  # (n_free,)
ens_var = np.var(ens_matrix, axis=0)  # (n_free,)
ens_err = np.sqrt(ens_var)  # (n_free,) -- std, comparable to MAD/QDIFF
ens_med = np.median(ens_matrix, axis=0)  # (n_free,)
ens_mad = np.median(
    np.abs(ens_matrix - ens_med[np.newaxis, :]), axis=0
)  # (n_free,)
ens_prc = np.percentile(ens_matrix, PERCENTILES, axis=0)  # (n_prc, n_free)

# --- Percentile-pair differences (robust spread, e.g. 1-sigma-equivalent IQR) ---
ens_qdiff = {}  # key -> (n_free,) array
for _lo, _hi in QDIFF_PAIRS:
    if _lo not in PERCENTILES or _hi not in PERCENTILES:
        print(
            f"  QDIFF_PAIRS: ({_lo}, {_hi}) not both in PERCENTILES — skipped."
        )
        continue
    _ilo = PERCENTILES.index(_lo)
    _ihi = PERCENTILES.index(_hi)
    _qkey = f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")
    ens_qdiff[_qkey] = np.abs(ens_prc[_ihi] - ens_prc[_ilo])  # (n_free,)

# --- Bootstrap variance estimate (optional alternative to the plug-in VAR) ---
ens_var_boot = None
ens_err_boot = None
ens_var_boot_se = None
if BOOTSTRAP_VAR:
    print(
        f"\nBootstrap variance estimation: {BOOTSTRAP_N} resamples "
        f"(seed={BOOTSTRAP_SEED if BOOTSTRAP_SEED is not None else '(fresh entropy)'}) …"
    )
    _boot_rng = np.random.default_rng(BOOTSTRAP_SEED)
    ens_var_boot, ens_var_boot_se = _bootstrap_variance(
        ens_matrix,
        BOOTSTRAP_N,
        _boot_rng,
        out=OUT,
    )
    ens_err_boot = np.sqrt(ens_var_boot)

print(f"\nStatistics (over {n_members} members, {ne[1]} free parameters):")
print(f"  mean   log10(ρ): [{ens_avg.min():.3f}, {ens_avg.max():.3f}]")
print(f"  var    log10(ρ): [{ens_var.min():.4f}, {ens_var.max():.4f}]")
print(
    f"  err    log10(ρ): [{ens_err.min():.4f}, {ens_err.max():.4f}]  (= sqrt(var))"
)
print(f"  median log10(ρ): [{ens_med.min():.3f}, {ens_med.max():.3f}]")
print(f"  MAD    log10(ρ): [{ens_mad.min():.4f}, {ens_mad.max():.4f}]")
for _qkey, _qval in ens_qdiff.items():
    print(f"  {_qkey}: [{_qval.min():.4f}, {_qval.max():.4f}]")
if BOOTSTRAP_VAR:
    print(
        f"  var_boot  log10(ρ): [{ens_var_boot.min():.4f}, {ens_var_boot.max():.4f}]"
    )
    print(
        f"  err_boot  log10(ρ): [{ens_err_boot.min():.4f}, {ens_err_boot.max():.4f}]  (= sqrt(var_boot))"
    )
    print(
        f"  var_boot_se       : [{ens_var_boot_se.min():.4f}, {ens_var_boot_se.max():.4f}]  "
        f"(bootstrap SE of var_boot itself)"
    )

# --- (2b) Sensitivity statistics: (1) per-member Jacobian aggregation ------
# Two aggregation orders, computed side by side -- see SENS_KIND's
# docstring above for why there is no single "correct" one now that the
# per-member sensitivity in results_iterX.h5 is confirmed NOT to already
# be normalised to that member's own max.
_SENS_STATS = ("mean", "median", "min", "max", "std")
#: key -> log10 array, e.g. sens["mean_na"], sens["cv_an"] (cv linear);
#: empty when COMPUTE_SENS is off or found no usable files.
sens = {}
if COMPUTE_SENS:
    if sens_matrix is None or sens_count == 0:
        print(
            "\n  COMPUTE_SENS: no results_iterX.h5 sensitivity files found — skipped."
        )
    else:
        if sens_missing_any or sens_count != n_members:
            print(
                f"\n  COMPUTE_SENS: sensitivity available for {sens_count}/"
                f"{n_members} accepted members — aggregating over those only."
            )

        def _aggregate(M):
            """mean/median/min/max/std/cv across members (rows), linear."""
            agg = {
                "mean": np.mean(M, axis=0),
                "median": np.median(M, axis=0),
                "min": np.min(M, axis=0),
                "max": np.max(M, axis=0),
                "std": (
                    np.std(M, axis=0, ddof=1)
                    if M.shape[0] > 1
                    else np.zeros(M.shape[1])
                ),
            }
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = agg["std"] / agg["mean"]
            cv[~np.isfinite(cv)] = 0.0
            agg["cv"] = cv
            return agg

        # "_raw": plain ensemble mean in SENS_KIND's native units.
        sens["mean_raw"] = _log10_pos(np.mean(sens_matrix, axis=0))

        # "_na": normalise each member by its own max, THEN aggregate.
        _row_max = np.max(sens_matrix, axis=1, keepdims=True)
        _row_max[_row_max == 0.0] = 1.0  # guard an all-zero member
        _agg = _aggregate(sens_matrix / _row_max)
        for _st in _SENS_STATS:
            sens[f"{_st}_na"] = _log10_pos(_agg[_st])
        sens["cv_na"] = _agg["cv"]

        # "_an": aggregate the RAW ensemble, THEN normalise by max(mean).
        _agg = _aggregate(sens_matrix)
        _renorm = np.nanmax(_agg["mean"])
        _renorm = _renorm if _renorm > 0 else 1.0
        for _st in _SENS_STATS:
            sens[f"{_st}_an"] = _log10_pos(_agg[_st] / _renorm)
        sens["cv_an"] = _agg["cv"]

        print(
            f"\n  Sensitivity (Jacobian, {sens_count} members, SENS_KIND="
            f"'{SENS_KIND}'), log10:"
        )
        for _k in ("mean_raw", "mean_na", "mean_an"):
            print(
                f"    sens_{_k:<9s}: [{np.nanmin(sens[_k]):.3f}, "
                f"{np.nanmax(sens[_k]):.3f}]  "
                f"({int(np.sum(~np.isfinite(sens[_k])))} non-positive -> NaN)"
            )
        print(
            f"    sens_cv (na / an, linear): "
            f"[{sens['cv_na'].min():.3f}, {sens['cv_na'].max():.3f}]  /  "
            f"[{sens['cv_an'].min():.3f}, {sens['cv_an'].max():.3f}]"
        )

# --- (2c) Sensitivity statistics: (2) ensemble-native SimRC ----------------
simrc_coef = simrc_corr = None  # log10 after computation
if COMPUTE_SIMRC:
    if ens_data_matrix is None or simrc_count == 0:
        print(
            "\n  COMPUTE_SIMRC: no member with usable calculated data in "
            "results_iterX.h5 — skipped (see skip summary above)."
        )
    else:
        if simrc_missing_any or simrc_count != n_members:
            print(
                f"\n  COMPUTE_SIMRC: forward response available for "
                f"{simrc_count}/{n_members} accepted members — SimRC "
                f"computed over that matched subset only."
            )
        # Model side must use exactly the members that also contributed a
        # forward-response row -- COMPUTE_SENS can drop
        # different members, so ens_matrix and ens_data_matrix need not
        # otherwise line up row-for-row.
        _Msub = ens_matrix[simrc_keep_idx, :]  # (simrc_count, n_free)
        _Mc = _Msub - np.mean(_Msub, axis=0, keepdims=True)
        _var_m = (
            np.var(_Msub, axis=0, ddof=1)
            if simrc_count > 1
            else np.ones(_Msub.shape[1])
        )
        _std_m = np.sqrt(_var_m)

        _Dc = ens_data_matrix - np.mean(ens_data_matrix, axis=0, keepdims=True)
        _std_d = (
            np.std(ens_data_matrix, axis=0, ddof=1)
            if simrc_count > 1
            else np.ones(ens_data_matrix.shape[1])
        )

        n_free_ = _Mc.shape[1]
        n_data_ = _Dc.shape[1]
        simrc_coef = np.zeros(n_free_)  # cumulative |regression coefficient|
        simrc_corr = np.zeros(n_free_)  # cumulative |correlation|
        _denom = max(simrc_count - 1, 1)

        print(
            f"\n  SimRC (ensemble-native sensitivity, {simrc_count} "
            f"members, {n_data_} data, chunk={SIMRC_CHUNK}) …"
        )
        # Chunked over data columns so the (n_data, n_free) cross-covariance
        # is never formed in full -- peak memory O(SIMRC_CHUNK * n_free).
        for lo in range(0, n_data_, SIMRC_CHUNK):
            hi = min(lo + SIMRC_CHUNK, n_data_)
            _cov_block = (_Dc[:, lo:hi].T @ _Mc) / _denom  # (chunk, n_free_)
            with np.errstate(divide="ignore", invalid="ignore"):
                _coef_block = _cov_block / _var_m[np.newaxis, :]
                _corr_block = _cov_block / (
                    _std_d[lo:hi, np.newaxis] * _std_m[np.newaxis, :]
                )
            _coef_block[~np.isfinite(_coef_block)] = 0.0
            _corr_block[~np.isfinite(_corr_block)] = 0.0
            simrc_coef += np.sum(np.abs(_coef_block), axis=0)
            simrc_corr += np.sum(np.abs(_corr_block), axis=0)

        simrc_coef = _log10_pos(simrc_coef)
        simrc_corr = _log10_pos(simrc_corr)
        print(
            f"    log10 cumulative |regression coeff.|: "
            f"[{np.nanmin(simrc_coef):.3f}, {np.nanmax(simrc_coef):.3f}]"
        )
        print(
            f"    log10 cumulative |correlation|      : "
            f"[{np.nanmin(simrc_corr):.3f}, {np.nanmax(simrc_corr):.3f}]"
        )

# --- (3) Empirical covariance (optional) -----------------------------------
ens_cov = None
ens_covs = None
ens_cov_eigval = None
ens_cov_eigvec = None

if COMPUTE_COV:
    if COV_METHOD == "low_rank":
        print("\nComputing low-rank covariance factorisation (thin SVD) …")
        _Xc = ens_matrix - ens_avg[np.newaxis, :]  # (m, n_free), centred
        _m = _Xc.shape[0]
        # Thin SVD of the (m, n_free) centred ensemble: cost O(m^2 * n_free),
        # memory O(m * n_free) — never forms the n_free x n_free covariance.
        # C = Xc^T Xc / (m-1) = Vt.T @ diag(S^2/(m-1)) @ Vt, exactly (rank <= m-1).
        _U, _S, _Vt = np.linalg.svd(_Xc, full_matrices=False)
        ens_cov_eigval = (_S**2) / max(_m - 1, 1)  # (r,)  r = min(m, n_free)
        ens_cov_eigvec = _Vt.T  # (n_free, r)
        print(
            f"  rank r={ens_cov_eigval.size}  "
            f"eigval range=[{ens_cov_eigval.min():.3e}, {ens_cov_eigval.max():.3e}]"
        )
        print(
            "  Full covariance can be reconstructed exactly as "
            "eigvec @ diag(eigval) @ eigvec.T"
        )
    else:
        print("\nComputing empirical covariance …")
        ens_cov = sklearn.covariance.empirical_covariance(ens_matrix)

        if SPARSIFY:
            tmp = ens_cov.copy()
            tmp[np.abs(tmp) / np.amax(np.abs(tmp)) <= SPARSE_THRESH] = 0.0
            ens_covs = scs.csr_array(tmp)
            nnz = ens_covs.nnz
            total = ens_cov.size
            print(
                f"  Sparse covariance: {nnz}/{total} non-zeros "
                f"({100.0*nnz/total:.2f}%), threshold={SPARSE_THRESH:.1e}"
            )
else:
    print("\nCOMPUTE_COV=False — skipping covariance estimation.")

# --- (4) Save .npz --------------------------------------------------------
ens_dict = {
    f"{P}_model_list": model_list,
    f"{P}_ens": ens_matrix,
    f"{P}_avg": ens_avg,
    f"{P}_var": ens_var,
    f"{P}_err": ens_err,
    f"{P}_med": ens_med,
    f"{P}_mad": ens_mad,
    f"{P}_prc": ens_prc,
    f"{P}_prc_levels": np.asarray(PERCENTILES),
}
for _qkey, _qval in ens_qdiff.items():
    ens_dict[f"{P}_{_qkey}"] = _qval
# best-nRMS member (same row as the "best" panel / QC figure)
_bi = min(range(len(model_list)), key=lambda i: model_list[i][2])
ens_dict[f"{P}_best"] = ens_matrix[_bi]  # log10(rho), free blocks
ens_dict[f"{P}_best_member"] = os.path.basename(os.path.dirname(model_list[_bi][0]))
ens_dict[f"{P}_best_file"] = str(model_list[_bi][0])
ens_dict[f"{P}_best_iter"] = int(model_list[_bi][1])
ens_dict[f"{P}_best_nrms"] = float(model_list[_bi][2])
# which model of each member the rows of <P>_ens are (see ENS_MODEL_SOURCE)
# ("both" = the converged run: <P>_ens is converged, the perturbed models
# are <P>_ens_perturbed, same member order)
ens_dict[f"{P}_model_source"] = str(_ENS_SRC)
ens_dict[f"{P}_model_file_name"] = (
    str(ENS_PERTURBED_FILE) if _ENS_SRC == "perturbed"
    else "resistivity_block_iter<N>.dat"
)
if pert_matrix is not None:
    ens_dict[f"{P}_ens_perturbed"] = pert_matrix  # log10(rho), free blocks
    ens_dict[f"{P}_perturbed_file_name"] = str(ENS_PERTURBED_FILE)
if BOOTSTRAP_VAR:
    ens_dict[f"{P}_var_boot"] = ens_var_boot
    ens_dict[f"{P}_err_boot"] = ens_err_boot
    ens_dict[f"{P}_var_boot_se"] = ens_var_boot_se
for _k, _v in sens.items():  # log10 (cv_* linear)
    ens_dict[f"{P}_sens_{_k}"] = _v
if simrc_coef is not None:  # log10
    ens_dict[f"{P}_simrc_coef"] = simrc_coef
    ens_dict[f"{P}_simrc_corr"] = simrc_corr
if ens_cov is not None:
    ens_dict[f"{P}_cov"] = ens_cov
if ens_cov_eigval is not None:
    ens_dict[f"{P}_cov_eigval"] = ens_cov_eigval
    ens_dict[f"{P}_cov_eigvec"] = ens_cov_eigvec

if subsample_idx is not None:
    # which of the accepted members (scan order) form this ensemble
    ens_dict[f"{P}_subsample_idx"] = np.asarray(subsample_idx, dtype=int)
    ens_dict[f"{P}_subsample_n_accepted"] = int(n_accepted_all)
    ens_dict[f"{P}_subsample_seed"] = (
        -1 if ENS_SUBSAMPLE_SEED is None else int(ENS_SUBSAMPLE_SEED)
    )

# nRMS table of ALL scanned members (accepted and rejected): read by
# femtic_data_misfit.py (method "nrms"), so the histogram needs no CSV. String array of
# shape (n, 4): member, best_iter, nrms, status.
ens_dict[f"{P}_nrms_table"] = np.array(
    [[str(r[0]), str(int(r[1])), f"{float(r[2]):.6f}", str(r[3])]
     for r in nrms_table],
    dtype=str,
).reshape(-1, 4)
ens_dict[f"{P}_nrms_max"] = float(NRMS_MAX)

os.makedirs(RESULTS_DIR, exist_ok=True)
np.savez_compressed(ENSEMBLE_RESULTS, **ens_dict)
print(f"\nResults saved → {ENSEMBLE_RESULTS}")

# --- (4b) nRMS table (CSV) ------------------------------------------------
# The histogram is made by femtic_data_misfit.py (method "nrms") from the .npz.
fem.write_nrms_table(NRMS_CSV, nrms_table)
print(f"nRMS table saved -> {NRMS_CSV}  ({len(nrms_table)} members)")

print("\nfemtic_ens_post.py complete.")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
_summary_path = utl.write_param_summary(__file__, out_dir=RESULTS_DIR)

# Directory scan report: why directories are not in the ensemble. Appended to
# the summary just written. Lowercase names so that nothing here shows up as a
# "user-set parameter".
_scan_md = fem.build_scan_report(
    not_accepted  = scan_not_accepted,
    fallback      = scan_fallback,
    matches       = scan_matches,
    n_scanned     = len(dir_list),
    n_accepted    = n_accepted_all,
    ensemble_dir  = ENSEMBLE_DIR,
    ensemble_name = ENSEMBLE_NAME,
    nrms_max      = NRMS_MAX,
    index_range   = ENSEMBLE_INDEX_RANGE,
)
fem.append_scan_report(_scan_md, _summary_path, __file__)
