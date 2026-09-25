#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_post.py — Ensemble postprocessing for FEMTIC

Collects all converged members of a FEMTIC ensemble (RTO, GST, or any
directory-based ensemble), computes summary statistics, assembles the
empirical covariance, and saves everything to a compressed ``.npz`` file.

Optionally produces slice figures for the **best-nRMS member** (QC) and
for the **ensemble statistics** (mean, variance, median, MAD).

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

    This script targets FEMTIC's current HDF5 output layout only.
    AI-generated code -- review before production use.
"""
from __future__ import annotations

import os
import re
import sys
import inspect
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
import ensembles as ens
import util as utl
from version import versionstrg

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None


def _km_to_m(val):
    """Convert a scalar distance in km to metres; None passes through.

    Used to convert MOD_PROJECTION_DIST/MOD_ROI_PAD_XY (and similar scalar
    spatial config parameters, now specified in km) to the metres expected
    by femtic_viz.py, which works in model-local/UTM metres throughout
    (matching mesh.dat).
    """
    return None if val is None else float(val) * 1000.0


def _lim_km_to_m(lim):
    """Convert a [min, max] km limit pair to metres; None passes through.

    Used to convert MOD_XLIM/MOD_YLIM/MOD_ZLIM/MOD_ROI_ZLIM (now specified
    in km) to the metres expected by femtic_viz.py.
    """
    return None if lim is None else [float(v) * 1000.0 for v in lim]


def _slices_km_to_m(slices):
    """Convert model-local km slice coordinates (x0/y0/z0) to metres.

    ``slices`` is the MOD_SLICES list of dicts, e.g.
    ``dict(kind="map", z0=5.0)`` or ``dict(kind="ns", x0=0.0)``. Plain
    numeric x0/y0/z0 values (model-local km) are converted to metres;
    (value, "latlon") tuples are left untouched since those are
    degrees, resolved separately by fem.resolve_slice_positions().
    None passes through.
    """
    if slices is None:
        return None
    out = []
    for spec in slices:
        spec = dict(spec)
        for key in ("x0", "y0", "z0"):
            if key in spec and isinstance(spec[key], (int, float)):
                spec[key] = spec[key] * 1000.0
        out.append(spec)
    return out


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
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/"
ENSEMBLE_NAME = "annecy_rnd_2_"
#: Prefix used for .npz output keys and default file/figure names.
#: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
ENSEMBLE_PREFIX = "annecy_rnd_2"

#: Maximum normalised RMS accepted from femtic.cnv.
NRMS_MAX = 1.5

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
#: .npz as f"{P}_qdiff_<lo>_<hi>" and available for MOD_STATS plotting
#: under the same key.
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
#                   response ensemble read from each member's
#                   result_XXX.txt files. Needs no Jacobian at all, and
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
assert SENS_KIND in ("raw", "volume_normalised"), \
    f"SENS_KIND must be 'raw' or 'volume_normalised', got {SENS_KIND!r}"

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
#:         [0, ~1] scale (log10 <= 0) for plotting and MOD_ALPHA_SOURCE.
#:
#: Both give a max-normalised statistic (_na's own max may be < 1, since
#: different members peak at different cells; _an's is exactly 1 by
#: construction), i.e. log10 values <= 0 after the log10 conversion. Both
#: are always computed when COMPUTE_SENS succeeds; use whichever (or both)
#: in MOD_STATS_WHAT / MOD_ALPHA_SOURCE.

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
SENS_ERROR_KEY    = "data_errors"   # None to disable error-weighting

#: All sensitivity aggregates (sens_mean/_median/_min/_max/_std, in their
#: _raw/_na/_an versions) and the SimRC sums (simrc_coef/_corr) are strictly
#: positive and span many decades, so they are converted to log10 right
#: after aggregation -- in memory, in the .npz, in the plotted block files,
#: and in MOD_ALPHA_SOURCE expressions (e.g. "sens_mean_an > -3." means
#: "an-normalised mean sensitivity above 1e-3 of its maximum"). Values <= 0
#: (e.g. a cell with exactly zero sensitivity) become NaN. sens_cv_na/_an
#: are ratios and stay linear.

# --- (2) Ensemble-native SimRC sensitivity ----------------------------------
#: Set False to skip entirely (no per-member result_XXX.txt files, or
#: forward-response ensemble not wanted).
COMPUTE_SIMRC = True

#: {data_type: filename} -- filenames are relative to each ensemble-member
#: run directory. Only entries whose file actually exists for a member
#: are used for that member (see fem.collect_forward_response()); a
#: member missing every listed file is dropped from the SimRC ensemble
#: with a warning.
#: Keys are also passed as the "data_type" arg to fem.get_femtic_data(),
#: so use "rhophas"/"imp"/"vtf"/"pt" (or a dict value understood by
#: SIMRC_DATA_KIND below) as appropriate for what FEMTIC wrote.
SIMRC_RESULT_FILES = {
    "imp": "result_MT.txt",
    "vtf":     "result_VTF.txt",
}

#: Site-metadata file shared across the whole ensemble (site geometry
#: does not vary between RTO/GST members, only the model/response does).
SIMRC_SITE_FILE = ENSEMBLE_DIR + "templates/site.dat"

#: Fallback data_type passed to fem.get_femtic_data() for any
#: SIMRC_RESULT_FILES key not already one of "imp"/"vtf"/"pt"/"rhophas".
SIMRC_DATA_KIND = "rhophas"

#: Number of data columns processed per chunk when accumulating the
#: (n_data, n_free) cross-covariance between the data and model
#: ensembles -- bounds peak memory to O(SIMRC_CHUNK * n_free) instead of
#: forming the full cross-covariance matrix at once.
SIMRC_CHUNK = 200

# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
#: Set False to skip covariance estimation entirely.  Mean/var/median/MAD/
#: percentiles and slice plots are unaffected — only the *_cov* /
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
SPARSIFY     = True
SPARSE_THRESH = 1.0e-8   # relative threshold for zeroing small entries

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
BOOTSTRAP_VAR  = True
BOOTSTRAP_N    = 500     # number of bootstrap resamples
BOOTSTRAP_SEED = None    # None = fresh OS entropy; int = reproducible

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
ENSEMBLE_RESULTS = ENSEMBLE_DIR + ENSEMBLE_PREFIX.upper() + "_results.npz"

# ---------------------------------------------------------------------------
# Mesh (required for any slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = ENSEMBLE_DIR + "templates/mesh.dat"

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN     = None
MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Figure output format
# ---------------------------------------------------------------------------
#: One or more file extensions. Every QC / MOD_STATS figure is saved once
#: per entry (slice geometry resolved once and reused; the figure itself
#: is still rebuilt per format, since plot_model_slices doesn't expose a
#: way to re-save an already-built figure), so e.g. ["pdf", "jpg"] writes
#: both a vector version for print and a raster preview in one run. A bare
#: string ("pdf") also works and is treated as a single-entry list.
#: MOD_QC_FILE / MOD_STATS_DIR filenames below are built without an
#: extension; _plot_slice() appends each MOD_PLOT_FORMAT entry in turn.
#: Supported values (Matplotlib Agg-backend savefig formats):
#:   "pdf"  -- vector, multi-page-safe, default; best for print/publication
#:   "svg"  -- vector, editable in Illustrator/Inkscape
#:   "eps"  -- vector, legacy PostScript (no transparency support)
#:   "png"  -- raster, lossless, transparency-capable; good for slides/web
#:   "jpg" / "jpeg" -- raster, lossy, no transparency; small file size
#:   "tif" / "tiff"  -- raster, lossless, common in publication workflows
#:   "webp" -- raster, modern lossy/lossless web format
#: Raster formats (png/jpg/tif/webp) are rendered at MOD_DPI; vector
#: formats (pdf/svg/eps) are resolution-independent and MOD_DPI is ignored
#: except for any embedded raster elements (e.g. rasterized markers).
MOD_PLOT_FORMAT = ["pdf", "jpg"]

#: Normalised to a list regardless of whether a bare string or a list was
#: set above.
_MOD_PLOT_FORMATS = (
    [MOD_PLOT_FORMAT] if isinstance(MOD_PLOT_FORMAT, str) else list(MOD_PLOT_FORMAT)
)

# ---------------------------------------------------------------------------
# QC slice plot — best-nRMS converged member
# ---------------------------------------------------------------------------
#: Set True to plot the best-nRMS member.
MOD_QC      = True
#: Extension-less base path; _plot_slice() appends ".<fmt>" per
#: MOD_PLOT_FORMAT entry.
MOD_QC_FILE = ENSEMBLE_DIR + ENSEMBLE_PREFIX + "_best"

# ---------------------------------------------------------------------------
# Statistics slice plots — mean / variance / median / MAD
# ---------------------------------------------------------------------------
#: Set True to write derived stat members as block files and plot them.
#: Requires MOD_MESH and a valid template file (taken from best member).
MOD_STATS      = True
#: Which statistics to plot.  Subset of: "avg", "var", "err", "med", "mad",
#: one auto-generated key per PERCENTILES level (e.g. 2.3 -> "p2_3",
#: 50.0 -> "p50", 97.7 -> "p97_7"), one per QDIFF_PAIRS entry (e.g.
#: (15.9, 84.1) -> "qdiff_15_9_84_1"), "err_boot" (+ "var_boot") when
#: BOOTSTRAP_VAR=True, and -- when COMPUTE_SENS / COMPUTE_SIMRC succeed --
#: the log10 sensitivity panels "sens_mean_raw", "sens_mean_na",
#: "sens_mean_an", "sens_median_na"/"_an", "sens_min_na"/"_an",
#: "sens_max_na"/"_an", "sens_std_na"/"_an", the linear ratios
#: "sens_cv_na"/"_an", and "simrc_coef"/"simrc_corr" (log10).
#: "err" = sqrt(var) is plotted by default instead of "var", since var is in
#: (log10 Ohm.m)^2 and not on the same scale as MAD/QDIFF (log10 Ohm.m).
#: Colour map, colour limits and colourbar label of every panel are set
#: per key in MOD_STATS_STYLE (shared slice block below).
MOD_STATS_WHAT = ["avg", "med", "err", "mad"] + [
    "p" + f"{_p:g}".replace(".", "_") for _p in PERCENTILES
] + [
    f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_") for _lo, _hi in QDIFF_PAIRS
] + (["err_boot"] if BOOTSTRAP_VAR else []) + (
    ["sens_mean_raw", "sens_mean_na", "sens_cv_na",
     "sens_mean_an", "sens_cv_an"] if COMPUTE_SENS else []
) + (
    ["simrc_corr"] if COMPUTE_SIMRC else []
)
#: Output directory for stat block files and figures.
MOD_STATS_DIR  = ENSEMBLE_DIR + "/stats_plots/"

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_mod_plot_slice.py — used by both MOD_QC and MOD_STATS below)
# ---------------------------------------------------------------------------

# --- Geographic / UTM origin of the mesh centre ----------------------------
#: Set to None when MOD_ORIGIN_METHOD will estimate the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT    = None   # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON    = None   # decimal degrees, positive = East
MOD_UTM_ORIGIN_E      = None   # UTM easting  [m]
MOD_UTM_ORIGIN_N      = None   # UTM northing [m]
MOD_UTM_ZONE_OVERRIDE = None   # override auto-derived zone; None = auto

#: "box"     → midpoint of UTM bounding box of all sites in MOD_SITE_DAT.
#: "average" → arithmetic mean of UTM coordinates in MOD_SITE_DAT.
#: None      → use the hard-coded literals above.
MOD_ORIGIN_METHOD = "box"

# --- Display coordinate system ---------------------------------------------
#: "model"  — axis ticks in model-local metres (default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees
MOD_DISPLAY_COORDS = "latlon"

# --- Site overlay ------------------------------------------------------------
#: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
#: Set to None to fall back to observe.dat / MOD_SITE_NUMBER.
MOD_SITE_DAT    = ENSEMBLE_DIR + "templates/site.dat"
MOD_SITE_NAMES  = None   # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS   = True    # show markers on map panels
MOD_PLOT_SITES_SLICES = False    # show markers on curtain / plane panels
#: Max distance [km] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 1.0    # km; None = show all sites on every panel

MOD_SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None
#: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
MOD_MAP_MARKERS = []
#: Mark the model origin on every "map" panel whenever MOD_DISPLAY_COORDS
#: is "utm"/"latlon" (no effect for "model"). Override style via a
#: MOD_MAP_MARKERS entry with "is_model_centre": True.
MOD_SHOW_MODEL_CENTRE = True

# --- Slice specification ----------------------------------------------------
#: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
#:   (value, "utm") | (value, "latlon")
#: Depth z0 is always model-local metres (no CRS tagging).
# MOD_SLICES = [    
#     dict(kind="map", z0=0.0),    # km
#     dict(kind="map", z0=5.0),    # km
#     dict(kind="map", z0=10.0),   # km
#     dict(kind="map", z0=15.0),   # km
#     dict(kind="map", z0=20.0),   # km
#     dict(kind="map", z0=25.0),   # km
#     dict(kind="ns",  x0=(-71.40723, 'latlon')),    # km
#     dict(kind="ew",  y0=(-16.299593, 'latlon')),    # km
# ]
MOD_SLICES = [    
    dict(kind="map", z0=-0.25),    # km
    dict(kind="map", z0=0.0),    # km
    dict(kind="map", z0=0.5),   # km
    dict(kind="map", z0=1.0),   # km
    dict(kind="map", z0=2.0),   # km
    dict(kind="map", z0=2.5),   # km
    dict(kind="map", z0=3.0),   # km
    dict(kind="map", z0=4.0),   # km
    dict(kind="map", z0=5.0),   # km
    dict(kind="ns",  x0=0.),    # km
    dict(kind="ew",  y0=0.),    # km
    #dict(kind="ns",  x0=(-71.40723, 'latlon')),    # km
    #dict(kind="ew",  y0=(-16.299593, 'latlon')),    # km
]

MOD_XLIM = None    # [xmin, xmax] model-local km; None = auto
MOD_YLIM = None    # [ymin, ymax] model-local km; None = auto
MOD_ZLIM = None    # [zmin, zmax] model-local km; None = auto

# --- Region of interest (auto xlim/ylim/zlim from site positions) ----------
#: When True and site positions are available (MOD_SITE_DAT / MOD_SITE_NUMBER,
#: subject to MOD_PLOT_SITES_MAPS/SLICES as usual), MOD_XLIM/MOD_YLIM are
#: derived automatically from the site bounding box + MOD_ROI_PAD_XY, and
#: MOD_ZLIM is set from MOD_ROI_ZLIM -- overriding the literal MOD_XLIM/
#: MOD_YLIM/MOD_ZLIM values above. Falls back to those literals (or to
#: full-mesh auto-scaling if they're also None) when no sites are found.
#: Also drives the per-panel aspect-ratio sizing below (MOD_PANEL_WIDTH),
#: since that sizing needs an actual extent to compute widths from.
#: The same three MOD_ROI_* variables (same defaults, same semantics) now
#: exist in femtic_rto_prep.py and femtic_gst_prep.py, so the plot extent
#: is identical between prep-time and post-time figures.
MOD_ROI_AUTO   = True
MOD_ROI_PAD_XY = 4.0             # km of padding around the site bbox
MOD_ROI_ZLIM   = [-1.0, 7.0]
#: depth range (km, positive-down) for ns/ew/plane panels; None = leave MOD_ZLIM as-is
#: Lower bound is negative (above the z=0 datum) to give ~1 km of headroom
#: so topography (mesh cells with z < 0) is not clipped out of the ns/ew/
#: plane panels. Previously [0.0, 20000.0] cut panels off exactly at the
#: datum, hiding any topography above it.

MOD_DPI         = 400            # figure DPI, used by both MOD_QC and MOD_STATS
MOD_CMAP        = "jet_r"
MOD_CLIM        = [0.0, 4.0]     # [log10_min, log10_max] Ω·m; None = auto
MOD_OCEAN_COLOR = "lightgrey"    # flat colour for ocean cells; None = colormap
MOD_AIR_COLOR   = "whitesmoke"
MOD_AIR_BGCOLOR = None           # axes facecolor for air; None = figure default

# --- Per-panel style (MOD_STATS) ------------------------------------------
#: Per-statistic overrides for every MOD_STATS panel, keyed as in
#: MOD_STATS_WHAT. Each value is a dict with any of
#:   "cmap"  : Matplotlib colormap name                (default MOD_CMAP)
#:   "clim"  : [vmin, vmax] in the panel's own units,
#:             None = auto-scale from the data         (default see below)
#:   "label" : colourbar label                         (default see below)
#: Keys/fields not given fall back to the defaults: value-scale panels
#: (avg, med, p*) use MOD_CMAP / MOD_CLIM / "log10(rho / Ohm*m)"; every
#: other panel uses MOD_CMAP / auto clim / the statistic's own name.
#: Sensitivity panels are log10 (see the Sensitivity statistics section),
#: so e.g. clim=[-4, 0] shows four decades below the normalised maximum.
MOD_STATS_STYLE = {
    "var": dict(clim=[0.0, 0.3], label="var log10(rho)"),
    "err": dict(clim=[0.0, 0.3], label="std log10(rho)"),
    "mad": dict(clim=[0.0, 0.3], label="MAD log10(rho)"),
}
for _lo, _hi in QDIFF_PAIRS:
    MOD_STATS_STYLE[f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")] = dict(
        clim=[0.0, 0.5], label=f"P{_hi:g} - P{_lo:g}  log10(rho)")
if BOOTSTRAP_VAR:
    MOD_STATS_STYLE["var_boot"] = dict(clim=[0.0, 0.3], label="bootstrap var log10(rho)")
    MOD_STATS_STYLE["err_boot"] = dict(clim=[0.0, 0.3], label="bootstrap std log10(rho)")
if COMPUTE_SENS:
    #: Native SENS_KIND units -- run- and mesh-dependent, auto-scaled.
    MOD_STATS_STYLE["sens_mean_raw"] = dict(
        cmap="viridis", clim=None, label=f"log10 S ({SENS_KIND})")
    for _sfx in ("na", "an"):
        for _st in ("mean", "median", "min", "max", "std"):
            MOD_STATS_STYLE[f"sens_{_st}_{_sfx}"] = dict(
                cmap="viridis", clim=[-4.0, 0.0],
                label=f"log10 S/Smax ({_st}, {_sfx})")
        MOD_STATS_STYLE[f"sens_cv_{_sfx}"] = dict(
            cmap="viridis", clim=None, label=f"CV of S ({_sfx})")
if COMPUTE_SIMRC:
    MOD_STATS_STYLE["simrc_coef"] = dict(cmap="viridis", clim=None,
                                         label="log10 sum|SimRC|")
    MOD_STATS_STYLE["simrc_corr"] = dict(cmap="viridis", clim=None,
                                         label="log10 sum|corr|")

# --- Blanking / fading (optional) ------------------------------------------
#: One condition "<name> <op> <value>" selecting the cells to SHOW; cells
#: failing it are blanked or faded. <name> is any statistic key known to
#: MOD_STATS (see MOD_STATS_WHAT; computed statistics are available here
#: even if not listed for plotting), <op> is one of > >= < <=, and <value>
#: is in that statistic's own units (log10 for sensitivities). Examples:
#:   "sens_mean_an > -3."    keep cells within 3 decades of max sensitivity
#:   "err < 0.2"             keep cells with ensemble std below 0.2 decades
#: None disables blanking/fading.
MOD_ALPHA_SOURCE = "sens_mean_an > -3."
#: "blank" -- failing cells are removed (hard cut).
#: "fade"  -- failing cells fade linearly from opaque at the threshold to
#:            fully transparent MOD_ALPHA_FADE_WIDTH beyond it (same units
#:            as <value>, e.g. 1.0 = one decade for log10 sensitivities).
MOD_ALPHA_MODE       = "blank"
MOD_ALPHA_FADE_WIDTH = 1.0
#: Apply the same mask to the MOD_QC (best-member) figure as well.
MOD_ALPHA_QC         = True

# --- Figure layout -----------------------------------------------------------
MOD_EQUAL_ASPECT = True
MOD_DEPTH_KM     = True
MOD_HORIZ_KM     = True
#: 2x2 grid matching the 4 default MOD_SLICES panels (2 maps + ns + ew).
#: Adjust to len(MOD_SLICES) if you change the number of panels; None/None
#: falls back to a single row of len(MOD_SLICES) columns.
MOD_NROWS        = 4      # None = auto (1 row)
MOD_NCOLS        = 3     # None = auto (len(MOD_SLICES) cols)
MOD_PANEL_HEIGHT = 18.0   # cm
#: None = auto per-column width from each panel's own aspect ratio (needs
#: MOD_EQUAL_ASPECT=True and real xlim/ylim/zlim -- supplied automatically
#: by MOD_ROI_AUTO above -- so map, ns, and ew panels naturally end up
#: different widths instead of being forced square).
MOD_PANEL_WIDTH  = None   # cm; None = auto from aspect ratio
MOD_FIGSIZE      = None   # [w, h] cm; overrides auto when set

#: Axis annotation font sizes, passed through to fviz.plot_model_slices.
#: Defaults match plot_model_slices' own defaults.
MOD_TICK_FONTSIZE  = 16   # axis tick labels, colourbar ticks
MOD_LABEL_FONTSIZE = 16    # axis labels, panel titles, colourbar label

#: Decimal digits shown on axis tick labels (depth, map/curtain
#: easting-northing, and lat/lon all share this one setting). None (default)
#: keeps plot_model_slices' own per-axis-type formatting unchanged.
MOD_TICK_DECIMALS = 2

#: When True (default) and this script is running inside Spyder, every
#: saved figure is also displayed inline (Spyder's Plots pane) via
#: plt.show(), in addition to being written to disk -- no change to what
#: gets saved. Has no effect outside Spyder (plain python / other IDEs
#: still save-only, matching prior behaviour); set False to disable even
#: under Spyder.
MOD_SHOW_IN_SPYDER = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

#: Detected once at import time; utl.runtime_env() returns 'spyder' when
#: running inside Spyder's IPython console (SPYDER_KERNEL env var / spyder_
#: kernels module), 'jupyter'/'ipython-*'/'python' otherwise.
_IN_SPYDER = (utl.runtime_env() == "spyder")
_SHOW_PLOTS = MOD_SHOW_IN_SPYDER and _IN_SPYDER
if _IN_SPYDER:
    print(f"Detected Spyder — inline figure display {'enabled' if _SHOW_PLOTS else 'disabled (MOD_SHOW_IN_SPYDER=False)'}.\n")

# ===========================================================================
# Helpers
# ===========================================================================

def _bootstrap_variance(ens_matrix: np.ndarray, n_boot: int,
                         rng: np.random.Generator, out: bool = True):
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
    sum_v  = np.zeros(n_free)
    sum_v2 = np.zeros(n_free)
    _report_every = max(n_boot // 10, 1)
    for _b in range(n_boot):
        _idx = rng.integers(0, n_members, size=n_members)
        _v = np.var(ens_matrix[_idx], axis=0)     # ddof=0, matches ens_var
        sum_v  += _v
        sum_v2 += _v * _v
        if out and ((_b + 1) % _report_every == 0 or _b == n_boot - 1):
            print(f"  bootstrap {_b + 1}/{n_boot}")
    var_boot    = sum_v / n_boot
    var_boot_se = np.sqrt(np.maximum(sum_v2 / n_boot - var_boot ** 2, 0.0))
    return var_boot, var_boot_se


#: Sentinel air/ocean resistivities used only in MOD_STATS block files: a
#: statistic (e.g. a log10 sensitivity of -0.6, or a raw log10 sensitivity
#: above 8) must never coincide with the values plot_model_slices uses to
#: recognise ocean (log10(ocean_value) +- 0.05) or air (log10 > threshold).
_STATS_AIR_RHO          = 1.0e30
_STATS_AIR_LOG10_THRESH = 29.0
_STATS_OCEAN_RHO        = 1.0e-30

_ALPHA_EXPR_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(>=|<=|>|<)\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def _log10_pos(x):
    """log10 of a strictly positive array; values <= 0 or non-finite -> NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0.0)
    out[ok] = np.log10(x[ok])
    return out


def _parse_alpha_source(expr: str):
    """Parse MOD_ALPHA_SOURCE ("name op value") -> (name, op, value).

    Only a single comparison is allowed; nothing is evaluated.
    """
    m = _ALPHA_EXPR_RE.match(expr)
    if m is None:
        raise ValueError(
            f"MOD_ALPHA_SOURCE={expr!r}: expected '<name> <op> <value>' "
            f"with op one of > >= < <=, e.g. 'sens_mean_an > -3.'")
    return m.group(1), m.group(2), float(m.group(3))


def _alpha_from_condition(arr, op: str, thr: float, mode: str,
                          width: float) -> np.ndarray:
    """Per-free-parameter alpha in [0, 1] from a single comparison.

    Cells satisfying ``arr <op> thr`` get alpha 1. Failing cells get 0
    ("blank") or fade linearly with distance beyond the threshold, reaching
    0 at ``width`` ("fade"). NaN values always get 0.
    """
    arr = np.asarray(arr, dtype=float)
    with np.errstate(invalid="ignore"):
        if op == ">":
            keep, dist = arr > thr, thr - arr
        elif op == ">=":
            keep, dist = arr >= thr, thr - arr
        elif op == "<":
            keep, dist = arr < thr, arr - thr
        else:  # "<="
            keep, dist = arr <= thr, arr - thr
    if mode == "blank":
        alpha = keep.astype(float)
    elif mode == "fade":
        if width <= 0.0:
            raise ValueError("MOD_ALPHA_FADE_WIDTH must be > 0 for 'fade'.")
        alpha = np.where(keep, 1.0, np.clip(1.0 - dist / width, 0.0, 1.0))
    else:
        raise ValueError(f"MOD_ALPHA_MODE must be 'blank' or 'fade', got {mode!r}")
    alpha[~np.isfinite(arr)] = 0.0
    return alpha


def _write_alpha_block(alpha: np.ndarray, template: str, path: str) -> None:
    """Write per-free-parameter alpha so the file's RAW values equal alpha.

    fem.insert_model() writes 10**v, and plot_model_slices reads alpha
    files as raw region values -- so v = log10(alpha) (alpha = 0 -> raw 0).
    Air/ocean are written as 1.0 (they are drawn separately anyway).
    """
    with np.errstate(divide="ignore"):
        v = np.where(alpha > 0.0, np.log10(np.maximum(alpha, 1e-300)), -np.inf)
    fem.insert_model(
        template   = template,
        model      = v,
        model_file = path,
        ocean      = MOD_OCEAN,
        air_rho    = 1.0,
        ocean_rho  = 1.0,
        out        = OUT,
    )


def _resolve_origin_and_sites():
    """Estimate UTM origin from MOD_SITE_DAT; collect site model-local coords.

    Mirrors the origin-resolution block in femtic_gst_prep.py /
    femtic_rto_prep.py so all three scripts behave identically, including
    the observe.dat / MOD_SITE_NUMBER fallback when MOD_SITE_DAT is absent.

    Returns
    -------
    utm_e, utm_n, utm_lat, utm_lon : float | None
    utm_zone, utm_northern : str | None, bool | None
    site_xys : list of (name, x_m, y_m, elev)
    obs_coords_only : bool
        True if site_xys was populated from observe.dat / MOD_SITE_NUMBER
        rather than MOD_SITE_DAT.
    """
    _e   = MOD_UTM_ORIGIN_E
    _n   = MOD_UTM_ORIGIN_N
    _lat = MOD_UTM_ORIGIN_LAT
    _lon = MOD_UTM_ORIGIN_LON
    _zone, _north = None, None

    if MOD_ORIGIN_METHOD is not None and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        _sdat = fem.read_site_dat(MOD_SITE_DAT)
        if _sdat:
            _Es  = np.array([d["easting"]  for d in _sdat])
            _Ns  = np.array([d["northing"] for d in _sdat])
            if MOD_ORIGIN_METHOD == "box":
                _e = 0.5 * (_Es.min() + _Es.max())
                _n = 0.5 * (_Ns.min() + _Ns.max())
            elif MOD_ORIGIN_METHOD == "average":
                _e = float(_Es.mean())
                _n = float(_Ns.mean())
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            _zone, _north = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=MOD_UTM_ZONE_OVERRIDE,
            )
            _lat, _lon = utl.utm_to_latlon_zn(_e, _n, _zone, _north)

    if _lat is not None and _lon is not None:
        _zone, _north = utl.utm_zone_from_latlon(
            _lat, _lon, override=MOD_UTM_ZONE_OVERRIDE
        )

    site_xys = []
    obs_coords_only = False
    _need_sites = MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES
    if _need_sites and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        for row in fem.read_site_dat(MOD_SITE_DAT, site_names=MOD_SITE_NAMES):
            sx, sy = fem.utm_to_model(
                row["easting"], row["northing"], _e, _n
            )
            site_xys.append(
                (row["name"], sx, sy, float(row.get("elev", 0.0)))
            )
    elif _need_sites and MOD_SITE_NUMBER is not None:
        _obs_file = ENSEMBLE_DIR + "templates/observe.dat"
        _site_nums = (MOD_SITE_NUMBER if isinstance(MOD_SITE_NUMBER, (list, tuple))
                      else [MOD_SITE_NUMBER])
        for _sn in _site_nums:
            sx, sy = fem.read_site_position(_obs_file, _sn)
            site_xys.append((_sn, sx, sy, 0.0))
        obs_coords_only = True

    return _e, _n, _lat, _lon, _zone, _north, site_xys, obs_coords_only


def _plot_slice(block_file: str, pdf_file: str,
                utm_e, utm_n, utm_lat, utm_lon,
                utm_zone, utm_north, site_xys: list,
                obs_coords_only: bool = False,
                clim="model", cmap=None, cbar_label=None,
                alpha_file=None, stats_panel: bool = False,
                ) -> None:
    """Call fviz.plot_model_slices once per MOD_PLOT_FORMAT entry.

    Mirrors the plotting call in femtic_gst_prep.py / femtic_rto_prep.py,
    so QC and statistics figures share the MOD_SLICES geometry, CRS
    handling, site overlay and figure layout.

    Parameters
    ----------
    pdf_file : str
        Output path *without* an extension; ".<fmt>" is appended for each
        MOD_PLOT_FORMAT entry.
    clim : "model" | [vmin, vmax] | None
        "model" (default) = MOD_CLIM; None = auto-scale from the data;
        a pair = explicit range (from MOD_STATS_STYLE). Previously None
        silently fell back to MOD_CLIM, so auto-scaling never happened.
    cmap, cbar_label : optional
        Per-panel overrides (from MOD_STATS_STYLE); None falls back to
        MOD_CMAP / plot_model_slices' default label.
    alpha_file : str | None
        Alpha block written by _write_alpha_block() (raw values = alpha in
        [0, 1]); plotted with alpha_mode="direct". None = no blanking.
    stats_panel : bool
        True for MOD_STATS panels. Their block files use sentinel
        resistivities for air/ocean (_STATS_AIR_RHO / _STATS_OCEAN_RHO) so
        that a statistic value can never be mistaken for air (log10 > 8)
        or ocean (log10 ~ log10(MOD_OCEAN_RHO)) by plot_model_slices.
    """
    if fviz is None:
        print("  plot_slice: femtic_viz not available — skipping.")
        return

    _clim = MOD_CLIM if (isinstance(clim, str) and clim == "model") else clim
    _cmap = MOD_CMAP if cmap is None else cmap
    _ocean_value = _STATS_OCEAN_RHO if stats_panel else MOD_OCEAN_RHO
    _air_thresh  = _STATS_AIR_LOG10_THRESH if stats_panel else 8.0

    _slices_resolved = fem.resolve_slice_positions(
        _slices_km_to_m(MOD_SLICES), utm_zone, utm_north,
        utm_e, utm_n, utm_lat, utm_lon,
        verbose=OUT,
    )
    for _fmt_i, _fmt in enumerate(_MOD_PLOT_FORMATS):
        _fmt_file = f"{pdf_file}.{_fmt}"
        # Only pop up the interactive window (if MOD_SHOW_IN_SPYDER) on the
        # first format; re-displaying the same figure once per extra
        # format would just spam the Plots pane for no benefit. Note the
        # slice geometry (_slices_resolved, above) is computed once and
        # reused, but plot_model_slices still builds and renders its own
        # figure fresh on each call -- it doesn't expose a way to re-save
        # an already-built figure under a second extension.
        _show_this = _SHOW_PLOTS if _fmt_i == 0 else False
        fviz.plot_model_slices(
            model_file          = block_file,
            mesh_file           = MOD_MESH,
            slices              = _slices_resolved,
            cmap                = _cmap,
            clim                = _clim,
            cbar_label          = cbar_label,
            xlim                = _lim_km_to_m(MOD_XLIM),
            ylim                = _lim_km_to_m(MOD_YLIM),
            zlim                = _lim_km_to_m(MOD_ZLIM),
            ocean_color         = MOD_OCEAN_COLOR,
            ocean_value         = _ocean_value,
            air_log10_thresh    = _air_thresh,
            air_color           = MOD_AIR_COLOR,
            air_bgcolor         = MOD_AIR_BGCOLOR,
            site_xys            = site_xys,
            obs_coords_only     = obs_coords_only,
            sites_in_maps       = MOD_PLOT_SITES_MAPS,
            sites_in_slices     = MOD_PLOT_SITES_SLICES,
            site_marker         = MOD_SITE_MARKER,
            site_marker_slices  = MOD_SITE_MARKER_SLICES,
            map_markers         = MOD_MAP_MARKERS,
            show_model_centre   = MOD_SHOW_MODEL_CENTRE,
            projection_dist     = _km_to_m(MOD_PROJECTION_DIST),
            display_coords      = MOD_DISPLAY_COORDS,
            utm_origin_e        = utm_e,
            utm_origin_n        = utm_n,
            utm_zone            = utm_zone,
            utm_northern        = utm_north,
            utm_to_latlon_fn    = utl.utm_to_latlon_zn,
            latlon_to_model_fn  = fem.latlon_to_model,
            depth_km            = MOD_DEPTH_KM,
            horiz_km            = MOD_HORIZ_KM,
            equal_aspect        = MOD_EQUAL_ASPECT,
            panel_height        = MOD_PANEL_HEIGHT / 2.54,
            panel_width         = MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None,
            figsize             = [v / 2.54 for v in MOD_FIGSIZE] if MOD_FIGSIZE is not None else None,
            nrows               = MOD_NROWS,
            ncols               = MOD_NCOLS,
            tick_fontsize       = MOD_TICK_FONTSIZE,
            label_fontsize      = MOD_LABEL_FONTSIZE,
            tick_decimals       = MOD_TICK_DECIMALS,
            alpha_file          = alpha_file,
            alpha_mode          = "direct",
            plot_file           = _fmt_file,
            dpi                 = MOD_DPI,
            show                = _show_this,
            out                 = OUT,
        )
        if OUT:
            print(f"  saved → {_fmt_file}")


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Scan ensemble directories ----------------------------------------
# get_filelist() matches on name only (fnmatch over os.listdir), so it can
# return non-directory matches (stray files, archives, etc.) alongside the
# actual ensemble-member sub-directories. Filter to directories immediately,
# before counting/looping, so downstream logic never has to think about
# non-directory entries.
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME+"*"],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
)
dir_list = [d for d in dir_list if os.path.isdir(d)]
print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

model_list  = []          # list of [block_file, n_iter, nRMS]
model_count = 0
ens_matrix  = None        # will become (n_members, n_free) float64

sens_matrix       = None  # will become (n_sens_members, n_free) float64
sens_count        = 0     # accepted members whose results_iterX.h5 was found
sens_missing_any  = False

ens_data_matrix   = None  # will become (n_simrc_members, n_data) float64
simrc_count       = 0     # accepted members whose result files were found
simrc_missing_any = False
simrc_manifest    = None  # (data_type, n_values) list from the first member
#: Row indices into ens_matrix (i.e. which accepted members, in append
#: order) that also contributed a row to ens_data_matrix -- needed so the
#: model/data cross-covariance below pairs up the *same* members, since
#: COMPUTE_SIMRC can drop members that COMPUTE_SENS did not (and vice
#: versa).
simrc_keep_idx    = []

for d in dir_list:
    print(f"\n  Inversion run: {d}")
    cnv_file = os.path.join(d, "femtic.cnv")
    if not os.path.isfile(cnv_file):
        print(f"    femtic.cnv not found — skipped.")
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
        continue

    # Scan *every* convergence-history row — not just the last — to find
    # the iteration with the smallest nRMS. The last iteration is not
    # necessarily the best: FEMTIC does not guarantee monotonic nRMS
    # reduction, and a run may drift upward again after its actual best
    # iteration.
    numit        = None
    nrms         = None
    iter0_nrms   = None   # tracked separately only to power the warning below
    for _row in _cnv["rows"]:
        _it, _nrms = int(_row["Iter"]), _row["RMS"]
        if _it == 0:
            # iter0 (prior/starting model) is not eligible as "best
            # iteration" — it precedes any inversion step.
            iter0_nrms = _nrms
            continue
        if nrms is None or _nrms < nrms:
            numit, nrms = _it, _nrms
            # ties: keep the first (lowest-iteration) occurrence of the
            # minimum, so a strict "<" (not "<=") comparison is correct
            # here — later equal values are simply not adopted.

    if numit is None:
        print(f"    no eligible (iter>0) convergence rows found in "
              f"{cnv_file} — skipped.")
        continue

    if iter0_nrms is not None and iter0_nrms <= nrms:
        print(f"    WARNING: iter0 nRMS={iter0_nrms:.4f} is <= the best "
              f"eligible iter>0 nRMS={nrms:.4f} (iter={numit}) — the "
              f"inversion did not improve on the starting model, but "
              f"iter0 is excluded from selection by design.")

    if nrms > NRMS_MAX:
        print(f"    best nRMS={nrms:.4f} (iter={numit}) > "
              f"NRMS_MAX={NRMS_MAX} — skipped.")
        continue

    mod_file = os.path.join(d, f"resistivity_block_iter{numit}.dat")
    if not os.path.isfile(mod_file):
        print(f"    {mod_file} not found — skipped.")
        continue

    print(f"    iter={numit}  nRMS={nrms:.4f}  {mod_file}")
    model_list.append([mod_file, numit, nrms])

    log_m = fem.read_model(model_file=mod_file, model_trans="log10", out=OUT)

    if ens_matrix is None:
        ens_matrix = log_m[np.newaxis, :]         # (1, n_free)
    else:
        ens_matrix = np.vstack((ens_matrix, log_m))   # (k, n_free)

    model_count += 1

    # --- (1) Per-member cumulative sensitivity (optional) -------------
    if COMPUTE_SENS:
        sens_h5 = os.path.join(d, SENS_H5_PATTERN.format(numit=numit))
        try:
            sens_vec = fem.read_h5_sensitivity(
                sens_h5,
                group        = SENS_H5_GROUP,
                cumsens_key  = SENS_KIND,
                jacobian_key = SENS_JACOBIAN_KEY,
                error_key    = SENS_ERROR_KEY,
                normalize    = False,
                out          = OUT,
            )
        except (FileNotFoundError, KeyError) as e:
            # No fallback to jacobian.h5 by design (see SENS_JACOBIAN_KEY
            # comment above) -- a missing SENS_KIND dataset at this
            # member's numit just means sensitivity wasn't computed for
            # that iteration; skip this member for sensitivity and move on.
            print(f"    COMPUTE_SENS: {e} — member skipped for sensitivity.")
            sens_missing_any = True
        else:
            if sens_vec.shape[0] != log_m.shape[0]:
                print(f"    COMPUTE_SENS: {sens_h5} gives {sens_vec.shape[0]} "
                      f"parameters, expected {log_m.shape[0]} — member "
                      f"skipped for sensitivity.")
                sens_missing_any = True
            else:
                if sens_matrix is None:
                    sens_matrix = sens_vec[np.newaxis, :]
                else:
                    sens_matrix = np.vstack((sens_matrix, sens_vec))
                sens_count += 1

    # --- (2) Per-member forward response for SimRC (optional) ---------
    if COMPUTE_SIMRC:
        _result_paths = {
            k: os.path.join(d, v) for k, v in SIMRC_RESULT_FILES.items()
        }
        try:
            d_cal, manifest = fem.collect_forward_response(
                _result_paths, SIMRC_SITE_FILE,
                data_kind = SIMRC_DATA_KIND, out = OUT,
            )
        except FileNotFoundError as e:
            print(f"    COMPUTE_SIMRC: {e} — member skipped for SimRC.")
            simrc_missing_any = True
        else:
            if simrc_manifest is None:
                simrc_manifest = manifest
            elif manifest != simrc_manifest:
                print(f"    COMPUTE_SIMRC: {d} response composition "
                      f"{manifest} != first member's {simrc_manifest} — "
                      f"member skipped for SimRC.")
                simrc_missing_any = True
                d_cal = None
            if d_cal is not None:
                if ens_data_matrix is None:
                    ens_data_matrix = d_cal[np.newaxis, :]
                else:
                    ens_data_matrix = np.vstack((ens_data_matrix, d_cal))
                simrc_keep_idx.append(model_count - 1)  # row just appended to ens_matrix
                simrc_count += 1

n_members = model_count
print(f"\nConverged members: {n_members}")

if n_members == 0:
    sys.exit("No converged members found. Nothing to do.")

# ens_matrix shape: (n_members, n_free)
# axis=0 → reduce over members  (correct for all aggregate statistics)
# axis=1 → reduce over free parameters (was the bug in the original script)

# --- (2) Summary statistics -----------------------------------------------
P        = ENSEMBLE_PREFIX
ne       = ens_matrix.shape

ens_avg  = np.mean  (ens_matrix, axis=0)                           # (n_free,)
ens_var  = np.var   (ens_matrix, axis=0)                           # (n_free,)
ens_err  = np.sqrt(ens_var)                                        # (n_free,) -- std, comparable to MAD/QDIFF
ens_med  = np.median(ens_matrix, axis=0)                           # (n_free,)
ens_mad  = np.median(np.abs(ens_matrix - ens_med[np.newaxis, :]),
                     axis=0)                                        # (n_free,)
ens_prc  = np.percentile(ens_matrix, PERCENTILES, axis=0)          # (n_prc, n_free)

# --- Percentile-pair differences (robust spread, e.g. 1-sigma-equivalent IQR) ---
ens_qdiff = {}   # key -> (n_free,) array
for _lo, _hi in QDIFF_PAIRS:
    if _lo not in PERCENTILES or _hi not in PERCENTILES:
        print(f"  QDIFF_PAIRS: ({_lo}, {_hi}) not both in PERCENTILES — skipped.")
        continue
    _ilo = PERCENTILES.index(_lo)
    _ihi = PERCENTILES.index(_hi)
    _qkey = f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")
    ens_qdiff[_qkey] = np.abs(ens_prc[_ihi] - ens_prc[_ilo])       # (n_free,)

# --- Bootstrap variance estimate (optional alternative to the plug-in VAR) ---
ens_var_boot    = None
ens_err_boot    = None
ens_var_boot_se = None
if BOOTSTRAP_VAR:
    print(f"\nBootstrap variance estimation: {BOOTSTRAP_N} resamples "
          f"(seed={BOOTSTRAP_SEED if BOOTSTRAP_SEED is not None else '(fresh entropy)'}) …")
    _boot_rng = np.random.default_rng(BOOTSTRAP_SEED)
    ens_var_boot, ens_var_boot_se = _bootstrap_variance(
        ens_matrix, BOOTSTRAP_N, _boot_rng, out=OUT,
    )
    ens_err_boot = np.sqrt(ens_var_boot)

print(f"\nStatistics (over {n_members} members, {ne[1]} free parameters):")
print(f"  mean   log10(ρ): [{ens_avg.min():.3f}, {ens_avg.max():.3f}]")
print(f"  var    log10(ρ): [{ens_var.min():.4f}, {ens_var.max():.4f}]")
print(f"  err    log10(ρ): [{ens_err.min():.4f}, {ens_err.max():.4f}]  (= sqrt(var))")
print(f"  median log10(ρ): [{ens_med.min():.3f}, {ens_med.max():.3f}]")
print(f"  MAD    log10(ρ): [{ens_mad.min():.4f}, {ens_mad.max():.4f}]")
for _qkey, _qval in ens_qdiff.items():
    print(f"  {_qkey}: [{_qval.min():.4f}, {_qval.max():.4f}]")
if BOOTSTRAP_VAR:
    print(f"  var_boot  log10(ρ): [{ens_var_boot.min():.4f}, {ens_var_boot.max():.4f}]")
    print(f"  err_boot  log10(ρ): [{ens_err_boot.min():.4f}, {ens_err_boot.max():.4f}]  (= sqrt(var_boot))")
    print(f"  var_boot_se       : [{ens_var_boot_se.min():.4f}, {ens_var_boot_se.max():.4f}]  "
          f"(bootstrap SE of var_boot itself)")

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
        print("\n  COMPUTE_SENS: no results_iterX.h5 sensitivity files found — skipped.")
    else:
        if sens_missing_any or sens_count != n_members:
            print(f"\n  COMPUTE_SENS: sensitivity available for {sens_count}/"
                  f"{n_members} accepted members — aggregating over those only.")

        def _aggregate(M):
            """mean/median/min/max/std/cv across members (rows), linear."""
            agg = {
                "mean":   np.mean  (M, axis=0),
                "median": np.median(M, axis=0),
                "min":    np.min   (M, axis=0),
                "max":    np.max   (M, axis=0),
                "std":    (np.std(M, axis=0, ddof=1) if M.shape[0] > 1
                           else np.zeros(M.shape[1])),
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
        _row_max[_row_max == 0.0] = 1.0            # guard an all-zero member
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

        print(f"\n  Sensitivity (Jacobian, {sens_count} members, SENS_KIND="
              f"'{SENS_KIND}'), log10:")
        for _k in ("mean_raw", "mean_na", "mean_an"):
            print(f"    sens_{_k:<9s}: [{np.nanmin(sens[_k]):.3f}, "
                  f"{np.nanmax(sens[_k]):.3f}]  "
                  f"({int(np.sum(~np.isfinite(sens[_k])))} non-positive -> NaN)")
        print(f"    sens_cv (na / an, linear): "
              f"[{sens['cv_na'].min():.3f}, {sens['cv_na'].max():.3f}]  /  "
              f"[{sens['cv_an'].min():.3f}, {sens['cv_an'].max():.3f}]")

# --- (2c) Sensitivity statistics: (2) ensemble-native SimRC ----------------
simrc_coef = simrc_corr = None   # log10 after computation
if COMPUTE_SIMRC:
    if ens_data_matrix is None or simrc_count == 0:
        print("\n  COMPUTE_SIMRC: no per-member forward-response files found "
              "— skipped.")
    else:
        if simrc_missing_any or simrc_count != n_members:
            print(f"\n  COMPUTE_SIMRC: forward response available for "
                  f"{simrc_count}/{n_members} accepted members — SimRC "
                  f"computed over that matched subset only.")
        # Model side must use exactly the members that also contributed a
        # forward-response row -- COMPUTE_SENS can drop
        # different members, so ens_matrix and ens_data_matrix need not
        # otherwise line up row-for-row.
        _Msub  = ens_matrix[simrc_keep_idx, :]                # (simrc_count, n_free)
        _Mc    = _Msub - np.mean(_Msub, axis=0, keepdims=True)
        _var_m = (np.var(_Msub, axis=0, ddof=1) if simrc_count > 1
                  else np.ones(_Msub.shape[1]))
        _std_m = np.sqrt(_var_m)

        _Dc    = ens_data_matrix - np.mean(ens_data_matrix, axis=0, keepdims=True)
        _std_d = (np.std(ens_data_matrix, axis=0, ddof=1) if simrc_count > 1
                  else np.ones(ens_data_matrix.shape[1]))

        n_free_ = _Mc.shape[1]
        n_data_ = _Dc.shape[1]
        simrc_coef = np.zeros(n_free_)   # cumulative |regression coefficient|
        simrc_corr = np.zeros(n_free_)   # cumulative |correlation|
        _denom = max(simrc_count - 1, 1)

        print(f"\n  SimRC (ensemble-native sensitivity, {simrc_count} "
              f"members, {n_data_} data, chunk={SIMRC_CHUNK}) …")
        # Chunked over data columns so the (n_data, n_free) cross-covariance
        # is never formed in full -- peak memory O(SIMRC_CHUNK * n_free).
        for lo in range(0, n_data_, SIMRC_CHUNK):
            hi = min(lo + SIMRC_CHUNK, n_data_)
            _cov_block = (_Dc[:, lo:hi].T @ _Mc) / _denom      # (chunk, n_free_)
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
        print(f"    log10 cumulative |regression coeff.|: "
              f"[{np.nanmin(simrc_coef):.3f}, {np.nanmax(simrc_coef):.3f}]")
        print(f"    log10 cumulative |correlation|      : "
              f"[{np.nanmin(simrc_corr):.3f}, {np.nanmax(simrc_corr):.3f}]")

# --- (3) Empirical covariance (optional) -----------------------------------
ens_cov       = None
ens_covs      = None
ens_cov_eigval = None
ens_cov_eigvec = None

if COMPUTE_COV:
    if COV_METHOD == "low_rank":
        print("\nComputing low-rank covariance factorisation (thin SVD) …")
        _Xc = ens_matrix - ens_avg[np.newaxis, :]           # (m, n_free), centred
        _m  = _Xc.shape[0]
        # Thin SVD of the (m, n_free) centred ensemble: cost O(m^2 * n_free),
        # memory O(m * n_free) — never forms the n_free x n_free covariance.
        # C = Xc^T Xc / (m-1) = Vt.T @ diag(S^2/(m-1)) @ Vt, exactly (rank <= m-1).
        _U, _S, _Vt = np.linalg.svd(_Xc, full_matrices=False)
        ens_cov_eigval = (_S ** 2) / max(_m - 1, 1)          # (r,)  r = min(m, n_free)
        ens_cov_eigvec = _Vt.T                               # (n_free, r)
        print(f"  rank r={ens_cov_eigval.size}  "
              f"eigval range=[{ens_cov_eigval.min():.3e}, {ens_cov_eigval.max():.3e}]")
        print("  Full covariance can be reconstructed exactly as "
              "eigvec @ diag(eigval) @ eigvec.T")
    else:
        print("\nComputing empirical covariance …")
        ens_cov = sklearn.covariance.empirical_covariance(ens_matrix)

        if SPARSIFY:
            tmp    = ens_cov.copy()
            tmp[np.abs(tmp) / np.amax(np.abs(tmp)) <= SPARSE_THRESH] = 0.0
            ens_covs = scs.csr_array(tmp)
            nnz      = ens_covs.nnz
            total    = ens_cov.size
            print(f"  Sparse covariance: {nnz}/{total} non-zeros "
                  f"({100.0*nnz/total:.2f}%), threshold={SPARSE_THRESH:.1e}")
else:
    print("\nCOMPUTE_COV=False — skipping covariance estimation.")

# --- (4) Save .npz --------------------------------------------------------
ens_dict = {
    f"{P}_model_list": model_list,
    f"{P}_ens":        ens_matrix,
    f"{P}_avg":        ens_avg,
    f"{P}_var":        ens_var,
    f"{P}_err":        ens_err,
    f"{P}_med":        ens_med,
    f"{P}_mad":        ens_mad,
    f"{P}_prc":        ens_prc,
    f"{P}_prc_levels": np.asarray(PERCENTILES),
}
for _qkey, _qval in ens_qdiff.items():
    ens_dict[f"{P}_{_qkey}"] = _qval
if BOOTSTRAP_VAR:
    ens_dict[f"{P}_var_boot"]    = ens_var_boot
    ens_dict[f"{P}_err_boot"]    = ens_err_boot
    ens_dict[f"{P}_var_boot_se"] = ens_var_boot_se
for _k, _v in sens.items():               # log10 (cv_* linear)
    ens_dict[f"{P}_sens_{_k}"] = _v
if simrc_coef is not None:                # log10
    ens_dict[f"{P}_simrc_coef"] = simrc_coef
    ens_dict[f"{P}_simrc_corr"] = simrc_corr
if ens_cov is not None:
    ens_dict[f"{P}_cov"] = ens_cov
if ens_cov_eigval is not None:
    ens_dict[f"{P}_cov_eigval"] = ens_cov_eigval
    ens_dict[f"{P}_cov_eigvec"] = ens_cov_eigvec

np.savez_compressed(ENSEMBLE_RESULTS, **ens_dict)
print(f"\nResults saved → {ENSEMBLE_RESULTS}")

# --- (5) Resolve UTM origin and sites (needed for any plot) ---------------
if MOD_QC or MOD_STATS:
    (utm_e, utm_n, utm_lat, utm_lon,
     utm_zone, utm_north, site_xys, obs_coords_only) = _resolve_origin_and_sites()

    # --- Region of interest: override MOD_XLIM/YLIM/ZLIM from site bbox ---
    if MOD_ROI_AUTO and site_xys:
        _sx = np.array([s[1] for s in site_xys])   # model-local metres
        _sy = np.array([s[2] for s in site_xys])   # model-local metres
        MOD_XLIM = [float(_sx.min() / 1000.0 - MOD_ROI_PAD_XY),
                    float(_sx.max() / 1000.0 + MOD_ROI_PAD_XY)]
        MOD_YLIM = [float(_sy.min() / 1000.0 - MOD_ROI_PAD_XY),
                    float(_sy.max() / 1000.0 + MOD_ROI_PAD_XY)]
        if MOD_ROI_ZLIM is not None:
            MOD_ZLIM = list(MOD_ROI_ZLIM)
        print(f"\nROI (from {len(site_xys)} sites, pad={MOD_ROI_PAD_XY:.2f} km):")
        print(f"  MOD_XLIM = {MOD_XLIM} km")
        print(f"  MOD_YLIM = {MOD_YLIM} km")
        print(f"  MOD_ZLIM = {MOD_ZLIM} km")
    elif MOD_ROI_AUTO:
        print("\nROI: MOD_ROI_AUTO=True but no sites available — "
              "using literal MOD_XLIM/MOD_YLIM/MOD_ZLIM instead.")

# --- (6) Statistic map, style, and blanking/fading mask ---------------------
# _stat_map: key -> (vector over free parameters, description). Used both
# for MOD_STATS panels and as the namespace of MOD_ALPHA_SOURCE.
_stat_map = {
    "avg": (ens_avg, "mean"),
    "var": (ens_var, "variance"),
    "err": (ens_err, "error (std = sqrt(var))"),
    "med": (ens_med, "median"),
    "mad": (ens_mad, "MAD"),
}
_value_scale_keys = {"avg", "med"}
for _i, _pval in enumerate(PERCENTILES):
    _pkey = "p" + f"{_pval:g}".replace(".", "_")
    _stat_map[_pkey] = (ens_prc[_i], f"{_pval:g}th percentile")
    _value_scale_keys.add(_pkey)
for _qkey, _qval in ens_qdiff.items():
    _stat_map[_qkey] = (_qval, f"|{_qkey}| spread")
if BOOTSTRAP_VAR:
    _stat_map["var_boot"] = (ens_var_boot, "bootstrap variance")
    _stat_map["err_boot"] = (ens_err_boot, "bootstrap error (std)")
for _k, _v in sens.items():
    _stat_map[f"sens_{_k}"] = (
        _v, f"sensitivity {_k} ({'linear' if _k.startswith('cv') else 'log10'})")
if simrc_corr is not None:
    _stat_map["simrc_coef"] = (simrc_coef, "log10 SimRC cumulative |regression coeff.|")
    _stat_map["simrc_corr"] = (simrc_corr, "log10 SimRC cumulative |correlation|")


def _panel_style(key):
    """(cmap, clim, label) for a MOD_STATS panel: MOD_STATS_STYLE over defaults."""
    st = MOD_STATS_STYLE.get(key, {})
    if key in _value_scale_keys:
        d_clim, d_label = MOD_CLIM, "log10(rho / Ohm*m)"
    else:
        d_clim, d_label = None, key
    return (st.get("cmap", MOD_CMAP),
            st.get("clim", d_clim),
            st.get("label", d_label))


_alpha_block = None
_need_plot = (MOD_QC or MOD_STATS) and fviz is not None and bool(model_list)
if _need_plot and MOD_ALPHA_SOURCE:
    _a_name, _a_op, _a_thr = _parse_alpha_source(MOD_ALPHA_SOURCE)
    if _a_name not in _stat_map:
        print(f"\n  MOD_ALPHA_SOURCE: '{_a_name}' not available for this run "
              f"(available: {sorted(_stat_map)}) — blanking disabled.")
    else:
        _alpha = _alpha_from_condition(_stat_map[_a_name][0], _a_op, _a_thr,
                                       MOD_ALPHA_MODE, MOD_ALPHA_FADE_WIDTH)
        os.makedirs(MOD_STATS_DIR, exist_ok=True)
        _alpha_block = os.path.join(MOD_STATS_DIR,
                                    f"resistivity_block_{P}_alpha.dat")
        _write_alpha_block(_alpha, min(model_list, key=lambda x: x[2])[0],
                           _alpha_block)
        print(f"\n  MOD_ALPHA_SOURCE '{MOD_ALPHA_SOURCE}' "
              f"(mode='{MOD_ALPHA_MODE}'): "
              f"{int(np.sum(_alpha >= 1.0))} shown, "
              f"{int(np.sum((_alpha > 0.0) & (_alpha < 1.0)))} faded, "
              f"{int(np.sum(_alpha <= 0.0))} blanked "
              f"of {_alpha.size} cells.")

# --- (7) QC slice plot — best-nRMS member ---------------------------------
if MOD_QC:
    if fviz is None:
        print("\n  MOD_QC: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_QC: no converged members — skipping.")
    else:
        _best      = min(model_list, key=lambda x: x[2])
        _best_file, _best_iter, _best_nrms = _best
        print(f"\nQC: best member  nRMS={_best_nrms:.4f}  "
              f"iter={_best_iter}")
        _plot_slice(
            block_file      = _best_file,
            pdf_file        = MOD_QC_FILE,
            utm_e           = utm_e,
            utm_n           = utm_n,
            utm_lat         = utm_lat,
            utm_lon         = utm_lon,
            utm_zone        = utm_zone,
            utm_north       = utm_north,
            site_xys        = site_xys,
            obs_coords_only = obs_coords_only,
            alpha_file      = _alpha_block if MOD_ALPHA_QC else None,
        )

# --- (8) Statistics slice plots -------------------------------------------
if MOD_STATS:
    if fviz is None:
        print("\n  MOD_STATS: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_STATS: no converged members — skipping.")
    else:
        os.makedirs(MOD_STATS_DIR, exist_ok=True)

        # Template = lowest-nRMS member (preserves header / flag columns)
        _best_file = min(model_list, key=lambda x: x[2])[0]

        for _key in MOD_STATS_WHAT:
            if _key not in _stat_map:
                print(f"  MOD_STATS: '{_key}' not available for this run — skipped.")
                continue
            _vec, _label = _stat_map[_key]
            _cmap, _clim, _cbl = _panel_style(_key)
            _block_out = os.path.join(
                MOD_STATS_DIR, f"resistivity_block_{P}_{_key}.dat")
            _pdf_out = os.path.join(MOD_STATS_DIR, f"{P}_{_key}")
            print(f"\nSTATS: writing {_label} → {_block_out}")
            # insert_model writes 10**v; plot_model_slices log10s it back,
            # so the panel shows _vec itself (NaN -> 0 raw -> not drawn).
            fem.insert_model(
                template   = _best_file,
                model      = np.where(np.isfinite(_vec), _vec, -np.inf),
                model_file = _block_out,
                ocean      = MOD_OCEAN,
                air_rho    = _STATS_AIR_RHO,
                ocean_rho  = _STATS_OCEAN_RHO,
                out        = OUT,
            )
            print(f"STATS: plotting {_label} → {_pdf_out}  "
                  f"(cmap={_cmap}, clim={_clim})")
            _plot_slice(
                block_file      = _block_out,
                pdf_file        = _pdf_out,
                utm_e           = utm_e,
                utm_n           = utm_n,
                utm_lat         = utm_lat,
                utm_lon         = utm_lon,
                utm_zone        = utm_zone,
                utm_north       = utm_north,
                site_xys        = site_xys,
                obs_coords_only = obs_coords_only,
                clim            = _clim,
                cmap            = _cmap,
                cbar_label      = _cbl,
                alpha_file      = _alpha_block,
                stats_panel     = True,
            )

print("\nfemtic_ens_post.py complete.")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(__file__)
