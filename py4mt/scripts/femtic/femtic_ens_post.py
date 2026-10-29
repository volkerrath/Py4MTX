#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_post.py — Ensemble postprocessing for FEMTIC

Collects all converged members of a FEMTIC ensemble (RTO, GST, or any
directory-based ensemble), computes summary statistics, assembles the
empirical covariance, and saves everything to a compressed ``.npz`` file.

Optionally produces slice figures for the **best-nRMS member** (QC) and
for the **ensemble statistics** (mean, variance, median, MAD).

With ``PLOT_ONLY = True`` nothing is scanned or recomputed: the script reads
an existing results ``.npz`` (``PLOT_ONLY_NPZ``, default ``ENSEMBLE_RESULTS``)
and only redraws the figures (best member, statistic panels, boreholes, nRMS
histogram) -- for quick iteration on styling, slices, vertical exaggeration or
the reference model. femtic_ens_plot.py can work from the same file
(``RESULTS_NPZ``).

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

    This script targets FEMTIC's current HDF5 output layout only.
    AI-generated code -- review before production use.
"""

from __future__ import annotations

import math
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


def _plane_spec_km_to_m(spec):
    """Convert a MOD_SLICES ``kind="plane"`` entry (km) to a plot spec (m).

    Accepted keys (km / degrees): ``point`` = [x, y] or [x, y, z] model-local
    km (default [0, 0] = model origin), ``strike`` (geographic azimuth,
    degrees clockwise from north, for dip = 90; default 0), ``dip``
    (default 90), ``invert_x``, ``title``, and optionally ``xlim`` =
    [min, max] along-strike km relative to ``point``.

    For a vertical plane (dip = 90) without an explicit ``xlim`` the
    along-strike range is derived from the same limits as the other panels:
    the corners of the MOD_XLIM x MOD_YLIM box (east x north, model-local
    km) are projected onto the strike direction, so the section covers the
    same ROI. If either of MOD_XLIM / MOD_YLIM is None the whole mesh is
    shown. The depth range follows MOD_ZLIM like every other curtain panel
    (applied inside femtic_viz.plot_model_slices).
    """
    sp = dict(spec)
    pt = list(sp.get("point", [0.0, 0.0]))
    pt += [0.0] * (3 - len(pt))
    sp["point"] = [float(v) * 1000.0 for v in pt[:3]]
    sp.setdefault("strike", 0.0)
    sp.setdefault("dip", 90.0)
    if "xlim" in sp:
        sp["xlim"] = _lim_km_to_m(sp["xlim"])
    elif abs(float(sp["dip"]) - 90.0) < 1e-6:
        if MOD_XLIM is not None and MOD_YLIM is not None:
            _s = math.radians(float(sp["strike"]))
            _u = [
                (ex - pt[0]) * math.sin(_s) + (ny - pt[1]) * math.cos(_s)
                for ex in MOD_XLIM
                for ny in MOD_YLIM
            ]
            sp["xlim"] = [min(_u) * 1000.0, max(_u) * 1000.0]
        else:
            sp["xlim"] = None
    return sp


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

# ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/misti2026/ensemble_gst_0/"
# ENSEMBLE_NAME = "misti_rnd_0_"
# #: Prefix used for .npz output keys and default file/figure names.
# #: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
# ENSEMBLE_PREFIX = "misti_rnd_0"

#: Name of the FEMTIC convergence file inside each ensemble member
#: directory. A single name or a list of candidates (first existing wins),
#: e.g. CNV_FILE = ["femtic.cnv", "femtic_run2.cnv"].
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

#: Maximum normalised RMS accepted from the convergence file (CNV_FILE).
NRMS_MAX = 2.

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
# Plot-only mode
# ---------------------------------------------------------------------------
#: False (default): full run -- scan the member directories, compute the
#: statistics, write the results .npz, then plot.
#: True: PLOT-ONLY -- read an existing results .npz (ENSEMBLE_RESULTS, i.e.
#: <RESULTS_DIR>/<ENSEMBLE_PREFIX.upper()>_results.npz) and run only the
#: plotting sections: "best" panel, MOD_STATS_WHAT statistic panels,
#: boreholes (MOD_BH) and the nRMS histogram. No member directory, cnv or
#: h5 file is read, nothing is recomputed (missing avg/var/err/med/mad/prc
#: are rebuilt from the stored members), and the .npz is NOT rewritten, so
#: styling, slices, MOD_VERT_EXAG, reference model, colour limits etc. can be
#: iterated in seconds. All plotting settings below act as in a full run.
#: What the file does not hold is not plotted: sens_* / simrc_* / *_boot
#: panels exist only if the full run computed them (COMPUTE_*, BOOTSTRAP_VAR),
#: and MOD_STATS_WHAT keys without data are skipped with a message. The
#: percentile levels are taken from the file. Still needed on disk: MOD_MESH,
#: and a block file for the region structure (see PLOT_ONLY_TEMPLATE).
PLOT_ONLY = False

#: Results .npz to read in plot-only mode. None = ENSEMBLE_RESULTS (see
#: above). If the file contains one ensemble under a different key prefix
#: than ENSEMBLE_PREFIX, that prefix is detected automatically (outputs are
#: still named after ENSEMBLE_PREFIX).
PLOT_ONLY_NPZ = None

#: Template block file in plot-only mode (None = the best member's block file
#: recorded in the .npz). Only its header / region structure / flag columns
#: are used; all panels are drawn from the stored vectors. Set it when the
#: results were moved to another machine or the member directories are gone:
#: any resistivity block file of this ensemble (same mesh, same free regions)
#: will do, e.g. a copy of the starting model. The "best" panel is then
#: rebuilt from the stored best vector instead of the member's own file.
PLOT_ONLY_TEMPLATE = None

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
SENS_ERROR_KEY = "data_errors"  # None to disable error-weighting

#: All sensitivity aggregates (sens_mean/_median/_min/_max/_std, in their
#: _raw/_na/_an versions) and the SimRC sums (simrc_coef/_corr) are strictly
#: positive and span many decades, so they are converted to log10 right
#: after aggregation -- in memory, in the .npz, in the plotted block files,
#: and in MOD_ALPHA_SOURCE expressions (e.g. "sens_mean_an > -3." means
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
#: Directory for ALL outputs: results .npz, "best" figure, stats_plots/ and
#: borehole figures/profiles. Created if missing. Absolute path, or build
#: it from ENSEMBLE_DIR (default: <ENSEMBLE_DIR>/results/).
RESULTS_DIR = os.path.join(ENSEMBLE_DIR, "results") + os.sep
ENSEMBLE_RESULTS = RESULTS_DIR + ENSEMBLE_PREFIX.upper() + "_results.npz"

#: Per-member best nRMS table (CSV): member, best iteration, nRMS, status
#: ("accepted", "accepted_fallback" (see ITER_FALLBACK), "rejected_nrms_max",
#: "no_model_file"). Always written.
NRMS_CSV = RESULTS_DIR + ENSEMBLE_PREFIX + "_nrms.csv"
#: Histogram of the per-member best nRMS (all members with a valid nRMS,
#: accepted and rejected shown stacked, NRMS_MAX marked). Extension-less
#: base path; one file per MOD_PLOT_FORMAT entry.
MOD_NRMS_HIST = True
MOD_NRMS_HIST_FILE = RESULTS_DIR + ENSEMBLE_PREFIX + "_nrms_hist"
MOD_NRMS_BINS = "auto"  # int or any numpy.histogram_bin_edges rule

# ---------------------------------------------------------------------------
# Mesh (required for any slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = ENSEMBLE_DIR + "templates/mesh.dat"

#: Optional swap of the first two node-coordinate columns when reading
#: MOD_MESH (femtic.read_femtic_mesh, which is used by every plot and by the
#: borehole block). None = leave femtic's default (no swap: coordinates as
#: stored in mesh.dat, with x = easting, y = northing); True = swap, for
#: meshes stored as x = northing, y = easting; False = force no swap.
MOD_MESH_SWAP_XY = None
if MOD_MESH_SWAP_XY is not None:
    fem.MESH_SWAP_XY = bool(MOD_MESH_SWAP_XY)

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN = None
MOD_AIR_RHO = 1.0e9  # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25  # Ω·m  (region 1 when treated as ocean)

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
    [MOD_PLOT_FORMAT]
    if isinstance(MOD_PLOT_FORMAT, str)
    else list(MOD_PLOT_FORMAT)
)

# ---------------------------------------------------------------------------
# Panels to plot -- ONE list of sources (MOD_STATS_WHAT) plus ONE blanking
# option (MOD_ALPHA_SOURCE, shared slice block below). There used to be a
# separate MOD_QC on/off switch to keep in sync with this list by hand;
# it is now just the special "best" entry below, so "what gets plotted"
# lives in exactly one place. MOD_QC / MOD_STATS still exist afterwards
# (derived, not user-set) purely so the rest of the script (sections
# (7)/(8) and the origin/site-resolution gate) is unchanged.
# ---------------------------------------------------------------------------
#: Which panels to plot. The special entry "best" plots the best-nRMS
#: converged member (to MOD_QC_FILE) -- formerly the standalone MOD_QC
#: switch. Every other entry is a statistic key: subset of "avg", "var",
#: "err", "med", "mad", one auto-generated key per PERCENTILES level (e.g.
#: 2.3 -> "p2_3", 50.0 -> "p50", 97.7 -> "p97_7"), one per QDIFF_PAIRS
#: entry (e.g. (15.9, 84.1) -> "qdiff_15_9_84_1"), "err_boot"
#: (+ "var_boot") when BOOTSTRAP_VAR=True, and -- when COMPUTE_SENS /
#: COMPUTE_SIMRC succeed -- the log10 sensitivity panels "sens_mean_raw",
#: "sens_mean_na", "sens_mean_an", "sens_median_na"/"_an",
#: "sens_min_na"/"_an", "sens_max_na"/"_an", "sens_std_na"/"_an", the
#: linear ratios "sens_cv_na"/"_an", and "simrc_coef"/"simrc_corr"
#: (log10). A key requested here but not actually available for this run
#: (e.g. its COMPUTE_* flag is off) is skipped with a printed message,
#: never an error -- so trimming a COMPUTE_* flag never requires editing
#: this list back down too.
#: "err" = sqrt(var) is plotted by default instead of "var", since var is in
#: (log10 Ohm.m)^2 and not on the same scale as MAD/QDIFF (log10 Ohm.m).
#: Colour map, colour limits and colourbar label of every statistic panel
#: are set per key in MOD_STATS_STYLE (shared slice block below); "best"
#: always plots at MOD_CLIM/MOD_CMAP directly (no MOD_STATS_STYLE entry).
MOD_STATS_WHAT = (
      ["best", "avg", "med", "err", "mad"]
#    + ["p" + f"{_p:g}".replace(".", "_") for _p in PERCENTILES]
    + [f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_") for _lo, _hi in QDIFF_PAIRS]
    + (["err_boot"] if (BOOTSTRAP_VAR or PLOT_ONLY) else [])
    + (
        [
            "sens_mean_raw",
            "sens_mean_an",
            "sens_median_an",
            "sens_cv_an",
        ]
        if (COMPUTE_SENS or PLOT_ONLY)
        else []
    )
    + (["simrc_corr"] if (COMPUTE_SIMRC or PLOT_ONLY) else [])
    + (["simrc_coef"] if (COMPUTE_SIMRC or PLOT_ONLY) else [])
)

#: Extension-less base path for the "best" panel; _plot_slice() appends
#: ".<fmt>" per MOD_PLOT_FORMAT entry.
MOD_QC_FILE = RESULTS_DIR + ENSEMBLE_PREFIX + "_best"

#: Output directory for the other panels' block files and figures.
#: Requires MOD_MESH and a valid template file (taken from best member).
MOD_STATS_DIR = RESULTS_DIR + "stats_plots/"

#: Derived from MOD_STATS_WHAT above -- not user-set. Remove "best" from
#: the list (or set MOD_STATS_WHAT = []) to turn either off, instead of a
#: separate boolean.
MOD_QC = "best" in MOD_STATS_WHAT
MOD_STATS = bool(set(MOD_STATS_WHAT) - {"best"})

#: ROI diagnostic for the log10 sensitivity panels ("sens_mean_raw",
#: "sens_{mean,median,min,max,std}_{na,an}" -- NOT "sens_cv_na"/"_an",
#: which are linear ratios, and NOT "simrc_coef"/"simrc_corr", left
#: untouched pending more experience with that method).
#:
#: Regardless of SENS_ROI_NORMALIZE below, every such panel that is both
#: listed in MOD_STATS_WHAT and actually plotted has its *original*
#: (whole-mesh-normalised, as currently stored/plotted) min/max printed
#: to the console, restricted to the free parameters inside the ROI (the
#: same box driving the MOD_STATS slice plots: MOD_XLIM/MOD_YLIM/MOD_ZLIM,
#: including any MOD_ROI_AUTO override) -- a quick read of the value range
#: actually in view, to help pick a MOD_ALPHA_SOURCE threshold.
#:
#: SENS_ROI_NORMALIZE (default False) additionally *re-centres* each such
#: panel, right before it is written/plotted, so that the ROI's own max
#: becomes log10 = 0 instead of the whole-mesh max (equivalent to dividing
#: the underlying linear quantity by its max within the ROI rather than
#: over the whole mesh). This is a display-only transform: it never
#: touches the .npz output (sens_mean_an etc. are saved exactly as before)
#: or MOD_ALPHA_SOURCE (which still reads the untransformed _stat_map
#: values). Cells outside the ROI can come out > 0 (log10) once
#: renormalised this way -- expected, since they may exceed the ROI's max.
SENS_ROI_NORMALIZE = True

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_mod_plot_slice.py — used by both MOD_QC and MOD_STATS below)
# ---------------------------------------------------------------------------

# --- Geographic / UTM origin of the mesh centre ----------------------------
#: Set to None when MOD_ORIGIN_METHOD will estimate the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT = None  # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON = None  # decimal degrees, positive = East
MOD_UTM_ORIGIN_E = None  # UTM easting  [m]
MOD_UTM_ORIGIN_N = None  # UTM northing [m]
MOD_UTM_ZONE_OVERRIDE = None  # override auto-derived zone; None = auto

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
MOD_SITE_DAT = ENSEMBLE_DIR + "templates/site.dat"
MOD_SITE_NAMES = None  # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS = True  # show markers on map panels
MOD_PLOT_SITES_SLICES = False  # show markers on curtain / plane panels
#: Max distance [km] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 1.0  # km; None = show all sites on every panel

MOD_SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10, label=None)
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
    dict(kind="map", z0=-0.25),  # km
    dict(kind="map", z0=0.0),  # km
    dict(kind="map", z0=0.5),  # km
    dict(kind="map", z0=1.0),  # km
    dict(kind="map", z0=2.0),  # km
    dict(kind="map", z0=2.5),  # km
    dict(kind="map", z0=3.0),  # km
    dict(kind="map", z0=4.0),  # km
    dict(kind="map", z0=5.0),  # km
    dict(kind="ns", x0=0.0),  # km
    dict(kind="ew", y0=0.0),  # km
    #: Vertical section along a strike through ``point`` (model-local km,
    #: [0, 0] = model origin). strike = geographic azimuth, degrees
    #: clockwise from north: 45 = SW (left) -> NE (right), 135 = NW -> SE.
    #: Horizontal axis = along-strike distance from ``point`` in model km;
    #: its range follows MOD_XLIM x MOD_YLIM (projected onto the strike),
    #: its depth range MOD_ZLIM. Optional: xlim=[min, max] (km) to override.
    dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),
    # dict(kind="ns",  x0=(-71.40723, 'latlon')),    # km
    # dict(kind="ew",  y0=(-16.299593, 'latlon')),    # km
]

# --- Reference model (optional) -----------------------------------------------
#: A reference model to compare the ensemble with: a resistivity block file
#: (resistivity_block_iterX.dat) on the SAME mesh as MOD_MESH -- e.g. a
#: deterministic inversion, the prior/starting model or a published model.
#: Its region structure may differ from the members'. None = no reference
#: (everything below is then ignored). A missing file, or one with another
#: element count than MOD_MESH, switches the reference off with a message.
# MOD_REF_FILE = None  # e.g. ENSEMBLE_DIR + "reference/resistivity_block_iter20.dat"
MOD_REF_FILE =ENSEMBLE_DIR + "/templates/referencemodel.dat" 
#: Slices: draw the reference as iso-resistivity lines on every slice panel
#: (the "best" panel and all MOD_STATS panels). The lines are the exact
#: boundaries between slice cells lying on different sides of a level; air
#: and ocean cells of the reference are ignored.
MOD_REF_SLICES = False
#: Boreholes: add the reference profile as a dashed line "reference model"
#: to every MOD_BH panel. A MOD_BH entry can switch it off with ref=False, or
#: list "ref" in its "what" to force it on and to set its drawing position.
#: Style: MOD_BH_STYLE["ref"].
MOD_REF_BH = True
#: Contour levels in log10(Ohm.m), e.g. [1.0, 2.0, 3.0]. None = the integer
#: decades inside MOD_CLIM (the reference model's own range if MOD_CLIM is
#: None). Always in the reference model's units, whatever quantity a
#: MOD_STATS panel shows.
MOD_REF_LEVELS = None
#: LineCollection keywords of the slice lines, e.g.
#: dict(colors="k", linewidths=0.8, linestyles="--").
MOD_REF_STYLE = dict(colors="k", linewidths=0.8)
#: Legend entry of the slice lines (lower right of each panel); None = none.
MOD_REF_LABEL = "reference model"

# --- Borehole profiles (optional) --------------------------------------------
#: 1-D resistivity-depth profiles through the ensemble at vertical "boreholes".
#: One panel per entry, all panels in one figure (MOD_BH_FILE). An empty list
#: ([]) switches the block off. Entries are dicts, in the same spirit as
#: MOD_SLICES (plain horizontal positions are model-local km, depths km):
#:
#:   "name"  : str    -- panel title / key in the .npz file         (required)
#:   "x","y" : horizontal position. Plain float = model-local km (origin =
#:             mesh centre); (value, "utm") = UTM metres in the mesh zone;
#:             (value, "latlon") = decimal degrees (x = longitude,
#:             y = latitude). x and y must carry the same tag. (required,
#:             unless "sites" is given)
#:   "sites" : instead of "x"/"y": "all" (every site in MOD_SITE_DAT) or a
#:             list of site names. The entry is then a template: it is
#:             expanded to one borehole per site, named after the site, with
#:             all other keys ("z_top", "z_bot", "dz", "what", "alpha",
#:             "xlim", "style") applied to each. Needs MOD_SITE_DAT.
#:   "n_random", "seed" : with "sites": use only N sites drawn at random
#:             (without replacement) from the selected ones; "seed" = int
#:             makes the draw reproducible, None (default) draws afresh
#:             every run. The chosen names are printed.
#:   "z_top" : float km (z-down, model datum) or "surface" = top of the first
#:             non-air element at (x, y) (sea level offshore). Default "surface".
#:   "z_bot" : float km, bottom of the profile. Default: MOD_ZLIM[1] if set,
#:             else 20 km.
#:   "dz"    : float km, sampling interval. Default 0.05 (50 m).
#:   "what"  : list of what to draw (order = drawing order). Any of
#:               "avg"          ensemble mean of log10(rho) (geometric mean)
#:               "med"          ensemble median
#:               "percentiles"  one line per level in PERCENTILES
#:               "p50", "p15_9" ... single percentile(s), keys as in
#:                              MOD_STATS_WHAT
#:               "sigma1"       the 1-sigma quantile pair, P15.87 and P84.13 of
#:                              the members (log10 rho), two lines
#:               "sigma2"       the 2-sigma pair, P2.28 and P97.72
#:               "ensemble"     every accepted member, thin lines drawn with
#:                              opacity "alpha"
#:               "best"         the best-nRMS member
#:               "heatmap"      density heat map of the ensemble: at every
#:                              depth level the members' log10(rho) are
#:                              histogrammed (MOD_BH_HEAT_BINS bins across
#:                              the panel's resistivity range) and shown as
#:                              a colour image behind all other items; one
#:                              colour bar per figure. See MOD_BH_HEAT_*.
#:                              Combine freely with "ensemble" (lines on
#:                              top of it), "med", "sigma1", ... Put it
#:                              anywhere in "what" -- it is always drawn at
#:                              the bottom.
#:               "ref"          the reference model (MOD_REF_FILE); added
#:                              automatically (drawn last, on top) when
#:                              MOD_REF_BH is True; list it here to choose its
#:                              position, or to force it on when MOD_REF_BH is
#:                              False
#:             Default ["avg", "percentiles"].
#:   "ref"   : False = no reference line in this panel although MOD_REF_BH is
#:             True. Default True.
#:   "alpha" : opacity of the "ensemble" lines, 0..1. Default MOD_BH_ALPHA.
#:   "xlim"  : [rho_min, rho_max] in Ohm.m for this panel (default MOD_BH_XLIM).
#:   "heat_bins" : bins of the "heatmap" for this entry: an int (number of
#:             equal bins in log10(rho) across the panel's xlim) or an
#:             increasing list of log10(rho) bin edges. Default
#:             MOD_BH_HEAT_BINS.
#:   "style" : dict of per-entry style overrides, keyed like MOD_BH_STYLE.
#: Air and out-of-mesh levels are left as gaps.
MOD_BH = [
    # dict(
    #     name="model centre",
    #     x=0.0,
    #     y=0.0,  # km, model-local
    #     z_top="surface",
    #     z_bot=4.0,
    #     dz=0.05,  # km
    #     what=["ensemble", "med", "sigma1". "sigma2",],  # ensemble first = underneath
    #     alpha=0.3,
    # ),
    # dict(sites="all", n_random=6, seed=1,   # 6 random sites, reproducible
    #      z_bot=5.0, what=["ensemble", "med", "sigma1"], alpha=0.05),
    #     what=["ensemble", "med", "sigma1". "sigma2",],  # ensemble first = underneath
    #     alpha=0.3,
    dict(sites="all", z_top="surface", z_bot=4.0, dz=0.05,   # one panel per site
             what=["ensemble", "med", "sigma1", "sigma2",],  # ensemble first = underneath
             alpha=0.4, xlim=[1., 10000.],),
    # dict(sites=["SITE01", "SITE07"], z_bot=5.0, what=["med", "best"]),
    # dict(name="BH-1",
    #      x=(-70.868025, "latlon"), y=(-16.363436, "latlon"),
    #      z_top="surface", z_bot=4.0, dz=0.05,
    #      what=["med", "p15_9", "p84_1", "best"]),
    # dict(name="BH-2",
    #      x=(229100.0, "utm"), y=(8184000.0, "utm"),
    #      z_top=0.0, z_bot=6.0, dz=0.1,
    #      what=["avg", "ensemble"], alpha=0.08,
    #      style=dict(avg=dict(color="navy", lw=3))),
]

#: Output path (no extension) of the borehole figure; one file per
#: MOD_PLOT_FORMAT entry is written.
MOD_BH_FILE = RESULTS_DIR + ENSEMBLE_PREFIX + "_boreholes"
#: Export the sampled profiles (depth, statistics, all members) to an .npz
#: next to MOD_BH_FILE (<MOD_BH_FILE>.npz). False = skip.
MOD_BH_NPZ = True
#: Default opacity of the "ensemble" lines (per-entry "alpha" overrides).
MOD_BH_ALPHA = 0.15
#: x-axis (resistivity) range in Ohm.m, log scale. None = 10**MOD_CLIM
#: (shared with the slice figures); if MOD_CLIM is None too, auto.
MOD_BH_XLIM = None
#: Panel size [width, height] in cm and number of panel columns
#: (None = up to 4 columns). MOD_BH_PER_FIG = maximum panels per figure
#: (None = all in one); further figures get a _01, _02, ... suffix. Useful
#: with "sites".
MOD_BH_PANEL_SIZE = [16.0, 18.0]
MOD_BH_NCOLS = 3
MOD_BH_PER_FIG = 12
#: True = all panels use one common depth range (union of the entries'
#: ranges), so profiles can be compared by eye.
MOD_BH_SHARE_DEPTH = True
#: Matplotlib legend location per panel; None = no legend.
MOD_BH_LEGEND_LOC = "lower right"
#: Heat map ("heatmap" in a MOD_BH entry's "what"): number of bins in
#: log10(rho) across the panel's resistivity range (the entry's "xlim", else
#: MOD_BH_XLIM, else 10**MOD_CLIM; if all are None, the range of the
#: members' values at that borehole). May also be a list of log10(rho) bin
#: edges. Members outside the range are not shown (e.g. ocean cells at
#: 0.25 Ohm.m with xlim [1, 1e4]); the fractions still refer to all valid
#: members, so such a level sums to less than 1. A per-entry "heat_bins"
#: overrides this.
MOD_BH_HEAT_BINS = 60
#: Colour scale of the heat map, one scale (and one colour bar) per figure:
#:   "fraction" : fraction of the valid members falling into a bin (a depth
#:                level sums to 1 if all members are inside the range) --
#:                comparable between depths, boreholes and ensembles of
#:                different size.
#:   "row_max"  : every depth level scaled to its own maximum (1 = most
#:                populated bin) -- shows the shape of the distribution
#:                even where it is broad (small fractions).
#: The .npz always holds the fractions; "row_max" is a display option only.
MOD_BH_HEAT_NORM = "fraction"
#: Colormap of the heat map (any Matplotlib name; light = few members). A
#: single-hue light-to-dark map ("Greys", "Blues", "Purples", ...) keeps the
#: lines on top readable. MOD_BH_STYLE["heatmap"] or an entry's
#: style=dict(heatmap=dict(cmap=...)) overrides it; the colour bar uses the
#: cmap of the first panel that draws a heat map.
MOD_BH_HEAT_CMAP = "Greys"
#: Shape of the colour scale between MOD_BH_HEAT_VMIN and MOD_BH_HEAT_VMAX:
#:   "linear" : colour proportional to the value (default; unchanged output)
#:   "sqrt"   : square-root scale -- lifts the low densities, keeps the
#:              peaks distinguishable
#:   "log"    : logarithmic scale, over the decades between VMIN and VMAX
#:              (default VMIN = smallest non-zero value of the figure, i.e.
#:              one member); empty bins (0) are left blank.
MOD_BH_HEAT_SCALE = "linear"
#: Lower end of the colour scale. None = 0 ("linear" / "sqrt"); the smallest
#: non-zero value of the figure ("log"; a log scale needs a value > 0). A
#: number sets it explicitly, e.g. 0.02 to blank out bins holding less than
#: 2 % of the members; values below it get the lowest colour (an arrow on the
#: colour bar), and with "log" they are left out.
MOD_BH_HEAT_VMIN = None
#: Upper end of the colour scale ("fraction" norm; "row_max" is always 1).
#: None = for "linear" / "sqrt" the 99th percentile of the non-zero fractions
#: of all panels of the figure, so that a few levels in which every member has
#: the same value (the ocean, fixed regions: fraction 1) do not wash out the
#: rest (the colour bar then ends in an arrow); for "log" the maximum. A
#: number sets it explicitly (1.0 = the full range).
MOD_BH_HEAT_VMAX = None
#: Draw the colour bar of the heat map (one per figure).
MOD_BH_HEAT_CBAR = True
#: Line style per drawn item, keyed "avg", "med", "best", "ensemble",
#: "percentiles" (all percentile lines), a single percentile key such as
#: "p50", or "sigma1" / "sigma2" (both lines of the pair). Any Matplotlib Line2D keyword is allowed. "percentiles" may carry
#: "cmap" instead of "color": the lines are then coloured by level from that
#: colormap. Per-entry "style" dicts override these.
MOD_BH_STYLE = {
    "avg": dict(color="black", lw=2.0, ls="-"),
    "med": dict(color="crimson", lw=2.0, ls="-"),
    "best": dict(color="tab:orange", lw=1.6, ls="-"),
    "ensemble": dict(color="0.5", lw=0.7, ls="-"),
    "percentiles": dict(cmap="viridis", lw=1.2, ls="--"),
    "sigma1": dict(color="navy", lw=1.6, ls="--"),
    "sigma2": dict(color="steelblue", lw=1.2, ls=":"),
    "ref": dict(color="black", lw=2.2, ls="--"),
    # heat map: any pcolormesh keyword, and "cmap" to override
    # MOD_BH_HEAT_CMAP (the colour scale itself: MOD_BH_HEAT_SCALE / _VMIN /
    # _VMAX; the colour bar uses the cmap of the first panel with a heat map)
    "heatmap": dict(rasterized=True),
}
if MOD_BH_HEAT_NORM not in ("fraction", "row_max"):
    raise ValueError(
        f"MOD_BH_HEAT_NORM must be 'fraction' or 'row_max', "
        f"got {MOD_BH_HEAT_NORM!r}."
    )
if MOD_BH_HEAT_SCALE not in ("linear", "sqrt", "log"):
    raise ValueError(
        f"MOD_BH_HEAT_SCALE must be 'linear', 'sqrt' or 'log', "
        f"got {MOD_BH_HEAT_SCALE!r}."
    )
if MOD_BH_HEAT_VMIN is not None and MOD_BH_HEAT_VMIN < 0:
    raise ValueError("MOD_BH_HEAT_VMIN must be >= 0 (or None).")
if MOD_BH_HEAT_SCALE == "log" and MOD_BH_HEAT_VMIN is not None:
    if MOD_BH_HEAT_VMIN <= 0:
        raise ValueError("MOD_BH_HEAT_SCALE='log' needs MOD_BH_HEAT_VMIN > 0.")

MOD_XLIM = None  # [xmin, xmax] model-local km; None = auto
MOD_YLIM = None  # [ymin, ymax] model-local km; None = auto
MOD_ZLIM = None  # [zmin, zmax] model-local km; None = auto

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
MOD_ROI_AUTO = True
MOD_ROI_PAD_XY = 4.0  # km of padding around the site bbox
MOD_ROI_ZLIM = [-1.0, 5.0]
#: depth range (km, positive-down) for ns/ew/plane panels; None = leave MOD_ZLIM as-is
#: Lower bound is negative (above the z=0 datum) to give ~1 km of headroom
#: so topography (mesh cells with z < 0) is not clipped out of the ns/ew/
#: plane panels. Previously [0.0, 20000.0] cut panels off exactly at the
#: datum, hiding any topography above it.

MOD_DPI = 400  # figure DPI, used by both MOD_QC and MOD_STATS
MOD_CMAP = "jet_r"
MOD_CLIM = [0.0, 4.0]  # [log10_min, log10_max] Ω·m; None = auto
MOD_OCEAN_COLOR = "lightgrey"  # flat colour for ocean cells; None = colormap
MOD_AIR_COLOR = "whitesmoke"
MOD_AIR_BGCOLOR = None  # axes facecolor for air; None = figure default

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
    "var": dict(clim=[0.0, 1.0], label="var log10(rho)"),
    "err": dict(clim=[0.0, 1.0], label="std log10(rho)"),
    "mad": dict(clim=[0.0, 1.0], label="MAD log10(rho)"),
}
for _lo, _hi in QDIFF_PAIRS:
    MOD_STATS_STYLE[f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")] = dict(
        clim=[0.0, 1.0], label=f"P{_hi:g} - P{_lo:g}  log10(rho)"
    )
if BOOTSTRAP_VAR or PLOT_ONLY:
    MOD_STATS_STYLE["var_boot"] = dict(
        clim=[0.0, 1.0], label="bootstrap var log10(rho)"
    )
    MOD_STATS_STYLE["err_boot"] = dict(
        clim=[0.0, 1.0], label="bootstrap std log10(rho)"
    )
if COMPUTE_SENS or PLOT_ONLY:
    #: Native SENS_KIND units -- run- and mesh-dependent, auto-scaled.
    MOD_STATS_STYLE["sens_mean_raw"] = dict(
        cmap="viridis", clim=None, label=f"log10 S ({SENS_KIND})"
    )
    for _sfx in ("na", "an"):
        for _st in ("mean", "median", "min", "max", "std"):
            MOD_STATS_STYLE[f"sens_{_st}_{_sfx}"] = dict(
                cmap="viridis",
                clim=[-5.0, 0.0],
                label=f"log10 S/Smax ({_st}, {_sfx})",
            )
        MOD_STATS_STYLE[f"sens_cv_{_sfx}"] = dict(
            cmap="viridis", clim=None, label=f"CV of S ({_sfx})"
        )
if COMPUTE_SIMRC or PLOT_ONLY:
    MOD_STATS_STYLE["simrc_coef"] = dict(
        cmap="viridis", clim=None, label="log10 sum|SimRC|"
    )
    MOD_STATS_STYLE["simrc_corr"] = dict(
        cmap="viridis", clim=None, label="log10 sum|corr|"
    )

# --- Blanking / fading (optional) ------------------------------------------
#: One condition "<name> <op> <value>" selecting the cells to SHOW; cells
#: failing it are blanked or faded. <name> is any statistic key known to
#: MOD_STATS (see MOD_STATS_WHAT; computed statistics are available here
#: even if not listed for plotting), <op> is one of > >= < <=, and <value>
#: is in that statistic's own units (log10 for sensitivities). Examples:
#:   "sens_mean_an > -3."    keep cells within 3 decades of max sensitivity
#:   "err < 0.2"             keep cells with ensemble std below 0.2 decades
#: None disables blanking/fading.
MOD_ALPHA_SOURCE = "sens_median_an > -5.5"
#: "blank" -- failing cells are removed (hard cut).
#: "fade"  -- failing cells fade linearly from opaque at the threshold to
#:            fully transparent MOD_ALPHA_FADE_WIDTH beyond it (same units
#:            as <value>, e.g. 1.0 = one decade for log10 sensitivities).
MOD_ALPHA_MODE = "blank"
MOD_ALPHA_SOURCE = "sens_median_an > -5."
#: "blank" -- failing cells are removed (hard cut).
#: "fade"  -- failing cells fade linearly from opaque at the threshold to
#:            fully transparent MOD_ALPHA_FADE_WIDTH beyond it (same units
#:            as <value>, e.g. 1.0 = one decade for log10 sensitivities).
MOD_ALPHA_MODE = "fade"
MOD_ALPHA_FADE_WIDTH = 1.

#: Apply the same mask to the MOD_QC (best-member) figure as well.
MOD_ALPHA_QC = True

#: Statistic panels whose key starts with one of these prefixes are drawn
#: WITHOUT the mask. Default: the sensitivity panels ("sens_*", "simrc_*") --
#: shading a sensitivity map by sensitivity is circular, and the unmasked map
#: is what shows where the MOD_ALPHA_SOURCE threshold cuts. All other panels
#: (avg, err, ..., and the best panel via MOD_ALPHA_QC) are masked as before.
#: () = mask every panel (the behaviour of earlier versions).
MOD_ALPHA_EXCLUDE = ("sens_", "simrc_")

# --- Figure layout -----------------------------------------------------------
MOD_EQUAL_ASPECT = True
#: Vertical exaggeration of the section panels (ns, ew and vertical "plane").
#: None = as set by MOD_EQUAL_ASPECT (unchanged). A number fixes the aspect of
#: every section from the real axis units: 1.0 = true equal axes, 2.0 = the
#: vertical scale twice the horizontal one. Works for every
#: MOD_DISPLAY_COORDS, also "latlon", where MOD_EQUAL_ASPECT alone does not
#: act (its test needs model/utm display); map panels stay 1:1 while
#: MOD_EQUAL_ASPECT is True.
MOD_VERT_EXAG = None
MOD_DEPTH_KM = True
MOD_HORIZ_KM = True
#: 2x2 grid matching the 4 default MOD_SLICES panels (2 maps + ns + ew).
#: Adjust to len(MOD_SLICES) if you change the number of panels; None/None
#: falls back to a single row of len(MOD_SLICES) columns.
MOD_NROWS = 4  # None = auto (1 row)
MOD_NCOLS = 3  # None = auto (len(MOD_SLICES) cols)
MOD_PANEL_HEIGHT = 18.0  # cm
#: None = auto per-column width from each panel's own aspect ratio (needs
#: MOD_EQUAL_ASPECT=True and real xlim/ylim/zlim -- supplied automatically
#: by MOD_ROI_AUTO above -- so map, ns, and ew panels naturally end up
#: different widths instead of being forced square).
MOD_PANEL_WIDTH = None  # cm; None = auto from aspect ratio
MOD_FIGSIZE = None  # [w, h] cm; overrides auto when set

#: Axis annotation font sizes, passed through to fviz.plot_model_slices.
#: Defaults match plot_model_slices' own defaults.
MOD_TICK_FONTSIZE = 16  # axis tick labels, colourbar ticks
MOD_LABEL_FONTSIZE = 16  # axis labels, panel titles, colourbar label
#: Font size of the figure title (the block-file name above the panels);
#: None = MOD_LABEL_FONTSIZE + 6.
MOD_TITLE_FONTSIZE = MOD_LABEL_FONTSIZE + 10

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
_IN_SPYDER = utl.runtime_env() == "spyder"
_SHOW_PLOTS = MOD_SHOW_IN_SPYDER and _IN_SPYDER
if _IN_SPYDER:
    print(
        f"Detected Spyder — inline figure display {'enabled' if _SHOW_PLOTS else 'disabled (MOD_SHOW_IN_SPYDER=False)'}.\n"
    )

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


#: Sentinel air/ocean resistivities used only in MOD_STATS block files: a
#: statistic (e.g. a log10 sensitivity of -0.6, or a raw log10 sensitivity
#: above 8) must never coincide with the values plot_model_slices uses to
#: recognise ocean (log10(ocean_value) +- 0.05) or air (log10 > threshold).
_STATS_AIR_RHO = 1.0e30
_STATS_AIR_LOG10_THRESH = 29.0
_STATS_OCEAN_RHO = 1.0e-30

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
            f"with op one of > >= < <=, e.g. 'sens_mean_an > -3.'"
        )
    return m.group(1), m.group(2), float(m.group(3))


def _alpha_from_condition(
    arr, op: str, thr: float, mode: str, width: float
) -> np.ndarray:
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
        raise ValueError(
            f"MOD_ALPHA_MODE must be 'blank' or 'fade', got {mode!r}"
        )
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
        template=template,
        model=v,
        model_file=path,
        ocean=MOD_OCEAN,
        air_rho=1.0,
        ocean_rho=1.0,
        out=OUT,
    )


_SENS_MAP = {}  # cached once: nreg, free_idx, chosen mapping (see below)


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


_ROI_MASK = {}  # cached once: free-parameter boolean mask inside the ROI


def _roi_free_mask(block_file, n_free):
    """Boolean mask (n_free,), True where a free parameter's volume-weighted
    centroid falls inside the current ROI (MOD_XLIM/MOD_YLIM/MOD_ZLIM, km,
    already resolved -- including any MOD_ROI_AUTO override -- by the time
    this is called from the MOD_STATS loop).

    Reuses the same region-index / free_idx convention as _sens_to_free()
    (populating _SENS_MAP from ``block_file`` if it is not already cached)
    plus fem.build_region_geometry() for volume-weighted free-region
    centroids, then tests each centroid's [x, y, z] directly against the
    resolved (metres) MOD_XLIM/MOD_YLIM/MOD_ZLIM box -- an axis-aligned
    box, so no rotation handling (fem.brick_mask()) is needed. Cached
    after the first call (same mesh/ROI for every panel in one run).
    Returns None (with a one-time printed warning) if the mesh or region
    geometry cannot be read -- callers should then skip the ROI
    diagnostic for this run rather than fail it.
    """
    if "mask" in _ROI_MASK:
        return _ROI_MASK["mask"]
    try:
        if not _SENS_MAP:
            st = fem._read_resistivity_block_struct(
                block_file, ocean=MOD_OCEAN, out=False
            )
            _SENS_MAP["nreg"] = int(st["nreg"])
            _SENS_MAP["free_idx"] = np.asarray(st["free_idx"], dtype=int)
            _SENS_MAP["elem_region"] = np.asarray(st["elem_region"], dtype=int)
        elif "elem_region" not in _SENS_MAP:
            st = fem._read_resistivity_block_struct(
                block_file, ocean=MOD_OCEAN, out=False
            )
            _SENS_MAP["elem_region"] = np.asarray(st["elem_region"], dtype=int)
        free_idx = _SENS_MAP["free_idx"]
        if free_idx.size != n_free:
            print(
                f"  ROI: free_idx size {free_idx.size} != n_free {n_free} "
                f"— skipping ROI diagnostic."
            )
            _ROI_MASK["mask"] = None
            return None
        nodes, conn = fem.read_femtic_mesh(MOD_MESH)
        region_ctr, _ = fem.build_region_geometry(
            nodes, conn, _SENS_MAP["elem_region"], free_idx
        )
        _xlim_m = _lim_km_to_m(MOD_XLIM)
        _ylim_m = _lim_km_to_m(MOD_YLIM)
        _zlim_m = _lim_km_to_m(MOD_ZLIM)
        mask = np.ones(n_free, dtype=bool)
        if _xlim_m is not None:
            mask &= (region_ctr[:, 0] >= min(_xlim_m)) & (
                region_ctr[:, 0] <= max(_xlim_m)
            )
        if _ylim_m is not None:
            mask &= (region_ctr[:, 1] >= min(_ylim_m)) & (
                region_ctr[:, 1] <= max(_ylim_m)
            )
        if _zlim_m is not None:
            mask &= (region_ctr[:, 2] >= min(_zlim_m)) & (
                region_ctr[:, 2] <= max(_zlim_m)
            )
        _ROI_MASK["mask"] = mask
        print(
            f"  ROI diagnostic: {int(mask.sum())} of {n_free} free "
            f"parameters inside MOD_XLIM/MOD_YLIM/MOD_ZLIM."
        )
        return mask
    except Exception as e:
        print(
            f"  ROI diagnostic: could not resolve free-parameter geometry "
            f"({type(e).__name__}: {e}) — skipping."
        )
        _ROI_MASK["mask"] = None
        return None


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
    _e = MOD_UTM_ORIGIN_E
    _n = MOD_UTM_ORIGIN_N
    _lat = MOD_UTM_ORIGIN_LAT
    _lon = MOD_UTM_ORIGIN_LON
    _zone, _north = None, None

    if (
        MOD_ORIGIN_METHOD is not None
        and MOD_SITE_DAT
        and os.path.isfile(MOD_SITE_DAT)
    ):
        _sdat = fem.read_site_dat(MOD_SITE_DAT)
        if _sdat:
            _Es = np.array([d["easting"] for d in _sdat])
            _Ns = np.array([d["northing"] for d in _sdat])
            if MOD_ORIGIN_METHOD == "box":
                _e = 0.5 * (_Es.min() + _Es.max())
                _n = 0.5 * (_Ns.min() + _Ns.max())
            elif MOD_ORIGIN_METHOD == "average":
                _e = float(_Es.mean())
                _n = float(_Ns.mean())
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            _zone, _north = utl.utm_zone_from_latlon(
                float(_lats.mean()),
                float(_lons.mean()),
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
            sx, sy = fem.utm_to_model(row["easting"], row["northing"], _e, _n)
            site_xys.append((row["name"], sx, sy, float(row.get("elev", 0.0))))
    elif _need_sites and MOD_SITE_NUMBER is not None:
        _obs_file = ENSEMBLE_DIR + "templates/observe.dat"
        _site_nums = (
            MOD_SITE_NUMBER
            if isinstance(MOD_SITE_NUMBER, (list, tuple))
            else [MOD_SITE_NUMBER]
        )
        for _sn in _site_nums:
            sx, sy = fem.read_site_position(_obs_file, _sn)
            site_xys.append((_sn, sx, sy, 0.0))
        obs_coords_only = True

    return _e, _n, _lat, _lon, _zone, _north, site_xys, obs_coords_only


# ---------------------------------------------------------------------------
# Reference model (MOD_REF_FILE)
# ---------------------------------------------------------------------------
_REF_STATE = {"checked": False, "ok": False, "struct": None}


def _ref_active() -> bool:
    """True if MOD_REF_FILE is set and exists (checked, reported once)."""
    if not _REF_STATE["checked"]:
        _REF_STATE["checked"] = True
        if MOD_REF_FILE is None:
            pass
        elif not os.path.isfile(MOD_REF_FILE):
            print(f"  MOD_REF_FILE not found: {MOD_REF_FILE} - reference off.")
        else:
            _REF_STATE["ok"] = True
    return bool(_REF_STATE["ok"])


def _ref_disable(why: str) -> None:
    """Switch the reference off for the rest of the run, with a message."""
    print(f"  reference model off: {why}")
    _REF_STATE["ok"] = False


def _block_nelem(path) -> int:
    """Element count from the header line of a resistivity block file."""
    with open(path, "r", encoding="utf-8", errors="replace") as _f:
        return int(_f.readline().split()[0])


def _ref_struct():
    """Region structure of the reference block file (read once); None = off."""
    if not _ref_active():
        return None
    if _REF_STATE["struct"] is None:
        _REF_STATE["struct"] = fem._read_resistivity_block_struct(
            MOD_REF_FILE, ocean=MOD_OCEAN, out=False
        )
    return _REF_STATE["struct"]


def _ref_levels() -> list:
    """Contour levels (log10 Ohm.m) of the reference lines in the slices.

    MOD_REF_LEVELS if given, else the integer decades strictly inside
    MOD_CLIM, else inside the range of the reference model's own free
    regions. The same levels serve every panel: the reference is always
    contoured in its own units, whatever a MOD_STATS panel shows.
    """
    if MOD_REF_LEVELS is not None:
        return [float(v) for v in np.atleast_1d(MOD_REF_LEVELS)]
    if MOD_CLIM is not None:
        lo, hi = float(min(MOD_CLIM)), float(max(MOD_CLIM))
    else:
        st = _ref_struct()
        rho = np.asarray(st["region_rho"], dtype=float)
        keep = np.isfinite(rho) & (rho > 0.0)
        keep[0] = False  # air
        if rho.size > 1 and st["meta"]["ocean_present"]:
            keep[1] = False  # ocean
        lg = np.log10(rho[keep])
        lg = lg[lg <= 8.0]
        if lg.size == 0:
            return []
        lo, hi = float(lg.min()), float(lg.max())
    lev = np.arange(math.floor(lo) + 1, math.ceil(hi), dtype=float)
    if lev.size == 0:
        lev = np.array([0.5 * (lo + hi)])
    return [float(v) for v in lev]


def _ref_slice_kwargs(block_file: str) -> dict:
    """plot_model_slices keywords for the reference lines ({} = no reference).

    The reference must have the element count of the plotted block file
    (same mesh); otherwise it is switched off with a message instead of
    stopping the run.
    """
    if not (MOD_REF_SLICES and _ref_active()):
        return {}
    try:
        n_ref, n_blk = _block_nelem(MOD_REF_FILE), _block_nelem(block_file)
    except (OSError, ValueError, IndexError) as _err:
        _ref_disable(f"cannot read the block-file headers ({_err}).")
        return {}
    if n_ref != n_blk:
        _ref_disable(
            f"MOD_REF_FILE has {n_ref} elements but the model has {n_blk} - "
            f"it must live on the same mesh as MOD_MESH."
        )
        return {}
    return dict(
        reference_file=MOD_REF_FILE,
        reference_levels=_ref_levels(),
        reference_style=dict(MOD_REF_STYLE or {}),
        reference_label=MOD_REF_LABEL,
        reference_ocean_value=MOD_OCEAN_RHO,
        reference_air_log10_thresh=8.0,
    )


def _plot_slice(
    block_file: str,
    pdf_file: str,
    utm_e,
    utm_n,
    utm_lat,
    utm_lon,
    utm_zone,
    utm_north,
    site_xys: list,
    obs_coords_only: bool = False,
    clim="model",
    cmap=None,
    cbar_label=None,
    alpha_file=None,
    stats_panel: bool = False,
    figure_title=None,
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
    figure_title : str | None
        Figure title passed to plot_model_slices; None = block-file name.

    Module settings used besides the MOD_* layout ones: MOD_REF_FILE /
    MOD_REF_SLICES / MOD_REF_LEVELS / MOD_REF_STYLE / MOD_REF_LABEL (reference
    iso-lines on every panel, stats panels included; the reference is read
    from its own block file and always contoured in its own resistivity) and
    MOD_VERT_EXAG (vertical exaggeration of the section panels).
    """
    if fviz is None:
        print("  plot_slice: femtic_viz not available — skipping.")
        return

    _ref_kw = _ref_slice_kwargs(block_file)
    _clim = MOD_CLIM if (isinstance(clim, str) and clim == "model") else clim
    _cmap = MOD_CMAP if cmap is None else cmap
    _ocean_value = _STATS_OCEAN_RHO if stats_panel else MOD_OCEAN_RHO
    _air_thresh = _STATS_AIR_LOG10_THRESH if stats_panel else 8.0

    # "plane" entries (vertical sections) are converted here (km -> m, no
    # CRS tagging); all other kinds go through fem.resolve_slice_positions
    # exactly as before. Original MOD_SLICES order is preserved.
    _others = [sp for sp in MOD_SLICES if sp.get("kind") != "plane"]
    _others_res = iter(
        fem.resolve_slice_positions(
            _slices_km_to_m(_others),
            utm_zone,
            utm_north,
            utm_e,
            utm_n,
            utm_lat,
            utm_lon,
            verbose=OUT,
        )
        if _others
        else []
    )
    _slices_resolved = [
        _plane_spec_km_to_m(sp) if sp.get("kind") == "plane" else next(_others_res)
        for sp in MOD_SLICES
    ]
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
            model_file=block_file,
            mesh_file=MOD_MESH,
            slices=_slices_resolved,
            cmap=_cmap,
            clim=_clim,
            cbar_label=cbar_label,
            xlim=_lim_km_to_m(MOD_XLIM),
            ylim=_lim_km_to_m(MOD_YLIM),
            zlim=_lim_km_to_m(MOD_ZLIM),
            ocean_color=MOD_OCEAN_COLOR,
            ocean_value=_ocean_value,
            air_log10_thresh=_air_thresh,
            air_color=MOD_AIR_COLOR,
            air_bgcolor=MOD_AIR_BGCOLOR,
            site_xys=site_xys,
            obs_coords_only=obs_coords_only,
            sites_in_maps=MOD_PLOT_SITES_MAPS,
            sites_in_slices=MOD_PLOT_SITES_SLICES,
            site_marker=MOD_SITE_MARKER,
            site_marker_slices=MOD_SITE_MARKER_SLICES,
            map_markers=MOD_MAP_MARKERS,
            show_model_centre=MOD_SHOW_MODEL_CENTRE,
            projection_dist=_km_to_m(MOD_PROJECTION_DIST),
            display_coords=MOD_DISPLAY_COORDS,
            utm_origin_e=utm_e,
            utm_origin_n=utm_n,
            utm_zone=utm_zone,
            utm_northern=utm_north,
            utm_to_latlon_fn=utl.utm_to_latlon_zn,
            latlon_to_model_fn=fem.latlon_to_model,
            depth_km=MOD_DEPTH_KM,
            horiz_km=MOD_HORIZ_KM,
            equal_aspect=MOD_EQUAL_ASPECT,
            vert_exag=MOD_VERT_EXAG,
            panel_height=MOD_PANEL_HEIGHT / 2.54,
            panel_width=(
                MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None
            ),
            figsize=(
                [v / 2.54 for v in MOD_FIGSIZE]
                if MOD_FIGSIZE is not None
                else None
            ),
            nrows=MOD_NROWS,
            ncols=MOD_NCOLS,
            tick_fontsize=MOD_TICK_FONTSIZE,
            label_fontsize=MOD_LABEL_FONTSIZE,
            figure_title_fontsize=MOD_TITLE_FONTSIZE,
            figure_title=figure_title,
            tick_decimals=MOD_TICK_DECIMALS,
            alpha_file=alpha_file,
            alpha_mode="direct",
            **_ref_kw,
            plot_file=_fmt_file,
            dpi=MOD_DPI,
            show=_show_this,
            out=OUT,
        )
        if OUT:
            print(f"  saved → {_fmt_file}")


# ---------------------------------------------------------------------------
# Borehole profiles (MOD_BH)
# ---------------------------------------------------------------------------
_BH_CACHE = {}  # mesh arrays, built once (first borehole) and reused


def _bh_mesh():
    """Mesh arrays for the borehole search, read once.

    Uses fem.read_femtic_mesh(), the package's only mesh reader, with its
    default axis handling (MOD_MESH_SWAP_XY): nodes[:, 0] = easting,
    nodes[:, 1] = northing, nodes[:, 2] = z-down (model-local metres).
    Per-element bounding boxes are cached for the lateral pre-filter.
    """
    if "verts" not in _BH_CACHE:
        nodes, conn = fem.read_femtic_mesh(MOD_MESH)
        verts = nodes[conn]  # (n_elem, 4, 3)
        _BH_CACHE["n_elem"] = int(conn.shape[0])
        _BH_CACHE["verts"] = verts
        for _i, _ax in enumerate("xyz"):
            _BH_CACHE[_ax + "min"] = verts[:, :, _i].min(axis=1)
            _BH_CACHE[_ax + "max"] = verts[:, :, _i].max(axis=1)
    return _BH_CACHE


def _bh_column(x_m, y_m):
    """Indices of the elements whose (x, y) bounding box holds the point."""
    c = _bh_mesh()
    return np.where(
        (c["xmin"] <= x_m)
        & (x_m <= c["xmax"])
        & (c["ymin"] <= y_m)
        & (y_m <= c["ymax"])
    )[0]


def _bh_locate(x_m, y_m, depths_m, cand=None):
    """Index of the tetrahedron containing (x_m, y_m, z) for each depth.

    Exact barycentric point-in-tetrahedron test (same tolerance as
    fem.extract_borehole_log), vectorised over the candidate elements of the
    column. Returns an int array, -1 where no element contains the point
    (outside the mesh). This is the expensive part of a borehole and is done
    once; every statistic and every member then reads its values from the
    returned element indices (see _bh_values()).
    """
    c = _bh_mesh()
    depths_m = np.asarray(depths_m, dtype=float)
    elem = np.full(depths_m.size, -1, dtype=np.int64)
    if cand is None:
        cand = _bh_column(x_m, y_m)
    if cand.size == 0:
        return elem
    V = c["verts"][cand]  # (nc, 4, 3)
    v0 = V[:, 0, :]
    T = np.transpose(V[:, 1:, :] - v0[:, None, :], (0, 2, 1))  # edge vectors
    det = np.linalg.det(T)
    ok = np.isfinite(det) & (np.abs(det) > 0.0)
    cand, v0, T = cand[ok], v0[ok], T[ok]
    if cand.size == 0:
        return elem
    Tinv = np.linalg.inv(T)
    zmin, zmax = c["zmin"][cand], c["zmax"][cand]
    tol = 1.0e-10
    for i, z in enumerate(depths_m):
        sel = np.where((zmin <= z) & (z <= zmax))[0]
        if sel.size == 0:
            continue
        lam = np.einsum(
            "nij,nj->ni", Tinv[sel], np.array([x_m, y_m, z]) - v0[sel]
        )
        inside = (lam.min(axis=1) >= -tol) & (lam.sum(axis=1) <= 1.0 + tol)
        hit = np.where(inside)[0]
        if hit.size:
            elem[i] = cand[sel[hit[0]]]
    return elem


def _bh_surface_m(x_m, y_m, dz_m, z_bot_m, elem_region):
    """z (m, z-down) of the top of the first non-air element at (x_m, y_m).

    Coarse scan from the top of the column (air included) downwards, then a
    refinement of the last coarse step at dz/5. Returns None if the column
    holds no non-air element above z_bot_m.
    """
    c = _bh_mesh()
    cand = _bh_column(x_m, y_m)
    if cand.size == 0:
        return None
    z0 = float(c["zmin"][cand].min())

    def _first_solid(depths):
        el = _bh_locate(x_m, y_m, depths, cand=cand)
        solid = (el >= 0) & (elem_region[np.maximum(el, 0)] != 0)
        idx = np.where(solid)[0]
        return int(idx[0]) if idx.size else None

    step = max(float(dz_m), 250.0)
    coarse = np.arange(z0, z_bot_m + 0.5 * step, step)
    j = _first_solid(coarse)
    if j is None:
        return None
    if j == 0:
        return float(coarse[0])
    fine = np.arange(coarse[j - 1], coarse[j] + 1e-9, max(float(dz_m) / 5.0, 1.0))
    k = _first_solid(fine)
    return float(fine[k]) if k is not None else float(coarse[j])


def _bh_resolve_xy(spec, utm_e, utm_n, zone, north):
    """(x_m, y_m, lat, lon) of a MOD_BH entry; lat/lon None if unknown.

    Plain floats are model-local km; (value, "utm") / (value, "latlon")
    tuples as for MOD_SLICES. x and y must carry the same CRS tag.
    """
    xv, xc = fem.parse_pos_crs(spec["x"])
    yv, yc = fem.parse_pos_crs(spec["y"])
    if xc != yc:
        raise ValueError(
            f"MOD_BH '{spec.get('name', '?')}': x CRS {xc!r} and y CRS "
            f"{yc!r} must match."
        )
    lat = lon = None
    if xc == "model":
        x_m, y_m = float(xv) * 1000.0, float(yv) * 1000.0
        if utm_e is not None:
            try:
                lat, lon = utl.utm_to_latlon_zn(
                    utm_e + x_m, utm_n + y_m, zone, north
                )
            except Exception:
                lat = lon = None
        return x_m, y_m, lat, lon
    if utm_e is None:
        raise ValueError(
            f"MOD_BH '{spec.get('name', '?')}': CRS {xc!r} needs the UTM "
            f"origin (MOD_SITE_DAT + MOD_ORIGIN_METHOD, or "
            f"MOD_UTM_ORIGIN_*)."
        )
    if xc == "utm":
        x_m, y_m = fem.utm_to_model(float(xv), float(yv), utm_e, utm_n)
        try:
            lat, lon = utl.utm_to_latlon_zn(float(xv), float(yv), zone, north)
        except Exception:
            lat = lon = None
        return x_m, y_m, lat, lon
    if xc == "latlon":
        lon, lat = float(xv), float(yv)
        x_m, y_m = fem.latlon_to_model(lat, lon, zone, north, utm_e, utm_n)
        return x_m, y_m, lat, lon
    raise ValueError(
        f"MOD_BH '{spec.get('name', '?')}': unknown CRS {xc!r}; use plain "
        f"km, 'utm' or 'latlon'."
    )


def _bh_pos_str(x_m, y_m, lat, lon, utm_e, utm_n):
    """Short position text for the panel title, following MOD_DISPLAY_COORDS."""
    if MOD_DISPLAY_COORDS == "latlon" and lat is not None:
        return f"{lat:.4f}, {lon:.4f} (lat, lon)"
    if MOD_DISPLAY_COORDS == "utm" and utm_e is not None:
        return f"E {(utm_e + x_m) / 1000.0:.2f}, N {(utm_n + y_m) / 1000.0:.2f} km"
    return f"x {x_m / 1000.0:.2f}, y {y_m / 1000.0:.2f} km"


#: Quantile levels (percent) of the "sigmaN" shortcuts: P(Phi(-N)), P(Phi(N)).
_BH_SIGMA_Q = {
    "sigma1_lo": 15.8655,
    "sigma1_hi": 84.1345,
    "sigma2_lo": 2.2750,
    "sigma2_hi": 97.7250,
}


def _bh_expand_what(what, value_keys):
    """Expand a MOD_BH "what" list to an ordered list of draw keys.

    "percentiles" -> one key per PERCENTILES level; "avg", "med", single
    percentile keys, "ensemble", "heatmap", "ref" and "best" pass through.
    Unknown names are
    reported and dropped.
    """
    pct_keys = [
        "p" + f"{_p:g}".replace(".", "_")
        for _p in PERCENTILES
        if "p" + f"{_p:g}".replace(".", "_") in value_keys
    ]
    keys = []
    for w in what:
        if w == "percentiles":
            new = pct_keys
        elif w in ("sigma1", "sigma2"):
            new = [w + "_lo", w + "_hi"]
        elif w in ("ensemble", "best", "ref", "heatmap") or w in value_keys:
            new = [w]
        else:
            print(
                f"  MOD_BH: '{w}' is not available (use 'avg', 'med', "
                f"'percentiles', a percentile key {pct_keys}, 'sigma1', "
                f"'sigma2', 'ensemble', 'heatmap', 'best' or 'ref') - skipped."
            )
            new = []
        keys += [k for k in new if k not in keys]
    return keys


def _bh_style(key, spec, n_pct_index=None, n_pct=None):
    """Line2D kwargs for draw key ``key`` (MOD_BH_STYLE < entry "style")."""
    key = key.split("_")[0] if key.startswith("sigma") else key
    is_pct = key.startswith("p") and key not in ("percentiles",)
    st = {}
    layers = [MOD_BH_STYLE, spec.get("style", {})]
    for layer in layers:
        if is_pct:
            st.update(layer.get("percentiles", {}))
        st.update(layer.get(key, {}))
    cmap = st.pop("cmap", None)
    if is_pct and cmap is not None and "color" not in st:
        import matplotlib

        frac = 0.0 if not n_pct or n_pct < 2 else n_pct_index / (n_pct - 1)
        st["color"] = matplotlib.colormaps[cmap](0.05 + 0.80 * frac)
    return st


def _bh_values(vec, reg_lvl, fixed_log10, free_pos, rho_out=True):
    """Map free-parameter vector(s) to values along a borehole.

    vec : (..., n_free) log10(rho) of the free regions (a statistic vector
        or the whole ens_matrix).
    reg_lvl : region index per depth level (-1 where no element).
    fixed_log10 : (nreg,) log10(rho) of fixed regions (NaN = air).
    free_pos : (nreg,) position of a region in the free vector, -1 = fixed.

    Returns (..., n_lvl) in Ohm.m (rho_out=True) or log10; NaN for air and
    for levels outside the mesh.
    """
    vec = np.asarray(vec, dtype=float)
    n_lvl = reg_lvl.size
    out = np.full(vec.shape[:-1] + (n_lvl,), np.nan)
    inside = reg_lvl >= 0
    fpos = np.where(inside, free_pos[np.maximum(reg_lvl, 0)], -1)
    is_free = inside & (fpos >= 0)
    is_fixed = inside & (fpos < 0)
    if is_free.any():
        out[..., is_free] = vec[..., fpos[is_free]]
    if is_fixed.any():
        out[..., is_fixed] = fixed_log10[reg_lvl[is_fixed]]
    return 10.0**out if rho_out else out


def _bh_glob_xlim():
    """Default resistivity range [min, max] (Ohm.m) of the borehole panels.

    MOD_BH_XLIM, else 10**MOD_CLIM, else None (auto).
    """
    if MOD_BH_XLIM is not None:
        return list(MOD_BH_XLIM)
    if MOD_CLIM is not None:
        return [10.0 ** MOD_CLIM[0], 10.0 ** MOD_CLIM[1]]
    return None


def _bh_edges(spec, ens_rho):
    """log10(rho) bin edges of the heat map of one borehole, or None.

    ``spec["heat_bins"]`` (default MOD_BH_HEAT_BINS): a list of edges is used
    as given; an int N gives N equal bins across the panel range (entry
    "xlim", else MOD_BH_XLIM / MOD_CLIM), or across the range of the finite
    member values (ens_rho, Ohm.m) when no range is set. None if there is
    nothing to bin.
    """
    bins = spec.get("heat_bins", MOD_BH_HEAT_BINS)
    if np.ndim(bins) > 0:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(
                f"MOD_BH '{spec.get('name', '?')}': heat_bins must be an int "
                f"or an increasing list of >= 2 log10(rho) edges."
            )
        return edges
    nb = int(bins)
    if nb < 1:
        raise ValueError(
            f"MOD_BH '{spec.get('name', '?')}': heat_bins must be >= 1."
        )
    xl = spec.get("xlim", _bh_glob_xlim())
    if xl is not None:
        lo, hi = np.log10(float(xl[0])), np.log10(float(xl[1]))
    else:
        ok = np.isfinite(ens_rho) & (ens_rho > 0)
        if not ok.any():
            return None
        lg = np.log10(ens_rho[ok])
        lo, hi = float(lg.min()), float(lg.max())
        if hi - lo < 1.0e-6:
            lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, nb + 1)


def _bh_heatmap(ens_rho, edges):
    """Per-depth histogram of the ensemble along a borehole.

    ens_rho : (n_mem, n_lvl) resistivity in Ohm.m, NaN for air / outside.
    edges   : (n_bins + 1,) increasing log10(rho) bin edges.

    Returns (n_lvl, n_bins): fraction of the valid members of each level
    falling into each bin. Members outside the edges are not binned but
    still count in the denominator, so a row sums to less than 1 where part
    of the ensemble lies outside the range (the last bin includes its upper
    edge). Levels without a valid member are NaN.
    """
    ens_rho = np.asarray(ens_rho, dtype=float)
    n_mem, n_lvl = ens_rho.shape
    nb = len(edges) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(ens_rho)
    valid = np.isfinite(lg)
    inside = valid & (lg >= edges[0]) & (lg <= edges[-1])
    idx = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, nb - 1)
    flat = idx + nb * np.arange(n_lvl)[None, :]
    counts = np.bincount(flat[inside], minlength=n_lvl * nb).reshape(n_lvl, nb)
    n_val = valid.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = counts / n_val[:, None]
    frac[n_val == 0, :] = np.nan
    return frac


def _bh_heat_display(frac):
    """Heat map values as shown (MOD_BH_HEAT_NORM) from the fractions."""
    if MOD_BH_HEAT_NORM == "row_max":
        import warnings

        with warnings.catch_warnings(), np.errstate(
            invalid="ignore", divide="ignore"
        ):
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN levels
            return frac / np.nanmax(frac, axis=1, keepdims=True)
    return frac


def _bh_heat_style(spec):
    """(cmap name, pcolormesh kwargs) of the heat map (MOD_BH_STYLE < entry)."""
    st = {}
    for layer in (MOD_BH_STYLE, spec.get("style", {})):
        st.update(layer.get("heatmap", {}))
    cmap = st.pop("cmap", MOD_BH_HEAT_CMAP)
    return cmap, st


def _bh_heat_log_ticks(norm):
    """1-2-5 colour-bar ticks for a log scale of less than 3 decades (the
    default shows only the powers of ten), else None."""
    if np.log10(norm.vmax / norm.vmin) >= 3.0:
        return None
    cand = np.array([m * 10.0**e for e in range(-8, 2) for m in (1.0, 2.0, 5.0)])
    tk = cand[(cand >= norm.vmin * (1 - 1e-9)) & (cand <= norm.vmax * (1 + 1e-9))]
    return tk if tk.size >= 2 else None


def _bh_heat_scale(heats):
    """Colour scale of the heat map of one figure.

    ``heats`` : the displayed arrays (_bh_heat_display) of its panels.
    Returns (norm, extend): a Matplotlib Normalize (MOD_BH_HEAT_SCALE between
    MOD_BH_HEAT_VMIN and MOD_BH_HEAT_VMAX / their automatic values) and the
    colour-bar ``extend`` ("neither" / "min" / "max" / "both").
    """
    from matplotlib import colors

    log = MOD_BH_HEAT_SCALE == "log"
    fin = np.concatenate([h[np.isfinite(h)] for h in heats])
    pos = fin[fin > 0]
    d_max = float(pos.max()) if pos.size else 1.0
    d_min = float(pos.min()) if pos.size else 1.0
    if MOD_BH_HEAT_NORM == "row_max":
        vmax = 1.0
    elif MOD_BH_HEAT_VMAX is not None:
        vmax = float(MOD_BH_HEAT_VMAX)
    elif log or not pos.size:
        vmax = d_max
    else:
        vmax = float(np.percentile(pos, 99.0))
    if MOD_BH_HEAT_VMIN is not None:
        vmin = float(MOD_BH_HEAT_VMIN)
    else:
        vmin = d_min if log else 0.0
    if not vmax > vmin:  # e.g. one value only, or an unlucky explicit pair
        vmax = vmin * 10.0 if log else vmin + 1.0
    # data beyond the scale -> arrows on the colour bar (zeros are blank on a
    # log scale, so they do not count there)
    lo = d_min if log else (float(fin.min()) if fin.size else 0.0)
    ext_min = lo < vmin * (1.0 - 1.0e-9)
    ext_max = d_max > vmax * (1.0 + 1.0e-9)
    extend = {(0, 0): "neither", (1, 0): "min", (0, 1): "max", (1, 1): "both"}[
        (int(ext_min), int(ext_max))
    ]
    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif MOD_BH_HEAT_SCALE == "sqrt":
        norm = colors.PowerNorm(0.5, vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return norm, extend


def _bh_depth_edges(depth):
    """Cell edges (n + 1) for the depth levels (cell centres), z-down."""
    d = np.asarray(depth, dtype=float)
    if d.size == 1:
        return np.array([d[0] - 0.5, d[0] + 0.5])
    mid = 0.5 * (d[1:] + d[:-1])
    first = d[0] - 0.5 * (d[1] - d[0])
    last = d[-1] + 0.5 * (d[-1] - d[-2])
    return np.concatenate(([first], mid, [last]))


def _plot_boreholes(profiles, plot_base):
    """One figure with a resistivity-depth panel per borehole profile.

    ``profiles`` : list of dicts built in step (9): name, pos_str, spec,
    depth (m, z-down), lines {key: rho(n_lvl)}, ens (n_mem, n_lvl) or None,
    heat (n_lvl, n_bins) fractions and heat_edges (log10 rho) or None.
    The figure is built once and saved once per MOD_PLOT_FORMAT entry.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(profiles)
    ncols = min(int(MOD_BH_NCOLS), n) if MOD_BH_NCOLS else min(n, 4)
    nrows = int(math.ceil(n / ncols))
    w, h = MOD_BH_PANEL_SIZE
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(w * ncols / 2.54, h * nrows / 2.54),
        squeeze=False,
    )
    axs = axs.ravel()
    scale = 1000.0 if MOD_DEPTH_KM else 1.0
    tick_fs, label_fs = MOD_TICK_FONTSIZE, MOD_LABEL_FONTSIZE

    glob_xlim = _bh_glob_xlim()

    # heat map: one colour scale for the whole figure
    heats = [
        _bh_heat_display(pr["heat"])
        for pr in profiles
        if "heatmap" in pr["order"] and pr.get("heat") is not None
    ]
    heat_norm, heat_extend = _bh_heat_scale(heats) if heats else (None, "neither")
    heat_mesh = None

    for k, (ax, pr) in enumerate(zip(axs, profiles)):
        spec, depth = pr["spec"], pr["depth"] / scale
        pct_keys = [q for q in pr["lines"] if q.startswith("p")]
        handles = []
        # heat map first (bottom layer, zorder 0), whatever its place in "what"
        has_heat = "heatmap" in pr["order"] and pr.get("heat") is not None
        if has_heat:
            hcmap, hkw = _bh_heat_style(spec)
            hkw.setdefault("shading", "flat")
            hval = _bh_heat_display(pr["heat"])
            # NaN (air) is always blank; on a log scale so are the empty bins
            hval = np.ma.masked_where(
                ~(hval > 0) if MOD_BH_HEAT_SCALE == "log" else ~np.isfinite(hval),
                hval,
            )
            mesh = ax.pcolormesh(
                10.0 ** pr["heat_edges"],
                _bh_depth_edges(depth),
                hval,
                cmap=hcmap,
                norm=heat_norm,
                zorder=0,
                **hkw,
            )
            if heat_mesh is None:
                heat_mesh = mesh
        # draw in the order given by "what"
        for key in pr["order"]:
            if key == "heatmap":
                continue
            if key == "ensemble":
                ens = pr["ens"]
                st = _bh_style("ensemble", spec)
                alpha = float(spec.get("alpha", MOD_BH_ALPHA))
                ax.plot(ens.T, depth, alpha=alpha, zorder=1, **st)
                proxy = dict(st)
                handles.append(
                    Line2D(
                        [],
                        [],
                        label=f"ensemble (n={ens.shape[0]})",
                        **{**proxy, "alpha": max(alpha, 0.6)},
                    )
                )
                continue
            if key not in pr["lines"]:
                continue
            if key.startswith("sigma"):
                st = _bh_style(key, spec)
                n_sig = key[5]
                label = (
                    f"{n_sig}-sigma quantiles"
                    if key.endswith("_lo")
                    else "_nolegend_"
                )
                z = 2
            elif key in pct_keys:
                st = _bh_style(key, spec, pct_keys.index(key), len(pct_keys))
                label = "P" + key[1:].replace("_", ".")
                z = 2
            elif key == "ref":
                st = _bh_style(key, spec)
                label = MOD_REF_LABEL if MOD_REF_LABEL else "_nolegend_"
                z = 4  # on top of the ensemble statistics
            else:
                st = _bh_style(key, spec)
                label = {"avg": "mean", "med": "median", "best": "best member"}[
                    key
                ]
                z = 3
            (ln,) = ax.plot(pr["lines"][key], depth, label=label, zorder=z, **st)
            if label != "_nolegend_":
                handles.append(ln)

        ax.set_xscale("log")
        xl = spec.get("xlim", glob_xlim)
        if xl is None and has_heat:  # auto range: the heat map's own range
            xl = [10.0 ** pr["heat_edges"][0], 10.0 ** pr["heat_edges"][-1]]
        if xl is not None:
            ax.set_xlim(xl)
        ax.set_title(f"{pr['name']}\n{pr['pos_str']}", fontsize=tick_fs - 2)
        ax.set_xlabel("resistivity (Ohm.m)", fontsize=label_fs)
        if k % ncols == 0:
            ax.set_ylabel(
                "depth (km)" if MOD_DEPTH_KM else "depth (m)", fontsize=label_fs
            )
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, which="both", axis="x", lw=0.3, alpha=0.5)
        ax.grid(True, which="major", axis="y", lw=0.3, alpha=0.5)
        if handles and MOD_BH_LEGEND_LOC is not None:
            ax.legend(
                handles=handles,
                fontsize=max(tick_fs - 4, 6),
                loc=MOD_BH_LEGEND_LOC,
                framealpha=0.85,
            )

    for ax in axs[n:]:
        ax.set_visible(False)

    # depth range: common (union) or per panel; y axis points down
    ranges = [
        (np.nanmin(pr["depth"]) / scale, np.nanmax(pr["depth"]) / scale)
        for pr in profiles
    ]
    for k, (ax, (lo, hi)) in enumerate(zip(axs, ranges)):
        if MOD_BH_SHARE_DEPTH:
            lo = min(r[0] for r in ranges)
            hi = max(r[1] for r in ranges)
        ax.set_ylim(hi, lo)

    fig.suptitle(
        os.path.basename(os.path.normpath(ENSEMBLE_DIR)) + "  " + ENSEMBLE_PREFIX,
        fontsize=MOD_TITLE_FONTSIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if heat_mesh is not None and MOD_BH_HEAT_CBAR:
        # one shared colour bar, added after tight_layout so that it only
        # takes its space from the panels
        cb = fig.colorbar(
            heat_mesh,
            ax=list(axs[:n]),
            location="right",
            fraction=0.03,
            pad=0.02,
            shrink=0.6,
            extend=heat_extend,
        )
        cb.set_label(
            "fraction of ensemble members per bin"
            if MOD_BH_HEAT_NORM == "fraction"
            else "fraction of the per-depth maximum",
            fontsize=label_fs,
        )
        cb.ax.tick_params(labelsize=tick_fs)
        if MOD_BH_HEAT_SCALE == "log" and _bh_heat_log_ticks(heat_norm) is not None:
            from matplotlib import ticker

            cb.set_ticks(_bh_heat_log_ticks(heat_norm))
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            cb.ax.minorticks_off()
    for _fmt in _MOD_PLOT_FORMATS:
        _f = f"{plot_base}.{_fmt}"
        fig.savefig(_f, dpi=MOD_DPI, bbox_inches="tight")
        if OUT:
            print(f"  saved -> {_f}")
    if _SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _plot_nrms_hist(rows, plot_base):
    """Histogram of per-member best nRMS; accepted/rejected stacked.

    ``rows`` : list of (member, iter, nrms, status) as written to NRMS_CSV.
    Saved once per MOD_PLOT_FORMAT entry.
    """
    import matplotlib.pyplot as plt

    acc = np.array(
        [r[2] for r in rows if r[3].startswith("accepted")], dtype=float
    )
    rej = np.array(
        [r[2] for r in rows if not r[3].startswith("accepted")], dtype=float
    )
    n_fb = sum(1 for r in rows if r[3] == "accepted_fallback")
    allv = np.concatenate((acc, rej))
    if allv.size == 0:
        print("  nRMS histogram: no members - skipped.")
        return
    edges = np.histogram_bin_edges(allv, bins=MOD_NRMS_BINS)
    fig, ax = plt.subplots(figsize=(10 / 2.54 * 1.6, 8 / 2.54 * 1.6))
    ax.hist(
        [acc, rej],
        bins=edges,
        stacked=True,
        color=["tab:blue", "tab:red"],
        edgecolor="k",
        label=[
            f"accepted (n={acc.size}"
            + (f", {n_fb} with fallback iter" if n_fb else "")
            + ")",
            f"rejected (n={rej.size})",
        ],
    )
    ax.axvline(NRMS_MAX, color="k", ls="--", lw=1.2, label=f"NRMS_MAX = {NRMS_MAX}")
    if acc.size:
        ax.axvline(
            acc.min(), color="tab:orange", ls="-", lw=1.5,
            label=f"best = {acc.min():.3f}",
        )
        ax.axvline(
            np.median(acc), color="crimson", ls=":", lw=1.5,
            label=f"median (accepted) = {np.median(acc):.3f}",
        )
    ax.set_xlabel("best nRMS per member", fontsize=MOD_LABEL_FONTSIZE)
    ax.set_ylabel("number of members", fontsize=MOD_LABEL_FONTSIZE)
    ax.tick_params(labelsize=MOD_TICK_FONTSIZE)
    ax.legend(fontsize=MOD_TICK_FONTSIZE - 2)
    fig.suptitle(
        os.path.basename(os.path.normpath(ENSEMBLE_DIR)) + "  " + ENSEMBLE_PREFIX,
        fontsize=MOD_TITLE_FONTSIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for _fmt in _MOD_PLOT_FORMATS:
        _f = f"{plot_base}.{_fmt}"
        fig.savefig(_f, dpi=MOD_DPI, bbox_inches="tight")
        if OUT:
            print(f"  saved -> {_f}")
    if _SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

#: Plot-only state, filled by the PLOT_ONLY loader below and read by the
#: best-member (QC) panel: name of the best member's block file as stored in
#: the results .npz, and whether that panel must be rebuilt from the stored
#: best vector (PLOT_ONLY_TEMPLATE given) instead of the member's own file.
_po_best_name = None
_plot_only_best_member = None
_po_qc_from_vector = False

if PLOT_ONLY:
    # --- (1)-(4b) replaced: read the existing results .npz ------------------
    # Nothing is scanned, recomputed or written (no member directories, no
    # cnv/h5 files, no new .npz); the variables that sections (5)-(9) use are
    # rebuilt from the file, so those sections run exactly as in a full run.
    _po_npz = PLOT_ONLY_NPZ if PLOT_ONLY_NPZ else ENSEMBLE_RESULTS
    if not os.path.isfile(_po_npz):
        sys.exit(f"PLOT_ONLY: results file not found: {_po_npz}")
    print(f"PLOT_ONLY: reading {_po_npz}")
    try:
        _z = np.load(_po_npz, allow_pickle=False)
    except (ValueError, OSError) as _e:
        sys.exit(f"PLOT_ONLY: cannot read {_po_npz}: {_e}")
    _keys = set(_z.files)

    # Key prefix inside the file: normally ENSEMBLE_PREFIX; if the file holds
    # exactly one other ensemble, that one is used (outputs keep P).
    _pk = ENSEMBLE_PREFIX
    if f"{_pk}_ens" not in _keys:
        _cand = sorted(
            k[: -len("_ens")]
            for k in _keys
            if k.endswith("_ens") and f"{k[: -len('_ens')]}_avg" in _keys
        )
        if len(_cand) == 1:
            print(
                f"  PLOT_ONLY: no '{_pk}_ens' in the file; using its only "
                f"ensemble, prefix '{_cand[0]}'."
            )
            _pk = _cand[0]
        else:
            sys.exit(
                f"PLOT_ONLY: no '{_pk}_ens' in {_po_npz}. Ensemble prefixes "
                f"found: {_cand if _cand else 'none'} -- set ENSEMBLE_PREFIX "
                f"to one of them."
            )

    def _po_get(name, recompute):
        """Array <_pk>_<name> from the file, else recompute() from the members."""
        _k = f"{_pk}_{name}"
        if _k in _keys:
            return np.asarray(_z[_k], dtype=float)
        print(f"  PLOT_ONLY: '{_k}' not in the file - recomputed from members.")
        return recompute()

    ens_matrix = np.asarray(_z[f"{_pk}_ens"], dtype=float)  # (n_members, n_free)
    n_members = int(ens_matrix.shape[0])
    if n_members == 0:
        sys.exit("PLOT_ONLY: the file holds no ensemble members. Nothing to do.")

    ens_avg = _po_get("avg", lambda: np.mean(ens_matrix, axis=0))
    ens_var = _po_get("var", lambda: np.var(ens_matrix, axis=0))
    ens_err = _po_get("err", lambda: np.sqrt(np.var(ens_matrix, axis=0)))
    ens_med = _po_get("med", lambda: np.median(ens_matrix, axis=0))
    ens_mad = _po_get(
        "mad",
        lambda: np.median(
            np.abs(ens_matrix - np.median(ens_matrix, axis=0)[np.newaxis, :]),
            axis=0,
        ),
    )

    # Percentile levels as stored (they define the p<level> panel keys).
    if f"{_pk}_prc_levels" in _keys:
        _lv = [float(v) for v in np.asarray(_z[f"{_pk}_prc_levels"]).ravel()]
        if _lv != [float(v) for v in PERCENTILES]:
            print(
                f"  PLOT_ONLY: using the stored percentile levels {_lv} "
                f"instead of PERCENTILES={list(PERCENTILES)}."
            )
        PERCENTILES = _lv
    ens_prc = _po_get(
        "prc", lambda: np.percentile(ens_matrix, PERCENTILES, axis=0)
    )
    if ens_prc.shape[0] != len(PERCENTILES):
        sys.exit(
            f"PLOT_ONLY: '{_pk}_prc' has {ens_prc.shape[0]} rows but there "
            f"are {len(PERCENTILES)} percentile levels - inconsistent file."
        )

    # Optional statistics: whatever the full run stored.
    ens_qdiff = {}
    _pre = f"{_pk}_qdiff_"
    for _k in sorted(_keys):
        if _k.startswith(_pre):
            ens_qdiff[_k[len(_pk) + 1 :]] = np.asarray(_z[_k], dtype=float)
    ens_var_boot = ens_err_boot = ens_var_boot_se = None
    if f"{_pk}_var_boot" in _keys:
        ens_var_boot = np.asarray(_z[f"{_pk}_var_boot"], dtype=float)
        ens_err_boot = (
            np.asarray(_z[f"{_pk}_err_boot"], dtype=float)
            if f"{_pk}_err_boot" in _keys
            else np.sqrt(ens_var_boot)
        )
        if f"{_pk}_var_boot_se" in _keys:
            ens_var_boot_se = np.asarray(_z[f"{_pk}_var_boot_se"], dtype=float)
    sens = {}
    _pre = f"{_pk}_sens_"
    for _k in sorted(_keys):
        if _k.startswith(_pre):
            sens[_k[len(_pre) :]] = np.asarray(_z[_k], dtype=float)
    simrc_coef = simrc_corr = None
    if f"{_pk}_simrc_coef" in _keys and f"{_pk}_simrc_corr" in _keys:
        simrc_coef = np.asarray(_z[f"{_pk}_simrc_coef"], dtype=float)
        simrc_corr = np.asarray(_z[f"{_pk}_simrc_corr"], dtype=float)
    # Only read by the failure message of MOD_ALPHA_SOURCE (which says
    # "not stored in the results file" in this mode); defined for safety.
    sens_count = simrc_count = 0
    sens_fail = []
    simrc_fail = []

    # Member list: [block file, best iteration, nRMS] per row of ens_matrix
    # (stored as a string array; mixed-type lists are saved that way).
    if f"{_pk}_model_list" not in _keys:
        sys.exit(f"PLOT_ONLY: '{_pk}_model_list' missing from {_po_npz}.")
    model_list = [
        [str(_r[0]), int(float(_r[1])), float(_r[2])]
        for _r in np.asarray(_z[f"{_pk}_model_list"]).reshape(-1, 3)
    ]
    if len(model_list) != n_members:
        sys.exit(
            f"PLOT_ONLY: model_list has {len(model_list)} entries but the "
            f"ensemble matrix has {n_members} rows - inconsistent file."
        )
    _bi = min(range(len(model_list)), key=lambda i: model_list[i][2])
    _po_best_name = os.path.basename(
        str(_z[f"{_pk}_best_file"]) if f"{_pk}_best_file" in _keys else model_list[_bi][0]
    )
    _plot_only_best_member = (
        str(_z[f"{_pk}_best_member"])
        if f"{_pk}_best_member" in _keys
        else os.path.basename(os.path.dirname(model_list[_bi][0]))
    )
    if f"{_pk}_best" in _keys:
        _sb = np.asarray(_z[f"{_pk}_best"], dtype=float)
        if _sb.shape != ens_matrix[_bi].shape or not np.allclose(_sb, ens_matrix[_bi]):
            print(
                "  PLOT_ONLY: WARNING - stored best vector differs from the "
                "lowest-nRMS row of the ensemble matrix."
            )

    # Template block file (header, region structure, flag columns): the best
    # member's own file, or PLOT_ONLY_TEMPLATE when that is gone / elsewhere.
    # model_list's best entry is pointed at it, so (5)-(9) use it unchanged.
    _tpl = PLOT_ONLY_TEMPLATE if PLOT_ONLY_TEMPLATE else model_list[_bi][0]
    if MOD_QC or MOD_STATS or MOD_BH:
        if not os.path.isfile(_tpl):
            sys.exit(
                f"PLOT_ONLY: template block file not found: {_tpl}\n"
                f"  The stored best-member file is {model_list[_bi][0]}.\n"
                f"  Set PLOT_ONLY_TEMPLATE to any resistivity block file of "
                f"this ensemble (same mesh and regions) -- the panels are "
                f"drawn from the stored vectors, only its structure is used."
            )
        _tst = fem._read_resistivity_block_struct(_tpl, ocean=MOD_OCEAN, out=False)
        if int(np.asarray(_tst["free_idx"]).size) != ens_matrix.shape[1]:
            sys.exit(
                f"PLOT_ONLY: template {_tpl} has "
                f"{int(np.asarray(_tst['free_idx']).size)} free regions but the "
                f"ensemble has {ens_matrix.shape[1]} - it is not a block file "
                f"of this ensemble."
            )
    _po_qc_from_vector = os.path.abspath(_tpl) != os.path.abspath(model_list[_bi][0])
    model_list[_bi][0] = _tpl

    P = ENSEMBLE_PREFIX
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(
        f"PLOT_ONLY: {n_members} members, {ens_matrix.shape[1]} free "
        f"parameters, best nRMS {model_list[_bi][2]:.4f} "
        f"({_plot_only_best_member}, iter {model_list[_bi][1]})."
    )
    _have = (
        ["avg", "var", "err", "med", "mad"]
        + ["p" + f"{_p:g}".replace(".", "_") for _p in PERCENTILES]
        + list(ens_qdiff)
        + (["var_boot", "err_boot"] if ens_var_boot is not None else [])
        + [f"sens_{_k}" for _k in sens]
        + (["simrc_coef", "simrc_corr"] if simrc_coef is not None else [])
    )
    print(f"  statistics in the file : {_have}")
    print(f"  requested (MOD_STATS_WHAT): {list(MOD_STATS_WHAT)}")

    # nRMS histogram from the CSV written by the full run (the .npz holds
    # accepted members only, the histogram also shows the rejected ones).
    if MOD_NRMS_HIST:
        import csv

        nrms_table = []
        _po_csv = NRMS_CSV
        _po_csv_alt = os.path.join(os.path.dirname(_po_npz), f"{_pk}_nrms.csv")
        if not os.path.isfile(_po_csv):
            _po_csv = _po_csv_alt
        if os.path.isfile(_po_csv):
            with open(_po_csv, newline="", encoding="utf-8") as _fh:
                for _r in csv.DictReader(_fh):
                    nrms_table.append(
                        (_r["member"], int(_r["best_iter"]), float(_r["nrms"]), _r["status"])
                    )
            _plot_nrms_hist(nrms_table, MOD_NRMS_HIST_FILE)
        else:
            print(
                f"  PLOT_ONLY: nRMS table not found ({NRMS_CSV}, "
                f"{_po_csv_alt}) - nRMS histogram skipped."
            )
    _z.close()
else:
    # --- (1) Scan ensemble directories ----------------------------------------
    # get_filelist() matches on name only (fnmatch over os.listdir), so it can
    # return non-directory matches (stray files, archives, etc.) alongside the
    # actual ensemble-member sub-directories. Filter to directories immediately,
    # before counting/looping, so downstream logic never has to think about
    # non-directory entries.
    dir_list = utl.get_filelist(
        searchstr=[ENSEMBLE_NAME + "*"],
        searchpath=ENSEMBLE_DIR,
        fullpath=True,
    )
    dir_list = [d for d in dir_list if os.path.isdir(d)]
    print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

    model_list = []  # list of [block_file, n_iter, nRMS]
    nrms_table = []  # (member, best_iter, nRMS, status) for NRMS_CSV / histogram
    model_count = 0
    ens_matrix = None  # will become (n_members, n_free) float64

    sens_matrix = None  # will become (n_sens_members, n_free) float64
    sens_count = 0  # accepted members whose results_iterX.h5 was found
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
        cnv_file = str(fem.resolve_cnv_path(d, CNV_FILE))
        if not os.path.isfile(cnv_file):
            print(f"    {os.path.basename(cnv_file)} not found — skipped.")
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
            continue

        mod_file = os.path.join(d, f"resistivity_block_iter{numit}.dat")
        _status = "accepted"
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
                    numit, nrms, _status = _i2, _n2, "accepted_fallback"
                    mod_file = os.path.join(
                        d, f"resistivity_block_iter{numit}.dat"
                    )
                else:
                    print(
                        f"    best available iteration iter{_i2} has "
                        f"nRMS={_n2:.4f} > NRMS_MAX={NRMS_MAX}."
                    )
        if not os.path.isfile(mod_file):
            print(f"    {mod_file} not found — skipped.")
            nrms_table.append(
                (os.path.basename(os.path.normpath(d)), numit, nrms, "no_model_file")
            )
            continue

        nrms_table.append(
            (os.path.basename(os.path.normpath(d)), numit, nrms, _status)
        )
        print(f"    iter={numit}  nRMS={nrms:.4f}  {mod_file}")
        model_list.append([mod_file, numit, nrms])

        log_m = fem.read_model(model_file=mod_file, model_trans="log10", out=OUT)

        if ens_matrix is None:
            ens_matrix = log_m[np.newaxis, :]  # (1, n_free)
        else:
            ens_matrix = np.vstack((ens_matrix, log_m))  # (k, n_free)

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

    n_members = model_count
    print(f"\nConverged members: {n_members}")


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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(ENSEMBLE_RESULTS, **ens_dict)
    print(f"\nResults saved → {ENSEMBLE_RESULTS}")

    # --- (4b) nRMS table (CSV) and histogram ----------------------------------
    import csv

    with open(NRMS_CSV, "w", newline="", encoding="utf-8") as _fh:
        _w = csv.writer(_fh)
        _w.writerow(["member", "best_iter", "nrms", "status"])
        for _r in sorted(nrms_table, key=lambda r: r[2]):
            _w.writerow([_r[0], _r[1], f"{_r[2]:.6f}", _r[3]])
    print(f"nRMS table saved -> {NRMS_CSV}  ({len(nrms_table)} members)")
    if MOD_NRMS_HIST:
        _plot_nrms_hist(nrms_table, MOD_NRMS_HIST_FILE)

# --- (5) Resolve UTM origin and sites (needed for any plot) ---------------
if MOD_QC or MOD_STATS or MOD_BH:
    (
        utm_e,
        utm_n,
        utm_lat,
        utm_lon,
        utm_zone,
        utm_north,
        site_xys,
        obs_coords_only,
    ) = _resolve_origin_and_sites()

    # --- Region of interest: override MOD_XLIM/YLIM/ZLIM from site bbox ---
    if MOD_ROI_AUTO and site_xys:
        _sx = np.array([s[1] for s in site_xys])  # model-local metres
        _sy = np.array([s[2] for s in site_xys])  # model-local metres
        MOD_XLIM = [
            float(_sx.min() / 1000.0 - MOD_ROI_PAD_XY),
            float(_sx.max() / 1000.0 + MOD_ROI_PAD_XY),
        ]
        MOD_YLIM = [
            float(_sy.min() / 1000.0 - MOD_ROI_PAD_XY),
            float(_sy.max() / 1000.0 + MOD_ROI_PAD_XY),
        ]
        if MOD_ROI_ZLIM is not None:
            MOD_ZLIM = list(MOD_ROI_ZLIM)
        print(
            f"\nROI (from {len(site_xys)} sites, pad={MOD_ROI_PAD_XY:.2f} km):"
        )
        print(f"  MOD_XLIM = {MOD_XLIM} km")
        print(f"  MOD_YLIM = {MOD_YLIM} km")
        print(f"  MOD_ZLIM = {MOD_ZLIM} km")
    elif MOD_ROI_AUTO:
        print(
            "\nROI: MOD_ROI_AUTO=True but no sites available — "
            "using literal MOD_XLIM/MOD_YLIM/MOD_ZLIM instead."
        )

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
if ens_var_boot is not None:  # = BOOTSTRAP_VAR in a full run
    _stat_map["var_boot"] = (ens_var_boot, "bootstrap variance")
    _stat_map["err_boot"] = (ens_err_boot, "bootstrap error (std)")
for _k, _v in sens.items():
    _stat_map[f"sens_{_k}"] = (
        _v,
        f"sensitivity {_k} ({'linear' if _k.startswith('cv') else 'log10'})",
    )
if simrc_corr is not None:
    _stat_map["simrc_coef"] = (
        simrc_coef,
        "log10 SimRC cumulative |regression coeff.|",
    )
    _stat_map["simrc_corr"] = (
        simrc_corr,
        "log10 SimRC cumulative |correlation|",
    )


def _panel_style(key):
    """(cmap, clim, label) for a MOD_STATS panel: MOD_STATS_STYLE over defaults."""
    st = MOD_STATS_STYLE.get(key, {})
    if key in _value_scale_keys:
        d_clim, d_label = MOD_CLIM, "log10(rho / Ohm*m)"
    else:
        d_clim, d_label = None, key
    return (
        st.get("cmap", MOD_CMAP),
        st.get("clim", d_clim),
        st.get("label", d_label),
    )


_alpha_block = None
_need_plot = (MOD_QC or MOD_STATS) and fviz is not None and bool(model_list)
if _need_plot and MOD_ALPHA_SOURCE:
    _a_name, _a_op, _a_thr = _parse_alpha_source(MOD_ALPHA_SOURCE)
    if _a_name not in _stat_map:
        if PLOT_ONLY:
            _why = "not stored in the results .npz (plot-only mode)"
        elif _a_name.startswith("sens_"):
            _why = (
                "COMPUTE_SENS=False"
                if not COMPUTE_SENS
                else f"no member had usable sensitivity ({sens_count}/"
                f"{n_members}; first reason: "
                f"{sens_fail[0][1] if sens_fail else 'n/a'})"
            )
        elif _a_name.startswith("simrc_"):
            _why = (
                "COMPUTE_SIMRC=False"
                if not COMPUTE_SIMRC
                else f"no member had usable forward responses ({simrc_count}/"
                f"{n_members}; first reason: "
                f"{simrc_fail[0][1] if simrc_fail else 'n/a'})"
            )
        else:
            _why = "unknown statistic name"
        print(
            f"\n  MOD_ALPHA_SOURCE: '{_a_name}' not available — {_why}. "
            f"Available: {sorted(_stat_map)}. Blanking disabled."
        )
    else:
        _alpha = _alpha_from_condition(
            _stat_map[_a_name][0],
            _a_op,
            _a_thr,
            MOD_ALPHA_MODE,
            MOD_ALPHA_FADE_WIDTH,
        )
        os.makedirs(MOD_STATS_DIR, exist_ok=True)
        _alpha_block = os.path.join(
            MOD_STATS_DIR, f"resistivity_block_{P}_alpha.dat"
        )
        _write_alpha_block(
            _alpha, min(model_list, key=lambda x: x[2])[0], _alpha_block
        )
        print(
            f"\n  MOD_ALPHA_SOURCE '{MOD_ALPHA_SOURCE}' "
            f"(mode='{MOD_ALPHA_MODE}'): "
            f"{int(np.sum(_alpha >= 1.0))} shown, "
            f"{int(np.sum((_alpha > 0.0) & (_alpha < 1.0)))} faded, "
            f"{int(np.sum(_alpha <= 0.0))} blanked "
            f"of {_alpha.size} cells."
        )

# --- (7) QC slice plot — best-nRMS member ---------------------------------
if MOD_QC:
    if fviz is None:
        print("\n  MOD_QC: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_QC: no converged members — skipping.")
    else:
        _best = min(model_list, key=lambda x: x[2])
        _best_file, _best_iter, _best_nrms = _best
        _best_member = _plot_only_best_member or os.path.basename(
            os.path.dirname(_best_file)
        )
        _best_name = _po_best_name or os.path.basename(_best_file)
        print(
            f"\nQC: best member  {_best_member}  nRMS={_best_nrms:.4f}  "
            f"iter={_best_iter}"
        )
        _qc_block, _qc_stats = _best_file, False
        if _po_qc_from_vector:
            # Plot-only with PLOT_ONLY_TEMPLATE: the template is not the best
            # member's own file, so draw the stored best vector into it (same
            # route and sentinels as the statistic panels).
            os.makedirs(MOD_STATS_DIR, exist_ok=True)
            _qc_block = os.path.join(
                MOD_STATS_DIR, f"resistivity_block_{P}_best.dat"
            )
            fem.insert_model(
                template=_best_file,
                model=ens_matrix[
                    min(range(len(model_list)), key=lambda i: model_list[i][2])
                ],
                model_file=_qc_block,
                ocean=MOD_OCEAN,
                air_rho=_STATS_AIR_RHO,
                ocean_rho=_STATS_OCEAN_RHO,
                out=OUT,
            )
            _qc_stats = True
        _plot_slice(
            block_file=_qc_block,
            stats_panel=_qc_stats,
            pdf_file=MOD_QC_FILE,
            utm_e=utm_e,
            utm_n=utm_n,
            utm_lat=utm_lat,
            utm_lon=utm_lon,
            utm_zone=utm_zone,
            utm_north=utm_north,
            site_xys=site_xys,
            obs_coords_only=obs_coords_only,
            alpha_file=_alpha_block if MOD_ALPHA_QC else None,
            figure_title=(
                f"{_best_name}\n"
                f"best member: {_best_member}  iter {_best_iter}  "
                f"nRMS {_best_nrms:.3f}"
            ),
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
            if _key == "best":
                continue  # handled by the QC block above (section 7)
            if _key not in _stat_map:
                print(
                    f"  MOD_STATS: '{_key}' not available for this run — skipped."
                )
                continue
            _vec, _label = _stat_map[_key]
            _roi_normalised = False

            # --- ROI diagnostic / optional ROI-max renormalisation --------
            # Print the original (whole-mesh-normalised, as stored) ROI
            # min/max for every sens_*/simrc_* panel actually plotted -- a
            # quick read of the value range in view, to help pick a
            # MOD_ALPHA_SOURCE threshold. SENS_ROI_NORMALIZE additionally
            # re-centres the panel to the ROI's own max, but only for
            # sens_* (not sens_cv_*, a linear ratio; not simrc_coef/_corr,
            # left as-is pending more experience with that method).
            _is_sens = _key.startswith("sens_") and not _key.startswith(
                "sens_cv"
            )
            _is_simrc = _key.startswith("simrc_")
            if _is_sens or _is_simrc:
                _roi_mask = _roi_free_mask(_best_file, _vec.shape[0])
                if _roi_mask is not None and np.any(_roi_mask):
                    _roi_vals = _vec[_roi_mask]
                    _roi_vals = _roi_vals[np.isfinite(_roi_vals)]
                    if _roi_vals.size:
                        print(
                            f"  {_key}: original ROI min={_roi_vals.min():.3f}, "
                            f"max={_roi_vals.max():.3f} (log10, whole-mesh-"
                            f"normalised, n={_roi_vals.size} free params)"
                        )
                        if _is_sens and SENS_ROI_NORMALIZE:
                            _roi_max = _roi_vals.max()
                            _vec = _vec - _roi_max
                            _label = f"{_label}, ROI-renormalised"
                            _roi_normalised = True
                    else:
                        print(
                            f"  {_key}: no finite values inside the ROI — "
                            f"skipping ROI diagnostic for this panel."
                        )

            _cmap, _clim, _cbl = _panel_style(_key)
            if _roi_normalised:
                _cbl = (
                    f"{_cbl} [ROI-renorm.]"
                    if _cbl
                    else "ROI-renormalised (log10)"
                )
                # ROI renormalisation can push values above the whole-mesh-
                # normalised default clim (e.g. [-4, 0]) since cells outside
                # the ROI may now exceed the ROI's own max -- auto-scale
                # unless the user explicitly overrode clim for this key in
                # MOD_STATS_STYLE.
                if (
                    _key not in MOD_STATS_STYLE
                    or "clim" not in MOD_STATS_STYLE[_key]
                ):
                    _clim = None
            _block_out = os.path.join(
                MOD_STATS_DIR, f"resistivity_block_{P}_{_key}.dat"
            )
            _pdf_out = os.path.join(MOD_STATS_DIR, f"{P}_{_key}")
            print(f"\nSTATS: writing {_label} → {_block_out}")
            # insert_model writes 10**v; plot_model_slices log10s it back,
            # so the panel shows _vec itself (NaN -> 0 raw -> not drawn).
            fem.insert_model(
                template=_best_file,
                model=np.where(np.isfinite(_vec), _vec, -np.inf),
                model_file=_block_out,
                ocean=MOD_OCEAN,
                air_rho=_STATS_AIR_RHO,
                ocean_rho=_STATS_OCEAN_RHO,
                out=OUT,
            )
            print(
                f"STATS: plotting {_label} → {_pdf_out}  "
                f"(cmap={_cmap}, clim={_clim})"
            )
            _plot_slice(
                block_file=_block_out,
                pdf_file=_pdf_out,
                utm_e=utm_e,
                utm_n=utm_n,
                utm_lat=utm_lat,
                utm_lon=utm_lon,
                utm_zone=utm_zone,
                utm_north=utm_north,
                site_xys=site_xys,
                obs_coords_only=obs_coords_only,
                clim=_clim,
                cmap=_cmap,
                cbar_label=_cbl,
                alpha_file=(
                    None
                    if _key.startswith(tuple(MOD_ALPHA_EXCLUDE))
                    else _alpha_block
                ),
                stats_panel=True,
            )

# --- (9) Borehole profiles (optional, MOD_BH) -----------------------------
# Resistivity-depth profiles at the MOD_BH positions: ensemble statistics
# (avg / med / percentiles), all members ("ensemble", drawn with opacity
# "alpha"), a density heat map of the members ("heatmap") and/or the best
# member. The element search is done once per
# borehole; each statistic / member then costs one array lookup.
_bh_ready = False
if MOD_BH:
    if fviz is None:
        print("\n  MOD_BH: femtic_viz not available - skipping.")
    elif not model_list:
        print("\n  MOD_BH: no converged members - skipping.")
    else:
        print("\nBOREHOLES:")
        _best_i = min(range(len(model_list)), key=lambda i: model_list[i][2])
        _best_file = model_list[_best_i][0]

        # Region structure of the template (lowest-nRMS member); all members
        # share it, as for the MOD_STATS block files.
        _st = fem._read_resistivity_block_struct(
            _best_file, ocean=MOD_OCEAN, out=False
        )
        _bh_er = np.asarray(_st["elem_region"], dtype=np.int64)
        _bh_nreg = int(_st["nreg"])
        _bh_free = np.asarray(_st["free_idx"], dtype=np.int64)
        if _bh_free.size != ens_matrix.shape[1]:
            print(
                f"  MOD_BH: template has {_bh_free.size} free regions but "
                f"the ensemble has {ens_matrix.shape[1]} - skipping."
            )
        elif _bh_er.size != _bh_mesh()["n_elem"]:
            print(
                f"  MOD_BH: block file has {_bh_er.size} elements but "
                f"MOD_MESH has {_bh_mesh()['n_elem']} - skipping."
            )
        else:
            _bh_ready = True

if _bh_ready:
    # log10(rho) of fixed regions: air = NaN (gap), ocean = MOD_OCEAN_RHO,
    # other fixed regions keep the template value.
    _bh_fixed = np.log10(np.asarray(_st["region_rho"], dtype=float))
    _bh_fixed[0] = np.nan
    if _bh_nreg > 1 and _st["meta"]["ocean_present"]:
        _bh_fixed[1] = np.log10(MOD_OCEAN_RHO)
    _bh_fpos = np.full(_bh_nreg, -1, dtype=np.int64)
    _bh_fpos[_bh_free] = np.arange(_bh_free.size)

    # keys usable for borehole lines: value-scale statistics
    _bh_value_keys = {k for k in _value_scale_keys if k in _stat_map}

    # Reference model (MOD_REF_FILE): log10(rho) per element, air = NaN,
    # ocean = MOD_OCEAN_RHO (as for the ensemble lines). Wanted when
    # MOD_REF_BH is on or a MOD_BH entry lists "ref" in its "what".
    _bh_ref_el = None
    if (
        MOD_REF_BH or any("ref" in (_t.get("what") or ()) for _t in MOD_BH)
    ) and _ref_active():
        _rs = _ref_struct()
        _rs_er = np.asarray(_rs["elem_region"], dtype=np.int64)
        if _rs_er.size != _bh_er.size:
            _ref_disable(
                f"MOD_REF_FILE has {_rs_er.size} elements but the model has "
                f"{_bh_er.size} - it must live on the same mesh as MOD_MESH."
            )
        else:
            _rs_l = np.log10(np.asarray(_rs["region_rho"], dtype=float))
            _rs_l[0] = np.nan
            if _rs["nreg"] > 1 and _rs["meta"]["ocean_present"]:
                _rs_l[1] = np.log10(MOD_OCEAN_RHO)
            _bh_ref_el = _rs_l[_rs_er]
            print(f"  reference model for the profiles: {MOD_REF_FILE}")

    _bh_profiles = []
    _bh_npz = {}
    _bh_hdr = []
    _bh_names = set()

    # Expand "sites" templates (one borehole per site, from MOD_SITE_DAT).
    _bh_specs = []
    for _tpl in MOD_BH:
        if "sites" not in _tpl:
            _bh_specs.append(_tpl)
            continue
        if not MOD_SITE_DAT or not os.path.isfile(MOD_SITE_DAT):
            print("  MOD_BH: 'sites' needs MOD_SITE_DAT - entry skipped.")
            continue
        if utm_e is None:
            print("  MOD_BH: 'sites' needs the UTM origin - entry skipped.")
            continue
        _sel = _tpl["sites"]
        if isinstance(_sel, str) and _sel.strip().lower() == "all":
            _want = None
        else:
            _want = [_sel] if isinstance(_sel, str) else [str(v) for v in _sel]
        _rows = fem.read_site_dat(MOD_SITE_DAT, site_names=_want)
        if _want is not None:
            _miss = sorted(set(_want) - {str(r["name"]) for r in _rows})
            if _miss:
                print(f"  MOD_BH: sites not found in MOD_SITE_DAT: {_miss}")
        _nr = _tpl.get("n_random")
        if _nr is not None and int(_nr) < len(_rows):
            _pick = np.random.default_rng(_tpl.get("seed")).choice(
                len(_rows), size=int(_nr), replace=False
            )
            _rows = [_rows[i] for i in sorted(_pick)]
            print(
                f"  n_random={int(_nr)}"
                f"{'' if _tpl.get('seed') is not None else ' (no seed, not reproducible)'}"
                f": {[str(r['name']) for r in _rows]}"
            )
        print(f"  sites template -> {len(_rows)} borehole(s)")
        for _r in _rows:
            _t = {
                k: v
                for k, v in _tpl.items()
                if k not in ("sites", "n_random", "seed")
            }
            _t.update(
                name=str(_r["name"]),
                x=(float(_r["easting"]), "utm"),
                y=(float(_r["northing"]), "utm"),
            )
            _bh_specs.append(_t)

    for _spec in _bh_specs:
        _nm = str(_spec.get("name", f"BH{len(_bh_profiles) + 1}"))
        _nm0, _dup = _nm, 1
        while _nm in _bh_names:  # names key the .npz arrays: keep unique
            _dup += 1
            _nm = f"{_nm0}_{_dup}"
        if _nm != _nm0:
            print(f"  MOD_BH: name '{_nm0}' already used - renamed '{_nm}'.")
        _bh_names.add(_nm)
        try:
            _x_m, _y_m, _lat, _lon = _bh_resolve_xy(
                _spec, utm_e, utm_n, utm_zone, utm_north
            )
        except KeyError as _e:
            print(f"  MOD_BH '{_nm}': missing key {_e} - skipped.")
            continue
        except ValueError as _e:
            print(f"  {_e} - skipped.")
            continue

        _dz = float(_spec.get("dz", 0.05)) * 1000.0
        _zb_km = _spec.get(
            "z_bot", MOD_ZLIM[1] if MOD_ZLIM is not None else 20.0
        )
        _zb = float(_zb_km) * 1000.0
        _zt_raw = _spec.get("z_top", "surface")
        if isinstance(_zt_raw, str) and _zt_raw.strip().lower() == "surface":
            _zt = _bh_surface_m(_x_m, _y_m, _dz, _zb, _bh_er)
            if _zt is None:
                print(
                    f"  MOD_BH '{_nm}': no non-air element above "
                    f"z_bot={_zb_km} km at this position - skipped."
                )
                continue
            print(f"  {_nm}: z_top='surface' -> {_zt / 1000.0:.3f} km")
        else:
            _zt = float(_zt_raw) * 1000.0
        if _zb <= _zt or _dz <= 0.0:
            print(f"  MOD_BH '{_nm}': need z_bot > z_top and dz > 0 - skipped.")
            continue

        _depth = np.arange(_zt, _zb + 0.5 * _dz, _dz)
        _elem = _bh_locate(_x_m, _y_m, _depth)
        _reg = np.where(_elem >= 0, _bh_er[np.maximum(_elem, 0)], -1)
        _n_hit = int(np.sum(_reg > 0))
        print(
            f"  {_nm}: x={_x_m / 1000.0:.3f} km, y={_y_m / 1000.0:.3f} km, "
            f"{_depth.size} levels, {_n_hit} in non-air elements"
        )
        if _n_hit == 0:
            print(f"  MOD_BH '{_nm}': profile is entirely air/outside - skipped.")
            continue

        _order = _bh_expand_what(
            _spec.get("what", ["avg", "percentiles"]), _bh_value_keys
        )
        # reference line: listed explicitly in "what" (forces it on), or added
        # last when MOD_REF_BH is on and the entry does not say ref=False
        if _bh_ref_el is None:
            _order = [_k for _k in _order if _k != "ref"]
        elif "ref" not in _order and MOD_REF_BH and _spec.get("ref", True):
            _order.append("ref")
        _lines = {}
        for _k in _order:
            if _k in ("ensemble", "heatmap"):
                continue
            if _k == "ref":
                _lines[_k] = np.where(
                    _elem >= 0,
                    10.0 ** _bh_ref_el[np.maximum(_elem, 0)],
                    np.nan,
                )
                continue
            if _k == "best":
                _vec = ens_matrix[_best_i]
            elif _k in _BH_SIGMA_Q:
                _vec = np.percentile(ens_matrix, _BH_SIGMA_Q[_k], axis=0)
            else:
                _vec = _stat_map[_k][0]
            _lines[_k] = _bh_values(_vec, _reg, _bh_fixed, _bh_fpos)
        # all members along the borehole: needed for the "ensemble" lines and
        # for the "heatmap" histogram
        _ens_all = (
            _bh_values(ens_matrix, _reg, _bh_fixed, _bh_fpos)
            if ("ensemble" in _order or "heatmap" in _order)
            else None
        )
        _ens = _ens_all if "ensemble" in _order else None
        _heat = _heat_edges = None
        if "heatmap" in _order:
            try:
                _heat_edges = _bh_edges(_spec, _ens_all)
                if _heat_edges is None:
                    print(f"  MOD_BH '{_nm}': heat map - nothing to bin.")
            except ValueError as _e:
                print(f"  {_e} - heat map skipped.")
            if _heat_edges is None:
                _order = [_k for _k in _order if _k != "heatmap"]
            else:
                _heat = _bh_heatmap(_ens_all, _heat_edges)
                print(
                    f"  {_nm}: heat map, {_heat.shape[1]} bins, "
                    f"{MOD_BH_HEAT_NORM} colour scale"
                )
        _bh_profiles.append(
            dict(
                name=_nm,
                pos_str=_bh_pos_str(_x_m, _y_m, _lat, _lon, utm_e, utm_n),
                spec=_spec,
                depth=_depth,
                order=_order,
                lines=_lines,
                ens=_ens,
                heat=_heat,
                heat_edges=_heat_edges,
            )
        )

        _safe = _nm.replace(" ", "_")
        _bh_npz[f"depth_{_safe}"] = _depth
        for _k, _v in _lines.items():
            _bh_npz[f"{_k}_{_safe}"] = _v
        if _ens is not None:
            _bh_npz[f"ensemble_{_safe}"] = _ens
        if _heat is not None:
            # fractions (n_levels, n_bins); edges in log10(Ohm.m)
            _bh_npz[f"heat_{_safe}"] = _heat
            _bh_npz[f"heat_edges_{_safe}"] = _heat_edges
        _bh_hdr.append(
            dict(
                name=_nm,
                x_m=_x_m,
                y_m=_y_m,
                lat=None if _lat is None else float(_lat),
                lon=None if _lon is None else float(_lon),
                z_top_m=float(_zt),
                z_bot_m=float(_zb),
                dz_m=float(_dz),
                n_levels=int(_depth.size),
                keys=list(_lines)
                + (["ensemble"] if _ens is not None else [])
                + (["heatmap"] if _heat is not None else []),
                heat_bins=None if _heat is None else int(_heat.shape[1]),
            )
        )

    if _bh_profiles:
        if MOD_BH_NPZ:
            import json

            _bh_npz["header"] = np.array(
                json.dumps(
                    dict(
                        ensemble_prefix=ENSEMBLE_PREFIX,
                        depth_unit="m, z positive-down, model datum",
                        rho_unit="Ohm.m (NaN = air / outside mesh)",
                        heat_unit=(
                            "heat_<name>: (n_levels, n_bins) fraction of the "
                            "valid members per bin (rows sum to 1 when all "
                            "members are inside the edges); "
                            "heat_edges_<name>: bin edges in log10(Ohm.m)"
                        ),
                        members=[m[0] for m in model_list],
                        best_member=_best_file,
                        reference_file=(
                            MOD_REF_FILE if _bh_ref_el is not None else None
                        ),
                        percentiles=list(PERCENTILES),
                        boreholes=_bh_hdr,
                    ),
                    indent=2,
                )
            )
            os.makedirs(os.path.dirname(MOD_BH_FILE) or ".", exist_ok=True)
            np.savez_compressed(MOD_BH_FILE + ".npz", **_bh_npz)
            print(f"  profiles saved -> {MOD_BH_FILE}.npz")
        os.makedirs(os.path.dirname(MOD_BH_FILE) or ".", exist_ok=True)
        _per = int(MOD_BH_PER_FIG) if MOD_BH_PER_FIG else len(_bh_profiles)
        _chunks = [
            _bh_profiles[i : i + _per] for i in range(0, len(_bh_profiles), _per)
        ]
        for _ic, _ch in enumerate(_chunks):
            _plot_boreholes(
                _ch,
                MOD_BH_FILE if len(_chunks) == 1 else f"{MOD_BH_FILE}_{_ic + 1:02d}",
            )
    else:
        print("  MOD_BH: no usable borehole - nothing plotted.")

print("\nfemtic_ens_post.py complete.")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(__file__)
