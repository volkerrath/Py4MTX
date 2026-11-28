#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_sitemap.py
======================

Site map of a FEMTIC run, or of a whole ensemble, on a DEM (Copernicus) background:
every site is an icon (marker) coloured by its normalized RMS misfit (nRMS), for
frequency bands and data components, or by another measure of data fit, or by the
strength of its galvanic distortion (see "Measures and distortion strength"), or
by the dimensionality of its observed impedances (see "Dimensionality"). For an
ensemble the colour of a fit or distortion measure is a statistic of the member
values at the site. Profile lines and other polylines (faults, coast lines, ...)
can be drawn on the maps.

What it does
------------
1. Reads one results_iter<N>.h5 from WORK_DIR (ITERATION = "best", "last" or
   a number; as in femtic_ens_misfit.py / femtic_mod_misfit.py) with
   femtic_misfit.resolve_run, and the site list site.dat (name, lat, lon,
   elev, sitenum, easting, northing) from WORK_DIR/templates (TEMPLATES_DIR;
   if absent there, WORK_DIR itself and ../templates, next to WORK_DIR). With
   several members (see "Ensemble") every member is read on its own.
2. Assigns the site ids of the results file to the sites of site.dat, by
   position (SITE_MATCH = "coords"; the results file holds x, y in metres
   relative to the model origin, site.dat the UTM easting / northing) or by
   number (SITE_MATCH = "id"; site_id = sitenum + SITE_ID_OFFSET). All data
   types of one site (MT, VTF, ...) count for the site; HTF and NMT rows have no
   station position and are only counted in the global values. Two stations at
   one place (several ids at one position and several site.dat entries there)
   get one entry each, in the order of the numbers.
3. nRMS for every combination of a frequency band (BANDS) and a component
   selection (COMPONENTS), per site and for the whole data set ("global"):

       nRMS = sqrt( mean( ((obs - cal) / err)**2 ) )

   real and imaginary parts separately (the imaginary part is left out for
   real-valued data types), non-finite values and err <= 0 skipped; this is
   the nRMS of femtic_misfit.site_nrms, and the global value for all bands
   and all components is femtic_misfit.data_rms.
4. Basemap: the extent comes from the user parameters (MAP_EXTENT_MODE: the
   sites plus a margin, a lon / lat box, or a centre and a size in km). The
   DEM is the Copernicus DEM GLO-30 (DEM_SOURCE = "copernicus30", the default;
   "copernicus90" for large maps): its 1 x 1 degree GeoTIFF tiles are
   downloaded into DEM_CACHE_DIR if they are not there yet and read with
   rasterio. Other sources: the AWS Terrain Tiles ("skadi"), tiles of your
   own ("tiles"), one file ("file": GeoTIFF with rasterio, .npz with lon,
   lat, z, or an .hgt tile), or none ("none"). It is shown as a shaded relief.
5. One figure per page (PAGE_BY): the panels are the maps of the selections,
   the sites drawn as icons coloured by their nRMS (colour map CMAP_NRMS,
   limits NRMS_VMIN / NRMS_VMAX, one colour scale for all pages; values
   beyond the limits get the end colours and arrows on the colour bar). Sites
   without data in a selection are drawn as open icons.
6. Tables: <OUTPUT_BASENAME>_nrms.csv and _n.csv (nRMS and number of values,
   one row per site, one column per selection) and _summary.txt (global nRMS
   and the spread of the site values per selection); for an ensemble also
   _members.csv (one row per member) and _ens_stats.csv (mean, median, std,
   min, max over the members per selection and site); for profiles
   _profiles.csv and _profile_sites.csv (see "Profiles and lines").

Measures and distortion strength
--------------------------------
SITE_MEASURES (command line --measures) lists what is mapped; each entry gets its
own maps, tables and summary section. Fit measures, of the error-normalized
residuals r = (obs - cal) / err of the rows of a selection (real and imaginary
parts together; modules/misfit_measures.py), with the value expected for
Gaussian errors of the stated size:

    "nrms"   sqrt(mean(r**2))                      1     (the default)
    "r_mae"  sqrt(pi/2) * mean(|r|)                1
    "r_med"  median(|r|) / 0.67449                 1
    "q95"    95th percentile of |r|                1.96
    "mean"   mean(r), the bias                     0
    "std"    standard deviation of r (1/n)         1
    "skew"   skewness                              0
    "kurt"   excess kurtosis                       0

Colour scales: nrms, r_mae, r_med, std use CMAP_NRMS, NRMS_REFERENCE, NRMS_VMIN
and NRMS_VMAX; q95 CMAP_NRMS centred on 1.96; mean, skew, kurt "RdBu_r" centred
on 0 (the largest absolute values drawn on top); distortion CMAP_SPREAD, linear
from 0. MEASURE_SCALES overrides cmap, reference, vmin, vmax per measure. With
ENS_STAT = "std" every measure but nRMS uses CMAP_SPREAD linear from 0 (nRMS as
before: NRMS_VMIN / NRMS_VMAX still apply to it).

"distortion": one map (no bands or components) of |C - I| = Frobenius norm
sqrt(sum C'ij**2) of the matrix difference C' = C - I that FEMTIC estimates with
the distortion type ESTIMATE_DISTORTION_MATRIX_DIFFERENCE (Z_obs = C Z_model).
0 = no distortion estimated at the site. This one number is a summary chosen
here (femtic_distcorr.py ranks sites by the same norm); it does not tell a gain
from a rotation or shear, and the estimated C' depends on the regularization of
the distortion in the inversion (see Usui et al., 2017), so compare runs with the
same settings only. C' is read from the /distortion group of the results file; if
there is none, from DISTORTION_FILE (distortion_iter<N>.dat) next to it, whose
type CANNOT be checked: do not use it for runs with gains / rotations. Runs
without distortion (or of another type) give a warning and no distortion map.
For an ensemble the value of a site is ENS_STAT over the members that have it.
Sites whose matrix was held fixed are included (C' as given) and marked in
_distortion.csv.

Dimensionality
--------------
Further SITE_MEASURES, of the impedances themselves (modules/dimens.py, which is
needed only for them), one value per site and frequency band (BANDS, BAND_AXIS,
BAND_PER_DECADE, BAND_MIN_N, with a site frequency counted as four rows;
COMPONENTS and the DATATYPES filter do not enter):

    "dim"     class of the WAL rotational invariants (WALDIM 1.2): undetermined,
              1D, 2D, 3D/2D, 3D, 3D/2Ddiag, 3D/1D2D, ANIS-HINT1, ANIS-HINT2
    "ptdim"   phase tensor class: undefined, 1D, 2D, 3D
    "ptskew"  median |beta| of the phase tensor [deg]
    "ellipt"  median ellipticity of the phase tensor
    "swift"   median Swift skew
    "bahr"    median Bahr phase-sensitive skew

DIM_SOURCE "results" (default) takes the observed impedances (data type "MT")
of the first used member and their errors as in the file (error floors
included; Ohm, converted to mV/km/nT); only frequencies with the four
components, finite values and errors > 0. A warning is given if other members
have other observed data (perturbed data of an RTO ensemble): then the values
are those of that one member, and DIM_SOURCE should name the original data: an
EDI directory, EDI files or a .npz collection, whose station names are matched
to site.dat without regard to case (positions are those of site.dat). Only
sites with data of the chosen DATATYPES in the results file are mapped.
Angles (strikes, azimuths) refer to the x axis of the data and are in the
tables only; no strike is drawn.
Maps: one figure per measure with a panel per band (DIM_PAGE "all") or one per
measure and band ("band"); classes have fixed colours (DIM_COLORS) and a colour
bar of names, the others CMAP_SPREAD linear from 0 to 10 deg (ptskew) or 0.5
(MEASURE_SCALES changes it). A panel title gives the number of sites n and the
most frequent classes or the median of the sites. Open icons: no data in the
band (a class "undetermined" is grey).
Output: <OUTPUT_BASENAME>_dimensionality.txt / .csv / .npz (all band values
per site, including strike, twist, shear and phase tensor azimuth), a section
in _summary.txt, and columns "<measure> | <band>" in _profile_sites.csv (classes
as codes). Thresholds: DIM_THRESHOLD, DIM_Q_THRESHOLD, DIM_PT_BETA_THRESHOLD,
DIM_PT_ELLIPTICITY_THRESHOLD; further keywords of
dimens.classify_dimensionality in DIM_OPTIONS.

Ensemble
--------
WORK_DIR = the root of an ensemble: <WORK_DIR>/templates/site.dat and one
directory per member with results_iter*.h5. If RESULTS_PATH (default WORK_DIR)
holds no results file, its sub-directories that do are the members; RUN_PATHS
(run directories, results files or globs, relative to WORK_DIR) names them
explicitly. Every member gets its own site assignment and nRMS, against its own
observed data (as in its .cnv file; the RMS1 of the ensemble mean is not
computed). The value of a site is ENS_STAT over the members: "mean", "median"
(default), "min", "max", "std" (sample std; own colour map CMAP_SPREAD, linear
scale from 0) or a percentile "pNN" ("p90"). MEMBER_NRMS_MAX leaves out members
with a larger nRMS of the iteration. A member that cannot be read is skipped
with a warning (and listed in _members.csv); with one member left or only one
found, the output is that of a single run.

Profiles and lines
------------------
PROFILES (cross-section lines) and LINES (faults, coast lines, caldera rims,
...) are lists of dicts (the keys are listed at the settings and in the readme)
that are drawn on every map. Positions are given in `coords`: "lonlat" (default),
"latlon", "utm" (easting, northing in km, in the UTM zone of site.dat, which is
found by comparing the easting / northing of the sites with their lat / lon) or
"model" (x, y in km of the FEMTIC model, the frame of site_x, site_y of the results file;
needs the sites assigned by position). Lines are given by points or read from a
file: text (GMT multi-segment, two columns), GeoJSON, KML or KMZ (the last three
always lon, lat). A profile is given by points, a file, start and end, or
centre, azimuth and length_km; ticks_km marks the distance along it, width_km
gives it a swath: the sites within width / 2 of the profile (not beyond its ends)
get a ring on the map and are listed with their distance along and across the
profile and their nRMS in _profile_sites.csv. Distances are flat-earth km on a
sphere (the scale of the map), about 0.3 % less east-west than on the WGS84
ellipsoid at mid latitudes; use profiles of up to a few 100 km.
The end points may carry their label: start=(x, y, "A"), end=(x, y, "A'");
PROFILE_END_MARKER adds a symbol there.
Style: every entry may set its own colour, line width and style, opacity,
drawing order, halo and markers (profiles also texts, ticks, swath and rings);
a key that is not given takes the setting PROFILE_* / LINE_*. A profile or line
that lies outside the map is named in a warning, with a note if it would be on
the map with longitude and latitude exchanged.
Profiles and lines are cut at the map
box made smaller by OVERLAY_CUT_MARGIN_KM (default 1 km), so that both ends of a
cut profile are inside the map; the tick distances and the tables still refer to
the whole profile (None = not cut). The map box itself does not depend on them
(MAP_INCLUDE_PROFILES = False): what goes beyond it is cut at it.

Appearance
----------
Fonts: FONT_FAMILY, FONT_RC (matplotlib rcParams) and the sizes in pt of tick
labels (TICK_FONTSIZE, also the base of the sizes that are None), axis labels,
figure and panel titles, site names, legend, colour bar, scale bar and DEM
credit; weights of the title and the site names. A figure title that is wider
than the figure is broken into lines. Sizes of figures: PANEL_SIZE (one panel),
CBAR_LENGTH and CBAR_WIDTH are in cm.

Selections
----------
BANDS      "all"      all frequencies only
           "decades"  all, and bands of BAND_PER_DECADE per decade (period or
                      frequency, BAND_AXIS), with at least BAND_MIN_N data
                      rows; the bands of femtic_misfit (after Baba, 2023)
           list       ["all", (0.01, 1.0), (1.0, 100.0)]: limits in s
                      (BAND_AXIS = "period") or Hz ("frequency"), lower limit
                      included, upper excluded; bands with fewer than
                      BAND_MIN_N rows are dropped
COMPONENTS list of selections: "all", a group of COMPONENT_GROUPS ("Z", "Zoff",
           "Zdiag", "T", ...), a component name ("Zxy"; any case, as written
           in the results file), or several joined by "+" ("Zxy+Zyx").
DATATYPES  None = all data types, or a list of names, e.g. ["MT", "VTF"].

DEM source
----------
DEM_SOURCE selects the elevation model of the relief (the routines are in
modules/dem.py):

  "copernicus30"  Copernicus DEM GLO-30 (the default): global, 1 arc second
                  (30 m; coarser in longitude at high latitudes), a surface
                  model (it includes buildings and vegetation) with heights
                  above the EGM2008 geoid. Cloud Optimized GeoTIFF tiles of
                  the open data bucket copernicus-dem-30m on AWS (no key, no
                  login), one file per 1 x 1 degree: https://copernicus-dem-30m
                  .s3.amazonaws.com/Copernicus_DSM_COG_10_N65_00_W017_00_DEM/
                  Copernicus_DSM_COG_10_N65_00_W017_00_DEM.tif. There are no
                  tiles over the open ocean (the maps get DEM_FILL_MISSING
                  there), and a few tiles of some countries are not public.
  "copernicus90"  GLO-90 (3 arc seconds), bucket copernicus-dem-90m: for maps
                  of a few 100 km, where GLO-30 needs more tiles than
                  DEM_MAX_TILES.
  "skadi"         The AWS Terrain Tiles (Mapzen), .hgt tiles of 1 arc second:
                  a composite of several elevation data sets, not one model
                  (the SRTM is the main source between about 60 N and 56 S,
                  other sources are used at high latitudes, and coarse sources
                  are interpolated). The former "srtm" setting. No rasterio.
  "tiles"         Your own tiles: DEM_URL_TEMPLATE = a URL or a path (relative
                  paths: from WORK_DIR) with the placeholders {name} (N65W017),
                  {NS}, {lat}, {EW}, {lon}; GeoTIFF (.tif) or .hgt tiles.
                  A path is used as it is (nothing is copied to the cache).
  "file"          One file (DEM_FILE): GeoTIFF, .npz or an .hgt tile.
  "none"          No relief.

The tiles of a URL are downloaded once into DEM_CACHE_DIR and never again
(tiles can also be put there by hand); DEM_DOWNLOAD = False uses only those.
DEM_URL_TEMPLATE replaces the URL of the source (a mirror of the bucket).
GeoTIFF files are read with rasterio, only the part inside the map, averaged
down to about DEM_MAX_PIXELS pixels per side; a raster in a projected
coordinate system (a national DEM, UTM) is warped to lon / lat first. The credit
line of the maps (DEM_CREDIT = "auto") is the notice that the Copernicus
licence asks for, or a note on the AWS Terrain Tiles; check the terms of any
other source you use.

Copernicus GLO-30 from Python (rasterio needed; the tiles of the map are
downloaded once into DEM_CACHE_DIR):

    import femtic_plot_sitemap as sm
    cfg = sm.settings("/data/krafla/krafla_rto_26", DEM_SOURCE="copernicus30",
                      DEM_CACHE_DIR="/data/dem_cache", DEM_STYLE="terrain")
    sm.run(cfg)

Usage
-----
    python femtic_plot_sitemap.py                  # settings below
    python femtic_plot_sitemap.py -w /data/run01   # another WORK_DIR
    python femtic_plot_sitemap.py --dem copernicus30   # Copernicus DEM GLO-30 (the default)
    python femtic_plot_sitemap.py --dem copernicus90   # GLO-90, for large maps
    python femtic_plot_sitemap.py --dem none       # no DEM (no download)
    python femtic_plot_sitemap.py -i last          # last iteration
    python femtic_plot_sitemap.py -w /data/ens     # ensemble root: median of the members
    python femtic_plot_sitemap.py -w /data/ens --stat std   # spread of the members
    python femtic_plot_sitemap.py --runs "kra*/" --stat p90 # members by glob, 90th percentile
    python femtic_plot_sitemap.py --profile "A-A'" -16.82 65.70 -16.74 65.74   # profile line
    python femtic_plot_sitemap.py --line gis/faults.txt    # polylines (text, GeoJSON, KML)
    python femtic_plot_sitemap.py --measures nrms r_mae distortion  # replaces SITE_MEASURES

All relative paths of the settings refer to WORK_DIR. Return code 0 on
success, 1 if the data could not be read or nothing could be plotted, 2 for a
wrong usage (WORK_DIR missing).

@author vrath
@project py4mt
@created 2026-11-21
@modified 2026-11-28

AI-GENERATED CODE NOTICE
-------------------------
Author      : Claude Sonnet 5.5 (Anthropic)
Generated   : 2026-11-21
Notice      : This code is AI-generated. Review and test it before any
              production use. The ensemble reading (2026-11-21, later the same
              day) was tested with synthetic ensembles (statistics against
              numpy and a brute-force nRMS per member and site) and with a
              pseudo-ensemble of three real Krafla iterations (1, 7, 8 as three
              members); it has NOT been run on a real RTO ensemble. The
              single-run part was tested with synthetic results files
              (the HDF5 reader replaced by a stand-in; h5py was not
              available) and with synthetic tiles, and run on a real FEMTIC
              results file (Krafla RTO member, iteration 8, read with a
              libhdf5 reader written for the test; the global nRMS agrees
              with the RMS of the .cnv file, and the 100 site ids agree with
              the site.dat by position and by number), but without a DEM
              there. The download from the server has NOT been run (no
              network where it was written; the URL pattern is the one of
              the AWS documentation), nor has the GeoTIFF reader.
              The profiles and lines (2026-11-22) were tested with synthetic
              data (UTM transform against a numerical meridian arc and the
              100 real Krafla site.dat positions; projection of the sites
              against a plain loop; readers with hand-made text, GeoJSON,
              KML and KMZ files) and run on the real Krafla results; the
              readers have NOT been tried on files from GIS programs.
              The GeoTIFF tiles of the Copernicus DEM (2026-11-22, later the
              same day) are read with rasterio, which was NOT installed where
              this was written: the code was tested with a stand-in module that
              imitates the few rasterio calls it makes (open, transform, read
              of a window, WarpedVRT), not with rasterio itself, and the
              download from the AWS bucket has NOT been run (the file names
              are those of the bucket documentation and of listings of it),
              and the warping of a raster in a projected system has not been
              tried on a real file.

Reference
---------
Usui, Y., Ogawa, Y., Aizawa, K., Kanda, W., Hashimoto, T., Koyama, T.,
Yamaya, Y., Kagiyama, T. (2017): Three-dimensional resistivity structure of
Asama Volcano revealed by data-space magnetotelluric inversion using
unstructured tetrahedral elements. Geophysical Journal International 208(3),
1359-1372. doi:10.1093/gji/ggw459

Copernicus GLO-30 Digital Elevation Model, data record of OpenTopography:
doi:10.5069/G9028PQB (data: ESA, DLR, Airbus Defence and Space; the notice of
the licence is the default credit line of the maps).

Farr, T. G., Rosen, P. A., Caro, E., Crippen, R., Duren, R., Hensley, S.,
Kobrick, M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S.,
Shimada, J., Umland, J., Werner, M., Oskin, M., Burbank, D., Alsdorf, D.
(2007): The Shuttle Radar Topography Mission. Reviews of Geophysics 45,
RG2004. doi:10.1029/2005RG000183

Changelog
---------
2026-11-21  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            created. Per-site nRMS for bands x components on an SRTM shaded
            relief, DEM download from the AWS Terrain Tiles with a cache.
            site.dat is searched in TEMPLATES_DIR, WORK_DIR and next to
            WORK_DIR (ensemble layout); ids at one position are given the
            site.dat entries at that position in the order of their numbers
            (two stations at one place in the Krafla data).
2026-11-21  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            a full ensemble is read. WORK_DIR = ensemble root (members and
            templates/) or RUN_PATHS; every member is read, matched to the
            sites and evaluated on its own; ENS_STAT (mean, median, min, max,
            std, percentile) over the members per site and selection;
            MEMBER_NRMS_MAX; unreadable members are skipped; new tables
            _members.csv and _ens_stats.csv; ensemble summary, titles ({M},
            {stat}), colour bar label and a sequential scale (CMAP_SPREAD) for
            the std; command line --runs, --stat. Output of a single run
            unchanged.
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            profile lines and other polylines on the maps. PROFILES (points,
            file, start + end, or centre + azimuth + length_km; swath with the
            sites in it, ticks with distances, end labels) and LINES (points,
            segments, text / GMT, GeoJSON, KML, KMZ; polygons with fill),
            coordinates lonlat, latlon, utm (own Kruger series, zone from
            site.dat) or model (frame and shift of the site assignment);
            legend with the labelled items (LEGEND_LOC); the map extent can
            include the profiles (MAP_INCLUDE_PROFILES); new tables
            _profiles.csv and _profile_sites.csv; command line --profile,
            --line, --no-overlays. Without PROFILES and LINES the output is
            unchanged.
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            the default DEM is the Copernicus DEM GLO-30 instead of the AWS
            Terrain Tiles. DEM_SOURCE "copernicus30" (default), "copernicus90",
            "skadi" (the former "srtm", now named for what the tiles are:
            "srtm" is refused with a hint), "tiles" (own URL or path pattern,
            GeoTIFF or .hgt), "file", "none". GeoTIFF tiles and files are read
            with rasterio, windowed and averaged down to DEM_MAX_PIXELS; a
            raster in a projected system is warped to lon / lat; tiles are
            merged (also with different pixel sizes; ocean tiles are filled
            over the map and a margin of three pixels only); credit line of the maps
            = the licence notice, wrapped to the width of the panel;
            DEM_URL_TEMPLATE is now None = the URL of the source;
            DEM_CACHE_DIR default "dem_cache" (was "srtm_cache"; tiles of the
            old cache can be copied there).
2026-11-22  Claude Sonnet 5.5 (Anthropic), later: example lines for the
            Copernicus GLO-30 in the CONFIG block, the Usage and the readme
            (cache, colour relief, offline, own tile names, one GeoTIFF).
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            CBAR_LENGTH, CBAR_WIDTH [cm] set the size of the colour bar (the
            length with the extension triangles; the figure is laid out without
            drawing and the bar made again until it has the size; CBAR_SHRINK
            as before without them; CBAR_SHRINK is checked, 0 < s <= 1).
            Profiles and lines are cut at the map box made smaller by
            OVERLAY_CUT_MARGIN_KM (default 1 km; None = not cut): both ends of
            a cut profile are inside the map, the end labels stand at the cut
            ends, the swath is flat there, rings and ticks follow, the ticks
            keep the distances of the whole profile (tables unchanged; message
            with the km shown). End labels in the start / end keys,
            start=(x, y, "A"), end=(x, y, "A'"); PROFILE_END_MARKER marks the
            ends with a symbol. Tests on synthetic data only.
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            MAP_INCLUDE_PROFILES now defaults to False: the map box (MAP_EXTENT_MODE
            "sites": sites + margin) is not enlarged by profiles that go beyond it;
            they are cut at that box (OVERLAY_CUT_MARGIN_KM). True restores the
            earlier behaviour (box large enough for the profiles).

2026-11-26 (km)  Claude Sonnet 5.5 (Anthropic)
              Settings in km: SITE_MATCH_TOL (0.015; before 15.0 m) and
              PROFILES / LINES positions with coords = 'utm' or 'model'
              typed in the settings; values that look like metres are
              refused. Coordinates read from files (site.dat, text files of
              a profile or line) stay in metres. Tested on synthetic files
              only. AI-generated; review before production use.

2026-11-27  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            (1) The DEM functions moved to the new module modules/dem.py,
            which this script now needs. (2) Style of profiles and lines:
            new settings PROFILE_* and LINE_* (colour, width, style,
            opacity, order, halo, markers; end markers, texts, ticks,
            swath, rings; fill), each also a key of an entry of PROFILES
            / LINES; the warning for an overlay outside the map says
            where it is and hints at exchanged lon / lat. The default
            profile "Vuache fault" did not show: its points are
            (lon, lat) but coords was "latlon"; now "lonlat" (also
            labels=False and a legend entry). (3) Fonts are settings (see
            "Appearance"). (4) PANEL_SIZE is in cm (before: inch; the
            default (12.2, 11.2) is about the former (4.8, 4.4)); panels
            below 6 cm get a note. Tested on synthetic data only.

2026-11-28  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            (1) Halos are off by default: PROFILE_HALO and LINE_HALO are now
            False; the halo of the site names, the scale bar and the DEM
            credit (before: always drawn) is the new setting MAP_HALO
            (False; MAP_HALO_COLOR, MAP_HALO_WIDTH). (2) New settings
            SITE_MEASURES, DISTORTION_FILE, MEASURE_SCALES and option
            --measures: maps, tables and summary sections of the fit
            measures of modules/misfit_measures.py and of the galvanic
            distortion strength |C - I|. With the default SITE_MEASURES =
            "nrms" the nRMS tables and summary are as before. (3) With
            SHOW = False every figure is closed as soon as it is saved.
            Tested on synthetic data only; the reading of a real /distortion
            group (h5py not installed) has not been run.

2026-11-28  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            (b) Dimensionality of the observed impedances (new module
            modules/dimens.py): SITE_MEASURES "dim", "ptdim", "ptskew",
            "ellipt", "swift", "bahr", settings DIM_*, option --dim-source,
            tables <OUTPUT_BASENAME>_dimensionality.*; a colour bar of class
            names (turned by 45 deg on a horizontal bar). Outputs of the
            other measures are unchanged. Tested with synthetic and sample
            data through a stand-in for the HDF5 results reader (h5py not
            installed); not run on a real results file.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import math
import os
import re
import sys
import inspect
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# py4mt environment: make the modules importable (femtic_misfit.py, femtic.py)
# ---------------------------------------------------------------------------
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")
if PY4MTX_ROOT:
    for _pth in (os.path.join(PY4MTX_ROOT, "py4mt", "modules"),
                 os.path.join(PY4MTX_ROOT, "py4mt", "scripts")):
        if os.path.isdir(_pth) and _pth not in sys.path:
            sys.path.insert(0, _pth)

import femtic_misfit as fm  # noqa: E402
import misfit_measures as mm  # noqa: E402
from dem import (DEM_SOURCES, DEM_STYLES, EARTH_RADIUS_M, dem_credit, dem_extent,  # noqa: E402
                 dem_template, is_url, load_dem, shade_dem, tile_kind)
import util as utl
from version import versionstrg

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# --- working directory and input ---
# Every relative path below is taken from WORK_DIR (an absolute path, or one
# that starts with ~, is used as given). None or "" = the current directory.
# WORK_DIR: Optional[str] = "./"
# RESULTS_PATH: str = "."               # run directory (searched for ITER_PATTERN) or a results file
WORK_DIR: Optional[str] = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"
RESULTS_PATH: str = WORK_DIR              # run directory (searched for ITER_PATTERN) or a results file


ITERATION: Union[str, int] = "best"   # "best" (smallest nRMS), "last", or a number
ITER_PATTERN: str = "results_iter*.h5"
TEMPLATES_DIR: str = "templates"      # sub-directory of WORK_DIR with site.dat
SITE_DAT: Optional[str] = "auto"      # "auto": TEMPLATES_DIR, WORK_DIR, then ../<TEMPLATES_DIR>; or a path
SITE_MATCH: str = "auto"              # "auto", "coords" (by position) or "id" (sitenum + SITE_ID_OFFSET)
SITE_MATCH_TOL: float = 0.015         # [km] largest distance for a match by position
SITE_ID_OFFSET: int = 0               # site_id in the results file = sitenum + SITE_ID_OFFSET
DATATYPES: Optional[Sequence[str]] = None   # None = all, or e.g. ["MT", "VTF"]

# --- ensemble ---
# WORK_DIR = the root of an ensemble (member directories and templates/): if
# RESULTS_PATH holds no results file, but sub-directories do, they are the
# members. RUN_PATHS names the members explicitly. With one member only, the
# ensemble settings have no effect.
RUN_PATHS: Optional[Sequence[str]] =  ["ann*/"] # None   # run directories / results files / globs, e.g. ["kra*/"]
ENS_STAT: str = "median"              # per site over the members: "mean", "median", "min", "max",
                                      # "std" (spread) or "pNN" (percentile, e.g. "p90")
MEMBER_NRMS_MAX: Optional[float] = None   # members with a larger nRMS of the iteration are left out

# --- nRMS selections ---
BANDS: Union[str, Sequence[Any]] = "decades"   # "all", "decades", or ["all", (lo, hi), ...]
BAND_AXIS: str = "period"             # "period" [s] or "frequency" [Hz]
BAND_PER_DECADE: int = 1              # for "decades"
BAND_MIN_N: int = 20                  # bands with fewer data rows are dropped
# COMPONENTS: Sequence[str] = ["all", "Zoff", "Zdiag", "T"]
COMPONENTS: Sequence[str] = ["all",  "T"]
COMPONENT_GROUPS: Dict[str, Sequence[str]] = {
    "Z": ("Zxx", "Zxy", "Zyx", "Zyy"),
    "Zoff": ("Zxy", "Zyx"),
    "Zdiag": ("Zxx", "Zyy"),
    "T": ("Tzx", "Tzy"),
    "rho": ("rhoXX", "rhoXY", "rhoYX", "rhoYY"),
    "phs": ("phsXX", "phsXY", "phsYX", "phsYY"),
    "PT": ("PTxx", "PTxy", "PTyx", "PTyy"),
}

# --- what the maps show ---
# One name or a list; each gets its own maps, tables and summary section:
#   "nrms", "r_mae", "r_med", "q95", "mean", "std", "skew", "kurt"
#                  fit measures of (obs - cal) / err (modules/misfit_measures.py)
#   "distortion"   strength of the galvanic distortion of a site, |C - I| (Frobenius
#                  norm of the matrix difference C' that FEMTIC estimates); one map,
#                  if the run has distortion estimates
SITE_MEASURES: Union[str, Sequence[str]] = "nrms"   # e.g. ["nrms", "r_mae", "distortion"]
# "distortion" of a run whose results file has no /distortion group: this text file
# next to the results file ({iter} = iteration; its type cannot be checked); None = no
DISTORTION_FILE: Optional[str] = "distortion_iter{iter}.dat"

# --- dimensionality (modules/dimens.py) ---
# Further SITE_MEASURES that describe the impedances, not the fit; one value per
# site and frequency band (BANDS, BAND_AXIS, BAND_PER_DECADE, BAND_MIN_N; the
# components of COMPONENTS and the DATATYPES filter do not enter):
#   "dim"     WAL class of the band (nine classes of WALDIM 1.2, and "undetermined")
#   "ptdim"   phase tensor class: 1D, 2D, 3D
#   "ptskew"  median |beta| of the phase tensor [deg]
#   "ellipt"  median ellipticity of the phase tensor
#   "swift"   median Swift skew
#   "bahr"    median Bahr phase-sensitive skew
DIM_SOURCE: Union[str, Sequence[str]] = "results"   # "results": the observed impedances of
                                      # the (first used) results file; else EDI directory,
                                      # EDI file or .npz collection, or a list of these
                                      # (relative to WORK_DIR; names = site names of site.dat)
DIM_PAGE: str = "all"                 # one figure per measure with a panel per band ("all"),
                                      # or one figure per measure and band ("band")
DIM_ERROR_PERCENT: Optional[float] = None   # error in percent of |Z| instead of the data errors
DIM_MISSING_ERROR_PERCENT: Optional[float] = 5.0   # DIM_SOURCE files without errors; None = stop
DIM_THRESHOLD: float = 0.15           # WAL invariants I3 - I7
DIM_Q_THRESHOLD: float = 0.10         # WAL invariant Q
DIM_PT_BETA_THRESHOLD: float = 3.0    # [deg] phase tensor: |beta| above it = 3D
DIM_PT_ELLIPTICITY_THRESHOLD: float = 0.1   # phase tensor: ellipticity above it = 2D, else 1D
DIM_OPTIONS: Dict[str, Any] = {}      # further keywords of dimens.classify_dimensionality, e.g.
                                      # dict(n_realizations=200, ft_convention="e-iwt",
                                      # edi_options=dict(manufacturer="phoenix"))
DIM_COLORS: Dict[str, str] = {}       # colours of classes, e.g. {"3D": "black"}

# --- basemap: extent and coordinates ---
MAP_EXTENT_MODE: str = "sites"        # "sites" (sites + margin), "box" or "center"
MAP_MARGIN_FRAC: float = 0.15         # "sites": margin as a fraction of the extent of the sites
MAP_MARGIN_MIN_KM: float = 2.0        # "sites": but at least this [km]
MAP_BOX: Optional[Tuple[float, float, float, float]] = None   # "box": (lon_min, lon_max, lat_min, lat_max) [deg]
MAP_CENTER: Optional[Tuple[float, float]] = None              # "center": (lon, lat) [deg]
MAP_SIZE_KM: Tuple[float, float] = (50.0, 50.0)               # "center": (width, height) [km]
DISPLAY_COORDS: str = "latlon"        # "latlon" (degrees) or "km" (km from the centre of the map)
MAP_GRID: bool = False
SCALE_BAR: bool = True
SCALE_BAR_KM: Optional[float] = None  # None = a round length of about 1/5 of the width

# --- DEM ---
# The relief is the hillshade of a digital elevation model. DEM_SOURCE:
#   "copernicus30"  Copernicus DEM GLO-30 (30 m, a surface model, heights above EGM2008): the
#                   Cloud Optimized GeoTIFF tiles of the AWS open data bucket copernicus-dem-30m,
#                   downloaded into DEM_CACHE_DIR and read with rasterio
#   "copernicus90"  Copernicus DEM GLO-90 (90 m; bucket copernicus-dem-90m): large maps
#   "skadi"         AWS Terrain Tiles (Mapzen): a composite of several DEMs in .hgt tiles
#                   (no rasterio needed)
#   "tiles"         tiles of your own: DEM_URL_TEMPLATE = URL or path pattern, GeoTIFF or .hgt
#   "file"          one DEM file, DEM_FILE: GeoTIFF (rasterio), .npz (lon, lat, z) or one .hgt tile
#   "none"          no relief
# Examples for the Copernicus DEM GLO-30 (copy the lines to the settings below):
#   DEM_SOURCE = "copernicus30"                      # the default: download once into DEM_CACHE_DIR
#   DEM_CACHE_DIR = "/data/dem_cache"                # a cache shared by all runs and maps
#   DEM_STYLE = "terrain"; DEM_CMAP = "gist_earth"   # colour + shading instead of the grey hillshade
#   DEM_MAX_PIXELS = 2400                            # finer relief of a small map (default 1600)
#   DEM_DOWNLOAD = False                             # offline: only the tiles that are in DEM_CACHE_DIR,
#                                                    # e.g. Copernicus_DSM_COG_10_N65_00_W017_00_DEM.tif
#   DEM_SOURCE = "tiles"
#   DEM_URL_TEMPLATE = "/data/cop30/Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"   # own folder / names
#   DEM_SOURCE = "file"; DEM_FILE = "/data/krafla_cop30.tif"   # one GeoTIFF (e.g. cut out with gdal_translate)
DEM_SOURCE: str = "copernicus30"
DEM_URL_TEMPLATE: Optional[str] = None   # None = the tiles of DEM_SOURCE; else a mirror, or the pattern of "tiles":
                                         # placeholders {name} N65W017, {NS}, {lat}, {EW}, {lon}; a path (relative: to WORK_DIR) is no download
DEM_DOWNLOAD: bool = True             # False: use only the tiles in DEM_CACHE_DIR
DEM_CACHE_DIR: str = "dem_cache"      # downloaded tiles are kept here (give an absolute path to share it between runs)
DEM_FILE: Optional[str] = None        # "file": GeoTIFF (needs rasterio), .npz (lon, lat, z) or one .hgt tile
DEM_TIMEOUT: float = 60.0             # [s] per tile
DEM_MAX_TILES: int = 16               # more tiles than this are refused (too large a map)
DEM_MAX_PIXELS: int = 1600            # the DEM is thinned out to at most this many pixels per side
DEM_FILL_MISSING: Optional[float] = 0.0   # elevation of tiles the server does not have (ocean); None = no value
DEM_STRICT: bool = False              # True: a failed download stops the script (else: map without relief)
DEM_STYLE: str = "terrain"          # "hillshade" (grey) or "terrain" (colour + shading, DEM_CMAP)
DEM_CMAP: str = "gist_earth"
HILLSHADE_AZIMUTH: float = 315.0
HILLSHADE_ALTITUDE: float = 45.0
HILLSHADE_VERT_EXAG: float = 1.0
BASEMAP_CMAP: str = "Greys"           # "hillshade": colours of the shading
BASEMAP_ALPHA: float = 0.9
DEM_CREDIT: Optional[str] = None    # text at the lower right of every map: "auto", None or text

# --- icons ---
SITE_MARKER: str = "o"
SITE_SIZE: float = 80.0               # [pt^2]
SITE_EDGECOLOR: str = "black"
SITE_LW: float = 0.8
SITE_NODATA_EDGECOLOR: str = "0.35"   # open icons: sites without data in the selection
SITE_LABELS: bool = True             # site names next to the icons
CMAP_NRMS: str = "jet"            # diverging colour map, centred on NRMS_REFERENCE
# CMAP_NRMS: str = "RdBu_r"             # diverging colour map, centred on NRMS_REFERENCE
CMAP_SPREAD: str = "YlOrRd"           # ENS_STAT = "std": sequential colour map, linear scale from 0
NRMS_REFERENCE: Optional[float] = 1.0   # target nRMS = centre colour; None = linear scale
NRMS_VMIN: Optional[float] = 0.6     # colour limits; None = from the data ("std": 0)
NRMS_VMAX: Optional[float] = 1.8
# colour scales of the other SITE_MEASURES, where they differ from the defaults:
#   {measure: dict(cmap=..., reference=..., vmin=..., vmax=...)}, any of the four keys
#   (reference = centre colour, None = linear scale; vmin / vmax None = from the data)
# defaults: r_mae, r_med, std as the nRMS settings above; q95 CMAP_NRMS about 1.96;
#   mean, skew, kurt "RdBu_r" about 0; distortion CMAP_SPREAD, linear from 0
# ENS_STAT = "std" (spread): every measure CMAP_SPREAD, linear from 0, MEASURE_SCALES unused
MEASURE_SCALES: Dict[str, Dict[str, Any]] = {}   # e.g. {"distortion": dict(vmax=0.5)}

# --- profiles and polylines ---
# Profile lines (cross-section lines) and other polylines / polygons (faults, coast
# lines, caldera rims, ...) on the maps; lists of dicts, see the readme.
#   profile: name, one of points | file | start + end | center + azimuth + length_km,
#            coords, width_km (swath), ticks_km, tick_labels, labels, label
#            start / end: (a, b) or (a, b, "A") = with the label ("marker") of that end; without
#            it labels = ("A", "A'") or, if there is none, the two halves of the name "A-A'";
#            labels=False = no text at the ends; label = entry in the legend
#   line:    label, one of file | points | segments, coords, closed, fill
#   coords:  how the number pairs are read:
#            "lonlat" (default) = (longitude, latitude), "latlon" = (latitude, longitude),
#            "utm" (easting, northing in km, zone of site.dat) or "model" (x, y in km of the
#            FEMTIC model); GeoJSON and KML are always lon, lat.
#            NOTE: in femtic_ens_plot_slices.py the tag "latlon" stands for [lon, lat]; a point
#            list copied from there needs coords="lonlat" here.
#            Only the numbers given here are km: "utm" / "model" coordinates read from a
#            text file (`file`) stay in m, as GIS programs and the results file write them
#   style:   every entry may set its own style; a key that is not given takes the setting below
#            (PROFILE_* / LINE_*):
#            both:     color, lw (or linewidth), ls (or linestyle), line_alpha, zorder, halo,
#                      halo_color, halo_width, marker, markersize (or ms)
#            profile:  end_marker, end_marker_size, fontsize, fontweight, text_color,
#                      label_offset, tick_length, tick_lw, tick_fontsize, swath_color,
#                      swath_alpha, mark_sites, ring_size, ring_lw
#            line:     alpha (opacity of the fill)
# PROFILES: Sequence[dict] = []         # e.g. [dict(name="A-A'", start=(-16.80, 65.68, "A"), end=(-16.70, 65.72, "A'"), width_km=4.0, ticks_km=5.0)]
#                                       # UTM: coords="utm", start=(405.0, 7291.8, "A"), end=(424.0, 7287.8, "A'")
#                                       # own style: color="magenta", lw=3.0, ls=":", line_alpha=0.9
PROFILES: Sequence[dict] = [
    # dict(name="A-A'", start=(405.0, 7291.8, "A"), end=(424.0, 7287.8, "A'"), coords="utm",
    #      width_km=1.0,) # ticks_km=5.0)
    dict(
        name="Vuache fault",
        start=(6.030869, 45.962166),          # (longitude, latitude)
        end=(6.115936, 45.910655),
        coords="lonlat",
        labels=False,                         # no text at the two ends
        label="Vuache fault",                 # entry in the legend
        # color="magenta", lw=3.0, ls=":",    # own style (else the PROFILE_* settings)
    ),
]

LINES: Sequence[dict] = [    ]            # e.g. [dict(label="Faults", file="gis/faults.txt", color="tab:red")]

# style of the profiles (an entry of PROFILES may override every one of these)
PROFILE_COLOR: str = "black"
PROFILE_LW: float = 1.8               # [pt]
PROFILE_LS: Any = "-"                 # "-", "--", "-.", ":" or (offset, (on, off, ...))
PROFILE_ALPHA: float = 1.0            # opacity of the line (key line_alpha)
PROFILE_ZORDER: float = 4.0           # drawing order; the sites are at 5: e.g. 5.5 = on top of them
PROFILE_HALO: bool = False            # outline under the line, the ticks and the texts
PROFILE_HALO_COLOR: str = "white"
PROFILE_HALO_WIDTH: float = 1.6       # [pt] the outline of the line is wider than the line by this
PROFILE_MARKER: Optional[str] = None  # symbol at every vertex of the profile; None = none
PROFILE_MARKER_SIZE: float = 4.0      # [pt]
PROFILE_END_MARKER: Optional[str] = None   # symbol ("o", "s", "D", "^", ...) at the two ends of every profile
                                      # (the ends of the cut part, see OVERLAY_CUT_MARGIN_KM); None = none
PROFILE_END_MARKER_SIZE: float = 6.0  # [pt]
PROFILE_FONTSIZE: float = 9.0         # [pt] end labels
PROFILE_FONTWEIGHT: str = "bold"      # end labels
PROFILE_TEXT_COLOR: Optional[str] = None   # end and distance labels; None = the colour of the line
PROFILE_LABEL_OFFSET: float = 10.0    # [pt] distance of an end label from the end
PROFILE_TICK_FONTSIZE: Optional[float] = None   # [pt] distance labels; None = PROFILE_FONTSIZE - 2
PROFILE_TICK_LENGTH: Optional[float] = None     # length of a tick as a fraction of the map width; None = 0.012
PROFILE_TICK_LW: Optional[float] = None         # [pt]; None = 0.8 x the line width, at least 0.8
PROFILE_SWATH_COLOR: Optional[str] = None       # shading of the swath (width_km); None = line colour
PROFILE_SWATH_ALPHA: float = 0.15
PROFILE_MARK_SITES: bool = False      # open circles around the sites in the swath
PROFILE_RING_SIZE: float = 1.9        # size of these circles, in units of SITE_SIZE
PROFILE_RING_LW: float = 1.0          # [pt]

# style of the lines (an entry of LINES may override every one of these)
LINE_COLOR: str = "tab:green"
LINE_LW: float = 1.2                  # [pt]
LINE_LS: Any = "-"
LINE_ALPHA: float = 1.0               # opacity of the line (key line_alpha)
LINE_ZORDER: float = 3.0              # the sites are at 5
LINE_HALO: bool = False               # outline under the lines
LINE_HALO_COLOR: str = "white"
LINE_HALO_WIDTH: float = 1.6          # [pt]
LINE_MARKER: Optional[str] = None     # symbol at every vertex; None = none
LINE_MARKER_SIZE: float = 4.0         # [pt]
LINE_FILL_ALPHA: float = 0.25         # opacity of the fill of a closed line (key alpha)

# outline of the other things on the maps: site names, scale bar and DEM credit
MAP_HALO: bool = False
MAP_HALO_COLOR: str = "white"
MAP_HALO_WIDTH: float = 1.5           # [pt] around the texts (scale bar: 1 pt more)

MAP_INCLUDE_PROFILES: bool = False    # MAP_EXTENT_MODE "sites": False = the map box does not depend on the profiles:
                                      # what goes beyond it is cut at it (OVERLAY_CUT_MARGIN_KM); True = the box is
                                      # made large enough for the profiles
OVERLAY_STRICT: bool = False          # True: a profile / line that cannot be made stops the script
OVERLAY_CUT_MARGIN_KM: Optional[float] = 1.0   # profiles and lines are cut at the map box, made smaller by this
                                      # margin [km] on every side, so that the ends of a cut profile (and its
                                      # labels) are inside the map; None = not cut (the map edge clips them)
LEGEND_LOC: Optional[str] = "upper left"   # legend (no-data icon, labelled profiles and lines); None = none

# --- fonts ---
# Sizes in pt. None = derived from TICK_FONTSIZE as noted (or matplotlib's default).
# The sizes of the profile texts are PROFILE_FONTSIZE and PROFILE_TICK_FONTSIZE above.
FONT_FAMILY: Union[None, str, Sequence[str]] = None   # None = matplotlib's; "serif", "sans-serif",
                                      # "monospace", or an installed font such as "DejaVu Serif"; a list
                                      # is a fall-back order, e.g. ["Arial", "DejaVu Sans"]
FONT_RC: Optional[Dict[str, Any]] = None   # further matplotlib rcParams for the text, e.g.
                                      # {"mathtext.fontset": "stix"} or {"font.style": "italic"}
TICK_FONTSIZE: float = 8.0            # tick labels of the maps (base of the sizes that are None)
AXIS_LABEL_FONTSIZE: Optional[float] = None    # "Longitude [deg]", ...; None = TICK_FONTSIZE
TITLE_FONTSIZE: Optional[float] = None         # figure title; None = TICK_FONTSIZE + 3
TITLE_FONTWEIGHT: str = "normal"
PANEL_TITLE_FONTSIZE: Optional[float] = None   # panel titles; None = TICK_FONTSIZE + 1
LABEL_FONTSIZE: float = 6.0           # site names
LABEL_FONTWEIGHT: str = "normal"
LEGEND_FONTSIZE: Optional[float] = None        # None = TICK_FONTSIZE - 1
CBAR_LABEL_FONTSIZE: Optional[float] = None    # colour bar title; None = matplotlib's (font.size)
CBAR_TICK_FONTSIZE: Optional[float] = None     # colour bar numbers; None = matplotlib's
SCALEBAR_FONTSIZE: Optional[float] = None      # None = TICK_FONTSIZE - 1
CREDIT_FONTSIZE: float = 5.0          # DEM credit line

# --- figures and output ---
PAGE_BY: str = "band"                 # one figure per "band" (panels: components), per "component"
                                      # (panels: bands), per "none" (one map each), or "all" (one figure)
PANEL_COLS: int = 2                   # panels per row
PANEL_SIZE: Tuple[float, float] = (12.2, 11.2)   # [cm] width, height of one panel (until 2026-11-26: inch)
TITLE: Optional[str] = "auto"         # "auto", None / "" (none) or text with {iter}, {nrms}, {M}, {stat}
CBAR_LOCATION: Optional[str] = None   # None / "right", "left", "top", "bottom"
CBAR_SHRINK: float = 0.7              # length of the colour bar: fraction (0 .. 1] of the panels along it
CBAR_LENGTH: Optional[float] = None   # [cm] length of the colour bar; None = CBAR_SHRINK x the panels
CBAR_WIDTH: Optional[float] = None    # [cm] width (thickness); None = 1/20 of the length
DPI: int = 400
SHOW: bool = False
OUTPUT_DIR: str = "sitemap"           # relative to WORK_DIR
OUTPUT_BASENAME: str = "femtic_sitemap"
OUTPUT_FORMATS: Sequence[str] = ["png", "pdf"]

# ---------------------------------------------------------------------------
# End of the configuration
# ---------------------------------------------------------------------------
_SETTING_NAMES = tuple(k for k in list(globals())
                       if k.isupper() and not k.startswith("_") and k != "PY4MTX_ROOT")

_TAG = "femtic_plot_sitemap:"
_USER_AGENT = "py4mt-femtic_plot_sitemap"    # of the DEM tile downloads
_KM_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0 / 1000.0     # one degree of latitude [km]
_PAGE_BY = ("band", "component", "none", "all")
_DISTORTION = "distortion"           # the measure that is no fit measure
_DISTORTION_NAMES = ("distortion", "dist", "distortion_strength")
_DISTORTION_LABEL = "distortion strength |C - I|"
_SCALE_KEYS = ("cmap", "reference", "vmin", "vmax")
_DIM_CLASS_MEASURES = ("dim", "ptdim")            # categorical maps
_DIM_VALUES = {                                    # measure: band value, scale maximum
    "ptskew": ("pt_abs_beta_deg", 10.0),
    "ellipt": ("pt_ellipticity_median", 0.5),
    "swift": ("swift_skew", 0.5),
    "bahr": ("bahr_skew", 0.5),
}
_DIM_MEASURES = _DIM_CLASS_MEASURES + tuple(_DIM_VALUES)
_DIM_ALIASES = {"dimensionality": "dim", "waldim": "dim"}
_DIM_LABELS = {
    "dim": "dimensionality (WAL class)",
    "ptdim": "phase tensor dimensionality",
    "ptskew": "median |beta| [deg]",
    "ellipt": "median phase tensor ellipticity",
    "swift": "median Swift skew",
    "bahr": "median Bahr skew",
}
_DIM_CODES = {                                     # classes on the colour bar
    "dim": (0, 1, 2, 4, 5, 6, 7, 8, 9),            # 3D/2Dtw counts as 3D/2D in a band
    "ptdim": (0, 1, 2, 3),
}
_DIM_CLASS_COLORS = {
    "undetermined": "#bdbdbd", "undefined": "#bdbdbd", "1D": "#2c7bb6", "2D": "#1a9641",
    "3D/2D": "#fee08b", "3D": "#d7191c", "3D/2Ddiag": "#fdae61", "3D/1D2D": "#7b3294",
    "ANIS-HINT1": "#a6611a", "ANIS-HINT2": "#f781bf",
}
_DIM_SETTINGS = {                                  # setting: keyword of classify_dimensionality
    "DIM_ERROR_PERCENT": "error_percent",
    "DIM_MISSING_ERROR_PERCENT": "missing_error_percent",
    "DIM_THRESHOLD": "threshold",
    "DIM_Q_THRESHOLD": "q_threshold",
    "DIM_PT_BETA_THRESHOLD": "pt_beta_threshold",
    "DIM_PT_ELLIPTICITY_THRESHOLD": "pt_ellipticity_threshold",
}
_DIM_NO_OPTIONS = ("bands", "bands_per_decade", "band_edges", "band_domain")   # BANDS counts
_OHM_PER_FIELD_UNIT = 4.0e-7 * math.pi * 1.0e3     # Z [Ohm] = this x Z [mV/km/nT]
_CBAR_LOCATIONS = (None, "right", "left", "top", "bottom")


def _say(msg: str) -> None:
    print(f"{_TAG} {msg}")


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
def resolve_work_dir(work_dir: Optional[str]) -> str:
    """Absolute WORK_DIR; None or "" = the current directory; ~ is expanded."""
    return os.path.abspath(os.path.expanduser(str(work_dir or ".")))


def anchor(path: str, work_dir: str) -> str:
    """`path` with ~ expanded; a relative path is joined to `work_dir`."""
    p = os.path.expanduser(str(path))
    return p if os.path.isabs(p) else os.path.join(work_dir, p)


def settings(work_dir: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
    """
    The settings of this script as a dict (uppercase names), with `overrides`
    applied and the paths anchored at WORK_DIR (`work_dir`, else the setting).
    A name that is no setting is an error (typo).
    """
    cfg = {k: globals()[k] for k in _SETTING_NAMES}
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"{_TAG} unknown setting {k!r}")
        cfg[k] = v
    wd = resolve_work_dir(cfg["WORK_DIR"] if work_dir is None else work_dir)
    cfg["WORK_DIR"] = wd
    for key in ("RESULTS_PATH", "TEMPLATES_DIR", "DEM_CACHE_DIR", "OUTPUT_DIR"):
        cfg[key] = anchor(cfg[key], wd)
    for key in ("DEM_FILE",):
        if cfg[key]:
            cfg[key] = anchor(cfg[key], wd)
    if cfg["DEM_URL_TEMPLATE"] and not is_url(cfg["DEM_URL_TEMPLATE"]):
        cfg["DEM_URL_TEMPLATE"] = anchor(cfg["DEM_URL_TEMPLATE"], wd)        # path pattern
    if cfg["SITE_DAT"] and cfg["SITE_DAT"] != "auto":
        cfg["SITE_DAT"] = anchor(cfg["SITE_DAT"], wd)
    src = cfg["DIM_SOURCE"]
    if src != "results" and src:
        if isinstance(src, (str, os.PathLike)):
            cfg["DIM_SOURCE"] = anchor(src, wd)
        else:
            cfg["DIM_SOURCE"] = [anchor(p, wd) for p in src]
    runs = cfg["RUN_PATHS"]
    if runs:
        cfg["RUN_PATHS"] = [anchor(p, wd) for p in ([runs] if isinstance(runs, str) else runs)]
    else:
        cfg["RUN_PATHS"] = None
    for key in ("PROFILES", "LINES"):                 # copies, with the files anchored
        items = cfg[key]
        items = [items] if isinstance(items, dict) else (list(items) if items else [])
        cfg[key] = [dict(it, file=anchor(it["file"], wd)) if isinstance(it, dict) and it.get("file")
                    else (dict(it) if isinstance(it, dict) else it) for it in items]
    return cfg


def check_settings(cfg: Dict[str, Any]) -> None:
    """Raise ValueError for a setting that cannot work (before any data is read)."""
    def one_of(key: str, allowed: Sequence[Any]) -> None:
        if cfg[key] not in allowed:
            raise ValueError(f"{key} = {cfg[key]!r}: must be one of {list(allowed)}")
    one_of("SITE_MATCH", ("auto", "coords", "id"))
    tol = cfg["SITE_MATCH_TOL"]
    if isinstance(tol, (int, float)) and tol >= 1.0:
        raise ValueError(f"SITE_MATCH_TOL = {tol!r}: given in km (0.015 = 15 m); this "
                         "looks like metres")
    one_of("BAND_AXIS", ("period", "frequency"))
    one_of("MAP_EXTENT_MODE", ("sites", "box", "center"))
    one_of("DISPLAY_COORDS", ("latlon", "km"))
    if cfg["DEM_SOURCE"] == "srtm":
        raise ValueError("DEM_SOURCE = 'srtm' does not exist any more: use 'copernicus30' (the "
                         "default), 'copernicus90', or 'skadi' (the AWS Terrain Tiles that 'srtm' "
                         "meant)")
    one_of("DEM_SOURCE", DEM_SOURCES)
    one_of("DEM_STYLE", DEM_STYLES)
    one_of("PAGE_BY", _PAGE_BY)
    one_of("CBAR_LOCATION", _CBAR_LOCATIONS)
    for key in ("CBAR_LENGTH", "CBAR_WIDTH"):
        v = cfg[key]
        if v is not None and not (isinstance(v, (int, float, np.integer, np.floating))
                                  and not isinstance(v, bool) and np.isfinite(v) and v > 0):
            raise ValueError(f"{key} = {v!r}: must be None or a number > 0 (cm)")
    check_appearance(cfg)
    mg = cfg["OVERLAY_CUT_MARGIN_KM"]
    if mg is not None and not (isinstance(mg, (int, float, np.integer, np.floating))
                               and not isinstance(mg, bool) and np.isfinite(mg) and mg >= 0):
        raise ValueError(f"OVERLAY_CUT_MARGIN_KM = {mg!r}: must be None or a number >= 0 (km)")
    sh = cfg["CBAR_SHRINK"]
    if not (isinstance(sh, (int, float, np.integer, np.floating)) and not isinstance(sh, bool)
            and 0.0 < sh <= 1.0):
        raise ValueError(f"CBAR_SHRINK = {sh!r}: must be a number in (0, 1]")
    if cfg["MAP_EXTENT_MODE"] == "box":
        b = cfg["MAP_BOX"]
        if b is None or len(b) != 4 or not (b[0] < b[1] and b[2] < b[3]):
            raise ValueError("MAP_BOX must be (lon_min, lon_max, lat_min, lat_max) "
                             "with lon_min < lon_max and lat_min < lat_max")
    if cfg["MAP_EXTENT_MODE"] == "center":
        c, s = cfg["MAP_CENTER"], cfg["MAP_SIZE_KM"]
        if c is None or len(c) != 2:
            raise ValueError("MAP_CENTER must be (lon, lat) for MAP_EXTENT_MODE = \"center\"")
        if len(s) != 2 or not (s[0] > 0 and s[1] > 0):
            raise ValueError("MAP_SIZE_KM must be (width, height) in km, both > 0")
    if cfg["DEM_SOURCE"] == "file" and not cfg["DEM_FILE"]:
        raise ValueError("DEM_SOURCE = \"file\" needs DEM_FILE")
    if cfg["DEM_SOURCE"] == "tiles" and not cfg["DEM_URL_TEMPLATE"]:
        raise ValueError("DEM_SOURCE = \"tiles\" needs DEM_URL_TEMPLATE")
    if cfg["DEM_SOURCE"] not in ("file", "none"):
        tile_kind(dem_template(cfg["DEM_SOURCE"], cfg["DEM_URL_TEMPLATE"]))      # ValueError
    if int(cfg["PANEL_COLS"]) < 1:
        raise ValueError("PANEL_COLS must be >= 1")
    if int(cfg["DEM_MAX_PIXELS"]) < 2:
        raise ValueError("DEM_MAX_PIXELS must be >= 2")
    parse_stat(cfg["ENS_STAT"])
    if cfg["MEMBER_NRMS_MAX"] is not None and not float(cfg["MEMBER_NRMS_MAX"]) > 0:
        raise ValueError("MEMBER_NRMS_MAX must be > 0 (or None)")
    one_of("LEGEND_LOC", _LEGEND_LOCS)
    check_measure_settings(cfg)
    check_dim_settings(cfg)
    check_overlay_specs(cfg)


# ---------------------------------------------------------------------------
# Measures: what the maps show, and their colour scales
# ---------------------------------------------------------------------------
def parse_measures(spec: Any) -> Tuple[str, ...]:
    """
    SITE_MEASURES as a tuple of names, in the order given and without repeats: the
    fit measures of misfit_measures (aliases "rms", "rmae", "rmed", "bias"),
    "distortion" (also "dist") and the dimensionality measures "dim" (also
    "dimensionality"), "ptdim", "ptskew", "ellipt", "swift", "bahr". A name,
    "a, b" or a list; None = ("nrms",).
    ValueError for an unknown name or an empty list.
    """
    if spec is None:
        return ("nrms",)
    names = [t for t in re.split(r"[,;\s]+", spec.strip()) if t] if isinstance(spec, str) else spec
    allowed = ", ".join(mm.FIT_MEASURE_NAMES + (_DISTORTION,) + _DIM_MEASURES)
    bad = f"SITE_MEASURES = {spec!r}: one or more of {allowed}"
    try:
        names = [str(t).strip().lower() for t in names]
    except TypeError:
        raise ValueError(bad) from None
    out: List[str] = []
    for t in names:
        t = _DIM_ALIASES.get(t, t)
        if t in _DISTORTION_NAMES:
            t = _DISTORTION
        elif t not in _DIM_MEASURES:
            try:
                t = mm.site_measures([t])[0]
            except ValueError:
                raise ValueError(bad) from None
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError(bad)
    return tuple(out)


def fit_names(measures: Sequence[str]) -> Tuple[str, ...]:
    """The fit measures of `measures` that are computed besides nRMS."""
    return tuple(m for m in measures if m not in ("nrms", _DISTORTION) + _DIM_MEASURES)


def dim_names(measures: Sequence[str]) -> Tuple[str, ...]:
    """The dimensionality measures of `measures`."""
    return tuple(m for m in measures if m in _DIM_MEASURES)


def measure_label(measure: str) -> str:
    """"nRMS", "rMAE", ..., "distortion strength |C - I|", "median Swift skew", ..."""
    if measure in _DIM_MEASURES:
        return _DIM_LABELS[measure]
    return _DISTORTION_LABEL if measure == _DISTORTION else mm.MEASURE_LABELS[measure]


def measure_text(value: float, measure: str) -> str:
    """A value for a panel title: 2 decimals, 3 for mean, skew and kurt (never "-0.000")."""
    text = f"{value:.{3 if measure in mm.SIGNED_MEASURES else 2}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def sel_values(s: dict, measure: str = "nrms") -> Tuple[float, np.ndarray]:
    """(global value, value per site) of `measure` in the selection `s`."""
    if measure == "nrms":
        return s["nrms"], s["nrms_site"]
    return s["glob_m"][measure], s["site_m"][measure]


def _scales_of(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """MEASURE_SCALES with canonical measure names; ValueError if it is malformed."""
    spec = cfg["MEASURE_SCALES"] or {}
    if not isinstance(spec, dict):
        raise ValueError("MEASURE_SCALES must be a dict {measure: dict(cmap, reference, vmin, vmax)}")
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in spec.items():
        try:
            name = parse_measures([key])[0]
        except ValueError:
            raise ValueError(f"MEASURE_SCALES: unknown measure {key!r}") from None
        if name in _DIM_CLASS_MEASURES:
            raise ValueError(f"MEASURE_SCALES[{key!r}]: a map of classes has no colour "
                             "scale; set colours with DIM_COLORS")
        if not isinstance(val, dict) or any(k not in _SCALE_KEYS for k in val):
            raise ValueError(f"MEASURE_SCALES[{key!r}] must be a dict with keys of "
                             f"{list(_SCALE_KEYS)}")
        out[name] = dict(val)
    return out


def check_measure_settings(cfg: Dict[str, Any]) -> None:
    """ValueError for a SITE_MEASURES, MEASURE_SCALES or DISTORTION_FILE that cannot work."""
    parse_measures(cfg["SITE_MEASURES"])
    scales = _scales_of(cfg)
    if scales:
        import matplotlib
        for name, val in scales.items():
            what = f"MEASURE_SCALES[{name!r}]"
            cmap = val.get("cmap")
            if "cmap" in val and not (isinstance(cmap, matplotlib.colors.Colormap)
                                      or cmap in matplotlib.colormaps):
                raise ValueError(f"{what}: cmap {cmap!r} is no matplotlib colour map")
            for k in ("reference", "vmin", "vmax"):
                v = val.get(k)
                if v is not None and not (isinstance(v, (int, float, np.integer, np.floating))
                                          and not isinstance(v, bool) and np.isfinite(v)):
                    raise ValueError(f"{what}: {k} = {v!r} must be None or a number")
    tpl = cfg["DISTORTION_FILE"]
    if tpl is not None:
        try:
            ok = isinstance(tpl, str) and bool(tpl.format(iter=1))
        except (KeyError, IndexError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"DISTORTION_FILE = {tpl!r}: None or a file name, {{iter}} = "
                             "iteration")


def measure_scale(measure: str, cfg: Dict[str, Any], *, spread: bool = False) -> Dict[str, Any]:
    """
    dict(cmap, reference, vmin, vmax) of the colour scale of `measure`. `spread` =
    ENS_STAT "std": CMAP_SPREAD, linear from 0 (nRMS keeps NRMS_VMIN / NRMS_VMAX).
    Else the default of the measure (see MEASURE_SCALES) updated by MEASURE_SCALES.
    """
    nrms = dict(cmap=cfg["CMAP_NRMS"], reference=cfg["NRMS_REFERENCE"], vmin=cfg["NRMS_VMIN"],
                vmax=cfg["NRMS_VMAX"])
    if measure in _DIM_VALUES:        # from the observed data: no ensemble spread
        out = dict(cmap=cfg["CMAP_SPREAD"], reference=None, vmin=0.0,
                   vmax=_DIM_VALUES[measure][1])
        out.update(_scales_of(cfg).get(measure, {}))
        return out
    if spread:
        if measure != "nrms":
            return dict(cmap=cfg["CMAP_SPREAD"], reference=None, vmin=0.0, vmax=None)
        return dict(nrms, cmap=cfg["CMAP_SPREAD"], reference=None,
                    vmin=0.0 if nrms["vmin"] is None else nrms["vmin"])
    if measure in ("nrms", "r_mae", "r_med", "std"):          # all expected to be 1
        out = nrms
    elif measure == "q95":
        out = dict(cmap=cfg["CMAP_NRMS"], reference=1.96, vmin=None, vmax=None)
    elif measure in mm.SIGNED_MEASURES:
        out = dict(cmap="RdBu_r", reference=0.0, vmin=None, vmax=None)
    else:                                                      # distortion strength
        out = dict(cmap=cfg["CMAP_SPREAD"], reference=None, vmin=0.0, vmax=None)
    out.update(_scales_of(cfg).get(measure, {}))
    return out


# ---------------------------------------------------------------------------
# Distortion strength
# ---------------------------------------------------------------------------
def distortion_strength(c_prime: Any) -> np.ndarray:
    """
    Strength of galvanic distortion |C - I|: the Frobenius norm of the matrix
    difference C' = C - I, for an array [..., 2, 2]; 0 = no distortion.
    """
    c = np.asarray(c_prime, float)
    return np.sqrt(np.sum(c * c, axis=(-2, -1)))


def read_distortion_text(path: str) -> Dict[int, Tuple[np.ndarray, bool]]:
    """
    {site id: (C' (2, 2), fixed)} of a distortion_iter<N>.dat (first line the
    number of sites, then: site id, C'xx, C'xy, C'yx, C'yy and, if there, the
    flag "fixed"); the matrices as femtic.read_distortion_file reads them.
    """
    _, c_prime = fm.fem.read_distortion_file(path)
    with open(path, "r", errors="replace") as fh:
        rows = [ln.split() for ln in fh.read().strip().splitlines()[1:]]
    if len(rows) != len(c_prime):
        raise ValueError(f"{len(rows)} lines but {len(c_prime)} matrices")
    return {int(p[0]): (np.asarray(c_prime[k], float),
                        len(p) > 5 and int(float(p[5])) != 0) for k, p in enumerate(rows)}


def read_run_distortion(results_path: str, iteration: int, *, file_template: Optional[str]
                        ) -> Tuple[Optional[Dict[int, Tuple[np.ndarray, bool]]], str]:
    """
    The distortion of one run: ({site id: (C' (2, 2), fixed)}, source) or (None, why
    there is none). First the /distortion group of the results file; without it the
    text file `file_template` ({iter} = `iteration`) next to the results file, whose
    type cannot be checked. Only the type "matrix difference" is used.
    """
    note = ""
    try:
        res = fm.fem.read_results_distortion(results_path, out=False)
    except Exception as exc:              # no h5py, damaged group, ...: try the text file
        res = None
        note = f" (/distortion of {os.path.basename(results_path)} not readable: {exc})"
    if res is not None:
        kind = int(res["type"])
        if kind == 0:
            return None, "the run has no distortion estimation"
        if kind != 1:
            return None, (f"distortion type {res['type_name']} is not implemented (only "
                          "ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)")
        out = {int(r["site_id"]): (np.array([[r["param1"], r["param2"]],
                                             [r["param3"], r["param4"]]], float),
                                   bool(r["isFixed"])) for r in res["rows"]}
        if not out:
            return None, "the /distortion group of the results file is empty"
        return out, "/distortion of the results file"
    if not file_template:
        return None, "no /distortion group in the results file (DISTORTION_FILE = None)" + note
    p = os.path.join(os.path.dirname(os.path.abspath(results_path)),
                     file_template.format(iter=iteration))
    if not os.path.isfile(p):
        return None, (f"no /distortion group in the results file and no {os.path.basename(p)}"
                      + note)
    try:
        out = read_distortion_text(p)
    except Exception as exc:
        return None, f"{os.path.basename(p)} not readable: {exc}"
    if not out:
        return None, f"{os.path.basename(p)} holds no site"
    shown = file_template.replace("{iter}", "<N>")
    return out, f"text file {shown}, type not checked" + note


def distortion_per_site(dist: Dict[int, Tuple[np.ndarray, bool]], id_to_site: Dict[int, int],
                        n_sites: int) -> dict:
    """
    The distortion of the site ids on the sites: dict(strength (n_sites, NaN = none),
    cprime (n_sites, 2, 2), fixed (n_sites, bool), unmatched (ids without a site),
    n_multi (sites with several ids: their mean is taken)).
    """
    found: Dict[int, List[Tuple[np.ndarray, bool]]] = {}
    unmatched: List[int] = []
    for sid_, (cp, fixed) in dist.items():
        j = id_to_site.get(sid_)
        if j is None:
            unmatched.append(sid_)
        elif np.isfinite(cp).all():
            found.setdefault(j, []).append((cp, fixed))
    strength = np.full(n_sites, np.nan)
    cprime = np.full((n_sites, 2, 2), np.nan)
    is_fixed = np.zeros(n_sites, dtype=bool)
    for j, items in found.items():
        stack = np.array([cp for cp, _ in items])
        strength[j] = float(np.mean(distortion_strength(stack)))
        cprime[j] = stack.mean(axis=0)
        is_fixed[j] = all(f for _, f in items)
    return dict(strength=strength, cprime=cprime, fixed=is_fixed, unmatched=sorted(unmatched),
                n_multi=sum(1 for v in found.values() if len(v) > 1))


def distortion_selection(members: Sequence[dict], stat: str, used: np.ndarray, *,
                         ens: bool) -> Optional[dict]:
    """
    The distortion strength as a selection for plot_page and the tables, from the
    members with "dist" (distortion_per_site); None if no site in `used` has a value.
    Single run: the strength per site, global = median of the sites. Ensemble: `stat`
    over the members per site; global = `stat` over the members of the site median
    of each member (not the median of the mapped values). X (members x sites), M
    (members with distortion).
    """
    have = [m for m in members if m.get("dist") is not None]
    if not have:
        return None
    x = np.vstack([m["dist"]["strength"] for m in have])
    n_sites = x.shape[1]
    med = np.array([np.median(r[used][np.isfinite(r[used])]) if np.isfinite(r[used]).any()
                    else np.nan for r in x])
    if ens:
        value, glob = ens_statistic(x, stat), float(ens_statistic(med, stat))
    else:
        value, glob = x[0], float(med[0])
    n = int(np.isfinite(x[:, used]).any(axis=0).sum())
    if n == 0:
        return None
    if not ens:
        sub = f"median of the sites = {glob:.3f}, {n} sites"
    elif parse_stat(stat)[0] == "std":
        sub = f"std of the site median = {glob:.3f} (M = {len(have)}), {n} sites"
    else:
        sub = (f"site median = {glob:.3f} ({stat_words(stat)} of M = {len(have)} members), "
               f"{n} sites")
    return dict(band="distortion", comp="strength", head=_DISTORTION_LABEL, n=n,
                nrms=float("nan"), nrms_site=np.full(n_sites, np.nan),
                glob_m={_DISTORTION: glob}, site_m={_DISTORTION: value},
                sub_m={_DISTORTION: sub}, X=x, M=len(have))


def format_distortion(sel: dict, used: np.ndarray, *, ens: bool, stat: str,
                      members: Sequence[dict]) -> str:
    """Plain-text section of the summary about the distortion strength."""
    v = sel["site_m"][_DISTORTION][used]
    v = v[np.isfinite(v)]
    have = [m for m in members if m.get("dist") is not None]
    fixed = np.all([m["dist"]["fixed"] | ~np.isfinite(m["dist"]["strength"]) for m in have],
                   axis=0) & np.isfinite(sel["X"]).any(axis=0)
    cols = ("sites", "min", "median", "max", "fixed")
    row = (str(v.size), _fmt(float(v.min()), 3) if v.size else "",
           _fmt(float(np.median(v)), 3) if v.size else "",
           _fmt(float(v.max()), 3) if v.size else "", str(int(fixed[used].sum())))
    w = [max(len(c), len(r)) for c, r in zip(cols, row)]
    line = "-" * (sum(w) + 2 * (len(w) - 1))
    head = ["distortion strength |C - I| of the sites (Frobenius norm of the estimated matrix "
            "difference C')",
            "source: " + "; ".join(sorted({m["dist_source"] for m in have}))
            + (f"; {len(have)} of {len(members)} members, statistic: {stat_words(stat)}"
               if ens else "")]
    if ens:
        head.append(f"site median of each member, {stat_words(stat)} over the members: "
                    f"{_fmt(sel['glob_m'][_DISTORTION], 3)}; table below: the mapped site values")
    out = head + ["", "=" * len(line), "  ".join(c.rjust(w[i]) for i, c in enumerate(cols)), line,
                  "  ".join(c.rjust(w[i]) for i, c in enumerate(row)), line,
                  "fixed = sites whose matrix was held fixed (not estimated) in the inversion"
                  + (", in every member that has them." if ens else ".")]
    return "\n".join(out) + "\n"


def write_distortion_table(stem: str, rows: Sequence[dict], sel: dict, used: np.ndarray,
                           members: Sequence[dict], *, ens: bool) -> str:
    """
    <stem>_distortion.csv: one row per site with a value: site, lat, lon, site_ids,
    distortion (the mapped value). Single run: also C' (cp_xx ... cp_yy) and fixed
    (1 / 0). Ensemble: also M, mean, median, std (ddof = 1), min, max of the member
    strengths and n_fixed (members in which the matrix was fixed).
    """
    p = f"{stem}_distortion.csv"
    kinds = ("mean", "median", "std", "min", "max")
    have = [m for m in members if m.get("dist") is not None]
    x = sel["X"]
    cnt = np.sum(np.isfinite(x), axis=0)
    stats_ = {k: ens_statistic(x, k) for k in kinds} if ens else {}
    n_fixed = np.sum([m["dist"]["fixed"] for m in have], axis=0)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "lat", "lon", "site_ids", "distortion"]
                   + (["M"] + list(kinds) + ["n_fixed"] if ens else
                      ["cp_xx", "cp_xy", "cp_yx", "cp_yy", "fixed"]))
        for j in used:
            if cnt[j] == 0:
                continue
            r = rows[j]
            out = [r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                   " ".join(str(i) for i in r.get("ids", [])),
                   _fmt(sel["site_m"][_DISTORTION][j])]
            if ens:
                out += [int(cnt[j])] + [_fmt(stats_[k][j]) for k in kinds] + [int(n_fixed[j])]
            else:
                d = have[0]["dist"]
                out += [_fmt(float(c)) for c in d["cprime"][j].ravel()] + [int(d["fixed"][j])]
            w.writerow(out)
    return p


# ---------------------------------------------------------------------------
# Sites: site.dat and the assignment of the site ids
# ---------------------------------------------------------------------------
def read_site_dat(path: str) -> List[Dict[str, Any]]:
    """
    Sites of a site.dat of mt_make_sitelist.py: columns name, lat, lon, elev,
    sitenum, easting, northing (comma or white-space separated; lines that
    do not fit, and comments, are skipped). Returns a list of dicts with these
    keys. ValueError if there is no site, or lat / lon are out of range.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = [x.strip() for x in (ln.split(",") if "," in ln else ln.split())]
            if len(p) < 7:
                continue
            try:
                out.append(dict(name=p[0], lat=float(p[1]), lon=float(p[2]),
                                elev=float(p[3]), sitenum=int(float(p[4])),
                                easting=float(p[5]), northing=float(p[6])))
            except ValueError:
                continue
    if not out:
        raise ValueError(f"no site (name, lat, lon, elev, sitenum, easting, northing) in {path}")
    lat = np.array([s["lat"] for s in out])
    lon = np.array([s["lon"] for s in out])
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise ValueError(f"{path}: lat / lon out of range (columns name, lat, lon, ... "
                         "expected, in degrees)")
    return out


def site_search_dirs(templates_dir: str, work_dir: str) -> List[str]:
    """
    Directories searched for site.dat: TEMPLATES_DIR (inside WORK_DIR), WORK_DIR
    itself, and a directory of the same name next to WORK_DIR (the layout of an
    ensemble: <ensemble>/templates and <ensemble>/<member>).
    """
    wd = os.path.abspath(work_dir)
    sibling = os.path.join(os.path.dirname(wd), os.path.basename(os.path.normpath(templates_dir)))
    out: List[str] = []
    for d in (templates_dir, wd, sibling):
        if d not in out:
            out.append(d)
    return out


def find_site_dat(spec: Optional[str], search_dirs: Sequence[str]) -> Optional[str]:
    """spec "auto": the first site.dat in `search_dirs`; a path: as given; None / "": None."""
    if not spec:
        return None
    if spec != "auto":
        return spec if os.path.isfile(spec) else None
    for d in search_dirs:
        p = os.path.join(d, "site.dat")
        if os.path.isfile(p):
            return p
    return None


def spread_colocated(info: Dict[int, dict], id_to_site: Dict[int, int],
                     sites: Sequence[Dict[str, Any]], tol: float) -> int:
    """
    Several site ids at exactly one position (in the results file) were all given
    the same nearest site.dat entry. If site.dat has several entries within `tol`
    of that one (two stations at one place), the ids, in ascending order, get
    these entries in order of their sitenum, skipping entries that other ids use;
    ids left over stay on the nearest entry. Changes `id_to_site` in place and
    returns the number of ids moved.
    """
    groups: Dict[Tuple[float, float], List[int]] = {}
    for sid_, p in info.items():
        if sid_ in id_to_site:
            groups.setdefault((round(p["x"], 2), round(p["y"], 2)), []).append(sid_)
    moved = 0
    for ids in groups.values():
        ids = sorted(ids)
        j0 = id_to_site[ids[0]]
        if len(ids) < 2 or any(id_to_site[i] != j0 for i in ids):
            continue
        e0, n0 = float(sites[j0]["easting"]), float(sites[j0]["northing"])
        used = {j for i, j in id_to_site.items() if i not in ids}
        cand = sorted((k for k, s_ in enumerate(sites)
                       if k not in used and math.hypot(float(s_["easting"]) - e0,
                                                       float(s_["northing"]) - n0) <= tol),
                      key=lambda k: (int(sites[k]["sitenum"]), k))
        for i, k in zip(ids, cand):
            if id_to_site[i] != k:
                id_to_site[i] = k
                moved += 1
    return moved


def assign_sites(ds: dict, sites: Sequence[Dict[str, Any]], *, match: str = "auto",
                 tol: float = 15.0, offset: int = 0) -> Tuple[np.ndarray, dict]:
    """
    Index of the site.dat site (position in `sites`) for every row of `ds`,
    -1 = not assigned.

    match "coords": by position (femtic_misfit.match_sites_by_coords, using the
    first row of each site id; data types without a station position, HTF and
    NMT, are never assigned); "id": site_id = sitenum + offset; "auto": by
    position, by id only if no row has a position. Several site ids at one
    place (MT and VTF) get the same site, unless site.dat has several entries
    there (two stations at one place): then the ids, in ascending order, get
    these entries in order of their sitenum (spread_colocated).

    Returns (site_index, stats) with stats = n_ids, n_hit, n_sites, n_rows,
    n_unassigned_rows, mode, id_to_site ({site id: index}), unmatched (ids),
    and for "coords" also frame, shift, dmax, n_spread (ids moved by
    spread_colocated).
    """
    sid = np.asarray(ds["site_id"]).astype(np.int64)
    dn = np.asarray(ds["datatype_name"])
    x = np.asarray(ds["site_x"], float)
    y = np.asarray(ds["site_y"], float)
    station = ~np.isin(dn, fm._NO_STATION_POSITION)        # rows that belong to one station
    has_pos = station & np.isfinite(x) & np.isfinite(y)

    ids_u = np.unique(sid[station])
    info: Dict[int, dict] = {}
    if has_pos.any():
        u, first = np.unique(sid[has_pos], return_index=True)
        rows = np.flatnonzero(has_pos)[first]
        info = {int(s): dict(x=float(x[r]), y=float(y[r])) for s, r in zip(u, rows)}

    mode = match
    if mode == "auto":
        mode = "coords" if info else "id"
    stats: Dict[str, Any] = dict(mode=mode, frame="", shift=(float("nan"), float("nan")),
                                 dmax=float("nan"))
    id_to_site: Dict[int, int] = {}
    if mode == "coords":
        sites_xy = [(str(j), int(s["sitenum"]), float(s["easting"]), float(s["northing"]))
                    for j, s in enumerate(sites)]
        names, st = fm.match_sites_by_coords(info, sites_xy, float(tol))
        id_to_site = {int(k): int(v) for k, v in names.items()}
        stats.update(frame=st["frame"], shift=st["shift"], dmax=st["dmax"],
                     n_spread=spread_colocated(info, id_to_site, sites, float(tol)))
    else:
        by_num: Dict[int, int] = {}
        for j, s in enumerate(sites):
            by_num.setdefault(int(s["sitenum"]) + int(offset), j)
        id_to_site = {int(s): by_num[int(s)] for s in ids_u if int(s) in by_num}

    uniq, inv = np.unique(sid, return_inverse=True)
    lut = np.array([id_to_site.get(int(u_), -1) for u_ in uniq], dtype=np.int64)
    site_index = lut[inv]
    site_index[~station] = -1
    stats.update(
        n_ids=int(ids_u.size), n_hit=int(sum(1 for s in ids_u if int(s) in id_to_site)),
        n_sites=len(set(id_to_site.values())), n_rows=int(sid.size),
        n_unassigned_rows=int(np.count_nonzero(site_index < 0)),
        id_to_site=id_to_site,
        unmatched=[int(s) for s in ids_u if int(s) not in id_to_site])
    return site_index, stats


# ---------------------------------------------------------------------------
# nRMS: residuals, selections, reduction
# ---------------------------------------------------------------------------
def row_residuals(ds: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Error-normalized residuals (obs - cal) / err of the real and the imaginary
    parts, one value per row; NaN where not usable (non-finite value, err <= 0,
    and the imaginary part of real-valued data types). The same arithmetic as
    femtic_misfit.site_nrms (calculated response = ref_re / ref_im).
    """
    n = np.asarray(ds["freq"]).size
    cpx = ~np.isin(ds["datatype_name"], fm.REAL_ONLY_TYPES)
    out = []
    for ko, kr, ke, msk in (("obs_re", "ref_re", "err_re", np.ones(n, bool)),
                            ("obs_im", "ref_im", "err_im", cpx)):
        obs, ref, err = (np.asarray(ds[k], float) for k in (ko, kr, ke))
        ok = msk & np.isfinite(err) & (err > 0.0) & np.isfinite(obs) & np.isfinite(ref)
        r = np.full(n, np.nan)
        r[ok] = (obs[ok] - ref[ok]) / err[ok]
        out.append(r)
    return out[0], out[1]


def _num(x: float) -> str:
    """Number for a label: 1000 (not 1e+03), 0.1, 3.16."""
    return f"{int(round(x))}" if x >= 1.0 and abs(x - round(x)) < 1e-9 * x else f"{x:.3g}"


def decade_bands(freq: np.ndarray, *, axis: str = "period", per_decade: int = 1,
                 min_n: int = 1) -> List[Tuple[str, np.ndarray]]:
    """
    [("all bands", mask), (label, mask), ...]: bands of `per_decade` per decade,
    anchored at whole decades, from short to long period (for axis "frequency":
    decreasing frequency); bands with fewer than `min_n` rows are dropped. The
    bands of femtic_misfit.select_bands, with labels like "1 - 10 s".
    """
    freq = np.asarray(freq, float)
    bidx = fm.band_index(freq, per_decade=per_decade, axis=axis)
    unit = "s" if axis == "period" else "Hz"
    out: List[Tuple[str, np.ndarray]] = [("all bands", np.ones(freq.size, bool))]
    ks = np.unique(bidx)
    if axis == "frequency":
        ks = ks[::-1]
    for k in ks:
        m = bidx == k
        if np.count_nonzero(m) >= int(min_n):
            lo, hi = 10.0 ** (int(k) / per_decade), 10.0 ** ((int(k) + 1) / per_decade)
            out.append((f"{_num(lo)} - {_num(hi)} {unit}", m))
    return out


def band_selections(freq: np.ndarray, spec: Union[str, Sequence[Any]], *, axis: str = "period",
                    per_decade: int = 1, min_n: int = 1) -> List[Tuple[str, np.ndarray]]:
    """
    [(label, row mask), ...] of the frequency bands. spec "all"; "decades" (all,
    then the bands of femtic_misfit.select_bands); or a list of "all" and
    (lo, hi) pairs, in s for axis "period", in Hz for "frequency" (lo <= x < hi).
    Bands with fewer than `min_n` rows are dropped ("all" is kept).
    """
    freq = np.asarray(freq, float)
    n = freq.size
    all_sel = ("all bands", np.ones(n, bool))
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "all":
            return [all_sel]
        if s in ("decades", "auto"):
            return decade_bands(freq, axis=axis, per_decade=int(per_decade), min_n=int(min_n))
        raise ValueError(f"BANDS = {spec!r}: \"all\", \"decades\" or a list")
    x = 1.0 / freq if axis == "period" else freq
    unit = "s" if axis == "period" else "Hz"
    out: List[Tuple[str, np.ndarray]] = []
    for item in spec:
        if isinstance(item, str):
            if item.strip().lower() != "all":
                raise ValueError(f"BANDS: {item!r} is not \"all\" or a (lo, hi) pair")
            out.append(all_sel)
            continue
        lo, hi = float(item[0]), float(item[1])
        if not (0.0 < lo < hi):
            raise ValueError(f"BANDS: ({lo}, {hi}) needs 0 < lo < hi")
        m = (x >= lo * (1.0 - 1e-9)) & (x < hi * (1.0 - 1e-9))
        label = f"{_num(lo)} - {_num(hi)} {unit}"
        if np.count_nonzero(m) < int(min_n):
            _say(f"band {label}: {int(np.count_nonzero(m))} data rows (BAND_MIN_N = "
                 f"{int(min_n)}), dropped.")
            continue
        out.append((label, m))
    return out


def component_selections(comp_names: np.ndarray, specs: Sequence[str],
                         groups: Dict[str, Sequence[str]]) -> List[Tuple[str, np.ndarray]]:
    """
    [(label, row mask), ...]. A spec is "all", a group of `groups`, a component
    name, or several joined by "+"; names and groups are not case sensitive.
    Selections that match no row are dropped (with a message).
    """
    names = np.char.lower(np.asarray(comp_names, dtype=str))
    glow = {str(k).lower(): [str(w).lower() for w in v] for k, v in groups.items()}
    out: List[Tuple[str, np.ndarray]] = []
    for spec in specs:
        sp = str(spec).strip()
        if sp.lower() == "all":
            out.append(("all", np.ones(names.size, bool)))
            continue
        m = np.zeros(names.size, bool)
        for tok in [t.strip() for t in sp.split("+") if t.strip()]:
            m |= np.isin(names, glow.get(tok.lower(), [tok.lower()]))
        if not m.any():
            _say(f"component selection {sp!r}: no data row, dropped.")
            continue
        out.append((sp, m))
    return out


def nrms_pooled(r_re: np.ndarray, r_im: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
    """(number of values, nRMS) of the residuals of the rows in `mask`."""
    r = np.concatenate([r_re[mask], r_im[mask]])
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0, float("nan")
    return int(r.size), float(np.sqrt(np.mean(r * r)))


def nrms_per_site(r_re: np.ndarray, r_im: np.ndarray, mask: np.ndarray,
                  site_index: np.ndarray, n_sites: int) -> Tuple[np.ndarray, np.ndarray]:
    """(number of values, nRMS) per site index; NaN nRMS where a site has no value."""
    rows = mask & (site_index >= 0)
    s2 = np.zeros(n_sites)
    cnt = np.zeros(n_sites, dtype=np.int64)
    for r in (r_re, r_im):
        ok = rows & np.isfinite(r)
        s2 += np.bincount(site_index[ok], weights=r[ok] ** 2, minlength=n_sites)
        cnt += np.bincount(site_index[ok], minlength=n_sites)
    nr = np.full(n_sites, np.nan)
    has = cnt > 0
    nr[has] = np.sqrt(s2[has] / cnt[has])
    return cnt, nr


def measures_pooled(r_re: np.ndarray, r_im: np.ndarray, mask: np.ndarray,
                    names: Sequence[str]) -> Dict[str, float]:
    """{name: value} of the fit measures `names` of the residuals of the rows in `mask`."""
    f = mm.fit_measures(np.concatenate([r_re[mask], r_im[mask]]))
    return {k: float(f[k]) for k in names}


def measures_per_site(r_re: np.ndarray, r_im: np.ndarray, mask: np.ndarray,
                      site_index: np.ndarray, n_sites: int,
                      names: Sequence[str]) -> Dict[str, np.ndarray]:
    """{name: value per site index} of the fit measures `names`; NaN where a site has no value."""
    out = {k: np.full(n_sites, np.nan) for k in names}
    rows = np.flatnonzero(mask & (site_index >= 0))
    if not names or rows.size == 0:
        return out
    rows = rows[np.argsort(site_index[rows], kind="stable")]
    for group in np.split(rows, np.flatnonzero(np.diff(site_index[rows])) + 1):
        r = np.concatenate([r_re[group], r_im[group]])
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        f = mm.fit_measures(r)
        j = site_index[group[0]]
        for k in names:
            out[k][j] = f[k]
    return out


def compute_nrms(ds: dict, site_index: np.ndarray, n_sites: int,
                 bands: Sequence[Tuple[str, np.ndarray]],
                 comps: Sequence[Tuple[str, np.ndarray]], *,
                 base: Optional[np.ndarray] = None,
                 measures: Sequence[str] = ()) -> List[dict]:
    """
    One dict per (band, component selection) that has data: ib, ic, band, comp,
    n and nrms (global: all rows of the selection, assigned to a site or not),
    n_site and nrms_site (arrays per site index), n_unassigned (values of rows
    that belong to no site). `base` = rows to use at all (data types). `measures`
    = further fit measures (misfit_measures): glob_m {name: value} and site_m
    {name: array per site index}, from the same residuals (empty dicts without).
    """
    r_re, r_im = row_residuals(ds)
    if base is None:
        base = np.ones(r_re.size, bool)
    out: List[dict] = []
    for ib, (bl, bm) in enumerate(bands):
        for ic, (cl, cm) in enumerate(comps):
            m = bm & cm & base
            n_all, nr_all = nrms_pooled(r_re, r_im, m)
            if n_all == 0:
                _say(f"selection {bl} | {cl}: no usable value, skipped.")
                continue
            n_s, nr_s = nrms_per_site(r_re, r_im, m, site_index, n_sites)
            out.append(dict(ib=ib, ic=ic, band=bl, comp=cl, n=n_all, nrms=nr_all,
                            n_site=n_s, nrms_site=nr_s,
                            glob_m=measures_pooled(r_re, r_im, m, measures) if measures else {},
                            site_m=measures_per_site(r_re, r_im, m, site_index, n_sites,
                                                     measures),
                            n_unassigned=int(n_all - int(n_s.sum()))))
    return out


# ---------------------------------------------------------------------------
# Dimensionality of the observed impedances (modules/dimens.py)
# ---------------------------------------------------------------------------
def check_dim_settings(cfg: Dict[str, Any]) -> None:
    """ValueError for a DIM_* setting that cannot work."""
    if cfg["DIM_PAGE"] not in ("all", "band"):
        raise ValueError(f"DIM_PAGE = {cfg['DIM_PAGE']!r}: \"all\" or \"band\"")
    src = cfg["DIM_SOURCE"]
    items = [src] if isinstance(src, (str, os.PathLike)) else src
    if not (isinstance(items, (list, tuple)) and items
            and all(isinstance(p, (str, os.PathLike)) and str(p) for p in items)):
        raise ValueError(f"DIM_SOURCE = {src!r}: \"results\", a path or a list of paths")
    for key in _DIM_SETTINGS:
        v = cfg[key]
        if v is None and key in ("DIM_ERROR_PERCENT", "DIM_MISSING_ERROR_PERCENT"):
            continue
        if not (isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, bool) and np.isfinite(v) and v >= 0):
            raise ValueError(f"{key} = {v!r}: must be a number >= 0")
    opts = cfg["DIM_OPTIONS"]
    if not isinstance(opts, dict):
        raise ValueError("DIM_OPTIONS must be a dict of keywords")
    for key in opts:
        if key in _DIM_NO_OPTIONS:
            raise ValueError(f"DIM_OPTIONS[{key!r}]: the bands are those of BANDS")
        if key in _DIM_SETTINGS.values():
            raise ValueError(f"DIM_OPTIONS[{key!r}]: use the setting DIM_...")
    colors = cfg["DIM_COLORS"]
    if not isinstance(colors, dict):
        raise ValueError("DIM_COLORS must be a dict {class name: colour}")
    if colors:
        from matplotlib.colors import is_color_like
        for key, value in colors.items():
            if key not in _DIM_CLASS_COLORS:
                raise ValueError(f"DIM_COLORS: unknown class {key!r} (one of "
                                 f"{', '.join(_DIM_CLASS_COLORS)})")
            if not is_color_like(value):
                raise ValueError(f"DIM_COLORS[{key!r}] = {value!r} is no colour")


def mt_rows(ds: dict) -> dict:
    """The impedance rows (data type MT) of a dataset, own observed data."""
    keep = np.asarray(ds["datatype_name"]) == "MT"
    keys = ("freq", "site_id", "component_name", "obs_re", "obs_im", "err_re", "err_im")
    return {k: np.asarray(ds[k])[keep] for k in keys}


def impedance_per_site(rows: dict, id_to_site: Dict[int, int]) -> Tuple[Dict[int, dict], dict]:
    """
    ({site index: data dict}, stats) from the impedance rows `rows` (mt_rows) of a
    results file, for dimens.classify_dimensionality: freq (descending), Z in
    mV/km/nT (the results file holds Ohm), Z_err = variance of the real and the
    imaginary part each (mean of the two squared errors), err_kind "var". Only
    frequencies with all four components, finite values and errors > 0 (as in
    the misfit). A row that repeats site, frequency and component is ignored (the
    first one counts). stats = n_repeated (rows), n_incomplete (frequencies).
    """
    comps = ("Zxx", "Zxy", "Zyx", "Zyy")
    ids = np.asarray(rows["site_id"]).astype(np.int64)
    uniq, inv = np.unique(ids, return_inverse=True)
    lut = np.array([id_to_site.get(int(u), -1) for u in uniq], dtype=np.int64)
    site = lut[inv] if ids.size else ids
    freq = np.asarray(rows["freq"], float)
    code = np.full(ids.size, -1)
    for k, name in enumerate(comps):
        code[np.asarray(rows["component_name"]) == name] = k
    vals = [np.asarray(rows[k], float) for k in ("obs_re", "obs_im", "err_re", "err_im")]
    ok = (site >= 0) & (code >= 0) & np.isfinite(freq) & (freq > 0.0)
    for v in vals:
        ok &= np.isfinite(v)
    ok &= (vals[2] > 0.0) & (vals[3] > 0.0)
    # every frequency that occurs at a site, usable or not: to count the incomplete
    seen = (site >= 0) & (code >= 0)
    n_seen = len(set(zip(site[seen].tolist(), freq[seen].tolist())))
    rows_ok = np.flatnonzero(ok)
    order = rows_ok[np.lexsort((code[rows_ok], -freq[rows_ok], site[rows_ok]))]  # stable
    first = np.ones(order.size, bool)
    if order.size > 1:
        first[1:] = ((np.diff(site[order]) != 0) | (np.diff(freq[order]) != 0)
                     | (np.diff(code[order]) != 0))
    n_repeated = int(order.size - first.sum())
    order = order[first]
    out: Dict[int, dict] = {}
    n_full = 0
    groups = np.split(order, np.flatnonzero(np.diff(site[order])) + 1) if order.size else []
    for g in groups:
        f, pos = np.unique(-freq[g], return_inverse=True)
        z = np.full((f.size, 4), np.nan, dtype=complex)
        var = np.full((f.size, 4), np.nan)
        z[pos, code[g]] = (vals[0][g] + 1j * vals[1][g]) / _OHM_PER_FIELD_UNIT
        var[pos, code[g]] = 0.5 * (vals[2][g] ** 2 + vals[3][g] ** 2) / _OHM_PER_FIELD_UNIT ** 2
        full = np.all(np.isfinite(var), axis=1)
        if full.any():
            out[int(site[g[0]])] = dict(freq=-f[full], Z=z[full].reshape(-1, 2, 2),
                                        Z_err=var[full].reshape(-1, 2, 2), err_kind="var",
                                        rot=0.0, Z_units="mV/km/nT")
            n_full += int(full.sum())
    return out, dict(n_repeated=n_repeated, n_incomplete=int(n_seen - n_full))


def same_impedances(a: Dict[int, dict], b: Dict[int, dict]) -> bool:
    """True if two results of impedance_per_site hold the same data (values, not errors)."""
    if set(a) != set(b):
        return False
    return all(a[j]["freq"].shape == b[j]["freq"].shape
               and np.allclose(a[j]["freq"], b[j]["freq"], rtol=1e-6, atol=0.0)
               and np.allclose(a[j]["Z"], b[j]["Z"], rtol=1e-6, atol=0.0) for j in a)


def classify_sites(cfg: Dict[str, Any], sites: Sequence[Dict[str, Any]],
                   data: Optional[Dict[int, dict]] = None) -> Tuple[Dict[int, dict], List[str]]:
    """
    ({site index: result of dimens.classify_dimensionality}, messages). `data` =
    impedance_per_site of the results file; None: the files of DIM_SOURCE, whose
    station names are matched to the names of site.dat (case and outer blanks
    ignored). Station name, lat and lon of the results are those of site.dat.
    Sites without a usable frequency are left out. OSError, ValueError,
    KeyError, ImportError of the reading and the analysis are passed on.
    """
    import dimens                      # only needed for these measures
    kw = {word: cfg[key] for key, word in _DIM_SETTINGS.items()}
    kw.update(cfg["DIM_OPTIONS"])
    notes: List[str] = []
    found: Dict[int, dict] = {}
    if data is not None:
        keys: Dict[str, int] = {}
        for j in sorted(data):
            name = str(sites[j]["name"]).strip() or f"site{j}"
            keys[name if name not in keys else f"{name}#{j}"] = j
        records = [dict(data[j], station=name) for name, j in keys.items()]
        if records:
            res = dimens.classify_dimensionality(records, **kw)
            found = {keys[name]: r for name, r in res.items()}
    else:
        res = dimens.classify_dimensionality(cfg["DIM_SOURCE"], **kw)
        by_name: Dict[str, int] = {}
        for j, st_ in enumerate(sites):
            by_name.setdefault(str(st_["name"]).strip().casefold(), j)
        lost = []
        for name, r in res.items():
            j = by_name.get(str(name).strip().casefold())
            if j is None:
                lost.append(name)
            elif j in found:
                notes.append(f"WARNING: DIM_SOURCE: {name!r} is a second data set of site "
                             f"{sites[j]['name']}; ignored.")
            else:
                found[j] = r
        if lost:
            notes.append(f"DIM_SOURCE: {len(lost)} of {len(res)} stations are not in site.dat: "
                         f"{', '.join(map(str, lost[:10]))}{' ...' if len(lost) > 10 else ''}.")
    out = {j: dict(r, station=sites[j]["name"], lat_deg=sites[j]["lat"],
                   lon_deg=sites[j]["lon"]) for j, r in sorted(found.items())
           if r["freq"].size}
    return out, notes


def dim_selections(results: Dict[int, dict], used: np.ndarray, n_sites: int,
                   measures: Sequence[str], cfg: Dict[str, Any]) -> Tuple[List[dict], dict]:
    """
    (selections, banded results) of the sites `used` that have a result. One
    selection per frequency band of BANDS (made as for nRMS, from the
    frequencies of these sites, a site frequency counting as four rows for
    BAND_MIN_N): ib, band, comp "", head, n (sites with data in the band),
    site_m {measure: array per site index; classes as positions on the colour
    bar}, codes {class measure: class codes}, glob_m, sub_m. Banded results =
    {name: result with "bands"} for dimens.write_summary.
    """
    import dimens
    have = [int(j) for j in used if int(j) in results]
    if not have:
        return [], {}
    freq = np.concatenate([results[j]["freq"] for j in have])
    bounds = np.cumsum([0] + [results[j]["freq"].size for j in have])
    bands = band_selections(np.repeat(freq, 4), cfg["BANDS"], axis=cfg["BAND_AXIS"],
                            per_decade=cfg["BAND_PER_DECADE"], min_n=cfg["BAND_MIN_N"])
    per_site: Dict[int, List[dict]] = {j: [] for j in have}
    sels: List[dict] = []
    for ib, (label, mask4) in enumerate(bands):
        mask = mask4[::4]
        if not mask.any():
            continue
        tmin, tmax = float(1.0 / freq[mask].max()), float(1.0 / freq[mask].min())
        values = {m: np.full(n_sites, np.nan) for m in measures}
        codes = {m: np.full(n_sites, np.nan) for m in measures if m in _DIM_CLASS_MEASURES}
        n = 0
        for k, j in enumerate(have):
            members = np.flatnonzero(mask[bounds[k]:bounds[k + 1]])
            summary = dimens.summarise_band(results[j], members)
            summary.update(label=label, period_min=tmin, period_max=tmax,
                           freq_min=1.0 / tmax, freq_max=1.0 / tmin)
            per_site[j].append(summary)
            if not members.size:
                continue
            n += 1
            for m in measures:
                if m in _DIM_CLASS_MEASURES:
                    code = summary["class_id" if m == "dim" else "pt_class"]
                    codes[m][j] = code
                    values[m][j] = _DIM_CODES[m].index(code)
                else:
                    values[m][j] = summary[_DIM_VALUES[m][0]]
        glob_m, sub_m = {}, {}
        for m in measures:
            v = values[m][used]
            v = v[np.isfinite(v)]
            if m in _DIM_CLASS_MEASURES:
                names = dimens.CLASS_NAMES if m == "dim" else dimens.PT_CLASS_NAMES
                c = codes[m][used]
                c = c[np.isfinite(c)].astype(int)
                found = sorted(((int(np.sum(c == k)), -k) for k in np.unique(c)), reverse=True)
                text = ", ".join(f"{names[-k]} {num}" for num, k in found[:3])
                glob_m[m] = float("nan")
                sub_m[m] = f"n = {n}: {text}" + (", ..." if len(found) > 3 else "")
            else:
                glob_m[m] = float(np.median(v)) if v.size else float("nan")
                shown = f"{glob_m[m]:.2f}" if v.size else "-"
                sub_m[m] = f"median of the sites = {shown}, n = {n}"
        sels.append(dict(ib=ib, ic=0, band=label, comp="", head=label, n=n, site_m=values,
                         codes=codes, glob_m=glob_m, sub_m=sub_m))
    banded: Dict[str, dict] = {}
    for j in have:
        name = str(results[j]["station"])
        banded[name if name not in banded else f"{name}#{j}"] = dict(results[j],
                                                                    bands=per_site[j])
    return sels, banded


def dim_style(measure: str, sels: Sequence[dict], used: np.ndarray,
              cfg: Dict[str, Any]) -> dict:
    """dict(cmap, norm, extend, ticks, reference) of the maps of a dimensionality measure."""
    if measure in _DIM_CLASS_MEASURES:
        import dimens
        from matplotlib.colors import BoundaryNorm, ListedColormap
        table = dimens.CLASS_NAMES if measure == "dim" else dimens.PT_CLASS_NAMES
        names = [table[k] for k in _DIM_CODES[measure]]
        colors = dict(_DIM_CLASS_COLORS, **cfg["DIM_COLORS"])
        cmap = ListedColormap([colors[t] for t in names])
        norm = BoundaryNorm(np.arange(len(names) + 1) - 0.5, cmap.N)
        return dict(cmap=cmap, norm=norm, extend="neither", reference=None,
                    ticks=(list(range(len(names))), names))
    scale = measure_scale(measure, cfg)
    allv = np.concatenate([s["site_m"][measure][used] for s in sels] or [[]])
    norm, extend = make_norm(allv, vmin=scale["vmin"], vmax=scale["vmax"],
                             reference=scale["reference"], floor=0.0,
                             name=f"the reference of {measure}")
    return dict(cmap=scale["cmap"], norm=norm, extend=extend, reference=scale["reference"],
                ticks=None)


def format_dimensionality(sels: Sequence[dict], banded: dict, *,
                          header: Sequence[str]) -> str:
    """Plain-text tables for the summary: sites per class and band, medians of the sites."""
    import dimens
    table = dimens.survey_summary(banded)
    labels = [str(v) for v in table["band_label"]]
    wide = max([len(v) for v in labels] + [4])
    short = {"undetermined": "undet", "3D/2Dtw": "3/2tw", "3D/2D": "3/2D",
             "3D/2Ddiag": "3/2dg", "3D/1D2D": "3/12D", "ANIS-HINT1": "ANIS1",
             "ANIS-HINT2": "ANIS2", "undefined": "undef"}
    out = list(header)
    for title, prefix, codes in (("WAL class", "class", _DIM_CODES["dim"]),
                                 ("phase tensor class", "pt_class", _DIM_CODES["ptdim"])):
        names = [str(table[prefix + "_names"][k]) for k in codes]
        head = f"{'band':<{wide}s} {'sites':>5s}" + "".join(
            f" {short.get(t, t):>6s}" for t in names)
        out += ["", f"number of sites per band: {title}", "=" * len(head), head, "-" * len(head)]
        for i, label in enumerate(labels):
            counts = table[prefix + "_site_counts"][i]
            out.append(f"{label:<{wide}s} {int(counts.sum()):5d}" + "".join(
                f" {int(counts[k]):6d}" for k in codes))
        out.append("-" * len(head))
    keys = [("pt_abs_beta_deg", "|beta|"), ("pt_ellipticity_median", "ellip"),
            ("swift_skew", "swift"), ("bahr_skew", "bahr")]
    head = f"{'band':<{wide}s}" + "".join(f" {t:>7s}" for _, t in keys)
    out += ["", "median over the sites of the band values (|beta| in deg)", "=" * len(head),
            head, "-" * len(head)]
    for i, label in enumerate(labels):
        line = f"{label:<{wide}s}"
        for key, _ in keys:
            v = table[key][:, i]
            v = v[np.isfinite(v)]
            line += f" {np.median(v):7.3g}" if v.size else f" {'-':>7s}"
        out.append(line)
    out.append("-" * len(head))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Ensemble: members and statistics over the members
# ---------------------------------------------------------------------------
class DataTypeError(ValueError):
    """No row of the requested data types (DATATYPES) in a results file."""


_STAT_WORDS = {"mean": "mean", "median": "median", "min": "minimum", "max": "maximum",
               "std": "std"}


def parse_stat(stat: str) -> Tuple[str, Optional[float]]:
    """
    ENS_STAT -> (kind, q): kind "mean", "median", "min", "max", "std" or
    "percentile" (q = 0..100 for "pNN", e.g. "p90"; else None). ValueError for
    anything else.
    """
    s = str(stat).strip().lower()
    if s in _STAT_WORDS:
        return s, None
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", s)
    if m and 0.0 <= float(m.group(1)) <= 100.0:
        return "percentile", float(m.group(1))
    raise ValueError(f"ENS_STAT = {stat!r}: must be one of mean, median, min, max, std, or "
                     "pNN (a percentile, e.g. p90)")


def stat_words(stat: str) -> str:
    """The statistic in words: "median", "std", "percentile 90", ..."""
    kind, q = parse_stat(stat)
    return f"percentile {q:g}" if kind == "percentile" else _STAT_WORDS[kind]


def ens_statistic(x: Any, stat: str, axis: int = 0) -> np.ndarray:
    """
    Statistic of `x` over `axis` (the members), ignoring NaN; NaN where there is
    no value (std: fewer than two; it is the sample std, ddof = 1).
    """
    kind, q = parse_stat(stat)
    a = np.asarray(x, float)
    a = np.where(np.isfinite(a), a, np.nan)
    cnt = np.sum(np.isfinite(a), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "mean":
            out = np.nanmean(a, axis=axis)
        elif kind == "median":
            out = np.nanmedian(a, axis=axis)
        elif kind == "min":
            out = np.nanmin(a, axis=axis)
        elif kind == "max":
            out = np.nanmax(a, axis=axis)
        elif kind == "std":
            out = np.nanstd(a, axis=axis, ddof=1)
        else:
            out = np.nanpercentile(a, q, axis=axis)
    return np.where(cnt >= (2 if kind == "std" else 1), out, np.nan)


def member_paths(cfg: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    (paths of the members, where they came from): RUN_PATHS (globs expanded); else
    the sub-directories of RESULTS_PATH that contain ITER_PATTERN files, if
    RESULTS_PATH itself contains none; else RESULTS_PATH (one member).
    """
    runs = cfg.get("RUN_PATHS")
    if runs:
        return fm.expand_paths([runs] if isinstance(runs, str) else list(runs)), "RUN_PATHS"
    top, pat = os.path.normpath(cfg["RESULTS_PATH"]), cfg["ITER_PATTERN"]
    if os.path.isdir(top) and not glob.glob(os.path.join(glob.escape(top), pat)):
        subs = [d for d in sorted(glob.glob(os.path.join(glob.escape(top), "*")))
                if os.path.isdir(d) and glob.glob(os.path.join(glob.escape(d), pat))]
        if subs:
            return subs, "sub-directories of RESULTS_PATH"
    return [top], "RESULTS_PATH"


def member_names(paths: Sequence[str]) -> List[str]:
    """Short unique names: the directory name (of the file, for a results file); repeats get #2, #3, ..."""
    out: List[str] = []
    count: Dict[str, int] = {}
    for p in paths:
        q = os.path.normpath(p)
        if os.path.isfile(q):
            q = os.path.dirname(os.path.abspath(q))
        nm = os.path.basename(q) or q
        count[nm] = count.get(nm, 0) + 1
        out.append(nm if count[nm] == 1 else f"{nm}#{count[nm]}")
    return out


def apply_datatypes(ds: dict, datatypes: Optional[Sequence[str]]) -> Tuple[dict, np.ndarray]:
    """
    (dataset, base): the rows of the data types `datatypes` (None = all). DataTypeError
    if there is none.
    """
    base = np.ones(int(np.asarray(ds["freq"]).size), bool)
    if datatypes:
        base = np.isin(ds["datatype_name"], list(datatypes))
        if not base.any():
            raise DataTypeError(f"no row of the data types {list(datatypes)} "
                                f"(found: {sorted(set(map(str, ds['datatype_name'])))}).")
        ds = {k: (v[base] if isinstance(v, np.ndarray) and v.shape[:1] == base.shape else v)
              for k, v in ds.items() if k not in ("cal_re_m", "cal_im_m")}
        base = np.ones(int(np.asarray(ds["freq"]).size), bool)
    return ds, base


def load_member(path: str, cfg: Dict[str, Any]) -> dict:
    """
    Read one member: dict(res, ds, base, n_rows) with the resolve_run result `res`
    (without its data), the dataset of the member (own observed data) restricted
    to DATATYPES, and the number of rows of the file; mt = the impedance rows (for
    the dimensionality measures with DIM_SOURCE "results", else None), all_ids =
    the ids of all rows if they are needed besides the filtered ones. OSError,
    ValueError, KeyError (DataTypeError) if it cannot be read.
    """
    res = fm.resolve_run(path, iteration=cfg["ITERATION"], pattern=cfg["ITER_PATTERN"])
    ds = fm.build_dataset([res["data"]])
    n_rows = int(np.asarray(ds["freq"]).size)
    res = {k: v for k, v in res.items() if k != "data"}
    measures = parse_measures(cfg["SITE_MEASURES"])
    dim_here = bool(dim_names(measures)) and cfg["DIM_SOURCE"] == "results"
    all_ids = None             # distortion and impedances are of MT ids, also if DATATYPES leaves MT out
    if cfg["DATATYPES"] and (_DISTORTION in measures or dim_here):
        all_ids = {k: ds[k] for k in ("freq", "site_id", "datatype_name", "site_x", "site_y")}
    mt = mt_rows(ds) if dim_here else None
    ds, base = apply_datatypes(ds, cfg["DATATYPES"])
    return dict(res=res, ds=ds, base=base, n_rows=n_rows, all_ids=all_ids, mt=mt)


def member_selections(lm: dict, site_index: np.ndarray, n_sites: int,
                      cfg: Dict[str, Any]) -> List[dict]:
    """compute_nrms of a loaded member for the BANDS, COMPONENTS and SITE_MEASURES of `cfg` (ValueError
    for a bad setting)."""
    ds = lm["ds"]
    bands = band_selections(ds["freq"], cfg["BANDS"], axis=cfg["BAND_AXIS"],
                            per_decade=cfg["BAND_PER_DECADE"], min_n=cfg["BAND_MIN_N"])
    comps = component_selections(ds["component_name"], cfg["COMPONENTS"], cfg["COMPONENT_GROUPS"])
    return compute_nrms(ds, site_index, n_sites, bands, comps, base=lm["base"],
                        measures=fit_names(parse_measures(cfg["SITE_MEASURES"])))


@contextlib.contextmanager
def _print_once(seen: set):
    """Lines printed in the block appear once: not again if they are in `seen` (which is updated)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        for ln in buf.getvalue().splitlines():
            if ln not in seen:
                seen.add(ln)
                print(ln)


def aggregate_members(members: Sequence[dict], stat: str, n_sites: int) -> List[dict]:
    """
    Selections of an ensemble: one dict per (band, components) found in any member,
    in the keys of compute_nrms. nrms_site = the statistic `stat` over the members
    of their nRMS at the site, nrms = the same for the global nRMS of the members;
    n, n_site, n_unassigned = median of the members (rounded). Further keys: M
    (members with a value), G (global nRMS per member), X (members x sites), stat,
    sub (second line of the panel title). The further fit measures of the members
    (site_m) in the same way: glob_m, site_m, G_m, X_m, sub_m. `members`: dicts
    with "sels".
    """
    bands: List[str] = []
    comps: List[str] = []
    keys: List[Tuple[str, str]] = []
    lookup: List[Dict[Tuple[str, str], dict]] = []
    for m in members:
        d: Dict[Tuple[str, str], dict] = {}
        for s in m["sels"]:
            if s["band"] not in bands:
                bands.append(s["band"])
            if s["comp"] not in comps:
                comps.append(s["comp"])
            if (s["band"], s["comp"]) not in keys:
                keys.append((s["band"], s["comp"]))
            d[(s["band"], s["comp"])] = s
        lookup.append(d)
    keys.sort(key=lambda k: (bands.index(k[0]), comps.index(k[1])))
    kind, _ = parse_stat(stat)
    nm = len(members)
    out: List[dict] = []
    for key in keys:
        X = np.full((nm, n_sites), np.nan)
        N = np.full((nm, n_sites), np.nan)
        G = np.full(nm, np.nan)
        nG = np.full(nm, np.nan)
        nU = np.full(nm, np.nan)
        for k, d in enumerate(lookup):
            s = d.get(key)
            if s is None:
                continue
            X[k] = s["nrms_site"]
            N[k] = np.where(s["n_site"] > 0, s["n_site"], np.nan)
            G[k], nG[k], nU[k] = s["nrms"], s["n"], s["n_unassigned"]
        names: List[str] = []
        for d in lookup:
            names += [k for k in d.get(key, {}).get("site_m", {}) if k not in names]
        Xm = {q: np.full((nm, n_sites), np.nan) for q in names}
        Gm = {q: np.full(nm, np.nan) for q in names}
        for k, d in enumerate(lookup):
            s = d.get(key)
            if s is not None:
                for q in s.get("site_m", {}):
                    Xm[q][k] = s["site_m"][q]
                    Gm[q][k] = s["glob_m"][q]
        gm = {q: float(ens_statistic(Gm[q], stat)) for q in names}
        n_have = int(np.sum(np.isfinite(G)))
        g = float(ens_statistic(G, stat))
        n_med = int(round(float(ens_statistic(nG, "median"))))
        what = "std of global nRMS" if kind == "std" else "global nRMS"
        words = "" if kind == "std" else f"{stat_words(stat)}, "
        out.append(dict(
            ib=bands.index(key[0]), ic=comps.index(key[1]), band=key[0], comp=key[1],
            n=n_med, nrms=g,
            n_site=np.nan_to_num(np.rint(ens_statistic(N, "median"))).astype(np.int64),
            nrms_site=ens_statistic(X, stat),
            n_unassigned=int(round(float(np.nan_to_num(ens_statistic(nU, "median"))))),
            M=n_have, G=G, X=X, stat=stat,
            sub=f"{what} = {g:.2f} ({words}M = {n_have}), n = {n_med}",
            glob_m=gm, site_m={q: ens_statistic(Xm[q], stat) for q in names}, G_m=Gm, X_m=Xm,
            sub_m={q: f"{what.replace('nRMS', measure_label(q))} = {measure_text(gm[q], q)} "
                      f"({words}M = {n_have}), n = {n_med}" for q in names}))
    return out


def iteration_text(iters: Sequence[int]) -> str:
    """"8" for one iteration, "5-8" for a range (smallest-largest)."""
    lo, hi = int(min(iters)), int(max(iters))
    return str(lo) if lo == hi else f"{lo}-{hi}"


# ---------------------------------------------------------------------------
# Map extent and coordinates
# ---------------------------------------------------------------------------
def map_extent(lon: np.ndarray, lat: np.ndarray, *, mode: str = "sites",
               box: Optional[Sequence[float]] = None,
               center: Optional[Sequence[float]] = None,
               size_km: Sequence[float] = (50.0, 50.0), margin_frac: float = 0.15,
               margin_min_km: float = 2.0) -> Tuple[float, float, float, float]:
    """
    (lon_min, lon_max, lat_min, lat_max) of the map. mode "sites": the box of the
    sites (`lon`, `lat`) plus a margin of `margin_frac` of its size on each side,
    at least `margin_min_km`; "box": `box` as given; "center": `size_km` (width,
    height) around `center` = (lon, lat).
    """
    if mode == "box":
        if box is None or len(box) != 4:
            raise ValueError("MAP_BOX = (lon_min, lon_max, lat_min, lat_max) needed")
        return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if mode == "center":
        if center is None or len(center) != 2:
            raise ValueError("MAP_CENTER = (lon, lat) needed")
        lon_c, lat_c = float(center[0]), float(center[1])
        kx = _KM_PER_DEG * math.cos(math.radians(lat_c))
        hw, hh = 0.5 * float(size_km[0]) / kx, 0.5 * float(size_km[1]) / _KM_PER_DEG
        return lon_c - hw, lon_c + hw, lat_c - hh, lat_c + hh
    if mode != "sites":
        raise ValueError(f"MAP_EXTENT_MODE = {mode!r}: \"sites\", \"box\" or \"center\"")
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if lon.size == 0:
        raise ValueError("no site to take the map extent from")
    lat_c = float(np.mean(lat))
    kx = _KM_PER_DEG * math.cos(math.radians(lat_c))
    span_lon, span_lat = float(lon.max() - lon.min()), float(lat.max() - lat.min())
    mlon = max(float(margin_frac) * span_lon, float(margin_min_km) / kx)
    mlat = max(float(margin_frac) * span_lat, float(margin_min_km) / _KM_PER_DEG)
    return (float(lon.min()) - mlon, float(lon.max()) + mlon,
            float(lat.min()) - mlat, float(lat.max()) + mlat)


def make_frame(display: str, lon0: float, lat0: float) -> dict:
    """Plot coordinates: "latlon" (degrees) or "km" (from (lon0, lat0), local scale)."""
    if display not in ("latlon", "km"):
        raise ValueError(f"DISPLAY_COORDS = {display!r}: \"latlon\" or \"km\"")
    c = math.cos(math.radians(lat0))
    return dict(display=display, lon0=float(lon0), lat0=float(lat0),
                kx=_KM_PER_DEG * c, ky=_KM_PER_DEG,
                aspect=(1.0 / c if display == "latlon" else 1.0))


def to_frame(frame: dict, lon: Any, lat: Any) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y) plot coordinates of (lon, lat) in degrees."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if frame["display"] == "latlon":
        return lon, lat
    return (lon - frame["lon0"]) * frame["kx"], (lat - frame["lat0"]) * frame["ky"]


# ---------------------------------------------------------------------------
# Overlays: profile lines and other polylines
# ---------------------------------------------------------------------------
class OverlayError(ValueError):
    """A profile or line that cannot be used (file, coordinates, geometry)."""


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_UTM_K0 = 0.9996
_UTM_TOL_M = 25.0                  # site.dat easting / northing must agree with lat / lon to this (median)
_COORDS = ("lonlat", "latlon", "utm", "model")
# style keys of the entries and the setting that holds the default of each
_PROFILE_STYLE = {
    "color": "PROFILE_COLOR", "lw": "PROFILE_LW", "ls": "PROFILE_LS",
    "line_alpha": "PROFILE_ALPHA", "zorder": "PROFILE_ZORDER", "halo": "PROFILE_HALO",
    "halo_color": "PROFILE_HALO_COLOR", "halo_width": "PROFILE_HALO_WIDTH",
    "marker": "PROFILE_MARKER", "markersize": "PROFILE_MARKER_SIZE",
    "end_marker": "PROFILE_END_MARKER", "end_marker_size": "PROFILE_END_MARKER_SIZE",
    "fontsize": "PROFILE_FONTSIZE", "fontweight": "PROFILE_FONTWEIGHT",
    "text_color": "PROFILE_TEXT_COLOR", "label_offset": "PROFILE_LABEL_OFFSET",
    "tick_fontsize": "PROFILE_TICK_FONTSIZE", "tick_length": "PROFILE_TICK_LENGTH",
    "tick_lw": "PROFILE_TICK_LW", "swath_color": "PROFILE_SWATH_COLOR",
    "swath_alpha": "PROFILE_SWATH_ALPHA", "mark_sites": "PROFILE_MARK_SITES",
    "ring_size": "PROFILE_RING_SIZE", "ring_lw": "PROFILE_RING_LW",
}
_LINE_STYLE = {
    "color": "LINE_COLOR", "lw": "LINE_LW", "ls": "LINE_LS", "line_alpha": "LINE_ALPHA",
    "zorder": "LINE_ZORDER", "halo": "LINE_HALO", "halo_color": "LINE_HALO_COLOR",
    "halo_width": "LINE_HALO_WIDTH", "marker": "LINE_MARKER",
    "markersize": "LINE_MARKER_SIZE", "alpha": "LINE_FILL_ALPHA",
}
_STYLE_ALIASES = {"linewidth": "lw", "linestyle": "ls", "ms": "markersize"}
_PROFILE_KEYS = ("name", "points", "file", "start", "end", "center", "azimuth", "length_km",
                 "coords", "width_km", "ticks_km", "tick_labels", "labels",
                 "label") + tuple(_PROFILE_STYLE) + tuple(_STYLE_ALIASES)
_LINE_KEYS = ("label", "file", "points", "segments", "coords", "closed",
              "fill") + tuple(_LINE_STYLE) + tuple(_STYLE_ALIASES)
# kinds of the style values (for the checks)
_STYLE_KINDS = {
    "color": "color", "halo_color": "color", "text_color": "color_or_none",
    "swath_color": "color_or_none", "lw": "pos", "ls": "ls", "line_alpha": "unit",
    "alpha": "unit", "swath_alpha": "unit", "zorder": "num", "halo": "bool",
    "halo_width": "nonneg", "marker": "marker", "end_marker": "marker", "markersize": "pos",
    "end_marker_size": "pos", "fontsize": "pos", "tick_fontsize": "pos_or_none",
    "fontweight": "weight", "label_offset": "nonneg", "tick_length": "pos_or_none",
    "tick_lw": "pos_or_none", "mark_sites": "bool", "ring_size": "pos", "ring_lw": "pos",
}
_FONT_WEIGHTS = ("ultralight", "light", "normal", "regular", "book", "medium", "roman",
                 "semibold", "demibold", "demi", "bold", "heavy", "extra bold", "black")
_GENERIC_FAMILIES = ("serif", "sans-serif", "sans", "sans serif", "cursive", "fantasy",
                     "monospace")
_SMALL_PANEL_CM = 6.0              # smaller panels get a note (PANEL_SIZE was in inches before)
_SCALE_BAR_LW = 2.0                # [pt] line of the scale bar
_LEGEND_LOCS = (None, "best", "upper right", "upper left", "lower left", "lower right", "right",
                "center left", "center right", "lower center", "upper center", "center")
_MAX_TICKS = 2000
_PROFILE_TICK_FRAC = 0.012         # length of a tick: this fraction of the width of the map
_END_TOL_KM = 0.01                 # a site this close beyond the end of a profile still counts (10 m)


def _krueger(n: float) -> Tuple[tuple, tuple, tuple]:
    """Series coefficients (alpha, beta, delta) of the Krueger transverse Mercator, to n**4."""
    a = (n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16 + 41 * n**4 / 180,
         13 * n**2 / 48 - 3 * n**3 / 5 + 557 * n**4 / 1440,
         61 * n**3 / 240 - 103 * n**4 / 140,
         49561 * n**4 / 161280)
    b = (n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96 - n**4 / 360,
         n**2 / 48 + n**3 / 15 - 437 * n**4 / 1440,
         17 * n**3 / 480 - 37 * n**4 / 840,
         4397 * n**4 / 161280)
    d = (2 * n - 2 * n**2 / 3 - 2 * n**3 + 116 * n**4 / 45,
         7 * n**2 / 3 - 8 * n**3 / 5 - 227 * n**4 / 45,
         56 * n**3 / 15 - 136 * n**4 / 35,
         4279 * n**4 / 630)
    return a, b, d


def utm_forward(lat: Any, lon: Any, zone: int, north: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    UTM (easting, northing) in m of (lat, lon) in degrees, WGS84, for `zone`
    (central meridian 6 * zone - 183 deg); `north` False = southern hemisphere
    (false northing 10000 km). Kruger series to 4th order (sub-millimetre inside
    the zone and some degrees beyond), no external library.
    """
    n = _WGS84_F / (2.0 - _WGS84_F)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0)
    al = _krueger(n)[0]
    e = 2.0 * math.sqrt(n) / (1.0 + n)
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float) - (6.0 * int(zone) - 183.0))
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi0 = np.arctan2(t, np.cos(lam))
    eta0 = np.arcsinh(np.sin(lam) / np.hypot(t, np.cos(lam)))
    xi, eta = xi0.copy(), eta0.copy()
    for j, aj in enumerate(al, start=1):
        xi = xi + aj * np.sin(2 * j * xi0) * np.cosh(2 * j * eta0)
        eta = eta + aj * np.cos(2 * j * xi0) * np.sinh(2 * j * eta0)
    return (500000.0 + _UTM_K0 * big_a * eta,
            (0.0 if north else 1.0e7) + _UTM_K0 * big_a * xi)


def utm_inverse(east: Any, north_m: Any, zone: int, north: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(lon, lat) in degrees of the UTM (easting, northing) in m; inverse of utm_forward."""
    n = _WGS84_F / (2.0 - _WGS84_F)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0)
    _, be, de = _krueger(n)
    xi0 = (np.asarray(north_m, float) - (0.0 if north else 1.0e7)) / (_UTM_K0 * big_a)
    eta0 = (np.asarray(east, float) - 500000.0) / (_UTM_K0 * big_a)
    xi, eta = xi0.copy(), eta0.copy()
    for j, bj in enumerate(be, start=1):
        xi = xi - bj * np.sin(2 * j * xi0) * np.cosh(2 * j * eta0)
        eta = eta - bj * np.cos(2 * j * xi0) * np.sinh(2 * j * eta0)
    chi = np.arcsin(np.sin(xi) / np.cosh(eta))
    phi = chi.copy()
    for j, dj in enumerate(de, start=1):
        phi = phi + dj * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(eta), np.cos(xi))
    return np.degrees(lam) + (6.0 * int(zone) - 183.0), np.degrees(phi)


def utm_zone_of(lon: float) -> int:
    """UTM zone (1 .. 60) of a longitude in degrees (no Norway / Svalbard exceptions)."""
    return int(math.floor((float(lon) + 180.0) / 6.0)) % 60 + 1


def make_geo(sites: Sequence[Dict[str, Any]], st: Optional[dict] = None) -> dict:
    """
    Georeferencing of the overlays: dict(zone, north, resid, frame, shift).
    zone / north = the UTM zone and hemisphere in which the easting / northing
    of the site.dat sites are the UTM coordinates of their lat / lon (the one
    with the smallest median distance, `resid` in m), zone None if no zone fits
    within 25 m (then "utm" and "model" coordinates cannot be used). frame /
    shift = those of the assignment of the site ids by position (assign_sites
    stats `st`: "x=north, y=east" or "x=east, y=north", and the shift (sx, sy)
    in m that turns model coordinates into UTM), "" and NaN otherwise.
    """
    geo: Dict[str, Any] = dict(zone=None, north=True, resid=float("nan"), frame="",
                               shift=(float("nan"), float("nan")))
    if st:
        sh = st.get("shift", geo["shift"])
        geo["frame"] = str(st.get("frame", "") or "")
        geo["shift"] = (float(sh[0]), float(sh[1]))
    if not sites:
        return geo
    lat = np.array([s["lat"] for s in sites], float)
    lon = np.array([s["lon"] for s in sites], float)
    east = np.array([s["easting"] for s in sites], float)
    north = np.array([s["northing"] for s in sites], float)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(east) & np.isfinite(north)
    if not ok.any():
        return geo
    lat, lon, east, north = lat[ok], lon[ok], east[ok], north[ok]
    zones = set()
    for lo in (lon.min(), lon.max(), lon.mean()):
        z0 = utm_zone_of(lo)
        for dz in (-1, 0, 1):
            zones.add((z0 - 1 + dz) % 60 + 1)
    best = None
    for z in sorted(zones):
        for nh in (True, False):
            e, n = utm_forward(lat, lon, z, nh)
            r = float(np.median(np.hypot(e - east, n - north)))
            if best is None or r < best[0]:
                best = (r, z, nh)
    geo["resid"] = best[0]
    if best[0] <= _UTM_TOL_M:
        geo["zone"], geo["north"] = best[1], best[2]
    return geo


def to_lonlat(x: Any, y: Any, coords: str, geo: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    (lon, lat) in degrees of the coordinates (x, y) given in `coords`: "lonlat"
    (lon, lat), "latlon" (lat, lon), "utm" (easting, northing in m, zone of
    site.dat) or "model" (x, y in m of the FEMTIC model, as site_x, site_y of
    the results file). OverlayError if the coordinates cannot be converted or
    are not finite or not on the globe.
    """
    if coords not in _COORDS:
        raise OverlayError(f"coords = {coords!r}: one of {list(_COORDS)}")
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise OverlayError("coordinates are not finite")
    if coords == "lonlat":
        lon, lat = x, y
    elif coords == "latlon":
        lon, lat = y, x
    else:
        if geo.get("zone") is None:
            res = geo.get("resid", float("nan"))
            raise OverlayError(
                f"coords = {coords!r} needs the UTM easting / northing of site.dat; they do not "
                "agree with its lat / lon in any zone"
                + (f" (median distance {res:.0f} m)" if np.isfinite(res) else ""))
        if coords == "utm":
            east, north = x, y
        else:
            sx, sy = geo.get("shift", (float("nan"), float("nan")))
            fr = geo.get("frame", "")
            if not (np.isfinite(sx) and np.isfinite(sy)) or fr not in ("x=north, y=east",
                                                                       "x=east, y=north"):
                raise OverlayError("coords = \"model\" needs the sites assigned by position "
                                   "(SITE_MATCH = \"coords\" or \"auto\" with site coordinates)")
            if fr == "x=north, y=east":
                north, east = x + sx, y + sy
            else:
                east, north = x + sx, y + sy
        lon, lat = utm_inverse(east, north, geo["zone"], geo["north"])
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise OverlayError(f"lon / lat out of range with coords = {coords!r} "
                           "(lonlat and latlon mixed up?)")
    return lon, lat


def local_km(lon: Any, lat: Any, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """(east, north) in km of (lon, lat) from (lon0, lat0), flat earth at the latitude lat0."""
    kx = _KM_PER_DEG * math.cos(math.radians(lat0))
    return ((np.asarray(lon, float) - lon0) * kx, (np.asarray(lat, float) - lat0) * _KM_PER_DEG)


def local_lonlat(x: Any, y: Any, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of the local (east, north) in km from (lon0, lat0); inverse of local_km."""
    kx = _KM_PER_DEG * math.cos(math.radians(lat0))
    return lon0 + np.asarray(x, float) / kx, lat0 + np.asarray(y, float) / _KM_PER_DEG


# --- readers: text (GMT multi-segment), GeoJSON, KML / KMZ ---
def read_text_polylines(path: str) -> List[dict]:
    """
    Polylines of a text file: two numbers per line (the first two columns; comma,
    semicolon or white space), segments separated by a line that starts with ">"
    (GMT multi-segment) or by a blank line; "#" and "%" start a comment, lines
    that are no numbers are skipped. Returns [dict(xy=(N, 2) array, closed)];
    closed if the first and the last point are equal (N >= 4). Segments of
    fewer than 2 points are dropped.
    """
    parts: List[dict] = []
    cur: List[Tuple[float, float]] = []

    def flush() -> None:
        if len(cur) >= 2:
            xy = np.array(cur, float)
            parts.append(dict(xy=xy, closed=bool(len(cur) >= 4 and np.array_equal(xy[0], xy[-1]))))
        cur.clear()

    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith(">"):
                flush()
                continue
            if s[0] in "#%":
                continue
            tok = s.replace(",", " ").replace(";", " ").split()
            try:
                a, b = float(tok[0]), float(tok[1])
            except (ValueError, IndexError):
                continue
            if math.isfinite(a) and math.isfinite(b):
                cur.append((a, b))
    flush()
    return parts


def read_geojson_polylines(path: str) -> List[dict]:
    """
    Lines and polygon rings of a GeoJSON file (FeatureCollection, Feature, a
    geometry or a GeometryCollection): LineString, MultiLineString, Polygon,
    MultiPolygon; points are ignored. Coordinates are lon, lat. Returns
    [dict(xy, closed)] as read_text_polylines (rings are closed).
    """
    import json
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        obj = json.load(fh)
    parts: List[dict] = []

    def add(coords: Any, closed: bool) -> None:
        try:
            xy = np.array([[float(c[0]), float(c[1])] for c in coords], float)
        except (TypeError, ValueError, IndexError):
            return
        if xy.shape[0] >= 2 and np.all(np.isfinite(xy)):
            parts.append(dict(xy=xy, closed=closed))

    def walk(g: Any) -> None:
        if not isinstance(g, dict):
            return
        t = g.get("type")
        if t == "FeatureCollection":
            for f in g.get("features") or []:
                walk(f)
        elif t == "Feature":
            walk(g.get("geometry"))
        elif t == "GeometryCollection":
            for h in g.get("geometries") or []:
                walk(h)
        elif t == "LineString":
            add(g.get("coordinates") or [], False)
        elif t == "MultiLineString":
            for c in g.get("coordinates") or []:
                add(c, False)
        elif t == "Polygon":
            for r in g.get("coordinates") or []:
                add(r, True)
        elif t == "MultiPolygon":
            for p in g.get("coordinates") or []:
                for r in p:
                    add(r, True)

    walk(obj)
    return parts


def read_kml_polylines(path: str) -> List[dict]:
    """
    LineString and LinearRing (polygon boundaries; closed) elements of a KML or
    KMZ file (the first .kml of the archive, doc.kml preferred). Coordinates are
    lon, lat[, alt]. Returns [dict(xy, closed)] as read_text_polylines.
    """
    import xml.etree.ElementTree as ET
    try:
        if path.lower().endswith(".kmz"):
            with zipfile.ZipFile(path) as zf:
                kml = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kml:
                    raise ValueError("no .kml in the archive")
                data = zf.read("doc.kml" if "doc.kml" in kml else kml[0])
            root = ET.fromstring(data)
        else:
            root = ET.parse(path).getroot()
    except (ET.ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"not a readable KML / KMZ file: {exc}") from exc
    parts: List[dict] = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""
        if tag not in ("LineString", "LinearRing"):
            continue
        for c in el:
            if isinstance(c.tag, str) and c.tag.rsplit("}", 1)[-1] == "coordinates" and c.text:
                pts = []
                for tok in c.text.split():
                    v = tok.split(",")
                    try:
                        pts.append((float(v[0]), float(v[1])))
                    except (ValueError, IndexError):
                        continue
                xy = np.array(pts, float)
                if xy.shape[0] >= 2 and np.all(np.isfinite(xy)):
                    parts.append(dict(xy=xy, closed=(tag == "LinearRing")))
    return parts


def read_geometry_file(path: str) -> Tuple[List[dict], bool]:
    """
    (parts, geographic) of a polyline file by its extension: .geojson / .json,
    .kml / .kmz (coordinates always lon, lat: geographic = True), anything else is
    text (GMT multi-segment; the coordinates are those of `coords`). OSError and
    ValueError for a file that cannot be read.
    """
    low = str(path).lower()
    if low.endswith((".geojson", ".json")):
        return read_geojson_polylines(path), True
    if low.endswith((".kml", ".kmz")):
        return read_kml_polylines(path), True
    return read_text_polylines(path), False


# --- profiles and lines ---
def _as_points(pts: Any, what: str) -> np.ndarray:
    """(N, 2) float array of a sequence of pairs; OverlayError if it is none."""
    try:
        a = np.array([[float(p[0]), float(p[1])] for p in pts], float)
    except (TypeError, ValueError, IndexError) as exc:
        raise OverlayError(f"{what}: a sequence of (a, b) pairs is needed") from exc
    if a.ndim != 2 or a.shape[1] != 2:
        raise OverlayError(f"{what}: a sequence of (a, b) pairs is needed")
    return a


def _read_parts(spec: dict, geo: dict) -> List[dict]:
    """Parts [dict(lon, lat, closed)] of the file of `spec`; coords as given, or lon / lat."""
    path = str(spec["file"])
    try:
        raw, geographic = read_geometry_file(path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OverlayError(f"{path}: {exc}") from exc
    if not raw:
        raise OverlayError(f"{path}: no polyline found")
    coords = "lonlat" if geographic else spec.get("coords", "lonlat")
    out = []
    for p in raw:
        try:
            lon, lat = to_lonlat(p["xy"][:, 0], p["xy"][:, 1], coords, geo)
        except OverlayError as exc:
            raise OverlayError(f"{path}: {exc}") from exc
        out.append(dict(lon=lon, lat=lat, closed=bool(p["closed"])))
    return out


def _end_point(p: Any, what: str) -> Tuple[Tuple[float, float], Optional[str]]:
    """((a, b), label) of an end point of a profile: (a, b), or (a, b, "A") with the label of that
    end (a third entry that is no text is ignored). OverlayError for anything else."""
    try:
        label = None
        if len(p) >= 3 and isinstance(p[2], str) and p[2].strip():
            label = p[2].strip()
        return (float(p[0]), float(p[1])), label
    except (TypeError, ValueError, IndexError) as exc:
        raise OverlayError(f"{what}: (a, b) or (a, b, \"label\") is needed") from exc


def default_end_labels(name: str) -> Tuple[str, str]:
    """End labels of a profile from its name: "A-A'" -> ("A", "A'"), "P1" -> ("P1", "P1'")."""
    m = re.match(r"^\s*([^-]+?)\s*-\s*([^-]+?)\s*$", str(name))
    return (m.group(1), m.group(2)) if m else (str(name), str(name) + "'")


def style_entries(spec: dict, table: Dict[str, str]) -> Dict[str, Any]:
    """
    The style keys that an entry of PROFILES / LINES gives (`table`: _PROFILE_STYLE
    or _LINE_STYLE), with the aliases (linewidth, linestyle, ms) turned into the
    short names. A value None is the same as a key that is not given. OverlayError
    if a key and its alias are both given.
    """
    out: Dict[str, Any] = {}
    for key, val in spec.items():
        name = _STYLE_ALIASES.get(key, key)
        if name not in table or val is None:
            continue
        if name in out:
            raise OverlayError(f"{name!r} is given twice (also as an alias)")
        out[name] = val
    return out


def overlay_style(item: dict, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    The complete style of a prepared profile or line: its own entries, else the
    settings PROFILE_* / LINE_* of `cfg` (a setting that `cfg` does not hold:
    that of this script). Derived values are filled in: text and swath colour
    (the line colour), size and width of the distance ticks, tick length. A
    marker "" / "none" / False is None.
    """
    table = _PROFILE_STYLE if item["kind"] == "profile" else _LINE_STYLE
    st = {k: (item[k] if item.get(k) is not None else cfg.get(setting, globals()[setting]))
          for k, setting in table.items()}
    for k in ("marker", "end_marker"):
        if k in st and (st[k] is False or st[k] in ("", "none", "None")):
            st[k] = None
    if item["kind"] == "profile":
        st["text_color"] = st["text_color"] or st["color"]
        st["swath_color"] = st["swath_color"] or st["color"]
        if st["tick_fontsize"] is None:
            st["tick_fontsize"] = max(float(st["fontsize"]) - 2.0, 1.0)
        if st["tick_length"] is None:
            st["tick_length"] = _PROFILE_TICK_FRAC
        if st["tick_lw"] is None:
            st["tick_lw"] = max(0.8 * float(st["lw"]), 0.8)
    return st


def check_style_value(value: Any, kind: str, what: str) -> None:
    """ValueError if `value` is no style value of the kind `kind` (see _STYLE_KINDS)."""
    from matplotlib.colors import is_color_like
    is_num = (isinstance(value, (int, float, np.integer, np.floating))
              and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value)))
    if kind.endswith("_or_none"):
        if value is None:
            return
        kind = kind[:-len("_or_none")]
    if kind == "color":
        if not is_color_like(value):
            raise ValueError(f"{what} = {value!r} is not a matplotlib colour")
    elif kind == "ls":
        from matplotlib.lines import Line2D
        try:
            Line2D([], [], linestyle=value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{what} = {value!r} is not a matplotlib line style: {exc}") from None
    elif kind == "marker":
        if value is None or value is False or value in ("", "none", "None"):
            return
        from matplotlib.markers import MarkerStyle
        try:
            MarkerStyle(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{what} = {value!r} is no matplotlib marker: {exc}") from None
    elif kind == "weight":
        ok = (value in _FONT_WEIGHTS) or (is_num and 0 <= value <= 1000)
        if not ok:
            raise ValueError(f"{what} = {value!r}: a font weight is 0 .. 1000 or one of "
                             f"{list(_FONT_WEIGHTS)}")
    elif kind == "bool":
        if not isinstance(value, (bool, np.bool_, int)):
            raise ValueError(f"{what} = {value!r} must be True or False")
    else:
        need = {"pos": "a number > 0", "nonneg": "a number >= 0", "unit": "a number in 0 .. 1",
                "num": "a number"}[kind]
        ok = is_num and {"pos": lambda v: v > 0, "nonneg": lambda v: v >= 0,
                         "unit": lambda v: 0 <= v <= 1, "num": lambda v: True}[kind](value)
        if not ok:
            raise ValueError(f"{what} = {value!r} must be {need}")


def font_rc(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The matplotlib rcParams of the font settings (FONT_FAMILY, FONT_RC)."""
    rc: Dict[str, Any] = {}
    fam = cfg.get("FONT_FAMILY")
    if fam:
        rc["font.family"] = [fam] if isinstance(fam, str) else list(fam)
    rc.update(cfg.get("FONT_RC") or {})
    return rc


def missing_fonts(family: Union[None, str, Sequence[str]]) -> List[str]:
    """The names in `family` that are no generic family and no font matplotlib can find."""
    if not family:
        return []
    from matplotlib import font_manager
    out = []
    for name in ([family] if isinstance(family, str) else family):
        if str(name).lower() in _GENERIC_FAMILIES:
            continue
        try:
            font_manager.findfont(font_manager.FontProperties(family=[str(name)]),
                                  fallback_to_default=False)
        except ValueError:
            out.append(str(name))
    return out


def font_size(cfg: Dict[str, Any], key: str, offset: float = 0.0) -> float:
    """The setting `key` [pt]; None = TICK_FONTSIZE + offset."""
    v = cfg.get(key)
    return max(float(cfg["TICK_FONTSIZE"]) + offset, 1.0) if v is None else float(v)


def check_appearance(cfg: Dict[str, Any]) -> None:
    """ValueError for a style, font or panel size setting that cannot work."""
    for table in (_PROFILE_STYLE, _LINE_STYLE):
        for key, setting in table.items():
            check_style_value(cfg[setting], _STYLE_KINDS[key], setting)
    for key in ("TICK_FONTSIZE", "LABEL_FONTSIZE", "CREDIT_FONTSIZE"):
        check_style_value(cfg[key], "pos", key)
    for key in ("AXIS_LABEL_FONTSIZE", "TITLE_FONTSIZE", "PANEL_TITLE_FONTSIZE",
                "LEGEND_FONTSIZE", "CBAR_LABEL_FONTSIZE", "CBAR_TICK_FONTSIZE",
                "SCALEBAR_FONTSIZE"):
        check_style_value(cfg[key], "pos_or_none", key)
    for key in ("TITLE_FONTWEIGHT", "LABEL_FONTWEIGHT"):
        check_style_value(cfg[key], "weight", key)
    for key, kind in (("MAP_HALO", "bool"), ("MAP_HALO_COLOR", "color"),
                      ("MAP_HALO_WIDTH", "nonneg")):
        check_style_value(cfg[key], kind, key)
    fam = cfg["FONT_FAMILY"]
    if fam is not None and not (isinstance(fam, str) or (
            isinstance(fam, (list, tuple)) and fam and all(isinstance(f, str) for f in fam))):
        raise ValueError(f"FONT_FAMILY = {fam!r}: must be None, a name or a list of names")
    rc = cfg["FONT_RC"]
    if rc is not None:
        if not isinstance(rc, dict):
            raise ValueError("FONT_RC must be None or a dict of matplotlib rcParams")
        import matplotlib
        for key, val in rc.items():
            if key not in matplotlib.rcParams:
                raise ValueError(f"FONT_RC: {key!r} is not a matplotlib rcParam")
            try:
                with matplotlib.rc_context({key: val}):
                    pass
            except (ValueError, TypeError) as exc:
                raise ValueError(f"FONT_RC[{key!r}] = {val!r}: {exc}") from None
    ps = cfg["PANEL_SIZE"]
    try:
        ok = len(ps) == 2 and all(float(v) > 0 and np.isfinite(float(v)) for v in ps)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"PANEL_SIZE = {ps!r}: must be (width, height) in cm, both > 0")


#: Largest |x|, |y| (km) of overlay positions given in the settings.
_SETTING_KM_LIMITS = {"utm": (2000.0, 20000.0), "model": (1000.0, 1000.0)}


def _settings_xy(a: np.ndarray, coords: str, what: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    (x, y) of the (N, 2) positions of a PROFILES / LINES entry in the units of
    to_lonlat: "utm" and "model" positions are km in the settings and m there;
    "lonlat" / "latlon" are passed on. ValueError if they look like metres.
    """
    x, y = a[:, 0], a[:, 1]
    if coords not in _SETTING_KM_LIMITS:
        return x, y
    for v, limit in zip((x, y), _SETTING_KM_LIMITS[coords]):
        v = v[np.isfinite(v)]
        if v.size and np.max(np.abs(v)) > limit:
            raise ValueError(
                f"{what}: positions with coords = {coords!r} are given in km; "
                f"{v[np.argmax(np.abs(v))]:g} looks like metres")
    return x * 1000.0, y * 1000.0


def prepare_profile(spec: dict, geo: dict, index: int = 0) -> dict:
    """
    One profile from its spec (see PROFILES): the vertices in lon / lat and in
    km (flat earth around their mean position, origin lon0 / lat0), cumulative
    distance cum, length, the swath polygons (lon / lat) if width_km > 0, and
    the style entries of the spec (keys of _PROFILE_STYLE; color, lw, ls are
    None if not given). OverlayError if the geometry cannot be made.
    """
    name = str(spec.get("name") or f"P{index + 1}")
    coords = spec.get("coords", "lonlat")
    end_labels: Tuple[Optional[str], Optional[str]] = (None, None)
    try:
        if "points" in spec:
            a = _as_points(spec["points"], "points")
            lon, lat = to_lonlat(*_settings_xy(a, coords, name), coords, geo)
            source = "points"
        elif "file" in spec and spec["file"]:
            parts = _read_parts(spec, geo)
            if len(parts) > 1:
                _say(f"WARNING: profile {name!r}: {spec['file']} holds {len(parts)} polylines; "
                     "the first is used.")
            lon, lat = parts[0]["lon"], parts[0]["lat"]
            source = str(spec["file"])
        elif "start" in spec:
            (p0, l0), (p1, l1) = _end_point(spec["start"], "start"), _end_point(spec["end"], "end")
            a = _as_points([p0, p1], "start / end")
            end_labels = (l0, l1)
            lon, lat = to_lonlat(*_settings_xy(a, coords, name), coords, geo)
            source = "start, end"
        else:
            c = _as_points([spec["center"]], "center")
            c_lon, c_lat = to_lonlat(*_settings_xy(c, coords, name), coords, geo)
            half = 0.5 * float(spec["length_km"])
            az = math.radians(float(spec["azimuth"]))
            kx = _KM_PER_DEG * math.cos(math.radians(float(c_lat[0])))
            lon = float(c_lon[0]) + np.array([-half, half]) * math.sin(az) / kx
            lat = float(c_lat[0]) + np.array([-half, half]) * math.cos(az) / _KM_PER_DEG
            source = "center, azimuth, length_km"
    except KeyError as exc:
        raise OverlayError(f"missing {exc}") from exc
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    x, y = local_km(lon, lat, lon0, lat0)
    keep = np.ones(x.size, bool)                        # no zero-length segments
    keep[1:] = np.hypot(np.diff(x), np.diff(y)) > 1e-6
    lon, lat, x, y = lon[keep], lat[keep], x[keep], y[keep]
    if x.size < 2:
        raise OverlayError("a profile needs two different points")
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.r_[0.0, np.cumsum(seg)]
    width = float(spec.get("width_km") or 0.0)
    ticks = spec.get("ticks_km")
    ticks = float(ticks) if ticks else None
    if ticks is not None and cum[-1] / ticks > _MAX_TICKS:
        raise OverlayError(f"ticks_km = {ticks:g} gives more than {_MAX_TICKS} ticks on "
                           f"{cum[-1]:.1f} km")
    lab = spec.get("labels")
    if lab is False:
        labels = None
    else:                                  # a label in start / end, else labels, else from the name
        base = default_end_labels(name) if lab is None else (str(lab[0]), str(lab[1]))
        labels = (end_labels[0] or base[0], end_labels[1] or base[1])
    prof = dict(kind="profile", name=name, source=source, labels=labels, lon=lon, lat=lat,
                x=x, y=y, lon0=lon0, lat0=lat0, cum=cum, length=float(cum[-1]), width=width,
                ticks_km=ticks, tick_labels=bool(spec.get("tick_labels", True)),
                color=None, lw=None, ls=None, label=spec.get("label"))
    prof.update(style_entries(spec, _PROFILE_STYLE))   # the own style of the entry
    prof["swath"] = profile_swath(prof) if width > 0 else []
    return prof


def profile_swath(prof: dict, n_circle: int = 24) -> List[np.ndarray]:
    """
    The swath of a profile as (N, 2) lon / lat polygons that all have the same
    orientation: one rectangle of the width per segment, and a disc at every
    inner vertex (rounded corners); together they are the set of points within
    width / 2 of the profile, without end caps.
    """
    w2 = 0.5 * float(prof["width"])
    x, y = prof["x"], prof["y"]
    polys: List[np.ndarray] = []
    for i in range(x.size - 1):
        d = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        nrm = np.array([-d[1], d[0]]) / np.hypot(*d) * w2
        a, b = np.array([x[i], y[i]]), np.array([x[i + 1], y[i + 1]])
        polys.append(np.array([a - nrm, b - nrm, b + nrm, a + nrm]))
    ang = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    circ = np.c_[np.cos(ang), np.sin(ang)] * w2
    for i in range(1, x.size - 1):
        polys.append(circ + np.array([x[i], y[i]]))
    out = []
    for p in polys:
        lo, la = local_lonlat(p[:, 0], p[:, 1], prof["lon0"], prof["lat0"])
        out.append(np.c_[lo, la])
    return out


def project_on_profile(prof: dict, lon: Any, lat: Any) -> dict:
    """
    Position of points relative to a profile: along (km from the start, along the
    polyline), offset (km, signed distance to the polyline, positive to the right
    of the direction start -> end), dist (|offset|) and inside (within width / 2
    of the polyline, and not beyond the start or the end of it; all False
    without a width). Flat earth around the profile (local_km).
    """
    x, y = local_km(lon, lat, prof["lon0"], prof["lat0"])
    px, py = prof["x"], prof["y"]
    ax_, ay_ = px[:-1, None], py[:-1, None]
    dx, dy = (px[1:] - px[:-1])[:, None], (py[1:] - py[:-1])[:, None]
    l2 = dx * dx + dy * dy
    vx, vy = x[None, :] - ax_, y[None, :] - ay_
    t_raw = (vx * dx + vy * dy) / l2
    t = np.clip(t_raw, 0.0, 1.0)
    dist = np.hypot(vx - t * dx, vy - t * dy)
    k = np.argmin(dist, axis=0)
    cols = np.arange(x.size)
    dist_k, t_k, traw_k = dist[k, cols], t[k, cols], t_raw[k, cols]
    seg_len = np.sqrt(l2[:, 0])
    along = prof["cum"][k] + t_k * seg_len[k]
    cross = dx[k, 0] * vy[k, cols] - dy[k, 0] * vx[k, cols]
    offset = np.where(cross > 0.0, -dist_k, dist_k)
    if prof["width"] > 0:
        nseg = px.size - 1
        past = (traw_k - np.clip(traw_k, 0.0, 1.0)) * seg_len[k]       # km beyond the segment
        beyond = ((k == 0) & (past < -_END_TOL_KM)) | ((k == nseg - 1) & (past > _END_TOL_KM))
        inside = (dist_k <= 0.5 * prof["width"] * (1.0 + 1e-9)) & ~beyond
    else:
        inside = np.zeros(x.size, bool)
    return dict(along=along, offset=offset, dist=dist_k, inside=inside)


def project_profiles(profiles: Sequence[dict], lon: np.ndarray, lat: np.ndarray) -> List[dict]:
    """
    The profiles with the projection of the sites (lon, lat): proj (project_on_profile),
    order (indices of the sites inside the swath, by increasing along) and ring (their
    lon, lat).
    """
    out = []
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    for p in profiles:
        pr = project_on_profile(p, lon, lat)
        idx = np.flatnonzero(pr["inside"])
        idx = idx[np.argsort(pr["along"][idx], kind="stable")]
        out.append(dict(p, proj=pr, order=idx, ring=(lon[idx], lat[idx])))
    return out


def profile_tick_points(prof: dict, spacing: float) -> dict:
    """
    Positions s = 0, spacing, 2 * spacing, ... <= length on a profile: s (km), x, y (km,
    local frame of the profile) and the unit direction ux, uy of the segment.
    """
    s = np.minimum(np.arange(int(math.floor(prof["length"] / float(spacing) + 1e-9)) + 1)
                   * float(spacing), prof["length"])
    nseg = prof["x"].size - 1
    i = np.clip(np.searchsorted(prof["cum"], s, side="right") - 1, 0, nseg - 1)
    dx = prof["x"][i + 1] - prof["x"][i]
    dy = prof["y"][i + 1] - prof["y"][i]
    ln = np.hypot(dx, dy)
    ux, uy = dx / ln, dy / ln
    t = np.minimum(s - prof["cum"][i], ln)
    return dict(s=s, x=prof["x"][i] + t * ux, y=prof["y"][i] + t * uy, ux=ux, uy=uy)


def prepare_line(spec: dict, geo: dict) -> dict:
    """
    One line (polylines and polygons) from its spec (see LINES): parts =
    [dict(lon, lat, closed)] and the style entries. OverlayError if the geometry
    cannot be made.
    """
    coords = spec.get("coords", "lonlat")
    if spec.get("file"):
        parts = _read_parts(spec, geo)
    else:
        if "points" in spec:
            groups = [spec["points"]]
        else:
            groups = list(spec["segments"])
        parts = []
        for g in groups:
            a = _as_points(g, "points")
            if a.shape[0] < 2:
                raise OverlayError("a polyline needs two points")
            lon, lat = to_lonlat(
                *_settings_xy(a, coords, str(spec.get("label") or "line")), coords, geo)
            parts.append(dict(lon=lon, lat=lat, closed=False))
    if spec.get("closed") is not None:
        for p in parts:
            p["closed"] = bool(spec["closed"])
    line = dict(kind="line", label=spec.get("label"), parts=parts, fill=spec.get("fill"),
                color=None, lw=None, ls=None, halo=None, alpha=None,
                n_vertices=int(sum(p["lon"].size for p in parts)))
    line.update(style_entries(spec, _LINE_STYLE))      # the own style of the entry
    return line


def prepare_overlays(cfg: Dict[str, Any], geo: dict) -> Tuple[List[dict], List[dict]]:
    """
    (profiles, lines) of the settings PROFILES and LINES. An item that cannot be made
    is skipped with a warning (OverlayError for the first one if OVERLAY_STRICT).
    """
    strict = bool(cfg.get("OVERLAY_STRICT"))
    profiles: List[dict] = []
    lines: List[dict] = []
    for i, spec in enumerate(cfg.get("PROFILES") or []):
        what = f"profile {str(spec.get('name') or 'P' + str(i + 1))!r}"
        try:
            profiles.append(prepare_profile(spec, geo, i))
        except OverlayError as exc:
            if strict:
                raise OverlayError(f"{what}: {exc}") from exc
            _say(f"WARNING: {what} skipped: {exc}")
    for i, spec in enumerate(cfg.get("LINES") or []):
        what = f"line {str(spec.get('label') or spec.get('file') or '#' + str(i + 1))!r}"
        try:
            lines.append(prepare_line(spec, geo))
        except OverlayError as exc:
            if strict:
                raise OverlayError(f"{what}: {exc}") from exc
            _say(f"WARNING: {what} skipped: {exc}")
    return profiles, lines


def check_overlay_specs(cfg: Dict[str, Any]) -> None:
    """ValueError for PROFILES / LINES entries that cannot work (keys, options, numbers)."""
    def number(spec: dict, key: str, what: str, sign: str = "pos") -> None:
        v = spec.get(key)
        if v is None:
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what}: {key} = {v!r} is not a number") from None
        need = {"pos": "> 0", "nonneg": ">= 0", "any": "finite"}[sign]
        bad = (not math.isfinite(f) or (sign == "pos" and f <= 0.0)
               or (sign == "nonneg" and f < 0.0))
        if bad:
            raise ValueError(f"{what}: {key} = {v!r} must be {need}")

    for key, allowed in (("PROFILES", _PROFILE_KEYS), ("LINES", _LINE_KEYS)):
        specs = cfg.get(key) or []
        if not isinstance(specs, (list, tuple)):
            raise ValueError(f"{key} must be a list of dicts")
        for i, spec in enumerate(specs):
            what = f"{key}[{i}]"
            if not isinstance(spec, dict):
                raise ValueError(f"{what} must be a dict")
            bad = [k for k in spec if k not in allowed]
            if bad:
                raise ValueError(f"{what}: unknown key {bad[0]!r}; allowed: {list(allowed)}")
            if spec.get("coords", "lonlat") not in _COORDS:
                raise ValueError(f"{what}: coords = {spec.get('coords')!r} must be one of "
                                 f"{list(_COORDS)}")
            table = _PROFILE_STYLE if key == "PROFILES" else _LINE_STYLE
            try:
                own = style_entries(spec, table)
            except OverlayError as exc:
                raise ValueError(f"{what}: {exc}") from None
            for name, val in own.items():
                check_style_value(val, _STYLE_KINDS[name], f"{what}: {name}")
            if spec.get("fill") is not None and spec.get("fill") is not False:
                check_style_value(spec["fill"], "color", f"{what}: fill")
            coords = spec.get("coords", "lonlat")
            groups = [spec.get("points")] + list(spec.get("segments") or [])
            groups += [[spec[k]] for k in ("start", "end", "center") if k in spec]
            for g in groups:
                if g is None:
                    continue
                try:
                    a = _as_points(g, what)
                except OverlayError:
                    continue                  # reported when the item is made
                _settings_xy(a, coords, what)
            if key == "PROFILES":
                modes = [m for m, on in (
                    ("points", "points" in spec),
                    ("file", bool(spec.get("file"))),
                    ("start + end", "start" in spec or "end" in spec),
                    ("center + azimuth + length_km",
                     any(k in spec for k in ("center", "azimuth", "length_km")))) if on]
                if len(modes) != 1:
                    raise ValueError(f"{what}: exactly one of points, file, start + end, center + "
                                     f"azimuth + length_km is needed (found {modes or 'none'})")
                if modes[0] == "start + end" and not ("start" in spec and "end" in spec):
                    raise ValueError(f"{what}: start and end are both needed")
                if modes[0].startswith("center") and not all(
                        k in spec for k in ("center", "azimuth", "length_km")):
                    raise ValueError(f"{what}: center, azimuth and length_km are all needed")
                number(spec, "length_km", what)
                number(spec, "azimuth", what, "any")
                number(spec, "width_km", what, "nonneg")
                number(spec, "ticks_km", what)
                lab = spec.get("labels")
                if lab is not None and lab is not False and not (
                        isinstance(lab, (list, tuple)) and len(lab) == 2):
                    raise ValueError(f"{what}: labels must be False or two texts (start, end)")
            else:
                modes = [m for m in ("file", "points", "segments") if spec.get(m)]
                if len(modes) != 1:
                    raise ValueError(f"{what}: exactly one of file, points, segments is needed "
                                     f"(found {modes or 'none'})")


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def make_norm(values: np.ndarray, *, vmin: Optional[float], vmax: Optional[float],
              reference: Optional[float], floor: Optional[float] = 0.0,
              name: str = "NRMS_REFERENCE"):
    """
    (norm, extend) of a colour scale (default: nRMS). With a `reference`, a two-slope
    scale centred on it (limits from the data, symmetric around the reference but not
    below `floor` (None = no limit), unless given); without, a linear one. `extend` =
    "min", "max", "both" or "neither" for values beyond the limits (colour bar
    arrows). `name` = the setting named in the warning.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    vlo = float(v.min()) if v.size else 0.0
    vhi = float(v.max()) if v.size else 1.0
    if reference is None:
        lo = vlo if vmin is None else float(vmin)
        hi = vhi if vmax is None else float(vmax)
        if not hi > lo:
            hi = lo + 1.0
        norm = Normalize(vmin=lo, vmax=hi)
    else:
        ref = float(reference)
        spread = max(abs(vhi - ref), abs(ref - vlo), 0.1)
        if vmin is None:
            lo = ref - spread if floor is None else max(ref - spread, floor)
        else:
            lo = float(vmin)
        hi = ref + spread if vmax is None else float(vmax)
        if lo < ref < hi:
            norm = TwoSlopeNorm(vcenter=ref, vmin=lo, vmax=hi)
        else:
            _say(f"WARNING: {name} = {ref:g} is not between the colour limits "
                 f"{lo:g} and {hi:g}; linear colour scale.")
            if not hi > lo:
                hi = lo + 1.0
            norm = Normalize(vmin=lo, vmax=hi)
    below = bool(v.size and vlo < lo)
    above = bool(v.size and vhi > hi)
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    return norm, extend


def _nice_length(target: float) -> float:
    """Largest of 1, 2, 5 x 10^k that is <= target."""
    if not target > 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(target))
    for f in (5.0, 2.0, 1.0):
        if f * mag <= target * (1.0 + 1e-12):
            return f * mag
    return mag


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_") or "x"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def clip_polyline(x: Any, y: Any, box: Sequence[float], cum: Optional[Any] = None) -> List[dict]:
    """
    The parts of the polyline (x, y) inside the rectangle box = (xa, xb, ya, yb)
    (edges included), as a list of dict(x, y, s0, s1): the vertices of the part
    and the distance along the polyline (in the units of x and y, or `cum`, the
    cumulative lengths of the vertices) at its first and last point. A polyline
    that leaves and enters again gives several parts; a part that touches the
    rectangle in a point is dropped. Every segment is clipped with the Liang-
    Barsky algorithm, the clipped segments that follow each other are joined.
    """
    xa, xb, ya, yb = (float(v) for v in box)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cum = (np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] if cum is None
           else np.asarray(cum, float))
    done: List[dict] = []
    cur: Optional[List[Tuple[float, float, float]]] = None

    def close() -> None:
        if cur is not None and len(cur) >= 2 and cur[-1][2] - cur[0][2] > 1e-9:
            a = np.array(cur)
            done.append(dict(x=a[:, 0], y=a[:, 1], s0=float(a[0, 2]), s1=float(a[-1, 2])))

    for i in range(x.size - 1):
        x0, y0, dx, dy = x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i]
        t0, t1, ok = 0.0, 1.0, True
        for p, q in ((-dx, x0 - xa), (dx, xb - x0), (-dy, y0 - ya), (dy, yb - y0)):
            if p == 0.0:
                if q < 0.0:
                    ok = False
                    break
                continue
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                ok = False
                break
        if not ok:
            close()
            cur = None
            continue
        ln = cum[i + 1] - cum[i]
        a = (x0 + t0 * dx, y0 + t0 * dy, cum[i] + t0 * ln)
        b = (x0 + t1 * dx, y0 + t1 * dy, cum[i] + t1 * ln)
        if cur is None or t0 != 0.0:                      # a new part (else: goes on from the vertex)
            close()
            cur = [a]
        if abs(b[2] - cur[-1][2]) > 1e-12:                # no repeated points
            cur.append(b)
        if t1 < 1.0:                                      # leaves the rectangle in this segment
            close()
            cur = None
    close()
    return done


def clip_polygon(x: Any, y: Any, box: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """The polygon (x, y) cut at the rectangle box = (xa, xb, ya, yb) (Sutherland-Hodgman);
    empty arrays if nothing is left."""
    xa, xb, ya, yb = (float(v) for v in box)
    pts = list(zip(np.asarray(x, float), np.asarray(y, float)))
    edges = ((lambda p: p[0] >= xa, lambda a, b: (xa, a[1] + (b[1] - a[1]) * (xa - a[0]) / (b[0] - a[0]))),
             (lambda p: p[0] <= xb, lambda a, b: (xb, a[1] + (b[1] - a[1]) * (xb - a[0]) / (b[0] - a[0]))),
             (lambda p: p[1] >= ya, lambda a, b: (a[0] + (b[0] - a[0]) * (ya - a[1]) / (b[1] - a[1]), ya)),
             (lambda p: p[1] <= yb, lambda a, b: (a[0] + (b[0] - a[0]) * (yb - a[1]) / (b[1] - a[1]), yb)))
    for inside, cross in edges:
        if not pts:
            break
        out = []
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
        pts = out
    if len(pts) < 3:
        return np.empty(0), np.empty(0)
    a = np.array(pts, float)
    return a[:, 0], a[:, 1]


def cut_box_lonlat(bbox: Sequence[float], margin_km: float) -> Tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of the map `bbox` made smaller by `margin_km` on every
    side. OverlayError if the margin leaves nothing."""
    dlat = float(margin_km) / _KM_PER_DEG
    dlon = float(margin_km) / (_KM_PER_DEG * math.cos(math.radians(0.5 * (bbox[2] + bbox[3]))))
    box = (bbox[0] + dlon, bbox[1] - dlon, bbox[2] + dlat, bbox[3] - dlat)
    if not (box[0] < box[1] and box[2] < box[3]):
        raise OverlayError(f"OVERLAY_CUT_MARGIN_KM = {margin_km:g} km is larger than the map allows")
    return box


def cut_profile(prof: dict, bbox: Optional[Sequence[float]], margin_km: Optional[float]) -> dict:
    """
    The profile `prof` cut at the map `bbox` (lon_min, lon_max, lat_min, lat_max) made smaller by
    `margin_km` on every side: a copy with `pieces` = [dict(x, y, lon, lat, s0, s1)] (the parts
    inside, x, y in km as prof["x"], s0 / s1 the distance from the start of the profile at their
    ends, so the distances of the ticks stay those of the whole profile), `cut` (True if
    anything was cut off) and, if it was cut, the swath of the parts (flat at the cut).
    Without `margin_km` or `bbox` the profile is not cut. OverlayError if the margin leaves
    no map.
    """
    whole = dict(x=prof["x"], y=prof["y"], lon=prof["lon"], lat=prof["lat"], s0=0.0,
                 s1=prof["length"])
    if margin_km is None or bbox is None:
        return dict(prof, pieces=[whole], cut=False)
    lo0, lo1, la0, la1 = cut_box_lonlat(bbox, margin_km)
    xa, ya = local_km(lo0, la0, prof["lon0"], prof["lat0"])
    xb, yb = local_km(lo1, la1, prof["lon0"], prof["lat0"])
    pieces = clip_polyline(prof["x"], prof["y"], (float(xa), float(xb), float(ya), float(yb)),
                           cum=prof["cum"])
    for pc in pieces:
        pc["lon"], pc["lat"] = local_lonlat(pc["x"], pc["y"], prof["lon0"], prof["lat0"])
    if (len(pieces) == 1 and pieces[0]["s0"] <= 1e-9
            and pieces[0]["s1"] >= prof["length"] - 1e-9):
        return dict(prof, pieces=[whole], cut=False)
    out = dict(prof, pieces=pieces, cut=True)
    if prof["width"] > 0:
        out["swath"] = [poly for pc in pieces for poly in profile_swath(
            dict(x=pc["x"], y=pc["y"], width=prof["width"], lon0=prof["lon0"], lat0=prof["lat0"]))]
    return out


def cut_line(ln: dict, bbox: Optional[Sequence[float]], margin_km: Optional[float]) -> dict:
    """
    The line `ln` (polylines and polygons) cut at the map `bbox` made smaller by `margin_km`: a
    copy with `drawn` = [dict(lon, lat, closed, fill = (lon, lat) of the polygon cut at the box
    or None)], the polylines to draw. A closed part is cut as its outline, its fill as a polygon.
    Without `margin_km` or `bbox` the parts are returned as they are.
    """
    if margin_km is None or bbox is None:
        return dict(ln, drawn=[dict(lon=p["lon"], lat=p["lat"], closed=p["closed"], fill=None,
                                    cut=False) for p in ln["parts"]])
    box = cut_box_lonlat(bbox, margin_km)
    rect = (box[0], box[1], box[2], box[3])
    drawn: List[dict] = []
    for part in ln["parts"]:
        lon, lat = np.asarray(part["lon"], float), np.asarray(part["lat"], float)
        if part["closed"]:
            lon, lat = np.r_[lon, lon[0]], np.r_[lat, lat[0]]
        pieces = clip_polyline(lon, lat, rect)
        inside = (len(pieces) == 1 and pieces[0]["x"].size == lon.size
                  and np.allclose(pieces[0]["x"], lon) and np.allclose(pieces[0]["y"], lat))
        fill = None
        if ln.get("fill") and part["closed"]:
            fx, fy = clip_polygon(part["lon"], part["lat"], rect)
            fill = (fx, fy)
        if inside:
            drawn.append(dict(lon=part["lon"], lat=part["lat"], closed=part["closed"], fill=fill,
                              cut=False))
            continue
        for k, pc in enumerate(pieces):
            drawn.append(dict(lon=pc["x"], lat=pc["y"], closed=False,
                              fill=fill if k == 0 else None, cut=True))
        if not pieces and fill is not None and fill[0].size:
            drawn.append(dict(lon=np.empty(0), lat=np.empty(0), closed=False, fill=fill, cut=True))
    return dict(ln, drawn=drawn)


def cut_report(profiles: Sequence[dict], lines: Sequence[dict], bbox: Sequence[float],
               margin_km: Optional[float]) -> List[str]:
    """Messages about what the cut at the map box takes off: one per profile (the km kept of
    its length) and per line that is cut."""
    if margin_km is None:
        return []
    out: List[str] = []
    for pr in profiles:
        c = cut_profile(pr, bbox, margin_km)
        if c["cut"]:
            kept = sum(pc["s1"] - pc["s0"] for pc in c["pieces"])
            out.append(f"profile {pr['name']!r} is cut at the map box (margin {float(margin_km):g} km): "
                       f"{kept:.1f} of {pr['length']:.1f} km are shown.")
    for ln in lines:
        c = cut_line(ln, bbox, margin_km)
        if any(d["cut"] for d in c["drawn"]):
            out.append(f"line {str(ln.get('label') or 'line')!r} is cut at the map box "
                       f"(margin {float(margin_km):g} km).")
    return out


def _cut_margin(cfg: Dict[str, Any], bbox: Optional[Sequence[float]]) -> Optional[float]:
    """OVERLAY_CUT_MARGIN_KM if the map box can have it (else a message, said once, and None)."""
    m = cfg.get("OVERLAY_CUT_MARGIN_KM")
    if m is None or bbox is None:
        return None
    try:
        cut_box_lonlat(bbox, m)
    except OverlayError as exc:
        _say_once(f"WARNING: {exc}: profiles and lines are not cut.")
        return None
    return float(m)


def draw_overlays(ax, *, frame: dict, overlays: Optional[dict], cfg: Dict[str, Any],
                  bbox: Optional[Sequence[float]] = None) -> None:
    """
    Draws the lines and the profiles on the map `ax` (limits already set): lines
    (and polygon fills) below the sites, profiles with swath, ticks, distance
    labels, end labels and rings around the sites in the swath. `overlays` =
    dict(profiles=[...], lines=[...]) as made by prepare_overlays / project_profiles.
    With `bbox` (the map box, lon / lat) and OVERLAY_CUT_MARGIN_KM they are cut at the
    box made smaller by the margin (cut_profile, cut_line): the end labels stand at the
    ends of the cut profile, the ticks keep the distances of the whole profile.
    """
    if not overlays:
        return
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    x0, x1 = ax.get_xlim()
    kx_plot = frame["kx"] if frame["display"] == "latlon" else 1.0
    map_km = abs(x1 - x0) * kx_plot                       # width of the map [km]
    margin = _cut_margin(cfg, bbox)

    from matplotlib.path import Path
    box = None if margin is None else cut_box_lonlat(bbox, margin)

    for ln0 in overlays.get("lines") or []:
        ln = cut_line(ln0, bbox, margin)
        sty = overlay_style(ln, cfg)
        color, lw, z = sty["color"], float(sty["lw"]), float(sty["zorder"])
        pe = ([patheffects.withStroke(linewidth=lw + float(sty["halo_width"]),
                                      foreground=sty["halo_color"])] if sty["halo"] else [])
        for part in ln["drawn"]:
            px, py = to_frame(frame, part["lon"], part["lat"])
            if part["closed"]:
                px, py = np.r_[px, px[0]], np.r_[py, py[0]]
            if ln.get("fill") and part["closed"]:
                ax.fill(px, py, facecolor=ln["fill"], edgecolor="none",
                        alpha=float(sty["alpha"]), zorder=z - 0.5)
            elif part["fill"] is not None and part["fill"][0].size:     # the polygon cut at the box
                fx, fy = to_frame(frame, part["fill"][0], part["fill"][1])
                ax.fill(fx, fy, facecolor=ln["fill"], edgecolor="none", alpha=float(sty["alpha"]),
                        zorder=z - 0.5)
            if px.size:
                ax.plot(px, py, color=color, lw=lw, ls=sty["ls"], alpha=float(sty["line_alpha"]),
                        zorder=z, path_effects=pe, solid_joinstyle="round")
        if sty["marker"] is not None:                  # at the vertices of the line (inside the cut box)
            vlon = np.concatenate([np.asarray(p["lon"], float) for p in ln0["parts"]])
            vlat = np.concatenate([np.asarray(p["lat"], float) for p in ln0["parts"]])
            if box is not None:
                keep = ((vlon >= box[0]) & (vlon <= box[1]) & (vlat >= box[2]) & (vlat <= box[3]))
                vlon, vlat = vlon[keep], vlat[keep]
            if vlon.size:
                vx, vy = to_frame(frame, vlon, vlat)
                ax.plot(vx, vy, ls="", marker=sty["marker"], ms=float(sty["markersize"]),
                        color=color, alpha=float(sty["line_alpha"]), zorder=z + 0.05)

    for pr0 in overlays.get("profiles") or []:
        pr = cut_profile(pr0, bbox, margin)
        if not pr["pieces"]:
            continue                                       # nothing of it inside the map
        sty = overlay_style(pr, cfg)
        color, lw, z = sty["color"], float(sty["lw"]), float(sty["zorder"])
        line_alpha = float(sty["line_alpha"])
        text_z = max(6.5, z + 0.2)
        pe = ([patheffects.withStroke(linewidth=lw + float(sty["halo_width"]),
                                      foreground=sty["halo_color"])] if sty["halo"] else [])
        halo_t = ([patheffects.withStroke(linewidth=2.0, foreground=sty["halo_color"])]
                  if sty["halo"] else [])
        if pr.get("swath"):
            verts: List[np.ndarray] = []
            codes: List[int] = []
            for poly in pr["swath"]:
                qx, qy = to_frame(frame, poly[:, 0], poly[:, 1])
                verts.append(np.c_[np.r_[qx, qx[0]], np.r_[qy, qy[0]]])
                codes += [Path.MOVETO] + [Path.LINETO] * (len(qx) - 1) + [Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(np.vstack(verts), codes), facecolor=sty["swath_color"],
                                   edgecolor="none", alpha=float(sty["swath_alpha"]),
                                   zorder=min(2.0, z - 0.5)))
        for pc in pr["pieces"]:
            px, py = to_frame(frame, pc["lon"], pc["lat"])
            ax.plot(px, py, color=color, lw=lw, ls=sty["ls"], alpha=line_alpha, zorder=z,
                    path_effects=pe, solid_capstyle="butt", solid_joinstyle="round")
        if sty["marker"] is not None:                      # at the vertices on the parts that are left
            cum = np.asarray(pr0["cum"], float)
            keep = np.zeros(cum.size, bool)
            for pc in pr["pieces"]:
                keep |= (cum >= pc["s0"] - 1e-6) & (cum <= pc["s1"] + 1e-6)
            if keep.any():
                vx, vy = to_frame(frame, pr0["lon"][keep], pr0["lat"][keep])
                ax.plot(vx, vy, ls="", marker=sty["marker"], ms=float(sty["markersize"]),
                        color=color, alpha=line_alpha, zorder=z + 0.05)
        tp = profile_tick_points(pr, pr["ticks_km"]) if pr.get("ticks_km") else None
        if tp is not None and pr["cut"]:                   # only the ticks on the parts that are left
            keep = np.zeros(tp["s"].size, bool)
            for pc in pr["pieces"]:
                keep |= (tp["s"] >= pc["s0"] - 1e-6) & (tp["s"] <= pc["s1"] + 1e-6)
            tp = {k: v[keep] for k, v in tp.items()}
        if tp is not None and tp["s"].size:
            h = 0.5 * float(sty["tick_length"]) * map_km
            a_lon, a_lat = local_lonlat(tp["x"] + h * tp["uy"], tp["y"] - h * tp["ux"],
                                        pr["lon0"], pr["lat0"])
            b_lon, b_lat = local_lonlat(tp["x"] - h * tp["uy"], tp["y"] + h * tp["ux"],
                                        pr["lon0"], pr["lat0"])
            ax1, ay1 = to_frame(frame, a_lon, a_lat)
            bx1, by1 = to_frame(frame, b_lon, b_lat)
            segs = [[(ax1[k], ay1[k]), (bx1[k], by1[k])] for k in range(ax1.size)]
            ax.add_collection(LineCollection(segs, colors=color, linewidths=float(sty["tick_lw"]),
                                             alpha=line_alpha, zorder=z, path_effects=pe))
            if pr.get("tick_labels", True):
                c_lon, c_lat = local_lonlat(tp["x"], tp["y"], pr["lon0"], pr["lat0"])
                cx, cy = to_frame(frame, c_lon, c_lat)
                tfs = float(sty["tick_fontsize"])
                off = tfs + 1.0                            # [pt] beside the tick
                for k in range(cx.size):
                    ax.annotate(f"{tp['s'][k]:g}", (cx[k], cy[k]),
                                xytext=(-off * tp["uy"][k], off * tp["ux"][k]),
                                textcoords="offset points", ha="center", va="center",
                                fontsize=tfs, color=sty["text_color"], zorder=text_z,
                                path_effects=halo_t, annotation_clip=True)
        first, last = pr["pieces"][0], pr["pieces"][-1]          # the ends of what is shown
        fx0, fy0 = to_frame(frame, first["lon"][:1], first["lat"][:1])
        fx1, fy1 = to_frame(frame, last["lon"][-1:], last["lat"][-1:])
        if sty["end_marker"] is not None:
            ax.plot([fx0[0], fx1[0]], [fy0[0], fy1[0]], ls="", marker=sty["end_marker"],
                    ms=float(sty["end_marker_size"]), mfc=color,
                    mec=sty["halo_color"] if sty["halo"] else color, mew=0.8,
                    zorder=max(6.0, z + 0.1))
        if pr.get("labels"):
            ends = ((fx0[0], fy0[0], -1.0, pr["labels"][0],
                     first["x"][1] - first["x"][0], first["y"][1] - first["y"][0]),
                    (fx1[0], fy1[0], 1.0, pr["labels"][1],
                     last["x"][-1] - last["x"][-2], last["y"][-1] - last["y"][-2]))
            for vx, vy, sgn, txt, dxk, dyk in ends:
                nrm = math.hypot(dxk, dyk) or 1.0
                d = sgn * float(sty["label_offset"])
                ax.annotate(txt, (vx, vy), xytext=(d * dxk / nrm, d * dyk / nrm),
                            textcoords="offset points", ha="center", va="center",
                            fontsize=float(sty["fontsize"]), fontweight=sty["fontweight"],
                            color=sty["text_color"], zorder=text_z, path_effects=halo_t)
        ring = pr.get("ring")
        if sty["mark_sites"] and ring is not None and len(ring[0]):
            rlon, rlat = np.asarray(ring[0], float), np.asarray(ring[1], float)
            if pr["cut"] and pr.get("proj") is not None and pr.get("order") is not None:
                al = np.asarray(pr["proj"]["along"])[np.asarray(pr["order"], int)]
                keep = np.zeros(al.size, bool)                  # only the sites along the parts left
                for pc in pr["pieces"]:
                    keep |= (al >= pc["s0"] - 1e-6) & (al <= pc["s1"] + 1e-6)
                rlon, rlat = rlon[keep], rlat[keep]
            if rlon.size:
                rx, ry = to_frame(frame, rlon, rlat)
                ax.scatter(rx, ry, s=float(sty["ring_size"]) * float(cfg["SITE_SIZE"]), marker="o",
                           facecolors="none", edgecolors=color, linewidths=float(sty["ring_lw"]),
                           zorder=4.5)


def overlay_handles(overlays: Optional[dict], cfg: Dict[str, Any]) -> list:
    """Legend handles of the profiles and lines that have a label (profiles first)."""
    from matplotlib.lines import Line2D
    out = []
    if not overlays:
        return out
    for kind in ("profiles", "lines"):
        for it in overlays.get(kind) or []:
            if it.get("label"):
                sty = overlay_style(it, cfg)
                out.append(Line2D([], [], color=sty["color"], lw=float(sty["lw"]), ls=sty["ls"],
                                  alpha=float(sty["line_alpha"]), marker=sty["marker"] or "",
                                  ms=float(sty["markersize"]), label=str(it["label"])))
    return out


def overlay_outside(items: Sequence[dict], bbox: Sequence[float]) -> List[str]:
    """Names of the profiles / lines whose bounding box does not touch the map `bbox`."""
    out = []
    for it in items:
        if it["kind"] == "profile":
            lon, lat, name = it["lon"], it["lat"], it["name"]
        else:
            lon = np.concatenate([p["lon"] for p in it["parts"]])
            lat = np.concatenate([p["lat"] for p in it["parts"]])
            name = str(it.get("label") or "line")
        if (lon.max() < bbox[0] or lon.min() > bbox[1] or lat.max() < bbox[2]
                or lat.min() > bbox[3]):
            out.append(name)
    return out


def outside_hints(items: Sequence[dict], bbox: Sequence[float]) -> List[str]:
    """
    One message per profile / line that is outside the map `bbox` (see
    overlay_outside): where it is, where the map is, and, if it would be on the
    map with longitude and latitude exchanged, a note on `coords`.
    """
    out = []
    for it in items:
        if not overlay_outside([it], bbox):
            continue
        if it["kind"] == "profile":
            lon, lat, what = it["lon"], it["lat"], f"profile {it['name']!r}"
        else:
            lon = np.concatenate([p["lon"] for p in it["parts"]])
            lat = np.concatenate([p["lat"] for p in it["parts"]])
            what = f"line {str(it.get('label') or 'line')!r}"
        msg = (f"  {what}: lon {lon.min():.4f} .. {lon.max():.4f}, lat {lat.min():.4f} .. "
               f"{lat.max():.4f}; the map: lon {bbox[0]:.4f} .. {bbox[1]:.4f}, lat "
               f"{bbox[2]:.4f} .. {bbox[3]:.4f}.")
        swapped = (lat.max() >= bbox[0] and lat.min() <= bbox[1]
                   and lon.max() >= bbox[2] and lon.min() <= bbox[3])
        if swapped:
            msg += (" With longitude and latitude exchanged it would be on the map: check its "
                    "coords (\"lonlat\" = (lon, lat) pairs, \"latlon\" = (lat, lon) pairs).")
        out.append(msg)
    return out


def draw_panel(ax, *, frame: dict, bbox: Tuple[float, float, float, float], relief: Optional[dict],
               relief_extent: Optional[Tuple[float, float, float, float]],
               lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
               names: Optional[Sequence[str]], cmap, norm, cfg: Dict[str, Any],
               credit: Optional[str] = None, overlays: Optional[dict] = None,
               signed: bool = False):
    """
    One map: relief, then the sites as icons coloured by `values` (NaN = open
    icon, no data), and the profiles and lines of `overlays` (draw_overlays).
    Returns the scatter of the coloured icons (None if there is none). bbox =
    (lon_min, lon_max, lat_min, lat_max) of the map. The largest values are drawn
    on top (`signed`: the largest absolute values).
    """
    from matplotlib import patheffects
    from matplotlib.ticker import MaxNLocator
    halo, bar_halo = [], []                    # MAP_HALO: texts, scale bar line
    if cfg["MAP_HALO"]:
        hw, hc = float(cfg["MAP_HALO_WIDTH"]), cfg["MAP_HALO_COLOR"]
        halo = [patheffects.withStroke(linewidth=hw, foreground=hc)]
        bar_halo = [patheffects.withStroke(linewidth=_SCALE_BAR_LW + hw + 0.5, foreground=hc)]
    x0, x1 = to_frame(frame, [bbox[0], bbox[1]], [bbox[2], bbox[2]])[0]
    y0, y1 = to_frame(frame, [bbox[0], bbox[0]], [bbox[2], bbox[3]])[1]
    if relief is not None and relief_extent is not None:
        ex0, ex1 = to_frame(frame, [relief_extent[0], relief_extent[1]],
                            [relief_extent[2], relief_extent[2]])[0]
        ey0, ey1 = to_frame(frame, [relief_extent[0], relief_extent[0]],
                            [relief_extent[2], relief_extent[3]])[1]
        if relief["kind"] == "gray":
            ax.imshow(np.ma.masked_invalid(relief["img"]), extent=(ex0, ex1, ey0, ey1),
                      origin="upper", cmap=cfg["BASEMAP_CMAP"], vmin=0.0, vmax=1.0,
                      alpha=float(cfg["BASEMAP_ALPHA"]), interpolation="bilinear", zorder=1)
        else:
            ax.imshow(relief["img"], extent=(ex0, ex1, ey0, ey1), origin="upper",
                      alpha=float(cfg["BASEMAP_ALPHA"]), interpolation="bilinear", zorder=1)
    x, y = to_frame(frame, lon, lat)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    sc = None
    kw = dict(marker=cfg["SITE_MARKER"], s=float(cfg["SITE_SIZE"]), zorder=5)
    if (~ok).any():
        ax.scatter(x[~ok], y[~ok], facecolors="none", edgecolors=cfg["SITE_NODATA_EDGECOLOR"],
                   linewidths=float(cfg["SITE_LW"]), **kw)
    if ok.any():
        order = np.argsort(np.abs(v[ok]) if signed else v[ok])     # the largest value on top
        sc = ax.scatter(x[ok][order], y[ok][order], c=v[ok][order], cmap=cmap, norm=norm,
                        edgecolors=cfg["SITE_EDGECOLOR"], linewidths=float(cfg["SITE_LW"]), **kw)
    if cfg["SITE_LABELS"] and names is not None:
        for xi, yi, nm in zip(x, y, names):
            ax.annotate(str(nm), (xi, yi), xytext=(4, 4), textcoords="offset points",
                        fontsize=float(cfg["LABEL_FONTSIZE"]), fontweight=cfg["LABEL_FONTWEIGHT"],
                        zorder=6, path_effects=halo)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(frame["aspect"])
    draw_overlays(ax, frame=frame, overlays=overlays, cfg=cfg, bbox=bbox)
    if cfg["MAP_GRID"]:
        ax.grid(True, color="0.5", lw=0.4, alpha=0.6)
    fs = float(cfg["TICK_FONTSIZE"])
    ax.tick_params(labelsize=fs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(useOffset=False, style="plain")
    fs_axis = font_size(cfg, "AXIS_LABEL_FONTSIZE")
    if frame["display"] == "latlon":
        ax.set_xlabel("Longitude [deg]", fontsize=fs_axis)
        ax.set_ylabel("Latitude [deg]", fontsize=fs_axis)
    else:
        ax.set_xlabel("East [km]", fontsize=fs_axis)
        ax.set_ylabel("North [km]", fontsize=fs_axis)
    if cfg["SCALE_BAR"]:
        width_km = abs(x1 - x0) * (frame["kx"] if frame["display"] == "latlon" else 1.0)
        length_km = (float(cfg["SCALE_BAR_KM"]) if cfg["SCALE_BAR_KM"]
                     else _nice_length(0.2 * width_km))
        length = length_km / (frame["kx"] if frame["display"] == "latlon" else 1.0)
        xa, ya = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([xa, xa + length], [ya, ya], color="black", lw=_SCALE_BAR_LW,
                solid_capstyle="butt", zorder=7, path_effects=bar_halo)
        ax.text(xa + 0.5 * length, ya + 0.012 * (y1 - y0), f"{length_km:g} km", ha="center",
                va="bottom", fontsize=font_size(cfg, "SCALEBAR_FONTSIZE", -1.0), zorder=7,
                path_effects=halo)
    if credit:
        w_in = ax.figure.get_figwidth() * ax.get_position().width
        fs_credit = float(cfg["CREDIT_FONTSIZE"])
        ax.text(0.99, 0.01, wrap_credit(credit, w_in, fs_credit), transform=ax.transAxes,
                ha="right", va="bottom", multialignment="right", fontsize=fs_credit, color="0.15",
                zorder=7, path_effects=halo)
    return sc


def wrap_credit(text: str, width_in: float, fontsize: float) -> str:
    """`text` broken into lines that fit (about) `width_in` inch at `fontsize` pt."""
    import textwrap
    n = max(20, int(0.9 * float(width_in) * 72.0 / (0.55 * float(fontsize))))
    return "\n".join(textwrap.wrap(str(text), width=n)) if len(str(text)) > n else str(text)


def build_pages(sels: Sequence[dict], page_by: str) -> List[Tuple[str, List[dict]]]:
    """[(file tag, [selections on the page]), ...] in the order of `sels`."""
    if page_by == "all":
        return [("all", list(sels))] if sels else []
    pages: Dict[str, List[dict]] = {}
    for s in sels:
        if page_by == "none":
            tag = f"b{s['ib']:02d}_c{s['ic']:02d}"
        elif page_by == "band":
            tag = f"b{s['ib']:02d}_{_slug(s['band'])}"
        elif page_by == "component":
            tag = f"c{s['ic']:02d}_{_slug(s['comp'])}"
        else:
            raise ValueError(f"PAGE_BY = {page_by!r}: one of {list(_PAGE_BY)}")
        pages.setdefault(tag, []).append(s)
    return list(pages.items())


def make_title(spec: Optional[str], *, iteration: int, nrms: float,
               ens: Optional[dict] = None, label: str = "nRMS") -> str:
    """
    Figure title: "auto", None / "" (none) or text with {iter}, {nrms}, {M} and
    {stat}. `ens` (an ensemble) = dict(M, stat, iters): number of members, ENS_STAT,
    iterations of the members; then {iter} is "8" or "5-8", {nrms} the statistic
    of the nRMS of the members, {M} their number and {stat} the statistic in
    words (for a single run: M = 1, stat = ""). `label` = what the maps show
    (placeholder {measure}); {nrms} is always the nRMS.
    """
    if spec is None or spec is False or spec == "":
        return ""
    if ens is None:
        if spec == "auto":
            return f"FEMTIC site {label} (iteration {iteration}, global nRMS = {nrms:.2f})"
        return str(spec).format(iter=iteration, nrms=nrms, M=1, stat="", measure=label)
    it = iteration_text(ens["iters"])
    words = stat_words(ens["stat"])
    if spec == "auto":
        what = "std of global nRMS" if parse_stat(ens["stat"])[0] == "std" else "global nRMS"
        word = "iteration" if "-" not in it else "iterations"
        return (f"FEMTIC site {label}, {words} of {ens['M']} members "
                f"({word} {it}, {what} = {nrms:.2f})")
    return str(spec).format(iter=it, nrms=nrms, M=ens["M"], stat=words, measure=label)


_CM_PER_INCH = 2.54
_CBAR_ASPECT = 20.0                # matplotlib's default: length / width of a colour bar
_CBAR_TOL = 0.01                   # the colour bar has the asked size within 1 %
_SAID: set = set()                 # messages that are said once (per run)


def _say_once(msg: str) -> None:
    if msg not in _SAID:
        _SAID.add(msg)
        _say(msg)


def _layout_pass(fig) -> None:
    """One layout pass of the figure without drawing it (the positions are then final)."""
    eng = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if eng is not None and hasattr(eng, "execute"):
        eng.execute(fig)
    else:                                          # older matplotlib: layout happens at draw time
        fig.canvas.draw()


def _cbar_size_in(cb, vertical: bool) -> Tuple[float, float]:
    """
    (length, width) in inch of the colour bar as it is drawn, after a layout pass.
    The length is that of the whole bar with its extension triangles (`extend`);
    the axes of the bar itself (the coloured rectangle) are shorter by that, which
    matplotlib's locator of the bar applies at draw time, so it is applied here.
    """
    ax = cb.ax
    full = ax.get_position(original=True)          # the slot: the bar with its extensions
    loc = ax.get_axes_locator()
    pos = full
    if loc is not None:
        try:
            renderer = ax.figure.canvas.get_renderer()
        except Exception:                          # a canvas without get_renderer
            renderer = None
        pos = loc(ax, renderer)
    ax.apply_aspect(pos)
    body = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    if vertical:
        ext = full.height / pos.height if pos.height > 0.0 else 1.0
        return body.height * ext * fh, body.width * fw
    ext = full.width / pos.width if pos.width > 0.0 else 1.0
    return body.width * ext * fw, body.height * fh


def add_colorbar(fig, mappable, axes, *, location: str, shrink: float, extend: str,
                 label: str, length_cm: Optional[float] = None,
                 width_cm: Optional[float] = None, label_fontsize: Optional[float] = None,
                 tick_fontsize: Optional[float] = None,
                 ticks: Optional[Tuple[Sequence[float], Sequence[str]]] = None):
    """
    The colour bar of the panels `axes`. Without `length_cm` and `width_cm` it is
    matplotlib's: the length is `shrink` x the length of the panels along the bar
    and the width 1/20 of its length. `length_cm` / `width_cm` [cm] fix the size
    that is drawn (the length with the extension triangles of `extend`, the width of
    the coloured bar): the figure is laid out (without drawing) and the bar made again
    with the length (`shrink`) and the width (`aspect` = length / width) that this
    pass shows are needed, repeated (up to 6 passes) until it is within 1 %. `width_cm` alone keeps the
    length, `length_cm` alone the proportions (1/20). A length that the panels
    cannot hold is reduced to the length of the panels, with a message. The size
    holds for the layout as it is when this is called (set the figure title first).
    `label_fontsize`, `tick_fontsize` [pt]: None = matplotlib's. `ticks` =
    (positions, texts) of a bar of classes; None = numbers.
    """
    kw = dict(location=location, extend=extend)

    def make(**size):
        bar = fig.colorbar(mappable, ax=axes, **size, **kw)
        bar.set_label(label, **({} if label_fontsize is None else {"fontsize": label_fontsize}))
        if ticks is not None:
            bar.set_ticks(list(ticks[0]), labels=list(ticks[1]))
            bar.ax.tick_params(which="both", length=0)
            if location in ("top", "bottom"):          # side by side the names would overlap
                for t in bar.ax.get_xticklabels():
                    t.set(rotation=45, rotation_mode="anchor",
                          ha="right" if location == "bottom" else "left")
        if tick_fontsize is not None:
            bar.ax.tick_params(labelsize=tick_fontsize)
        return bar

    cb = make(shrink=shrink)
    if length_cm is None and width_cm is None:
        return cb
    vertical = location in ("right", "left")
    want_l = None if length_cm is None else float(length_cm) / 2.54
    want_w = None if width_cm is None else float(width_cm) / 2.54
    aspect = _CBAR_ASPECT
    frac = 0.5 if want_w is not None else 0.15      # the width may be more than 15 % of the panels
    clipped = False
    keep_l = None                                   # no CBAR_LENGTH: the length of the plain bar
    for it in range(6):
        _layout_pass(fig)
        have_l, have_w = _cbar_size_in(cb, vertical)
        if want_l is None and keep_l is None:
            keep_l = have_l
        goal_l = have_l if clipped else (keep_l if want_l is None else want_l)
        goal_w = want_w if want_w is not None else goal_l / aspect
        if abs(have_l - goal_l) <= _CBAR_TOL * goal_l and abs(have_w - goal_w) <= _CBAR_TOL * goal_w:
            break
        new_shrink = shrink * goal_l / have_l
        if new_shrink > 1.0:                        # longer than the panels: all of their length
            new_shrink, clipped = 1.0, True
        if want_w is not None:
            aspect = have_l * new_shrink / shrink / want_w
        shrink = new_shrink
        cb.remove()
        cb = make(shrink=shrink, aspect=aspect, fraction=frac)
    else:
        _say(f"the colour bar has {have_l * 2.54:.2f} x {have_w * 2.54:.2f} cm, not the asked "
             f"size (CBAR_LENGTH, CBAR_WIDTH).")
    if clipped:
        msg = (f"CBAR_LENGTH = {length_cm:g} cm is longer than the panels "
               f"({have_l * 2.54:.1f} cm): the colour bar is as long as the panels.")
        _say_once(msg)
    return cb


def plot_page(page: Sequence[dict], *, site_lon: np.ndarray, site_lat: np.ndarray,
              site_names: Sequence[str], frame: dict,
              bbox: Tuple[float, float, float, float], relief: Optional[dict],
              relief_extent: Optional[Tuple[float, float, float, float]], norm, extend: str,
              cfg: Dict[str, Any], title: str, credit: Optional[str],
              cbar_label: str = "nRMS", overlays: Optional[dict] = None,
              measure: str = "nrms",
              cbar_ticks: Optional[Tuple[Sequence[float], Sequence[str]]] = None):
    """
    One figure with the maps of `measure` of the selections of `page`; returns the
    figure. CMAP_NRMS and NRMS_REFERENCE of `cfg` are the colour map and the
    reference of this measure (see measure_scale). The
    second line of a panel title is s["sub"] if the selection has one (ensemble),
    else the global nRMS and n (other measures: s["sub_m"][measure] or their global
    value). The first line is s["head"] if given, else band | components. `overlays` (profiles, lines) are drawn in every
    panel; the legend (LEGEND_LOC, first panel) holds the no-data icon and the
    profiles and lines that have a label. `cbar_ticks` = (positions, texts) of
    the colour bar of a map of classes.
    """
    import matplotlib.pyplot as plt
    n = len(page)
    nc = min(int(cfg["PANEL_COLS"]), n)
    nr = int(math.ceil(n / nc))
    pw, ph = (float(v) / _CM_PER_INCH for v in cfg["PANEL_SIZE"])        # cm -> inch
    with plt.rc_context(font_rc(cfg)):
        fig, axs = plt.subplots(nr, nc, figsize=(pw * nc + 1.0, ph * nr + (0.5 if title else 0.0)),
                                squeeze=False, constrained_layout=True)
        cmap = plt.get_cmap(cfg["CMAP_NRMS"])
        used = []
        any_open = False
        for k, s in enumerate(page):
            ax = axs[k // nc][k % nc]
            used.append(ax)
            glob_v, vals = sel_values(s, measure)
            any_open = any_open or bool(np.isnan(vals).any())
            draw_panel(ax, frame=frame, bbox=bbox, relief=relief, relief_extent=relief_extent,
                       lon=site_lon, lat=site_lat, values=vals, names=site_names, cmap=cmap,
                       norm=norm, cfg=cfg, credit=credit, overlays=overlays,
                       signed=measure in mm.SIGNED_MEASURES)
            if measure == "nrms":
                second = s.get("sub") or f"global nRMS = {s['nrms']:.2f}, n = {s['n']}"
            else:
                second = (s.get("sub_m", {}).get(measure)
                          or f"global {measure_label(measure)} = {measure_text(glob_v, measure)}, "
                             f"n = {s['n']}")
            head = s.get("head") or f"{s['band']} | {s['comp']}"
            ax.set_title(f"{head}\n{second}",
                         fontsize=font_size(cfg, "PANEL_TITLE_FONTSIZE", 1.0))
        for k in range(n, nr * nc):
            axs[k // nc][k % nc].set_visible(False)
        if title:                                  # before the colour bar: its size is fitted to the layout
            fit_title(fig.suptitle(title, fontsize=font_size(cfg, "TITLE_FONTSIZE", 3.0),
                                   fontweight=cfg["TITLE_FONTWEIGHT"]))
        loc = cfg["CBAR_LOCATION"] or "right"
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = add_colorbar(fig, sm, used, location=loc, shrink=float(cfg["CBAR_SHRINK"]),
                          extend=extend, label=cbar_label, length_cm=cfg["CBAR_LENGTH"],
                          width_cm=cfg["CBAR_WIDTH"], label_fontsize=cfg["CBAR_LABEL_FONTSIZE"],
                          tick_fontsize=cfg["CBAR_TICK_FONTSIZE"], ticks=cbar_ticks)
        ref = cfg["NRMS_REFERENCE"]
        if ref is not None and norm.vmin <= ref <= norm.vmax:
            (cb.ax.axhline if loc in ("right", "left") else cb.ax.axvline)(
                ref, color="black", lw=0.8, ls="--")
        handles = []
        if any_open:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], ls="", marker=cfg["SITE_MARKER"], mfc="none",
                                  mec=cfg["SITE_NODATA_EDGECOLOR"], label="no data"))
        handles += overlay_handles(overlays, cfg)
        leg_loc = cfg.get("LEGEND_LOC", "upper left")
        if handles and leg_loc:
            used[0].legend(handles=handles, loc=leg_loc,
                           fontsize=font_size(cfg, "LEGEND_FONTSIZE", -1.0), framealpha=0.8)
    return fig


def fit_title(text) -> None:
    """Break the figure title `text` into lines if it is wider than its figure."""
    import textwrap
    fig = text.get_figure()
    try:
        width = text.get_window_extent(fig.canvas.get_renderer()).width
    except AttributeError:                         # a canvas without get_renderer
        return
    room = 0.97 * fig.get_figwidth() * fig.dpi
    s = text.get_text()
    if width <= room or "\n" in s:
        return
    n = max(10, int(len(s) * room / width))
    text.set_text("\n".join(textwrap.wrap(s, n)))


def save_figure(fig, stem: str, formats: Sequence[str], dpi: int) -> List[str]:
    """Write `fig` to stem.<fmt> for every format; returns the file names."""
    out = []
    for fmt in ([formats] if isinstance(formats, str) else formats):
        p = f"{stem}.{str(fmt).lstrip('.')}"
        fig.savefig(p, dpi=int(dpi))
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _fmt(v: float, nd: int = 4) -> str:
    return "" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_site_tables(stem: str, rows: Sequence[dict], sels: Sequence[dict],
                      used: np.ndarray, measures: Sequence[str] = ()) -> List[str]:
    """
    <stem>_nrms.csv and <stem>_n.csv: one row per site (name, lat, lon, site
    ids), one column per selection ("band | components"): nRMS, and the number
    of values. `rows` = the sites of site.dat (name, lat, lon, ids), `used` =
    their indices (the sites with data). `measures` = further fit measures: one
    table <stem>_<measure>.csv each.
    """
    paths = []
    head = ["site", "lat", "lon", "site_ids"] + [f"{s['band']} | {s['comp']}" for s in sels]
    jobs = [("nrms", lambda s: s["nrms_site"], _fmt),
            ("n", lambda s: s["n_site"], lambda v: str(int(v)))]
    jobs += [(m, lambda s, m=m: s["site_m"][m], _fmt) for m in measures]
    for suffix, values, fmt in jobs:
        p = f"{stem}_{suffix}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for j in used:
                r = rows[j]
                w.writerow([r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                            " ".join(str(i) for i in r.get("ids", []))]
                           + [fmt(values(s)[j]) for s in sels])
        paths.append(p)
    return paths


def write_members_table(stem: str, members: Sequence[dict], sels: Sequence[dict]) -> str:
    """
    <stem>_members.csv: one row per member (in the order found): member, path
    (the results file read), iteration, nrms (of the iteration), used (1 / 0), note
    (why it is not used), then the global nRMS of the member for every selection
    of `sels` ("band | comp"). `members`: dicts with name, path, res (None if not
    read), keep, note, sels.
    """
    p = f"{stem}_members.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["member", "path", "iteration", "nrms", "used", "note"]
                   + [f"{s['band']} | {s['comp']}" for s in sels])
        for m in members:
            res = m.get("res")
            look = {(t["band"], t["comp"]): t["nrms"] for t in m.get("sels", [])}
            w.writerow([m["name"], res["path"] if res else m["path"],
                        res["iter"] if res else "", _fmt(res["rms"]) if res else "",
                        int(bool(m.get("keep"))), m.get("note", "")]
                       + [_fmt(look.get((s["band"], s["comp"]), float("nan"))) for s in sels])
    return p


def write_ens_stats(stem: str, rows: Sequence[dict], sels: Sequence[dict],
                    used: np.ndarray, measure: str = "nrms") -> str:
    """
    <stem>_ens_stats.csv: for every selection and site with data, the number of
    members M and the mean, median, std (ddof = 1), min and max of the member nRMS.
    `sels` = aggregate_members result (key X), `rows`, `used` as in write_site_tables.
    Another fit `measure`: <stem>_<measure>_ens_stats.csv (keys X_m).
    """
    p = f"{stem}_ens_stats.csv" if measure == "nrms" else f"{stem}_{measure}_ens_stats.csv"
    kinds = ("mean", "median", "std", "min", "max")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "components", "site", "lat", "lon", "M"] + list(kinds))
        for s in sels:
            x = s["X"] if measure == "nrms" else s["X_m"][measure]
            cnt = np.sum(np.isfinite(x), axis=0)
            vals = {k: ens_statistic(x, k) for k in kinds}
            for j in used:
                if cnt[j] == 0:
                    continue
                r = rows[j]
                w.writerow([s["band"], s["comp"], r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                            int(cnt[j])] + [_fmt(vals[k][j]) for k in kinds])
    return p


def overlay_summary(profiles: Sequence[dict], lines: Sequence[dict]) -> List[str]:
    """Text lines about the profiles (length, swath, sites in it) and lines, for the summary."""
    out = []
    for p in profiles:
        txt = f"profile {p['name']}: {p['length']:.1f} km, {p['lon'].size} vertices"
        if p["width"] > 0:
            n = int(p["proj"]["inside"].sum()) if "proj" in p else 0
            txt += f", swath {p['width']:g} km wide with {n} sites"
        out.append(txt)
    for ln in lines:
        nm = ln.get("label") or "(no label)"
        out.append(f"line {nm}: {len(ln['parts'])} parts, {ln['n_vertices']} vertices")
    return out


def write_profile_tables(stem: str, profiles: Sequence[dict], rows: Sequence[dict],
                         used: np.ndarray, sels: Sequence[dict],
                         extra: Sequence[Tuple[str, np.ndarray]] = ()) -> List[str]:
    """
    <stem>_profiles.csv: one row per profile (name, vertices, length_km, width_km,
    n_sites in the swath, lon / lat of the start and the end, source).
    <stem>_profile_sites.csv (profiles with a swath): the sites in the swath, by
    profile and increasing distance: profile, site, lat, lon, along_km (from the
    start), offset_km (positive to the right of the direction start -> end) and the
    nRMS of every selection of `sels`, then the columns `extra` = (heading, value per
    site of site.dat). `profiles` = project_profiles result, `rows`
    = the sites of site.dat (name, lat, lon), `used` = their indices behind the
    projection.
    """
    paths = []
    p1 = f"{stem}_profiles.csv"
    with open(p1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "vertices", "length_km", "width_km", "n_sites", "lon_start",
                    "lat_start", "lon_end", "lat_end", "source"])
        for p in profiles:
            w.writerow([p["name"], p["lon"].size, f"{p['length']:.3f}", f"{p['width']:g}",
                        len(p["order"]), f"{p['lon'][0]:.6f}", f"{p['lat'][0]:.6f}",
                        f"{p['lon'][-1]:.6f}", f"{p['lat'][-1]:.6f}", p["source"]])
    paths.append(p1)
    if any(p["width"] > 0 for p in profiles):
        p2 = f"{stem}_profile_sites.csv"
        with open(p2, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["profile", "site", "lat", "lon", "along_km", "offset_km"]
                       + [f"{s['band']} | {s['comp']}" for s in sels]
                       + [name for name, _ in extra])
            for p in profiles:
                for k in p["order"]:
                    j = int(used[k])
                    r = rows[j]
                    w.writerow([p["name"], r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                                f"{p['proj']['along'][k]:.3f}", f"{p['proj']['offset'][k]:.3f}"]
                               + [_fmt(s["nrms_site"][j]) for s in sels]
                               + [_fmt(v[j]) for _, v in extra])
        paths.append(p2)
    return paths


def format_summary(sels: Sequence[dict], *, header: Sequence[str],
                   measure: str = "nrms") -> str:
    """
    Plain-text table: global nRMS and the spread of the site values per selection.
    For an ensemble (selections of aggregate_members) with the number of members M
    and the smallest and largest global nRMS of a member (m_min, m_max). Another
    fit `measure` in place of nRMS.
    """
    ens = bool(sels) and "G" in sels[0]
    label = measure_label(measure)
    cols = ("band", "components", "n", label) + (("M", "m_min", "m_max") if ens else ()) \
        + ("sites", "min", "median", "max", "n_unassigned")
    body = []
    for s in sels:
        glob_v, v = sel_values(s, measure)
        v = v[np.isfinite(v)]
        row = [s["band"], s["comp"], str(s["n"]), _fmt(glob_v, 3)]
        if ens:
            g = s["G"] if measure == "nrms" else s["G_m"][measure]
            g = g[np.isfinite(g)]
            row += [str(s["M"]), _fmt(float(g.min()), 3) if g.size else "",
                    _fmt(float(g.max()), 3) if g.size else ""]
        row += [str(v.size), _fmt(float(v.min()), 3) if v.size else "",
                _fmt(float(np.median(v)), 3) if v.size else "",
                _fmt(float(v.max()), 3) if v.size else "", str(s["n_unassigned"])]
        body.append(tuple(row))
    w = [max(len(c), *(len(r[i]) for r in body)) if body else len(c) for i, c in enumerate(cols)]
    line = "-" * (sum(w) + 2 * (len(w) - 1))
    fmtrow = lambda r: "  ".join(                                    # noqa: E731
        (str(x).ljust(w[i]) if i < 2 else str(x).rjust(w[i])) for i, x in enumerate(r))
    out = list(header) + ["", "=" * len(line), fmtrow(cols), line]
    out += [fmtrow(r) for r in body]
    out += [line,
            "n = number of values (real and imaginary parts), global = all rows of the "
            "selection;",
            f"sites, min, median, max = {label} of the sites; n_unassigned = values of rows "
            "that belong to no site."]
    if ens:
        stat = stat_words(sels[0]["stat"])
        out += [f"{label} and the {label} of the sites = {stat} over the members (M = members with a "
                "value; n, n_unassigned = median of the members);",
                f"m_min, m_max = smallest and largest global {label} of a member."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(cfg: Dict[str, Any]) -> int:
    """
    Run with the settings `cfg` (see settings()). Return code: 0 success, 1 data
    problem (nothing to plot), 2 usage (WORK_DIR missing).
    """
    wd = cfg["WORK_DIR"]
    if not os.path.isdir(wd):
        print(f"{_TAG} WORK_DIR is not a directory: {wd}", file=sys.stderr)
        return 2
    try:
        check_settings(cfg)
    except ValueError as exc:
        print(f"{_TAG} {exc}", file=sys.stderr)
        return 2
    import matplotlib
    if not cfg["SHOW"]:
        matplotlib.use("Agg")
    _say(f"WORK_DIR = {wd}")

    # --- members: the first one that can be read ---
    paths, how = member_paths(cfg)
    if not paths:
        print(f"{_TAG} RUN_PATHS matches no path: {list(cfg['RUN_PATHS'] or [])}", file=sys.stderr)
        return 1
    n_paths = len(paths)
    names = member_names(paths)
    if n_paths > 1:
        _say(f"ensemble of {n_paths} members ({how}); statistic over the members: "
             f"{stat_words(cfg['ENS_STAT'])}.")
    failed: Dict[int, str] = {}              # member index -> why it is not read
    first, k0, err0 = None, 0, None
    for k, pth in enumerate(paths):
        try:
            first = load_member(pth, cfg)
            k0 = k
            break
        except (OSError, ValueError, KeyError) as exc:       # DataTypeError is a ValueError
            if err0 is None:
                err0 = exc
            failed[k] = str(exc)
            if n_paths > 1:
                _say(f"WARNING: member {names[k]} skipped: {exc}")
    if first is None:
        if isinstance(err0, DataTypeError):
            print(f"{_TAG} {err0}", file=sys.stderr)
        else:
            print(f"{_TAG} cannot read the results{' of any member' if n_paths > 1 else ''}: "
                  f"{err0}", file=sys.stderr)
        return 1
    res, ds, n_rows = first["res"], first["ds"], first["n_rows"]
    _say(f"{res['path']}: iteration {res['iter']}, {n_rows} data rows, nRMS = {res['rms']:.3f}.")

    # --- sites ---
    site_dirs = site_search_dirs(cfg["TEMPLATES_DIR"], wd)
    site_path = find_site_dat(cfg["SITE_DAT"], site_dirs)
    if site_path is None:
        print(f"{_TAG} no site.dat found (SITE_DAT = {cfg['SITE_DAT']!r}; looked in "
              f"{', '.join(site_dirs)}).", file=sys.stderr)
        return 1
    try:
        sites = read_site_dat(site_path)
    except (OSError, ValueError) as exc:
        print(f"{_TAG} cannot read site.dat: {exc}", file=sys.stderr)
        return 1
    site_index, st = assign_sites(ds, sites, match=cfg["SITE_MATCH"],
                                  tol=1000.0 * cfg["SITE_MATCH_TOL"],
                                  offset=cfg["SITE_ID_OFFSET"])
    _say(f"{len(sites)} sites in {site_path}; assigned by {st['mode']}: {st['n_hit']} of "
         f"{st['n_ids']} site ids ({st['n_sites']} sites)"
         + (f"; frame {st['frame']}, shift {st['shift'][0]:.1f}, {st['shift'][1]:.1f} m, "
            f"largest distance {st['dmax']:.1f} m" if st["mode"] == "coords" else "") + ".")
    if st.get("n_spread"):
        _say(f"{st['n_spread']} site ids share a position with another id; they were given the "
             "site.dat entries at that position in the order of their numbers.")
    if st["unmatched"]:
        _say(f"WARNING: no site for the site ids {st['unmatched'][:10]}"
             f"{' ...' if len(st['unmatched']) > 10 else ''}; their data count in the global "
             "values only.")
    if st["n_hit"] == 0:
        print(f"{_TAG} no site id could be assigned to a site of {site_path}; "
              "check SITE_MATCH, SITE_MATCH_TOL, SITE_ID_OFFSET and the site.dat.", file=sys.stderr)
        return 1
    if st["n_ids"] and st["n_hit"] < 0.5 * st["n_ids"]:
        _say("WARNING: fewer than half of the site ids assigned -- is this the site.dat of the "
             "data set?")

    # --- nRMS (and the other measures) of every member ---
    measures = parse_measures(cfg["SITE_MEASURES"])
    fits = fit_names(measures)
    dims = dim_names(measures)

    def ids_of_all_rows(member: dict, lm: dict) -> Dict[int, int]:
        """{site id: site index} of all rows of a member (also those DATATYPES leaves out)."""
        if lm.get("all_ids") is None:
            return member["id_to_site"]
        with _print_once(seen):                        # DATATYPES: match the ids of all rows
            return assign_sites(lm["all_ids"], sites, match=cfg["SITE_MATCH"],
                                tol=1000.0 * cfg["SITE_MATCH_TOL"],
                                offset=cfg["SITE_ID_OFFSET"])[1]["id_to_site"]

    def add_impedances(member: dict, lm: dict) -> None:
        """member["z"] = impedance_per_site of a loaded member (DIM_SOURCE "results")."""
        if lm.get("mt") is not None:
            member["z"], member["z_stats"] = impedance_per_site(lm["mt"],
                                                                ids_of_all_rows(member, lm))

    def add_distortion(member: dict, lm: dict) -> None:
        """member["dist"] (None = not available) of a loaded member."""
        member.update(dist=None, dist_source="", dist_why="")
        if _DISTORTION not in measures:
            return
        found, text = read_run_distortion(lm["res"]["path"], int(lm["res"]["iter"]),
                                          file_template=cfg["DISTORTION_FILE"])
        if found is None:
            member["dist_why"] = text
            return
        ids = ids_of_all_rows(member, lm)
        member.update(dist=distortion_per_site(found, ids, len(sites)), dist_source=text)

    seen: set = set()                        # messages of the selections: once
    try:
        with _print_once(seen):
            sels0 = member_selections(first, site_index, len(sites), cfg)
    except ValueError as exc:
        print(f"{_TAG} {exc}", file=sys.stderr)
        return 1
    members = [dict(k=k0, name=names[k0], path=paths[k0], res=res, n_rows=n_rows, sels=sels0,
                    keep=True, note="", used=np.unique(site_index[site_index >= 0]),
                    id_to_site=st["id_to_site"])]
    add_distortion(members[0], first)
    add_impedances(members[0], first)
    for k in range(k0 + 1, n_paths):
        try:
            lm = load_member(paths[k], cfg)
            idx, st_k = assign_sites(lm["ds"], sites, match=cfg["SITE_MATCH"],
                                     tol=1000.0 * cfg["SITE_MATCH_TOL"],
                                     offset=cfg["SITE_ID_OFFSET"])
            if st_k["n_hit"] == 0:
                raise ValueError("no site id could be assigned to a site of the site.dat")
            with _print_once(seen):
                sels_k = member_selections(lm, idx, len(sites), cfg)
        except (OSError, ValueError, KeyError) as exc:
            failed[k] = str(exc)
            _say(f"WARNING: member {names[k]} skipped: {exc}")
            continue
        if (st_k["n_hit"], st_k["n_ids"]) != (st["n_hit"], st["n_ids"]):
            _say(f"WARNING: member {names[k]}: {st_k['n_hit']} of {st_k['n_ids']} site ids "
                 f"assigned ({st['n_hit']} of {st['n_ids']} in {names[k0]}).")
        members.append(dict(k=k, name=names[k], path=paths[k], res=lm["res"],
                            n_rows=lm["n_rows"], sels=sels_k, keep=True, note="",
                            used=np.unique(idx[idx >= 0]), id_to_site=st_k["id_to_site"]))
        add_distortion(members[-1], lm)
        add_impedances(members[-1], lm)
        if n_paths <= 20 or len(members) % 10 == 0 or k == n_paths - 1:
            _say(f"member {names[k]} ({k + 1} of {n_paths}): iteration {lm['res']['iter']}, "
                 f"nRMS = {lm['res']['rms']:.3f}.")

    # --- which members count ---
    mmax = cfg["MEMBER_NRMS_MAX"]
    if mmax is not None:
        for m in members:
            if not m["res"]["rms"] <= float(mmax):
                m["keep"] = False
                m["note"] = f"nRMS above MEMBER_NRMS_MAX = {float(mmax):g}"
        out_ = [m["name"] for m in members if not m["keep"]]
        if out_:
            _say(f"{len(out_)} members with nRMS above MEMBER_NRMS_MAX = {float(mmax):g} left "
                 f"out: {', '.join(out_[:10])}{' ...' if len(out_) > 10 else ''}.")
    kept = [m for m in members if m["keep"]]
    if not kept:
        print(f"{_TAG} no member left: the nRMS of all {len(members)} members is above "
              f"MEMBER_NRMS_MAX = {float(mmax):g}.", file=sys.stderr)
        return 1
    ens = len(kept) > 1
    if n_paths > 1:
        _say(f"{len(kept)} of {n_paths} members used"
             + ("" if ens else " (plotted as a single run)") + ".")
    used = np.unique(np.concatenate([m["used"] for m in kept]))
    ids_of: Dict[int, set] = {}              # site.dat entry -> its site ids in the members
    for m in kept:
        for sid_, j in m["id_to_site"].items():
            ids_of.setdefault(j, set()).add(sid_)
    site_rows = [dict(name=s["name"], lat=s["lat"], lon=s["lon"], ids=sorted(ids_of.get(j, ())))
                 for j, s in enumerate(sites)]
    kind = parse_stat(cfg["ENS_STAT"])[0]
    if ens:
        sels = aggregate_members(kept, cfg["ENS_STAT"], len(sites))
    else:
        res = kept[0]["res"]
        sels = kept[0]["sels"]
    plot_sels = [s for s in sels if np.isfinite(s["nrms_site"][used]).any()]
    if not plot_sels:
        print(f"{_TAG} no selection (band, components) has data at a site.", file=sys.stderr)
        return 1
    for s in sels:
        if s not in plot_sels:
            _say(f"selection {s['band']} | {s['comp']}: no data at a site, no map.")

    # --- distortion strength ---
    dsel = None
    if _DISTORTION in measures:
        have = [m for m in kept if m["dist"] is not None]
        why = [m for m in kept if m["dist"] is None]
        dsel = distortion_selection(kept, cfg["ENS_STAT"], used, ens=ens)
        if not have:
            _say(f"WARNING: distortion strength not available: {why[0]['dist_why']}"
                 + (f" (member {why[0]['name']}; none of {len(kept)} members has it)"
                    if ens else "") + ". No distortion map.")
        else:
            _say("distortion from " + "; ".join(sorted({m["dist_source"] for m in have}))
                 + (f": {len(have)} of {len(kept)} members" if ens else "") + ".")
            if why:
                _say(f"WARNING: {len(why)} members without distortion, e.g. {why[0]['name']}: "
                     f"{why[0]['dist_why']}.")
            lost = sorted({i for m in have for i in m["dist"]["unmatched"]})
            if lost:
                _say(f"WARNING: distortion: no site for the site ids {lost[:10]}"
                     f"{' ...' if len(lost) > 10 else ''}.")
            if any(m["dist"]["n_multi"] for m in have):
                _say("distortion: sites with several site ids get the mean of their strengths.")
            if dsel is None:
                _say("WARNING: distortion: no value at a site with data. No distortion map.")

    # --- dimensionality ---
    dim_sels: List[dict] = []
    dim_banded: dict = {}
    dim_from = ""
    if dims:
        data = None
        if cfg["DIM_SOURCE"] == "results":
            ref = kept[0]
            data = ref.get("z", {})
            dim_from = "observed impedances of " + (f"member {ref['name']}" if ens
                                                    else ref["res"]["path"])
            zs = ref.get("z_stats", {})
            if zs.get("n_repeated"):
                _say(f"WARNING: dimensionality: {zs['n_repeated']} repeated impedance rows "
                     "(several ids of one site?); the first one counts.")
            if zs.get("n_incomplete"):
                _say(f"dimensionality: {zs['n_incomplete']} site frequencies without four "
                     "usable impedance components left out.")
            other = [m["name"] for m in kept[1:] if not same_impedances(data, m.get("z", {}))]
            if other:
                _say(f"WARNING: dimensionality: the observed impedances of {len(other)} members "
                     f"differ from those of {ref['name']} (perturbed data?); those of "
                     f"{ref['name']} are used. DIM_SOURCE can name the original data.")
        else:
            src = cfg["DIM_SOURCE"]
            dim_from = "data of " + ", ".join(
                map(str, [src] if isinstance(src, (str, os.PathLike)) else src))
        try:
            found, notes = classify_sites(cfg, sites, data)
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            print(f"{_TAG} dimensionality: {exc}", file=sys.stderr)
            return 1
        for msg in notes:
            _say(msg)
        try:
            with _print_once(seen):
                dim_sels, dim_banded = dim_selections(found, used, len(sites), dims, cfg)
        except ValueError as exc:
            print(f"{_TAG} dimensionality: {exc}", file=sys.stderr)
            return 1
        if dim_sels:
            _say(f"dimensionality: {len(dim_banded)} of {used.size} sites with impedances, "
                 f"{dim_from}.")
        else:
            _say("WARNING: dimensionality: no impedances at a site with data (data type MT "
                 "in the results file, or DIM_SOURCE). No dimensionality map.")

    # --- profiles and lines ---
    lon_s = np.array([site_rows[j]["lon"] for j in used])
    lat_s = np.array([site_rows[j]["lat"] for j in used])
    profiles: List[dict] = []
    lines: List[dict] = []
    if cfg["PROFILES"] or cfg["LINES"]:
        try:
            profiles, lines = prepare_overlays(cfg, make_geo(sites, st))
        except OverlayError as exc:
            print(f"{_TAG} {exc}", file=sys.stderr)
            return 1
        profiles = project_profiles(profiles, lon_s, lat_s)
    overlays = dict(profiles=profiles, lines=lines) if (profiles or lines) else None

    # --- output: tables ---
    out_dir = cfg["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, cfg["OUTPUT_BASENAME"])
    files = write_site_tables(stem, site_rows, sels, used, fits)
    if dsel is not None:
        files.append(write_distortion_table(stem, site_rows, dsel, used, kept, ens=ens))
    if profiles:
        extra = [(f"{m} | {s['band']} | {s['comp']}", s["site_m"][m]) for m in fits for s in sels]
        if dsel is not None:
            extra.append((_DISTORTION, dsel["site_m"][_DISTORTION]))
        extra += [(f"{m} | {s['band']}", s["codes"].get(m, s["site_m"][m]))
                  for m in dims for s in dim_sels]
        files += write_profile_tables(stem, profiles, site_rows, used, sels, extra)
    if ens:
        rms_k = np.array([m["res"]["rms"] for m in kept], float)
        its = [int(m["res"]["iter"]) for m in kept]
        n_drop = len(members) - len(kept)
        header = [f"femtic_plot_sitemap.py: nRMS of the sites, ensemble of {len(kept)} members "
                  f"({how})",
                  f"statistic over the members: {stat_words(cfg['ENS_STAT'])}; {n_paths} members "
                  f"found, {len(kept)} used, {n_drop} above MEMBER_NRMS_MAX, {len(failed)} "
                  "not readable",
                  f"iterations {iteration_text(its)}; nRMS of the iteration: min "
                  f"{np.nanmin(rms_k):.4f}, median {np.nanmedian(rms_k):.4f}, max "
                  f"{np.nanmax(rms_k):.4f}",
                  f"{len(used)} sites with data of {len(sites)} in {site_path}; "
                  f"site ids assigned by {st['mode']}"]
    else:
        header = [f"femtic_plot_sitemap.py: nRMS of the sites ({res['path']})",
                  f"iteration {res['iter']}, {kept[0]['n_rows']} data rows, "
                  f"nRMS of the iteration = {res['rms']:.4f}",
                  f"{len(used)} sites with data of {len(sites)} in {site_path}; "
                  f"site ids assigned by {st['mode']}"]
    if n_paths > 1:
        every = sorted([dict(m) for m in members] +
                       [dict(k=k, name=names[k], path=paths[k], res=None, sels=[], keep=False,
                             note=msg) for k, msg in failed.items()], key=lambda m: m["k"])
        files.append(write_members_table(stem, every, sels))
    if ens:
        files.append(write_ens_stats(stem, site_rows, sels, used))
        files += [write_ens_stats(stem, site_rows, sels, used, m) for m in fits]
    header += overlay_summary(profiles, lines)
    txt = format_summary(sels, header=header)
    for m in fits:
        txt += "\n" + format_summary(sels, header=[f"{measure_label(m)} of the sites"], measure=m)
    if dsel is not None:
        txt += "\n" + format_distortion(dsel, used, ens=ens, stat=cfg["ENS_STAT"], members=kept)
    if dim_sels:
        import dimens
        about = [f"dimensionality of the sites: {dim_from}",
                 f"WAL thresholds {cfg['DIM_THRESHOLD']:g} (I3-I7), {cfg['DIM_Q_THRESHOLD']:g} "
                 f"(Q); phase tensor thresholds |beta| {cfg['DIM_PT_BETA_THRESHOLD']:g} deg, "
                 f"ellipticity {cfg['DIM_PT_ELLIPTICITY_THRESHOLD']:g}",
                 f"errors: {cfg['DIM_ERROR_PERCENT']:g} percent of |Z|"
                 if cfg["DIM_ERROR_PERCENT"] is not None else "errors: those of the data",
                 "angles in the .csv and .npz files refer to the x axis of the data"]
        if cfg["DIM_OPTIONS"]:
            about.append(f"DIM_OPTIONS: {cfg['DIM_OPTIONS']}")
        files += [str(q) for q in dimens.write_summary(
            dim_banded, out_dir, name=f"{cfg['OUTPUT_BASENAME']}_dimensionality",
            header=about)]
        txt += "\n" + format_dimensionality(dim_sels, dim_banded, header=about)
    with open(stem + "_summary.txt", "w") as fh:
        fh.write(txt)
    files.append(stem + "_summary.txt")
    print(txt)

    # --- basemap ---
    ext_lon, ext_lat = lon_s, lat_s
    if cfg["MAP_INCLUDE_PROFILES"] and profiles:
        ext_lon = np.concatenate([lon_s] + [p["lon"] for p in profiles])
        ext_lat = np.concatenate([lat_s] + [p["lat"] for p in profiles])
    try:
        bbox = map_extent(ext_lon, ext_lat, mode=cfg["MAP_EXTENT_MODE"], box=cfg["MAP_BOX"],
                          center=cfg["MAP_CENTER"], size_km=cfg["MAP_SIZE_KM"],
                          margin_frac=cfg["MAP_MARGIN_FRAC"],
                          margin_min_km=cfg["MAP_MARGIN_MIN_KM"])
    except ValueError as exc:
        print(f"{_TAG} {exc}", file=sys.stderr)
        return 1
    inside = ((lon_s >= bbox[0]) & (lon_s <= bbox[1]) & (lat_s >= bbox[2]) & (lat_s <= bbox[3]))
    if not inside.all():
        _say(f"WARNING: {int(np.count_nonzero(~inside))} of {inside.size} sites lie outside the "
             "map (not shown; see the tables).")
    out_of_map = overlay_outside(list(profiles) + list(lines), bbox)
    if out_of_map:
        _say(f"WARNING: outside the map (not shown): {', '.join(out_of_map)}.")
        for msg in outside_hints(list(profiles) + list(lines), bbox):
            _say(msg)
    try:
        for msg in cut_report(profiles, lines, bbox, cfg["OVERLAY_CUT_MARGIN_KM"]):
            _say(msg)
    except OverlayError as exc:
        _say(f"WARNING: {exc}: profiles and lines are not cut.")
    frame = make_frame(cfg["DISPLAY_COORDS"], 0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3]))
    relief = None
    relief_extent = None
    credit = None
    if cfg["DEM_SOURCE"] != "none":
        try:
            dem = load_dem(cfg["DEM_SOURCE"], bbox, file=cfg["DEM_FILE"],
                           url_template=cfg["DEM_URL_TEMPLATE"], cache_dir=cfg["DEM_CACHE_DIR"],
                           download=cfg["DEM_DOWNLOAD"], timeout=cfg["DEM_TIMEOUT"],
                           max_tiles=cfg["DEM_MAX_TILES"], fill=cfg["DEM_FILL_MISSING"],
                           strict=cfg["DEM_STRICT"], max_pixels=cfg["DEM_MAX_PIXELS"],
                           user_agent=_USER_AGENT, say=_say)
            if dem is not None:
                relief = shade_dem(dem, style=cfg["DEM_STYLE"], azdeg=cfg["HILLSHADE_AZIMUTH"],
                                   altdeg=cfg["HILLSHADE_ALTITUDE"],
                                   vert_exag=cfg["HILLSHADE_VERT_EXAG"], cmap=cfg["DEM_CMAP"])
                relief_extent = dem_extent(dem)
                _say(f"DEM: {dem['z'].shape[1]} x {dem['z'].shape[0]} pixels, "
                     f"{np.nanmin(dem['z']):.0f} .. {np.nanmax(dem['z']):.0f} m.")
            else:
                _say("WARNING: no DEM -- the maps have no relief.")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            if cfg["DEM_STRICT"]:
                print(f"{_TAG} DEM: {exc}", file=sys.stderr)
                return 1
            _say(f"WARNING: DEM: {exc} -- the maps have no relief.")
    if relief is not None:
        credit = dem_credit(cfg["DEM_SOURCE"], cfg["DEM_URL_TEMPLATE"], cfg["DEM_CREDIT"])

    # --- maps ---
    spread = ens and kind == "std"           # spread of the members: sequential scale from 0
    if ens:
        n_k = len(kept)
        run_title = dict(iteration=its[0], nrms=float(ens_statistic(rms_k, cfg["ENS_STAT"])))
    else:
        run_title = dict(iteration=int(res["iter"]), nrms=float(res["rms"]))
    import matplotlib.pyplot as plt
    for name in missing_fonts(cfg["FONT_FAMILY"]):
        _say(f"WARNING: FONT_FAMILY: matplotlib finds no font {name!r}; another font is used.")
    if max(float(v) for v in cfg["PANEL_SIZE"]) < _SMALL_PANEL_CM:
        _say(f"NOTE: PANEL_SIZE = {tuple(cfg['PANEL_SIZE'])} is in cm (inch until 2026-11-26): "
             "small panels.")
    figs, n_figs = [], 0                     # figs: kept open for SHOW only

    def emit(fig, tag: str) -> None:
        """Save a figure and close it, or keep it for SHOW."""
        nonlocal n_figs
        files.extend(save_figure(fig, f"{stem}_{tag}", cfg["OUTPUT_FORMATS"], cfg["DPI"]))
        n_figs += 1
        if cfg["SHOW"]:
            figs.append(fig)
        else:
            plt.close(fig)                   # many measures x pages: do not keep them all

    with plt.rc_context(font_rc(cfg)):             # also for the text made when saving / showing
        for measure in measures:
            if measure in _DIM_MEASURES:
                if not dim_sels:
                    continue
                if not any(np.isfinite(s["site_m"][measure][used]).any() for s in dim_sels):
                    _say(f"{measure_label(measure)}: no value at a site, no map.")
                    continue
                style = dim_style(measure, dim_sels, used, cfg)
                pcfg = dict(cfg, CMAP_NRMS=style["cmap"], NRMS_REFERENCE=style["reference"])
                label = measure_label(measure)
                if cfg["TITLE"] == "auto":
                    title = (f"Site {label}, observed impedances"
                             if cfg["DIM_SOURCE"] == "results" else f"Site {label}")
                else:
                    m_ens = dict(M=len(kept), stat=cfg["ENS_STAT"], iters=its) if ens else None
                    title = make_title(cfg["TITLE"], ens=m_ens, label=label, **run_title)
                for tag, page in build_pages(dim_sels, cfg["DIM_PAGE"]):
                    page_sels = [dict(s, site_m={q: v[used] for q, v in s["site_m"].items()})
                                 for s in page]
                    fig = plot_page(page_sels, site_lon=lon_s, site_lat=lat_s,
                                    site_names=[site_rows[j]["name"] for j in used],
                                    frame=frame, bbox=bbox, relief=relief,
                                    relief_extent=relief_extent, norm=style["norm"],
                                    extend=style["extend"], cfg=pcfg, title=title,
                                    credit=credit, cbar_label=label, overlays=overlays,
                                    measure=measure, cbar_ticks=style["ticks"])
                    emit(fig, f"{measure}_{tag.rstrip('_')}")
                continue
            if measure == _DISTORTION:
                if dsel is None:
                    continue
                m_sels, pages = [dsel], [(_DISTORTION, [dsel])]
            else:
                m_sels = [s for s in plot_sels
                          if np.isfinite(sel_values(s, measure)[1][used]).any()]
                prefix = "" if measure == "nrms" else f"{measure}_"
                pages = [(prefix + tag, page)
                         for tag, page in build_pages(m_sels, cfg["PAGE_BY"])]
            allv = np.concatenate([sel_values(s, measure)[1][used] for s in m_sels] or [[]])
            if not np.isfinite(allv).any():
                _say(f"{measure_label(measure)}: no value at a site, no map.")
                continue
            scale = measure_scale(measure, cfg, spread=spread)
            pcfg = dict(cfg, CMAP_NRMS=scale["cmap"], NRMS_REFERENCE=scale["reference"])
            norm, extend = make_norm(
                allv, vmin=scale["vmin"], vmax=scale["vmax"], reference=scale["reference"],
                floor=None if measure in mm.SIGNED_MEASURES else 0.0,
                name="NRMS_REFERENCE" if measure == "nrms" else f"the reference of {measure}")
            label = measure_label(measure)
            short = "|C - I|" if measure == _DISTORTION else label
            m_ens = None
            if ens:
                m_k = dsel["M"] if measure == _DISTORTION else n_k
                m_ens = dict(M=m_k, stat=cfg["ENS_STAT"], iters=its)
                cbar_label = (f"std of {short} ({m_k} members)" if spread
                              else f"{short} ({stat_words(cfg['ENS_STAT'])} of {m_k} members)")
            else:
                cbar_label = _DISTORTION_LABEL if measure == _DISTORTION else label
            title = make_title(cfg["TITLE"], ens=m_ens, **run_title,
                               label="distortion strength" if measure == _DISTORTION else label)
            for tag, page in pages:
                page_sels = [dict(s, nrms_site=s["nrms_site"][used],
                                  site_m={q: v[used] for q, v in s["site_m"].items()})
                             for s in page]
                fig = plot_page(page_sels, site_lon=lon_s, site_lat=lat_s,
                                site_names=[site_rows[j]["name"] for j in used], frame=frame,
                                bbox=bbox, relief=relief, relief_extent=relief_extent, norm=norm,
                                extend=extend, cfg=pcfg, title=title, credit=credit,
                                cbar_label=cbar_label, overlays=overlays, measure=measure)
                emit(fig, tag)
        _say(f"{len(files)} files in {out_dir}:")
        for p in files:
            print("   " + os.path.basename(p))
        if cfg["SHOW"]:
            plt.show()
    for fig in figs:
        plt.close(fig)
    if not n_figs:
        print(f"{_TAG} no map: nothing to show for SITE_MEASURES = {list(measures)}.",
              file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str]) -> int:
    """
    Command line: -w, -i, --dem, --runs, --stat, --measures, --dim-source, --profile,
    --line, --no-overlays.
    """
    ap = argparse.ArgumentParser(
        prog="femtic_plot_sitemap.py",
        description="Site map of a FEMTIC run or ensemble: nRMS (or other fit measures per "
                    "site, band and component, the distortion strength, or the dimensionality "
                    "of the impedances per site and band) on a shaded relief "
                    "(Copernicus DEM by default). All paths are taken relative to WORK_DIR.")
    ap.add_argument("-w", "--work-dir", default=None, metavar="DIR",
                    help="base of all relative paths (replaces WORK_DIR)")
    ap.add_argument("-i", "--iteration", default=None, metavar="N",
                    help="\"best\", \"last\" or a number (replaces ITERATION)")
    ap.add_argument("--dem", default=None, choices=DEM_SOURCES,
                    help="DEM source (replaces DEM_SOURCE)")
    ap.add_argument("--runs", nargs="+", default=None, metavar="PATH",
                    help="members of the ensemble: run directories, results files or globs "
                         "(replaces RUN_PATHS)")
    ap.add_argument("--stat", default=None, metavar="STAT",
                    help="statistic over the members: mean, median, min, max, std or pNN "
                         "(replaces ENS_STAT)")
    ap.add_argument("--measures", nargs="+", default=None, metavar="NAME",
                    help="what the maps show: " + ", ".join(mm.FIT_MEASURE_NAMES)
                         + ", distortion, " + ", ".join(_DIM_MEASURES)
                         + " (replaces SITE_MEASURES)")
    ap.add_argument("--dim-source", nargs="+", default=None, metavar="PATH",
                    help="data of the dimensionality measures: \"results\", EDI directories "
                         "or files, .npz collections (replaces DIM_SOURCE)")
    ap.add_argument("--profile", nargs=5, action="append", default=None,
                    metavar=("NAME", "LON1", "LAT1", "LON2", "LAT2"),
                    help="a profile line from (LON1, LAT1) to (LON2, LAT2), in degrees; can be "
                         "repeated (added to PROFILES)")
    ap.add_argument("--line", action="append", default=None, metavar="FILE",
                    help="a polyline file (text / GMT with lon lat columns, GeoJSON, KML, KMZ); "
                         "can be repeated (added to LINES)")
    ap.add_argument("--no-overlays", action="store_true",
                    help="no profiles and lines (empties PROFILES and LINES)")
    args = ap.parse_args(list(argv))
    over: Dict[str, Any] = {}
    if args.no_overlays:
        if args.profile or args.line:
            ap.error("--no-overlays cannot be combined with --profile or --line")
        over["PROFILES"] = []
        over["LINES"] = []
    if args.profile:
        new = []
        for name, *num in args.profile:
            try:
                lon1, lat1, lon2, lat2 = (float(v) for v in num)
            except ValueError:
                ap.error("--profile: LON1 LAT1 LON2 LAT2 must be numbers")
            new.append(dict(name=name, start=(lon1, lat1), end=(lon2, lat2)))
        over["PROFILES"] = list(PROFILES) + new
    if args.line:
        over["LINES"] = list(LINES) + [dict(file=f, label=os.path.basename(f)) for f in args.line]
    if args.iteration is not None:
        if args.iteration in ("best", "last"):
            over["ITERATION"] = args.iteration
        else:
            try:
                over["ITERATION"] = int(args.iteration)
            except ValueError:
                ap.error("--iteration: \"best\", \"last\" or an integer")
    if args.dem is not None:
        over["DEM_SOURCE"] = args.dem
    if args.runs is not None:
        over["RUN_PATHS"] = args.runs
    if args.dim_source is not None:
        src = args.dim_source
        over["DIM_SOURCE"] = src[0] if len(src) == 1 else src
    if args.stat is not None:
        try:
            parse_stat(args.stat)
        except ValueError as exc:
            ap.error(str(exc))
        over["ENS_STAT"] = args.stat
    if args.measures is not None:
        try:
            over["SITE_MEASURES"] = list(parse_measures(args.measures))
        except ValueError as exc:
            ap.error(str(exc))
    return run(settings(args.work_dir, **over))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
