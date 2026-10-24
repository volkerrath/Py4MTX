# femtic_add_dummy_vtf

Convert a simple FEMTIC `observe.dat` into the layout with dummy VTF
stations, so that the electric field and the magnetic field of every MT
station are interpolated in different elements (`OWNER_ELEMENT -1`).

Provenance: written by Claude (Anthropic), 2026-10-20, updated 2026-10-22.
AI-generated code; review and test before production use.

---

## 1. Why

FEMTIC interpolates E and H of a station in one "owner element". Two cases
are better served by using different elements for E and H:

| Case     | E from                  | H from                  | Manual slides |
|----------|-------------------------|-------------------------|---------------|
| seafloor | upward element (1)      | downward element (0)    | 55-56         |
| land     | downward element (0)    | upward element (1)      | 57-58         |

Land: H varies less in the air, and using the element below the surface for
H can overestimate the impedance amplitude at high frequencies unless that
element is very small. Seafloor: the manual cites Usui et al. (2018) on
slide 55 (see section 8).

The manual's recipe, which this script automates:

1. set `OWNER_ELEMENT -1` in `control.dat`;
2. add a dummy VTF (or HTF) station at each MT location. If a VTF or HTF
   station is already there, no new dummy is needed (slides 55, 57);
3. give that station the owner type used for H, and the MT station the owner
   type used for E;
4. set the MT station's "Station ID for horizontal magnetic field" to the
   ID of that station.

The script decides this per MT site (section 5), so files with VTF stations
at only some of the sites are handled in one run.

## 2. Files

| File                              | Purpose                                   |
|-----------------------------------|-------------------------------------------|
| `femtic_add_dummy_vtf.py`         | the converter                             |
| `femtic_add_dummy_vtf_test.py`    | 49 tests                                  |
| `femtic_add_dummy_vtf_readme.md`  | this file                                 |
| `observe_dummyVTF.dat`            | result for the supplied 42-station file   |

Python 3.7+, standard library only.

## 3. Quick start

```
python femtic_add_dummy_vtf.py                         # uses the USER SECTION
python femtic_add_dummy_vtf.py in.dat out.dat          # override file names
python femtic_add_dummy_vtf.py in.dat out.dat seafloor # override SITE_TYPE
```

Then, in `control.dat`:

```
OWNER_ELEMENT
-1
```

Without this change FEMTIC reads the new owner-type column as the X
coordinate and the run fails or gives garbage.

## 4. USER SECTION (top of the script)

| Parameter                | Default                  | Meaning                                                                 |
|--------------------------|--------------------------|-------------------------------------------------------------------------|
| `INPUT_FILE`             | `"observe.dat"`          | input file                                                              |
| `OUTPUT_FILE`            | `"observe_dummyVTF.dat"` | output file; must differ from the input                                 |
| `SITE_TYPE`              | `"land"`                 | `"land"` or `"seafloor"` (table in section 1)                           |
| `DUMMY_ID_OFFSET`        | `1000`                   | dummy ID = MT ID + offset; collisions with existing IDs are an error    |
| `REUSE_EXISTING_STATIONS`| `True`                   | use a VTF/HTF station already at an MT location instead of adding a dummy |
| `RENUMBER_EXISTING_VTF`  | `True`                   | give such a station the ID MT ID + offset, as a dummy would get (location and data unchanged) |
| `REUSE_BLOCKS`           | `("VTF", "HTF")`         | blocks searched for such stations, in order of preference               |
| `MATCH_TOL_KM`           | `1.0e-3`                 | stations are co-located if \|dX\| and \|dY\| are both within this (km)  |
| `IMPROVE_ALL_VTF_HTF`    | `True`                   | all existing VTF/HTF stations get the H-field owner type (see section 5)|
| `INPUT_HAS_OWNER_COLUMN` | `False`                  | `True` if the input already has the owner-type column                   |
| `HAS_ELEC_COLUMN`        | `False`                  | `True` if stations carry the E-field-type column (`ELEC_FIELD -1`)      |
| `DEFAULT_OWNER_OTHER`    | `0`                      | owner type for PT stations, and for VTF/HTF stations not covered by `IMPROVE_ALL_VTF_HTF` (0 = old behaviour) |
| `VERIFY_OUTPUT`          | `True`                   | re-read the output and compare it with the input                        |

## 5. What changes in the file

Input (simple layout, station line: `ID  H-ID  X  Y`):

```
MT  42
1  1  -2.671  19.713
31
...
END
```

Output, `land` mode (station line: `ID  H-ID  owner  X  Y`):

```
MT  42
1  1001  0  -2.671  19.713
31
...
VTF  42
1001  1001  1  -2.671  19.713
0
1002  1002  1  3.167  14.288
0
...
END
```

Strategy, applied to every station of the `MT` and `APP_RES_AND_PHS` blocks
on its own:

| Situation at the MT site                          | What happens                                                     |
|---------------------------------------------------|------------------------------------------------------------------|
| no VTF/HTF station                                | dummy VTF created: ID = MT ID + offset, H-ID = own ID, 0 freq.   |
| real VTF (or HTF) station, within `MATCH_TOL_KM`  | no dummy; the real station gets ID = MT ID + offset              |
| both (some sites with, some without)              | both of the above, site by site                                  |

In every case the MT station's H-ID is then set to that station's new ID, and
owner types follow the table of section 1 (MT/APP: E type, VTF/HTF: H type).

Details for real stations:

- A VTF station is preferred over an HTF station at the same site. Its
  location and data are never changed; only the ID (renumbering) and the
  owner type can change. Several MT stations at one site share one station,
  which is named after the first of them.
- If it already has the ID MT ID + offset, nothing is renamed. A real VTF
  with the same ID as its MT station (a common case) is renamed too, so the
  MT station's H-ID no longer points at itself.
- H-IDs of other stations (MT, PT, ...) that pointed to the old ID are
  updated. If the old ID is also used by another station, they cannot be told
  apart; they are left unchanged and a warning is printed.
- The new ID must be free. A clash with any ID in the file is an error;
  change `DUMMY_ID_OFFSET` then.
- `RENUMBER_EXISTING_VTF = False` keeps the real station's ID and points the
  MT station to it (with a warning if that ID equals the MT station's ID).
- **Existing VTF/HTF stations use the improved H field as well.** With
  `IMPROVE_ALL_VTF_HTF = True` every existing VTF and HTF station, at an MT
  site or not, gets the H type (land 1, seafloor 0). With `False` only the
  stations used for MT sites do; the others get `DEFAULT_OWNER_OTHER`. If the
  input already has an owner column, its values are kept and one warning lists
  the stations that differ from the H type.
- `PT` stations keep `DEFAULT_OWNER_OTHER`.
- A VTF/HTF station is only used if its H-ID equals its own ID. Otherwise a
  warning is printed and a dummy is created instead.
- New dummies go into one `VTF` block. If the file already has a `VTF`
  block, they are appended after the existing stations; otherwise a new
  block is inserted before the first `HTF`/`PT` block, or before `END`. If
  every MT site already has a VTF/HTF station, no block is added.

Set `REUSE_EXISTING_STATIONS = False` to always add dummies, even where a
VTF/HTF station exists.

Example, land mode, real VTF stations 900 and 901 at the sites of MT 1 and
MT 2 (none at MT 3):

```
MT  3                                  VTF  3
1  1001  0  -2.671  19.713             1001  1001  1  -2.671  19.713   <- was 900
...                                    ...                              (same data)
2  1002  0   3.167  14.288             1002  1002  1   3.167  14.288   <- was 901
3  1003  0  ...                        1003  1003  1  ...              <- new dummy, 0 freq.
```

Layout of the other blocks in the output (column order follows the MT table
on manual slide 36):

```
MT, APP_RES_AND_PHS, PT : ID  H-ID  owner  [E-type]  X  Y
VTF                     : ID  H-ID  owner  X  Y
HTF                     : ID  ID    owner  X  Y
```

All numbers (frequencies, data, standard deviations, X, Y) are copied as
text. Nothing is rounded or reformatted.

## 6. Checks built in

- Parser errors (with token position) on: missing `END`, unknown or
  duplicated block keyword, truncated data, non-integer IDs.
- Error if a new ID (dummy or renumbered station) is already in use, if the
  file has no MT / `APP_RES_AND_PHS` stations, or if nothing needs doing.
- Warning (and a dummy instead) if a co-located VTF/HTF station has an H-ID
  different from its own ID.
- Refuses to overwrite the input file.
- `VERIFY_OUTPUT`: re-parses the written file and checks that location and
  all data lines of every original station are unchanged, that IDs changed
  only where intended, that owner types and H-IDs are as intended, that every
  H-ID used by an MT station belongs to a VTF/HTF station at its location,
  that new IDs are unique, and that each dummy matches its MT station.

Result for the supplied file: 42 MT stations, no VTF/HTF stations in the
input, so 42 dummies (IDs 1001-1042); verification passed. The `VTF` block
starts at line 1182. The summary printed at the end also reports how many
sites use an existing station, which IDs were renumbered, how many
references were updated and how many existing stations got the H owner type.

Tests: `python -I femtic_add_dummy_vtf_test.py` runs 49 cases: land and
seafloor mode, merge into an existing VTF block, block insertion order,
E-type column pass-through, ID collision, bad offset, truncated input,
missing END, unknown and duplicate keywords, foreign H-ID, in == out, bad
`SITE_TYPE`, and for existing stations: partly existing VTF, renumbering,
HTF fallback, VTF preferred, tolerance, shared stations, ID already in the
dummy numbering, VTF with the same ID as its MT station, updated and
ambiguous references, ID clash, unchanged location and data, owner types for
land and seafloor, the three switches `IMPROVE_ALL_VTF_HTF`,
`REUSE_EXISTING_STATIONS`, `RENUMBER_EXISTING_VTF`, owner column in the
input, all MT sites already covered, and a mixed file with all situations.

## 7. Limits and things to check

- Giving existing VTF/HTF stations the H owner type changes which element
  their own response is computed in (for land: upward instead of downward).
  I applied it to all of them (`IMPROVE_ALL_VTF_HTF = True`) because the
  manual's argument for the H field (slides 55, 57) holds for any magnetic
  field station. The manual only states it for the dummy stations, so set
  the switch to `False` if you want to limit it to the stations used for MT
  sites.
- Renumbering a real VTF station changes its ID in the output file. If other
  files of your workflow (plots, scripts, notes) refer to the old VTF IDs,
  they have to follow; the script prints old and new IDs of the first three.
  Set `RENUMBER_EXISTING_VTF = False` to keep the IDs.
- `PT` stations keep `DEFAULT_OWNER_OTHER`; the PT block is not part of the
  manual's recipe.
- An MT station whose H-ID differs from its own ID keeps that H-ID. A
  warning is printed and no dummy is created for it.
- The E-field-type column (`ELEC_FIELD -1`) is placed after the owner column,
  following slide 36. The manual examples (slides 56, 58) show only the owner
  column, so test that case on a small model first.
- The number of MT stations is unchanged by the script. If you use
  distortion (`DISTORTION > 0`), `distortion_iter0.dat` must still match the
  MT stations in `observe.dat`; the version seen with this data set listed 33
  stations against 42 MT stations, so regenerate it.
- The script handles the block keywords `MT`, `APP_RES_AND_PHS`, `VTF`,
  `HTF`, `PT`. Other blocks cause an "unknown keyword" error rather than
  being passed through silently.

## 8. Sources

- FEMTIC manual, `Manual_Of_FEMTIC_v4_3.pdf`:
  - slides 55-56, "Things to be careful when seafloor MT data is used" (+ Example)
  - slides 57-58, "How to improve response functions at high frequencies" (+ Example)
  - slides 36-45, observe.dat block formats (MT, APP_RES_AND_PHS, VTF, HTF, PT)
  - slide 8, `OWNER_ELEMENT` and `ELEC_FIELD` keywords
- "Usui et al., 2018" is cited on slide 55 for the seafloor case. The
  manual gives no full reference for it, and none was looked up here, so no
  DOI is given. Verify the reference before citing it.
