# femtic_add_dummy_vtf

Convert a simple FEMTIC `observe.dat` into the layout with dummy VTF
stations, so that the electric field and the magnetic field of every MT
station are interpolated in different elements (`OWNER_ELEMENT -1`).

Provenance: written by Claude (Anthropic), 2026-10-20. AI-generated code;
review and test before production use.

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
2. add a dummy VTF (or HTF) station at each MT location;
3. give the dummy the owner type used for H, and the MT station the owner
   type used for E;
4. set the MT station's "Station ID for horizontal magnetic field" to the
   ID of the dummy.

## 2. Files

| File                              | Purpose                                   |
|-----------------------------------|-------------------------------------------|
| `femtic_add_dummy_vtf.py`         | the converter                             |
| `femtic_add_dummy_vtf_test.py`    | 18 edge-case tests                        |
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
| `INPUT_HAS_OWNER_COLUMN` | `False`                  | `True` if the input already has the owner-type column                   |
| `HAS_ELEC_COLUMN`        | `False`                  | `True` if stations carry the E-field-type column (`ELEC_FIELD -1`)      |
| `DEFAULT_OWNER_OTHER`    | `0`                      | owner type for pre-existing VTF/HTF/PT stations (0 = old behaviour)     |
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

Rules applied to every station of the `MT` and `APP_RES_AND_PHS` blocks:

- a dummy VTF station is created at the same X, Y, with ID = MT ID + offset,
  H-ID = its own ID, and 0 frequencies (as in the manual example);
- the MT station's H-ID is set to the dummy ID;
- owner types are set as in the table of section 1;
- all dummies go into one `VTF` block. If the file already has a `VTF`
  block, they are appended; otherwise a new block is inserted before the
  first `HTF`/`PT` block, or before `END`.

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
- Error if a dummy ID is already in use, or if the file has no MT /
  `APP_RES_AND_PHS` stations.
- Refuses to overwrite the input file.
- `VERIFY_OUTPUT`: re-parses the written file and checks that all data
  lines, coordinates and original IDs are unchanged, that owner types and
  H-IDs are as intended, and that each dummy matches its MT station.

Result for the supplied file: 42 MT stations, 42 dummies (IDs 1001-1042),
verification passed. The `VTF` block starts at line 1182.

Tests: `python -I femtic_add_dummy_vtf_test.py` runs 18 cases (land and
seafloor mode, merge into an existing VTF block, block insertion order, E-type
column pass-through, ID collision, bad offset, truncated input, missing END,
unknown and duplicate keywords, foreign H-ID, in == out, bad `SITE_TYPE`).

## 7. Limits and things to check

- Existing VTF/HTF stations at the MT locations are not reused; dummies are
  added anyway. The manual (slides 55, 57) says new dummies are unnecessary
  in that case; if you prefer to reuse them, edit those stations by hand.
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
