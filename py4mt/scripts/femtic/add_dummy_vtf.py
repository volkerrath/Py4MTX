#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_dummy_vtf.py
================

Convert a simple FEMTIC observe.dat (layout for OWNER_ELEMENT 0) into the
layout with dummy VTF stations, so that the electric field and the magnetic
field of each MT station are interpolated in different elements.

Provenance
----------
Author    : Claude (Anthropic)
Generated : 2026-10-20
Notice    : AI-generated code. Review and test before production use.
Basis     : FEMTIC manual slides "Things to be careful when seafloor MT data
            is used" and "How to improve response functions at high
            frequencies" (Manual_Of_FEMTIC_v4.3.pdf), observe.dat format
            slides of the same manual.

What the script does
--------------------
For every station of the blocks MT and APP_RES_AND_PHS:

  1. a dummy VTF station is created at the same X, Y location. Its station ID
     is  <MT ID> + DUMMY_ID_OFFSET,  its H-field ID is its own ID, and it has
     0 frequencies (as in the manual example);
  2. the MT station gets the dummy ID as "Station ID for horizontal magnetic
     field";
  3. both stations get a "type of owner element" column:

        SITE_TYPE   MT station (E field)     dummy VTF (H field)
        ---------   ---------------------    ---------------------
        land        0 = downward element     1 = upward element
        seafloor    1 = upward element       0 = downward element

All dummy stations are written into one VTF block (appended to an existing
VTF block if the file already has one).

Resulting station lines (OWNER_ELEMENT -1 layout)
-------------------------------------------------
    MT block / APP block : ID  H-ID  owner  [E-type]  X  Y
    VTF block            : ID  H-ID  owner  X  Y
    HTF block            : ID  ID    owner  X  Y
    PT block             : ID  H-ID  owner  [E-type]  X  Y

REQUIRED CHANGE IN control.dat
------------------------------
The new file is only valid together with

    OWNER_ELEMENT
    -1

in control.dat. With OWNER_ELEMENT 0 or 1 FEMTIC reads the extra column as
the X coordinate and the run is garbage or fails.

Notes and limits
----------------
- All numbers (frequencies, data, standard deviations, X, Y) are copied as
  text, so nothing is rounded or reformatted.
- Existing stations of other blocks (VTF, HTF, PT) are kept unchanged and get
  the owner type DEFAULT_OWNER_OTHER (0 reproduces the old OWNER_ELEMENT 0).
  Existing VTF/HTF stations at the MT locations are NOT reused; dummies are
  added anyway.
- An MT station whose H-field ID differs from its own ID is left pointing to
  that ID (warning printed, no dummy created for it).
- If the file carries the E-field type column (ELEC_FIELD -1), set
  HAS_ELEC_COLUMN = True. The column order  owner, E-type  follows the table
  order of the manual; the manual example only shows the owner column.
- Keywords are matched in upper case; each block keyword may appear once.

Usage
-----
    python add_dummy_vtf.py                       # uses the USER SECTION
    python add_dummy_vtf.py in.dat out.dat        # override file names
    python add_dummy_vtf.py in.dat out.dat seafloor
"""

import sys
from dataclasses import dataclass, field
from typing import List, Tuple

# ===========================================================================
# USER SECTION - edit here only
# ===========================================================================

# Input file in the simple layout, and output file (must be different).
INPUT_FILE = "observe.dat"
OUTPUT_FILE = "observe_dummyVTF.dat"

# "land"     : E from downward element (0), H from upward element (1)
# "seafloor" : E from upward element (1),   H from downward element (0)
SITE_TYPE = "land"

# Dummy VTF station ID = MT station ID + DUMMY_ID_OFFSET.
# Must not collide with any station ID already in the file (checked).
DUMMY_ID_OFFSET = 1000

# Layout of the INPUT file.
# True if the input already has the "type of owner element" column
# (OWNER_ELEMENT -1 layout). False for the simple layout.
INPUT_HAS_OWNER_COLUMN = False

# True if the file has the "type of electric field" column per station
# (ELEC_FIELD -1 in control.dat). False for ELEC_FIELD 0 or 1.
HAS_ELEC_COLUMN = False

# Owner type given to pre-existing stations of the blocks VTF, HTF, PT when
# the input has no owner column. 0 keeps the meaning of the old OWNER_ELEMENT 0.
DEFAULT_OWNER_OTHER = 0

# Re-read the written file and compare it with the input (recommended).
VERIFY_OUTPUT = True

# ===========================================================================
# END OF USER SECTION
# ===========================================================================

# (owner type of MT station, owner type of dummy VTF station)
OWNER_TYPES = {
    "land": (0, 1),
    "seafloor": (1, 0),
}

# keyword -> (station line may carry the E-field type column,
#             number of tokens per frequency line = frequency + data + SDs)
BLOCK_SPECS = {
    "MT": (True, 17),
    "APP_RES_AND_PHS": (True, 17),
    "VTF": (False, 9),
    "HTF": (False, 17),
    "PT": (True, 9),
}

# Blocks whose stations get a dummy VTF partner.
DUMMY_SOURCE_BLOCKS = ("MT", "APP_RES_AND_PHS")

OWNER_NAMES = {0: "downward element", 1: "upward element"}


@dataclass
class Station:
    sid: int
    hid: int
    owner: int
    elec: str
    x: str
    y: str
    freqs: List[List[str]] = field(default_factory=list)


Blocks = List[Tuple[str, List[Station]]]


class TokenStream:
    """Whitespace token reader with readable error messages."""

    def __init__(self, text: str):
        self.tok = text.split()
        self.pos = 0

    def take(self, n: int) -> List[str]:
        if self.pos + n > len(self.tok):
            raise ValueError("unexpected end of file after token %d" % self.pos)
        out = self.tok[self.pos:self.pos + n]
        self.pos += n
        return out

    def one(self) -> str:
        return self.take(1)[0]


def to_int(token: str, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError("expected an integer for %s, got %r" % (what, token))


def parse_observe(text: str, has_owner: bool, has_elec: bool,
                  default_owner: int) -> Blocks:
    """Parse observe.dat into a list of (keyword, stations)."""
    ts = TokenStream(text)
    blocks: Blocks = []
    seen = set()
    while True:
        if ts.pos >= len(ts.tok):
            raise ValueError("keyword END is missing at the end of the file")
        key = ts.one()
        if key == "END":
            break
        if key not in BLOCK_SPECS:
            raise ValueError("unknown block keyword %r (token %d)"
                             % (key, ts.pos))
        if key in seen:
            raise ValueError("block %s appears more than once" % key)
        seen.add(key)
        elec_possible, ntok = BLOCK_SPECS[key]
        nsta = to_int(ts.one(), key + " station count")
        stations = []
        for i in range(nsta):
            what = "%s station %d" % (key, i + 1)
            sid = to_int(ts.one(), what + " ID")
            hid = to_int(ts.one(), what + " H-field ID")
            owner = (to_int(ts.one(), what + " owner type")
                     if has_owner else default_owner)
            elec = ts.one() if (has_elec and elec_possible) else ""
            x, y = ts.take(2)
            nf = to_int(ts.one(), what + " number of frequencies")
            freqs = [ts.take(ntok) for _ in range(nf)]
            stations.append(Station(sid, hid, owner, elec, x, y, freqs))
        blocks.append((key, stations))
    return blocks


def add_dummy_vtf(blocks: Blocks, site_type: str, offset: int) -> List[Station]:
    """Add dummy VTF stations in place and return them."""
    mt_owner, dummy_owner = OWNER_TYPES[site_type]

    used = set()
    for _, sts in blocks:
        for st in sts:
            used.add(st.sid)
            used.add(st.hid)

    dummies: List[Station] = []
    new_ids = set()
    for name, sts in blocks:
        if name not in DUMMY_SOURCE_BLOCKS:
            continue
        for st in sts:
            st.owner = mt_owner
            if st.hid != st.sid:
                print("WARNING: %s station %d uses H-field of station %d; "
                      "no dummy created for it." % (name, st.sid, st.hid))
                continue
            did = st.sid + offset
            if did in used or did in new_ids:
                raise ValueError(
                    "dummy ID %d (station %d + offset %d) is already in use; "
                    "choose another DUMMY_ID_OFFSET" % (did, st.sid, offset))
            new_ids.add(did)
            st.hid = did
            dummies.append(Station(did, did, dummy_owner, "", st.x, st.y, []))

    if not dummies:
        raise ValueError("no MT / APP_RES_AND_PHS stations found, nothing to do")

    for name, sts in blocks:
        if name == "VTF":
            sts.extend(dummies)
            break
    else:
        pos = len(blocks)
        for i, (name, _) in enumerate(blocks):
            if name in ("HTF", "PT"):
                pos = i
                break
        blocks.insert(pos, ("VTF", dummies))
    return dummies


def render(blocks: Blocks, has_elec: bool) -> str:
    """Write blocks in the OWNER_ELEMENT -1 layout."""
    lines = []
    for name, sts in blocks:
        elec_possible = BLOCK_SPECS[name][0]
        lines.append("%s  %d" % (name, len(sts)))
        for st in sts:
            cols = [str(st.sid), str(st.hid), str(st.owner)]
            if has_elec and elec_possible:
                cols.append(st.elec)
            cols += [st.x, st.y]
            lines.append("  ".join(cols))
            lines.append(str(len(st.freqs)))
            for f in st.freqs:
                lines.append(" ".join(f))
    lines.append("END")
    return "\n".join(lines) + "\n"


def verify(text: str, out_text: str, site_type: str, offset: int,
           n_dummies: int) -> None:
    """Re-parse the output and compare it with the input."""
    mt_owner, dummy_owner = OWNER_TYPES[site_type]
    orig = dict(parse_observe(text, INPUT_HAS_OWNER_COLUMN, HAS_ELEC_COLUMN,
                              DEFAULT_OWNER_OTHER))
    new = dict(parse_observe(out_text, True, HAS_ELEC_COLUMN,
                             DEFAULT_OWNER_OTHER))

    def fail(msg):
        raise RuntimeError("verification failed: " + msg)

    for name, osts in orig.items():
        if name not in new:
            fail("block %s missing in output" % name)
        nsts = new[name]
        if len(nsts) < len(osts):
            fail("block %s lost stations" % name)
        for o, n in zip(osts, nsts):
            if (o.sid, o.x, o.y, o.freqs) != (n.sid, n.x, n.y, n.freqs):
                fail("%s station %d changed" % (name, o.sid))
            if name in DUMMY_SOURCE_BLOCKS:
                want_hid = o.sid + offset if o.hid == o.sid else o.hid
                if n.owner != mt_owner or n.hid != want_hid:
                    fail("%s station %d has wrong owner/H-ID" % (name, o.sid))
            elif (o.hid, o.owner) != (n.hid, n.owner):
                fail("%s station %d changed" % (name, o.sid))

    extras = new["VTF"][len(orig.get("VTF", [])):]
    if len(extras) != n_dummies:
        fail("expected %d dummy stations, found %d" % (n_dummies, len(extras)))
    src = {st.sid: st for nm in DUMMY_SOURCE_BLOCKS if nm in orig
           for st in orig[nm]}
    for e in extras:
        s = src.get(e.sid - offset)
        if s is None or (s.x, s.y) != (e.x, e.y):
            fail("dummy %d does not match its MT station" % e.sid)
        if e.freqs or e.hid != e.sid or e.owner != dummy_owner:
            fail("dummy %d is malformed" % e.sid)


def main(argv: List[str]) -> int:
    in_path = argv[1] if len(argv) > 1 else INPUT_FILE
    out_path = argv[2] if len(argv) > 2 else OUTPUT_FILE
    site_type = argv[3] if len(argv) > 3 else SITE_TYPE

    if site_type not in OWNER_TYPES:
        sys.exit("ERROR: SITE_TYPE must be one of %s, got %r"
                 % (sorted(OWNER_TYPES), site_type))
    if in_path == out_path:
        sys.exit("ERROR: output file must differ from the input file")

    try:
        with open(in_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        blocks = parse_observe(text, INPUT_HAS_OWNER_COLUMN, HAS_ELEC_COLUMN,
                               DEFAULT_OWNER_OTHER)
        counts = {name: len(sts) for name, sts in blocks}
        dummies = add_dummy_vtf(blocks, site_type, DUMMY_ID_OFFSET)
        out_text = render(blocks, HAS_ELEC_COLUMN)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out_text)
        if VERIFY_OUTPUT:
            verify(text, out_text, site_type, DUMMY_ID_OFFSET, len(dummies))
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit("ERROR: %s" % exc)

    mt_owner, dummy_owner = OWNER_TYPES[site_type]
    print("=" * 70)
    print("add_dummy_vtf: %s" % site_type)
    print("=" * 70)
    print("Input   : %s  (%s)" % (in_path, ", ".join(
        "%s %d" % (k, v) for k, v in counts.items())))
    print("Output  : %s" % out_path)
    print("-" * 70)
    print("MT owner type        : %d (%s)" % (mt_owner, OWNER_NAMES[mt_owner]))
    print("Dummy VTF owner type : %d (%s)" % (dummy_owner,
                                              OWNER_NAMES[dummy_owner]))
    print("Dummy VTF stations   : %d  (IDs %d ... %d)"
          % (len(dummies), min(d.sid for d in dummies),
             max(d.sid for d in dummies)))
    if VERIFY_OUTPUT:
        print("Verification         : passed")
    print("-" * 70)
    print("REMINDER - control.dat must contain:")
    print("    OWNER_ELEMENT")
    print("    -1")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
