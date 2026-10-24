#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_add_dummy_vtf.py
=======================

Convert a simple FEMTIC observe.dat (layout for OWNER_ELEMENT 0) into the
layout with dummy VTF stations, so that the electric field and the magnetic
field of each MT station are interpolated in different elements.

Provenance
----------
Author    : Claude (Anthropic)
Generated : 2026-10-20 (updated 2026-10-22)
Notice    : AI-generated code. Review and test before production use.
Basis     : FEMTIC manual (Manual_Of_FEMTIC_v4_3.pdf), slides 55-56 "Things to
            be careful when seafloor MT data is used", slides 57-58 "How to
            improve response functions at high frequencies", slides 36-45
            observe.dat format.

What the script does
--------------------
Each MT / APP_RES_AND_PHS station is treated on its own, so a file in which
only some sites have a VTF station gets both treatments, site by site:

  1. NO VTF/HTF station at the site: a dummy VTF station is created at the
     same X, Y. Its station ID is  <MT ID> + DUMMY_ID_OFFSET,  its H-field ID
     is its own ID, and it has 0 frequencies (as in the manual example).
  2. A REAL VTF station (second choice: HTF) at the site, within
     MATCH_TOL_KM: no dummy is created. The real station gets the ID
     <MT ID> + DUMMY_ID_OFFSET (RENUMBER_EXISTING_VTF = True), its location and
     data stay as they are, and H-field IDs of other stations that pointed to
     its old ID are updated. This is what the manual allows (slides 55, 57).
  3. In both cases the MT station gets that ID as "Station ID for horizontal
     magnetic field", and both stations get a "type of owner element":

        SITE_TYPE   MT station (E field)     VTF/HTF station (H field)
        ---------   ---------------------    -------------------------
        land        0 = downward element     1 = upward element
        seafloor    1 = upward element       0 = downward element

New dummy stations are written into one VTF block (appended to an existing
VTF block if the file already has one).

Existing VTF and HTF stations also get the improved H field: with
IMPROVE_ALL_VTF_HTF = True every existing VTF/HTF station, not only those at
MT sites, gets the H-field type of the table above, because their magnetic
field benefits in the same way. With False only the stations used for MT sites
get it.

On an existing VTF/HTF station only the ID (renumbering) and the owner type
ever change. If the input has no owner column (simple layout), the type is set
as described. If the input already has an owner column, its values are kept
and one warning lists the stations whose type differs from the table.

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
- Existing stations of the PT block, and of the VTF/HTF blocks when
  IMPROVE_ALL_VTF_HTF is False (stations used for MT sites excepted), are kept
  unchanged and get the owner type DEFAULT_OWNER_OTHER (0 reproduces the old
  OWNER_ELEMENT 0).
- A VTF/HTF station is only used if its H-field ID equals its own ID and it
  lies within MATCH_TOL_KM of the MT station in X and Y. Otherwise a warning
  is printed and a dummy is created. If both a VTF and an HTF station match,
  the one in the earlier block of REUSE_BLOCKS is used. Several MT stations
  at one site share one existing station; it is renumbered after the first.
- References to a renumbered ID (H-field IDs of other stations) are updated
  unless the old ID is also used by another station (e.g. an MT station with
  the same number); then a warning is printed and they are left unchanged.
- The new ID must be free: a clash with any ID in the file is an error, and
  DUMMY_ID_OFFSET has to be changed.
- RENUMBER_EXISTING_VTF = False keeps the ID of existing stations and points
  the MT station to it (warning if that ID equals the MT station's ID).
- Set REUSE_EXISTING_STATIONS = False to always add dummies.
- An MT station whose H-field ID differs from its own ID is left pointing to
  that ID (warning printed, no dummy created for it).
- If the file carries the E-field type column (ELEC_FIELD -1), set
  HAS_ELEC_COLUMN = True. The column order  H-ID, owner, E-type, X, Y  is
  the one of the MT table in the manual (slide 36); the manual examples
  (slides 56, 58) only show the owner column.
- Keywords are matched in upper case; each block keyword may appear once.

Usage
-----
    python femtic_add_dummy_vtf.py                       # uses the USER SECTION
    python femtic_add_dummy_vtf.py in.dat out.dat        # override file names
    python femtic_add_dummy_vtf.py in.dat out.dat seafloor
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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

# True : a VTF / HTF station that already sits at the location of an MT
#        station is used as its H-field station instead of adding a dummy
#        there (manual slides 55, 57). Sites without a VTF/HTF station get a
#        dummy, so files with only some VTF sites are handled site by site.
# False: always add a dummy, even if a VTF / HTF station is already there.
REUSE_EXISTING_STATIONS = True

# Blocks searched for such stations, in order of preference.
REUSE_BLOCKS = ("VTF", "HTF")

# Two stations are at the same location if |dX| and |dY| are both <= this (km).
MATCH_TOL_KM = 1.0e-3

# True : a used existing station gets the ID  <MT ID> + DUMMY_ID_OFFSET, the
#        same ID a dummy would get, and the MT station is pointed to it.
#        Location and data are not touched. H-field IDs of other stations that
#        pointed to the old ID are updated. This also removes the clash when a
#        VTF station has the same ID as its MT station.
# False: the existing station keeps its ID; the MT station is pointed to it.
RENUMBER_EXISTING_VTF = True

# True : every existing VTF / HTF station gets the H-field owner type of
#        SITE_TYPE (land 1, seafloor 0), so their magnetic field is also
#        taken from the improved element.
# False: only the reused stations get it; the others get DEFAULT_OWNER_OTHER.
# This switch only sets the owner type (and only if the input has no owner
# column); locations and data of existing stations are never touched.
IMPROVE_ALL_VTF_HTF = True

# Layout of the INPUT file.
# True if the input already has the "type of owner element" column
# (OWNER_ELEMENT -1 layout). False for the simple layout.
INPUT_HAS_OWNER_COLUMN = False

# True if the file has the "type of electric field" column per station
# (ELEC_FIELD -1 in control.dat). False for ELEC_FIELD 0 or 1.
HAS_ELEC_COLUMN = False

# Owner type given to pre-existing stations of the block PT (and of VTF/HTF,
# if not covered by IMPROVE_ALL_VTF_HTF) when the input has no owner column.
# 0 keeps the meaning of the old OWNER_ELEMENT 0.
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

# Blocks whose stations are magnetic-field stations.
H_BLOCKS = ("VTF", "HTF")

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


@dataclass
class Plan:
    """What add_dummy_vtf() did, needed for the summary and the verification."""
    dummies: List[Station] = field(default_factory=list)
    # MT/APP station ID -> (block, ORIGINAL ID) of the existing VTF/HTF
    # station it uses
    reused: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    # MT/APP station ID -> H-field ID written to the output
    final_hid: Dict[int, int] = field(default_factory=dict)
    # (block, original ID) -> new ID of renumbered existing VTF/HTF stations
    renamed: Dict[Tuple[str, int], int] = field(default_factory=dict)
    # old ID -> new ID, applied to H-field IDs of other stations
    ref_remap: Dict[int, int] = field(default_factory=dict)
    # number of H-field IDs of other stations that were updated
    refs_updated: int = 0
    # IDs of existing VTF/HTF stations whose owner type was set to the H type
    owner_set: List[int] = field(default_factory=list)


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


def to_float(token: str) -> float:
    """float() that also accepts Fortran D exponents."""
    return float(token.replace("D", "E").replace("d", "e"))


def same_place(a: Station, b: Station, tol: float) -> bool:
    """True if the two stations are at the same X, Y within tol (km)."""
    try:
        return (abs(to_float(a.x) - to_float(b.x)) <= tol
                and abs(to_float(a.y) - to_float(b.y)) <= tol)
    except ValueError:
        return a.x == b.x and a.y == b.y


def find_existing(st: Station, blocks: Blocks, reuse_blocks, tol: float):
    """Return (block name, station) of an existing VTF/HTF station at the
    location of st that can be used for the magnetic field, or None."""
    for want in reuse_blocks:
        for name, sts in blocks:
            if name != want:
                continue
            for cand in sts:
                if not same_place(st, cand, tol):
                    continue
                if cand.hid != cand.sid:
                    print("WARNING: %s station %d is at the location of %d "
                          "but has H-field ID %d; not reused."
                          % (name, cand.sid, st.sid, cand.hid))
                    continue
                return name, cand
    return None


def add_dummy_vtf(blocks: Blocks, site_type: str, offset: int,
                  reuse_blocks=(), tol: float = 0.0,
                  input_has_owner: bool = False,
                  improve_all: bool = False,
                  renumber: bool = False) -> Plan:
    """Set up the H-field station of every MT / APP station in place.

    Per MT / APP station (H-field ID equal to its own ID):
      - if a VTF/HTF station (blocks in reuse_blocks) exists at the same
        location, it is used; with renumber=True it gets the ID
        <MT ID> + offset, exactly the ID a dummy would get, and H-field IDs
        of other stations that pointed to its old ID are updated;
      - otherwise a new dummy VTF station with that ID is created.
    The H-field ID of the MT station is set to the ID of that station.
    Existing VTF/HTF stations get the H-field owner type if they are used or
    if improve_all is True. Only their ID (renumber) and owner type are ever
    changed, never their location or data.
    """
    mt_owner, dummy_owner = OWNER_TYPES[site_type]

    all_sts = [st for _, sts in blocks for st in sts]
    used = set()
    sid_count: Dict[int, int] = {}
    for st in all_sts:
        used.add(st.sid)
        used.add(st.hid)
        sid_count[st.sid] = sid_count.get(st.sid, 0) + 1
    orig_sid = {id(st): st.sid for st in all_sts}
    foreign = [st for st in all_sts if st.hid != st.sid]
    source_objs = {id(st) for name, sts in blocks
                   if name in DUMMY_SOURCE_BLOCKS for st in sts}

    plan = Plan()
    new_ids = set()
    n_source = 0
    for name, sts in blocks:
        if name not in DUMMY_SOURCE_BLOCKS:
            continue
        for st in sts:
            n_source += 1
            st.owner = mt_owner
            if st.hid != st.sid:
                print("WARNING: %s station %d uses H-field of station %d; "
                      "no dummy created for it." % (name, st.sid, st.hid))
                plan.final_hid[st.sid] = st.hid
                continue
            target = st.sid + offset

            found = find_existing(st, blocks, reuse_blocks, tol)
            if found is not None:
                bname, cand = found
                oid = orig_sid[id(cand)]
                plan.reused[st.sid] = (bname, oid)
                if renumber:
                    # rename only once, and only if it is not already the target
                    if cand.sid == oid and oid != target:
                        if target in used or target in new_ids:
                            raise ValueError(
                                "ID %d for %s station %d (MT station %d + "
                                "offset %d) is already in use; choose "
                                "another DUMMY_ID_OFFSET"
                                % (target, bname, oid, st.sid, offset))
                        new_ids.add(target)
                        cand.sid = cand.hid = target
                        plan.renamed[(bname, oid)] = target
                elif oid == st.sid:
                    print("WARNING: %s station %d has the same ID as MT "
                          "station %d; the H-field ID cannot tell them apart. "
                          "Set RENUMBER_EXISTING_VTF = True."
                          % (bname, oid, st.sid))
                st.hid = cand.sid
                plan.final_hid[st.sid] = cand.sid
                continue

            if target in used or target in new_ids:
                raise ValueError(
                    "dummy ID %d (station %d + offset %d) is already in use; "
                    "choose another DUMMY_ID_OFFSET" % (target, st.sid, offset))
            new_ids.add(target)
            st.hid = target
            plan.final_hid[st.sid] = target
            plan.dummies.append(
                Station(target, target, dummy_owner, "", st.x, st.y, []))

    if n_source == 0:
        raise ValueError("no MT / APP_RES_AND_PHS stations found, nothing to do")
    if not plan.dummies and not plan.reused:
        raise ValueError("every MT / APP_RES_AND_PHS station already has its "
                         "own H-field ID, nothing to do")

    # H-field IDs of other stations that pointed to a renumbered station
    for (bname, oid), new_id in plan.renamed.items():
        if sid_count.get(oid, 0) > 1:
            n_refs = sum(1 for st in foreign if st.hid == oid)
            if n_refs:
                print("WARNING: ID %d of %s station is also used by another "
                      "station; %d H-field ID reference(s) to %d are left "
                      "unchanged." % (oid, bname, n_refs, oid))
        else:
            plan.ref_remap[oid] = new_id
    for st in foreign:
        if st.hid in plan.ref_remap:
            st.hid = plan.ref_remap[st.hid]
            plan.refs_updated += 1
            if id(st) in source_objs:
                plan.final_hid[st.sid] = st.hid

    # Owner type of the EXISTING VTF/HTF stations (before dummies are added).
    reused_now = {(b, plan.renamed.get((b, oid), oid))
                  for b, oid in plan.reused.values()}
    differs = []
    for name, sts in blocks:
        if name not in H_BLOCKS:
            continue
        for st in sts:
            if not (improve_all or (name, st.sid) in reused_now):
                continue
            if not input_has_owner:
                st.owner = dummy_owner
                plan.owner_set.append(st.sid)
            elif st.owner != dummy_owner:
                differs.append(st.sid)
    if differs:
        shown = ", ".join(str(i) for i in differs[:10])
        print("WARNING: %d existing VTF/HTF station(s) keep an owner type "
              "different from the H-field type %d for %s (IDs: %s%s)."
              % (len(differs), dummy_owner, site_type, shown,
                 ", ..." if len(differs) > 10 else ""))

    if plan.dummies:
        for name, sts in blocks:
            if name == "VTF":
                sts.extend(plan.dummies)
                break
        else:
            pos = len(blocks)
            for i, (name, _) in enumerate(blocks):
                if name in ("HTF", "PT"):
                    pos = i
                    break
            blocks.insert(pos, ("VTF", list(plan.dummies)))
    return plan


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
           plan: Plan) -> None:
    """Re-parse the output and compare it with the input."""
    mt_owner, dummy_owner = OWNER_TYPES[site_type]
    orig = dict(parse_observe(text, INPUT_HAS_OWNER_COLUMN, HAS_ELEC_COLUMN,
                              DEFAULT_OWNER_OTHER))
    new = dict(parse_observe(out_text, True, HAS_ELEC_COLUMN,
                             DEFAULT_OWNER_OTHER))
    reused_now = {(b, plan.renamed.get((b, oid), oid))
                  for b, oid in plan.reused.values()}

    def fail(msg):
        raise RuntimeError("verification failed: " + msg)

    for name, osts in orig.items():
        if name not in new:
            fail("block %s missing in output" % name)
        nsts = new[name]
        if len(nsts) < len(osts):
            fail("block %s lost stations" % name)
        for o, n in zip(osts, nsts):
            # Location and all data of every original station unchanged; the
            # ID only changes for renumbered VTF/HTF stations.
            exp_sid = o.sid
            if name in H_BLOCKS:
                exp_sid = plan.renamed.get((name, o.sid), o.sid)
            if (exp_sid, o.x, o.y, o.freqs) != (n.sid, n.x, n.y, n.freqs):
                fail("%s station %d changed" % (name, o.sid))
            if name in DUMMY_SOURCE_BLOCKS:
                if n.owner != mt_owner or n.hid != plan.final_hid[o.sid]:
                    fail("%s station %d has wrong owner/H-ID" % (name, o.sid))
            else:
                if o.hid == o.sid:
                    exp_hid = exp_sid
                else:
                    exp_hid = plan.ref_remap.get(o.hid, o.hid)
                want_owner = o.owner
                if (name in H_BLOCKS and not INPUT_HAS_OWNER_COLUMN
                        and (IMPROVE_ALL_VTF_HTF
                             or (name, exp_sid) in reused_now)):
                    want_owner = dummy_owner
                if (exp_hid, want_owner) != (n.hid, n.owner):
                    fail("%s station %d changed" % (name, o.sid))

    mt_orig = {st.sid: st for nm in DUMMY_SOURCE_BLOCKS if nm in orig
               for st in orig[nm]}
    for sid, (bname, oid) in plan.reused.items():
        cands = [st for st in orig.get(bname, []) if st.sid == oid]
        if not cands or not same_place(mt_orig[sid], cands[0], MATCH_TOL_KM):
            fail("station %d does not match its reused %s station %d"
                 % (sid, bname, oid))

    # every H-field ID in use must belong to a VTF/HTF station
    h_ids = {st.sid for nm in H_BLOCKS for st in new.get(nm, [])}
    for nm in DUMMY_SOURCE_BLOCKS:
        for o, n in zip(orig.get(nm, []), new.get(nm, [])):
            if o.hid == o.sid and n.hid not in h_ids:
                fail("%s station %d: H-field ID %d is not a VTF/HTF station"
                     % (nm, o.sid, n.hid))

    # IDs created or changed by the script must be unique in the output
    count: Dict[int, int] = {}
    for sts in new.values():
        for st in sts:
            count[st.sid] = count.get(st.sid, 0) + 1
    for i in list(plan.renamed.values()) + [d.sid for d in plan.dummies]:
        if count.get(i, 0) != 1:
            fail("ID %d is not unique in the output" % i)

    extras = new.get("VTF", [])[len(orig.get("VTF", [])):]
    if len(extras) != len(plan.dummies):
        fail("expected %d dummy stations, found %d"
             % (len(plan.dummies), len(extras)))
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
        plan = add_dummy_vtf(
            blocks, site_type, DUMMY_ID_OFFSET,
            reuse_blocks=REUSE_BLOCKS if REUSE_EXISTING_STATIONS else (),
            tol=MATCH_TOL_KM, input_has_owner=INPUT_HAS_OWNER_COLUMN,
            improve_all=IMPROVE_ALL_VTF_HTF,
            renumber=RENUMBER_EXISTING_VTF)
        out_text = render(blocks, HAS_ELEC_COLUMN)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out_text)
        if VERIFY_OUTPUT:
            verify(text, out_text, site_type, DUMMY_ID_OFFSET, plan)
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit("ERROR: %s" % exc)

    mt_owner, dummy_owner = OWNER_TYPES[site_type]
    print("=" * 70)
    print("femtic_add_dummy_vtf: %s" % site_type)
    print("=" * 70)
    print("Input   : %s  (%s)" % (in_path, ", ".join(
        "%s %d" % (k, v) for k, v in counts.items())))
    print("Output  : %s" % out_path)
    print("-" * 70)
    print("MT owner type        : %d (%s)" % (mt_owner, OWNER_NAMES[mt_owner]))
    print("H-field owner type   : %d (%s)" % (dummy_owner,
                                              OWNER_NAMES[dummy_owner]))
    if plan.dummies:
        print("New dummy VTF sites  : %d  (IDs %d ... %d)"
              % (len(plan.dummies), min(d.sid for d in plan.dummies),
                 max(d.sid for d in plan.dummies)))
    else:
        print("New dummy VTF sites  : none")
    by_block: Dict[str, int] = {}
    for bname, _ in plan.reused.values():
        by_block[bname] = by_block.get(bname, 0) + 1
    print("Existing sites reused: %d%s"
          % (len(plan.reused), "  (" + ", ".join(
              "%s %d" % (k, v) for k, v in by_block.items()) + ")"
             if by_block else ""))
    if plan.renamed:
        ex = ", ".join("%d->%d" % (oid, nid) for (_, oid), nid
                       in list(plan.renamed.items())[:3])
        print("  renumbered         : %d  (%s%s)"
              % (len(plan.renamed), ex,
                 ", ..." if len(plan.renamed) > 3 else ""))
    if plan.refs_updated:
        print("  H-ID references of other stations updated: %d"
              % plan.refs_updated)
    print("Existing VTF/HTF given H owner type: %d" % len(plan.owner_set))
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
