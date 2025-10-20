#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:02:04 2025

@author: vrath
"""

import math
from pyproj import CRS, database

def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Return the EPSG code for the UTM zone (WGS84) covering the point (lon, lat).
    Uses EPSG:326## for northern hemisphere, EPSG:327## for southern hemisphere.
    """
    # compute 1..60 UTM zone
    zone = int((math.floor((lon + 180) / 6) % 60) + 1)
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    # Validate that this EPSG exists
    try:
        CRS.from_epsg(epsg)
    except Exception:
        raise ValueError(f"UTM EPSG {epsg} not recognized")
    return epsg

def find_local_projected_crs(lon: float, lat: float, buffer_deg: float = 0.1, max_results: int = 10):
    """
    Query pyproj database for projected CRSs overlapping a bbox around the point.
    Returns a list of dicts: [{'epsg': int, 'name': str, 'extent': (west,south,east,north)}...]
    buffer_deg is the half-width/half-height of the bbox in degrees (approx).
    """
    # build small bbox (west, south, east, north)
    west = lon - buffer_deg
    east = lon + buffer_deg
    south = lat - buffer_deg
    north = lat + buffer_deg

    # query CRSs from the pyproj database (authority = 'EPSG')
    info_list = database.query_crs_info(auth_name='EPSG', bbox=(west, south, east, north))
    results = []
    for info in info_list[:max_results]:
        # info is a pyproj.database.CRSInfo object with attributes: name, code, auth_name, area_of_use, bbox
        try:
            epsg_code = int(info.code)
            crs = CRS.from_epsg(epsg_code)
            # keep only projected CRSs (not geographic)
            if crs.is_projected:
                results.append({
                    'epsg': epsg_code,
                    'name': info.name,
                    'area_of_use': info.area_of_use,
                    'bbox': info.bbox
                })
        except Exception:
            continue
    return results

# --- Example usage ---
if __name__ == "__main__":
    lon, lat = 6.0, 45.6   # example coordinates (replace with your own)
    utm_epsg = get_utm_epsg(lon, lat)
    print("UTM EPSG (WGS84):", utm_epsg)

    candidates = find_local_projected_crs(lon, lat, buffer_deg=0.2, max_results=20)
    print("\nLocal projected CRS candidates (EPSG, name):")
    for c in candidates:
        print(f"EPSG:{c['epsg']}\t{c['name']}")
