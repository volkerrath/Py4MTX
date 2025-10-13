"""Summary
"""
import numpy as np
import os
from copy import deepcopy
from math import log10, floor
import pyproj
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import median_filter
# from pyMT.IO import debug_print


MU = 4 * np.pi * 1e-7


def remove_bad_data(data, error, tolerance=0.5):
    good_idx = []
    bad_idx = []
    for ii, d, e in enumerate(zip(data, error)):
        if d / e < tolerance:
            good_idx.append(ii)
        else:
            bad_idx.append(ii)
    return good_idx, bad_idx


def is_all_empties(dictionary):
    empties = []
    for key, val in dictionary.items():
        if not any(True for ii in val):
            empties.append(key)
    if set(empties) == set(dictionary.keys()):
        return True
    else:
        return False


def edge2center(x):
    y = np.zeros(len(x) - 1)
    for ii in range(len(x) - 1):
        y[ii] = (x[ii] + x[ii + 1]) / 2
    return y


def percdiff(val1, val2):
    """Summary

    Args:
        val1 (TYPE): Description
        val2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    return abs(val1 - val2) / (0.5 * (val1 + val2))


def remove_outliers(data, size=7, threshold=1):
    med = median_filter(data, size=size)
    diff = percdiff(data, med)
    idx = abs(diff) > threshold
    output = np.array(data)
    output[idx] = med[idx]
    return output


def skin_depth(resistivity, period):
    return 500 * np.sqrt(resistivity * period)


def strike_direction(site):
    zxx = site.data['ZXXR'] + 1j * site.data['ZXXI']
    zxy = site.data['ZXYR'] + 1j * site.data['ZXYI']
    zyx = site.data['ZYXR'] + 1j * site.data['ZYXI']
    zyy = site.data['ZYYR'] + 1j * site.data['ZYYI']
    C = ((zxx - zyy) * np.conj(zxy + zyx) + np.conj(zxx - zyy) * (zxy + zyx)) / \
        ((zxx - zyy) * np.conj(zxx - zyy) - (zxy - zyx) * np.conj(zxy - zyx))
    return np.rad2deg(0.25 * np.arctan(np.real(C)))


def dms_to_dd(dms_string):
    degrees, minutes, seconds = str.split(dms_string, ':')
    degrees = float(degrees)
    minutes = float(minutes)
    try:
        seconds = float(seconds)
    except ValueError:
        seconds = float(seconds[:-1])
    return degrees + minutes / 60 + seconds / 3600

def dd_to_dms(dd):
    mult = -1 if dd < 0 else 1
    mnt,sec = divmod(abs(dd)*3600, 60)
    deg,mnt = divmod(mnt, 60)
    return mult*deg, mnt, sec


def generate_zmesh(min_depth=1, max_depth=500000, NZ=None):
    print('This function is depreciated.\n')
    print('It functions as it used to, however, the function has been updated and included as a method within the pyMT Model object.')
    num_decade = int(np.ceil(np.log10(max_depth)) - floor(np.log10(min_depth)))
    try:
        if num_decade != len(NZ):
            print('Number of decades is: {}'.format(num_decade))
            print('Number of parameters (NZ) given is: {}'.format(len(NZ)))
            print('Length of NZ must be equal to the number of decades.')
            return
        decade = np.log10(min_depth)
        depths = []
        for n in NZ:
            dDecade = np.logspace(decade, min(np.floor(decade + 1), np.log10(max_depth)), int(n + 1))
            decade = floor(decade + 1)
            depths.append(dDecade)
        depths = flatten_list(depths)
        depths = np.array(np.unique(depths))
    except TypeError:
        depths = np.logspace(np.log10(min_depth), np.log10(max_depth), int(NZ))
    ddz = np.diff(np.diff(depths))
    if any(ddz < 0):
        print('Warning! Second derivative of depths is not always positive.')
    zCS = np.diff(depths)
    zCS = list(zCS)
    depths = list(depths)
    depths.insert(0, 0)
    zCS.insert(0, depths[1])
    return depths, zCS, ddz


def generate_lateral_mesh(site_locs, regular=True, min_x=None, model=None, max_x=None,
                          num_pads=None, pad_mult=None, DEBUG=True):
    """Summary

    Args:
        site_locs (TYPE): Description
        min_x (TYPE): Description
        model (None, optional): Description
        num_pads (None, optional): Description
        pad_mult (None, optional): Description
        DEBUG (bool, optional): Description

    Returns:
        TYPE: Description
    """
    def regular_mesh(x_size, bounds):
        print(bounds, x_size)
        xmesh = list(np.arange(bounds[0], bounds[1], x_size))
        nmesh = len(xmesh)
        return xmesh, nmesh

    def j2_mesh(site_locs, min_x, max_x):
        MAX_X = 150
        xlocs = site_locs  # South-North
        # ylocs = site_locs[:, 1]  # West-East
        xloc_sort = np.sort(xlocs)
        # yloc_sort = np.sort(ylocs)
        xoff = xloc_sort[0] + (xloc_sort[-1] - xloc_sort[0]) / 2
        xloc_sort = (xloc_sort - xoff).astype(int)
        x_seperation = np.diff(xloc_sort)
        max_sep = np.max(x_seperation)
        xmesh = np.zeros(MAX_X)
        # median_seperation = np.median(x_seperation)
        avg_sep = np.mean(x_seperation)
        max_xmin = 3 * avg_sep
        if not min_x:
            min_x = max_xmin / 2
        if not max_x:
            max_x = min_x * 2
        if min_x > max_xmin:
            print('Minimum cell size shouldn\'t be more than {}'.format(max_xmin))
            # return
        dist_without_mesh = 0
        is_right_ofmesh = 0
        imesh = 1
        xmesh[0] = xloc_sort[0] - min_x * 0.5
        for ii in range(len(xloc_sort) - 1):
            # Look to station location on right
            dist_right = xloc_sort[ii + 1] - xloc_sort[ii]
            ifact = 0
            dist = 0
            while dist < dist_right:
                ifact += 1
                dist += min_x + min_x * ifact
            ifact -= ifact
            if ifact >= 2:
                # if (imesh + ifact * 2 + 1 > MAX_X):
                #     resp = input('Number of cells exceeds {}. Continue? (y/n)'.format(MAX_X))
                #     if resp == 'n':
                #         return
                for jj in range(1, max(ifact - 2, 1) + 1):
                    imesh += 1
                    xmesh[imesh - 1] = xmesh[imesh - 2] + min_x * jj
                imesh += 1
                xmesh[imesh - 1] = xloc_sort[ii] + (xloc_sort[ii + 1] - xloc_sort[ii]) / 2
                imesh += ifact - 1
                imesh_save = imesh
                xmesh[imesh - 1] = xloc_sort[ii + 1] - min_x / 2
                for jj in range(1, ifact - 1):
                    imesh -= 1
                    xmesh[imesh - 1] = xmesh[imesh] - min_x * jj
                imesh = imesh_save
                dist_without_mesh = 0
                is_right_ofmesh = ii + 1
            elif (dist_right >= min_x) and (dist_right <= 5 * min_x):
                if DEBUG:
                    print('Gap is small, splitting sites')
                # if (imesh + 1 >= MAX_X):
                #     resp = input('Number of cells exceeds {}. Continue? (y/n)'.format(MAX_X))
                #     if resp == 'n':
                #         return
                imesh += 1
                xmesh[imesh - 1] = xloc_sort[ii] + (xloc_sort[ii + 1] - xloc_sort[ii]) / 2
                dist_without_mesh = 0
                is_right_ofmesh = ii + 1
            else:
                # no room for mesh?
                if DEBUG:
                    print('Sites too close, splitting mesh')
                dist_without_mesh = xloc_sort[ii + 1] - xmesh[imesh - 1]
            if dist_without_mesh > max_x:
                if DEBUG:
                    print("Gone too far without adding mesh")
                max_sep = 0
                # if is_right_ofmesh == 0:
                # is_right_ofmesh = 1
                for is_check in range(is_right_ofmesh, ii + 1):
                    check_sep = xloc_sort[is_check + 1] - xloc_sort[is_check]
                    if check_sep > max_sep:
                        is_save = is_check
                        max_sep = check_sep
                if imesh + 1 > MAX_X:
                    resp = input('Number of cells exceeds {}. Continue? (y/n)'.format(MAX_X))
                    if resp == 'n':
                        return
                imesh += 1
                xmesh[imesh - 1] = xloc_sort[is_save] + (xloc_sort[is_save + 1] - xloc_sort[is_save]) / 2
                dist_without_mesh = 0
                is_right_ofmesh = is_save + 1
        imesh += 1
        if imesh > MAX_X:
            resp = input('Number of cells exceeds {}. Continue? (y/n)'.format(MAX_X))
            if resp == 'n':
                return
        xmesh[imesh - 1] = xloc_sort[-1] + xloc_sort[-1] - xmesh[imesh - 2]
        nmesh = imesh
        return xmesh[:nmesh], nmesh

    # def add_pads(mesh, num_pads, pad_mult):
    #     pad = pad_mult * mesh[0]
    #     self.model.dy_insert(0, self.model.dy[0] - pad)
    #     if self.padRight.checkState():
    #         pad = pad_mult * mesh[-1]
    #         self.model.dy_insert(self.model.ny + 1, pad + self.model.dy[-1])
    #     if self.padTop.checkState():
    #         pad = pad_mult * mesh[-1]
    #         self.model.dx_insert(self.model.nx + 1, self.model.dx[-1] + pad)
    #     if self.padBottom.checkState():
    #         pad = pad_mult * mesh[0]
    #         self.model.dx_insert(0, self.model.dx[0] - pad)

    if regular:
        min_x = max(min_x, 10)
        bounds = [np.min(site_locs) - 500, np.max(site_locs) + 500]
        # print(min_x, bounds)
        xmesh, nmesh = regular_mesh(min_x, bounds)
        if nmesh < 2:
            xmesh.append(xmesh[0] + 1)
        print(xmesh, nmesh)
    else:
        xmesh, nmesh = j2_mesh(site_locs=site_locs, min_x=min_x, max_x=max_x)
    # if num_pads:
    #     xmesh, nmesh = add_padding(mesh, num_pads, pad_mult)
    return xmesh, nmesh


def enforce_input(**types):
    """Summary

    Args:
        **types (TYPE): Description

    Returns:
        TYPE: Description
    """
    def decorator(func):
        """Summary

        Args:
            func (TYPE): Description

        Returns:
            TYPE: Description
        """
        def new_func(*args, **kwargs):
            """Summary

            Args:
                *args (TYPE): Description
                **kwargs (TYPE): Description

            Returns:
                TYPE: Description
            """
            newargs = {}
            for k, t in types.items():
                arg = kwargs.get(k, None)
                if isinstance(arg, t) or arg is None:
                    newargs.update({k: arg})
                else:
                    if t == list:
                        newargs.update({k: to_list(kwargs[k])})
                    elif t is np.ndarray:
                        newargs.update({k: np.array(kwargs[k])})
                    else:
                        newargs.update({k: t(kwargs[k])})
            return func(*args, **newargs)
        return new_func
    return decorator


def enforce_output(*types):
    """Summary

    Args:
        *types (TYPE): Description

    Returns:
        TYPE: Description
    """
    def decorator(func):
        """Summary

        Args:
            func (TYPE): Description

        Returns:
            TYPE: Description
        """
        def new_outputs(*args, **kwargs):
            """Summary

            Args:
                *args (TYPE): Description
                **kwargs (TYPE): Description

            Returns:
                TYPE: Description
            """
            newouts = []
            outs = func(*args, **kwargs)
            for (t, a) in zip(types, outs):
                if isinstance(a, t):
                    newouts.append(a)
                else:
                    if t == list:
                        newouts.append(to_list(a))
                    elif t == np.array:
                        newouts.append(np.array(a))
                    else:
                        newouts.append(t(a))
            return newouts
        return new_outputs
    return decorator


def closest_periods(available_p, wanted_p):
    """Summary

    Args:
        available_p (TYPE): Description
        wanted_p (TYPE): Description

    Returns:
        TYPE: Description
    """
    retval = []
    for p in wanted_p:
        ind = np.argmin([abs(ap - p) for ap in available_p])
        retval.append(available_p[ind])
    retval = sorted(set(retval))
    if isinstance(available_p, np.ndarray):
        retval = np.array(retval)
    return retval


def linear_distance(x, y):
    nodes = np.array([x, y]).T
    linear_x = np.zeros(x.shape)
    linear_x[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 +
                           (y[1:] - y[:-1]) ** 2)
    linear_x = np.cumsum(linear_x)
    linear_site = np.ones((x.shape))
    for ii, (x1, y1) in enumerate(zip(x, y)):
        dist = np.sum((nodes - np.array([x1, y1]).T) ** 2, axis=1)
        idx = np.argmin(dist)
        linear_site[ii] = linear_x[idx]
    return linear_site


def list_or_numpy(func):
    """Summary

    Args:
        func (TYPE): Description

    Returns:
        TYPE: Description
    """
    def inner(obj):
        """Summary

        Args:
            obj (TYPE): Description

        Returns:
            TYPE: Description
        """
        made_change = False
        if isinstance(obj, np.ndarray):
            obj = [x for x in obj]
            made_change = True
        retval = func(obj)
        if made_change:
            retval = np.array(retval)
        return retval
    return inner


def check_file(outfile):
    """Summary

    Args:
        outfile (TYPE): Description

    Returns:
        TYPE: Description
    """
    try:
        open(outfile, 'r')
    except FileNotFoundError:
        return False
    else:
        return True


def row_or_col(func):
    """Summary

    Args:
        func (TYPE): Description

    Returns:
        TYPE: Description
    """
    def inner(vec):
        """Summary

        Args:
            vec (TYPE): Description

        Returns:
            TYPE: Description
        """
        made_change = False
        if np.shape(vec)[0] < np.shape(vec)[1]:
            vec = np.transpose(vec)
            made_change = True
        retval = func(vec)
        if made_change:
            retval = np.transpose(retval)
        return retval
    return inner


# def model_origin


def compute_longorigin(Long=0, long_origin=999):
    if Long == 0:
        origin = -3
    else:
        origin = float(int(Long / 6) * 6) + 3 * int(Long / 6) / abs(int(Long / 6))
    return origin


def geo2utm(Lat, Long, long_origin=999):
    """Summary

    Args:
        Lat (TYPE): Description
        Long (TYPE): Description

    Returns:
        TYPE: Description
    """
    # AA = 6378206.4
    # ECC2 = 0.00676866
    AA = 6378137
    ECC2 = 0.00669438
    K0 = 0.9996
    if long_origin == 999:
        origin = compute_longorigin(Long)
    else:
        origin = long_origin
    LatR, LongR = np.deg2rad([Lat, Long])
    eccPrimeSquared = ECC2 / (1 - ECC2)
    N = AA / (np.sqrt(1 - ECC2 * np.sin(LatR) ** 2.0))
    T = np.tan(LatR) ** 2
    C = eccPrimeSquared * np.cos(LatR) ** 2
    A = np.cos(LatR) * np.deg2rad((Long - origin))
    M = AA * ((1 -
               ECC2 / 4 -
               3 * ECC2 * ECC2 / 64 -
               5 * ECC2 * ECC2 * ECC2 / 256) * LatR -
              (3 * ECC2 / 8 +
               3 * ECC2 * ECC2 / 32 +
               45 * ECC2 * ECC2 * ECC2 / 1024) * np.sin(2 * LatR) +
              (15 * ECC2 * ECC2 / 256 + 45 * ECC2 * ECC2 * ECC2 / 1024) * np.sin(4 * LatR) -
              (35 * ECC2 * ECC2 * ECC2 / 3072) * np.sin(6 * LatR))
    # M = 111132.0894 * Lat - 16216.94 * np.sin(2 * LatR) + 17.21 * \
    #     np.sin(4 * LatR) - 0.02 * np.sin(6 * LatR)
    Easting = K0 * N * \
        (A + (1 - T + C) * ((A ** 3) / 6) +
         (5 - 18 * T ** 3 + 72 * C - 58 * eccPrimeSquared) *
            (A ** 5) / 120) + 500000

    Northing = K0 * \
        (M + N * np.tan(LatR) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * ((A ** 4) / 24)) +
         (61 - 58 * T + T * T + 600 * C -
          330 * eccPrimeSquared) * ((A ** 6) / 720))
    return Easting, Northing, origin


def to_list(obj):
    """Summary

    Args:
        obj (TYPE): Description

    Returns:
        TYPE: Description
    """
    if not hasattr(obj, '__iter__') or isinstance(obj, str):
        obj = [obj]
    else:
        obj = list(obj)
        # obj = [x for x in obj]
    return obj


def fileparts(filename):
    """Summary

    Args:
        filename (TYPE): Description

    Returns:
        TYPE: Description
    """
    path = os.path.dirname(filename)
    filename = os.path.basename(filename)
    ext = os.path.splitext(filename)[1]
    return path, filename, ext


@list_or_numpy
def truncate(nums):
    """Summary

    Args:
        nums (TYPE): Description

    Returns:
        TYPE: Description

    Deleted Parameters:
        sig (int, optional): Description
    """
    sig = 5
    if nums is not None:
        if hasattr(nums, '__iter__'):
            retval = []
            for x in nums:
                if x == 0:
                    retval.append(0)
                else:
                    retval.append(round(x, -int(floor(log10(abs(x)))) + (sig - 1)))
            # retval = [round(x, -int(floor(log10(abs(x)))) + (sig - 1)) for x in nums]
            # print('This is a list')
        else:
            if nums == 0:
                retval = nums
            else:
                retval = round(nums, -int(floor(log10(abs(nums)))) + (sig - 1))
        return retval


def center_locs(locations):
    """Summary

    Args:
        locations (TYPE): Description

    Returns:
        TYPE: Description
    """
    X, Y = locations[:, 0], locations[:, 1]
    # center = (np.mean(X), np.mean(Y))
    center = ((np.max(X) + np.min(X)) / 2, (np.max(Y) + np.min(Y)) / 2)
    X, Y = X - center[0], Y - center[1]
    return np.array([[x, y] for x, y in zip(X, Y)]), center


def rotate_locs(locs, azi=0):
    """Summary

    Args:
        locs (TYPE): Description
        azi (int, optional): Description

    Returns:
        TYPE: Description
    """
    if azi == 0:
        return locs
    azi = np.deg2rad(azi)
    locs, center = center_locs(locs)
    R = np.array([[np.cos(azi), -np.sin(azi)], [np.sin(azi), np.cos(azi)]])
    # Rotates with +ve azis moving clockwise from North
    rot_locs = np.matmul(locs, R.T)
    # Rotates with +ve azimuth moving counter-clockwise from East
    # rot_locs = np.matmul(locs, R)
    rot_locs += center
    return rot_locs


def project_to_line(locs, azi):
    a = np.array([0, 0])
    b = np.array([np.tan(np.deg2rad(azi)), 1])
    projected_locs = np.zeros(locs.shape)
    for ii in range(len(locs)):
        ap = locs[ii, :] - a
        ab = b - a
        projected_locs[ii, :] = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return projected_locs


def list2np(data):
    """Summary

    Args:
        data (TYPE): Description

    Returns:
        TYPE: Description
    """
    try:
        return np.array(data)
    except ValueError as e:
        print('Error here')
        print(data)
        print('Error over')
        raise(e)


def np2list(data):
    """Summary

    Args:
        data (TYPE): Description

    Returns:
        TYPE: Description
    """
    return [x for x in data]


def flatten_list(List):
    """Summary

    Args:
        List (TYPE): Description

    Returns:
        TYPE: Description
    """
    try:
        ret = [float(point) for sublist in List for point in sublist]
    except ValueError:
        ret = [point for sublist in List for point in sublist]
    return  ret


def validate_input(inval, expected_type):
    """Summary

    Args:
        inval (TYPE): Description
        expected_type (TYPE): Description

    Returns:
        TYPE: Description
    """
    try:
        retval = expected_type(inval)
    except ValueError:
        print('Conversion not possible. Please enter a valid input (of type {})'.format(expected_type))
        return False
    else:
        return retval


def compute_rho(site, calc_comp=None, errtype='none'):
    """Summary
    Computes apparent resistivity from impedance data for a given site.
    Args:
        site (Site): wsinv3dmt.data_structures Site object containing impedance data to be converted
        calc_comp (string, optional): Type of apparent resistivity to calculate.
        (Rxx, Rxy, Ryy, Ryx, Rdet {Default})

    Returns:
        np.array: Apparent resistivity of the chosen type
        np.array: Associated errors. Note that the errors are set to 0 for the determinant apparent
                  resistivities (Proper calculatation not implemented yet)
    """
    def compute_rho_error(site, errtyped, compd):
        # compd = ''.join(['Z', compd.upper()])
        zeR = getattr(site, errtyped)[''.join([compd, 'R'])]
        zeI = getattr(site, errtyped)[''.join([compd, 'I'])]
        z = site.data[''.join([compd, 'R'])] - 1j * site.data[''.join([compd, 'I'])]
        if len(zeR) == 0:
            zeR = zeI = z * 0
        C = 2 * site.periods / (MU * 2 * np.pi)
        rho_error = np.sqrt((C * np.real(z) * zeR) ** 2 + (C * np.imag(z) * zeI) ** 2)
        rho_log10Err = (rho_error * np.sqrt(2) /
                        ((C * np.real(z) ** 2 + C * np.imag(z) ** 2) * np.log(10)))
        idx = zeR == site.REMOVE_FLAG
        # Preserve remove flags
        rho_error[idx] = site.REMOVE_FLAG
        rho_log10Err[idx] = site.REMOVE_FLAG
        return rho_error, rho_log10Err

    COMPS = ('XX', 'XY',
             'YY', 'YX',
             'DET', 'GAV', 'AAV', 'SSQ')
    if not calc_comp:
        calc_comp = 'det'
    calc_comp = calc_comp.lower()
    if 'rho' in calc_comp:
        calc_comp = calc_comp.replace('rho', '')
    if calc_comp.upper() not in COMPS:
        print('{} not a valid option'.format(calc_comp))
        return
    # mu = 0.2 / (4 * np.pi * 1e-7)
    if calc_comp.upper() not in COMPS[:4]:
        if calc_comp.lower() == 'det':
            det, err = compute_MT_determinant(site)

        elif calc_comp == 'gav':
            det, err = compute_gav(site)
        elif calc_comp == 'aav':
            det, err = compute_aav(site)
        elif calc_comp == 'ssq':
            det, err = compute_ssq(site)
        rho_data = det * np.conj(det) * site.periods / (MU * 2 * np.pi)
        # print(err)
        if errtype.lower() != 'none':

            eXY, eXY_log10 = compute_rho_error(deepcopy(site), errtype, 'ZXY')
            eYX, eYX_log10 = compute_rho_error(site, errtype, 'ZYX')
            rho_error = np.max((eXY, eYX), axis=0)
            rho_log10Err = np.max((eXY_log10, eYX_log10), axis=0)
            # rho_error = rho_data * 0
            # rho_log10Err = rho_data * 0
        else:
        # rho_max = rho_data * (1 + 2*err)
        # rho_min = rho_data / (1 + 2*err)
        # rho_error = rho_max - rho_min
        # rho_log10Err = np.log10(rho_error)
            rho_error = rho_data * 0
            rho_log10Err = rho_data * 0
        # Do errors here...
    else:
        comp = ''.join(['Z', calc_comp.upper()])
        z = site.data[''.join([comp, 'R'])] - 1j * site.data[''.join([comp, 'I'])]
        rho_data = z * np.conj(z) * site.periods / (MU * 2 * np.pi)
        if errtype.lower() != 'none':
            rho_error, rho_log10Err = compute_rho_error(site, errtype, comp)
            # zeR = getattr(site, errtype)[''.join([comp, 'R'])]
            # zeI = getattr(site, errtype)[''.join([comp, 'I'])]
            # if len(zeR) == 0:
            #     zeR = zeI = z * 0
            # C = 2 * site.periods / (MU * 2 * np.pi)
            # rho_error = np.sqrt((C * np.real(z) * zeR) ** 2 + (C * np.imag(z) * zeI) ** 2)
            # rho_log10Err = (rho_error * np.sqrt(2) /
            #                 ((C * np.real(z) ** 2 + C * np.imag(z) ** 2) * np.log(10)))
            # idx = zeR == site.REMOVE_FLAG
            # # Preserve remove flags
            # rho_error[idx] = site.REMOVE_FLAG
            # rho_log10Err[idx] = site.REMOVE_FLAG
        else:
            rho_error = rho_log10Err = rho_data * 0
    return np.real(rho_data), np.real(rho_error), np.real(rho_log10Err)


def compute_phase(site, calc_comp=None, errtype=None, wrap=0):
    def compute_phase_error(phase_data, z_real, z_imag, r_error, im_err):
        # Phase error is calculated by first calculating the phase at
        # the 8 possible points when considering the complex errors,
        # and then averaging the results. This is probably not strictly speaking correct,
        # but it makes sense as an approximation at least. Overestimates phase errors
        # when impedance errors are large
        # pha_error = np.zeros(z_real.shape, dtype=np.complex128)
        zE = np.zeros((len(z_real), 8))
        zE[:, 0] = np.angle(z_real - r_error + (z_imag + im_err), deg=True)
        zE[:, 1] = np.angle(z_real + r_error + (z_imag + im_err), deg=True)
        zE[:, 2] = np.angle(z_real + r_error + (z_imag - im_err), deg=True)
        zE[:, 3] = np.angle(z_real - r_error + (z_imag - im_err), deg=True)
        zE[:, 4] = np.angle(z_real + (z_imag - im_err), deg=True)
        zE[:, 5] = np.angle(z_real + (z_imag + im_err), deg=True)
        zE[:, 6] = np.angle(z_real + r_error + z_imag, deg=True)
        zE[:, 7] = np.angle(z_real - r_error + z_imag, deg=True)
        pha_error = np.mean(abs(phase_data[:, np.newaxis] - zE), 1)
        # print(np.angle(z_real - r_error + (z_imag + im_err)))
        # print(phase_data)
        # Preserve remove flags
        idx = r_error == site.REMOVE_FLAG
        pha_error[idx] = site.REMOVE_FLAG
        return pha_error

    COMPS = ('XX', 'XY',
             'YY', 'YX',
             'DET', 'GAV', 'AAV', 'SSQ')
    if not errtype:
        errtype = 'none'
    if not calc_comp:
        calc_comp = 'det'
    calc_comp = calc_comp.lower()
    if 'pha' in calc_comp:
        calc_comp = calc_comp.replace('pha', '')
    if calc_comp.upper() not in COMPS:
        print('{} not a valid option'.format(calc_comp))
        return
    if calc_comp.upper() not in COMPS[:4]:
        if calc_comp == 'det':
            det, err = compute_MT_determinant(site)
        elif calc_comp == 'gav':
            det, err = compute_gav(site)
        elif calc_comp == 'aav':
            det, err = compute_aav(site)
        elif calc_comp == 'ssq':
            det, err = compute_ssq(site)
        # pha_data = (np.angle(det, deg=True) + 90) % 90
        pha_data = np.rad2deg(np.arctan2(np.imag(det), np.real(det)))
        if calc_comp == 'aav' or calc_comp == 'ssq':
            pha_data = pha_data % 90
            # zR = np.real(det)
            # zI = 1j*np.imag(det)
        else:
            # zR = np.real(det)
            # zI = 1j*np.imag(det)
            pha_data += 90
        # pha_error = pha_data * 0
        
        
        if errtype.lower() != 'none':
            zR = site.data['ZXYR']
            zI = -1j*site.data['ZXYI']
            z = zR + zI
            zeR = getattr(site, errtype)['ZXYR']
            zeI = -1j*getattr(site, errtype)['ZXYI']
            eXY = compute_phase_error(np.angle(z, deg=True), zR, zI, zeR, zeI)
            zR = site.data['ZYXR']
            zI = -1j*site.data['ZYXI']
            z = zR + zI
            zeR = getattr(site, errtype)['ZYXR']
            zeI = -1j*getattr(site, errtype)['ZYXI']
            eYX = compute_phase_error(np.angle(z, deg=True), zR, zI, zeR, zeI)
            pha_error = np.max((eXY, eYX), axis=0)
        else:
            pha_error = 0 * pha_data
        # pha_error = np.rad2deg(np.arctan(err))
    else:
        comp = ''.join(['Z', calc_comp.upper()])
        zR = site.data[''.join([comp, 'R'])]
        zI = -1j * site.data[''.join([comp, 'I'])]
        z = zR + zI
        pha_data = np.angle(z, deg=True)
        pha_error = pha_data * 0
        # if comp[1:] == 'YX':
        #     print('hi')
        #     pha_data = pha_data
        if errtype.lower() != 'none':
            zeR = getattr(site, errtype)[''.join([comp, 'R'])]
            zeI = -1j * getattr(site, errtype)[''.join([comp, 'I'])]
            pha_error = compute_phase_error(pha_data, zR, zI, zeR, zeI)
        if 'yx' in calc_comp and wrap:
            pha_data += 180
    return pha_data, pha_error


def compute_gav(site):
    try:
        gav = np.sqrt((site.data['ZXYR'] - 1j * site.data['ZXYI']) *
                      (site.data['ZYXR'] - 1j * site.data['ZYXI']))
        gav_err = np.sqrt(abs((site.used_error['ZXYR'] + 1j * site.used_error['ZXYI']) *
                              (site.used_error['ZYXR'] + 1j * site.used_error['ZYXI'])))
    except KeyError as e:
        print('Missing component needed for computation')
        raise e
    else:
        return gav, gav_err


def compute_aav(site):
    try:
        aav = ((abs(site.data['ZXYR']) + 1j * abs(site.data['ZXYI'])) +
               (abs(site.data['ZYXR']) + 1j * abs(site.data['ZYXI']))) / 2
        aav_err = ((abs(site.used_error['ZXYR']) + abs(site.used_error['ZXYI'])) +
                   (abs(site.used_error['ZYXR']) + abs(site.used_error['ZYXI']))) / 4
    except KeyError as e:
        print('Missing component needed for computation')
        raise e
    else:
        return aav, aav_err


def compute_MT_determinant(site):
    try:
        det = np.sqrt((site.data['ZXYR'] - 1j * site.data['ZXYI']) *
                      (site.data['ZYXR'] - 1j * site.data['ZYXI']) -
                      (site.data['ZXXR'] - 1j * site.data['ZXXI']) *
                      (site.data['ZYYR'] - 1j * site.data['ZYYI']))
        det_err = (site.used_error['ZXXR'] * (abs(site.data['ZYYR'] + 1j*site.data['ZYYI'])) +
                   site.used_error['ZYYR'] * (abs(site.data['ZXXR'] + 1j*site.data['ZXXI'])) +
                   site.used_error['ZXYR'] * (abs(site.data['ZYXR'] + 1j*site.data['ZYXI'])) +
                   site.used_error['ZYXR'] * (abs(site.data['ZXYR'] + 1j*site.data['ZXYI']))) / (2 * abs(det))
    except KeyError as e:
        print('Determinant cannot be computed unless all impedance components are available')
        raise e
    else:
        return det, det_err


def compute_ssq(site):
    try:
        zxy = (site.data['ZXYR'] - 1j * site.data['ZXYI'])
        zyx = (site.data['ZYXR'] - 1j * site.data['ZYXI'])
        zxx = (site.data['ZXXR'] - 1j * site.data['ZXXI'])
        zyy = (site.data['ZYYR'] - 1j * site.data['ZYYI'])
        ssq = np.sqrt((zxx * zxx + zxy * zxy + zyx * zyx + zyy * zyy) / 2)
        det_err = (site.used_error['ZXXR'] * (abs(site.data['ZYYR'] + 1j*site.data['ZYYI'])) +
                   site.used_error['ZYYR'] * (abs(site.data['ZXXR'] + 1j*site.data['ZXXI'])) +
                   site.used_error['ZXYR'] * (abs(site.data['ZYXR'] + 1j*site.data['ZYXI'])) +
                   site.used_error['ZYXR'] * (abs(site.data['ZXYR'] + 1j*site.data['ZXYI']))) / (2 * abs(ssq))
    except KeyError as e:
        print('SSQ cannot be computed unless all impedance components are available')
        raise e
    else:
        return ssq, det_err


def geotools_filter(x, y, fwidth=1, use_log=True):

    RTD = 180 / np.pi
    DIFLIMIT = 0.1
    DSLLIMIT = 90.0
    # print(use_log)
    if any(x < 0):
        logx = x
    else:
        logx = np.log10(x)
    if use_log:
        logy = np.log10(y)
        if any(np.isnan(logx)) or any(np.isnan(logy)):
            logx = x
            logy = y
            use_log = False
    else:
        # logx = x
        logy = y
    hfwidth = fwidth / 2
    min_x = min(logx)
    min_y = min(logy)
    max_x = max(logx)
    max_y = max(logy)
    X = np.maximum(np.minimum((logx - min_x) / (max_x - min_x), 1), 0)
    Y = np.maximum(np.minimum((logy - min_y) / (max_y - min_y), 1), 0)

    ldif = np.zeros(X.shape)
    lslope = np.zeros(X.shape)
    rslope = np.zeros(X.shape)
    rdif = np.zeros(X.shape)
    ret_y = np.zeros(X.shape)
    score = np.zeros(X.shape)
    # Left and right differences
    for ii, (x, y) in enumerate(zip(X, Y)):
        if ii > 0:
            ldif[ii] = Y[ii] - Y[ii - 1]
            if ldif[ii] == 0 and X[ii] == X[ii - 1]:
                lslope[ii] = 0
            else:
                lslope[ii] = RTD * np.arctan2(ldif[ii], X[ii] - X[ii - 1])

        if ii < len(X) - 1:
            rdif[ii] = Y[ii + 1] - Y[ii]
            if rdif[ii] == 0 and X[ii + 1] == X[ii]:
                rslope[ii] = 0
            else:
                rslope[ii] = RTD * np.arctan2(rdif[ii], X[ii + 1] - X[ii])
    ldif[0] = ldif[1]
    rdif[-1] = rdif[-2]
    lslope[0] = lslope[1]
    rslope[-1] = rslope[-2]

    # Generate score for each point

    for ii in range(len(X)):
        pen1 = 0
        pen2 = 0
        totaldif = abs(ldif[ii]) + abs(rdif[ii])
        if (((abs(ldif[ii]) > DIFLIMIT) and (abs(rdif[ii]) > DIFLIMIT) and
                (ldif[ii] * rdif[ii] < 0)) or (totaldif > 4 * DIFLIMIT)):
            pen1 = abs(ldif[ii]) - DIFLIMIT + abs(rdif[ii]) - DIFLIMIT
        dslope = rslope[ii] - lslope[ii]
        if (abs(dslope > DSLLIMIT)):
            pen2 = 0.01 * (abs(dslope) - DSLLIMIT)
        score[ii] = np.maximum(np.minimum(1 - pen1 - pen2, 1), 0.1)

    # Convolve filter with weighted, normalized Y data
    for ii in range(len(X)):
        point = 0
        WTSUM = 0
        for jj in range(len(X)):
            if (abs(X[ii] - X[jj]) <= fwidth):
                FILTER = max(min((hfwidth - abs(X[ii] - X[jj]) / hfwidth), 1), 0)
                WTSUM += FILTER * score[jj]
                point += FILTER * score[jj] * Y[jj]
        ret_y[ii] = (point / WTSUM) * (max_y - min_y) + min_y
    if use_log:
        ret_y = 10 ** ret_y
    return ret_y


def compute_bost1D(site, method='phase', comp=None, filter_width=1):
    if not comp:
        comp = 'det'
    if 'bost' in comp.lower():
        comp = comp.lower().replace('bost', '')
    rho = compute_rho(site, calc_comp=comp)[0]
    # Flag bad points
    idx = rho != 0
    rho = rho[idx]
    periods = site.periods[idx]
    # coefs = np.polyfit(np.log10(periods), np.log10(rho), 4)
    # f = np.poly1d(coefs)
    # rhofit = 10 ** f(np.log10(periods))
    rhofit = geotools_filter(periods, rho, fwidth=filter_width)
    log_rho = np.log10(rhofit)
    log_freq = np.log10(1 / periods)
    depth = np.sqrt(rhofit * periods / (2 * np.pi * MU)) / 1000
    if method.lower() == 'bostick':
        slope = np.diff(log_rho) / abs(np.diff(log_freq))
        slope = list(slope)
        slope.append((log_rho[-1] - log_rho[-2]) / abs(log_freq[-1] - log_freq[-2]))
        slope = np.array(slope)
        slope[abs(slope) > 0.95] = np.sign(slope[abs(slope) > 0.95]) * 0.95
        bostick = rhofit * ((1 + slope) / (1 - slope))
        phase = compute_phase(site, calc_comp=comp)[0]
    elif method.lower() == 'phase-bostick':
        phase = compute_phase(site, calc_comp=comp)[0]
        phase = phase[idx]
        phase = geotools_filter(periods, phase, use_log=False, fwidth=filter_width)
        slope = 1 - phase / 45
        bostick = rho * ((1 + slope) / (1 - slope))
    elif method.lower() == 'phase':
        phase = compute_phase(site, calc_comp=comp)[0]
        phase = phase[idx]
        phase = geotools_filter(periods, phase, use_log=False, fwidth=filter_width)
        bostick = rho * ((np.pi / (2 * np.deg2rad(phase % 90))) - 1)
    return bostick, depth, rhofit, phase


# @enforce_input(files=list)
# def sort_files(files):
#     ret_dict = {}
#     print(files)
#     types = ('model', 'dat', 'resp', 'lst', 'reso')
#     for file in files:
#         try:
#             file_type = (next(x for x in types if '.'+x in file))
#         except StopIteration:
#             if '.rho' in file.lower():
#                 ret_dict.update({'model': file})
#             else:
#                 print('{} does not correspond to a recognized file type'.format(file))
#         else:
#             ret_dict.update({file_type: file})
#     return ret_dict
@enforce_input(files=list)
def sort_files(files):
        ret_dict = {}
        # types = ('model', 'dat', 'resp', 'lst', 'reso')

        for file in files:
            name, ext = os.path.splitext(file)
            if ext in ('.rho', '.model', '.zani'):
                ret_dict.update({'model': file})
            elif ext in ('.dat', '.data'):
                with open(file, 'r') as f:
                    line = f.readline()
                if 'response' in line:
                    ret_dict.update({'response': file})
                else:
                    ret_dict.update({'data': file})
            elif ext in ('.lst', '.list'):
                ret_dict.update({'list': file})
            elif ext in ('.reso', '.resolution'):
                ret_dict.update({'resolution': file})
        return ret_dict


def calculate_misfit(data_site, response_site):
    components = response_site.components
    NP = len(data_site.periods)
    NR = len(response_site.components)
    # N_flagged = np.zeros(NP)
    misfit = {comp: np.zeros((NP)) for comp in components}
    # comp_misfit = {comp: 0 for comp in components}
    comp_misfit = {comp: np.zeros((NP)) for comp in components}
    period_misfit = np.zeros((NP))
    if NP != len(response_site.periods):
        print('Number of periods in data and response not equal!')
        return
    for comp in components:
        if comp in data_site.components:
            try:
                misfit[comp] = (np.abs(response_site.data[comp] - data_site.data[comp]) /
                                data_site.used_error[comp]) ** 2
                comp_misfit[comp] = ((misfit[comp]))
                period_misfit += misfit[comp]
                # Account for points that have been flagged for removal
                # N_flagged += [int(x) for x in data_site.used_error[comp] == data_site.REMOVE_FLAG] 
            except ValueError:
                a+=1
    # period_misfit = (period_misfit / NR - np.array(N_flagged))
    period_misfit = (period_misfit / NR)
    comp_misfit.update({'Total': period_misfit})
    return misfit, comp_misfit


def convert2impedance(rho, phase, periods, component):
    phase = np.deg2rad(phase)
    tp = np.tan(phase)
    re = np.sign(tp) * np.sqrt(5 * rho * MU / (periods * (1 + tp**2)))
    im = np.sign(tp) * np.sqrt(5 * rho * MU / (periods * (1 + 1 / (tp**2))))
    if 'xy' in component.lower():
        im = -im
    elif 'yx' in component.lower():
        re = -re
    return re, im


def check_extention(outfile, expected=''):
    path, file, ext = fileparts(outfile)
    if expected in ext:
        return outfile
    else:
        return '.'.join([outfile, expected])


def normalize_resolution(resolution, reqmean=0.5, reqvar=0.5):
    # Takes a model object. Should probably just be a part of the Model class
    [xCS, yCS, zCS] = np.meshgrid(resolution.xCS, resolution.yCS, resolution.zCS, indexing='ij')
    volumes = xCS * yCS * zCS
    resolution.vals = resolution.vals / (volumes ** (1 / 3))
    resolution.vals += -np.mean(resolution.vals)
    resolution.vals = resolution.vals / np.std(resolution.vals)
    resolution.vals = reqmean + resolution.vals * np.sqrt(reqvar)
    return resolution
# def brute_compute_strike(site, increment=1, band=None):
#     if band is None:
#         band = (site.periods[0], site.periods[-1])
#     working_site = deepcopy(site)
#     degrees = np.linspace(0, 359, increment)
#     XXPow = np.array((len(degrees), 1))
#     YYPow = np.array((len(degrees), 1))
#     idx = np.bitwise_and(site.periods >= band[0], site.periods <= band[1])
#     for ii, theta in enumerate(degrees):

# Note this section (and all other coordinate related things) should really be in its own module.
_projections = {}


def rms(data):
    return np.sqrt(np.nanmean(data ** 2))


def zone_convert(coordinates):
    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32
    if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35
        return 37
    return int((coordinates[0] + 180) / 6) + 1

def generic_projection(x, y, from_proj=None, to_proj=None, utm_zone=None):
    if not (from_proj and to_proj):
        print('Have to define your projections...')
        return False
    proj_dict = {'lambert': 'epsg:3979',
                 'latlong': 'epsg:4326',
                 'utm': 'epsg:4269'}
    # if isinstance(utm_zone, str):
        # proj_dict['utm'] += utm_zone
    try:                 
        transformer = pyproj.Transformer.from_crs(proj_dict[from_proj], proj_dict[to_proj])
        out_x, out_y = np.zeros(len(x)), np.zeros(len(y))
        for ii, (xx, yy) in enumerate(zip(x, y)):
            out_x[ii], out_y[ii] = transformer.transform(xx, yy)
    except KeyError:
        print('Requested projection not recognized')
        return x, y
    return out_x, out_y

def project(coordinates, zone=None, letter=None):
    def letter_convert(coordinates):
        return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]
    if not(zone and letter):
        z = zone_convert(coordinates)
        L = letter_convert(coordinates)
    else:
        z = zone
        L = letter
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    x, y = _projections[z](coordinates[0], coordinates[1])
    if y < 0:
        y += 10000000
    return z, L, x, y


def unproject(z, l, x, y):
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    if l < 'N':
        y -= 10000000
    lng, lat = _projections[z](x, y, inverse=True)
    return (lng, lat)


def to_lambert(x, y):
    transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3979')
    lam_x = np.zeros(x.shape)
    lam_y = np.zeros(y.shape)
    for ii, (lat, lon) in enumerate(zip(x, y)):
        lam_x[ii], lam_y[ii] = transformer.transform(lat, lon)
    return lam_x, lam_y

def parse_dms(dms):
    '''
        Parse a string of DMS where the degrees, minutes, seconds are separated by ':'
    '''
    d, m, s = [float(x) for x in dms.split(':')]
    return d, m, s


def dms2dd(dms):
    '''
        Converts strings containing dms lat longs to decimal degrees
    '''
    try:
        d, m, s = parse_dms(dms)
        dd = abs(d) + abs(m) / 60 + abs(s) / 3600
        val = dd * np.sign(d)
    except ValueError: # In case its already in decimal degrees
        return float(dms)
    return val

def dd2dms(dd):
    mult = -1 if dd < 0 else 1
    mnt, sec = divmod(abs(dd) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return mult * deg, mult * mnt, mult * sec

def normalize(vals, lower=0, upper=1, explicit_bounds=False):
    if explicit_bounds:
        norm_vals = (vals - lower) / (upper - lower)
    else:
        norm_vals = (vals - np.min(vals)) / \
                    (np.max(vals) - np.min(vals))
        norm_vals = norm_vals * (upper - lower) + lower
    return norm_vals


def normalize_range(vals, lower_range=0, upper_range=1, lower_norm=0, upper_norm=1):
    vals = list(vals)
    vals.append(lower_range)
    vals.append(upper_range)
    norm_vals = normalize(np.array(vals), lower=lower_norm, upper=upper_norm)
    norm_vals = norm_vals[:-2]
    return norm_vals


def regrid_model(mod, new_x, new_y, new_z):
    x, y, z = (edge2center(arr) for arr in (mod.dx, mod.dy, mod.dz))
    X, Y, Z = (edge2center(arr) for arr in (new_x, new_y, new_z))
    X_grid, Y_grid, Z_grid = np.meshgrid(X, Y, Z)
    X_grid, Y_grid, Z_grid = (np.ravel(arr) for arr in (X_grid, Y_grid, Z_grid))
    # interp = griddata((x_grid, y_grid, z_grid),
    #                   np.ravel(mod.vals),
    #                   (X_grid, Y_grid, Z_grid),
    #                   method='nearest')
    interp = RGI((y, x, z),
                 np.transpose(mod.vals, [1, 0, 2]),
                 method='nearest', bounds_error=False, fill_value=mod.background_resistivity)
    query_points = np.array((Y_grid, X_grid, Z_grid)).T
    new_vals = interp(query_points)
    new_vals = np.reshape(new_vals, [len(Y), len(X), len(Z)])
    new_vals = np.transpose(new_vals, [1, 0, 2])
    return new_vals
