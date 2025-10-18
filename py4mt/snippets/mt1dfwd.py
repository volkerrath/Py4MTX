def mt1dfwd(freq, sig, d, inmod="r", out="imp", magfield="b"):
    """
    Calulate 1D magnetotelluric forward response.

    based on A. Pethik's script at www.digitalearthlab.com
    Last change vr Nov 20, 2020
    """
    import numpy as np
    mu0 = 4.0e-7 * np.pi  # Magnetic Permeability (H/m)

    sig = np.array(sig)
    freq = np.array(freq)
    d = np.array(d)

    if inmod[0] == "c":
        sig = np.array(sig)
    elif inmod[0] == "r":
        sig = 1.0 / np.array(sig)

    if sig.ndim > 1:
        exit("IP not yet implemented")

    n = np.size(sig)

    Z = np.zeros_like(freq) + 1j * np.zeros_like(freq)
    w = np.zeros_like(freq)

    ifr = -1
    for f in freq:
        ifr = ifr + 1
        w[ifr] = 2.0 * np.pi * f
        imp = np.array(range(n)) + np.array(range(n)) * 1j

        # compute basement impedance
        imp[n - 1] = np.sqrt(1j * w[ifr] * mu0 / sig[n - 1])

        for layer in range(n - 2, -1, -1):
            sl = sig[layer]
            dl = d[layer]
            # 3. Compute apparent rho from top layer impedance
            # Step 2. Iterate from bottom layer to top(not the basement)
            #   Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt(1j * w[ifr] * mu0 * sl)
            wj = dj / sl
            #   Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = np.exp(-2 * dl * dj)

            #   Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance
            impb = imp[layer + 1]
            rj = (wj - impb) / (wj + impb)
            re = rj * ej
            Zj = wj * ((1 - re) / (1 + re))
            imp[layer] = Zj

        Z[ifr] = imp[0]
        # print(Z[ifr])

    if out.lower() == "imp":

        if magfield.lower() =="b":
            return Z/mu0
        else:
            return Z

    elif out.lower() == "rho":
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))

        return rhoa, phase
    else:
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))
        return Z, rhoa, phase


def mt1dfwd_simple(periods=None, thick=None, res=None):
    import numpy as np
    scale = 1 / (4 * np.pi / 10000000)
    mu = 4 * np.pi * 1e-7 * scale
    omega = 2 * np.pi / periods

    cond = 1 / np.array(res)

    sp = np.size(periods)
    Z = np.zeros(sp, dtype=complex)
    rhoa = np.zeros(sp)
    phi = np.zeros(sp)

    for freq, w in enumerate(omega):
        prop_const = np.sqrt(1j*mu*cond[-1] * w)
        C = np.zeros(sp, dtype=complex)
        C[-1] = 1 / prop_const
        if len(thick) > 1:
            for k in reversed(range(len(res) - 1)):
                prop_layer = np.sqrt(1j*w*mu*cond[k])
                k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * thick[k]))
                k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * thick[k])) + 1)
                C[k] = (1 / prop_layer) * (k1 / k2)
        Z[freq] = 1j * w * mu * C[0]

    rhoa = 1/omega*np.abs(Z)**2
    phi = np.angle(Z, deg=True)
    return rhoa, phi, np.real(Z), np.imag(Z)
