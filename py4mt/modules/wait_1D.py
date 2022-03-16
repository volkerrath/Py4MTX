import numpy as np



scale = 1 / (4 * np.pi / 10000000)
mu = 4 * np.pi * 1e-7 * scale
periods = np.logspace(-4, 4, 80)
omega = 2 * np.pi / periods
# d = np.cumsum(self.thickness + [100000])
thickness = [1000, 1000]
r = [1000, 1000]
cond = 1 / np.array(r)
# r = 1 / np.array(r)
Z = np.zeros(len(periods), dtype=complex)
rhoa = np.zeros(len(periods))
phi = np.zeros(len(periods))

for nfreq, w in enumerate(omega):
    prop_const = np.sqrt(1j*mu*cond[-1] * w)
    C = np.zeros(len(r), dtype=complex)
    C[-1] = 1 / prop_const
    if len(thickness) > 1:
        for k in reversed(range(len(r) - 1)):
            prop_layer = np.sqrt(1j*w*mu*cond[k])
            k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * thickness[k]))
            k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * thickness[k])) + 1)
            C[k] = (1 / prop_layer) * (k1 / k2)
# #         k2 = np.sqrt(1j*omega[nfreq]*C*mu0/r[k+1]);
#         g = (g*k2+k1*np.tanh(k1*d[k]))/(k1+g*k2*np.tanh(k1*d[k]));
    Z[nfreq] = 1j * w * mu * C[0]
rhoa = 1/omega*np.abs(Z)**2;
phi = np.angle(Z, deg=True);
