# resistivity_models.py

Utilities for petrophysical conductivity, resistivity, permeability, and effective-medium calculations.

---

## Included models

- Archie (1942)
- Simandoux (1963)
- Dual-porosity / fractured parallel-network model
- Sen & Goode (1992) NaCl brine conductivity/resistivity
- Arps/Hilchie legacy brine resistivity approximation
- RGPZ permeability model after Glover et al. (2006)
- Glover, Hole & Pous (2000) two-phase conductivity mixing law
- Glover (2010) generalized n-phase conductivity mixing law
- Hashin-Shtrikman bounds for two-phase and n-phase composites
- Kozeny-Carman permeability for comparison
- Sinmyo & Keppler (2017) NaCl-H2O conductivity model
- Guo & Keppler (2019) NaCl-H2O conductivity model
- Sakuma & Ichiki (2016) NaCl-H2O conductivity model (low-T and high-T/P formulations)

---

## High-P/T fluid models

Explicit NaCl-H2O conductivity models:

- brine_conductivity_sakuma_high(temp_c, salinity_wt_pct, pressure_mpa)
- brine_conductivity_sakuma_low(temp_c, salinity_wt_pct)
- brine_conductivity_sinmyo_keppler(salinity_wt_pct, temp_c, density_g_cm3, pressure_gpa=None)
- brine_conductivity_guo_keppler(salinity_wt_pct, temp_c, density_g_cm3, pressure_gpa=None)

### Notes

- Sakuma (2016): polynomial parameterization (no EOS required)
- Sinmyo/Guo: require external water density
- Sakuma low-T model may produce non-physical values outside calibration range

---

## Example: water density (IAPWS)

```python
from iapws import IAPWS95

def water_density_iapws(T_c, P_MPa):
    w = IAPWS95(T=T_c + 273.15, P=P_MPa)
    return w.rho  # kg/m^3
```

---

## Main functions

### Brine / fluid functions

- brine_conductivity_sen_goode(salinity_ppm, temp_c)
- brine_conductivity_sen_goode_mol(C, temp_c)
- brine_resistivity_sen_goode(salinity_ppm, temp_c)
- brine_resistivity(salinity_ppm, temp_c)
- brine_conductivity_sakuma_high(temp_c, salinity_wt_pct, pressure_mpa)
- brine_conductivity_sakuma_low(temp_c, salinity_wt_pct)
- brine_conductivity_sinmyo_keppler(...)
- brine_conductivity_guo_keppler(...)

### Rock / bulk models

- archie(phi, Sw, Rw, m=2.0, n=2.0, a=1.0)
- simandoux(phi, Sw, Rw, Vsh, Rsh, m=2.0, n=2.0, a=1.0)
- dual_porosity(...)
- solve_Sw_archie(...)
- solve_Sw_simandoux(...)

### Permeability models

- rgpz(...)
- rgpz_from_formation_factor(...)
- kozeny_carman(...)

### Effective-medium / mixing models

- glover_two_phase(...)
- glover_n_phase(...)
- hashin_shtrikman_two_phase(...)
- hashin_shtrikman_n_phase(...)

---

## Units and conventions

- Conductivity: S/m
- Resistivity: Ω·m
- Temperature: °C
- Salinity:
  - ppm → Sen-Goode
  - wt% → Sakuma, Sinmyo, Guo
- Density: g/cm³ (Sinmyo, Guo)
- Pressure:
  - MPa → Sakuma
  - GPa → Sinmyo, Guo
- Grain size: µm
- Permeability: m² (internally), optionally mD

---

## Notes

- Sakuma: direct polynomial fit (crustal P–T conditions)
- Sinmyo/Guo: thermodynamic high P–T models
- Hashin-Shtrikman: rigorous bounds
- Archie/Simandoux: require connected pore space
- Module includes __main__ demo block

---

## References

Archie, G.E. (1942). The electrical resistivity log as an aid in determining some reservoir characteristics. Transactions of the AIME, 146, 54–62.

Simandoux, P. (1963). Dielectric measurements on porous media. Revue de l’Institut Français du Pétrole, 18, 193–220.

Warren, J.E. & Root, P.J. (1963). The behavior of naturally fractured reservoirs. SPE Journal, 3(3), 245–255.

Sen, P.N. & Goode, P.A. (1992). Influence of temperature on electrical conductivity on shaly sands. Geophysics, 57(1), 89–96.

Batzle, M. & Wang, Z. (1992). Seismic properties of pore fluids. Geophysics, 57(11), 1396–1408.

Glover, P.W.J., Hole, M.J. & Pous, J. (2000). A modified Archie's law. Geophysical Journal International, 142, 516–526.

Glover, P.W.J., Zadjali, I.I. & Frew, K.A. (2006). Permeability prediction. Geophysics, 71(4), F49–F60.

Glover, P.W.J. (2010). A generalized Archie's law for n phases. Solid Earth, 1, 85–91.

Hashin, Z. & Shtrikman, S. (1962). Variational approach to multiphase materials. Journal of Applied Physics, 33(10), 3125–3131.

Berryman, J.G. (1995). Mixture theories for rock properties. AGU Reference Shelf 3.

Sinmyo, R. & Keppler, H. (2017). Electrical conductivity of NaCl-bearing aqueous fluids. Contributions to Mineralogy and Petrology, 172, 4.

Guo, H. & Keppler, H. (2019). Electrical conductivity of NaCl–H₂O fluids. Journal of Geophysical Research: Solid Earth, 124, 1760–1771.

Sakuma, H. & Ichiki, M. (2016). Electrical conductivity of NaCl–H₂O fluid in the crust. Journal of Geophysical Research: Solid Earth, 121, 577–594.
