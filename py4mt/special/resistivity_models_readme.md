# resistivity_models.py

Utilities for petrophysical conductivity, resistivity, permeability, and effective-medium calculations.

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

## Removed partial implementations

The following provisional or placeholder implementations were deliberately removed:

- Sakuma & Ichiki (2016) placeholder wrapper
- Manthilake-style generic Arrhenius helper

They were removed because they were only partial or proxy implementations rather than exact published model equations.

## High-P/T fluid models retained

Two explicit high-P/T NaCl-H2O fluid conductivity models remain:

- `brine_conductivity_sinmyo_keppler(...)`
- `brine_conductivity_guo_keppler(...)`

Both require the pure-water density at target P-T conditions as an input (`density_g_cm3`). This file does not embed a separate equation of state.

conda/pip install iapws

Example:

from iapws import IAPWS95

def water_density_iapws(T_c, P_MPa):
    w = IAPWS95(T=T_c + 273.15, P=P_MPa)
    return w.rho  # kg/m^3
    
    
## Main functions

### Brine / fluid functions

- `brine_conductivity_sen_goode(salinity_ppm, temp_c)`
- `brine_conductivity_sen_goode_mol(C, temp_c)`
- `brine_resistivity_sen_goode(salinity_ppm, temp_c)`
- `brine_resistivity(salinity_ppm, temp_c)`
- `brine_conductivity_sinmyo_keppler(salinity_wt_pct, temp_c, density_g_cm3, pressure_gpa=None)`
- `brine_conductivity_guo_keppler(salinity_wt_pct, temp_c, density_g_cm3, pressure_gpa=None)`

### Rock / bulk models

- `archie(phi, Sw, Rw, m=2.0, n=2.0, a=1.0)`
- `simandoux(phi, Sw, Rw, Vsh, Rsh, m=2.0, n=2.0, a=1.0)`
- `dual_porosity(phi_matrix, Sw_matrix, Rw, phi_frac, ...)`
- `solve_Sw_archie(...)`
- `solve_Sw_simandoux(...)`

### Permeability models

- `rgpz(phi, d_geom_um, m=2.0, a_pack=PACKING_SPHERICAL)`
- `rgpz_from_formation_factor(F, phi, d_geom_um, a_pack=PACKING_SPHERICAL)`
- `kozeny_carman(phi, d_geom_um, tortuosity=2.5)`

### Effective-medium / mixing models

- `glover_two_phase(...)`
- `glover_n_phase(...)`
- `hashin_shtrikman_two_phase(...)`
- `hashin_shtrikman_n_phase(...)`

## Units and conventions

- Conductivity in S/m
- Resistivity in ohm m
- Temperature in degree C on the public API unless noted otherwise
- Salinity for Sen-Goode functions in ppm
- Salinity for Sinmyo/Guo functions in wt%
- Density for Sinmyo/Guo functions in g/cm^3
- Grain size in micrometres
- Permeability returned internally in m^2 and commonly exposed also in mD

## Notes

- The Sinmyo and Guo functions validate their published temperature ranges and optional pressure ranges.
- Hashin-Shtrikman functions return bounds in both conductivity and resistivity form.
- The module includes a `__main__` demo block for quick sanity checks.
