"""Functions for calculating different heat transfer modes."""

from copy import deepcopy
from math import log
import numpy as np
from pyfluids import Fluid, FluidsList, Input, HumidAir, InputHumidAir


def conduction(edge:dict, link:dict) -> np.ndarray:
    """Calculates common interface temperature between conducting faces."""

    na = link['k_bar'] / link['hn']
    nb = edge['k_bar'] / edge['hn']

    return (na*link['u_in'] + nb*edge['u_in']) / (na + nb)



def radiation(edge:dict, link:dict) -> np.ndarray:
    """Calculates thermal flux due to radiation between edge and link object"""
    q = 5.67e-8*((2*edge['emissivity'] - 1)*link['u4_mean'] -\
                link['emissivity']*edge['u']**4)

    return q



def convection(edge:dict, link:dict) -> np.ndarray:
    """Calculates thermal flux due to convection, with different cases."""

    # TODO: create fluid in each case, they have different reference temps
    # TODO: flip original sign of du, so that q is positive hk / l

    u_film = 0.5*(link['temperature'] + np.mean(edge['u']))
    beta = 1 / (u_film if link['phase'] != 'gas' else link['temperature'])
    fluid = calc_fluid(deepcopy(link) | {'temperature':u_film})
    du = link['temperature'] - edge['u']
    l = np.arange(0, (edge['indices'][1] + 1 - edge['indices'][0])*edge['hp'], edge['hp'])
    reynolds = abs(link['velocity']*l[-1] / fluid['kinematic_viscosity'])
    alpha = fluid['conductivity'] / (fluid['density']*fluid['specific_heat'])
    prandtl = fluid['kinematic_viscosity'] / alpha

    if reynolds > 100: # Forced convection, orientation invariant
        q = np.zeros_like(l, float)

    elif edge['direction'][1] == 0: # Natural convection, vertical
        l[0] = edge['hp']  # prevent undefined flux at l == 0
        rayleigh = 9.81*beta*np.abs(du)*l**3 / (fluid['kinematic_viscosity']*alpha)
        nusselt = 0.508*rayleigh**0.25 *(prandtl / (0.952 + prandtl))**0.25
        q = nusselt*du*fluid['conductivity'] / l

    elif sum(edge['direction']) < 0 == np.mean(du) < link['temperature']: # Horizontal, stable
        diameter = 2*l[-1]
        rayleigh = 9.81*beta*np.abs(du)*diameter**3 / (fluid['kinematic_viscosity']*alpha)
        nusselt = 0.82*rayleigh**0.2 *prandtl**(0.034)
        q = du*nusselt*fluid['conductivity'] / diameter

    else: # Horizontal, unstable
        if not 0.024 <= prandtl <= 2000:
            print("Warning, Prandtl number is outside of correlation domain.")

        # using the blending formula (Raithby and Hollands)
        l_star = np.sum(edge['areas']) / edge['perimeter']
        rayleigh = 9.81*beta*abs(np.mean(du))*l_star**3 / (fluid['kinematic_viscosity']*alpha)

        nu_turb = 0.14*rayleigh**(1 / 3) *(1 + 0.0107*prandtl) / (1 + 0.01*prandtl)
        nu_lam = 0.56*rayleigh**0.25 / ((1 + 0.492 / prandtl)**(9 / 16))**(4 / 9)
        nu_corrected = 1.4 / log(1 + 1.4 / nu_lam)

        q = du*((nu_corrected**10 + nu_turb**10)**0.1)*fluid['conductivity'] / l_star

    return q



def calc_fluid(params:dict):
    """Returns fluid properties as a dictionary using pyfluids."""

    # Temporary Fix: convert temperature to celsius
    params['temperature'] -= 273

    if params['name'] == 'humid_air':
        fluid = HumidAir().with_state(
                InputHumidAir.pressure(params['pressure']),
                InputHumidAir.temperature(params['temperature']),
                InputHumidAir.relative_humidity(0))

        return fluid.as_dict()

    fluid = Fluid(FluidsList[params['name']]).with_state(
            Input.pressure(params['pressure']),
            Input.temperature(params['temperature']))

    return fluid.as_dict()
