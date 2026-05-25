"""Functions for calculating different heat transfer modes."""

import numpy as np



def conduction(edge:dict, link:dict) -> np.ndarray:
    """Calculates thermal flux due to conduction between an edge and link object."""

    na = link['k_bar'] / link['h_norm']
    nb = edge['k_bar'] / edge['h_norm']

    u_int = (na*link['u_in'] + nb*edge['u_in']) / (na + nb)
    q = edge['k_bar']*(u_int - edge['u_in']) / edge['h_norm']

    return q



def radiation(edge:dict, link:dict) -> np.ndarray:
    """Calculates thermal flux due to radiation between edge and link object"""
    q = 5.67e-8*((2*edge['emissivity'] - 1)*link['u4_mean'] -\
                link['emissivity']*edge['u']**4)

    return q



def convection(edge:dict, link:dict, mode:str) -> np.ndarray:
    """Calculates thermal flux due to convection, with different cases."""
    q = np.zeros_like(edge['u'], float)
    return q
