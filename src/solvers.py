"""Mesh utility functions."""

from math import copysign
import numpy as np



def tdma(x, a, b, c, d) -> np.ndarray:
    """Solves for x given a tridiagonal matrix, following Thomas' Algorithm."""

    # forward substitution
    p = np.zeros_like(x, float)
    q = np.zeros_like(x, float)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, x.size):
        dn = b[i] + a[i]*p[i - 1]
        p[i] = - c[i] / dn
        q[i] = (d[i] - a[i]*q[i - 1]) / dn

    # back substitution for x
    x[-1] = q[-1]
    for i in range(x.size - 2, -1, -1):
        x[i] = p[i]*x[i + 1] + q[i]

    return x



def find_regions_2d(p:str, ind_n:int, ind_p:np.ndarray, lines:list) -> list[list]:
    """
    Uses a transition method to identify all regions in this slice.

    Regions are either 'i', internal or 'b', bounding.

    Internal regions are solved for in the TDMA steps.
    Bounding regions have boundary conditions applied in the BC step, in the correct direction.
    
    boundary lines have a direction, 'l' or 'r' in which to apply the boundary.
    'l' is descending index, 'r' is ascending index.
    
    ind_n is the slice's index in the normal axis
    ind_p is the array of indices along the parallel direction

    each region has a type and bounding points. If edge it has a direction and boundary number
    {'type':'internal', 'bounds':(pa, pb)}
    {'type':'edge', 'bounds':(pa, pb), 'direction':1, 'bc_index':5}
    direction is +1 for ascending order, -1 for descending. (towards next internal region)
    """

    a = 0 if p == 'x' else 1 # gives line coordinate to inspect for normality
    transitions = []
    m = 0

    for k in ind_p:

        for l, line in enumerate(lines):

            is_normal = (line[0][a] == line[1][a]) and k in (line[0][a], line[1][a])

            da = line[0][1 - a] - ind_n
            db = line[1][1 - a] - ind_n

            spans = copysign(1, da) != copysign(1, db) or ind_n in (line[0][1 - a], line[1][1 - a])

            if is_normal and spans:

                direction = -1

                if da*db != 0:
                    direction = 2
                elif da > ind_n or db > ind_n:
                    direction = 1

                if m > 1 and transitions[-1][3] == transitions[-2][3] == 2:
                    m = 0

                transitions.append([k, l, m, direction])
                m += 1
                break

    regions = []
    for i, transition in enumerate(transitions):  # construct regions from transition array

        if transition[2] == 0:
            continue

        pa = (transitions[i-1][0], ind_n) if a == 0 else (ind_n, transitions[i-1][0])
        pb = (transition[0], ind_n) if a == 0 else (ind_n, transition[0])
        reg = {'bounds':(pa[a], pb[a])}

        if 2 in (transition[-1], transitions[i-1][-1]) or\
                (len(regions) > 0 and regions[-1]['type'] == 'edge' and m > 1):

            reg.update({'type':'internal', 'bc_s':transitions[i-1][1], 'bc_e':transition[1]})
            regions.append(reg)
            continue

        d = transitions[i - 1][3] if transition[2] == 1 else -transitions[i - 1][3]

        for l, line in enumerate(lines):  # find the boundary condition (contains both pa, pb)
            if pa in line and pb in line:
                break

        reg.update({'type':'edge', 'direction':d, 'bc':l})
        regions.append(reg)

    return regions
