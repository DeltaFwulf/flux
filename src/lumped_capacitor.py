"""Contains functions for initialising lumped capacitors."""

from copy import deepcopy
from math import pi, copysign

from util import get_material, update_properties, calc_bc_relations



def init_lc(lc_def:dict) -> dict:
    """Initialises a lumped capacitor for use in a simulation."""

    lc = deepcopy(lc_def)

    lc.update({'material':get_material(lc_def['material'])})
    lc.update({'u':lc_def['u0']})
    lc.update({'u_latest':lc_def['u0']})
    lc.update({'u_last':lc_def['u0']})

    find_edges(lc)
    update_properties(lc)
    calc_bc_relations(lc)

    return lc



def find_edges(lc:dict) -> None:
    """Initialises capacitor edges. 
    
    Running this function updates the lumped-capacitor
    given to the function directly, returning nothing.
    """

    edges = []

    for l, line in enumerate(lc['lines']):

        edge = {}

        # edge direction
        p = 0 if line[0][1] == line[1][1] else 1
        n = 1 - p

        # how many parallel overlapping lines to either side?
        # inside is the direction with an odd number
        # only one side can have odd number, only store one direction
        lefts = 0
        direction = [0, 0]
        for lb, line_b in enumerate(lc['lines']):
            if line_b[0][p] == line_b[1][p] or lb == l: # line_b is normal or same, skip
                continue

            da = line_b[0][p] - line[0][p]
            db = line_b[1][p] - line[0][p]
            spans = da*db == 0 or copysign(1, da) != copysign(db)
            if spans and line_b[0][n] < line[0][n]:
                lefts += 1

        direction[n] = 1 if (lefts % 2 == 0) else -1
        edge.update({'direction':direction})

        # edge length
        length = abs(line[1][p] - line[0][p])

        if lc['curvature'] == 0:
            area = length*lc['depth']
            perimeter = 2*(length + lc['depth'])

        elif p == 0: # edge parallel to x
            area = pi*abs(line[1][p]**2 - line[0][p]**2)
            perimeter = 2*pi*(line[0][p] + line[1][p])
        else:
            area = length*2*pi*line[0][n]
            perimeter = 4*pi*(line[0][n])

        edge.update({'length':abs(line[1][p] - line[0][p])})
        edge.update({'area':area})
        edge.update({'perimeter':perimeter})

        edges.append(edge)

    lc.update({'edges':edges})
