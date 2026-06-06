"""Common utility functions."""

from os import getcwd
from os.path import join
from math import pi
import numpy as np
import yaml



def get_material(name:str) -> dict:
    """Gets material properties from 'materials.yaml'."""

    with open(join(getcwd(), 'src', 'data', 'materials.yaml'), encoding='utf-8') as f:
        materials = yaml.load(stream=f, Loader=yaml.SafeLoader)

    mat = materials.get(name)
    if mat is None:
        print(f"No material found with name {name}.")
        raise ValueError

    return mat


# TODO: just accept temperature arrays or value and return the dict
#       don't need to give the whole dictionary, this adds weird
#       dependency on the dictionary structure.
def update_properties(solid:dict) -> None:
    """Updates a solid object's material properties (k, cp, rho) given a temperature."""

    u = solid['u_prev']
    mat = solid['material']

    k = np.interp(x=u, xp=mat['u'], fp=mat['k'])
    cp = np.interp(x=u, xp=mat['u'], fp=mat['cp'])
    rho = np.interp(x=u, xp=mat['u'], fp=mat['rho'])

    solid.update({'k':k, 'cp':cp, 'rho':rho})



def calc_bc_relations(solid:dict):
    """Returns a list of boundary condition indices relevant to each edge in a mesh."""

    edge_bcs = []
    for l in range(len(solid['edges'])):

        # iterate through all boundary conditions, add relevant entries to list
        relevant = [i for i, bc in enumerate(solid['boundary_conditions']) if bc['edge'] == l]
        edge_bcs.append(relevant)

    solid.update({'edge_bcs':edge_bcs})



def get_decimal_resolution(num) -> int:
    """Returns the number of significant trailing digits in a number."""

    if round(num) == num:
        return 0

    return len(str(num).split('.')[1])



def calc_face_perimeter(bounds:tuple, normal:tuple, curvature:int, depth:float=0.0) -> float:
    """Calculates a mesh edge face's perimeter."""

    # planar
    if curvature == 0:
        perimeter = 2*(bounds[1] - bounds[0] + depth)

    # curved, horizontal
    elif normal[0] == 0:
        perimeter = 2*pi*(bounds[0] + bounds[1])

    # curved, vertical
    else:
        perimeter = 4*pi*bounds[2]

    return perimeter
