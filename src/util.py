"""Common utility functions."""

from os import getcwd
from os.path import join
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



def update_properties(solid:dict) -> None:
    """Updates a solid object's material properties (k, cp, rho) given a temperature."""

    u = solid['u_last']
    mat = solid['material']

    k = np.interp(x=u, xp=mat['u'], fp=mat['k'])
    cp = np.interp(x=u, xp=mat['u'], fp=mat['cp'])
    rho = np.interp(x=u, xp=mat['u'], fp=mat['rho'])
    alpha = k / (rho*cp)

    solid.update({'k':k, 'cp':cp, 'rho':rho, 'diffusivity':alpha})



def calc_bc_relations(solid:dict):
    """Returns a list of boundary condition indices relevant to each edge in a mesh."""

    edge_bcs = []
    for l in range(len(solid['edges'])):

        # iterate through all boundary conditions, add relevant entries to list
        relevant = [i for i, bc in enumerate(solid['boundary_conditions']) if bc['edge'] == l]
        edge_bcs.append(relevant)

    solid.update({'edge_bcs':edge_bcs})
