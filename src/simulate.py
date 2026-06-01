"""Simulates multiple 2D meshes via linked boundary conditions."""

import numpy as np

from util import update_mesh, calc_edge_states, update_properties



def simulate_2d(inputs:dict) -> None:
    """Sets up the mesh definitions and handles simulation and plotting.
    
    The inputs dictionary must contain:
    - a dict of runtime settings (tf, dt_store, courant etc), named 'runtime'
    
    and any combination of:
    - a dict of meshes, named 'meshes'
    - a dict of lumped capacitors, named 'lumped_capacitors'
    - a dict of environmental conditions, named 'environment'

    This function returns a 'results' dict which contains:
    - whatever combination of objects fed into the simulation, with state arrays inside
    - the global time array, named 't'
    """

    t = np.array([0.0], float)
    t_now = 0.0

    runtime = inputs['runtime']
    meshes = inputs.get('meshes')
    # caps = inputs.get('lumped_capacitors')

    while t_now < runtime['tf']:
        # calculate next timestep to satisfy courant constraint
        dt = min(runtime['dt_storage'], min(runtime['max_courant']*min(m['dx'],\
                 m['dy'])**2 / np.max(m['diffusivity']) for m in meshes.values()))

        t_now += dt
        print("                                     ", end='\r')
        print(f"t = {t_now:0.3f} s, dt = {dt:0.6f} s", end='\r')
        store = t_now - t[-1] >= runtime['dt_storage']
        if store:
            t = np.hstack((t, t_now))

        for m in meshes.values():

            m['u_last'] = m['u_latest']
            update_properties(m)
            calc_edge_states(cfg=inputs)

            m['u_latest'] = update_mesh(mesh=m,
                                        dt=dt,
                                        curv=m['curvature'],
                                        theta=runtime['theta'])

            if store:
                m['u'] = np.dstack((m['u'], m['u_latest']))

    outputs = {}
    outputs.update({'meshes':meshes})
    outputs.update({'t':t})

    return outputs
