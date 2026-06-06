"""2D heat transfer simulation script."""

from os import getcwd
from os.path import join
import yaml
import numpy as np

from .lumped_capacitor import init_lc
from .mesh import create_mesh, update_mesh, calc_edge_states
from .util import update_properties



def run_simulation(config:str):
    """Runs a 2D heat transfer simulation, then plots results.
    
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

    # load runtime settings and simulation variables
    with open(join(getcwd(), 'src', 'data', 'examples', config), 'r', encoding='utf-8') as cfg:
        cfg = yaml.load(cfg, Loader=yaml.SafeLoader)

    runtime = {}
    runtime.update({'tf':cfg['tf']})
    runtime.update({'dt_storage':cfg['dt_storage']})
    runtime.update({'dt_max':cfg['dt_max']})
    runtime.update({'theta':cfg['theta']})
    runtime.update({'max_courant':cfg['max_courant']})

    if runtime['dt_max'] > runtime['dt_storage']:
        print("ensure that dt_max <= dt_storage")
        return

    config = {'runtime':runtime}

    meshes = {}
    if cfg.get('meshes') is not None:
        for key, mesh in cfg['meshes'].items():
            meshes.update({key:create_mesh(mesh, cfg['force_finer'])})
        config.update({'meshes':meshes})

    lumped_capacitors = {}
    if cfg.get('lumped_capacitors') is not None:
        for key, lc in cfg['lumped_capacitors'].items():
            lumped_capacitors.update({key:init_lc(lc)})
        config.update({'lumped_capacitors':lumped_capacitors})

    environment = {}
    if cfg.get('environment') is not None:
        environment = cfg['environment']
        config.update({'environment':environment})

    # Run Simulation -----------------------------------------------------------------------------#
    #                                                                                             #
    # --------------------------------------------------------------------------------------------#

    t = np.zeros(1, float)
    t_now = t[-1]
    energy_change = 0.0

    while t_now < runtime['tf']:

        # calculate timestep
        dt = min(runtime['dt_max'], min(runtime['max_courant']*min(m['dx'], m['dy'])**2\
                 / np.max(m['k'] / (m['rho']*m['cp'])) for m in meshes.values()))

        t_now += dt
        print("                                     ", end='\r')
        print(f"t = {t_now:0.3f} s, dt = {dt:0.6f} s", end='\r')
        store = t_now - t[-1] >= runtime['dt_storage']
        if store:
            t = np.hstack((t, t_now))

        for m in meshes.values():
            m['u_prev'] = m['u_latest']

            update_properties(m)
            calc_edge_states(cfg=config)

            m['u_latest'] = update_mesh(mesh=m,
                                        dt=dt,
                                        curv=m['curvature'],
                                        theta=runtime['theta'])

            energy_change += dt*sum(m['edge_powers_latest'])

            if store:
                m['u'] = np.dstack((m['u'], m['u_latest']))

                for e in range(len(m['edges'])):
                    m['edge_fluxes'][e] = np.append(m['edge_fluxes'][e],m['edge_fluxes_latest'][e])
                    m['edge_powers'][e] = np.append(m['edge_powers'][e],m['edge_powers_latest'][e])

                m['net_energy'] = np.append(m['net_energy'], energy_change)

    results = config
    results.update({'t':t})
    return results
