"""Top-level script for running heat transfer simulations."""

from os import getcwd
from os.path import join
import yaml

from mesh_setup import init_mesh
from simulate import simulate_2d
from plotter import animate_temp_2d

def run_simulation(config:str):
    """Runs a heat transfer simulation, then plots results."""

    sim_inputs = {}

    # load runtime settings and mesh definitions
    with open(join(getcwd(), config), 'r', encoding='utf-8') as cfg:
        cfg = yaml.load(cfg, Loader=yaml.SafeLoader)

    runtime = {}
    runtime.update({'tf':cfg['tf']})
    runtime.update({'dt_storage':cfg['dt_storage']})
    runtime.update({'theta':cfg['theta']})
    runtime.update({'max_courant':cfg['max_courant']})
    sim_inputs.update({'runtime':runtime})

    # initialise meshes
    if cfg.get('meshes') is not None:
        sim_inputs.update({'meshes':{}})
        for key, mesh in cfg['meshes'].items():
            sim_inputs['meshes'].update({key:init_mesh(mesh, cfg['force_finer'])})

    # initialise lumped capacitors

    # initialise environment
    sim_inputs.update({'environment':cfg['environment']})

    # sim_inputs = setup_2d(config)
    results = simulate_2d(inputs=sim_inputs)
    animate_temp_2d(results, save=False)



run_simulation('octoforge.yaml')
