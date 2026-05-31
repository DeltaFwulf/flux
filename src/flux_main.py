"""Top-level script for running heat transfer simulations."""

from setup import setup_2d
from mesh_2d import simulate_2d
from plotter import plot2d_flat
from plotter import animate_temp_2d

def run_simulation(config:str):
    """Runs a heat transfer simulation, then plots results."""

    sim_inputs = setup_2d(config)
    results = simulate_2d(inputs=sim_inputs)
    # animate_temp_2d(results)
    plot2d_flat(results)



run_simulation('octoforge.yaml')
