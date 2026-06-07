"""Test script for running the flux library."""

from src.simulate import run_simulation
from src.plotter import animate_temp_2d, plot_total_powers, plot_steady_slice



results = run_simulation("octoforge.yaml")
animate_temp_2d(results, save=True)
plot_total_powers(results)
# plot_steady_slice(results)
