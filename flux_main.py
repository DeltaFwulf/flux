"""Test script for running the flux library."""

from src.simulate import run_simulation
from src.plotter import animate_temp_2d, plot_total_powers



results = run_simulation("hotshoe.yaml")
animate_temp_2d(results, save=False)
plot_total_powers(results)
