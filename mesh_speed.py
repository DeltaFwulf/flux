"""Tests for code performance evaluation."""

import cProfile
import pstats
from pstats import SortKey

from src.simulate import run_simulation



cProfile.run('run_simulation("pipe.yaml")', 'stats')
p = pstats.Stats('stats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)