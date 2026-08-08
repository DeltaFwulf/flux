"""Tests for code performance evaluation."""

import cProfile
import pstats
from pstats import SortKey

#pylint: disable=W0611
from src.simulate import run_simulation



cProfile.run('run_simulation("octoforge.yaml")', 'stats')
p = pstats.Stats('stats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
