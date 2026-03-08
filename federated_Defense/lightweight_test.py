#!/usr/bin/env python3
"""
Lightweight CD-CFL experiments — thin wrapper around run_cd_cfl_experiments.py.

Same experiments (E1-E7) with fewer rounds (100) and more data (50%)
for quick validation. Drops krum/median/trimmed_mean baselines.

Usage:
    python lightweight_test.py                          # All E1-E7
    python lightweight_test.py -d femnist               # One dataset
    python lightweight_test.py -d femnist -a gradient_scaling
    python lightweight_test.py -e E3                    # E3 only
    python lightweight_test.py -e E2,E3 -d femnist      # E2+E3 on FEMNIST
"""

import run_cdcfl_experiments as exp

# Lightweight overrides
exp.ROUNDS = {'FEMNIST': 100, 'Shakespeare': 100, 'Sentiment140': 100}
exp.DATASET_FRACTION = 0.50
exp.OUTPUT_PREFIX = 'cdcfl_lightweight'

# Fewer baselines (drop krum, multi_krum, median, trimmed_mean)
exp.E2_METHODS = ['fedavg', 'cmfl', 'cmfl_ii', 'cdcfl_i', 'cdcfl_ii']

if __name__ == '__main__':
    exp.main()
