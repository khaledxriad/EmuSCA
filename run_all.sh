#!/usr/bin/env bash
# Run from the code/ directory: bash ../run_all.sh
set -euo pipefail
python3 e6_cost.py
python3 e5_validation.py
python3 e1_unprotected.py 60
python3 e2_shuffle.py 30
python3 e3_masked.py 30
python3 e3c_masked_shuffled_s2.py 30
python3 e4_tvla.py
python3 e7_strict.py e1
python3 e7_strict.py e3
python3 e8_sign.py
python3 make_figs.py
