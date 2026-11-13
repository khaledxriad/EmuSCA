# EmuSCA: Emulated Side-Channel Evaluation of Masked and Shuffled Kyber on Arm Cortex-M

EmuSCA (Emulated Side-Channel Analysis) is the code and data package that accompanies the article
"EmuSCA: Emulated Side-Channel Evaluation of Masked and Shuffled Kyber on Arm Cortex-M" (Khaled Riad,
King Faisal University). It emulates the data-dependent leakage of a 32-bit Arm Cortex-M core while a
bit-exact Kyber/ML-KEM arithmetic model runs, and compares unprotected, shuffled, masked, and
masked-and-shuffled implementations of the number-theoretic transform (NTT) and the coefficient
multiplication under identical conditions.

**Important scope note.** Everything here is an *emulation*. No hardware was measured. The cycle counts
come from an analytical Cortex-M4 cost model (assumptions are listed in `code/e6_cost.py`), not from a
device. Masked results count recovery of the secret coefficient up to its sign unless stated otherwise
(see `data/e8_sign.csv` and `data/e7_strict_*.csv` for exact-key success).

Repository links (to be filled in before publication):
GitHub: [GitHub Link] | Zenodo archive: [Zenodo URL]

## Layout

| Path | Content |
|---|---|
| `code/lbsca.py`, `code/engine.py`, `code/ntt_emul.py` | Kyber arithmetic (signed Montgomery), leakage model (HW + 0.25 HD + Gaussian noise), CPA / second-order CPA / TVLA engines, 896-butterfly NTT emulator |
| `code/e1_unprotected.py` | Experiment 1: first-order CPA on unprotected code vs. noise |
| `code/e2_shuffle.py` | Experiment 2: shuffling, fixed-slot and window-integrating attackers |
| `code/e3_masked.py`, `code/e3c_masked_shuffled_s2.py` | Experiment 3: second-order attack on masked code, masked + shuffled |
| `code/e4_tvla.py` | Experiment 4: TVLA versus number of traces and per NTT layer |
| `code/e5_validation.py` | Experiment 5: numerical check of the closed-form predictions |
| `code/e6_cost.py` | Experiment 6: analytical Cortex-M4 cycle model |
| `code/e7_strict.py`, `code/e8_sign.py` | Strict (exact key) vs. sign-tolerant success |
| `code/make_figs.py` | Generates all result charts as PDF files in `figures/` from `data/` |
| `data/` | All raw result CSV files, run logs, and `chart_data.json` (the data behind every chart) |
| `figures/` | Result charts (PDF) as used in the article |

## Requirements

Python 3.10 or newer with NumPy, SciPy, and Matplotlib (`pip install -r requirements.txt`).
No GPU and no special hardware are needed.

## Reproducing the results

All scripts use fixed random seeds, so a re-run reproduces the CSV files in `data/`.
Run every script from inside `code/`:

```bash
cd code
bash ../run_all.sh          # all experiments, then the charts (long: see below)
# or one at a time:
python3 e6_cost.py          # seconds
python3 e5_validation.py    # about a minute
python3 e1_unprotected.py 60
python3 e2_shuffle.py 30
python3 e3_masked.py 30
python3 e3c_masked_shuffled_s2.py 30
python3 e4_tvla.py
python3 e7_strict.py e1 ; python3 e7_strict.py e3
python3 e8_sign.py
python3 make_figs.py        # writes ../figures/*.pdf from ../data/*.csv
```

The numeric argument of `e1`, `e2`, `e3`, and `e3c` is the number of independent trials per point.
The experiments with masking and large shuffling windows are the slowest (minutes to tens of
minutes each on a laptop CPU; see the `time` columns in `data/log_*.txt`).

## Data dictionary

* `e1_unprotected_*.csv`: success rate (`_sr`) and traces for 90% success (`_mtd`, column `MTD90`) of first-order CPA, noise sigma = 1, 2, 4, 8, 16.
* `e2_shuffle_*.csv`: shuffling windows w = 1, 2, 4, 8, 16, fixed-slot (`MTD_fixed`) and integrated (`MTD_integrated`) attackers at sigma = 2.
* `e3a_masked_*.csv`, `e3b_*`, `e3c_*`: second-order attack on masked code, with and without shuffling. `inf` means that 90% success was not reached within the largest trace budget.
* `e4_tvla_vs_N.csv`, `e4_tvla_layers.csv`: maximum |t| of the fixed-versus-random Welch t-test (threshold 4.5) versus traces per class and per NTT layer.
* `e5_validation.csv`: closed-form prediction versus emulation.
* `e6_cost.csv`: analytical cycle counts per configuration (unprotected base: 20,352 cycles).
* `e7_strict_*.csv`, `e8_sign.csv`: exact-key versus sign-tolerant success.

## Citation

Please cite the article and, once available, the Zenodo archive (see `CITATION.cff`).

## License

MIT License (see `LICENSE`). Copyright (c) 2026 Khaled Riad.
