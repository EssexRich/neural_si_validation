# Neural Structural Intelligence — Reproducibility Package

Reproduction code and data for the paper:

**Structural Topology Monitoring Predicts and Steers Neural Network Phase Transitions**  
Richard Benfield, 2026.

---

## Overview

Five sequential experiments on toy tasks (modular arithmetic, simple sequence prediction) and small architectures (128–512 unit MLPs, 40K-parameter transformer), demonstrating that the Fiedler value (second-smallest eigenvalue of the weight graph Laplacian), combined with Scheffer critical slowing down indicators, provides a predictive signal for phase transitions during training that precedes the loss function. Results are specific to the tasks and architectures tested; generalisation to production-scale training is unvalidated.

---

## Experiments

| Script | Experiment | Key result |
|---|---|---|
| `poc_grokking_spectral.py` | Step 1: Detection | 21,000-step lead time before grokking |
| `poc_classification.py` | Step 2: Classification | λ₂ rate 3.7× faster for forgetting vs grokking |
| `poc_steering.py` | Step 3: Steering | 91.7% knowledge retained vs 2.6% unsteered |
| `poc_compounding_v6_multihead.py` | Step 4: Compounding | 100%/100%/97.5% on three tasks, 48× acceleration |
| `poc_step5_transformer.py` | Step 5: Preemptive curriculum | 100% preserved (bridged) vs 0% (direct) |

All experiments run on CPU. Total compute: under 24 hours on a consumer laptop (Intel i7).

---

## Requirements

```
torch>=2.0
numpy
scipy
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Running the experiments

Each script is self-contained. Run in order (each builds on checkpoints from the previous):

```bash
python poc_grokking_spectral.py       # ~2 hours, saves results/ and checkpoint
python poc_classification.py          # ~1 hour, loads Phase 1 checkpoint
python poc_steering.py                # ~1 hour, loads Phase 1 checkpoint
python poc_compounding_v6_multihead.py  # ~30 mins, warm-starts from Step 1 checkpoint
python poc_step5_transformer.py       # ~2 hours, trains transformer from scratch
```

Results (CSV + PNG) are saved to `results/`.

---

## Pre-computed results

The `results/` directory contains all plots and CSVs from the paper runs. Checkpoints are not included due to size but are reproducible from the scripts above.

---

## Citation

```bibtex
@article{benfield2026nsi,
  title={Structural Topology Monitoring Predicts and Steers Neural Network Phase Transitions},
  author={Benfield, Richard},
  year={2026},
  note={arXiv preprint}
}
```

---

Patent application GB2611542.8 filed 18 May 2026 (pending). Copyright © 2025-2026 Richard Benfield. Code released for academic reproducibility.
