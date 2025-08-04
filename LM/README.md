# EM-based Interpolation Weight Estimation

This repository contains scripts to estimate and apply interpolation weights for n-gram language models using the Expectation-Maximisation (EM) algorithm.

## 🧠 Overview

The scripts support several EM strategies:

- **Regularised EM** (single global weight vector)
- **Group-based EM** (per-group weights based on n-gram difficulty)
- **Hybrid EM** (interpolates global and group weights)

The objective is to optimise interpolation weights to minimise perplexity on development data and optionally evaluate them on held-out data.

---

## 📁 Input Format

- Each language model’s probability file: One probability per line (e.g., `dev/*.probs`, `eval/*.probs`)
- Development and evaluation sets must be aligned in position and size.

---

## 🚀 Estimate Weights

### My solution
Hybrid EM is the final approach which I decided to use for this task. As per the assignment description, run this by:
```bash
python estimate.py dev\*  weight.txt
python apply.py eval\*  weight.txt
```

### Other EM explored
There were other implementations explored during this practical. The results were included in the report to support my final decision with Hybrid EM approach so I have included these as well. 

To run these:
```bash
python <approach_name>_estimate.py dev\*  weight.txt
python <approach_name>_apply.py eval\*  weight.txt
```