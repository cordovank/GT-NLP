# 1 — Introduction to Neural Networks with PyTorch

[![Python](https://img.shields.io/badge/Python-3.10-informational)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![Task](https://img.shields.io/badge/Task-Regression%20%7C%20MLP-blue)]()
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](../LICENSE)

A warm-up assignment to implement, train, and evaluate a small feed-forward neural network in PyTorch on a synthetic supervised task (self-driving style: predict 2 continuous controls from 4 proximity sensors).

## Goals
- Build an end-to-end MLP in PyTorch (tensors, batching, training loop).
- Understand autograd, gradients, and parameter updates.
- Evaluate using a proxy safety metric (crash threshold).

## Data
- `make_data(num_data)` simulates:
  - Inputs: 4 normalized sensors (front, back, left, right).
  - Targets: 2 controls in [-1, 1] (accelerate, turn) derived from simple rules.
- Batching targets: `X` shape `(B, 4)`, `Y` shape `(B, 2)`.

## Model & Techniques
- **Architecture:** CarNet with tanh activations, linear layers `4 → 16 → 8 → 2`, outputs constrained to `[-1, 1]`.
- **Optimization:** Adam + MSELoss.
- **Loop:** zero_grad → forward → loss → backward → step.
- Optional: inspect weights/gradients via autograd; torchviz graph.

## Evaluation
- **Crash metric:** if either predicted accel/turn differs from truth by > 0.1 (absolute), count a crash.
- **Accuracy:** `1 − crashes / N` on a hold-out test set.

## Delivered
- Implemented `get_batch` that returns correctly shaped tensors `(X: B×4, Y: B×2)`.
- Built `CarNet` (`4 → 16 → 8 → 2`) with `tanh` activations; outputs constrained to `[-1, 1]`.
- Wrote full training loop (Adam + MSELoss); included loss curve and basic diagnostics.
- Verified parameters/gradients via autograd; optional graph visualization hooks.
- Evaluated with the “crash” metric and achieved 100% accuracy on provided test runs.


## License
**All Rights Reserved.** No use, copying, modification, distribution, or model training is permitted without prior written permission.  
See the [LICENSE](./LICENSE) file for details.
