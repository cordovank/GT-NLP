# 5 — Key-Value Memory Networks (KVMemNet) for QA

[![Python](https://img.shields.io/badge/Python-3.10-informational)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
![Task](https://img.shields.io/badge/Task-QA%20%7C%20Attention%20over%20Facts-blue)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](../LICENSE)

Implement a Key-Value Memory Network for question answering by attending over an external dictionary of facts (keys/values) to retrieve answers. Focus on attention mechanics related to transformers while avoiding full transformer complexity.

## Goals
- Build a single-hop KVMemNet end-to-end for QA over structured facts.
- Learn batched dot-product attention and value aggregation.

## Core Idea
Embed question `q`, keys `K`, and values `V`. Compute attention
`p = softmax(q · K^T)`, then output `o = p V` as the aggregated value
representation; project to answer space.

## Architecture
- Two trainable projections:
  - **A** for embedding `q`, `k`, `v`.
  - **B** for projecting value representations for final similarity scoring.
- **Attention:** dot-product over keys, softmax to weights, weighted sum over values.
- **Batching:** use `torch.bmm` for attention and value aggregation.

## Data & Practical Constraints
- Select a compact subset of key-value pairs per question (entity-scoped slice + distractors) rather than the whole DB.
- For full-data runs: template questions (e.g., "When was [name] born?"), reduced relations/vocabulary, balanced distractors.

### This Run (documented)
- People: 500 train / 100 test.
- Relations: top-5.
- Templates: ~4 per relation.
- Distractors: 15 per question (mix of same-person other-relations + other-person facts).
- Vocab: ~2.8k tokens after reduction.
- Samples: ~8.7k train / ~1.7k test.

## Training & Evaluation
- **Loss:** CrossEntropy on the index of the correct value (after B-projection).
- **Optimizer:** AdamW (5e-4); batch=32; embedding dim=200; 20 epochs.
- **Results (this run):**
  - Train loss: ~2.49 → ~0.58.
  - Test loss: ~1.31 (epoch 7) rising to ~1.60 (epoch 20).
  - **Test accuracy peak:** **71.4%** (epoch 17), final **70.6%**.
- Insights: overfitting after ~7 epochs; improved retrieval via natural-language key formatting and balanced distractors.


## License
**All Rights Reserved.** No use, copying, modification, distribution, or model training is permitted without prior written permission.  
See the [LICENSE](./LICENSE) file for details.
