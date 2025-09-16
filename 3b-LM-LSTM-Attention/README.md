# 3b — Language Modeling (LSTM + Attention)

[![Python](https://img.shields.io/badge/Python-3.10-informational)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
![Task](https://img.shields.io/badge/Task-Language%20Modeling%20%7C%20Attention-blue)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](../LICENSE)

Extend a basic RNN LM to (a) an LSTM LM using `nn.LSTM` and (b) a from-scratch `LSTMCell` LM, then add a seq2seq-style attention decoder over encoder states for more coherent generations.

## Goals
- Implement `nn.LSTM` LM and a custom `LSTMCell` LM.
- Add an encoder–decoder attention mechanism and compare decoding quality.

## Data
- Same normalized corpus and `<eos>` vocabulary as HW3a.
- Create sequential pairs and standard train/valid/test splits.

## Models & Techniques
- **LSTM LM (easy path):** Embedding/one-hot → `nn.LSTM` → Linear → log_softmax.
- **LSTM LM (from scratch):** Implement gate mechanics:
  - Input/forget/output gates `i, f, o = σ(...)`, candidate cell `ĉ = tanh(...)`.
  - State updates: `c' = f ⊙ c + i ⊙ ĉ`, `h' = o ⊙ tanh(c')`.
- **Attention decoder (encoder–decoder LM):**
  - Encoder runs `LSTMCell` to collect hidden states.
  - Decoder computes attention scores → weights, forms a context vector (e.g., via `bmm` over encoder states), concatenates with input, steps its own `LSTMCell`, then projects to vocab with log_softmax.
- **Decoding:** greedy and temperature sampling.

## Delivered
- Implemented an LSTM LM using `nn.LSTM`; added greedy and temperature-controlled sampling.
- Built a from-scratch `LSTMCell` LM (gates + state updates) and trained to target perplexity.
- Added an encoder–decoder attention module: dot-product scores, softmax weights, context vector via `bmm`, and fused decoding.
- Trained the attention model and compared decoding quality with worked examples.
- Included prompt continuation utilities and runtime-conscious configs for notebook execution.


## License
**All Rights Reserved.** No use, copying, modification, distribution, or model training is permitted without prior written permission.  
See the [LICENSE](./LICENSE) file for details.
