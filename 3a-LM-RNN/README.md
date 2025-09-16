# 3a — Language Modeling (RNN)

[![Python](https://img.shields.io/badge/Python-3.10-informational)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
![Task](https://img.shields.io/badge/Task-Language%20Modeling-blue)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](../LICENSE)

Build a token-level language model that predicts the next word given context, then generate text via greedy and temperature-scaled sampling. Emphasis on LM plumbing: vocab, data prep, a simple RNN cell, loss, and decoding.

## Goals
- Implement a minimal RNN-based LM with correct tensor shapes and training loop.
- Track loss and report perplexity; implement multiple decoding strategies.

## Data
- A normalized text corpus, tokenized to a vocabulary (includes `<eos>`).
- Create sequential (input, target) pairs where target = next token.

## Model & Techniques
- **Vocabulary** with word2index/index2word; frequency-based indexing.
- **Inputs:** one-hot vectors over the vocabulary.
- **RNN cell:** concat(one-hot x, prev h) → Linear → sigmoid → Linear → log_softmax (token scores).
- **Training:** NLL/CrossEntropy with Adam; iterate over corpus.
- **Decoding:** greedy and stochastic sampling with temperature scaling.

## Delivered
- Implemented text normalization, tokenization, and a `<eos>`-aware vocabulary; built corpus tensors.
- Converted sequences into one-hot inputs and next-token targets with helper utilities.
- Built a minimal RNN LM (custom cell + projection to vocab) with verified forward/gradient flow.
- Trained with Adam; tracked decreasing loss and reported perplexity on validation/test.
- Implemented greedy and temperature-scaled sampling; generated prompt-conditioned continuations.


## License
**All Rights Reserved.** No use, copying, modification, distribution, or model training is permitted without prior written permission.  
See the [LICENSE](./LICENSE) file for details.
