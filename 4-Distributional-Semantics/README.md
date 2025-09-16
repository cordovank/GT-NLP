# 4 — # NLP Project — Distributional Semantics (GloVe, CBOW, Skip-Gram)

[![Python](https://img.shields.io/badge/Python-3.10-informational)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
![Task](https://img.shields.io/badge/Task-Embeddings%20%7C%20Analogies%20%7C%20Retrieval-blue)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](../LICENSE)

Use and learn word vectors. Evaluate **analogies**, **nearest neighbors**, and **document retrieval**. Train **CBOW** and **Skip-Gram** from scratch, and compare to **pretrained GloVe**.

---

## Goals
- Build intuition for **distributional semantics** and embedding spaces.
- Reproduce classic evaluations: **nearest neighbors**, **word analogies**, and **doc retrieval**.
- Implement and train **CBOW/Skip-Gram** with negative sampling; compare to **GloVe**.

## Models & Techniques
- **CBOW:** predict center word from the average of context embeddings in a window.
- **Skip-Gram:** predict surrounding context words from the center word.
- **Negative Sampling:** logistic loss with `k` negatives per positive pair (e.g., `k=5`).
- **Window size:** typical `2–5`; **embedding dim:** `100–200` (configurable).
- **Optimization:** Adam/SGD; batched training with PyTorch `DataLoader`.

## Evaluations
- **Nearest Neighbors:** cosine similarity; sanity-check neighborhoods (`food → pizza, burger, …`).
- **Analogies:** classic “king − man + woman ≈ queen”; report top-k accuracy.
- **Document Retrieval:** mean of word vectors per document; cosine retrieval for query.

## Delivered
- Loaded pretrained GloVe vectors and ran analogy and document-retrieval baselines.
- Implemented CBOW and Skip-Gram with windowing, batching, and negative sampling in PyTorch.
- Trained embeddings; evaluated nearest neighbors, analogies (top-k), and cosine-based retrieval.
- Compared learned vs. pretrained vectors and documented effects of corpus size and domain.


## License
**All Rights Reserved.** No use, copying, modification, distribution, or model training is permitted without prior written permission.  
See the [LICENSE](./LICENSE) file for details.
