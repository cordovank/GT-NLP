# 2 — Text Classification (IMDb Sentiment & AG News Topics)

[![Python](https://img.shields.io/badge/Python-3.10-informational)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
![Tasks](https://img.shields.io/badge/Tasks-Sentiment%20%7C%20Topic%20Classification-blue)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](../LICENSE)

End-to-end text classifiers for **IMDb** (binary sentiment) and **AG News** (4-class topics). Compare classical probabilistic baselines with neural, embedding-based models and study how preprocessing/representation affects accuracy.

## Goals
- Build full pipelines: cleaning → tokenization → vectorization → modeling → evaluation.
- Compare **BOW + Naive Bayes/Logistic** vs. **GloVe-based neural** classifiers.

## Datasets
- **IMDb** (pos/neg), **AG News** (4 topics).
- Loaded via the `datasets` library (Hugging Face) with train/valid/test splits.

## Preprocessing & Representations
- Remove HTML, lowercase, tokenize, optional stopwords/stemming; build vocabulary.
- **BOW counts** and **pretrained GloVe (100-d)** embeddings (average or shallow NN over sequences).

## Models & Training
- **Naive Bayes** (Laplace smoothing) on BOW.
- **Logistic regression** (IMDb): `Linear(|V| → 1)` + sigmoid + BCELoss.
- **Multiclass softmax** (AG News): `Linear(|V| → 4)` + CrossEntropyLoss.
- Optimizers: SGD/Adam; mini-batch training with gradient updates.

## Evaluation
- Accuracy and classification report (precision/recall/F1), with train/valid curves.
- IMDb: sigmoid thresholding for probability outputs.

## Delivered
- Implemented complete preprocessing and vectorization: cleaning/tokenization, vocabulary, BOW, and batching (`X: B×|V|`).
- Built Naive Bayes baseline with Laplace smoothing; produced classification reports (IMDb macro-F1 ≈ 0.80+ in provided runs).
- Trained logistic regression for IMDb; improved over NB baseline.
- Implemented multiclass softmax classifier for AG News; reported accuracy and macro-F1.
- Added embedding-based classifier using pretrained GloVe (100-d) with a shallow NN head; observed better generalization on noisier samples.
- Logged train/validation curves and summarized trade-offs across models.


## License
**All Rights Reserved.** No use, copying, modification, distribution, or model training is permitted without prior written permission.  
See the [LICENSE](./LICENSE) file for details.
