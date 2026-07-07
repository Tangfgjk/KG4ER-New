# KG4ER-New

This repository contains the portable training, testing, evaluation, and ablation code for the SemanticConvE exercise recommendation model.

Current version: **V6 state-aware continuous-relation SemanticConvE**.

V6 adds:

- state-aware learner encoder: learner entities are represented by cognitive state features rather than a learned `uid` ID embedding;
- continuous relation encoding: relations use `relation type + continuous strength` instead of relation ID embedding in the full model;
- statistical pedagogical features for comparing IRT-based and statistics-based educational signals;
- evaluation separation: `test_triples.txt` is used for evaluation by default, not for training;
- an 8-experiment V6 suite through `--ablations all`.

Data files are not included. To run experiments on a new computer, clone this repository and copy the prepared dataset folders into:

```text
data/
```

The expected dataset structure is documented in:

```text
docs-for-git/SemanticConvE-V6运行命令.md
```

The code is designed to support `score(StateEncoder(uid_state), rec, exercise)` while keeping the existing KG4ER graph files and experiment runner interfaces.
