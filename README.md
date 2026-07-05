# KG4ER-New

This repository contains the portable training, testing, evaluation, and ablation code for the SemanticConvE exercise recommendation model.

Current version: **V3.1 masked gated fusion**. Compared with V3, V3.1 keeps semantic-quality priors and learnable type-aware gates, but adds stricter type masks so learner-cluster features only affect learner entities and pedagogical numeric features only affect learner/exercise entities. It also scales count features into `[0, 1]` and initializes auxiliary-feature gates conservatively.

Data files are not included. To run experiments on a new computer, clone this repository and copy the prepared dataset folders into:

```text
data/
```

The expected dataset structure is documented in:

```text
docs-for-git/RUN_COMMANDS.md
```

The code is designed to preserve the existing `score(uid, rec, exercise)` experiment flow while adding gated semantic/pedagogical feature fusion, cognitive-factor ablations, and semantic/pedagogical model-component ablations.
