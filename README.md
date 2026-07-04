# KG4ER-New

This repository contains the portable training, testing, evaluation, and ablation code for the SemanticConvE exercise recommendation model.

Current version: **V3 gated fusion**. Compared with V2, V3 adds semantic-quality priors and learnable type-aware gates for semantic, pedagogical, and learner-cluster feature fusion while keeping the existing recommendation scoring flow unchanged.

Data files are not included. To run experiments on a new computer, clone this repository and copy the prepared dataset folders into:

```text
data/
```

The expected dataset structure is documented in:

```text
docs-for-git/RUN_COMMANDS.md
```

The code is designed to preserve the existing `score(uid, rec, exercise)` experiment flow while adding gated semantic/pedagogical feature fusion, cognitive-factor ablations, and semantic/pedagogical model-component ablations.
