# KG4ER-New

This repository contains the portable training, testing, evaluation, and ablation code for the SemanticConvE exercise recommendation model.

Current version: **V4 fine-grained ablation**. Compared with V3.1, V4 keeps masked gated fusion and adds fine-grained ablations for concept semantics, exercise semantics, exercise IRT features, learner IRT features, learner clusters, and relation representation variants (`discrete_relation` and `hybrid_relation`). The one-command runner supports `--ablations all` to run the full model and all ablations together.

Data files are not included. To run experiments on a new computer, clone this repository and copy the prepared dataset folders into:

```text
data/
```

The expected dataset structure is documented in:

```text
docs-for-git/RUN_COMMANDS.md
```

The code is designed to preserve the existing `score(uid, rec, exercise)` experiment flow while adding gated semantic/pedagogical feature fusion, cognitive-factor ablations, fine-grained feature ablations, and relation-representation ablations.
