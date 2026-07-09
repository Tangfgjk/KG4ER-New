# KG4ER-New

This repository contains the portable training, testing, evaluation, and ablation code for the SemanticConvE exercise recommendation model.

Current branch: **V9**. V9 uses the V8 no-Q MIRT preprocessing outputs, keeps the raw ConvE recommendation score as the final score, and defines a compact ConvE ablation suite:

```text
full
hybrid_relation
id_only
relation_id_only
no_type_aware_scoring
```

Data files are not included. To run experiments on a new computer, clone this repository and copy the prepared dataset folders into:

```text
data/
```

The expected dataset structure is documented in:

```text
docs-for-git/RUN_COMMANDS.md
```

The code is designed to preserve the existing `score(uid, rec, exercise)` experiment flow while adding semantic/pedagogical feature fusion, MIRT-derived educational features, relation-aware encoding, and type-aware exercise scoring.
