# KG4ER-New V-Fin

This branch contains the current end-to-end experiment workflow for
SemanticConvE exercise recommendation:

1. Canonical raw datasets in `Data_Fin/<dataset>/raw/`.
2. Front-feature generation: Q-constrained MIRT, EKTM-MIRT, forgetting
   features, semantic features, and ER-graph building.
3. SemanticConvE full and ablation experiments.
4. ID-only KGE and traditional recommendation baselines.

`Data_Fin/` is versioned on this branch. It contains all five canonical raw
datasets, plus any already generated front features and ER graphs. Training
checkpoints remain intentionally excluded.

The V11 compatibility pipeline is retained because the current EKTM-MIRT
front trainer imports its model definition from it. The retired V10 pipeline
and historical command documents are not part of this branch.

Complete setup, generation, training, resumption, and result-summary commands
are in [docs/V-Fin-全流程运行命令.md](docs/V-Fin-%E5%85%A8%E6%B5%81%E7%A8%8B%E8%BF%90%E8%A1%8C%E5%91%BD%E4%BB%A4.md).
