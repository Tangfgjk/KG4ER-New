# KG4ER-New Code Overview

This document introduces the code uploaded to the `KG4ER-New` repository. The repository contains only the code required for SemanticConvE training, testing, evaluation, result summarization, and ablation experiments. Dataset files and generated results are intentionally excluded.

## 1. Repository Structure

```text
KG4ER-New/
  codes-New-ConvE/
    feature_loader.py
    semantic_conve_model.py
    run_semantic_conve.py
    test_semantic_conve.py
    run_semantic_experiments.py
    summarize_semantic_results.py
    validate_semantic_ready.py
    semantic_ablation_data.py
    evaluate_recommendations.py
    ep_sim.py
    experiment_utils.py
  docs-for-git/
    CODE_OVERVIEW.md
    RUN_COMMANDS.md
  requirements.txt
  README.md
  .gitignore
```

## 2. Core Scripts

| File | Purpose |
| --- | --- |
| `feature_loader.py` | Loads entity IDs, relation IDs, semantic text embeddings, pedagogical numeric features, entity types, learner clusters, relation types, and continuous relation strengths. |
| `semantic_conve_model.py` | Defines SemanticConvE with semantic-aware entity encoding, pedagogical feature fusion, continuous relation-aware encoding, and type-aware tail scoring. |
| `run_semantic_conve.py` | Trains one SemanticConvE model for one dataset, one seed, and one ablation setting. |
| `test_semantic_conve.py` | Loads a trained checkpoint and exports learner-exercise recommendation scores. |
| `run_semantic_experiments.py` | One-command runner: validation, training, testing, evaluation, logs, checkpointing, and resume. |
| `summarize_semantic_results.py` | Aggregates per-seed metrics and outputs paper-ready tables. |
| `validate_semantic_ready.py` | Checks whether prepared data and semantic features are ready before training. |
| `semantic_ablation_data.py` | Builds independent graph data for cognitive-factor ablations. |
| `evaluate_recommendations.py` | Computes ACC, NOV, and Ep_sim from exported recommendation scores. |

## 3. Model Components

SemanticConvE uses:

```text
entity representation =
ID embedding
+ text semantic embedding
+ pedagogical numeric embedding
+ entity type embedding
+ learner cluster embedding
```

and:

```text
relation representation =
relation type embedding
+ projected continuous relation strength
```

At recommendation time, the model scores only exercise entities:

```text
score(uid, rec, exercise)
```

## 4. Supported Ablations

The code supports:

```text
full
no_mastery
no_forgetting
no_seq
no_semantic
no_pedagogical
no_relation_strength
id_only
```

The cognitive-factor ablations generate independent graph data:

| Ablation | Recommendation terms | Removed relation |
| --- | --- | --- |
| `no_mastery` | forgetting + sequence | `mlkc` |
| `no_forgetting` | mastery + sequence | `exfr` |
| `no_seq` | mastery + forgetting | `pkc` |

The model-component ablations use the full graph and only disable model inputs:

| Ablation | Disabled component |
| --- | --- |
| `no_semantic` | text semantic embeddings |
| `no_pedagogical` | IRT/pedagogical numeric features and learner cluster |
| `no_relation_strength` | continuous relation strength |
| `id_only` | semantic, pedagogical, entity type, cluster, and relation strength components |

## 5. Data Assumption

The repository does not contain data. Copy prepared datasets into:

```text
KG4ER-New/data/
```

Each dataset should already contain KG graph files, KT outputs, IRT features, semantic features, and text embeddings. The validation script checks this before training.

