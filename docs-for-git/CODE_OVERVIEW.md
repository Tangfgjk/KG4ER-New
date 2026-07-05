# KG4ER-New Code Overview

This document introduces the code uploaded to the `KG4ER-New` repository. The repository contains only the code required for SemanticConvE training, testing, evaluation, result summarization, and ablation experiments. Dataset files and generated results are intentionally excluded.

Current code branch: **V4 fine-grained ablation**.

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
    ABLATION_EXPERIMENTS.md
  requirements.txt
  README.md
  .gitignore
```

## 2. Core Scripts

| File | Purpose |
| --- | --- |
| `feature_loader.py` | Loads entity IDs, relation IDs, semantic text embeddings, pedagogical numeric features, entity types, learner clusters, semantic-quality priors, relation types, and continuous relation strengths. |
| `semantic_conve_model.py` | Defines SemanticConvE with type-aware gated semantic/pedagogical entity fusion, continuous relation-aware encoding, and type-aware tail scoring. |
| `run_semantic_conve.py` | Trains one SemanticConvE model for one dataset, one seed, and one ablation setting. |
| `test_semantic_conve.py` | Loads a trained checkpoint and exports learner-exercise recommendation scores. |
| `run_semantic_experiments.py` | One-command runner: validation, training, testing, evaluation, logs, checkpointing, and resume. |
| `summarize_semantic_results.py` | Aggregates per-seed metrics and outputs paper-ready tables. |
| `validate_semantic_ready.py` | Checks whether prepared data and semantic features are ready before training. |
| `semantic_ablation_data.py` | Builds independent graph data for cognitive-factor ablations. |
| `evaluate_recommendations.py` | Computes ACC, NOV, and Ep_sim from exported recommendation scores. |

## 3. Model Components

SemanticConvE V4 uses:

```text
entity representation =
ID embedding
+ entity type embedding
+ gate_sem[type] * semantic_quality(entity) * text semantic embedding
+ mask_ped(type) * gate_ped[type] * pedagogical numeric embedding
+ mask_cluster(type) * gate_cluster[type] * learner cluster embedding
```

where `mask_ped(type)` is 1 only for learner and exercise entities, and
`mask_cluster(type)` is 1 only for learner entities. This avoids injecting
learner-cluster or numeric-projector bias into concept/exercise entities that
do not own those features.

and:

```text
relation representation =
relation type embedding
+ projected continuous relation strength
```

V4 also supports relation-representation variants:

```text
discrete_relation: relation ID embedding
hybrid_relation:   relation ID embedding + relation type embedding + projected relation strength
```

At recommendation time, the model scores only exercise entities:

```text
score(uid, rec, exercise)
```

The gate values are learned during training and exported to:

```text
runs/{dataset}/{run_id}/SemanticConvE/seed{seed}/gate_values.json
```

The result summarizer also outputs:

```text
runs/{dataset}/{run_id}/summary/gate_values_per_seed.csv
runs/{dataset}/{run_id}/summary/gate_values_mean_std.csv
```

## 4. V4 Changes Compared with V2/V3/V3.1

| Area | V2 | V3 | V3.1 | V4 |
| --- | --- | --- | --- | --- |
| Entity fusion | Direct addition of ID, semantic, pedagogical, type, and cluster embeddings | Type-aware gated fusion after semantic/numeric projection | Type-aware gated fusion with entity-type masks | Kept unchanged |
| Semantic reliability | All available text embeddings contributed equally | Each entity has a fixed `semantic_quality` prior based on semantic source | Kept unchanged | Kept unchanged |
| Gate initialization | Not used | Gate sigmoid initialized at 0.5 | Gate sigmoid initialized at 0.1 | Kept unchanged |
| Numeric counts | Raw log-count feature | Raw log-count feature | Log-count scaled into `[0, 1]` | Kept unchanged |
| Cluster feature | Directly fused | Gated but kept a non-learner placeholder vector | Applied only to learner entities | Adds `no_cluster` ablation |
| Pedagogical numeric feature | Directly fused | Gated for every entity type | Applied only to learner and exercise entities | Adds learner/exercise IRT ablations |
| Semantic feature | Directly fused | Gated by entity type and quality prior | Kept unchanged | Adds concept/exercise semantic ablations |
| Relation encoding | Continuous relation type + strength | Kept unchanged | Kept unchanged | Adds `discrete_relation` and `hybrid_relation` variants |
| Explainable diagnostics | Only recommendation explanations and metrics | Adds learned gate values for semantic, pedagogical, and cluster contributions | Kept unchanged | Kept unchanged |
| Recommendation scoring | `score(uid, rec, exercise)` over exercise tails | Kept unchanged | Kept unchanged | Kept unchanged |

## 5. Supported Ablations

The code supports:

```text
full
no_mastery
no_forgetting
no_seq
no_semantic
no_concept_semantic
no_exercise_semantic
no_pedagogical
no_exercise_irt
no_learner_irt
no_cluster
no_relation_strength
discrete_relation
hybrid_relation
id_only
```

Use the shortcut below to run the full model and every ablation:

```powershell
--ablations all
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
| `no_concept_semantic` | concept-name and concept-definition semantic embeddings |
| `no_exercise_semantic` | exercise text or structured exercise semantic embeddings |
| `no_pedagogical` | IRT/pedagogical numeric features and learner cluster |
| `no_exercise_irt` | exercise difficulty, discrimination, correct rate, and interaction-count features |
| `no_learner_irt` | learner ability, mastery, correctness, history-length, and concept-mastery features |
| `no_cluster` | learner cluster embedding only |
| `no_relation_strength` | continuous relation strength |
| `discrete_relation` | uses relation ID embedding only |
| `hybrid_relation` | uses relation ID + relation type + continuous strength |
| `id_only` | semantic, pedagogical, entity type, cluster, and relation strength components |

## 6. Data Assumption

The repository does not contain data. Copy prepared datasets into:

```text
KG4ER-New/data/
```

Each dataset should already contain KG graph files, KT outputs, IRT features, semantic features, and text embeddings. The validation script checks this before training.
