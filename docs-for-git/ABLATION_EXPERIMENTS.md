# SemanticConvE V4 Ablation Experiments

This document explains every experiment included in:

```powershell
--ablations all
```

`all` expands to:

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

## 1. Full Model

| Name | Meaning |
| --- | --- |
| `full` | Complete SemanticConvE V4 model. |

The full model uses ID embeddings, entity type embeddings, concept semantics, exercise semantics, exercise IRT features, learner IRT/mastery features, learner cluster embeddings, relation type embeddings, and continuous relation strength.

## 2. Cognitive-Factor Graph Ablations

These ablations rebuild the graph data.

| Name | Removed Component | Graph Change |
| --- | --- | --- |
| `no_mastery` | Mastery term | Removes `mlkc` relations and rebuilds recommendation triples without mastery. |
| `no_forgetting` | Forgetting term | Removes `exfr` relations and rebuilds recommendation triples without forgetting. |
| `no_seq` | Sequence/progress term | Removes `pkc` relations and rebuilds recommendation triples without sequence/progress. |

Purpose: test the contribution of the three educational factors used in graph construction and recommendation-edge generation.

## 3. Semantic Feature Ablations

| Name | Removed Component | Kept Component |
| --- | --- | --- |
| `no_semantic` | All text semantic embeddings | ID, pedagogical features, cluster, and relation encoding. |
| `no_concept_semantic` | Concept name and definition embeddings | Exercise semantics remain enabled. |
| `no_exercise_semantic` | Exercise text or structured exercise semantic embeddings | Concept semantics remain enabled. |

Purpose: determine whether text semantics are useful, and whether concept semantics or exercise semantics contribute more.

## 4. Pedagogical Feature Ablations

| Name | Removed Component | Kept Component |
| --- | --- | --- |
| `no_pedagogical` | All pedagogical numeric features and learner cluster | ID, semantics, entity type, and relation encoding. |
| `no_exercise_irt` | Exercise difficulty, discrimination, correct rate, and interaction count | Learner IRT/mastery and cluster remain enabled. |
| `no_learner_irt` | Learner ability, overall mastery, KT mastery, correct rate, history length, and concept-mastery features | Exercise IRT and cluster remain enabled. |
| `no_cluster` | Learner cluster embedding only | Exercise and learner IRT features remain enabled. |

Purpose: split the former coarse `no_pedagogical` ablation into exercise-side, learner-side, and cluster-side components.

## 5. Relation Representation Ablations

The default full model uses:

```text
r = LayerNorm(relation_type_embedding + MLP(continuous_strength))
```

V4 adds:

| Name | Relation Representation | Purpose |
| --- | --- | --- |
| `no_relation_strength` | `LayerNorm(relation_type_embedding)` | Test whether continuous relation strength is useful. |
| `discrete_relation` | `LayerNorm(relation_id_embedding)` | Compare against the older discrete relation representation capacity. |
| `hybrid_relation` | `LayerNorm(relation_id_embedding + relation_type_embedding + MLP(continuous_strength))` | Keep both discrete fitting capacity and continuous-strength interpretability. |

If `hybrid_relation` outperforms `full`, the default continuous relation representation may be too restrictive.

## 6. ID-only Lower Bound

| Name | Meaning |
| --- | --- |
| `id_only` | Entity representation uses only entity ID embedding; relation representation uses only relation ID embedding. |

Purpose: test whether semantic, pedagogical, entity-type, cluster, and continuous-relation components improve over pure ID memorization.

## 7. How to Interpret Results

Use ACC-Avg as the primary metric and NOV-Avg as a secondary metric.

```text
full > ablation:
    the removed component is helpful.

full ≈ ablation:
    the removed component has weak or unstable contribution.

full < ablation:
    the removed component may introduce noise and should be removed or weakened.
```

For relation variants:

```text
hybrid_relation > full:
    use hybrid relation encoding in the next main model.

discrete_relation > full:
    the dataset benefits from discrete relation memorization.

full > discrete_relation:
    continuous relation strength is useful.
```

