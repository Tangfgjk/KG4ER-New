# KG4ER-New V6 Code Overview

This branch contains the portable code for **SemanticConvE V6**. Dataset files, model checkpoints, and run outputs are intentionally excluded.

## Core Changes In V6

| Area | V6 behavior |
|---|---|
| Learner representation | Learner entities use `StateEncoder(state(uid))` instead of relying on learned `uid` ID embedding. |
| Relation representation | Full model uses `relation type + continuous strength`; relation ID embedding is not used in the full model. |
| Educational features | Loader separates IRT features and statistical pedagogical features for controlled ablations. |
| Train/test split | `test_triples.txt` is evaluation-only by default. It is not added to training unless `--include-test-triples` is explicitly passed. |
| Scoring | Recommendation still scores exercise candidates only through type-aware ConvE scoring. |

## Main Files

| File | Purpose |
|---|---|
| `codes-New-ConvE/feature_loader.py` | Loads entities, relations, text embeddings, IRT/statistical features, state features, relation types, and relation strengths. |
| `codes-New-ConvE/semantic_conve_model.py` | Defines SemanticConvE V6, including StateEncoder, gated entity features, continuous relation encoding, and type-aware scoring. |
| `codes-New-ConvE/run_semantic_conve.py` | Trains one model for one dataset, one seed, and one ablation. |
| `codes-New-ConvE/test_semantic_conve.py` | Exports learner-exercise recommendation scores from a trained checkpoint. |
| `codes-New-ConvE/run_semantic_experiments.py` | One-command train/test/evaluate runner with resume support. |
| `codes-New-ConvE/summarize_semantic_results.py` | Aggregates five-seed metrics and gate values. |
| `codes-New-ConvE/semantic_ablation_data.py` | Builds independent graph variants for `no_mastery`, `no_forgetting`, and `no_seq`. |
| `codes-New-ConvE/evaluate_recommendations.py` | Computes ACC, NOV, and Ep_sim from exported scores. |
| `codes-New-ConvE/validate_semantic_ready.py` | Checks whether the copied dataset contains all required graph and semantic feature files. |

## V6 Representation

Learner representation:

```text
h_uid = StateEncoder([
  stu2know_mastery,
  stu2know_seq,
  stu2know_forget,
  learner IRT features,
  learner statistical features,
  learner cluster feature
])
```

Relation representation:

```text
r = LayerNorm(type_embedding(relation_type) + gate * MLP(continuous_strength))
```

Recommendation score:

```text
score(uid, rec, ex)
= sigmoid(ConvETransform(h_uid, r_rec)^T h_ex + b_ex)
```

## Default V6 Ablations

`--ablations all` expands to:

```text
full_state_hybrid
irt_only_ped
stat_only_ped
no_irt
no_stat_ped
no_mastery
no_forgetting
no_seq
```

See `docs-for-git/SemanticConvE-V6运行命令.md` for full commands.

