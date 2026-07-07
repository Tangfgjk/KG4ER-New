# SemanticConvE V7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SemanticConvE V7 with feature-token attention fusion, concept text enabled in the full model, and a compact ablation suite for Eedi and later multi-dataset runs.

**Architecture:** V7 replaces V5's gated direct-add entity fusion with feature-token attention. Entity-side ID/text/pedagogical/type/cluster tokens are normalized, masked by entity type and ablation, fused by attention, projected back to the ConvE embedding dimension, and added to the ID embedding through a small residual scale. Relation representation keeps relation ID as the anchor and fuses relation type plus continuous strength.

**Tech Stack:** Python, PyTorch, existing KG4ER graph files, existing `semantic_kg_features` text/statistical feature files.

---

### Task 1: Add Feature-Token Attention Fusion

**Files:**
- Modify: `codes-New-ConvE/semantic_conve_model.py`

- [x] Add a `FeatureAttentionFusion` module that accepts token tensors shaped `[batch, token_count, dim]` and boolean masks shaped `[batch, token_count]`.
- [x] Replace entity gate-add logic with token construction for semantic, pedagogical, type, and cluster features.
- [x] Keep an ID residual path: `LayerNorm(id_emb + residual_scale * fused_extra)`.
- [x] Implement `direct_sum_fusion` as the V5-style comparison branch using the same active features.
- [x] Enable concept text in `full`; disable it only for `no_concept_text`, `no_text_semantic`, and `id_only`.

### Task 2: Update Ablation Names and Defaults

**Files:**
- Modify: `codes-New-ConvE/semantic_conve_model.py`
- Modify: `codes-New-ConvE/semantic_experiment_utils.py`

- [x] Add V7 ablations: `no_concept_text`, `no_exercise_text`, `no_text_semantic`, `direct_sum_fusion`, `no_pedagogical`, `no_relation_aware`, `no_type_aware_scoring`, `id_only`, `no_mastery`, `no_forgetting`, `no_seq`.
- [x] Keep backward-compatible aliases used by previous code where inexpensive.
- [x] Set `MODEL_VERSION = semantic_conve_v7_attention_fusion`.
- [x] Set default `all` ablations to the V7 compact suite.

### Task 3: Preserve Relation ID Anchor

**Files:**
- Modify: `codes-New-ConvE/semantic_conve_model.py`

- [x] Keep relation representation as `relation ID + relation type + relation strength`.
- [x] For `no_relation_aware`, return only normalized relation ID embedding.

### Task 4: Add V7 Run Documentation

**Files:**
- Add: `docs-for-git/SemanticConvE-V7三台电脑运行命令.md`
- Add: `docs-for-git/SemanticConvE-V7修改说明.md`

- [x] Document required copied dataset folders.
- [x] Split Eedi V7 experiments into three machine groups.
- [x] Include resume and summary commands.

### Task 5: Verify

**Commands:**

```powershell
python -m pytest codes-New-ConvE/test_semantic_conve_model.py codes-New-ConvE/test_semantic_experiment_utils.py -q
python codes-New-ConvE/run_semantic_experiments.py --dataset Eedi --run-id Eedi_v7_dryrun --seeds 2024 --ablations full,id_only,direct_sum_fusion --epochs 1 --dry-run --skip-validation
```

Expected:

```text
pytest passes
dry-run writes command/status files without training
```

Current verification:

```text
18 passed in local V7 unit tests
```
