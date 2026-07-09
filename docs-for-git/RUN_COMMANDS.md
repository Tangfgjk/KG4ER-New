# KG4ER-New V9 Run Commands

This branch contains the V9 SemanticConvE experiment code. Data files are not stored in GitHub.

For TransE, TransE-adv, RotatE, DistMult, ComplEx, EB-CF, SB-CF, CBF, and KCP-ER, see:

```text
docs-for-git/COMPARISON_MODELS.md
```

Copy each prepared dataset into:

```text
data/<dataset>/er_v8/
data/<dataset>/er_v8/semantic_kg_features/
```

## Default V9 Ablations

`--ablations all` expands to:

```text
full
hybrid_relation
id_only
relation_id_only
no_type_aware_scoring
```

## Validate Eedi

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --allow-template
```

## Run Eedi V9 ConvE Experiments

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_conve_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0 `
  --cuda auto
```

Resume:

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_conve_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0 `
  --cuda auto `
  --resume
```

Summarize:

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v9_conve_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --runs-root runs
```
