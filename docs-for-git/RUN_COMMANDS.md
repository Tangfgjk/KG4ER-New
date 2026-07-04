# KG4ER-New Run Commands

This document explains how to run **SemanticConvE V3 gated fusion** and its ablation experiments after cloning the `KG4ER-New` repository on a new computer.

## 1. Install Environment

Create or activate a Python environment with PyTorch installed. For example:

```powershell
conda create -n kg4er_cuda python=3.10 -y
conda activate kg4er_cuda
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your GPU and driver from the official PyTorch instructions.

## 2. Copy Data

Copy prepared dataset folders into:

```text
KG4ER-New/data/
```

Expected examples:

```text
KG4ER-New/data/Eedi/
KG4ER-New/data/algebra2005/prepared_for_kt/
KG4ER-New/data/assist2009-sub/prepared_for_kt/
KG4ER-New/data/statics2011/prepared_for_kt/
KG4ER-New/data/XES3G5M-sub-small/prepared_for_kt/
```

Each formal graph directory should contain:

```text
entities.dict
relations.dict
triples.txt
test_triples.txt
Q.txt
stu2know_mastery.json
stu2know_seq.json
stu2ex_forget.json
*_uid_kc_response.txt
semantic_kg_features/
```

## 3. Validate Data

Run this before training:

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets "Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small"
```

The status should be:

```text
"status": "passed"
```

## 4. Run Full SemanticConvE

Example for one dataset:

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset algebra2005 `
  --run-id algebra2005_semantic_conve_v3_gated_fusion_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

Resume:

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset algebra2005 `
  --run-id algebra2005_semantic_conve_v3_gated_fusion_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 5. Run All Ablations

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset algebra2005 `
  --run-id algebra2005_semantic_ablation_v3_gated_fusion_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,no_mastery,no_forgetting,no_seq,no_semantic,no_pedagogical,no_relation_strength,id_only `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

Resume ablations:

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset algebra2005 `
  --run-id algebra2005_semantic_ablation_v3_gated_fusion_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,no_mastery,no_forgetting,no_seq,no_semantic,no_pedagogical,no_relation_strength,id_only `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 6. Summarize Results

Full model:

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset algebra2005 `
  --run-id algebra2005_semantic_conve_v3_gated_fusion_5seeds `
  --seeds 2024,2025,2026,2027,2028
```

Ablation results:

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset algebra2005 `
  --run-id algebra2005_semantic_ablation_v3_gated_fusion_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,no_mastery,no_forgetting,no_seq,no_semantic,no_pedagogical,no_relation_strength,id_only
```

Summary outputs are saved under:

```text
runs/{dataset}/{run_id}/summary/
```

V3 additionally writes gate diagnostics:

```text
runs/{dataset}/{run_id}/SemanticConvE/seed{seed}/gate_values.json
runs/{dataset}/{run_id}/summary/gate_values_per_seed.csv
runs/{dataset}/{run_id}/summary/gate_values_mean_std.csv
```

## 7. Output Directories

Full model:

```text
runs/{dataset}/{run_id}/SemanticConvE/seed{seed}/
```

Ablations:

```text
runs/{dataset}/{run_id}/SemanticConvE_{ablation}/seed{seed}/
```

Graph-level ablation data:

```text
ablation_data/{dataset}/{ablation}_top10/
```

These directories are generated automatically and are ignored by Git.
