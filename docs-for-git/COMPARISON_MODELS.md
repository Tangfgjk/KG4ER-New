# V9 Comparison Model Commands

The `comparison_models/` folder contains the ID-only baseline code copied from the original KG4ER implementation.

It is intended for running comparison models on the same V9 graph directory:

```text
data/<dataset>/er_v8/
```

These comparison models do **not** use SemanticConvE features directly. They read the graph files, ID dictionaries, recommendation labels, KT state files, and optional `sequence_interactions.csv` from the selected `er_v8` directory.

## Included Models

KGE baselines:

```text
TransE
TransE-adv
RotatE
DistMult
ComplEx
```

Traditional baselines:

```text
EB-CF
SB-CF
CBF
KCP-ER
```

## Required Data

Copy the prepared dataset directory to:

```text
data/<dataset>/er_v8/
```

The directory should include:

```text
Q.txt
entities.dict
relations.dict
triples.txt
test_triples.txt
stu2know_mastery.json
stu2know_seq.json
stu2know_forget.json
stu2ex_forget.json
stu2ex_recommend.json
sequence_interactions.csv
```

`sequence_interactions.csv` is important for EB-CF and SB-CF.

## Run One Dataset

Example for Eedi:

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_baselines_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models all `
  --cuda auto
```

Resume:

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_baselines_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models all `
  --cuda auto `
  --resume
```

## Run Only KGE Baselines

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_kge_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models TransE,TransE-adv,RotatE,DistMult,ComplEx `
  --cuda auto
```

## Run Only Traditional Baselines

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_cf_baselines `
  --seeds 2024 `
  --models EB-CF,SB-CF,CBF,KCP-ER `
  --cuda false
```

## Summaries

After a run finishes, summary files are written under:

```text
runs/<dataset>/<run-id>/summaries/
```

The most useful files are:

```text
dataset_summary.csv
dataset_summary.md
dataset_summary_stats.csv
dataset_summary_stats.md
dataset_summary.json
```

