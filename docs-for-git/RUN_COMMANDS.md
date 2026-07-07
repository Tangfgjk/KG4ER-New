# KG4ER-New V6 Run Commands

This repository branch contains **SemanticConvE V6**.

For the complete new-computer workflow, use:

```text
docs-for-git/SemanticConvE-V6运行命令.md
```

That document covers:

- cloning and checking out the `V6` branch;
- copying prepared datasets into `data/`;
- validating semantic features;
- running the full Eedi V6 experiment suite on one computer;
- resuming interrupted runs;
- summarizing ACC/NOV/Ep_sim and gate values.

The default V6 suite is:

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

Quick Eedi command:

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v6_full_suite_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --data-root .\data `
  --runs-root .\runs `
  --resume
```

Summarize:

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v6_full_suite_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --runs-root .\runs
```

