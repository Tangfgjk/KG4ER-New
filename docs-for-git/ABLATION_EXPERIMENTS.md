# SemanticConvE V6 Ablation Experiments

V6 uses a compact first-round ablation suite. The goal is to identify which newly added feature groups are useful before expanding to more expensive experiments.

## Default Suite

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

## Experiment Meanings

| Ablation | What changes | Purpose |
|---|---|---|
| `full_state_hybrid` | Full V6 model | Uses StateEncoder, semantic features, IRT features, statistical pedagogical features, continuous relation encoding, and type-aware scoring. |
| `irt_only_ped` | Removes statistical pedagogical features | Tests whether IRT features alone are useful. |
| `stat_only_ped` | Removes IRT features | Tests whether simple statistical educational features are more stable than IRT. |
| `no_irt` | Removes IRT from the full model | Measures the marginal value of IRT when statistical features remain. |
| `no_stat_ped` | Removes statistical pedagogical features from the full model | Measures the marginal value of statistical features when IRT remains. |
| `no_mastery` | Removes `mlkc` graph relations, removes mastery from recommendation-edge generation, and masks mastery state input | Tests mastery contribution. |
| `no_forgetting` | Removes `exfr` graph relations, removes forgetting from recommendation-edge generation, masks forgetting state input, and disables explicit forgetting score | Tests forgetting contribution. |
| `no_seq` | Removes `pkc` graph relations, removes sequence/progress from recommendation-edge generation, and masks sequence state input | Tests sequence/progress contribution. |

## Train And Evaluation Separation

For graph ablations, V6 trains on the ablated graph but evaluates with the original complete graph files.

This matters because metrics such as ACC need complete cognitive-state references. For example, `no_mastery` removes `mlkc` from the training graph, but ACC should still be computed against the original mastery-based evaluation standard.

## Test Triples Policy

V6 default:

```text
triples.txt      -> training
test_triples.txt -> evaluation only
```

Use `--include-test-triples` only for diagnostic comparison with older transductive settings.

