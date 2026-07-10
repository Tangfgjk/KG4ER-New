# SemanticConvE V10 消融与对比实验运行命令

本文档用于新电脑拉取 `V10` 分支后运行 SemanticConvE 消融实验和传统对比实验。

## 1. 拉取代码

```powershell
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V10
git pull origin V10
```

如果本地已经存在仓库：

```powershell
cd KG4ER-New
git fetch origin
git checkout V10
git pull origin V10
```

## 2. 拷贝数据

把已经生成好的 V10 前置文件拷贝到新电脑仓库下：

```text
KG4ER-New/data/Eedi/v10/
KG4ER-New/data/algebra2005/v10/
KG4ER-New/data/assist2009-sub/v10/
KG4ER-New/data/statics2011/v10/
KG4ER-New/data/XES3G5M-sub-small/v10/
```

也就是说，最终目录应类似：

```text
KG4ER-New/
  codes-New-ConvE/
  comparison_models/
  data/
    Eedi/
      v10/
        entities.dict
        relations.dict
        triples.txt
        test_triples.txt
        semantic_kg_features/
```

## 3. 验证前置文件

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --data-root data `
  --graph-subdir v10 `
  --allow-template
```

## 4. SemanticConvE 消融实验

V10 默认消融实验包括：

| ablation | 含义 |
|---|---|
| `full` | 完整 V10-attn 模型 |
| `id_only` | 实体和关系只使用 ID embedding |
| `no_pedagogical` | 去掉学生 theta/cluster 和题目 MIRT 难度、区分度 |
| `no_text_semantic` | 去掉题目文本和知识点文本语义 |
| `no_concept_semantic` | 只去掉知识点文本 embedding，保留题目文本 embedding |
| `no_relation_aware` | 关系只使用 relation ID，不融合 type 和 strength |
| `no_type_aware_scoring` | 测试时对所有实体打分，再筛选习题 |
| `no_mastery` | 图级消融，去掉 mastery 项和 `mlkc` |
| `no_forgetting` | 图级消融，去掉 forgetting 项和 `exfr` |
| `no_seq` | 图级消融，去掉 sequence/progress 项和 `pkc` |

其中 `no_concept_semantic` 就是“只去掉知识点文本 Embedding”的消融实验。

### 4.1 单个数据集完整消融

以 Eedi 为例：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v10 `
  --run-id Eedi_v10_attn_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

### 4.2 续跑

同一个命令加 `--resume`：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v10 `
  --run-id Eedi_v10_attn_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

### 4.3 五个数据集一键运行

```powershell
$datasets = @(
  "Eedi",
  "algebra2005",
  "assist2009-sub",
  "statics2011",
  "XES3G5M-sub-small"
)

foreach ($ds in $datasets) {
  python codes-New-ConvE\run_semantic_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir v10 `
    --run-id "${ds}_v10_attn_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
    --epochs 25 `
    --bs 1024 `
    --negative-ratio 5 `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "SemanticConvE failed for $ds"
  }
}
```

## 5. SemanticConvE 结果统计

单个数据集：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v10_attn_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq
```

五个数据集：

```powershell
foreach ($ds in $datasets) {
  python codes-New-ConvE\summarize_semantic_results.py `
    --dataset $ds `
    --run-id "${ds}_v10_attn_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq
}
```

## 6. 对比模型实验

对比模型使用 ID-only 图表示，不使用 SemanticConvE 新增实体/关系特征。

以 Eedi 为例：

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v10 `
  --run-id Eedi_v10_comparison_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF,KCP-ER `
  --cuda auto
```

续跑：

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v10 `
  --run-id Eedi_v10_comparison_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF,KCP-ER `
  --cuda auto `
  --resume
```

五个数据集：

```powershell
foreach ($ds in $datasets) {
  python comparison_models\run_v9_comparison_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir v10 `
    --run-id "${ds}_v10_comparison_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --models TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF,KCP-ER `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "Comparison failed for $ds"
  }
}
```
