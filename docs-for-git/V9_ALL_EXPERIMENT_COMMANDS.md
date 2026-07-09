# KG4ER-New V9 全实验运行命令

本文档用于新电脑拉取 V9 代码后，运行五个数据集上的 SemanticConvE 消融实验和对比模型实验。

V9 代码只上传训练、测试、统计所需代码，不上传数据和运行结果。因此新电脑需要先拉取代码，再拷贝已经生成好的 `er_v8` 前置文件。

## 1. 拉取 V9 代码

```powershell
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V9
git pull origin V9
```

如果仓库已经存在：

```powershell
cd KG4ER-New
git fetch origin
git checkout V9
git pull origin V9
```

## 2. 需要拷贝哪些数据

只需要拷贝每个数据集已经生成好的 `er_v8` 文件夹。

新电脑上的目录结构应为：

```text
KG4ER-New/
  data/
    Eedi/
      er_v8/
    algebra2005/
      er_v8/
    assist2009-sub/
      er_v8/
    statics2011/
      er_v8/
    XES3G5M-sub-small/
      er_v8/
```

每个 `er_v8` 文件夹至少应包含：

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
semantic_kg_features/
```

其中：

- `semantic_kg_features/` 是 SemanticConvE 使用的语义和教育学特征目录。
- `sequence_interactions.csv` 是 EB-CF、SB-CF 等传统对比模型需要的学生作答序列文件。
- 不需要拷贝 `runs/`，运行时会自动生成。
- 不需要拷贝原始数据、MIRT 训练代码、DeepSeek 生成脚本或 BGE-M3 模型缓存；V9 训练测试阶段只读取 `er_v8`。

## 3. 安装环境

进入仓库根目录后安装依赖：

```powershell
pip install -r requirements.txt
```

建议使用支持 CUDA 的 PyTorch 环境。运行命令中的 `--cuda auto` 会优先使用 GPU；如果当前环境没有可用 CUDA，则自动使用 CPU。

## 4. 检查数据是否完整

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --data-root data `
  --graph-subdir er_v8 `
  --allow-template
```

输出中每个数据集都应显示：

```text
"status": "passed"
```

## 5. SemanticConvE V9 消融实验

### 5.1 V9 默认消融实验包含哪些模型

`--ablations all` 会运行以下 5 个实验：

```text
full
hybrid_relation
id_only
relation_id_only
no_type_aware_scoring
```

含义如下：

| 实验名 | 含义 |
|---|---|
| `full` | V9 主模型，实体使用 ID + 文本语义 + MIRT 教育学特征，关系使用类型 + 连续强度，推荐时只对习题实体打分 |
| `hybrid_relation` | 在 `full` 基础上，把关系表示改为 relation ID + relation type + relation strength |
| `id_only` | 所有实体和关系只使用 ID embedding，相当于最接近原始 ConvE 的设置 |
| `relation_id_only` | 实体仍使用 V9 特征，但关系只使用 relation ID embedding |
| `no_type_aware_scoring` | 不限制只对习题实体打分，而是恢复为对所有实体打分后再评估 |

### 5.2 单独运行一个数据集

以 Eedi 为例：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_semantic_conve_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0 `
  --cuda auto
```

### 5.3 续跑一个数据集

如果中断，使用相同 `--run-id` 并增加 `--resume`：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir er_v8 `
  --run-id Eedi_v9_semantic_conve_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0 `
  --cuda auto `
  --resume
```

续跑会检查已完成的模型和随机种子，已完成的会跳过，未完成的继续运行。

### 5.4 统计一个数据集的 SemanticConvE 结果

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v9_semantic_conve_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --runs-root runs
```

结果保存在：

```text
runs/Eedi/Eedi_v9_semantic_conve_5seeds/summaries/
```

## 6. 对比模型实验

### 6.1 V9 对比模型包含哪些

`comparison_models/` 中包含原 KG4ER 的 ID-only 对比模型代码。

KGE 对比模型：

```text
TransE
TransE-adv
RotatE
DistMult
ComplEx
```

传统 baseline：

```text
EB-CF
SB-CF
CBF
KCP-ER
```

这些模型默认读取同一个 `data/<dataset>/er_v8/` 图数据，但不使用 SemanticConvE 的额外语义和教育学特征。

### 6.2 单独运行一个数据集的全部对比模型

以 Eedi 为例：

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

### 6.3 续跑一个数据集的对比模型

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

### 6.4 只运行 KGE 对比模型

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

### 6.5 只运行传统 baseline

传统 baseline 没有神经网络训练过程，通常不需要 5 个随机种子。可以只跑一个种子：

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

### 6.6 对比模型结果保存位置

```text
runs/<dataset>/<run-id>/summaries/
```

重点看：

```text
dataset_summary.csv
dataset_summary.md
dataset_summary_stats.csv
dataset_summary_stats.md
dataset_summary.json
```

## 7. 一键运行五个数据集的 SemanticConvE 消融实验

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
    --graph-subdir er_v8 `
    --run-id "${ds}_v9_semantic_conve_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations all `
    --epochs 25 `
    --bs 1024 `
    --negative-ratio 5 `
    --forgetting-score-weight 0 `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "SemanticConvE failed for $ds"
  }
}
```

## 8. 一键运行五个数据集的对比模型

```powershell
$datasets = @(
  "Eedi",
  "algebra2005",
  "assist2009-sub",
  "statics2011",
  "XES3G5M-sub-small"
)

foreach ($ds in $datasets) {
  python comparison_models\run_v9_comparison_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir er_v8 `
    --run-id "${ds}_v9_baselines_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --models all `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "Comparison baselines failed for $ds"
  }
}
```

## 9. 一键运行五个数据集的所有实验

这个命令会先跑 SemanticConvE V9 消融实验，再跑所有对比模型。时间较长，建议在确认单个数据集能跑通后再使用。

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
    --graph-subdir er_v8 `
    --run-id "${ds}_v9_semantic_conve_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations all `
    --epochs 25 `
    --bs 1024 `
    --negative-ratio 5 `
    --forgetting-score-weight 0 `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "SemanticConvE failed for $ds"
  }

  python comparison_models\run_v9_comparison_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir er_v8 `
    --run-id "${ds}_v9_baselines_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --models all `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "Comparison baselines failed for $ds"
  }
}
```

## 10. 常见问题

### 10.1 `runs/` 文件夹需要手动创建吗？

不需要。训练和测试脚本会自动创建：

```text
runs/<dataset>/<run-id>/
```

### 10.2 能不能只跑一个随机种子？

可以，例如：

```powershell
--seeds 2024
```

### 10.3 五个随机种子建议用哪些？

当前统一使用：

```text
2024,2025,2026,2027,2028
```

### 10.4 如果换电脑继续跑，怎么保证不是重新开始？

需要满足两个条件：

1. 拷贝原电脑对应的 `runs/<dataset>/<run-id>/` 运行结果目录；
2. 重新运行命令时使用相同 `--run-id` 并添加 `--resume`。

如果不拷贝 `runs/`，新电脑会从头开始跑，这是正常的。

