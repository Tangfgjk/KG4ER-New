# SemanticConvE V13 完整运行命令

本文档用于 V13 分支。V13 的核心变化是重新生成 `stu2know_seq.json`：

- 不再使用“全 1 标签”的 pyKT patch 导出方式；
- 使用 DKT/LSTM 结构做 **下一步知识点多标签预测**；
- 每个时间步的标签为下一道题对应的 Q-matrix multi-hot 知识点向量；
- 最终输出格式仍为原来的 `stu2know_seq.json`，即 `学生数 × 知识点数`。

所有命令默认在仓库根目录执行：

```powershell
cd "C:\Users\29694\Desktop\ER\KG4ER-New"
conda activate kg4er_cuda
```

如果在新电脑运行，先拉取 V13：

```powershell
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V13
git pull origin V13
```

五个数据集名称如下：

```powershell
$datasets = @(
  "Eedi",
  "algebra2005",
  "assist2009-sub",
  "statics2011",
  "XES3G5M-sub-small"
)
```

## 1. 生成 V13 前置文件

V13 前置文件统一保存在：

```text
data/<dataset>/v13/
```

原始数据仍从旧目录读取：

```text
ER/KG4ER/data/<dataset>/
```

### 1.1 准备 MIRT 输入

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\prepare_mirt_inputs_v13.py `
    --dataset $ds `
    --force
}
```

### 1.2 训练 no-Q MIRT 并导出 a/b/theta 参数

MIRT 在 V13 中作为教育学特征估计器，用于生成题目难度、区分度和学生能力参数。

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\train_mirt_noq_v13.py `
    --dataset $ds `
    --epoch 70 `
    --batch-size 1024 `
    --lr 0.001 `
    --device cuda `
    --force
}
```

### 1.3 训练 next-KC DKT 并导出 `stu2know_seq.json`

这是 V13 重点修复步骤。输出仍是：

```text
data/<dataset>/v13/stu2know_seq.json
```

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\train_pkc_dkt_v13.py `
    --dataset $ds `
    --epochs 30 `
    --batch-size 128 `
    --learning-rate 0.001 `
    --positive-weight auto `
    --device cuda `
    --force
}
```

如果只想从已有 checkpoint 重新导出：

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\train_pkc_dkt_v13.py `
    --dataset $ds `
    --device cuda `
    --export-only
}
```

### 1.4 训练 EKTM_mirt 并导出 mastery 与文本表示

该步骤导出：

- `stu2know_mastery.json`
- `semantic_kg_features/text_embeddings/exercise_text_embeddings.npy`
- `semantic_kg_features/text_embeddings/concept_text_embeddings.npy`

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\train_ektm_mirt_v13.py `
    --dataset $ds `
    --epochs 30 `
    --batch-size 16 `
    --device cuda `
    --force
}
```

如果只想从已有 EKTM checkpoint 重新导出：

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\train_ektm_mirt_v13.py `
    --dataset $ds `
    --device cuda `
    --export-only
}
```

### 1.5 生成遗忘、语义教育特征和 ER 图

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\generate_forgetting_v13.py `
    --dataset $ds `
    --theta 10000000 `
    --timestamp-unit auto `
    --force

  python v13_pipeline\export_mirt_features_v13.py `
    --dataset $ds `
    --force

  python v13_pipeline\build_v13_graph.py `
    --dataset $ds `
    --sequence-term one_minus_cos_sq `
    --force
}
```

说明：

- `build_v13_graph.py` 默认使用 `(1 - cos(Q_j, seq_i))^2` 作为 sequence 项；
- 排序使用完整精度的推荐距离；
- `stu2ex_recommend.json` 只用于存储和查看，保留 6 位小数；
- 关系仍固定为 304 个：`rec + mlkc/pkc/exfr 的 0.00~1.00 离散关系`。

### 1.6 校验 V13 前置文件

```powershell
python v13_pipeline\validate_v13_front_files.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --output-file data\v13_validation_report.json
```

### 1.7 提前统计前置推荐分数

```powershell
foreach ($ds in $datasets) {
  python v13_pipeline\score_v13_front_files.py `
    --dataset $ds
}
```

## 2. 运行 SemanticConvE 主模型和消融实验

V13 消融实验建议使用：

```text
full
id_only
no_pedagogical
no_text_semantic
no_concept_semantic
no_relation_aware
no_type_aware_scoring
no_mastery
no_forgetting
no_seq
```

### 2.1 单个数据集运行

以 Eedi 为例：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v13 `
  --run-id Eedi_v13_attn_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --include-test-triples
```

续跑：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v13 `
  --run-id Eedi_v13_attn_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --include-test-triples `
  --resume
```

### 2.2 五个数据集一键运行

```powershell
foreach ($ds in $datasets) {
  python codes-New-ConvE\run_semantic_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir v13 `
    --run-id "${ds}_v13_attn_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
    --epochs 25 `
    --bs 1024 `
    --negative-ratio 5 `
    --cuda auto `
    --include-test-triples

  if ($LASTEXITCODE -ne 0) {
    throw "SemanticConvE V13 failed on $ds"
  }
}
```

五个数据集续跑：

```powershell
foreach ($ds in $datasets) {
  python codes-New-ConvE\run_semantic_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir v13 `
    --run-id "${ds}_v13_attn_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
    --epochs 25 `
    --bs 1024 `
    --negative-ratio 5 `
    --cuda auto `
    --include-test-triples `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "SemanticConvE V13 resume failed on $ds"
  }
}
```

## 3. 运行对比模型

对比模型使用 ID-only 图嵌入和传统 baseline，不使用 SemanticConvE 的新增语义/教育特征。

### 3.1 单个数据集

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v13 `
  --run-id Eedi_v13_comparison_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF,KCP-ER `
  --cuda auto
```

续跑：

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v13 `
  --run-id Eedi_v13_comparison_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --models TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF,KCP-ER `
  --cuda auto `
  --resume
```

### 3.2 五个数据集一键运行

```powershell
foreach ($ds in $datasets) {
  python comparison_models\run_v9_comparison_experiments.py `
    --dataset $ds `
    --data-root data `
    --graph-subdir v13 `
    --run-id "${ds}_v13_comparison_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --models TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF,KCP-ER `
    --cuda auto

  if ($LASTEXITCODE -ne 0) {
    throw "Comparison V13 failed on $ds"
  }
}
```

## 4. 统计结果

SemanticConvE 结果统计：

```powershell
foreach ($ds in $datasets) {
  python codes-New-ConvE\summarize_semantic_results.py `
    --dataset $ds `
    --run-id "${ds}_v13_attn_5seeds" `
    --seeds 2024,2025,2026,2027,2028
}
```

对比模型结果统计仍使用各自 run 目录中的 summary 文件。如果需要统一论文表格，可以继续使用已有的 `summarize_dataset_results.py` 或后续单独汇总脚本。

## 5. 新电脑只训练测试时需要拷贝什么

如果 V13 前置文件已经在主电脑生成，新电脑只跑训练/测试，则每个数据集只需要拷贝：

```text
data/<dataset>/v13/
```

可删除或不拷贝其中的中间模型目录：

```text
mirt/
pkc_dkt/
ektm_mirt/
forgetting/
front_oracle_eval/
```

但必须保留：

```text
semantic_kg_features/
entities.dict
relations.dict
Q.txt
*_uid_kc_response.txt
stu2know_mastery.json
stu2know_seq.json
stu2know_forget.json
stu2ex_forget.json
stu2ex_recommend.json
stu2ex_recommend_full_precision.json
triples.txt
test_triples.txt
v13_graph_manifest.json
```

