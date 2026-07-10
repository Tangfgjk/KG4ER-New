# SemanticConvE V10 前置文件生成与运行命令

本文档说明 V10 前置文件如何从 `ER/KG4ER/data` 中读取原始/已处理数据，并统一输出到：

```text
ER/KG4ER-New/data/<dataset>/v10/
```

原始数据目录 `ER/KG4ER/data` 不会被覆盖。

## 1. 数据集

```powershell
$datasets = @(
  "Eedi",
  "algebra2005",
  "assist2009-sub",
  "statics2011",
  "XES3G5M-sub-small"
)
```

## 2. 生成 V10 MIRT 输入

这一步只保留 ER 图中对应的 learner 子集，并保持题目 ID 不重新压缩，避免 Eedi-sub 错用完整 Eedi。

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\prepare_mirt_inputs_v10.py `
    --dataset $ds `
    --force
}
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/mirt/inputs/
```

## 3. 训练 MIRT 教育学特征估计器

MIRT 只作为教育学特征估计器，用来估计题目 difficulty/discrimination 和学习者 theta，不作为最终推荐模型。

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\train_mirt_noq_v10.py `
    --dataset $ds `
    --epoch 50 `
    --batch-size 2048 `
    --device cuda `
    --force
}
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/mirt/outputs/
```

## 4. 导出 DKT Sequence/Progress

每个数据集需要先训练 pyKT DKT，然后用下面命令导出 `stu2know_seq.json`。`$ckpt` 换成对应数据集训练好的 pyKT DKT checkpoint 目录。

```powershell
$ds = "algebra2005"
$ckpt = "ER/pykt-toolkit-main/examples/saved_model/<your_dkt_checkpoint_dir>"

python ER\KG4ER-New\v10_pipeline\export_seq_from_dkt_v10.py `
  --dataset $ds `
  --pykt-root ER/pykt-toolkit-main `
  --checkpoint-dir $ckpt `
  --checkpoint-file qid_model.ckpt `
  --test-sequences ER/KG4ER/data/$ds/prepared_for_kt/dkt_concept/test_sequences_full.csv `
  --expected-students 144 `
  --expected-concepts 112 `
  --device cuda
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/kt_dkt/stu2know_seq.json
ER/KG4ER-New/data/<dataset>/v10/stu2know_seq.json
```

## 5. 导入 EKTM_mirt Mastery

严格 V10 要求 `stu2know_mastery.json` 来自 EKTM_mirt 的 `know_output`。当 EKTM_mirt 导出矩阵后，用下面命令导入：

```powershell
python ER\KG4ER-New\v10_pipeline\import_ektm_mastery_v10.py `
  --dataset algebra2005 `
  --know-output-file "C:\path\to\algebra2005_ektm_mirt_know_output.json" `
  --expected-students 144 `
  --expected-concepts 112
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/ektm_mirt/stu2know_mastery.json
ER/KG4ER-New/data/<dataset>/v10/stu2know_mastery.json
```

## 6. 重新生成遗忘率

`stu2know_forget.json` 由测试 learner 的时间序列重新计算；`stu2ex_forget.json` 由 `stu2know_forget.json + Q.txt` 按知识点平均得到，不再使用求和或事后 clip。

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\generate_forgetting_v10.py `
    --dataset $ds `
    --force
}
```

## 7. 构建 V10 ER 图

默认推荐距离使用：

```text
sqrt((delta1 - mastery_product)^2 + (1 - cos(Q, seq))^2 + (delta2 - forget)^2)
```

排序使用完整精度，最终保存时只做可读性四舍五入。

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\build_v10_graph.py `
    --dataset $ds `
    --sequence-term one_minus_cos_sq `
    --top-k-rec 10 `
    --force
}
```

调试时如果暂时没有新 DKT/EKTM_mirt 导出，可加 `--allow-existing-state`，但正式实验不要使用。

## 8. 导入 EKTM_mirt topic_v 文本向量

知识点 definition、题目文本或结构化文本元数据可以复用已有固定文件；但正式 V10 的题目文本向量不再使用早期 BGE-M3 向量，而必须来自 EKTM_mirt 中 `TopicRNNModel` / Bi-GRU 导出的 `topic_v`。

当 EKTM_mirt 的题目文本向量矩阵导出后，使用下面命令导入。矩阵行顺序必须与 `ex0, ex1, ..., exN` 一致。

```powershell
python ER\KG4ER-New\v10_pipeline\import_ektm_topic_embeddings_v10.py `
  --dataset Eedi `
  --topic-embedding-file "C:\path\to\Eedi_ektm_topic_v.npy" `
  --force
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/semantic_kg_features/text_embeddings/
```

说明：知识点文本 embedding 会写成零向量，因为 EKTM_mirt 没有使用知识点 definition 作为文本编码输入；知识点 definition 仍保留在 `concept_semantics.json` 中用于解释和文字描述。

## 9. 生成 V10 语义与教育学特征

学习者和题目教育学特征从 V10 MIRT 与 V10 mastery 重新生成。

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\export_mirt_features_v10.py `
    --dataset $ds `
    --force
}
```

## 10. 校验前置文件

```powershell
python ER\KG4ER-New\v10_pipeline\validate_v10_front_files.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small
```

校验输出：

```text
ER/KG4ER-New/data/v10_validation_report.json
```

## 11. 训练 SemanticConvE

```powershell
python ER\KG4ER-New\codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root ER\KG4ER-New\data `
  --graph-subdir v10 `
  --run-id Eedi_v10_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,hybrid_relation,id_only,relation_id_only,no_type_aware_scoring `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑：

```powershell
python ER\KG4ER-New\codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root ER\KG4ER-New\data `
  --graph-subdir v10 `
  --run-id Eedi_v10_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,hybrid_relation,id_only,relation_id_only,no_type_aware_scoring `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```
