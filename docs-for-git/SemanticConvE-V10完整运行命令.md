# SemanticConvE V10 完整运行命令

本文档对应本地 V10 代码：

```text
C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New\ER\KG4ER-New
```

V10 的核心约定：

- 原始数据仍放在 `ER/KG4ER/data/`，不移动、不覆盖。
- 新生成的 V10 前置文件放在 `ER/KG4ER-New/data/<dataset>/v10/`。
- 训练测试默认读取 `data/<dataset>/v10/`。
- 主模型使用 `ID embedding + feature-token attention residual`。
- 消融实验包含新增的 `no_type_aware_scoring`，用于验证“只对习题实体打分”的创新点。

## 1. 环境和目录

```powershell
conda activate kg4er_cuda
cd "C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New"
```

五个正式数据集：

```powershell
$datasets = @(
  "Eedi",
  "algebra2005",
  "assist2009-sub",
  "statics2011",
  "XES3G5M-sub-small"
)
```

## 2. 生成 V10 前置文件

### 2.1 准备 MIRT 输入

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\prepare_mirt_inputs_v10.py `
    --dataset $ds `
    --seed 2024 `
    --force
}
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/mirt/inputs/
```

### 2.2 训练 no-Q MIRT

MIRT 在这里作为教育学特征估计器，用于估计题目难度、区分度和学生能力参数。

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

主要输出：

- `a_param_K.csv`
- `b_param_K.csv`
- `theta_param_K.csv`

### 2.3 训练 V10 PKC-DKT 并导出 `stu2know_seq.json`

该 DKT 不再直接复用外部 checkpoint，而是在 V10 中重新训练。任务是根据学生历史答题序列预测下一步知识点出现概率。

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\train_pkc_dkt_v10.py `
    --dataset $ds `
    --epochs 50 `
    --batch-size 128 `
    --device cuda `
    --force
}
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/pkc_dkt/
```

主要输出：

- `best.pt`
- `last.pt`
- `stu2know_seq.json`
- `pkc_dkt_manifest.json`

根目录同步文件：

```text
ER/KG4ER-New/data/<dataset>/v10/stu2know_seq.json
```

### 2.4 从 EKTM_mirt checkpoint 导出 mastery 和文本表示

这一步需要你先完成 EKTM_mirt 训练，并准备每个数据集的 best checkpoint 路径。

```powershell
$ektm = @{
  "Eedi" = "请替换为\Eedi\EKTM_mirt_best.pth"
  "algebra2005" = "请替换为\algebra2005\EKTM_mirt_best.pth"
  "assist2009-sub" = "请替换为\assist2009-sub\EKTM_mirt_best.pth"
  "statics2011" = "请替换为\statics2011\EKTM_mirt_best.pth"
  "XES3G5M-sub-small" = "请替换为\XES3G5M-sub-small\EKTM_mirt_best.pth"
}

foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\export_ektm_outputs_v10.py `
    --dataset $ds `
    --checkpoint $ektm[$ds] `
    --device cuda `
    --force
}
```

输出内容：

- `stu2know_mastery.json`：来自 EKTM_mirt 的 `know_output`。
- `exercise_text_embeddings.npy`：来自 EKTM_mirt 的题目 `topic_v`。
- `concept_text_embeddings.npy`：来自 EKTM_mirt 的知识点 embedding，并对齐到 `topic_v` 维度。

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/ektm_mirt/exports/
ER/KG4ER-New/data/<dataset>/v10/semantic_kg_features/text_embeddings/
```

### 2.5 生成遗忘文件

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\generate_forgetting_v10.py `
    --dataset $ds `
    --theta 10000000 `
    --timestamp-unit auto `
    --force
}
```

输出：

- `stu2know_forget.json`
- `stu2ex_forget.json`

V10 已将 `stu2ex_forget` 改成题目涉及知识点遗忘率的平均值，而不是求和后截断。

### 2.6 导出 MIRT 教育学特征

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\export_mirt_features_v10.py `
    --dataset $ds `
    --force
}
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/semantic_kg_features/
```

主要输出：

- `entity_features/learner_pedagogy.json`
- `irt_features/exercise_irt_features.json`

### 2.7 构建 V10 ER 图

默认使用新的 sequence 项：

```text
(1 - cos(Q_j, seq_i))^2
```

命令：

```powershell
foreach ($ds in $datasets) {
  python ER\KG4ER-New\v10_pipeline\build_v10_graph.py `
    --dataset $ds `
    --sequence-term one_minus_cos_sq `
    --top-k-rec 10 `
    --seed 2024 `
    --force
}
```

输出位置：

```text
ER/KG4ER-New/data/<dataset>/v10/
```

同时会生成旧公式对照目录：

```text
ER/KG4ER-New/data/<dataset>/v10_rec_legacy_cos/
```

### 2.8 校验 V10 前置文件

```powershell
python ER\KG4ER-New\v10_pipeline\validate_v10_front_files.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small
```

再校验 SemanticConvE 训练所需特征：

```powershell
python ER\KG4ER-New\codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --data-root ER\KG4ER-New\data `
  --graph-subdir v10
```

## 3. 新电脑拉取代码和拷贝数据

```powershell
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V10
git pull origin V10
```

然后把主电脑生成好的数据目录拷贝到新电脑仓库内：

```text
KG4ER-New/data/Eedi/v10/
KG4ER-New/data/algebra2005/v10/
KG4ER-New/data/assist2009-sub/v10/
KG4ER-New/data/statics2011/v10/
KG4ER-New/data/XES3G5M-sub-small/v10/
```

只训练某个数据集时，只拷贝对应数据集的 `v10` 文件夹即可。

## 4. SemanticConvE 主模型和消融实验

V10 默认消融列表：

```text
full,
id_only,
no_pedagogical,
no_text_semantic,
no_concept_semantic,
no_relation_aware,
no_type_aware_scoring,
no_mastery,
no_forgetting,
no_seq
```

其中：

- `full`：完整 V10 模型。
- `id_only`：最原始的实体和关系 ID embedding，不融合任何额外特征。
- `no_pedagogical`：去掉 MIRT 教育学特征。
- `no_text_semantic`：去掉题目和知识点文本/语义表示。
- `no_concept_semantic`：只去掉知识点语义表示。
- `no_relation_aware`：关系只保留 relation ID，不融合关系类型和强度。
- `no_type_aware_scoring`：测试时对所有实体打分，再筛选习题，用于验证 restricted-space/type-aware scoring。
- `no_mastery`：重新构图，去掉掌握度关系。
- `no_forgetting`：重新构图，去掉遗忘关系。
- `no_seq`：重新构图，去掉序列关系。

### 4.1 单数据集五个随机种子

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

续跑同一个 `run-id`，加 `--resume`：

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

### 4.3 五个数据集一键跑

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
    --cuda auto

  if ($LASTEXITCODE -ne 0) {
    throw "SemanticConvE dataset $ds failed."
  }
}
```

续跑时同样加 `--resume`。

## 5. 对比模型实验

对比模型仍然使用 ID-only 方式，不使用 V10 新增的语义、教育学、关系强度融合特征。它们在同一个 V10 图上运行，保证前置文件一致。

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

## 6. 统计结果

### 6.1 统计 SemanticConvE 消融结果

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v10_attn_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq
```

### 6.2 统计对比模型结果

```powershell
python comparison_models\summarize_dataset_results.py `
  --dataset Eedi `
  --run-dir runs\Eedi\Eedi_v10_comparison_5seeds
```

## 7. 输出目录

SemanticConvE 结果：

```text
KG4ER-New/runs/<dataset>/<run-id>/<model>/seed<seed>/
```

对比模型结果：

```text
KG4ER-New/runs/<dataset>/<run-id>/<model>/seed<seed>/
```

每个 seed 下常用文件：

- `train.log`
- `runner_train.log`
- `runner_test.log`
- `runner_eval.log`
- `best.pt`
- `last.pt`
- `SemanticConvE_uid_ex_scores.pkl`
- `eval/metrics.json`
- `eval/metrics.csv`

统计结果通常写入：

```text
KG4ER-New/runs/<dataset>/<run-id>/summaries/
```
