# SemanticConvE V8 运行命令

本文档对应 `KG4ER-New` 的 V8 代码。V8 目标是把前置教育测量特征统一到修改后的 EduCDM no-Q MIRT 输出，避免继续使用早期自定义 2PL-IRT 特征。

当前 V8 使用的 MIRT 代码路径为：

```text
C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New\EduCDM_MIRT_noQ_export_modified\EduCDM-main
```

注意：当前 `export_mastery_from_mirt_v8.py` 使用的是 `mirt_proxy` 导出方式，即使用 no-Q MIRT 的学习者能力、题目难度和学生作答历史生成 `stu2know_mastery.json`。如果后续有兼容的 EKTM_mirt checkpoint，可以在同一输出目录下替换为严格 EKTM_mirt 推理导出的 mastery 文件。

## 1. 进入项目目录

```powershell
cd "C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New"
conda activate kg4er_cuda
```

如果不使用 `kg4er_cuda`，至少需要当前环境已安装：

```text
torch
pandas
numpy
scikit-learn
tqdm
```

## 2. 单个数据集完整前置流程

以下以 `Eedi` 为例。其他数据集把 `Eedi` 替换成：

```text
algebra2005
assist2009-sub
statics2011
XES3G5M-sub-small
```

### 2.1 生成 MIRT 输入文件

```powershell
python ER\KG4ER-New\v8_pipeline\prepare_mirt_inputs.py `
  --dataset Eedi `
  --strategy all `
  --seed 2024 `
  --force
```

输出位置：

```text
ER/KG4ER/data/Eedi/mirt_v8/inputs/
```

会生成：

```text
train.csv
valid.csv
test.csv
all.csv
Q_matrix.csv
mirt_input_manifest.json
```

### 2.2 训练 no-Q MIRT 并导出参数

```powershell
python ER\KG4ER-New\v8_pipeline\train_mirt_noq.py `
  --dataset Eedi `
  --epoch 50 `
  --batch-size 2048 `
  --device cuda `
  --force
```

输出位置：

```text
ER/KG4ER/data/Eedi/mirt_v8/outputs/
```

会生成：

```text
a_param_K.csv
b_param_K.csv
theta_param_K.csv
mirt_no_q.params
mirt_export_manifest.json
```

其中 `K` 默认等于该数据集的知识点数量，但 no-Q MIRT 中它表示潜在维度，不强制等同于知识点维度。

### 2.3 生成 V8 mastery 文件

```powershell
python ER\KG4ER-New\v8_pipeline\export_mastery_from_mirt_v8.py `
  --dataset Eedi `
  --history-weight 0.5 `
  --force
```

输出位置：

```text
ER/KG4ER/data/Eedi/kt_exports_v8/
```

会生成：

```text
stu2know_mastery.json
mastery_export_manifest.json
```

### 2.4 生成 V8 SemanticConvE 特征文件

```powershell
python ER\KG4ER-New\v8_pipeline\mirt_feature_export.py `
  --dataset Eedi `
  --n-clusters 5 `
  --force
```

输出位置：

```text
ER/KG4ER/data/Eedi/semantic_kg_features_v8/
```

会生成或复制：

```text
entity_features/concept_semantics.json
entity_features/exercise_semantics.json
entity_features/learner_pedagogy.json
irt_features/exercise_irt_features.json
text_embeddings/
feature_generation_manifest.json
```

### 2.5 生成 ER v8 图目录

```powershell
python ER\KG4ER-New\v8_pipeline\build_er_v8.py `
  --dataset Eedi `
  --seed 2024 `
  --top-k-rec 10 `
  --force
```

输出位置：

```text
ER/KG4ER/data/Eedi/er_v8/
```

该目录是后续训练测试读取的正式图目录。

### 2.6 验证 V8 图目录

```powershell
python ER\KG4ER-New\codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi `
  --graph-subdir er_v8 `
  --allow-template
```

## 3. 运行 V8 SemanticConvE 实验

单随机种子快速验证：

```powershell
python ER\KG4ER-New\codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --graph-subdir er_v8 `
  --run-id Eedi_v8_noq_mirt_seed2024 `
  --seeds 2024 `
  --ablations full_state_hybrid `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

五个随机种子：

```powershell
python ER\KG4ER-New\codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --graph-subdir er_v8 `
  --run-id Eedi_v8_noq_mirt_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full_state_hybrid `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

V8 对照/消融实验可以这样跑：

```powershell
python ER\KG4ER-New\codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --graph-subdir er_v8 `
  --run-id Eedi_v8_noq_mirt_ablation_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full_state_hybrid,irt_only_ped,stat_only_ped,no_irt,no_stat_ped,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑命令是在原命令后加 `--resume`：

```powershell
python ER\KG4ER-New\codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --graph-subdir er_v8 `
  --run-id Eedi_v8_noq_mirt_ablation_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full_state_hybrid,irt_only_ped,stat_only_ped,no_irt,no_stat_ped,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 4. 统计结果

```powershell
python ER\KG4ER-New\codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v8_noq_mirt_ablation_5seeds `
  --seeds 2024,2025,2026,2027,2028
```

## 5. 五个数据集批量前置命令

```powershell
$datasets = @("Eedi", "algebra2005", "assist2009-sub", "statics2011", "XES3G5M-sub-small")

foreach ($ds in $datasets) {
  python ER\KG4ER-New\v8_pipeline\prepare_mirt_inputs.py --dataset $ds --strategy all --seed 2024 --force
  python ER\KG4ER-New\v8_pipeline\train_mirt_noq.py --dataset $ds --epoch 50 --batch-size 2048 --device cuda --force
  python ER\KG4ER-New\v8_pipeline\export_mastery_from_mirt_v8.py --dataset $ds --history-weight 0.5 --force
  python ER\KG4ER-New\v8_pipeline\mirt_feature_export.py --dataset $ds --n-clusters 5 --force
  python ER\KG4ER-New\v8_pipeline\build_er_v8.py --dataset $ds --seed 2024 --top-k-rec 10 --force
}
```

