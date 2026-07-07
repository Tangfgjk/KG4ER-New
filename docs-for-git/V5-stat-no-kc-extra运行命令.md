# V5-stat-no-kc-extra 运行命令

本文档对应 GitHub 分支：

```text
V5-stat-no-kc-extra
```

本分支从 V5 改来，主要用于验证：

```text
知识点只用 ID embedding
题目难度/区分度使用统计特征
```

## 1. 拉取代码

如果新电脑还没有仓库：

```powershell
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git checkout V5-stat-no-kc-extra
```

如果已经有仓库：

```powershell
cd KG4ER-New
git fetch origin
git checkout V5-stat-no-kc-extra
git pull
```

如需配置代理：

```powershell
git config --global http.proxy http://127.0.0.1:10090
git config --global https.proxy http://127.0.0.1:10090
```

取消代理：

```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 2. 准备环境

推荐使用之前安装的环境：

```powershell
conda activate kg4er_cuda
```

如果是新环境：

```powershell
conda create -n kg4er_cuda python=3.10 -y
conda activate kg4er_cuda
pip install -r requirements.txt
```

CUDA 版 PyTorch 请按当前电脑显卡驱动安装匹配版本。

## 3. 拷贝数据

把数据集文件夹拷贝到：

```text
KG4ER-New/data/
```

最终目录示例：

```text
KG4ER-New/data/Eedi/
KG4ER-New/data/algebra2005/prepared_for_kt/
KG4ER-New/data/assist2009-sub/prepared_for_kt/
KG4ER-New/data/statics2011/prepared_for_kt/
KG4ER-New/data/XES3G5M-sub-small/prepared_for_kt/
```

每个正式图目录至少应包含：

```text
entities.dict
relations.dict
triples.txt
test_triples.txt
Q.txt
stu2know_mastery.json
stu2know_seq.json
stu2ex_forget.json
sequence_interactions.csv
semantic_kg_features/
```

## 4. 生成统计题目教育特征

五个数据集一起生成：

```powershell
python codes-New-ConvE\build_statistical_pedagogical_features.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --force
```

只生成 Eedi：

```powershell
python codes-New-ConvE\build_statistical_pedagogical_features.py `
  --datasets Eedi `
  --force
```

生成后会得到：

```text
data/{dataset}/.../semantic_kg_features/stat_features/exercise_stat_features.json
```

说明：

```text
alpha 默认是 10
如果 sequence_interactions.csv 有 source_split 列，默认不使用 test 行
如果没有 source_split 列，则使用可用交互记录
```

## 5. 校验数据

五个数据集：

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small
```

只校验 Eedi：

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi
```

看到：

```text
"status": "passed"
```

即可开始训练。

## 6. 跑 Eedi 完整实验

推荐先跑 Eedi 的 8 个实验：

```text
full
concept_extra
no_content_entity
no_relation_aware
no_type_aware_scoring
no_mastery
no_forgetting
no_seq
```

命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_stat_no_kc_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_stat_no_kc_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 7. 跑单个模型

只跑新 full：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_stat_no_kc_full_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

只跑 `concept_extra`：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_stat_no_kc_concept_extra_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations concept_extra `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

## 8. 统计结果

完整实验：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v5_stat_no_kc_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all
```

只统计 full：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v5_stat_no_kc_full_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full
```

结果保存在：

```text
runs/{dataset}/{run_id}/summary/
```

关键文件：

```text
summary.md
paper_table.md
paper_table.csv
gate_values_per_seed.csv
gate_values_mean_std.csv
```

## 9. 多数据集循环命令

```powershell
$datasets = @(
  "Eedi",
  "algebra2005",
  "assist2009-sub",
  "statics2011",
  "XES3G5M-sub-small"
)

foreach ($ds in $datasets) {
  python codes-New-ConvE\build_statistical_pedagogical_features.py `
    --datasets $ds `
    --force

  python codes-New-ConvE\run_semantic_experiments.py `
    --dataset $ds `
    --run-id "${ds}_v5_stat_no_kc_5seeds" `
    --seeds 2024,2025,2026,2027,2028 `
    --ablations all `
    --epochs 25 `
    --bs 1024 `
    --negative-ratio 5 `
    --cuda auto `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "Dataset $ds failed."
  }
}
```

## 10. 结果目录

训练、测试和评价结果：

```text
runs/{dataset}/{run_id}/SemanticConvE/seed{seed}/
runs/{dataset}/{run_id}/SemanticConvE_concept_extra/seed{seed}/
runs/{dataset}/{run_id}/SemanticConvE_no_mastery/seed{seed}/
...
```

图级认知消融数据：

```text
ablation_data/{dataset}/no_mastery_top10/
ablation_data/{dataset}/no_forgetting_top10/
ablation_data/{dataset}/no_seq_top10/
```

这些运行结果目录不会上传 GitHub。
