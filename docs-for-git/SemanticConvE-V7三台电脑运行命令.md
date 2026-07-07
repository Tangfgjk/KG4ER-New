# SemanticConvE V7 三台电脑运行命令

本文档用于在三台电脑上分组运行 Eedi 数据集的 V7 完整消融实验。每台电脑只跑一部分模型，最后把三个结果文件夹合并后统一统计。

## 1. 拉取 V7 代码

```powershell
cd "C:\Users\你的用户名\Desktop\ER"

git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V7
git pull origin V7
```

如果文件夹已经存在：

```powershell
cd "C:\Users\你的用户名\Desktop\ER\KG4ER-New"
git fetch origin
git checkout V7
git pull origin V7
```

## 2. 拷贝数据

在每台电脑上，把本地已经生成好的 Eedi 数据集文件夹拷贝到：

```text
KG4ER-New\data\Eedi
```

拷贝后至少应包含：

```text
data\Eedi\entities.dict
data\Eedi\relations.dict
data\Eedi\triples.txt
data\Eedi\test_triples.txt
data\Eedi\Q.txt
data\Eedi\stu2know_mastery.json
data\Eedi\stu2know_seq.json
data\Eedi\stu2know_forget.json
data\Eedi\stu2ex_forget.json
data\Eedi\semantic_kg_features\
```

## 3. 激活环境

本机之前新建的 CUDA 环境名是：

```powershell
conda activate kg4er_cuda
```

如果在其他电脑上使用已有环境，也需要确保安装了 `requirements.txt` 中的依赖：

```powershell
pip install -r requirements.txt
```

## 4. 公共参数

五个随机种子统一使用：

```text
2024,2025,2026,2027,2028
```

统一 run id：

```text
Eedi_v7_attention_5seeds
```

所有命令都在仓库根目录执行：

```powershell
cd "C:\Users\你的用户名\Desktop\ER\KG4ER-New"
```

## 5. 电脑 1：主模型、ID-only、直接相加、全部文本消融

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,direct_sum_fusion,no_text_semantic `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,direct_sum_fusion,no_text_semantic `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 6. 电脑 2：知识点文本、题目文本、教育特征、关系表示消融

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_concept_text,no_exercise_text,no_pedagogical,no_relation_aware `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_concept_text,no_exercise_text,no_pedagogical,no_relation_aware `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 7. 电脑 3：type-aware scoring 和认知项消融

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 8. 结果合并

三台电脑运行完成后，把各自的结果子文件夹合并到同一个目录：

```text
KG4ER-New\runs\Eedi\Eedi_v7_attention_5seeds
```

不要只覆盖整个 `Eedi_v7_attention_5seeds` 文件夹，建议按模型子文件夹合并，例如：

```text
SemanticConvE\
SemanticConvE_id_only\
SemanticConvE_direct_sum_fusion\
...
```

## 9. 统一统计结果

在合并后的电脑上运行：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v7_attention_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,direct_sum_fusion,no_text_semantic,no_concept_text,no_exercise_text,no_pedagogical,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq
```

统计结果会保存到：

```text
runs\Eedi\Eedi_v7_attention_5seeds\summary\
```

重点查看：

```text
paper_table.md
paper_table.csv
mean_std_metrics.csv
per_seed_metrics.csv
gate_values_mean_std.csv
```

