# Eedi V5 两台电脑实验运行命令

本文档用于在两台电脑上并行运行 Eedi-sub 数据集的 SemanticConvE V5 实验。V5 正式实验共 7 个：

```text
full
no_content_entity
no_relation_aware
no_type_aware_scoring
no_mastery
no_forgetting
no_seq
```

两台电脑分工如下：

| 电脑 | 实验组 | 模型 |
|---|---|---|
| 电脑 1 | 主模型 + 三个创新点消融 | `full`, `no_content_entity`, `no_relation_aware`, `no_type_aware_scoring` |
| 电脑 2 | 三个认知因素消融 | `no_mastery`, `no_forgetting`, `no_seq` |

## 1. 拉取 V5 代码

如果电脑上还没有仓库：

```powershell
cd "C:\Users\你的用户名\Desktop"
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V5
```

如果电脑上已经有仓库：

```powershell
cd "C:\Users\你的用户名\Desktop\KG4ER-New"
git fetch origin
git checkout V5
git pull origin V5
```

如果 GitHub 连接代理端口为 `127.0.0.1:10090`，可以先设置：

```powershell
git config --global http.proxy http://127.0.0.1:10090
git config --global https.proxy http://127.0.0.1:10090
```

## 2. 拷贝 Eedi 数据

GitHub 仓库不上传数据。需要手动把本地已有的 Eedi 数据集复制到新仓库：

源文件夹：

```text
C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New\ER\KG4ER\data\Eedi
```

目标文件夹：

```text
KG4ER-New\data\Eedi
```

复制后目标目录中至少应包含：

```text
entities.dict
relations.dict
triples.txt
test_triples.txt
Q.txt
stu2know_mastery.json
stu2know_seq.json
stu2know_forget.json
stu2ex_forget.json
semantic_kg_features/
```

其中 `semantic_kg_features/` 是 V5 所需的语义和教育学特征目录，必须一起复制。

## 3. 进入运行目录

所有命令都在仓库根目录下执行：

```powershell
cd "C:\Users\你的用户名\Desktop\KG4ER-New"
```

激活环境，例如：

```powershell
conda activate kg4er_cuda
```

如果你的环境名是 `rtx5060`，则使用：

```powershell
conda activate rtx5060
```

## 4. 电脑 1：主模型 + 三个创新点消融

电脑 1 运行：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_part1_innovation_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,no_content_entity,no_relation_aware,no_type_aware_scoring `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0.2 `
  --cuda auto
```

续跑命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_part1_innovation_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,no_content_entity,no_relation_aware,no_type_aware_scoring `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0.2 `
  --cuda auto `
  --resume
```

## 5. 电脑 2：三个认知因素消融

电脑 2 运行：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_part2_cognitive_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0.2 `
  --cuda auto
```

续跑命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v5_part2_cognitive_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --forgetting-score-weight 0.2 `
  --cuda auto `
  --resume
```

## 6. 结果保存位置

电脑 1 结果：

```text
KG4ER-New\runs\Eedi\Eedi_v5_part1_innovation_5seeds
```

电脑 2 结果：

```text
KG4ER-New\runs\Eedi\Eedi_v5_part2_cognitive_5seeds
```

每个模型和随机种子会保存到：

```text
runs\Eedi\{run-id}\SemanticConvE_{ablation}\seed{seed}
```

其中 `full` 的目录名仍为：

```text
SemanticConvE
```

## 7. 单独统计结果

电脑 1 跑完后统计：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v5_part1_innovation_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,no_content_entity,no_relation_aware,no_type_aware_scoring
```

电脑 2 跑完后统计：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v5_part2_cognitive_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations no_mastery,no_forgetting,no_seq
```

统计结果位于：

```text
runs\Eedi\{run-id}\summary\
```

主要查看：

```text
paper_table.md
paper_table.csv
summary.json
per_seed_metrics.csv
gate_values_mean_std.csv
```

## 8. 运行注意事项

1. V5 的 `MODEL_VERSION` 已改变，不能用 V4 的 checkpoint 续跑。
2. 如果已有同名 run-id，但里面是旧代码结果，建议换新 run-id 或删除旧目录。
3. `--resume` 会自动跳过已完成的 seed，并从 `last.pt` 尝试续跑未完成训练。
4. `no_mastery`、`no_forgetting`、`no_seq` 会自动生成独立消融图数据，保存在：

```text
KG4ER-New\ablation_data\Eedi\
```

5. `no_forgetting` 会自动关闭显式 forgetting score 分支。
6. `no_type_aware_scoring` 会在测试阶段先对全实体打分，再筛选习题实体结果。
