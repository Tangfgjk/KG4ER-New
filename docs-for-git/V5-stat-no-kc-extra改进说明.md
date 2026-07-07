# V5-stat-no-kc-extra 改进说明

本分支从 `origin/V5` 创建，不包含 V6 的 state-aware head encoder 改动。目标是在保留 V5 训练、测试、负采样、评价流程的基础上，只验证两个更小、更可控的改动：

1. 知识点实体只使用 ID embedding；
2. 题目难度和区分度由 IRT 特征改为统计特征。

## 1. 知识点实体只用 ID

V5 原始做法中，所有实体都会以 ID embedding 为主体，再通过 gate 融合文本语义、教育学数值、实体类型和学习者聚类等补充特征。

本分支中，默认 `full` 模型对知识点实体采用：

```text
h_kc = LayerNorm(ID_kc)
```

也就是说，知识点不再融合：

```text
concept name / definition semantic embedding
entity type embedding
numeric feature projection
cluster embedding
```

这样做的原因是：当前 KT 状态文件主要使用知识追踪模型输出的学生-知识点掌握、序列和遗忘信息，并没有直接使用知识点文本定义。若继续把知识点 definition 文本注入图嵌入，可能会引入和 KT 状态来源不一致的噪声。

为了保留对照实验，本分支新增：

```text
concept_extra
```

该 ablation 会恢复 V5 旧逻辑，即知识点也融合额外语义和类型特征，可用于观察“知识点额外特征是否真的有帮助”。

## 2. 题目统计教育特征

V5 原始题目教育特征使用 `irt_features/exercise_irt_features.json` 中的：

```text
difficulty_norm
discrimination_norm
correct_rate
interaction_count
```

本分支新增脚本：

```text
codes-New-ConvE/build_statistical_pedagogical_features.py
```

脚本从 `sequence_interactions.csv` 统计每道题的作答表现，生成：

```text
semantic_kg_features/stat_features/exercise_stat_features.json
```

核心特征包括：

```text
difficulty_stat_norm
discrimination_stat_norm
correct_rate
interaction_count
high_group_correct_rate
low_group_correct_rate
error_rate
```

其中：

```text
difficulty_stat_norm = smoothed error rate
```

即题目的平滑错误率。默认平滑参数：

```text
alpha = 10
```

区分度使用高能力学生组和低能力学生组的正确率差：

```text
discrimination_stat_norm = max(0, correct_rate_high_group - correct_rate_low_group)
```

高、低能力学生组由学生总体正确率的 73% 和 27% 分位划分。

如果交互文件中存在 `source_split` 列，脚本默认排除 `test` 行，只用训练侧交互统计题目特征；如果没有该列，则使用可用交互记录。

## 3. Feature Loader 行为

`feature_loader.py` 现在优先读取：

```text
semantic_kg_features/stat_features/exercise_stat_features.json
```

如果该文件不存在，才回退到：

```text
semantic_kg_features/irt_features/exercise_irt_features.json
```

正式实验建议先运行统计特征生成脚本，再运行 `validate_semantic_ready.py`。校验脚本已将统计特征文件列为必需文件。

## 4. 模型版本

本分支模型版本为：

```text
semantic_conve_v5_stat_no_kc_extra
```

由于模型版本和 V5/V6 均不同，旧 checkpoint 不建议复用。正式运行请使用新的 `run-id`。

## 5. 建议对比

建议至少运行：

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

其中：

```text
full
```

表示本分支的新完整模型。

```text
concept_extra
```

表示恢复知识点额外语义/类型特征，用来判断知识点额外特征是否引入噪声。

```text
no_content_entity
no_relation_aware
no_type_aware_scoring
```

分别对应三个新增创新点的消融。

```text
no_mastery
no_forgetting
no_seq
```

对应原论文认知状态三因素消融。
