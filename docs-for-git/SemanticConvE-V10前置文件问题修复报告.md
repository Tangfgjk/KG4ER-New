# SemanticConvE V10 前置文件问题修复报告

本文记录 V10 相对 V8/V9 前置文件生成流程解决的问题、具体处理方式和涉及代码。V10 的核心原则是：原始数据继续保留在 `ER/KG4ER/data/`，新的前置文件统一输出到 `ER/KG4ER-New/data/<dataset>/v10/`，不覆盖旧实验数据。

## 1. Eedi 的 stu2know_seq.json 异常

### 问题

此前统计发现 Eedi 的 `stu2know_seq.json` 几乎全部集中在 0.95 以上，且不同行数极少，明显不同于其他数据集。这说明 sequence/progress 特征没有形成有效区分。

### V10 处理

V10 不再直接沿用旧 `stu2know_seq.json` 作为正式文件，而是提供独立导出脚本：

```text
ER/KG4ER-New/v10_pipeline/export_seq_from_dkt_v10.py
```

该脚本要求重新训练或指定 pyKT DKT checkpoint，然后从 DKT 在测试学生序列上的预测结果导出 `stu2know_seq.json`。

导出结果保存到：

```text
ER/KG4ER-New/data/<dataset>/v10/kt_dkt/stu2know_seq.json
ER/KG4ER-New/data/<dataset>/v10/stu2know_seq.json
```

同时新增校验脚本会检查 `stu2know_seq.json` 的最小值、最大值、均值、标准差、不同行比例。如果再次出现 Eedi 这种集中在 0.95 以上且行重复严重的情况，会在报告中给出 warning。

## 2. Mastery 与论文描述不一致

### 问题

论文描述中，知识掌握度应来自 text-enhanced KT 模型，即结合学生历史答题序列、题目文本表示、题目知识点、MIRT 参数和 GRU 状态后输出的知识点掌握概率。但之前流程中部分数据集使用了简化或替代方式，和论文描述不完全一致。

### V10 处理

V10 增加严格导入入口：

```text
ER/KG4ER-New/v10_pipeline/import_ektm_mastery_v10.py
```

正式 V10 中，`stu2know_mastery.json` 应由 EKTM_mirt 或其 know-output 导出脚本产生的 `know_output` 矩阵导入。`know_output` 表示模型在测试学生最后状态上输出的知识点掌握概率。

导入后保存到：

```text
ER/KG4ER-New/data/<dataset>/v10/ektm_mirt/stu2know_mastery.json
ER/KG4ER-New/data/<dataset>/v10/stu2know_mastery.json
```

如果暂时还没有 EKTM_mirt 导出矩阵，可以使用图构建脚本的 `--allow-existing-state` 做调试，但不能作为正式实验。

## 3. stu2ex_forget.json 超过 1

### 问题

旧流程中 `stu2ex_forget.json` 由 `stu2know_forget.json` 与 Q 矩阵做矩阵乘法得到，本质是“题目涉及知识点遗忘率求和”。如果一道题涉及多个知识点，结果可能超过 1，例如 Algebra 2005 和 XES3G5M-sub-small 出现过范围到 30 或 6 的情况。

只做 `clip(0, 1)` 不合理，因为 1.5、6、30 都会被压成 1，仍然丢失区分度。

### V10 处理

V10 从 `stu2know_forget.json + Q.txt` 重新计算题目遗忘率，改为当前题目所涉及知识点遗忘率的平均值：

```text
exercise_forget(u, e) = mean_{k in KC(e)} know_forget(u, k)
```

涉及代码：

```text
ER/KG4ER-New/v10_pipeline/build_v10_graph.py
ER/KG4ER-New/v10_pipeline/generate_forgetting_v10.py
```

这样 `stu2ex_forget.json` 理论范围保持在 0 到 1，同时保留多知识点题目的差异。

## 4. 推荐距离中 sequence 项方向问题

### 问题

旧推荐距离中 sequence 项为：

```text
cos(Q_j, seq_i)^2
```

由于推荐距离越小越优，若题目知识点 Q 与学生进展向量 seq 越相似，cos 越大，距离反而越大。这和“更符合学习进展的题目应更容易推荐”的直觉相反。

### V10 处理

V10 默认改为：

```text
(1 - cos(Q_j, seq_i))^2
```

因此 Q 与 seq 越相似，该项越小，题目越容易进入推荐集合。

涉及代码：

```text
ER/KG4ER-New/v10_pipeline/build_v10_graph.py
```

同时脚本也会额外输出一个 `v10_rec_legacy_cos/` 目录，用于保存旧公式构图结果，方便后续做公式方向对照实验。

## 5. 推荐距离排序精度问题

### 问题

如果推荐距离先四舍五入再排序，会产生大量并列分数。并列后排序可能退化为按题目 ID 排序，而不是按真实距离排序。

### V10 处理

V10 在排序时使用完整精度距离；保存到 `stu2ex_recommend.json` 时只做可读性四舍五入。用于构图的推荐 Top-K 来源于完整精度排序结果。

同时 `relations.dict` 仍保持 304 个关系，即 `rec` 加 `mlkc/pkc/exfr` 三类 0.00 到 1.00 的两位小数关系标签，保证和原论文图结构规模一致。

## 6. Eedi-sub 与完整 Eedi 混用问题

### 问题

之前 MIRT 输入曾出现 Eedi 有 4870 个学生的问题，而当前 ER 图中的 Eedi-sub 只有 935 个 learner。这说明 MIRT 特征估计时误用了完整 Eedi 或未对齐 ER 子集。

### V10 处理

V10 新增 MIRT 输入准备脚本：

```text
ER/KG4ER-New/v10_pipeline/prepare_mirt_inputs_v10.py
```

该脚本只保留 ER 图中出现的 learner，并保持题目 ID 与 ER 图中的 `ex0...exN` 对齐，不再重新压缩题目 ID。

MIRT 输入保存到：

```text
ER/KG4ER-New/data/<dataset>/v10/mirt/inputs/
```

## 7. 语义和教育学特征的处理

### 固定复用

知识点 definition、题目文本或结构化文本、BGE 文本向量属于固定语义文件。DeepSeek 已生成的 definition 不需要重复调用，BGE 向量也不需要重复生成。

### 重新生成

与学生状态或教育学参数相关的文件需要基于 V10 重新生成，包括：

```text
semantic_kg_features/entity_features/learner_pedagogy.json
semantic_kg_features/irt_features/exercise_irt_features.json
feature_generation_manifest.json
```

涉及代码：

```text
ER/KG4ER-New/v10_pipeline/export_mirt_features_v10.py
```

## 8. 新增统计与校验

V10 新增统一校验脚本：

```text
ER/KG4ER-New/v10_pipeline/validate_v10_front_files.py
```

校验内容包括：

- learner、exercise、concept、relation 数量；
- `stu2know_mastery.json`、`stu2know_seq.json`、`stu2know_forget.json`、`stu2ex_forget.json` 的形状；
- 数值是否超出 0 到 1；
- `stu2know_seq.json` 是否过度集中或重复；
- `stu2ex_forget.json` 是否仍然超过 1；
- `relations.dict` 是否保持 304；
- 推荐分数排序是否存在大量四舍五入并列风险；
- train/test triples 的关系分布；
- 语义特征文件和文本向量是否存在。

输出位置：

```text
ER/KG4ER-New/data/v10_validation_report.json
```

## 9. 训练策略暂不纳入 V10 前置文件

参考 KG4EX 代码后可以确认，其训练阶段对所有关系做负采样，并使用 subsampling weight 降低高频三元组影响。但这属于训练策略，不属于前置文件生成。

V10 先专注前置文件修复。后续如需做训练策略消融，建议比较：

```text
rec-only negative sampling
all-relation type-constrained negative sampling
```

其中更推荐类型约束负采样：

- `rec` 替换 tail 时只采 exercise；
- `mlkc/pkc/exfr` 替换 tail 时只采 learner。

这样比 KG4EX 的全实体负采样更符合当前图中实体类型语义。

