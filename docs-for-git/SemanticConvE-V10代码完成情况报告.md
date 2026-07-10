# SemanticConvE V10 代码完成情况报告

## 1. 已完成的主要修改

### 1.1 V10 前置文件目录

新增并完善：

```text
ER/KG4ER-New/v10_pipeline/
```

所有 V10 生成结果保存到：

```text
ER/KG4ER-New/data/<dataset>/v10/
```

旧数据目录不覆盖。

### 1.2 MIRT 教育学特征

已保留并修复：

```text
v10_pipeline/prepare_mirt_inputs_v10.py
v10_pipeline/train_mirt_noq_v10.py
v10_pipeline/export_mirt_features_v10.py
```

修复点：

- `common.py` 新增 `minmax()`，解决 `export_mirt_features_v10.py` 无法导入的问题。
- MIRT 输出继续作为教育学特征估计器，不作为最终推荐模型。
- 学生侧最终只使用 `theta_norm` 和 `cluster_id`，不再将 `overall_mastery_mirt` 融入实体表示，避免和 mastery 关系重复。

### 1.3 PKC-DKT 重新训练

新增：

```text
v10_pipeline/pkc_dkt_model_v10.py
v10_pipeline/train_pkc_dkt_v10.py
```

作用：

- 不再直接依赖外部未知训练目标的 pyKT DKT checkpoint。
- 在 V10 中重新训练下一步知识点出现概率预测模型。
- 输出新的 `stu2know_seq.json`。

### 1.4 EKTM_mirt 导出

完善：

```text
v10_pipeline/export_ektm_outputs_v10.py
```

当前导出：

- 题目 `topic_v` 作为 `exercise_text_embeddings.npy`。
- EKTM_mirt 的知识点 embedding 作为 `concept_text_embeddings.npy`，并对齐到 `topic_v` 维度。
- EKTM_mirt 的 `know_output` 作为 `stu2know_mastery.json`。

说明：

- 已删除 BGE/SentenceTransformer 文本 embedding 依赖。
- 文本/语义表示优先来自 EKTM_mirt 内部模型。

### 1.5 遗忘文件

完善：

```text
v10_pipeline/generate_forgetting_v10.py
```

修复点：

- 数值时间戳统一转换到秒。
- 新增 `--timestamp-unit auto|seconds|milliseconds|minutes|days`。
- `stu2ex_forget.json` 使用题目关联知识点遗忘率平均值，不再使用求和后截断。
- manifest 中记录时间间隔分布。

### 1.6 推荐公式和构图

完善：

```text
v10_pipeline/build_v10_graph.py
```

修复点：

- 默认 sequence 项使用 `(1 - cos(Q_j, seq_i))^2`。
- 推荐排序使用完整精度，最终存储再保留小数。
- 关系数仍固定为 304。
- `v10_rec_legacy_cos/` 现在也会生成完整图文件，而不只是推荐分数文件。

### 1.7 V10-attn SemanticConvE

重写核心模型：

```text
codes-New-ConvE/semantic_conve_model.py
```

实现：

- 实体表示：`ID embedding + Attention(side feature tokens)`。
- 关系表示：`relation ID embedding + Attention(relation type, relation strength)`。
- 去掉 V9 的 state-aware head 和实体 type token。
- 保留 restricted-space scoring，并新增 `no_type_aware_scoring` 消融。

## 2. V10 完整模型特征

### 学生实体

```text
uid = ID + Attention(theta_mirt_norm, cluster_id)
```

### 题目实体

```text
ex = ID + Attention(topic_v/text, difficulty_mirt_norm, discrimination_mirt_norm)
```

### 知识点实体

```text
kc = ID + Attention(concept_semantic)
```

这里的 `concept_semantic` 来自 EKTM_mirt 知识点 embedding。

### 关系

```text
r = relation ID + Attention(relation_type, relation_strength)
```

## 3. V10 消融实验

默认 `--ablations all` 等价于：

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

其中 `no_type_aware_scoring` 是新增的 restricted-space scoring 消融。

## 4. 仍按计划暂不修改的内容

### no-Q MIRT 维度不严格对应知识点

状态：暂时不需要修改。

原因：

- V10 当前使用 `a` 向量的 L2 norm 作为题目区分度。
- 不直接把每一维解释为具体知识点。
- 因此该问题主要是命名和解释风险，不阻断当前实验。

## 5. 已完成测试

已通过：

```text
python -m py_compile ...
python v10_pipeline/export_mirt_features_v10.py --help
python v10_pipeline/train_pkc_dkt_v10.py --help
python v10_pipeline/export_ektm_outputs_v10.py --help
python codes-New-ConvE/run_semantic_experiments.py --help
python -m unittest test_semantic_conve_model test_semantic_experiment_utils
python -m pytest v10_pipeline/test_v10_pipeline.py
```

已完成一次 dry-run：

```text
run_semantic_experiments.py --dry-run --ablations full,no_type_aware_scoring,id_only
```

结果：runner 能生成训练、测试和评估阶段命令。
