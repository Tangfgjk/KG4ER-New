# SemanticConvE V8 代码说明

V8 新增代码位于：

```text
ER/KG4ER-New/v8_pipeline/
```

V8 的核心目标是：把题目难度、区分度、学习者能力等教育测量特征统一到修改后的 EduCDM no-Q MIRT 输出，减少早期自定义 2PL-IRT 与 KT/MIRT 口径不一致的问题。

## 1. 新增脚本

| 文件 | 作用 | 主要输入 | 主要输出 |
|---|---|---|---|
| `common.py` | 公共路径、JSON、Q 矩阵、归一化和拷贝工具 | 数据集目录 | 被其他脚本调用 |
| `prepare_mirt_inputs.py` | 从 `sequence_interactions.csv` 生成 EduCDM MIRT 输入 | `sequence_interactions.csv`、`Q.txt` | `mirt_v8/inputs/*.csv` |
| `train_mirt_noq.py` | 调用 `EduCDM_MIRT_noQ_export_modified` 训练 no-Q MIRT | `mirt_v8/inputs/train.csv` | `a_param_K.csv`、`b_param_K.csv`、`theta_param_K.csv` |
| `export_mastery_from_mirt_v8.py` | 生成 V8 `stu2know_mastery.json` | MIRT `theta/b`、Q、作答历史 | `kt_exports_v8/stu2know_mastery.json` |
| `mirt_feature_export.py` | 生成 SemanticConvE 侧特征文件 | MIRT `a/b/theta`、V8 mastery、旧语义文件 | `semantic_kg_features_v8/` |
| `build_er_v8.py` | 组装 V8 ER 图目录并构造推荐边/三元组 | V8 mastery、seq/forget、V8 特征 | `er_v8/` |
| `test_v8_pipeline.py` | V8 纯函数单元测试 | 小型内存数据 | pytest 通过 |

## 2. 数据目录

V8 不覆盖旧数据，而是在每个数据集下新增：

```text
ER/KG4ER/data/<dataset>/mirt_v8/
ER/KG4ER/data/<dataset>/kt_exports_v8/
ER/KG4ER/data/<dataset>/semantic_kg_features_v8/
ER/KG4ER/data/<dataset>/er_v8/
```

训练和测试最终读取：

```text
ER/KG4ER/data/<dataset>/er_v8/
```

## 3. 特征来源

### 题目特征

| 特征 | 来源 |
|---|---|
| `difficulty_mirt` / `difficulty_norm` | no-Q MIRT 的 `b_param_K.csv` |
| `discrimination_mirt` / `discrimination_norm` | no-Q MIRT 的 `a_param_K.csv` 每行 L2 范数 |
| `kc_discrimination_vector` | no-Q MIRT 的整行 `a` 向量，当前主要用于保存和分析 |
| `correct_rate`、`interaction_count` | `mirt_v8/inputs/all.csv` |
| 文本/结构化语义 | 优先复制旧 `semantic_kg_features/text_embeddings` 和 `exercise_semantics.json` |

### 学习者特征

| 特征 | 来源 |
|---|---|
| `theta_mirt_vector` | no-Q MIRT 的 `theta_param_K.csv` |
| `theta_norm` | `theta` 均值归一化 |
| `overall_mastery_irt` | 当前等于归一化后的 `theta` 能力代理 |
| `overall_mastery_kt_mean` | V8 `stu2know_mastery.json` 的知识点均值 |
| `cluster_id` | 基于 `theta` 与 mastery 聚合特征做 K-Means |

### 知识点特征

知识点仍使用旧的 `concept_semantics.json` 与文本 embedding。V8 默认不把 MIRT 的 `a/b` 直接当知识点特征，因为 no-Q MIRT 的潜在维度不严格对应知识点。

## 4. 当前 mastery 导出说明

当前版本的 `export_mastery_from_mirt_v8.py` 使用 `mirt_proxy`：

```text
theta_param_K.csv + b_param_K.csv + Q.txt + 学生作答历史
-> stu2know_mastery.json
```

它不是严格的 EKTM_mirt checkpoint 推理结果。这样做的原因是：当前还没有针对五个数据集统一训练好的 EKTM_mirt checkpoint 和文本字典适配文件。代码保留了独立输出目录，后续可以用严格 EKTM_mirt 导出的 `stu2know_mastery.json` 直接替换：

```text
ER/KG4ER/data/<dataset>/kt_exports_v8/stu2know_mastery.json
```

替换后重新执行：

```text
mirt_feature_export.py
build_er_v8.py
run_semantic_experiments.py --graph-subdir er_v8
```

## 5. 训练入口修改

`run_semantic_experiments.py` 新增：

```text
--graph-subdir er_v8
```

该参数会让训练、测试、评估统一读取：

```text
ER/KG4ER/data/<dataset>/er_v8/
```

旧命令不加 `--graph-subdir` 时仍按原逻辑读取旧数据目录。

