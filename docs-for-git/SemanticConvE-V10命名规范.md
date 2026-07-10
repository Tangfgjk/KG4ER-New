# SemanticConvE V10 命名规范

## 1. 前置文件目录

V10 所有新生成的前置文件统一放在：

```text
ER/KG4ER-New/data/<dataset>/v10/
```

不要覆盖旧目录：

```text
ER/KG4ER/data/<dataset>/
ER/KG4ER/data/<dataset>/prepared_for_kt/
```

## 2. 前置文件子目录

| 目录 | 含义 |
|---|---|
| `mirt/inputs/` | MIRT 输入文件 |
| `mirt/outputs/` | no-Q MIRT 输出参数 |
| `pkc_dkt/` | V10 PKC-DKT checkpoint 和 `stu2know_seq.json` |
| `ektm_mirt/exports/` | EKTM_mirt 导出的 `topic_v` 和 `know_output` |
| `forgetting/` | 重新计算的遗忘文件 |
| `semantic_kg_features/` | SemanticConvE 训练读取的实体/关系侧特征 |

## 3. 关键前置文件

| 文件 | 来源 |
|---|---|
| `stu2know_mastery.json` | EKTM_mirt 的 `know_output` |
| `stu2know_seq.json` | V10 PKC-DKT 的下一步知识点出现概率 |
| `stu2know_forget.json` | 历史答题时间间隔的知识点遗忘率 |
| `stu2ex_forget.json` | 题目关联知识点遗忘率平均值 |
| `stu2ex_recommend.json` | 推荐距离，保留 6 位小数用于存储 |
| `stu2ex_recommend_full_precision.json` | 完整精度推荐距离，用于排序检查 |
| `triples.txt` | 训练三元组 |
| `test_triples.txt` | 测试学生状态三元组 |
| `entities.dict` | 实体 ID 映射 |
| `relations.dict` | 固定 304 个关系 |

## 4. 推荐公式目录

主目录：

```text
data/<dataset>/v10/
```

使用：

```text
(1 - cos(Q_j, seq_i))^2
```

旧公式对照目录：

```text
data/<dataset>/v10_rec_legacy_cos/
```

使用：

```text
cos(Q_j, seq_i)^2
```

两个目录都会生成完整 ER 图文件，可以分别训练。

## 5. SemanticConvE 消融模型名

| 模型名 | 含义 |
|---|---|
| `full` | 完整 V10 模型 |
| `id_only` | 实体和关系都只用 ID embedding |
| `no_pedagogical` | 去掉学生 theta、cluster 和题目 MIRT 难度/区分度 |
| `no_text_semantic` | 去掉题目和知识点文本/语义表示 |
| `no_concept_semantic` | 只去掉知识点语义表示 |
| `no_relation_aware` | 关系只用 relation ID，不融合类型和强度 |
| `no_type_aware_scoring` | 测试时对所有实体打分，再筛选习题 |
| `no_mastery` | 构图时去掉掌握度关系 |
| `no_forgetting` | 构图时去掉遗忘关系 |
| `no_seq` | 构图时去掉序列关系 |

## 6. 运行 ID 命名

推荐主实验：

```text
<dataset>_v10_attn_5seeds
```

推荐对比实验：

```text
<dataset>_v10_comparison_5seeds
```

单次调试：

```text
<dataset>_v10_debug_<说明>
```

## 7. 结果目录

SemanticConvE：

```text
runs/<dataset>/<run-id>/SemanticConvE/seed2024/
runs/<dataset>/<run-id>/SemanticConvE_no_pedagogical/seed2024/
```

对比模型：

```text
runs/<dataset>/<run-id>/TransE/seed2024/
runs/<dataset>/<run-id>/RotatE/seed2024/
```

## 8. GitHub 分支

V10 代码提交到：

```text
V10
```

发布仓库只上传代码和必要文档，不上传：

- `data/`
- `runs/`
- `__pycache__/`
- checkpoint 模型文件
