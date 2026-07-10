# SemanticConvE V10 创新特征来源核查

本文核查当前 V10 / SemanticConvE 代码中三个创新点所使用的特征来源，并标出仍需修正的地方。

## 1. 总体结论

当前 V10 已经将文本向量来源改为硬约束：SemanticConvE 不再接受早期 BGE-M3 / SentenceTransformer 向量。

正式 V10 必须先通过：

```text
ER/KG4ER-New/v10_pipeline/import_ektm_topic_embeddings_v10.py
```

导入 EKTM_mirt 中 `TopicRNNModel` / Bi-GRU 生成的题目文本表示 `topic_v`，再进入 SemanticConvE 训练。

## 2. 当前新增特征来源表

| 创新点 | 实体/关系 | 当前使用的特征 | 当前来源 | 是否与 EKTM_mirt 一致 | 结论 |
|---|---|---|---|---|---|
| Pedagogically content-aware Entity Representation | 知识点 kc | ID embedding | `entities.dict` 中的 `kc*` | 是，基础 ID | 保留 |
| Pedagogically content-aware Entity Representation | 知识点 kc | concept semantic text / definition metadata | `concept_semantics.json`，definition 由 DeepSeek 或模板生成 | 部分一致 | definition 可作为补充解释文本，但 EKTM_mirt 本身没有使用知识点 definition |
| Pedagogically content-aware Entity Representation | 知识点 kc | concept text embedding | V10 写入零向量 | 是 | EKTM_mirt 本身不使用知识点 definition 做文本编码，因此不再引入 BGE 知识点向量 |
| Pedagogically content-aware Entity Representation | 题目 ex | ID embedding | `entities.dict` 中的 `ex*` | 是，基础 ID | 保留 |
| Pedagogically content-aware Entity Representation | 题目 ex | exercise semantic text | `exercise_semantics.json` 中的题干文本或结构化文本 | 输入文本本身一致 | 保留文本元数据 |
| Pedagogically content-aware Entity Representation | 题目 ex | exercise text embedding | EKTM_mirt `TopicRNNModel(topic)` 输出的 `topic_v` | 是 | 由 `import_ektm_topic_embeddings_v10.py` 导入，BGE 已禁用 |
| Pedagogically content-aware Entity Representation | 题目 ex | difficulty | V10 MIRT 输出 `b_param_K.csv`，归一化后写入 `exercise_irt_features.json` | 部分一致 | 与 EKTM_mirt 所需 MIRT 参数体系一致，但需确认 EKTM_mirt 使用同一批 a/b 参数 |
| Pedagogically content-aware Entity Representation | 题目 ex | discrimination | V10 MIRT 输出 `a_param_K.csv`，取 L2 norm 后归一化 | 部分一致 | 与 MIRT 参数体系一致；如论文写区分度，建议说明由 MIRT `a` 向量聚合得到 |
| Pedagogically content-aware Entity Representation | 学习者 uid | ID embedding | `entities.dict` 中的 `uid*` | 是，基础 ID | 保留 |
| Pedagogically content-aware Entity Representation | 学习者 uid | theta / ability | V10 MIRT 输出 `theta_param_K.csv` | 部分一致 | 与 MIRT 特征体系一致，需保持与 EKTM_mirt 使用的 a/b/theta 同源 |
| Pedagogically content-aware Entity Representation | 学习者 uid | 综合掌握度 | `stu2know_mastery.json` 的均值；严格 V10 要来自 EKTM_mirt `know_output` | 计划一致 | 只有当 mastery 严格由 EKTM_mirt 导出时才完全一致 |
| Pedagogically content-aware Entity Representation | 学习者 uid | cluster_id | 基于 theta、mastery 均值、mastery 方差等 KMeans 聚类 | 不属于 EKTM_mirt 原生输出 | 是我们后处理生成的学习者画像特征 |
| Relation-aware Encoding | 关系 r | relation type | 从关系名解析 `rec/mlkc/pkc/exfr` | 与图结构一致 | 保留 |
| Relation-aware Encoding | 关系 r | relation strength | 从关系名解析 0.00 到 1.00 的连续强度 | 与 V10 关系强度设计一致 | 保留 |
| Relation-aware Encoding | 关系 r | relation ID embedding | 当前 full 默认不使用；`hybrid_relation` 使用 | 原始 ConvE 使用 | 建议后续 full 使用 `ID + type + strength`，减少表达能力损失 |
| Type-aware ConvE | 推荐打分 | 只对 exercise tail 打分 | `score_tails(..., tail_ids=exercise_entity_ids)` | 与推荐任务一致 | 保留 |

## 3. 当前代码中各特征实际读取位置

### 3.1 文本向量

当前读取位置：

```text
ER/KG4ER-New/codes-New-ConvE/feature_loader.py
```

核心函数：

```text
_load_text_embeddings(feature_dir)
```

它读取：

```text
semantic_kg_features/text_embeddings/text_embedding_manifest.json
semantic_kg_features/text_embeddings/concept_text_embeddings.npy
semantic_kg_features/text_embeddings/exercise_text_embeddings.npy
```

V10 要求这些文件由 `import_ektm_topic_embeddings_v10.py` 写入。若 manifest 中出现 BGE-M3 / SentenceTransformer，校验和训练加载都会失败。

### 3.2 EKTM_mirt 中真正的题目文本表示

EKTM_mirt 定义在：

```text
KT/MMKT/src/model_EKT.py
```

关键逻辑：

```text
topic_model = TopicRNNModel(...)
topic_v, _ = topic_model(topic)
```

其中 `TopicRNNModel` 是 Bi-GRU 文本编码器，`topic_v[0]` 是题目文本语义表示。它随后与知识点 embedding、MIRT 参数 `topic_ab = concat(a, b)` 一起进入 `EKTSeqModel_cdm`。

因此，V10 从 EKTM_mirt 导出每道题的 `topic_v`，保存为：

```text
semantic_kg_features/text_embeddings/exercise_text_embeddings.npy
```

或新命名为：

```text
semantic_kg_features/text_embeddings/exercise_ektm_topic_embeddings.npy
```

同时在 `text_embedding_manifest.json` 中标明：

```json
{
  "model": "EKTM_mirt.TopicRNNModel",
  "source": "topic_v",
  "embedding_dim": 200
}
```

## 4. 哪些特征应该继续复用，哪些必须重算

### 可以复用

- `concept_semantics.json`：知识点名称和 DeepSeek definition，可用于解释或知识点文本元数据；
- `exercise_semantics.json`：题目文本或结构化文本元数据；
- `entities.dict`、`Q.txt` 等图结构文件；
- MIRT 训练输入中的作答记录过滤逻辑。

### 必须重算或重新导出

- `exercise_text_embeddings.npy`：由 EKTM_mirt 的 Bi-GRU `topic_v` 导出，不再使用 BGE；
- `stu2know_mastery.json`：应由 EKTM_mirt 的 `know_output` 导出；
- `exercise_irt_features.json`：应与 EKTM_mirt 使用的 `a/b` 参数同源；
- `learner_pedagogy.json`：应与同源 theta 和 EKTM_mirt mastery 对齐；
- `stu2know_seq.json`：应由重新训练的 pyKT DKT 或明确选定的序列模型导出；
- `stu2know_forget.json`、`stu2ex_forget.json`：应按 V10 修正逻辑重新生成。

## 5. 当前 V10 仍需修正的代码点

### 5.1 `export_mirt_features_v10.py`

当前代码不再复制旧的 `text_embeddings` 目录，避免把 BGE 向量带入 V10。

需要修改为：

```text
必须先导入 EKTM_mirt topic_v；如果没有，则报错。
```

### 5.2 `validate_v10_front_files.py`

当前只检查 text embedding 是否存在，没有检查来源模型。

需要新增检查：

```text
text_embedding_manifest.json 中 model/source 是否为 EKTM_mirt.TopicRNNModel / topic_v。
```

如果仍为 `BAAI/bge-m3`，V10 直接报错。

### 5.3 `feature_loader.py`

读取接口可以保持不变，因为最终都是 `.npy` 矩阵；但要兼容 EKTM_mirt 的 200 维 `topic_v`，不要再假设文本 embedding 是 1024 维。

当前代码中 `text_dim` 是从 manifest 或 npy shape 自动读取，理论上可以兼容 200 维。

## 6. 当前已实现导入脚本

V10 已新增脚本：

```text
ER/KG4ER-New/v10_pipeline/import_ektm_topic_embeddings_v10.py
```

作用：

1. 读取已经由 EKTM_mirt 导出的 `topic_v` 矩阵；
2. 检查行数是否与 `ex0...exN` 题目数一致；
3. 按题目 ID 顺序保存 `exercise_text_embeddings.npy`；
4. 为知识点保存零向量 `concept_text_embeddings.npy`，避免引入 EKTM_mirt 未使用的知识点文本特征；
5. 写入 manifest，明确来源为 `EKTM_mirt.TopicRNNModel` 和 `topic_v`。

这样 V10 的语义特征才和论文中“text-enhanced KT module”完全一致。
