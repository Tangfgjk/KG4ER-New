# SemanticConvE V10 DKT 任务改造与剩余问题处理计划

本文记录 V10 后续需要处理的 DKT/PKC-DKT 改造，以及当前仍需完善的 11 个问题。本文只作为后续实现计划，当前不直接修改代码。

## 1. 当前判断

目前 `ER/KG4ER-New/v10_pipeline/export_seq_from_dkt_v10.py` 只是从已有 pyKT DKT checkpoint 导出最后时间步的全知识点输出向量，并写成：

```text
stu2know_seq.json
```

但是 pyKT 原始 DKT 的训练目标仍是标准知识追踪任务：

```text
根据历史作答，预测下一题/下一知识点是否答对。
```

也就是说，当前代码只做了“导出方式适配”，没有完成说明文档中要求的“训练任务适配”。

根据 `KG4ER说明文档.pdf` 中截图说明，序列预测 DKT 应该修改训练阶段的标签构造，使其更接近“下一步知识点出现/进展预测”任务，而不是仅仅预测下一题答对概率。

## 2. DKT 需要怎么改

### 2.1 不再直接调用已有 pyKT checkpoint

后续不应再默认使用：

```text
ER/pykt-toolkit-main/examples/saved_model/<old_dkt_checkpoint>
```

因为旧 checkpoint 的训练目标未必符合本项目的 `stu2know_seq.json` 语义。

应改为：

```text
在 KG4ER-New 中复制一份最小可控的 DKT 训练代码，
按 KG4ER 说明文档修改训练目标，
再重新训练每个数据集自己的 DKT/PKC-DKT。
```

### 2.2 代码放置建议

不修改原始 pyKT 目录：

```text
ER/pykt-toolkit-main/
```

新增 KG4ER-New 自己的 DKT 代码：

```text
ER/KG4ER-New/v10_pipeline/pkc_dkt/
  dkt.py
  train_model_pkc.py
  evaluate_model_pkc.py
  dataset.py
  train_pkc_dkt_v10.py
  export_pkc_seq_v10.py
```

其中：

- `dkt.py`：从 pyKT 的 `pykt/models/dkt.py` 复制基础模型；
- `train_model_pkc.py`：修改 loss 和 label 构造；
- `evaluate_model_pkc.py`：按说明文档导出每个学生最后时间步的全知识点预测；
- `train_pkc_dkt_v10.py`：V10 专用训练入口；
- `export_pkc_seq_v10.py`：从 best checkpoint 导出 `stu2know_seq.json`。

### 2.3 训练目标修改思路

原 pyKT DKT 训练逻辑大致是：

```python
y = model(c, r)
y = (y * one_hot(cshft, model.num_c)).sum(-1)
loss = binary_cross_entropy(y, rshft)
```

含义是：

```text
只取下一步真实知识点对应的预测值，
拟合下一步是否答对。
```

说明文档要求的修改方向是：

```python
t = torch.masked_select(torch.ones(size=rshft.shape).cuda(), sm)
loss = binary_cross_entropy(y.double(), t.double())
```

也就是把目标从“答对/答错标签”改成“下一步真实出现知识点为正例”。更直观地说：

```text
模型输出的是各知识点下一步出现/进展概率；
实际下一步出现过的知识点 label 设为 1；
然后训练模型提高这些真实出现知识点的预测概率。
```

需要注意：这一步必须仔细设计负样本或未出现知识点的处理。如果所有被选中的位置都设为 1，而没有合理负样本，模型可能学成全 1。因此后续实现时建议至少保留两种版本用于验证：

1. `pkc_positive_only`：严格复现说明文档截图做法；
2. `pkc_multilabel`：下一步出现知识点为 1，其余知识点为 0 或采样负例。

优先实现说明文档版本，再根据分布检查是否需要改进。

### 2.4 导出 `stu2know_seq.json`

训练完成后，使用 best checkpoint 对测试学生序列推理：

```python
y = model(c, r)
stu2know_seq = y[:, -1, :]
```

输出：

```text
ER/KG4ER-New/data/<dataset>/v10/kt_pkc_dkt/stu2know_seq.json
ER/KG4ER-New/data/<dataset>/v10/stu2know_seq.json
```

导出时必须校验：

- 行数 = V10 图中的 learner 数；
- 列数 = 知识点数；
- 行顺序 = `uid0, uid1, ...`；
- 数值范围在 `[0, 1]`；
- 不能出现 Eedi 那种所有值集中在 0.95 以上、几乎无区分度的问题。

## 3. V10 前置文件后续推荐流程

后续完整 V10 前置文件生成应为：

```text
1. prepare_mirt_inputs_v10.py
2. train_mirt_noq_v10.py
3. 训练 EKTM_mirt
4. export_ektm_outputs_v10.py
5. 训练 PKC-DKT
6. export_pkc_seq_v10.py
7. generate_forgetting_v10.py
8. export_mirt_features_v10.py
9. build_v10_graph.py
10. validate_v10_front_files.py
```

其中：

- `EKTM_mirt` 负责 `stu2know_mastery.json` 和 `topic_v`；
- `PKC-DKT` 负责 `stu2know_seq.json`；
- forgetting 负责 `stu2know_forget.json` 和 `stu2ex_forget.json`；
- graph builder 负责推荐距离、推荐边和最终三元组。

## 4. V10-attn 模型特征融合与实验口径

V10 后续模型层采用 `V10 前置文件修复 + V8 attention 融合结构` 的方案。也就是说，前置文件仍使用 V10 重新生成和校验后的数据，但 SemanticConvE 的实体/关系表示使用 feature-token self-attention 进行融合。

### 4.1 实体表示

所有实体都保留原始 ID embedding 作为稳定锚点，但实体侧不再额外加入 entity type token。实体类型信息只通过不同实体使用不同特征集合体现，不再作为实体 token 输入 attention。

学生实体 `uid`：

```text
ID embedding
+ Attention(theta_mirt_norm, cluster_id)
```

说明：

- 学生保留 IRT/MIRT 能力参数 `theta_mirt_norm`；
- 学生保留基于画像或能力分布得到的 `cluster_id`；
- 学生不再融合 `overall_mastery_mirt`，避免与 `stu2know_mastery.json` 和推荐构图阶段的掌握度信息重复；
- 学生不再融合 entity type token。

题目实体 `ex`：

```text
ID embedding
+ Attention(topic_v/text, difficulty_mirt_norm, discrimination_mirt_norm)
```

说明：

- `topic_v/text` 来自 EKTM_mirt 中题目文本编码模块导出的题目语义表示；
- `difficulty_mirt_norm` 来自 MIRT 难度参数归一化；
- `discrimination_mirt_norm` 来自 MIRT 区分度参数归一化；
- 不再额外使用 entity type token。

知识点实体 `kc`：

```text
ID embedding
+ Attention(concept_semantic)
```

说明：

- `concept_semantic` 来自知识点名称和 definition 的语义表示；
- 知识点不再加入 entity type token；
- 如果后续实验发现知识点文本信息噪声较大，则通过 `no_concept_semantic` 消融单独验证。

实体融合统一形式为：

```text
final_entity = LayerNorm(ID_embedding + lambda_entity * AttentionFusion(feature_tokens))
```

其中 `feature_tokens` 只包含该实体实际拥有的额外特征。ID embedding 不作为 attention token，而是作为 residual anchor 保留。

### 4.2 关系表示

关系表示保留 relation ID embedding，并在此基础上融合关系类型和连续强度：

```text
relation r:
ID embedding + Attention(relation_type, relation_strength)
```

关系类型包括：

```text
mlkc -> mastery
pkc  -> sequence/progress
exfr -> forgetting
rec  -> recommendation
```

关系强度来自关系名中的数值，例如：

```text
mlkc_0.83 -> strength = 0.83
pkc_0.64  -> strength = 0.64
exfr_0.41 -> strength = 0.41
```

`rec` 没有天然连续强度，暂时使用固定默认值，例如 `1.0`。关系侧不去掉 type，因为 relation type 是区分 mastery、sequence、forgetting 和 recommendation 的核心语义。

关系融合形式为：

```text
final_relation = LayerNorm(ID_relation + lambda_relation * AttentionFusion(type_token, strength_token))
```

### 4.3 未见过习题的处理

未见过的习题相关遗忘需求暂时保持旧方案中的最大值 `1` 不变。短期这样做可以保持与原 KG4ER/KG4EX 构图逻辑一致，避免一次性引入过多变量。后续如果需要区分“从未学习”和“曾经学习但遗忘”，再增加显式 mask 文件，例如：

```text
stu2know_seen_mask.json
stu2ex_seen_mask.json
```

### 4.4 后续消融实验设计

后续主模型和消融实验固定为以下 9 个：

| 实验名 | 含义 |
|---|---|
| `full` | 完整 V10-attn 模型，实体使用对应语义/教育特征，关系使用 ID + type + strength |
| `id_only` | 最原始模型，只使用实体 ID embedding 和关系 ID embedding，不融合任何额外特征 |
| `no_pedagogical` | 去掉教育领域特征，包括学生 `theta_mirt_norm`、`cluster_id`，以及题目 `difficulty_mirt_norm`、`discrimination_mirt_norm` |
| `no_text_semantic` | 去掉全部文本语义信息，包括题目 `topic_v/text` 和知识点 `concept_semantic` |
| `no_concept_semantic` | 只去掉知识点文本语义信息，保留题目文本语义 |
| `no_relation_aware` | 去掉关系融合特征，只保留 relation ID embedding |
| `no_mastery` | 去掉掌握度相关认知关系/构图信息 |
| `no_forgetting` | 去掉遗忘相关认知关系/构图信息 |
| `no_seq` | 去掉序列/进展相关认知关系/构图信息 |

其中 `id_only` 同时作为“最原始只用 ID 的 ConvE”对照模型，用于评估新增语义特征、教育学特征和关系感知表示是否整体有效。

### 4.5 对比模型运行口径

后续 TransE、TransE-adv、RotatE、DistMult、ComplEx 等对比模型不使用 SemanticConvE 的新增实体/关系特征，而是在最原始 ID-only 图表示上运行。也就是说：

```text
对比模型使用实体 ID + 关系 ID；
不融合题目文本、知识点文本、MIRT 参数、cluster、relation strength 等新增特征。
```

这样可以保证对比模型仍对应传统 KGE baseline，而不是把本文新增的三个创新点同步加到所有 baseline 中。

## 5. 11 个遗留问题与处理方案

### 问题 1：`export_mirt_features_v10.py` 缺少 `minmax`

状态：未解决，直接阻断。

处理：

在 `v10_pipeline/common.py` 中补充：

```python
def minmax(values: np.ndarray, default: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    if maximum - minimum <= 1e-12:
        return np.full_like(arr, float(default), dtype=np.float64)
    return (arr - minimum) / (maximum - minimum)
```

优先级：最高。

### 问题 2：V10 不是完全端到端

状态：部分解决。

含义：

当前 V10 已经可以严格对齐和导入前置文件，但还没有做到一条命令自动完成：

```text
原始数据 -> MIRT -> EKTM_mirt -> PKC-DKT -> forgetting -> ER 图 -> SemanticConvE
```

处理：

短期接受分阶段运行；长期新增总控脚本：

```text
run_v10_front_pipeline.py
```

优先级：中。

### 问题 3：DKT checkpoint 训练目标不确定

状态：确认存在。

处理：

不再依赖旧 pyKT checkpoint。复制 DKT 代码到 `KG4ER-New/v10_pipeline/pkc_dkt/`，按说明文档改 loss 和导出逻辑，重新训练每个数据集自己的 PKC-DKT。

优先级：最高。

### 问题 4：学生和题目 ID 只检查数量，没有显式映射

状态：部分解决。

处理：

正式流程优先使用 `export_ektm_outputs_v10.py` 和后续 `export_pkc_seq_v10.py`，由脚本根据 V10 manifest 显式对齐 `uid0...uidN`。旧 import 脚本只作为备用。

优先级：高。

### 问题 5：Eedi seq 导出未自动执行子集筛选

状态：未完全解决。

处理：

新的 `export_pkc_seq_v10.py` 必须读取 V10 manifest 中的 Eedi 子集映射，确保导出的 935 行顺序严格对应 `uid0...uid934`，不能直接读取完整 Eedi test file。

优先级：高。

### 问题 6：Forgetting 时间单位未统一

状态：未解决。

处理：

`generate_forgetting_v10.py` 后续需要增加：

```text
timestamp_unit
theta_unit
median_delta_t
p90_delta_t
```

并在计算前统一转成秒或天。

优先级：高。

### 问题 7：未接触知识点仍设为 1

状态：保留旧方案。

处理：

短期保留，解释为“最大复习需求/最大不确定性”。长期增加：

```text
stu2know_seen_mask.json
```

区分：

```text
从未学习
曾经学习但遗忘
```

优先级：中。

### 问题 8：MIRT valid/test 与 train 重叠

状态：设计取舍。

处理：

因为 MIRT 只是教育学特征估计器，不作为最终推荐模型，所以允许用全量交互估计稳定参数。但论文中不能把 MIRT 的 AUC/accuracy 当成无泄漏泛化性能报告。

优先级：中。

### 问题 9：no-Q MIRT 维度不严格对应知识点

状态：暂时不需要修改。

处理：

当前 V10/V10-attn 主要使用 MIRT 的 `a` 向量 L2 norm 作为题目区分度特征，不直接把 no-Q MIRT 的每一维解释为某个具体知识点。因此虽然 no-Q MIRT 的潜在维度不严格对应知识点，但暂时不会影响当前模型使用的核心特征。

后续如果论文或代码中不显式使用 `kc_discrimination_vector` 作为知识点级解释证据，则该命名风险暂不处理。若后续需要展示或解释 MIRT 潜在维度，再考虑改名为：

```text
latent_discrimination_vector
```

优先级：暂不处理。

### 问题 10：负采样和训练不平衡

状态：暂不修改。

处理：

先保持原方案：只对 `rec` 做负采样。后续作为消融实验比较：

```text
rec-only negative sampling
all-relation type-constrained negative sampling
```

优先级：后续实验。

### 问题 11：legacy 公式目录不是完整图

状态：按需处理。

处理：

如果只是比较推荐距离，保留当前 `v10_rec_legacy_cos/` 即可。如果要训练旧公式对照模型，需要补完整：

```text
entities.dict
relations.dict
triples.txt
test_triples.txt
```

优先级：低。

## 6. 下一步建议

后续代码修改建议按以下顺序：

1. 修复 `minmax`；
2. 复制并改造 DKT 为 `pkc_dkt`；
3. 新增 `train_pkc_dkt_v10.py`；
4. 新增 `export_pkc_seq_v10.py`；
5. 修复 Eedi 子集顺序对齐；
6. 重新生成 `stu2know_seq.json`；
7. 重新验证五个数据集的 seq 分布；
8. 再跑 SemanticConvE V10。
