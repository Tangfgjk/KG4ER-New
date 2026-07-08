# SemanticConvE V8：EduCDM-MIRT 与前置文件改造方案

本文档规划 V8 版本的代码与数据流程改造。目标是在不覆盖现有数据集、不修改已生成文件的前提下，重新梳理 IRT/MIRT 参数、KT 前置文件、ER 构图文件、SemanticConvE 训练与测试之间的关系。

## 1. V8 改造目标

V8 的核心变化是：不再使用当前 `generate_semantic_pedagogical_features.py` 中自定义的 2PL IRT 估计结果作为教育学特征来源，而是使用 `EduCDM` 中的 MIRT 模型生成题目参数和学习者能力参数。

具体目标如下：

1. 使用 `EduCDM_MIRT_noQ_export_modified/EduCDM-main/EduCDM/MIRT/MIRT.py` 训练修改后的 no-Q MIRT。
2. 为每个数据集生成新的 `a_param_K.csv`、`b_param_K.csv`、`theta_param_K.csv`。
3. 使用新的 MIRT 参数支持后续 EKTM_mirt 或 V8 KT 导出流程，重新生成 `stu2know_mastery.json`。
4. 基于新的 mastery 文件重新生成推荐边、三元组与 V8 ER 训练目录。
5. 后续 SemanticConvE V8 训练只读取 V8 新目录，不覆盖旧目录。

## 2. 不修改旧数据的目录策略

所有新生成文件都放在各数据集目录下的新文件夹中。

以 `Eedi` 为例：

```text
ER/KG4ER/data/Eedi/
  原有文件保持不变

  mirt_v8/
    inputs/
      train.csv
      valid.csv
      test.csv
      Q_matrix.csv
      user_id_map.json
      item_id_map.json
      split_manifest.json
    outputs/
      mirt_no_q.params
      a_param_57.csv
      b_param_57.csv
      theta_param_57.csv
      mirt_metrics.json
      export_manifest.json

  kt_exports_v8/
    stu2know_mastery.json
    mastery_export_manifest.json
    kt_model/
      best.pt
      last.pt

  semantic_kg_features_v8/
    irt_features/
      exercise_mirt_features.json
      learner_mirt_features.json
    entity_features/
      entity_feature_index.json
    feature_generation_manifest.json

  er_v8/
    entities.dict
    relations.dict
    Q.txt
    stu2know_mastery.json
    stu2know_seq.json
    stu2know_forget.json
    stu2ex_forget.json
    stu2ex_recommend.json
    triples.txt
    test_triples.txt
    semantic_kg_features_v8/
```

其他数据集同理：

```text
ER/KG4ER/data/<dataset>/mirt_v8/
ER/KG4ER/data/<dataset>/kt_exports_v8/
ER/KG4ER/data/<dataset>/semantic_kg_features_v8/
ER/KG4ER/data/<dataset>/er_v8/
```

## 3. EduCDM-MIRT 输入输出

实现口径更新：当前 V8 使用的是用户修改后的 `EduCDM_MIRT_noQ_export_modified`。该版本不使用 Q-matrix 约束，`latent_dim=K` 只表示 K 个潜在因子，不强制与 K 个知识点一一对应。因此 V8 默认把 MIRT 的 `a/b/theta` 作为教育测量特征来源，而不再声称 `a` 的每一维都是严格知识点区分度。

核心模型文件：

```text
EduCDM_MIRT_noQ_export_modified/EduCDM-main/EduCDM/MIRT/MIRT.py
```

已有导出脚本：

```text
EduCDM/EduCDM-main/examples/MIRT/MIRT_Q_export.py
```

该 MIRT 模型输入：

| 输入 | 说明 |
|---|---|
| `user_num` | 学生数量，要求 `user_id` 从 0 连续编号 |
| `item_num` | 题目数量，要求 `item_id` 从 0 连续编号 |
| `latent_dim` | 潜在维度，V8 中设置为知识点数量 |
| `q_matrix` | 题目-知识点矩阵，形状为 `item_num × knowledge_num` |
| `train_data` | 三元组 `(user_id, item_id, score)` |

模型公式为：

```text
P(correct | u, e) = sigmoid(D * sum(a_e * theta_u) - b_e)
```

使用 no-Q MIRT 后：

```text
a_e = softplus(a_raw_e) * Q_e
```

因此，题目未涉及的知识点维度会被强制置 0。

输出：

| 文件 | 说明 |
|---|---|
| `a_param_K.csv` | 题目区分度潜在因子矩阵，形状 `item_num × latent_dim` |
| `b_param_K.csv` | 题目难度/偏置参数，形状 `item_num × 1` |
| `theta_param_K.csv` | 学生能力潜在因子向量，形状 `user_num × latent_dim` |
| `mirt_no_q.params` | MIRT 模型权重 |

## 4. V8 数据流程

### 4.1 阶段一：准备 MIRT 输入

新增 V8 代码应读取每个数据集已有的：

```text
sequence_interactions.csv
Q.txt
```

转换为 EduCDM MIRT 所需格式：

```csv
user_id,item_id,score
0,12,1
0,35,0
1,12,1
```

需要完成的处理：

1. 将原始 `uid` 重编码为连续整数 `user_id`。
2. 校验 `question` 是否已经是连续整数题目 ID。
3. 将 `response` 改名为 `score`。
4. 将 `Q.txt` 复制或规范化为 `Q_matrix.csv`。
5. 保存 `user_id_map.json`，保证后续 `theta_param_K.csv` 可以映射回 ER 的 `uid`。

### 4.2 阶段二：训练 EduCDM-MIRT 并导出参数

每个数据集单独训练 MIRT：

```text
input:
  mirt_v8/inputs/train.csv
  mirt_v8/inputs/valid.csv
  mirt_v8/inputs/test.csv
  mirt_v8/inputs/Q_matrix.csv

output:
  mirt_v8/outputs/a_param_K.csv
  mirt_v8/outputs/b_param_K.csv
  mirt_v8/outputs/theta_param_K.csv
```

推荐默认设置：

| 参数 | 默认值 |
|---|---|
| `epoch` | 50 |
| `batch_size` | 1024 或 2048 |
| `lr` | 0.001 |
| `weight_decay` | 1e-4 |
| `device` | `cuda` 优先 |
| `latent_dim` | 数据集知识点数量 |

需要注意：

1. 这里的“训练侧覆盖全部题目”只针对 MIRT 参数估计阶段。意思是：如果只用 `sequence_interactions.csv` 中的训练交互来训练 MIRT，题目是否都至少出现过一次。只有出现过的题目，其 `a_param_K.csv` 和 `b_param_K.csv` 才能被充分更新。
2. 当前按本地文件统计，`Eedi`、`algebra2005`、`statics2011`、`XES3G5M-sub-small` 的训练交互基本覆盖全部题目。
3. `assist2009-sub` 的 Q 矩阵共有 1987 道题，但当前训练交互中只覆盖 1982 道题，少量题目只在测试侧或非训练侧出现。如果 MIRT 只用训练交互拟合，这些缺失题目的 MIRT 参数会停留在随机初始化或非常不稳定的状态。
4. 因此 V8 代码需要提供策略：
   - 方案 A：MIRT 使用全量交互估计 item 参数，即把 MIRT 当作题目教育学特征估计器；
   - 方案 B：缺失题目使用同知识点均值或全局均值初始化；
   - 推荐先采用方案 A，因为 MIRT 参数本身是前置教育学特征，不直接作为最终 ER 推荐测试标签。这样可以避免少数题目的 `a/b` 参数无效。

### 4.3 阶段三：重新生成 `stu2know_mastery.json`

这是 V8 最关键的一步。

当前问题是：原 ER 前置文件中的 `stu2know_mastery.json` 并没有严格使用论文描述中的文本 + MIRT + 知识点模型生成。V8 应改为：

```text
题目文本 / 结构化题目信息
+ 知识点 ID / Q-matrix
+ EduCDM-MIRT a/b 参数
+ 学生历史作答序列
-> EKTM_mirt / V8 KT exporter
-> stu2know_mastery.json
```

为避免修改 `KT/MMKT` 原代码，建议在 `ER/KG4ER-New` 下新增 V8 KT 导出脚本：

```text
ER/KG4ER-New/kt_export_v8/export_mastery_from_ektm_mirt_v8.py
```

该脚本负责：

1. 加载每个数据集的 V8 MIRT 参数。
2. 加载题目文本或结构化题目信息。
3. 加载题目-知识点映射。
4. 运行 EKTM_mirt 风格的知识追踪模型。
5. 导出新的：

```text
ER/KG4ER/data/<dataset>/kt_exports_v8/stu2know_mastery.json
```

`stu2know_seq.json` 和 `stu2know_forget.json` 暂时可以沿用已有流程生成的结果，因为 V8 当前重点修正的是 mastery 与 MIRT 参数来源。

### 4.4 阶段四：重新生成 ER 构图文件

由于 `stu2know_mastery.json` 会改变，推荐边 `rec` 和三元组也必须重新生成。

V8 不应覆盖原来的 `triples.txt` / `test_triples.txt`，而是生成：

```text
ER/KG4ER/data/<dataset>/er_v8/
```

生成逻辑：

1. 从旧 ER 数据目录复制静态文件：
   - `Q.txt`
   - `entities.dict`
   - `stu2know_seq.json`
   - `stu2know_forget.json`
   - `stu2ex_forget.json`
2. 从 `kt_exports_v8/` 复制新的：
   - `stu2know_mastery.json`
3. 重新运行推荐分数计算：
   - 输出 `stu2ex_recommend.json`
4. 重新构造三元组：
   - 输出 `triples.txt`
   - 输出 `test_triples.txt`
5. 重新生成或复制关系字典：
   - `relations.dict`

关系数量仍保持 KG4ER 设计中的 304，不因 V8 改动而改变。

### 4.5 阶段五：重新生成 SemanticConvE V8 特征

当前旧版本中 `exercise_irt_features.json` 和 `learner_pedagogy.json` 来自自定义 2PL IRT。V8 需要替换为 EduCDM-MIRT 结果。

建议生成：

```text
semantic_kg_features_v8/irt_features/exercise_mirt_features.json
semantic_kg_features_v8/irt_features/learner_mirt_features.json
```

题目特征来源：

| 特征 | 来源 | 用途 |
|---|---|---|
| `difficulty_mirt` | `b_param_K.csv` | 原始难度值，主要用于记录、解释和可视化，不建议直接进入模型 |
| `difficulty_mirt_norm` | 对 `b` 归一化 | 默认进入模型的题目难度特征 |
| `discrimination_mirt_norm` | 对 `a` 的 L2 norm 归一化 | 默认进入模型的题目区分度强度特征 |
| `kc_discrimination_vector` | `a_param_K.csv` 的整行向量 | 可选保存；默认不直接进入 SemanticConvE，以避免高维稀疏向量带来噪声 |

说明：上表不是表示四个字段都必须同时进入模型。V8 默认进入模型的是归一化后的紧凑数值特征，即 `difficulty_mirt_norm` 和 `discrimination_mirt_norm`。原始 `difficulty_mirt` 用于解释和复查，`kc_discrimination_vector` 用于后续扩展或分析。

学习者特征来源：

| 特征 | 来源 | 用途 |
|---|---|---|
| `theta_mirt_vector` | `theta_param_K.csv` | 学习者知识维度能力向量，默认用于聚类和解释，不直接完整进入模型 |
| `theta_mirt_mean` | 学生 theta 均值 | 可作为学习者整体能力统计特征 |
| `theta_mirt_norm` | 归一化能力 | 默认进入模型的学习者能力特征 |
| `overall_mastery_mirt` | 由 theta 或 V8 mastery 聚合得到 | 默认进入模型的综合掌握度特征 |
| `cluster_id` | 基于 V8 学习者画像重新聚类 | 默认进入模型，使用 `nn.Embedding` 表示学习者群体类型 |

说明：`theta_mirt_vector` 是高维向量，默认不直接整段输入 SemanticConvE。V8 默认使用 `theta_mirt_norm`、`overall_mastery_mirt` 和 `cluster_id`，这样既保留 MIRT 能力信息，又避免高维 theta 直接扰动 ID embedding。

### 4.5.1 V8 中各类实体使用的特征

V8 的实体仍然以 ID embedding 为基础，不把外部特征作为替代品。不同实体使用的补充特征不同：

| 实体类型 | 默认使用的特征 | 来源 |
|---|---|---|
| 学习者 `uid` | ID embedding、实体类型 embedding、`theta_mirt_norm`、`overall_mastery_mirt`、`cluster_id` | `entities.dict`、`theta_param_K.csv`、V8 `stu2know_mastery.json`、K-Means 聚类 |
| 题目 `ex` | ID embedding、实体类型 embedding、题目文本/结构化语义 embedding、`difficulty_mirt_norm`、`discrimination_mirt_norm` | `entities.dict`、题目文本或结构化字段、`a_param_K.csv`、`b_param_K.csv` |
| 知识点 `kc` | ID embedding、实体类型 embedding、知识点名称与 definition 的语义 embedding | `entities.dict`、`concept_semantics.json` |

知识点实体默认不直接使用 `a_param_K.csv` 或 `b_param_K.csv`，因为 MIRT 的 `a/b` 是题目参数，不是知识点参数。若后续需要知识点教育学特征，可以再从关联题目上聚合平均难度、平均区分度，但这不是 V8 的默认设计。

关系表示继续采用 relation-aware encoding：

```text
relation representation
= relation ID embedding
+ relation type embedding
+ relation strength / coarse strength embedding
```

其中 relation ID embedding 保留原有关系的可学习记忆能力，relation type 和 strength 用来表达 `mlkc/pkc/exfr/rec` 的类型和强度。

### 4.5.2 V8 的特征融合方式

V8 不建议继续使用早期版本的“所有向量直接相加”：

```text
entity = ID + semantic + pedagogical + type + cluster
```

这种方式的问题是：所有特征一开始就以同等形式扰动 ID embedding，模型还没有机会判断哪些特征可靠、哪些特征是噪声。

V8 推荐采用“向量拼接 + attention + 全连接压缩 + ID residual”的方式：

```text
1. 每一类补充特征先投影成同维度 feature token
   semantic token
   pedagogical token
   type token
   cluster token

2. 将这些 feature token 在 token 维度拼接
   feature_tokens = concat([semantic, pedagogical, type, cluster])

3. 使用 attention 学习不同特征的重要性
   attended_feature = Attention(feature_tokens)

4. 使用全连接层压回 ConvE 需要的固定维度
   feature_enhance = FC(attended_feature)

5. 与 ID embedding 做 residual 融合
   final_entity = LayerNorm(ID_embedding + gate * feature_enhance)
```

这里的“拼接”发生在 attention 之前，模型可以先看到不同来源的特征；最后仍然和 ID embedding 做 residual 加法，是为了保留原 ConvE/ConvER 中强有效的 ID 记忆能力。这个 residual 加法不是旧版本的直接相加，而是经过 attention 和 FC 过滤后的增强项。

### 4.6 阶段六：V8 训练与测试

V8 训练不读取旧数据目录，而读取：

```text
ER/KG4ER/data/<dataset>/er_v8/
```

运行结果保存到：

```text
ER/KG4ER-New/runs/<dataset>/<dataset>_v8_*/
```

模型命名建议：

| 模型名 | 说明 |
|---|---|
| `SemanticConvE_v8_full` | V8 完整模型 |
| `SemanticConvE_v8_no_content_entity` | 去掉实体侧新增内容特征 |
| `SemanticConvE_v8_no_relation_aware` | 去掉 relation type/strength |
| `SemanticConvE_v8_no_type_aware_scoring` | 恢复全实体打分 |
| `SemanticConvE_v8_no_mastery` | 去掉 mastery 关系 |
| `SemanticConvE_v8_no_forgetting` | 去掉 forgetting 关系 |
| `SemanticConvE_v8_no_seq` | 去掉 sequence/progress 关系 |

## 5. V8 需要新增或修改的代码

### 5.1 新增代码

```text
ER/KG4ER-New/semantic_features_v8/prepare_educdm_mirt_inputs.py
```

职责：从 `sequence_interactions.csv` 和 `Q.txt` 生成 EduCDM MIRT 输入。

```text
ER/KG4ER-New/semantic_features_v8/train_export_educdm_mirt.py
```

职责：调用 EduCDM MIRT，训练并导出 `a/b/theta`。

```text
ER/KG4ER-New/kt_export_v8/export_mastery_from_ektm_mirt_v8.py
```

职责：基于 V8 MIRT 参数和 EKTM_mirt 风格模型导出新的 mastery。

```text
ER/KG4ER-New/semantic_features_v8/build_v8_mirt_features.py
```

职责：把 `a/b/theta` 转成 SemanticConvE 可读取的 JSON 特征。

```text
ER/KG4ER-New/data_pipeline_v8/build_er_v8_dataset.py
```

职责：将 V8 mastery、旧 seq/forget、推荐计算与三元组构造串起来，生成 `er_v8/`。

```text
ER/KG4ER-New/codes-New-ConvE/run_semantic_experiments_v8.py
```

职责：V8 训练入口，默认读取 `er_v8/` 和 `semantic_kg_features_v8/`。

### 5.2 修改代码

尽量少改原有 V7 文件。若必须复用已有模型代码，建议只做兼容参数：

```text
ER/KG4ER-New/codes-New-ConvE/feature_loader.py
ER/KG4ER-New/codes-New-ConvE/semantic_conve_model.py
ER/KG4ER-New/codes-New-ConvE/summarize_semantic_results.py
```

修改原则：

1. V7 逻辑不删除。
2. 新增 `--feature-version v8` 或新入口 `run_semantic_experiments_v8.py`。
3. 默认读取 V8 文件夹，不覆盖旧结果。

## 6. 数据集扩展注意事项

### 6.1 Eedi

已有：

```text
ER/KG4ER/data/Eedi/sequence_interactions.csv
ER/KG4ER/data/Eedi/Q.txt
```

Q 矩阵形状：

```text
948 × 57
```

Eedi 是 V8 首个打通数据集。

### 6.2 algebra2005

可用目录：

```text
ER/KG4ER/data/algebra2005/prepared_for_kt/
```

Q 矩阵形状：

```text
1084 × 112
```

该数据集多知识点题目较多，V8 中 Q 矩阵主要用于把题目级 MIRT 难度聚合到知识点级 mastery proxy，而不是约束 MIRT 训练。

### 6.3 assist2009-sub

可用目录：

```text
ER/KG4ER/data/assist2009-sub/prepared_for_kt/
```

Q 矩阵形状：

```text
1987 × 149
```

训练交互中存在少量题目覆盖缺失，V8 需要显式记录处理策略。

### 6.4 statics2011

可用目录：

```text
ER/KG4ER/data/statics2011/prepared_for_kt/
```

Q 矩阵形状：

```text
633 × 97
```

规模较小，适合作为 V8 验证数据集。

### 6.5 XES3G5M-sub-small

可用目录：

```text
ER/KG4ER/data/XES3G5M-sub-small/prepared_for_kt/
```

Q 矩阵形状：

```text
1928 × 408
```

知识点维度较高，MIRT 训练可能更慢，建议优先测试较小 epoch。

## 7. V8 推荐执行顺序

1. 先在 Eedi 上生成 `mirt_v8/inputs/`。
2. 在 Eedi 上训练 EduCDM-MIRT，导出 `a/b/theta`。
3. 验证 `a_param_57.csv` 和 `b_param_57.csv` 形状。
4. 使用 V8 MIRT 参数生成 `stu2know_mastery.json`。
5. 生成 Eedi 的 `er_v8/`。
6. 跑 Eedi 单种子 smoke test。
7. 跑 Eedi 5 seeds。
8. 再扩展到其余四个数据集。

## 8. 关键验证项

每个数据集生成后必须检查：

| 检查项 | 期望 |
|---|---|
| `a_param_K.csv` | 行数等于题目数，列数等于知识点数 |
| `b_param_K.csv` | 行数等于题目数，列数为 1 |
| `theta_param_K.csv` | 行数等于 MIRT 学生数，列数等于知识点数 |
| `stu2know_mastery.json` | 学生数和测试学生一致 |
| `er_v8/triples.txt` | 存在且非空 |
| `er_v8/test_triples.txt` | 存在且非空 |
| `relations.dict` | 关系数仍为 304 |
| V8 训练结果 | 保存到 `runs/<dataset>/<dataset>_v8_*` |

## 9. 与旧版本的关系

V8 不删除 V7，也不覆盖旧实验结果。

V8 与旧版本的主要区别：

| 模块 | V7 或之前 | V8 |
|---|---|---|
| IRT 来源 | 自定义 2PL IRT 或统计特征 | 修改后的 EduCDM no-Q MIRT |
| `a/b` 参数 | 不与 EKTM_mirt 原始设计完全一致 | 显式生成 `a_param_K.csv` 和 `b_param_K.csv` |
| mastery 来源 | 可能未严格使用文本 + MIRT + 知识点模型 | 当前使用 V8 no-Q MIRT proxy；后续可替换为严格 EKTM_mirt checkpoint 导出 |
| ER 构图 | 使用旧 mastery 构造 rec | 使用 V8 mastery 重新构造 rec |
| 数据目录 | 原目录 | 新建 `mirt_v8/`、`kt_exports_v8/`、`er_v8/` |

## 10. 当前暂不执行的内容

本方案阶段暂不做：

1. 不训练 MIRT。
2. 不生成新前置文件。
3. 不修改 `KT/MMKT` 原始代码。
4. 不覆盖任何已有数据集文件。
5. 不运行 SemanticConvE 实验。

后续确认方案后，再进入 V8 代码实现阶段。
