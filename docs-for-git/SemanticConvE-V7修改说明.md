# SemanticConvE V7 修改说明

## 1. 版本目标

V7 主要解决 V5 中“不同来源特征直接相加，可能互相干扰”的问题。新版本保留原 ConvE 的 ID embedding 作为稳定主干，把文本语义、教育学数值、实体类型、学习者聚类等额外信息作为独立 feature token 输入 attention，再通过全连接层压回 ConvE 所需维度，并以较小残差比例加入 ID embedding。

## 2. 实体表示改动

V5 的实体表示近似为：

```text
ID embedding + semantic embedding + pedagogical embedding + type embedding + cluster embedding
```

V7 改为：

```text
feature tokens = [semantic, pedagogical, entity type, cluster]
fused feature = FC(Attention(feature tokens))
final entity embedding = LayerNorm(ID embedding + alpha * fused feature)
```

其中 `alpha` 是可学习残差系数，初始值为 `0.05`。这意味着模型一开始更信任 ID embedding，训练过程中再学习是否需要更多使用额外特征。

## 3. 不同实体使用的特征

| 实体类型 | ID embedding | 文本语义 | 教育学数值 | 实体类型 | cluster |
| --- | --- | --- | --- | --- | --- |
| 知识点 `kc` | 使用 | 使用知识点名称和 definition 的文本 embedding | 不使用 | 使用 | 不使用 |
| 题目 `ex` | 使用 | 使用题目文本或结构化题目信息的文本 embedding | 使用题目统计难度、区分度、正确率等 | 使用 | 不使用 |
| 学习者 `uid` | 使用 | 不使用 | 使用学习者综合掌握度、正确率、作答量等 | 使用 | 使用 |

## 4. 关系表示改动

V7 保留 relation ID embedding，不再删除原始关系 ID 能力。关系表示为：

```text
relation tokens = [relation type, relation strength]
fused relation = FC(Attention(relation tokens))
final relation embedding = LayerNorm(relation ID embedding + beta * fused relation)
```

其中 `relation type` 表示 `rec/mlkc/pkc/exfr` 等关系类型，`relation strength` 表示从关系名中解析出的连续强度值，例如 `mlkc0.83` 中的 `0.83`。

## 5. Type-aware Scoring

默认推荐阶段只对题目实体 `ex` 打分：

```text
score(uid, rec, ex)
```

`no_type_aware_scoring` 消融会恢复为对全部实体打分后再筛选题目，用于验证类型约束打分是否有效。

## 6. V7 消融实验

| 消融名称 | 含义 |
| --- | --- |
| `full` | V7 完整模型 |
| `id_only` | 只使用 ID embedding，不使用任何额外特征 |
| `direct_sum_fusion` | 使用 V5 风格的直接相加融合，不使用 attention |
| `no_text_semantic` | 去掉全部文本语义特征 |
| `no_concept_text` | 只去掉知识点文本语义 |
| `no_exercise_text` | 只去掉题目文本语义 |
| `no_pedagogical` | 去掉教育学数值特征和学习者 cluster |
| `no_relation_aware` | 关系只使用 relation ID embedding |
| `no_type_aware_scoring` | 推荐阶段不限制 tail 类型 |
| `no_mastery` | 重新生成图，去掉掌握度推荐项和 `mlkc` 关系 |
| `no_forgetting` | 重新生成图，去掉遗忘推荐项和 `exfr` 关系 |
| `no_seq` | 重新生成图，去掉序列推荐项和 `pkc` 关系 |

## 7. 主要代码文件

| 文件 | 作用 |
| --- | --- |
| `codes-New-ConvE/semantic_conve_model.py` | V7 模型主体、attention 融合、relation-aware 表示、type-aware scoring |
| `codes-New-ConvE/feature_loader.py` | 加载实体文本 embedding、统计教育特征、关系类型和关系强度 |
| `codes-New-ConvE/run_semantic_conve.py` | 单个模型训练 |
| `codes-New-ConvE/test_semantic_conve.py` | 单个模型测试与打分 |
| `codes-New-ConvE/run_semantic_experiments.py` | 一键运行训练、测试、评估 |
| `codes-New-ConvE/summarize_semantic_results.py` | 汇总五个随机种子的结果 |

