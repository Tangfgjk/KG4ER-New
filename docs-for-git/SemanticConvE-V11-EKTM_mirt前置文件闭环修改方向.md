# SemanticConvE V11：基于原始 EKTM_mirt 的前置文件闭环修改方向

本文档记录 V11 相对当前本地 V10 代码的修改方向。V11 的核心目标不是重新设计一个新的知识掌握度生成方法，而是在现有 V10 流程基础上，将 `stu2know_mastery.json`、习题文本表示和知识点文本表示统一回到原始 MMKT/EKT 体系中的 `EKTM_mirt` 逻辑。

## 1. 当前判断

当前本地代码可视为 V10 基线。V10 已经完成了 SemanticConvE 的实体/关系融合、V10 图构建、消融实验和对比实验入口，但前置文件中最关键的 `stu2know_mastery.json` 仍存在方法一致性问题。

V10 之前的实现曾使用自定义的 `mastery_head` 或 MIRT proxy 来生成 mastery，这不符合我们现在的目标。正确方向应是：

```text
使用原始 EKTM_mirt 模型结构
→ 重新训练每个数据集对应的 EKTM_mirt checkpoint
→ 从 EKTM_mirt 的 know_output 导出 stu2know_mastery.json
```

因此，V11 主要修改对象是前置文件生成流程，而不是直接修改 SemanticConvE 的推荐模型主体。

## 2. V11 总体目标

V11 需要实现以下闭环：

```text
原始交互数据 / ER 对齐子集
        ↓
训练 no-Q MIRT，得到题目 a/b 与学生 theta
        ↓
训练轻量改造后的 pyKT DKT，得到 DKT checkpoint
        ↓
训练 EKTM_mirt
        ↓
从 EKTM_mirt 导出：
  1. stu2know_mastery.json
  2. exercise_text_embeddings.npy
  3. concept_text_embeddings.npy
        ↓
继续生成：
  stu2know_seq.json
  stu2know_forget.json
  stu2ex_forget.json
  stu2ex_recommend.json
  triples.txt / test_triples.txt
        ↓
训练和测试 SemanticConvE
```

其中最重要的是：

```text
stu2know_mastery.json 必须来自 EKTM_mirt 的 know_output
stu2know_seq.json 必须来自轻量改造后的 pyKT DKT checkpoint
```

不能再来自自定义 mastery head、简单统计掌握度或 MIRT proxy。
同时，`stu2know_seq.json` 不再使用 V10 中自写的 `train_pkc_dkt_v10.py` 路线。

## 3. 为什么必须使用 EKTM_mirt

我们的论文方法强调以下信息共同参与认知状态提取：

```text
学生历史答题序列
题目文本语义
题目对应知识点
MIRT 难度/区分度参数
GRU 更新后的学生知识状态
```

原始 `EKTM_mirt` 正好覆盖这些输入：

| 信息 | EKTM_mirt 中的对应模块 |
|---|---|
| 学生历史答题序列 | 按时间步输入题目编号及作答结果 |
| 题目文本 | `TopicRNNModel` / Bi-GRU 文本编码器 |
| 知识点 | `knowledge embedding` 与题目-知识点映射 |
| MIRT 参数 | `a_dict`、`b_dict`，拼接为 `topic_ab` |
| 学生状态 | `EKTSeqModel_cdm` 中的 GRU hidden state |
| 知识掌握度 | `know_output` |

因此，V11 应该复用原始 EKTM_mirt 的建模逻辑，而不是另写一套近似模型。

## 4. 原始代码需要复刻的部分

原始相关文件位于：

```text
KT/MMKT/src/model_EKT.py
KT/MMKT/src/utils/run_ekt.py
KT/MMKT/src/test_knowmastery.py
```

V11 不直接修改这些旧文件，而是将必要代码复制到 KG4ER-New 的 V11 前置文件模块中，例如：

```text
ER/KG4ER-New/v11_pipeline/ektm_mirt_model.py
ER/KG4ER-New/v11_pipeline/train_ektm_mirt_v11.py
ER/KG4ER-New/v11_pipeline/export_ektm_mirt_v11.py
```

这样可以避免污染旧 KT/MMKT 代码，同时方便在五个数据集上统一运行。

## 5. 必须修复的原始 EKTM_mirt 细节

原始 `EKTSeqModel_cdm.forward()` 内部已经计算并返回：

```python
return predict_score.view(1), h, know_output.view(-1)
```

但是原始 `EKTM_mirt.forward()` 中曾写成：

```python
s, h = self.seq_model(topic_v[0], k, topic_ab, score, hidden)
return s, h
```

这会丢失 `know_output`。V11 复制版需要改成：

```python
s, h, ko = self.seq_model(topic_v[0], k, topic_ab, score, hidden)
return s, h, ko
```

这是最小且必要的修复。

## 6. EKTM_mirt 训练目标

V11 训练 EKTM_mirt 时，训练目标仍然是原始 EKT 思路：

```text
根据学生历史状态预测当前题目答对概率
```

损失函数使用二分类交叉熵：

```math
\mathcal{L}_{resp}
= - \left[
y_t \log \hat{y}_t
+ (1-y_t)\log(1-\hat{y}_t)
\right]
```

其中：

```text
y_t      表示学生在当前题目上的真实作答结果
\hat y_t 表示 EKTM_mirt 预测的答对概率
```

V11 不再额外添加如下自定义监督：

```text
hidden → mastery_head → mastery_loss
```

因为 mastery 应该是模型内部知识状态通过 `know_output` 自然导出的结果，而不是单独新建标签强行监督。

## 7. mastery 导出方式

训练完成后，V11 使用每个数据集的 best EKTM_mirt checkpoint 导出 mastery。

对每个测试学生：

```text
按时间顺序读取其答题序列
逐题输入 EKTM_mirt
持续更新 hidden state
在最后一个时间步取 know_output
保存为该学生的知识点掌握度向量
```

形式为：

```math
\mathbf{m}_u = \operatorname{know\_output}(h_{u,T})
```

其中：

```text
h_{u,T} 表示学生 u 在最后一个交互时刻的知识状态
\mathbf{m}_u 表示学生 u 对全部知识点的掌握度
```

最终保存：

```text
ER/KG4ER-New/data/<dataset>/v11/stu2know_mastery.json
```

## 8. 文本 embedding 导出方式

V11 需要保证习题文本和知识点文本位于同一语义空间。

因此，词表构建时同时纳入：

```text
习题文本
知识点名称
知识点 definition
```

训练 EKTM_mirt 时，习题文本通过同一个 `TopicRNNModel / Bi-GRU` 编码为 `topic_v`。

训练完成后，用同一个 Bi-GRU 导出：

```text
exercise_text_embeddings.npy
concept_text_embeddings.npy
```

其中：

```text
exercise_text_embeddings.npy：
  每道题目的文本语义表示

concept_text_embeddings.npy：
  每个知识点 name + definition 的文本语义表示
```

注意：知识点文本 embedding 主要用于后续 SemanticConvE 中的知识点实体表示。为了尽量保持原始 EKTM_mirt 结构，V11 暂不把知识点文本直接加入 EKTM_mirt 的状态更新逻辑。

## 9. MIRT 参数来源

V11 仍然需要每个数据集对应的 MIRT 参数：

```text
a_param.csv
b_param.csv
theta_param.csv
```

这些参数作为 EKTM_mirt 的输入特征，而不是最终推荐模型的标签。

V11 暂不修改 no-Q MIRT 的模型结构和参数导出逻辑，仅调整默认训练配置，使其在保持 V10 严格 ER 图对齐策略的同时获得更充分的训练：

```text
MIRT epoch = 70
MIRT batch-size = 1024
MIRT lr = 0.001
```

该调整只属于训练配置调整，不改变 MIRT 模块的算法定义。MIRT 仍然作为教育学特征估计器使用，后续通过 EKTM_mirt 和 SemanticConvE 的融合模块决定其贡献大小。

题目教育学特征来源：

| 特征 | 来源 |
|---|---|
| difficulty | MIRT 的 b 参数 |
| discrimination | MIRT 的 a 向量范数 |
| theta | MIRT 学生能力参数 |

需要注意，MIRT 只是教育学参数估计器，不作为最终推荐模型。

## 10. V11 与 V10 的核心区别

| 模块 | V10 当前问题 | V11 修改方向 |
|---|---|---|
| mastery 来源 | 曾使用自定义 mastery head / proxy | 改为 EKTM_mirt 的 `know_output` |
| seq 来源 | 自写 `PKCDKT` 使用全知识点 multi-label BCE，负样本过多导致整体概率偏低 | 改为复制 pyKT DKT 训练代码，仅按说明文档轻量修改 loss |
| EKTM_mirt | 未严格复刻原始模型导出逻辑 | 复制原始结构并最小修复 |
| 文本 embedding | 存在 BGE 或独立文本向量历史方案 | 改为同一 Bi-GRU 导出 |
| 习题文本 | 可用于文本表示 | 进入 EKTM_mirt 训练和导出 |
| 知识点文本 | 仅用于后续实体表示 | 使用同一 Bi-GRU 导出 |
| MIRT 参数 | 可能和 mastery 生成脱节 | 作为 EKTM_mirt 输入统一使用 |

## 11. V11 需要新增或修改的代码

建议新增：

```text
ER/KG4ER-New/v11_pipeline/ektm_mirt_model.py
ER/KG4ER-New/v11_pipeline/prepare_ektm_inputs_v11.py
ER/KG4ER-New/v11_pipeline/train_ektm_mirt_v11.py
ER/KG4ER-New/v11_pipeline/export_ektm_mirt_outputs_v11.py
ER/KG4ER-New/v11_pipeline/build_v11_graph.py
ER/KG4ER-New/v11_pipeline/validate_v11_front_files.py
```

其中：

| 文件 | 作用 |
|---|---|
| `ektm_mirt_model.py` | 复制并最小修复原始 EKTM_mirt |
| `prepare_ektm_inputs_v11.py` | 准备文本 token、序列、Q 矩阵、MIRT 参数 |
| `train_ektm_mirt_v11.py` | 训练每个数据集的 EKTM_mirt |
| `export_ektm_mirt_outputs_v11.py` | 导出 mastery 和文本 embedding |
| `build_v11_graph.py` | 基于 V11 前置文件重新构图 |
| `validate_v11_front_files.py` | 检查分布、维度、映射和异常值 |

## 12. V11 前置文件输出目录

V11 不覆盖 V10 文件，建议新建目录：

```text
ER/KG4ER-New/data/<dataset>/v11/
```

每个数据集最终至少包含：

```text
stu2know_mastery.json
stu2know_seq.json
stu2know_forget.json
stu2ex_forget.json
stu2ex_recommend.json
stu2ex_recommend_full_precision.json
exercise_text_embeddings.npy
concept_text_embeddings.npy
semantic_kg_features/
triples.txt
test_triples.txt
entities.dict
relations.dict
Q.txt
```

MIRT 与 EKTM_mirt 训练中间文件建议保存到：

```text
ER/KG4ER-New/data/<dataset>/v11_mirt/
ER/KG4ER-New/data/<dataset>/v11_ektm_mirt/
```

## 13. 需要重点验证的统计项

V11 导出后必须检查：

```text
stu2know_mastery.json:
  shape = 测试学生数 × 知识点数
  value range 应在 [0,1]
  不应大面积集中在 0 或 1

exercise_text_embeddings.npy:
  shape = 题目数 × text_dim

concept_text_embeddings.npy:
  shape = 知识点数 × text_dim

stu2ex_forget.json:
  value range 应在 [0,1]
  应使用题目涉及知识点遗忘率的平均值，而不是求和

stu2ex_recommend.json:
  排序应使用完整精度
  保存时可以保留 rounded version
```

## 14. V11 后续实验关系

V11 只修正前置文件来源和一致性。训练策略仍可沿用 V10：

```text
rec-only negative sampling
include / exclude test triples 可配置
type-aware scoring 可消融
attention fusion 可保留
```

对比模型仍建议在 ID-only 图上运行，避免把 SemanticConvE 的新增特征泄露到传统 baseline 中。

## 15. 当前优先级

V11 优先级如下：

1. 复制并修复原始 `EKTM_mirt`。
2. 完成五个数据集 EKTM_mirt 训练入口。
3. 用 best checkpoint 导出 `stu2know_mastery.json`。
4. 用同一个 Bi-GRU 导出习题和知识点文本 embedding。
5. 重新生成 V11 图文件。
6. 用统计脚本验证前置文件分布。
7. 再运行 SemanticConvE 主模型和消融实验。

暂时不优先处理：

```text
全关系负采样
冷启动实验
知识点文本直接进入 EKTM_mirt 状态更新
MIRT 参数泛化性能评价
```

这些可以作为 V11 之后的训练策略或论文补充实验。

## 16. stu2know_seq 的 V11 修正方向

V10 中曾新增 `train_pkc_dkt_v10.py`，其任务是：

```text
当前 concept-response → 下一步知识点 multi-hot
```

该实现会把下一步真实知识点设为 1，同时把其他所有知识点设为 0。由于每个时间步只有少量正知识点、大量负知识点，普通 BCE 会被负样本主导，导致五个数据集的 `stu2know_seq.json` 整体偏低。

V11 不再使用这一路线。正式流程改为：

```text
不再使用 train_pkc_dkt_v10.py
复制 pyKT DKT 训练相关代码到 KG4ER-New
只做说明文档中的最小 loss 修改
训练得到 pyKT DKT checkpoint
继续使用 export_seq_from_dkt.py / export_seq_from_dkt_v10.py 导出 stu2know_seq.json
```

说明文档中的关键修改是将下一步真实知识点位置的标签改为 1：

```python
t = torch.masked_select(torch.ones(size=rshft.shape).cuda(), sm)
```

也就是说，pyKT DKT 仍然输出所有知识点的预测向量，但 loss 只监督下一步真实出现的知识点位置，并将这些位置视为正标签。它不会把其他全部知识点都强制作为负样本，因此可以避免 V10 自写 PKCDKT 中的概率整体塌缩问题。

V11 的 `stu2know_seq.json` 生成流程应为：

```text
ER 对齐后的 DKT 训练序列
        ↓
轻量改造后的 pyKT DKT
        ↓
保存 DKT checkpoint
        ↓
加载 checkpoint
        ↓
输入测试学生完整历史序列
        ↓
取最后一个时间步的 DKT 输出
        ↓
保存 stu2know_seq.json
```

因此，V11 中 `stu2know_seq.json` 的含义是：

```text
根据学生当前历史学习序列，由轻量改造后的 pyKT DKT 预测得到的下一步知识点趋势向量。
```

该文件继续用于构造 `pkc` 关系：

```text
kc --pkc_strength--> uid
```

其中 `pkc_strength` 来自 `stu2know_seq.json` 中对应知识点的数值。
