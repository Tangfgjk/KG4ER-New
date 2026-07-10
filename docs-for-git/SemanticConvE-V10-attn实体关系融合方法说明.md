# SemanticConvE V10-attn 实体与关系表示融合方法说明


## 1. 总体回答：选择哪一种 Q/K/V 方案

```text
Option 1: Q = semantic, K/V = difficulty + discrimination
Option 2: Q = difficulty + discrimination, K/V = semantic
Option 3: Q/K/V 都来自 semantic、difficulty、discrimination 等多源特征
```
采用的是 **Option 3 的 feature-token self-attention 形式**。

不人为指定“语义信息一定作为 Query”，也不指定“教育学参数一定作为 Query”。不同来源的特征会先被投影到同一维度，形成若干个 feature tokens，然后由 self-attention 自适应学习它们之间的关系。

给定某个实体或关系的特征 token 矩阵：

$$
\mathbf{X} = [\mathbf{x}_1; \mathbf{x}_2; \cdots; \mathbf{x}_m] \in \mathbb{R}^{m \times d},
$$

其中 \(m\) 表示该实体或关系拥有的特征 token 数量，\(d\) 是 ConvE 使用的 embedding 维度。

self-attention 中的 Query、Key 和 Value 由同一组 token 学习得到：

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V.
$$

attention 输出为：

$$
\mathrm{Attn}(\mathbf{X}) =
\mathrm{softmax}
\left(
\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}
\right)
\mathbf{V}.
$$

随后对 attention 输出进行池化，并通过 MLP 压回 ConvE 所需的固定维度：

$$
\mathbf{f} =
\mathrm{MLP}
\left(
\mathrm{Pool}(\mathrm{Attn}(\mathbf{X}))
\right).
$$

最终表示保留 ID embedding 作为 residual anchor：

$$
\mathbf{h} =
\mathrm{LayerNorm}
\left(
\mathbf{h}^{id} + \lambda \mathbf{f}
\right),
$$

其中 \(\mathbf{h}^{id}\) 是实体或关系的 ID embedding，\(\lambda\) 是可学习的融合强度参数。


```text
Q、K、V 都由 semantic、pedagogical、relation-type、relation-strength 等特征 token 共同生成，
而不是人为指定某一类特征固定作为 Q 或 K/V。
```

## 2. 实体表示融合

所有实体都保留 ID embedding，不同实体类型使用不同的特征集合，实体类型信息通过“特征集合差异”体现，而不是作为单独 token 输入 attention。

### 2.1 题目实体 Exercise

题目实体 \(e\) 的输入特征包括：

| 特征 | 含义 | 来源 |
|---|---|---|
| ID embedding | 题目 ID 表示 | KG 中的 `ex` 实体 ID |
| topic/text semantic | 题目文本语义表示 | EKTM_mirt 中 Bi-GRU 文本编码器导出的 `topic_v` |
| difficulty | 题目难度 | MIRT 的 \(b\) 参数归一化 |
| discrimination | 题目区分度 | MIRT 的 \(a\) 参数向量范数归一化 |

题目文本语义 token 为：

$$
\mathbf{x}_e^{text}
= \mathbf{W}_{text}\mathbf{v}_e^{topic},
$$

其中 \(\mathbf{v}_e^{topic}\) 是 EKTM_mirt 文本编码模块输出的题目文本表示。

题目教育学参数 token 为：

$$
\mathbf{x}_e^{ped}
=
\mathrm{MLP}_{ex}
\left(
[
d_e^{norm}, a_e^{norm}
]
\right),
$$

其中 \(d_e^{norm}\) 表示归一化后的题目难度，\(a_e^{norm}\) 表示归一化后的题目区分度。

因此题目实体的 feature token 集合为：

$$
\mathbf{X}_e =
[
\mathbf{x}_e^{text};
\mathbf{x}_e^{ped}
].
$$

最终题目表示为：

$$
\mathbf{h}_e =
\mathrm{LayerNorm}
\left(
\mathbf{e}_e^{id}
+ \lambda_E
\mathrm{Fusion}(\mathbf{X}_e)
\right),
$$

其中 \(\mathbf{e}_e^{id}\) 是题目 ID embedding，\(\mathrm{Fusion}(\cdot)\) 表示 feature-token self-attention + MLP。

也就是说，题目不是简单拼接或直接相加，而是：

```text
ID + Attention(topic_v/text, difficulty_mirt_norm, discrimination_mirt_norm)
```

### 2.2 学生实体 Student

学生实体 \(u\) 的输入特征包括：

| 特征 | 含义 | 来源 |
|---|---|---|
| ID embedding | 学生 ID 表示 | KG 中的 `uid` 实体 ID |
| theta | 学生能力参数 | MIRT 的 \(\theta\) 参数归一化 |
| cluster_id | 学生群体类别 | 基于学生画像或能力分布聚类 |


学生能力 token 为：

$$
\mathbf{x}_u^{\theta}
=
\mathrm{MLP}_{\theta}
\left(
[
\theta_u^{norm}
]
\right).
$$

学生群体 token 为：

$$
\mathbf{x}_u^{cluster}
=
\mathrm{Emb}(c_u),
$$

其中 \(c_u\) 是学生聚类编号。

学生实体的 feature token 集合为：

$$
\mathbf{X}_u =
[
\mathbf{x}_u^{\theta};
\mathbf{x}_u^{cluster}
].
$$

最终学生表示为：

$$
\mathbf{h}_u =
\mathrm{LayerNorm}
\left(
\mathbf{e}_u^{id}
+ \lambda_U
\mathrm{Fusion}(\mathbf{X}_u)
\right).
$$

对应实现口径为：

```text
ID + Attention(theta_mirt_norm, cluster_id)
```

### 2.3 知识点实体 Knowledge Concept

知识点实体 \(k\) 的输入特征包括：

| 特征 | 含义 | 来源 |
|---|---|---|
| ID embedding | 知识点 ID 表示 | KG 中的 `kc` 实体 ID |
| concept semantic | 知识点语义表示 | 知识点名称 + definition |

知识点语义 token 为：

$$
\mathbf{x}_k^{sem}
=
\mathbf{W}_{kc}
\mathbf{v}_k^{concept},
$$

其中 \(\mathbf{v}_k^{concept}\) 表示知识点名称与 definition 的语义向量。

知识点实体的 feature token 集合为：

$$
\mathbf{X}_k =
[
\mathbf{x}_k^{sem}
].
$$

最终知识点表示为：

$$
\mathbf{h}_k =
\mathrm{LayerNorm}
\left(
\mathbf{e}_k^{id}
+ \lambda_K
\mathrm{Fusion}(\mathbf{X}_k)
\right).
$$

对应实现口径为：

```text
ID + Attention(concept_semantic)
```


## 3. 关系表示融合

V10-attn 中，关系表示保留 relation ID embedding，并在此基础上融合关系类型和关系强度。

图中主要关系包括：

| 关系前缀 | 关系类型 | 含义 |
|---|---|---|
| `mlkc` | mastery | 学生对知识点的掌握关系 |
| `pkc` | sequence/progress | 学生学习进展或下一步知识点倾向关系 |
| `exfr` | forgetting | 题目或知识点相关遗忘关系 |
| `rec` | recommendation | 学生-题目的推荐关系 |

对于带数值的关系，例如：

```text
mlkc_0.83
pkc_0.64
exfr_0.41
```

我们将其拆分为：

$$
\mathrm{type}(r) \in
\{
\mathrm{mastery},
\mathrm{sequence},
\mathrm{forgetting},
\mathrm{recommend}
\},
$$

以及关系强度：

$$
s_r \in [0,1].
$$

例如：

$$
\mathrm{type}(\mathrm{mlkc\_0.83}) = \mathrm{mastery},
\quad
s_{\mathrm{mlkc\_0.83}} = 0.83.
$$

关系类型 token 为：

$$
\mathbf{x}_r^{type}
=
\mathrm{Emb}(\mathrm{type}(r)).
$$

关系强度 token 为：

$$
\mathbf{x}_r^{str}
=
\mathrm{MLP}_{r}
([s_r]).
$$

关系 token 集合为：

$$
\mathbf{X}_r =
[
\mathbf{x}_r^{type};
\mathbf{x}_r^{str}
].
$$

最终关系表示为：

$$
\mathbf{h}_r =
\mathrm{LayerNorm}
\left(
\mathbf{r}^{id}
+ \lambda_R
\mathrm{Fusion}(\mathbf{X}_r)
\right).
$$

对应实现口径为：

```text
relation ID + Attention(relation_type, relation_strength)
```

其中 `rec` 关系没有天然连续强度，因此设置为固定默认强度，例如：

$$
s_{\mathrm{rec}} = 1.0.
$$

## 4. 为什么 ID embedding 不进入 attention

V10-attn 保留 ID embedding 作为 residual anchor，而不是把 ID embedding 与文本、难度、区分度一起作为普通 token 输入 attention。

原因是当前任务不是严格冷启动任务。学生、题目和知识点大多已经在图中出现过，因此 ID embedding 对图结构记忆非常重要。如果直接让 ID embedding 和其他特征完全混合，模型可能失去原 ConvE 对实体身份和图结构关系的稳定建模能力。

因此我们采用：

$$
\mathbf{h}
=
\mathrm{LayerNorm}
\left(
\mathbf{h}^{id}
+ \lambda \cdot \mathrm{Fusion}(\mathbf{X})
\right),
$$

而不是：

$$
\mathbf{h}
=
\mathrm{Fusion}
\left(
[
\mathbf{h}^{id};
\mathbf{X}
]
\right).
$$

这使得新增特征起到“补充和调节”作用，而不是完全替代 ID 表示。


## 5. 简要回答

### Q1：exercise 的 semantic、difficulty、discrimination 怎么融合？

采用 feature-token self-attention，即 Option 3。

题目表示为：

$$
\mathbf{h}_e =
\mathrm{LayerNorm}
\left(
\mathbf{e}_e^{id}
+ \lambda_E
\mathrm{Fusion}
\left(
[
\mathbf{x}_e^{text};
\mathbf{x}_e^{ped}
]
\right)
\right).
$$

其中：

$$
\mathbf{x}_e^{text}
=
\mathbf{W}_{text}
\mathbf{v}_e^{topic},
$$

$$
\mathbf{x}_e^{ped}
=
\mathrm{MLP}_{ex}
\left(
[d_e^{norm}, a_e^{norm}]
\right).
$$

### Q2：student 的 ID 和 \(\theta\) 怎么处理？Q/K/V 是什么？

学生同样使用 feature-token self-attention。学生 ID embedding 作为 residual anchor，不进入 attention。学生侧 feature tokens 包括：

$$
\mathbf{X}_u =
[
\mathbf{x}_u^{\theta};
\mathbf{x}_u^{cluster}
].
$$

Q/K/V 均由该 token 集合生成：

$$
\mathbf{Q}_u = \mathbf{X}_u\mathbf{W}_Q,
\quad
\mathbf{K}_u = \mathbf{X}_u\mathbf{W}_K,
\quad
\mathbf{V}_u = \mathbf{X}_u\mathbf{W}_V.
$$

最终：

$$
\mathbf{h}_u =
\mathrm{LayerNorm}
\left(
\mathbf{e}_u^{id}
+ \lambda_U
\mathrm{Fusion}(\mathbf{X}_u)
\right).
$$

### Q3：knowledge concept 是否只有 ID？

不是只用 ID。V10-attn 中知识点使用：

```text
ID + Attention(concept_semantic)
```

其中 `concept_semantic` 来自知识点名称和 definition。对应公式为：

$$
\mathbf{h}_k =
\mathrm{LayerNorm}
\left(
\mathbf{e}_k^{id}
+ \lambda_K
\mathrm{Fusion}
\left(
[
\mathbf{x}_k^{sem}
]
\right)
\right).
$$

### Q4：relation type 和 strength 怎么处理？

关系保留 relation ID embedding，同时融合 relation type 和 strength：

$$
\mathbf{h}_r =
\mathrm{LayerNorm}
\left(
\mathbf{r}^{id}
+ \lambda_R
\mathrm{Fusion}
\left(
[
\mathbf{x}_r^{type};
\mathbf{x}_r^{str}
]
\right)
\right).
$$

其中：

$$
\mathbf{x}_r^{type}
=
\mathrm{Emb}(\mathrm{type}(r)),
\quad
\mathbf{x}_r^{str}
=
\mathrm{MLP}_{r}([s_r]).
$$

对于 `mlkc`、`pkc` 和 `exfr`，strength 由关系名中的数值获得；对于 `rec`，strength 使用固定默认值。



