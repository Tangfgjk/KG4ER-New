# Data_Fin 前置文件管线

该目录是当前唯一的前置文件生产入口。它只读取项目内的：

```text
Data_Fin/<dataset>/raw/
```

所有输出都写入同一数据集目录，并且不读取或修改旧 `data/`、`v10_pipeline/`、`v11_pipeline/` 的数据：

```text
Data_Fin/<dataset>/front_features/
Data_Fin/<dataset>/er_graph/
```

## 输入与输出

| 阶段 | 训练数据与约束 | 输出 |
| --- | --- | --- |
| `prepare_front_protocol.py` | `raw/interactions_all.csv` 与 `raw/student_split.csv` | 固定外层训练/测试学生队列及内层验证划分 |
| `train_q_mirt.py` | 仅外层训练学生更新共享 `a/b`；Q 矩阵约束题目区分度；测试学生只优化自身 `theta` | Q 约束 MIRT 的 `a/b/theta`、难度、区分度 |
| `train_multilabel_seq.py` | 仅外层训练学生训练；输入为题目 Q 向量与正误；标签是下一题的完整多标签 Q 向量 | `stu2know_seq.json` |
| `train_ektm_mirt.py` | 仅外层训练学生训练；冻结后对所有学生前向导出 | `stu2know_mastery.json`、题目/知识点 Bi-GRU 文本向量 |
| `generate_forgetting.py` | 所有学生各自历史；不更新共享参数 | `stu2know_forget.json`、按题目知识点平均的 `stu2ex_forget.json` |
| `build_semantic_features.py` | 复用上述 MIRT、EKTM 与元数据 | SemanticConvE 需要的语义/教育学特征 |
| `build_er_graph.py` | 训练学生写入 `triples.txt`，测试学生写入 `test_triples.txt` | `er_graph/` |

测试学生从不更新 MIRT、序列模型和 EKTM-MIRT 的共享参数。测试学生的 `theta` 仅在冻结题目参数下进行个体适配；其他状态均由冻结模型前向导出。

## 一键运行

在项目根目录运行：

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets Eedi `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --sequence-epochs 30 `
  --ektm-epochs 30 `
  --device cuda `
  --force
```

五个数据集：

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --sequence-epochs 30 `
  --ektm-epochs 30 `
  --device cuda `
  --force
```

仅重建最终图并校验：

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets Eedi `
  --data-fin-root Data_Fin `
  --stages graph,validate `
  --force
```

## 后续训练入口

SemanticConvE：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id Eedi_raw_front_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full `
  --cuda auto
```

对比模型同样传入 `--data-root Data_Fin --graph-subdir er_graph`。它们只使用 `entities.dict`、`relations.dict`、`triples.txt`、`test_triples.txt` 与 `Q.txt`，不读取 SemanticConvE 的额外特征。
