# SemanticConvE V11：Eedi 数据集完整运行命令

本文档只写 Eedi 数据集的 V11 运行流程。所有命令默认在项目根目录执行：

```powershell
cd "C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New"
conda activate kg4er_cuda
```

V11 的目标是重新闭环生成前置文件，并把结果保存到：

```text
ER/KG4ER-New/data/Eedi/v11/
```

原始数据仍从旧目录读取，不移动、不覆盖：

```text
ER/KG4ER/data/Eedi/
```

## 1. 生成 MIRT 输入

```powershell
python ER\KG4ER-New\v11_pipeline\prepare_mirt_inputs_v11.py `
  --dataset Eedi `
  --force
```

作用：

- 使用 ER 图已对齐的 Eedi-sub 学生、题目和作答记录；
- 生成 no-Q MIRT 训练所需输入；
- 避免把完整 Eedi 当成 Eedi-sub 使用。

## 2. 训练 no-Q MIRT

```powershell
python ER\KG4ER-New\v11_pipeline\train_mirt_noq_v11.py `
  --dataset Eedi `
  --epoch 70 `
  --batch-size 1024 `
  --lr 0.001 `
  --device cuda `
  --force
```

作用：

- 将 MIRT 作为教育学特征估计器；
- 输出题目难度、区分度和学生能力参数；
- 后续 EKTM_mirt 和 SemanticConvE 都复用这组教育学参数。

## 3. 训练修正版 pyKT DKT 并导出 stu2know_seq.json

```powershell
python ER\KG4ER-New\v11_pipeline\train_pykt_dkt_v11.py `
  --dataset Eedi `
  --epochs 50 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --device cuda `
  --force
```

作用：

- 复制 pyKT DKT 到 V11 工作目录；
- 只在复制版 pyKT 中修改 DKT loss，使标签改为“下一步知识点出现”；
- 训练新的 DKT checkpoint；
- 导出：

```text
ER/KG4ER-New/data/Eedi/v11/stu2know_seq.json
```

如果训练已完成，只想从已有 checkpoint 重新导出：

```powershell
python ER\KG4ER-New\v11_pipeline\train_pykt_dkt_v11.py `
  --dataset Eedi `
  --device cuda `
  --skip-train
```

## 4. 训练 EKTM_mirt

```powershell
python ER\KG4ER-New\v11_pipeline\train_ektm_mirt_v11.py `
  --dataset Eedi `
  --epochs 30 `
  --batch-size 16 `
  --device cuda `
  --force
```

作用：

- 使用学生作答序列、题目文本、知识点映射和 MIRT 参数训练 EKTM_mirt；
- 保存 best checkpoint；
- 训练结束后导出 `stu2know_mastery.json`、习题文本 embedding 和知识点文本 embedding。

如果训练已完成，只想使用 best checkpoint 重新导出：

```powershell
python ER\KG4ER-New\v11_pipeline\train_ektm_mirt_v11.py `
  --dataset Eedi `
  --device cuda `
  --export-only
```

## 5. 生成遗忘相关文件

```powershell
python ER\KG4ER-New\v11_pipeline\generate_forgetting_v11.py `
  --dataset Eedi `
  --theta 10000000 `
  --timestamp-unit auto `
  --force
```

输出：

```text
stu2know_forget.json
stu2ex_forget.json
```

V11 中 `stu2ex_forget.json` 按题目涉及知识点的遗忘率平均值重新计算，避免旧流程中“求和后超过 1”的问题。

## 6. 导出 SemanticConvE 特征文件

```powershell
python ER\KG4ER-New\v11_pipeline\export_mirt_features_v11.py `
  --dataset Eedi `
  --force
```

作用：

- 汇总学生、题目、知识点、关系所需特征；
- 使用 EKTM_mirt/Bi-GRU 导出的文本 embedding；
- 不再使用 BGE/SentenceTransformer；
- 写入：

```text
ER/KG4ER-New/data/Eedi/v11/semantic_kg_features/
```

## 7. 构建 V11 ER 图

默认推荐公式使用更直观的 sequence 项：

```text
(1 - cos(Q_j, seq_i))^2
```

运行：

```powershell
python ER\KG4ER-New\v11_pipeline\build_v11_graph.py `
  --dataset Eedi `
  --sequence-term one_minus_cos_sq `
  --force
```

输出：

```text
stu2ex_recommend.json
stu2ex_recommend_full_precision.json
triples.txt
test_triples.txt
entities.dict
relations.dict
```

如需保留旧公式对照：

```powershell
python ER\KG4ER-New\v11_pipeline\build_v11_graph.py `
  --dataset Eedi `
  --sequence-term legacy_cos_sq `
  --force
```

## 8. 校验 V11 前置文件

```powershell
python ER\KG4ER-New\v11_pipeline\validate_v11_front_files.py `
  --datasets Eedi
```

重点检查：

- 学生、题目、知识点数量是否与 ER 图一致；
- `stu2know_mastery.json`、`stu2know_seq.json` 是否维度正确；
- 遗忘率是否在合理范围；
- 文本 embedding 是否来自 EKTM_mirt/TopicRNNModel；
- 是否仍混入 BGE embedding。

## 9. 前置推荐分数预评估

```powershell
python ER\KG4ER-New\v11_pipeline\score_v11_front_files.py `
  --dataset Eedi
```

作用：

- 直接用 V11 手工推荐距离做一次“前置分数”评估；
- 距离越小越推荐，脚本会自动转成评估器需要的高分优先格式；
- 输出：

```text
ER/KG4ER-New/data/Eedi/v11/front_oracle_eval/
```

这一步用于提前判断前置文件本身是否异常，不等价于 SemanticConvE 最终结果。

## 10. 训练与测试 SemanticConvE 主模型和消融实验

进入新代码目录：

```powershell
cd "C:\Users\29694\Desktop\我的文件\陆子欣师姐\2025陆子欣\Code-New\ER\KG4ER-New"
```

运行 Eedi 五个随机种子的完整实验：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v11 `
  --run-id Eedi_v11_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

续跑命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v11 `
  --run-id Eedi_v11_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

如果需要保持旧代码 transductive 设置，即训练时加入 `test_triples.txt`：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v11 `
  --run-id Eedi_v11_include_test_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --include-test-triples
```

## 11. 统计实验结果

在 `ER/KG4ER-New` 目录下运行：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v11_5seeds `
  --seeds 2024,2025,2026,2027,2028
```

如果统计 include-test 版本：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v11_include_test_5seeds `
  --seeds 2024,2025,2026,2027,2028
```

结果目录：

```text
ER/KG4ER-New/runs/Eedi/<run-id>/summaries/
```

## 12. 新电脑运行需要拷贝什么

如果在新电脑只跑 Eedi，拉取 V11 代码后，拷贝这个目录：

```text
ER/KG4ER-New/data/Eedi/v11/
```

放到新电脑仓库中：

```text
KG4ER-New/data/Eedi/v11/
```

然后进入：

```powershell
cd KG4ER-New
```

即可运行第 10 节和第 11 节命令。

