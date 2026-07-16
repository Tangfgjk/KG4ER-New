# V-Fin AutoDL 五 GPU 运行指南

本文用于在五台 AutoDL GPU 实例上并行运行五个数据集的正式 V-Fin 实验。每台实例负责一个数据集，依次完成：前置文件生成、SemanticConvE 完整模型与消融实验、ID-only 对比模型、统计结果。

## 1. 实例与分工

建议为每个数据集创建一台独立实例。RTX 4090 24 GB、Python 3.10、PyTorch + CUDA 12.1 的镜像即可；不需要在一台实例上同时跑两个数据集。

| 实例 | 数据集 | 启动命令中的数据集名 |
| --- | --- | --- |
| GPU 1 | Eedi-sub | `Eedi` |
| GPU 2 | Algebra 2005 | `algebra2005` |
| GPU 3 | ASSISTments 2009-sub | `assist2009-sub` |
| GPU 4 | Statics 2011 | `statics2011` |
| GPU 5 | XES3G5M-sub-small | `XES3G5M-sub-small` |

请把仓库放在 `/root/autodl-tmp/`，不要放在系统盘临时目录。浏览器关闭或 SSH 断开不会停止实例中的 `tmux` 会话；但不要点击实例的“关机”或“释放”。

## 2. 首次准备实例

在 AutoDL 的 JupyterLab Terminal 中依次执行：

```bash
cd /root/autodl-tmp
git clone -b V-Fin --single-branch https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
```

安装除 PyTorch 外的依赖。AutoDL PyTorch 镜像通常已经含有与 CUDA 匹配的 GPU 版 `torch`；不要让 `requirements.txt` 再下载或替换它。

```bash
sed '/^torch$/d' requirements.txt > requirements_no_torch.txt
pip install -r requirements_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

最后一条必须显示 `True` 和实际 GPU 名称，例如 `NVIDIA GeForce RTX 4090`。

若系统没有 `tmux`，安装它：

```bash
apt-get update && apt-get install -y tmux
```

仓库的 `Data_Fin/` 含五个数据集的标准 `raw/` 输入。前置模型、checkpoint、图文件和实验结果均不应从其他电脑直接混用；每台实例会为自己的数据集正式重建。

## 3. 创建断线不中断的会话

以 Eedi 为例：

```bash
cd /root/autodl-tmp/KG4ER-New
tmux new -s kg4er-eedi
```

在 tmux 内运行任务后，按 `Ctrl+B`，松开，再按 `D`，即可退出会话但训练继续运行。之后可用：

```bash
tmux attach -t kg4er-eedi
```

重新查看。若仅需检查 GPU：

```bash
nvidia-smi
```

## 4. 一键运行一个数据集

仓库提供脚本：

```text
scripts/run_autodl_vfin_dataset.sh
```

首次运行应使用 `--with-front`。它会先以正式参数重建该数据集的前置文件和 `er_graph`，再顺序执行：

1. SemanticConvE `full` 与 9 个消融模型；
2. 每个 SemanticConvE 模型的随机种子 `2024,2025,2026,2027,2028`；
3. 对比模型 `TransE`、`TransE-adv`、`RotatE`、`DistMult`、`ComplEx`、`EB-CF`、`SB-CF`、`CBF`、`KCP-ER`；
4. 自动汇总 SemanticConvE 和对比模型结果。

SemanticConvE 的完整模型和消融实验会传入 `--include-test-triples`；该文件不含 `rec` 边，仅提供测试学生的认知状态边。对比模型严格只训练 `triples.txt`。

### GPU 1：Eedi

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi --with-front
```

### GPU 2：Algebra 2005

```bash
bash scripts/run_autodl_vfin_dataset.sh algebra2005 --with-front
```

### GPU 3：ASSISTments 2009-sub

```bash
bash scripts/run_autodl_vfin_dataset.sh assist2009-sub --with-front
```

### GPU 4：Statics 2011

```bash
bash scripts/run_autodl_vfin_dataset.sh statics2011 --with-front
```

### GPU 5：XES3G5M-sub-small

```bash
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small --with-front
```

所有输出同时写入屏幕和 `logs/<dataset>_vfin_<timestamp>.log`。`RotatE` 的训练显著慢于其他对比模型，这是预期现象；脚本会等待其完成后继续。

## 5. 续跑

前置文件已成功生成、但后续 ER 训练被中断时，不要再次传 `--with-front`。从仓库根目录运行：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi --resume
```

将 `Eedi` 替换成该实例负责的数据集。`--resume` 会跳过已有完成标记的训练、测试与评估任务，只继续未完成部分。

如果中断发生在前置文件生成阶段，则应使用同一条首次命令重新开始：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi --with-front
```

`--with-front` 会覆盖该数据集的 `front_features/` 并完整重建 `er_graph/`，但不会改动 `Data_Fin/<dataset>/raw/`。

## 6. 单独运行或统计

脚本已经自动统计。需要手动重复统计时，以 Eedi 为例：

```bash
DS=Eedi
SEEDS=2024,2025,2026,2027,2028
ABLATIONS=full,id_only,no_theta,no_text,no_exercise_ped,no_relation_features,no_mastery,no_forgetting,no_seq,no_type_aware_scoring

python codes-New-ConvE/summarize_semantic_results.py \
  --dataset "$DS" \
  --run-id "${DS}_vfin_semantic_5seeds" \
  --runs-root runs \
  --seeds "$SEEDS" \
  --ablations "$ABLATIONS"

python comparison_models/summarize_dataset_results.py \
  --dataset "$DS" \
  --run-dir "runs/${DS}/${DS}_vfin_comparison_5seeds"
```

生成的主要文件为：

```text
runs/<dataset>/<dataset>_vfin_semantic_5seeds/summaries/paper_table.md
runs/<dataset>/<dataset>_vfin_comparison_5seeds/summaries/
```

对比模型中的 KGE 模型会按五个随机种子独立训练。`EB-CF`、`SB-CF`、`CBF`、`KCP-ER` 属于确定性传统基线，当前实现只运行一次，不存在随机初始化的五次重复训练。

## 7. 打包并取回结果

在每台实例完成后：

```bash
DS=Eedi
tar -czf "/root/autodl-tmp/${DS}_vfin_results.tar.gz" "runs/${DS}"
```

随后在 JupyterLab 左侧文件浏览器中下载 `/root/autodl-tmp/Eedi_vfin_results.tar.gz`。不要下载或提交 `Data_Fin/<dataset>/front_features/*/*.pt` 等 checkpoint；论文统计只需要 `runs/<dataset>/` 下的结果。

## 8. 常见问题

- `tmux: command not found`：执行 `apt-get update && apt-get install -y tmux`。
- `torch.cuda.is_available()` 为 `False`：当前镜像不是 GPU PyTorch，重新选择 PyTorch + CUDA 12.1 镜像，或安装与实例 CUDA 兼容的 PyTorch。
- `Graph directory not found`：先使用 `--with-front` 生成 `Data_Fin/<dataset>/er_graph/`。
- 任务失败后续跑：先阅读 `logs/` 和对应 `runs/<dataset>/.../runner_*.log`，修复环境问题后使用 `--resume`。
- 关闭浏览器或 JupyterLab 标签页：tmux 中的任务继续运行；关闭或释放 AutoDL 实例则会停止计算。
