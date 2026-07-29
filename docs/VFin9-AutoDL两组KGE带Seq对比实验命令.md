# VFin9 AutoDL 两组 KGE 带 Seq 对比实验命令

本文档用于在两台 AutoDL 实例上运行 VFin9 的带 `seq` 图对比实验。

本轮只跑 KGE 对比模型：

```text
TransE, TransE-adv, RotatE, DistMult, ComplEx
```

每个数据集跑 3 个随机种子：

```text
2024, 2025, 2026
```

使用的数据图是：

```text
Data_Fin/<dataset>/er_graph_with_seq
```

该图包含 304 个关系：

```text
rec + mlkc + pkc + exfr
```

其中 `pkc` 只用于 KGE 图训练与 `rec` 推荐边构建，最终推荐打分仍使用各个 KGE 模型自己的打分函数。

## 1. 拉取代码

在 AutoDL 的 `/root/autodl-tmp` 下操作：

```bash
cd /root/autodl-tmp

git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New

git fetch origin
git checkout VFin9
git pull origin VFin9
```

如果已经克隆过：

```bash
cd /root/autodl-tmp/KG4ER-New
git fetch origin
git checkout VFin9
git pull origin VFin9
```

## 2. 拷贝数据

需要把本地的整个 `Data_Fin` 拷贝到 AutoDL 仓库根目录：

```text
/root/autodl-tmp/KG4ER-New/Data_Fin
```

至少要包含这五个目录：

```text
Data_Fin/Eedi/er_graph_with_seq
Data_Fin/algebra2005/er_graph_with_seq
Data_Fin/statics2011/er_graph_with_seq
Data_Fin/assist2009-sub/er_graph_with_seq
Data_Fin/XES3G5M-sub-small/er_graph_with_seq
```

如果要重新校验或后续重建图，建议直接拷贝完整 `Data_Fin`。

## 3. 安装依赖

AutoDL PyTorch 镜像一般已经带 CUDA 版 `torch`，不要重复安装 `torch`。

```bash
cd /root/autodl-tmp/KG4ER-New

sed '/^torch$/d' requirements.txt > requirements_no_torch.txt

pip install -r requirements_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install -r requirements_no_torch.txt -i https://pypi.org/simple

python -c "import numpy,pandas,sklearn,torch,tqdm,wandb,einops,matplotlib; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果遇到 `pandas` 下载失败，可改用 conda：

```bash
conda install -y pandas scikit-learn tqdm matplotlib
pip install wandb einops -i https://pypi.org/simple
```

## 4. 处理脚本换行

Windows 上传后的 `.sh` 可能带 CRLF，运行前执行一次：

```bash
cd /root/autodl-tmp/KG4ER-New
sed -i 's/\r$//' scripts/run_autodl_vfin9_kge_seq_group.sh
chmod +x scripts/run_autodl_vfin9_kge_seq_group.sh
```

## 5. 两台 AutoDL 分组运行

### GPU 1：Eedi、algebra2005、statics2011

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin9_kge_seq_group.sh group1
```

### GPU 2：assist2009-sub、XES3G5M-sub-small

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin9_kge_seq_group.sh group2
```

## 6. 续跑命令

如果中断后继续：

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin9_kge_seq_group.sh group1 resume
```

另一台：

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin9_kge_seq_group.sh group2 resume
```

## 7. 只重新统计

如果模型已经跑完，只想重新生成 summaries：

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin9_kge_seq_group.sh group1 summarize-only
bash scripts/run_autodl_vfin9_kge_seq_group.sh group2 summarize-only
```

## 8. 输出位置

每个数据集结果保存在：

```text
runs/<dataset>/<dataset>_kge_with_seq_3seeds
```

统计结果保存在：

```text
runs/<dataset>/<dataset>_kge_with_seq_3seeds/summaries
```

例如：

```text
runs/Eedi/Eedi_kge_with_seq_3seeds/summaries
runs/algebra2005/algebra2005_kge_with_seq_3seeds/summaries
```

## 9. 单独跑某几个数据集

脚本也支持逗号分隔的数据集列表：

```bash
bash scripts/run_autodl_vfin9_kge_seq_group.sh Eedi,statics2011
```

续跑：

```bash
bash scripts/run_autodl_vfin9_kge_seq_group.sh Eedi,statics2011 resume
```

只统计：

```bash
bash scripts/run_autodl_vfin9_kge_seq_group.sh Eedi,statics2011 summarize-only
```
