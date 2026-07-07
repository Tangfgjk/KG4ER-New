# SemanticConvE V6 运行命令

本文档用于在一台电脑上运行 V6 的完整训练、测试和结果统计。

V6 的默认实验包括 8 个模型：

| 实验名 | 含义 |
|---|---|
| `full_state_hybrid` | V6 完整模型，使用 state-aware learner encoder、语义特征、IRT 特征、统计教育特征、type/strength 关系编码、type-aware scoring |
| `irt_only_ped` | 只使用 IRT 教育特征，去掉统计教育特征 |
| `stat_only_ped` | 只使用统计教育特征，去掉 IRT 特征 |
| `no_irt` | 在完整模型基础上去掉 IRT 特征 |
| `no_stat_ped` | 在完整模型基础上去掉统计教育特征 |
| `no_mastery` | 去掉 mastery 认知关系和 mastery 状态输入 |
| `no_forgetting` | 去掉 forgetting 认知关系和 forgetting 状态输入 |
| `no_seq` | 去掉 sequence/progress 认知关系和 sequence 状态输入 |

V6 默认不把 `test_triples.txt` 加入训练。测试学生由推荐前的状态文件经过 StateEncoder 生成表示，`test_triples.txt` 只用于评估指标计算。

---

## 1. 拉取 V6 代码

```powershell
cd "C:\Users\你的用户名\Desktop"
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V6
git pull origin V6
```

如果文件夹已经存在：

```powershell
cd "C:\Users\你的用户名\Desktop\KG4ER-New"
git fetch origin
git checkout V6
git pull origin V6
```

如果 GitHub 网络需要代理，可先执行：

```powershell
git config --global http.proxy http://127.0.0.1:10090
git config --global https.proxy http://127.0.0.1:10090
```

不用代理时可取消：

```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

## 2. 拷贝数据

GitHub 仓库不上传数据。请把已经生成好前置文件和语义特征的数据集拷贝到仓库根目录下的 `data` 文件夹。

推荐结构：

```text
KG4ER-New/
  codes-New-ConvE/
  docs-for-git/
  data/
    Eedi/
      entities.dict
      relations.dict
      triples.txt
      test_triples.txt
      Q.txt
      stu2know_mastery.json
      stu2know_seq.json
      stu2know_forget.json
      stu2ex_forget.json
      semantic_kg_features/
```

如果后续运行其他数据集，则结构类似：

```text
data/
  algebra2005/prepared_for_kt/...
  assist2009-sub/prepared_for_kt/...
  statics2011/prepared_for_kt/...
  XES3G5M-sub-small/prepared_for_kt/...
```

---

## 3. 激活环境

如果已经有之前安装好的环境，例如：

```powershell
conda activate kg4er_cuda
```

如果是在新电脑上首次配置，至少需要安装：

```powershell
conda create -n kg4er_cuda python=3.10 -y
conda activate kg4er_cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tqdm
```

如果只训练和统计，不重新生成文本 embedding，一般不需要安装 `sentence-transformers`。

---

## 4. 检查数据是否完整

进入仓库根目录：

```powershell
cd "C:\Users\你的用户名\Desktop\KG4ER-New"
```

检查 Eedi：

```powershell
python codes-New-ConvE\validate_semantic_ready.py `
  --datasets Eedi `
  --data-root .\data
```

如果输出 `status: passed`，说明训练前置文件完整。

---

## 5. 运行 Eedi 的 V6 完整实验

这条命令会在一台电脑上运行：

```text
8 个实验 × 5 个随机种子
```

随机种子为：

```text
2024, 2025, 2026, 2027, 2028
```

运行命令：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v6_full_suite_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --data-root .\data `
  --runs-root .\runs `
  --resume
```

说明：

- `--ablations all` 会自动展开为 V6 的 8 个实验；
- `--cuda auto` 会优先使用 CUDA，如果没有 GPU 则使用 CPU；
- `--resume` 可以直接保留，首次运行时不会影响结果；
- 默认不加入 `test_triples.txt` 训练；
- 训练输出保存在：

```text
runs/Eedi/Eedi_v6_full_suite_5seeds/
```

---

## 6. 续跑实验

如果中断，重新运行同一条命令即可：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v6_full_suite_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --data-root .\data `
  --runs-root .\runs `
  --resume
```

续跑逻辑：

- 已完成的 seed 会跳过；
- 未完成训练但存在 `last.pt` 的 seed 会从上一轮继续；
- 已完成训练但未完成测试或评估的 seed，会继续后面的阶段；
- 不要更换 `--run-id`，否则会被视为新实验。

---

## 7. 统计结果

实验全部跑完后执行：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v6_full_suite_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations all `
  --runs-root .\runs
```

统计结果会生成在：

```text
runs/Eedi/Eedi_v6_full_suite_5seeds/summaries/
```

重点查看：

```text
summaries/paper_table.md
summaries/paper_table.csv
summaries/per_seed_metrics.csv
summaries/mean_std_metrics.csv
summaries/gate_values_per_seed.csv
summaries/gate_values_mean_std.csv
summaries/summary.json
```

其中：

- `paper_table.md` 适合快速查看论文表格；
- `paper_table.csv` 适合复制到 Excel 或进一步排版；
- `gate_values_mean_std.csv` 可用于分析 V6 自动学习到的特征权重；
- `per_seed_metrics.csv` 可查看每个随机种子的原始结果。

---

## 8. 单个随机种子调试命令

如果想先快速测试一个 seed：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --run-id Eedi_v6_debug_seed2024 `
  --seeds 2024 `
  --ablations full_state_hybrid `
  --epochs 1 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --data-root .\data `
  --runs-root .\runs `
  --resume
```

确认能跑通后，再运行完整 5 seeds 实验。
