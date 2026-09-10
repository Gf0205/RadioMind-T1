# T1 任务包：数据管线 + O'Shea CNN 可复现基线

> 本文件是给 Codex 的完整开工契约。请整段照做，不要自创目录结构、配置格式或指标输出约定；如有偏离，必须在交付说明里逐条列出。
> 项目归属人：小高（西电通信，研二）。项目定位见文末"背景"，动手前请先读一遍。

---

## 0. 一句话目标

在 `RadioMind/` 下交付：**可复现的 RML2016.10a 数据管线 + O'Shea CNN 调制识别基线**，同一份代码能在本机（CPU）跑 smoke、在远端 GPU（AutoDL）跑全量，结果制品统一归档。

## 1. 环境事实（实测，勿假设）

| 项 | 本机（Windows，开发用） | 远端（AutoDL，全量训练） |
|---|---|---|
| OS | Windows（Git Bash / PowerShell） | Ubuntu（SSH） |
| GPU | ❌ 无 | ✅ CUDA（3090/4090 级别） |
| Python | `C:\Users\gaofe\.workbuddy\binaries\python\versions\3.13.12\python.exe`（3.13） | 3.10 / 3.11 |
| torch | 需装 **CPU 版**（index-url=https://download.pytorch.org/whl/cpu） | GPU 版（pip 默认源即可，镜像常自带） |
| 数据 | `RML2016.10a_dict.pkl` 在仓库根目录 | 上传到 `/root/autodl-tmp/RML2016.10a_dict.pkl` |
| 角色 | 写代码 + `--smoke` 冒烟验收 | 同一份代码跑全量训练 |

**推论（必须写进代码/README）**：任何路径不得写死；`device` 自动探测（cuda → mps → cpu）；训练可随时中断恢复；全量训练产物能 scp 拉回本机归档。

## 2. 硬性约束（违反即返工）

1. **`RML2016.10a_dict.pkl` 只读**：禁止修改、删除、移动、git 提交或重新分发。加载必须 `pickle.load(f, encoding='latin1')`。读取后转换为项目自有安全格式再使用。
2. **禁止引入 strict-split / "跨信道泛化"结论**：本 pickle 无 waveform seed / capture session / 源波形信息，物理上无法做跨信道切分。切分只有一种协议（见 §5）。README/文档中不得出现"验证了跨信道泛化"类表述。
3. **代码必须 device-agnostic**：`device = "cuda" if torch.cuda.is_available() else "cpu"`（可加 mps）。不允许写死 `cuda` 或假设 GPU 存在。本地跑不了不等于可以写只在远端能跑的代码——本地 `--smoke` 必须能端到端通过（含出图）。
4. **数据切分禁止全局随机 shuffle 混切**：必须按 (调制, SNR) 格子内分层切（见 §5），这是 AMC 论文最常见的泄漏错误，本项目红线。
5. **不要为凑数调参**：全 SNR 平均准确率落在文献值 83–84% 的 ±3% 区间（即 80.5–87%）即算达标。超出该区间时，先检查代码/切分 bug，再考虑调参，并在交付说明中报告。
6. **禁止在本任务引入**：RF-Net 多任务、数据增强、RMS 归一化消融、OOD/校准、LoRA/GRPO、MiniMind 集成（全部是后续任务）。RMS/标准化开关可以留配置位（默认关），但不要实现消融逻辑。

## 3. 目录契约（照此结构，不增不减必要模块）

```
RadioMind/
├── RML2016.10a_dict.pkl        # 只读源数据（勿动）
├── requirements.txt            # 锁定版本
├── README.md                   # 项目说明：结构、本机/远端两段跑法（§9）
├── configs/
│   ├── default.yaml            # 全量训练默认配置
│   └── smoke.yaml              # smoke 用配置（或由 --smoke 参数派生）
├── src/radio_mind/
│   ├── __init__.py
│   ├── config.py               # dataclass + YAML 加载；DATA_ROOT 支持 env/config 覆盖
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py             # pkl → manifest(npy 内存映射/缓存) ，latin1 加载
│   │   ├── split.py            # (mod, snr) 格子内分层 70/15/15，固定 seed
│   │   ├── dataset.py          # torch Dataset/DataLoader；仅 train 统计的预处理开关
│   │   └── check.py            # `python -m radio_mind.data --check` 入口（§8 验收 1）
│   ├── models/
│   │   ├── __init__.py
│   │   └── oshea_cnn.py        # O'Shea et al. 2016 复现（出处写 docstring）
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          # 训练循环：early stop、best+last ckpt、断点续跑
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py          # 指标计算 + 出图 + metrics.json 写入
├── scripts/
│   ├── train.py                # 入口：python scripts/train.py [--config ...] [--smoke] [--resume <run_id>]
│   └── fetch_results.sh        # 从远端拉取 results/ 到本机（远端用）
├── results/
│   └── <run_id>/               # run_id = YYYYMMDD_HHMMSS_<tag>（自动生成）
│       ├── config.yaml         # 本次运行的配置副本（含 DATA_ROOT、seed、commit hash 若有）
│       ├── metrics.json        # §7 定义的结构
│       ├── acc_vs_snr.png      # 逐 SNR 准确率曲线
│       ├── confusion_matrix.png
│       ├── loss_curve.png
│       └── checkpoints/
│           ├── best.pt         # val 最优
│           └── last.pt         # 最后一轮（续跑用）
└── data/                       # DATA_ROOT 默认值（.gitignore）
```

约定：
- **运行目录**：所有命令从仓库根目录执行；`python -m radio_mind...` 需 `PYTHONPATH=src`（scripts/ 内自行处理 sys.path 或安装为 editable 均可，README 写清）。
- **run_id 每次训练自动生成**，不接受固定目录覆盖旧结果；同一 run 续跑沿用原 run_id。
- `.gitignore` 至少忽略：`data/`、`results/`、`*.pt`、`__pycache__/`、`.venv/`。**pkl 本身不提交**（641MB，且许可未明）。

## 4. 数据模块规格

**源数据实测规格（已核对，勿改数字）**：220,000 样本；11 类调制 = `8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM`；SNR = -20..18 dB 步长 2 共 20 档；每 (调制, SNR) 恰好 1,000 条；单样本 `(2, 128)` float32（通道 0=实部 I，通道 1=虚部 Q）；无 NaN/Inf；全局 |x|max≈0.164。

**加载（load.py）**：
- 从 `DATA_ROOT/RML2016.10a_dict.pkl` 读取（`encoding='latin1'` 必须）。
- pickle 键为 `(mod_str, snr)`，值为 `(1000, 2, 128)` ndarray——**按此结构读取**，不要依赖 dict 里的 'X'/'Y'（本文件无）。
- 转成可复现流式格式：建议内存映射 npy 或 torch 能流式读的格式；manifest 含列：`sample_id, iq_path(或数组), modulation_id(0-10), mod_str, snr_db, split`。sample_id 全局唯一、可溯源到 (mod, snr, 格内序号)。
- 预处理模块：仅提供两个开关位（默认值见 default.yaml）：`preprocess.normalize: none|global|rms`（默认 **none**，贴近原版；global=用 train 统计量的标准化、rms=逐样本单位功率，后两者本期只留开关与正确实现，不跑消融）；不做其他变换。I/Q 双通道原样作为模型输入。

**切分（split.py）——本项目红线逻辑**：
- 对每个 (调制, SNR) 格子内的 1,000 条：按固定 seed（config `data.split_seed`，默认 20260907）打乱后取 **700 / 150 / 150** → train / val / test。
- **绝不**做跨格子的全局 shuffle 再切。
- split 结果写入 manifest（或缓存切分索引），保证多次加载一致；训练与切分 seed 分离（`train.seed`）。

**校验（check.py，`python -m radio_mind.data --check` 输出）**：
1. 每格三份数量 = (700, 150, 150)，11×20 格全绿；
2. 三份合起来 = 1,000，无样本丢失/重复/跨界；
3. 类别覆盖：三份均含全部 11 类；SNR 覆盖：三份均含全部 20 档；
4. 打印总样本数、每份总数、数值统计（无 NaN/Inf）供人工核对。

## 5. 评测协议（只此一套，report 只用它）

- test 集 = 全部 20 档 SNR 的每格 150 条（33,000 条）。
- **必报四件套**：
  1. **全 SNR 平均准确率**（目标 80.5–87%，文献 ~83–84%）；
  2. **SNR≥0 dB 平均准确率**（参考锚点 ≥90%，本期只报数不设门槛）；
  3. **逐 SNR 准确率**（PNG 曲线 + metrics.json 内数组）；
  4. **混淆矩阵**（PNG；建议同时存归一化矩阵数组进 JSON）。
- 附加：macro-F1（全 SNR）、每类 per-SNR F1 可选（进 metrics.json）。
- 不报：无 val 调参曲线之外的任何"隐藏集"成绩；不得把 val 当 test 用。

## 6. 模型与训练规格

**模型（oshea_cnn.py）**：复现 O'Shea, Corgan, Clancy, "Convolutional Radio Modulation Recognition Networks"（2016）。参考结构（实现时以公开复现为准并标注出处，允许小幅调整，在 README 记录差异）：
- 输入 `(2, 128)` → Conv1d 序列（in_ch=2 → 256 → 256 → 256 → 80，kernel 3，带 padding 处理），ReLU + Dropout；尾接全局池化/Flatten → 全连接 → 11 类 logits。
- 结构里**只允许一个分类头**（本期不做多任务）。损失 = CrossEntropy。
- 给 `forward` 写清楚形状注释，便于后续 RF-Net 复用代码风格。

**训练（trainer.py / scripts/train.py）**：
- 默认超参（configs/default.yaml 可覆盖）：Adam（lr 1e-3，可选 cosine/step 衰减）、batch 512、epochs 上限 50、early stop 看 val 准确率 patience 8、按 val 最优存 `best.pt`、每轮结束覆盖存 `last.pt`。
- **断点续跑**：`--resume <run_id>` 从该 run 的 `last.pt` + optimizer state + epoch 计数继续，写入同一 run_id 目录。
- **smoke 模式** `--smoke`：每格取 20 条（即从 manifest 中每格 train 抽 14/val 3/test 3 或简化每格 20 全量含三份），1 epoch，batch 64，必须完整走完"训练→评估→出 metrics.json+两张 PNG"，用于本地验证整条链路正确。smoke 结果写 `results/<run_id>_smoke/`（或 run_id 带 smoke 标记），不得污染正式 run。
- 训练过程打印：每 epoch train loss / val acc；结束打印四件套指标。
- seed：`train.seed`（默认 20260907）固定 torch/numpy/random。

**可复现性要求**：CPU 上同 seed 两次训练，metrics.json 中全 SNR 平均准确率应一致（浮点差 ≤0.1pp）；数据切分完全一致。GPU 允许微小非确定差异，核心指标差 ≤0.2pp。

## 7. metrics.json 结构（定死，勿自创）

```json
{
  "run_id": "20260908_153000_oshea",
  "config": { "data": {"data_root": "...", "split_seed": 20260907, "normalize": "none"},
              "model": {"name": "oshea_cnn"},
              "train": {"seed": 20260907, "batch_size": 512, "lr": 0.001, "epochs": 50} },
  "num_parameters": 0,
  "best_epoch": 0,
  "overall_accuracy": 0.0,
  "accuracy_snr_ge0": 0.0,
  "macro_f1": 0.0,
  "per_snr_accuracy": {"-20": 0.0, "-18": 0.0},
  "confusion_matrix": [[0.0]],
  "per_class_f1": {"8PSK": 0.0},
  "train_time_seconds": 0.0,
  "device": "cpu",
  "git_commit": ""
}
```

## 8. 验收自证清单（交付时逐条贴出命令输出）

1. `python -m radio_mind.data --check` —— §4 校验全绿。
2. 同 seed 两次全量训练（或同 seed 两次 smoke），metrics.json 核心指标差 ≤ 0.1pp（CPU）。
3. `python scripts/train.py --smoke` —— 端到端跑通，产出 metrics.json + acc_vs_snr.png + confusion_matrix.png + best.pt/last.pt，exit code 0。
4. 全量训练（可在远端 GPU 完成）metrics.json 中 `overall_accuracy` ∈ [80.5, 87.0]，report 含四件套。
5. 断点续跑验证：训练中途 Ctrl-C，`--resume` 后能继续并正常收敛/结束。
6. README 按 §9 写清两段跑法。

## 9. 本机 / 远端两段跑法（必须写进 README）

**本机（CPU 冒烟）**：
```bash
python -m venv .venv
# Windows: .venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Linux/mac: .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
set PYTHONPATH=src        # Linux: export PYTHONPATH=src
python scripts/train.py --smoke
```

**AutoDL（GPU 全量）**：
```bash
# 1) 数据一次性传数据盘（此后常驻）
scp RML2016.10a_dict.pkl root@<实例>:/root/autodl-tmp/
# 2) 代码 + 环境
git clone <repo> && cd RadioMind
pip install -r requirements.txt            # GPU torch 随镜像或默认源
# 3) 训练（数据路径由 env 覆盖，不写死）
export DATA_ROOT=/root/autodl-tmp
export PYTHONPATH=src
nohup python scripts/train.py > run.log 2>&1 &   # 或 tmux
# 4) 中断后续跑
python scripts/train.py --resume <run_id>
# 5) 拉回本机归档
bash scripts/fetch_results.sh <run_id>     # scp results/<run_id> 回本机 results/
```

**要求**：config 中 `data.data_root` 优先级为 `环境变量 DATA_ROOT > config 值 > 默认 ./data`；README 里两端各给一个可直接复制的最小命令块。

## 10. 不做清单（本期明确不做）

RF-Net 多任务（分类+SNR+校准）、任何数据增强、RMS/global 归一化消融实验、OOD/拒识、置信度校准、LoRA/SFT/GRPO、MiniMind 集成、strict split、真实 SDR 数据、RML2018.01a。交付 README 时可提"后续任务见 docs/T2…"，但不要展开实现。

## 11. 交付物清单

1. 上述目录契约下的完整代码 + requirements.txt + .gitignore + README.md；
2. 本机 `--smoke` 通过记录（§8.3）；
3. 至少一份全量训练 run（远端 GPU 或本机 CPU 均可）的 `results/<run_id>/` 完整产物（含 metrics.json 四件套达标）；若全量只在远端跑，附 fetch 后的本地路径；
4. 验收自证清单 §8 的逐条输出或说明；
5. 交付说明（若对契约有任何偏离：结构、指标、结构差异等，逐条列出理由）。

## 12. 背景（30 秒版）

RadioMind = 基于 MiniMind 的 RF 信号分析 Agent（秋招项目，通信背景）。架构：**冻结的 MiniMind 做工具调用编排 + 专用 RF-Net 做信号感知**，两者解耦。RML2016.10a 仅作为第一版感知内核的训练与评测底座，后续会有自研信道损伤/增强与开集数据。T1 只做"数据管线 + 复现 O'Shea 基线"，把数字基准和工程骨架立起来。验收口径要能讲清：切分无泄漏、指标可复现、代码一份两端跑。
