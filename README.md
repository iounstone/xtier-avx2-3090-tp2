# xtier-avx2-3090-tp2

[XTier](https://github.com/a775828b-dot/xtier) 的个人 fork —— 把"**消费级 Ampere 双卡 + 纯 PCIe 无 NVLink + Zen2 无 AVX-512**"这一档跑通，并把这条路上必须改的东西整理成可以直接往上提交的补丁。

主线支持的是 SM120（RTX 50 系 / 专业卡）和 AVX-512 服务器 CPU。30 系卡既没有硬件 FP8 转换，也没有 P2P；双路 EPYC 7532 是 Zen2，只有 AVX2。这个 fork 就是补这几处的缺口：**5 个分支，2 个阻塞启动的修复、2 个 decode 加速、1 个本机生产配置。**

## 硬件

| | |
|---|---|
| GPU | 2 × RTX 3090 24 GB（GA102，SM86，82 SM，936 GB/s） |
| 卡间 | **纯 PCIe，无 NVLink / 无 P2P**，`nvidia-smi topo -m` 显示 `SYS` |
| CPU | 2 × EPYC 7532（Zen2，32C/64T ×2 = **128 线程**），**无 AVX-512 / 无 VNNI，只有 AVX2** |
| 内存 | 251 GiB DDR4，NPS4，4 个 NUMA node |
| 模型 | DeepSeek-V4.1-Flash-1017，`--tp-size 2`，`context_length = max_total_tokens = 262144`，`max_running_requests=1` |
| 存储 | Engram 嵌入表行放 NVMe（`XTIER_ENGRAM_ROWS=nvme`），换 ~190 GB 主机内存，prefill −25% |

三个约束决定了这一档的所有取舍：

1. **48 GB 显存装不下 720 专家 × 40 层**（约 260 GB 权重）→ 必须走 CPU/GPU 分层 MoE，专家常驻内存、按热度换进显存槽位。每槽 0.35 GiB/层，26 个主槽就吃掉 9.1 GiB，还要给 262144 token 的 KV 池（1.21 GB）和满配 prefill 工作区留位置。
2. **没有 P2P**→ TP=2 的 all-reduce 走 NCCL 时每步 83 次代理调用、67.4 µs/次。这是改开关解决不了的，必须换成 PCIe-IPC all-reduce（见下）。
3. **没有 AVX-512**→ CPU 专家内核只能用 `XTIER_ISA=avx2`。Zen2 每 socket 的双通道带宽吃不满，CPU 侧一步要 436 µs × 40 层 ≈ 17.4 ms，是 decode 的主要矛盾。

## 与主线的差异

5 个分支全部重基于上游最新 `main`（`behind=0`），`tools/repo_check.py` 零 findings。

| 分支 | 类型 | 规模 | 做了什么 |
|---|---|---|---|
| `fix/marlin-mxfp4-format` | **阻塞启动** | 1 文件 +5 | 30 系上 SGLang 对 MXFP4 专家选 `marlin`，format 表里没有 `Mxfp4MarlinMoEMethod` 这一条 → 模型起不来就报 `no CPU engine format for the MoE method`。补 `Fp8Fp4Experts` 映射（打包的 E2M1 权重 + UE8M0 scale，名字与 Fp8MoEMethod 相同） |
| `fix/flashmla-hook-and-engram-peer` | 修复 | 1 文件 +25/−1 | FlashMLA 已搬出 sgl-kernel，改在 `sglang.kernels.ops.attention.flash_mla_interface` 后面，`DeepseekV4AttnBackend.__init__` 用 `functools.partial` 绑它；只 hook 旧的 `sgl_kernel.flash_mla` 会让 sparse-MLA 的 SM86 kernel 白装。注册加了存在性探测：`register()` 不解析目标，失败发生在 `apply_hooks()`，每个不存在的目标每次启动每 rank 留一条带 traceback 的 error（本机实测 **322 条**） |
| `perf/wo-a-fast-e4m3` | decode 加速 | 3 文件 +101/−5 | 用整数位运算重写 e4m3 解码（`_e4m3_fast`），并推广到 `tl.dot` / `einsum` 两条路径。穷举 256 个 E4M3 码逐位比对：**255 个完全相同**，唯一差异是 `0xFF` 那个 NaN 的符号位（`0x7fc00000` vs `0xffc00000`，两者都是 NaN）。附 `tests/wo_a_e4m3_fast_check.py` |
| `perf/wo-a-splitk-gemv` | decode 加速 | 4 文件 +324/−5 | decode T=1 时 wo_a 退化成 GEMV，`tl.dot` 的网格受 **CTA 数量**而非算力限制。改成 `(BLOCK_R, group, split)` 三维网格沿 D 切分累加到 FP32 scratch，`split = max(1, min(8, ceil(4*SM/ctas)))`，只在 `T == 1` 且网格 < 3×SM 时启用（prefill 不受影响）。附 `tests/wo_a_splitk_check.py` |
| `config/ds41-avx2-nps4-3090-tp2` | 配置 | 3 文件 +177 | 本机生产配置，见下 |

split-K 那个分支顺手修了两个自己跑测试才发现的 bug：`_gemv_kernel` 没有 T 维而 dispatch 写了 `T <= 2`（T=2 时只有第 0 行正确，其余是 `torch.empty` 残留）；`_scratch_f32` 按 `numel` 而非 `shape` 做 key（同元素数的不同形状会串用对方 stride）。前者本机日志里**从未出现 T=2**（只有 T=1/6/7），所以历史跑分没错，但并发 2 请求就会踩。

还没有提上去、只在本地的一处：`_convert` 里 `w.shape == (sn*BLOCK, sk*BLOCK)` 在 TP>1 下永不成立，导致 wo_a 永不进 FP8、白留 BF16。

## 速度

口径：`bench_ascii.py` 走 `/v1/chat/completions`，每档跑 ≥4 次，方差约 2%。**不要用单次 `gen_once`**，TTFT 占比大，同配置能漂 25.8 → 35.1。

**decode（tok/s，单请求，262144 满配上下文）**

| 版本 | decode | 相对上一版 |
|---|---|---|
| e85 起点 | 33.25 | — |
| e90 +位运算解码 / `GB_R=4` / 60 线程 | 35.34 | +6.3% |
| **e93 +FASTDEC 推广** | **36.55** | **+10.0%** |
| 现生产（26 主槽 + elastic） | 35.78 / 34.86 | 满配 prefill 有 35 次非致命 OOM 警告 |

三个加速开关叠在无开关的基座上是 **28.59 → 35.78（+25%）**，但前提是 `DSPARK=0`：开草稿时每步要先等 CPU 算出 6 个草稿 token（约 23 ms，占步长 60%），GPU 侧的优化全被这段等待盖住，只剩 +7%。

**分项**

| 改动 | 实测 |
|---|---|
| `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`（上游自带，只是打开） | NCCL 每步 83 次 × 67.4 µs → IPC 79 次 × p50 **2.9 µs**；decode all-reduce 降到 **1.39 ms/步** |
| split-K GEMV | 四个 decode shape 带宽 535→567 / 481→607 / 423→532 / 396→506 GB/s；wo_a 合计 **6.41 → 5.53 ms/步（−13.7%）** |
| `_e4m3_fast` | wo_a kernel **44.7 → 33.6 µs**，`tl.dot` 路径 **53.3 → 38.8 µs（1.37×）**，prefill **1625.9 → 1810.9 tok/s**，输出逐位相同 |

**上下文**：262144（261073 token 实测）单次 prefill 156.7 s 通过，输出正常；KV 池只占 1.21 GB（`fp8_e4m3`），**没有为了速度砍上下文**，也没有为了上下文砍槽位到不可用的程度。

## 生产配置

`configs/ds41-avx2-nps4-3090-tp2/run-ds41.sh`，默认值就是上表那条 35.78 的定档：

- **槽位**：26 主槽 + elastic（reserve 2.5 GiB，可长到 32 槽）。每槽 0.35 GiB/层，主槽边际效率 14.9 ms/GiB
- **显存瘦身**：`SGLANG_DSV4_SWA_CHUNKS_IN_FLIGHT=1.2`（SWA 池 35840 → 22784 token）、`SGLANG_TORCH_SCORE_BUDGET_MIB=256`（indexer score chunk 1024 → 256 MiB，能把崩溃栈从 `scoring.py` 移走，存活 prefill 6.2K → 15.5K token）
- **CPU 引擎**：`XTIER_THREADS=60`（4 NUMA node × 15 worker + 1 空闲核给 dispatcher）、`XTIER_SPIN_COUNT=1048576`（只作用于 decode；prefill 另走 2048）
- **chunked prefill 必须 16384**，降到 4096 是 732 vs 1664 tok/s 的纯亏

**踩过的坑**：启动时 cwd 必须在仓库父目录，不能在 `sglang-1030/` 这一层 —— 该层的 `xtier/` 没有 `__init__.py`，`import xtier` 会命中它变成 namespace package，插件静默加载失败，然后 Engram 走原生 GPU 路径、单层申请 45.78 GiB 直接 OOM。自查：`grep -c "Failed to load plugin"` 必须为 0，`grep -c "Engram layer.*on NVMe"` 应该是 4（2 层 × 2 rank）。

## 边界

- **只在 3090 / SM86 上验证过**。SM89+ 有硬件 FP8 转换，应保留 `arch.e4m3_f32`，不要用 `_e4m3_fast`。
- **单请求**（`max_running_requests=1`），没有并发数据。split-K 在 T>1 已禁用、回退 `tl.dot`。
- `XTIER_THREADS=64` 会把 decode 打到 **2.16 tok/s**（dispatcher 被 worker 的 SMT 兄弟饿死）。60 是硬上限。
- 26 主槽 + 262144 那档满配 prefill 会有约 35 次非致命 OOM 警告（申请 20 MiB 时只剩 6–8 MiB），allocator 重试后成功、无异常、输出正常。要清零把主槽降到 24，或走 20 主槽 + 32 草稿那档（decode 26.6，预填充零警告）。
- **`XTIER_SPIN_COUNT` 不往下调**：动机是省空转功耗换 CPU 频率，但降到 131072 以下可能让 worker 掉进 futex 睡眠，属 decode 副作用，主动放弃。

## 复现

```bash
git -C /path/to/sglang checkout 6162b07199          # 上游测试过的 main
SGLANG_BUILD_RUST_EXTS=none pip install -e /path/to/sglang/python
tools/build_engine.sh /path/to/site-packages/nvidia/cu13
pip install --no-deps -e /path/to/xtier-avx2-3090-tp2      # 分支 config/ds41-avx2-nps4-3090-tp2
cd ..   # cwd 铁律，见上
bash xtier/configs/ds41-avx2-nps4-3090-tp2/run-ds41.sh
```

配置在分支 `config/ds41-avx2-nps4-3090-tp2` 上，合入上游前需先 checkout 该分支；脚本里每个路径都可以用环境变量覆盖。

## 分支

```
fix/marlin-mxfp4-format            1 文件  +5      阻塞 SM86 启动
fix/flashmla-hook-and-engram-peer  1 文件  +25/−1  入口搬家 + 去掉 322 条 traceback
perf/wo-a-fast-e4m3                3 文件  +101/−5 位运算解码，逐位等价
perf/wo-a-splitk-gemv              4 文件  +324/−5 叠在 perf/wo-a-fast-e4m3 上
config/ds41-avx2-nps4-3090-tp2     3 文件  +177    本机生产配置
```

上游：<https://github.com/a775828b-dot/xtier>。每个分支都重基于当天的最新 `main`，`python tools/repo_check.py` → `0 findings`。

---

# 上游 XTier README（以下为原文，未改动）

[English](docs/README.en.md) · [文档目录](docs/README.md) · [快速开始](docs/quickstart.md) · [支持情况](docs/support.md)

XTier 是官方 [SGLang](https://github.com/sgl-project/sglang) 的插件（跟 main 分支），做 CPU/GPU 分层 MoE 推理：专家分布在显存和内存两层，按热度放置、按需调度；NVMe 只放大嵌入表的行（DeepSeek 的 Engram、Qwen 的 PLE）。内存放不下全部专家时，可选的冷专家池把少用的专家放到硬盘、用到时读回内存，池子大小和速度受限于硬盘速度和类型。

**状态：早期开发**（[工作计划](docs/PLAN.md)）。Qwen3.8 Flash Next、GLM-5.3 Flash、DeepSeek V4.1 Flash 1017 跑在 SGLang main 上；MiniMax-M3.1-Flash 跑在 SGLang 的 M3.1 分支（PR #45925），已暂停开发（本机太慢、智力不强）。

- **不改 SGLang 源码**：只用官方插件框架的钩子（接在官方 KTransformers 混合推理的入口上），加载后自检是否接管成功。
- **与模型无关**：混合推理包在官方 `FusedMoE` 的量化方法外面，用标准 MoE 层的模型装上就能跑（NVFP4、MXFP4 专家）。
- **加速包可选**：某个模型独有的修正和内核放在 `xtier/packs/` 下。

## 结构

```
xtier/
├── backend/   混合 MoE 后端：接管官方量化方法，权重加载完即交给 CPU 引擎；解码走信箱（CUDA 图可捕获），prefill 走 GPU 分组 GEMM 或 CPU
├── cache/     热专家缓存：每层槽数可不同，全部常驻的层不经过 CPU；自适应换入换出；常驻层的 prefill 直接在槽上算；
│              第二张卡当热专家池（可选，`XTIER_GPU2_SLOTS`）
│              冷专家池（可选，`XTIER_COLD=1`）：少用的专家放到硬盘（[设计](docs/design-cold-experts.md)）
├── planner/   规划器：本机探测的 GPU / CPU 耗时 + 路由档案 → 显存预算内每层的热专家槽数（按 CPU 和 GPU 同时算完分配），可在启动时自动算
├── engine/    CPU 引擎的 Python 绑定、绑核、锁页内存（C++ 源码在 csrc/）
├── packs/     加速包：qwen4_exp（PLE 表、PLE 行放 NVMe、hc_mix 重叠内核、QSA 读 NVFP4 KV）、glm5_next（量化排除、DSA、NextN）、
│              deepseek_v4（Engram 行放 NVMe、SM120 候选索引器、FP8 GEMM 调优配置、FP8 wo_a、recipe 分词、分页索引器）、
│              minimax_m3（SM120 通用栈、CUDA 上的 DSpark verify、草稿词表头 FP8、SM120 上读 NVFP4 KV）、
│              speculative（通用投机解码：EAGLE 的热词表配无损拒绝采样、草稿温度、按置信度截断验证；DSpark 按置信度截断验证）
├── tables/    大嵌入表的行放在 NVMe：多段多文件、任意行宽、行缓存、CUDA 图内查表
├── console/   启动台：一个网页选模型、调启动参数、启动 / 停止，跟踪加载阶段与 decode 遥测（只用标准库）
└── platform.py  与模型无关的主机修正（TileLang 目标选择）
csrc/          CPU 引擎：AVX-512 / AVX2 专家内核、NUMA 线程池、CPU 与 GPU 之间的信箱通信
configs/       每个"模型 × 硬件"的启动配置，console.py 是它在启动台里的条目（选项、默认值、启动命令）
tests/         单层测试（引擎、信箱、缓存、prefill、格式适配、hc_mix、投机截断、QSA NVFP4、GPT-OSS SwiGLU、wo_a、MiniMax NVFP4 KV、快速加载、未命中代算）、启动台与规划器检查
tools/         构建、内存看门狗、规划器（plan_probe 探测、plan_trace 路由档案、plan_auto 出方案、plan_calibrate 校准，见 docs/planner.md）
bench/         验收脚本（PPL、照抄、验证步、并发、prefill 延迟）与结果
```

## 快速上手

从空机器到第一次请求的完整路线（环境、模型准备、启动命令、测试请求）见[快速开始](docs/quickstart.md)。已经有 SGLang 环境时：

```bash
git -C /path/to/sglang checkout 6162b07199      # 测试过的官方 SGLang main（2026-10-30）
SGLANG_BUILD_RUST_EXTS=none pip install -e /path/to/sglang/python   # 没装 Rust 时跳过 SGLang 可选的 Rust 扩展
tools/build_engine.sh /path/to/site-packages/nvidia/cu13   # CPU 引擎与信箱库 -> xtier/engine/lib
pip install --no-deps -e .                      # 注册插件入口 sglang.srt.plugins: xtier
python tools/check_sglang.py --strict           # 不用模型：核对 SGLang 的提交、XTier 的钩子目标和 import
XTIER=1 XTIER_THREADS=112 XTIER_MAIN_CPUS=auto XTIER_CACHE_SLOTS=155 \
  python -m sglang.launch_server --model-path <MoE 检查点> --cuda-graph-backend-prefill disabled ...
```

XTier 跟着官方 SGLang main 走，每次只认一个测试过的提交（[xtier/__init__.py](xtier/__init__.py) 的 `SGLANG_COMMIT`，更新时写进
版本更新）；装的 SGLang 不是这个提交时，启动日志会警告，`tools/check_sglang.py` 列出挂不上的钩子（加 `--strict` 时提交不同也算失败）。
依赖的版本见 [constraints.txt](constraints.txt)。

开关见 [xtier/config.py](xtier/config.py)；按模型和硬件的完整启动参数见 [configs/](configs/)，也可以用下面的启动台。

快速加载（可选，`XTIER_FAST_LOAD=1`；默认关，即 SGLang 原来的加载方式）：专家直接从模型文件（开冷专家时从冷存储）读进
CPU 引擎，全部读完、检查完才报在线，DeepSeek V4.1 Flash 1017 启动 472 → 186 s（[报告](docs/reports/2026-10-27-fast-load.md)）。
MoE 张量并行（`--tp-size` > 1）时每个 rank 只读自己那份（双 RTX 4090 实测 Qwen3.8 权重加载 170 → 133 s，分片与普通加载逐字节相同）。

## 启动台

![XTier 启动台](docs/images/console.png)

在仓库根目录运行 `python3 -m xtier.console`（只用 Python 标准库），打开 http://127.0.0.1:8100/ ：

- **左边模型栏**：每个模型一张小卡片，写着端口和 README 实测速度，标签注明测法（AGENT = agent 速度，文本 = 文本速度），
  模型多了在栏内滚动。
- **中间启动参数**：默认就是实测配置，改过的项会标出。公共的有加速包开关（关 = 测速表里的「基线」）、投机解码、上下文、
  CPU 专家线程，其余是各模型自己的（热专家槽、分块、弹性热专家、冷专家卸载、未命中专家 GPU 代算等）。
- **右边运行状态**：加载阶段（权重 → 草稿 → 专家池 → CUDA 图 → 预热 → 在线），在线后显示 decode 速度、接受长度、显存和日志。

一次只跑一个模型；模型服务在自己的端口上提供 OpenAI 兼容接口，启动台退出不影响它。加模型只要在 `configs/<模型>-<硬件>/` 里
加一个 `console.py`（卡片、选项、启动命令）。细节见 [docs/console.md](docs/console.md)。

## 支持的模型

| 模型 | 官方模块 | 状态 |
|---|---|---|
| Qwen3.8 Flash Next | `qwen4_exp` | 官方 main 上跑通：精度、照抄、512K 召回与生产一致；加速包比基线 冷 prefill 快 5% / 24%（64K / 512K）；KV 存 NVFP4（与生产同参，加速包让官方 QSA 能读），省下的显存用来多放常驻层，启动台默认由规划器自动布局；PLE 表（48 GiB）可放 NVMe，放傲腾时速度与锁页表持平 |
| GLM-5.3 Flash | `glm5_next` | 官方 main 上跑通（含 EAGLE）：加速包（FP8 权重、热词表拒绝采样、截断验证、弹性热专家槽）比基线 长回复每步快 22%，PPL 3.987 → 3.998 |
| DeepSeek V4.1 Flash 1017 | `deepseek_v4` | 官方 main 上跑通（含 DSpark）：Engram 表（189 GB）的行放 NVMe（[设计](docs/design-tables-nvme.md)），照抄 0 错；加速包（含 FP8 wo_a、分页索引器、弹性热专家槽）比基线 长回复每步快 39%、冷 prefill 快 33% |
| MiniMax-M3.1-Flash | `minimax_m3` | **暂停开发**（2026-10-26，本机太慢、智力不强）。SGLang 的 M3.1 分支（PR #45925）上跑通（含 DSpark），加速包比官方代码长回复每步快 34% |

## 支持的显卡

每个模型、显卡和多卡模式实际验证到什么程度（整模型、截层、单层、模拟）见[支持情况](docs/support.md)。

| 显卡 | 状态 |
|---|---|
| Blackwell（SM120） | 开发和测速平台（RTX PRO 5000），下面的测速都在这上面测 |
| RTX 40 系（SM89） | 支持。RTX 4090 实测：Qwen3.8 整模型在单卡 24 GB + 120 GB 内存上照抄 0 错、120K 上下文（[配置](configs/qwen38-rtx4090-3090/README.md)）；GLM-5.3 截层、DeepSeek V4.1 单层测试通过，整模型在本机模拟验证；GLM 要关投机解码。详见[报告](docs/reports/2026-10-29-rtx40.md) |
| RTX 30 系（SM86） | 支持。RTX 3090 实测：Qwen3.8 整模型在单卡 24 GB 上照抄 0 错、120K 上下文，decode 约 42 tok/s（云主机 28 个 CPU 线程）；GLM-5.3 截层、DeepSeek V4.1 单层测试通过，整模型在本机模拟验证。FP8 全部软件解码（与硬件逐位相同），GLM 用 BF16 KV、要关投机解码。详见[报告](docs/reports/2026-10-29-rtx30.md) |

## 多卡（张量并行）

两张卡用张量并行（`TP=2`）：每张卡放每个热专家的一半和一半 KV，每卡能放的热专家槽数是单卡的三倍多；CPU 引擎每个 rank 一个进程，
各用自己 GPU 那组 NUMA 节点（每颗 CPU 带一张卡时正好一卡一 CPU）。两卡之间不需要 P2P / NVLink，SGLang 会自动改走 NCCL。
双 RTX 3090 实测 Qwen3.8：decode 42.1 → 59.6–61.7 tok/s（+42–48%，开 MTP 69–72 tok/s），120K 首字 72.6 → 47.0 s，照抄 0 错。
两张卡显存不一样大时，SGLang 默认拒绝启动，加 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` 可以跑，但两张卡都只能按小卡的显存放
（24 + 16 GiB 实测每层 64 槽，decode 51 tok/s）。
用法见[配置](configs/qwen38-rtx4090-3090/README.md#two-gpus-tp2)，详见[报告](docs/reports/2026-10-30-tp.md)。

### 异构卡：快卡带慢卡，第二张卡纯当热专家池

两张卡显存或速度不同时，TP 只能按小卡、慢卡的节奏跑。第二张卡可以不放模型，只放热专家（`XTIER_GPU2_SLOTS`，默认关）：

```
XTIER_GPU2_SLOTS=auto CUDA_VISIBLE_DEVICES=快卡,慢卡 bash configs/qwen38-rtx4090-3090/run-qwen.sh
```

- **做法**：主卡没命中、第二张卡有的专家由它算，结果经内存加回 CPU，不需要 P2P；prefill 只用主卡。
- **分配**：`auto` 留 1.5 GiB，其余显存放专家，分工自动调整。
- **精度**：Qwen、GLM、DeepSeek 单层结果和主卡逐位相同。
- **限制**：只支持 `TP=1`，不能和冷专家同用；主卡有硬件 FP4 解码时第二张卡也要有。

测速只比较四种用法：同规格卡 TP、异构卡 TP、同规格卡热池、异构卡热池，单卡作基准。都用 RTX 4090 模拟（Qwen3.8，单请求，输出 512 token），方法见报告。

| 配置 | 每层热专家槽 | 32K decode tok/s | 120K decode tok/s | 120K 首字 |
|---|---|---:|---:|---:|
| 单卡 24 GiB（基准） | 48 | 41.0（100%） | 43.6（100%） | 60.9 s（100%） |
| 同规格卡 TP 24 + 24 GiB | 180（半个专家） | 62.1（151%） | 63.4（145%） | 68.9 s（88%） |
| 异构卡 TP 24 + 16 GiB | 64（半个专家） | 51.1（125%） | 49.4（113%） | 60.0 s（101%） |
| 同规格卡热池 主 24 + 热池 24 GiB | 48 + 178 | 58.2（142%） | 56.2（129%） | 59.3 s（103%） |
| **异构卡热池 主 24 + 热池 16 GiB** | **48 + 113** | **57.3（140%）** | **55.9（128%）** | **58.9 s（103%）** |
| 异构卡热池 主 24 + 热池 16 GiB，热池卡耗时约 1.2 倍 | 48 + 113 | 54.5（133%） | 54.2（124%） | 59.0 s（103%） |

括号是相对单卡的速度，首字按耗时反算。异构卡热池比异构卡 TP 快 12–13%；同规格卡 TP 的 decode 最快。

热池不参与 prefill，首字 103% 不是加速；同规格卡 TP 的 88% 是那次 prefill 卡了约 9 s。

只看趋势：这台 CPU 偏弱，CPU 越强第二张卡省得越少；真实异构卡还没测过，DeepSeek 和 GLM 只测了单层和截层。

照抄：开启时 9 次启动有 3 次翻同一个 token，其余 0 错；单卡反复重算提示词 KV 也会翻。详见[报告](docs/reports/2026-10-31-second-gpu-pool.md)。

## 测速

单卡 RTX PRO 5000 72GB + 2× EPYC 9334（NPS4）+ 384 GiB DDR5，单位 tok/s，只放最新一次的结果。这些表是 XTier 跟 SGLang main 8cd3c494 时
（2026-10 下旬）测的，后面的版本没有重测，每个版本只回归 64K / 512K 速度和精度（最近一次见[报告](docs/reports/2026-11-01-audit-fixes.md)）。

用 [bench/agent_bench.py](bench/agent_bench.py) 模拟编码代理：5 个会话从 64K 到约 500K，铺满 512K KV 池，多步调用工具，同一次启动
连跑两轮。冷 prefill = 会话第一个请求里未命中缓存的 token / 首字，decode 只算长回复（≥ 300 token）。基线 = 启动台「关加速包」：
官方 SGLang main 加 XTier 的混合推理和跑起来必需的修正，不开提速项，**不是原版 SGLang**。配置、精度和原始数据见[测速细节](docs/benchmarks.md)。

### DeepSeek V4.1 Flash 1017

| DSpark | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 3,812 | 4,768 | 141.8 | 4.50 | 31.7 ms |
|  | 2 | 5,027 | 4,844 | 140.8 | 4.30 | 30.6 ms |
| 基线（关加速包） | 1 | 3,318 | 3,627 | 65.5 | 3.25 | 50.9 ms |

加速包对基线：每步 31.2 对 50.9 ms（快 39%），冷 prefill 快 33%；精度无损，照抄 0 错。

### Qwen3.8 Flash Next

| MTP | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 5,715 | 5,170 | 181.7 | 2.57 | 14.2 ms |
|  | 2 | 5,610 | 5,150 | 207.2 | 2.68 | 13.0 ms |
| 基线（关加速包） | 1 | 5,319 | 4,078 | 139.5 | 2.13 | 15.2 ms |

加速包（规划器自动布局）对基线：每步 13.6 对 15.2 ms（快 11%），冷 prefill 64K / 500K 快 6% / 27%；精度轻微损失
（KV 存 NVFP4，与生产相同），PPL 3.481，照抄 0 错。

### GLM-5.3 Flash

| EAGLE | 轮 | 冷 prefill 64K 会话 | 冷 prefill 500K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 2,648 | 2,640 | 96.3 | 2.54 | 26.4 ms |
|  | 2 | 2,715 | 2,635 | 98.2 | 2.66 | 27.1 ms |
| 基线（关加速包） | 1 | 2,663 | 2,518 | 76.8 | 2.61 | 34.4 ms |

加速包（58 槽 + 弹性热专家槽）对基线：每步 26.9 对 34.4 ms（快 22%），冷 prefill 64K / 500K 快 1% / 5%；精度轻微损失
（注意力 / 稠密层存 FP8，PPL 3.987 → 3.998），照抄 0 错。

### MiniMax-M3.1-Flash（暂停开发）

| DSpark | 轮 | 冷 prefill 64K 会话 | decode（长回复） | 接受长度 | 每步 |
|---|:-:|---:|---:|---:|---:|
| XTier + 加速包 | 1 | 1,854 | 53.4 | 3.04 | 56.9 ms |
|  | 2 | 1,877 | 50.0 | 2.70 | 54.0 ms |
| 官方代码 + 必要修正 | 1 | 1,824 | 46.8 | 3.94 | 84.3 ms |
|  | 2 | 1,883 | 45.5 | 3.82 | 84.0 ms |

SGLang 的 M3.1 分支（PR #45925），固定 512K，只跑 64K 会话；加速包对官方代码 + 必要修正每步 55.4 对 84.1 ms（快 34%）。

## 冷专家卸载到硬盘

内存放不下全部专家时，每层只把一部分专家放内存，其余放到 NVMe 上的冷专家池，路由到时读回内存再算（`XTIER_COLD=1`，默认关，
[设计](docs/design-cold-experts.md)）。冷存储存全部专家，每个专家按 NUMA 节点切成 2.35 MB 的片分到各块盘，一次缺失所有盘同时读；
池子大小和速度受限于硬盘的速度和类型。启动台的 DeepSeek 卡片有「冷专家卸载」（关 / 10% / 20% / 30%）。

测试用的冷存储是 **4 块 PCIe 5.0 NVMe**：三星 9100 PRO 1TB ×4（一张 x16 转接卡，四盘同读 50.6 GB/s），机器同上。下面四个模型的
回归是之前在 4 块 PCIe 4.0 盘（傲腾 P5800X ×2、三星 990 PRO、HP FX700）上跑的。

**四个模型都能用**，各按卸载 30% 跑过端到端回归（照抄、开 / 关冷专家逐 token 对比 logprob、15 分钟 dsh 会话：上下文填到约
20 万 token、召回、编码），全部通过：

| 模型 | 每层在内存的专家 | 冷存储（每块盘） | 照抄 / dsh 15 分钟 | 所需内存（不开冷专家） | 所需内存（卸载 30%） |
|---|---|---|---|---:|---:|
| DeepSeek V4.1 Flash 1017 | 269 / 384 | 72.2 GB | 0 错 / 18 次 0 失败 | 约 298 GiB | 约 224 GiB（−74） |
| GLM-5.3 Flash | 202 / 288 | 42.8 GB | 0 错 / 15 次 0 失败 | 约 174 GiB | 约 129 GiB（−45） |
| Qwen3.8 Flash Next | 358 / 512（CPU 上的 18 层） | 6.4 GB | 0 错 / 65 次 0 失败 | 约 121 GiB | 约 116 GiB（−5） |
| MiniMax-M3.1-Flash | 90 / 128 | 58.1 GB | 0 错 / 10 次 0 失败 | 约 227 GiB | 约 165 GiB（−62） |

所需内存 = 空闲时整机的可用内存（369 GiB）减去运行期间（含加载）可用内存的最低点（`tools/mem_guard.sh` 每秒采样
MemAvailable），不含显存；DeepSeek 不开冷专家取 6 次启动的中位数。Qwen 大部分层的专家在显存里，只有 CPU 上的 18 层卸载，所以省得少。

### DeepSeek V4.1 Flash 1017：卸载比例与速度

4 块 PCIe 5.0 盘上的实测，最后一列是之前 4 块 PCIe 4.0 盘的每步耗时。单位 tok/s。

| 卸载 | 放到硬盘的专家 | 每步缺失 | decode（每步 / 速度） | 64K 冷 prefill | 增量 prefill 首字 | PCIe 4.0 每步 |
|---|---:|---:|---|---|---|---:|
| 不卸载 | 0 | — | 30.0 ms / 142 | 4,300 | 0.84 s | 30.5 ms |
| 10% | 27 GiB | 2.5 | 31.6 ms / 135（−5%） | 3,784（−12%） | 1.06 s | 32.8 ms |
| 20% | 54 GiB | 8.3 | 34.4 ms / 124（−13%） | 3,742（−13%） | 1.13 s | 37.8 ms |
| 30% | 81 GiB | 17.2 | 37.9 ms / 112（−21%） | 3,731（−13%） | 1.70 s | 43.9 ms |
| 40% | 108 GiB | 29.5 | 44.5 ms / 96（−32%） | 3,672（−15%） | 1.37 s | 约 58 ms |
| 50% | 134 GiB | 49.2 | 53.5 ms / 80（−44%） | 3,537（−18%） | 1.77 s | 73.8 ms |

- 实测用 agent_bench 的 64K 会话：decode 是长回复（≥ 300 token）的每步耗时（不卸载取前后两次的平均），速度统一按每个 DSpark 验证步
  4.26 token 折算，便于逐行比较；64K 冷 prefill 是会话第一个请求（66K token）；增量 prefill 是后续请求（每次新增约 300 token）首字时间的
  中位数。都是单次运行，有 ±5–10% 波动。
- 每次缺失多花约 0.5 ms（18.8 MB ÷ 四盘 50 GB/s 约 0.38 ms），和盘的带宽对得上。长 prefill 在 10% 时已经慢 12%、到 50% 也只慢 18%：
  主要是冷模式每层同步一次的固定开销，不是盘不够快。
- 盘要做几分钟的持续读测试：无 DRAM 的 HP FX700 满速读约 100 秒后会塌到约 60 MB/s（停下约 7 分钟才恢复），一次缺失要等最慢的
  那块盘，一块这样的盘会拖住整个池子；几秒的短测看不出来。
- 每档明细和精度（照抄 0 错，NLL 在两次启动之间的波动内）见[报告](docs/reports/2026-11-01-cold-pcie5.md)。

### 大模型只存一份

冷存储本来就存全部专家，所以模型文件里的专家可以不留：第一次按 `XTIER_COLD=1` 启动写好冷存储，停服务后运行 `tools/cold_punch.py`，
逐字节核对后在模型文件里打洞释放这部分空间（文件大小、修改时间、文件头和其他张量都不变）。DeepSeek V4.1 Flash 1017 的副本实测：占用
290 → 21 GiB，核对加打洞 42 秒；打洞后启动，内存里的专家逐层校验一致，照抄 0 错，速度和精度都在波动内。之后这些层只能从冷存储加载，
配置不对时启动直接报错，不会读到全零；`--restore` 可以写回（174 秒，sha256 全部一致）。代价：冷存储成了唯一一份，任何一块盘坏了都要
重新下载模型；没有 XTier 的官方 SGLang 不能加载打洞后的目录。

### 最小化运行：32 GB 显存 + 128 GB 内存跑 DeepSeek V4.1 Flash 1017

模拟一张 RTX 5090（32 GB）+ 128 GB 内存 + 4 块 NVMe（冷存储当时是 4 块 PCIe 4.0 盘），启动台卡片「DeepSeek V4.1 Flash 1017 ·
最小化」：冷专家 70%（每层内存只放 384 个里的 115 个）、常驻热专家每层 4 槽 + 弹性、8K 分块、128K 上下文、DSpark。显存和内存都是
硬上限（GPU 仍是 RTX PRO 5000，只给它 32 GB）：`tools/vram_limit.py` 占住 32,000 MiB 以外的显存，服务所在的 systemd scope 设
`MemoryMax=128G` 且不许用 swap。内存带宽是本机的 24 通道；换成 8 通道（约 200 GB/s）速度也差不多（按日志估算：prefill 基本不变、
decode 约慢 10%），主要受限于硬盘速度。

| 单位 tok/s | 32K | 64K | 128K |
|---|---:|---:|---:|
| prefill（32 GB + 128 GB） | 1,186 | 1,165 | 1,155 |
| decode（32 GB + 128 GB） | 28.5 | 28.1 | 32.7 |
| prefill（正常配置：72 GB 显存，专家全在内存） | 3,886 | 4,609 | 4,655 |
| decode（正常配置） | 130.7 | 127.9 | 122.4 |

- 测法：真实文档拼成的提示，清前缀缓存后测冷 prefill（提示 token / 首字），再 decode 512 token（温度 0）；两次启动各两轮
  （prefill 取每次启动的第 2 轮）。内存峰值 110.5 GiB、显存 29.5 GB，照抄 0 错。
- 约为正常配置的 25%，瓶颈在读盘：prefill 每个 8K 分块都要把约 70% 的专家从 4 块盘重读一遍，decode 每步约 100 个缺失。详见
  [报告](docs/reports/2026-10-27-ds41-minimal.md)。

## 版本更新

- 2026-11-01（v0.1.0）：全面代码审计后修了换槽序号回绕、后台线程报错被吞、首个请求现编内核被内存守护杀掉等问题，失效的加速项启动时直接报错；
  冷专家池换到 4 块 PCIe 5.0 盘（卸载 30% 时 decode −21%），大模型的专家可以只存一份。详见[修复](docs/reports/2026-11-01-audit-fixes.md)、[冷专家](docs/reports/2026-11-01-cold-pcie5.md)。
- 2026-10-31：第二张卡可以不放模型、纯当热专家池（可选，`XTIER_GPU2_SLOTS`，默认关），给显存或速度不一样的两张卡用：双 RTX 4090 模拟 24 + 16 GiB 时 Qwen3.8 decode 比同样配置的 TP 快 12–13%，比单卡快约 28%。详见[文档](docs/reports/2026-10-31-second-gpu-pool.md)。
- 2026-10-30（下午）：跟进官方 SGLang main 6162b07，安装步骤写明测试过的提交，装的版本不同时启动会警告，`tools/check_sglang.py`
  不用模型就能核对钩子；快速加载支持 MoE 张量并行（双 RTX 4090 实测通过）；英伟达官方 GLM NVFP4 检查点开 MTP 不再崩溃。详见[文档](docs/reports/2026-10-30-sglang-6162b07.md)。
- 2026-10-30：双卡张量并行（`TP=2`）实测通过：双 RTX 3090 上 Qwen3.8 decode +42–48%、120K 首字 −35%，照抄 0 错；修了
  `~/.cache` 是软链接时 SGLang 的 JIT 内核每次启动都重编（30 系单卡启动省约 5 分钟）。详见[文档](docs/reports/2026-10-30-tp.md)。
- 2026-10-29（晚）：支持 RTX 30 系（SM86）：没有 FP8 硬件，FP8 一律读时软件解码（与硬件逐位相同），DeepSeek 的 FP8
  投影只量化权重；RTX 3090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](docs/reports/2026-10-29-rtx30.md)。

更早的版本更新见 [docs/changelog.md](docs/changelog.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
