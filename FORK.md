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
