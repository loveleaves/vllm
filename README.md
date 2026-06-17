# v0.15.1 学习
[官方readme](https://github.com/vllm-project/vllm/tree/v0.15.1)

## 本仓库的学习资料

- [`example.py`](example.py)：最小离线推理示例，对标 `nano-vllm/example.py`，演示真实 vLLM 的 `llm.generate` / `llm.chat` 两种写法。
- [`开发与测试指南.md`](开发与测试指南.md)：从零搭建可开发/调试/测试环境（安装、依赖、跑测试、源码阅读路线）。
- [`architecture.md`](architecture.md)：整体架构梳理。
- [`core_concepts.md`](core_concepts.md)：核心概念详解。

## 说明
### vLLM 的架构演进：三个时代

vLLM 的代码库大致经历了三个架构阶段，理解这个分期是回答你问题的关键：

**V0（legacy，~v0.7.x 及更早）**：经典的同步 `LLMEngine`，调度器明确区分 prefill/decode 两个阶段，KV cache 支持 GPU↔CPU 换出。这套架构从 v0.9.x 起被冻结，vLLM 团队提议正式废弃 V0 并将其实现从代码库中移除，从 v0.9 开始 V0 代码库被冻结只允许小修复，v0.10 标志着 V0 代码开始被移除。也就是说 V0 是一条正在死去的分支，跟最新代码的差距会越来越大，**不建议作为学习对象**。

**V1（v0.8.0 起成为默认引擎，至今仍是主体架构）**：这是一次彻底的核心重写，引入了 scheduler、KV cache manager、worker、sampler、API server 的全面重构，但仍与 V0 共享模型实现、GPU kernel 等基础组件。V1 取消了 prefill/decode 的传统区分，把 prompt token 和生成 token 统一处理，调度结果就是一个简单的 {request_id: num_tokens} 字典。这套 EngineCore / AsyncLLM / Scheduler / KVCacheManager / Executor / GPUModelRunner 的分层，**到今天（v0.22.x）依然是整个系统的骨架**，没有发生根本性变化。

**Model Runner V2（MRV2，v0.18.0 起，2026年4月引入，目前仍在迁移中）**：这是对 V1 内部 worker 层的局部重构，把原本纠缠在一起的 gpu_model_runner.py 巨石文件拆分成约 1168 行的核心加 40 个子模块，并采用 async-first、零 CPU-GPU 同步点的设计。但 截至 v0.18.0，线性注意力模型（Qwen3.5、Nemotron 3 Super）等仍不支持 MRV2，MRV2 仍处于实验阶段、尚未功能完整，直到最近的版本仍有大量针对 MRV2 的 bug fix 和迁移 PR 在合并（如让 Llama/Mistral 等 dense 模型也启用 MRV2）。换句话说，**最新主干代码里 V1 的 GPUModelRunner 老实现并没有消失**，很多模型路径目前还走的是它，MRV2 是叠加在 V1 骨架之上、还在逐步铺开的"局部换件"。

### 结论：选哪个版本

**结论是：选一个"V1 已成唯一架构、但 MRV2 还没出现"的版本，大致是 v0.9.x ~ v0.16.x 这个区间。** 理由分两层：

第一，架构相似度。最新代码的整体形态（EngineCore + AsyncLLM + Scheduler + KVCacheManager + Executor + GPUModelRunner 的进程/模块划分，统一调度而非 prefill/decode 分离）是 V1 在 v0.8.0 就定下来的，这一层"骨架"在 v0.9~v0.16 和 v0.22 之间几乎没差别。MRV2 只是把 GPUModelRunner 这一个模块内部拆得更细、做了 async-first 改造，并不改变你对整个系统的宏观理解。所以学这个区间的版本，得到的架构心智模型对理解最新代码依然成立——甚至最新代码里很多模型走的还是这条路径。

第二，学习压力。v0.9~v0.16 这段时间里：
- V0 的遗留分支逻辑已经清理掉了（不用同时理解两套引擎共存的判断逻辑）；
- `gpu_model_runner.py` 还是一个相对单一、完整的大文件，可以一口气读下来理清"输入准备 → forward → 采样"的全流程，而不是在 MRV2 的 40 个子模块间跳转、再去理解 `StagedWriteTensor`、`tmp_states`/`states` 双缓冲这类为零同步设计服务的复杂技巧；
- async scheduler、persistent batch 等特性此时已经存在（作为可选项），所以你接触到的调度概念跟最新版是一致的，只是还没有 MRV2 引入的那层额外抽象。

如果想再往"更接近最新"靠一点，可以选 v0.13~v0.16 这一段——异步调度等特性更成熟，但仍是 MRV2 引入（v0.18.0）之前，GPUModelRunner 还是单文件。

### 跟 nano-vllm 的映射关系

nano-vllm（GeeeekExplorer 的极简实现，~1200 行）本质上就是对 vLLM V1 架构的"教学版复刻"，对象基本能一一对应：

- `Sequence` / 请求对象 ↔ vLLM `v1/request.py` 的 `Request`
- `Scheduler` ↔ `v1/core/sched/scheduler.py`（同样是不区分 prefill/decode 的统一调度）
- `BlockManager`（KV cache 块分配） ↔ `v1/core/kv_cache_manager.py` / `block_pool.py`
- `ModelRunner`（forward + 采样） ↔ `v1/worker/gpu_model_runner.py`（V1 时代的单文件版本）
- `LLMEngine` 主循环 ↔ `v1/engine/core.py`（EngineCore）+ `v1/engine/llm_engine.py`

nano-vllm 里没有的、也是真实 vLLM 里"新增的学习内容"主要在：多进程解耦的 `AsyncLLM`/`EngineCore` 通信（msgpack 序列化、ZeroMQ IPC）、`MultiprocExecutor`（张量并行/流水线并行的分布式执行）、CUDA Graph 捕获、prefix caching 的完整实现细节。这些恰恰是 v0.9~v0.16 版本里就已经齐备、且形态稳定的部分。

### 不建议的两个方向

不建议啃 V0：它正在被移除，prefill/decode 分离的调度模型和 nano-vllm 的统一调度心智模型不一致，学了之后还要"反学习"，性价比低。

也不建议直接啃最新 main 分支 / MRV2：一是它本身还在快速变动（大量 bugfix PR 还在合并），文档和讲解资源相对少；二是 40 个子模块 + 零同步设计的复杂度对一个刚从 nano-vllm 上来的人来说跳跃太大，容易在还没建立宏观认知前就陷入细节。比较合理的路径是：先用 v0.9~v0.16 这类版本把"骨架 + GPUModelRunner 单体"读透，建立起从 nano-vllm 到真实 vLLM 的桥梁，之后再单独读 [Model Runner V2 设计文档](https://docs.vllm.ai/en/latest/design/model_runner_v2/) 作为"增量补丁"去理解最新的 worker 层改造，这样学习曲线会平滑很多。
## v0.15.1说明
v0.15.1 是个相当不错的选择，正好落在我们上一轮分析的"甜蜜区间"（V1 纯架构、MRV2 尚未全面铺开）的靠后端，比 v0.9 系列新很多但又没踩进 MRV2 全面接管的复杂期。下面具体说说它的定位和特点。

**时间和版本距离**：根据 PyPI 发布记录，v0.15.1 发布于 2026 年 2 月 5 日，v0.16.0 是 2 月 26 日，v0.17.0 是 3 月 7 日，最新的 v0.23.0 发布于 2026 年 6 月 13 日，v0.22.1 是 6 月 5 日，v0.21.0 是 5 月 15 日，v0.20.0 是 4 月 27 日。也就是说 v0.15.1 距离当前最新版大约 4 个月、8 个小版本。自 v0.12.0 起 vLLM 改为常规版本每 2 周发布一次的节奏，所以这个差距并不算大——相比之下 v0.9 系列已经是一年多以前的代码了，v0.15.1 在 API、配置类、CLI 参数等方面会更贴近你现在看到的最新文档。

**架构层面**：到 v0.15.1 时，V0 早已被清除（移除工作从 v0.10 就开始了），所以代码库是纯 V1 架构——EngineCore/AsyncLLM/Scheduler/KVCacheManager/Executor/GPUModelRunner 这套骨架完整且没有 V0 遗留分支干扰。对于你关心的"标准 decoder-only LLM"路径（也就是 nano-vllm 对标的场景），worker 层在这个版本基本还是经典的单文件 `gpu_model_runner.py`，可以一口气读完，跟 nano-vllm 的 `Scheduler`/`BlockManager`/`ModelRunner`/`LLMEngine` 映射关系依然成立。

**关于 MRV2**：值得注意的是，v0.15.0 的 release notes 里已经出现"Model Runner V2: VLM support，architecture improvements"这一条——说明 MRV2 在这个版本就已经开始进入代码库，但当时的范围主要是给 VLM（多模态模型）用的，而让 Llama/Mistral 这类 dense 文本模型也启用 MRV2 是更晚（4 月份左右）才陆续合并的 PR。所以在 v0.15.1 里，如果你只关注纯文本 decoder-only 模型的推理路径，默认走的依然是传统 GPUModelRunner，不会被 MRV2 的 40 文件拆分和 async-no-sync 设计打断主线学习；但代码库里已经能"预览"到 MRV2 刚起步时的形态（主要在多模态相关代码路径里），等你把核心架构学透之后，可以顺着这条线索过渡到后面版本的 MRV2 设计文档。

**调度器**：async scheduler 是在 v0.19.0（2026 年 4 月）才被设为默认，所以 v0.15.1 的默认调度路径仍偏"传统/同步"，概念上跟 nano-vllm 的单步调度循环更接近，理解成本更低；异步调度作为可选项已经存在，等你打好基础后可以单独打开研究。

**该版本的其他特性概览**（多数是外围功能，不影响你研究引擎主线，但可以作为了解）：新增 score/render 等 API 端点，FIPS 140-3 安全选项，自动根据 dp_size 设置 api_server_count，移除了一批废弃的量化方式（DeepSpeedFp8、RTN）和废弃指标，Blackwell 上 FlashInfer MLA 成为默认 MLA backend，FP4 量化 kernel 有显著加速，并新增了 Kimi-K2.5、Molmo2 等模型架构支持。这些大多是"生产特性"层面的更新，跟架构骨架关系不大。

**结论**：v0.15.1 适合作为学习版本。优点是离最新足够近（API/配置基本是当前主流形态），但又赶在 MRV2 对常规 LLM 大规模生效之前，核心 worker 层仍是可读的单体文件，调度器默认仍是同步、心智负担小。建议的阅读顺序不变：先看 `v1/engine`（EngineCore/AsyncLLM）→ `v1/core/sched`（Scheduler）→ `v1/core/kv_cache_manager`（block pool/prefix cache）→ `v1/worker/gpu_model_runner.py`（针对纯文本模型路径），逐一对照 nano-vllm 里你已经熟悉的对应模块；如果路上碰到带 `model_runner_v2` / MRV2 相关 flag 或文件，先跳过即可，等主线打通后再去读 [Model Runner V2 设计文档](https://docs.vllm.ai/en/latest/design/model_runner_v2/) 作为增量补充。

## 