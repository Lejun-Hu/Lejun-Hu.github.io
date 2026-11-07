---
title: "【未完成】DeepEP 源码学习笔记"
permalink: /notes/deepep-source-study/
date: 2026-12-20
category: "集合通信"
tags: ["DeepEP", "MoE", "Expert Parallel", "NCCL", "GIN", "对称内存", "NVSHMEM", "源码分析"]
description: "DeepEP 源码学习笔记（初稿）：从 MoE 的通信特征出发，沿 Python 入口、C++ runtime、JIT 编译到 GPU 内核逐级拆解调用链，说明 DeepEP 如何复用 NCCL 的通信器、对称内存与 GIN 后端完成不规则 all-to-all。"
published: true
---

本文是《[NCCL 源码学习笔记](/notes/nccl-source-study/)》的姊妹篇。NCCL 笔记讲的是通用集合通信库如何建网、如何调度、以及 GIN 与对称内存这些底层能力的机制；本文讲的是**一个上层库如何把这些能力当作接口使用**，去解决 NCCL 的集合操作本身不擅长的问题。

DeepEP 是 DeepSeek 开源的专家并行（Expert Parallelism, EP）通信库，全名 DeepEveryParallel（`README.md:3`）。它面向 MoE 模型训练与推理中的 token 分发与回收，提供高吞吐、低延迟的 all-to-all GPU 内核。本笔记基于本地的 **DeepEP v2.1.0** 源码（`deep_ep/__init__.py:97` 的 `__version__ = '2.1.0'`），并对照 NCCL master 分支核实所有 NCCL 侧 API。文中每个函数、字段都标注源码位置，方便打开源码对照。

按初学者的阅读顺序，全文分三章：先讲清楚「MoE 的通信为什么需要专门的库」，再沿真实调用链逐层读代码，最后归纳设计要点与源码阅读路径。

| 章节 | 关注的问题 | 读完应掌握 |
|------|-----------|-----------|
| 一、问题背景 | MoE 的通信与常规集合通信差在哪里、V1/V2 有什么区别 | 为什么要另写一个库、两代架构的差异 |
| 二、代码骨架 | 一次 `dispatch` 从 Python 调用到 GPU 内核经过了哪些层 | 完整调用链、每层的职责、NCCL 能力在何处被消费 |
| 三、设计要点 | 这些实现解决了什么工程问题 | 定制内核、通信计算重叠、低延迟设计与阅读路径 |

前置知识：第二章大量引用 NCCL 的 `ncclDevComm`、`ncclWindow_t`、GIN 与 Team 语义，建议先读 NCCL 笔记的第四章（GIN 详解），特别是设备侧 context、窗口注册与 QP 参数传递部分。

---

## 一、问题背景：MoE 为什么需要专门的通信库

本章不进入 DeepEP 的代码，先把问题本身说清楚：MoE 的通信模式与常规集合通信有什么本质差别，由此产生了哪几类场景，以及为什么读代码前必须先分清 V1 与 V2。这三节分别对应「是什么」「有哪些形态」「读哪一套代码」。

### 1.1 版本背景：V1 与 V2

DeepEP 目前有两代实现，**架构差异很大，尤其是对 NCCL 的依赖**。这一点必须先交代，否则把两代代码混着读会得出完全错误的结论。当前源码是 v2.1.0（`deep_ep/__init__.py:97`），两代接口并存：`deep_ep/buffers/legacy.py` 的 `Buffer` 是 V1，`deep_ep/buffers/elastic.py` 的 `ElasticBuffer` 是 V2。

官方 README 对 V2 的定位（`README.md:9`）：V2 是对专家并行的完整重构，用少得多的 SM 资源达到极致性能，支持更大的 scale-up 与 scale-out 规模，并把后端从 NVSHMEM 切换为更轻量的 NCCL Gin 后端。README 的 New features 部分还给出了可核对的具体变化（`README.md:13-25`）：全量 JIT 编译、复用已有 NCCL communicator、高吞吐与低延迟 API 统一为单个 `ElasticBuffer` 接口、EP 规模上限到 2048、SM 占用从 24 降到 4–6。

下表按「依赖什么」而不是「性能数字」对比两代，因为前者决定读代码时该打开哪些文件：

| 维度 | V1（`Buffer`） | V2（`ElasticBuffer`） |
|------|----------------|----------------------|
| 跨节点 RDMA 后端 | NVSHMEM + IBGDA | NCCL Gin |
| 机内通信 | CUDA IPC 共享内存（`csrc/legacy/buffer.hpp:49,132-136`） | NCCL 对称内存窗口 + LSA 指针 |
| NCCL communicator | **不使用** | 复用 PyTorch 的 `comm`；旧版 PyTorch 或需要 CPU 段时自建（`deep_ep/utils/comm.py:42-75`） |
| 内存分配 | `nvshmem_align` + CUDA IPC | `ncclMemAlloc` 或 GPU/CPU 混合的弹性内存 |
| API 形态 | 高吞吐 `dispatch`/`combine` 与低延迟 `low_latency_*` 分开 | 统一为 `ElasticBuffer` 的 `dispatch` / `combine` |
| 内核交付方式 | 安装期编译 | 运行期 JIT 编译（`README.md:13`） |

需要特别指出一处容易误判的点：**V1 代码里完全没有 `ncclComm_t`**。`legacy.py` 的初始化只做两件事——交换 CUDA IPC handles 与 NVSHMEM unique ID（`deep_ep/buffers/legacy.py:92-136`）；而 `csrc/kernels/backend/nccl.cu:20-41` 里的 `get_local_unique_id` / `create_nccl_comm` 虽然名字像自建通信器，实际是 V2 在 `comm.py` 里走兜底分支时才调用的。第二章会沿调用链把这一点坐实。

本笔记第二章以 **V2 主路径** 为主，因为它完整使用了 NCCL 的对称内存与 GIN；V1 的 NVSHMEM 与低延迟内核放在 2.8 和 3.3 单独说明。

### 1.2 MoE 的通信特征

理解 DeepEP 的设计，先看它要解决的通信模式。稠密模型训练里的代表性操作是 AllReduce：每个 GPU 算一部分梯度，再做一次全局归约。数据流是对称且规整的——每个 rank 发出与接收的字节数相同，通信对象覆盖全体 rank。NCCL 的 Ring/Tree 算法、channel 切分、带宽模型都是围绕这种规整模式设计的。

MoE 不一样。每一层的 token 由路由结果决定交给哪几个专家（如 top-2），而专家分布在不同 GPU 上，于是产生两个阶段：

- **Dispatch（分发）**：本地这批 token 要按各自的路由结果，发往**不同的目标 rank**。token A 去 rank 1 的专家 3，token B 去 rank 5 的专家 7，去向逐 token 不同。
- **Combine（回收）**：专家算完后，结果要送回 token 原来的 rank 做加权求和。它是 dispatch 的逆向路由。

这带来三个 NCCL 通用集合操作难以直接处理的性质：

1. **数据量不均衡**：各 rank 发出的 token 数、各专家收到的 token 数随每步路由变化，可能相差数十倍。
2. **数据相关且稀疏**：谁发给谁、发多少取决于本步 `topk_idx`，无法在初始化时固定。
3. **需要与计算深度重叠**：专家 GEMM 很贵，通信必须尽量与它并行，否则直接拉低吞吐。

NCCL 确实提供 `ncclAlltoAll`，但它是**等量规整**的：头文件注释写明「每个设备向所有其他设备发送 `count` 个值，发往 rank j 的数据取自 `sendbuff+j*count`」（`nccl.h.in:534-545`）。每个 (rank, peer) 对的量固定为 `count`、偏移固定为 `j*count`，无法表达 MoE 这种逐 token 变化、各对不等量的发送布局。这正是 DeepEP 要自己写内核的原因。

### 1.3 DeepEP 面向的三类场景

明确了通信特征，再看 DeepEP 把问题切成几类。V1 的 `Buffer` 类在 docstring 里直接列出支持的三类 all-to-all（`deep_ep/buffers/legacy.py:14-19`），这三类也是理解后续内核分流的框架：

| 场景 | 数据路径 | 优化目标 |
|------|---------|---------|
| 高吞吐节点内（intranode） | NVLink | 机内带宽跑满 |
| 高吞吐节点间（internode） | RDMA 走跨机 + NVLink 走机内 | 两级带宽叠加、吞吐优先 |
| 低延迟（low-latency） | 纯 RDMA | 推理 decode 阶段的延迟优先 |

前两类面向训练与 prefill，追求带宽；第三类面向推理 decode，追求延迟。`docs/legacy.md:9-11` 补充说明了这两条路线：高吞吐内核针对 NVLink 域到 RDMA 域的非对称带宽转发做了优化，低延迟内核用纯 RDMA 并采用 hook 式的通信-计算重叠，不占用 SM。

V2 把三者收进统一的 `ElasticBuffer` 接口，但底层仍按拓扑分流——第二章 2.4、2.5 会看到 `is_nvlink_accessible` 决定机内直写还是走 GIN put，这是同一套分流逻辑在新架构里的延续。

---

## 二、从 Python 调用到 GPU 内核：完整代码骨架

这是全文核心章节。上一章说明了「为什么需要专门的库」，本章回答「一次调用具体怎么跑起来」。DeepEP 对 NCCL 的使用方式与直觉不同：它**不调用 NCCL 的集合通信 API**，而是复用 NCCL 提供的四项底层能力——communicator 句柄、`ncclDevComm` 设备上下文、对称内存窗口与 LSA 指针、GIN 设备侧 API。

先给出全链路总览。下表中「时机」一列区分初始化期与每次调用期，这是避免把不同阶段的调用误读成连续流程的关键：

| 步骤 | 时机 | 入口 | 产物 |
|------|------|------|------|
| 1. 取得 NCCL communicator | `ElasticBuffer` 构造 | `get_nccl_comm_handle`（`deep_ep/utils/comm.py:42`） | `NCCLCommHandle` |
| 2. 建设备上下文、对称内存、窗口 | `ElasticBuffer` 构造 | `NCCLSymmetricMemoryContext`（`csrc/kernels/backend/nccl.cu:62`） | `dev_comm` / `window` / `nvl_window_ptrs` |
| 3. Python 层参数整理 | 每次 `dispatch` | `ElasticBuffer.dispatch`（`deep_ep/buffers/elastic.py:855`） | `runtime.dispatch` 的入参 |
| 4. C++ 层启动准备 | 每次 `dispatch` | `ElasticBuffer::dispatch` → `launch_dispatch`（`csrc/elastic/buffer.hpp:702`、`csrc/kernels/elastic/dispatch.hpp:141`） | `DispatchRuntime::Args` |
| 5. JIT 生成、编译、启动 | 每次 `dispatch` | `generate` → `build` → `launch`（`dispatch.hpp:227-229`） | 特化后的 CUDA 内核 |
| 6. GPU 内核执行 | 每次 `dispatch` | `dispatch_impl`（`deep_ep/include/deep_ep/impls/dispatch.cuh:32`） | 接收缓冲区 + `EPHandle` |

调用链按层级展开如下，箭头表示调用关系而非时间上的连续执行：

```
ElasticBuffer.dispatch()                     deep_ep/buffers/elastic.py:855
  → self.runtime.dispatch(...)              :976  （pybind 绑定见 csrc/elastic/buffer.hpp:1364）
    → ElasticBuffer::dispatch()             csrc/elastic/buffer.hpp:702
      → launch_dispatch(...)                :980 → csrc/kernels/elastic/dispatch.hpp:141
        → DispatchRuntime::generate(args)   :227  生成模板实例化源码
        → jit::compiler->build("dispatch")  :228  NVRTC 编译
        → DispatchRuntime::launch(...)      :229  启动内核
          → dispatch_impl<...>()            deep_ep/include/deep_ep/impls/dispatch.cuh:32
            → handle::NCCLGin::put / put_value / get_sym_ptr
              → ncclGin::put（NCCL 设备侧 GIN API）
```

### 2.1 初始化期：取得 NCCL communicator

先解决「DeepEP 手里为什么要有 `ncclComm_t`」。它是后续 `ncclDevCommCreate`、`ncclCommWindowRegister` 等底层 API 的必需参数，而不是用来发起 `ncclAllReduce` 之类集合操作的。V2 的获取逻辑集中在 `get_nccl_comm_handle`（`deep_ep/utils/comm.py:42-75`），有三条分支：

```python
# deep_ep/utils/comm.py:55-75（节选）
if not force_new_comm and group in _storage:
    return _storage[group]                                  # 分支 1：命中缓存

backend = group._get_backend(torch.device('cuda'))
if not force_new_comm and hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
    _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)   # 分支 2：复用 PyTorch 的 comm
    return _storage[group]

# 分支 3：旧版 PyTorch 或强制新建，自己建一个
nccl_unique_ids = [None, ] * group.size()
dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)
_storage[key] = NCCLCommHandle(
    _C.create_nccl_comm(nccl_unique_ids[0], group.size(), group.rank()), True)
```

分支 2 是主路径：`backend._comm_ptr()` 拿到 PyTorch 已经建好的 NCCL communicator 裸指针，`managed=False` 表示 DeepEP 不负责销毁它。分支 3 才调用 `csrc/kernels/backend/nccl.cu:20-41` 的 `get_local_unique_id` + `create_nccl_comm`（内部是 `ncclGetUniqueId` + `ncclCommInitRank`），`managed=True` 表示由 `NCCLCommHandle.__del__` 负责 `ncclCommAbort`。

触发分支 3 有两种情况：PyTorch 没暴露 `_comm_ptr` 或 `EP_REUSE_NCCL_COMM=0`；以及 `ElasticBuffer.__init__` 传入 `force_new_comm=num_cpu_bytes > 0`（`deep_ep/buffers/elastic.py:301`），即启用了 CPU 弹性内存段时必须新建通信器。

### 2.2 初始化期：设备上下文、对称内存与窗口注册

拿到 communicator 后，`ElasticBuffer.__init__` 构造 C++ runtime（`deep_ep/buffers/elastic.py:344-352`），后者在 `csrc/elastic/buffer.hpp:108-114` 创建 `NCCLSymmetricMemoryContext`。这个构造函数（`csrc/kernels/backend/nccl.cu:62-148`）是 DeepEP 与 NCCL 衔接最密集的一段，按真实执行顺序分四步。

**第 1 步：查询能力并声明 GIN 需求。**

```cpp
// csrc/kernels/backend/nccl.cu:82-108（节选）
ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
NCCL_CHECK(ncclCommQueryProperties(comm, &props));
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
    EP_HOST_ASSERT((allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE);
    reqs.ginContextCount    = num_allocated_qps;
    reqs.ginExclusiveContexts = true;
    reqs.ginQueueDepth      = kGinQPDepth;
    reqs.ginTrafficClass    = sl_idx;
    reqs.ginSignalCount     = num_ranks + 2 * 2;   // 自定义 RDMA barrier 需要额外信号
    reqs.ginConnectionType  = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL
                                                : NCCL_GIN_CONNECTION_FULL;
}
NCCL_CHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm.ptr)));  // :108
```

`ncclDevCommCreate` 声明于 `src/include/nccl_device/core.h:195`。它按 `reqs` 为上层库准备好设备侧上下文：QP、signal/counter 表、GIN 后端句柄等（机制见 NCCL 笔记第四章）。这里 `ginContextCount` 取 `num_allocated_qps`，其默认值由 `elastic.py:326-335` 按是否 hybrid 模式算出（65 / 129 / 17）；`ginConnectionType` 决定连接覆盖全体 rank 还是仅同 rail，对应 `gin_host.cc:141-155` 的两条分支。

**第 2 步：划分物理域与逻辑域。** `nccl.cu:110-125` 用 `ncclTeamLsa(comm)` 得到 NVLink 域大小，再按 `allow_hybrid_mode` 决定 `num_scaleout_ranks` / `num_scaleup_ranks`。这一步决定后续内核是按机内路径还是混合路径分流。

**第 3 步：分配对称内存。** `nccl.cu:127-132` 调用 `symmetric::alloc`（`csrc/kernels/backend/symmetric.hpp:291-317`），按 CPU 段字节数选择三种实现之一（详见 2.6）。最基础的实现直接包一层 NCCL 的 `ncclMemAlloc`：

```cpp
// csrc/kernels/backend/symmetric.hpp:123-132（节选）
// Wraps ncclMemAlloc/ncclMemFree (pure GPU, current default)
class GPUSymmetricMemory final : public SymmetricMemory {
public:
    explicit GPUSymmetricMemory(const int64_t& num_bytes) {
        EP_HOST_ASSERT(num_bytes > 0 and num_bytes % kNumAlignmentBytes == 0);
        NCCL_CHECK(ncclMemAlloc(&ptr, num_bytes));
        ...
    }
};
```

`ncclMemAlloc` 声明于 `src/nccl.h.in:156`，要求 2 MB 对齐（`symmetric.hpp:16` 的 `kNumAlignmentBytes`），分配出的内存会在各 rank 上被映射到**相同的虚拟地址**，这是窗口注册与跨 rank 指针换算的前提。

**第 4 步：注册窗口并取 LSA 指针。**

```cpp
// csrc/kernels/backend/nccl.cu:134-147（节选）
// NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
// across all ranks, so no explicit barrier is needed after this call.
raw_window_ptr = this->symmetric_memory->ptr;
NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr,
                                  this->symmetric_memory->num_bytes, &window,
                                  NCCL_WIN_STRICT_ORDERING));
NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));
nvl_window_ptrs.resize(num_nvl_ranks);
for (int i = 0; i < num_nvl_ranks; ++i)
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
```

两个 NCCL API 的含义需要分清：

| API | 声明位置 | 作用 |
|-----|---------|------|
| `ncclCommWindowRegister` | `src/nccl.h.in:351` | 把对称内存注册为 window；内部跨所有 rank 做 bootstrapBarrier，是集合操作 |
| `ncclGetLsaDevicePointer` | `src/include/nccl_device/core.h:202` | 返回本进程 VA 空间中指向 LSA 域内第 i 个 GPU 该窗口的指针 |

LSA 是 **Load/Store Access** 的缩写，指同节点内对其他 GPU 内存的直接 load/store 访问能力（`docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md:94`）。窗口注册后，NCCL 会建立统一的对称虚拟地址视图，把各 peer 的物理段映射进本进程的 VA（`NCCL_Gin_and_Symmetric_Memory.md:281-290`），于是内核访问某个 peer 的窗口只需做地址算术，无需 CPU 参与。`nvl_window_ptrs[i]` 就是这张地址表在 host 侧的缓存。

到这四步结束，初始化期完成：设备侧拿到了 `dev_comm`、`window`、以及机内各 peer 的窗口基址。后面每次 `dispatch` 都复用这些资源。

### 2.3 调用期：从 `dispatch()` 到内核启动

现在进入每次调用的路径。Python 侧 `ElasticBuffer.dispatch`（`deep_ep/buffers/elastic.py:855-1014`）先做参数整理：自动推断 SM 与 QP 数（`:926-930`）、解包 FP8 的 `(x, sf)`、复用已有 `handle` 中的缓存布局（`:937-956`），然后把一长串参数交给 C++ runtime（`:976`）：

```python
# deep_ep/buffers/elastic.py:963-996（节选）
(recv_x, recv_sf, recv_topk_idx, recv_topk_weights, ...,
 psum_num_recv_tokens_per_scaleup_rank, psum_num_recv_tokens_per_expert,
 recv_src_metadata, dst_buffer_slot_idx, token_metadata_at_forward,
 channel_linked_list, event) = self.runtime.dispatch(
     x, sf, topk_idx, topk_weights, ..., num_sms, num_qps, ...)
```

pybind 把 `runtime.dispatch` 绑定到 C++ 的 `ElasticBuffer::dispatch`（`csrc/elastic/buffer.hpp:1364`）。后者（`buffer.hpp:702` 起）负责校验缓冲区大小、清理 host workspace（`:962-976`），再把 `nccl_context->dev_comm`、`window`、各类布局指针一起传给 `launch_dispatch`（`:980-1003`）。注意这里传的是 2.2 节初始化好的句柄，说明单次调用不再触碰 NCCL 的控制面。

`launch_dispatch`（`csrc/kernels/elastic/dispatch.hpp:141-230`）是 Host 侧启动器，做三件事：按共享内存容量算出 warp 分工（`:170-196`）、填充 `DispatchRuntime::Args`（`:199-226`）、执行 JIT 三步：

```cpp
// csrc/kernels/elastic/dispatch.hpp:227-229
const auto code = DispatchRuntime::generate(args);            // 生成源码
const auto runtime = jit::compiler->build("dispatch", code);  // NVRTC 编译
DispatchRuntime::launch(runtime, args, stream);               // 启动
```

V2 的内核是**运行期 JIT 编译**的（`README.md:13`）。`generate` 并非凭空生成算法，而是把运行期才知道的模板参数拼成一段仅用于实例化的源码（`dispatch.hpp:51-89`）：

{% raw %}
```cpp
// csrc/kernels/elastic/dispatch.hpp:80-88（节选）
return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
```
{% endraw %}

`func_name` 由 `is_scaleup_nvlink`、`num_sms`、warp 数、`num_experts`、`num_topk`、`num_qps` 等十几个参数拼成模板实参（`:55-77`），`header_name` 在单机与跨机两种情况下分别选 `dispatch` 与 `hybrid_dispatch`。因此算法主体写在 `.cuh` 里，运行期只决定「实例化哪一个模板组合」，再用 NVRTC 编译（`csrc/jit/compiler.hpp:22` 起）并缓存。这样既避免了安装期编译，又让内核里的循环边界、warp 数量都是编译期常量。

### 2.4 内核期：`dispatch_impl` 的三个阶段

`dispatch_impl`（`deep_ep/include/deep_ep/impls/dispatch.cuh:17-45`）的模板参数就是上一节 JIT 拼进去的那些，内核签名接收 `x`、`topk_idx`、workspace、缓冲区以及 `ncclDevComm_t` / `ncclWindow_t`。内核内部按 warp 角色分三段，下表给出各段的职责与产物：

| 阶段 | 参与的 warp | 职责 | 产物 |
|------|------------|------|------|
| 1. 计数与通知 | `warp_idx < kNumNotifyWarps` | 统计每个 token 的目标 rank/expert，全 SM 归约，把计数发给各 rank | `rank_count` / `expert_count` |
| 2. 前缀和 | SM 0 的 warp 0/1 | 由计数算接收布局的起始偏移 | `psum_num_recv_tokens_per_*` |
| 3. 数据搬运 | 其余 dispatch warps | TMA 装入共享内存、分配目标 slot、按路径发出数据 | 接收缓冲区中的 token |
| 收尾 | 全体 | `gpu_barrier` 确认数据到达 | 可安全读取的接收区 |

**第 1 阶段：运行时动态计数**（`dispatch.cuh:79-115`）。notify warps 逐 token 读 `topk_idx`，用共享内存原子加统计「每个 expert 收到几个 token」和「每个 rank 收到几个 token」，后者对同一 token 的多个专家做去重：

```cpp
// deep_ep/include/deep_ep/impls/dispatch.cuh:98-106（节选）
const auto dst_expert_idx = lane_idx < kNumTopk ?
    static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
if (dst_expert_idx >= 0)
    atomicAdd_block(expert_count + dst_expert_idx, 1);
const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
    atomicAdd_block(rank_count + dst_rank_idx, 1);
```

计数随后通过 `gin.put_value<team_t>` 写到各对端的 workspace（`dispatch.cuh:152-177`），也就是**用 GIN 完成控制信息的交换**。SM 0 还负责等待对端计数、对齐 expert 数量、写 host workspace（`:183-230`）。

**第 2 阶段：前缀和定位**（`dispatch.cuh:232-257`）。用 warp 内的 inclusive sum 把计数前缀和写进 `psum_num_recv_tokens_per_scaleup_rank` 与 `psum_num_recv_tokens_per_expert`。这两个数组是后面 kernel 和用户侧 `EPHandle` 判断「某 rank 的 token 放在接收缓冲区哪一段」的依据，也是不规则布局得以精确定位的关键。

**第 3 阶段：逐 token 搬运与路径分流**（`dispatch.cuh:259-394`）。每个 dispatch warp 先按 `get_qp_mode` 取自己的 QP 与共享模式，构造 `NCCLGin`（`:67-71`）；随后循环处理 token：TMA 把 hidden 与 SF 搬进共享内存（`:284-312`）、读 top-k 索引、用原子计数分配目标 slot（`:336-351`），然后按目标 rank 是否在 NVLink 域内分两条路：

```cpp
// deep_ep/include/deep_ep/impls/dispatch.cuh:371-393（节选）
// Issue TMA NVLink stores
const auto dst_ptr = stored_dst_slot_idx >= 0 ?
    gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                            stored_dst_rank_idx) : nullptr;
if (dst_ptr != nullptr)
    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
ptx::tma_store_commit();
...
// Issue RDMA put
if constexpr (not kIsScaleupNVLink) {
    ptx::tma_store_wait<1>();
    if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
        gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
    }
}
```

`get_sym_ptr` 返回非空表示目标在同 NVLink 域内，直接用 TMA store 写对端窗口；返回空则把数据先落到本地 send buffer，再由 `gin.put` 经网卡 RDMA 写过去。最后 `comm::gpu_barrier`（`dispatch.cuh:397-400`）确认所有 rank 的数据到达。这一分流的判定依据来自 `NCCLGin` 封装，2.5 节展开。

### 2.5 设备侧 GIN 封装：Team 判定与两条数据路径

上一节出现了 `NCCLGin::put` / `get_sym_ptr` / `put_value`，这里展开这层封装，它是 DeepEP 把 NCCL 设备侧 GIN API 变成「按 rank 自动选路」的关键。代码在 `deep_ep/include/deep_ep/common/handle.cuh`。

构造时（`handle.cuh:11-35`）保存 `ncclDevComm_t` 与 `ncclWindow_t`，用 `qp_idx` 构造 NCCL 提供的 `ncclGin` 对象，并一次性取好三个 Team 与本 rank 的 LSA 窗口基址：

```cpp
// deep_ep/include/deep_ep/common/handle.cuh:26-35（节选）
NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
        const int& qp_idx = 0,
        const ncclGinResourceSharingMode& resource_sharing_mode = NCCL_GIN_RESOURCE_SHARING_GPU):
    gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),
    team_world(ncclTeamWorld(nccl_dev_comm)),
    team_lsa(ncclTeamLsa(nccl_dev_comm)),
    team_rail(ncclTeamRail(nccl_dev_comm)) {
    lsa_base_ptr = reinterpret_cast<uint64_t>(ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));
}
```

三个 Team 对应三种 rank 集合（`src/include/nccl_device/core.h:55-57` 定义了对应 tag）：World 是全体 rank，Lsa 是同 NVLink 域内的 rank，Rail 是跨节点中处于相同 rail 位置的 rank。`is_nvlink_accessible`（`handle.cuh:37-54`）依据 Team 类型判断目标 rank 是否能用对称指针直达；`get_sym_ptr`（`:64-92`）则在可达时用 `ncclGetLsaPointer` 把本地窗口偏移换算成该 peer 的窗口地址，不可达时返回 `nullptr`。2.4 节里 `dst_ptr != nullptr` 的分支正是以此为依据。

其余三个封装方法把不同形态的远程操作映射到 NCCL 设备侧 API：

| 方法 | 位置 | 映射到的 NCCL API |
|------|------|------------------|
| `put` | `handle.cuh:175-198` | `gin.put`，参数用窗口 + 相对 `lsa_base_ptr` 的偏移 |
| `get` | `handle.cuh:127-144` | `gin.get` |
| `put_value` | `handle.cuh:200-220` | 可达时 `st_relaxed_sys` 直写，否则 `gin.putValue` |
| `signal` / `flush` / `flush_async` | `handle.cuh:122-173` | `gin.wait` / `gin.flush` / `gin.flushAsync` / `gin.signal` |

`red_add_rel`（`handle.cuh:96-120`）体现了「对称指针优先、RDMA 兜底」这一原则：目标可达时用 `red_add_rel_sys` 做原子加，不可达时改用 `gin.signal` 配 `ncclGin_VASignalAdd` 发一个带值的信号。控制信息与数据信息都遵循同一套分流。

QP 的分配由 `comm::get_qp_mode`（`deep_ep/include/deep_ep/common/comm.cuh:56-86`）决定，它按 SM 数、QP 数、warp 角色返回 `(qp_idx, resource_sharing_mode)`。需要澄清一处常见误解：warp 与 QP **不是一一对应**的；当 SM 数多于可用 QP 时（`comm.cuh:81-85`），所有 SM 共享全部 QP 并使用 GPU 级共享模式，QP 下标由全局 channel 索引取模得到。只有 SM 数少于 QP 数时，单个 SM 才能独占一条 QP。

内核的跨 rank 同步也不再依赖 CPU：`comm::gpu_barrier`（`comm.cuh:214-265`）按 scaleup/scaleout 域分别做 barrier，其中 GIN 路径先用 `gin.flush` 刷完所有 QP，再用 `gin.signal` 递增各 rank 的信号并等待（`comm.cuh:143-165`）。这解释了 2.2 节里 `ginSignalCount = num_ranks + 4` 为什么要多留出信号。

### 2.6 combine 与 `EPHandle`：布局信息如何跨两次调用传递

dispatch 把 token 送到目标 rank，combine 要把专家算完的结果按原路由送回并归约。两者之间靠 `EPHandle` 传递布局信息，这是 V2 接口设计里值得单独看的一点。

`EPHandle`（`deep_ep/buffers/elastic.py:25-98`）由 `dispatch` 返回（`elastic.py:1001-1014`）。下表列出它保存的主要字段及其在 combine 中的用途：

| 字段 | 含义 | combine 中的用途 |
|------|------|-----------------|
| `topk_idx` | dispatch 时的 top-k 专家索引副本 | 确定每个 token 要送回哪些源 rank |
| `recv_src_metadata` | 源 token 索引与缓冲区 slot 索引 | 定位数据应写回的位置 |
| `psum_num_recv_tokens_per_scaleup_rank` | 各 scaleup rank 接收量的前缀和 | 确定接收区有效长度 |
| `dst_buffer_slot_idx` | dispatch 分配的目标 slot | 复用布局，跳过重新计数 |
| `num_sms` | dispatch 使用的 SM 数 | combine 默认复用（`elastic.py:1086`） |
| `token_metadata_at_forward` / `channel_linked_list` | hybrid 模式的转发元数据 | 跨机路径的通道组织 |

`ElasticBuffer.combine`（`elastic.py:1046-1105`）从 handle 取出这些字段传给 C++ runtime，并把 `num_sms` 默认设为 `handle.num_sms`。`EPHandle` 还能作为**缓存 handle** 传回下一次 `dispatch`（`elastic.py:28-29`、`:937-956`），此时 `topk_idx` 等参数从 handle 读取、`do_cpu_sync` 强制为 False，内核走 `reuse_slot_indices` 分支跳过重复的布局计算（`dispatch.cuh:338-342`）。这就是 3.4 节所说的「通信 handle 复用」在 V2 里的具体实现。

combine 内核 `combine_impl`（`deep_ep/include/deep_ep/impls/combine.cuh:15-35`）与 dispatch 结构对称：同样的模板特化、同样的 `NCCLGin` 与 `get_qp_mode`（`:71-74`）、前后各一次 `gpu_barrier`（`:76-80`、`:239-242`）。数据搬运同样按 NVLink 可达性分流——可达时 `tma_store_1d` 直写对端，不可达时先落本地 send buffer 再 `gin.put`（`:192-207`、`:232-235`）。差别在于 combine 读的是专家输出、写回原 rank 并做归约，因此它依赖 `EPHandle` 里的源索引与布局信息来反向寻址。

### 2.7 对称内存的三种实现

2.2 节第 3 步提到 `symmetric::alloc` 会按条件选实现，这里展开。`csrc/kernels/backend/symmetric.hpp:291-317` 的选择逻辑是：无 CPU 段用 `GPUSymmetricMemory`；有 CPU 段且 `allow_hybrid_mode` 用 `HybridElasticSymmetricMemory`，否则用 `ElasticSymmetricMemory`。三者的内存构成与适用场景如下：

| 实现类 | 定义位置 | 内存构成 | 适用场景 |
|--------|---------|---------|---------|
| `GPUSymmetricMemory` | `symmetric.hpp:124-140` | 纯 GPU 显存（`ncclMemAlloc`） | 默认路径，数据全在显存 |
| `ElasticSymmetricMemory` | `symmetric.hpp:145-186` | GPU 段（前）+ NUMA 本地 CPU 段（后） | 用 CPU 内存扩充容量 |
| `HybridElasticSymmetricMemory` | `symmetric.hpp:191-289` | GPU 段 + 各 scaleup rank 各一段 CPU 内存 | hybrid 模式，各 rank 的 CPU 段都映射进 VA |

三者共同点是构造出一段**连续且 2 MB 对齐**的虚拟地址空间（`symmetric.hpp:16`），所有 rank 映射到相同地址，因而都能传给 `ncclCommWindowRegister`。后两者通过 CUDA Driver API 的 `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemSetAccess` 逐段映射（`symmetric.hpp:162-175`），并为 GPU 段设置 `gpuDirectRDMACapable`（`:52-61`）。

当选中 `ElasticSymmetricMemory` 时，`alloc` 还会设置环境变量启用 NCCL 的弹性缓冲注册（`symmetric.hpp:312-315`）。这是 DeepEP 影响 NCCL 行为的一个例子：它不只调用 NCCL 的分配器，还会通过环境与窗口注册参数改变 NCCL 对这段内存的处理方式。`elastic.py:284-298` 在 CPU 段很大时设置 `NCCL_WIN_STRIDE` 也是同一类做法。

### 2.8 两代后端：从 NVSHMEM 到 NCCL Gin

前面各节都沿 V2 路径，这里补齐 V1 的跨节点实现，以便理解「V2 切换后端」具体改了什么。

V1 的初始化分两条线，都不经过 NCCL：机内用 CUDA IPC 共享内存，跨节点用 NVSHMEM。Python 侧 `Buffer.__init__`（`deep_ep/buffers/legacy.py:92-136`）先构造 C++ runtime，再交换设备 ID、IPC handles 与 NVSHMEM unique ID；其中启用 **IBGDA**（InfiniBand GPU Direct Async，GPU 直接访问网卡）的环境变量设置在 `legacy.py:108-110`：

```python
# deep_ep/buffers/legacy.py:108-110（节选）
os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

C++ 侧 `csrc/legacy/buffer.hpp:232-280` 完成 IPC handle 的导入映射，并在需要时调 `nvshmem::init` 与 `nvshmem::alloc`（`:262-268`）。`nvshmem::init` 的实现在 `csrc/kernels/backend/nvshmem.cu:49-76`：用 unique ID 初始化 NVSHMEM，再按 `team_split_stride` 切出跨节点 RDMA 子 team。机内部分则由 `shared_memory::MemHandle` 走 CUDA IPC（`csrc/legacy/buffer.hpp:49`、`:132-136`）。

V2 去掉了这条依赖：`deep_ep/buffers/elastic.py` 中不再有任何 `NVSHMEM` 初始化，跨节点数据统一经 `gin.put` / `gin.get` 走 NCCL Gin。下表按通信域对比两代，注意机内路径也一并改变了：

| 通信域 | V1 | V2 |
|--------|----|----|
| 机内（NVLink 域） | CUDA IPC 共享内存 | NCCL 对称内存窗口 + LSA 指针 / TMA store |
| 跨机（RDMA） | NVSHMEM + IBGDA | NCCL Gin（`gin.put` / `gin.get`） |
| GPU 侧发起网络 | NVSHMEM 的 put/get | 统一走 NCCL GIN 设备 API |
| 进程间控制面 | IPC handles + NVSHMEM unique ID 交换 | 复用或自建 NCCL communicator |

README 对这次切换的表述是 "V2 has switched from the NVSHMEM backend to the more lightweight NCCL Gin backend"（`README.md:9`）。同时要注意 V2 的能力边界：`README.md:30` 明确写了 "0 SM RDMA low-latency EP is no longer supported"，即 V1 那种零 SM 占用的 RDMA 低延迟 EP 在 V2 不再提供，不能把「接口统一」理解成「能力等价」。

至此第二章的链条闭合：初始化期由 Host 准备好 communicator、设备上下文、窗口与地址表；每次调用期由 Python 整理参数、C++ 启动器做 JIT 实例化、GPU 内核按路由动态计数并分流搬运，全程不调用 NCCL 的集合通信 API。

---

## 三、设计要点与源码阅读路径

第二章沿调用链读完了代码，本章从实现细节里抽身，归纳几个决定 DeepEP 性能特征的设计选择，并给出后续自己读源码的建议顺序。前三节对应「内核怎么适配不规则通信」「通信与计算怎么重叠」「低延迟场景做了什么取舍」，最后两节是工程细节与阅读路径。

### 3.1 面向不规则 all-to-all 的定制内核

最根本的一点是：DeepEP 没有把 MoE 的 dispatch/combine 硬套进 `ncclAlltoAll` 的等量模型，而是自己写了针对动态布局的内核。第二章 2.4 已经看到具体做法，这里按设计意图归纳三条：

1. **布局在内核里现算**：notify warps 用共享内存原子加统计逐 token 的 rank/expert 计数，全 SM 归约后由 warp 前缀和算出各段起始偏移（`dispatch.cuh:92-115`、`:232-257`）。V2 的产物是 `psum_num_recv_tokens_per_scaleup_rank` 与 `psum_num_recv_tokens_per_expert`；V1 里对应的概念叫 `rank_prefix_matrix` / `channel_prefix_matrix`（`legacy.py:391`），两者名字不同但目的一致。
2. **warp 即 channel**：每个 dispatch warp 独立处理一批 token，配一个 `NCCLGin` 实例，并按需分配 QP 与共享模式（`dispatch.cuh:67-71`、`comm.cuh:56-86`）。这是细粒度并行的来源，但 QP 与 warp 不是一一对应，多 SM 会共享 QP 池。
3. **数据与布局分离**：token 的 hidden/SF 用 TMA 搬入共享内存，目标 slot 由原子计数分配，源信息记入 `recv_src_metadata` 供 combine 反向寻址。布局信息因此能在 `EPHandle` 中缓存复用。

### 3.2 内核级通信带来的重叠

常规集合通信里，通信通常作为独立的内核或流任务出现，计算与通信的边界比较粗。DeepEP 借 GIN 让 GPU 内核里的线程在搬运过程中直接发起 RDMA（`dispatch.cuh:389-390`、`handle.cuh:183-196`），边界细化到 warp 与 token 级。

收益有两层：一是省去「内核结束 → Host 代理线程提交 → 网卡执行」的 Host 参与，二是 warp 可以一边为下一个 token 做 TMA 搬运，一边让已发出的 `gin.put` 在后台完成。第二章 2.5 提到的 `gpu_barrier`（`comm.cuh:214-265`）则负责在这条细粒度流水线末端给出全体 rank 的到达保证——它用 GIN signal 实现，同样不需要 CPU 同步点。

### 3.3 面向推理的低延迟设计

V1 提供了独立的 `low_latency_dispatch` / `low_latency_combine` 接口（`deep_ep/buffers/legacy.py:553`、`:624`），专为推理 decode 的小批量、延迟敏感场景设计。几个可核对的取舍：

- **纯 RDMA 路径**：所有 rank 经 RDMA 互访，不经过 NVLink 转发，把跨机延迟压到最低（`docs/legacy.md:28-30`）。
- **固定缓冲 + 不做 CPU 同步**：接收缓冲按 `num_max_dispatch_tokens_per_rank` 预先固定分配，且 docstring 明确说明「不同步 CPU 侧接收计数，否则与 CUDA Graph 不兼容」（`legacy.py:591-600`）。
- **接收 hook**：`return_recv_hook=True` 时内核只发起 RDMA 请求、不等待数据到达，由调用方在真正需要数据时执行 hook（`legacy.py:584-585`）。`docs/legacy.md:11` 指出这种方式不占用 SM，可把网络流量与计算重叠。
- **FP8 与统计**：`use_fp8` 在通信过程中做量化以减少传输量（`legacy.py:557`）；`dispatch_wait_recv_cost_stats` / `combine_wait_recv_cost_stats` 统计等待接收的耗时，用于定位慢节点（`legacy.py:556,577-578`、`:627,653-654`）。

V2 把高吞吐与低延迟 API 统一进 `ElasticBuffer`（`README.md:18`），通过 `async_with_compute_stream`、`do_cpu_sync` 等参数表达重叠诉求（`elastic.py:866,870`），没有独立的 `low_latency_*` 方法。但要注意 `README.md:30` 的说明：**零 SM 占用的 RDMA 低延迟 EP 在 V2 不再支持**，因此 V1 的这一能力并非被等价迁移。

### 3.4 工程细节

以下几项是 V1/V2 共有的工程性设计，只是接口名不同（V1 用元组 handle，V2 用 `EPHandle`）：

- **布局缓存**：`EPHandle` 可作为 cached handle 传回 `dispatch`，复用前缀和与 slot 索引，跳过重算（`elastic.py:28-29,937-956`；内核侧 `dispatch.cuh:338-342`）。
- **专家对齐**：`expert_alignment` 把每个专家收到的 token 数对齐到指定倍数，便于后续 GEMM 访存对齐（`elastic.py:896`；内核侧 `dispatch.cuh:215`）。
- **rank 动态屏蔽**：`low_latency_update_mask_buffer`（`legacy.py:672`）支持在通信中屏蔽指定 rank，用于在线服务的故障隔离。
- **超时与可观测性**：内核用 `kNumTimeoutCycles` 与 `comm::timeout_while` 做等待超时检查并打印诊断（`dispatch.cuh:123-146`），避免分布式场景下静默挂死。

### 3.5 建议的源码阅读路径

读完本笔记后若想自己深入，建议按第二章的调用链顺序由浅入深，先走完 V2 主路径再回头看 V1 作对比：

| 顺序 | 文件 | 关注点 |
|------|------|--------|
| 1 | `deep_ep/buffers/elastic.py` | `ElasticBuffer.__init__` / `dispatch` / `combine` 的参数整理与 `EPHandle` |
| 2 | `deep_ep/utils/comm.py`、`csrc/kernels/backend/nccl.cu` | communicator 三条分支；`NCCLSymmetricMemoryContext` 四步 |
| 3 | `csrc/kernels/backend/symmetric.hpp` | `symmetric::alloc` 与三种对称内存实现 |
| 4 | `csrc/elastic/buffer.hpp`、`csrc/kernels/elastic/dispatch.hpp` | Host 侧启动器、`DispatchRuntime::Args`、JIT 三步 |
| 5 | `deep_ep/include/deep_ep/common/handle.cuh`、`comm.cuh` | `NCCLGin` 封装、Team 判定、`get_qp_mode`、`gpu_barrier` |
| 6 | `deep_ep/include/deep_ep/impls/dispatch.cuh` / `combine.cuh` | 内核三阶段、NVLink 直写与 GIN put 分流、combine 反向路由 |
| 7 | `deep_ep/buffers/legacy.py`、`csrc/legacy/buffer.hpp`、`csrc/kernels/backend/nvshmem.cu` | V1 的 IPC + NVSHMEM 初始化与低延迟接口 |

---

## 参考资料

- DeepEP 官方源码仓库：[github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)（本笔记基于 v2.1.0）
- DeepEP V1 legacy 文档：官方仓库 `docs/legacy.md`（NVSHMEM 时代的接口、性能数据与示例）
- NCCL 官方源码仓库：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)（device API 声明见 `src/nccl.h.in`、`src/include/nccl_device/core.h`）
- NCCL 对称内存与 GIN 设计文档：`docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md`（LSA 定义、窗口注册与统一 VA 视图）
- [《NCCL 源码学习笔记》](/notes/nccl-source-study/)： communicator 初始化、设备侧 context、GIN 后端与 QP 参数传递

---

*声明：本文是作者学习 DeepEP 源码过程中的阶段性笔记，内容基于 DeepEP v2.1.0 与 NCCL master 分支源码整理。由于这是一份仍在持续完善的初稿，文中可能存在表述不准确或理解不完整之处，欢迎通过邮件联系指正。*
