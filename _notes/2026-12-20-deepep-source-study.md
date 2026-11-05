---
title: "【未完成】DeepEP 源码学习笔记"
permalink: /notes/deepep-source-study/
date: 2026-12-20
category: "集合通信"
tags: ["DeepEP", "MoE", "Expert Parallel", "NCCL", "GIN", "对称内存", "NVSHMEM", "源码分析"]
description: "DeepEP 源码学习笔记（初稿）：从 MoE 的通信特征讲起，拆解 DeepEP 如何复用 NCCL 的通信器与 GIN 后端完成不规则 all-to-all，以及它对 NCCL 的真实调用关系。"
published: true
---

> 如果让你基于 NCCL 去构建一个 MoE（Mixture of Experts，混合专家）模型的通信层，你会怎么做？
>
> DeepSeek 开源的 **DeepEP** 就是对这个问题的回答。它是一个专门为 MoE 专家并行（Expert Parallelism）设计的通信库，因为解决了 MoE 训练/推理中最棘手的通信问题，而成为各方积极参考的对象（包括各家大模型团队和自研通信库）。
>
> 这份笔记假设你**还没读过 DeepEP 的代码**，从"MoE 到底需要什么样的通信"讲起，再一步步拆解 DeepEP 的架构，以及它和 NCCL 之间真实、精确的调用关系。
>
> **前置知识**：本笔记大量引用 NCCL 的内部机制（通信器、GIN 后端、对称内存、Proxy 线程），建议先读姊妹篇《[NCCL 源码学习笔记](/notes/nccl-source-study/)》，特别是其中关于 GIN 与初始化流程的部分。

---

## 一、为什么 MoE 需要专门的通信库

### 1.1 版本背景：先搞清楚 V1 与 V2

在读代码之前，有一件事必须先交代清楚：**DeepEP 目前有 V1 和 V2 两代，两者架构差异很大，尤其是对 NCCL 的调用方式**。如果混着读，很容易被绕晕。

本笔记基于的源码是 **DeepEP v2.1.0**（`deep_ep/__init__.py:97` 的 `__version__ = '2.1.0'`）。这个版本同时保留了两套 API：`deep_ep/buffers/legacy.py` 里的 `Buffer` 是 **V1 的旧接口**，`deep_ep/buffers/elastic.py` 里的 `ElasticBuffer` 是 **V2 的新接口**。

官方 README 里对 V2 的定位（`README.md:9`，原文翻译）：

> V2 是对专家并行（EP）的一次**完整重构**——相比 V1，用**少得多的 SM 资源**达到极致性能，同时支持**大得多的 scale-up 与 scale-out 规模**。V2 还把后端从 **NVSHMEM 切换到了更轻量的 NCCL Gin 后端**。

**V1 与 V2 的核心区别，尤其是对 NCCL 的调用：**

| 维度 | V1（legacy `Buffer`） | V2（`ElasticBuffer`） |
|------|----------------------|----------------------|
| **跨节点 RDMA 后端** | **NVSHMEM**（+ IBGDA，GPU 直接访问网卡） | **NCCL Gin**（不再用 NVSHMEM 做主后端） |
| **NCCL communicator** | 自己用 `ncclGetUniqueId` + `ncclCommInitRank` **新建一个** | **复用 PyTorch 已有的 communicator**（`backend._comm_ptr()`，见 `deep_ep/utils/comm.py:62`） |
| **对称内存** | 纯 GPU（`ncclMemAlloc`） | GPU + CPU 混合的"弹性"内存（`ElasticSymmetricMemory` 等） |
| **API 形态** | 高吞吐与低延迟**分开**（`dispatch`/`combine` + `low_latency_*`） | 统一为单个 `ElasticBuffer` 接口 |
| **SM 占用** | 较多（V1 低延迟约 24 SM） | 大幅降低（同场景约 4–6 SM） |
| **规模上限** | 较小 | 更大（号称支持到 EP2048） |
| **额外能力** | 无 | 新增 Engram（KV cache）、PP、CP、AGRS 等 |

> **一句话把握主线**：V1 靠"NVSHMEM 做 RDMA + 自己建 NCCL comm"，V2 则"**全面倒向 NCCL Gin** + **复用 PyTorch 的 comm**"。所以"DeepEP 怎么调用 NCCL"这个问题，**答案因版本而异**——这也是为什么我们必须先分清版本。

> **本笔记的讲述策略**：第二章的"DeepEP 怎么调用 NCCL"主要沿着 V2 的 `ElasticBuffer` 主路径讲（因为这是当前推荐、也是最能体现"GIN 底层能力"的路径），但在涉及 V1 特有的地方（比如 NVSHMEM 跨节点、`low_latency_*` 接口）会**明确标注是 V1**，避免混淆。

### 1.2 先理解 MoE 的通信特征

在普通的稠密模型（Dense）训练里，AllReduce 是主角：每个 GPU 算一部分梯度，然后所有人做一次全局归约。数据流是**对称的、规整的**——每个 GPU 发出去的字节数和收进来的字节数都一样，而且每个 GPU 都要和"所有"其他 GPU 通信。这正是 NCCL 最擅长处理的场景。

但 MoE 完全不一样。MoE 模型里，每一层的计算会被路由到不同的"专家"（expert）上，而每个 token 只会被少数几个（比如 top-2）专家处理。于是出现了一种全新的通信模式——**all-to-all**（全对全）：

- **Dispatch（分发）阶段**：每个 GPU 手里的一批 token，需要按"路由结果"发送到**各自不同的目标专家所在的 GPU**。token A 去 GPU 1 的专家 3，token B 去 GPU 5 的专家 7……**每个 token 的去向都不同**。
- **Combine（回收）阶段**：专家算完之后，每个 token 的结果又要被送回它**原来的 GPU**，做加权求和（因为一个 token 可能被多个专家处理）。

这套通信有几个让 NCCL 的通用集合通信"使不上力"的特点：

1. **数据量严重不均衡**：每个 GPU 分到的 token 数、每个专家收到的 token 数，在每一步都是动态变化的、且极不均匀。可能 GPU 0 要发 1000 个 token，GPU 3 只发 10 个。
2. **通信模式是稀疏的、数据相关的**：谁发给谁、发多少，取决于这步的路由结果，无法预先固定。
3. **通信和计算需要深度重叠**：专家计算（GEMM）很贵，通信必须尽量和它 overlap，否则通信会拖垮整体吞吐。

NCCL 本身其实也有 `AllToAll` 集合操作，但它是"规整的" all-to-all（每个 rank 发给每个 rank 等量数据），无法直接表达 MoE 这种"不规则、动态、稀疏"的 all-to-all。于是，需要一个专门的库来做这件事——这就是 DeepEP 的用武之地。

### 1.3 DeepEP 解决的三个核心场景

DeepEP 的 `Buffer` 类（V1，`deep_ep/buffers/legacy.py`）在文档里开门见山地列出了它支持的三类通信（翻译如下）：

1. **高吞吐节点内 all-to-all**（intranode，用 NVLink）——同机多卡之间，追求最大带宽；
2. **高吞吐节点间 all-to-all**（internode，用 RDMA + NVLink）——跨机场景，RDMA 走跨机、NVLink 走机内，两者组合；
3. **低延迟 all-to-all**（low-latency，用 RDMA）——专门为**推理**场景优化，追求最小延迟而非最大带宽。

> **一句话记忆**：训练场景看重"吞吐"（带宽要跑满），推理场景看重"延迟"（首 token 要快）。DeepEP 为两者分别做了高度定制化的内核。

---

## 二、DeepEP 到底怎么调用 NCCL（关键：读代码）

这是本部分最重要的一节。很多人的直觉是"DeepEP 会调用 NCCL 的 all-to-all 函数"，**但事实并非如此**。DeepEP 对 NCCL 的使用方式非常"底层"、也非常巧妙——它**几乎不用 NCCL 的集合通信 API，而是复用了 NCCL 提供的三个更底层的能力**：

1. **对称内存（Symmetric Memory）**：一块所有 GPU 都映射到**相同虚拟地址**的内存；
2. **GIN（GPU Initiated Network）**：让 GPU 内核里的线程**直接发起 RDMA 网络操作**，绕开 CPU；
3. **Team 与 LSA 窗口**：查询 GPU 间的 NVLink 拓扑关系，并拿到"对称内存里对方的那块地址"。

> **⚠️ 先纠正一个版本误区**：很多资料说"DeepEP 跨节点用 NVSHMEM"，这其实是 **V1 的做法**。在 **V2** 里，DeepEP 已经把跨节点 RDMA 也统一交给了 **NCCL Gin**（即上面第 2 点的 `gin.put`/`gin.get`，它们本身就支持跨节点），NVSHMEM 仅在 legacy 兼容路径里保留。所以本节的讲述以 **V2 的 NCCL Gin 主路径**为准，V1 的 NVSHMEM 会在 7.5 节单独说明。

下面逐条拆解，每条都标出对应的源码位置，请你打开源码对照着看。

### 2.1 第一步：拿到 NCCL communicator（但为了当"钩子"而非做集合通信）

DeepEP 需要先有一个 `ncclComm_t` 作为后续所有底层 API 的"把手"。这里 **V1 和 V2 的做法不同**，正是上一节对比表里最关键的一处差异：

- **V1（legacy）**：自己动手建——在 C++ 侧 `csrc/kernels/backend/nccl.cu` 里用 `ncclGetUniqueId` + `ncclCommInitRank` 新建一个独立的 communicator：

```cpp
// csrc/kernels/backend/nccl.cu:20-41（V1 路径）
pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    ...
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    ncclUniqueId root_unique_id;
    ...
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    return reinterpret_cast<int64_t>(comm);
}
```

- **V2（elastic）**：**直接复用 PyTorch 已有的 communicator**，不再自己建。关键在 `deep_ep/utils/comm.py:42-64` 的 `get_nccl_comm_handle`：

```python
# deep_ep/utils/comm.py:60-64（V2 路径，节选）
backend = group._get_backend(torch.device('cuda'))
if not force_new_comm and hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
    # 新版 PyTorch 暴露了内部 NCCL communicator 指针，直接拿过来复用
    _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)
    return _storage[group]
```

只有在新版 PyTorch **没有** `_comm_ptr` 这个接口时，V2 才退回到"自己建 comm"的 V1 老路（`comm.py:67-74`）。

无论哪条路径，拿到这个 `ncclComm_t` 之后，DeepEP **都不是**用来调用 `ncclAllReduce` 之类的集合操作，而是把它作为后续一系列底层 API 的"把手（handle）"——这就引出了下面 7.2 的对称内存、窗口、LSA 指针，以及 7.3 的内核级 GIN。

### 2.2 第二步：分配对称内存 + 注册窗口 + 拿 LSA 指针

这是 DeepEP 复用 NCCL 能力最核心的一段。同样在 `nccl.cu` 的 `NCCLSymmetricMemoryContext` 构造函数里（`csrc/kernels/backend/nccl.cu:62-148`），依次发生了：

**（1）创建 device communicator（含 GIN 配置）：**

```cpp
// csrc/kernels/backend/nccl.cu:83-108（节选）
ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
ncclCommQueryProperties(comm, &props);
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
    // 配置 GIN：上下文数量、队列深度、流量类别、信号数量等
    reqs.ginContextCount = num_allocated_qps;
    reqs.ginExclusiveContexts = true;
    reqs.ginQueueDepth = kGinQPDepth;
    reqs.ginTrafficClass = sl_idx;
    reqs.ginSignalCount = num_ranks + 2 * 2;
    reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
}
NCCL_CHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm.ptr)));
```

这里出现的 `ncclDevCommCreate` 和 GIN（`ginContextCount`、`ginQueueDepth`、`ginSignalCount`……），正是我们前面 NCCL 笔记里讲的 **GIN 机制**。DeepEP 在创建 device communicator 时，就要求 NCCL 为它准备好"GPU 直接发起网络"所需的一切资源。

**（2）分配对称内存：**

```cpp
// csrc/kernels/backend/symmetric.hpp:124-140（GPUSymmetricMemory 节选）
explicit GPUSymmetricMemory(const int64_t& num_bytes) {
    NCCL_CHECK(ncclMemAlloc(&ptr, num_bytes));
    ...
}
```

`symmetric.hpp` 里定义了三种对称内存实现（后面 7.4 节会展开），最基础的一种 `GPUSymmetricMemory` 直接调用 NCCL 的 `ncclMemAlloc`——这块内存会被 NCCL 保证在所有 rank 上映射到**相同的虚拟地址**，这是"对称内存"的关键。

**（3）注册窗口 + 获取 LSA 指针：**

```cpp
// csrc/kernels/backend/nccl.cu:135-147（节选）
// `ncclCommWindowRegister` 是集合操作：内部会跨所有 rank 调 bootstrapBarrier
NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, this->symmetric_memory->num_bytes,
                                  &window, NCCL_WIN_STRICT_ORDERING));
NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));

// 获取所有 LSA 对端（同 NVLink 域内各 GPU）的指针
nvl_window_ptrs.resize(num_nvl_ranks);
for (int i = 0; i < num_nvl_ranks; ++ i)
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
```

这里有两个关键 NCCL 概念，值得你停下来理解：

- **`ncclCommWindowRegister`（窗口注册）**：把一块对称内存"注册"成一个 window，让它可以被跨 GPU 访问；
- **`ncclGetLsaDevicePointer`（LSA 指针）**：LSA = "local NVLink 可达域"（可以理解为同一节点内通过 NVLink 互相可见的一组 GPU）。这个 API 能返回"**在 LSA 域内第 `i` 个 GPU 上，这块 window 内存对应的本地地址**"。

有了这些指针，DeepEP 就实现了一个强大的能力：**同一 NVLink 域内的任意 GPU，可以直接通过指针访问其他 GPU 的缓冲区**，无需任何 CPU 参与。

### 2.3 第三步：内核里直接用 GIN 收发数据（而非调用集合通信）

这是 DeepEP 最"反直觉"也最精彩的地方。打开内核的辅助头文件 `deep_ep/include/deep_ep/common/handle.cuh`，你会看到一个 `NCCLGin` 结构体，它封装了对 NCCL 设备侧 GIN API 的直接调用：

```cpp
// deep_ep/include/deep_ep/common/handle.cuh:27-35（节选）
NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
        const int& qp_idx = 0, ...)
    : gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),
      team_world(ncclTeamWorld(nccl_dev_comm)),
      team_lsa(ncclTeamLsa(nccl_dev_comm)),
      team_rail(ncclTeamRail(nccl_dev_comm)) {
    lsa_base_ptr = reinterpret_cast<uint64_t>(ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));
}
```

然后，`NCCLGin` 提供了 `put` / `get` / `signal` 等操作，它们在 GPU 内核内部直接发起网络操作：

```cpp
// deep_ep/include/deep_ep/common/handle.cuh:175-198（put 节选）
template <typename team_t, ...>
void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes, const int& dst_rank_idx, ...) {
    // local or NVLink put 也会通过这个 API 走 NIC
    gin.put(TEAM_WORLD_RAIL(), dst_rank_idx,
            nccl_window, reinterpret_cast<int64_t>(recv_sym_ptr) - lsa_base_ptr,
            nccl_window, reinterpret_cast<int64_t>(send_sym_ptr) - lsa_base_ptr,
            num_bytes, ...);
}
```

**这意味着什么？** 在 DeepEP 的 dispatch/combine 内核执行过程中，**GPU 里的线程自己就能决定"把这个 token 的数据，直接 RDMA 写到目标 GPU 的对称内存里"**，完全不需要先结束内核、再让 CPU 去提交一次通信。这正是 GIN（GPU Initiated Network）的价值，也是 DeepEP 能做到极致低延迟的关键。

> **对比一下你就会有体感**：传统 NCCL 的流程是"GPU 内核算完 → CPU 的 Proxy 线程轮询 → CPU 提交 RDMA → 网卡执行"；而 DeepEP 的流程是"GPU 内核里的线程直接 `gin.put` → 网卡执行"。省掉了 CPU 这一环的多次同步与调度。

实际的 dispatch 内核 `deep_ep/include/deep_ep/impls/dispatch.cuh` 里，每个 warp 都会构造一个 `NCCLGin`，并把它当作"channel"来收发数据：

```cpp
// deep_ep/include/deep_ep/impls/dispatch.cuh:67-71（节选）
// 我们把每个 warp 当作一个 "channel"
const auto [qp_idx, sharing_mode] = comm::get_qp_mode<...>(sm_idx, warp_idx - kNumNotifyWarps, ...);
const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
```

### 2.4 对称内存的三种实现（`symmetric.hpp`）

前面提到 `symmetric.hpp` 里有三种对称内存，这里展开一下，它们对应不同的部署形态（源码在 `csrc/kernels/backend/symmetric.hpp`）：

| 实现类 | 内存构成 | 适用场景 |
|--------|----------|----------|
| `GPUSymmetricMemory` | 纯 GPU 显存（`ncclMemAlloc`） | 常规场景，数据全放显存 |
| `ElasticSymmetricMemory` | GPU 显存（前段）+ CPU 内存（后段，NUMA-local） | "弹性"场景，用 CPU 内存扩充容量，缓解显存不足 |
| `HybridElasticSymmetricMemory` | GPU 显存 + 每个 scaleup rank 各一段 CPU 内存 | 混合模式，每个 rank 的 CPU 段都映射进来 |

它们的共同点是：**构造出一段连续的虚拟地址空间，所有 rank 映射到相同地址**，从而兼容 `ncclCommWindowRegister`（窗口注册要求地址对齐且对称）。其中 `ElasticSymmetricMemory` 还会设置环境变量 `NCCL_ELASTIC_BUFFER_REGISTER=1` 来启用 NCCL 的弹性缓冲注册（见 `symmetric.hpp:312-316`）。

> 这一节体现了一个重要工程思想：**DeepEP 不只是"调用" NCCL，它甚至会反过来影响 NCCL 的行为**（通过环境变量、通过复用 NCCL 的对称内存分配器 `ncclMemAlloc`）。两者是深度耦合、互相配合的关系。

### 2.5 跨节点 RDMA 的版本演进：从 NVSHMEM（V1）到 NCCL Gin（V2）

细心的你可能会问：跨节点的数据搬运（internode dispatch/combine），DeepEP 到底用的是什么？

答案是：**这也是 V1 和 V2 的分水岭**。

**V1 用 NVSHMEM**：V1 的跨节点 RDMA 走 **NVSHMEM**（NVIDIA 的 PGAS 编程模型库）+ **IBGDA**。证据在 `csrc/kernels/backend/nvshmem.cu`：

```cpp
// csrc/kernels/backend/nvshmem.cu:49-76（V1 路径，init 节选）
int init(const std::vector<uint8_t>& root_unique_id_val,
         const int& rank, const int& num_ranks, const int& team_split_stride) {
    nvshmemx_uniqueid_t root_unique_id;
    ...
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    // 创建子 RDMA team（用于跨节点 RDMA）
    if (team_split_stride > 0 and num_ranks > team_split_stride) {
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, ...);
    }
    ...
}
```

而 Python 侧（V1，`deep_ep/buffers/legacy.py:103-132`）会设置一堆 NVSHMEM 环境变量来启用 **IBGDA**（InfiniBand GPU Direct Async，GPU 直接访问网卡）：

```python
os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

**V2 改用 NCCL Gin**：V2 把跨节点 RDMA 也统一交给了 NCCL Gin。这体现在两处：

1. 在 V2 的内核 `NCCLGin` 封装里（`deep_ep/include/deep_ep/common/handle.cuh`），`put`/`get` 用的是 `team_t` 参数 + `TEAM_WORLD_RAIL()`，它既能表示"机内 NVLink 可达"，也能表示"跨节点走 RDMA"——`is_nvlink_accessible` 判断不成立时，就走 `gin.put` 把数据经网卡发到远端（见 `handle.cuh:96-120` 的 `red_add_rel`，那里明确写了"NVLink 可达就走对称指针，否则走 `gin.signal`/RDMA"）；
2. V2 构造时，`ElasticBuffer` 里已经**不再初始化 NVSHMEM**（对比 V1 的 `Buffer` 会设置一堆 `NVSHMEM_*` 环境变量），跨节点依赖的 GIN 资源全部由 7.2 节里 `ncclDevCommCreate` 的 `ginContextCount`/`ginQueueDepth` 等配置提前备好。

> 官方 README 对此的总结是一句："V2 has switched from the NVSHMEM backend to the more lightweight NCCL Gin backend"（V2 已从 NVSHMEM 后端切换到更轻量的 NCCL Gin 后端）。NVSHMEM 仅作为 legacy 兼容保留，不再走主路径。

**于是，两代"通信底座"的对比可以这样看：**

| 通信域 | V1 用什么 | V2 用什么 |
|--------|----------|----------|
| **机内（NVLink 域）** | NCCL 对称内存 + LSA 指针 | NCCL 对称内存 + LSA 指针（基本一致） |
| **跨机（RDMA）** | **NVSHMEM + IBGDA** | **NCCL Gin**（`gin.put`/`gin.get`） |
| **GPU 侧发起网络** | NVSHMEM 的 put/get + 部分 GIN | 统一走 NCCL Gin |
| **进程间控制面** | 自建 NCCL communicator | 复用 PyTorch 的 NCCL communicator |

> **一句话总结 DeepEP 与 NCCL 的关系**：DeepEP **复用**了 NCCL 的对称内存、GIN、窗口/LSA 这些"底层积木"，但**不用** NCCL 的 `AllToAll` 等集合通信接口，而是自己写了针对 MoE 不规则通信量身定制的 dispatch/combine 内核。V1 时代跨节点靠 NVSHMEM 补位，V2 时代则连跨节点也一并收编进了 NCCL Gin——**对 NCCL 的依赖反而更深了**。

---

## 三、DeepEP 的独创之处：为什么各方都在参考它

理解了 DeepEP 对 NCCL 的调用方式之后，我们再站高一点，看看它到底"独创"在哪里、为什么值得学习。

### 3.1 专为"不规则 all-to-all"定制内核

这是最根本的一点。NCCL 的 `AllToAll` 是规整的（每个 rank 间等量），而 MoE 的 dispatch/combine 是**动态、稀疏、不均衡**的。DeepEP 为此专门设计了 dispatch 和 combine 内核（`deep_ep/include/deep_ep/impls/dispatch.cuh`、`combine.cuh`），核心思路：

- **运行时动态计算路由**：内核里先统计"每个 token 要去哪个 rank/哪个专家"，用原子操作在共享内存里累加计数（`dispatch.cuh:92-107` 的 `atomicAdd_block`），实时算出收发布局；
- **前缀和（prefix sum）定位**：用前缀和矩阵（`rank_prefix_matrix`、`channel_prefix_matrix`）精确定位每个 token 应该写到目标缓冲区的哪个位置，避免冲突；
- **warp 即 channel**：把每个 warp 当作一个通信 channel，对应一个 GIN 的 QP（队列对），实现细粒度的并行收发。

### 3.2 内核级通信（GIN）带来的极致重叠

传统集合通信里，通信和计算是"串行"的（算完再传、传完再算）。DeepEP 通过 GIN，让 GPU 内核在**计算过程中随时发起 RDMA**，实现了真正的**计算-通信细粒度重叠**。这对于 MoE 尤其重要，因为专家 GEMM 非常昂贵，任何一点通信停顿都会被放大。

### 3.3 低延迟模式的针对性设计（面向推理）

> 说明：下面讲的 `low_latency_dispatch` / `low_latency_combine` 是 **V1（legacy）** 的专用接口（`deep_ep/buffers/legacy.py:553-713`）。在 V2 里，低延迟能力已经**合并进统一的 `ElasticBuffer` 接口**，不再有独立的 `low_latency_*` API（V2 的 `dispatch`/`combine` 通过 `async_with_compute_stream` 等参数表达同样的"少 SM、与计算重叠"诉求）。这里仍以 V1 的低延迟内核为例，因为它最能讲清楚"面向推理的低延迟"到底做了哪些针对性设计：

`low_latency_dispatch` / `low_latency_combine`（V1，`deep_ep/buffers/legacy.py:553-713`）是 DeepEP 面向推理的招牌功能，几个关键设计：

- **纯 RDMA（IBGDA）**：所有 rank 都通过 RDMA 直接互访，不走 NVLink，把延迟压到最低；
- **固定缓冲 + 无 CPU 同步**：接收缓冲是预先固定分配的（`num_max_dispatch_tokens_per_rank` 决定上限），并且**不把"收到多少 token"同步回 CPU**（`legacy.py:600` 注释明确说明），这样才兼容 CUDA Graph；
- **接收 hook**：`return_recv_hook` 允许"只发起 RDMA 请求、不等待数据到达"，把等待推迟到真正需要数据的那一刻，进一步和计算重叠；
- **FP8 转换**：通信过程中直接做 FP8 量化（`use_fp8` 参数），在低延迟模式下顺带减少传输数据量。

### 3.4 丰富的工程细节

> 说明：下面这些工程优化是 DeepEP 的**共性设计思想**，V1/V2 都有对应实现，只是接口命名不同（V1 是 `Buffer` + 元组 handle，V2 是 `ElasticBuffer` + `EPHandle` 对象）。这里以 V1 的接口名为例标注源码位置，便于你在 legacy 代码里定位；V2 里同样能找到等价物。

- **通信 handle 复用**：dispatch 返回的 `handle` 里缓存了前缀和矩阵、源索引等布局信息，下一次 combine 或同布局的 dispatch 可以直接复用，省去重复计算（V1 见 `legacy.py:385-405`，V2 见 `EPHandle` + `elastic.py` 的 cached handle 逻辑）；
- **专家对齐（expert_alignment）**：把每个专家收到的 token 数对齐到某个倍数，方便后续 GEMM 内核的访存对齐；
- **rank 动态屏蔽（mask）**：`low_latency_update_mask_buffer` 等接口支持在通信中动态屏蔽某个 rank，用于在线服务的故障隔离（`legacy.py:672-698`）；
- **性能统计内置**：`dispatch_wait_recv_cost_stats`、`combine_wait_recv_cost_stats` 等参数能在通信中直接统计"等待接收"的耗时，用于定位慢节点（`legacy.py:577-579`）。

### 3.5 建议你重点阅读的源码路径

如果读完这一部分你想深入，建议按下面的顺序读（由浅入深）。注意分清两条线——**先读 V2 主路径**（当前推荐），再回头看 V1 的 legacy 路径作对比：

**V2 主路径（优先）：**

1. **`deep_ep/buffers/elastic.py`**：V2 的 Python 入口，先理解 `ElasticBuffer` 的 `dispatch` / `combine` 以及它如何通过 `get_nccl_comm_handle` 复用 PyTorch 的 communicator（`deep_ep/utils/comm.py`）；
2. **`csrc/kernels/backend/nccl.cu`**：看 DeepEP 如何初始化 NCCL、注册窗口、拿 LSA 指针（`NCCLSymmetricMemoryContext`，承接我们前面 NCCL 的知识）；
3. **`csrc/kernels/backend/symmetric.hpp`**：看对称内存的三种实现（V2 重点看 `ElasticSymmetricMemory` / `HybridElasticSymmetricMemory`）；
4. **`deep_ep/include/deep_ep/common/handle.cuh`**：看 `NCCLGin` 如何封装 GIN 的 put/get/signal（这是 V2"GPU 直接通信、跨节点也走 GIN"的核心）；
5. **`deep_ep/include/deep_ep/impls/dispatch.cuh` / `combine.cuh`**：看 V2 的不规则 all-to-all 内核（`namespace deep_ep::elastic`）。

**V1 legacy 路径（对比参考）：**

6. **`deep_ep/buffers/legacy.py`**：V1 的 `Buffer`，重点看 `low_latency_dispatch` / `low_latency_combine` 这两个面向推理的专用接口；
7. **`csrc/kernels/backend/nvshmem.cu`**：看 V1 跨节点 RDMA 的 NVSHMEM 初始化，体会它和 V2 用 NCCL Gin 的差异。

---

## 参考资料

- DeepEP 官方源码仓库：[github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)（本笔记基于 v2.1.0）
- DeepEP 技术博客：DeepSeek 官方发布的 DeepEP 解读文章（含低延迟内核、混合通信等设计细节）
- DeepEP V1 legacy 文档：官方仓库 `docs/legacy.md`（NVSHMEM 时代的接口与性能数据，用于和 V2 对比）
- NCCL 官方源码仓库：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)（V2 的 GIN 后端、对称内存窗口、通信器初始化均来自它，机制详见[《NCCL 源码学习笔记》](/notes/nccl-source-study/)）

---

*声明：本文是作者学习 DeepEP 源码过程中的阶段性笔记，内容基于 NCCL master 分支与 DeepEP 官方仓库源码整理。由于这是一份仍在持续完善的初稿，文中可能存在表述不准确或理解不完整之处，欢迎通过邮件联系指正。*
