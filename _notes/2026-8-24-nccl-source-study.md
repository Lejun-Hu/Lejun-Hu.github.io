---
title: "【未完成】NCCL 源码学习笔记"
permalink: /notes/nccl-source-study/
date: 2026-09-20
category: "集合通信"
tags: ["NCCL", "集合通信", "源码分析", "AllReduce", "Ring", "Tree", "GIN", "GPU互联", "CUDA", "DeepEP", "MoE", "Expert Parallel"]
description: "NCCL 源码学习笔记（初稿）：从零开始讲清 NCCL 是什么、它如何支撑 PyTorch 分布式训练、底层打通了哪些硬件链路；再以 MoE 架构为引，深入 DeepEP 库及其对 NCCL 的真实调用依赖。"
published: true
---

> 很多同学第一次接触 NCCL，是在写 PyTorch 分布式训练的 `all_reduce` 时。代码里只写了一行 `dist.all_reduce(tensor)`，训练却能跨 8 张卡、甚至跨几百台机器的几千张卡完成梯度同步——这背后到底发生了什么？
>
> 答案就藏在 NCCL（NVIDIA Collective Communications Library）里。它是一套专门为 GPU 间集合通信而生的底层库，由 NVIDIA 开源。它不关心你的模型是什么、损失函数是什么，它只做一件极致的事：**在尽可能短的时间里，把数据在参与训练的所有 GPU 之间正确地搬来搬去**。
>
> 本文是一份**面向初学者的、仍在持续完善的笔记**。我会尽量假设你"还没读过源码"，从"为什么需要 NCCL"讲起，逐步深入到源码结构、初始化流程、执行路径，以及一个在实际生产中被广泛参考的上层库——DeepEP（专为 MoE 专家并行设计的通信库），看它如何站在 NCCL 的肩膀上做出自己的独创设计。
>
> 文中涉及的每个文件、函数、数据结构，都会标注它在源码中的位置，方便你随时打开源码对照阅读。建议的阅读方式：**先通读建立骨架，再挑你感兴趣的一节，跟着代码位置去源码里"钻"进去看**。

---

## 〇、背景引入：NCCL 在软件栈中的位置

在翻开 NCCL 源码之前，先花一点时间搞清楚三件事：NCCL 到底是什么、它给上游（PyTorch）提供了什么、它往下打通了哪些硬件链路。理解了这张"位置图"，后面看源码才不会迷路。

### 0.1 NCCL 是什么，解决什么问题

一句话：**NCCL 是 GPU 集群上做"集合通信"（collective communication）的标准库**。

"集合通信"指的不是两个点之间传数据（那是点对点），而是一群进程/GPU 之间协调地交换数据。下面先列出几种最常见的集合通信原语（primitive），帮你建立直观印象。

> **关于算法本身的说明**：这里只是简单介绍这些原语"做什么"，并不展开"怎么做"。事实上，集合通信**算法**本身就是一个极其值得深入探究的领域——同一种 AllReduce，用 Ring、Tree 还是其他算法，在不同的网络拓扑下，其通信步数（step）和总延迟都会发生显著变化。我们计划**单独开一篇笔记**专门讲集合通信算法的细节与性能权衡；本篇笔记暂时先以**理解代码的大致架构**为主，不深入算法性能层面的讨论。

| 集合操作 | 含义 | 典型用途 |
|----------|------|----------|
| **AllReduce** | 每个 GPU 都有一个数据，所有人拿到"所有数据的归约结果"（如求和） | 梯度同步：把各卡算出的梯度求平均 |
| **Broadcast** | 一个 GPU 的数据广播给所有人 | 分发模型参数 |
| **AllGather** | 每人的数据拼起来，所有人拿到完整拼接结果 | 收集各卡的部分结果 |
| **ReduceScatter** | 归约 + 切分，每人只拿到自己那份结果 | 大模型并行里常和 AllGather 搭配 |
| **Send / Recv** | 点对点收发 | 流水线并行里的张量传递 |

为什么需要专门的库？因为"把数据搬得快"这件事，在 GPU 集群上远比想象中复杂：

- GPU 之间可能通过 **NVLink**（同机高速直连）、**PCIe**（同机较慢直连）、**RDMA 网卡（InfiniBand / RoCE）**（跨机）等多种物理链路相连；
- 同一种 AllReduce，用"环形（Ring）"还是"树形（Tree）"算法，在不同数据量、不同拓扑下性能差异巨大；
- 还要处理拓扑感知、多流并发、故障恢复等一堆工程细节。

NCCL 把这些复杂性全部封装起来，向上暴露一组简洁的 C API（`ncclAllReduce`、`ncclBroadcast`……），向下自动探测硬件拓扑并选择最优算法。于是上层框架只需要"我要做 AllReduce"，剩下的交给 NCCL。

### 0.2 向上：PyTorch 是如何调用到 NCCL 的（代码视角）

这是理解 NCCL 价值的关键一环。让我们沿着一条真实的调用链，从你写的 Python 代码走到 NCCL：

```
你的代码: torch.distributed.all_reduce(tensor)
    ↓
Python 层: torch/distributed/distributed_c10d.py
    ↓  (ProcessGroup 抽象)
C++ 层:   torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
    ↓  (调用 NCCL C API)
NCCL 库:  libnccl.so → ncclAllReduce(...)
    ↓
底层硬件: NVLink / PCIe / RDMA 网卡
```

**关键分层（从上到下）：**

1. **`torch.distributed`（Python API）**：你写的 `all_reduce`、`broadcast` 等，最终都会落到一个 `ProcessGroup` 对象上。PyTorch 支持多种后端（Gloo、MPI、NCCL……），当你用 `dist.init_process_group(backend="nccl")` 时，选中的就是 `ProcessGroupNCCL`。

2. **`ProcessGroupNCCL`（C++ 层）**：这是 PyTorch 里专门对接 NCCL 的胶水层，源码在 `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`。它负责：
   - 把 PyTorch 的 Tensor 地址、数据类型、Reduction 操作翻译成 NCCL 的参数；
   - 管理 NCCL communicator（`ncclComm_t`）的创建与复用；
   - 处理 CUDA stream 的同步（确保通信和计算在同一流上有正确的先后关系）。

3. **NCCL C API**：`ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)`。看到这个签名你就明白了——NCCL 不关心张量形状、不关心梯度，它只关心"从哪块内存搬到哪块内存、多少字节、什么归约操作、在哪个 CUDA 流上执行"。

   > 你可以去 PyTorch 源码里验证：在 `ProcessGroupNCCL.cpp` 里搜 `ncclAllReduce(`，会看到 `allreduce` 相关的函数里直接调用了它，把 `tensor.data_ptr()` 作为 `sendbuff`/`recvbuff` 传进去。

**所以，"不同库承担的角色"可以这样概括：**

| 库/层 | 角色 | 它关心的事 |
|-------|------|-----------|
| PyTorch / 训练框架 | 定义"要做什么" | 模型、优化器、训练循环；决定"何时同步梯度" |
| `torch.distributed` + `ProcessGroupNCCL` | 翻译与适配 | 把框架语义翻译成集合通信调用；管理设备、流、进程组 |
| **NCCL** | **执行通信** | 探测拓扑、选算法、真正把数据在 GPU 间搬完 |
| 底层驱动（CUDA driver、NIC 驱动） | 硬件抽象 | 提供对 NVLink、RDMA 网卡等硬件的访问接口 |

### 0.3 向下：NCCL 打通了哪些硬件链路

NCCL 的核心价值之一，就是它把下面这些异构的物理链路**统一抽象成了几种"传输层（transport）"**，并根据 GPU 间的拓扑自动选择最快的那条：

| 传输层 | 底层硬件 | 适用场景 |
|--------|----------|----------|
| **P2P** | NVLink / PCIe / C2C（同机 GPU 直连） | 同一节点内 GPU 之间，直接读写对方显存 |
| **SHM** | 共享内存（Host 内存 + CPU memcpy） | 同进程多卡、或无法 P2P 时的兜底 |
| **NET** | RDMA 网卡（InfiniBand / RoCE） | 跨节点，GPU 数据经网卡直接搬远 |
| **CollNet** | 支持 SHARP 的交换机（网内计算） | 跨节点大消息，把归约"卸载"到交换机 |

对应到源码，这些 transport 分别实现在 `src/transport/` 目录下的 `p2p.cc`、`net.cc`、`coll_net.cc` 等文件里。后文"传输层"部分会详细讲。

> **带着这张地图读源码**：NCCL 的源码虽然庞大，但主线很清晰——"探测拓扑 → 选 transport → 选算法 → 启动内核 → 内核里通过 transport 搬数据"。后面每一节，本质上都是在展开这条主线的某个环节。

---

## 一、源码目录架构

在深入具体机制之前，先建立对 NCCL 源码整体组织方式的认知。NCCL 的源码结构按照功能域清晰分层：`src/` 下的每个一级 `.cc` 文件通常对应一个核心子系统，其头文件放在 `src/include/` 中；设备侧代码统一在 `src/device/`，传输层实现在 `src/transport/`，插件适配层在 `src/plugin/`。

### 1.1 顶层目录

| 目录/文件 | 用途 |
|-----------|------|
| `src/` | 核心源码（Host + Device），全部 `.cc`/`.cu`/`.h` 文件 |
| `src/include/` | 内部头文件（分功能模块组织） |
| `src/device/` | GPU 端内核代码（`.cu`, `.h`），集合通信原语 |
| `src/graph/` | 拓扑发现、Ring/Tree 图搜索、XML 解析 |
| `src/transport/` | 四种传输层实现：P2P/SHM/NET/CollNet |
| `src/gin/` | GIN（GPU Initiated Network）Host 侧实现 |
| `src/nccl_device/` | 对称运行时（symmetric runtime） |
| `src/rma/` | RMA（Remote Memory Access） |
| `src/ras/` | 可靠性/可用性/可维护性（RAS） |
| `src/scheduler/` | 任务调度、AllGatherV 调度 |
| `src/plugin/` | 插件系统（Net/GIN/Tuner/Profiler 各版本编解码） |
| `src/misc/` | 工具函数（参数检查、CUDA/IB 封装、Socket 等） |
| `src/os/` | 操作系统抽象层（Linux/Windows） |
| `src/param/` | 参数系统（环境变量注册与解析） |
| `src/register/` | 缓冲区注册（集合通信和 SendRecv） |
| `src/devcomm/` | 多版本设备通信结构体定义（v22902/v22907/v23000） |
| `bindings/` | 多语言绑定 |
| `nccl.h.in` | 公共 API 头文件模板 → `nccl.h` |
| `CMakeLists.txt` | 构建系统入口 |

### 1.2 src/ 核心源码详解

| 文件 | 执行位置 | 功能 |
|------|----------|------|
| `init.cc`（146KB） | Host | 初始化入口：`ncclCommInitRank`、`ncclCommInitRankDev`、`initTransportsRank`、通信器分配/释放 |
| `enqueue.cc`（144KB） | Host | 集合操作入队：`ncclEnqueueCheck`、Kernel Plan 构建、Work Batch 组装、算法/协议选择 |
| `group.cc`（34KB） | Host | Group 操作：`ncclGroupStart/End`、异步任务管理、Preconnect |
| `collectives.cc`（16KB） | Host | 所有集合操作 API 入口：AllReduce、AllGather、ReduceScatter、Broadcast、SendRecv、RMA 等 |
| `proxy.cc`（84KB） | Host | Proxy 线程机制：负责 Host 侧数据搬运（Network → GPU / SHM → GPU） |
| `transport.cc`（22KB） | Host | 传输层调度：`selectTransport`、`ncclTransportP2pConnect`、传输类型检查 |
| `bootstrap.cc`（57KB） | Host | 带外 Bootstrap 机制：TCP Socket 通信，用于初始化阶段的 AllGather/Barrier |
| `channel.cc`（9KB） | Host | Channel 初始化与释放 |
| `mem_manager.cc`（39KB） | Host | 显存管理器：动态内存分配/释放、Suspend/Resume |
| `allocator.cc`（16KB） | Host | 内存分配器：MemoryStack、MemoryPool |
| `device/common.cu` | Device | 设备侧统一内核入口：`ncclDevKernel_Generic` |
| `device/prims_ll.h` | Device | LL 协议原语（小数据量） |
| `device/prims_ll128.h` | Device | LL128 协议原语（中等数据量） |
| `device/prims_simple.h` | Device | Simple 协议原语（大数据量） |
| `device/all_reduce.h` | Device | AllReduce 设备端实现模板 |
| `device/reduce_scatter.h` | Device | ReduceScatter 设备端实现模板 |
| `device/all_gather.h` | Device | AllGather 设备端实现模板 |
| `device/broadcast.h` | Device | Broadcast 设备端实现模板 |
| `device/reduce.h` | Device | Reduce 设备端实现模板 |
| `device/sendrecv.h` | Device | SendRecv 设备端实现模板 |
| `gin/gin_host.cc` | Host | GIN Host 侧核心：初始化、Connect、Progress 线程 |
| `gin/gin_host_proxy.cc` | Host | GIN Proxy Backend 的 Host 侧实现 |
| `graph/topo.cc` | Host | 拓扑图构建：GPU/PCI/NIC/NVSwitch 节点与链路 |
| `graph/rings.cc` | Host | Ring 搜索算法 |
| `graph/trees.cc` | Host | Tree 搜索算法 |
| `graph/paths.cc` | Host | 路径计算（最短路径与带宽） |
| `graph/xml.cc` | Host | XML 拓扑解析（NVML 导出） |
| `graph/tuning.cc` | Host | Tuner 集成 |

### 1.3 include/ 头文件组织

| 路径 | 内容 |
|------|------|
| `include/comm.h` | **核心数据结构**：`ncclComm`（523-797 行）、`ncclChannel`、`ncclKernelPlan`、`ncclKernelPlanner`、`ncclSharedResources` |
| `include/transport.h` | 传输层框架：`ncclTransport` 虚函数表、四种 transport 类型常量、`ncclPeerInfo` |
| `include/graph.h` | 拓扑图基础类型：`ncclTopoNode`、`ncclTopoLink`、`ncclTopoSystem` |
| `include/proxy.h` | Proxy 机制：`ncclProxyOp`、`ncclProxyArgs`、`ncclProxyState` |
| `include/enqueue.h` | 入队接口：`ncclEnqueueCheck`、`ncclLaunchKernel`、协议对齐常量 |
| `include/bootstrap.h` | Bootstrap：`bootstrapInit`、`bootstrapAllGather`、`bootstrapBarrier` |
| `include/channel.h` | Channel 管理 |
| `include/net.h` | 网络抽象（内部） |
| `include/gin.h` | GIN 内部接口 |
| `include/device.h` | 设备侧数据结构与常量 |
| `include/p2p.h` | P2P 连接器定义（`ncclConnector`） |
| `include/plugin/nccl_net.h` | **Net Plugin API**：`ncclNet_t` (v12)、网络插件接口定义 |
| `include/plugin/nccl_gin.h` | **GIN Plugin API**：`ncclGin_t` 插件接口 |
| `include/plugin/nccl_tuner.h` | Tuner 插件接口 |
| `include/nccl_device/` | 设备侧 NCCL 运行时：comm、barrier、GIN 设备 API、向量操作等 |
| `include/param/` | 参数系统：注册、解析器（枚举/列表/位集） |
| `include/rma/` | RMA 内部接口 |
| `include/compiler/` | 编译器适配（GCC/MSVC） |
| `include/os/` | 操作系统适配 |

### 1.4 关键数据结构总览

**`ncclComm`**（定义于 `src/include/comm.h:523-797`）是 NCCL 中最重要的数据结构，包含：

- **拓扑信息**：`ncclTopoSystem* topo`、`rank`、`nRanks`、`cudaDev`、`busId`
- **Channel 数组**：`ncclChannel channels[MAXCHANNELS]`，每个 Channel 承载 Ring/Tree 结构
- **Graph 数组**：`ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]`，分别为 Ring/Tree/CollNet/NVLS 算法存储图结构
- **网络**：`ncclNet_t* ncclNet`、`netContext`、`ginContext`
- **内核调度**：`ncclKernelPlanner planner`、workFifo 机制
- **共享资源**：`ncclSharedResources* sharedRes`（跨 Split 子通信器共享）
- **配置**：`ncclConfig_t config`

**`ncclChannel`**（定义于 `src/include/comm.h:150-172`）代表一条独立的通信通道：

- `ncclRing ring`：Ring 拓扑连接信息
- `ncclTree tree`：Tree 拓扑连接信息
- `ncclChannelPeer** peers`：指向各 Peer 的连接器数组
- `workFifoProduced`：Work FIFO 生产指针

**`ncclTransport`**（定义于 `src/include/transport.h:136-142`）是传输层虚函数表：

- `canConnect`：判断两节点间是否可用此传输
- `send.setup` / `recv.setup`：建立连接（交换 handle）
- `send.connect` / `recv.connect`：完成连接
- `send.free` / `recv.free`：释放连接
- `proxySetup` / `proxyConnect` / `proxyProgress`：Proxy 线程侧操作

四种实例：`p2pTransport`、`shmTransport`、`netTransport`、`collNetTransport`。

---

## 二、初始化流程：从 API 调用到网络就绪

### 2.1 总体调用链路

完整调用链：

```
ncclGetUniqueId → ncclCommInitRank → ncclCommInitRankDev → ncclAsyncLaunch → ncclCommInitRankFunc
```

**代码位置：**

- `ncclGetUniqueId`：`src/init.cc:183` — 创建 Root bootstrap handle，生成 128 字节唯一 ID
- `ncclCommInitRank`：`src/init.cc:2562` — 公共 API 入口，解析配置、调用 `ncclCommInitRankDev`
- `ncclCommInitRankDev`：`src/init.cc:2477` — 创建异步任务 `ncclCommInitRankAsyncJob`
- `ncclCommInitRankFunc`：`src/init.cc:1831` — 异步任务执行体，完成全部初始化

### 2.2 Bootstrap 阶段

**功能：** 在所有 Rank 间建立带外通信通道，用于交换初始化信息。基于 TCP Socket，不依赖任何 GPU 间直接互联。

关键调用（在 `ncclCommInitRankFunc` 中）：

```cpp
// src/init.cc:1916 — 常规初始化
bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm, job->parent);

// src/init.cc:1891 — Split 初始化（从父通信器派生）
bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks);
```

Bootstrap 提供的原语（`src/include/bootstrap.h`）：

- `bootstrapAllGather`：所有 Rank 交换数据
- `bootstrapSend` / `bootstrapRecv`：点对点数据交换
- `bootstrapBarrier`：全局栅栏同步
- `bootstrapBroadcast`：广播
- `bootstrapIntraNodeBarrier` / `bootstrapIntraNodeAllGather`：节点内优化版本

Bootstrap 实现位于 `src/bootstrap.cc`。核心思路：Root Rank 创建 TCP 监听 Socket，其余 Rank 连接到 Root 形成星形拓扑，通过 Root 中转实现 AllGather。节点内通过共享文件描述符（Unix Domain Socket）建立直连，减少 Root 中转开销。

### 2.3 AllGather 信息交换

在 `initTransportsRank`（`src/init.cc:965`）中通过 Bootstrap 完成两轮 AllGather：

**第一轮 AllGather：**

```cpp
// src/init.cc:1036-1037
fillInfo(comm, comm->peerInfo + rank, comm->commHash);
bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo));
```

交换 `ncclPeerInfo`（`src/include/transport.h:43-65`），包含：rank、cudaDev、nvmlDev、busId、GPU UUID、hostHash、pidHash、计算能力、GIN 支持类型、RMA 可用性等。

**第二轮 AllGather：**

```cpp
// src/init.cc:1283 — 第二轮 AllGather：交换图计算结果
bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data));
```

交换图计算结果：各算法的 channel 数、带宽、跨 NIC 信息、拓扑排序 ranks。

### 2.4 拓扑发现与路径计算

```cpp
// src/init.cc:1141 — 构建系统拓扑图
ncclTopoGetSystem(comm, &comm->topo);

// src/init.cc:1143 — 计算 GPU↔NIC 路径
ncclTopoComputePaths(comm->topo, comm);

// src/init.cc:1145 — 删除不可达 GPU 和未使用 NIC
ncclTopoTrimSystem(comm->topo, comm);

// src/init.cc:1147 — 重新计算路径
ncclTopoComputePaths(comm->topo, comm);

// src/init.cc:1149 — 初始化搜索状态
ncclTopoSearchInit(comm->topo);
```

**拓扑节点类型**（`src/graph/topo.cc:32-34`）：GPU, PCI, NVS (NVSwitch), CPU, NIC, NET, GIN, RMA, DEV, CXB。

**链接类型**：LOC (Local), NVL (NVLink), C2C (Chip-to-Chip), PCI, SYS (跨 CPU Socket), NET。

**拓扑名类型**：LOC, NVL, NVB (NVLink Bridge), C2C, PIX (PCIe 同一 Switch), PXB (跨 PCIe Switch), P2C, PXN (PCIe 同一 CPU 通过 NVLink), PHB, SYS, NET。

拓扑发现主要通过 NVML API 查询 GPU 互联信息，构建以 GPU 为中心的图结构。具体实现在 `src/graph/topo.cc` 和 `src/graph/xml.cc`。

### 2.5 图计算：Ring/Tree/CollNet/NVLS

```cpp
// src/init.cc:1173-1178 — Ring
ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
ncclTopoCompute(comm->topo, ringGraph);

// src/init.cc:1181-1187 — Tree
treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
ncclTopoCompute(comm->topo, treeGraph);

// src/init.cc:1202-1207 — CollNet（若启用）
collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;
collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;
ncclTopoCompute(comm->topo, collNetChainGraph);
ncclTopoCompute(comm->topo, collNetDirectGraph);

// src/init.cc:1209-1217 — NVLS（若可用）
nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;
ncclTopoCompute(comm->topo, nvlsGraph);
```

支持的算法枚举（`src/init.cc:53-54`）：`Tree, Ring, CollNetDirect, CollNetChain, NVLS, NVLSTree, PAT`。

搜索算法实现在 `src/graph/search.cc`（通用搜索框架）、`src/graph/rings.cc`（Ring 搜索）、`src/graph/trees.cc`（Tree 搜索）。

### 2.6 Transport 连接建立

**传输类型：**

| 常量 | 传输 | 适用场景 |
|------|------|----------|
| `TRANSPORT_P2P` (0) | GPU 直接访问 | NVLink / PCIe P2P（同节点） |
| `TRANSPORT_SHM` (1) | 共享内存 | 同进程多 Rank / CE memcpy |
| `TRANSPORT_NET` (2) | 网络（RDMA） | 跨节点 InfiniBand / RoCE |
| `TRANSPORT_COLLNET` (3) | 网内计算 | SHARP / SHARPv2 |
| `TRANSPORT_PROFILER` (4) | Profiler 代理 | 轮询 Profiler 计数器 |

连接建立流程：

```cpp
// P2P 连接建立（含 SHM fallback）
ncclTransportP2pSetup(comm, NULL, 1);

// Ring 连接：逐一建立 send/recv connector
ncclTransportRingConnect(comm);
// Tree 连接：建立父子连接
ncclTransportTreeConnect(comm);
// CollNet：网内计算资源
ncclCollNetSetup(comm, parent, graphs);
// NVLS：Multicast 缓冲区
ncclNvlsBufferSetup(comm);

// Proxy 初始化
ncclProxyInit(comm);
```

在 `ncclTransportP2pSetup` 中（`src/transport.cc:20-42`），`selectTransport` 模板函数按优先级遍历 `ncclTransports[]`（P2P → SHM → NET → CollNet），调用 `canConnect` 判断两节点是否可用当前传输，然后调用对应的 `setup`。

**P2P Transport**（`src/transport/p2p.cc`）支持四种模式：

- `P2P_DIRECT`：GPU 直接 P2P 访问（NVLink/C2C）
- `P2P_INTERMEDIATE`：通过中间 GPU 中转
- `P2P_IPC`：CUDA IPC（同主机不同进程）
- `P2P_CUMEM`：cuMem API（CUDA 12.0+ 虚拟地址）

**NET Transport**（`src/transport/net.cc`）通过 Net Plugin API（v12）与 RDMA 网卡交互。Setup 阶段交换 `ncclNetHandle_t`（含 QP 信息），Connect 阶段修改 QP 状态到 RTS。

### 2.7 国产芯片适配要点

> 本节梳理了 NCCL 中 NV 闭源组件对应的抽象层，以及可能的替换策略，供国产芯片适配时参考。

| NV 闭源组件 | NCCL 中的抽象层 | 替换策略 |
|-------------|-----------------|----------|
| **NVLink** | `TRANSPORT_P2P`（`src/transport/p2p.cc`），通过 `cudaDeviceEnablePeerAccess` 建立 P2P | 若国产芯片提供类似 P2P Direct Access API，在 `p2pTransport.canConnect` 中判断拓扑后返回 true，在 `setup` 中调用对应 API 建立直连 |
| **NVSwitch** | `ncclTopoNode` 中的 `NVS` 类型，NVLS (NVLink SHARP) 算法 | 走 NET transport fallback；若自研交换机提供 SHARP 类网内计算能力，通过 `TRANSPORT_COLLNET` 接入 |
| **InfiniBand / ConnectX** | `TRANSPORT_NET`（`src/transport/net.cc` + `net_ib/`），Net Plugin API v12（`src/include/plugin/nccl_net.h`） | **关键替换点**：实现自定义 Net Plugin（实现 `ncclNet_v12_t` 接口），在 `devices()`、`listen()`、`connect()`、`regMr()`、`isend()`、`irecv()` 等接口中对接自研 RDMA 协议栈 |
| **GDRCopy** | `src/misc/gdrwrap.cc`，GPU 内存 CPU 端直接读写 | 若自研芯片支持 CPU 直接访问设备内存，实现等效的 GDR API 包装 |
| **Broadcom Tomahawk**（Scale-out 交换机） | `TRANSPORT_NET` | Tomahawk 是标准 Ethernet 交换机，走 RoCEv2 路径；只需确保 Net Plugin 的 RoCE 实现与 Tomahawk 兼容。若需网内计算，通过 `TRANSPORT_COLLNET` 和 CollNet Plugin 接入 |
| **GIN (GPU Initiated Network)** | `src/gin/` + GIN Plugin API（`src/include/plugin/nccl_gin.h`） | 可实现三种 Backend 之一：GPI（芯片支持 GPU 直接注入网络包）/ GDAKI（通过 DOCA GPUNetIO 等框架）/ Proxy（最通用，Host 侧代理线程） |

**最小适配路径：**

1. 实现 Net Plugin（最关键）：对接自研 RDMA 协议栈和交换机
2. 适配拓扑发现：提供等效 NVML 的 API 来查询芯片间互联拓扑
3. 实现 P2P transport：如果芯片支持片间直连
4. GIN 可先走 Proxy backend，后续升级到 GPI

NCCL 的插件架构设计良好，`ncclNet_t`、`ncclGin_t`、`ncclTuner_t` 均已通过版本化的虚函数表定义了清晰的 API 边界。

---

## 三、集合通信的执行路径

### 3.1 从 ncclAllReduce 到内核启动

以 AllReduce 为例的完整执行链：

```
ncclAllReduce → ncclEnqueueCheck → ncclGroupStart/End → ncclLaunchPrepare → ncclLaunchKernel → Device Kernel
```

**Step 1: API 入口**

```cpp
// src/collectives.cc:129-150
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff, count, datatype, op, 0, comm, stream, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
```

所有集合操作 API 都遵循相同模式：构造 `ncclInfo` 结构体，然后调用 `ncclEnqueueCheck`。

**Step 2: ncclEnqueueCheck**（`src/enqueue.cc`）核心逻辑：

- 参数校验（`src/misc/argcheck.cc`）
- 若在 Group 内（`ncclGroupDepth > 0`），将任务添加到 `ncclKernelPlanner` 的任务队列
- 若不在 Group 内，立即完成 Schedule → Launch 全流程

**Step 3: Schedule 阶段**（`ncclPrepareTasks`，`src/enqueue.cc`）：

- 选择算法（RING/TREE/COLLNET/NVLS/PAT）和协议（LL/LL128/SIMPLE）
- 划分数据块到各 Channel（`nChannels`）
- 为每个 Channel 创建 `ncclProxyOp` 描述数据搬运方式
- 构建 Work Batch 描述设备端执行单元

**Step 4: Launch 阶段**（`src/group.cc` 的 `ncclGroupEndInternal` 中）：

```cpp
// 将任务转换为 KernelPlan
ncclLaunchPrepare(comm);
// 启动 CUDA Kernel
ncclLaunchKernel(comm, plan);
// 清理
ncclLaunchFinish(comm);
```

### 3.2 任务分组与 Planner

**`ncclGroupStart` / `ncclGroupEnd`**（`src/group.cc`）实现对多个集合操作的批处理。

Group 深度由线程局部变量 `ncclGroupDepth` 控制（`src/group.cc:27`）：

- `ncclGroupStart()` 递增深度
- 在 Group 内的集合操作调用通过 `ncclAsyncLaunch` 将任务加入 `ncclAsyncJobs` 队列
- `ncclGroupEnd()` 递减深度，深度归零时执行全部累积任务

**`ncclKernelPlanner`**（`src/include/comm.h:429-502`）是任务调度核心：

- `peers[R]`：按 Peer 组织的 P2P 任务队列（send/recv/bcast）
- `collSorter`：按数据量排序的集合任务
- `wipPlan.channels[C]`：每个 Channel 的进行中 Plan 状态
- `planQueue`：已完成的 Kernel Plan 队列，等待 Launch

### 3.3 Kernel Launch 与 Device 侧调度

**Host 侧 Launch**：`ncclLaunchKernel`（`src/enqueue.cc`）将 `ncclKernelPlan` 中的 kernel 函数指针和参数启动为 CUDA Kernel。

**Device 侧入口**：

```cpp
// src/device/common.cu:23-25 — 统一内核入口
__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K ... args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}
```

`ncclKernelMain` 是设备侧的核心调度函数（位于设备头文件中）：

1. 读取 Work FIFO 中的 Work 项
2. 根据 Work 类型分发到对应的函数（AllReduce/ReduceScatter/AllGather/Broadcast/SendRecv/P2P 等）
3. 每种集合操作的实现使用 C++ 模板，根据算法（Ring/Tree）、协议（LL/LL128/SIMPLE）、数据类型和 Reduction 操作进行编译期特化

**协议选择**（`src/enqueue.cc` 和 `device/prims_*.h`）：

| 协议 | 适用数据量 | 特征 |
|------|-----------|------|
| LL | 极小消息 | 单线程处理，低延迟；8 字节对齐（`NCCL_LL_ALIGNMENT_PER_THREAD`） |
| LL128 | 小到中等 | 单 Warp 处理，128 字节粒度的流水线；480 字节 Warp 对齐 |
| SIMPLE | 大消息 | 多 CTA 并行，利用全部带宽；16 × Warp × 8 × 16 对齐 |

### 3.4 Transport 层数据路径

设备端的集合操作代码（如 `src/device/all_reduce.h`）在完成本地计算后，通过以下方式将数据发送到对端：

**P2P Transport**（NVLink/PCIe）：

- 通过 `ncclConnector` 中的 `connFifo`（生产者-消费者环形缓冲）传递控制信息
- 通过 `ptrExchange`（`src/include/comm.h:58`）交换数据缓冲区的 GPU 虚拟地址
- 设备端直接使用 `ld.global`/`st.global` 指令读写远程 GPU 内存（P2P Direct Access）
- 控制信息：`ncclSendMem.head` / `ncclRecvMem.tail` 指针跟踪数据进度

**NET Transport**（RDMA 网卡）：

- 设备端不直接操作网卡，而是将数据放入发送缓冲区
- Proxy 线程（`src/proxy.cc`）从发送缓冲区取出数据，通过 Net Plugin 的 `isend`/`irecv` 提交 RDMA 操作
- GDRCopy（若启用）让 Proxy 线程直接读写 GPU 内存

**GIN Transport**：

- 设备端通过 `ncclGinCall<ApiFn>` 分发到对应 Backend
- `Put` / `Get` 操作直接由 GPU Thread 发起，通过 GIN 后端（GPI/GDAKI/Proxy）将数据包注入网络

```cpp
// 设备端 Ring AllReduce 简化伪代码 (src/device/all_reduce.h)
// 每个 Channel 的 Ring 连接：prev ← 当前Rank → next
// NCCL_PROTO_SIMPLE 模式：
//   1. ReduceScatter 阶段：从 prev 接收 → 本地 reduce → 发送到 next
//   2. AllGather 阶段：从 prev 接收 → 转发到 next
//      当前 Rank 在完成自己的 chunk 后切换到直接拷贝模式
```

---

## 四、GIN 详解

### 4.1 GIN 概念与动机

**GIN (GPU Initiated Network)** 允许 GPU SM 上的线程直接发起网络通信操作（Put/Get/信号量），避免以下开销：

- Host 发起模式（传统）：GPU Kernel → CUDA Stream 完成 → Host 提交 RDMA → 网卡执行。需要 Kernel 结束才能启动网络操作
- GIN 模式：GPU Kernel 内的线程直接通过设备端 API 提交网络请求，实现 Kernel 计算与网络通信的重叠（overlap）

核心收益：**消除 Host-Device 同步开销，实现计算-通信 overlap 的细粒度流水线**。

### 4.2 三种 Backend

| Backend | 枚举值 | 原理 | 硬件依赖 |
|---------|--------|------|----------|
| **GPI**（GPU Packet Injection） | `NCCL_NET_DEVICE_GIN_GPI` | GPU 直接构造网络包并通过 PCIe 写入 NIC 发送队列 | 需要 NIC 硬件支持 GPU 直接访问发送队列 |
| **GDAKI**（GPU Direct Async Kernel Initiated） | `NCCL_NET_DEVICE_GIN_GDAKI` | 基于 NVIDIA DOCA GPUNetIO 框架，GPU 通过专用通道提交网络描述符 | ConnectX-6 Dx 及以上 + DOCA |
| **PROXY** | `NCCL_NET_DEVICE_GIN_PROXY` | GPU 写描述符到共享内存，Host 侧 Proxy 线程轮询并代理执行网络操作 | 无特殊硬件要求（最通用） |

Backend 选择机制（`src/include/nccl_device/gin/gin_device_common.h:40-49`）：通过编译期 `backendMask` 位掩码支持多 Backend 并存，运行时根据 `ncclGinCtx.backend` 在 `ncclGinCallImpl` 中分发。

### 4.3 主机侧 GIN 初始化

```cpp
// src/include/gin.h:13
ncclResult_t ncclGinInit(struct ncclComm* comm);
```

`ncclGinInit` → `ncclGinConnectOnce`（`src/gin/gin_host.cc:80`）的核心流程：

1. 加载 GIN Plugin（`ncclGin_t` 接口，`src/include/plugin/nccl_gin.h`）
2. 遍历所有网卡设备，创建 `ginComm_t` 通信上下文
3. 通过 Bootstrap 交换各 Rank 的 GIN 连接信息
4. 对所有 Peer 建立 GIN 连接
5. 启动 GIN Progress 线程（`ncclGinProgress`，`src/gin/gin_host.cc:43`）

GIN 共享资源存储在 `ncclSharedResources::ginState`（`src/include/comm.h:146`）：

```cpp
// ncclGinState 包含：
//   ncclGin_t* ncclGin      — GIN 插件接口
//   int ginCommCount         — GIN Comm 数量
//   ncclGinType_t ginType   — GPI / GDAKI / PROXY
//   GinStateDevComm* devComms — 设备侧上下文链表
//   thread progressThread    — Progress 线程
//   mutex/cv                 — 同步原语
```

### 4.4 设备侧 GIN API

设备侧 GIN API 定义在：

- `src/include/nccl_device/gin/gin_device_common.h` — 通用定义与分发机制
- `src/include/nccl_device/gin/gin_device_api.h` — 统一 API 声明
- `src/include/nccl_device/gin/gpi/gin_gpi.h` — GPI Backend 实现
- `src/include/nccl_device/gin/gdaki/gin_gdaki.h` — GDAKI Backend 实现
- `src/include/nccl_device/gin/proxy/gin_proxy.h` — Proxy Backend 实现

核心 API：

| API | 功能 |
|-----|------|
| `ncclGinApi_Put::call(ctx, coop, peer, dstWin, ...)` | GPU 发起 Put 操作（本地→远程），可附带信号量操作 |
| `ncclGinApi_Get::call(ctx, coop, peer, remoteWin, localWin, ...)` | GPU 发起 Get 操作（远程→本地） |
| `ncclGinApi_PutValue::call(ctx, peer, dstWin, srcData, ...)` | Put 一个标量值 |
| `ncclGinApi_Wait::call(ctx, request, ...)` | 等待异步操作完成 |
| `ncclGinApi_Flush::call(ctx, coop, ...)` | 刷新所有未完成操作 |
| `ncclGinApi_FlushAsync::call(ctx, peer, ...)` | 异步刷新 |

分发机制（`gin_device_common.h:177-209`）：

```cpp
// 编译期根据 backendMask 分发
template <template <ncclNetDeviceType> typename ApiFn, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCallImpl(unsigned beMask, ncclGinCtx ctx, Arg&&... arg) {
  switch (ctx.backend) {
    case NCCL_NET_DEVICE_GIN_PROXY: return ApiFn<PROXY>::call(...);
    case NCCL_NET_DEVICE_GIN_GDAKI: return ApiFn<GDAKI>::call(...);
    case NCCL_NET_DEVICE_GIN_GPI:   return ApiFn<GPI>::call(...);
  }
}
```

通过模板 + switch 实现零开销分发，编译期即可消除不活跃 Backend 的代码路径。

### 4.5 GIN Progress 线程

GIN Progress 线程（`src/gin/gin_host.cc:43-76`）是 GIN 架构中 Host 侧的核心组件：

```cpp
// src/gin/gin_host.cc:43-76
void* ncclGinProgress(struct ncclGinState* ginState) {
  while (1) {
    if (ginState->ginProgress == 1) {
      // 遍历所有 Device Comm，对每个 GIN Comm 调用进度函数
      for (auto* dc = ginState->devComms; dc; dc = dc->next) {
        for (int n = 0; n < ginState->ginCommCount; n++) {
          ginState->ncclGin->ginProgress(dc->ginCtx[n]);
        }
      }
    } else if (ginState->ginProgress == -1) {
      // 退出
      return NULL;
    }
    // 状态 == 0 时等待条件变量
  }
}
```

Progress 线程的作用因 Backend 而异：

- **Proxy Backend**：轮询 GPU 写入的共享内存描述符，调用 Net Plugin 的 `isend`/`irecv` 提交网络操作，完成后写回完成信号
- **GPI/GDAKI Backend**：轮询 NIC 的完成队列（CQ），更新设备侧可用的 Credit/Window 状态

Progress 线程的启停由 `ginState->ginProgress` 状态控制：`0`（暂停等待）、`1`（运行中）、`-1`（停止）。

### 4.6 Host 发起 vs GIN 发起的对比

| 维度 | Host 发起（传统 NCCL） | GIN 发起 |
|------|------------------------|----------|
| 通信触发者 | Host 侧 Proxy 线程通过 `ncclProxyProgress` | GPU Kernel 内直接调用 `ncclGinApi_Put` / `ncclGinApi_Get` |
| Kernel 与通信的 overlap | 粗粒度：Kernel 结束 → Proxy 处理 → 下一个 Kernel | 细粒度：同一 Kernel 内计算与通信交替执行 |
| 同步机制 | 通过 `ncclSendMem.head` / `ncclRecvMem.tail` 跟踪进度，Host 可读写 | 通过 `ncclGinSignalDescriptor` 和 `ncclGinCounter`，设备端直接操作 |
| 数据路径 | GPU → (GDR) → Host Buffer → Net Plugin → NIC | GPU → GIN Backend → NIC（尽可能短） |
| 代码位置 | `src/proxy.cc`、`src/transport/net.cc` | `src/gin/gin_host.cc` + `src/include/nccl_device/gin/` |
| 延迟 | 较高（Host-Device 同步 + 线程调度） | 较低（消除 Host 参与） |

**关键代码差异：**

- 传统模式：设备端操作完成后设置 `ncclSendMem.head`，Host Proxy 线程轮询该值并调用 `ncclNet->isend()`
- GIN 模式：设备端在 Kernel 内直接调用 `ncclGinCall<ncclGinApi_Put>(ctx, coop, peer, ...)` 提交网络请求，`ncclGinApi_Flush` 确保请求发出

---

## 五、组件交互全景

### 5.1 Channel 与 Connector

**Channel** 是 NCCL 并行通信的基本单位。每个 Channel 独立承载一个 Ring 或 Tree 连接子图。

- 一个 NCCL 操作的数据被切分为 `nChannels` 份，每个 Channel 独立处理一个数据块
- Channel 的数量由 Ring 和 Tree 图搜索决定，受 `minCTAs`/`maxCTAs` 配置限制（最大 `MAXCHANNELS`）
- 每个 Channel 对应一个 CUDA CTA（Cooperative Thread Array），实现 Kernel 级并行

**Connector**（`src/include/p2p.h`）是 Channel 内 Peer 间连接的具体描述：

- `ncclConnector.conn`：指向具体的 Transport 资源（P2P/SHM/NET）
- `ncclConnector.transportComm`：指向该传输的虚函数表
- `ncclConnector.connected`：连接状态标志
- 每个 Peer 对在一个 Channel 中有一个 send Connector 和一个 recv Connector

### 5.2 Proxy 机制

Proxy 是 NCCL 中负责 Host 侧数据搬运的线程机制（`src/proxy.cc`）。

**为什么需要 Proxy：**

- NET transport 不支持 GPU 直接发起 RDMA 操作（不使用 GIN 时）
- SHM transport 需要通过 CPU 在共享内存和 GPU 之间拷贝数据
- CollNet transport 需要在 Host 侧管理网内计算资源

**Proxy 工作模式：**

1. 设备端 Kernel 将数据传输描述写入 Proxy Queue（共享内存或设备内存）
2. Host 侧 Proxy 线程轮询 Queue，取出操作描述符
3. 根据 `ncclPattern_t`（Ring/Tree/CollNet 等）选择合适的执行路径
4. 调用 `transportComm->proxyProgress` 完成实际数据搬运

**关键数据结构：**

- `ncclProxyOp`（`src/include/proxy.h:73-100`）：单个 Proxy 操作描述符，含数据指针、大小、模式、Channel ID 等
- `ncclProxyArgs`：一批 Proxy 操作参数，传递给 Proxy 线程
- `ncclProxyState`：Proxy 全局状态，含线程管理、连接表

Proxy 机制在 `ncclTransportP2pSetup` 完成后初始化（`ncclProxyInit`），其生命周期与通信器绑定。

### 5.3 算法与协议的选择

**算法选择**（在 `ncclPrepareTasks` 中），NCCL 根据以下因素选择算法：

- 数据量大小（与 `threadThresholds[algo][proto]` 比较）
- Tuner 插件返回的延迟和带宽表（`comm->latencies` / `comm->bandwidths`）
- 硬件能力（CollNet 可用性、NVLS 可用性、PAT 可用性）
- 配置覆盖（用户可通过环境变量强制选择）

**协议选择（Protocol）：**

| 协议 | 选择条件 | 设备端实现 |
|------|----------|-----------|
| LL | 数据量 < `threadThresholds[algo][LL]` | `src/device/prims_ll.h` |
| LL128 | `LL <= 数据量 < LL128` | `src/device/prims_ll128.h` |
| SIMPLE | 数据量 >= `threadThresholds[algo][LL128]` | `src/device/prims_simple.h` |

**Cross-NIC 优化**：Ring 和 Tree 图支持跨 NIC 路由（`crossNic`），在 AllGather3 阶段确定是否需要跨 NIC 来充分利用多网卡带宽。

**PAT (Parallel Aggregated Trees) 算法**（`NCCL_ALGO_PAT`）：针对 AllReduce 在小数据量下的优化，利用 PCIe 交换机或 NVSwitch 的聚合能力。实现在 `src/graph/` 和 `src/transport/`。

### 5.4 值得深入研究的算法

以下算法在 NCCL 中属于核心技术，不需要在初学阶段深入细节，但需要知道它们的存在、用途和在代码中的位置：

| 算法/机制 | 位置 | 说明 |
|-----------|------|------|
| **Ring 搜索** | `src/graph/rings.cc` + `src/graph/search.cc` | 在拓扑图中搜索最优 Ring 排列。目标是最小化 Ring 中相邻节点间的路径总代价（综合考虑带宽和跳数） |
| **Tree 搜索** | `src/graph/trees.cc` + `src/graph/search.cc` | 搜索平衡二叉树或多叉树。Tree 算法延迟为 O(log N)，适合小消息；Ring 延迟为 O(N)，适合大消息 |
| **Double Binary Tree (DBT)** | 配置在 Tree 图中 | 通过两棵互补的二叉树实现全带宽利用，每棵树的叶子节点在另一棵树中为内部节点 |
| **拓扑发现** | `src/graph/topo.cc` + `src/graph/xml.cc` | 通过 NVML API 获取 GPU ↔ NVSwitch ↔ NIC 的 PCIe 拓扑，构建加权图 |
| **路径搜索** | `src/graph/paths.cc` | 计算节点间所有可能路径及其带宽，用于 Ring/Tree 搜索时的代价评估 |
| **NVLS (NVLink SHARP)** | `src/transport/nvls.cc` | 利用 NVSwitch 的 Multicast/Reduction 硬件能力实现网内计算 |
| **CollNet (SHARP)** | `src/transport/coll_net.cc` | 利用 InfiniBand 交换机的 SHARP 能力实现网内 Reduction |
| **Tuner** | `src/graph/tuning.cc` + `src/plugin/tuner.cc` | 运行时调优：为不同大小的操作自动选择最优算法/协议组合 |
| **PXN (PCIe + NVLink)** | 散布在 graph/ 和 transport/ | 利用 NVLink 跨 CPU Socket 的能力优化跨 NUMA 节点间通信 |

---

# 第二部分：DeepEP —— MoE 架构下的集合通信利器

> 前面我们用很大篇幅理解了 NCCL 本身。现在换一个角度：**如果让你基于 NCCL 去构建一个 MoE（Mixture of Experts，混合专家）模型的通信层，你会怎么做？**
>
> DeepSeek 开源的 **DeepEP** 就是对这个问题的回答。它是一个专门为 MoE 专家并行（Expert Parallelism）设计的通信库，因为解决了 MoE 训练/推理中最棘手的通信问题，而成为各方积极参考的对象（包括各家大模型团队和自研通信库）。
>
> 这一部分会假设你**还没读过 DeepEP 的代码**，从"MoE 到底需要什么样的通信"讲起，再一步步拆解 DeepEP 的架构，以及它和 NCCL 之间真实、精确的调用关系。

---

## 六、为什么 MoE 需要专门的通信库

### 6.1 先理解 MoE 的通信特征

在普通的稠密模型（Dense）训练里，AllReduce 是主角：每个 GPU 算一部分梯度，然后所有人做一次全局归约。数据流是**对称的、规整的**——每个 GPU 发出去的字节数和收进来的字节数都一样，而且每个 GPU 都要和"所有"其他 GPU 通信。这正是 NCCL 最擅长处理的场景。

但 MoE 完全不一样。MoE 模型里，每一层的计算会被路由到不同的"专家"（expert）上，而每个 token 只会被少数几个（比如 top-2）专家处理。于是出现了一种全新的通信模式——**all-to-all**（全对全）：

- **Dispatch（分发）阶段**：每个 GPU 手里的一批 token，需要按"路由结果"发送到**各自不同的目标专家所在的 GPU**。token A 去 GPU 1 的专家 3，token B 去 GPU 5 的专家 7……**每个 token 的去向都不同**。
- **Combine（回收）阶段**：专家算完之后，每个 token 的结果又要被送回它**原来的 GPU**，做加权求和（因为一个 token 可能被多个专家处理）。

这套通信有几个让 NCCL 的通用集合通信"使不上力"的特点：

1. **数据量严重不均衡**：每个 GPU 分到的 token 数、每个专家收到的 token 数，在每一步都是动态变化的、且极不均匀。可能 GPU 0 要发 1000 个 token，GPU 3 只发 10 个。
2. **通信模式是稀疏的、数据相关的**：谁发给谁、发多少，取决于这步的路由结果，无法预先固定。
3. **通信和计算需要深度重叠**：专家计算（GEMM）很贵，通信必须尽量和它 overlap，否则通信会拖垮整体吞吐。

NCCL 本身其实也有 `AllToAll` 集合操作，但它是"规整的" all-to-all（每个 rank 发给每个 rank 等量数据），无法直接表达 MoE 这种"不规则、动态、稀疏"的 all-to-all。于是，需要一个专门的库来做这件事——这就是 DeepEP 的用武之地。

### 6.2 DeepEP 解决的三个核心场景

DeepEP 的 `Buffer` 类（`deep_ep/buffers/legacy.py`）在文档里开门见山地列出了它支持的三类通信（翻译如下）：

1. **高吞吐节点内 all-to-all**（intranode，用 NVLink）——同机多卡之间，追求最大带宽；
2. **高吞吐节点间 all-to-all**（internode，用 RDMA + NVLink）——跨机场景，RDMA 走跨机、NVLink 走机内，两者组合；
3. **低延迟 all-to-all**（low-latency，用 RDMA）——专门为**推理**场景优化，追求最小延迟而非最大带宽。

> **一句话记忆**：训练场景看重"吞吐"（带宽要跑满），推理场景看重"延迟"（首 token 要快）。DeepEP 为两者分别做了高度定制化的内核。

---

## 七、DeepEP 到底怎么调用 NCCL（关键：读代码）

这是本部分最重要的一节。很多人的直觉是"DeepEP 会调用 NCCL 的 all-to-all 函数"，**但事实并非如此**。DeepEP 对 NCCL 的使用方式非常"底层"、也非常巧妙——它**几乎不用 NCCL 的集合通信 API，而是复用了 NCCL 提供的三个更底层的能力**：

1. **对称内存（Symmetric Memory）**：一块所有 GPU 都映射到**相同虚拟地址**的内存；
2. **GIN（GPU Initiated Network）**：让 GPU 内核里的线程**直接发起 RDMA 网络操作**，绕开 CPU；
3. **Team 与 LSA 窗口**：查询 GPU 间的 NVLink 拓扑关系，并拿到"对称内存里对方的那块地址"。

而真正跨节点的 RDMA，DeepEP 用的是 **NVSHMEM**（NVIDIA 的 PGAS 库），不是 NCCL。

下面逐条拆解，每条都标出对应的源码位置，请你打开源码对照着看。

### 7.1 第一步：创建 NCCL communicator（但为了拿"钩子"而非做集合通信）

DeepEP 在 C++ 侧有一个专门的 NCCL 封装文件：`csrc/kernels/backend/nccl.cu`。看它做的第一件事：

```cpp
// csrc/kernels/backend/nccl.cu:20-41
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

这里调用的 `ncclGetUniqueId` 和 `ncclCommInitRank`，正是我们前面"初始化流程"章节讲过的那两个 NCCL 入口。但注意：DeepEP 拿到这个 `ncclComm_t` 之后，**不是**用来调用 `ncclAllReduce` 之类的，而是作为后续一系列底层 API 的"把手（handle）"。

### 7.2 第二步：分配对称内存 + 注册窗口 + 拿 LSA 指针

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

### 7.3 第三步：内核里直接用 GIN 收发数据（而非调用集合通信）

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

### 7.4 对称内存的三种实现（`symmetric.hpp`）

前面提到 `symmetric.hpp` 里有三种对称内存，这里展开一下，它们对应不同的部署形态（源码在 `csrc/kernels/backend/symmetric.hpp`）：

| 实现类 | 内存构成 | 适用场景 |
|--------|----------|----------|
| `GPUSymmetricMemory` | 纯 GPU 显存（`ncclMemAlloc`） | 常规场景，数据全放显存 |
| `ElasticSymmetricMemory` | GPU 显存（前段）+ CPU 内存（后段，NUMA-local） | "弹性"场景，用 CPU 内存扩充容量，缓解显存不足 |
| `HybridElasticSymmetricMemory` | GPU 显存 + 每个 scaleup rank 各一段 CPU 内存 | 混合模式，每个 rank 的 CPU 段都映射进来 |

它们的共同点是：**构造出一段连续的虚拟地址空间，所有 rank 映射到相同地址**，从而兼容 `ncclCommWindowRegister`（窗口注册要求地址对齐且对称）。其中 `ElasticSymmetricMemory` 还会设置环境变量 `NCCL_ELASTIC_BUFFER_REGISTER=1` 来启用 NCCL 的弹性缓冲注册（见 `symmetric.hpp:312-316`）。

> 这一节体现了一个重要工程思想：**DeepEP 不只是"调用" NCCL，它甚至会反过来影响 NCCL 的行为**（通过环境变量、通过复用 NCCL 的对称内存分配器 `ncclMemAlloc`）。两者是深度耦合、互相配合的关系。

### 7.5 跨节点 RDMA：为什么用 NVSHMEM 而不是 NCCL

细心的你可能会问：跨节点的数据搬运（internode dispatch/combine），DeepEP 用的还是 NCCL 吗？

答案是：**不是**。跨节点的 RDMA 通信，DeepEP 用的是 **NVSHMEM**（NVIDIA 的 PGAS 编程模型库）。证据在 `csrc/kernels/backend/nvshmem.cu`：

```cpp
// csrc/kernels/backend/nvshmem.cu:49-76（init 节选）
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

而 Python 侧（`deep_ep/buffers/legacy.py:103-132`）在初始化时，会设置一堆 NVSHMEM 环境变量来启用 **IBGDA**（InfiniBand GPU Direct Async，GPU 直接访问网卡），例如：

```python
os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

**所以 DeepEP 的通信底座其实是"双引擎"：**

| 通信域 | 用什么 | 说明 |
|--------|--------|------|
| **机内（NVLink 域）** | NCCL 的对称内存 + LSA 指针 | 直接指针访问同机 GPU 显存 |
| **跨机（RDMA）** | NVSHMEM + IBGDA | GPU 直接发起 RDMA |
| **GPU 侧发起网络** | NCCL 的 GIN | 内核里直接 put/get/signal |
| **进程间控制面** | NCCL 的 communicator + bootstrap | 初始化时交换信息 |

> **一句话总结 DeepEP 与 NCCL 的关系**：DeepEP **复用**了 NCCL 的对称内存、GIN、窗口/LSA 这些"底层积木"，但**不用** NCCL 的 `AllToAll` 等集合通信接口，而是自己写了针对 MoE 不规则通信量身定制的 dispatch/combine 内核。跨节点部分则交给 NVSHMEM。

---

## 八、DeepEP 的独创之处：为什么各方都在参考它

理解了 DeepEP 对 NCCL 的调用方式之后，我们再站高一点，看看它到底"独创"在哪里、为什么值得学习。

### 8.1 专为"不规则 all-to-all"定制内核

这是最根本的一点。NCCL 的 `AllToAll` 是规整的（每个 rank 间等量），而 MoE 的 dispatch/combine 是**动态、稀疏、不均衡**的。DeepEP 为此专门设计了 dispatch 和 combine 内核（`deep_ep/include/deep_ep/impls/dispatch.cuh`、`combine.cuh`），核心思路：

- **运行时动态计算路由**：内核里先统计"每个 token 要去哪个 rank/哪个专家"，用原子操作在共享内存里累加计数（`dispatch.cuh:92-107` 的 `atomicAdd_block`），实时算出收发布局；
- **前缀和（prefix sum）定位**：用前缀和矩阵（`rank_prefix_matrix`、`channel_prefix_matrix`）精确定位每个 token 应该写到目标缓冲区的哪个位置，避免冲突；
- **warp 即 channel**：把每个 warp 当作一个通信 channel，对应一个 GIN 的 QP（队列对），实现细粒度的并行收发。

### 8.2 内核级通信（GIN）带来的极致重叠

传统集合通信里，通信和计算是"串行"的（算完再传、传完再算）。DeepEP 通过 GIN，让 GPU 内核在**计算过程中随时发起 RDMA**，实现了真正的**计算-通信细粒度重叠**。这对于 MoE 尤其重要，因为专家 GEMM 非常昂贵，任何一点通信停顿都会被放大。

### 8.3 低延迟模式的针对性设计（面向推理）

`low_latency_dispatch` / `low_latency_combine`（`deep_ep/buffers/legacy.py:553-713`）是 DeepEP 面向推理的招牌功能，几个关键设计：

- **纯 RDMA（IBGDA）**：所有 rank 都通过 RDMA 直接互访，不走 NVLink，把延迟压到最低；
- **固定缓冲 + 无 CPU 同步**：接收缓冲是预先固定分配的（`num_max_dispatch_tokens_per_rank` 决定上限），并且**不把"收到多少 token"同步回 CPU**（`legacy.py:600` 注释明确说明），这样才兼容 CUDA Graph；
- **接收 hook**：`return_recv_hook` 允许"只发起 RDMA 请求、不等待数据到达"，把等待推迟到真正需要数据的那一刻，进一步和计算重叠；
- **FP8 转换**：通信过程中直接做 FP8 量化（`use_fp8` 参数），在低延迟模式下顺带减少传输数据量。

### 8.4 丰富的工程细节

- **通信 handle 复用**：dispatch 返回的 `handle` 里缓存了前缀和矩阵、源索引等布局信息，下一次 combine 或同布局的 dispatch 可以直接复用，省去重复计算（`legacy.py:385-405`）；
- **专家对齐（expert_alignment）**：把每个专家收到的 token 数对齐到某个倍数，方便后续 GEMM 内核的访存对齐；
- **rank 动态屏蔽（mask）**：`low_latency_update_mask_buffer` 等接口支持在通信中动态屏蔽某个 rank，用于在线服务的故障隔离（`legacy.py:672-698`）；
- **性能统计内置**：`dispatch_wait_recv_cost_stats`、`combine_wait_recv_cost_stats` 等参数能在通信中直接统计"等待接收"的耗时，用于定位慢节点（`legacy.py:577-579`）。

### 8.5 建议你重点阅读的源码路径

如果读完这一部分你想深入，建议按下面的顺序读（由浅入深）：

1. **`deep_ep/buffers/legacy.py`**：Python 入口，先理解 `Buffer` 的 `dispatch` / `combine` / `low_latency_dispatch` / `low_latency_combine` 这几个 API 的签名和语义（这是理解一切的起点）；
2. **`csrc/kernels/backend/nccl.cu`**：看 DeepEP 如何初始化 NCCL、注册窗口、拿 LSA 指针（承接我们前面 NCCL 的知识）；
3. **`csrc/kernels/backend/symmetric.hpp`**：看对称内存的三种实现；
4. **`deep_ep/include/deep_ep/common/handle.cuh`**：看 `NCCLGin` 如何封装 GIN 的 put/get/signal（这是"GPU 直接通信"的核心）；
5. **`deep_ep/include/deep_ep/impls/dispatch.cuh` / `combine.cuh`**：看不规则 all-to-all 内核的具体实现；
6. **`csrc/kernels/backend/nvshmem.cu`**：看跨节点 RDMA 的 NVSHMEM 初始化。

---

## 参考资料

- NCCL 官方源码仓库：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)（版本 2.30.x）
- NCCL 官方文档：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- NCCL 开发者文档：[NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2206/nccl-developer-guide/)
- DeepEP 官方源码仓库：[github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- DeepEP 技术博客：DeepSeek 官方发布的 DeepEP 解读文章（含低延迟内核、混合通信等设计细节）

---

*声明：本文是作者学习 NCCL 与 DeepEP 源码过程中的阶段性笔记，内容基于 NCCL master 分支与 DeepEP 官方仓库源码整理。由于这是一份仍在持续完善的初稿，文中可能存在表述不准确或理解不完整之处，欢迎通过邮件联系指正。*
