---
title: "【未完成】NCCL 源码学习笔记"
permalink: /notes/nccl-source-study/
date: 2026-09-20
category: "集合通信"
tags: ["NCCL", "集合通信", "源码分析", "AllReduce", "Ring", "Tree", "GIN", "GPU互联", "CUDA"]
description: "NCCL 源码学习笔记（初稿）：从零开始讲清 NCCL 是什么、它如何支撑 PyTorch 分布式训练、底层打通了哪些硬件链路，以及初始化流程、执行路径与 GIN 机制。"
published: true
---

> 很多同学第一次接触 NCCL，是在写 PyTorch 分布式训练的 `all_reduce` 时。代码里只写了一行 `dist.all_reduce(tensor)`，训练却能跨 8 张卡、甚至跨几百台机器的几千张卡完成梯度同步——这背后到底发生了什么？
>
> 答案就藏在 NCCL（NVIDIA Collective Communications Library）里。它是一套专门为 GPU 间集合通信而生的底层库，由 NVIDIA 开源。它不关心你的模型是什么、损失函数是什么，它只做一件极致的事：**在尽可能短的时间里，把数据在参与训练的所有 GPU 之间正确地搬来搬去**。
>
> 本文是一份**面向初学者的、仍在持续完善的笔记**。我会尽量假设你"还没读过源码"，从"为什么需要 NCCL"讲起，逐步深入到源码结构、初始化流程、执行路径与 GIN 机制。
>
> MoE 专家并行下的通信库 DeepEP 已拆分为独立笔记，见[《DeepEP 源码学习笔记》](/notes/deepep-source-study/)——它大量依赖本文讲到的通信器、GIN 后端与对称内存。
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

- GPU 之间可能通过 **NVLink**（同机高速直连）、**PCIe**（同机较慢直连）、**RDMA 网卡（InfiniBand / RoCE）**（跨机）等多种物理链路相连。此外，其他公司自研的底层互联方案——例如 AMD 的 Infinity Fabric、国产网卡/IP 厂商自研的链路——也会在这一层接入，成为 xCCL（广义的CCL） 需要适配的"传输层"之一；
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

   > 上面提到了两个初学者容易混淆、却又贯穿 NCCL 全程的概念，这里单独解释一下：
   >
   > **Communicator（通信器，`ncclComm_t`）**：可以理解为"一次集合通信的参与范围 + 所有相关状态"的集合体。它记录了这次通信包含哪些 rank（进程/GPU）、它们之间已经建好的连接、拓扑信息、缓冲区等。只有**在同一个 communicator 里的 rank 才能互相通信**。它的创建（`ncclCommInitRank`）比较昂贵，所以框架会尽量复用，而不是每次通信都重新建一个——这正是 `ProcessGroupNCCL` 要"管理 communicator 的创建与复用"的原因。
   >
   > 可以这样类比：一个 communicator 就是**一个"通信域"**。这个域里所有 rank 的地位是平等的、彼此可达的，域外的 rank 则无法参与进来。但要注意，**并不是"一个通信域只能有一个通信器"**——恰恰相反，同一个进程（GPU）可以同时属于多个 communicator。例如：8 卡做数据并行需要"全体 8 卡的 communicator"，同时每 2 卡之间又要做张量并行需要"局部 2 卡的 communicator"，这两个 communicator 可以并存，互不干扰。更准确的说法是：**每一次集合通信操作，必须在"某一个确定的 communicator"上进行，所有参与者拿到的必须是"同一个 communicator"**——这是它们能正确协作的前提。
   >
   > 那么这个 communicator 里到底存了什么？NCCL 源码里 `ncclComm` 结构体（定义于 `src/include/comm.h:523`）就是它的真身，字段非常多，我们挑几组最有代表性的来看：

```cpp
// src/include/comm.h:523-797（节选，保留关键字段并加上注释）
struct ncclComm {
  // ---- 拓扑信息：这个通信域的基本盘 ----
  struct ncclTopoSystem* topo;              // 整个通信域的硬件拓扑图（GPU/NVLink/PCIe/NIC 等节点与链路）
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]; // 各算法（Ring/Tree/CollNet/NVLS）算出的通信图
  struct ncclPeerInfo* peerInfo;             // 每个 rank 的信息（GPU UUID、busId、hostHash 等）

  // ---- 网络与传输插件 ----
  ncclNet_t* ncclNet;                        // 网络插件接口（RDMA 网卡等）
  void* netContext;                          // 网络上下文
  void* ginContext;                          // GIN 上下文（详见下方首次解释）
  ncclCollNet_t* ncclCollNet;                // 网内计算（SHARP）插件接口

  // ---- 我在这个域里的身份 ----
  int rank;      // 我在 communicator 里的编号
  int nRanks;    // communicator 里一共有多少个 rank
  int cudaDev;   // 我对应的 CUDA 设备号
  int nvmlDev;   // 我对应的 NVML 设备号
  int compCap;   // 我这块 GPU 的计算能力（compute capability）
  int64_t busId; // 我的 PCI bus ID

  // ---- 节点/本地视角 ----
  int node;           // 我在哪个节点
  int nNodes;         // 一共多少个节点
  int localRank;      // 我在本节点内的编号
  int localRanks;     // 本节点内有多少个 rank

  // ---- 通道与缓冲区 ----
  struct ncclChannel channels[MAXCHANNELS];  // 并行通信通道（每个 channel 承载一条 Ring/Tree 连接）
  int nChannels;                             // 实际使用的 channel 数
  int buffSizes[NCCL_NUM_PROTOCOLS];         // 各协议（LL/LL128/Simple）的缓冲区大小

  // ---- 算法/协议选择用的调优数据 ----
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];   // 延迟表
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];  // 带宽表

  // ---- 设备侧与内核调度 ----
  struct ncclKernelComm* devComm;            // 设备侧（GPU 可见的）communicator
  void* workFifoBuf;                         // Work FIFO：CPU 向 GPU 内核下发任务的队列
  struct ncclKernelPlanner planner;          // 任务调度器

  // ---- 杂项 ----
  uint64_t commHash;                         // 该通信域的唯一标识哈希
  ncclConfig_t config;                       // 用户配置（超时、阻塞模式等）
  ...
};
```

   > **第一次出现 GIN，这里先解释一下**：GIN 全称 **GPU Initiated Network**，直译是"由 GPU 发起的网络通信"。它要解决的是传统通信里的一个痛点——**通信必须由 CPU（host）来"牵头"**：GPU 算完后，得先让 CPU 知道"算完了"，再由 CPU 去提交网络请求，中间有多次 CPU↔GPU 的同步和调度开销。
   >
   > GIN 的思路则反过来：**让 GPU 内核里的线程自己直接发起网络操作**（比如 `put` 把数据写到远程、`get` 从远程读、`signal` 发信号），全程不需要 CPU 参与。这样通信的触发点就从"CPU 调度"下沉到了"GPU 内核内部"，能够实现更细粒度的计算-通信重叠。
   >
   > 你可以在脑中先建立一个粗浅的印象：**传统路径是"CPU 下发通信任务"，GIN 路径是"GPU 自己发起通信"**。上面 `ncclComm` 里的 `ginContext` 字段，就是 GIN 所需的上下文句柄。至于 GIN 具体怎么工作、有哪些后端（GPI/GDAKI/Proxy），我们在第 0.4 节会先做个概念性收尾，到第四部分"GIN 详解"再完整展开。

   > 读这段代码时，你可以抓住一条主线：**`ncclComm` 就是"一次通信所需的全部上下文"的容器**——它既记录了成员与位置信息（`rank`/`nRanks`/`node`/`busId`），也记录了域内硬件连接（`topo`/`graphs`），还记录了数据通路（`channels`/`buffSizes`/`planner`/`workFifoBuf`）。后面讲初始化流程时，你会看到这个结构体是**一步步被填充起来**的——先建 bootstrap、再 AllGather 交换 peerInfo、再算拓扑和图、再建连接，最后才是一个"可用"的 communicator。
   >
   > 顺带一提，`ncclComm` 的首尾两个字段 `startMagic` 和 `endMagic` 是"魔数哨兵"（`comm.h:524` 与 `comm.h:796`），用于在内存越界或结构体被误用（比如传错了指针类型）时快速发现问题。
   >
   > 它的原理正是你想的那样——**"哨兵值被改动"意味着有人越界写了内存**。具体机制是这样的：
   >
   > - **哨兵值**：`NCCL_MAGIC` 定义在 `comm.h:504`，值是 `0x0280028002800280`。这个数字有个小彩蛋——源码注释写的是 "Nickel atomic number is 28"（镍 Ni 的原子序数是 28）。注意 `0x28` 是十六进制、等于十进制的 40，但这里玩的是"NVIDIA 的 Ni"梗：8 字节里每个字节都取 `0x28`，拼成 `0x0280028002800280` 这样一个规律性极强的、现实中几乎不会自然出现的魔数值。
   > - **写入**：communicator 初始化完成的那一刻，NCCL 才把这两个字段设为 `NCCL_MAGIC`（`init.cc:2510`，注释原话就是 "Used to detect comm corruption"——"用于检测 comm 是否被破坏"）。而在此之前，它们被初始化为 `0`（`init.cc:210`）。
   > - **检查**：每次集合通信 API 的入口，都会先做一次校验（`argcheck.cc:40`）：`if (comm->startMagic != NCCL_MAGIC || comm->endMagic != NCCL_MAGIC)`，只要首尾任意一个哨兵不对，就打印 `"Error: corrupted comm object detected"` 并返回 `ncclInvalidArgument`，拒绝继续执行。
   >
   > 于是形成了三重防线：**（1）** 如果某段代码发生**越界写**（比如写穿了 `ncclComm` 相邻的内存），最外层的 `startMagic` 或 `endMagic` 极可能被顺带破坏，下次调用就会被拦住；**（2）** 如果**传错了指针**（把别的对象的地址当成 `ncclComm_t` 传进来），那个地址处根本不会有 `NCCL_MAGIC`，同样会被识别出来；**（3）** 如果 communicator **尚未初始化完成**或**已经被释放/作废**，哨兵要么是 0、要么已被清掉，也能挡住误用。
   >
   > 需要注意：这是一种**低成本、非 100% 可靠**的防御手段——它只能"大概率"发现越界/误用，无法精确定位越界发生在哪里、也防不住"恰好把别的合法数据写进哨兵位置"的极端情况。要精确定位内存错误，还得靠 AddressSanitizer（ASan）、CUDA 的 compute-sanitizer 这类工具。但作为每次调用都执行的一道轻量闸门，它已经足够在"错误刚发生不久"就暴露出来，价值很大。
   >
   > **CUDA Stream（流，`cudaStream_t`）**：是 GPU 上任务执行的"队列"。同一流里的操作按顺序执行，不同流里的操作则可以并行。把通信操作放到哪个流上，决定了它相对计算操作是"串行等待"还是"并行重叠"。`ncclAllReduce` 的最后一个参数就是 stream，NCCL 会把通信内核"入队"到这个流上，从而与计算在正确的先后顺序下协调执行。理解了这两个概念，后文看源码时对 `comm` 和 `stream` 这两个反复出现的参数就不会再陌生了。



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

### 0.4 通算融合、通信算子，与 GIN 的关系

前面我们理解的，是**一次"单纯"的通信**：算完之后，专门发起一次集合通信，等它完成，再继续算。但真实的大模型训练里，通信往往是大头开销——如果每次都"算完停下来等通信"，GPU 就浪费了。于是诞生了两个很流行的概念，这里作为第 0 节的收尾把它们理清楚。

#### （1）通算融合：让"算"和"通"在时间上重叠

**通算融合（计算-通信融合 / 重叠）** 的核心思想很简单：**不要"算完→通信→再算"这样串行，而是让计算和通信在时间上并行起来**。

它大致分成两个层次：

| 层次 | 做法 | 粒度 |
|------|------|------|
| **kernel 间重叠** | 用多条 CUDA stream，让"计算 kernel"和"通信 kernel"跑在不同的流上并行执行 | 粗粒度（整个 kernel 级别） |
| **kernel 内重叠** | 在一个 kernel 内部，把数据切块，一边算某几块、一边把已算好的块发出去 | 细粒度（块级别） |

经典的例子是：反向传播里，"当前层的梯度计算"和"上一层的梯度 AllReduce"其实是可以重叠的——当你在算第 2 层的梯度时，第 3 层的梯度已经可以开始做通信了。这样通信耗时就被计算时间覆盖。

#### （2）通信算子：把"通信"也当成一个算子来调度

"通信算子（communication operator / comm kernel）"是一个视角上的转变。

在传统认知里，通信是"独立于计算之外的一次系统调用"。但如果你把通信也看成**一个 GPU kernel**——它有输入张量、输出张量、有延迟和带宽成本、可以被放进一个有向无环图（DAG）里和其他算子一起调度——那它就和 `matmul`、`softmax` 一样，是一个**算子（operator）**了。

这样做的好处：编译器和调度器可以用统一的框架来做优化，比如：

- **算子融合**：把"通信算子"和紧邻的"计算算子"合并，减少中间数据的显存读写；
- **依赖分析**：分析哪个通信必须等哪个计算、哪个可以提前，从而自动安排重叠；
- **图级调度**：像 Megatron 里"把 AllReduce 拆成 ReduceScatter + AllGather"这类优化，本质上就是在"算子图"层面重排通信算子。

#### （3）这和 GIN 是什么关系？算不算 GIN 的内容？

**结论先说：通算融合是一个"概念/目标"，GIN 是达成"细粒度通算融合"的一种底层"手段"，两者不是一回事。**

- 前面 0.1～0.3 讲的"host 下发任务到 device"（CPU 调用 `ncclAllReduce` → NCCL 把通信 kernel 入队），是**传统、也是最经典**的路径。它**已经能实现"kernel 间"层面的通算融合**——靠多条 stream 把计算 kernel 和通信 kernel 并行起来。但它的极限是"kernel 粒度"：一个通信 kernel 必须等上一个计算 kernel 整体结束（或至少整块结束）才能开始。

- **GIN（GPU Initiated Network）** 则更进一步，允许 **GPU 内核里的线程直接发起网络操作**（`put`/`get`/`signal`），不需要等 CPU、不需要 kernel 结束。这才能实现**"kernel 内"层面的细粒度融合**——计算和通信在同一个 kernel 里交替进行、逐块重叠。

所以可以这样理解它们的层级关系：

```
通算融合（目标）
├── kernel 间重叠（较粗）  ← 传统 host 下发 + 多流 就能实现
└── kernel 内重叠（更细）  ← 需要 GIN 这类"设备发起通信"的能力
```

> **一句话**：通算融合是"目的"，通信算子是"把它工程化的视角"，GIN 是"把重叠做到极致（kernel 内）的关键技术"。你现在看到的主流训练框架（Megatron、DeepSpeed 等）里各类"通算融合"优化，很多底层仍走的是传统 host 下发 + 多流重叠；而像 DeepEP 这种追求极致低延迟的库，才进一步用上了 GIN——它在通信库中的具体用法，见[《DeepEP 源码学习笔记》](/notes/deepep-source-study/)。

---

## 一、源码目录架构

在深入具体机制之前，先建立对 NCCL 源码整体组织方式的认知。NCCL 的源码结构按照功能域清晰分层：`src/` 下的每个一级 `.cc` 文件通常对应一个核心子系统，其头文件放在 `src/include/` 中；设备侧代码统一在 `src/device/`，传输层实现在 `src/transport/`，插件适配层在 `src/plugin/`。

> **阅读提示**：本章罗列的目录和各目录里的结构体、文件，只是作为一个**大致的框架介绍**，目的是让你先有一张"地图"，知道什么在哪儿。如果看到某个概念（比如 `ncclComm`、Channel、GIN、某个 transport）一时没看懂，**不必纠结，可以先跳过**——它们都会在后面的正式流程中（初始化、执行路径等）随着讲解逐步展开，到那时你对这些结构体和数据结构的理解会自然加深。现在先建立"哪个词大概在哪"的印象即可。

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
| `bindings/` | 多语言绑定（详见 1.2 节） |
| `nccl.h.in` | 公共 API 头文件模板 → `nccl.h` |
| `CMakeLists.txt` | 构建系统入口 |

### 1.2 src 之外：仓库顶层的其他目录

在聚焦 `src/` 之前，先把 `src/` **同级**的其他目录过一遍，这样你对整个仓库才有一个完整的版图。这些目录大多不参与 `libnccl` 核心库的构建，但各有各的职责：

**`bindings/` —— 对外的语言桥接层**

目的是让非 C/C++ 的开发者也能调用 NCCL：

| 子目录 | 作用 |
|--------|------|
| `nccl4py/` | **官方新一代 Python 绑定（核心）**。内部用 Cython 的 `.pyx`/`.pxd` 文件把 NCCL 的 C 接口封装成 Python 可调用的模块，再往上提供纯 Python 的 `core/`、`ep/` 等封装，其中 `ep` 对应端到端集合通信场景。这是当前官方最活跃、最推荐的 Python 接入方式，用来替代老的 nccl-python |
| `ir/` | 存放 `nccl_device_wrapper.h` 等头文件（含 `clang_compat/`），作用是对设备端（device-side）接口做 Clang 编译兼容与包装，让主机端能顺利把设备端内核代码编译出来——本质是**编译器适配层**，不是给用户直接调用的 API |

**`contrib/` —— 社区贡献与实验性项目**

这些项目默认**不会被编进 `libnccl` 核心库**，每个子目录都有自己的 `Makefile` 和 `README`，需要单独编译使用：

| 子目录 | 内容 |
|--------|------|
| `custom_algos/` | 自定义算法示例：教你用 NCCL 设备端 API 手写 allreduce、alltoall 等内核，附性能图，适合想深入设备端编程的开发者 |
| `nccl_ep/` | 低延迟通信 endpoint 架构的实验，含独立的设备内核、测试和 benchmark |
| `nccl_ubx/` | 在 nccl4py 之上再包一层更易用的 Python API，带 C++ 内核、bench 和测试 |
| `nccl_m2n/` | 跨 GPU 组（cross-group）数据搬运的独立实验库，核心是 **reshard**（数据重排）：把全局张量在两个不相交的 GPU 进程组之间重新分布，基于 NCCL 的 user-window API（`ncclWindow_t` + `ncclMemAlloc`），实现零拷贝、单边（one-sided）传输 |
| `nccl_checkpoint/` | 检查点 KV-store 客户端（`kv_store_client.cc`），用于保存和恢复 NCCL 运行状态 |

> 可以把 `contrib/` 整体理解为"与 NCCL 相关的周边实验项目"——不参与核心库构建，但展示了 NCCL 能力的多种扩展方向。
>
> ⚠️ 一个容易踩的坑：网上部分资料把上面"跨 GPU 数据重排"的项目写成 `nccl_xfer/`，但本仓库里**实际目录名是 `nccl_m2n/`**（M2N 即 many-to-n / 跨组搬运）。功能描述没错，目录名以源码为准。

**`plugins/` —— 插件化扩展点（2.17 起正式引入）**

存放每种插件的接口头文件和参考示例，每个子目录都包含一个可编译的最小插件 `example/` 以及一组版本化的 `.h` 接口：

| 子目录 | 插件类型 |
|--------|----------|
| `net/` | 网络插件：自定义 RDMA 或其他网络后端，接口从 `net_v2` 演进到 v12 |
| `tuner/` | 调优插件：替换 NCCL 默认的算法和带宽选择策略 |
| `profiler/` | 性能剖析插件：官方提供 inspector 实现，可把性能数据导出 JSON 或推送 Prometheus/Grafana 监控 |
| `env/` | 环境变量插件：自定义环境变量的读取逻辑 |
| `mixed/` | 混合插件示例：展示多种插件能力如何协同 |
| `gin/` | GIN（GPU Initiated Networking）插件：对应 GPU 主动发起网络通信的场景 |

这些插件的共同特点是：**用户不需要修改 NCCL 核心代码**，只要编译自己的动态库并在运行时加载，就能扩展 NCCL 的行为。

**`docs/` —— 完整文档体系**

| 子目录 | 内容 |
|--------|------|
| `userguide/` | Sphinx 用户手册源码：API 参考、环境变量说明、使用教程、故障排查指南，是查阅官方文档的源头 |
| `examples/` | 从基础到高级的可运行示例，编号 `01`~`07`：通信器创建、点对点、集合通信、缓冲区注册、对称内存、设备端 API、内核融合，每个示例都提供 C 和 Python 两套版本 |
| `dev_guide/` | 开发者规范（主要是代码风格约定） |
| `contrib/` | GIN 等技术文档，补充核心手册之外的设计细节 |

**`makefiles/` 与 `cmake/` —— 两套并行的构建体系**

- `makefiles/`：NCCL 手写 Makefile 体系的核心片段，`src/` 里的 Makefile 会通过 `include` 引入。`common.mk`（通用编译规则）、`version.mk`（版本号）、`formatting.mk`（代码格式化）、`examples.mk`（编译示例）。
- `cmake/`：`NCCLConfig.cmake.in`——下游项目用 `find_package(NCCL)` 查找 NCCL 时，CMake 基于这个模板生成配置文件。
- 根目录同时存在 `CMakeLists.txt` 和 `Makefile`，说明 NCCL 维护着**老的 Makefile 体系**和**新的 CMake 体系**两套并行构建入口。

**`pkg/` —— 打包发行**

`debian/`、`redhat/`、`srctxz/`、`txz/` 分别用来生成 `.deb`、`.rpm`、`.tar.xz` 源码包或二进制包，`python_wheel/` 用来生成可 `pip install` 的 Python wheel。这些目录里包含 `*.spec.in`、`control.in`、`*.install.in` 等打包模板，配合发行版打包工具使用。

**`maint/`、`.github/` 与根目录文件**

- `maint/`：维护和开发工具目录，主要是 `run-clang-format.sh`（批量格式化代码，配合根目录 `.clang-format` 使用）。
- `.github/`：GitHub 相关的 PR 模板和 Issue 模板，属仓库协作层面配置，不参与构建。
- 根目录散落文件：`README.md`、`LICENSE.txt`、`ThirdPartyNotices.txt`、`CONTRIBUTING.md` 提供说明与许可证信息，`Makefile`/`CMakeLists.txt` 是构建入口，`.clang-format` 是代码风格配置。

### 1.3 src/ 核心源码详解

| 文件 | 执行位置 | 功能 |
|------|----------|------|
| `init.cc`（146KB） | Host | 初始化入口：`ncclCommInitRank`、`ncclCommInitRankDev`、`initTransportsRank`、通信器分配/释放 |
| `enqueue.cc`（144KB） | Host | 集合操作入队：`ncclEnqueueCheck`、Kernel Plan 构建、Work Batch 组装、算法/协议选择 |
| `group.cc`（34KB） | Host | Group 操作：`ncclGroupStart/End`、异步任务管理、Preconnect |
| `collectives.cc`（16KB） | Host | 所有集合操作 API 入口：AllReduce、AllGather、ReduceScatter、Broadcast、SendRecv、RMA 等 |
| `proxy.cc`（84KB） | Host | Proxy 线程机制：当 GPU 无法直接驱动网卡/CPU 拷贝时，由 Host 侧常驻线程代劳完成数据搬运（详见 5.3 节） |
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

### 1.4 include/ 头文件组织

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

### 1.5 关键数据结构总览

NCCL 的源码虽然庞大，但整个框架其实是**围绕少数几个核心结构体**搭建起来的。这一节就介绍其中最基础、也最能体现 NCCL 架构设计的三个结构体——`ncclComm`（通信器）、`ncclChannel`（通道）、`ncclTransport`（传输层）。你可以把它们理解成 NCCL 的"骨架"：理解了这三者各自代表什么、彼此什么关系，后面读初始化流程、执行路径时，就能时刻知道"现在这段代码在操作哪一层"。这里先建立直觉，字段细节不必强记，后面的正式流程会反复遇见它们。

#### 1. 通信器 `ncclComm`：一个"通信域"的完整描述

**`ncclComm`**（定义于 `src/include/comm.h:523-797`）是 NCCL 最重要的数据结构，它就是**通信器（communicator）**。它对应的抽象概念是**通信域（communication domain）**——一组彼此建立了连接、可以互相做集合通信的 rank（进程/GPU）。

先厘清几个容易混的概念：

- **通信域**：一个通信器所定义的"可以互相通信的范围"。域内的 rank 地位平等、彼此可达；域外的 rank 无法参与进来。
- **通信组（process group）**：这是上层框架（PyTorch 的 `ProcessGroup`）的叫法。实际上，**一个 PyTorch 通信组的底层，就对应一个 NCCL communicator**——"通信组"和"通信域"是同一件事在不同层次的两个名字。

一个进程（GPU）**可以同时属于多个通信域**，也就是持有多个 `ncclComm`。为什么需要这样？因为这正好对应大模型训练里**不同并行维度需要不同的通信范围**：

| 并行维度 | 通信范围 | 对应通信器 | 典型操作 |
|----------|----------|-----------|----------|
| **DP（数据并行）** | 全体 rank | 一个大 communicator | 梯度 AllReduce（跨所有卡） |
| **TP（张量并行）** | 节点内几个 rank | 一个小 communicator | AllReduce / AllGather（延迟敏感，希望走 NVLink） |
| **PP（流水线并行）** | 相邻 stage（2 个 rank） | 更小的 communicator | 点对点 Send/Recv |

于是 TP/PP/DP 会各自使用**不同的通信器、不同的通信域**，互不干扰，各自维护自己的连接和拓扑。

更进一步，NCCL 还支持**子通信器**：通过 `ncclCommSplit` 从一个父通信器"分裂"出子通信器。子通信器拥有独立的通信域，但可以通过 `ncclSharedResources* sharedRes` 字段**复用父通信器的部分资源**（这正是字段注释里"跨 Split 子通信器共享"的含义），避免重复初始化。

**而在一个 `ncclComm` 内部，又存在"Team"的概念**，这里要特别区分清楚：

> **communicator 是"域"的边界**（不同 communicator 之间不能通信），**Team 是"域内"的子划分**。同一个 communicator 里，NCCL 会按硬件拓扑自动划分出逻辑子域（team），用来决定某对 rank 之间该走哪条物理链路。最常见的三种 Team：
>
> - `ncclTeamWorld`：全体 rank；
> - `ncclTeamLsa`（Local NVLink Accessible）：同一 NVLink 域内、通过 NVLink 直接可达的一组 GPU（通常就是同一节点内的卡）；
> - `ncclTeamRail`：跨节点的"同一位置" GPU 组（例如每个节点的第 0 号 GPU 组成一个 rail，共享同一条网络路径）。
>
> 一句话记忆：**"通信域 vs 通信组"是跨层次的同义词；"communicator vs Team"则是"域边界 vs 域内拓扑分组"的关系。**

回到字段本身，`ncclComm` 里装的都是"这一个通信域"的全部上下文：

- **拓扑信息**：`ncclTopoSystem* topo`、`rank`、`nRanks`、`cudaDev`、`busId`——描述本域有哪些 GPU、它们如何连接、本 rank 的编号与位置；
- **Channel 数组**：`ncclChannel channels[MAXCHANNELS]`——并行通信通道（见下面第 2 点）；
- **Graph 数组**：`ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]`——分别为 Ring/Tree/CollNet/NVLS 等算法算出的通信图；
- **网络**：`ncclNet_t* ncclNet`、`netContext`、`ginContext`——跨节点的网络插件与 GIN 上下文；
- **内核调度**：`ncclKernelPlanner planner`、workFifo 机制——把任务调度给 GPU 内核；
- **共享资源**：`ncclSharedResources* sharedRes`——跨 Split 子通信器共享的资源；
- **配置**：`ncclConfig_t config`——用户配置（超时、阻塞模式等）。

#### 2. 通道 `ncclChannel`：并行通信的最小单位

**`ncclChannel`**（定义于 `src/include/comm.h:150-172`）代表**一条独立的通信通道**，可以理解为"通信域里的一条并行管道"。

为什么一个通信器里要有**多个** Channel？答案是**并行**。一个集合操作的数据会被切成 `nChannels` 份，每个 Channel 独立负责其中一份——每个 Channel 对应一条 Ring 或 Tree 连接，并由独立的 CUDA CTA 并行执行（**CTA 即 Cooperative Thread Array，也就是我们常说的 block/线程块**——它是 CUDA 里线程调度的基本单元，一个 CTA 内的线程在同一个 SM 上协同工作）。Channel 越多，能同时搬运的数据块就越多，带宽利用就越充分（当然也受硬件限制，最大 `MAXCHANNELS`）。

> **补充：Channel 是"逻辑通道"，不是物理链路绑定。** 一个 Channel 本身只包含"一条逻辑 Ring/Tree 连接 + 一份数据切片 + 一个 CUDA CTA"，它并不绑定某条具体物理链路——数据实际走 NVLink 还是 RDMA、走哪块网卡，是由每个连接（connector）的传输层和拓扑决定的。
>
> 那"切成多个 Channel 为什么就能并行、更快"？有两层原因：
> 1. **GPU 侧并行**：一个 Channel 对应一个 CTA，只能占一个 SM；多个 Channel 就是多个 CTA 同时占多个 SM，把 GPU 的并行计算单元用满；
> 2. **带宽叠加（更关键）**：单条 Ring 的吞吐受限于"环上最慢的一条链路"（比如单条 NVLink 端口、或单块网卡），而 GPU 总带宽远高于单条链路。用多条 Channel（多条并行的 Ring），每条走**不同的物理链路**（不同 NVLink 端口、不同网卡），带宽就能叠加，最终占满 GPU 总带宽。NCCL 源码里 `ncclTopoDupChannels`（`graph/search.cc:1036`）干的就是"把一条 Ring 复制成多条并行跑"。
>
> 至于"是不是对应不同的数据平面"——直觉接近，但不完全等同。Channel 的首要目的是 GPU 侧并行；只有跨节点多网卡时，多个 Channel 分散到**不同网卡（不同 rail）**上传输，才真正有"多数据平面"的意味。NCCL 里更贴近"物理数据平面/路径"的概念其实是 **rail**（`ncclTeamRail`，见本节第 1 点 里 Team 的说明）：每个 rail 对应一条网络路径（一块网卡），多个 Channel 可分布到多个 rail 上叠加多网卡带宽。

它的核心字段：

- `ncclRing ring` / `ncclTree tree`：这个 Channel 承载的 Ring / Tree 拓扑连接信息（即"我的上一跳是谁、下一跳是谁"）；
- `ncclChannelPeer** peers`：指向各 Peer 的连接器（connector）数组，描述"我连到每个 peer 的具体连接"；
- `workFifoProduced`：Work FIFO 的生产指针，用于 CPU 向 GPU 内核下发任务；
- 此外还有 `collnetChain` / `collnetDirect` / `ncclNvls nvls` 等字段，分别对应 CollNet、NVLS 这类"网内计算"算法在该 Channel 上的连接。

#### 3. 传输层 `ncclTransport`：异构硬件的统一抽象

**`ncclTransport`**（定义于 `src/include/transport.h:136-142`）是**传输层（transport）的虚函数表**，NCCL 用它把 NVLink、PCIe、RDMA 网卡、共享内存这些**异构物理链路，统一成同一套接口**。

一个 `ncclTransport` 实例封装了"某一类链路"的全部操作：

- `canConnect`：判断两个节点之间能否用这类传输连通；
- `send` / `recv`（两个 `ncclTransportComm` 结构体）：分别管发送侧和接收侧，内含 `setup`（建连接、交换 handle）、`connect`（完成连接）、`free`（释放连接）等生命周期函数；
- `proxySetup` / `proxyConnect` / `proxyProgress`：**Proxy 线程侧**的操作（正好对应 5.3 节讲的 Proxy 机制——当 GPU 不能直接驱动该链路时，由 Host 侧线程代劳）。

NCCL 提供了四种实例，分别对应四类链路：

| 实例 | 底层链路 | 场景 |
|------|----------|------|
| `p2pTransport` | NVLink / PCIe / C2C 直连 | 机内 GPU 直接读写对方显存 |
| `shmTransport` | 共享内存 + CPU 拷贝 | 同进程多卡、或无法 P2P 时的兜底 |
| `netTransport` | RDMA 网卡（InfiniBand / RoCE） | 跨节点 |
| `collNetTransport` | 支持 SHARP 的交换机 | 跨节点大消息的网内计算 |

> **这四种实例在代码中的位置**：它们都是**全局的 `struct ncclTransport` 变量**，各自定义在对应的 transport 实现文件里——`p2pTransport` 在 `src/transport/p2p.cc:1466`、`shmTransport` 在 `src/transport/shm.cc:467`、`netTransport` 在 `src/transport/net.cc:2085`、`collNetTransport` 在 `src/transport/coll_net.cc:1867`（各自的 `name` 字段分别是 `"P2P"`/`"SHM"`/`"NET"`/`"COL"`），声明集中在 `src/include/transport.h:28-32`。
>
> 真正把它们"串起来"的是一个数组 `ncclTransports[]`（`src/transport.cc:15-18`），按优先级顺序注册：
>
> ```c
> struct ncclTransport* ncclTransports[NTRANSPORTS + 1] = {
>   &p2pTransport, &shmTransport, &netTransport, &collNetTransport,
>   &profilerTransport // 仅用于创建轮询 profiler 计数器的 proxy op，并非真正的传输
> };
> ```
>
> 而"选哪一个"的逻辑在 `selectTransport`（`src/transport.cc:20-34`）：它遍历 `ncclTransports[]`，对每个 transport 调用 `canConnect` 判断两节点能否用这类链路，能用就选中，并把它填进 `connector->transportComm`。因为数组顺序是 P2P → SHM → NET → CollNet，所以这本质是一条**"优先 P2P，不行退 SHM，再不行 NET"的兜底链**——这正好和下面第 4 点 讲的 connector 衔接起来：`selectTransport` 就是"给每个 connector 挑选 transport"的地方。

#### 4. 连接器 `ncclConnector`：Channel 与 Transport 之间的桥梁

前面说 Channel 是"逻辑通道"、Transport 是"物理链路的抽象"，那这两者是怎么衔接起来的？答案就是 **connector（连接器）**。

**`ncclConnector`**（定义于 `src/include/device.h:165-173`）描述的是"**本 rank 在某个 Channel 里，与某个具体 peer 之间的一条连接**"。它把 Channel 里那句逻辑的"我的下一跳是 rank X"，落实成"具体用哪套 transport、哪条物理链路、哪块连接资源"。

核心字段：

- `transportComm`：指向传输层的虚函数表（`ncclTransportComm*`，正是 `ncclTransport` 里的 `send`/`recv`）——决定这条连接"走 P2P 还是 NET"；
- `transportResources`：传输层的具体连接资源（比如 RDMA 的 QP、P2P 的映射地址等）；
- `conn`：连接信息（`ncclConnInfo`，含具体传输类型、地址等）；
- `connected`：连接是否已建立的状态标志；
- `proxyConn`：对应的 Proxy 连接器（呼应 5.3 节的 Proxy 机制）。

在 `ncclChannelPeer` 里，每个 peer 都有 `send[NCCL_MAX_CONNS]` 和 `recv[NCCL_MAX_CONNS]` 两组 connector（`device.h:233-234`）——即对每个 peer，发送和接收各有一套连接描述。

> **那"一个 Channel 里，本 rank 与不同 rank 之间的连接是怎么划分的？"** 答案藏在 `ncclChannel.peers` 这个字段里：它是一个**按 rank 编号索引的指针数组**（长度 `nRanks`），`peers[r]` 专门存放"本 rank 在这个 channel 里，与 rank r 之间的连接载体"（见 `channel.cc:35-38` 的分配逻辑：`channel->peers[r] = sharedRes->peers[channelId] + topParentRanks[r]`）。
>
> 换句话说，**每个 Channel 都维护一张覆盖全部 rank 的"连接表"**，下标就是 rank 编号，天然把"和谁连"分好了格。但**不是每个格子都有真实连接**——Channel 的 Ring/Tree 结构（`ring.prev`/`ring.next`/`tree.up`/`tree.down`）决定了"本 rank 应该连哪些 rank"：Ring 里只连 `prev`、`next` 两个邻居，所以只有 `peers[prev]`、`peers[next]` 这两个格子里的 connector 是真正建好的（`connected=1`），其余格子的 connector 都是空的、未建连接。
>
> ```text
> comm->channels[16]                 // 16 个 channel
>   └─ channels[k].peers[128]        // 指针数组，下标 = rank 编号
>        ├─ peers[prev] ──→ ncclChannelPeer（真实连接：Ring 前驱）
>        ├─ peers[next] ──→ ncclChannelPeer（真实连接：Ring 后继）
>        └─ peers[其他] ──→ ncclChannelPeer（空，未建连接）
>             └─ send[2] / recv[2]   // connector（connIndex 0=集合通信，1=点对点）
> ```
>
> **补充：Tree 算法的邻居用 `up`（父）和 `down[]`（子）体现，而不是 prev/next。** `ncclTree`（`device.h:193-197`）的结构是：
>
> ```c
> struct ncclTree {
>   int depth;
>   int up;                        // 父节点（1 个）
>   int down[NCCL_MAX_TREE_ARITY]; // 子节点数组（最多 3 个）
> };
> ```
>
> - Ring 是"线性环"：每个 rank 固定 2 个邻居 `prev`（前驱）+ `next`（后继）；
> - Tree 是"树"：每个 rank 有 **1 个 `up`（父）** + **最多 3 个 `down`（子）**。`up`/`down[]` 里存的是 rank 编号，`-1` 表示"没有该邻居"（树根 `up=-1`，叶子 `down=-1`）。
>
> 所以落到 `peers` 数组上，Tree 算法里"真正有连接的格子"是 `peers[up]` 和 `peers[down[0..2]]`，其余是空的。
>
> 为什么 `down` 最多是 3 个？这源于 NCCL Tree 的**两层结构**——它其实是"节点间二叉树 + 节点内链"的组合：
>
> - **节点间层（走网络，慢）**：把每个节点（机器）抽象成一个"虚拟节点"，节点之间组成**二叉树**（每个节点最多 2 个跨节点子节点）。用二叉树是为了把网络跳数压到 O(log N)。
> - **节点内层（走 NVLink，快）**：同一节点里的多个 GPU 被排成一条**链**，链上每个 GPU 的 `down[0]` 存"链上的下一个 GPU"、`up` 存"链上的上一个"（见 `connect.cc:61-62`）。
>
> 所以 `down` 数组 3 个槽位的分工是：`down[0]` = 节点内链的下一个 GPU（NVLink），`down[1]` / `down[2]` = 跨节点的 2 个子节点（网络）。这就是 `all_reduce.h:96` 注释 "binary tree + local" 的确切含义——**二叉树（2 个跨节点子）+ 本地（1 个节点内链的下一个）**。
>
> ```text
> 跨节点（二叉树，走网络）：
>         [节点A] ← 树根
>          ├─ down[1] ─▶ [节点B]
>          └─ down[2] ─▶ [节点C]
>
> 节点内（链，走 NVLink），以节点A为例：
>    rank0 ─down[0]─▶ rank1 ─down[0]─▶ rank2 ─down[0]─▶ rank3
>    (链头，同时承担跨节点的 down[1]/down[2])
> ```
>
> 一个容易误解的点：**不是"每个节点只有一个 GPU 参与树"**。恰恰相反——节点内**所有 GPU** 都参与树形结构，只是通过"节点内链"参与；跨节点的二叉树，才由**链头（head）这一个 GPU** 代表整个节点参与。于是 `down` 数组要同时容纳两种连接：
>
> - `down[0]`（节点内链）：链上**每个非尾部 GPU** 都用到；
> - `down[1]` / `down[2]`（跨节点子）：**只有链头**用到（且仅当它不是叶子）。
>
> 以"2 节点 × 4 GPU"（rank0~3 在节点 A，rank4~7 在节点 B，rank0/rank4 为链头）为例，每个 rank 的 `down` 数组实际填的是：
>
> | rank | 角色 | `down[0]` | `down[1]`/`down[2]` |
> |------|------|-----------|---------------------|
> | rank0 | 链头 + 树根 | rank1（节点内） | rank4（跨节点子） |
> | rank1 | 链中 | rank2 | -1 |
> | rank2 | 链中 | rank3 | -1 |
> | rank3 | 链尾 | -1 | -1 |
> | rank4 | 链头 + 叶子 | rank5 | -1 |
> | rank5~6 | 链中 | 下一个 | -1 |
> | rank7 | 链尾 | -1 | -1 |
>
> 可以看到：`down[0]` 是"链上几乎人人都有"的槽位，`down[1]`/`down[2]` 是"链头专属"的槽位。所以 3 个槽位不是"多给某个节点留的"，而是**节点内链（1 个）和节点间二叉树（2 个）两种层次的连接必须共存于同一个 `down` 数组**。
>
> 而树根的上限 `NCCL_MAX_TREE_ARITY_TOP = 2`（比内部节点少 1），是因为 NCCL 用的是 Double Binary Tree（双树）：每棵树的根只有 1 个跨节点子（+1 节点内 = 2），内部节点才有 2 个跨节点子（+1 节点内 = 3）。
>
> 连接建立时（`ncclTransportTreeConnect`，`generic.cc:55-58`），Tree 是"recv 来自 `down[]`（最多 3 个子）+ send 给 `up`（1 个父）"，再反向一套；而 Ring（`ncclTransportRingConnect`，`generic.cc:26`）是"recv 来自 `prev` + send 给 `next`"。

> 到这里，完整的层次链就补全了：**`ncclComm`（通信域）→ `ncclChannel`（并行通道）→ `ncclConnector`（对某个 peer 的具体连接）→ `ncclTransport`（该连接走的传输层）**。Channel 负责"逻辑上谁连谁"，Connector 负责"物理上怎么连"，Transport 负责"连上之后怎么搬"。
>
> 为了把**数量关系**也讲透，举一个具体例子——假设 1 个通信器、128 个 rank、拓扑搜索后 `nChannels = 16`（Ring 算法）：
>
> - **Channel 是"全体 rank 共享"的全局结构，不是"每对 rank 之间"各有一套。** 这 128 个 rank 共享同一套 16 个 channel（channel 0~15），`nChannels` 是通信器级别的属性，所有 rank 相同（上限 `MAXCHANNELS = 64`）。
> - **每个 Channel 是一条遍历全部 128 个 rank 的逻辑 Ring。** 16 个 channel = 16 条并行的、排列各异的 Ring，每条都串起全部 128 个 rank。数据被切成 16 份，第 k 份在第 k 条 Ring 上流动。
> - **每个 rank 在每个 Channel 里只连 2 个邻居**（Ring 的 `prev` / `next`，见 `ncclRing` 定义 `device.h:175-187`），而不是连全部 127 个 rank。所以每个 rank 在 16 个 channel 里共有 16×2 = 32 条"邻居连接"，分散在 16 条不同的 Ring 里。
> - **"每对 rank 之间有多少条连接"**：只有"在某条 Ring 里恰好相邻"的两个 rank 之间，才存在该 channel 的一条连接；若它们同时在多个 channel 里相邻，就有多条，否则没有。所以连接数取决于"它们在几条 Ring 里相邻"，而不是"每对 rank 之间固定有 16 条"。
>
> 落到数据结构上，以某个 rank 的视角看：
>
> ```c
> comm->channels[16];                        // 本 rank 的 16 个 channel（comm.h:534）
> channels[k].ring.prev / .next;             // 本 rank 在第 k 条 Ring 里的两个邻居
> channels[k].peers[...];                    // 本 rank 在第 k 个 channel 里连到的 peer 数组
> channels[k].peers[p]->send[...] / recv[...]; // 连到 peer p 的连接器
> ```
>
> 其中每个 peer 的 `send`/`recv` 数组长度 `NCCL_MAX_CONNS = 2`（`device.h:231`），是因为一个 communicator 既要支持**集合通信**、又要支持**点对点通信**，两者是两套独立的连接：`connIndex = 0` 留给集合通信（Ring、Tree、CollNet 等**共用**这一个槽位，见 `ncclTransportRingConnect` 传 `0`、`ncclTransportTreeConnect` 也传 `0`，`generic.cc:26/55`），`connIndex = 1` 留给点对点 Send/Recv（见 `init.cc:1621/1625` 的 `send[1]`/`recv[1]`）。所以并不是"Ring 和 Tree 各占一个"，而是"集合通信与点对点通信各占一个"。

---

## 二、初始化流程：从 ncclCommInitRank 到网络就绪

> 第一章我们认清了 NCCL 的几个核心结构体（通信器、通道、连接子、传输层）。这一章回答一个更具体的问题：**当你（或 PyTorch）调用一次 `ncclCommInitRank` 之后，一个"可用的通信器"到底是怎么被一步步造出来的？**
>
> 阅读建议：先读 **2.1**（认清 API 家族——先弄懂"五个公共 API 如何收敛到同一个执行体"这条主线）和 **2.2**（全景图——总共分四个阶段），再逐个阶段深入。若只想快速把握骨架，读完 2.1、2.2、2.6 即可。

### 2.1 先认清 API 家族：五个公共 API 如何收敛到同一个执行体

初始化相关的函数名字一大堆（`ncclCommInitRank`、`ncclCommInitRankConfig`、`ncclCommInitRankScalable`、`ncclCommInitAll`、`ncclCommInitRankDev`、`ncclCommInitRankFunc`……），第一次看极易迷路。

**但只要抓住一条主线，立刻就清楚了：这些函数不是并列关系，而是"四个层次、逐层收敛"的关系。**

#### 核心主线：四个层次，一条调用链

```
【门面层】对外的公共 API（用户 / 框架调用）
    ncclCommInitRank   ncclCommInitRankConfig   ncclCommInitRankScalable   ncclCommInitAll
              │                   │                        │                    │
              └───────────────────┴────────────┬───────────┴────────────────────┘
                                               │  都调用同一个函数，只是参数不同
                                               ▼
【实现层】ncclCommInitRankDev(...)      ← 唯一的实现入口（static，外部看不见）
                                               │  1. 校验 + 归一化参数
                                               │  2. 分配 ncclComm 骨架
                                               │  3. 构造 AsyncJob 任务结构体，转交调度层
                                               │  ncclAsyncLaunch(job, ncclCommInitRankFunc, ...)
                                               ▼
【调度层】ncclAsyncLaunch(...)          ← 决定"什么时候、在哪个线程执行"
                                               │  不在 Group → 立即执行；在 Group → 入队；非阻塞 → 后台线程
                                               ▼
【执行层】ncclCommInitRankFunc(job)     ← 真正干活：本章 2.3~2.6 的四个阶段全在它体内
```

**各层职责**：

| 层 | 代表函数 | 只负责 | 外部可见 |
|----|----------|--------|----------|
| 门面层 | `ncclCommInitRank` 等 5 个 | 参数预处理（取 device、准备默认 config、性能标记）后**转交** | ✅ 导出 |
| 实现层 | `ncclCommInitRankDev` | 把各门面 API 的参数差异**归一化**，备好 comm 骨架与任务，再转交 | ❌ `static` |
| 调度层 | `ncclAsyncLaunch` | 决定**同步执行 / 入队 / 后台线程**三种执行方式 | ❌ `static`（辅助函数） |
| 执行层 | `ncclCommInitRankFunc` | **真正建立通信器**（拓扑、图、连接、资源全在这里完成） | ❌ `static` |

> 记住这张图，后面每一步都能对号入座：**门面层只做"预处理"，实现层做"归一化 + 备料"，调度层管"何时执行"，执行层干"实际建通信器"。**

#### 2.1.1 门面层：5 个公共 API，差异只在三处参数

| 公共 API | 一句话场景 | 关键差异 |
|----------|-----------|----------|
| `ncclGetUniqueId` | 生成全局唯一 ID（严格说不属于 init 家族，但**必须先做**） | 无 |
| `ncclCommInitRank` | 最基础的初始化，多进程/多线程各自建 comm | 不包 Group → **同步阻塞** |
| `ncclCommInitRankConfig` | 带用户配置的初始化，**上层框架常用** | 多一个 `ncclConfig_t` 入参 |
| `ncclCommInitRankScalable` | 超大规模训练，用多个 uniqueId 加速 | 多一个 `nId`（uniqueId 个数）入参 |
| `ncclCommInitAll` | 单进程管多卡时，一次建出"一整串"communicator | 内部**循环**调用实现层 |

**门面函数的共同特征：它们的函数体几乎都只有"预处理 + 转交"两步。** 看最基础的 `ncclCommInitRank` 全文（只有 4 件事）：

```cpp
// src/init.cc:2562-2578（完整函数体）
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclInitEnv());                        // 1. 进程级一次性初始化（见下方说明）
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)          // 2. 性能标记（见 5.1 节）
  (void)ncclCudaLibraryInit();                     // 3. 加载 CUDA 驱动、解析 dlsym hook
  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;   // 4. 准备默认配置
  CUDACHECK(cudaGetDevice(&cudaDev));              //    取当前 CUDA 设备号
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));
  // ↑ 转交实现层。注意倒数第一个参数是本函数名（__func__），供下层打日志/报错时定位
  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  return ncclSuccess;
}
```

> 这两点最能体现"门面"性质：**（1）** 它把"当前是哪个设备""用什么配置"这些**上下文信息**补全后传下去（因为 `ncclCommInitRank` 的签名里根本没有 device 参数，它默认用当前设备）；**（2）** 它把 `__func__`（调用者函数名）一路传下去，让下层在出错时能报出"是哪个 API 调用的"。

##### `ncclInitEnv()` / `ncclInit()`：进程级一次性初始化

这两个函数你在后面每个公共 API 入口都会反复看到，这里讲一次，后续不再重复。

**1. 为什么需要这样一个"进程级初始化"？**

NCCL 是被加载进**你的进程**里的一个库。在它能够建立任何通信器之前，有一批东西必须先准备好，而这些准备事项有三个共同特征：

1. **属于整个进程，不属于某个 comm**——比如"用哪块网卡做带外通道"、"环境变量从哪儿读"；
2. **只需做一次**——比如"把 GPU 显存映射到 CPU 可访问地址"；
3. **成本高或必须唯一**——`dlopen` 一个库、探测网卡、改进程级线程属性都是重操作，重复做既浪费也可能出错。

这类东西显然不能塞进"每个 comm 各自的初始化"里：`ncclCommInitRank` 在真实训练里会被调用**很多次**（数据并行一个 comm、张量并行又一个、MoE 再一个），每次都去 dlopen、探网卡、改线程默认属性，纯属浪费。所以 NCCL 把它们抽成**一次性初始化**，用 `std::call_once` 保证全进程只执行一次。

而 `ncclInitEnv()` 只做一件事：初始化**环境变量插件**（`ncclEnvPluginInit()`，`init.cc:170-173`）——决定环境变量"从哪里读"（默认直接读进程环境，也可由 `plugins/env` 接管）。它比 `ncclInit()` 更靠前，因为后面那些初始化本身可能就要读环境变量。

**2. `ncclInit()` 具体做了哪四件事**

源码注释里那句 "Bootstrap network, CPU stack, GDR" 就是这四件事的缩写（`initOnceFunc`，`init.cc:147-155`）：

| 步骤 | 做了什么 | 为什么需要 |
|------|---------|-----------|
| `ncclOsInitialize()` | 把**新建 pthread 的默认栈大小**抬到至少 8MB（`SAFE_STACK_SIZE`），通过 `pthread_setattr_default_np` 改进程默认值 | Linux 默认线程栈可能偏小（尤其 `RLIMIT_STACK` 为 unlimited 时），NCCL 内部线程栈溢出会直接崩。受 `NCCL_SET_CPU_STACK_SIZE` 控制（`src/os/linux.cc:152-157`） |
| `initGdrCopy()` | 若 `NCCL_GDRCOPY_ENABLE=1`（**默认关**），动态加载 GDRCopy 库，句柄存进全局 `ncclGdrCopy` | 让 CPU 能低延迟读写显存，见下面第 3 点（`init.cc:137-142`） |
| `bootstrapNetInit()` | 选定 bootstrap 带外通道用**哪块网卡/IP**：设了 `NCCL_COMM_ID` 就找同子网的本地网卡，否则自动挑一张；结果存进静态变量 `bootstrapNetIfName` / `bootstrapNetIfAddr` | 所有 rank 要靠这条 TCP/UDS 通道交换地址、QP 号等信息，才能建出 NVLink/RDMA 链路。你看到的 `Bootstrap: Using eth0:10.0.0.5` 就是它打的（`src/bootstrap.cc:105-138`） |
| `initNvtxRegisteredEnums()` | 注册 NVTX 所需的枚举与描述符 | 供 5.1 节的性能标记使用 |

> 注意"CPU stack"是**字面义**：CPU 侧线程栈大小，不是 CPU 拓扑。

**3. GDRCopy 是什么（先把缩写说清楚）**

- **GDR = GPUDirect RDMA**；**GDRCopy = GDR + Copy**，即"基于 GPUDirect RDMA 技术做的 Copy 库"
- 两者共用同一套机制——**GPU 显存通过 PCIe BAR 暴露到 PCIe 地址空间**，区别只在**谁发起访问**：网卡（GPUDirect RDMA）还是 CPU（GDRCopy）

**关键：别把二者当成二选一。** 放到一次跨机传输里看：

```
A 机 GPU 显存
   │ 1. 前提：显存要有"PCIe 可寻址"地址  ← BAR 映射（nvidia-peermem）
   ▼
A 机 RDMA 网卡
   │ 2. 网卡 DMA 引擎按该地址直接读显存  ← GPUDirect RDMA = 1 + 2
   │ 3. 网络
   ▼
B 机 RDMA 网卡 ──4. 反向 DMA──▶ B 机 GPU 显存
```

- **BAR 映射负责"可寻址"**：把 GPU 显存暴露到 PCIe 地址空间，让显存对 GPU 之外可见，但**它自身不搬运任何字节**；
- **网卡负责"搬运"**：由 DMA 引擎把字节真正送到对端。没有 BAR 映射，网卡没有可写的目标地址；没有网卡，显存对外可见也出不了机器。

所以差别只在于**网卡从哪拿数据**：

| 场景 | 路径 | 额外拷贝 |
|------|------|---------|
| 有 GDRDMA（NCCL 置 `NCCL_DIRECT_NIC`，`device.h:122`） | GPU 显存 --(网卡 DMA)--> 网卡 | 0 次 |
| 无 GDRDMA | GPU 显存 → 系统内存 → 网卡 | 多 1 次 |

**GDRCopy 为什么不在这条链上**：它不搬 payload，只是让 CPU（proxy 线程）低延迟读到 GPU 写的标志位（`workFifo` 在 `init.cc:631-634`；`gdcSync`/`gdcFlush` 在 `net.cc:966-970`、`1137-1140`、`1658`）。分工可以概括为：**搬数据靠网卡，看状态靠 GDRCopy，两者都依赖 BAR 映射**。实现上是 `pin` 住显存再映射进用户态，CPU 直接 load/store 约**几百纳秒**，远快于 `cudaMemcpy` 的**微秒**级。

> 内核组件：GPUDirect RDMA 依赖 `nvidia-peermem`（NCCL 在 `src/transport/net_ib/gdr.cc:18-20` 检查），GDRCopy 依赖 `gdrcopy.ko` + `libgdrapi.so`，可同时启用。GDRCopy 默认关闭，需 `NCCL_GDRCOPY_ENABLE=1`，不满足条件会自动降级回 host 内存（`gdrwrap.h:287-293`）。

> **NCCL 源码里 "GDR" 有两个含义**，读代码别混：
>
> | 代码写法 | 实际指 |
> |---|---|
> | `ncclGdrCopy`、`initGdrCopy()`、`NCCL_GDRCOPY_ENABLE` | **GDRCopy**（CPU 直访显存）——本节 `ncclInit()` 注释里的 "GDR" 就是这个 |
> | `peerInfo->gdrSupport`、`ncclIbGdrSupport()`、`ncclGpuGdrSupport()` | **GPUDirect RDMA**（网卡直访显存，`src/include/net.h:28` 注释原文 "Test whether the current GPU support GPU Direct RDMA"） |

**4. 为什么每个 API 入口都无脑调一遍？**

因为实现是 `std::call_once` + 静态变量缓存结果（`init.cc:157-173`）：

```cpp
static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);   // 只有第一次真正执行
  return initResult;                            // 之后每次只是读一个已完成的标志
}
```

所以它**幂等且近乎零成本**。NCCL 在每个入口调它，要的不是"再初始化一次"，而是"**确保底层已经就绪**"——哪怕这个 API 恰好是用户调用的第一个 NCCL 函数。

> 一个容易看漏的点：`ncclCommInitRank` 这里**只调了 `ncclInitEnv()`，没有 `ncclInit()`**。`ncclInit()` 是在更下层的 `ncclCommInitRankDev` 里调的（`init.cc:2489`）；此外 `ncclGetUniqueId`（2.3 节）和 `ncclCommGrow` 也各会调一次——反正有 `call_once` 兜底，重复调用没有代价。

#### 2.1.2 命名规范：`Dev` / `Func` / `AsyncJob` 后缀的含义

（插叙一段：这几个后缀很容易被误读成"设备端"，先把这个命名约定讲清楚，后面看名字就不会被绊住；2.1.3 接回主线。）

这三个后缀是**"按层命名"**的产物，对应上面链条图里的三层：

| 后缀 | 含义 | 例子 |
|------|------|------|
| 无后缀（公共名） | 对外的门面函数：只做参数预处理（取当前 device、准备默认 config）就转交下去 | `ncclCommInitRank` |
| `...Dev` | **实现层的唯一入口**：接收已确定好的 `cudaDev`、`config`、以及调用者函数名（`funcName`，用于日志/报错定位），把各公共 API 的参数差异**归一化**，然后创建异步任务 | `ncclCommInitRankDev`（`src/init.cc:2477`） |
| `...Func` | **执行层的函数体**：真正完成初始化全部工作的那个函数，由调度层调用 | `ncclCommInitRankFunc`（`src/init.cc:1831`） |
| `...AsyncJob` | 传递参数的**任务结构体**（把门面层的差异参数打包，随任务带到执行层） | `ncclCommInitRankAsyncJob`（`src/init.cc:1734-1753`） |

> **怎么一眼分辨"公共"还是"内部"？** 看 `src/init.cc` 里的声明形式：
> - 写成 `NCCL_API(ncclResult_t, ncclCommInitRank, ...)` 的是**导出符号**，外部可调用；
> - 写成 `static ncclResult_t ncclCommInitRankDev(...)` 的是**内部函数**——`static` 意味着**文件作用域**，链接期外部根本找不到它（所以框架想调也调不到）。
>
> 再看 `NCCL_API` 宏本身（`src/include/core.h:16-27`）——它并不是简单的导出标记：
>
> ```cpp
> // src/include/core.h:16-27（Linux 分支节选）
> #if defined(NCCL_OS_LINUX)
> #ifdef PROFAPI
> #define NCCL_API(ret, func, args...) \
>   extern "C" __attribute__((visibility("default"))) __attribute__((alias(#func))) ret p##func(args); \
>   extern "C" __attribute__((visibility("default"))) __attribute__((weak)) ret func(args)
> #else
> #define NCCL_API(ret, func, args...) extern "C" __attribute__((visibility("default"))) ret func(args)
> #endif
> ```
>
> 两点值得注意：**（1）** 在定义了 `PROFAPI` 的构建里，它会用 GCC 的 `alias` 属性额外声明一个 `p##func`（即 `pncclCommInitRank`）作为 `func` 的**别名**——这正是 `nccl.h.in` 里每个 API 都有 `ncclXxx` / `pncclXxx` 两份声明的原因（`p` 版本供性能分析插桩层使用）；**（2）** 真正的实现符号被标成 `weak`，允许外部覆盖——这也是为什么"打桩/替换 NCCL API"能做到。

#### 2.1.3 实现层：所有公共 API 都收敛到同一个函数

**结论先摆出来：5 个门面 API，无论各自动机是什么，最终都调用同一个 `ncclCommInitRankDev`——只是传入的参数不同。** 看三个 API 的收尾调用（`src/init.cc`）：

```cpp
// :2572 —— ncclCommInitRank：nId 固定为 1，config 用默认值，不包 Group
NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));

// :2674 —— ncclCommInitRankConfig：nId 仍为 1，但 config 由用户传入
NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

// :2711 —— ncclCommInitRankScalable：nId 也由用户传入（提供多个 uniqueId）
NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);
```

**所以几个公共 API 的差异，全部浓缩在这两处参数（`nId` 与 `config`）+ 一个控制流（是否包 Group）上**：

| 公共 API | `nId`（用几个 uniqueId） | `config` 来源 | 是否包在 Group 里 | 阻塞行为 |
|----------|------------------------|---------------|------------------|----------|
| `ncclCommInitRank` | 固定 1 | 默认值 | ❌ 不包 | **同步阻塞**，返回即完成 |
| `ncclCommInitRankConfig` | 固定 1 | 用户传入 | ✅ 包（`ncclGroupStartInternal`/`EndInternal`） | 取决于 `config.blocking`，可为非阻塞 |
| `ncclCommInitRankScalable` | 用户传入（≤ nranks） | 用户传入 | ✅ 包 | 同上 |
| `ncclCommInitAll` | 固定 1 | 默认值 | ✅ 包 | 内部对每个设备循环调用 `ncclCommInitRankDev` |

> 官方头文件（`src/nccl.h.in`）对这几个 API 的定位说得很直白：
> - `ncclCommInitRank`（`:180-186`）："ncclCommInitRank **implicitly synchronizes with other ranks**, so it must be called by different threads/processes or use ncclGroupStart/ncclGroupEnd"——它天然阻塞。
> - `ncclCommInitRankConfig`（`:175-177`）："Create a new communicator (multi thread/process version) **with a configuration set by users**"。
> - `ncclCommInitRankScalable`（`:259-263`）："similar to ncclCommInitRankConfig. Allows to use **more than one ncclUniqueId** (up to one per rank), indicated by nId, to **accelerate the init operation**. The number of ncclUniqueIds and their order must be the same for every rank."
> - `ncclCommInitAll`（`:189-195`）："Creates a clique of communicators (**single process version**)"。

#### 2.1.4 调度层：`ncclAsyncLaunch` 决定"什么时候、在哪个线程执行"

`ncclCommInitRankDev` 的最后一步是启动异步任务（`src/init.cc:2544`）：

```cpp
NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);
```

而 `ncclAsyncLaunch` 的**行为取决于当前是否在 Group 内**（`src/group.cc:35-43`）：

```cpp
// src/group.cc:35-43（节选）
ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*), ...) {
  job->destroyFlag = comm->destroyFlag;
  if (ncclGroupDepth == 0) {
    ret = func(job);              // ← 不在 Group 里：直接同步执行，返回即完成
    ...
  } else {
    job->func = func;             // ← 在 Group 里：只是入队，等 ncclGroupEnd 统一处理
    ...
  }
}
```

结合门面层的差异，执行方式其实分成**三种情况**（正好对应"用户想不想异步"的需求）：

1. **不在 Group 里（`ncclCommInitRank` 走这条）** → `ncclGroupDepth == 0` → **立即同步执行、阻塞到完成**。这就是官方文档说它"隐式同步"的底层原因；
2. **在 Group 里 + `blocking != 0`（默认）** → 任务先入队，在 `ncclGroupEndInternal` 里**同步**执行完再返回；
3. **在 Group 里 + `blocking == 0`** → group end **另起一个线程**在后台执行，自己立刻返回 `ncclInProgress`（`src/group.cc:854-857`）：

```cpp
// src/group.cc:854-857（节选）
groupJob->base.func = groupLaunchNonBlocking;
STDTHREADCREATE_GOTO(groupJob->base.thread, ncclAsyncJobMain, ret, fail, &groupJob->base);
groupJob->nonBlockingInit = true;
ret = ncclInProgress;   // ← 立即返回"进行中"
```

##### 先厘清：这个 `ncclInProgress` 到底是"谁"返回的？

这是读这段代码最容易踩的坑。答案是：

> **`ncclInProgress` 是 `ncclGroupEndInternal()`（`src/group.cc:857`）返回的**——也就是统一启动 Group 内全部任务的触发点。
> 上面代码里的 `ret` 是 `ncclGroupEndInternal` 的**局部变量**，随后由函数末尾的 `return ret` 抛出。

**不是** `ncclAsyncLaunch` 返回的，也**不是** `ncclCommInitRankDev` 返回的。这两个函数在 Group 模式下返回的都是 `ncclSuccess`——那个成功**只代表"任务已成功入队"**，不代表初始化已经完成：

```cpp
// src/group.cc:44-69（ncclAsyncLaunch 的 Group 分支，节选）
} else {
  job->func = func;
  job->state = ncclGroupJobRunning;
  ncclIntruQueueEnqueue(&ncclAsyncJobs, job);   // ← 只是把任务挂到这个 thread_local 队列
  ...
}
return ret;   // ret 仍是 ncclSuccess
```

所以完整的两跳传播链是：

```
1.  ncclGroupEndInternal()
        │  （非阻塞分支里 ret = ncclInProgress 后 return）
        ▼
2.  ncclCommInitRankConfig 的收尾：
    NCCLCHECK( ncclGroupEndInternal() )
        │  （NCCLCHECK 对 ncclInProgress 放行，不中断）
        ▼
3.  ncclCommGetAsyncError( *newcomm, &ret )
        │  （用 comm->asyncResult 覆盖 ret）
        ▼
4.  return ret   →   用户拿到 ncclInProgress
```

要特别注意 `ncclCommInitRankConfig` 的收尾这段（`src/init.cc:2677-2687`）：

```cpp
exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());          // 1. 这里可能返回 ncclInProgress
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void)ncclCommGetAsyncError(*newcomm, &ret);   // 2. 用 comm 自己的状态覆盖 ret
    }
    ...
  }
  return ret;
```

第 1 步返回 `ncclInProgress` 却**不会**提前结束函数，是因为 `NCCLCHECK` 对 `ncclInProgress` 是**放行**的（`src/include/checks.h:148-156`）：

```cpp
#define NCCLCHECK(call) \
  do { \
    ncclResult_t RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      return RES;   // ← 只有真正的错误才返回；inProgress 会继续往下走
    } \
  } while (0)
```

于是控制流走到第 2 步，用 `comm` 自己的异步状态覆盖 `ret`——**这才是函数最终返回给用户的值**。

##### `ncclCommGetAsyncError` 只是"读一个原子变量"，它不干活

这是第二个容易误解的点。它的实现只有几行（`src/init.cc:3448-3456`）：

```cpp
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = COMPILER_ATOMIC_LOAD(&comm->asyncResult, std::memory_order_acquire);  // ← 纯读
  if (*asyncError == ncclSuccess && comm->proxyState) {
    *asyncError = COMPILER_ATOMIC_LOAD(&comm->proxyState->asyncResult, std::memory_order_acquire);
  }
  // 后面还会顺带检查 GIN progress 线程的状态……
}
```

它**不推进、不阻塞、不等待**任何工作——就是一次 `acquire` 语义的原子读。所以它是"**查询**"，不是"**推进**"。

进度是由**另一个线程**推进的，形成一个标准的"生产者—消费者"模型：

| 阶段 | 谁在做 | 动作 | 源码位置 |
|------|--------|------|----------|
| **置为进行中** | 主线程（`ncclGroupEndInternal` 起线程前） | `ncclCommSetAsyncError(comm, ncclInProgress)` 把 `comm->asyncResult` 写成 `ncclInProgress` | `group.cc:830` / `group.cc:843` |
| **后台推进** | 新起的后台线程 | `groupLaunchNonBlocking` → `groupLaunch` → `ncclCommInitRankFunc` 真正建通信器 | `group.cc:855` → `group.cc:762` |
| **写入终态** | 后台线程（`groupLaunch` 收尾） | `if (!comm->config.blocking) ncclCommSetAsyncError(comm, ret)`，把状态改成 `ncclSuccess` 或错误码 | `group.cc:748-750` |
| **读取状态** | 用户线程 | `ncclCommGetAsyncError(comm, &err)` 原子读 `comm->asyncResult` | `init.cc:3453` |

> 顺便区分两个容易混淆的字段：`comm->asyncResult`（`comm.h:647`）是 **Group 层对外暴露**的异步状态，`ncclCommGetAsyncError` 读的是它；而 `comm->initState`（`comm.h:741`）是**初始化函数内部**的进度标记（`init.cc:2514` 置为 `ncclInProgress`，`init.cc:1924` 在 `initTransportsRank` 成功后置为 `ncclSuccess`）。两者分工不同，`ncclCommEnsureReady` 检查的是前者。

##### 谁必须轮询？（直接回答"哪个流程需要保证初始化完成"）

**不需要"某个额外流程"去轮询——谁想用这个 comm，谁就得先确认它 ready。** 具体分两类：

**（1）直接以 `blocking = 0` 调用 `ncclCommInitRankConfig` 的用户代码**

API 只是"把状态告诉你一次"就返回了，**不会替你等**。你必须自己写轮询循环：

```c
// 非阻塞初始化的典型用法（伪代码，语义与 NCCL 官方示例一致）
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;
ncclComm_t comm;
ncclCommInitRankConfig(&comm, nranks, id, rank, &config);   // 返回 ncclInProgress

// 主线程可以先去做别的事，比如加载数据、建 pipeline
load_data();

ncclResult_t asyncErr = ncclInProgress;
while (asyncErr == ncclInProgress) {
  ncclCommGetAsyncError(comm, &asyncErr);   // 反复读那个原子变量
}
if (asyncErr != ncclSuccess) { /* 初始化失败，处理错误 */ }
```

**（2）任何后续要使用这个 comm 的代码——NCCL 会强制拦住你**

这一点很关键：NCCL 在几乎所有 API 的入口都埋了一道检查 `ncclCommEnsureReady()`（`src/init.cc:422-439`）：

```cpp
ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));       // ← 内部也是调它去读状态
    if (ret == ncclInProgress) {
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      ret = ncclInvalidArgument;                        // ← 直接判为非法使用
      goto exit;
    }
  }
  return ret;
}
```

`ncclCommCount`、`ncclCommCuDevice`、`ncclCommUserRank`、集合通信、`ncclCommFinalize`……这些函数的第一句基本都是 `NCCLCHECK(ncclCommEnsureReady(comm))`。所以如果你在初始化还没完成时就去用这个 comm，会直接拿到 `ncclInvalidArgument` 和那条醒目的 WARN。

**换句话说**：轮询这件事不是"可选的性能优化"，而是**非阻塞模式下的强制契约**——要么你自己轮询到 `ncclSuccess`，要么你就会被 `ncclCommEnsureReady` 挡在门外。

##### Group 机制的作用

到这儿可以回头理解 Group 到底为什么存在了。`ncclGroupStart()` / `ncclGroupEnd()` 的本质是**"攒批 + 统一启动"**：

- 在 `ncclGroupStart` 与 `ncclGroupEnd` **之间**，所有 API 调用都只是**登记**：把任务塞进 `thread_local` 队列 `ncclAsyncJobs`（`group.cc:31`），**一个都还没真正执行**；
- 只有到了 `ncclGroupEnd`，才一次性决定怎么启动它们（同步 `groupLaunch` 或后台线程）；
- 由于队列是 `thread_local` 的，每个进程各自攒自己那一批，然后**所有 rank 在同一个 Group 边界上统一启动**。

这个设计的价值在于：集合通信/初始化都要求"所有 rank 都到齐才能开始"。如果 rank 0 早早开始 Send 而 rank 1 还没准备好，就会卡死或者走成低效路径。**用 Group 把"下发"和"启动"分离**，能保证所有 rank 把该登记的都登记完，再同时起跑——既避免了时序错配导致死锁，也让多流、多 comm 的操作可以并发启动。

而"非阻塞"只是这个机制上的一个开关（`config.blocking = 0`）：启动之后不等结果，**先把控制权还给调用者**，进度交给后台线程，状态通过 `asyncResult` 暴露给 `ncclCommGetAsyncError`。这就是"非阻塞初始化"的全貌。

##### 反过来说：默认的阻塞模式会怎样？

不设 `blocking = 0`（即默认 `blocking = 1`）时，同样由 `ncclGroupEndInternal` 统一启动，但走的是另一个分支：**不另起线程，而是在当前线程里把 `groupLaunch` 跑完才返回**（`group.cc:858-866`）。

所以两者的区别在于：

> **阻塞 = "启动后在调用线程等待结果"；非阻塞 = "启动后立即返回，结果通过 `asyncResult` 异步通知"。**

| 维度 | 阻塞（默认） | 非阻塞（`blocking = 0`） |
|------|-------------|------------------------|
| 函数返回时 | 初始化**已完成**（或已失败） | 可能**仍在进行** |
| 返回值 | 真实结果（`ncclSuccess` / 错误码） | `ncclInProgress` |
| 需要轮询吗 | 不需要 | 必须（否则后续 API 被 `ncclCommEnsureReady` 拦下） |
| 主线程 | 被占住直到结束 | 立刻交还，可去做别的事 |

> **阻塞的代价**：初始化要做 bootstrap 交换、拓扑探测、建连接、注册内存，跨机场景动辄几十毫秒到秒级。这段时间主线程完全被占住。非阻塞模式存在的意义就是**把这段等待利用起来**——先去加载数据、建 pipeline、准备下一批输入。

> ⚠️ 一个约束：**同一个 Group 里不允许混用阻塞和非阻塞 comm**（`group.cc:60-63` 有明确检查，违反直接返回 `ncclInvalidArgument`）。

（补充：阻塞模式下 `asyncResult` 不会被更新——因为"函数返回"本身就是完成信号，不需要第二个通知渠道。它靠分配时清零即等于 `ncclSuccess`，所以后续 `ncclCommEnsureReady` 也不会误拦。）

> 补充一句：`ncclCommInitRankConfig` 自己内部就包了一层 `ncclGroupStartInternal()` / `ncclGroupEndInternal()`（`init.cc:2667` / `init.cc:2679`）。所以即便用户代码里没有写 `ncclGroupStart`，走进 `ncclAsyncLaunch` 时 `ncclGroupDepth` 也已经是 1 —— 这就是为什么它一定会经过"入队 → 统一启动"这条路，而不是直接同步执行。

#### 2.1.5 执行层：`ncclCommInitRankFunc` 才是真正干活的那个

前两层都在"准备和转交"，真正建立通信器的代码在**执行层**。它的入口长这样（`src/init.cc:1831-1833`）：

```cpp
// src/init.cc:1831-1833
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;  // 取回任务结构体
  ncclComm_t comm = job->comm;
  ...   // ← 本章 2.3 ~ 2.6 讲的四个阶段，全部发生在这个函数体内
}
```

它接收的参数是**基类指针** `ncclAsyncJob*`（因为调度层要统一处理各种任务），进函数后第一件事就是**向下转型**回具体的 `ncclCommInitRankAsyncJob`，再取出 `comm`、`nranks`、`commId`、`config`、`cudaDev` 等参数。

> **为什么需要一个"任务结构体"来传参，而不是直接传一堆参数？** 因为执行可能发生在**另一个线程**（非阻塞初始化，见 2.1.4）或**稍后的 Group 结束时刻**——那时门面函数的栈帧早就销毁了，参数必须随任务对象保存。`ncclCommInitRankAsyncJob`（`src/init.cc:1734-1753`）就是这个任务结构体，它还容纳了**多个门面 API 共用的字段**：
>
> | 字段分组 | 服务于哪个门面 API |
> |----------|-------------------|
> | `nranks, myrank, nId, commId, config` | `ncclCommInitRank` / `Config` / `Scalable` |
> | `parent, color, key, childCount` | `ncclCommSplit`（从父通信器派生） |
> | `excludeRanksList, excludeRanksCount` | `ncclCommShrink`（剔除某些 rank） |
> | `isGrow` | `ncclCommGrow`（扩容） |
> | `funcName[NCCL_COMMINIT_FUNCNAME_LEN]` | 记录"是哪个 API 发起的"，用于日志与报错定位 |
>
> 也就是说，`split` / `shrink` / `grow` 复用这条链**不是"顺带省了点代码"，而是这套分层的有意设计**。把当前仓库里所有走这条链的入口数一遍，会看到一个很清晰的收敛结构：
>
> **7 个公共 API → 3 个创建路径 → 1 个执行体（`ncclCommInitRankFunc`）**
>
> | 公共 API（门面层） | 走哪个创建路径（实现层） |
> |-------------------|------------------------|
> | `ncclCommInitRank` / `Config` / `Scalable` / `InitAll` | `ncclCommInitRankDev`（`init.cc:2544`） |
> | `ncclCommSplit` / `ncclCommShrink` | `ncclCommInitChildComm`（`init.cc:3155`，源码注释即 "for both split and shrink"） |
> | `ncclCommGrow` | `ncclCommGrow` 自身（`init.cc:3355`） |
>
> 这个结构的价值不在于"以后 API 会越来越多"，而在于**把变动隔离在一侧**：今后无论是新增一种派生方式，还是调整初始化的内部流程，改动都只需落在"门面层怎么组装 `ncclCommInitRankAsyncJob`"或"执行层怎么建通信器"其中一边，另一边基本不用跟着动。

#### 2.1.6 上层框架用哪个？为什么要提供这么多变体

**框架只能看见公共 API**（`Dev`/`Func` 是 `static` 的）。实践上，PyTorch 的 `ProcessGroupNCCL` 走的是 **`ncclCommInitRankConfig`**——因为它需要传递 `ncclConfig_t`（超时、blocking、splitShare 等），而基础版 `ncclCommInitRank` 没有配置入口，只作为老版本 NCCL 的回退路径。

> 你可以自己验证：在 PyTorch 源码 `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` 里搜索 `ncclCommInitRankConfig`，会看到它在创建 communicator 的辅助函数里被调用。
>
> **这也解释了 NCCL 为什么要提供这么多变体**：不同调用者的需求不同（要不要传配置、规模大到需要多 root、是不是单进程管多卡），但**实现只有一份**（全部收敛到 `ncclCommInitRankDev` → `ncclCommInitRankFunc`）。这是典型的"门面函数 + 统一内核"设计。

### 2.2 初始化全景图：四个阶段

搞清 API 家族后，再看整个初始化的**阶段划分**。`ncclCommInitRankFunc`（真正干活的执行体）做的事可以归纳为四个阶段：

| 阶段 | 目标（职责） | 关键函数 | 产物 |
|------|-------------|----------|------|
| **1. 唯一 ID 生成** | 生成全局唯一的 ID，让所有 rank 定位到同一个引导服务（root） | `ncclGetUniqueId` → `bootstrapGetUniqueId` | `ncclUniqueId`（含 root 的 IP:port + magic） |
| **2. 骨架 + 控制面** | 分配 `ncclComm` 空壳；各 rank 建立**不依赖 GPU 互联**的带外通道 | `ncclCommInitRankDev` 分配骨架 → `bootstrapInit` | 空的 `ncclComm` + `comm->bootstrap` |
| **3. 信息交换与拓扑图** | 三次信息交换：第 1 次（全体）交换 peerInfo；第 2 次（**仅机内**）融合同机 XML 拓扑；探测本机拓扑并搜索 Ring/Tree/CollNet/NVLS 通信图；第 3 次（全体）交换图参数并聚合出全局一致 | `bootstrapAllGather` ×2 + `bootstrapIntraNodeAllGather` ×1 + `ncclTopoGetSystem` / `ncclTopoCompute` | 全量 peer 信息 + 整机拓扑 + 各算法图参数 + `comm->nChannels` |
| **4. 连接与资源** | 按图把 Channel → Connector → Transport 一层层建起来，映射真实物理资源；启动 Proxy 线程 | `setupChannel`、`ncclTransportP2pSetup`、`ncclTransportRingConnect`、`ncclProxyCreate` | 可用的 `comm->channels[]` |

> **为什么必须先有"控制面"（阶段 2、3）？** 换个角度看：高性能数据面网络（NVLink 拓扑、RDMA QP、IPC 显存句柄）的建立过程，本身需要**大量的元信息交换与协商**——拓扑探测结果、每个 rank 的设备与网卡信息、对端显存的 IPC handle、QP 号、内存注册句柄……这些消息**小、杂、多**，而且建网期间数据面还不存在。
>
> 所以 NCCL 的做法是：**先用带外网络（TCP socket，同机用 Unix Domain Socket）建一个轻量的"管理小网络"（Bootstrap），专职负责建网期间的一切元信息交换**。这样分工带来两个好处：
>
> - **管理面足够简单可靠**：TCP 天然有序、可靠、跨机通用，用它传"建网指令"几乎不会出错，同机走 UDS 更是零成本；
> - **数据面的建立因此变得高效且可编排**：元信息交换有了专属通道，建网流程可以按"探测 → 交换 → 建连"的固定节奏推进，不必为传几个小消息专门依赖任何高性能链路。
>
> 换句话说，**阶段 2、3 先搭的这个小网络，为后面那个更高效、更专一的高性能数据面网络创造了建立的可能性**。数据面建成之后，bootstrap 不再参与数据面，只继续承担少量控制消息（如错误通知、abort）。

下面按 1 → 4 的顺序，把每个阶段逐一展开。

### 2.3 阶段一：生成唯一 ID —— `ncclGetUniqueId`

**职责**：生成一个全局唯一 ID，让分处不同进程/机器的所有 rank 获知彼此属于同一个通信域。它是初始化流程里**唯一需要用户自己做一次跨进程分发**的东西（通常由 rank 0 生成，再通过 MPI/文件/env 广播给其他 rank）。

```cpp
// src/init.cc:182-196
NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInitEnv());
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  struct ncclBootstrapHandle handle;
  NCCLCHECK(bootstrapGetUniqueId(&handle, NULL));
  // ncclUniqueId 和 bootstrapHandle 的大小/对齐不同，先清零再 memcpy
  memset(out, 0, sizeof(*out));
  memcpy(out, &handle, sizeof(handle));
  ...
}
```

> **几个值得注意的点**：
> - `ncclUniqueId` 本质是 **bootstrap handle 的拷贝**——整个结构只有三个字段（`src/include/bootstrap.h:14-18`）：`uint64_t magic`（校验值，`bootstrap.cc:446` 写入 `NCCL_MAGIC`）、`union ncclSocketAddress addr`（root 的地址，即 IP:端口）、`int nRanks`。所以它**不包含任何 GPU 信息**，只描述"引导服务的地址 + 预期成员数"。
> - 它必须能塞进 128 字节：`nccl.h.in:40-41` 定义 `NCCL_UNIQUE_ID_BYTES = 128`，并由 `static_assert(sizeof(ncclBootstrapHandle) <= sizeof(ncclUniqueId))` 在**编译期**保证装得下（`bootstrap.h:19-20`）。由于两者大小/对齐不同，函数里先 `memset` 清零再 `memcpy`，避免残留的未初始化数据被当作有效内容。
> - 它**必须早于** `ncclCommInitRank` 调用，且**每个通信域都要用新生成的 ID**——同一个 ID **不能**发起第二次 init。原因：ID 里的 bootstrap root 是**一次性的集中登记服务**，第一次 init 全体 rank 上报地址后监听 socket 即关闭（`bootstrap.cc:392-396`），再连只会被拒。所以 split/grow 派生新通信器时，都会生成**新** ID + **新** root（`bootstrap.cc:448-456`），从不复用。真正"每次 init 各自建一套"的是**数据面连接**（QP / IPC handle），通信器之间的隔离由"随机 magic + `commHash` 身份"保证。
> - 日志里那行 `TRACE_CALL(... getHash(...))` 是在打印这个 ID 的**指纹**。`getHash` 是 NCCL 通用的 64 位哈希（`src/include/bitops.h:539`），意义在于：uniqueId 是 128 字节二进制块，没法读也没法比对，哈希成短代号后，**同一个 ID 在任何 rank 上指纹相同**，排查"各 rank 是不是拿着同一个 ID"一眼可辨；且日志只留指纹、不留原文——拿到原文就等于能加入通信域。这个函数影响面很广：**通信器的身份标识 `commHash` 就是用它算的**（`init.cc:1905`），你在日志和 NVTX 里反复看到的 `commHash` 源头即此，`hostHash`、MNNVL 的 `cliqueId` 同理。

ID 生成并分发到位后，初始化的控制权回到 NCCL 内部——阶段二开始：先建骨架，再建控制面。

### 2.4 阶段二：骨架 + 控制面

这一阶段做两件事：**分配一个"空壳" `ncclComm`**，以及**建立带外控制通道（Bootstrap）**。

#### （1）分配骨架：`ncclComm` 的初始字段

所有公共 API 最终都进 `ncclCommInitRankDev`，它在这里把 comm 的骨架和"安全阀"先建起来（`src/init.cc:2506-2510`）：

```cpp
// src/init.cc:2506-2510（节选）
NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);                      // ncclComm 主体
NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);           // 中断标志（host）
NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail); // 中断标志（device 可见）
comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
```

> 注意 `startMagic`/`endMagic` 在这里就被设为 `NCCL_MAGIC`（还记得 1.5 节讲的"魔数哨兵"吗？就是这里写入的）。而 `abortFlag` 之所以要准备 **host 和 device 两份**，是因为 NCCL 允许"一个 rank 出错时通知所有 rank 的内核提前退出"——这个标志必须能被 CPU 写、被 GPU 读，所以要分配在**页锁定（pinned）主机内存**上（`ncclCudaHostCalloc`）。
>
> 初读容易疑惑：第 1 行已经把整个 `ncclComm` calloc 出来了，为什么还要对它内部字段再 calloc？因为 `abortFlag`/`abortFlagDev` 在结构体里只是**指针字段**（`comm.h:650-652`）——第一次分配只给了"装指针的槽位"（清零为 NULL），真正的标志位内存必须**另行分配**再把地址填进去。这么做有两个原因：一是 `abortFlagDev` 必须是 pinned 内存（GPU 可见），只能走 CUDA 分配器，没法内嵌在普通堆里；二是 `abortFlag` 会被多个通信器**共享**——子通信器直接指向父通信器的同一份标志（配合 `abortFlagRefCount` 引用计数），"指针 + 独立分配 + 引用计数"才能实现这种多对一共享：abort 一个通信器，所有共享该标志的通信器同时被通知，内存由最后一个使用者释放。

注意此时 comm 里**还什么都没有**：没有拓扑、没有 channel、没有连接。这些都要在后面的阶段逐步填充。

#### （2）建立带外控制面：Bootstrap

**职责**：在"GPU 之间的高速链路还不存在"的前提下，先建一条**只依赖网络栈（TCP / Unix Socket）的可靠通道**，用于后续交换建链所需的句柄信息。关键调用在 `ncclCommInitRankFunc` 中：

```cpp
// src/init.cc:1916 — 常规初始化（从 uniqueId 建 bootstrap）
bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm, job->parent);

// src/init.cc:1891 — Split 初始化（从父通信器派生，不需要新 root）
bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks);
```

Bootstrap 之后向上提供的原语（`src/include/bootstrap.h`）：

| 原语 | 用途 |
|------|------|
| `bootstrapAllGather` | **全体**交换（第 1、3 次信息交换的主力） |
| `bootstrapSend` / `bootstrapRecv` | 点对点数据交换 |
| `bootstrapBarrier` | 全局栅栏同步 |
| `bootstrapBroadcast` | 广播 |
| `bootstrapIntraNodeBarrier` / `bootstrapIntraNodeAllGather` | **仅机内**的优化版（走 UDS，不绕 root）——第 2 次交换（机内 XML 融合）用的就是它 |

**实现思路**（`src/bootstrap.cc`）：root rank 创建 TCP 监听 socket，其余 rank 连向 root，形成**星形拓扑**；跨 rank 的 AllGather 由 root 中转。同一个节点内的 rank 之间则通过**共享文件描述符（Unix Domain Socket）**建立直连，避免都挤到 root 上。

> **多 root 的意义**：`ncclCommInitRankScalable` 传的 `nId` 就是"用几个 uniqueId"，即**建多个 root**——因为在大规模集群里，让几千个 rank 都连向同一个 root 会形成瓶颈。这是"Scalable"这个名字的由来。

### 2.5 阶段三：信息交换与全局拓扑 —— 从交换成员信息到生成通信图

**职责**：把"每个 rank 只掌握本地信息"变成"掌握全体成员信息"，再据此把硬件画成拓扑图、搜出各算法的通信路线，最后聚合成全局一致的参数。这一步在 `initTransportsRank`（`src/init.cc:965`）里完成，靠的就是阶段二建好的 Bootstrap 通道。

先建立一个全局视角：**这个阶段里其实有三次信息交换**，它们的规模、范围、内容各不相同——这也是理解后面所有内容的主线：

| 次序 | 范围 | 交换什么 | 时机 |
|------|------|---------|------|
| **第 1 次**（源码注释 AllGather1） | **全体**（跨机 + 机内） | `ncclPeerInfo`：rank、设备、所在主机、能力协商字段 | 2.5.1 |
| **第 2 次** | **仅机内**（同主机 rank 之间） | 各自探测出的本机 XML 拓扑 → 融合成整机图 | 2.5.2 |
| **第 3 次**（源码注释 AllGather3） | **全体**（跨机 + 机内） | 各自算出的图参数 + 预设邻居 → 聚合出全局一致 | 2.5.4 |

```
第 1 次交换【全体】：ncclPeerInfo —— 交换成员信息（rank / 设备 / 能力）
      ↓
拓扑发现 + 图搜索（2.5.2 ~ 2.5.3）
      ├─ 本机探测（机内，NVML/PCIe 真探测）
      ├─ 第 2 次交换【仅机内】：同机 XML 融合成整机图
      └─ 图搜索（纯本地计算，零通信）
      ↓
第 3 次交换【全体】：allGather3Data —— 交换各自算出的图参数，聚合出全局一致结果（2.5.4）
```

> 范围的规律值得记住：**第 1、3 次是全体**——因为要建立/对齐的是**全局认知**；**第 2 次只有机内**——因为"整机精确拓扑"只需要同主机的进程合作，远程机器反正只能当"叶子"（见 2.5.2）。

#### 2.5.1 第 1 次交换【范围：全体】：`ncclPeerInfo` —— 交换成员描述

```cpp
// src/init.cc:1036-1037
fillInfo(comm, comm->peerInfo + rank, comm->commHash);        // 填自己的信息
bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo));  // 收齐全员信息
```

交换的结构体是 `ncclPeerInfo`（`src/include/transport.h:43-65`），字段就是"为了让别人能连上我，我必须告诉别人的一切"：

| 字段 | 作用 |
|------|------|
| `rank` / `cudaDev` / `nvmlDev` | 身份与设备号 |
| `busId` / `gpuUuid` | **物理定位**：靠它判断两块 GPU 在不在同一台机器、能不能走 NVLink |
| `hostHash` / `pidHash` | 是否同主机、是否同进程——决定能否用 UDS 快通道、能否用 CUDA IPC |
| `cudaCompCap` | 计算能力（决定可用内核特性） |
| `gdrSupport` / `fabricInfo` | GDR（GPU Direct RDMA）、MNNVL 等能力协商 |
| `comm` | 回指通信器 |

> **`hostHash`/`pidHash` 为什么关键**：UDS 是同主机进程间的内核级快通道（不走 TCP/IP 栈）；CUDA IPC 则让同主机的不同进程**直接读写对方的 GPU 显存**（一方 `cudaIpcGetMemHandle` 拿句柄传给对方，对方 open 后零拷贝访问）。两者都以"同主机"为前提。由于这些 hash 在 AllGather 阶段就换到手了，**NCCL 在建任何连接之前就知道每对 rank 是什么关系**，选 transport 只是查表：
>
> | 两 rank 的关系 | 能走的路 |
> |---------------|---------|
> | 同主机 + 同进程（单进程多线程管多卡） | `P2P_DIRECT`：地址空间本来就是同一体，直接指针访问（`p2p.cc:414-416`） |
> | 同主机 + 不同进程 | CUDA IPC：把对端显存映射进自己地址空间，NVLink/PCIe 直连（`p2p.cc:248/318`） |
> | 同主机、P2P 不通 | SHM：CPU 经共享内存搬运（`shm.cc:105`） |
> | 跨主机 | 只剩 NET（RDMA/Socket），P2P/SHM/UDS 全部出局 |
>
> 判断本身就是一个宏：`P2P_SAME_PID = hostHash 相同 && pidHash 相同`（`p2p.cc:330-331`）。

> 这一轮的产物是 `comm->peerInfo[]`——一个**长度为 nRanks 的数组**，之后"我能不能和 rank r 直连走 NVLink"这类判断全靠它。

#### 2.5.2 拓扑发现：三层建成全图

**职责**：回答"本机与全体的 GPU、NVSwitch、网卡是怎么连的"，产物是 `comm->topo`（硬件拓扑图）。（图搜索的产物 `comm->graphs[]`、`comm->nChannels` 见 2.5.3。）

```cpp
// src/init.cc:1141-1149（节选）
ncclTopoGetSystem(comm, &comm->topo);          // 构建系统拓扑图
ncclTopoComputePaths(comm->topo, comm);        // 计算 GPU ↔ NIC 路径
ncclTopoTrimSystem(comm->topo, comm);          // 剔除不可达 GPU / 未使用 NIC
ncclTopoComputePaths(comm->topo, comm);        // 裁剪后重算路径
ncclTopoSearchInit(comm->topo);                // 初始化搜索状态
```

实现主要靠 **NVML API** 查询 GPU 互联信息，构建一张**以 GPU 为中心的加权图**（`src/graph/topo.cc`、`src/graph/xml.cc`）：

- **节点类型**：GPU、PCI（PCIe 桥）、NVS（NVSwitch）、CPU、NIC、NET、GIN、RMA、DEV、CXB（`src/graph/topo.cc:32-34`）
- **链路类型**：LOC（本地）、NVL（NVLink）、C2C（片间直连）、PCI、SYS（跨 CPU Socket）、NET
- **链路"名字"（用于表达路径质量）**：LOC、NVL、NVB（NVLink Bridge）、C2C、PIX（同一 PCIe Switch）、PXB（跨 PCIe Switch）、P2C、PXN（跨 CPU 但经 NVLink）、PHB、SYS、NET

> 用"节点 + 加权边"表达硬件，好处是后面可以套用**通用图搜索算法**：把"带宽"当作边的代价，就能用统一的搜索框架找"总代价最低的环/树"。

**这张"全图"分三层建成——每层范围、信息来源、精确度都不同：**

| 层 | 范围 | 信息来源 | 精确度 |
|----|------|---------|--------|
| 1. 本机自身 | 机内 | NVML / PCIe 实时探测（`topo.cc:1800`，注释原话："Detect only the GPU managed by this process"） | **精确** |
| 2. 同主机的其他 GPU | 机内 | **XML 融合**：同机各 rank 交换各自探测出的 XML 再融合 | **精确**（拼出来的） |
| 3. 远程机器的 GPU | 跨机 | 只有第 1 次交换得来的 `peerInfo`（存在、busId、能力） | **粗糙**：内部接线不可知，到它的路径一律记为 NET |

第 2 层就是初始化里的**第 2 次信息交换**——范围**仅限机内**：

```cpp
// src/graph/topo.cc:1908-1924（节选）— 同机 XML 融合
// 按 hostHash 找出同主机的 ranks，然后：
bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, ...);   // 机内交换各自的 XML
for (int i = 0; i < nLocalRanks; i++)
  NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);       // 逐个融合成整机图
```

**为什么第 2 层只需机内、第 3 层却只能"将就"？** 整机拓扑（谁和谁用 NVLink 连着）只有同机的进程**探得到**；远程机器的内部接线谁也看不见，只能把它当"挂在网络上的叶子"。于是每台机器拿到的全图都是：**自己精确 + 同机精确（融合）+ 远程粗糙（叶子）**——机内段可以精细优化，跨机段只能按 NET 带宽近似。这也为 2.5.4 的聚合埋下伏笔：**各机的"精确部分"互不重叠，谁也没有完整真相**。

#### 2.5.3 图搜索：为每种算法找一条最优路径

整机拓扑图就绪后，下一步是在它上面为每种算法搜出一条通信路线。以 AllReduce 为例，NCCL 会为 Ring、Tree 两种算法**各算一份图**（还有可选的 CollNet、NVLS）：

```cpp
// src/init.cc:1173-1217（节选）
ringGraph->pattern = NCCL_TOPO_PATTERN_RING;                  // 环形
ncclTopoCompute(comm->topo, ringGraph);

treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;         // 平衡二叉树
ncclTopoCompute(comm->topo, treeGraph);

collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;          // CollNet（若启用交换机网内计算）
collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;
ncclTopoCompute(comm->topo, collNetChainGraph);

nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;                  // NVLS（若支持 NVSwitch 网内计算）
ncclTopoCompute(comm->topo, nvlsGraph);
```

支持的算法枚举（`src/init.cc:53-54`）：`Tree, Ring, CollNetDirect, CollNetChain, NVLS, NVLSTree, PAT`。

搜索算法实现在 `src/graph/search.cc`（通用搜索框架——含超时、带宽递减重试等策略）、`src/graph/rings.cc`（Ring 搜索）、`src/graph/trees.cc`（Tree 搜索）。

> **为什么"图"要分算法各算一份？** 因为它们的**优化目标不同**：
> - **Ring**：目标是环上相邻 rank 之间路径代价之和最小（适合大消息、带宽优先）；
> - **Tree**：目标是树的形状让跳数 ≤ O(log N)（适合小消息、延迟优先），且要区分"节点间二叉树 + 节点内链"两层结构（见 1.5 节）；
> - **CollNet / NVLS**：目标是利用交换机/NVSwitch 的**网内计算**能力，让归约在交换节点完成，减少数据往返。
>
> 这个阶段的本质是**"把硬件拓扑翻译成若干条候选通信路线"**，供运行时（第三、五章）根据消息大小挑选。

注意：**图搜索是纯本地计算，零通信**——但它的输入里含本机网卡（只有自己知道），所以各 rank 独立算出的结果**天然可能不同**。怎么收敛成全局一致？这正是下一次交换的任务。

#### 2.5.4 第 3 次交换【范围：全体】：把各自的图对齐

```cpp
// src/init.cc:1283 — 第 3 次全体交换（缓冲区就叫 allGather3Data，"3" 与 AllGather1 呼应）
bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data));
```

> 注意这次调用**没有"参与者名单"参数**——"全体"的范围由第一个参数 `comm->bootstrap`（`bootstrapState`）决定，它在阶段 2 建控制面时就定型了：记录着本通信域的 `rank`/`nranks` 和环上邻居连接。实现上就是沿 bootstrap 环转圈（`bootstrap.cc:1144`，注释原话 "Simple ring based AllGather"）：数据每步前进一格，`nranks-1` 步后人人拿到全员数据——**能参与这个环的，就是全体成员**。对照第 2 次交换的 `bootstrapIntraNodeAllGather` 需要显式传 rank 名单，反证了不带名单的 `bootstrapAllGather` 范围就是全体。

交换的是各 rank **各自算出的图参数**（`init.cc:1256-1281`），缓冲区结构 `allGatherInfo`（`init.cc:994-1007`）把内容分成三类：

| 类别 | 字段 | 含义 |
|------|------|------|
| **图参数** | `graphInfo[算法]`：`pattern` / `nChannels` / `sameChannels` / `bwIntra` / `bwInter` / `typeIntra` / `typeInter` / `crossNic` | 我为每种算法搜出的图长什么样、值多少带宽 |
| **预设邻居** | `topoRanks`：`ringRecv`/`ringSend`/`ringPrev`/`ringNext`、`treeToParent`/`treeToChild0/1`、`nvlsHeads`（各 `MAXCHANNELS` 条） | 基于我的图，我认为我在这条 channel 上的上/下一跳是谁（`graph.h:195-206`） |
| **本地事实** | `cpuArch`/`cpuVendor`、`localNetDeviceCount`/`Bw`、`localCollNetCount`、`minNetBw`、`isAllNvlink`、`p2pnChannelsPerPeer`/`p2pMaxPeers` | 只有我自己知道的机器事实 |

AllGather 之后，聚合按字段逐个进行（`init.cc:1291-1316`、`1435-1451`），**产出正好是阶段四建连所需的全部全局参数**：

| 聚合 | 规则 | 产出（交给阶段四） |
|------|------|------------------|
| 图参数 | `nChannels`/`sameChannels`/`bw*` → **min**；`type*`/`crossNic` → **max** | 全局统一的 `comm->nChannels` → 阶段四建几条 channel；带宽 → `computeBuffSizes` 定各协议（LL/LL128/Simple）缓冲区大小 |
| `topoRanks` | `ncclTopoPostset`（`:1480`）把所有人的预设**拼装** | 最终 `rings[]`（每 channel 完整环序）→ `setupChannel` 按它建连接表；`ring.prev/next`、`tree.up/down` → 阶段四按它们与邻居握手 |
| `ringRecv[0]` | 相同者归为同一**节点** | `nNodes` / `rankToNode` / 各节点 `localRanks` 布局（`:1291-1301`）——"哪些 rank 同机"由此划定 |
| 本地设备数 | min / max 对比 | 不一致 → WARN 甚至报错（`NCCL_IGNORE_NET_MISMATCH`，`:1324-1344`） |
| 能力与带宽 | `isAllNvlink` 取 AND；`minNetBw` 取 min | 运行时调优表；CollNet/NVLS 若聚合后 channel 数为 0 则整体禁用（`:1452-1453`） |

> **表中前两行最抽象，以 2 机 × 4 卡为例展开**：
>
> - **`topoRanks` = 每人申报"我这一段"**：图搜索给出的 `intra` 是本机机内链序，Preset（`connect.cc:20-93`）让每个 rank 申报 `ringRecv/Send`（本机链的**入口/出口**）、`ringPrev/Next`（机内我的**前/后**）、`treeToParent/Child0/1`（本机树段的父/子）——**跨机那段谁也没申报**。Postset 收齐后缝合（`connect.cc:103-110`）：节点 n 的入口的 prev ← 上一节点的出口、出口的 next ← 下一节点的入口，再把缝好的 prev/next 写回各 rank 的 `channel->ring`，产出 `rings[]` 交给 `setupChannel`。例：A 机报"链 A1→A2→A3→A4"、B 机报"B1→…→B4"，缝合后全局环 A1→…→A4→B1→…→B4→A1，跨机边只有 A4→B1、B4→A1 两条。
> - **`ringRecv[0]` = 节点指纹**：它是本机链的入口 rank；同机所有 rank 必然报同一个值，不同机器几乎必然不同 → **值相同的一组 = 同一节点**（`init.cc:1291-1301`），组内第一个 rank 即 `nodesFirstRank[node]`（缝合时的节点锚点）。不用 hostHash 划分，是因为这里的"节点"是图意义概念——"环上连续的一段"，与缝合所需锚点是同一个量。

**第 3 次交换的产物，就是阶段四建连所需的全部全局参数**——建几条 channel、每条环序如何、每对邻居是谁、缓冲区多大、NVLS/CollNet 开不开。没有它，阶段四的每一步都会因为"各说各话"而失败。

**为什么必须再交换一次？** 一个自然的疑问是："大家的输入不都一致吗，算出来的结果不应该自动一样？"——**前提就不成立**：

1. **各 rank 的搜索输入并不相等**。图搜索的输入 = 全局 GPU 拓扑（一致，来自第 1 次交换）+ **本机网卡与 CPU**（不一致！）。网卡的数量、带宽、GPU-NIC 亲和是纯本地事实：异构集群里 A 机 4 张 400G、B 机 2 张 200G，两边算出的 `bwInter`、`nChannels` 天然不同。源码甚至为"网卡数不一致"准备了**报错路径**（`init.cc:1324-1344`，可用 `NCCL_IGNORE_NET_MISMATCH` 放行）——NCCL 从未假设大家算出来会一样。

2. **需要全局一致的量，正确值是"聚合值"，任何单个 rank 都算不出来**：
   - `nChannels` 取所有人的 **min**（`:1439`、`:1459`）——channel 数必须全环一致，只能就低；取完还要把 Preset 阶段多复制的 channel 搬回去（`:1460-1466`）；
   - 带宽估计取 **min**（`:1440-1442`）——图的真实性能由**最慢一跳**决定，而"最慢一跳在哪"只有把所有人的数据摆在一起才知道；
   - 能力类（`typeIntra`/`crossNic`）取 **max** = 最保守降级（`:1443-1445`）——只要有一个人不支持，整条 channel 就不能按那个模式建。

3. **一些全局结构只能从汇总数据里"长出来"**：比如"这个通信域有几个节点、每个节点由哪些 rank 组成"（`nNodes`/`rankToNode`，`:1291-1301`）要看到所有人的 `topoRanks.ringRecv[0]` 才能划分；ring/tree 的最终邻居也是 `ncclTopoPostset` 用所有人的 Preset 结果拼装的（`:1480`）。

> 这里也澄清一个容易产生的误解：Ring 的相邻关系**不靠"各自以为"**——阶段四建连时每对 prev/next 要直接握手确认，接错了根本连不上。真正需要全局对齐的是**结构参数**：否则会出现"有人建 8 条 channel、有人建 6 条""同一 channel 上收发双方对带宽/协议的预期不一致"这类握手之后才发现的错位。

### 2.6 阶段四：连接与资源 —— Channel / Connector / Transport 是怎么落地的

**职责**：把阶段三算出的"**逻辑**通信图"（Ring 的环序、Tree 的父子关系）翻译成**真实的连接与物理资源**——为每个 channel 分配连接表、为每条连接挑一种传输方式、和对端交换句柄、映射显存/网卡资源，最后把连接信息下发到 GPU。

这是整章最"硬"的一节，我们从结构层次讲起，再讲建链过程。

#### 2.6.1 全景：一条五层结构链

前面 1.5 节我们分别认识了这些结构体，现在把它们按**"逻辑 → 物理"的顺序串起来**：

```
ncclComm（通信器）                          ← 一个通信域的全部上下文
  └── ncclChannel channels[MAXCHANNELS]     ← 并行通道数组（nChannels 条）
        └── ncclChannelPeer* peers[nRanks]  ← "连接表"：下标 = peer 的 rank
              └── ncclConnector send[2]/recv[2]   ← 连接子：一条具体连接
                    └── ncclTransportComm* transportComm   ← 走哪类物理链路
                          └── 物理资源（P2P 显存映射 / SHM 主机内存 / NET QP …）
```

逐层的职责与数量关系：

| 层 | 数据结构 | 职责 | 数量 | 代码位置 |
|----|----------|------|------|----------|
| **1. 通信器** | `ncclComm` | 一个通信域的全局上下文 | 1 | `comm.h:523-797` |
| **2. 通道** | `ncclChannel` | 一条独立的并行通信管道（承载一份数据切片） | `nChannels`（≤ `MAXCHANNELS`=64） | `comm.h:150-172` |
| **3. 连接表** | `ncclChannelPeer* peers[]` | "本 rank 在这个通道里，与各 peer 的连接载体"；**下标就是 rank 号** | 每通道 `nRanks` 项 | `comm.h:151` |
| **4. 连接子** | `ncclConnector` | 一条具体连接：走哪个传输、有哪些资源、进度指针在哪 | 每 peer `2 send + 2 recv` | `device.h:165-173` |
| **5. 传输虚函数表** | `ncclTransportComm` | 这条连接用**哪类**物理链路（P2P/SHM/NET/CollNet）及其操作集 | 全局 4 种实例 | `transport.h:117-134` |
| **6. 物理资源** | `p2pResources` / `shm*Resources` / `sendNetResources` … | 真正的东西：显存映射、主机内存、网卡 QP | 每条连接一份 | 各 `transport/*.cc` |

> **这个层次的职责分工**：**通信器**划定"谁能和谁说话"；**通道**把数据切成几份并行处理；**连接表**用 rank 号索引"我和每个人的连接"；**连接子**描述"这一条连接"；**传输**决定"这条连接走什么物理链路"；**物理资源**才是内存里真实的句柄、地址和队列。

#### 2.6.2 第一步：建通道，同时分配它的"连接表"

阶段三算出了每个 channel 的 ring 环序，接下来按它建通道并分配连接表（`src/init.cc:791-808`）：

```cpp
// src/init.cc:791-793（节选）
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  NCCLCHECK(initChannel(comm, channelId));   // 1. 分配该通道的连接表（下面展开）
  struct ncclRing* ring = &comm->channels[channelId].ring;
  ...                                        // 2. 按 ringRanks 填 ring->userRanks / rankToIndex / index
}
```

`initChannel` 里分配连接表的关键几行（`src/channel.cc:17-39`）：

```cpp
// src/channel.cc:17-39（节选）
int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
for (int r = 0; r < nRanks; r++) {
  channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
  ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
}
```

要点：

- `peers` 是**按 rank 编号索引**的指针数组，长度是 `nRanks + 1（CollNet 用）+ 本节点 GPU 数（NVLS 用）`；
- **此刻表里的每一项都还是"空连接载体"**——`ncclChannelPeer` 里的 connector 尚未建连（`connected == 0`）。真正填内容是接下来的步骤；
- 数组元素来自 `comm->sharedRes->peers[channelId]`（**共享资源**），并用 `refCount` 计数——所以 `ncclCommSplit` 出来的子通信器能复用父通信器已分配的这些载体（呼应 1.5 节 `sharedRes` 的作用）；
- 同一段代码还分配了 **device 侧镜像** `channel->devPeers`（`src/channel.cc:42-56`）——连接信息最终要出现在 GPU 能读到的地方，所以 host 侧和 device 侧各有一套。

**那行指针运算到底在做什么？** 关键是看清这里其实有**两级表**：

**第一级：全局共享表 `sharedRes->peers[channelId]`**（`channel.cc:32-34`）

```cpp
ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks);
// = 给 channel channelId 分配一块连续内存，装 tpNRanks 个 ncclChannelPeer 结构体
//    下标 = 顶层（父）通信器的全局 rank
```

这是**真正的连接状态存储**：按"顶层全局 rank"索引、每 channel 一份。它挂在 `sharedRes` 上而不是挂在单个 comm 上（源码注释原话：*"this can be shared between communicators hence should not be tied to comm"*）——父通信器和共享资源的子通信器（split 时 `shareResources`）看到的是**同一份**；`if (... == NULL)` 的判断保证**第一个到达的 comm 分配、后来者直接复用**。

**第二级：本 channel 的"指针目录" `channel->peers`**（`channel.cc:35`）

`ncclMemoryStackAlloc` 分配的只是 **nPeers 个指针**——注意它是"指针的数组"，不是结构体数组。

于是那行赋值就是一次"**编号翻译 + 指针定位**"：

```cpp
channel->peers[r] = sharedRes->peers[channelId] + comm->topParentRanks[r];
// 等价于
channel->peers[r] = &sharedRes->peers[channelId][ topParentRanks[r] ];
```

含义：本 channel 里"我视角下的第 r 个对端"，指向全局表里下标为 `topParentRanks[r]` 的那条记录。为什么要翻译、不能直接用 `r` 做下标？因为**子通信器（split 出来的）的 rank 编号从 0 重排过**，与顶层全局 rank 对不上；共享表按全局 rank 索引，必须经 `topParentRanks[r]`（本子通信器 rank r 在顶层通信器里的编号）做一次映射。

**这样设计的意义**：同一通信域内，"指向同一个对端"的所有引用收敛到**同一份 `ncclChannelPeer`**——无论多少个 comm、多少个 channel 引用它，连接状态只有一份（single source of truth），不会出现两份副本不一致；`refCount` 记录"有多少个目录条目指向它"，最后一个使用者退出时才释放（`init.cc:333-338` 里对 `sharedRes->peers[c]` 的 free 就发生在共享引用归零后）。

#### 2.6.3 第二步："登记"要建哪些连接

Ring/Tree 的建链入口会先**登记**（而不是立刻建）需要建立的连接（`src/transport/generic.cc:24-28`）：

```cpp
// src/transport/generic.cc:24-28（Ring，节选）
for (int c = 0; c < comm->nChannels; c++) {
  struct ncclChannel* channel = comm->channels + c;
  NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
}
NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
```

`ncclTransportP2pConnect`（`src/transport.cc:44-69`）做的事很轻：按 ring 的 `prev`/`next`（Tree 则按 `up`/`down[]`）把"我要和哪个 peer 建连"记进 `comm->connectSend` / `connectRecv` 两个**位图**（每个 bit 代表一个 channel），并跳过 `connected` 已为 1 的连接。

> **为什么先"登记"再"建"？** 因为建连是**成对**的：A 要连 B，B 也要连 A。先把全体要建的连接汇总成位图，才能高效地做"批量握手"（下一步的 setup/connect 会遍历这些位图，一次把同 peer 的多条连接一起处理），避免每个 channel 各自来回通信。

#### 2.6.4 调度中枢：`ncclTransportP2pSetup` 把登记变成连接

上一节的登记只写了两个位图；真正把登记变成连接的，是**所有拓扑共用**的执行器 `ncclTransportP2pSetup`（`src/transport.cc:123-287`）。Ring/Tree/PAT 三种拓扑最终都调用它，区别只在登记内容（位图）和传入的 `graph`。

它是"**两段式建链**"的执行者。NCCL 把"建立一条连接"拆成**两段**，对应 `ncclTransportComm` 里的两个函数指针：

| 段 | 函数 | 做什么 | 是否跨 rank 通信 |
|----|------|--------|------------------|
| **setup** | `transportComm->setup(...)` | 生成"**让对方能连上我**"的连接信息（交换数据），并创建 `transportResources`（物理资源容器） | ✅ 交换（用阶段二的控制面） |
| **connect** | `transportComm->connect(...)` | 拿到对端的连接信息后，**真正完成映射/连接**，把结果填进 `connector->conn` | 部分（如 NET 需改 QP 状态） |

下面的三步流程正好对应这两段：**第一步执行 setup 段，第二步交换两段之间的连接信息，第三步执行 connect 段**。

```cpp
// src/transport.cc:123-130（节选）
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex) {
  struct ncclConnect** data;      // 每个"距离对"的 send/recv 交换数据
  ...
  int maxPeers = ncclParamConnectRoundMaxPeers();   // 每轮处理的对端数上限
```

**第一步：选路与 setup**（`transport.cc:149-186`）——按与对端的"距离"遍历，逐 channel 选传输：

```cpp
// src/transport.cc:149-185（节选）
for (int i = 1; i < comm->nRanks; i++) {                    // 按距离遍历所有对端（先近后远）
  int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
  int sendPeer = (comm->rank + i) % comm->nRanks;
  uint64_t recvMask = comm->connectRecv[recvPeer];           // ← 登记阶段留下的位图
  uint64_t sendMask = comm->connectSend[sendPeer];
  ...
  for (int c = 0; c < MAXCHANNELS; c++) {
    if (recvMask & (1ULL << c)) {                            // 该 channel 与该对端有 recv 连接
      NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p] + recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
    }
  }
  ...                                                        // send 方向同理
}
```

> **这两行为什么要这样写？** `recvPeer`/`sendPeer` 是沿 rank 编号环的正反两个方向各走 `i` 步：
>
> ```c
> int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;   // 反方向走 i 步
> int sendPeer = (comm->rank + i) % comm->nRanks;                  // 正方向走 i 步
> ```
>
> `+ nRanks` 防止 `rank - i` 变负，`% nRanks` 保证结果落在 `[0, nRanks-1]`。`i` 从 1 扫到 `nRanks-1`，两个方向合起来恰好枚举除自己外的全部 rank（以 rank 3、nRanks=8 为例：i=1 → send 4 / recv 2，i=2 → send 5 / recv 1……i=4 → send = recv = 7，即"正对面"，由 `:189` 的 `sendPeer == recvPeer` 分支合并收发）。
>
> **按距离逐轮处理，而不是一次性连所有 peer**，原因有二：
>
> - **对称轮次防死锁**：同一轮里，rank x 的 `sendPeer` 是 x+i，而 x+i 这一轮的 `recvPeer` 恰是 x——双方在同一轮互为对端、同时收发，bootstrap 的 send/recv 严格配对；若各 rank 以任意顺序互连，`bootstrapRecv` 可能阻塞在一个还没进行到该轮的对端上，造成互相等待；
> - **枚举与实际连接解耦**：`send`/`recv` 的命名来自 Ring 的数据流向，但这里只充当**枚举顺序**——真正连谁由登记位图（`connectRecv`/`connectSend`）决定：Ring 只连相邻、Tree 的父子可能相距很远、NVLS/P2P Direct 任意，距离枚举保证任何距离的对端都会被处理到，位图为空的组合自然跳过。
>
> 这也解释了 tag 为什么含 `i`（`(i << 8) + graph->id + 1`）：每一轮距离独立编号，bootstrap 消息不会跨轮混淆。

`selectTransport`（`transport.cc:20-41`）就是"按优先级选路 + 现场调 setup"：

```cpp
// src/transport.cc:27-37（节选）
for (int t = 0; t < NTRANSPORTS; t++) {                      // 顺序：P2P → SHM → NET → CollNet
  struct ncclTransport* transport = ncclTransports[t];
  int ret = 0;
  NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));   // 这对 rank 之间走不走得通？
  if (ret) {
    connector->transportComm = transportComm;                // 1. 定下这条连接用哪套传输
    NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
    //                                                       ↑ 2. 调它的 setup，产出 ncclConnect 交换数据
    return ncclSuccess;
  }
}
```

> **补充：`ncclTransports[]` 里其实还有第 5 个成员 `profilerTransport`，但选路循环根本不会碰它。** 数组声明是 `NTRANSPORTS + 1` 项，而 `selectTransport` 只遍历前 `NTRANSPORTS` 项（P2P/SHM/NET/CollNet）；且它的 `canConnect` 干脆是 NULL——整个结构里只实现了 `recv` 侧的 `profilerProxyConnect` / `profilerProxyProgress` 两个函数（`src/transport/profiler.cc:57-61`），没有 `setup`、没有 `connect`，不参与任何数据搬运。
>
> **它存在的目的是复用 Proxy 框架来轮询 Profiler 插件的计数器**。计数器需要被周期性读取并上报，而"周期性后台轮询 + 生命周期管理 + 错误处理"正是 Proxy 线程已经解决的问题。NCCL 的做法是：借用 Transport 的结构外壳，把这两个函数挂在 proxy 槽位上——于是计数器轮询就成了一种特殊的 proxy op，完全复用既有的线程管理与调度（呼应 5.3 节）。代价是多了一个不参与通信的伪传输，换来的是无需另造一套轮询机制。

> **为什么找到第一个可用的就直接返回，不继续比较后面的？** 这是"静态选最优、不维护多路径"的设计决策，分三层理解：
>
> **1. 选路粒度决定了"一种"**。选路的粒度是 `(peer, channel, connIndex) → 一个 Connector → 一种 Transport`（`NCCL_MAX_CONNS = 2`，`device.h:231`，即每个 peer 的 send/recv 各只有 connIndex 0/1 两个槽位）。设备内核通过 `connector->transportComm` 这一个指针调用收发函数，**只能指向一种传输**——同一个 Connector 同时挂多种传输在结构上就不成立。
>
> **2. 不维护多路径：收益为零、代价显著**。初始化时拓扑已定：P2P 可用就选 P2P（带宽最高），不可用才降级 SHM、NET——最优链路已经确定且够用，运行时不存在"换路"的需求。而若同一对 peer 并行建多条传输：要付出双份资源（QP、缓冲区），内核热路径上要做路径选择，接收端还要处理多路径乱序——全是开销，没有收益。
>
> **3. "多种连接"体现在其他维度，而不是同一个 Connector 里**：
>
> | 维度 | 效果 |
> |------|------|
> | 不同 **channel** | 可绑不同物理链路（如不同网卡），形成 channel 级并行 |
> | 同一 peer 的两个 **connIndex** | 0 = 集合通信、1 = 点对点，两者**各自独立选路**，可走不同 transport |
> | 不同 **peer** | 各自独立判断——同机走 P2P/SHM，跨机走 NET |
>
> `canConnect` 的语义是"**这对 peer 之间能否走该传输**"（依据 2.5.1 换来的 `peerInfo` 与拓扑判断）；`selectTransport` 对每个 Connector 独立做一次"选最优"。而"多路径"的需求，被 NCCL 拆成了"多 channel / 多 connIndex / 多 peer"三个正交维度，而不是在一个 Connector 里塞多种传输。

setup 产出的连接信息是一个**不透明缓冲区**：

```cpp
// src/include/transport.h:70-72
struct ncclConnect {
  char data[CONNECT_SIZE];   // 内容由各 transport 自己解释
};
```

> 为什么设计成不透明缓冲区？因为不同传输要交换的东西完全不同：P2P 交换的是**显存缓冲区的 IPC/cuMem 句柄**，NET 交换的是**网卡 handle（含 QP 信息）**，SHM 交换的是**共享内存 IPC 描述符**。用固定大小的 `char[]` 统一交换格式，传输层就能横向插拔——这正是 4 种 transport 能共用一套流程的原因。

**第二步：经 bootstrap 交换连接信息**（`transport.cc:188-221`）——与每个对端互换各自的 `ncclConnect`：

```cpp
// src/transport.cc:201-219（节选）
if (recvChannels) bootstrapSend(comm->bootstrap, recvPeer, tag, recvData[p], ...);   // 把"连我的方式"发给对端
if (sendChannels) bootstrapSend(comm->bootstrap, sendPeer, tag, sendData[p], ...);   // 把"我要连你的方式"发出去
if (sendChannels) bootstrapRecv(comm->bootstrap, sendPeer, tag, sendData[p], ...);   // 收回对端"允许我连入的方式"
if (recvChannels) bootstrapRecv(comm->bootstrap, recvPeer, tag, recvData[p], ...);
// tag = (i << 8) + graph->id + 1 —— 不同拓扑轮次的交换数据互不混淆
```

**第三步：connect 落地 + 下发 GPU**（`transport.cc:224-287`）——骨架是"**攒满一轮对端 → 按位图逐连接落地 → 存在进行中就整轮重扫**"：

```cpp
// src/transport.cc:224-287（节选）
if (i - done == maxPeers || i == comm->nRanks - 1) {      // 1. 攒满一轮对端（或已到最后一个）
  bool allChannelsConnected = false;
  while (!allChannelsConnected) {                          // 2. 存在 inProgress 就整轮重扫
    allChannelsConnected = true;
    for (int j = done + 1; j <= i; j++) {                  // 3. 本轮内的每个对端
      ...
      for (int c = 0; c < MAXCHANNELS; c++) {              // 4. 逐 channel
        if (sendMask & (1ULL << c)) {                      //    位图登记过才连
          struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
          if (conn->connected == 0) {                      //    尚未连上
            NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
            if (ret == ncclSuccess) {
              conn->connected = 1;                         //    5. connect 成功，标记已连
              CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex],
                                            &conn->conn, sizeof(struct ncclConnInfo),
                                            cudaMemcpyHostToDevice, hostStream), ret, fail);   // 6. 连接信息同步下发 GPU
            } else if (ret == ncclInProgress) {
              allChannelsConnected = false;                //    该连接异步进行中 → 整轮重扫
            }
          }
          sendDataOffset++;
        }
      }
      ...                                                  // recv 方向同理（recvMask / recvData / recvPeer）
    }
    ...
  }
  done = i;                                                // 7. 本轮全部完成，推进已处理边界
}
```

读这段代码的三个要点：

- **`connect` 的输入是对端交换数据**：`sendData[p] + offset` 就是第二步从对端 bootstrap 收到的 `ncclConnect`（对端 `setup` 的产物）；`connect` 用它完成真正的资源映射——P2P 映射显存、NET 激活 QP、SHM attach 共享内存；
- **`connected` 标志 + 整轮重扫**：`ncclInProgress` 不算错误，只表示"该连接尚未就绪"——`while` 会重扫本轮全部连接，直到每一条都 `connected == 1`；
- **host / device 双份连接信息**：GPU 内核只能读取 device 可访问的内存，host 侧那份 `conn->conn` 是普通堆内存，内核物理上读不到——内核运行时需要的参数（对端缓冲区基址、fifo、协议类型）必须放进 device 侧的 `devPeers`。host 份也不能省：`connect` 过程本身、proxy 线程和后续的资源释放都在 host 侧操作它。连接信息在建连后不再变化，所以建连结束时一次 `cudaMemcpyAsync` 同步两份即可，此后各司其职——host 管理生命周期与进度，device 供内核读取。

> 另外注意 `:286-299` 有一段进度打印：`NCCL_REPORT_CONNECT_PROGRESS` 开启且 rank 0 时，每秒输出一次"P2p connect: xx% Remaining ..."——大集群建连时你在日志里看到的进度条就是它。

**各拓扑是怎么组织起来的**（`init.cc:1561-1594`，`ncclCommInitRankFunc` 阶段四）：

```
setupChannel（全部 channel 建连接表）
→ ncclTransportRingConnect    // 每 channel 登记 ring.prev / ring.next → P2pSetup(graph=RING)
→ ncclTransportTreeConnect    // 每 channel 登记两轮（子连父、父收子）→ P2pSetup(graph=TREE)
→ ncclTransportPatConnect     // 仅单 GPU 节点：按 mask=1,2,4… 登记距离对 → 正反两个方向各一次 P2pSetup
→ ncclNvlsSetup / ncclNvlsTreeConnect
→ CollNet setup（若启用）
→ 连接本地 NET proxy（SharedInit）
```

三个 Connect 变体（`src/transport/generic.cc`）没有公共逻辑要复用，**差异只在"登记哪些邻居"**：

| 拓扑 | 登记内容 |
|------|---------|
| `ncclTransportRingConnect` | 每 channel 一对邻居：`ring.prev` / `ring.next` |
| `ncclTransportTreeConnect` | 每 channel 两次登记：先"以 `tree.down` 为 recv、`tree.up` 为 send"，再反向一次（树的父子要双向各建一条） |
| `ncclTransportPatConnect` | 仅 `maxLocalRanks == 1`（单卡节点）专用：按 `mask = 1, 2, 4…` 的二进制距离逐对登记，ReduceScatter / AllGather 两个方向各登记一次、各跑一次 P2pSetup |

> **PAT 是什么？** PAT = **Parallel Aggregated Trees**（NVIDIA 在 NCCL 2.23 引入的算法，论文 arXiv:2506.20252），**只用于 AllGather / ReduceScatter** 两种集合操作。它是 **Bruck 算法的变体**（NVIDIA 官方博客原话），按 rank 执行"偏移二项树"，与递归翻倍类算法相比的优势是**可在任意数量的 rank 上运行**（不要求 2 的幂）。核心特征：**网络交换步数只有 log₂(节点数)**——每一步与距离 `2^k` 的对端交换，而 Ring 需要线性步数；官方预期**小到中消息**场景收益最大。这也解释了它的连接模式：`mask = 1, 2, 4…` 正是二项树的每一步，`±mask` 距离的对端就是该步的交换对象；ReduceScatter（全量收敛到分片）与 AllGather（分片扩展到全量）是互逆的两个方向，所以各建一套连接。
>
> **为什么当前仅单 GPU 节点（`maxLocalRanks == 1`）？** 启用检查在 `ncclPatEnable`（`tuning.cc:216-223`），源码注释原话就是 `// PAT only supports 1 GPU per node`（条件 `nNodes != nRanks` 则禁用）。这是当前实现的**阶段性边界**——PAT 把每个节点当作一个参与者做节点间对数步交换，节点内的聚合/分发尚未覆盖；并非算法原理上不能支持多 GPU 节点。另外两个附加条件：`minCompCap >= 60`（内核用到 CUDA atomics）、不支持 net device offload。
>
> **这个限制的实际影响**：多 GPU 节点用户完全不受影响——`ncclPatEnable` 返回 0 后调优器直接跳过 PAT，AllGather/ReduceScatter 照常走 Ring/NVLS（由延迟模型决定，`tuning.cc:429-433`：PAT 的模型是 `log2(节点数)` 次网络往返 + 线性项，带宽打 75 折）。它面向的目标场景是**大规模单卡节点集群**（rank 数 = 节点数，PAT 提供对数步数选项）；是否真用由调优器按拓扑与消息大小自动决定。

**分工总结**：**拓扑知识（连谁）在各 Connect 变体里，执行流程（怎么连）在 P2pSetup 里**。新增一种拓扑只需写一个 Connect 变体登记邻居，P2pSetup 一行不改——这是"逻辑与物理分离"在连接阶段的又一次体现。

> 补充：`comm->runtimeConn`（cuMem 支持且参数开启）时走另一条路——只做 `setupChannel` + NVLS/CollNet，P2P 连接推迟到运行时建立（`init.cc:1552-1560`），这里不展开。

#### 2.6.5 连接子里面到底装了什么：`ncclConnInfo`

`connect` 的成果全部体现在 `connector->conn`（类型 `ncclConnInfo`，`src/include/device.h:133-151`）里。这个结构体是**内核真正使用的东西**，值得逐字段看：

| 字段 | 含义 |
|------|------|
| `buffs[NCCL_NUM_PROTOCOLS]` | 三个协议（LL/LL128/SIMPLE）各自的**缓冲区地址**——发送方向指向对端显存（或本地发缓冲），接收方向指向本地缓冲 |
| `tail` / `head` | **进度指针**：head 由发送侧推进、tail 由接收侧推进；GPU 内核与 Host Proxy 线程就靠这两个指针交换进度状态（呼应 5.3 节的 Proxy 机制） |
| `connFifo` | GPU ↔ Proxy 之间传递控制信息的**环形缓冲** |
| `ptrExchange` / `redOpArgExchange` | 直连（direct pointer）模式下的**指针交换区**与 reduce 参数交换区 |
| `mhandles[]` / `netDeviceHandle` | 内存**注册句柄**（RDMA 的 MR）、网卡设备句柄 |
| `flags` | 连接属性（如 `NCCL_P2P_WRITE` / `NCCL_P2P_READ` / `NCCL_DIRECT_NIC`） |
| `stepSize` / `step` / `llLastCleaning` | 流水线的步进与内部状态 |

**以 P2P（同机 GPU 直连）为例，看 connect 具体填了什么**（`src/transport/p2p.cc:536-557`）：

```cpp
// src/transport/p2p.cc:536-557（节选）
struct p2pResources* resources = (struct p2pResources*)send->transportResources;
// 1. 把对端的进度标记内存映射进本进程地址空间
NCCLCHECK(p2pMap(comm, &send->proxyConn, comm->peerInfo + rank, comm->peerInfo + info->rank, &info->p2pBuff,
                 (void**)&remDevMem, &resources->recvMemIpc));
// 2. 切分缓冲区：把对端显存里的各协议缓冲区首地址，逐个填进 conn.buffs[]
char* buff = (char*)(remDevMem + 1);
for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
  send->conn.buffs[p] = buff;
  buff += comm->buffSizes[p];
}
send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
```

> 这两句就是 **"P2P 直连"的全部魔法**：`p2pMap` 把对端显存映射到了本进程的地址空间（用的是 CUDA IPC / cuMem / 或对端在同进程时直接给指针），于是 `conn.buffs[p]` 就是**一个本进程可写的、指向对端显存的地址**——之后 GPU 内核用普通的 `st.global` 指令写这个地址，数据就出现在对端显存里了。**不需要网卡、不需要 CPU 搬运**。

#### 2.6.6 四类传输各自对应哪些物理资源

| 传输（虚函数表实例） | 常量 | resources 结构 | 物理资源 | 代码位置 |
|---------------------|------|----------------|----------|----------|
| `p2pTransport` | `TRANSPORT_P2P` (0) | `p2pResources` | 对端显存映射（IPC/cuMem 句柄）、进度标记内存（`ncclSendMem`/`ncclRecvMem`） | `p2p.cc:80-95` |
| `shmTransport` | `TRANSPORT_SHM` (1) | `shmSendResources` / `shmRecvResources` | **主机内存**（共享内存段）+ IPC 描述符 | `shm.cc:28-42` |
| `netTransport` | `TRANSPORT_NET` (2) | `sendNetResources` / `recvNetResources` | 网卡通信对象（**QP**）、内存注册句柄、GDR 描述符、网卡设备句柄 | `net.cc:89-137` |
| `collNetTransport` | `TRANSPORT_COLLNET` (3) | CollNet 相关资源 | 交换机侧 SHARP 资源 | `coll_net.cc` |
| （`profilerTransport`） | `TRANSPORT_PROFILER` (4) | — | 仅为轮询 profiler 计数器而存在，不是真正的传输 | — |

其中 P2P 内部还细分为四种模式（`src/transport/p2p.cc:20-25` 的 `enum p2pType`），由 `p2pGetInfo` 依据拓扑和 cuMem 可用性自动选择：

| 模式 | 含义 |
|------|------|
| `P2P_DIRECT` | 对端显存可直接访问（NVLink / C2C / 同进程） |
| `P2P_INTERMEDIATE` | 两者不能直连，需**经一张中间的 GPU 中转** |
| `P2P_IPC` | 同主机不同进程，用传统 CUDA IPC 共享显存 |
| `P2P_CUMEM` | 用 cuMem API（CUDA 12+ 虚拟地址管理，支持 MNNVL 多节点 NVLink） |

而 NET transport 与网卡的交互完全通过 **Net Plugin API**（`src/include/plugin/nccl_net.h`，v12）：**setup** 阶段向对端提供网卡的 `handle`（内含 QP 信息），**connect** 阶段把对端 QP 置为 RTS（Ready To Send）状态——自研 RDMA 协议栈也是通过这个接口接入的。

#### 2.6.7 第四步：收尾 —— Proxy 线程、调优器与 device 侧镜像

先澄清一个容易混淆的点：**connect 完成的只是 rank 之间的互联，它不等于"通信器可用"**。connect 结束后还有几件**本 rank 内部**的收尾工作；而且时序和直觉不同——Proxy 线程其实在建链**之前**就启动了（`init.cc:1519-1696`）：

**1. Proxy 线程先于建链启动**（`init.cc:1524-1530`）：

```cpp
// Launch proxy service thread, after this, the proxy calls can be used.
NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);   // ← 在 setupChannel / Connect 系列之前
```

原因：接下来的建连过程就要用到它——NET transport 连接完成后要立刻通过 proxy 做 `SharedInit`（把 `p2pnChannels` 等共享参数注册给 proxy，`init.cc:1590-1593`）。Proxy 是本 rank 的常驻服务（host↔GPU 数据搬运），与"和哪个 rank 建链"无关，所以先起服务、再建链。

**2. Connect 系列**（connIndex=0，rank 间互联）：Ring → Tree → PAT（单卡节点）→ NVLS → CollNet；随后 NET 通过 proxy 做 SharedInit。若开启 NVB preconnect，还会用 `ncclTransportP2pSetup(comm, NULL, 1)` 预连一批点对点连接（connIndex=1，`:1607-1632`，可选路径）。

**3. 建链全部完成后的本 rank 收尾**（均不涉及对端交互）：

- **调优器就位**（`:1638-1644`）：加载 tuner 插件，`ncclTopoTuneModel` 把各算法/协议的延迟与带宽模型算好——运行时选算法/协议靠它；
- **`devCommSetup`**（`:1690`）：把 `ncclComm` 的关键结构镜像到 GPU 显存（`devComm`）——GPU 内核运行时读的是 device 份；
- **机内 barrier**（`:1694`）：同机 rank 互相同步一次，确认所有缓冲区分配完毕。

到 `devCommSetup` + 机内 barrier 完成，通信器才正式"就绪"。

> **connIndex 的分配**：`0` 给集合通信连接（Ring/Tree/CollNet 共用），`1` 给点对点 Send/Recv——这正是 1.5 节 `NCCL_MAX_CONNS = 2` 的由来。

#### 2.6.8 回看这条链：一次调用的完整旅程

把本章四个阶段串成一条线：

```
ncclGetUniqueId
   → 生成引导句柄（root IP:port + magic）
ncclCommInitRank → ncclCommInitRankDev
   → 分配 ncclComm 空壳（含 abortFlag、魔数哨兵）
ncclCommInitRankFunc（异步任务体，真正干活）
   ├─ 阶段 2 bootstrapInit ──────────── 建带外控制面（TCP/UDS）
   ├─ 阶段 3 AllGather1【全体】 ──────── 交换 peerInfo（成员身份与位置信息）
   ├─ 阶段 3 ncclTopoGetSystem（内部含机内 XML 融合【仅机内】）
   │    └─ ncclTopoCompute ─────────── 搜索 Ring/Tree/CollNet/NVLS 图（纯本地计算）
   ├─ 阶段 3 AllGather3【全体】 ──────── 交换图参数，min/max 聚合出全局一致
   └─ 阶段 4 Connect 系列与收尾
             ├─ ncclProxyCreate        起 Proxy 线程（建链期间就要用）
             ├─ channels[c].peers[]    分配连接表（按 rank 索引）
             ├─ ncclTransportP2pConnect 登记要建的连接（位图）
             ├─ setup（交换句柄 via bootstrap）
             ├─ connect（映射显存/QP，填 conn，下发到 device）
             └─ 调优器 + devCommSetup + 机内 barrier
   → 通信器可用，可以开始 ncclAllReduce
```

> **这条链体现的设计哲学**：**逻辑与物理分离**。阶段三产出的是纯粹的"逻辑图"（谁和谁相邻），阶段四才把它落到具体传输和资源上。好处是：换一种物理链路（比如国产网卡、自研互联）只需要替换阶段四的 transport 实现，前面的拓扑/图逻辑完全不用动。

---

## 三、集合通信的执行路径

前两章拆的是 NCCL 的**静态骨架**：一是核心结构体的形状（通信器 / 通道 / 连接子 / 传输层），二是一个通信器如何从零被建起来——探测拓扑、搜索图、逐条建立连接、映射物理资源。第二章结束时，连接已经全部就绪，但"就绪"只意味着**可以使用**，还没有真正被使用过。

**NCCL 的网络从来不是静态的。** 前面看到的每一个设计，目的都是为集合通信的执行服务：`channels[]` 是为了把数据切开、并行搬运；`connector` 是为了让 GPU 内核知道往哪读、往哪写；协议（LL/LL128/Simple）是为了匹配不同消息大小下的延迟与带宽诉求。脱离执行过程去看这些结构，只能知道它们"长什么样"，看不出"为什么长成这样"。

所以从本章开始，视角切换到**调用方**：用户（或 PyTorch）调用一次 `ncclAllReduce` 之后，NCCL 要把这个调用翻译成什么、经过哪几层、最终在 GPU 上跑起来的又是什么。本章按这条链路依次展开：

| 小节 | 关注点 |
|------|--------|
| 3.1 | 从 `ncclAllReduce` 到内核启动：API → enqueue → schedule → launch |
| 3.2 | 任务分组与 Planner：多个调用如何合并成一次 launch |
| 3.3 | Kernel Launch 与 Device 侧调度：内核内部怎么分工执行 |
| 3.4 | Transport 层数据路径：数据沿哪条链路真正流动 |

也是从这里开始，"集合通信"这件事才真正发生：前两章建立的网络被实际驱动起来，而第二章讨论的拓扑、算法选择、带宽估计，也才有了实际作用的载体。

### 3.1 从 ncclAllReduce 到内核启动

以 AllReduce 为例的完整执行链：

```
ncclAllReduce
  → ncclEnqueueCheck                     （enqueue.cc）
      ├─ ncclGroupStartInternal()        隐式 GroupStart，depth 0→1
      ├─ taskAppend → collTaskAppend     task 入 planner（只是登记）
      └─ ncclGroupEndInternal()          隐式 GroupEnd，depth 1→0
          └─ groupLaunch                 ← 启动器（group.cc）
              ├─ ncclPrepareTasks        选算法/协议、切 channel、产 proxy op
              ├─ ncclTasksRegAndEnqueue  注册 buffer、写 work
              └─ doLaunches
                  ├─ ncclLaunchPrepare   task → ncclKernelPlan（含 work batch）
                  └─ ncclLaunchKernel    ← cudaLaunchKernel，内核真正跑起来
  → Device Kernel
```

> 注意 `groupLaunch` **不是** `ncclAllReduce` 直接调用的——它藏在 `ncclGroupEndInternal` 里：阻塞分支在当前线程直接调用（`group.cc:862`）；非阻塞分支先起后台线程，由 `groupLaunchNonBlocking`（`:762`）再调。

**Step 1: API 入口**

```cpp
// src/collectives.cc:129-150
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff, count, datatype, op, 0, comm, stream, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
```

所有集合操作 API 都遵循相同模式：构造 `ncclInfo` 结构体，然后调用 `ncclEnqueueCheck`。

**Step 2: `ncclEnqueueCheck` —— 隐式 Group + 任务登记**（`src/enqueue.cc:3124-3172`）

先纠正一个容易误解的点：**NCCL 里并没有"不在 Group 就直接执行"的独立分支**。每次独立的集合调用，都会给自己包一层"隐式 Group"：

```cpp
// src/enqueue.cc（节选）
NCCLCHECK(ncclGroupStartInternal());                       // 1. depth: 0 → 1
...
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);    // 2. 把这次调用登记成一个 task
...
NCCLCHECK(ncclGroupEndInternal());                         // 3. depth: 1 → 0，在这里触发 launch
```

所以"在不在 Group 内"只决定 **depth 何时归零**：用户自己写了 `ncclGroupStart/End`，任务先攒着，等用户的 `ncclGroupEnd` 让 depth 归零；没写，本次调用的第 3 步就让 depth 归零、立即启动。**两种情况的后续代码路径完全一致**。

> 隐式 Group 的完整机制——`depth` 怎么增减、`== 0` 为什么是唯一启动点、为什么单次调用也走 Group——放到 **3.2 节**和 Planner 一起讲（它本就和"任务分组"是一体的）。这里只需记住：**`ncclEnqueueCheck` 是"登记入口"，真正启动发生在 `ncclGroupEndInternal` 里。**

2. 的 `taskAppend`（`:3014`）按操作类型分派，集合通信走 `collTaskAppend`（`:2690`）。它做的事就是"登记"，具体是**按数据量排序插入**：

```cpp
// src/enqueue.cc:2725-2748（节选）
struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, ...);
t->func = info->coll;      t->sendbuff = info->sendbuff;
t->recvbuff = info->recvbuff;  t->count = info->count;   // 填齐参数
...
planner->nTasksColl += 1;
ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);   // 按流量大小有序插入
```

注意最后一行：task 不是简单追加到队尾，而是**按 `trafficBytes` 有序插入 sorter**——后面的调度会优先处理流量大的任务。

**此刻什么都没启动**，只是 `comm->planner` 里多了一个 task。

**Step 3: `ncclGroupEndInternal` → `groupLaunch`：task 变 plan**（`src/group.cc:766` / `:598`）

depth 归零后进入 `ncclGroupEndInternal`；在默认的 blocking 分支里，它在**当前线程**调用 `groupLaunch`，依次做三件事：

| 顺序 | 调用 | 阶段 | 做什么 |
|------|------|------|--------|
| 1. | `ncclPrepareTasksAndCollPreconnect`（`:674`）→ `ncclPrepareTasks`（`:574`） | **schedule** | 遍历 planner 里的 task：选算法（RING/TREE/COLLNET/NVLS/PAT）与协议（LL/LL128/SIMPLE）、把数据切到各 channel、为每个 channel 生成 `ncclProxyOp`；若某条连接尚未建立，顺手发起建连异步任务 |
| 2. | `ncclTasksRegAndEnqueue`（`:690`） | 注册 + 入队 | 按需注册用户 buffer，把 work 描述写进 FIFO |
| 3. | `doLaunches`（`:723`） | **launch** | 把 plan 变成真正的 kernel 启动（见 Step 4） |

真正把 task 翻译成 `ncclKernelPlan` 的，是第 3 点里的 `ncclLaunchPrepare`（`enqueue.cc:1568`）：

```cpp
// src/enqueue.cc:1586-1639（节选）
struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(...);   // 新建一个 plan
...
NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);    // 从 sorter 里"取任务填 plan"
NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, &p2pEpoch, &p2pRound, plan, &budget), ...);
finishPlan(comm, plan);                                                          // 收尾
ncclIntruQueueEnqueue(&planner->planQueue, plan);                                // 挂到 plan 队列
...
planner->unlaunchedPlansHead = planHead;    // 已备好、待启动的链表头
```

一个关键设计：**plan 按"预算"切分**（`budget`）——不是所有 task 都塞进同一个 plan，而是按 work 大小分批，切出多少个 plan 就启动多少轮 kernel（`enqueue.cc:1612-1615` 的注释还解释了为什么必须先排空 coll 任务：否则切分点会因 rank 而异）。

**Step 4: `doLaunches` —— 逐个 plan 启动 kernel**（`group.cc:309-384`）

```cpp
// 每个 comm：先把 task 全部转成 plan
NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);          // :323

while (true) {                                                    // 一轮启动一个 plan
  struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead; // :353 取出一个待启动的 plan
  if (plan != nullptr) {
    comm->planner.unlaunchedPlansHead = plan->next;
    ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan);          // :357 前置准备（barrier 等）
    ncclLaunchKernel(comm, plan);                                 // :363 ← cudaLaunchKernel 在这里
    ncclLaunchKernelAfter_NoCuda(comm, plan);                     // :369 收尾
  }
  if (!moreRounds) ncclLaunchFinish(comm);                        // :374 最后一轮：清空 planner
}
```

到这里，"一次 `ncclAllReduce` 的调用"才真正变成"一个在 GPU 上跑的 kernel"。

**小结**

```
用户调用 ──taskAppend──▶ ncclTaskColl（参数登记，按流量排序）
                              │ ncclLaunchPrepare + scheduleCollTasksToPlan
                              ▼
                        ncclKernelPlan（work batch 集合，按预算分批）
                              │ doLaunches 逐轮启动
                              ▼
                        CUDA Kernel（设备端执行）
```

一次调用要经过两级中间对象才落到内核：先由 `taskAppend` 登记成 `ncclTaskColl`，再经 `ncclLaunchPrepare` 打包成 `ncclKernelPlan`，最后由 `doLaunches` 启动为 CUDA Kernel。

> 关于"为什么单次调用也要走 Group"、depth 语义与配对规则，见 **3.2 节**。

### 3.2 任务分组与 Planner

**本节目的**：回答 3.1 留下的两个问题——"调用被登记进了哪里""什么时候统一启动"。一次 `ncclAllReduce` 不会直接跑起来，而是先变成 `comm->planner` 里的一个 task；多个调用（或嵌套 Group）攒在一起，由最外层 Group 边界统一启动。本节拆两半：**Group 攒批机制**（谁决定"一起启动"）和 **`ncclKernelPlanner`**（任务以什么形态被组织）。

**本节在整条链上的位置**：

```
taskAppend（登记到 planner）
      │  （若干次调用，可能被用户 Group 包裹）
      ▼
ncclGroupEndInternal → groupLaunch          ← 最外层边界统一启动
                          ├─ ncclPrepareTasks（选算法/协议）
                          └─ doLaunches → ncclLaunchPrepare（task → plan）
```

3.1 里反复出现的"隐式 Group""depth 归零"，机制在 3.2.1 展开；任务到 plan 的组织在 3.2.2。

#### 3.2.1 Group 攒批：`ncclGroupDepth` 全景

`ncclGroupDepth` 是 **thread_local 计数器**（`group.cc:27`，注释原话 "depth of ncclGroupStart nesting"），全仓**只有两个函数**会改它：

```cpp
// 自增：src/include/group.h:87-90
inline ncclResult_t ncclGroupStartInternal() { ncclGroupDepth++; return ncclSuccess; }

// 自减：src/group.cc:788
if ((--ncclGroupDepth) > 0) goto exit;    // 还没退到最外层 → 直接返回，不启动
```

**谁在加减**：

| 调用方 | 加减时机 |
|--------|---------|
| 用户 `ncclGroupStart()` / `ncclGroupEnd()` | 显式开 / 关 Group |
| `ncclEnqueueCheck`（`:3137` / `:3164`） | **每次独立 API 调用**的隐式包装 |
| `ncclCommInitRankConfig`（`init.cc:2667` / `:2679`） | 初始化通信器也走 Group 攒批 |
| 框架 / 库内部包装 | 与用户 Group 形成嵌套 |

**为什么需要 depth，而不能只用一个 bool**

1. **嵌套安全**：Group 可以嵌套，而 NCCL 自己就在制造嵌套（`ncclCommInitRankConfig` 内部就包了一层）。若内层 `End` 也启动，就会把外层**尚未登记完**的批次提前 launch，破坏"一批一起启动"的语义。
2. **单发与批量共用一条路径**：单发调用是 `0→1→0`，立即启动；用户 Group 内是 `1→2→1`，不启动，等外层 `1→0` 统一启动。同一套代码覆盖两种用法，无需维护两个分支。
3. **攒批是正确性要求，不只是性能**：所有 rank 必须在同一个 Group 边界把任务登记完再一起启动——否则会出现"某 rank 的内核已经在等对端，另一 rank 的任务还没下发"的错位（p2p 任务对此尤其敏感）。

**`== 0` 的意义：唯一允许"启动"的状态**

0 表示"当前线程不在任何 Group 内"。这个判断在代码里出现在四处，构成完整语义：

| 位置 | `depth == 0` 时做什么 |
|------|----------------------|
| `ncclGroupEndInternal`（`group.cc:788`） | 自减后归零 → 越过最外层边界 → **构造 groupJob、触发 `groupLaunch`** |
| `ncclAsyncLaunch`（`group.cc:40`） | 不在 Group 内 → 异步任务**直接同步执行**，不入队 |
| `ncclGroupErrCheck`（`group.h:96-100`） | `depth > 0` 时把错误**暂存**进 `ncclGroupError`，等 End 时统一处理——错误也要攒批 |
| `ncclGroupEndInternal`（`:775`） | 在 0 处再 End → 配对错误，`WARN("ncclGroupEnd: not in a group call.")` |

**"到处都在 Start" 的真相：每一层都成对、互相抵消**

| Start 位置 | 配对 End | 净效果 |
|-----------|---------|--------|
| 用户 `ncclGroupStart()` | 用户 `ncclGroupEnd()` | **真正决定攒批窗口的就是这一对** |
| `ncclEnqueueCheck`（`:3137`） | `ncclEnqueueCheck`（`:3164`） | 单发时 `0→1→0`，立即启动 |
| `ncclCommInitRankConfig`（`:2667`） | 同函数（`:2679`） | 同理 |

深度轨迹（单发 vs 用户 Group）：

```
单发一次 ncclAllReduce
  ncclEnqueueCheck: Start   0 → 1
  taskAppend                      （登记）
  ncclEnqueueCheck: End     1 → 0    → 立即启动

用户写 Group，内含两个 allreduce
  ncclGroupStart             0 → 1        ← 外层攒批生效
    ncclAllReduce #1        1 → 2 → 1     （内层抵消，不启动）
    ncclAllReduce #2        1 → 2 → 1     （内层抵消，不启动）
  ncclGroupEnd               1 → 0        ← 两个 task 一起 launch
```

**为什么每个 API 都要自己包一层**：为了让"单发"和"批内调用"复用同一段代码，而不是写 `if (在组内) 只登记; else 登记 + 启动;` 两套分支——包一层之后**照常登记、照常 End**，是否此刻启动由"End 后 depth 是否为 0"自动决定，调用方无需知道自己是否在批内。这也是 `ncclCommInitRankConfig` 自带一对 Start/End 的原因：否则 depth 恒为 0，`ncclAsyncLaunch` 会直接同步执行，非阻塞初始化能力就失效了。

**为什么必须是计数器而不是 bool**：要支持嵌套。若用 `bool inGroup`，内层 End 会把它清成 `false`，可外层还开着——再进来的调用就会误判"可以启动"，把尚未登记完的批次提前 launch。计数器天然支持"层数抵消"。

> **一个容易混淆的点：一个 Group 里可以装两类东西，走两条队列。** 初始化类任务通过 `ncclAsyncLaunch` 进 **`ncclAsyncJobs`** 队列（`group.cc:31`，init / preconnect 等异步 job）；集合通信任务则通过 `taskAppend` 进各自的 **`comm->planner`**（见 3.2.2）。二者共享同一个 depth 计数器和 `groupLaunch` 启动入口，但队列、任务类型、处理路径都不同——这正是"初始化网络"和"allreduce"会在代码上"重合"的原因：它们只是复用同一套攒批机制，不会混进同一批。

#### 3.2.2 `ncclKernelPlanner`：任务的调度核心

`ncclKernelPlanner`（`src/include/comm.h:429-502`）是**每个 comm 各自一个**的调度核心，负责把"已登记的 task"组织成"可启动的 plan"：

- `peers[R]`：按 Peer 组织的 P2P 任务队列（send/recv/bcast）
- `collSorter`：按数据量排序的集合任务（就是 3.1 里 `collTaskAppend` 写入的那个）
- `wipPlan.channels[C]`：每个 Channel 的进行中 Plan 状态
- `planQueue`：已完成的 Kernel Plan 队列，等待 Launch

它与 3.1 的衔接关系：`taskAppend` 把 task 写进 `collSorter` / `peers[]` → `ncclPrepareTasks` 做算法/协议选择 → `ncclLaunchPrepare` 里的 `scheduleCollTasksToPlan` 把 task 装进 `ncclKernelPlan` → 挂 `planQueue` → `doLaunches` 逐个启动。

### 3.3 Kernel Launch 与 Device 侧调度

**本节目的**：3.2 结束时，`ncclKernelPlan` 已经备好（挂在 `unlaunchedPlansHead`）。这一节追"最后一公里"：plan 里的内核函数指针 + 参数，怎么变成 GPU 上一个正在跑的内核；内核启动后，又怎么找到"该干的那份活"。分两端看：Host 侧启动、Device 侧调度。

**Host 侧：`ncclLaunchKernel`（`src/enqueue.cc:1753`）**

```cpp
void* sym = plan->kernelFn;                      // 编译期特化好的内核符号
dim3 grid  = {(unsigned)countOneBits(plan->channelMask), 1, 1};  // 每个 channel 一个 block
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
cudaLaunchKernel(..., sym, grid, block, plan->kernelArgs, ...);  // ← 真正的启动点
```

三个要点：

- **grid.x = 参与本次的 channel 数**：一个 block 负责一个 channel——channel 是并行执行的最小单元；
- **`kernelFn` 是"特化"出来的**：`DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, ...)`（`device/common.h:438-441`）为每种 (集合操作, 数据类型, 规约, 算法, 协议) 组合都生成了一个内核入口，启动前就定死了，而不是运行时 if/else；
- 启动前后各有一步：`ncclLaunchKernelBefore_NoUncapturedCuda`（`uploadWork`，把 work 写进 device 侧的 FIFO，`enqueue.cc:1740`）和 `ncclLaunchKernelAfter_NoCuda`（收尾）。

**Device 侧：从 `ncclDevKernel_Generic` 到 `ncclKernelMain`**

统一入口（`src/device/common.cu:23`）：

```cpp
__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}
```

`ncclKernelMain`（`device/common.h:355`）三步：

1. **映射 block → channel**：按 `channelMask` 找第 n 个置位 bit，`blockIdx.x == n` 的那个 block 接管 channel（`common.h:370-373`）；
2. **把 comm / channel / work 搬进共享内存**：前两个 warp 加载 `ncclKernelComm` / `ncclDevChannel`，其余 warp 用 `loadWorkBatchToShmem` 从 work FIFO 读本 channel 的 work（`common.h:382-414`）；
3. **按 `funcId` 分发，并循环处理 work**（`common.h:417-431`）：

```cpp
while (ncclShmem.aborted == 0) {
  if (0 <= SpecializedFnId && ncclShmem.funcId == SpecializedFnId)
    SpecializedRunWorkBatch().run();        // 特化入口直接跑
  else
    ncclDevFuncTable[ncclShmem.funcId]();    // 通用入口：查函数表分发
  if (ncclShmem.nextBatchIx == -1) break;    // 没有下一批 work → 退出
  loadWorkBatchToShmem(tid, tn, args, batchIx);  // 加载下一批
}
```

**`funcId` 决定"跑哪个集合操作"**：`ncclDevFuncTable` 是函数指针数组，每个元素由宏 `DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto)`（`common.h:446-449`）生成，内部调 `RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run()`。`funcId` 是 host 侧 `ncclPrepareTasks` 选好算法/协议后编码进 work 的——于是"算法/协议选择"这一步，最终落成"取函数表哪个下标"。

**协议表**（协议在 host 侧选好、以模板参数固化进内核，列出来是为了理解"为什么有这么多内核变体"）：

| 协议 | 适用数据量 | 特征 |
|------|-----------|------|
| LL | 极小消息 | 单线程处理，低延迟；8 字节对齐（`NCCL_LL_ALIGNMENT_PER_THREAD`） |
| LL128 | 小到中等 | 单 Warp 处理，128 字节粒度流水线；480 字节 Warp 对齐 |
| SIMPLE | 大消息 | 多 CTA 并行，利用全部带宽；16 × Warp × 8 × 16 对齐 |

> 一句话：**Host 把"算法 / 协议 / 类型"在编译期特化成一个内核符号，用 grid/block/args 启动它；内核内按 `funcId` 查 `ncclDevFuncTable` 分发到对应集合操作的模板实现**。协议表不是运行时的 if/else，而是"到底生成了多少个内核变体"的根源。

### 3.4 Transport 层数据路径

**本节目的**：前 3 节追的是"任务怎么从一次调用变成 GPU 上正在跑的内核"——到 3.3 结束时内核已经启动，但**数据还没有真正流动**。这一节补上最后一块：内核跑起来之后，字节到底怎么从这块 GPU 走到那块 GPU。它把第二章建好的连接在这一刻"用起来"——`ncclConnector` 里的 `conn`/`connFifo`、head/tail 进度指针、P2P 映射、NET QP，前面都是"建好了但没用"，这里就是它们被内核消费的地方。换句话说：**3.1~3.3 是"控制流"（怎么调度），3.4 是"数据流"（怎么搬运）**，两者合起来才是一条完整的集合通信执行路径。

内核侧的搬运统一由 `Primitives`（`src/device/prims_simple.h`）封装：它持有 `ncclConnInfo`（`conn`）和 `ncclConnFifo`（`connFifo`），进度靠 `conn->head`/`conn->tail` 两个指针推进（`st_relaxed_sys_global(connStepPtr, step)`），数据本身分两条路：**同机 P2P 直写**、**跨机经 Proxy 走 NET**。

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

**GIN = GPU Initiated Network，即"由 GPU 发起的网络通信"。** 它回答一个具体问题：跨节点的数据搬运，能不能不经过 Host CPU、而是由 GPU kernel 里的线程直接发起？

回顾第三章的 NET 路径：GPU 内核算完 → 更新 `head`/`tail` 进度标记 → Host 侧的 Proxy 线程轮询到这个标记 → 调用 `ncclNet->isend()/irecv()` 提交 RDMA → 网卡才真正开始搬数据。这条链里，**通信的"发起权"在 Host**：GPU 只能把数据准备好、把标记写好，然后等 Host 来"接棒"。每次从"GPU 算完"到"网卡开动"，中间隔着 Host 的轮询周期和线程调度，这对 MoE、KV cache 这类延迟敏感场景已经不够。

GIN 把发起权下放给 GPU：kernel 内的线程直接调用设备端 API 提交 Put/Get/信号量。收益有两点：**消除 Host-Device 往返与线程调度延迟**；**让"计算-通信重叠"从 kernel 级细化到线程级**——同一个 kernel 里可以一边算一边发，而不必等整个 kernel 结束、下一轮 proxy 介入。

NVIDIA 之所以在 NCCL 与自家硬件上做这件事，是因为硬件前提已经具备：Volta（SM 7.0）起的 GPU 能直接写网卡的发送队列或描述符。GIN 的作用，就是把这份硬件能力包装成软件可调的 API。它对软硬件配合方式的改变是根本性的——过去是"GPU 写标记、Host 读标记、Host 驱动网卡"，现在变成"GPU 直接驱动网卡（GPI/GDAKI），或写描述符交 Proxy 代发（PROXY）"。为此 NCCL 在两层各加了一套结构：**主机侧**建连接、注册窗口（`ncclGinState` + `ncclGin_t` 插件接口），**设备侧**发起操作（`ncclGinPut`/`ncclGinGet` 等 + 三种 backend 的分发）。

本章按以下顺序展开：先讲清楚"为什么需要 GIN"（4.1），再看它落地的三种硬件形态（4.2），然后分别拆主机侧（4.3）与设备侧（4.4）的代码骨架，最后看谁在推进这些异步操作（4.5）以及与 Host 发起模式的对比（4.6）。

### 4.1 GIN 解决什么问题：从 Host 发起谈起

NCCL 的跨节点通信默认由 Host 发起。以一次 send 为例，完整时序是：

```
GPU kernel：写数据 → 写 head 标记（ncclSendMem.head）
      │
Host Proxy 线程：轮询 head → 发现推进 → 调用 ncclNet->isend()
      │
网卡：DMA 读取数据 → 发出 → 写完成队列（CQ）
      │
Host Proxy 线程：轮询 CQ → 更新 tail / credit
```

这条链有两个延迟来源：**Host-Device 往返**（GPU 要等 Host 看到标记，Host 提交后 GPU 还要等网卡完成）和**轮询/调度延迟**（Proxy 线程的轮询周期受 CPU 调度影响）。它们叠加起来，决定了"GPU 准备好数据"到"网卡真正开动"之间的固定开销。

GIN 的目标就是砍掉这段。对比一下发起路径：

| | 传统（Host 发起） | GIN（GPU 发起） |
|---|---|---|
| 谁提交网络操作 | Host Proxy 线程（`ncclNet->isend`） | GPU kernel 内线程（`ncclGinPut` 等） |
| GPU 需要做什么 | 写 head 标记后**等 Host** | 直接提交，提交完继续算 |
| 重叠粒度 | kernel 级（等 kernel 结束 + proxy 介入） | 线程级（kernel 内边算边发） |
| 典型延迟 | Host-Device 往返 + 线程调度 | 仅 GPU 到网卡的总线写 |

这就是 GIN 存在的根本原因：**把通信发起权从 Host 移进 GPU kernel**，从而让计算与通信的 overlap 细到线程粒度。

### 4.2 三种 Backend：同一套 API，三种硬件落点

"GPU 直接驱动网卡"有几种物理实现方式，NCCL 把它们抽象成三个 backend。设备侧 API 是同一套（`ncclGinPut`/`ncclGinGet` 等），区别只在每个 backend 怎么把这套 API 落到硬件上：

| Backend | 枚举值 | GPU 实际做什么 | 硬件前提 |
|---------|--------|---------------|----------|
| **GPI**（GPU-Push Interface） | `NCCL_NET_DEVICE_GIN_GPI` | 只提交 64 字节抽象描述符（GFD），WQE 生成与报文构造由 NIC 固件接管 | Spectrum-X 平台的固件能力，当前为实验特性 |
| **GDAKI**（GPUDirect Async Kernel-Initiated，常写作 GDA-KI） | `NCCL_NET_DEVICE_GIN_GDAKI` | GPU 自行构造完整 RDMA WQE，提交到 QP 发送队列并敲 doorbell | ConnectX-6 Dx 及以上 + DOCA |
| **PROXY** | `NCCL_NET_DEVICE_GIN_PROXY` | 把请求写成描述符放进共享内存，Host Proxy 线程代发 | 无（最通用，也是缺省回退） |

两个缩写的官方出处：`GPI` 见 `docs/userguide/source/api/device_setup.rst:229-233`（"GPU-Push Interface (GPI) ... Requires SpectrumX ... Added as an experimental feature in NCCL 2.30.6"）；`GDAKI` 见 `docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md:57`（"GPUDirect Async Kernel-Initiated (GDA-KI) ..."，该术语源自 NVSHMEM，NCCL 沿用）。GPI 的含义是 GPU 线程把描述符直接推入网卡可见的 MMIO 队列（`bindings/nccl4py/nccl/core/typing.py:154`）；GDAKI 则强调通信由 CUDA kernel 内的线程异步发起，而非 Host。

**GPI 与 GDAKI 的本质区别：由谁把传输请求翻译成网卡可执行的 WQE 与报文。** GDAKI 沿用标准 InfiniBand verbs 模型，GPU 线程经 DOCA GPUNetIO 的设备端函数（`doca_gpu_dev_verbs_put/get/signal`，`gin_gdaki.h:100-120`）自行构造完整 WQE（地址、rkey、长度）写入 QP 发送队列，QP 选择与报文格式由 GPU 侧控制——内部实现 `ncclGinIbGdaki` 正是复用标准 `ncclNetIb`（`src/transport/net_ib/gin.cc:314`）。GPI 把这一步卸载：GPU 只写一条 64 字节 GFD（操作码、目的 PE、窗口句柄与偏移）到 `gpu_memic_ptr` 环形队列（`gin_gpi.h:180-228`），后续的 QP 选择、WQE 生成、报文构造全部由 NIC 固件完成。由此形成取舍：GDAKI 控制粒度细、生态成熟；GPI 的 GPU 侧提交开销更小，但控制粒度更粗，且依赖更新的硬件——源码中内置插件只有 GDAKI 与 PROXY（`gin.cc:245-254`），GPI 需外部插件，可用 `NCCL_GIN_TYPE` 显式指定（`transport/net_ib/gin.cc:46`）。

类型定义在 `src/include/nccl_device/core.h:168-173`，枚举值特意与设备类型对齐（`src/include/nccl_device/net_device.h:18-24`）：

```cpp
typedef enum {
  NCCL_GIN_TYPE_NONE = 0,
  NCCL_GIN_TYPE_PROXY = 2,  // = NCCL_NET_DEVICE_GIN_PROXY
  NCCL_GIN_TYPE_GDAKI = 3,  // = NCCL_NET_DEVICE_GIN_GDAKI
  NCCL_GIN_TYPE_GPI = 4,    // = NCCL_NET_DEVICE_GIN_GPI
} ncclGinType_t;
```

三个 backend 的编译开关是独立的（`src/include/nccl_device/gin/gin_device_common.h:24-38`）：PROXY 恒启用；GPI/GDAKI 要求 `CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700`。运行时到底用哪个，由 `ncclGinState.ginType` 记录——它在插件分配时写入（见 4.3），设备侧分发时据此选择（见 4.4）。

### 4.3 主机侧：从插件加载到连接与窗口建立

前面讲了 GIN 是什么、有哪几种硬件形态，这一节看主机侧代码怎么把它变成"可用"。先说清时机：**这些准备工作并不在同一个阶段完成**。

- **通信器初始化期**：第二章的阶段四里，`ncclCommInitRank` 会顺带调用 `ncclGinInit`（`init.cc:477`，与 `ncclNetInit`/`ncclRmaInit` 并列）。这一步只加载插件、选定一个后端，**不建立任何连接**——因为此时还不知道上层库会不会用 GIN。
- **上层库激活期**：当上层库（如 DeepEP）调用公共 API `ncclDevCommCreate`（`src/dev_runtime.cc:1502`）申请设备侧通信上下文、并在 `reqs` 里声明需要 GIN 时，才触发剩下的三步——建连接、注册内存窗口、创建设备侧 context（`ginActivated` 控制，首次激活时执行一次，`dev_runtime.cc:1137`）。

这也解释了为什么 GIN 不是 AllReduce 的主路径：它由上层库按需激活，而不是每个通信器初始化时都建一套。按代码执行顺序，主机侧共四步：

| 步骤 | 时机 | 入口函数 | 产物 |
|------|------|----------|------|
| 1. 加载并选定插件 | 通信器初始化 | `ncclGinInit`（`plugin/gin.cc:269`） | `ginState` 里记下用哪个后端 |
| 2. 建立 rank 间连接 | `ncclDevCommCreate` | `ncclGinConnectOnce`（`gin/gin_host.cc:80`） | `ginComms[]` |
| 3. 注册对称内存窗口 | `ncclDevCommCreate` | `symMemoryRegisterGin` → `ncclGinRegister`（`dev_runtime.cc:1152`、`gin_host.cc:374`） | `ginHostWins` / `ginDevWins` |
| 4. 创建设备侧 context | `ncclDevCommCreate` | `ncclGinDevCommSetup`（`dev_runtime.cc:1244`、`gin_host.cc:190`） | 下发到设备侧的句柄；按需起 progress 线程 |

四步都读写同一个结构 `ncclGinState`（`src/include/gin/gin_host.h:27-48`，挂在 `sharedRes` 上，因此父子通信器共享同一份）：

```cpp
struct ncclGinState {
  ncclGin_t* ncclGin;                              // 插件接口（v14）
  void* ginInstance;                               // 插件 init 出来的实例
  ncclGinType_t ginType;                           // GPI / GDAKI / PROXY
  int ginCommCount;                                // 建了几条 GIN 连接（≤ 4）
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];        // 每条连接的 collComm
  int needsProxyProgress;                          // 是否需要 Host 线程推进
  int ginProgress;                                 // 0 暂停 / 1 运行 / -1 退出
  std::thread thread; std::mutex mutex; std::condition_variable cond;
  struct ncclGinStateDevComm* devComms;            // 设备侧上下文链表
  ncclGinConnectionType_t ginConnectionType;       // FULL / RAIL
};
```

**第一步：加载并选定插件 —— `ncclGinInit`（`src/plugin/gin.cc:269`）**

要理解这一步，先要明白"插件"指什么。NCCL 不直接实现操作网卡的代码，而是把 GIN 需要的一组操作定义成函数指针表 `ncclGin_t`（`src/include/plugin/gin/gin_v14.h:27-67`）：`listen`、`connect`、`createContext`、`regMrSym`、`ginProgress` 等。**提供这组函数的一个实现，就是一个 GIN 插件**。三种 backend 之所以能共用同一套流程，是因为它们都以插件形式交出这组函数；`ginState->ncclGin` 指向选定的那一个，后面三步全都通过它调用。

为什么需要"遍历"？因为 NCCL 在启动前不知道这台机器上哪个后端能用——取决于装了什么网卡、有没有厂商提供的库。`initPluginLibsOnceFunc`（`gin.cc:194-256`，`std::call_once` 只构造一次进程级列表）按顺序列出四类候选：

1. **外部插件**：`NCCL_GIN_PLUGIN` 环境变量指定的动态库（默认 `libnccl-gin.so`）——厂商的 GPI 实现就走这条路；
2. **NET 插件自带的 GIN 支持**（若该 NET 插件提供）；
3. **内置 `ncclGinIbGdaki`**：基于标准 IB verbs 的 GDAKI 实现，无需外部库；
4. **内置 `ncclGinProxy`**：PROXY 回退实现，同样无需外部库。

后两个是 NCCL 自己编进来的，保证即使没有任何外部库，GIN 也至少能回退到 GDAKI 或 PROXY。

`ncclGinInit` 的遍历逻辑（`gin.cc:269-304`）：先检查 `compCap >= 70`（Volta 起支持 GIN）；然后对列表中每一项，若是外部插件就先 `dlopen` 取 v14/v13 符号，再调用 `init(&ginContext, commHash, ...)` 初始化、`devices(&ndev)` 探测是否有可用设备——`ndev <= 0` 就标记禁用、继续看下一个。**第一个成功启用的插件**由 `ncclGinPluginAssignToComm`（`gin.cc:133-174`）把 `ncclGin` 和 `ginType`（取自该插件 `getProperties(0).netDeviceType`）写进 `ginState`，随后禁用其余外部插件并结束遍历。用户可用 `NCCL_GIN_TYPE` 强制指定后端类型（`gin.cc:141-148`）；选不中任何插件时 `ncclGinState` 保持空，GIN 不可用。

**第二步：建立连接 —— `ncclGinConnectOnce`（`src/gin/gin_host.cc:80`）**

这一步才真正把 rank 之间的 GIN 连接建起来：

```cpp
// gin_host.cc:100-172（节选）
ginState->ncclGin->devices(&ndev);                          // 有 GIN 能力的设备数
ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs); // 本机 GIN 设备
...
for (int n = 0; n < ginState->ginCommCount; n++) {
  ginState->ncclGin->listen(ginInstance, localGinDevs[n % nLocalGinDevs],
                            allHandles + ..., &listenComm);   // 1. 生成监听句柄
  ginState->ncclGin->getProperties(..., ginState->ginProps + n);
  bootstrapAllGather(comm->bootstrap, allHandles, ...);       // 2. 全体交换句柄
  ginState->ncclGin->connect(ginContext, handles, nGinRanks, myGinRank,
                             listenComm, ginState->ginComms + n);  // 3. 建立连接
  ginState->ncclGin->closeListen(listenComm);
}
```

三个要点：**`ginCommCount` 是全体协商出来的**（每个 rank 上报自己的数，`bootstrapAllGather` 后取 min，上限 `NCCL_GIN_MAX_CONNECTIONS = 4`）；**连接是多路的**（`ginComms[]` 数组，对应多网卡/多 rail）；**复用阶段二建好的 bootstrap** 交换句柄——与 NET transport 建连是同一套路（呼应 2.6.4）。

**第三步：注册对称内存窗口 —— `ncclGinRegister`（`gin_host.cc:374`）**

连接建好后，还要把"哪些内存可以被远端访问"登记下来。GIN 通信的对象是窗口（window）：`ncclGinRegister` 对一块地址调 `regMrSym`，拿到 host 侧句柄（`ginHostWins`）和设备侧窗口（`ginDevWins`）——设备端 `ncclGinPut` 用的 `dstWin`/`srcWin` 就是这里注册出来的。`ncclDevCommCreate` 在首次激活时会遍历已有的对称内存逐一注册（`dev_runtime.cc:1150-1154`，源码注释说明窗口句柄要等拿到 stream 后再更新）；跨多个物理 segment 的地址需要 DMABUF 支持（`gin_host.cc:379-387`）。

**第四步：创建设备侧 context —— `ncclGinDevCommSetup`（`gin_host.cc:190`）**

这一步为上层库创建的 `ncclDevComm` 分配设备侧 context 并下发句柄，核心是循环 `createContext`（`gin_host.cc:277-292`），之后**按需启动 progress 线程**：

```cpp
// gin_host.cc:294-298
if (ginState->needsProxyProgress && ginState->ginProgress == 0) {
  ginState->ginProgress = 1;
  ginState->thread = std::thread(ncclGinProgress, ginState);  // 只有需要才起线程
}
```

progress 线程**不是 GIN 初始化就启动**，而是由所选 backend 在 `createContext` 时声明需要才起（详见 4.5）。

**四步完成后，Host 与 Device 的分界就确定了。** Host 侧在 kernel 启动前完成了全部控制面准备：选定插件、建立 rank 间连接（`ginComms[]`）、注册对称内存窗口（取得 rkey/lkey 与窗口句柄）、分配设备侧 context（QP 数组或 GFD 队列、signal/counter 表、共享内存队列指针）。这些结果通过 `ncclDevComm` 的字段下发给设备侧：

```cpp
// src/include/nccl_device/impl/comm__types.h:48-57（节选）
uint8_t ginConnectionCount;
uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];  // 每条连接使用哪个 backend
void*   ginHandles[NCCL_GIN_MAX_CONNECTIONS];         // 设备侧 context 的地址
int     ginSignalCount, ginCounterCount;
uint64_t* ginSignalShadows;
uint32_t ginContextCount;
```

这些字段由 `ncclGinDevCommSetup` 从 `devHandles[n]->handle` 填入（`gin_host.cc:289-291`）；插件文档也明确区分了两者：`ginCtx` 是 host 侧描述符，`ncclNetDeviceHandle_t.handle` 是设备侧描述符（`plugins/gin/README.md:127-131`）。设备侧构造 `ncclGin_C` 时，`ncclGinInitCommon` 直接从 `ncclDevComm` 读出 backend 类型与句柄（`src/include/nccl_device/impl/gin__funcs.h:81-98`），不再向 Host 查询。

因此 kernel 执行期间不需要 Host 参与控制面操作：连接、内存注册、队列与信号的分配都已就绪，且窗口句柄里的 rkey/lkey、context 里的 QP 与队列地址都是 GPU 可直接读写的地址，GPU 线程按 4.4 的 API 提交请求即可。这正是 GIN 能消除 Host-Device 往返的前提。

数据面则需要区分 backend。GPI 与纯 GPU 直驱的 GDAKI 下，网络操作的提交与完成完全由 GPU 侧完成；PROXY 下 GPU 只把描述符写入共享队列，仍由 Host progress 线程代发；GDAKI 在 QP 需要 CPU 代理时同样如此。更准确的表述是：**控制面准备一次完成并下发到设备侧，kernel 不再为此与 Host 交互；数据面是否经过 Host，取决于所选 backend。**

### 4.4 设备侧：GPU kernel 里怎么发起一次通信

4.3 结束时，Host 侧已经完成连接建立、内存注册与设备侧 context 分配，并把 backend 类型、句柄写进了 `ncclDevComm`。这些字段是留给 GPU kernel 用的：上层库在自己写的 kernel 里调用 `ncclGinPut` 时，正是通过它们确定该往哪个 QP 或哪条队列提交请求。本节站到这条分界的另一侧，追一次设备侧发起的通信调用怎样一步步落到硬件上。

设备侧代码之所以分成三层，来自一个必须解决的矛盾：上层库需要一套与硬件无关的稳定接口，而 4.2 讲的三种 backend 在设备侧要做的事完全不同——GDAKI 由 GPU 组装 WQE 并投递到 QP，GPI 写 64 字节 GFD 交给网卡固件，PROXY 把描述符放进共享内存等 Host 代发。如果让上层库直接面对这些差异，每换一种 backend 就得改调用代码。NCCL 的做法是把一次调用拆成三段：一层统一 API 承接所有调用，一层按 `backendMask` 与设备侧 context 里记录的 backend 分发，剩下的差异全部收进各 backend 的模板特化里。

于是沿一次调用的流向，读代码时依次要回答三个问题：kernel 里能调用哪些操作；一次调用怎样找到 4.2 选定的那个 backend；找到之后最终写出的是标准 RDMA WQE 还是 GFD 描述符。下表把这三个问题与代码中对应的层次列出，后续三小节按此顺序展开：

| 层 | 回答的问题 | 代码位置 |
|----|-----------|---------|
| 第一层 · 对外 API | kernel 能调用哪些操作、操作对象是什么 | `src/include/nccl_device/gin.h` |
| 第二层 · 分发 | 一次调用怎样找到选定的 backend | `gin_device_common.h` 的 `ncclGinCall` / `ncclGinCallImpl` |
| 第三层 · 内部 API 与 backend 实现 | 选定之后写出 WQE 还是 GFD | `gin_device_common.h` 的 `ncclGinApi_*` 模板 + `proxy/`、`gpi/`、`gdaki/` 三个目录 |

三层走完之后，4.4.1 与 4.4.2 再往下一层追问 GDAKI 的具体参数：一次 `Put` 用哪条 QP（4.4.1）、远端地址与 rkey 从哪里取（4.4.2）——这部分直接说明 4.3 下发的句柄如何被消费。本节只覆盖「提交」这一侧；请求提交后如何确认完成、以及为什么部分 backend 还需要 Host 线程推进，放在 4.5。

先看最靠近调用方的第一层。上层库（如 DeepEP）在自己写的 kernel 里包含 `nccl_device/gin.h` 后，看到的就是下面这组函数；它们的签名带 `ncclGin_C*`（GIN 上下文）和 `ncclTeam`，不含任何 backend 相关参数，这正是「与硬件无关」在接口上的体现：

**第一层：对外 API（`src/include/nccl_device/gin.h`）**

```cpp
// gin.h:150-203（节选）
ncclGinPut(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, ...);     // 本地 → 远程
ncclGinGet(net, team, peer, remoteWin, remoteOff, localWin, localOff, bytes, ...); // 远程 → 本地
ncclGinSignal(net, team, peer, ..., signalId, signalOp, ...);                      // 只发信号
ncclGinFlush(net, coop, ord);                                                     // 冲刷未完成请求
ncclGinWaitSignal(net, coop, signal, least, bits, ord);                           // 等信号
```

操作对象是三类句柄：**窗口** `ncclGinWindow_t`（`void*`，`gin_device_host_common.h:18`）、**信号** `ncclGinSignal_t`（`uint32_t`，`nccl_device/core.h:29`）、**计数器** `ncclGinCounter_t`（`uint32_t`）。信号分强/弱两种语义（`gin.h:23-26`）：**Strong** signal 的可见性隐含"它之前所有 put 都已落定"；**Weak** 只保证捆绑的那次 put 落定——DeepEP 用 strong signal 做跨步同步、weak 做单次通知。

第一层的接口里没有 backend 参数，但同一份上层代码必须能在 GDAKI、GPI、PROXY 三种实现上运行。这个矛盾由第二层解决：对外 API 内部都汇到 `ncclGinCall`，再进 `ncclGinCallImpl` 分发到具体的 backend 实现。分发有两条依据——编译期常量 `beMask` 说明**这次编译把哪几种 backend 编进了内核**，`ctx.backend` 说明**当前这个 context 实际在用哪一种**（来自 4.3 下发的 `ginNetDeviceTypes`）。下面这段代码就是分发的全部逻辑，读的时候重点看前两行如何决定走哪条 case、以及每个 case 里的条件如何帮编译器裁掉无关代码；中文注释是笔记补注，代码本身保持源码原样：

**第二层：分发（`src/include/nccl_device/gin/gin_device_common.h`）**

```cpp
// gin_device_common.h:177-199（节选，中文注释为笔记所加）
template <template <ncclNetDeviceType> typename ApiFn, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCallImpl(unsigned beMask, ncclGinCtx ctx, Arg&&... arg) {
  // ApiFn 是第三层那个「按 backend 特化的操作模板」（如 ncclGinApi_Put）；
  // beMask 是编译期位掩码，第 k 位为 1 表示第 k 号 backend 被编进本内核。
  // 位号取 backend 枚举值：PROXY=2、GDAKI=3、GPI=4（net_device.h:18-24）。
  bool singleton = (beMask & (beMask - 1)) == 0;
  // x & (x-1) 会清掉 x 最低位的那个 1；结果为 0 说明 beMask 只启用了一种 backend。

  switch (singleton ? __popc(beMask - 1) : (int)ctx.backend) {
    // 只启用一种时无需读内存：__popc 数出 beMask-1 里 1 的个数，
    // 恰好等于唯一置位位的位号（也就是 backend 枚举值），编译期即可定死分支；
    // 启用多种时才退回运行时读取 ctx.backend。
#if NCCL_GIN_PROXY_ENABLE
  case (int)NCCL_NET_DEVICE_GIN_PROXY:
    // 该 backend 不在 beMask 中时告知编译器「此分支不可达」，便于整段裁掉。
    if (!(1 & (beMask >> (int)NCCL_NET_DEVICE_GIN_PROXY))) __builtin_unreachable();
    return ApiFn<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, static_cast<Arg&&>(arg)...);
    // 把调用参数原样转发给第三层该 backend 的特化实现。
#endif
    // GDAKI / GPI 两个 case 结构完全相同，各自受对应的 ENABLE 宏控制（:186-195）。
  default:
    __builtin_unreachable();
  }
}
```

把两个位运算展开说一下：设 `beMask = 1 << 3`（只编入 GDAKI），则 `beMask & (beMask - 1) = 8 & 7 = 0`，`singleton` 为真；同时 `beMask - 1 = 7 = 0b111`，`__popc(7) = 3`，正好得到 GDAKI 的枚举值，于是 switch 直接落到 GDAKI 分支，连 `ctx.backend` 都不必读。若编入了多种 backend，`singleton` 为假，才按 `ctx.backend` 在运行时选择。

`ncclGinCtx`（`gin_device_common.h:62-70`）携带 `backendMask`/`backend`/`rank`/`nRanks`/`handle`/`contextId`，其中 `backend` 与 `backendMask` 由 `ncclGin_C_makeCtx` / `_makeCtx` 从设备侧 context 填入（`impl/gin__funcs.h:126-147`）。`ncclGinCall` 有两个重载（`gin_device_common.h:201-209`）：一个传运行时的 `ctx.backendMask`，另一个传模板参数 `beMask`（编译期常量）。配合每个 case 里的 `__builtin_unreachable()`，未启用的 backend 代码会被编译器整段裁掉，内核里不留下运行时判断的开销。

分发层只负责「选谁」，并不产生任何网络请求。真正把一次 `Put` 变成 WQE 或 GFD 的，是被选中的那个 backend 的特化实现。为了让分发层能用统一的 `ApiFn<backend>::call(...)` 形式调用，NCCL 把每个操作也定义成按 backend 特化的模板结构，这就是第三层：

**第三层：内部 API（`gin_device_common.h:102-173`）**

下面贴出其中三个最常用操作的真实声明，中文注释为笔记所加。可以看到它们都是只带 `backend` 一个模板参数的结构体，成员函数 `call` 的第一个参数统一是 `ncclGinCtx`——这正是第二层能用 `ApiFn<backend>::call(ctx, ...)` 统一调用的原因：

```cpp
// gin_device_common.h:114-131（节选，中文注释为笔记所加）
template <ncclNetDeviceType backend>
struct ncclGinApi_Get {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags = ncclGinOptFlagsDefault);
  // 参数里已经带上远端窗口、本地窗口、偏移与长度——第一层 API 的语义在这里落成具体数据
};

template <ncclNetDeviceType backend>
struct ncclGinApi_Put {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault);
  // 除数据外还可附带 signal / counter：这解释了为什么 Put 的参数比 Get 多一截
};

template <ncclNetDeviceType backend>
struct ncclGinApi_Wait {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinRequest_t& outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag);
  // 等待异步请求完成，与 4.5 讲的「提交不等于完成」对应
};
```

同一文件里还有 `Flush` / `FlushAsync` / `PutValue` / `GetSignalPtr` / `GetCounterPtr` / `ResetSignal` / `ResetCounter` 等操作，声明形式相同。这里只是**声明**，每个操作的具体实现由三个 backend 各自特化提供（`gin_device_common.h:100-173` 只给出模板骨架）。

三个 backend 各自实现这套接口：`proxy/gin_proxy.h`、`gpi/gin_gpi.h`、`gdaki/gin_gdaki.h`。以 proxy 为例（`gin_proxy.h:68`），`ncclGinApi_Put<PROXY>::call` 最终调 `postGfd`——把请求写成一个 GFD 描述符放进共享内存（`proxyCtx->pis`/`cis` 是生产者/消费者索引），真正的网络操作由 4.5 的 progress 线程消费。

到这里，一次 `ncclGinPut` 的完整路径就走通了：统一 API 接收调用，分发层按 context 里的 backend 选中实现，第三层的模板特化把请求写成硬件可消费的形式。三层各自只解决一个问题，因此更换 backend 时上层库与第一层都不需要改动。接下来 4.4.1、4.4.2 沿着第三层的 GDAKI 实现继续往下看，分别追问一次 `Put` 具体用哪条 QP、远端地址与 rkey 从哪里取。

#### 4.4.1 QP 选择：设备侧怎么定位本端的发送队列

前三层回答的是「调用怎么走到 backend」。走到 GDAKI 这一层时，第一个具体问题是：一次 `Put` 用哪条 QP。这正好说明 4.3 里 Host 下发的那些句柄在设备侧如何被消费，因此放在设备侧这一节展开。以下以 GDAKI 为主，因为它由 GPU 自己选定 QP；另外两个 backend 的选择权不在 GPU。至于数据要写到对端哪块内存（远端地址与 rkey），放到 4.4.2 单独展开。

**三种 backend 的选择权归属**——同一套 `ncclGinPut` 调用，决定 QP 的位置完全不同：

| Backend | QP 由谁选 | 依据 |
|---------|-----------|------|
| GDAKI | GPU 内核 | 设备侧持有按 peer 索引的 QP 数组（`gdqp`），发起时自己取用并投递 WQE |
| GPI | 网卡固件 | GPU 只写 64 字节 GFD，字段为目的 PE 等抽象信息，不含 QP；WQE 生成与 QP 选择由固件完成 |
| PROXY | Host 侧 | GPU 写 GFD 到共享队列，progress 线程读出后交给 `rmaBackend->iput` / `iget` / `iputSignal`（`gin_host_proxy.cc:245-262`），QP 由 RMA 后端负责 |

**GDAKI：GPU 侧定位本地 QP**。设备侧 context 由 `ctx.handle` 指向，`ctx.contextId` 选第几个 context，`gdqp` 则是该 context 内按 peer 索引的 QP 数组：

```cpp
// src/include/nccl_device/gin/gdaki/gin_gdaki.h:59-61（节选）
ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;   // 按 peer 取本地 QP
```

逻辑上「GPU 侧有一个 `qp[peer]`」是成立的，物理形态是 `ncclGinGdakiGPUContext`（`gin_gdaki_device_host_common.h:25-36`）里位于 GPU 可见内存的 `doca_gpu_dev_verbs_qp` 数组——Host 在 `createContext` 阶段用 `doca_gpu_verbs_export_multi_qps_dev` 把这批 QP 导出到设备（`transport/net_ib/gdaki/gin_host_gdaki.cc:747`）。GPU 运行时按索引直接取，不经过 Host。

前面看到设备侧用 `ctx.handle[ctx.contextId].gdqp + peer` 定位 QP，但 `handle` 与 `contextId` 都不是凭空来的：它们由 `ncclGinInitCommon` 从一个 `contextIndex` 拆出。之所以要拆两级，是因为设备侧 GIN 资源本身就是两级组织的——先按「连接」分，再在每条连接内按「context」分。下面先说明两级各自对应什么资源，再看拆分规则。

**连接**对应 `ginConnectionCount`，其值是 `ginState->ginCommCount`（`gin_host.cc:215`），来源为 `ncclTopoGetLocalGinDevs` 给出的 `nLocalGinDevs`（`:113`、`:121`）。这里先说清「GIN 设备」指什么，因为它容易被误认为 GPU：**它是网卡侧的能力标记**。拓扑里 `NIC`/`NET`/`GIN`/`RMA` 是平级的节点类型（`src/graph/topo.h:45-50`），`ncclTopoAddNic` 遍历每个物理网卡下的 `<net>` 节点，只有 `gin="1"` 的才创建 GIN 节点（`topo.cc:488-506`、`:448-465`）；源码注释写明 NET/GIN/CollNet 共用同一批物理网卡节点（`:1811`）。以内置 GDAKI 插件为例，插件的 `devices(&ndev)` 遍历 IB 设备、只把 `ibProvider == IB_PROVIDER_MLX5` 的网卡计入（`transport/net_ib/gin.cc:63-69`、`:236-240`）——即「GIN 设备数」是支持 GPU 直驱的网卡数。

为什么不是每个 GPU 对应一个 GIN 设备？因为 `nLocalGinDevs` 来自 `ncclTopoGetLocalGinDevs`（`topo.cc:2084-2092`）：它以当前 rank 所在 GPU 为起点，按 channel 依次查**与该 GPU 有本地路径**的 GIN 节点，直到设备号重复为止，得到的是这块 GPU 就近可达的网卡数。`topo.cc:2056-2072` 给出分配规则：`localGpuCount` 块 GPU 共享 `localNetCount` 张网卡，AUTO 策略下 `netsPerGpu = DIVUP(localNetCount, localGpuCount)`，起始网卡由 `gpu.dev % localGpuCount` 决定，再按 channelId 轮转。8 GPU 配 8 网卡时接近 1:1，8 GPU 配 4 网卡时每块 GPU 就近只有 2 张，这个数由 PCIe/NVLink 拓扑决定，与 GPU 数量没有必然对应关系。

上限 4 来自 `NCCL_GIN_MAX_CONNECTIONS`（`gin_device_host_common.h:15`），它是 `ncclDevComm` 中固定长度数组的编译期容量（`ginHandles[4]`、`ginNetDeviceTypes[4]`，`impl/comm__types.h:48-50`），`gin__funcs.h:84` 的 `static_assert` 也依赖它做按位与取模——属于软件结构限制，不是硬件最多只能有 4 张网卡。此外 `gin_host.cc:130-134` 会 bootstrap 全 gather 后取所有 rank 的**最小值**：连接数必须全体一致，否则跨 rank 的通道与 QP 映射无法对齐。用户也可以通过 `NCCL_GIN_NCONNECTIONS` 显式指定（`:78`、`:127`）。

每条连接是一次 `listen`/`connect` 建起来的 `ginComms[n]`，绑定一个本地 GIN 设备（`:157-161`）；4.3 第四步的 `createContext` 循环对每条连接各调一次，产出的句柄分别写入 `devComm->ginHandles[n]`、backend 类型写入 `ginNetDeviceTypes[n]`（`:289-291`）。因此选 `connectionId` 决定的是：**这次操作走哪条 GIN 连接（哪张网卡）、用哪一份设备侧资源表、以及该连接的 backend 类型**。多条连接用于把流量分散到多张网卡。

**context**是连接内部的并发工作区。`ginContextCount` 是上层库申请的总数（DeepEP 里即 `num_allocated_qps`），Host 把它向上取整到连接数的倍数，得到 `nContextsPerComm = nContextsTotal / ginCommCount`（`gin_host.cc:210-225`），再作为 `ginConfig.nContexts` 传给该连接的 `createContext`（`:265-268`）。一个 context 拥有的独立资源因 backend 而异：GDAKI 里每个 context 一个 `ncclGinGdakiGPUContext`，含自己的 `gdqp`（QP 数组）、`counters_table` 与 `signals_table`，分配大小是 `ncontexts * NCCL_GIN_GDAKI_GPU_CONTEXT_MAX_SIZE`（`gin_host_gdaki.cc:516-518`），QP 总数为 `nqps_for_comm = ncontexts * nranks`（`:476-478`）；PROXY 里每个 context 一条独立的 GFD 队列与 `sis`/`cis` 索引（`gin_host_proxy.cc:32-38`、`:174`），signal 槽位按 `contextId * nSignalsPerContext` 分开（`:309`）。因此选 `contextId` 决定的是：**该连接内用哪一组 QP 或哪一条 GFD 队列、哪一段 signal 与 counter 槽位**。多个 context 让不同 CTA 各用一套队列与信号，避免争用同一个发送队列。

这里值得把「避免争用」的机制展开，因为它是多 context 设计的性能根源。先澄清一个易混点：**QP 不是「目的地相同」的副本，而是「每个 peer 一条」**——`gdqp + peer` 意味着一个 context 内 peer 0 到 peer N-1 各有一条 QP，每条两端固定配对，这是 RDMA 的 QP 语义。所以「多 context」带来的不是同一目的地的多条 QP，而是「每个 (context, peer) 组合各自一条独立 QP」。那么同一个 peer 为什么还要在多个 context 里各建一条？因为**一条 QP 是串行提交的共享资源**：它上面有一个发送队列 SQ、一个门铃寄存器 `sq_db`、一个 WQE 槽位计数器 `sq_rsvd_index`、以及一把 `sq_lock`。每次 `put` 都要占 WQE 槽、更新生产者索引、敲门铃，这几步必须互斥（`doca-gpunetio/include/device/doca_gpunetio_dev_verbs_qp.cuh:95-102`、`:443-469`）：

```cpp
// 提交 WQE 到一条 QP 的关键步骤（节选）
doca_gpu_dev_verbs_reserve_wq_slots(...);            // 原子加 sq_rsvd_index，占一个 WQE 槽
doca_gpu_dev_verbs_lock<mode>(&qp->sq_lock);          // 锁住门铃
doca_gpu_dev_verbs_atomic_max(...&qp->sq_wqe_pi, prod_index);
doca_gpu_dev_verbs_ring_db(...);                     // 敲 doorbell，通知网卡取 WQE
doca_gpu_dev_verbs_unlock<mode>(&qp->sq_lock);
```

若所有 CTA 共用同一条 QP，它们会在 `sq_lock` 上自旋互等（`while (atomicCAS_block(lock, 0, 1) != 0) continue;`，`doca_gpunetio_dev_verbs_common.cuh:338-339`），门铃也退化成串行瓶颈。多 context 的意义，就是给每个并发提交单元（CTA）一套**独占的 QP**，让门铃、槽位计数、锁各自独立，提交互不干扰。这和「多网卡」是同一思路的两个粒度：多网卡分散的是**物理带宽**，多 context 分散的是**同一条 QP 上的提交串行化**。

这一点也体现在 `ncclGinResourceSharingMode` 的三种取值上（`gin_device_common.h:56-60`），它们决定了同一 QP 被共享时的同步代价（`doca_gpunetio_dev_verbs_common.cuh:334-345`、`:305-318`）：

| 模式 | 含义 | 提交时的同步代价 |
|------|------|-----------------|
| `EXCLUSIVE` | 单线程独占该 QP | 直接读写 `sq_lock`/槽位计数，无锁无原子 |
| `CTA` | 一个 CTA 内共享 | `atomicCAS_block` + block 级 fence |
| `GPU` | 全 GPU 共享 | 全局原子 + 全 GPU 级 fence |

DeepEP 的 `get_qp_mode`（`comm.cuh:56-86`）正是按「SM 数 ≤ QP 数时每个 SM 独占一条 QP（走 `CTA`），SM 数 > QP 数时才退化到 `GPU` 共享」来分配的——**优先独占，就是为了避免锁与原子带来的串行化**。

把上面两层资源与映射画成排布图更直观。下图以「2 张 GIN 网卡、每条连接 3 个 context、每个 context 4 个 peer」为例，展示从网卡到 QP 的并列结构，以及 kernel 侧 `contextIndex` 到 `connectionId` / `contextId` 的映射。其中 channelId 是 Host 建连期 `ncclTopoGetLocalGinDevs` 枚举本 GPU 就近网卡时用的计数器：`localGinDevs[i]` 即 channelId=i 枚举到的那张网卡，连接号 i 直接用 `localGinDevs[i]`，因此 channelId 与 connectionId 在数组下标上一一对应（`topo.cc:2084-2092`）。

```text
Host 建连期：channelId 枚举本 GPU 就近的 GIN 网卡 → localGinDevs[]
  channelId=0 ──► 网卡 dev=5 ──┐
  channelId=1 ──► 网卡 dev=9 ──┴──► localGinDevs[] = {5, 9}（2 张并列，netsPerGpu=2）

连接层 connectionId（每条连接 = 一张网卡 + 一份设备侧资源表，上限 4）
  connectionId=0 ──► localGinDevs[0]=dev5 ──► ginHandles[0]
  connectionId=1 ──► localGinDevs[1]=dev9 ──► ginHandles[1]

context 层 contextId（每条连接内 3 组并列，每组 nranks 个 QP，按 peer 索引）
  ginHandles[0]（网卡 dev5）
    contextId=0 ──► GPUContext{ gdqp→[QP(peer0)  QP(peer1)  QP(peer2)  QP(peer3)] }
    contextId=1 ──► GPUContext{ gdqp→[QP(peer0)  QP(peer1)  QP(peer2)  QP(peer3)] }
    contextId=2 ──► GPUContext{ gdqp→[QP(peer0)  QP(peer1)  QP(peer2)  QP(peer3)] }
  ginHandles[1]（网卡 dev9）：同样 3 组，结构相同，从略

kernel 侧定位：contextIndex → (connectionId, contextId) → 某组内的 QP
  contextIndex = blockIdx.x % ginContextCount            （共 6 个：0..5）
  connectionId = contextIndex & (ginConnectionCount-1)   （低位：0,1,0,1,0,1 交错选网卡）
  contextId    = contextIndex >> 1                        （高位：0,0,1,1,2,2 分组选 QP 组；连接数=2 时右移 1 位）
  QP = ((ncclGinGdakiGPUContext*)ginHandles[connectionId])[contextId].gdqp + peer
  例 contextIndex=5 → connectionId=1（网卡 dev9）、contextId=2（第 2 组），发 peer2 即用该组 QP(peer2)
```

图中三层都是**建连期固定的静态绑定**：网卡与连接、连接与其下的 context、context 与其下的 QP 数组，运行时 kernel 只需用 `contextIndex` 拆出两个下标、再加一个 `peer`，就能落到唯一的 QP。下面这段是拆分逻辑，中文注释为笔记所加，与图里的低位/高位规则一一对应：

```cpp
// src/include/nccl_device/impl/gin__funcs.h:81-97（节选，中文注释为笔记所加）
template <typename GinType>
NCCL_DEVICE_INLINE void ncclGinInitCommon(GinType* gin, ncclDevComm const& comm, int contextIndex) {
  gin->nConnections = comm.ginConnectionCount;
  // 低位交错：相邻 contextIndex 落到不同连接，把 CTA 流量分散到多张网卡
  gin->connectionId = comm.ginConnectionCount == 3 ? uint32_t(contextIndex) % 3
                                                     : contextIndex & (comm.ginConnectionCount - 1);
  // 高位分组：同一连接内再区分第几套 QP / 队列（连接数为 3 时用除法，其余用移位）
  gin->contextId = comm.ginConnectionCount == 3 ? uint32_t(contextIndex) / 3
                                                  : contextIndex >> (comm.ginConnectionCount == 4 ? 2 : comm.ginConnectionCount - 1);
  gin->_ginBackend = comm.ginNetDeviceTypes[gin->connectionId];  // backend 类型按连接选
  gin->_ginHandle  = comm.ginHandles[gin->connectionId];          // 设备侧资源表按连接选
  gin->_signalShadows = comm.ginSignalShadows + contextIndex * comm.ginSignalCount;  // 信号影子按完整 contextIndex 分片
}
```

两个下标各自的作用归纳如下（具体换算已在上图与代码注释中展示，这里不再重复数值）：

| 下标 | 计算方式 | 选中的资源 | 带来的影响 |
|------|---------|-----------|-----------|
| `connectionId` | `contextIndex` 低位（取模或按位与） | `ginComms[]`、`ginHandles[]`、`ginNetDeviceTypes[]` | 决定走哪张网卡、哪条 GIN 连接；多条连接实现跨 NIC 带宽聚合与 rail 覆盖 |
| `contextId` | `contextIndex` 高位（整除或移位） | 该连接内的 QP 组 / GFD 队列 / signal 与 counter 槽位 | 决定并发工作区；多个 context 避免 CTA 之间争用同一发送队列与信号 |

两级合起来，把 `ginContextCount` 个 `contextIndex` 映射到「连接 × context」的二维资源网格；再叠加上一节的 `peer` 下标，一次 Put 的完整定位就是 `ginHandles[connectionId]` → `[contextId]` → `gdqp + peer`。内核用 `blockIdx.x % ginContextCount` 选出 `contextIndex`（`device/symmetric/all_gather_gin.cuh:17`），因此 CTA 编号直接决定了它使用哪张网卡上的哪组 QP。

**从 CTA 到 QP：并行发送的完整路径。** 上面这句里的 `blockIdx.x` 是关键，它把 CUDA 的并行性与 GIN 的并行性接了起来。CUDA 里 CTA（Cooperative Thread Array）就是 thread block：一个 kernel 以 grid 形式启动，grid 中多个 block 在多个 SM 上并行执行，每个 block 有唯一编号 `blockIdx.x`。GIN 要做的事，是把这些「并行执行单元」映射到「并行通信资源」上。整条映射链如下：

```text
CUDA 并行层：一个 kernel 启动多个 block（CTA），并行执行，各有唯一 blockIdx.x
      │  blockIdx.x % ginContextCount     取模：把 block 编号折进 context 的有限范围
      ▼
contextIndex                             扁平 context 编号（0 .. ginContextCount-1）
      │  ncclGinInitCommon 拆低位 / 高位
      ▼
connectionId（选哪张网卡）+ contextId（选该网卡上第几组 QP）
      │  + peer
      ▼
唯一一条 QP
```

`ncclGin` 构造器（`gin.h:204-215`、`gin__funcs.h:100-105`）接收 `contextIndex` 并原样交给 `ncclGinInitCommon` 拆分，NCCL 的 symmetric 内核都在函数开头这样构造（`all_gather_gin.cuh:17`、`reduce_scatter_gin.cuh:13`），因此**一个 block 内的所有 GIN 操作固定走 `blockIdx.x` 定出的那一个 context**：block 内通过 `coop` 协同，block 之间用不同 context 隔离。

这里要说明两点，它们合起来回答了「为什么这样能并行发送」：

1. **取模是「并发单元 → 有限资源」的轮转分配器。** 一个 kernel 的 block 数通常远大于 context 数（context 才几十个，block 可能有上千个）。取模让相邻 `blockIdx.x` 落到相邻 `contextIndex`，进而交错到不同网卡、不同 QP 组，于是多个 block 的发送被分散到多路资源上并行进行。这正是「多组数据发散到多个 CTA、多个 CTA 再打散到多张网卡、多个 context 上的 QP 组」的实现机制——CUDA 的并行性（多 block）经过这一层映射，被翻译成了 GIN 的并行性（多网卡 × 多 QP 组）。
2. **分配粒度由并发单元决定，且要匹配资源数。** 并发度（block 数）不超过资源数（context 数）时，每个 block 独占一组 QP，走 `CTA`/`EXCLUSIVE` 模式、提交无需锁；超过后，多个 block 绕回起点共享同一 context，才退化为 `GPU` 模式、需要锁与原子（对应上一节的争用代价）。所以 `blockIdx.x % ginContextCount` 既是并行发送的入口，也是决定「独占还是共享」的开关。

分配单位因内核而异，但机制相同：NCCL 的 symmetric 内核是「一个 CTA = 一个 context」（block 级），DeepEP 的 dispatch 内核粒度更细，是「一个 warp = 一个 context」（`dispatch.cuh:67-71`，每个 warp 用 `get_qp_mode` 分一个 QP）。无论 block 级还是 warp 级，「并发单元取模轮转到有限资源」这一逻辑不变。

**信号影子（signal shadow）：GPU 本地维护的信号进度副本。** 这里补一个前面 `_signalShadows` 提到的概念。先看「信号」本身：它是 GIN 的完成通知机制——`put`/`get` 可以附带一个 signal，网卡在数据落定后对信号槽做原子加，接收方轮询该槽位判断之前的操作是否已完成。信号槽放在对称内存里（每个 context 一段 `signals_table`），值是单调递增的。而「影子」是这块 GPU 本地记录的一份「我已经等到哪个信号值」，按 `contextIndex` 分片、每个 signal 一个 uint64，分配大小为 `nGinContexts * ginSignalTotal * sizeof(uint64_t)`（`dev_runtime.cc:1237`），地址位于对称内存窗口内（`:1298-1299`），设备端可直接读写。

为什么需要它，有两层原因：

1. **进度记忆**：信号值随多次操作累加，每次等待必须知道「从哪个值继续等」，不能从头轮询。影子就是这份进度，体现在三个函数里（`gin__funcs.h:1031-1082`、`:1097-1105`）：
   - `waitSignalMeetShadow`：`least = _signalShadows[signal] + sig.offset`，轮询信号槽直到达到 `least`——从上次进度出发，只等增量；
   - `waitSignalFollowShadow`：等到之后把读回的值减去 `offset` 写回 `_signalShadows[signal]`（`:1066-1068`），影子跟上新进度；
   - `resetSignal`：重置信号槽后把 `_signalShadows[signal] = 0`（`:1103-1104`），两者一起归零。
2. **剥离物理偏移**：信号槽的物理值带一个 `offset`（`sig.offset`，是该 context 信号表在统一窗口里的基址偏移），不是从 0 开始。影子存的是扣除 offset 之后的逻辑值，等待时再加回 offset 换算成物理值，于是调用方无需关心信号槽的绝对地址，只关心相对增量。

这正解释了为什么 `_signalShadows` 按**全局** `contextIndex` 而非 `contextId` 分片（`gin__funcs.h:97`）：不同 context 的信号表 offset 不同、进度也互相独立，共用一份会互相覆盖进度；按完整二维组合隔离，同一 `contextId` 落在不同连接上时也不会共用槽位。

#### 4.4.2 远端参数：地址与 rkey 从哪来、何时固化

4.4.1 回答了「用哪条本地 QP 发」——`ctx.handle[contextId]` 与 `gdqp + peer` 选出的是本端的发送队列。但一次 RDMA 写是单边操作，接收方 CPU 不参与，发送方投递 WQE 时还必须再告诉网卡「数据写到对端哪块内存、用什么 rkey 校验」，否则对端网卡无从定位目标。所以除本地 QP 外，Put 还有另一半参数——远端地址与 rkey。本节回答它们从哪来、何时固化，分两个阶段看：

- **建连阶段**：`gdakiFillExchInfo` 把本端的 `qpn`、LID、GID 填入 `gdaki_exch_info`（`gin_host_gdaki.cc:353-357`），全体交换后由 `gdakiConnectQp` 调用 `doca_verbs_qp_attr_set_dest_qp_num` 把**对端 QPN 写进本地 QP 的属性**（`:416`）。一次 RDMA 写涉及通信两端各自独立的 QP：本端 QP 负责投递 WQE，对端 QP 由这个绑定关系确定，配对工作在初始化时一次完成。
- **运行阶段**：设备侧提交时传给 `doca_gpu_dev_verbs_put` 的参数是 `qp`、远端地址与 rkey、本地地址与 lkey、长度，**不需要再带远端 QP 号**（`gin_gdaki.h:67-71`、`:110`）：

```cpp
raddr.addr = dstOff;
raddr.key  = loadConst(loadConst(&dstMh->rkeys) + peer);   // 远端 rkey，按 peer 索引
laddr.addr = srcOff;
laddr.key  = loadConst(&srcMh->lkey);                      // 本地 lkey
```

`rkeys` 来自 4.3 的窗口注册：`regMrSym` 对每块窗口做一次 AllGather 换得各 peer 的 rkey（`gin_host_gdaki.cc:1022-1025`），再把这份数组 `copy_h_to_d()` 拷进 GPU 内存、指针存进 `ncclGinGdakiMemHandle`（`gin_host_gdaki.cc:1027-1030`、`gin_gdaki_device_host_common.h:38-41`）。RDMA 是单边写，接收方 CPU 不参与匹配，因此 WQE 必须显式给出远端虚拟地址与 rkey；网卡据此完成地址转换与内存保护校验，再把数据 DMA 到对端。带信号或计数器的操作会使用独立的 rkey（`signalKey`、`counters_table.rkeys`）以及 `companion_gdqp + peer` 这条伴随 QP（`gin_gdaki.h:75-89`）。

把这一段和前面「本地 QP 的复杂定位」合起来看，能澄清一个关键问题：设备侧发起一次 Put 所需的全部信息——本地 QP、对端 QP、远端 rkey——**是否都在初始化期备好、随句柄交到设备侧，而不是 kernel 临时向 Host 查询？** 答案是肯定的，但三者的获取方式不同，不能都套用「按编号取」：

| 信息 | 初始化期如何准备 | 运行时如何取用 |
|------|----------------|---------------|
| 本地 QP | `createContext` 里 `doca_gpu_verbs_export_multi_qps_dev` 把 `ncontexts * nranks` 条 QP 导出到 GPU 可见内存（`gin_host_gdaki.cc:747`） | `gdqp + peer`，按 peer 下标直接取 |
| 对端 QP 号 | 建连时 `doca_verbs_qp_attr_set_dest_qp_num` 把对端 QPN 写进本地 QP 的属性（`gin_host_gdaki.cc:416`） | **无需再取**：本地 QP 属性已绑定对端，`doca_gpu_dev_verbs_put` 只传本地 `qp` 指针 |
| 远端 rkey | 窗口注册 `regMrSym` 做 AllGather 后 `copy_h_to_d()` 拷进 GPU 内存，指针存进窗口句柄（`gin_host_gdaki.cc:1022-1030`） | `dstMh->rkeys[peer]`，按 peer 下标从窗口句柄取 |

由此得到三点结论：

1. **对端 QP 不是「按编号查表」来的，而是固化在本地 QP 属性里。** 前面那套复杂定位（`connectionId`、`contextId`、`peer` 三级下标）只用于选出**本端**的发送队列；一旦拿到本地 QP 指针，对端配对已隐含其中，运行时不需要再关心远端 QP 号。这正是「运行阶段不带远端 QP 号」的原因。
2. **rkey 与本地 QP 用同一个 `peer` 下标对齐。** `gdqp + peer` 与 `dstMh->rkeys[peer]` 里的 `peer` 是同一个 rank 索引，因此一次 Put 的两半参数——本端 QP、远端 rkey——由同一个 peer 对齐，不存在额外的映射表。
3. **「下发」发生在初始化期，不是每次 launch kernel。** 窗口注册与 `createContext` 完成时，rkey 数组与 QP 数组就已放进 GPU 可见内存，句柄随 `ncclDevComm` 与窗口一次性交到设备侧；kernel 启动时只是按下标读这些已就绪的指针。这正是 4.3 结尾「控制面一次准备、kernel 不再向 Host 查询」在 GDAKI 参数上的具体体现。

最后补一个决定「peer 覆盖哪些 rank」的配置——连接范围。前面说的按 peer 索引的 QP 数组、rkey 数组，其 peer 到底覆盖哪些 rank，由 `ginConnectionType` 决定，取 FULL 或 RAIL 两种，依据全体 rank 是否支持跨 NIC 自动判定（`init.cc:1668-1671`）。

rail 是跨节点网络拓扑里的「轨道」概念。以 rail-optimized 组网为例：每个节点有 `lsaSize` 块 GPU，每块 GPU 绑定一张网卡，这张网卡对应的 rail 贯穿所有节点、连接每个节点的「同一槽位」GPU。`ncclTeamRail`（`nccl_device/core.cc:36-45`）返回的正是与自己同 rail 的 rank 集合：`nRanks = 总 rank 数 / lsaSize`（即节点数）、`rank = 本 rank / lsaSize`（自己在第几个节点）、`stride = lsaSize`。也就是说，rail 成员是「各节点中与自己同槽位的 GPU」，成员数等于节点数。

两种连接范围在 GDAKI 里的区别（`gin_host.cc:141-155`）：

- **FULL**：`nGinRanks = comm->nRanks`，为全体 rank 建连接。每个 context 的 QP 数组、rkey 数组长度都是「总 rank 数」，任意两 rank 都能直连。要求全体 rank 支持跨 NIC（GPU 能访问多张网卡）。
- **RAIL**：`nGinRanks = railTeam.nRanks`（即节点数），只为同 rail 的 rank 建连接。每个 context 的 QP 数与 rkey 数降为「节点数」，QP 总数从 `ncontexts * nRanks` 降到 `ncontexts * 节点数`（`gin_host_gdaki.cc:475-478`）。跨 rail 的 rank 不能直连，要发数据给它，需先经节点内 NVLink 转发到目标 rail 上的 GPU，再由那张网卡发出——这正是 DeepEP hybrid 模式里「scaleup 走 NVLink、scaleout 走 rail」分流的原因。

区分这两者的意义在于：rail-optimized 拓扑下 GPU 不能跨 NIC，它只能直连自己那张网卡对应的 rail。此时若强行按 FULL 为全体 rank 建连接，多数 QP 也走不通；RAIL 用更少的 QP（节点数而非总 rank 数）覆盖了真正可直连的那部分，把跨 rail 的流量留给 NVLink 转发。

### 4.5 异步请求的落定：progress 线程与错误上报

4.4 走到第三层时，一次 `ncclGinPut` 已经把请求写成 WQE 或 GFD，但**提交不等于完成**。设备端 API 返回后，网卡还要实际搬数据、写回完成状态，上层库也还需要一个途径判断操作何时可用、出错时如何得知。4.4 开头把这一侧留给了本节，4.3 第四步那句 `if (ginState->needsProxyProgress && ginState->ginProgress == 0)` 同样留下了一个尚未解释的判断。本节把这条线补完，依次回答三个问题：哪些 backend 需要 Host 侧协助推进、这个线程具体做什么、错误如何上报给调用方。

先回到 4.2 的三种硬件落点。GDAKI 与 GPI 下，GPU 提交的内容由网卡直接执行，落定过程可以在设备侧完成；PROXY 下 GPU 只把 GFD 描述符写进共享队列，队列另一端必须有 Host 线程读取并调用 RMA 后端，网络操作才会真正发出。因此是否需要 Host 侧线程，取决于所选 backend 的**数据路径**，与「有没有 GIN 能力」无关——GPI 这类完全由 GPU 直驱的后端就不需要它。

这个判断在代码里落成一个字段：`ncclNetDeviceHandle_t::needsProxyProgress`，由插件在 `createContext` 返回句柄时填好，`ncclGinDevCommSetup` 遍历所有 context 汇总（`gin_host.cc:291`），任一为 1 就把 `ginState->needsProxyProgress` 置 1。下表列出各 backend 的取值与依据，用来判断线程何时会出现：

| Backend | `needsProxyProgress` | 依据 |
|---------|---------------------|------|
| PROXY | 恒为 `1` | `gin_host_proxy.cc:554` 直接写死；设备端只写 GFD 到共享队列，必须由 Host 读出后交给 RMA 后端提交 |
| GDAKI | 取决于 QP 的 `cpu_proxy` 标志 | `transport/net_ib/gdaki/gin_host_gdaki.cc:745`、`:794`：遍历所有 QP 与伴随 QP，任一需要 CPU 代理则置 1 |
| 外部插件（含 GPI 实现） | 由插件自定，通常为 `0` | 示例插件 `plugins/gin/example/plugin.c:170` 置 0；GPU 直接写网卡可见队列，无需 Host 推进 |

读表可知，PROXY 是唯一无条件需要 progress 线程的后端；GDAKI 只在个别 QP 需要 CPU 代理时才需要；外部插件多数不需要。这也解释了 4.3 第四步那个条件判断为什么出现在 `createContext` 循环之后：`needsProxyProgress` 不是初始化时写死的常量，而是遍历完插件返回的句柄才确定，所以线程只能在第四步（创建设备侧 context）之后创建。

线程本身由 `ncclGinDevCommSetup` 在满足条件时启动（`gin_host.cc:294-298`），执行体是 `ncclGinProgress`（`gin_host.cc:43-76`）。下面这段代码是它的循环骨架；注意它不包含任何网络逻辑，只按状态遍历所有 device comm 与 GIN 连接，逐个调用插件的 `ginProgress` 回调：

```cpp
// src/gin/gin_host.cc:48-75（节选）
while (1) {
  std::unique_lock<std::mutex> lock(ginState->mutex);
  if (ginState->ginProgress == 1) {
    struct ncclGinStateDevComm* dc = ginState->devComms;
    while (dc) {
      for (int n = 0; n < ginState->ginCommCount; n++) {
        ncclResult_t ret = ginState->ncclGin->ginProgress(dc->ginCtx[n]);  // 调到插件
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
          ginState->ginProgress = -2;
          return NULL;                                                     // 出错则退出
        }
      }
      dc = dc->next;
    }
    lock.unlock();
    std::this_thread::yield();
  } else if (ginState->ginProgress == -1) return NULL;                      // 正常退出
  else if (ginState->ginProgress == 0) ginState->cond.wait(lock);            // 暂停
}
```

`ginProgress` 是 `ncclGin_t` 接口里的可选回调（`gin_v14.h:59-60`，注释 "Will be called if devHandle.needsProxyProgress=1"）。线程只是驱动者，真正的推进逻辑在插件实现里。以 PROXY 为例，`ncclGinProxyProgress`（`gin_host_proxy.cc:616-636`）对每个 context 做三件事：轮询完成状态（`proxyGinPollCompletions`）、按 rank 轮询 GFD 队列（`proxyGinPollGfd`）、把取出的描述符交给 `proxyGinProcessGfd` 转成 `rmaBackend->iput` / `iputSignal` / `iget` 提交（`:245-262`、`:303-313`），最后调用 RMA 后端自身的 `rmaProgress`（`:633`）。完成信息由 `proxyGinPollCompletions` 写回消费索引 `cis`（`:141-145`），设备侧据此判断请求已落定。

线程的启停由 `ginProgress` 状态控制：`0` 暂停（`cond.wait`）、`1` 运行、`-1` 退出（`ncclGinHostFinalize` 置 -1 并 `join`）、`-2` 表示回调报错后自行退出。错误上报有两条路径，`ncclCommGetAsyncError` 都会检查（`init.cc:3459-3474`）：若 `needsProxyProgress` 为真，先读 `ginState->asyncResult`，即 progress 线程写下的结果；随后**无论有没有 progress 线程**，都调用 `ncclGinQueryLastError` 走插件的 `queryLastError`（`gin_v14.h:62-63`）。后者保证 GPI/GDAKI 这类不启动线程的后端，错误同样能被上层库取到。

到这里，GIN 从主机侧准备（4.3）、设备侧提交（4.4）到请求落定（4.5）的路径就完整了。下一节把这条路径与第三章的 Host 发起模式并排比较，看发起权下放给 GPU 之后，控制流与数据流分别改变了什么。

### 4.6 与 Host 发起模式的对比

| 维度 | Host 发起（传统 NET） | GIN 发起 |
|------|----------------------|----------|
| 通信触发者 | Host Proxy 线程轮询 head/tail 后调 `ncclNet->isend` | GPU kernel 内线程直接调 `ncclGinPut`/`ncclGinGet` |
| 重叠粒度 | kernel 级（等 kernel 结束 + proxy 介入） | 线程级（kernel 内边算边发） |
| 同步机制 | `ncclSendMem.head`/`ncclRecvMem.tail`（Host 读写） | 窗口 + 信号/计数器（设备端直接操作） |
| 数据路径 | GPU 显存 →（GDRDMA 直读 或 Host buffer 中转）→ 网卡 | GPI/GDAKI：GPU 直接驱动网卡；PROXY：经 Host 代理 |
| 代码位置 | `src/proxy.cc`、`src/transport/net.cc` | `src/gin/gin_host.cc` + `src/include/nccl_device/gin/` |
| 延迟 | 较高（Host-Device 往返 + 线程调度） | 较低（GPI/GDAKI 消除 Host 参与） |

两者的核心差异：Host 发起模式下，GPU 是"数据生产者 + 进度标记写入者"，网卡要靠 Host 来驱动；GIN 模式下，GPU 是"数据生产者 + 网络操作发起者"，Host 只在 PROXY 回退时充当代理。

> 需要澄清的一点：GIN 目前**不是 NCCL 自身集合通信（AllReduce 等）的主路径**——那些仍走第三章的 proxy 机制。GIN 是 NCCL 暴露给上层库（DeepEP、自研 MoE 通信层）的能力，让它们能在自定义 kernel 里直接发起网络操作。理解这一章，是为了看懂 DeepEP 这类库怎么绕过 NCCL 的集合接口、直接拿 GIN 当通信原语。

---

## 五、组件交互全景

组件交互不只是"数据怎么流动"，还包括"我们怎么**看见**这些流动"。在拆解 Channel/Connector/Proxy 这些通信组件之前，先讲一套贯穿所有 NCCL API 的可观测性基础设施——NVTX 插桩。它是性能分析和调优（后续所有章节的实验手段）的前提，也是一份非常好的 C++ 惯用法教材。

### 5.1 性能可观测性：NVTX 插桩与零开销抽象

#### （1）NVTX 是什么

NVTX（NVIDIA Tools Extension）是 NVIDIA 提供的**性能标记 API**：在代码里插入时间区间标记（Range），Profiler（如 Nsight Systems）就能在时间轴上显示"这段时间在做什么"。

| 概念 | 含义 |
|------|------|
| **Range** | 一个时间区间标记，有名字（如 `"AllReduce"`），在时间轴上显示为色块 |
| **Payload** | 附加在 Range 上的**结构化数据**（如通信器哈希、rank 号、消息大小），展开详情或做筛选分析时用 |
| **Schema** | 定义 Payload 的数据格式（字段名、类型），Profiler 据此解析 |
| **Domain** | Range 的命名空间，NCCL 用名为 `"NCCL"` 的独立 domain，避免和其他库的标记混在一起 |

> **Range 名字 vs Payload 的分工**：名字是字符串，用于时间轴上的快速识别；Payload 是结构化数据，用于"按通信器筛选""按消息大小排序"这类**机器可读**的分析。两者分开存储、各司其职。

#### （2）NCCL 的两层 RAII 封装（`src/include/nvtx.h`）

NCCL 没有在代码里散落 `nvtxRangePush`/`nvtxRangePop`，而是封装了两个 RAII 类。先看最基础的：

```cpp
// src/include/nvtx.h:93-116（节选）
class ncclOptionalNvtxScopedRange {
public:
  void push(const nvtx3::event_attributes& attr) noexcept {
    pushed = true;
    nvtxDomainRangePushEx(nvtx3::domain::get<nccl_domain>(), attr.get());
  }

  ~ncclOptionalNvtxScopedRange() noexcept {   // 析构时自动 pop
    if (!pushed) return;
    nvtxDomainRangePop(nvtx3::domain::get<nccl_domain>());
  }

  ncclOptionalNvtxScopedRange() = default;
  ncclOptionalNvtxScopedRange(ncclOptionalNvtxScopedRange const&) = delete;      // 禁止拷贝
  ncclOptionalNvtxScopedRange& operator=(ncclOptionalNvtxScopedRange const&) = delete;
  ncclOptionalNvtxScopedRange(ncclOptionalNvtxScopedRange&&) = delete;           // 禁止移动
  ncclOptionalNvtxScopedRange& operator=(ncclOptionalNvtxScopedRange&&) = delete;

private:
  bool pushed = false;
};
```

> **用到的 C++ 惯用法，逐个拆解**：
>
> - **RAII（Resource Acquisition Is Initialization）**：资源的获取放在构造函数、释放放在析构函数，利用"栈对象离开作用域自动析构"保证资源一定被释放。它不是 C++ 内置功能，而是用"构造/析构 + 作用域 + 栈展开"组合出的惯用法。这里"资源"就是"一个打开着的 NVTX Range"——忘了 pop 会让时间轴错乱，而 RAII 让你**想忘都忘不掉**。
> - **`= delete`（禁止拷贝/移动）**：RAII 对象必须独占资源。如果允许拷贝，一个 Range 会被 pop 两次。相比把拷贝构造声明成 `private`（只在链接期报错、易被绕过），`= delete` 让误用直接**编译期报错**。
> - **`noexcept`（析构不抛异常）**：C++11 起析构函数默认就是 `noexcept`，这里显式写出是自我文档化。若析构抛异常且发生在**栈展开**（异常传播时逐层退出调用栈、逐层析构局部对象的过程）中，会直接 `std::terminate()`。
> - **`pushed` 标志**：push 是条件执行的（NVTX 被禁用时跳过，见下面的宏），析构时用这个标志判断"到底 push 没有"，避免不平衡的 pop。成员注释也写了：正常运行中它不该为 false，但逐一检查太贵，所以只在析构兜底。

第二层是带 Payload 的模板版本：

```cpp
// src/include/nvtx.h:147-187（节选）
template <typename PayloadType>
class ncclOptionalNvtxPayloadRange {
public:
  void push(const nvtx3::event_attributes& attr) noexcept { /* 同上，RangePushEx */ }

  ~ncclOptionalNvtxPayloadRange() noexcept {
    if (!pushed) return;
    if (payloadData.payload) {
      nvtxRangePopPayload(nvtx3::domain::get<nccl_domain>(), &payloadData, 1); // pop 时把 payload 一起交出去
    } else {
      nvtxDomainRangePop(nvtx3::domain::get<nccl_domain>());
    }
  }

  void setPayloadData(const uint64_t schemaId) noexcept {
    payloadData = {schemaId, sizeof(PayloadType), &payload};
  }

  ncclOptionalNvtxPayloadRange() = default;
  ncclOptionalNvtxPayloadRange(const ncclOptionalNvtxPayloadRange&) = delete;   // 同样禁止拷贝/移动
  ... // 移动也 = delete

  PayloadType payload{};   // payload 就存在对象里，稍后填充

private:
  bool pushed = false;
  nvtxPayloadData_t payloadData = {NVTX_PAYLOAD_ENTRY_TYPE_INVALID, 0, NULL};
};
```

> **模板（`template <typename PayloadType>`）在这里的价值**：不同 API 的 Payload 类型不同（AllReduce 是 `{comm, bytes, op}`，CommInitRank 是 `{commHash, nranks, myrank, cudaDev}`），模板让每种 Payload 生成一份专属的类实例——**编译期确定类型、运行期零开销**，且 `sizeof(PayloadType)`、字段布局都在编译期可知。代价是代码膨胀和编译时间，但对这种基础设施级的小类完全值得。

#### （3）宏家族：把 RAII 对象"串"进函数

裸用上面的类还是很啰嗦，NCCL 用几个宏把它们一行塞进函数。**这里有一个非常精妙的"两段式"设计**，先看用于通信器生命周期的这对宏：

```cpp
// src/include/nvtx.h:191-197 —— 第一段：函数入口，先 push 一个只有名字的 Range
#define NVTX3_RANGE(T) \
  ncclOptionalNvtxPayloadRange<T> nvtx3_range__; \
  if (!ncclParamNvtxDisable()) { \
    static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
    ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
    nvtx3_range__.push(nvtx3_func_attr__); \
  }

// src/include/nvtx.h:204-214 —— 第二段：执行完毕后，给已存在的 Range 补上 Payload
#define NVTX3_RANGE_ADD_PAYLOAD(N, S, P) \
  do { \
    if (!nvtx3_range__.isPushed()) break; \
    constexpr uint64_t schema_id = NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##N; \
    static const payload_schema schema{S, std::extent<decltype(S)>::value - 1, schema_id, \
                                       sizeof(nvtx3_range__.payload)}; \
    nvtx3_range__.payload = {P}; \
    nvtx3_range__.setPayloadData(schema_id); \
  } while (0)
```

看它在 `ncclCommInitRank` 里的真实用法（`src/init.cc:2562-2578`）：

```cpp
// src/init.cc:2562-2578
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)          // 1. 入口：先 push（此时还不知道 payload！）
  ...
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));  // 2. 执行初始化
  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));  // 3. 尾部：补 payload
  return ncclSuccess;
}  // 4. 函数退出：RAII 析构，pop 时把 payload 一起交给 Profiler
```

> **为什么要"先 push、后补 payload"，而不是构造时一起给？** 因为 **`commHash`、`cudaDev` 这些信息在函数入口时根本还不存在**——它们要等初始化完成后才能从 `*newcomm` 里读出来。所以 Range 必须从入口就开始计时，payload 只能等数据就绪后再填，最后析构时一并交给 Profiler。这是"**占坑在先、填数据在后**"的两段式设计，`NVTX3_RANGE_ADD_PAYLOAD` 宏里的 `do { ... } while (0)` 和 `isPushed()` 检查都是为了安全地配合这套流程。
>
> （集合操作 API 没有这个问题——`commHash`、字节数在入口就知道，所以用的是一步到位的 `NVTX3_FUNC_WITH_PARAMS` 宏，如 `src/collectives.cc:170` 的 `ncclAllReduce`：`NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce, NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op))`。）

#### （4）Payload Schema：编译期"声明数据格式"

Payload 要能被 Profiler 解析，必须先注册 schema（字段名、类型、顺序）。NCCL 用一个宏**同时生成 C struct 和 schema 描述**（`src/include/nvtx_payload_schemas.h:41-52`）：

```cpp
// src/include/nvtx_payload_schemas.h:41-45
NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommInitRank, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr)))
```

> 这个宏做两件事：定义 `struct NcclNvtxParamsCommInitRank { uint64_t newcomm; int nranks; int myrank; int cudaDev; }`，并生成对应的 schema entry 数组（含每个字段的可读名，如 "No. of ranks"）。schema ID 是**静态分配**的常量（`nvtx.h:22-45`，如 `NVTX_SID_CommInitRank = 0`、`NVTX_SID_AllReduce = 5`）；语义相同的 API 复用同一 schema（`CommDestroy`/`CommAbort`/`CommRevoke` 都 typedef 自 `NcclNvtxParamsCommInitRank`）。
>
> 注册动作藏在 `payload_schema` 类的**构造函数**里（`nvtx.h:60-68`，调 `nvtxPayloadSchemaRegister`），而宏里它被声明为 `static const` 局部变量——**首次执行到该行才构造、且全程只构造一次**，后续调用零成本。

#### （5）让 Profiler 显示 "Sum" 而不是数字：枚举注册

Reduction 操作（`ncclSum`/`ncclMax`…）本质是个整数枚举，直接显示没有可读性。NCCL 在 `src/init_nvtx.cc:19-35` 的 `initNvtxRegisteredEnums` 里把这些枚举值注册成可读字符串（"Sum"、"Product"、"Max"…），Profiler 展示 Payload 时就能翻译。整个过程受环境变量 `NCCL_NVTX_DISABLE` 控制（`NCCL_PARAM(NvtxDisable, "NVTX_DISABLE", 0)`），设为 1 时所有宏都短路，插桩几乎零开销。

#### （6）小结：这套设计的启发

| 设计 | 手段 | 收益 |
|------|------|------|
| 自动管理 Range 生命周期 | RAII（构造 push、析构 pop） | 永不泄漏一个 Range，异常栈展开时也安全 |
| 独占性 | `= delete` 拷贝/移动 | 误用编译期报错 |
| 异常安全 | `noexcept` 析构 + `pushed` 兜底 | 栈展开不 terminate、不重复 pop |
| 零开销抽象 | 模板 + `static` 首次注册 + `NCCL_NVTX_DISABLE` 短路 | 不开 Profiler 时几乎无成本 |
| 结构化数据 | 编译期 schema 宏 + 两段式 payload 填充 | Profiler 可按 comm/rank/消息大小筛选分析 |

> **和本章主题的关系**：接下来要讲的 Channel/Connector/Proxy 都是"数据平面"的组件，而 NVTX 这一套是"控制平面 + 观测平面"的横切基础设施——它包在**每一个**对外 API（初始化、集合操作、Group、内存分配）的入口和出口上。读代码时你会反复撞见 `NVTX3_RANGE(...)`，现在你知道：它不是业务逻辑，只是"我进入/离开了这个函数"的时间轴标记，可以直接跳过不影响理解主线。

### 5.2 Channel 与 Connector

**Channel** 是 NCCL 并行通信的基本单位。每个 Channel 独立承载一个 Ring 或 Tree 连接子图。

- 一个 NCCL 操作的数据被切分为 `nChannels` 份，每个 Channel 独立处理一个数据块
- Channel 的数量由 Ring 和 Tree 图搜索决定，受 `minCTAs`/`maxCTAs` 配置限制（最大 `MAXCHANNELS`）
- 每个 Channel 对应一个 CUDA CTA（Cooperative Thread Array），实现 Kernel 级并行

**Connector**（`src/include/p2p.h`）是 Channel 内 Peer 间连接的具体描述：

- `ncclConnector.conn`：指向具体的 Transport 资源（P2P/SHM/NET）
- `ncclConnector.transportComm`：指向该传输的虚函数表
- `ncclConnector.connected`：连接状态标志
- 每个 Peer 对在一个 Channel 中有一个 send Connector 和一个 recv Connector

### 5.3 Proxy 机制

先正面回答一个常见疑问：**Proxy 是不是"本卡 GPU 为了通过 Host 侧网卡发消息的一个交互"？** —— 你的直觉方向是对的，但可以更精确地说：

> **Proxy 的本质是"代劳"**：当 GPU **无法直接驱动某个硬件链路**时（典型是跨节点的 RDMA 网卡、同机 SHM 的 CPU 拷贝），NCCL 在 Host 侧拉起一个**常驻线程（proxy 线程）**，由它来"代理"完成那部分物理搬运。GPU 内核负责"算好数据 + 标记就绪"，Proxy 线程负责"真正把数据经网卡/CPU 搬走或搬进来"。

**为什么需要 Proxy：**

- **NET transport（跨节点 RDMA）**：传统路径下 GPU 不能直接写网卡发送队列（doorbell），必须由 CPU 通过网卡驱动发起 `isend`/`irecv`；
- **SHM transport**：需要通过 CPU 在共享内存和 GPU 显存之间做 `memcpy`；
- **CollNet transport**：需要在 Host 侧管理网内计算（SHARP）资源。

（补充一句：有了 GIN 之后，NET 场景下 GPU 才能直接发起网络操作，从而绕开 Proxy——这正是前面第 0.4 节和第四部分讲 GIN 时的对比点。Proxy 和 GIN 本质上是"谁发起网络操作"的两种答案。）

**两侧代码各在哪（这是理解 Proxy 的关键）：**

Proxy 是一个典型的 **Host/Device 协作机制**，两侧代码分布如下：

| 侧 | 文件 | 具体职责 |
|----|------|----------|
| **Host 侧**（proxy 线程） | `src/proxy.cc` | Proxy 线程主体：`ncclProxyProgress`（`proxy.cc:954`，线程主循环）、`progressOps`（`proxy.cc:801`）、`ncclProxyGetPostedOps`（`proxy.cc:835`）——轮询 op 队列并推进 |
| Host 侧 | `src/transport/net.cc` | NET 的 proxy 实现：`sendProxyProgress`（`net.cc:1304`）、`recvProxyProgress`（`net.cc:1470`），真正调用 net plugin 的 `isend`/`irecv` |
| Host 侧 | `src/transport/shm.cc` | SHM 的 proxy 实现：CPU `memcpy` 搬运 |
| Host 侧 | `src/include/proxy.h` | 核心数据结构：`ncclProxyOp`（`proxy.h:73`）、`ncclProxyArgs`（`proxy.h:185`）、`ncclProxyState`（`proxy.h:333`）、`ncclProxyOpsPool`（`proxy.h:229`） |
| **Device 侧**（GPU kernel） | `src/device/prims_simple.h` | 集合内核里更新 fifo 进度标记，例如 `peerPtr->send[connIndex].step += steps; st_relaxed_sys_global(...tail, ...)`（`prims_simple.h:307-308`） |
| Device 侧 | `src/device/prims_ll.h` / `prims_ll128.h` | 同理，各协议内核里维护进度标记 |
| Device 侧 | `src/include/comm.h` | 进度标记结构体：`ncclSendMem`（`comm.h:53`，含 `head`）、`ncclRecvMem`（`comm.h:67`，含 `tail`），就是 GPU 内核与 Proxy 线程之间交换进度状态的共享内存 |

**工作模式（以 NET 跨节点发送为例，串起上面两侧）：**

1. **Device 侧**：GPU 内核把数据算好放进发送缓冲区，然后**写共享内存里的进度标记**（`ncclSendMem.head` / 对应 fifo 的 `step`），表示"这一块数据已经就绪，可以发了"；
2. **Host 侧**：proxy 线程（`ncclProxyProgress`）**轮询**这些进度标记，发现新数据后，构造/取出 `ncclProxyOp`，走对应的 `proxyProgress`（NET 是 `sendProxyProgress`）；
3. **Host 侧**：`sendProxyProgress` 调用 net plugin 的 `isend()`，把 GPU 缓冲区里的数据经 RDMA 网卡发出去；
4. **接收侧反向**：对端 proxy 线程 `irecv()` 收数据到 GPU 接收缓冲区，更新 `ncclRecvMem.tail`；对端 GPU 内核轮询到 `tail` 前进后，才消费数据。

**关键数据结构（Host 侧）：**

- `ncclProxyOp`（`src/include/proxy.h:73`）：单个 Proxy 操作描述符，含连接指针、字节数、opCount、root 等；
- `ncclProxyArgs`（`proxy.h:185`）：一批 Proxy 操作参数，带 `progress` 函数指针，传递给 Proxy 线程；
- `ncclProxyState`（`proxy.h:333`）：Proxy 全局状态，含线程句柄、连接池、op 池等。

Proxy 线程在建链开始**之前**就由 `ncclProxyCreate` 启动（`init.cc:1524-1530`，建连期间 NET 的 SharedInit 就需要它），其生命周期与通信器绑定。

### 5.4 算法与协议的选择

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

### 5.5 值得深入研究的算法

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

## 参考资料

- NCCL 官方源码仓库：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)（版本 2.30.x）
- NCCL 官方文档：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- NCCL 开发者文档：[NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2206/nccl-developer-guide/)

---

## 延伸阅读

- [DeepEP 源码学习笔记](/notes/deepep-source-study/) —— MoE 专家并行下的通信库，以及在它之上如何复用 NCCL 通信器与 GIN 后端完成不规则 all-to-all。

---

*声明：本文是作者学习 NCCL 源码过程中的阶段性笔记，内容基于 NCCL master 分支源码整理。由于这是一份仍在持续完善的初稿，文中可能存在表述不准确或理解不完整之处，欢迎通过邮件联系指正。*
