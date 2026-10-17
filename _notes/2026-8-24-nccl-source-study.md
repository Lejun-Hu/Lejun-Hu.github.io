---
title: "【未完成】NCCL 与 DeepEP 源码学习笔记"
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

   > 读这段代码时，你可以抓住一条主线：**`ncclComm` 就是"一次通信所需的全部上下文"的容器**——它既记录了"我是谁、我在哪"（`rank`/`nRanks`/`node`/`busId`），也记录了"这个域的硬件长什么样"（`topo`/`graphs`），还记录了"数据怎么搬"（`channels`/`buffSizes`/`planner`/`workFifoBuf`）。后面讲初始化流程时，你会看到这个结构体是**一步步被填充起来**的——先建 bootstrap、再 AllGather 交换 peerInfo、再算拓扑和图、再建连接，最后才是一个"可用"的 communicator。
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

经典的例子是：反向传播里，"当前层的梯度计算"和"上一层的梯度 AllReduce"其实是可以重叠的——当你在算第 2 层的梯度时，第 3 层的梯度已经可以开始做通信了。这样通信的耗时就被"藏"进了计算的时间里。

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

> **一句话**：通算融合是"目的"，通信算子是"把它工程化的视角"，GIN 是"把重叠做到极致（kernel 内）的关键技术"。你现在看到的主流训练框架（Megatron、DeepSpeed 等）里那些花哨的"通算融合"，很多底层仍走的是传统 host 下发 + 多流重叠；而像 DeepEP 这种追求极致低延迟的库，才进一步用上了 GIN。我们在本笔记第二部分的 DeepEP 章节里，会具体看到 GIN 是怎么被用起来的。

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

#### ① 通信器 `ncclComm`：一个"通信域"的完整描述

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

- **拓扑信息**：`ncclTopoSystem* topo`、`rank`、`nRanks`、`cudaDev`、`busId`——"这个域有哪些 GPU、它们怎么连的、我是谁"；
- **Channel 数组**：`ncclChannel channels[MAXCHANNELS]`——并行通信通道（见下面的 ②）；
- **Graph 数组**：`ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]`——分别为 Ring/Tree/CollNet/NVLS 等算法算出的通信图；
- **网络**：`ncclNet_t* ncclNet`、`netContext`、`ginContext`——跨节点的网络插件与 GIN 上下文；
- **内核调度**：`ncclKernelPlanner planner`、workFifo 机制——把任务调度给 GPU 内核；
- **共享资源**：`ncclSharedResources* sharedRes`——跨 Split 子通信器共享的资源；
- **配置**：`ncclConfig_t config`——用户配置（超时、阻塞模式等）。

#### ② 通道 `ncclChannel`：并行通信的最小单位

**`ncclChannel`**（定义于 `src/include/comm.h:150-172`）代表**一条独立的通信通道**，可以理解为"通信域里的一条并行管道"。

为什么一个通信器里要有**多个** Channel？答案是**并行**。一个集合操作的数据会被切成 `nChannels` 份，每个 Channel 独立负责其中一份——每个 Channel 对应一条 Ring 或 Tree 连接，并由独立的 CUDA CTA 并行执行（**CTA 即 Cooperative Thread Array，也就是我们常说的 block/线程块**——它是 CUDA 里线程调度的基本单元，一个 CTA 内的线程在同一个 SM 上协同工作）。Channel 越多，能同时搬运的数据块就越多，带宽利用就越充分（当然也受硬件限制，最大 `MAXCHANNELS`）。

> **补充：Channel 是"逻辑通道"，不是物理链路绑定。** 一个 Channel 本身只包含"一条逻辑 Ring/Tree 连接 + 一份数据切片 + 一个 CUDA CTA"，它并不绑定某条具体物理链路——数据实际走 NVLink 还是 RDMA、走哪块网卡，是由每个连接（connector）的传输层和拓扑决定的。
>
> 那"切成多个 Channel 为什么就能并行、更快"？有两层原因：
> 1. **GPU 侧并行**：一个 Channel 对应一个 CTA，只能占一个 SM；多个 Channel 就是多个 CTA 同时占多个 SM，把 GPU 的并行计算单元用满；
> 2. **带宽叠加（更关键）**：单条 Ring 的吞吐受限于"环上最慢的一条链路"（比如单条 NVLink 端口、或单块网卡），而 GPU 总带宽远高于单条链路。用多条 Channel（多条并行的 Ring），每条走**不同的物理链路**（不同 NVLink 端口、不同网卡），带宽就能叠加，最终占满 GPU 总带宽。NCCL 源码里 `ncclTopoDupChannels`（`graph/search.cc:1036`）干的就是"把一条 Ring 复制成多条并行跑"。
>
> 至于"是不是对应不同的数据平面"——直觉接近，但不完全等同。Channel 的首要目的是 GPU 侧并行；只有跨节点多网卡时，多个 Channel 分散到**不同网卡（不同 rail）**上传输，才真正有"多数据平面"的意味。NCCL 里更贴近"物理数据平面/路径"的概念其实是 **rail**（`ncclTeamRail`，见本节 ① 里 Team 的说明）：每个 rail 对应一条网络路径（一块网卡），多个 Channel 可分布到多个 rail 上叠加多网卡带宽。

它的核心字段：

- `ncclRing ring` / `ncclTree tree`：这个 Channel 承载的 Ring / Tree 拓扑连接信息（即"我的上一跳是谁、下一跳是谁"）；
- `ncclChannelPeer** peers`：指向各 Peer 的连接器（connector）数组，描述"我连到每个 peer 的具体连接"；
- `workFifoProduced`：Work FIFO 的生产指针，用于 CPU 向 GPU 内核下发任务；
- 此外还有 `collnetChain` / `collnetDirect` / `ncclNvls nvls` 等字段，分别对应 CollNet、NVLS 这类"网内计算"算法在该 Channel 上的连接。

#### ③ 传输层 `ncclTransport`：异构硬件的统一抽象

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
> 而"选哪一个"的逻辑在 `selectTransport`（`src/transport.cc:20-34`）：它遍历 `ncclTransports[]`，对每个 transport 调用 `canConnect` 判断两节点能否用这类链路，能用就选中，并把它填进 `connector->transportComm`。因为数组顺序是 P2P → SHM → NET → CollNet，所以这本质是一条**"优先 P2P，不行退 SHM，再不行 NET"的兜底链**——这正好和下面 ④ 讲的 connector 衔接起来：`selectTransport` 就是"给每个 connector 挑选 transport"的地方。

#### ④ 连接器 `ncclConnector`：Channel 与 Transport 之间的桥梁

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
> 阅读建议：先读 **2.1**（认清 API 家族——先弄懂"五个公共 API 如何收敛到同一个执行体"这条主线）和 **2.2**（全景图——总共分五个阶段），再逐个阶段深入。若只想快速把握骨架，读完 2.1、2.2、2.7 即可。

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
                                               │  ① 校验 + 归一化参数
                                               │  ② 分配 ncclComm 骨架
                                               │  ③ 建 AsyncJob 集装箱，转交调度层
                                               │  ncclAsyncLaunch(job, ncclCommInitRankFunc, ...)
                                               ▼
【调度层】ncclAsyncLaunch(...)          ← 决定"什么时候、在哪个线程执行"
                                               │  不在 Group → 立即执行；在 Group → 入队；非阻塞 → 后台线程
                                               ▼
【执行层】ncclCommInitRankFunc(job)     ← 真正干活：本章 2.3~2.7 的五个阶段全在它体内
```

**各层职责一句话**：

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
| `ncclGetUniqueId` | 先生成"入场券"（严格说不属于 init 家族，但**必须先做**） | 无 |
| `ncclCommInitRank` | 最基础的初始化，多进程/多线程各自建 comm | 不包 Group → **同步阻塞** |
| `ncclCommInitRankConfig` | 带用户配置的初始化，**上层框架常用** | 多一个 `ncclConfig_t` 入参 |
| `ncclCommInitRankScalable` | 超大规模训练，用多个 uniqueId 加速 | 多一个 `nId`（uniqueId 个数）入参 |
| `ncclCommInitAll` | 单进程管多卡时，一次建出"一整串"communicator | 内部**循环**调用实现层 |

**门面函数的共同特征：它们的函数体几乎都只有"预处理 + 转交"两步。** 看最基础的 `ncclCommInitRank` 全文（只有 4 件事）：

```cpp
// src/init.cc:2562-2578（完整函数体）
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclInitEnv());                        // ① 环境准备（加载环境变量等）
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)          // ② 性能标记（见 5.1 节）
  (void)ncclCudaLibraryInit();                     // ③ 加载 CUDA 驱动、解析 dlsym hook
  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;   // ④ 准备默认配置
  CUDACHECK(cudaGetDevice(&cudaDev));              //    取当前 CUDA 设备号
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));
  // ↑ 转交实现层。注意倒数第一个参数是本函数名（__func__），供下层打日志/报错时定位
  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  return ncclSuccess;
}
```

> 这两点最能体现"门面"性质：**（1）** 它把"当前是哪个设备""用什么配置"这些**上下文信息**补全后传下去（因为 `ncclCommInitRank` 的签名里根本没有 device 参数，它默认用当前设备）；**（2）** 它把 `__func__`（调用者函数名）一路传下去，让下层在出错时能报出"是哪个 API 调用的"。

#### 2.1.2 命名规范：`Dev` / `Func` / `AsyncJob` 后缀的含义

（插叙一段：这几个后缀很容易被误读成"设备端"，先把这个命名约定讲清楚，后面看名字就不会被绊住；2.1.3 接回主线。）

这三个后缀是**"按层命名"**的产物，对应上面链条图里的三层：

| 后缀 | 含义 | 例子 |
|------|------|------|
| 无后缀（公共名） | 对外的门面函数：只做参数预处理（取当前 device、准备默认 config）就转交下去 | `ncclCommInitRank` |
| `...Dev` | **实现层的唯一入口**：接收已确定好的 `cudaDev`、`config`、以及调用者函数名（`funcName`，用于日志/报错定位），把各公共 API 的参数差异**归一化**，然后创建异步任务 | `ncclCommInitRankDev`（`src/init.cc:2477`） |
| `...Func` | **执行层的函数体**：真正完成初始化全部工作的那个函数，由调度层调用 | `ncclCommInitRankFunc`（`src/init.cc:1831`） |
| `...AsyncJob` | 传递参数的**任务结构体**（"集装箱"：把门面层的差异参数装进去，带到执行层） | `ncclCommInitRankAsyncJob`（`src/init.cc:1734-1753`） |

> **怎么一眼分辨"公共"还是"内部"？** 看 `src/init.cc` 里的声明形式：
> - 写成 `NCCL_API(ncclResult_t, ncclCommInitRank, ...)` 的是**导出符号**，外部可调用；
> - 写成 `static ncclResult_t ncclCommInitRankDev(...)` 的是**内部函数**——`static` 意味着**文件作用域**，链接期外部根本找不到它（所以框架想调也调不到）。
>
> 顺带看清 `NCCL_API` 宏本身（`src/include/core.h:16-27`）——它并不是简单的导出标记：
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

> **`ncclInProgress` 是 `ncclGroupEndInternal()`（`src/group.cc:857`）返回的**——也就是 Group 的"发令枪"。
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
①  ncclGroupEndInternal()
        │  （非阻塞分支里 ret = ncclInProgress 后 return）
        ▼
②  ncclCommInitRankConfig 的收尾：
    NCCLCHECK( ncclGroupEndInternal() )
        │  （NCCLCHECK 对 ncclInProgress 放行，不中断）
        ▼
③  ncclCommGetAsyncError( *newcomm, &ret )
        │  （用 comm->asyncResult 覆盖 ret）
        ▼
④  return ret   →   用户拿到 ncclInProgress
```

要特别注意 `ncclCommInitRankConfig` 的收尾这段（`src/init.cc:2677-2687`）：

```cpp
exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());          // ① 这里可能返回 ncclInProgress
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void)ncclCommGetAsyncError(*newcomm, &ret);   // ② 用 comm 自己的状态覆盖 ret
    }
    ...
  }
  return ret;
```

第 ① 步返回 `ncclInProgress` 却**不会**提前结束函数，是因为 `NCCLCHECK` 对 `ncclInProgress` 是**放行**的（`src/include/checks.h:148-156`）：

```cpp
#define NCCLCHECK(call) \
  do { \
    ncclResult_t RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      return RES;   // ← 只有真正的错误才返回；inProgress 会继续往下走
    } \
  } while (0)
```

于是控制流走到 ②，用 `comm` 自己的异步状态覆盖 `ret`——**这才是函数最终返回给用户的值**。

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

##### 顺带把 Group 机制的作用讲清楚

到这儿可以回头理解 Group 到底为什么存在了。`ncclGroupStart()` / `ncclGroupEnd()` 的本质是**"攒批 + 统一发令"**：

- 在 `ncclGroupStart` 与 `ncclGroupEnd` **之间**，所有 API 调用都只是**登记**：把任务塞进 `thread_local` 队列 `ncclAsyncJobs`（`group.cc:31`），**一个都还没真正执行**；
- 只有到了 `ncclGroupEnd`，才一次性决定怎么启动它们（同步 `groupLaunch` 或后台线程）；
- 由于队列是 `thread_local` 的，每个进程各自攒自己那一批，然后**所有 rank 在同一个 Group 边界上一起发令**。

这个设计的价值在于：集合通信/初始化都要求"所有 rank 都到齐才能开始"。如果 rank 0 早早开始 Send 而 rank 1 还没准备好，就会卡死或者走成低效路径。**用 Group 把"下发"和"启动"分离**，能保证所有 rank 把该登记的都登记完，再同时起跑——既避免了时序错配导致死锁，也让多流、多 comm 的操作可以并发启动。

而"非阻塞"只是这个机制上的一个开关（`config.blocking = 0`）：发令之后不等结果，**先把控制权还给调用者**，进度交给后台线程，状态通过 `asyncResult` 暴露给 `ncclCommGetAsyncError`。这就是"非阻塞初始化"的全貌。

##### 反过来说：默认的阻塞模式会怎样？

不设 `blocking = 0`（即默认 `blocking = 1`）时，同样是 `ncclGroupEndInternal` 发令，但走的是另一个分支：**不另起线程，而是在当前线程里把 `groupLaunch` 跑完才返回**（`group.cc:858-866`）。

所以两者的区别，本质就一句话：

> **阻塞 = "发令 + 站在原地等结果"；非阻塞 = "发令 + 先把控制权还给你，结果我另外通知"。**

| 维度 | 阻塞（默认） | 非阻塞（`blocking = 0`） |
|------|-------------|------------------------|
| 函数返回时 | 初始化**已完成**（或已失败） | 可能**仍在进行** |
| 返回值 | 真实结果（`ncclSuccess` / 错误码） | `ncclInProgress` |
| 需要轮询吗 | 不需要 | 必须（否则后续 API 被 `ncclCommEnsureReady` 拦下） |
| 主线程 | 被占住直到结束 | 立刻交还，可去做别的事 |

> **阻塞的代价**：初始化要做 bootstrap 交换、拓扑探测、建连接、注册内存，跨机场景动辄几十毫秒到秒级。这段时间主线程完全被占住。非阻塞模式存在的意义就是**把这段等待利用起来**——先去加载数据、建 pipeline、准备下一批输入。

> ⚠️ 一个约束：**同一个 Group 里不允许混用阻塞和非阻塞 comm**（`group.cc:60-63` 有明确检查，违反直接返回 `ncclInvalidArgument`）。

（顺带一提：阻塞模式下 `asyncResult` 不会被更新——因为"函数返回"本身就是完成信号，不需要第二个通知渠道。它靠分配时清零即等于 `ncclSuccess`，所以后续 `ncclCommEnsureReady` 也不会误拦。）

> 补充一句：`ncclCommInitRankConfig` 自己内部就包了一层 `ncclGroupStartInternal()` / `ncclGroupEndInternal()`（`init.cc:2667` / `init.cc:2679`）。所以即便用户代码里没有写 `ncclGroupStart`，走进 `ncclAsyncLaunch` 时 `ncclGroupDepth` 也已经是 1 —— 这就是为什么它一定会经过"入队 → 发令"这条路，而不是直接同步执行。

#### 2.1.5 执行层：`ncclCommInitRankFunc` 才是真正干活的那个

前两层都在"准备和转交"，真正建立通信器的代码在**执行层**。它的入口长这样（`src/init.cc:1831-1833`）：

```cpp
// src/init.cc:1831-1833
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;  // 取回"集装箱"
  ncclComm_t comm = job->comm;
  ...   // ← 本章 2.3 ~ 2.7 讲的五个阶段，全部发生在这个函数体内
}
```

它接收的参数是**基类指针** `ncclAsyncJob*`（因为调度层要统一处理各种任务），进函数后第一件事就是**向下转型**回具体的 `ncclCommInitRankAsyncJob`，再取出 `comm`、`nranks`、`commId`、`config`、`cudaDev` 等参数。

> **为什么需要一个"任务结构体"来传参，而不是直接传一堆参数？** 因为执行可能发生在**另一个线程**（非阻塞初始化，见 2.1.4）或**稍后的 Group 结束时刻**——那时门面函数的栈帧早就销毁了，参数必须"打包带走"。`ncclCommInitRankAsyncJob`（`src/init.cc:1734-1753`）就是这个集装箱，它还容纳了**多个门面 API 共用的字段**：
>
> | 字段分组 | 服务于哪个门面 API |
> |----------|-------------------|
> | `nranks, myrank, nId, commId, config` | `ncclCommInitRank` / `Config` / `Scalable` |
> | `parent, color, key, childCount` | `ncclCommSplit`（从父通信器派生） |
> | `excludeRanksList, excludeRanksCount` | `ncclCommShrink`（剔除某些 rank） |
> | `isGrow` | `ncclCommGrow`（扩容） |
> | `funcName[NCCL_COMMINIT_FUNCNAME_LEN]` | 记录"是哪个 API 发起的"，用于日志与报错定位 |
>
> 也就是说：**门面层的 API 数量还会继续增加（split/shrink/grow 都复用了这条链），但实现层和执行层不用重复造轮子**——这正是分层收敛的价值。

#### 2.1.6 上层框架用哪个？为什么要提供这么多变体

**框架只能看见公共 API**（`Dev`/`Func` 是 `static` 的）。实践上，PyTorch 的 `ProcessGroupNCCL` 走的是 **`ncclCommInitRankConfig`**——因为它需要传递 `ncclConfig_t`（超时、blocking、splitShare 等），而基础版 `ncclCommInitRank` 没有配置入口，只作为老版本 NCCL 的回退路径。

> 你可以自己验证：在 PyTorch 源码 `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` 里搜索 `ncclCommInitRankConfig`，会看到它在创建 communicator 的辅助函数里被调用。
>
> **这也解释了 NCCL 为什么要提供这么多变体**：不同调用者的需求不同（要不要传配置、规模大到需要多 root、是不是单进程管多卡），但**实现只有一份**（全部收敛到 `ncclCommInitRankDev` → `ncclCommInitRankFunc`）。这是典型的"门面函数 + 统一内核"设计。

### 2.2 初始化全景图：五个阶段

搞清 API 家族后，再看整个初始化的**阶段划分**。`ncclCommInitRankFunc`（真正干活的执行体）做的事可以归纳为五个阶段：

| 阶段 | 目标（职责） | 关键函数 | 产物 |
|------|-------------|----------|------|
| **① 入场券** | 生成全局唯一的 ID，让各 rank 能找到同一个"集会地点" | `ncclGetUniqueId` → `bootstrapGetUniqueId` | `ncclUniqueId`（含 root 的 IP:port + magic） |
| **② 骨架 + 控制面** | 分配 `ncclComm` 空壳；各 rank 建立**不依赖 GPU 互联**的带外通道 | `ncclCommInitRankDev` 分配骨架 → `bootstrapInit` | 空的 `ncclComm` + `comm->bootstrap` |
| **③ 信息交换** | 通过控制面做两轮 AllGather：交换各自的 peerInfo、以及图计算结果 | `bootstrapAllGather` ×2（在 `initTransportsRank` 中） | 全量 peer 信息 + 各算法的图参数 |
| **④ 拓扑与图** | 探测本机硬件拓扑，搜索出 Ring/Tree/CollNet/NVLS 各自的通信图 | `ncclTopoGetSystem`、`ncclTopoCompute` | `comm->topo`、`comm->graphs[]`、`comm->nChannels` |
| **⑤ 连接与资源** | 按图把 Channel → Connector → Transport 一层层建起来，映射真实物理资源；初始化 Proxy | `setupChannel`、`ncclTransportP2pSetup`、`ncclTransportRingConnect`、`ncclProxyInit` | 可用的 `comm->channels[]` |

> **为什么必须先有"控制面"（阶段②③）才能做后面的？** 因为一个典型的先有鸡还是先有蛋问题：GPU 之间的高速链路（NVLink/RDMA）要建立连接，双方必须先交换各自的**地址/句柄信息**（比如对端显存的 IPC handle、网卡的 QP 号）。可这些信息怎么交换？此时 GPU 链路还不存在——所以 NCCL 先用最朴素可靠的 **TCP socket**（同机则用 Unix Domain Socket）建一条"带外通道"，用它把该交换的信息交换完，再据此建立高速链路。这就是阶段②③存在的意义。

### 2.3 阶段一：入场券 —— `ncclGetUniqueId`

**职责**：生成一个全局唯一 ID，让分处不同进程/机器的所有 rank 知道"我们要加入的是同一个通信域"。它是初始化流程里**唯一需要用户自己做一次跨进程分发**的东西（通常由 rank 0 生成，再通过 MPI/文件/env 广播给其他 rank）。

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
> - `ncclUniqueId` 本质是 **bootstrap handle 的拷贝**——整个结构只有三个字段（`src/include/bootstrap.h:14-18`）：`uint64_t magic`（校验值，`bootstrap.cc:446` 写入 `NCCL_MAGIC`）、`union ncclSocketAddress addr`（root 的地址，即 IP:端口）、`int nRanks`。所以它**不包含任何 GPU 信息**，只是个"集会地点 + 规模"的描述。
> - 它必须能塞进 128 字节：`nccl.h.in:40-41` 定义 `NCCL_UNIQUE_ID_BYTES = 128`，并由 `static_assert(sizeof(ncclBootstrapHandle) <= sizeof(ncclUniqueId))` 在**编译期**保证装得下（`bootstrap.h:19-20`）。由于两者大小/对齐不同，函数里先 `memset` 清零再 `memcpy`，避免残留的未初始化数据被当作有效内容。
> - 它**必须早于** `ncclCommInitRank` 调用，且同一个 ID 可以被**多个** communicator 使用（每次 init 会各自建一套连接）。

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
| `bootstrapAllGather` | 所有 rank 交换数据（阶段三的主力） |
| `bootstrapSend` / `bootstrapRecv` | 点对点数据交换 |
| `bootstrapBarrier` | 全局栅栏同步 |
| `bootstrapBroadcast` | 广播 |
| `bootstrapIntraNodeBarrier` / `bootstrapIntraNodeAllGather` | 同节点内的优化版本（走 UDS，不绕 root） |

**实现思路**（`src/bootstrap.cc`）：root rank 创建 TCP 监听 socket，其余 rank 连向 root，形成**星形拓扑**；跨 rank 的 AllGather 由 root 中转。同一个节点内的 rank 之间则通过**共享文件描述符（Unix Domain Socket）**建立直连，避免都挤到 root 上。

> **多 root 的意义**：`ncclCommInitRankScalable` 传的 `nId` 就是"用几个 uniqueId"，即**建多个 root**——因为在大规模集群里，让几千个 rank 都连向同一个 root 会形成瓶颈。这是"Scalable"这个名字的由来。

### 2.5 阶段三：信息交换 —— 两轮 AllGather

**职责**：把"每个 rank 只有自己的局部认知"变成"所有 rank 拥有一致的全局认知"。这一步在 `initTransportsRank`（`src/init.cc:965`）里完成，靠的就是 Bootstrap 的 AllGather。**为什么恰好是两轮？** 因为中间隔着"拓扑发现与图计算"这个必须各 rank 独立完成的步骤：

```
第一轮 AllGather（交换"我是谁、我在哪"）
      ↓
各 rank 独立探测本机拓扑、搜索通信图（阶段四，见 2.6）
      ↓
第二轮 AllGather（交换"我算出来的图长什么样"，收敛到一致）
```

#### 第一轮：交换 `ncclPeerInfo`（认识彼此）

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

> 这一轮的产物是 `comm->peerInfo[]`——一个**长度为 nRanks 的数组**，之后"我能不能和 rank r 直连走 NVLink"这类判断全靠它。

#### 第二轮：交换图计算结果（达成一致）

```cpp
// src/init.cc:1283 — 第二轮 AllGather：交换图计算结果
bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data));
```

交换的是各 rank **各自算出的图参数**：各算法（Ring/Tree/CollNet/NVLS）的 channel 数、带宽估计、跨 NIC 信息、拓扑排序后的 rank 序列等。之所以要再交换一轮，是因为**阶段四的拓扑探测和图搜索是"各 rank 独立算"的**：

- 每台机器只看得见自己的本地拓扑（自己那几张卡、自己的网卡）；
- 但一个集合通信图（比如 Ring 的环序、Tree 的父子关系）必须是**全局一致**的——否则 rank A 以为下一跳是 B，B 却以为下一跳是 C，通信立刻错乱。

所以第二轮 AllGather 的作用是"**对答案**"：把各自的局部计算结果汇总，取全局最优/一致的方案（例如各 rank 上报自己到 NIC 的带宽，取最小值作为公共可用带宽），最终确定 `comm->nChannels` 等全局参数。

### 2.6 阶段四：拓扑发现与图搜索

**职责**：回答两个问题——**（1）** 这台机器/整个集群里的 GPU、NVSwitch、网卡是怎么连的？（拓扑发现）**（2）** 在这种连接关系上，Ring/Tree/CollNet/NVLS 各自应该按什么顺序组织 rank、用几条 channel？（图搜索）

产物是 `comm->topo`（硬件拓扑图）和 `comm->graphs[]`（各算法的通信图），以及全局参数 `comm->nChannels`。

#### 2.6.1 拓扑发现：把硬件画成一张图

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

#### 2.6.2 图搜索：为每种算法找一条最优路径

以 AllReduce 为例，NCCL 会为 Ring、Tree 两种算法**各算一份图**（还有可选的 CollNet、NVLS）：

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

### 2.7 阶段五：连接与资源 —— Channel / Connector / Transport 是怎么落地的

**职责**：把阶段四算出的"**逻辑**通信图"（Ring 的环序、Tree 的父子关系）翻译成**真实的连接与物理资源**——为每个 channel 分配连接表、为每条连接挑一种传输方式、和对端交换句柄、映射显存/网卡资源，最后把连接信息下发到 GPU。

这是整章最"硬"的一节，我们从结构层次讲起，再讲建链过程。

#### 2.7.1 全景：一条五层结构链

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
| **① 通信器** | `ncclComm` | 一个通信域的全局上下文 | 1 | `comm.h:523-797` |
| **② 通道** | `ncclChannel` | 一条独立的并行通信管道（承载一份数据切片） | `nChannels`（≤ `MAXCHANNELS`=64） | `comm.h:150-172` |
| **③ 连接表** | `ncclChannelPeer* peers[]` | "本 rank 在这个通道里，与各 peer 的连接载体"；**下标就是 rank 号** | 每通道 `nRanks` 项 | `comm.h:151` |
| **④ 连接子** | `ncclConnector` | 一条具体连接：走哪个传输、有哪些资源、进度指针在哪 | 每 peer `2 send + 2 recv` | `device.h:165-173` |
| **⑤ 传输虚函数表** | `ncclTransportComm` | 这条连接用**哪类**物理链路（P2P/SHM/NET/CollNet）及其操作集 | 全局 4 种实例 | `transport.h:117-134` |
| **⑥ 物理资源** | `p2pResources` / `shm*Resources` / `sendNetResources` … | 真正的东西：显存映射、主机内存、网卡 QP | 每条连接一份 | 各 `transport/*.cc` |

> **一句话理解这个层次**：**通信器**划定"谁能和谁说话"；**通道**把数据切成几份并行处理；**连接表**用 rank 号索引"我和每个人的连接"；**连接子**描述"这一条连接"；**传输**决定"这条连接走什么物理链路"；**物理资源**才是内存里真实的句柄、地址和队列。

#### 2.7.2 第一步：建通道，同时分配它的"连接表"

阶段四算出了每个 channel 的 ring 环序，接下来按它建通道并分配连接表（`src/init.cc:791-808`）：

```cpp
// src/init.cc:791-793（节选）
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  NCCLCHECK(initChannel(comm, channelId));   // ① 分配该通道的连接表（下面展开）
  struct ncclRing* ring = &comm->channels[channelId].ring;
  ...                                        // ② 按 ringRanks 填 ring->userRanks / rankToIndex / index
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

#### 2.7.3 第二步："登记"要建哪些连接

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

#### 2.7.4 第三步：两段式建链 —— setup（换句柄）→ connect（映射资源）

这是整个阶段五最核心的机制，也是最容易被忽略的细节。NCCL 把"建立一条连接"拆成了**两段**，对应 `ncclTransportComm` 里的两个函数指针：

| 段 | 函数 | 做什么 | 是否跨 rank 通信 |
|----|------|--------|------------------|
| **setup** | `transportComm->setup(...)` | 生成"**让对方能连上我**"的信息（信封），并通过 **bootstrap** 与对端交换；同时创建 `transportResources`（物理资源容器） | ✅ 交换（用阶段二的控制面） |
| **connect** | `transportComm->connect(...)` | 拿到对端信封后，**真正完成映射/连接**，把结果填进 `connector->conn` | 部分（如 NET 需改 QP 状态） |

两段之间传递的"信封"是一个**不透明缓冲区**：

```cpp
// src/include/transport.h:70-72
struct ncclConnect {
  char data[CONNECT_SIZE];   // 内容由各 transport 自己解释
};
```

> 为什么设计成不透明缓冲区？因为不同传输要交换的东西完全不同：P2P 交换的是**显存缓冲区的 IPC/cuMem 句柄**，NET 交换的是**网卡 handle（含 QP 信息）**，SHM 交换的是**共享内存 IPC 描述符**。用固定大小的 `char[]` 统一信封，传输层就能横向插拔——这正是 4 种 transport 能共用一套流程的原因。

**connect 之后，连接信息会被下发到 GPU**（`src/transport.cc:242-253`）：

```cpp
// src/transport.cc:242-253（节选）
struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
if (conn->connected == 0) {
  NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
  if (ret == ncclSuccess) {
    conn->connected = 1;
    /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is a device memory access. */
    CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex],
                                  &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                  ret, fail);
  } else if (ret == ncclInProgress) {
    allChannelsConnected = false;   // ← 连接可能异步进行，未完成则稍后重试
  }
}
```

注意 `ret == ncclInProgress` 这一分支：**建连是允许异步的**（比如某些传输的连接需要等网卡状态），所以 `ncclTransportP2pSetup` 内部会循环重试直到所有 channel 都连上。

#### 2.7.5 连接子里面到底装了什么：`ncclConnInfo`

`connect` 的成果全部体现在 `connector->conn`（类型 `ncclConnInfo`，`src/include/device.h:133-151`）里。这个结构体是**内核真正使用的东西**，值得逐字段看：

| 字段 | 含义 |
|------|------|
| `buffs[NCCL_NUM_PROTOCOLS]` | 三个协议（LL/LL128/SIMPLE）各自的**缓冲区地址**——发送方向指向对端显存（或本地发缓冲），接收方向指向本地缓冲 |
| `tail` / `head` | **进度指针**：head 由发送侧推进、tail 由接收侧推进；GPU 内核与 Host Proxy 线程就是靠这两个指针"对话"（呼应 5.3 节的 Proxy 机制） |
| `connFifo` | GPU ↔ Proxy 之间传递控制信息的**环形缓冲** |
| `ptrExchange` / `redOpArgExchange` | 直连（direct pointer）模式下的**指针交换区**与 reduce 参数交换区 |
| `mhandles[]` / `netDeviceHandle` | 内存**注册句柄**（RDMA 的 MR）、网卡设备句柄 |
| `flags` | 连接属性（如 `NCCL_P2P_WRITE` / `NCCL_P2P_READ` / `NCCL_DIRECT_NIC`） |
| `stepSize` / `step` / `llLastCleaning` | 流水线的步进与内部状态 |

**以 P2P（同机 GPU 直连）为例，看 connect 具体填了什么**（`src/transport/p2p.cc:536-557`）：

```cpp
// src/transport/p2p.cc:536-557（节选）
struct p2pResources* resources = (struct p2pResources*)send->transportResources;
// ① 把对端的进度标记内存映射进本进程地址空间
NCCLCHECK(p2pMap(comm, &send->proxyConn, comm->peerInfo + rank, comm->peerInfo + info->rank, &info->p2pBuff,
                 (void**)&remDevMem, &resources->recvMemIpc));
// ② 切分缓冲区：把对端显存里的各协议缓冲区首地址，逐个填进 conn.buffs[]
char* buff = (char*)(remDevMem + 1);
for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
  send->conn.buffs[p] = buff;
  buff += comm->buffSizes[p];
}
send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
```

> 这两句就是 **"P2P 直连"的全部魔法**：`p2pMap` 把对端显存映射到了本进程的地址空间（用的是 CUDA IPC / cuMem / 或对端在同进程时直接给指针），于是 `conn.buffs[p]` 就是**一个本进程可写的、指向对端显存的地址**——之后 GPU 内核用普通的 `st.global` 指令写这个地址，数据就出现在对端显存里了。**不需要网卡、不需要 CPU 搬运**。

#### 2.7.6 四类传输各自对应哪些物理资源

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

而 NET transport 与网卡的交互完全通过 **Net Plugin API**（`src/include/plugin/nccl_net.h`，v12）：**setup** 阶段向对端提供网卡的 `handle`（内含 QP 信息），**connect** 阶段把对端 QP 置为 RTS（Ready To Send）状态——这也是"国产网卡替换"最关键的一个接口（见 2.8 节）。

#### 2.7.7 第四步：收尾 —— Proxy 与运行时资源

各算法/传输的建链依次完成后，还有一批"运行时资源"要初始化：

```cpp
ncclTransportP2pSetup(comm, NULL, 1);      // P2P 连接（connIndex=1，点对点 Send/Recv 用）
ncclTransportRingConnect(comm);            // Ring 连接（connIndex=0，集合通信用）
ncclTransportTreeConnect(comm);            // Tree 连接（connIndex=0）
ncclCollNetSetup(comm, parent, graphs);    // CollNet 网内计算资源（若启用）
ncclNvlsBufferSetup(comm);                 // NVLS Multicast 缓冲区（若支持）
ncclProxyInit(comm);                       // ★ Proxy 线程 + 共享内存 op 池
```

> **注意这里 `connIndex` 的差异**：`0` 给集合通信（Ring/Tree/CollNet 共用），`1` 给点对点 Send/Recv——这正是 1.5 节里 `NCCL_MAX_CONNS = 2` 的由来。
>
> `ncclProxyInit` 是"连接层"和"运行时"的分界点：它把 5.3 节讲的 Proxy 线程和共享内存 op 池建起来，从此 GPU 内核与 Host 之间就有了数据搬运的通道。到这一步，`comm->channels[]` 里的连接**全部可用**，通信器正式"就绪"。

#### 2.7.8 回看这条链：一次调用的完整旅程

把本章五个阶段串成一条线：

```
ncclGetUniqueId
   → 生成"集会地点"（root IP:port + magic）
ncclCommInitRank → ncclCommInitRankDev
   → 分配 ncclComm 空壳（含 abortFlag、魔数哨兵）
ncclCommInitRankFunc（异步任务体，真正干活）
   ├─ 阶段② bootstrapInit ──────────── 建带外控制面（TCP/UDS）
   ├─ 阶段③ 第一轮 AllGather ───────── 交换 peerInfo（我是谁、我在哪）
   ├─ 阶段④ ncclTopoGetSystem + ncclTopoCompute
   │         ──────────────────────── 探测拓扑 → 搜索 Ring/Tree/CollNet/NVLS 图
   ├─ 阶段③ 第二轮 AllGather ───────── 交换图计算结果（全局达成一致）
   └─ 阶段⑤ setupChannel / Connect 系列
             ├─ channels[c].peers[]    分配连接表（按 rank 索引）
             ├─ ncclTransportP2pConnect 登记要建的连接（位图）
             ├─ setup（交换句柄 via bootstrap）
             ├─ connect（映射显存/QP，填 conn，下发到 device）
             └─ ncclProxyInit           建 Proxy 线程与 op 池
   → 通信器可用，可以开始 ncclAllReduce
```

> **这条链体现的设计哲学**：**逻辑与物理分离**。阶段四产出的是纯粹的"逻辑图"（谁和谁相邻），阶段五才把它落到具体传输和资源上。好处是：换一种物理链路（比如国产网卡、自研互联）只需要替换阶段五的 transport 实现，前面的拓扑/图逻辑完全不用动。

### 2.8 国产芯片适配要点

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
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)          // ① 入口：先 push（此时还不知道 payload！）
  ...
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));  // ② 执行初始化
  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));  // ③ 尾部：补 payload
  return ncclSuccess;
}  // ④ 函数退出：RAII 析构，pop 时把 payload 一起交给 Profiler
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
| Device 侧 | `src/include/comm.h` | 进度标记结构体：`ncclSendMem`（`comm.h:53`，含 `head`）、`ncclRecvMem`（`comm.h:67`，含 `tail`），就是 GPU 内核和 Proxy 线程之间"对话"用的共享内存 |

**工作模式（以 NET 跨节点发送为例，串起上面两侧）：**

1. **Device 侧**：GPU 内核把数据算好放进发送缓冲区，然后**写共享内存里的进度标记**（`ncclSendMem.head` / 对应 fifo 的 `step`），表示"这一块数据已经就绪，可以发了"；
2. **Host 侧**：proxy 线程（`ncclProxyProgress`）**轮询**这些进度标记，发现新数据后，构造/取出 `ncclProxyOp`，走对应的 `proxyProgress`（NET 是 `sendProxyProgress`）；
3. **Host 侧**：`sendProxyProgress` 调用 net plugin 的 `isend()`，把 GPU 缓冲区里的数据经 RDMA 网卡发出去；
4. **接收侧反向**：对端 proxy 线程 `irecv()` 收数据到 GPU 接收缓冲区，更新 `ncclRecvMem.tail`；对端 GPU 内核轮询到 `tail` 前进后，才消费数据。

**关键数据结构（Host 侧）：**

- `ncclProxyOp`（`src/include/proxy.h:73`）：单个 Proxy 操作描述符，含连接指针、字节数、opCount、root 等；
- `ncclProxyArgs`（`proxy.h:185`）：一批 Proxy 操作参数，带 `progress` 函数指针，传递给 Proxy 线程；
- `ncclProxyState`（`proxy.h:333`）：Proxy 全局状态，含线程句柄、连接池、op 池等。

Proxy 机制在 `ncclTransportP2pSetup` 完成后初始化（`ncclProxyInit`），其生命周期与通信器绑定。

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

# 第二部分：DeepEP —— MoE 架构下的集合通信利器

> 前面我们用很大篇幅理解了 NCCL 本身。现在换一个角度：**如果让你基于 NCCL 去构建一个 MoE（Mixture of Experts，混合专家）模型的通信层，你会怎么做？**
>
> DeepSeek 开源的 **DeepEP** 就是对这个问题的回答。它是一个专门为 MoE 专家并行（Expert Parallelism）设计的通信库，因为解决了 MoE 训练/推理中最棘手的通信问题，而成为各方积极参考的对象（包括各家大模型团队和自研通信库）。
>
> 这一部分会假设你**还没读过 DeepEP 的代码**，从"MoE 到底需要什么样的通信"讲起，再一步步拆解 DeepEP 的架构，以及它和 NCCL 之间真实、精确的调用关系。

---

## 六、为什么 MoE 需要专门的通信库

### 6.1 版本背景：先搞清楚 V1 与 V2

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

> **本笔记的讲述策略**：第七章的"DeepEP 怎么调用 NCCL"主要沿着 V2 的 `ElasticBuffer` 主路径讲（因为这是当前推荐、也是最能体现"GIN 底层能力"的路径），但在涉及 V1 特有的地方（比如 NVSHMEM 跨节点、`low_latency_*` 接口）会**明确标注是 V1**，避免混淆。

### 6.2 先理解 MoE 的通信特征

在普通的稠密模型（Dense）训练里，AllReduce 是主角：每个 GPU 算一部分梯度，然后所有人做一次全局归约。数据流是**对称的、规整的**——每个 GPU 发出去的字节数和收进来的字节数都一样，而且每个 GPU 都要和"所有"其他 GPU 通信。这正是 NCCL 最擅长处理的场景。

但 MoE 完全不一样。MoE 模型里，每一层的计算会被路由到不同的"专家"（expert）上，而每个 token 只会被少数几个（比如 top-2）专家处理。于是出现了一种全新的通信模式——**all-to-all**（全对全）：

- **Dispatch（分发）阶段**：每个 GPU 手里的一批 token，需要按"路由结果"发送到**各自不同的目标专家所在的 GPU**。token A 去 GPU 1 的专家 3，token B 去 GPU 5 的专家 7……**每个 token 的去向都不同**。
- **Combine（回收）阶段**：专家算完之后，每个 token 的结果又要被送回它**原来的 GPU**，做加权求和（因为一个 token 可能被多个专家处理）。

这套通信有几个让 NCCL 的通用集合通信"使不上力"的特点：

1. **数据量严重不均衡**：每个 GPU 分到的 token 数、每个专家收到的 token 数，在每一步都是动态变化的、且极不均匀。可能 GPU 0 要发 1000 个 token，GPU 3 只发 10 个。
2. **通信模式是稀疏的、数据相关的**：谁发给谁、发多少，取决于这步的路由结果，无法预先固定。
3. **通信和计算需要深度重叠**：专家计算（GEMM）很贵，通信必须尽量和它 overlap，否则通信会拖垮整体吞吐。

NCCL 本身其实也有 `AllToAll` 集合操作，但它是"规整的" all-to-all（每个 rank 发给每个 rank 等量数据），无法直接表达 MoE 这种"不规则、动态、稀疏"的 all-to-all。于是，需要一个专门的库来做这件事——这就是 DeepEP 的用武之地。

### 6.3 DeepEP 解决的三个核心场景

DeepEP 的 `Buffer` 类（V1，`deep_ep/buffers/legacy.py`）在文档里开门见山地列出了它支持的三类通信（翻译如下）：

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

> **⚠️ 先纠正一个版本误区**：很多资料说"DeepEP 跨节点用 NVSHMEM"，这其实是 **V1 的做法**。在 **V2** 里，DeepEP 已经把跨节点 RDMA 也统一交给了 **NCCL Gin**（即上面第 2 点的 `gin.put`/`gin.get`，它们本身就支持跨节点），NVSHMEM 仅在 legacy 兼容路径里保留。所以本节的讲述以 **V2 的 NCCL Gin 主路径**为准，V1 的 NVSHMEM 会在 7.5 节单独说明。

下面逐条拆解，每条都标出对应的源码位置，请你打开源码对照着看。

### 7.1 第一步：拿到 NCCL communicator（但为了当"钩子"而非做集合通信）

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

### 7.5 跨节点 RDMA 的版本演进：从 NVSHMEM（V1）到 NCCL Gin（V2）

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

> 说明：下面讲的 `low_latency_dispatch` / `low_latency_combine` 是 **V1（legacy）** 的专用接口（`deep_ep/buffers/legacy.py:553-713`）。在 V2 里，低延迟能力已经**合并进统一的 `ElasticBuffer` 接口**，不再有独立的 `low_latency_*` API（V2 的 `dispatch`/`combine` 通过 `async_with_compute_stream` 等参数表达同样的"少 SM、与计算重叠"诉求）。这里仍以 V1 的低延迟内核为例，因为它最能讲清楚"面向推理的低延迟"到底做了哪些针对性设计：

`low_latency_dispatch` / `low_latency_combine`（V1，`deep_ep/buffers/legacy.py:553-713`）是 DeepEP 面向推理的招牌功能，几个关键设计：

- **纯 RDMA（IBGDA）**：所有 rank 都通过 RDMA 直接互访，不走 NVLink，把延迟压到最低；
- **固定缓冲 + 无 CPU 同步**：接收缓冲是预先固定分配的（`num_max_dispatch_tokens_per_rank` 决定上限），并且**不把"收到多少 token"同步回 CPU**（`legacy.py:600` 注释明确说明），这样才兼容 CUDA Graph；
- **接收 hook**：`return_recv_hook` 允许"只发起 RDMA 请求、不等待数据到达"，把等待推迟到真正需要数据的那一刻，进一步和计算重叠；
- **FP8 转换**：通信过程中直接做 FP8 量化（`use_fp8` 参数），在低延迟模式下顺带减少传输数据量。

### 8.4 丰富的工程细节

> 说明：下面这些工程优化是 DeepEP 的**共性设计思想**，V1/V2 都有对应实现，只是接口命名不同（V1 是 `Buffer` + 元组 handle，V2 是 `ElasticBuffer` + `EPHandle` 对象）。这里以 V1 的接口名为例标注源码位置，便于你在 legacy 代码里定位；V2 里同样能找到等价物。

- **通信 handle 复用**：dispatch 返回的 `handle` 里缓存了前缀和矩阵、源索引等布局信息，下一次 combine 或同布局的 dispatch 可以直接复用，省去重复计算（V1 见 `legacy.py:385-405`，V2 见 `EPHandle` + `elastic.py` 的 cached handle 逻辑）；
- **专家对齐（expert_alignment）**：把每个专家收到的 token 数对齐到某个倍数，方便后续 GEMM 内核的访存对齐；
- **rank 动态屏蔽（mask）**：`low_latency_update_mask_buffer` 等接口支持在通信中动态屏蔽某个 rank，用于在线服务的故障隔离（`legacy.py:672-698`）；
- **性能统计内置**：`dispatch_wait_recv_cost_stats`、`combine_wait_recv_cost_stats` 等参数能在通信中直接统计"等待接收"的耗时，用于定位慢节点（`legacy.py:577-579`）。

### 8.5 建议你重点阅读的源码路径

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

- NCCL 官方源码仓库：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)（版本 2.30.x）
- NCCL 官方文档：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- NCCL 开发者文档：[NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2206/nccl-developer-guide/)
- DeepEP 官方源码仓库：[github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)（本笔记基于 v2.1.0）
- DeepEP 技术博客：DeepSeek 官方发布的 DeepEP 解读文章（含低延迟内核、混合通信等设计细节）
- DeepEP V1 legacy 文档：官方仓库 `docs/legacy.md`（NVSHMEM 时代的接口与性能数据，用于和 V2 对比）

---

*声明：本文是作者学习 NCCL 与 DeepEP 源码过程中的阶段性笔记，内容基于 NCCL master 分支与 DeepEP 官方仓库源码整理。由于这是一份仍在持续完善的初稿，文中可能存在表述不准确或理解不完整之处，欢迎通过邮件联系指正。*
