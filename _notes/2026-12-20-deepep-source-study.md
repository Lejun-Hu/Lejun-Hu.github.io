---
title: "【未完成】DeepEP 源码学习笔记"
permalink: /notes/deepep-source-study/
date: 2026-12-20
category: "集合通信"
tags: ["DeepEP", "MoE", "Expert Parallel", "NCCL", "GIN", "对称内存", "NVSHMEM", "源码分析"]
description: "DeepEP 源码学习笔记（初稿）：从 MoE 的通信特征出发，沿 Python 入口、C++ runtime、JIT 编译到 GPU 内核逐级拆解调用链，说明 DeepEP 如何复用 NCCL 的通信器、对称内存与 GIN 后端完成不规则 all-to-all。"
published: true
---

本文是《[NCCL 源码学习笔记](/notes/nccl-source-study/)》的姊妹篇。NCCL 笔记讲的是通用集合通信库如何建网、如何调度、以及 GIN 与对称内存这些底层能力的机制；本文讲的是**一个上层库如何把这些能力当作接口使用**，去解决 NCCL 的集合操作本身不擅长的问题。前者回答「能力从哪来」，后者回答「能力被谁用、怎么用」，两篇合起来才是一条完整的链路。

DeepEP 是 DeepSeek 开源的专家并行（Expert Parallelism, EP）通信库，全名 DeepEveryParallel（`README.md:3`）。它面向 MoE（Mixture of Experts，混合专家）模型训练与推理中的 token 分发与回收，提供高吞吐、低延迟的 all-to-all GPU 内核；all-to-all 指每个 rank 都向其余 rank 发送数据、同时从其余 rank 接收数据的通信模式，`README.md:3` 还说明该库另外提供 PP、CP、Engram 的实验性原语。本笔记基于本地的 **DeepEP v2.1.0** 源码（`deep_ep/__init__.py:97` 的 `__version__ = '2.1.0'`），并对照 NCCL master 分支核实所有 NCCL 侧 API。文中 DeepEP 侧路径相对 `F:/User/Desktop_2/DeepEP-main`，NCCL 侧路径相对 `F:/User/Desktop_2/nccl-master`，每个函数、字段都标注源码位置，方便打开源码对照。

全文反复用到几个底层缩写，先在此给出定义与官方出处，后文涉及具体机制时再展开其作用。GIN 是 GPU-Initiated Networking（NCCL `docs/userguide/source/usage/deviceapi.rst:20`），指由 GPU 内核线程直接发起网络传输的设备侧 API；LSA 是 Load/Store Accessible（同文档 `:13`，GIN 设计文档 `docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md:94` 写作 Load/Store Access），指同节点内用 load/store 指令访问对端显存的能力。SM 是 Streaming Multiprocessor（`docs/legacy.md:9`），QP 是 InfiniBand/RDMA 的 Queue Pair（NCCL `docs/userguide/source/env.rst:1015` 称 queue pairs，DOCA 头文件 `src/transport/net_ib/gdaki/doca-gpunetio/include/device/doca_gpunetio_dev_verbs_qp.cuh:52` 给出全称），TMA 是 Tensor Memory Accelerator（NCCL `contrib/nccl_ep/README.md:36`）。这些能力的内部机制见 NCCL 笔记第四章，本文只关心 DeepEP 调用它们的接口与时机。

按初学者的阅读顺序，全文分三章：先讲清楚「MoE 的通信为什么需要专门的库」，再沿真实调用链逐层读代码，最后归纳设计要点与源码阅读路径。下表给出三章各自的问题与阅读收获，读表时抓住一条因果链：第一章确认通信形态，第二章定位实现层次，第三章归纳设计取舍；第二章是全文重点，后续两章都建立在它的调用链之上。

| 章节 | 关注的问题 | 读完应掌握 |
|------|-----------|-----------|
| 一、问题背景 | MoE 的通信与常规集合通信差在哪里、V1/V2 有什么区别 | 为什么要另写一个库、两代架构的差异 |
| 二、代码骨架 | 一次 `dispatch` 从 Python 调用到 GPU 内核经过了哪些层 | 完整调用链、每层的职责、NCCL 能力在何处被消费 |
| 三、设计要点 | 这些实现解决了什么工程问题 | 定制内核、通信计算重叠、低延迟设计与阅读路径 |

读表结论：三章按「问题 → 实现 → 归纳」推进，第二章的调用链是后两章所有判断的依据；若只关心 DeepEP 如何复用 NCCL，可以从第二章直接进入。

前置知识：第二章大量引用 NCCL 的 `ncclDevComm`、`ncclWindow_t`、GIN 与 Team 语义。`ncclDevComm` 是 NCCL 为设备侧准备的资源上下文，`ncclWindow_t` 是对称内存窗口句柄，Team 是 communicator 内的 rank 子集（`src/include/nccl_device/core.h:45-47` 定义 `ncclTeam`，`docs/userguide/source/usage/deviceapi.rst:211-228` 给出 World/Lsa/Rail 三个预定义 Team）。建议先读 NCCL 笔记的第四章（GIN 详解），特别是设备侧 context、窗口注册与 QP 参数传递部分。

---

## 一、问题背景：MoE 为什么需要专门的通信库

本章位于全文起点，不进入 DeepEP 的代码，先把问题本身说清楚：MoE 的通信模式与常规集合通信有什么本质差别，由此产生了哪几类场景，以及为什么读代码前必须先分清 V1 与 V2。这三节分别对应「是什么」「有哪些形态」「读哪一套代码」；只有先确认通信形态，第二章的调用链与设计选择才有判断依据。

### 1.1 版本背景：V1 与 V2

上一节末尾提出「读代码前要先分清版本」，本节回答的正是这个问题：当前仓库里两套接口并存，哪一套才是全文的主路径。它从「DeepEP 依赖 NCCL 能力」这一全文前提引出来——若把两代代码混着读，会得到完全相反的结论。读完应能回答三个具体问题：当前版本号是多少、两代实现各在哪些文件、后端差异决定了要打开哪些源码。

DeepEP 目前有两代实现，**架构差异很大，尤其是对 NCCL 的依赖**。当前源码是 v2.1.0（`deep_ep/__init__.py:97`），两代接口并存：`deep_ep/buffers/legacy.py` 的 `Buffer` 是 V1，`deep_ep/buffers/elastic.py` 的 `ElasticBuffer` 是 V2。

官方 README 对 V2 的定位（`README.md:9`）：V2 是对专家并行的完整重构，用少得多的 SM 资源达到极致性能，支持更大的 scale-up 与 scale-out 规模，并把后端从 NVSHMEM 切换为更轻量的 NCCL Gin 后端。README 的 New features 部分还给出了可核对的具体变化（`README.md:13-25`）：全量 JIT（Just-In-Time，运行期编译）编译、复用已有 NCCL communicator、高吞吐与低延迟 API 统一为单个 `ElasticBuffer` 接口、EP 规模上限到 2048、SM 占用从 24 降到 4–6。其中 scale-up 指节点内 NVLink 域，scale-out 指跨节点 RDMA 域，这一划分由 `csrc/kernels/backend/nccl.cu:56-60` 的 `get_logical_domain_size` 决定，2.2 节会给出具体赋值。

下表按「依赖什么」而不是「性能数字」对比两代，因为前者决定读代码时该打开哪些文件；读表时重点看「跨节点后端」与「NCCL communicator」两行，它们直接决定了第二章的主路径。

| 维度 | V1（`Buffer`） | V2（`ElasticBuffer`） |
|------|----------------|----------------------|
| 跨节点 RDMA 后端 | NVSHMEM + IBGDA | NCCL Gin |
| 机内通信 | CUDA IPC 共享内存（`csrc/legacy/buffer.hpp:49,132-136`） | NCCL 对称内存窗口 + LSA 指针 |
| NCCL communicator | **不使用** | 复用 PyTorch 的 `comm`；旧版 PyTorch 或需要 CPU 段时自建（`deep_ep/utils/comm.py:42-75`） |
| 内存分配 | `nvshmem_align` + CUDA IPC | `ncclMemAlloc` 或 GPU/CPU 混合的弹性内存 |
| API 形态 | 高吞吐 `dispatch`/`combine` 与低延迟 `low_latency_*` 分开 | 统一为 `ElasticBuffer` 的 `dispatch` / `combine` |
| 内核交付方式 | 安装期编译 | 运行期 JIT 编译（`README.md:13`） |

读表结论是：V2 把对 NCCL 的依赖从「无」变成「贯穿初始化与内核」，因此本文必须以 V2 为主路径。需要特别指出一处容易误判的点：**V1 代码里完全没有 `ncclComm_t`**。V1 的 `Buffer.__init__`（`deep_ep/buffers/legacy.py:33-136`）先构造 C++ runtime（`:92-93`），随后交换设备 ID（`:95-97`）、CUDA IPC handles（`:99-101`）与 NVSHMEM unique ID（`:103-132`），并在 `:106-126` 设置一批 NVSHMEM/IBGDA 环境变量，最后调用 `self.runtime.sync(device_ids, ipc_handles, root_unique_id)`（`:135`）；整个过程不涉及任何 NCCL communicator。相对地，`csrc/kernels/backend/nccl.cu:20-41` 里的 `get_local_unique_id` / `create_nccl_comm` 虽名字像自建通信器，实际只在 V2 的 `comm.py` 走自建分支时调用。2.1 节会沿三条分支说明 V2 何时才需要它们，2.8 节再回到 V1 的初始化路径。

本笔记第二章以 **V2 主路径** 为主，因为它完整使用了 NCCL 的对称内存与 GIN；V1 的 NVSHMEM 与低延迟内核放在 2.8 和 3.3 单独说明。要理解 V2 为何值得单独读，下一节先看它要处理的通信形态。

### 1.2 MoE 的通信特征

上一节确定了读 V2 的代码，本节回答「V2 到底在解决什么通信问题」：把 MoE 的数据流与常规集合操作对照，指出后者无法直接表达的性质。读完应能回答：dispatch 与 combine 各自做什么、这种数据流有哪三个特征、为什么 `ncclAlltoAll` 不能直接套用。

理解 DeepEP 的设计，先看它要解决的通信模式。稠密模型训练里的代表性操作是 AllReduce：每个 GPU（对应一个 rank，即通信组内的进程编号）算一部分梯度，再做一次全局归约。数据流是对称且规整的——每个 rank 发出与接收的字节数相同，通信对象覆盖全体 rank。NCCL 的 Ring/Tree 算法、channel 切分、带宽模型都是围绕这种规整模式设计的。

MoE 不一样。MoE 层里每个 token（一个输入向量）由路由结果决定交给哪几个专家（如 top-2，即取权重最高的 2 个专家），专家分布在不同 GPU 上，于是产生两个阶段：

- **Dispatch（分发）**：本地这批 token 要按各自的路由结果，发往**不同的目标 rank**。token A 去 rank 1 的专家 3，token B 去 rank 5 的专家 7，去向逐 token 不同。
- **Combine（回收）**：专家算完后，结果要送回 token 原来的 rank 做加权求和。它是 dispatch 的逆向路由。

这带来三个 NCCL 通用集合操作难以直接处理的性质：

1. **数据量不均衡**：各 rank 发出的 token 数、各专家收到的 token 数随每步路由变化，可能相差数十倍。
2. **数据相关且稀疏**：谁发给谁、发多少取决于本步 `topk_idx`，无法在初始化时固定。
3. **需要与计算深度重叠**：专家 GEMM（通用矩阵乘）很贵，通信必须尽量与它并行，否则直接拉低吞吐。

NCCL 确实提供 `ncclAlltoAll`，但它是**等量规整**的。头文件注释写明「每个设备向所有其他设备发送 `count` 个值，发往 rank j 的数据取自 `sendbuff+j*count`，从 rank i 收到的数据放在 `recvbuff+i*count`」（`src/nccl.h.in:534-541`，函数声明在 `:542-543`）。每个 (rank, peer) 对的量固定为 `count`、偏移固定为 `j*count`，无法表达 MoE 这种逐 token 变化、各对不等量的发送布局。这正是 DeepEP 要自己写内核的原因。

明确了「规整集合操作不够用」，下一节接着看 DeepEP 把这个不规则问题切成了几类场景，以及不同场景各自优化什么。

### 1.3 DeepEP 面向的三类场景

上一节得到的不规则通信特征，在不同部署形态下需要不同的优化目标，本节回答「DeepEP 按什么维度切分问题」。它从 1.2 节的三个性质引出来：数据量不均衡主要影响吞吐，延迟敏感则与 batch 大小有关。读完应能回答：三类场景各自的数据路径与优化目标、V1 与 V2 在接口上如何处理这种切分。

明确了通信特征，再看 DeepEP 把问题切成几类。V1 的 `Buffer` 类在 docstring 里直接列出支持的三类 all-to-all（`deep_ep/buffers/legacy.py:15-19`，三条列表项在 `:17-19`），这三类也是理解后续内核分流的框架；读表时关注「数据路径」一列，它决定了后续内核走哪条物理链路。

| 场景 | 数据路径 | 优化目标 |
|------|---------|---------|
| 高吞吐节点内（intranode） | NVLink | 充分利用机内带宽 |
| 高吞吐节点间（internode） | RDMA 走跨机 + NVLink 走机内 | 两级带宽叠加、吞吐优先 |
| 低延迟（low-latency） | 纯 RDMA | 推理 decode 阶段的延迟优先 |

读表结论是：前两类面向训练与 prefill，追求带宽；第三类面向推理 decode，追求延迟。`docs/legacy.md:9-11` 补充说明了这两条路线：`docs/legacy.md:9` 的高吞吐内核针对 NVLink 域到 RDMA 域的非对称带宽转发做了优化，`docs/legacy.md:11` 的低延迟内核用纯 RDMA 并采用 hook 式的通信-计算重叠，不占用 SM。

V2 把三者收进统一的 `ElasticBuffer` 接口，但底层仍按拓扑分流——第二章 2.4、2.5 会看到 `is_nvlink_accessible` 决定机内直写还是走 GIN put，这是同一套分流逻辑在新架构里的延续。至此第一章说明了「问题是什么」，第二章接着回答「一次调用具体怎么跑起来」。

---

## 二、从 Python 调用到 GPU 内核：完整代码骨架

这是全文核心章节。上一章说明了「为什么需要专门的库」，本章回答「一次调用具体怎么跑起来」，并按初始化期与调用期两个时机把代码分层。读完应能回答：一次 `dispatch` 从 Python 到 GPU 内核经过哪几层、每层消费哪一类 NCCL 能力、哪些步骤只在构造时做一次。DeepEP 对 NCCL 的使用方式与直觉不同：它**不调用 NCCL 的集合通信 API**，而是复用 NCCL 提供的四项底层能力——communicator 句柄、`ncclDevComm` 设备上下文、对称内存窗口与 LSA 指针、GIN 设备侧 API，分别在 2.1、2.2 与 2.5 展开。

先给出全链路总览。下表回答「哪些步骤属于哪个时机」这一阅读主线，其中「时机」一列区分初始化期与每次调用期，这是避免把不同阶段的调用误读成连续流程的关键。

| 步骤 | 时机 | 入口 | 产物 |
|------|------|------|------|
| 1. 取得 NCCL communicator | `ElasticBuffer` 构造 | `get_nccl_comm_handle`（`deep_ep/utils/comm.py:42`） | `NCCLCommHandle` |
| 2. 建设备上下文、对称内存、窗口 | `ElasticBuffer` 构造 | `NCCLSymmetricMemoryContext`（`csrc/kernels/backend/nccl.cu:62`） | `dev_comm` / `window` / `nvl_window_ptrs` |
| 3. Python 层参数整理 | 每次 `dispatch` | `ElasticBuffer.dispatch`（`deep_ep/buffers/elastic.py:855`） | `runtime.dispatch` 的入参 |
| 4. C++ 层启动准备 | 每次 `dispatch` | `ElasticBuffer::dispatch` → `launch_dispatch`（`csrc/elastic/buffer.hpp:702`、`csrc/kernels/elastic/dispatch.hpp:141`） | `DispatchRuntime::Args` |
| 5. JIT 生成、编译、启动 | 每次 `dispatch` | `generate` → `build` → `launch`（`dispatch.hpp:227-229`） | 特化后的 CUDA 内核 |
| 6. GPU 内核执行 | 每次 `dispatch` | `dispatch_impl`（`deep_ep/include/deep_ep/impls/dispatch.cuh:32`） | 接收缓冲区 + `EPHandle` |

读表结论：第 1、2 步只在 `ElasticBuffer` 构造时执行一次，产物被后续所有调用复用；第 3 至 6 步每次 `dispatch` 都重复，其中第 5 步有编译缓存，并非每次都真的重新编译。下表之后的调用链按层级展开，箭头表示调用关系而非时间上的连续执行，读图时重点看第 1、2 步的产物（`dev_comm`、`window`）如何在第 4、6 步被当作参数传入。

```
ElasticBuffer.dispatch()                     deep_ep/buffers/elastic.py:855
  → self.runtime.dispatch(...)              :976  （pybind 绑定见 csrc/elastic/buffer.hpp:1364）
    → ElasticBuffer::dispatch()             csrc/elastic/buffer.hpp:702
      → launch_dispatch(...)                :980 → csrc/kernels/elastic/dispatch.hpp:141
        → DispatchRuntime::generate(args)   :227  生成模板实例化源码
        → jit::compiler->build("dispatch")  :228  NVCC 编译生成 CUBIN 并加载
        → DispatchRuntime::launch(...)      :229  启动内核
          → dispatch_impl<...>()            deep_ep/include/deep_ep/impls/dispatch.cuh:32
            → handle::NCCLGin::put / put_value / get_sym_ptr
              → ncclGin::put（NCCL 设备侧 GIN API）
```

读图结论：Python 与 C++ 两层只做参数整理与启动准备，真正的通信动作全部发生在 GPU 内核里；Host 侧不再参与单次通信的数据搬运。下面按时机分节，先看初始化期。

### 2.1 初始化期：取得 NCCL communicator

上一节总览的第 1 步是取得 communicator，本节回答「DeepEP 手里为什么要有 `ncclComm_t`、它从哪里来」。这个问题从 1.1 节「V1 完全没有 `ncclComm_t`、V2 却重度依赖」的对比引出：`ncclComm_t` 是 NCCL communicator 的句柄类型，也是后续 `ncclDevCommCreate`、`ncclCommWindowRegister` 等底层 API 的必需参数，而不是用来发起 `ncclAllReduce` 之类集合操作的。读完应能回答：三条获取分支的触发条件、复用与自建在生命周期上的差别、以及哪些情况会强制自建。

V2 的获取逻辑集中在 `get_nccl_comm_handle`（`deep_ep/utils/comm.py:42-75`），有三条分支。下面这段代码要重点看 `:57`、`:62` 与 `:68-74` 三个判断点：`comm.py` 的参数 `group` 是 PyTorch 的 `ProcessGroup`（`torch.distributed` 中一组 rank 的通信组），`backend._comm_ptr()` 是较新 PyTorch 暴露的 NCCL communicator 裸指针 getter。

```python
# 中文注释为笔记所加（以「笔记：」开头）；以下为 deep_ep/utils/comm.py:55-75 原文，未省略
    # Check cache hit
    global _storage
    if not force_new_comm and group in _storage:
        # 笔记：分支 1，同一 ProcessGroup 已建过句柄时直接复用，_storage 定义于 comm.py:39
        return _storage[group]

    # New PyTorch has such API
    backend = group._get_backend(torch.device('cuda'))
    if not force_new_comm and hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
        # 笔记：分支 2，复用 PyTorch 已建好的 communicator；managed=False 表示不交由 DeepEP 销毁
        _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)
        return _storage[group]

    # For old PyTorch, we have to recreate a NCCL comm
    # 笔记：分支 3，各 rank 用 all_gather_object 汇集 rank 0 生成的 unique ID，再各自 ncclCommInitRank
    nccl_unique_ids = [None, ] * group.size()
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)
    root_unique_id = nccl_unique_ids[0]

    # Create a new communicator
    # 笔记：强制新建时改用时间戳作缓存键，避免覆盖已有的 group 缓存
    key = time.time_ns() if force_new_comm else group
    _storage[key] = NCCLCommHandle(
        _C.create_nccl_comm(root_unique_id, group.size(), group.rank()), True)
    return _storage[key]
```

读码结论：分支 2 是主路径，`managed=False` 表示 DeepEP 不负责销毁；分支 3 才调用 `csrc/kernels/backend/nccl.cu:20-41` 的 `get_local_unique_id`（内部 `ncclGetUniqueId`，`:22`）与 `create_nccl_comm`（内部 `ncclCommInitRank`，`:37`），`managed=True` 表示由 `NCCLCommHandle.__del__` 负责销毁。生命周期的差别写在 `comm.py:11-37`：`NCCLCommHandle.__init__`（`:21-24`）把 `self.destroy` 绑定到 `_C.destroy_nccl_comm`，`__del__`（`:26-28`）仅在 `managed` 为真时调用它，最终落到 `nccl.cu:43-47` 的 `ncclCommAbort`。另外 `EP_REUSE_NCCL_COMM` 默认值是 `'1'`（`:62`），即默认开启复用。

触发分支 3 有两种情况：PyTorch 没暴露 `_comm_ptr` 或 `EP_REUSE_NCCL_COMM=0`；以及 `ElasticBuffer.__init__` 传入 `force_new_comm=num_cpu_bytes > 0`（`deep_ep/buffers/elastic.py:301`），即启用了 CPU 弹性内存段时必须新建通信器，该参数的语义在 2.7 节回扣。

至此初始化期的第一项资源就绪；下一步要用这个句柄向 NCCL 申请设备侧资源，2.2 节接着说明。

### 2.2 初始化期：设备上下文、对称内存与窗口注册

上一节拿到 communicator，本节回答「有了句柄之后还要准备哪些资源、按什么顺序准备」。它从 2.1 节的结论引出：communicator 只是控制面句柄，GPU 内核还不能直接用它发 RDMA。读完应能回答：构造 `NCCLSymmetricMemoryContext` 的四步各自依赖什么、每步产出的字段被谁消费、以及 GIN 需求参数从哪里来。

拿到 communicator 后，`ElasticBuffer.__init__` 构造 C++ runtime（`deep_ep/buffers/elastic.py:346-354`），后者在 `csrc/elastic/buffer.hpp:107-114`（创建语句在 `:111-114`）创建 `NCCLSymmetricMemoryContext`。这个构造函数（`csrc/kernels/backend/nccl.cu:62-148`）是 DeepEP 与 NCCL 衔接最密集的一段，按真实执行顺序分四步。之所以必须分成四步，是因为存在一条时序依赖链：设备上下文要靠 communicator 查询能力后才能声明；通信域划分排在 `ncclDevCommCreate` 之后执行，但查询本身用的是 Host 侧 communicator（`ncclTeamLsa(comm)`，`nccl.cu:111`），并不依赖设备上下文；内存分配要按域的大小决定形态；窗口注册又必须等内存就绪并让全体 rank 同步，最后才能逐 peer 取 LSA 指针。下表给出这条依赖链的总览，读表时关注「依赖」一列，它解释了为什么不能把四步合并成一次调用。

| 步骤 | 依赖 | 主要产物 |
|------|------|---------|
| 1. 查询能力、声明 GIN 需求 | `comm` 句柄 | `reqs` → `dev_comm` |
| 2. 划分物理域与逻辑域 | 时序上排在 `dev_comm` 之后，查询用 `comm` | `num_scaleout_ranks` / `num_scaleup_ranks` / `is_scaleup_nvlink` |
| 3. 分配对称内存 | 逻辑域大小、CPU 段字节数 | `symmetric_memory->ptr` |
| 4. 注册窗口并取 LSA 指针 | 对称内存、全体 rank 同步 | `window` / `mapped_window_ptr` / `nvl_window_ptrs` |

读表结论：四步是单向依赖的构造过程，前一步的产物恰好是后一步的输入；因此它们只在 `ElasticBuffer` 构造时执行一次，2.3 节开始的每次 `dispatch` 直接消费第 1、4 步产出的句柄。

**第 1 步：查询能力并声明 GIN 需求。** 第 1 步先确认当前 NCCL 是否具备 GIN 能力，再把 DeepEP 需要的 QP、signal、连接类型等参数写进 `reqs`。下面这段代码要重点看 `:86` 的开关条件、`:92-98` 的六项需求字段，以及末尾 `:108` 的 `ncclDevCommCreate`。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 csrc/kernels/backend/nccl.cu:82-108 原文，未省略
    // Initialize NCCL device communicator
    // 笔记：两个 INITIALIZER 宏来自 NCCL core.h:161-166 与 :118-141，用于把结构体清零并填入 magic/version/默认值
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    // 笔记：仅多 rank 且未设 EP_DISABLE_GIN 时申请 GIN 资源；NCCL_CHECK 定义于 common/exception.cuh:84-94
    if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
        EP_HOST_ASSERT(
            // 笔记：hybrid 模式查 railedGinType（RAIL 连接能力），direct 模式查 ginType（FULL 连接能力）
            (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
            "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
            "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");

        reqs.ginContextCount = num_allocated_qps;
        reqs.ginExclusiveContexts = true;
        reqs.ginQueueDepth = kGinQPDepth;
        reqs.ginTrafficClass = sl_idx;
        // Customized RDMA barrier needs extra signals
        // 笔记：信号数按 num_ranks + 4 申请，其中前 num_ranks 个给 barrier，后 4 个给 PP，详见 2.5 节
        reqs.ginSignalCount = num_ranks + 2 * 2;
        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
    }
// 笔记：:100-106 是条件编译，按 NCCL 版本决定按运行时结构体大小还是按编译期大小分配 dev_comm
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true;
    dev_comm.ptr = malloc(props.devCommRuntimeVersionSize);
#else
    EP_HOST_ASSERT(NCCL_VERSION_CODE == nccl_runtime_version and "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. Please re-compile DeepEP.");
    dev_comm.ptr = malloc(sizeof(ncclDevComm_t));
#endif
    EP_HOST_ASSERT(dev_comm.ptr != nullptr);
    // 笔记：把 reqs 交给 NCCL 建立设备侧上下文；声明见 src/include/nccl_device/core.h:195-196
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm.ptr)));
```

读码结论：这一步不搬运任何数据，只把「需要多少个 GIN 上下文、多少 signal、多深的队列、什么连接类型」告诉 NCCL，由 `ncclDevCommCreate` 生成内核可用的 `dev_comm`。

其中几处写法需要展开。`NCCL_CHECK` 是 DeepEP 自己定义的宏（`deep_ep/include/deep_ep/common/exception.cuh:84-94`）：执行 NCCL 调用，返回值不是 `ncclSuccess` 时把错误码与 `ncclGetLastError` 文本包装成 `EPException` 抛出，不要与 NCCL 自带的 `NCCLCHECK`（`src/include/checks.h:148-156`）混淆。两个 `INITIALIZER` 宏（NCCL `src/include/nccl_device/core.h:118-141`、`:161-166`）除清零外还写入 `NCCL_API_MAGIC` 与版本号，NCCL 在 `src/dev_runtime.cc:1466-1509` 校验 magic，因此不能手写结构体初始化。`#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)` 是编译期条件：2.31 之前要求编译期与运行时 NCCL 版本严格一致，之后可按 `props.devCommRuntimeVersionSize` 分配，`reqs.useRuntimeVersion = true` 告诉 NCCL 采用运行时布局。`ginSignalCount` 的注释「Guaranteed to start at id=0」见 `core.h:99`，`ginContextCount` 的「This is a hint, the actual context count in the devcomm may not match」见 `core.h:98`，各字段的官方说明另见 `docs/userguide/source/api/device_setup.rst:94-132`。

关键参数的来源也需要交代，避免把它们当成凭空出现的常量：`num_allocated_qps` 是 `ElasticBuffer.__init__` 的参数（`deep_ep/buffers/elastic.py:244` 默认 0 表示自动），在 `:328-335` 按 `allow_hybrid_mode` 与网卡是否支持 fast RDMA atomic 取 65 / 129 / 17，再经 `csrc/elastic/buffer.hpp:87,114` 传入构造函数；`kGinQPDepth = 1024` 是编译期常量（`deep_ep/include/deep_ep/common/compiled.cuh:84`）；`sl_idx` 是 RDMA service level 索引（`elastic.py:243` 默认 3，`:322-324` 可被 `EP_OVERRIDE_RDMA_SL` 覆盖）；`allow_hybrid_mode`（`elastic.py:240` 默认 True）决定 `ginConnectionType` 取 `RAIL` 还是 `FULL`。这两条分支在 NCCL `src/gin/gin_host.cc:141-155` 落实为建链范围：`FULL` 时 `nGinRanks = comm->nRanks`（连接全体 rank，`:141-146`），否则取 `ncclTeamRail(comm)` 并用 `ncclTeamRankToWorld` 换算同 rail 的 rank（`:147-154`）。

**第 2 步：划分物理域与逻辑域。** 第 1 步只声明了资源需求，还没回答「哪些 rank 之间是 NVLink、哪些要走 RDMA」，因此第 2 步用 `ncclTeamLsa(comm)` 查询。`nccl.cu:110-125` 先取 `lsaTeam.nRanks`（`:111-113`）得到物理 NVLink 域大小，再按 `allow_hybrid_mode` 决定逻辑域：hybrid 模式下 `num_scaleout_ranks = num_rdma_ranks`、`num_scaleup_ranks = num_nvl_ranks`（`:118-120`），direct 模式下把全体 rank 视作一个 scaleup 域、`num_scaleout_ranks = 1`（`:121-124`）；`:125` 的 `is_scaleup_nvlink` 标记 scaleup 域是否恰好等于 NVLink 域。这一划分直接决定 2.3 节选择 `dispatch.cuh` 还是 `hybrid_dispatch.cuh`。

**第 3 步：分配对称内存。** 域划分完成后，内存形态才能确定：CPU 段是否存在、hybrid 模式下要映射几个 rank 的 CPU 段，都依赖第 2 步的 `num_scaleup_ranks`。`nccl.cu:127-132` 调用 `symmetric::alloc`（`csrc/kernels/backend/symmetric.hpp:291-317`），按 CPU 段字节数选择三种实现之一（详见 2.7）。最基础的实现直接包一层 NCCL 的 `ncclMemAlloc`，下面这段代码重点看两处对齐断言与 `:128` 的分配调用。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 csrc/kernels/backend/symmetric.hpp:123-132 节选，省略 :134-140 的析构函数
// Wraps ncclMemAlloc/ncclMemFree (pure GPU, current default)
class GPUSymmetricMemory final : public SymmetricMemory {
public:
    explicit GPUSymmetricMemory(const int64_t& num_bytes) {
        // 笔记：kNumAlignmentBytes = 2097152（2 MB），定义于 symmetric.hpp:16
        EP_HOST_ASSERT(num_bytes > 0 and num_bytes % kNumAlignmentBytes == 0);
        // 笔记：ncclMemAlloc 声明于 src/nccl.h.in:156，由 NCCL 分配并保证对称语义
        NCCL_CHECK(ncclMemAlloc(&ptr, num_bytes));
        // 笔记：分配后再校验返回地址本身也满足 2 MB 对齐
        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAlignmentBytes == 0);
        this->num_bytes = num_bytes;
        this->num_gpu_bytes = num_bytes;
    }
```

读码结论：`2097152 = 2 × 1024 × 1024` 字节，`num_bytes % kNumAlignmentBytes == 0` 要求请求大小是它的整数倍；这是后续窗口注册与各 rank 统一地址换算的前提，而不只是分配器的偏好。

**第 4 步：注册窗口并取 LSA 指针。** 内存就绪后还差最后一步：让 NCCL 把这段内存纳入所有 rank 共享的窗口，并取回本进程可直接使用的地址。下面这段代码重点看 `:140` 的注册调用与 `:141,145-147` 两次 `ncclGetLsaDevicePointer` 的差别。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 csrc/kernels/backend/nccl.cu:134-147 原文，未省略
    // Create window
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    // 笔记：注册是集合操作，内部已做跨 rank 同步，因此调用后无需再写显式 barrier
    raw_window_ptr = this->symmetric_memory->ptr;
    this->num_gpu_bytes = this->symmetric_memory->num_gpu_bytes;
    this->num_cpu_bytes = this->symmetric_memory->num_cpu_bytes;
    // 笔记：NCCL_WIN_STRICT_ORDERING 是 #define 宏（src/nccl.h.in:66），值为 0x02，要求严格顺序
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, this->symmetric_memory->num_bytes, &window, NCCL_WIN_STRICT_ORDERING));
    // 笔记：先取本 rank 在 LSA 域内该窗口的映射地址，作为后续偏移计算的基准
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));

    // Get LSA pointers for all LSA peers
    // TODO: check whether this is correct for network with RDMA
    // 笔记：再为每个 LSA peer 各取一次，循环变量 i 是 NVLink 域内 rank 编号
    nvl_window_ptrs.resize(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++ i)
        NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
```

读码结论：`mapped_window_ptr` 用于把窗口内任一指针换算成相对偏移（`nccl.cu:150-153` 的 `get_sym_ptr`），`nvl_window_ptrs[i]` 则是逐个 peer 的窗口基址表，两者都与 `nvl_rank_idx` 一样来自 `ncclTeamLsa`（`:111-112`）。注意窗口注册标志 `NCCL_WIN_STRICT_ORDERING` 不是枚举，而是 `src/nccl.h.in:63-66` 里的 `#define`，同组还有 `NCCL_WIN_DEFAULT 0x00` 与 `NCCL_WIN_COLL_SYMMETRIC 0x01`。

两个 NCCL API 的含义需要分清，下表回答「注册与取指针各自做什么」，读表时关注「作用」一列；结论是注册建立全局视图、取指针把视图变成内核可用的地址。

| API | 声明位置 | 作用 |
|-----|---------|------|
| `ncclCommWindowRegister` | `src/nccl.h.in:350-351` | 把对称内存注册为 window；内部跨所有 rank 做 bootstrapBarrier，是集合操作 |
| `ncclGetLsaDevicePointer` | `src/include/nccl_device/core.h:202-203` | 返回本进程 VA 空间中指向 LSA 域内第 i 个 GPU 该窗口的指针 |

读表结论：注册是一次集合操作，负责建立全局视图；取指针是逐 peer 的本地查询，负责把视图变成可参与地址算术的具体地址，两者缺一不可。

LSA 的定义见 `docs/userguide/source/usage/deviceapi.rst:13`（Load/Store Accessible），GIN 设计文档 `:94` 写作 Load/Store Access，指同节点内对其他 GPU 内存的直接 load/store 访问能力。窗口注册后由 NCCL 建立统一的对称 VA 视图（`docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md:306-309`），把各 peer 的物理段映射进本进程的 VA（同文档 `:232`、`:243`、`:265-266`），设备侧访问 peer 的窗口因此只需 `lsaFlatBase + peer*stride + offset` 这样的地址算术（同文档 `:289-290`），无需 CPU 参与。`nvl_window_ptrs[i]` 就是这张地址表在 Host 侧的缓存。

到这四步结束，初始化期完成：设备侧拿到了 `dev_comm`、`window`、以及机内各 peer 的窗口基址。后面每次 `dispatch` 都复用这些资源，2.3 节接着进入调用期。

### 2.3 调用期：从 `dispatch()` 到内核启动

上一节把初始化期资源准备完毕，本节回答「每次调用时 Host 侧做了什么、内核是怎么被编译出来的」。它从 2.2 节的结论引出：控制面只需一次，调用期只做参数整理与启动。读完应能回答：Python 层整理哪些参数、C++ 启动器如何填 `Args`、JIT 三步各自由哪个文件实现、以及运行期才知道的信息如何变成编译期常量。

现在进入每次调用的路径。Python 侧 `ElasticBuffer.dispatch`（`deep_ep/buffers/elastic.py:855-1033`）先做参数整理：自动推断 SM 与 QP 数（`:926-930`，未指定时调用 `get_theoretical_num_sms` / `get_theoretical_num_qps`，并断言 `num_qps <= self.num_allocated_qps`）、解包 FP8 的 `(x, sf)`（`:932-933`，FP8 是 8 位浮点格式，输入为「数据张量 + SF 缩放因子张量」的二元组）、复用已有 `handle` 中的缓存布局（`:937-956`），然后把一长串参数交给 C++ runtime（`:976`）。下面这段代码重点看 `:964-976` 的返回值解包与 `:976` 的调用入口。

```python
# 中文注释为笔记所加（以「笔记：」开头）；以下为 deep_ep/buffers/elastic.py:963-976 节选，省略 :977-996 的 cached 参数与调用期选项
        # Do dispatch
        # 笔记：左侧元组是 C++ runtime 一次返回的全部产物，前四个是接收张量，其余是布局与元数据
        (recv_x, recv_sf,
         recv_topk_idx, recv_topk_weights,
         cloned_topk_idx,
         num_recv_tokens, num_expanded_tokens,
         num_recv_tokens_per_expert_list,
         psum_num_recv_tokens_per_scaleup_rank,
         psum_num_recv_tokens_per_expert,
         num_unaligned_recv_tokens_per_expert,
         recv_src_metadata,
         dst_buffer_slot_idx,
         token_metadata_at_forward,
         channel_linked_list,
         event) = self.runtime.dispatch(x, sf, topk_idx, topk_weights,
                                        cumulative_local_expert_recv_stats,
```

读码结论：Python 层不改变算法，只是把「本步的张量、上一步的缓存布局、SM/QP 数、事件与同步选项」整理成一次调用；`:977-987` 继续传入 10 个 `cached_*` 参数，`:988-996` 传入 `num_max_tokens_per_rank`、`num_experts`、`num_sms`、`num_qps` 与 `do_cpu_sync` 等，这些字段的来源见 2.6 节的 `EPHandle`。

pybind 把 `runtime.dispatch` 绑定到 C++ 的 `ElasticBuffer::dispatch`（`csrc/elastic/buffer.hpp:1364`）。后者（`buffer.hpp:702` 起）负责校验缓冲区大小（`:962-966`）、清理 Host workspace（`:968-976`，把接收计数清零并以 `std::atomic_thread_fence` 发布），再把 `nccl_context->dev_comm`、`window`、各类布局指针一起传给 `launch_dispatch`（`:980-1003`）。注意这里传的是 2.2 节初始化好的句柄，说明单次调用不再触碰 NCCL 的控制面。

`launch_dispatch`（`csrc/kernels/elastic/dispatch.hpp:141-230`）是 Host 侧启动器，做三件事：按共享内存容量算出 warp 分工（`:170-196`）、填充 `DispatchRuntime::Args`（`:199-226`）、执行 JIT 三步。下面这三行是全章的关键衔接，重点看 `generate` / `build` / `launch` 各自接收什么。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 csrc/kernels/elastic/dispatch.hpp:227-229 原文，未省略
    // 笔记：generate 由基类 LaunchRuntime::generate 提供（csrc/jit/launch_runtime.hpp:33-46），内部调用派生类 generate_impl
    const auto code = DispatchRuntime::generate(args);
    // 笔记：build 查缓存、必要时调用 NVCC 生成 CUBIN 并加载（csrc/jit/compiler.hpp:111-160）
    const auto runtime = jit::compiler->build("dispatch", code);
    // 笔记：launch 由 LaunchRuntime::launch 构造启动配置后调用派生类 launch_impl（launch_runtime.hpp:49-71）
    DispatchRuntime::launch(runtime, args, stream);
```

读码结论：运行期才知道的参数先被拼成源码，再交给编译器生成可加载内核，最后才启动；这三步分别落在 `csrc/jit/launch_runtime.hpp`、`csrc/jit/compiler.hpp` 与 `dispatch.hpp:91-127`。

V2 的内核是**运行期 JIT 编译**的（`README.md:13`），包级初始化 `deep_ep/__init__.py:71-84` 的 `init_jit` 会把 CUDA/NCCL 根路径传给 C++ 编译器对象。`generate` 并非凭空生成算法，而是把运行期才知道的模板参数拼成一段仅用于实例化的源码（`dispatch.hpp:51-89` 的 `generate_impl`）。下面这段代码要重点看单花括号占位符、双花括号转义与 `reinterpret_cast<void*>(&{})` 这一句。

{% raw %}
```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 csrc/kernels/elastic/dispatch.hpp:80-88 原文，未省略
return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
```
{% endraw %}

读码结论：这段源码里没有任何算法语句，唯一的作用是让 NVCC 实例化 `func_name` 指定的那个模板特化，并把它发射进 CUBIN。

其中几处写法容易误读，逐一展开。`{}` 是 `fmt::format` 的占位符，按实参顺序替换为 `header_name` 与 `func_name`；源码里连续两个左花括号与连续两个右花括号是 fmt 的转义，输出单个字面量花括号，若少写一层会被当成格式字段而报错。`func_name` 由 `dispatch.hpp:55-65` 用 15 个 `{}` 拼出 `dispatch_impl<...>`，或由 `:68-77` 用 16 个 `{}` 拼出 `hybrid_dispatch_impl<...>`，实参依次是 `is_scaleup_nvlink`、`do_cpu_sync`、`reuse_slot_indices`、`num_sms`、warp 数、域大小、token 布局、`num_experts`、`num_topk`、`expert_alignment`、`num_qps`、`num_timeout_cycles`。`header_name` 按 `args.num_scaleout_ranks == 1` 选择：单机（direct 模式）选 `dispatch`（`:53-54`），跨机（hybrid 模式）选 `hybrid_dispatch`（`:67`）。`reinterpret_cast<void*>(&func_name)` 对 `__global__` 函数模板的特化取地址，迫使编译器实例化并保留该 kernel 符号；这是必需的，因为 `csrc/jit/kernel_runtime.hpp:31-55` 用 `cuobjdump -symbols` 从 CUBIN 中找出唯一的 kernel 符号，而仅仅 `#include` 模板定义不会生成任何 kernel。`num_timeout_cycles` 的来源链也在此闭合：`elastic.py:245` 的 `num_gpu_timeout_secs=100` 经 `buffer.hpp:121-123` 乘以 GPU 时钟率变成周期数，再经 `dispatch.hpp:210` 填入 `Args`、`:65` 拼成模板实参。

`generate` 与 `build` 的真实实现需要与前文的旧表述区分开，这里给出核实后的链路。`generate` 是基类 `jit::LaunchRuntime::generate`（`csrc/jit/launch_runtime.hpp:33-46`），先调用派生类 `DispatchRuntime::generate_impl`（`:34`），再把 include 部分的哈希写到源码首行（`:37-42`）。`build`（`csrc/jit/compiler.hpp:111-160`）按 `name$$signature$$flags$$code` 计算缓存键（`:112-113`），命中 `kernel_runtime_cache` 就直接返回（`:115-117`）；未命中时把源码写入临时目录，调用 `NVCCCompiler::compile`（`:222-262`）执行 `nvcc ... -cubin -o ...`（`:232-233`），随后 `fsync_dir`（`:141`）并以 `std::filesystem::rename` 原子重命名整个缓存目录（`:143-154`），最后在 `KernelRuntime` 构造时加载 CUBIN（`csrc/jit/handle.hpp:36` 的 `cudaLibraryLoadFromFile`，或 Driver API 路径 `:97` 的 `lazy_cuModuleLoad`）。默认编译器对象就是 `NVCCCompiler`（`compiler.hpp:265-267`），可用 `EP_JIT_NVCC_COMPILER` 覆盖（`README.md:347`）。

> 辨析：初稿曾把 `build` 写成「NVRTC 编译」。回源码核对后修正：`compiler.hpp:6` 虽包含 `<nvrtc.h>`，但全库没有任何 `nvrtc*` API 调用，实际路径是 NVCC 编译出 CUBIN 再加载；`<nvrtc.h>` 的存在不构成运行时编译器的证据。同理，CUBIN 的加载发生在 `build` 末尾，而不是 `launch` 阶段。

这样既避免了安装期编译，又让内核里的循环边界、warp 数量都是编译期常量。配合 `if constexpr` 使用效果更明显：`dispatch.cuh:338` 的 `if constexpr (kReuseSlotIndices)` 与 `:382` 的 `if constexpr (not kIsScaleupNVLink)` 都在编译期裁掉不成立的分支，运行期不留判断指令；`reuse_slot_indices` 来自 `dispatch.hpp:176` 的 `cached_mode`，`is_scaleup_nvlink` 来自 `nccl.cu:125`，两者都在 2.4、2.6 节出现。至此调用期的 Host 侧工作结束，下一节进入内核内部。

### 2.4 内核期：`dispatch_impl` 的三个阶段

上一节把模板特化编译并启动，本节回答「内核拿到逐 token 的路由后怎样算出接收布局并搬运数据」。它从 2.3 节的结论引出：既然 `topk_idx` 只在运行期可知、布局又必须精确，内核就必须自己计算，而不能依赖 Host 预先排好。读完应能回答：三个阶段各自解决什么、哪些 warp 参与、产物被谁使用、以及机内与跨机两条数据路径如何分流。

`dispatch_impl`（`deep_ep/include/deep_ep/impls/dispatch.cuh:17-45`，模板参数在 `:17-30`、`__global__` 函数声明在 `:31-45`、函数体至 `:409`）的模板参数就是上一节 JIT 拼进去的那些，内核签名接收 `x`、`topk_idx`、workspace、缓冲区以及 `ncclDevComm_t` / `ncclWindow_t`。内核内部按 warp 角色分三段，这样分段的动机是一条数据依赖：不知道各 rank 收多少 token，就无法确定接收区的分段；没有分段偏移，就无法给 token 分配目标 slot；slot 未定，数据搬运就不能开始；搬运完成还需要一个全体 rank 的到达保证。下表给出各段的职责与产物，读表时关注「产物」一列，它串起了后面的 `EPHandle` 与 combine。

| 阶段 | 参与的 warp | 职责 | 产物 |
|------|------------|------|------|
| 1. 计数与通知 | `warp_idx < kNumNotifyWarps` | 统计每个 token 的目标 rank/expert，全 SM 归约，把计数发给各 rank | `rank_count` / `expert_count` |
| 2. 前缀和 | SM 0 的 warp 0/1 | 由计数算接收布局的起始偏移 | `psum_num_recv_tokens_per_*` |
| 3. 数据搬运 | 其余 dispatch warps | TMA 装入共享内存、分配目标 slot、按路径发出数据 | 接收缓冲区中的 token |
| 收尾 | 全体 | `gpu_barrier` 确认数据到达 | 可安全读取的接收区 |

读表结论：第 1 阶段产生「量」，第 2 阶段把「量」变成「位置」，第 3 阶段按「位置」搬运，收尾阶段保证搬运已完成；四者缺一不可，且顺序不能调换。另外只有 SM 0 的 notify warps 继续执行等待、对齐、前缀和与写 Host workspace，其他 SM 的 notify warps 在 `:79-115` 的规约后即结束，`kNumNotifyWarps = 4` 定义在 `dispatch.hpp:130`。

**第 1 阶段：运行时动态计数**（`dispatch.cuh:79-115`）。notify warps 逐 token 读 `topk_idx`，用共享内存原子加统计「每个 expert 收到几个 token」和「每个 rank 收到几个 token」，后者对同一 token 的多个专家做去重。下面这段代码要重点看 `:98-99` 的 lane 分工、`:101` 的专家计数与 `:105-106` 的去重计数。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 deep_ep/include/deep_ep/impls/dispatch.cuh:98-106 原文，未省略
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            // 笔记：一个 warp 有 32 个 lane，kNumTopk <= 32（:93 断言），每个 lane 负责该 token 的一个 top-k 专家
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            // 笔记：atomicAdd_block 是 block 作用域原子加，这里作用于共享内存中的 expert_count
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            // 笔记：kNumExpertsPerRank = kNumExperts / kNumRanks（:46），把专家编号换算成目标 rank
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
```

读码结论：每个 top-k 选择都给对应 expert 计数加 1，但一个 token 的多个专家可能落在同一 rank，因此 rank 计数必须先去重再加，否则接收端会多算一个 slot。

这里两个 CUDA 内建与一个库函数需要展开。`__ldg` 是只读数据缓存加载（等价 PTX `ld.global.nc`，库内封装见 `deep_ep/include/deep_ep/common/ptx.cuh:244-246`），适用于 `topk_idx` 这种内核期间不会被修改的全局输入；`atomicAdd_block` 是 block 作用域的原子加，本工程中作用于动态共享内存（`dispatch.cuh:59` 的 `extern __shared__`、`:86` 把 `rank_count` / `expert_count` 指向该区域），不涉及全局内存。`ptx::deduplicate` 的实现是 `ptx.cuh:419-421`：先用 `__match_any_sync(0xffffffff, value)`（`:394-396`）得到「与本 lane 取值相同的所有 lane」的 32 位掩码，再用 `bfind.u32`（`:412-417`，等价 `31 - __clz(mask)`）取掩码中最高置位的 lane 编号，只有该 lane 返回 `true`。代入一个具体例子：若 lane 0..3 的 `dst_rank_idx` 依次是 `1, 1, 3, -1`，则 `match(1) = 0b0011`、最高置位 lane 为 1，`match(3) = 0b0100`、最高置位 lane 为 2，其余 lane 对 `-1` 的掩码最高位为 31；因此只有 lane 1 与 lane 2 通过，配合 `dst_rank_idx >= 0` 过滤，rank 1 与 rank 3 各被计一次。

计数随后要发给各对端，这里按数据形态分成两类：`dispatch.cuh:151-158` 用 `gin.put_value<team_t>` 把 rank 计数逐个写到各 peer 的 workspace（`:157` 另传 `ncclGinOptFlagsAggregateRequests`），`:161-170` 在 NVLink 可达时同样用 `put_value` 逐元素写 expert 计数，而 `:171-177` 的 RDMA 分支改用 `gin.put<team_t>` 把整段计数值批量拷到对端（`:176`）——两段合起来才是 `:151-178` 的完整通知过程，不能笼统说成「全部用 put_value」。这就是**用 GIN 完成控制信息的交换**。SM 0 还负责等待对端计数、对齐 expert 数量、写 Host workspace（`:183-230`）：`:215` 的 `expert_count[i] = math::align(sum, kExpertAlignment)` 把每个专家的 token 数对齐到 `kExpertAlignment` 的倍数，`:224` 的 `if constexpr (kDoCPUSync)` 决定要不要把计数写进 mapped host workspace 供 CPU 读取。

**第 2 阶段：前缀和定位**（`dispatch.cuh:232-257`）。第 1 阶段得到的是「各段有多少个 token」，本阶段把它换算成「各段从哪里开始」。`do_psum`（`:234-250`）用 `ptx::warp_inclusive_sum` 做 warp 内前缀和，warp 0 对 rank 计数做包含式前缀和写入 `psum_num_recv_tokens_per_scaleup_rank`（`:251-253`），warp 1 对 expert 计数做排他式前缀和写入 `psum_num_recv_tokens_per_expert`（`:254-256`）。这两个数组的语义写在 `EPHandle` 的 docstring（`deep_ep/buffers/elastic.py:38-48`）：前者最后一项等于接收 token 总数，且同一 token 的多个专家落在同一 rank 时只计一次；后者按 `expert_alignment` 填充。它们是后面 copy epilogue 内核与用户侧 `EPHandle` 判断「某 rank 的 token 放在接收缓冲区哪一段」的依据，也是不规则布局得以精确定位的关键。

**第 3 阶段：逐 token 搬运与路径分流**（`dispatch.cuh:259-395`）。偏移确定后，dispatch warps 才能开始搬运。每个 warp 先按 `comm::get_qp_mode` 取自己的 QP 与共享模式，构造 `NCCLGin`（`:67-71`，这层封装在 2.5 节展开）；随后循环处理 token：用 TMA 把 hidden 与 SF 搬入共享内存（`:283-312`，`tma_load_1d` 封装在 `ptx.cuh:124-136`，用 mbarrier 的完成字节数通知；SF 部分走 `cp_async`）、读 top-k 索引（`:314-326`）、用原子计数分配目标 slot（`:336-351`）。`atomicAdd` 作用于 workspace 中的发送计数器（`:345`），`if constexpr (kReuseSlotIndices)`（`:338-342`）则在传入缓存 handle 时直接读取上次的 `dst_buffer_slot_idx`，跳过重新计数。

搬运的最后按目标 rank 是否在 NVLink 域内分两条路。下面这段代码要重点看 `:373-375` 的 `get_sym_ptr` 判定、`:376-378` 的 TMA 直写与 `:381-393` 的 RDMA 分支。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 deep_ep/include/deep_ep/impls/dispatch.cuh:371-393 原文，未省略
            // Issue TMA NVLink stores
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");
            // 笔记：get_sym_ptr 返回非空表示目标 rank 可用对称指针直达，否则走 RDMA
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?
                gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                nullptr;
            if (dst_ptr != nullptr)
                // 笔记：tma_store_1d 对应 cp.async.bulk.global.shared::cta（ptx.cuh:138-148），异步写对端窗口
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            // 笔记：commit 把已发出的 store 封成一个 bulk_group，供后续 wait 追踪（ptx.cuh:150-152）
            ptx::tma_store_commit();
            __syncwarp();

            // Issue RDMA put
            // 笔记：if constexpr 是编译期分支，纯 NVLink 模式下这一段被整体裁掉
            if constexpr (not kIsScaleupNVLink) {
                // Wait the send buffer store to arrive
                // 笔记：<1> 表示等到只剩 1 个未完成 group（ptx.cuh:114-117），即先让 send buffer 那批 store 落定
                ptx::tma_store_wait<1>();
                __syncwarp();

                // NOTES: we should skip the NVLink accessible ranks
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
                    // 笔记：数据先由 :362-369 落到本地 send buffer，再由 GIN put 经 RDMA 写到对端窗口
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                __syncwarp();
            }
```

读码结论：`get_sym_ptr` 返回非空表示目标在同 NVLink 域内，直接用 TMA store 写对端窗口；返回空则把数据先落到本地 send buffer（`:362-369`），再由 `gin.put` 经网卡 RDMA 写过去。两条路径互斥，`dst_ptr == nullptr` 就是 RDMA 分支的判定条件。这正是 1.3 节所说「V2 底层仍按拓扑分流」的落点，其判定依据来自 `NCCLGin` 封装，2.5 节展开。

最后 `comm::gpu_barrier`（`dispatch.cuh:397-400`，用 `comm::kDispatchTag1`）确认所有 rank 的数据到达；`:403` 的 `cudaTriggerProgrammaticLaunchCompletion()` 用于 Programmatic Dependent Launch，通知 CUDA 后续的 copy epilogue 内核可以提前启动（该内核的 JIT 运行时见 `csrc/kernels/elastic/dispatch.hpp:232-340`），`:405-408` 清理本轮的原子计数器。至此内核产出接收缓冲区与布局信息，下一节展开它依赖的 `NCCLGin` 封装。

### 2.5 设备侧 GIN 封装：Team 判定与两条数据路径

上一节出现了 `NCCLGin::put` / `get_sym_ptr` / `put_value`，也留下了「分流判定依据」这个未闭合点；1.3 节末同样预告过 `is_nvlink_accessible` 的作用，本节一并回答：DeepEP 如何把 NCCL 设备侧 GIN API 包装成「按目标 rank 自动选路」的接口。读完应能回答：三个 Team 各自的 rank 集合、对称指针何时可用、各封装方法映射到哪个 NCCL API、以及 QP 与 warp 的对应关系。代码在 `deep_ep/include/deep_ep/common/handle.cuh`。

先交代术语。Team 是 communicator 内的 rank 子集（`src/include/nccl_device/core.h:45-47` 的 `ncclTeam { nRanks, rank, stride; }`），NCCL 预定义 World（全体 rank）、Lsa（同 NVLink 域）、Rail（跨节点中处于相同 rail 位置的 rank）三个，`core.h:55-57` 定义对应的 tag 类型 `ncclTeamTagWorld` / `ncclTeamTagLsa` / `ncclTeamTagRail`；rail 的定义见 `docs/userguide/source/usage/deviceapi.rst:221-222`，即各节点内编号相同的 GPU 组成的集合。构造时（`handle.cuh:11-35`）保存 `ncclDevComm_t` 与 `ncclWindow_t`，用 `qp_idx` 构造 NCCL 提供的 `ncclGin` 对象，并一次性取好三个 Team 与本 rank 的 LSA 窗口基址。下面这段代码重点看初始化列表里三个 Team 的取得方式与 `:34` 的 `lsa_base_ptr`。

```cpp
// 中文注释为笔记所加（以「笔记：」开头）；以下为 deep_ep/include/deep_ep/common/handle.cuh:26-35 原文，未省略
    __device__ __forceinline__
    // 笔记：qp_idx 来自 comm::get_qp_mode；resource_sharing_mode 默认 NCCL_GIN_RESOURCE_SHARING_GPU（GPU 级共享）
    NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
            const int& qp_idx = 0,
            const ncclGinResourceSharingMode& resource_sharing_mode = NCCL_GIN_RESOURCE_SHARING_GPU):
        nccl_dev_comm(nccl_dev_comm), nccl_window(nccl_window),
        // 笔记：ncclGin 是 NCCL 设备侧 GIN 句柄（src/include/nccl_device/gin.h:120 的类型别名）
        gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),
        team_world(ncclTeamWorld(nccl_dev_comm)), team_lsa(ncclTeamLsa(nccl_dev_comm)), team_rail(ncclTeamRail(nccl_dev_comm)) {
        // TODO: what if we only have 1 NVLink rank
        // 笔记：LSA 窗口基址，配合 get_sym_offset（:59-61）把本地指针换算成窗口内相对偏移
        lsa_base_ptr = reinterpret_cast<uint64_t>(ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));
    }
```

读码结论：构造一次就把设备上下文、三个 Team 与本 rank 窗口基址固定下来，后续每个方法都靠这几项做地址换算与选路，不需要再向 Host 查询。

`is_nvlink_accessible`（`handle.cuh:37-54`）依据 Team 类型判断目标 rank 是否能用对称指针直达。这里用到了 `handle.cuh:12-17` 定义的一组宏，例如 `IS_TEAM_LSA(code)` 展开为 `if constexpr (std::is_same_v<team_t, ncclTeamTagLsa>) { code }`，是编译期按模板 tag 选择分支，运行期不产生判断；`TEAM_WORLD_RAIL()`（`:17`）则按 tag 在 `team_world` 与 `team_rail` 之间选一个作为 GIN 调用的目标 Team。三种判定分别是：Lsa 恒为可达（`:39-41`）；World 下要求 `team_rail.rank * team_lsa.nRanks <= dst_rank_idx < (team_rail.rank + 1) * team_lsa.nRanks`（`:46-47`），例如 2 个节点各 8 个 GPU 时，节点 1 的 rank 编号是 8..15，只有落在这个区间的目标才与本地同 NVLink 域；Rail 下只有 `team_rail.rank == dst_rank_idx` 可达（`:50-53`）。`get_sym_ptr`（`:64-92`）在可达时用 `ncclGetLsaPointer(nccl_window, get_sym_offset(ptr), dst_nvl_rank_idx)`（`:88-89`）把本地窗口偏移换算成该 peer 的窗口地址，本地 rank 则直接返回原指针（`:84-85`），不可达时返回 `nullptr`。2.4 节里 `dst_ptr != nullptr` 的分支正是以此为依据。

其余封装方法把不同形态的远程操作映射到 NCCL 设备侧 API。下表回答「每个方法最终落到哪个 NCCL 调用」，读表时关注「映射」一列；结论是所有方法都遵循同一原则——对称指针可达时直接访存，不可达时才动用 GIN。

| 方法 | 位置 | 映射到的 NCCL API |
|------|------|------------------|
| `put` | `handle.cuh:175-198` | `gin.put`，参数用窗口 + 相对 `lsa_base_ptr` 的偏移（`:187-188`） |
| `get` | `handle.cuh:127-144` | `gin.get`，同样是窗口 + 偏移（`:135-136`） |
| `put_value` | `handle.cuh:200-220` | 可达时 `ptx::st_relaxed_sys` 直写（`:206`），否则 `gin.putValue`（`:209-218`） |
| `wait` / `flush` / `flush_async` / `signal` | `handle.cuh:122-173` | `gin.wait` / `gin.flush` / `gin.flushAsync` / `gin.signal` |

读表结论：这些方法覆盖数据面（put/get）、控制面（put_value）与同步面（wait/flush/signal），映射关系说明 `NCCLGin` 本身不实现网络协议，只负责参数换算与路径选择。

其中两处写法需要展开。`ptx::st_relaxed_sys`（`ptx.cuh:314-324`）是 system 作用域的 relaxed 存储（PTX `st.relaxed.sys.global.u32/u64`），只保证写入的原子性与对系统内其他设备的可见性，不提供 acquire/release 排序，适合发布单个计数值。`put` 中的 `gin.put(TEAM_WORLD_RAIL(), dst_rank_idx, nccl_window, reinterpret_cast<int64_t>(recv_sym_ptr) - lsa_base_ptr, ...)`（`:183-188`）说明 NCCL 设备 API 接收的是「窗口句柄 + 窗口内偏移」而非裸虚拟地址，所以 `lsa_base_ptr` 必须是每个 rank 自己的 LSA 基址。`red_add_rel`（`handle.cuh:96-120`）体现了「对称指针优先、RDMA 回退」这一原则：目标可达时用 `red_add_rel_gpu` 或 `red_add_rel_sys` 做原子加（`:103-107`），不可达时改用 `gin.signal` 配 `ncclGin_VASignalAdd` 发一个带值的信号（`:112-114`）。控制信息与数据信息都遵循同一套分流。

> 辨析：`ncclGin_VASignalAdd` 在 NCCL 中已标注弃用（`src/include/nccl_device/gin.h:51-56` 的注释，`docs/userguide/source/api/device_gin.rst:168-170` 标注 `deprecated:: 2.30.5`），建议改用 `ncclGin_StrongVASignalAdd` / `ncclGin_WeakVASignalAdd`。DeepEP v2.1.0 的 `handle.cuh:113` 仍使用旧名，这是当前源码的实际状态，不是推荐写法。

QP 的分配由 `comm::get_qp_mode`（`deep_ep/include/deep_ep/common/comm.cuh:56-86`）决定，它按 SM 数、QP 数、warp 角色返回 `(qp_idx, resource_sharing_mode)`。需要澄清一处常见误解：warp 与 QP **不是一一对应**的。完整分支是：只有 1 条 QP 时全部共享 QP 0（`:63-64`）；notify warp 固定用 QP 0 且 CTA 级共享（`:67-68`）；数据通道先把 QP 0 让给 notify warp，`kQPStartIdx = kWithNotifyWarps`、`kNumAvailableQPs = kNumQPs - kQPStartIdx`（`:71-72`），当 `kNumSMs > kNumAvailableQPs` 时所有 SM 共享全部可用 QP，QP 下标为 `kQPStartIdx + (global_channel_idx % kNumAvailableQPs)`（`:81-85`），并采用 `kSharingGrid`（GPU 级共享，`:59-60`）；只有 `kNumSMs <= kNumAvailableQPs` 时单个 SM 才能独占若干 QP（`:73-80`）。换句话说，共享分支的条件与取模对象都是「可用 QP 数」，不是 `kNumQPs` 本身。

内核的跨 rank 同步也不再依赖 CPU：`comm::gpu_barrier`（`comm.cuh:214-270`）按 scaleup/scaleout 域分别做 barrier，其中 GIN 路径（`comm.cuh:143-187`）先 `flush` 刷完所有 QP（`:143-147`）、World Team 下补一次 `ptx::fence_acq_rel_sys()`（`:149-152`），再由 SM 0 用 `gin.signal` 递增各 rank 的信号（`:158-164`，`ncclGin_SignalInc`），最后以 `ld_acquire_sys` 轮询信号值直到到达目标（`:167-185`）。等待期间的超时检查由 `comm::timeout_while` 完成，其实现与参数来源在 3.4 节展开。`gpu_barrier` 只使用下标 `0..kNumRanks-1` 的 `num_ranks` 个信号（`comm.cuh:163-168`），这一点需要用来订正 2.2 节留下的一个问题：

> 辨析：`nccl.cu:96-97` 的 `ginSignalCount = num_ranks + 2 * 2` 中，额外的 4 个信号并非给 `gpu_barrier` 使用。`comm.cuh:163-168` 显示 barrier 只占用前 `num_ranks` 个；索引 `num_ranks..num_ranks+3` 由 PP（pipeline parallelism）的 send/recv 内核使用，见 `deep_ep/include/deep_ep/impls/pp_send_recv.cuh:145`、`:162`、`:196`、`:208`，其中 `:145` 与 `:208` 用 `kNumRanks + 2`。源码注释 `// Customized RDMA barrier needs extra signals`（`nccl.cu:96`）与 `comm.cuh` 中 barrier 实际只占用前 `num_ranks` 个信号这一事实不一致；按代码可核对的用途，多出的 4 个由 PP 使用，不能按注释字面理解成 barrier 需要多留 4 个。

至此 `NCCLGin` 的选路、QP 分配与同步都解释完毕；下一节看这些布局信息如何跨 dispatch 与 combine 两次调用传递。

### 2.6 combine 与 `EPHandle`：布局信息如何跨两次调用传递

上一节闭合了内核的选路问题，本节回答「dispatch 算出的布局怎样被 combine 复用」。这个问题从 2.4 节的产物引出：前缀和、slot 索引、源元数据都是内核算出来的，若 combine 再算一遍既浪费又可能与前次不一致。读完应能回答：`EPHandle` 保存哪些字段、各字段在 combine 中起什么作用、缓存 handle 如何跳过重复计算。

dispatch 把 token 送到目标 rank，combine 要把专家算完的结果按原路由送回并归约。两者之间靠 `EPHandle` 传递布局信息，这是 V2 接口设计里值得单独看的一点。

`EPHandle`（`deep_ep/buffers/elastic.py:25-192`，其中 `:25-57` 是类 docstring、`:59-98` 是 `__init__`、`:100-192` 是确定性排序用的 `deterministic_sort`）由 `dispatch` 构造（`:1001-1014`）并在 `:1033` 返回。下表列出它保存的主要字段及其在 combine 中的用途，字段语义来自类 docstring（`:38-55`），读表时关注「用途」一列：它们共同构成反向路由所需的全部信息。

| 字段 | 含义 | combine 中的用途 |
|------|------|-----------------|
| `topk_idx` | dispatch 时的 top-k 专家索引副本 | 确定每个 token 要送回哪些源 rank |
| `recv_src_metadata` | 源 token 索引与缓冲区 slot 索引 | 定位数据应写回的位置 |
| `psum_num_recv_tokens_per_scaleup_rank` | 各 scaleup rank 接收量的前缀和 | 确定接收区有效长度 |
| `dst_buffer_slot_idx` | dispatch 分配的目标 slot | 复用布局，跳过重新计数 |
| `num_sms` | dispatch 使用的 SM 数 | combine 默认复用（`elastic.py:1086`） |
| `token_metadata_at_forward` / `channel_linked_list` | hybrid 模式的转发元数据 | 跨机路径的通道组织 |

读表结论：这些字段全部是布局与路由信息，不含通信资源句柄；通信资源（`dev_comm`、`window`）由 2.2 节初始化并按 buffer 复用，因此 handle 可以跨调用传递而不涉及资源所有权。

`ElasticBuffer.combine`（`elastic.py:1046-1107`）从 handle 取出这些字段传给 C++ runtime（`:1091-1106`），并把 `num_sms` 默认设为 `handle.num_sms`（`:1086` 的运行时回退取值，函数签名默认值其实是 `:1051` 的 `num_sms: int = 0`）。`EPHandle` 还能作为**缓存 handle** 传回下一次 `dispatch`：类 docstring（`:28-29`）说明可复用，`dispatch` 的参数是 `:868` 的 `handle`，README 的 decode 示例（`README.md:271,284-301`）也演示了这种用法。传入后 `:937-956` 从 handle 读取 `topk_idx` 等参数、把 `do_cpu_sync` 强制设为 False，`launch_dispatch` 再把 `cached_mode` 转成 `reuse_slot_indices`（`dispatch.hpp:176`、`:202`），内核走 `dispatch.cuh:338-342` 分支跳过重复的布局计算。这就是 3.4 节所说的「通信 handle 复用」在 V2 里的具体实现。

combine 内核 `combine_impl`（`deep_ep/include/deep_ep/impls/combine.cuh:15-35`）与 dispatch 结构对称：同样的模板特化、同样的 `NCCLGin` 与 `get_qp_mode`（`:71-74`）、前后各一次 `gpu_barrier`（`:76-80`、`:239-242`）。数据搬运同样按 NVLink 可达性分流——`:192-207` 在展开发送模式下可达时 `tma_store_1d` 直写对端、不可达时先落本地 send buffer 再 `gin.put`（`:206-207`），非展开模式在 `:230-236` 用 `gin.put` 发 RDMA。差别在于 combine 读的是专家输出、写回原 rank 并做归约，因此它依赖 `EPHandle` 里的源索引与布局信息来反向寻址；接收 token 总数则由 `:45-46` 从 `psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1` 读出。

至此 dispatch 与 combine 的调用链闭合；下一节回到 2.2 节第 3 步留下的伏笔，展开 `symmetric::alloc` 的三种实现。

### 2.7 对称内存的三种实现

2.2 节第 3 步提到 `symmetric::alloc` 会按条件选实现，本节回答「为什么需要三种内存形态、各自解决什么约束」。它从 2.2 节的结论引出：内存必须同时满足「可供 NCCL 注册为窗口」「2 MB 对齐」「能容纳可能超出显存容量的数据」。读完应能回答：三种实现的选择条件、内存构成、以及 DeepEP 通过哪些手段影响 NCCL 对这段内存的处理。

之所以分成三种，是因为容量与拓扑产生了矛盾：默认路径只需显存，而启用 CPU 弹性内存段后需要把 GPU 与 CPU 物理内存拼成一段连续 VA；hybrid 模式下每个 rank 还要能访问同节点其他 rank 的 CPU 段，映射关系更复杂。`csrc/kernels/backend/symmetric.hpp:291-317` 的选择逻辑是：无 CPU 段用 `GPUSymmetricMemory`；有 CPU 段且 `allow_hybrid_mode` 用 `HybridElasticSymmetricMemory`，否则用 `ElasticSymmetricMemory`（`:299-310`）。三者的内存构成与适用场景如下，读表时关注「内存构成」一列。

| 实现类 | 定义位置 | 内存构成 | 适用场景 |
|--------|---------|---------|---------|
| `GPUSymmetricMemory` | `symmetric.hpp:124-140` | 纯 GPU 显存（`ncclMemAlloc`） | 默认路径，数据全在显存 |
| `ElasticSymmetricMemory` | `symmetric.hpp:145-186` | GPU 段（前）+ NUMA 本地 CPU 段（后） | 用 CPU 内存扩充容量 |
| `HybridElasticSymmetricMemory` | `symmetric.hpp:191-289` | GPU 段 + 各 scaleup rank 各一段 CPU 内存 | hybrid 模式，各 rank 的 CPU 段都映射进 VA |

读表结论：三者都构造出一段**连续且 2 MB 对齐**的虚拟地址空间（`symmetric.hpp:16` 的 `kNumAlignmentBytes = 2097152`，`:129`、`:175`、`:251` 各有对齐断言），因而都能传给 `ncclCommWindowRegister`（`nccl.cu:140`）。后两者通过 CUDA Driver API 逐段映射：`lazy_cuMemAddressReserve`（`:164`、`:215`）、`cumem_create_with_fallback`（`:167`、`:171`，内部调用 `lazy_cuMemCreate`，`:73-87`）、`lazy_cuMemMap`（`:168`、`:172`）与 `set_access`（`:169`、`:173`，内部调用 `lazy_cuMemSetAccess`，`:90-111`）；GPU 段在 `:52-61` 设置 `gpuDirectRDMACapable = 1`，使网卡可以直接读写这段显存。

有一处表述需要按源码订正：初稿写「所有 rank 映射到相同地址」，这并非 DeepEP 代码给出的保证。`lazy_cuMemAddressReserve(&addr, this->num_bytes, kNumAlignmentBytes, 0, 0)`（`:164`、`:215`）的第 4 个参数是 `0`，表示不指定固定基址，各 rank 各自 reserve；`nccl.cu:145-147` 为每个 peer 单独调用 `ncclGetLsaDevicePointer`，也说明基址不必相同。统一对称 VA 视图是 NCCL 在窗口注册阶段建立的（`NCCL_Gin_and_Symmetric_Memory.md:306-309`），内核通过 `nvl_window_ptrs[i]` 与 `ncclGetLsaPointer` 逐 peer 换算地址；纯 GPU 路径的 `ncclMemAlloc`（`symmetric.hpp:128`）则由 NCCL 分配器隐含对称语义。因此准确说法是「交给 NCCL 注册后形成统一的逻辑视图」，而不是「各 rank 的 reserve 地址相等」。

Hybrid 实现还多一层进程间协作：每个 rank 用 `create_cpu_handle`（`symmetric.hpp:271-288`）导出自己 CPU 段的 POSIX 文件描述符，Python 侧 `elastic.py:337-342` 用 `dist.all_gather_object` 汇集 `(pid, fd)`，构造时再按 `cpu_comm` 逐段 `lazy_cuMemImportFromShareableHandle` + `lazy_cuMemMap` 映射进来（`symmetric.hpp:224-249`）。这也回扣了 2.1 节留下的伏笔：一旦启用 CPU 段，`elastic.py:301` 就强制 `force_new_comm=True`。源码没有说明这一强制的原因；可以核对的是 `get_nccl_comm_handle` 对该参数的定义——既不使用缓存、也不复用 PyTorch 的 communicator，始终新建一个由 DeepEP 管理的 communicator（`deep_ep/utils/comm.py:49-50`）。

当选中 `ElasticSymmetricMemory` 时，`alloc` 还会设置环境变量启用 NCCL 的弹性缓冲注册（`symmetric.hpp:312-315` 的 `NCCL_ELASTIC_BUFFER_REGISTER=1`，注意只对 Elastic 分支生效）。这是 DeepEP 影响 NCCL 行为的一个例子：它不只调用 NCCL 的分配器，还会通过环境与窗口注册参数改变 NCCL 对这段内存的处理方式。`elastic.py:284-298` 在 `num_cpu_bytes > 0` 且估算的注册字节数超过设备总显存时（`:291-293`）设置 `NCCL_WIN_STRIDE`（`:298`），也是同一类做法。

至此第二章的初始化与调用两条线都已补齐；下一节回到 1.1 节的另一半内容，说明 V1 的跨节点实现，以便理解 V2 切换后端具体改了什么。

### 2.8 两代后端：从 NVSHMEM 到 NCCL Gin

前面各节都沿 V2 路径，本节补齐 V1 的跨节点实现，回扣 1.1 节「V1 完全没有 `ncclComm_t`」与 2.1 节「V2 何时自建 communicator」两处伏笔。读完应能回答：V1 初始化交换了哪些资源、IBGDA 的作用、V2 切换后端后在两个通信域里各改用哪条路径、以及这次切换损失了什么能力。

V1 的初始化分两条线，都不经过 NCCL：机内用 CUDA IPC 共享内存，跨节点用 NVSHMEM。Python 侧 `Buffer.__init__`（`deep_ep/buffers/legacy.py:33-136`，其中 `:92-136` 是构造 runtime 与资源交换的部分）先构造 C++ runtime（`:92-93`），再交换设备 ID（`:95-97`）、IPC handles（`:99-101`）与 NVSHMEM unique ID（`:103-132`），最后调用 `runtime.sync`（`:135`）。其中启用 **IBGDA**（InfiniBand GPU Direct Async，GPU 直接访问网卡）的环境变量设置在 `legacy.py:108-110`，下面这段代码重点看三个变量分别控制什么。

```python
# 中文注释为笔记所加（以「笔记：」开头）；以下为 deep_ep/buffers/legacy.py:108-110 原文，未省略
            # 笔记：是否在低延迟模式允许 NVLink P2P；关闭时 NVSHMEM 不经 NVLink
            os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
            # 笔记：开启 IBGDA，使 GPU 内核线程可直接驱动网卡发起 RDMA
            os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
            # 笔记：每个 PE（NVSHMEM 中的处理单元，对应一个 rank）的 RC QP 数量
            os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

读码结论：这三行决定了 V1 跨节点通信由 GPU 侧直接发起 RDMA；`:113-126` 还设置了 `NVSHMEM_QP_DEPTH`、`NVSHMEM_MAX_TEAMS`、`NVSHMEM_CUMEM_GRANULARITY` 等变量，用于保证在途请求数、限制 team 数量与分配粒度。

C++ 侧 `csrc/legacy/buffer.hpp:232-280` 完成 IPC handle 的导入映射（`:232-253`，其中 `:242` 调用 `open_mem_handle`），并在需要时调 `nvshmem::init` 与 `nvshmem::alloc`（`:255-285`，调用点在 `:264-268`）。`nvshmem::init` 的实现在 `csrc/kernels/backend/nvshmem.cu:49-76`：用 unique ID 初始化 NVSHMEM（`:53-57`），再按 `team_split_stride` 切出跨节点 RDMA 子 team（`:60-71` 的 `nvshmem_team_split_strided`，条件是 `num_ranks > team_split_stride`），最后做一次全体 barrier（`:74`）。机内部分则由 `shared_memory::MemHandle` 走 CUDA IPC（`csrc/legacy/buffer.hpp:49`、`:132-136`），默认 `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`（`csrc/utils/shared_memory.hpp:96`、`:111`）；`allow_mnnvl` 为真时还有 `CUmemFabricHandle` 变体（同文件 `:91-94`、`:101-108`）。

V2 去掉了这条依赖：`deep_ep/buffers/elastic.py` 中不再有任何 NVSHMEM 初始化，跨节点数据统一经 `gin.put` / `gin.get` 走 NCCL Gin。下表按通信域对比两代，注意机内路径也一并改变了；读表时关注「机内」一行，它说明后端切换不只是换了一个跨节点 API。

| 通信域 | V1 | V2 |
|--------|----|----|
| 机内（NVLink 域） | CUDA IPC 共享内存 | NCCL 对称内存窗口 + LSA 指针 / TMA store |
| 跨机（RDMA） | NVSHMEM + IBGDA | NCCL Gin（`gin.put` / `gin.get`） |
| GPU 侧发起网络 | NVSHMEM 的 put/get | 统一走 NCCL GIN 设备 API |
| 进程间控制面 | IPC handles + NVSHMEM unique ID 交换 | 复用或自建 NCCL communicator |

读表结论：V1 的两条路径由两套独立的内存与同步机制支撑，V2 把它们收敛到同一套 NCCL 对称内存窗口与设备侧 GIN API 之下。README 对这次切换的表述是 "V2 has switched from the NVSHMEM backend to the more lightweight NCCL Gin backend"（`README.md:9`）。同时要注意 V2 的能力边界：`README.md:30` 明确写了 "0 SM RDMA low-latency EP is no longer supported"，即 V1 那种零 SM 占用的 RDMA 低延迟 EP 在 V2 不再提供，不能把「接口统一」理解成「能力等价」。

至此第二章的链条闭合：初始化期由 Host 准备好 communicator、设备上下文、窗口与地址表；每次调用期由 Python 整理参数、C++ 启动器做 JIT 实例化、GPU 内核按路由动态计数并分流搬运，全程不调用 NCCL 的集合通信 API。第三章不再逐行跟随这条链路，改为归纳决定性能特征的设计选择。

---

## 三、设计要点与源码阅读路径

第二章沿调用链读完了代码，本章不再逐行跟随实现细节，回答「这些实现解决了什么工程问题、自己继续读源码该按什么顺序」。它建立在第二章的结论之上：既然 DeepEP 不调用集合通信 API、而是把通信写进内核，性能特征就由内核设计决定。前三节对应「内核怎么适配不规则通信」「通信与计算怎么重叠」「低延迟场景做了什么取舍」，最后两节是工程细节与阅读路径。

### 3.1 面向不规则 all-to-all 的定制内核

本章先回答最根本的一点：为什么 DeepEP 不复用现成的集合操作。这个问题从 1.2 节「`ncclAlltoAll` 无法表达不等量布局」的结论引出，第二章 2.4 已给出内核里的具体做法，这里按设计意图归纳。读完应能回答：不规则布局在内核里如何现算、并行度来自哪里、布局信息为什么能与数据分离。

最根本的一点是：DeepEP 没有把 MoE 的 dispatch/combine 直接用 `ncclAlltoAll` 的等量模型表达，而是自己写了针对动态布局的内核。第二章 2.4 已经看到具体做法，这里按设计意图归纳三条：

1. **布局在内核里现算**：notify warps 用共享内存原子加统计逐 token 的 rank/expert 计数，全 SM 归约后由 warp 前缀和算出各段起始偏移（`dispatch.cuh:92-115`、`:232-257`）。V2 的产物是 `psum_num_recv_tokens_per_scaleup_rank` 与 `psum_num_recv_tokens_per_expert`；V1 里对应的概念叫 `rank_prefix_matrix` / `channel_prefix_matrix`（`legacy.py:391`），两者名字不同但目的一致。
2. **warp 即 channel**：每个 dispatch warp 独立处理一批 token，配一个 `NCCLGin` 实例，并按需分配 QP 与共享模式（`dispatch.cuh:67-71`、`comm.cuh:56-86`）。这是细粒度并行的来源，但 QP 与 warp 不是一一对应，多 SM 会共享 QP 池（2.5 节）。
3. **数据与布局分离**：token 的 hidden/SF 用 TMA 搬入共享内存，目标 slot 由原子计数分配，源信息记入 `recv_src_metadata` 供 combine 反向寻址。布局信息因此能在 `EPHandle` 中缓存复用（2.6 节）。

这三条共同说明：不规则的「量」与「位置」都被转化为内核内的局部计算，而不是 Host 侧预先排好的静态表。下一节接着看这种内核级通信给通信-计算重叠带来了什么变化。

### 3.2 内核级通信带来的重叠

上一节说明通信动作被写进了内核，本节回答「边界细化之后重叠粒度有什么不同」。它从 3.1 节的第三条引出来：既然数据搬运与布局计算都在 GPU 线程里完成，就不必再以「整个内核」为单位与计算划界。读完应能回答：省去了哪一段 Host 参与、warp 级重叠如何实现、以及到达保证由谁给出。

常规集合通信里，通信通常作为独立的内核或流任务出现，计算与通信的边界比较粗。DeepEP 借 GIN 让 GPU 内核里的线程在搬运过程中直接发起 RDMA（`dispatch.cuh:389-390`、`handle.cuh:183-196`），边界细化到 warp 与 token 级。

收益有两层：一是省去「内核结束 → Host 代理线程提交 → 网卡执行」的 Host 参与，二是 warp 可以一边为下一个 token 做 TMA 搬运，一边让已发出的 `gin.put` 在后台完成。第二章 2.5 提到的 `gpu_barrier`（`comm.cuh:214-270`）则负责在这条细粒度流水线末端给出全体 rank 的到达保证——它用 GIN signal 实现，同样不需要 CPU 同步点。用户侧把重叠表达出来的接口是 `ElasticBuffer.dispatch` 的 `async_with_compute_stream`（`elastic.py:866`）与返回的 `EventOverlap`（`elastic.py:1017`），README 的示例（`README.md:255-267`）用 `event.current_stream_wait()` 在真正使用接收数据前等待通信流。

粒度变细之后，推理 decode 的小批量场景还需要进一步压缩延迟，下一节看 V1 为此做了哪些取舍、V2 是否等价承接。

### 3.3 面向推理的低延迟设计

1.3 节把低延迟列为第三类场景，本节回答「这一场景的具体取舍是什么、两代接口如何变化」。它从 3.2 节的「重叠粒度」引出来：decode 阶段 batch 小，等待与同步的固定开销比带宽更关键。读完应能回答：V1 低延迟内核的四项取舍、V2 用哪些参数表达同类诉求、以及哪项能力并未迁移。

V1 提供了独立的 `low_latency_dispatch` / `low_latency_combine` 接口（`deep_ep/buffers/legacy.py:553`、`:624`），专为推理 decode 的小批量、延迟敏感场景设计。几个可核对的取舍：

- **纯 RDMA 路径**：所有 rank 经 RDMA 互访，不经过 NVLink 转发，把跨机延迟压到最低（`docs/legacy.md:28-30`；`legacy.py:561-563` 还要求 IBGDA 必须启用）。
- **固定缓冲 + 不做 CPU 同步**：接收缓冲按 `num_max_dispatch_tokens_per_rank` 预先固定分配，且 docstring 明确说明「不同步 CPU 侧接收计数，否则与 CUDA Graph 不兼容」（`legacy.py:599-600`、`legacy.py:602`）。
- **接收 hook**：`return_recv_hook=True` 时内核只发起 RDMA 请求、不等待数据到达，由调用方在真正需要数据时执行 hook（`legacy.py:584-585`）。`docs/legacy.md:11` 指出这种方式不占用 SM，可把网络流量与计算重叠。
- **FP8 与统计**：`use_fp8` 在通信过程中做量化以减少传输量（`legacy.py:557`、`:580`）；`dispatch_wait_recv_cost_stats` / `combine_wait_recv_cost_stats` 统计等待接收的耗时，用于定位慢节点（`legacy.py:556,577-578`、`:627,653-654`）。

V2 把高吞吐与低延迟 API 统一进 `ElasticBuffer`（`README.md:18`），通过 `async_with_compute_stream`、`do_cpu_sync` 等参数表达重叠诉求（`elastic.py:866,870`），没有独立的 `low_latency_*` 方法。但要注意 `README.md:30` 的说明：**零 SM 占用的 RDMA 低延迟 EP 在 V2 不再支持**，因此 V1 的这一能力并非被等价迁移。

这些取舍之外还有一批不依赖具体场景的工程性设计，下一节归纳，并闭合 2.5、2.6 两节留下的两处伏笔。

### 3.4 工程细节

2.6 节把 `EPHandle` 的缓存复用称为「通信 handle 复用」，2.5 节则把超时检查留到本节展开。本节回答「哪些机制是两代共有、只是接口名不同」，读完应能回答：布局缓存、专家对齐、rank 屏蔽、超时诊断各自的实现位置与动机。

以下几项是 V1/V2 共有的工程性设计，只是接口名不同（V1 用元组 handle，V2 用 `EPHandle`）：

- **布局缓存**：`EPHandle` 的 docstring（`elastic.py:28-29`）说明可复用，`dispatch` 的 `handle` 参数在 `:868`、复用逻辑在 `:937-956`，内核侧对应 `dispatch.cuh:338-342`。
- **专家对齐**：`expert_alignment`（`elastic.py:862` 的签名参数，`:896` 的 docstring 说明）把每个专家收到的 token 数对齐到指定倍数，便于后续 GEMM 访存对齐；内核侧是 `dispatch.cuh:215` 的 `math::align`。
- **rank 动态屏蔽**：`low_latency_update_mask_buffer`（`legacy.py:672`）支持在通信中屏蔽指定 rank，用于在线服务的故障隔离。
- **超时与可观测性**：内核用 `kNumTimeoutCycles` 与 `comm::timeout_while` 做等待超时检查并打印诊断（`dispatch.cuh:123-146`），避免分布式场景下静默挂死。

最后一项的写法值得展开。`comm::timeout_while`（`comm.cuh:30-49`）以 `clock64()` 计时：循环内用 `clock64() - start_clock >= kNumTimeoutCycles` 判断超时（`:38`），超时那一轮先把 `timeout=true` 传给回调做最后一次带现场信息的检查（`:39-40`，`dispatch.cuh:139-144` 打印 `status/expected`），若仍不满足则空转 `kNumOneSecCycles = 2000000000`（`comm.cuh:13`）让所有线程打印完诊断，再执行 `ptx::trap()` 终止内核（`:42-47`）。`kNumTimeoutCycles` 不是全局常量而是模板参数（`dispatch.cuh:26`），其来源链是：`elastic.py:245` 的 `num_gpu_timeout_secs=100` → `buffer.hpp:121-123` 乘以 GPU 时钟率得到周期数 → `dispatch.hpp:210` 填入 `Args.num_timeout_cycles` → `dispatch.hpp:65` 拼成模板实参。这正是 2.3 节「运行期参数变成编译期常量」在超时机制上的体现。

至此各项设计选择已归纳完毕；最后一节给出自己继续读源码时的顺序建议。

### 3.5 建议的源码阅读路径

前面四节归纳了设计选择，本节回答「脱离笔记后按什么顺序自己读」。它从第二章的调用链引出来：按由浅入深的层次走完 V2 主路径，再回头看 V1 作对比。读完应能回答：每一步打开哪个文件、关注什么、以及为什么把这个文件放在这一步。

读完本笔记后若想自己深入，建议按第二章的调用链顺序由浅入深，先走完 V2 主路径再回头看 V1 作对比；下表按顺序列出文件与关注点，读表时注意第 2、4、5 行分别对应初始化期、Host 启动期与设备侧封装，与 2.1 至 2.5 节一一对应。

| 顺序 | 文件 | 关注点 |
|------|------|--------|
| 1 | `deep_ep/buffers/elastic.py` | `ElasticBuffer.__init__` / `dispatch` / `combine` 的参数整理与 `EPHandle` |
| 2 | `deep_ep/utils/comm.py`、`csrc/kernels/backend/nccl.cu` | communicator 三条分支；`NCCLSymmetricMemoryContext` 四步 |
| 3 | `csrc/kernels/backend/symmetric.hpp` | `symmetric::alloc` 与三种对称内存实现 |
| 4 | `csrc/elastic/buffer.hpp`、`csrc/kernels/elastic/dispatch.hpp`、`csrc/jit/compiler.hpp`、`csrc/jit/launch_runtime.hpp` | Host 侧启动器、`DispatchRuntime::Args`、JIT 的 generate / build / launch 三步 |
| 5 | `deep_ep/include/deep_ep/common/handle.cuh`、`comm.cuh`、`ptx.cuh` | `NCCLGin` 封装、Team 判定、`get_qp_mode`、`gpu_barrier`、PTX 封装与去重/超时写法 |
| 6 | `deep_ep/include/deep_ep/impls/dispatch.cuh` / `combine.cuh` | 内核三阶段、NVLink 直写与 GIN put 分流、combine 反向路由 |
| 7 | `deep_ep/buffers/legacy.py`、`csrc/legacy/buffer.hpp`、`csrc/kernels/backend/nvshmem.cu` | V1 的 IPC + NVSHMEM 初始化与低延迟接口 |

读表结论：前六行构成 V2 的完整链路，第 7 行只用于对比；若时间有限，优先读第 2、5、6 行，它们分别对应 NCCL 能力入口、设备侧选路与内核主体。

---

## 参考资料

- DeepEP 官方源码仓库：[github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)（本笔记基于 v2.1.0）
- DeepEP V1 legacy 文档：官方仓库 `docs/legacy.md`（NVSHMEM 时代的接口、性能数据与示例）
- NCCL 官方源码仓库：[github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)（本笔记对照 master 分支；device API 声明见 `src/nccl.h.in`、`src/include/nccl_device/core.h`、`src/include/nccl_device/gin.h`）
- NCCL 设备 API 用户文档：`docs/userguide/source/usage/deviceapi.rst`（GIN、LSA、Team 定义）、`docs/userguide/source/api/device_setup.rst`（`ncclDevCommRequirements_t` 各字段）、`docs/userguide/source/api/device_gin.rst`（GIN 设备 API 与弃用标记）
- NCCL 对称内存与 GIN 设计文档：`docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md`（LSA 定义、窗口注册与统一 VA 视图）
- [《NCCL 源码学习笔记》](/notes/nccl-source-study/)： communicator 初始化、设备侧 context、GIN 后端与 QP 参数传递

---

*声明：本文是作者学习 DeepEP 源码过程中的阶段性笔记，内容基于 DeepEP v2.1.0 与 NCCL master 分支源码整理。由于这是一份仍在持续完善的初稿，文中可能存在表述不准确或理解不完整之处，欢迎通过邮件联系指正。*
