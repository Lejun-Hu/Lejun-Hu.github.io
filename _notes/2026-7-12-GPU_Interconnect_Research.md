---
title: "GPU卡间互联协议深度调研：NVLink/NVSwitch及其替代方案全景分析（未完成）"
permalink: /notes/GPU_Interconnect_Research/
date: 2026-08-15
category: "AI 硬件"
tags: ["GPU互联", "NVLink", "NVSwitch", "UALink", "InfiniBand", "Scale-up", "Scale-out", "交换芯片"]
description: "从NVLink/NVSwitch出发，全景拆解GPU卡间互联协议与交换芯片技术栈——涵盖英伟达scale-up/scale-out体系、国内外替代协议竞争格局、交换芯片市场格局与成本拆解。"
published: true
---

> 2022年底，ChatGPT发布，将世界推入大模型时代。此后短短四年间，模型参数量从千亿膨胀到万亿，训练集群从百卡跃升到十万卡——但这背后有一个越来越尖锐的矛盾：**单颗芯片的算力增长，远远追不上模型规模膨胀的速度。**
>
> 解法看似简单：堆更多的芯片一起算。但问题随之而来——芯片之间如何通信？如果一万张GPU之间传输一次数据的延迟过高，算力利用率将急剧下降。这就是现代AI集群的关键瓶颈：**互联通信。**
>
> 这个领域长期以来被英伟达的NVLink+NVSwitch方案统治，但近年来发生了剧烈的变化：在协议层，AMD、Intel、Google、AWS等国际巨头纷纷推出替代方案，寒武纪、华为昇腾、壁仞、摩尔线程、沐曦等国产力量也在紧追，UALink和OISA等开放标准联盟试图打破封闭垄断；在交换层，博通的Tomahawk Ultra以SUE（Scale-Up Ethernet）架构正面叫板NVSwitch，UALink联盟定义了开放的ULS交换标准，英伟达则推出NVLink Fusion试图拉拢Marvell、Astera Labs等厂商共建生态。底层的交换芯片市场由博通、Marvell等巨头把控，形成了复杂的产业链博弈。
>
> 而对这些技术的深入理解，对于从事集合通信库（XCCL、NCCL、HCCL、RCCL等）开发、大规模模型训练以及推理部署的工程师来说，都是必修课——通信库的本质，就是要在给定的硬件互联拓扑之上，把数据传输的效率压榨到极致；而模型的分布式训练策略与部署方案，同样需要对底层组网有清晰认知，才能合理规划张量并行、流水线并行和数据并行的切分与调度。

---

## 一、英伟达的Scale-up与Scale-out体系拆解

在任何一个AI算力集群中，网络实际上分为**两个层次**：

**Scale-up（纵向扩展/超节点互联）：** 在**单个NVLink域**内工作——从一台8卡服务器内部的GPU直连，到通过NVSwitch将跨节点GPU编织成全互联的统一计算域。英伟达最新的NVL72机柜将72颗Blackwell GPU置于同一NVLink域中，任意两卡之间1.8TB/s带宽，通过NVSwitch实现all-to-all无阻塞全互联。这一层的特点是**TB/s级带宽、纳秒级延迟、内存语义通信**——大模型的张量并行（TP）和专家并行（EP）重度依赖它。代表技术：**NVLink（GPU直连协议）+ NVSwitch（交换芯片，可扩展到跨节点全互联）**。

**Scale-out（横向扩展/跨Pod互联）：** 当一个NVLink域内的GPU数量达到上限（如NVL72的72颗），仍需要更多算力时，就需要将多个NVLink域连接起来。这就是scale-out的领域——不同机柜、不同Pod甚至不同数据中心之间的网络。这一层使用**InfiniBand或以太网（RoCEv2/UEC）**协议，带宽在几百Gb/s量级，主要承载数据并行（DP）和流水线并行（PP）等对延迟相对宽容的通信模式。InfiniBand本质上是基于**RDMA**的一种私有化协议，由IBTA（InfiniBand Trade Association）定义标准，但实际实现和生态高度集中于英伟达手中。英伟达在这层使用的Quantum InfiniBand交换机和Spectrum以太网交换机均采用**自研Mellanox ASIC芯片**。

> 一个容易混淆的点：**NVSwitch通过第二层交换可以实现跨节点互联（如Hopper时代的256 GPU NVLink域），但这仍然是scale-up，不是scale-out。** 因为所有GPU在一个NVLink域内共享统一地址空间，走的是内存语义。真正的scale-out始于NVLink域的边界——当你需要用InfiniBand或Spectrum-X以太网连接不同的NVLink域时，才进入scale-out的领地。

### 1.1 NVLink技术演进简述

NVLink是一种**自研的专有高速串行互联协议**，用于GPU到GPU以及GPU到CPU之间的直接通信。自2014年第一代推出以来，经历了六代演进：

| 代次 | 架构 | 每GPU链路数 | 每GPU总带宽(双向) | 关键特性 |
|---|---|---|---|---|
| NVLink 1 | Pascal P100 (2016) | 4 | 160 GB/s | 首代NVLink，替代PCIe P2P |
| NVLink 2 | Volta V100 (2017) | 6 | 300 GB/s | 引入NVSwitch (DGX-2) |
| NVLink 3 | Ampere A100 (2020) | 12 | 600 GB/s | sub-link改为4对差分信号线 |
| NVLink 4 | Hopper H100 (2022) | 18 | 900 GB/s | NVSwitch支持256 GPU，引入SHARP网络内计算 |
| NVLink 5 | Blackwell B200 (2024) | 18 | 1,800 GB/s | 224G PAM4 SerDes，NVL72支持72 GPU全互联 |
| NVLink 6 | Vera Rubin (CES 2026发布，已全面量产) | 36 | 3,600 GB/s | 链路数翻倍，NVL144聚合带宽520TB/s |

**第五代NVLink关键指标：** 每条链路100GB/s双向速率，Blackwell GPU配备18条链路，单GPU总带宽达到**1.8TB/s**，是PCIe 5.0 x16带宽（128GB/s）的**14倍**。GB200 NVL72系统在单个NVLink域内连接72个GPU，聚合带宽达**130TB/s**。

### 1.2 英伟达产品命名规则简述

在深入技术细节之前，有必要先厘清英伟达的产品命名体系，因为类似"GB200""NVL72""VR200"这样的代号贯穿本文。

**GPU 架构命名：致敬科学巨匠。** 英伟达每代 GPU 架构以著名科学家命名：Ampere（A）、Hopper（H）、Blackwell（B）、Rubin（R），下一代为 Feynman。数据中心 GPU 通常以"架构首字母 + 性能层级"命名，如 A100、H100、B200。

**"G+B" 组合：从 GPU 到超级芯片。** 从 GH200（Grace Hopper）开始，英伟达推出整合 CPU 与 GPU 的"超级芯片"。GB200 即 **G**race CPU + **B**lackwell GPU 的组合，由一个 Grace CPU 与两个 Blackwell GPU 封装而成。下一代 Vera Rubin 平台延续此逻辑——**V**era CPU + **R**ubin GPU，VR200 为其代表型号。

**NVL + 数字：NVLink 域规模。** "NVL72"指通过 NVSwitch 将 **72** 颗 GPU 全互联的机柜系统，NVL144 则为 144 颗。此前 Hopper 时代的 256 GPU NVLink 域同理，只是未以 NVL 命名。

| 代号 | 含义 |
|---|---|
| **G** | Grace CPU（ARM 架构自研数据中心 CPU） |
| **B** | Blackwell GPU |
| **H** | Hopper GPU |
| **V** | Vera CPU（下一代自研 CPU） |
| **R** | Rubin GPU |
| **NVL72** | 72 GPU 通过 NVLink 全互联的机柜系统 |
| **GH200** | Grace + Hopper 超级芯片 |
| **GB200** | Grace + Blackwell 超级芯片 |
| **VR200** | Vera + Rubin 超级芯片 |

### 1.3 NVLink、NVSwitch 与 Scale-out 交换网络的分工

在讨论 GPU 互联时，一个常见的混淆是把 NVLink、NVSwitch、InfiniBand、Spectrum-X 这些名词混为一谈。它们实际上分属**三个不同层次**，各自承担不同的角色，共同构成了英伟达从单机到万卡集群的完整互联体系。下面自底向上逐一梳理。

**第一层：NVLink — GPU 之间的点对点直连协议。**

NVLink 定义了相邻两颗 GPU 之间以什么格式、以什么速度传输数据。它是纯粹的物理链路层协议：每颗 Blackwell GPU 配备 18 条 NVLink 链路，每条 100GB/s 双向，单 GPU 总带宽 1.8TB/s。NVLink 工作在服务器机箱内部，解决的是"两颗紧挨着的 GPU 如何高效通信"的问题。它的核心特征是**内存语义**（load/store）——GPU A 可以直接用一条访存指令 `ld.global [addr_on_B]` 读写 GPU B 的 HBM，NVLink 硬件自动完成地址翻译，全程不需要软件介入，延迟在纳秒级。

**第二层：NVSwitch — 将多颗 GPU 编织成全互联域。**

NVLink 只能做点对点连接，而 NVSwitch 是交换芯片，将多颗 GPU 的多条 NVLink 链路汇聚成一个 **all-to-all 全互联交换网络**。单个服务器内的 NVSwitch 让 8 颗 GPU 彼此全速通信；外部 NVSwitch 则跨节点扩展，将一个 NVLink 域扩展到最多 576 颗 GPU（Blackwell 理论值）。NVSwitch 定义了多链路之间的无阻塞交换能力（例如 72×72 全互联），它和 NVLink 共同构成了英伟达的 **scale-up 体系**——追求极致带宽、纳秒延迟和内存语义通信。在同一个 NVLink 域内，所有 GPU 共享由硬件维护的统一地址空间和缓存一致性协议，任何一张卡都可以像访问本地显存一样直接 load/store 远端数据。

**第三层：Scale-out 交换网络 — InfiniBand 与 Spectrum 以太网两条产品线。**

当一个 NVLink 域内的 GPU 数量达到物理上限、仍需更多算力时，就需要将多个 NVLink 域连接起来，进入 scale-out 的领域。这一层与 NVLink 域有本质区别：**不同 NVLink 域之间没有共享地址空间，GPU 不能像域内那样直接用 load/store 跨域读写。** 取而代之的是消息传递模型——数据被显式地打包、寻址，通过网卡和交换机在域间路由。英伟达在这一层布局了两条产品线，面向不同的客户场景和生态策略。

**InfiniBand** 是英伟达 2019 年以 69 亿美元收购 Mellanox 后获得的核心资产。它是一种基于 RDMA 的私有化协议，Quantum 系列交换机（NDR 400Gbps / XDR 800Gbps）通过 ConnectX 网卡连接各 GPU，采用 Fat-Tree 胖树拓扑将数百个 NVLink 域组成万卡集群。跨域通信的具体流程是：发送端 GPU 提前向自己的 ConnectX 网卡注册一块本地显存区域并获得 rkey，然后发起 RDMA Write 请求，网卡通过 IB 网络直接将数据推送到远端 GPU 已注册的显存区域——全程不经过远端 CPU，接收端 GPU 甚至不参与数据传输。InfiniBand 的核心优势在于极低的端到端延迟和成熟的在网计算 INC（In-Network Computing，以 SHARP 为代表）：交换机在转发数据时可执行随路归约（网侧 Reduction），梯度聚合在交换芯片内完成，数据无需返回 GPU 再转发，对大规模训练的梯度同步至关重要。

**Spectrum-X 以太网** 则是英伟达的另一条产品线，采用自研 Spectrum ASIC 交换芯片，支持 RoCEv2（RDMA over Converged Ethernet）和 UEC 超以太网协议。它的战略定位是为不想绑定 InfiniBand 私有生态的云厂商提供基于标准以太网的 scale-out 替代方案。与 InfiniBand 相比，Spectrum-X 的延迟稍高但生态更开放——可以利用已有的以太网运维工具和人才储备，成本更低、供应链更灵活。这一层也正是英伟达与博通、Marvell 等外部交换芯片厂商**直接竞争**的主战场：博通的 Tomahawk/Jericho 系列和 Marvell 的 Teralynx 系列都是以太网交换芯片，与 Spectrum-X 争夺同一批云数据中心客户。

**那么 NVLink 域内的 load/store 和跨域的 RDMA 之间是如何协作的？** 答案是 NCCL 通信库在中间做了抽象。NCCL 会自动感知集群的物理拓扑：同一个 NVLink 域内的 GPU 之间使用 NVLink transport，走高效的内存语义通道；跨域通信则走 IB/RoCE transport，把数据传输封装为 RDMA 消息。对上层 PyTorch 用户来说，一行 `all_reduce(gradients)` 在 8 卡 NVLink 域内和在 8192 卡的 IB 网络上都能正确执行，底层通信方式的切换是透明的——只是后者的延迟和带宽会不可避免地受域间物理链路的限制。

三条线的关系可以概括为：**NVLink 定义了单条链路的带宽，NVSwitch 决定了多条链路如何无阻塞交换，而 InfiniBand / Spectrum-X 负责将不同的 NVLink 域连接为万卡集群。** NVLink + NVSwitch 属于 scale-up，域内共享地址空间、走内存语义；InfiniBand / Spectrum-X 属于 scale-out，域间走消息语义，通过 RDMA 和 NCCL 抽象层无缝衔接。

当前最具代表性的应用是 **GB200 NVL72**：18 个计算托盘（72 颗 Blackwell GPU）通过 9 个 NVSwitch 托盘实现全部互联，任意两颗 GPU 之间均为 1.8TB/s 全速通信。如果要将多个 NVL72 机柜连接起来组成更大的集群，则需要通过每颗 GPU 配备的 ConnectX-7 InfiniBand 网卡接入 Quantum 交换机，进入 scale-out 网络。

### 1.4 为什么NVSwitch不能直接扩展到万卡？

核心原因可以归结为**物理极限、拓扑约束和语义差异**三个层面：

**1. 铜缆的物理距离极限。** NVLink 5在1.8TB/s的带宽下使用无源铜缆（passive copper），信号有效传输距离仅**2-2.5米**。这就是为什么NVL72机柜中NVSwitch必须放在正中央——被上下各9个计算托盘夹在中间。

**2. 全互联拓扑不可扩展。** NVSwitch构建的是**all-to-all全互联拓扑**——域内任意两颗GPU之间都是直连的。这要求交换芯片的端口数 ≥ GPU数。如果要做1,000 GPU的全互联，需要NVSwitch芯片数量和互联线缆呈**O(n²)增长**，在物理上不可行。而InfiniBand使用的是**Fat-Tree或Dragonfly拓扑**，通过多层交换机级联来实现对数级别的扩展。

**3. 内存语义 vs 消息语义。** NVLink域内的GPU共享**统一地址空间**，走的是内存语义（load/store）——这种模式要求极低的延迟和极强的缓存一致性，维护大规模一致性协议的复杂度呈指数增长。InfiniBand走的是**消息传递语义**——数据被显式地打包、寻址、路由、传输，天然适合大规模分布式系统。

**4. 功耗和散热。** NVL72已经是120kW的功耗怪兽，需要全液冷。Scale-up域的GPU数量每翻一倍，交换芯片和互联缆的功耗都超线性增长。

> **总结：这是一个"分工不同"的设计，不是性能优劣的问题。** NVSwitch/NVLink负责的是**在一个NVLink域内实现GPU间极致带宽和内存语义通信**。InfiniBand/Spectrum-X负责的是**将多个NVLink域连接为大规模集群**——追求可扩展性、容错性、标准化管理。两者在架构上是互补关系，而非竞争关系。

### 1.5 NVLink底层协议栈

- **物理层**：采用自研224G PAM4 SerDes技术。PAM4（4-Level Pulse Amplitude Modulation，四电平脉冲幅度调制）是一种通过 4 个不同电压电平（00/01/10/11）在单个符号周期内传输 2 比特数据的高阶调制方式——相比传统的 NRZ（Non-Return-to-Zero，不归零编码）每个符号仅传输 1 比特，PAM4 在同等频率下带宽翻倍，但代价是信噪比容限降低、对信号完整性要求更高。从 Ampere 的 56Gbps → Hopper 的 112Gbps → Blackwell 的 224Gbps，SerDes 速率持续翻倍。
- **信令协议**：通过 4 对差分信号线构成一个"sub-link"作为基本物理单元。信号通过铜缆 Twinax 介质传输，在接收端使用 ADC + 高级 DSP 进行信号均衡和时钟恢复。
- **数据链路层**：使用FLIT（Flow Control Unit）固定大小包格式，包含FEC前向纠错。
- **协议层**：专有协议，支持GPU间直接内存访问（P2P RDMA-like操作），但与标准RDMA（如RoCEv2）不同。NVLink 3.0起采用分组交换架构，通过NVSwitch芯片实现all-to-all全互联。

### 1.6 成本拆解

作为 AI 技术从业者，我们日常接触最多的是模型 API 的调用成本——每百万 token 多少钱、不同厂商之间如何比价。但 API 定价的背后，是数千颗 GPU 组成的算力集群、TB 级互联带宽、全液冷散热系统以及高昂的 HBM 显存采购成本在支撑。理解一台 AI 服务器机柜的成本构成，不仅有助于把握"算力为什么这么贵"，也能对行业供应链格局建立更清晰的认知。

以下数据主要来自摩根士丹利、SemiAnalysis 等机构的公开研报，具体数字因采购规模、渠道和时点不同可能存在差异，但其量级和比例关系可以帮助读者建立一个可靠的参考框架。

以GB300 NVL72（Blackwell Ultra）和 VR200 NVL72（Vera Rubin）机架为例（数据来源：摩根士丹利 2026 年 5 月研报、SemiAnalysis）：

| 核心组件 | GB300 NVL72 | 占比 | VR200 NVL72 | 占比 | 增幅 |
|---|---|---|---|---|---|
| GPU | $2,513,700（72颗 B300） | 63.0% | $3,959,000（72颗 Rubin） | 50.8% | +57% |
| 显存/内存 | $373,500（HBM3e） | 9.4% | $2,001,600（HBM4 + LPDDR5X） | 25.7% | +435% |
| NVLink Switch 交换芯片 | $64,800（18颗 NVSwitch 5） | 1.6% | $144,000（36颗 NVSwitch 6） | 1.8% | +122% |
| 网络芯片（ConnectX / BlueField 等） | $261,000 | 6.5% | $576,000（含 ConnectX-9、BlueField-4 DPU 等） | 7.4% | +121% |
| CPU | —（Grace CPU 集成于 GPU 托盘） | — | ~$180,000（36颗 Vera CPU） | 2.3% | 新增 |
| 液冷系统（含 CDU） | $49,860 | 1.2% | $122,000 | 1.6% | +145% |
| PCB | $35,100 | 0.9% | $116,730 | 1.5% | +233% |
| 其他（电源/MLCC/ODM组装/结构件等） | ~$692,040 | 17.4% | ~$700,670 | 9.0% | +1% |
| **整柜总BOM** | **~$3,990,000** | **100%** | **~$7,800,000** | **100%** | **+95%** |

> 注：BOM（Bill of Materials，物料清单）指产品所有零部件的采购成本明细汇总，不包含研发、营销、运输等间接费用。以上数据均为投行估算的硬件采购成本，并非英伟达官方报价。

> **关键变化：** GPU 在 BOM 中的比例从 GB300 的 **63%** 下降至 VR200 的 **51%**，而显存/内存从 **9%** 暴涨至 **26%**，成为最大增量来源。这意味着 AI 算力硬件的价值正在从单一 GPU 扩散至存储、互连、PCB 等全链条。VR200 中新增的 Vera CPU、BlueField-4 DPU 以及翻倍的 NVSwitch 和 ConnectX 数量，进一步推高了非 GPU 组件的价值占比。

---

## 二、替代互联协议全景：Scale-up 层

上一节我们拆解了英伟达以 NVLink + NVSwitch 为核心的 scale-up 体系。但英伟达并非这一层的唯一玩家——随着大模型训练对 GPU 间带宽的需求持续攀升，越来越多的厂商开始推出自己的 scale-up 互联方案，试图在 GPU 直连这一核心环节打破 NVLink 的垄断。这些方案既有 AMD、Google、AWS 等国际巨头的自研协议，也有 UALink 这样的开放标准联盟，还有华为昇腾、寒武纪、壁仞等国产力量在紧追。

下文将依次对各家方案进行简要分析和对比，重点关注每项技术的核心设计思路、关键带宽指标、扩展规模上限以及商用成熟度，帮助读者快速建立 scale-up 层的竞争全景。需要说明的是，本文对各协议的介绍属于概览性总结，具体的协议层细节繁琐且枯燥，读者可快速浏览以获取大致印象，在此不必深究细节实现。

### 2.1 国外厂商方案

下表汇总了当前海外主要 scale-up 互联协议的关键参数。除了 NVLink 之外，各家的方案在开放程度和生态策略上差异明显——有的走专有封闭路线（Google ICI、AWS NeuronLink），有的试图建立开放标准联盟（UALink），还有的基于既有通用标准做延伸（PCIe 6.0、CXL 4.0）。

| 互联协议 | 提出方 | 每GPU/芯片带宽 | 最大扩展规模 | 开放程度 | 商用状态 |
|---|---|---|---|---|---|
| **AMD Infinity Fabric** | AMD | ~1.075 TB/s (MI300X, 7链路) | 8 GPU (OAM Mesh) | AMD专有，软件开源(ROCm) | 已量产 (MI300X/MI355X) |
| **UALink 1.0** | UALink联盟 (AMD/Intel/Google/Meta/Microsoft等) | 800 GB/s (4通道) | 1,024加速器 | 开放标准 | 规范已发布(2025.4)，首批硬件预计2026-2027 |
| **Google ICI** | Google | 数百GB/s (TPU v5p/v7) | 9,216芯片 (Ironwood) | Google专有，不对外 | 已量产多代，仅限 Google Cloud |
| **AWS NeuronLink** | Amazon (AWS) | 1,024 GB/s (Trainium2) | 64芯片 (UltraServer) | AWS专有，不对外 | 已量产 (Trainium2) |
| **PCIe 6.0** | PCI-SIG | 256 GB/s (x16双向) | 点对点，需 Switch 扩展 | 完全开放标准 | 已商用 |
| **CXL 4.0** | CXL联盟 | 128 GT/s | 跨机架内存池化 | 开放标准 | 规范已发布(2025.11) |

**1. AMD Infinity Fabric (XGMI)**

AMD 的 Infinity Fabric 最初是为 CPU 与 GPU 之间以及 Chiplet 芯粒之间的高速通信而设计的内部互联总线，后来被扩展为 XGMI（External Global Memory Interconnect），专门用于 Instinct 系列 GPU 之间的卡间直连。这一需求源于 AMD 在数据中心 GPU 市场对 NVIDIA 的追赶——如果不能在多卡互联带宽上与 NVLink 抗衡，即使单卡算力追平，集群扩展效率也会严重受限。

XGMI 的设计思路与 NVLink 类似，都是专有的 GPU 直连协议，但两者在规模上有明显差距。以 MI300X 为例，每颗芯片配备 7 条第四代 Infinity Fabric 链路，每条 128 GB/s 双向，总带宽约 **1.075 TB/s**，与 H100 的 NVLink 4（900 GB/s）基本处于同一量级。但 XGMI 目前仅支持 **8 颗 GPU** 以环形或网格拓扑互联，超出 8 卡就需要切换到以太网，这在实际大模型训练中是一个明显的规模瓶颈。

软件栈方面，AMD 的 **ROCm + RCCL** 对标 CUDA + NCCL，且 ROCm 完全开源，对部分云厂商有较强吸引力。AMD 也意识到专用协议的局限：下一代 MI400 系列（2026 年下半年）将放弃纯自研路线，转而支持开放的 **UALink** 协议，XGMI 则退居为 Chiplet 片内互联的角色。从量产状态来看，MI300X 和 MI355X 均已大规模出货，MI355X 采用 3nm 工艺和 CDNA 4 架构，配备 288 GB HBM3e，互联规格与 MI300X 相近但制程明显领先。

| 对比维度 | AMD XGMI (MI300X) | NVIDIA NVLink 4 (H100) | NVIDIA NVLink 5 (B200) |
|---|---|---|---|
| 单 GPU 总带宽 | ~1.075 TB/s (7 链路) | 900 GB/s (18 链路) | 1,800 GB/s (18 链路) |
| 最大互联规模 | 8 GPU | 256 GPU | 576 GPU |
| 拓扑 | 环形 / 网格 | Switch 全互联 | Switch 全互联 |
| 开放程度 | AMD 专有 | NVIDIA 专有 | NVIDIA 专有 |

**2. UALink（Ultra Accelerator Link）**

UALink 诞生于一个清晰的行业共识：超大规模云厂商（AWS、Google、Meta、Microsoft）不希望被 NVLink 单一供应商锁定。如果每部署一个 AI 集群都要连带采购整套 NVLink + NVSwitch + InfiniBand，议价权和供应链灵活性将完全丧失。于是 2024 年，AMD、Intel、Google、Meta、Microsoft、AWS、Cisco、HPE 八家联合发起 UALink 联盟，目标是定义一个**开放的、多供应商的 scale-up 互联标准**，让任何加速器厂商都可以使用同一套协议和交换芯片。截至 2026 年 1 月，联盟成员已超过 100 家。

UALink 1.0 规范于 2025 年 4 月发布，核心技术规格如下：每通道 200 GT/s（Giga Transfers per second，此处 1 Transfer = 1 Byte 有效数据，即每通道 200 GB/s），4 通道聚合 **800 GB/s** 每 GPU，支持最多 **1,024 个加速器**在单一网络架构中互联——这一数字超过了 NVLink 5 的 576 GPU。协议支持加速器之间直接读写内存（内存语义），在物理层复用了成熟以太网 SerDes 技术以降低实现成本，但上层协议栈完全重新设计以适配 GPU 的数据流特征。更重要的是，UALink 同时定义了开放的交换芯片标准 **ULS（UALink Switch）**，允许博通、Marvell、Astera Labs 等第三方厂商生产兼容芯片，从而创建一个类似以太网交换芯片的多供应商市场。

UALink 2.0 于 2026 年 4 月发布，新增了在网计算、芯粒定义和可管理性等企业级特性。商用进度方面：AMD 的 MI400 系列将是首批原生支持 UALink 的 GPU，预计 2026 年下半年出货；原生 UALink 交换 ASIC 则要等到 2027 年左右面世。这意味着当前存在一个"协议先行、硬件跟进"的窗口期——早期系统可能通过博通 Tomahawk 等以太网交换芯片隧道承载 UALink 流量。

与 NVLink 5 的 1,800 GB/s 相比，UALink 1.0 的 800 GB/s 在单卡带宽上仍有差距，但其开放性和 1,024 卡的扩展上限构成了差异化竞争力。超大规模云厂商更看重的是"不被锁定"而非"绝对性能"。

**3. Google ICI（Inter-Chip Interconnect）**

Google 的 ICI 是本文所有方案中最"异类"的一个——它不属于任何开放标准，也不对标 NVLink 的设计范式，而是一套面向 TPU 脉动阵列架构**从零开始全栈自研**的芯片间互联体系。

Google 开发 TPU 的初衷非常直接：搜索排序、广告推荐、Gemini 大模型等内部 AI 工作负载规模巨大，完全依赖外购 GPU 既不经济也无法针对自身业务做深度优化。ICI 作为 TPU 的专用互联协议，从第一代 TPU（2015 年）就开始内部部署，至今已经过七代演进。最新的 Ironwood（TPU v7）代表了当前 ICI 的工程极限：单个 Pod 可连接 **9,216 颗芯片**，最大通信跳数仅 7 跳。

ICI 的设计与 NVLink 几乎走的是完全相反的路线。NVLink 依赖 NVSwitch 构建 all-to-all 全互联交换网络，而 ICI 采用 **4×4×4 三维环面拓扑（3D Torus）**——每颗 TPU 只与六个方向上的邻居直接相连，数据经过中间节点逐跳转发。这种规则的邻居结构表面上不如交换机灵活，但对于 Transformer 训练中高度规整的 AllReduce 和 AllGather 通信模式，Torus 网络反而能获得极高的带宽利用率，且链路数量不随规模增长。

更关键的区别在物理层。TPU Pod 内部的基本构建块是 64 颗芯片组成的 4×4×4 立方体，立方体内部用铜缆互联。立方体之间则通过 **OCS（Optical Circuit Switch，光电路交换机）** 连接——这是一种基于 3D MEMS 反射镜阵列的纯光学交换设备，直接重定向光束而无需光电转换，可在约 10 秒内重新配置网络拓扑。这种铜光混合设计使 Google 能构建超大规模的单一逻辑计算域，而不会陷入 NVSwitch 那种全互联带来的平方级布线复杂度。

ICI 完全专有、仅通过 Google Cloud 对外提供，不向第三方出售芯片或授权协议。因此在开放性和生态兼容性上，ICI 几乎为零。但从纯粹的技术角度看，它证明了一条独立于 NVLink 的架构路线是可行的。

**4. AWS NeuronLink**

AWS 对自研 AI 芯片的投入逻辑与 Google 相似，但动机更偏向成本优化——作为全球最大的云计算厂商，AWS 需要为内部 AI 工作负载和对外推理服务提供比采购 NVIDIA GPU 更经济的方案。NeuronLink 是 AWS 为 Trainium 和 Inferentia 系列自研芯片定制的卡间互联协议。

Trainium2 搭载的 **NeuronLink-v3** 每芯片提供 **1,024 GB/s** 双向带宽，这一数字介于 NVLink 4（900 GB/s）和 NVLink 5（1,800 GB/s）之间。在拓扑上，16 颗 Trainium2 芯片通过 4×4 二维 Torus 组成一个 Trn2 计算实例；4 个 Trn2 实例再通过 NeuronLink 串联成 **Trn2 UltraServer**——一个包含 **64 颗芯片**的 Ring 拓扑计算集群。

Trainium3 将进一步升级为交换式拓扑结构（不再依赖纯 Ring），带宽和规模预计有明显提升。更长远的路线图中，AWS 已加入 UALink 联盟，Trainium4 将转向 UALink 开放标准，不再使用自研 NeuronLink。目前 Trainium2 已大规模量产并对外提供服务，Trn2 实例成为 AWS 上 AI 训练的主力产品之一。

**5. PCIe 6.0、CXL 4.0 与 NVLink-C2C**

前面的讨论集中在 GPU 之间的互联，但 AI 系统中还有另一条关键的数据通路：**CPU 与 GPU 之间如何通信**。这一层涉及三种技术——PCIe、CXL 以及英伟达最新的 NVLink-C2C，三者之间既有依赖关系也有定位差异，放在一起对比才能看出各自的分工。

**PCIe 6.0** 是整个体系的物理基础。它由 PCI-SIG 组织于 2022 年发布，采用 PAM4 信令和 FLIT 包格式，x16 双向带宽 **256 GB/s**。作为最成熟的通用互联标准，PCIe 的优势在于生态庞大、成本低廉，但带宽远低于 NVLink，且不支持 GPU 间内存语义访问。在 AI 系统中，它最适合承担 CPU 到 GPU 的数据搬运通道，而非 GPU 间的高速直连。

**CXL（Compute Express Link）** 则是在 PCIe 物理层之上构建的高层协议。CXL 4.0 于 2025 年 11 月发布，基于 PCIe 6.0 物理层，速率 128 GT/s，双向带宽约 **512 GB/s**。CXL 对 PCIe 的依赖关系非常明确：CXL 复用 PCIe 的 SerDes 和电气规范作为物理传输介质，但在协议层额外增加了缓存一致性（CXL.cache）、内存语义（CXL.mem）等能力。它的核心价值在于**跨机架内存池化**——让多台服务器通过 CXL 交换机共享同一块海量内存池，按需分配、弹性伸缩。这是 PCIe 本身做不到的。

**NVLink-C2C** 则是英伟达在最新 Vera Rubin 平台中引入的自研 CPU-GPU 直连技术。在上一代 Grace Hopper（GH200）中，Grace CPU 与 Hopper GPU 之间同样通过 NVLink-C2C 连接，但 Vera Rubin 将这一通道的带宽推向了一个新高度——双向 **1.8 TB/s**，延迟在纳秒级。它是 NVLink 协议在 Chiplet 片间互联层面的延伸，让 Vera CPU 能像访问本地内存一样直接读写 Rubin GPU 的 HBM。NVLink-C2C 是英伟达的私有技术，仅用于自家芯片之间的互联。根据英伟达官方 2026 年 7 月发布的 [Vera Rubin 平台部署进展](https://mp.weixin.qq.com/s/G88gqBIF8reG1WMp4cVTiw) 披露，全球已有超过 300 家合作伙伴开始部署该平台，每兆瓦 Token 吞吐量达到 Grace Blackwell 的 10 倍。

表面上看，NVLink-C2C 的 1.8 TB/s 远超 CXL 4.0 的 512 GB/s，似乎 CXL 被"碾压"了。但实际上两者解决的是**完全不同的问题**。NVLink-C2C 追求的是单节点内 CPU 和 GPU 之间**最快的点到点通道**——它的物理范围仅限于同一芯片封装或基板内部，距离极短、性能极致，但不可扩展。CXL 追求的是**跨机架的海量内存共享**——几十台服务器可以按需接入同一个 CXL 内存池，在数百 TB 级别弹性分配资源，速度和距离的权衡截然不同。

可以这样理解：在 Vera Rubin NVL72 机柜内部，Vera CPU 与 Rubin GPU 之间最频繁、最紧急的数据交换走 **NVLink-C2C** 这条专属高速公路；而当多个机柜需要共享一份超大的模型参数或 KV 缓存时，**CXL** 提供了跨节点的弹性内存池，作为整个集群的共享"仓库"按需存取。两者在架构上是共存关系：NVIDIA 用私有技术锁死自家 CPU 和 GPU 的配合效率，同时为更大规模、更灵活的集群组网保留了 CXL 这类开放标准的接入空间。

### 2.2 国产GPU互联方案

国产 GPU 的 scale-up 互联起步较晚，但在过去几年取得了显著进展。受出口管制和供应链安全需求的双重驱动，华为昇腾、寒武纪、壁仞、摩尔线程、沐曦等厂商在过去几年密集推出了各自的互联协议，并在 2025—2026 年间实现了从实验室到大规模集群部署的跨越。与此同时，以中国移动主导的 OISA 开放标准正在尝试构建国产 GPU 间的统一互联生态，目标类似 UALink 在中国的落地。以下是各家方案的具体分析。

**6. 华为昇腾：从 HCCS 到 UB 统一总线的演进**

华为在 AI 芯片领域的投入规模和垂直整合能力在国产厂商中独树一帜。昇腾系列从 910B 开始采用 Chiplet 封装，910C 在此基础上进一步将两颗 910B 芯片异构集成，通过自研的 **HCCS（Huawei Cache Coherence System）** 实现片间高速缓存一致性互联。HCCS 在 910 代际的单卡互联总带宽约为 90 GB/s，虽然与同期 NVLink 的 900 GB/s 存在数量级差距，但华为的策略并非在单卡带宽上硬拼——它把重心放在了**超大规模组网**上。

华为于 2025 年发布的 **CloudMatrix 384 超节点**是这一策略的集中体现：将 384 颗昇腾 910C NPU 和 192 颗鲲鹏 CPU 封装进单一逻辑计算域，通过下一代 **UB（Unified Bus，统一总线，中文名为"灵衢"）** 实现 TB/s 级的高带宽、低延迟互联。UB 架构融合了此前 HCCS 的缓存一致性能力和华为自研的交换芯片技术，目标是让超节点内的数百颗昇腾芯片像一个整体一样工作——这与 NVLink + NVSwitch 将 72 颗 GPU 组成 NVL72 机柜的思路异曲同工，但华为在规模上直接瞄准了更大的数量级。

软件栈方面，华为的 **CANN（Compute Architecture for Neural Networks）** 全栈自研，HCCL 通信库对标 NCCL。目前昇腾 910C 已大规模量产，CloudMatrix 384 已在多家运营商和互联网企业的数据中心落地。昇腾芯片在大模型训练领域的实际验证也在快速推进。2026 年 4 月，DeepSeek 发布的 V4 系列在全球范围内首次将**昇腾 950 系列（950PR 面向 Prefill & Recommendation 推理场景，950DT 面向 Decode & Training）** 引入万亿参数模型的训练流程——其中 V4-Flash 的后训练完全由昇腾 950 系列完成，尽管 V4 和 V4-Pro 的主体预训练仍在英伟达 H800/H100 上运行。两个月后的 2026 年 6 月，深圳河套学院联合哈工大（深圳）、深圳市大数据研究院及华为，使用至少 1,000 块昇腾 910C 芯片，独立完成了 **DeepSeek V4-Pro（1.6 万亿参数）** 的全参数后训练验证。这两个事件标志着国产算力栈正在从"推理可用"向"训练可用"过渡。华为的路线图显示将在 2026—2027 年推出 910D/910E 等后续芯片，UB 的带宽和规模将持续提升。

**7. 寒武纪：MLU-Link 的全自研路线**

寒武纪是国内最早一批进入 AI 芯片领域的公司，从终端 IP 起家，逐步扩展到云端训练/推理芯片。其互联协议 **MLU-Link** 完全自研，不依赖 PCIe Switch 或第三方交换芯片，目前已在思元 370、思元 590 以及最新旗舰思元 690 三代产品上量产。

MLU-Link 在思元 590 上的具体参数为 4 端口、16 lane、单 lane 50 Gbps，双向总带宽 **200 GB/s**。思元 590 采用 7nm 工艺和 MLUarch05 架构，INT8 算力 512 TOPS，FP16 算力约 256 TFLOPS。2026 年初量产的旗舰款**思元 690** 则是一次大的架构跨越——采用双 Die Chiplet 封装（2× 思元 590 规模），FP16 算力超过 **700 TFLOPS**，搭配 **MLU-Link 4.0** 实现更高带宽的卡间互联，目标直指 NVIDIA H100（2022 年发布，FP16 989 TFLOPS，NVLink 4 带宽 900 GB/s）。思元 690 在单卡算力上相当于 H100 的约 70%，在互联带宽方面仍有明显差距，但相较于思元 590 对比 A100 那代的代际差已在快速缩小。

软件栈为 **NeuWare** 全栈平台，包含 AI 加速算子库 CNNL 和集合通信库 **CNCL**。在商用落地上，寒武纪已从政府和科研机构向互联网大厂渗透——字节跳动是最大客户，2026 年订单规模达 250-300 亿元；阿里云正在测试思元 690 用于推理平台；腾讯则将思元 590 批量部署于特定推理场景。不过目前大规模训练集群的部署仍处于早期验证阶段，主力营收以推理业务为主。寒武纪的客户主要集中在阿里、字节等互联网大厂的推理业务，训练侧的大规模集群部署尚处于验证阶段。

**8. 壁仞科技：Blink 协议与光交换超节点的差异化路线**

壁仞科技是国产 GPU 厂商中少数同时布局电互联和光互联的企业。其首款通用 GPU **BR100** 于 2022 年发布，采用 7nm 工艺，在 INT8（2,048 TOPS）和 BF16（1,024 TFLOPS）指标上直接对标甚至超越了同年发布的 NVIDIA H100（FP16 约 989 TFLOPS），一度创下国产芯片算力纪录。但由于美国出口管制升级，BR100 的后续量产和生态推广受到显著影响。2025 年，其面向通用计算场景的 **BR104** 率先实现量产，首批交付百度、字节跳动等互联网大厂，单卡 FP32 算力约 256 TFLOPS，搭载 32 GB HBM2e 显存。

互联方面，壁仞的自研 **Blink** 电互联协议每通道 64 GB/s，支持 4 到 8 通道配置，单卡互联带宽约 256—512 GB/s。与 NVLink 4 的 900 GB/s 和 NVLink 5 的 1,800 GB/s 相比仍有明显差距，这也是壁仞押注光互连路线的原因之一——电互联在带宽和距离上的天花板，在国产制程受限的情况下更难突破。

壁仞最大的技术亮点在于**光互连超节点**方案。2025 年 7 月，壁仞联合仪电、曦智科技、中兴通讯发布了国内首个光互连光交换 GPU 超节点 **光跃 LightSphere X**，基于曦智科技的分布式光交换技术实现 128 卡互联。2026 年 7 月 WAIC 期间，壁仞进一步推出了下一代 **NPO（Near-Packaged Optics，近封装光学）光互连**方案，支持单个超节点 **1,024 卡** Scale-up 扩展，并配套发布了 **BLink 2.0** 协议，在延迟和带宽上较上一代有显著提升。

光互连的核心优势在于突破电信号在铜缆上的距离和密度限制，这对于超大规模组网意义重大——Google 的 TPU OCS 和英伟达 Rubin 平台的 CPO 路线都在向光互连方向演进。壁仞是国内唯一将光交换超节点作为核心产品路线推进的厂商，在国产算力集群的组网层面占据了差异化先发优势。

**9. 摩尔线程：MTLink 4.0 与 MUSA 的"CUDA 兼容"策略**

摩尔线程是国产 GPU 中在软件生态兼容性上走得最远的一家。其核心策略是 **MUSA（Moore Threads Unified System Architecture）**——一套 API 级兼容 CUDA 的全栈软件平台，能够通过 MUSIFY 工具自动转换 90% 以上的 CUDA 代码，大幅降低了从 NVIDIA 生态迁移的工程成本。

2025 年 12 月，摩尔线程在首届 MUSA 开发者大会上发布了全新 GPU 架构 **"花港"** 及首款基于该架构的云端 AI 加速芯片 **"华山"**。华山定位为 **AI 训推一体**芯片——即同一款芯片既能用于大模型训练也能用于推理部署，而非像以往那样将训练和推理拆分为两条产品线；同时支持超智融合场景（AI + 高性能计算的混合负载），覆盖 FP4 到 FP64 的全精度计算。华山支持超十万卡级 AI 工厂部署，其 Scale-up 系统兼容 **MTLink 4.0**，整卡互联带宽达到 **1,314 GB/s**，支持最多 **1,024 卡**互联。这一带宽指标在国产方案中处于领先水平，已接近 2024 年随 Blackwell B200 发布的 **NVLink 5（1,800 GB/s）**。

MTLink 同时兼容多种以太网协议，使得摩尔线程的芯片可以灵活接入现有的数据中心网络。MUSA 5.0 全栈软件同步升级，覆盖从通信库到算子库再到 PyTorch 框架适配的完整链路。华山芯片于 2026 年进入量产阶段，摩尔线程也是 OISA 生态的核心成员之一。

**10. 沐曦股份：MetaXLink 与全自研 MXMACA 软件栈**

沐曦股份成立于 2020 年，创始团队拥有丰富的 GPU 研发经验，产品线覆盖云端训练（曦云 C 系列）、推理（曦思 N 系列）和图形渲染（曦彩 G 系列）。其互联方案 **MetaXLink** 已迭代到第二代，在旗舰产品曦云 C600（2025 年 7 月于 WAIC 发布）上提供约 **900 GB/s** 的互联带宽，配备 7 个接口——这一规格与 2023 年发布的 H200 所搭载的 NVLink 4（900 GB/s）基本处于同一量级，但距 2024 年随 Blackwell B200 发布的 NVLink 5（1,800 GB/s）仍有约两年代际差距。

C500 采用 7nm 工艺和自研 XCORE 1.0 架构，C600 在此基础上升级至 XCORE 1.5 架构，新增 FP8 精度支持，搭载 144 GB HBM3e 显存，FP8 峰值算力约 1,000 TFLOPS。沐曦的 **MXMACA** 软件栈全栈自研，包含 MCCL 通信库，已适配 PyTorch 等主流框架。

在商用落地方面，沐曦是目前国产 GPU 中出货量较大的一家——截至 2025 年底累计出货超过 **25,000 颗**，已在万卡级别集群中完成部署验证。沐曦也是 OISA 生态的创始成员之一，深度参与国产互联标准的制定。

**11. OISA：中国的开放 Scale-up 互联标准**

如果说 UALink 是海外的"反 NVLink 联盟"，那么 **OISA（Omni-directional Intelligent Sensing Express Architecture，全向智感互联架构）** 就是中国移动主导的国产对标方案。

OISA 由中国移动联合北京市科委发起，昆仑芯、摩尔线程、沐曦、海光、天数智芯、登临科技、太初、南湖研究院等 8 家 GPU 企业以及腾讯、百度共同参与，目标与 UALink 高度一致：构建国产 GPU 卡间**开放、多供应商的 scale-up 互联标准**，避免每家厂商各自定义一套私有协议导致的生态碎片化。

OISA 1.0 于 2025 年 8 月发布，2.0 进一步将规格提升至单点 **896 GB/s** 带宽、交换容量 51.2 Tbps、支持最多 **1,024 张 AI 芯片**互联，时延缩短至数百纳秒，已具备原生内存语义。2025 年，昆仑芯基于 OISA 的 64 卡超节点在国际会议上完成首秀；OISA 商用产品预计 **2026 年 Q2** 开始出货，盛科通信等国产交换芯片厂商将在其中扮演核心角色。

与 UALink 相比，OISA 在带宽和标准化程度上仍有一定差距，但它在国内的意义更为特殊——它第一次将碎片化的国产 GPU 生态整合到一个共同的互联标准框架下，这对于尚处于追赶阶段的国产算力集群而言，其生态聚合价值可能比单纯的性能指标更重要。

---

## 三、替代交换方案全景：Scale-out 层

### 3.1 全球以太网交换芯片市场格局

根据QYResearch 2024年数据，全球以太网交换芯片市场规模约**48.6亿美元**，预计2031年达83.7亿美元（CAGR 7.26%）。市场高度集中：

| 厂商 | 2024年市场份额 | 代表产品 | 最新性能 | 策略定位 |
|---|---|---|---|---|
| **Broadcom** | 54.59% | Tomahawk 6, Jericho 4, Trident 5 | Tomahawk 6: 102.4Tbps, 3nm, 512×200G端口 | 商用芯片领导者，覆盖全场景 |
| **Marvell** | 12.95% | Teralynx 10, T100 | T100: 102.4Tbps, 3nm; Teralynx 10: 51.2T, 500ns延迟 | 低延迟+可编程，AI云场景 |
| **Cisco** | 9.60%（下降中） | Silicon One | 自研自用 | 从自研转向混合商用芯片策略 |
| **NVIDIA (Mellanox)** | 7-8% | Spectrum-4, Spectrum-X | 51.2Tbps | 自研ASIC，自用+外销，AI集群专用优化 |
| **Microchip** | ~2.5% | SparX-5 | 工业/企业级 | 中小企业市场 |
| **盛科通信 (国产)** | ~2.5% | Arctic系列 | 25.6Tbps, 800G端口 | 国产替代领军，对标博通 |

> **关键事实：** 英伟达在scale-out层使用的交换机（InfiniBand Quantum系列和以太网Spectrum系列）均采用**自研ASIC芯片**，并非采购博通或Marvell的商用芯片。英伟达在2019年以69亿美元收购Mellanox后，获得了完整的交换芯片设计能力。

### 3.2 国外交换机方案竞争

#### NVSwitch vs Tomahawk Ultra vs UALink ULS

| 方案 | 主导方 | 交换容量 | 最大规模 | 开放性 | 成熟度 |
|---|---|---|---|---|---|
| NVSwitch 5 | NVIDIA | 28.8 Tbps | 576 GPU（理论） | 完全封闭 | 已大规模量产 |
| Tomahawk Ultra (SUE) | Broadcom | 51.2 Tbps | 1,024加速器 | SUE开放规范 | 已量产出货 |
| UALink Switch (ULS) | UALink联盟 | 待定（多厂商） | 1,024加速器 | 完全开放标准 | 2027年量产 |
| NVLink Fusion | NVIDIA + 合作伙伴 | 依赖NVSwitch | 依赖NVSwitch | "半开放"（需搭配NVIDIA产品） | 生态构建中 |

#### Broadcom Tomahawk Ultra：以太网的"越界"挑战

博通在2025年推出的Tomahawk Ultra是迄今对NVSwitch最直接的挑战。它的核心理念是：**用以太网技术做Scale-up互联**，把原本属于NVLink+NVSwitch专用领域的GPU直连市场，纳入开放的以太网生态。博通为此定义了SUE（Scale-Up Ethernet）开放接口规范。

| 参数 | Broadcom Tomahawk Ultra | NVIDIA NVSwitch 5 |
|---|---|---|
| 交换容量 | **51.2 Tbps** | 28.8 Tbps |
| 制程 | TSMC 5nm | TSMC 4nm |
| 延迟（64B包） | **250 ns** | 未公开（推测更高） |
| 最大互联规模 | **1,024加速器** | 576 GPU（理论） |
| 协议标准 | Scale-Up Ethernet (SUE)，开放规范 | NVLink，专有封闭 |
| 网内计算 | 支持INC (In-Network Collective) | 支持SHARP |
| 兼容性 | 任意加速器 | 仅NVIDIA GPU |

#### NVLink Fusion：英伟达的"半开放"反击

面对UALink和SUE的围攻，英伟达在2025年5月推出了**NVLink Fusion**——一个策略性的"半开放"方案。核心思路：允许第三方加速器（非NVIDIA GPU）通过NVLink协议与NVIDIA的CPU、GPU和网络组件通信，但**必须搭配至少一款NVIDIA产品**。

已加入NVLink Fusion生态的合作伙伴：Marvell（英伟达投资20亿美元）、三星晶圆厂、ARM、联发科、Astera Labs。而AMD、Intel、博通——这三家英伟达的直接竞争对手全部缺席NVLink Fusion，转而主推UALink/SUE。

### 3.3 国产交换机方案

#### 盛科通信

国产商用交换芯片唯一上市公司，覆盖100G-25.6Tbps。Arctic系列25.6Tbps支持800G端口，对标博通Tomahawk 4/Marvell Teralynx 7级别。深度绑定新华三、锐捷网络等头部设备商。研发费用率42%，押注RISC-V CPU集成。在OISA生态中扮演交换芯片核心角色。

#### OISA Switch

| 方案 | 主导方 | 交换容量 | 最大规模 | 开放性 | 成熟度 |
|---|---|---|---|---|---|
| OISA Switch | 中国移动 + 国产厂商 | 51.2T交换容量 | 1,024卡 (2.0) | 国产开放标准 | 商用产品2026Q2 |

### 3.4 华为昇腾384超节点供应链

| 环节 | 核心供应商 | 地位与份额 |
|---|---|---|
| AI芯片 | 华为海思 (昇腾910B/C/950PR) | 全自研 |
| 高速背板连接器 | 华丰科技 (华为持股2.95%) | 昇腾910C背板连接器主供，市占率超60% |
| PCB | 深南电路、沪电股份 | 深南电路供应910C 60%封装基板 |
| ABF载板 | 兴森科技 | 核心供应商 |
| 交换机代工 | 共进股份 | 独家代工华为800G数据中心交换机 |
| 交换芯片 | 华为海思 (自研) | 自用，不外售 |
| 光模块 | 光迅科技、中际旭创 | 光迅为384超节点核心供应商 |
| 服务器整机 | 拓维信息(兆瀚)、华鲲振宇、神州数码 | 华鲲振宇占昇腾服务器出货50%+份额 |
| 液冷 | 申菱环境(一供)、川润股份、高澜股份 | 申菱联合开发"风液同源大液冷系统" |

---

## 四、总结与展望

**1. 英伟达方案的本质：** NVLink+NVSwitch是一套从SerDes物理层到协议层全栈自研的封闭体系，不基于RDMA或以太网。其昂贵的根源在于全栈封闭的锁定效应。GB300 NVL72整柜BOM约$399万，VR200（Vera Rubin）已达$780万，其中NVSwitch交换芯片+铜缆互联占机柜成本的10-12%。

**2. 替代协议竞争格局：** 目前形成了"三大开放阵营 + 两大云厂商自研"的格局。UALink联盟（100+成员）是挑战NVLink的最强开放力量；OISA是中国主导的国产开放标准；Google ICI和AWS NeuronLink是云厂商自研的封闭方案。AMD的Infinity Fabric在8 GPU规模上与NVLink 4处于同一量级。国产厂商中，摩尔线程MTLink（1314GB/s）和沐曦MetaXLink（~900GB/s）在带宽指标上已相当进取。

**3. CUDA兼容性：** 这是一个分层适配问题。在通信库（XCCL/NCCL）层面，各厂商的自研通信库正在快速追赶。摩尔线程的MUSA通过API级兼容提供了最低的迁移门槛。Google TPU是最彻底的"异类"——从编译链到框架全部自研，性能强但生态隔离度最高。

**4. 交换芯片格局：** 博通以54.6%的份额统治市场，Tomahawk 6（102.4Tbps/3nm）代表了业界巅峰。Marvell以低延迟+可编程差异化竞争。国产盛科通信已完成25.6Tbps芯片的小批量交付，但与博通仍有2-3代差距。NVIDIA Mellanox（7-8%份额）走自用优先+部分外销路线。

**5. 交换机竞争：** 形成了NVSwitch（封闭）、Tomahawk Ultra/SUE（以太网开放）、UALink ULS（标准开放）三阵营格局。博通的Tomahawk Ultra以51.2Tbps交换容量和250ns延迟在参数上已超越NVSwitch 5，ULAink ULS预计2027年量产。英伟达以NVLink Fusion（半开放）作为反击。

**6. 国产算力集群：** 已形成以华为昇腾为核心、OISA生态为开放补充的格局。供应链方面，在服务器组装、液冷、PCB、连接器、光模块等环节已有较强的国产替代能力，但GPU芯片本身和高端交换芯片仍有代差。超节点架构（384卡/1024卡）通过OISA等协议在规模上正在追赶，但在单卡算力密度和软件生态方面仍需时间沉淀。

**7. 核心趋势：** 交换芯片层面的竞争正从"NVIDIA一家独大"转向"三阵营对垒"——NVIDIA的NVSwitch（封闭）+ NVLink Fusion（半开放），博通的Tomahawk Ultra/SUE（以太网开放路线），和UALink联盟的ULS（全新开放标准）。谁是赢家取决于一个关键问题：UALink交换机能否在2027年赶上NVSwitch 6（Vera Rubin平台，3,600GB/s per GPU）的性能迭代速度。

---

## 参考文献

- [NVIDIA NVLink 和 NVLink 交换机 官方页面](https://www.nvidia.com/zh-cn/data-center/nvlink)
- [NVLink与纵向扩展网络：当800G以太网不够用时 - Introl](https://introl.com/zh/blog/nvlink-scale-up-networking-gpu-interconnect-infrastructure-2025)
- [NVIDIA GB200 NVL72官方技术博客](https://developer.nvidia.com/zh-cn/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference)
- [PCIe、NVLink、CXL……谁才是芯片互联界的"真·海王"? - 与非网](https://m.eefocus.com/article/1992270.html)
- [UALink与CXL 4.0：重塑GPU集群架构的开放标准 - Introl](https://introl.com/zh/blog/ualink-cxl-4-gpu-interconnect-memory-pooling-guide-2025)
- [GPU Interconnects and Rack-Scale Topology: The Complete Guide - Prompt20](https://blog.prompt20.com/posts/nvlink-and-rack-scale-topology/)
- [Confusion Grows With More Interconnect Options And Tradeoffs - SemiEngineering](https://semiengineering.com/confusion-grows-with-more-interconnect-options-and-tradeoffs/)
- [GB200 Hardware Architecture - Component Supply Chain & BOM - SemiAnalysis](https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component)
- [TPUv7: Google Takes a Swing at the King - SemiAnalysis](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the)
- [深度解析谷歌第八代TPU架构 - ICViews](https://www.icviews.cn/news/30613/7)
- [UALink vs NVLink vs InfiniBand comparison - MassAPI Blog](https://blog.massapi.com/categories/interconnects/)
- [UALink and the Battle for Rack-Scale GPU Interconnect - Bitsilica](https://bitsilica.com/ualink-and-the-battle-for-rack-scale-gpu-interconnect/)
- [OISA全向智感互联 官方网站](https://www.oisa.org.cn/)
- [Marvell Teralynx 数据中心交换机 官方页面](https://cn.marvell.com/products/data-center-switches.html)
- [Broadcom Tomahawk 6 官方页面](https://www.broadcom.com/products/ethernet-connectivity/switching/strataxgs/bcm78910-series)
- [盛科通信 官方网站](https://www.centec.com/)
- [AWS Trainium2 Trn2 Architecture - AWS Neuron Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/about-neuron/arch/neuron-hardware/trn2-arch.html)
- [沐曦 MetaXLink 开发者文档](https://developer.metax-tech.com/doc/101)
- [AI算力的隐形高速公路：PCIe芯片与互联技术全景解析 - 与非网](https://m.eefocus.com/article/2027960.html)
- [国产交换芯片大盘点 - 与非网](https://m.eefocus.com/article/2024370.html)
- [交换芯片：AI超节点驱动二次成长 - 华泰证券](https://finance.sina.com.cn/wm/2026-05-24/doc-inhyyaqq6875917.shtml)
- [PCIe、NVLink、CXL互联技术对比 - EET-China](https://www.eet-china.com/mp/a488295.html)
- [大模型基础设施工程04：互联与网络 - 掘金](https://juejin.cn/post/7632354324076757046)
- [CUDA围墙之下，国产GPU软件栈的绝地反击 - CSDN](https://blog.csdn.net/qq_42255328/article/details/160241953)
- [2025年全球以太网交换IC市场规模 - 格隆汇](https://m.gelonghui.com/p/3381807)

---

*声明：本文基于公开资料整理，数据来源包括英伟达官网、SemiAnalysis、QYResearch、摩根士丹利研报、各公司官方文档及技术媒体分析。所有价格数据均为估算值。技术参数截至2026年7月。本文尚在编写中，内容将持续完善与修订。*
