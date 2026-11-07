---
layout: learning
title: 学习资料 & 计划
description: 分门别类的学习计划与精选学习资料
permalink: /learning/
---

# 📚 学习资料 & 计划

把学习当成一个项目来管理。这里记录我当前的学习路线和阶段性计划，也收藏一些自己读过、觉得值得反复看的资料——希望能给同样在 AI Infra 路上的朋友提供一份参考。

## 🗓 学习计划

以下是计划中的学习主题（状态会随学习进度更新）：

### 集合通信与网络

| 主题 | 状态 |
|------|:----:|
| NCCL 源码精读1：Ring/Tree 搜索算法、Channel 与 Connector 机制、协议选择（LL/LL128/Simple） | 计划中 |
| NCCL 源码精读2：P2P/SHM/NET 三种 Transport 的连接建立与数据路径 | 计划中 |
| DeepEP 源码精读：对 NCCL 的依赖以及创新 | 计划中 |
| 集合通信算法与拓扑实现的基础量化分析，例如各种 allreduce 实现的开销以及 BUF（Bandwidth Utilization Factor） | 计划中 |
| GIN（GPU Initiated Network）机制详解与国产芯片适配 | 计划中 |
| RDMA / InfiniBand / RoCEv2 协议栈与 Verbs 编程 | 计划中 |
| 在网计算（SHARP / CollNet）的原理与实现 | 计划中 |
| 国产集合通信库（HCCL 等）与 NCCL 的对比分析 | 计划中 |

### 加速器架构与 CUDA

| 主题 | 状态 |
|------|:----:|
| CUDA kernel 优化：Memory Coalescing、Shared Memory、Warp 调度 | 计划中 |
| Tensor Core 编程模型与矩阵乘优化 | 计划中 |
| 国产 AI 芯片架构（昇腾达芬奇 / 寒武纪 / 壁仞等）的软硬件协同 | 计划中 |

### 大模型分布式系统

| 主题 | 状态 |
|------|:----:|
| 张量并行 / 流水线并行 / 数据并行 / 专家并行的通信模式拆解 | 计划中 |
| 通信-计算重叠（overlap）与通算融合技术 | 计划中 |
| 大模型训练/推理中的通信瓶颈分析与优化 | 计划中 |

> 计划会随着学习和工作重心的变化不断调整，欢迎通过邮件交流指正。

## 🖥 算力卡 & 选型

| 资料 | 说明 | 标签 |
|------|------|------|
| [MirrorFrog — AI 算力卡百科](https://mirrorfrog.com/) | 收录 233 款 AI 芯片的规格对比与排行，覆盖 NVIDIA / AMD / 昇腾 / 寒武纪 等，附路线图与 TCO 计算器。 | 选型 |

## 📖 在线书

| 资料 | 说明 | 标签 |
|------|------|------|
| [How to Scale Your Model（JAX Scaling Book）](https://jax-ml.github.io/scaling-book/) | Google DeepMind 出品的大规模 LLM 训练系统教程：Roofline 分析、TPU 原理、sharding 与各类并行策略、LLaMA 3 训练/服务成本估算，含 GPU 专题章。 | 分布式训练 |

## 🌐 网络与通信

| 资料 | 说明 | 标签 |
|------|------|------|
| [Everything You Wanted to Know About RDMA（SNIA, 2025）](https://www.snia.org/educational-library/everything-you-wanted-know-about-rdma-were-too-proud-ask-2025) | SNIA 教育库的 RDMA 体系化入门长文：从 RDMA 的基本概念、RoCE / InfiniBand / iWARP 三种传输的差异，到 NVMe over Fabrics 与大规模存储网络，适合作为 RDMA 协议的导读资料。 | RDMA |
