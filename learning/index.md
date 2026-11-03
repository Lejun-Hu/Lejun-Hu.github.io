---
layout: learning
title: 学习资料 & 计划
description: 分门别类的学习计划与精选学习资料
permalink: /learning/
---

# 📚 学习资料 & 计划

把学习当成一个项目来管理。这里记录我当前的学习路线和阶段性计划，也收藏一些自己读过、觉得值得反复看的资料——希望能给同样在 AI Infra 路上的朋友提供一份参考。

## 🗓 学习计划

| 计划 | 状态 | 周期 | 说明 |
|------|:----:|------|------|
| [NCCL 源码精读](/notes/2026-8-24-nccl-source-study/) | 进行中 | 2026.08 — | 从 ncclCommInitRank 入口出发，逐层读通初始化、Channel/连接建立与运行时通信路径，笔记持续更新中。 |
| CUDA Kernel 开发进阶 | 进行中 | 2026 — | 围绕 Tensor Core、WMMA 与性能调优，补齐从 CUDA C++ 到高吞吐 kernel 的工程细节。 |
| RDMA / verbs 深入 | 计划中 | — | 从 ibverbs API 到 RoCEv2 网络行为，理解 QP、MR、Completion 语义，以及 NCCL NET 插件与它们的对接方式。 |
| LLM 推理系统 | 计划中 | — | 以 vLLM 为切入点，梳理 PagedAttention、Continuous Batching 与分布式推理的工程实现。 |

## 📘 官方文档

| 资料 | 说明 | 标签 |
|------|------|------|
| [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) | User Guide 与 API 参考，理解语义与调优参数的第一手资料。 | NVIDIA |
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | CUDA 编程模型与运行时行为的权威参考。 | NVIDIA |
| [rdma-core](https://github.com/linux-rdma/rdma-core) | ibverbs / RDMA 用户态 API 的参考实现，学 RDMA 绕不开的源码。 | Linux |

## 📄 论文

| 资料 | 说明 | 标签 |
|------|------|------|
| [PAT — Parallel Aggregated Trees（arXiv:2506.20252）](https://arxiv.org/abs/2506.20252) | NCCL 2.23 新增的 AllGather / ReduceScatter 算法，Bruck 变体，对数级网络步数（笔记 2.6.4 有解读）。 | 集合通信 |
| [Megatron-LM（arXiv:1909.08053）](https://arxiv.org/abs/1909.08053) | 张量并行与数据并行组合训练大模型的开山之作。 | 分布式训练 |
| [PagedAttention / vLLM（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) | LLM 推理显存管理的分页设计，理解 KV Cache 工程化的起点。 | LLM 推理 |

## 🧩 源码仓库

| 资料 | 说明 | 标签 |
|------|------|------|
| [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | 本笔记的精读对象——集合通信库的参考实现。 | C++ |
| [NVIDIA/gdrcopy](https://github.com/NVIDIA/gdrcopy) | CPU 直读显存的用户态库，NCCL GDRCopy 路径的底层依赖。 | CUDA |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | ProcessGroupNCCL 是观察"框架如何使用 NCCL"的最佳样本。 | C++/Python |

## 🎓 课程 & 视频

| 资料 | 说明 | 标签 |
|------|------|------|
| [GPU MODE（原 CUDA MODE）](https://www.youtube.com/@GPUMODE) | 社区驱动的 GPU 编程系列讲座，覆盖 Triton、PTX 与性能调优实战。 | 视频 |

## 📚 书籍

| 资料 | 说明 | 标签 |
|------|------|------|
| Programming Massively Parallel Processors（4th Ed.） | GPU 体系结构与 CUDA 编程的经典教材，适合系统性打底。 | 书籍 |
| Computer Architecture: A Quantitative Approach | 理解互联、缓存一致性与存储墙的底层视角。 | 体系结构 |
