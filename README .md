# Lejun-Hu.github.io

> 个人技术主页 & 学习笔记仓库 —— 记录一个 AI Infra 工程师在集合通信、加速器架构与分布式系统上的学习与思考。

## 关于这个库

这是我的个人网站与博客仓库，基于 **Jekyll** + **GitHub Pages** 构建，访问地址为 [lejun-hu.github.io](https://lejun-hu.github.io)。

网站包含两部分内容：

- **个人主页**（`index.md`）：简历式自我介绍，涵盖工作经历、技术栈、教育背景等信息。
- **学习笔记**（`_notes/`）：长期更新的技术博客，是我在学习 AI 全栈过程中的笔记沉淀。目前主要围绕 **AI 计算系统** 这条主线展开。

我的技术方向是 **AI Infra（AI 基础设施）**，日常工作涉及集合通信库（NCCL / HCCL / XCCL 等）、RDMA 与在网计算、CUDA kernel 开发，以及大模型分布式训练/推理中的通信与数据流优化。这个仓库的笔记，正是我在这些领域学习与实战的记录。

## 已发布的笔记

| 笔记 | 主题 |
|------|------|
| [GPU、TPU 与 LPU：AI 时代计算架构的分野](https://lejun-hu.github.io/notes/gpu-tpu-lpu-architecture/) | 三种加速器的架构哲学与"通用性 vs 专用性"权衡 |
| [GPU 卡间互联协议深度调研](https://lejun-hu.github.io/notes/GPU_Interconnect_Research/) | NVLink/NVSwitch 及 scale-up / scale-out 替代方案全景 |
| [CUDA 软件栈全路径拆解与国产 AI 芯片生态兼容分析](https://lejun-hu.github.io/notes/cuda-ecosystem-compatibility/) | 从框架到硬件的 CUDA 编译执行链路，及国产芯片兼容策略 |
| [NCCL 源码学习笔记](https://lejun-hu.github.io/notes/nccl-source-study/) | NCCL 源码的初始化、执行路径与 GIN 机制（持续完善中） |

## 将来的学习计划

以下是计划中的学习主题（状态会随学习进度更新）：

### 集合通信与网络

- [ ] NCCL 源码精读：Ring/Tree 搜索算法、Channel 与 Connector 机制、协议选择（LL/LL128/Simple）
- [ ] NCCL 传输层深入：P2P/SHM/NET 三种 Transport 的连接建立与数据路径
- [ ] GIN（GPU Initiated Network）机制详解与国产芯片适配
- [ ] RDMA / InfiniBand / RoCEv2 协议栈与 Verbs 编程
- [ ] 在网计算（SHARP / CollNet）的原理与实现
- [ ] 国产集合通信库（HCCL 等）与 NCCL 的对比分析

### 加速器架构与 CUDA

- [ ] CUDA kernel 优化：Memory Coalescing、Shared Memory、Warp 调度
- [ ] Tensor Core 编程模型与矩阵乘优化
- [ ] 国产 AI 芯片架构（昇腾达芬奇 / 寒武纪 / 壁仞等）的软硬件协同

### 大模型分布式系统

- [ ] 张量并行 / 流水线并行 / 数据并行 / 专家并行的通信模式拆解
- [ ] 通信-计算重叠（overlap）与通算融合技术
- [ ] 大模型训练/推理中的通信瓶颈分析与优化

> 计划会随着学习和工作重心的变化不断调整，欢迎通过邮件交流指正。

## 快速开始

### 本地预览

```bash
# 安装依赖
bundle install

# 启动本地服务器
bundle exec jekyll serve

# 访问 http://localhost:4000
```

### 添加一篇新笔记

1. 在 `_notes/` 下新建 Markdown 文件，命名格式 `YYYY-M-D-标题.md`
2. 添加 front matter：

```markdown
---
title: "笔记标题"
permalink: /notes/your-slug/
date: 2026-08-24
category: "分类"
tags: ["标签1", "标签2"]
description: "笔记简介"
published: true
---
```

3. 如有配图，放入 `assets/images/notes/your-slug/` 目录，正文中用 `{{ '/assets/images/notes/your-slug/xxx.png' | relative_url }}` 引用

## 文件结构

```
.
├── _config.yml          # 网站配置
├── _data/               # 数据文件（教育背景、工作经历、技能等）
├── _includes/           # 可复用组件（head/header/footer）
├── _layouts/            # 页面布局（default/note）
├── _notes/              # 学习笔记（Markdown 集合）
├── notes/               # 笔记列表页
├── assets/
│   ├── css/             # 主样式表
│   ├── js/              # 交互脚本
│   └── images/          # 图片资源（含 notes/ 配图目录）
└── index.md             # 主页（个人简历）
```

## 技术栈

- [Jekyll](https://jekyllrb.com/) - 静态网站生成器
- [GitHub Pages](https://pages.github.com/) - 免费托管
- [SCSS](https://sass-lang.com/) - CSS 预处理器
- [Liquid](https://shopify.github.io/liquid/) - 模板语言

## 许可证

MIT License
