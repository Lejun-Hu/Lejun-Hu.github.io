---
title: "【未完成】CUDA软件栈全路径拆解与国产AI芯片生态兼容深度分析"
permalink: /notes/cuda-ecosystem-compatibility/
date: 2026-08-20
category: "AI芯片软件栈"
tags: ["CUDA", "GPU软件栈", "PTX", "算子编译器", "DSA", "GPGPU", "华为CANN", "昇腾", "Google TPU", "摩尔线程MUSA", "国产AI芯片", "PyTorch", "cuBLAS", "NCCL", "Triton"]
description: "从PyTorch框架调用到GPU硬件执行，逐层拆解CUDA软件栈的完整编译与执行链路；分析GPGPU与DSA两种架构路线在兼容CUDA生态时的工作量差异与技术权衡；以华为昇腾CANN（DSA）、Google TPU（DSA自主生态）和摩尔线程MUSA（GPGPU）为案例，详解各厂商的软件栈实现与开发者迁移路径。"
published: true
---

> 如果你是一名 AI 开发者，日常工作可能是这样：在 PyTorch 里写几行 Python 代码，定义一个模型，调用 `model.to("cuda")`，然后 `loss.backward()` 等训练完成。整个过程大多数时候不需要关心 GPU 到底在做什么——直到某一天你尝试把模型部署到一张国产算力卡上。
>
> 此时你会发现一个令人不安的事实：**CUDA 并非一个"API 层"那么简单，而是一个从硬件指令集延伸到上层框架的完整软件帝国。** 那些看似理所当然的 `cudaMalloc`、`<<<>>>` 语法糖、自动梯度同步，背后是经过了近二十年打磨的编译链路、运行时系统和数千个闭源优化的算子库。要想让一张全新的硬件卡无缝承接这个生态，远不是在 API 层"改几个函数名字"就能解决的。
>
> 本文试图回答以下几个问题：**从 PyTorch 框架下发一个计算任务，到 GPU 设备真实执行，中间到底发生了什么？** CUDA 软件栈的每一层分别承担什么职责，它们的编译依赖和运行时行为是怎样的？如果我们要自研一套 AI 硬件，选择 GPGPU 架构（如摩尔线程、壁仞）还是 DSA 架构（如华为昇腾、Google TPU），对于兼容现有 CUDA 生态分别意味着什么？通过对 NVIDIA 完整软件栈的逐层拆解，以及对华为昇腾（DSA）、Google TPU（DSA 自建生态）和摩尔线程（GPGPU）三个案例的横向对比，本文试图为"国产 AI 芯片如何融入现有开发者生态"这一核心问题提供一个结构化的分析框架。
>
> 本文的目标读者是对 CUDA 和 GPU 编程有一定了解的技术人员，但并不要求是编译器或底层驱动专家。文中涉及的每个概念都会给出清晰的定义，所有代码示例均基于真实 API 和实际工程实践。

---

## 一、NVIDIA CUDA 软件栈的完整逐层拆解

在分析任何"如何兼容 CUDA"的问题之前，必须首先准确理解"CUDA"本身涵盖了什么。一个常见的误解是把 CUDA 等同于 `cudaMalloc`、`cudaMemcpy` 这几个 Runtime API 函数——实际上这些只是最表层的一小部分。从 PyTorch 的框架调用到 GPU 物理硬件的执行，CUDA 构建了一条多层级的编译与执行链路，每一层都有自己独立的职责、产物和闭源组件。

下面沿着 **"框架层 → 算子库层 → Runtime层 → 编译层 → 硬件层"** 的顺序，逐一拆解每一个环节中发生的过程、涉及的工具和库，以及每一步的输入与输出。

### 1.1 起点：PyTorch 框架中的一行代码

以一段最简单的 PyTorch 代码为例：

```python
import torch
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = torch.matmul(x, y)
```

在这三行 Python 背后，PyTorch 内部发生了什么？首先，`torch.matmul` 并不是一个"普通的 Python 函数"——它是 PyTorch 的 ATen 张量运算库中上千个算子之一。当 Python 解释器执行到 `torch.matmul(x, y)` 时，PyTorch 的 dispatcher（调度器）会根据输入张量的设备类型（`device="cuda"`）和数据类型，动态路由到对应的后端实现。以默认 eager 模式下 CUDA 设备上的矩阵乘法为例，dispatcher 通常会命中 **cuBLAS** 库的 `cublasGemmEx` 函数；但在 `torch.compile` 图编译模式下，Inductor 后端可能改用 CUTLASS 或 Triton 生成的 kernel，不同的矩阵尺寸和精度组合也可能触发不同的底层实现路径。

再举一个更贴近日常训练的例子——开篇提到的 `loss.backward()` 中的"自动梯度同步"。当你在 PyTorch 中调用 `loss.backward()` 时，框架并非简单地"在 GPU 上把梯度算完就完了"。实际的执行序列是：PyTorch 的 autograd 引擎遍历计算图，为每个算子调用其反向传播 kernel——这些 kernel 在 GPU 上以**异步**方式提交到 CUDA stream 中执行。也就是说，`backward()` 返回时，GPU 可能还没算完任何梯度。真正保证"梯度已经就绪"的是 PyTorch 在读取梯度数据（如 `param.grad`）之前自动插入的隐式同步点——底层调用的是 `cudaDeviceSynchronize` 或 stream 级的同步 API。这个同步开销是 GPU 编程中常见的"隐形成本"之一，在后续 §1.6 和 §1.8 讨论异步执行与 CUDA Graphs 优化时还会再次涉及。

> **阅读提示**：上面这段出现了几个可能不太熟悉的名词——eager 模式、图编译、Inductor、CUTLASS、Triton。不用担心，本文后续章节会逐一深入介绍它们。这里先给一个速览表，方便你快速建立印象：
>
> | 名词 | 简要含义 | 本文详细位置 |
> |------|---------|------------|
> | **eager 模式** | PyTorch 的默认执行方式：每写一行 Python，框架就立即调用底层算子执行，不做全局优化。类似于"解释执行"。 | §1.1 本节 |
> | **图编译模式** | 先将整个计算过程记录为一幅计算图，再对整图做优化（如算子融合），最后一次性生成并执行高效代码。 | §1.8 |
> | **cuBLAS** | NVIDIA 提供的闭源 GPU 线性代数加速库，是 PyTorch 在 CUDA 设备上执行矩阵乘法的默认后端，内部包含针对不同 GPU 架构高度手工调优的 kernel。 | §1.1 本节 |
> | **Inductor** | PyTorch 2.0 `torch.compile` 的默认后端编译器，负责将计算图转化为 Triton/C++ kernel。 | §1.8 |
> | **CUTLASS** | NVIDIA 开源的 CUDA C++ 模板库，提供了高度可定制的 GEMM/Conv 等算子模板，适合作为自定义 kernel 的性能参照。 | §1.1 表格 |
> | **Triton** | OpenAI 开源的 GPU kernel 编程语言与编译器，用 tile 级抽象替代 CUDA 的线程级编程，也是 Inductor 默认的 kernel 生成后端。（注：NVIDIA 另有一个同名产品 Triton Inference Server，是推理部署工具，与本文讨论的 Triton 语言/编译器无关。） | §1.7, §3.2 |

这里的第一个关键概念是 **算子（Operator）**。在深度学习框架中，"算子"指代对一个或多个张量执行的特定数学运算——矩阵乘法（matmul）、卷积（conv2d）、逐元素加法（add）、归一化（layernorm）等均属于算子。PyTorch 中有超过 2,000 个 ATen 算子，每个算子需要针对不同设备的多个后端（CPU / CUDA / MPS / XPU 等）提供独立的实现[^1]。对于 CUDA 后端，绝大多数算子的实现并不是手写的 CUDA C++ kernel——PyTorch 团队会优先调用 NVIDIA 提供的**闭源高性能算子库**，因为经过二十年代代优化的闭源库性能远超通用 kernel 实现。

[^1]: **MPS**（Metal Performance Shaders）是 Apple 为 M 系列芯片提供的 GPU 加速后端，在 PyTorch 中通过 `device="mps"` 调用，使 Mac 也能进行 GPU 训练/推理。**XPU** 是 Intel GPU 的后端标识，通过 Intel Extension for PyTorch（IPEX）支持 Intel 数据中心 GPU 和集显。

在继续介绍具体算子库之前，有必要先厘清"开源"与"闭源"这两个概念。在软件工程中，**开源**意味着源代码公开，任何人都可以阅读、修改和重新分发；**闭源**则只提供编译后的二进制文件（如 `.so` 动态库），源码不公开。闭源库之所以仍然能被 PyTorch 调用，是因为 NVIDIA 公开了这些库的**头文件**（`.h`）——头文件中声明了所有函数的名称、参数类型和返回值类型，PyTorch 在编译时只需 `#include` 这些头文件，链接时再连接到 NVIDIA 提供的闭源 `.so` 文件即可。这类似于你知道一台自动售货机的按钮标签和投币口位置（头文件/API），也知道它能吐出什么饮料（函数签名），但无法打开机器看到内部齿轮和电路是如何运作的（源码）。那么能否通过**反汇编**（将二进制机器码逆向为汇编代码）来窥探闭源库的内部实现呢？技术上可以——`cuBLAS`、`cuDNN` 的 `.so` 文件可以通过 `objdump`、`Ghidra` 等工具反汇编为 SASS 指令。但这样做的实际价值非常有限：一方面，cuBLAS 内部包含数千个针对不同矩阵尺寸和 GPU 架构高度手工调优的 kernel 变体，反汇编出的 SASS 代码量极其庞大且缺乏符号信息，几乎无法还原为可理解的算法逻辑；另一方面，即便还原了部分实现，这些代码也深度绑定了 NVIDIA GPU 的特定硬件特性（如 Tensor Core 的 wmma 指令编码、Shared Memory 的 bank 布局），无法直接移植到其他硬件上。因此，对于国产 AI 芯片厂商而言，"兼容 cuBLAS"的策略并不是去逆向工程它的内部实现，而是在 API 层做**干净室重实现**（clean-room reimplementation）——对照 NVIDIA 公开的头文件，用自研 kernel 实现完全相同的数学语义。这个概念在 §2.1 中会有更详细的讨论。具体来说：

| 算子类型 | 调用的 NVIDIA 库 | 库是否开源 | 说明 |
|---------|-----------------|-----------|------|
| 矩阵乘法（matmul, linear） | cuBLAS | **闭源** | 稠密线性代数运算的 GPU 加速实现，经过多代架构调优 |
| 卷积、池化、激活函数 | cuDNN | **闭源** | 深度神经网络原语的极致优化实现，支持自动算法选择（heuristics）|
| 快速傅里叶变换 | cuFFT | **闭源** | FFT 的 GPU 加速实现 |
| 稀疏矩阵运算 | cuSPARSE | **闭源** | 稀疏 BLAS 操作 |
| 随机数生成 | cuRAND | **闭源** | GPU 侧随机数生成 |
| 集合通信（AllReduce等） | NCCL | **开源** | 多 GPU 之间的 AllReduce/Broadcast/AllGather 等通信原语 |
| 通用矩阵乘法模板 | CUTLASS | **开源** | CUDA C++ 模板抽象，用于实现高性能 GEMM，可作性能参照但非框架默认路径 |

这里的 **cuBLAS** 和 **cuDNN** 尤其需要着重理解。cuBLAS 是 BLAS（Basic Linear Algebra Subprograms，基础线性代数子程序）标准的 GPU 实现，提供 `cublasSgemm`（单精度矩阵乘）、`cublasGemmEx`（混合精度矩阵乘）等数百个 API。它是几乎所有深度学习框架在 CUDA 设备上执行矩阵运算的底层依赖。cuBLAS 本身是闭源动态库（`libcublas.so`），其内部包含针对不同 GPU 架构、不同矩阵尺寸高度手工调优的 kernel 实现——这些 kernel 的具体代码从未公开。

cuDNN 的情况类似：它提供了 `cudnnConvolutionForward`、`cudnnBatchNormalizationForwardInference` 等深度神经网络原语，闭源且内部包含针对 Tensor Core 等硬件特性的精细优化。它通过内置的 heuristics 引擎在运行时根据输入尺寸、卷积核大小、可用显存等参数自动选择最优 kernel 实现——这种自动调优能力是 cuDNN 的核心壁垒之一，要在一个自有硬件上达到同样的自动调优覆盖率，工作量极其巨大。

NCCL 和 CUTLASS 是上述表格中仅有的两个开源库。NCCL 采用 BSD 许可（Berkeley Software Distribution，一种非常宽松的开源协议，允许商用和闭源再发布，仅要求保留版权声明）。它提供了 `ncclAllReduce`、`ncclBroadcast`、`ncclAllGather` 等集合通信原语，支持 ring、tree、collnet 等多种算法变体。虽然 NCCL 源码公开，但其传输层深度绑定了 NVLink（GPU 间直连协议）、NVSwitch（交换芯片，将多颗 GPU 编织成全互联域）和 InfiniBand（跨节点 RDMA 网络协议）等 NVIDIA 自有互联硬件，要在不同硬件上移植必须重写整个传输层。

与 cuBLAS/cuDNN 这类"单卡计算库"不同，NCCL 是**跨卡通信库**——它解决的是多 GPU 之间如何高效交换数据的问题。在一个典型的分布式训练步骤中，计算库和通信库是协作关系：cuBLAS 负责在前向/反向传播中完成每张卡上的矩阵乘法（计算），NCCL 负责在反向传播后将各卡的梯度做 AllReduce 求和（通信），使得每张卡都拿到全局平均梯度。两者的交互发生在 PyTorch 的分布式框架层——`torch.distributed` 会在反向传播完成后自动触发 AllReduce，而 `torch.distributed` 内部正是通过 NCCL 的 API 来完成数据传输的。关于 NCCL 的编译方式以及在 CUDA 编译链路中的位置，将在 §1.9 全链路总览之后专门讨论。

### 1.2 自定义 Kernel：当框架算子不够用时

以上是 PyTorch 框架自动调用的标准算子路径。但在实际工程中，内置算子并不能覆盖所有需求——新的激活函数（如 SwiGLU）、融合算子（将 Scale + Add + Activation 合并为一个 kernel）、自定义量化方案等场景需要开发者手写 kernel。在 CUDA 生态中，这通过 CUDA C++ 实现：

```cpp
// vector_add.cu —— CUDA C++ 自定义 kernel 示例
#include <cuda_runtime.h>

// __global__ 声明该函数在 GPU 上执行，由 CPU 端调用
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // threadIdx.x: 当前线程在其所属 block 内的索引
    // blockIdx.x: 当前 block 在 grid 中的索引
    // blockDim.x: 每个 block 中的线程数
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1024;
    float *d_a, *d_b, *d_c;  // 设备端（GPU）指针

    // Runtime API：分配 GPU 显存
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // ...初始化数据（略）...

    // <<<grid, block>>> 是 CUDA 的 kernel 启动语法糖
    // grid: 256个block, block: 每个block 256个线程
    vectorAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    // 等待 GPU 完成所有任务
    cudaDeviceSynchronize();

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

这段代码中包含了 CUDA 编程模型中的几个核心抽象：

- **`__global__` 函数**：声明在 GPU 设备端执行的函数，由 CPU 主机端通过 `<<< >>>` 语法调用。
- **线程层次**：CUDA 使用 `threadIdx`（线程在 block 内的索引）、`blockIdx`（block 在 grid 中的索引）和 `blockDim`（每个 block 的线程数）来组织大规模并行计算。GPU 以 **warp（32 线程）** 为最小调度单位，同一 warp 内所有线程锁步执行相同指令。这是整个 CUDA 编程模型的基础，也是后续所有编译优化的起点——编译器需要知道哪些线程共享数据、哪些操作需要同步。
- **`cudaMalloc` / `cudaMemcpy` / `cudaFree`**：CUDA Runtime API，管理 GPU 显存的分配、主机-设备数据传输和释放。这些是 `libcudart.so` 动态库提供的高层封装。回顾开篇的问题——`cudaMalloc` 并不是一个"魔法函数"，它内部实际上是通过 Driver API 的 `cuMemAlloc` 完成的，而 `cuMemAlloc` 再进一步通过内核驱动向 GPU 硬件请求显存地址空间。这一层层调用的详细路径将在 §1.6 中展开。
- **`<<<grid, block>>>` 语法糖**：CUDA 特有的 kernel 启动语法，它会在编译时被 nvcc 展开为实际的 `cudaLaunchKernel` Runtime API 调用。具体展开过程是：nvcc 前端在解析 `.cu` 文件时，将 `kernel<<<grid, block>>>(args)` 替换为一系列设置 kernel 参数和 grid/block 维度的 Runtime API 调用，最终调用 `cudaLaunchKernel`。这个编译期替换过程将在 §1.3 的 nvcc 编译链路中详细展示。
- **`cudaDeviceSynchronize`**：阻塞主机端直到 GPU 所有已提交任务完成，是最基本的同步原语。这也是开篇提到的"自动梯度同步"的底层依赖之一——`loss.backward()` 中的梯度计算在 GPU 上是异步的，PyTorch 会在读取梯度前自动插入同步点，确保 GPU 计算完成后再将梯度数据传回 CPU。

### 1.3 编译链路第一站：nvcc 前端 —— CUDA C++ → PTX（Parallel Thread Execution，并行线程执行）

上面的 `.cu` 源文件如何变成 GPU 可执行的代码？答案是通过 **nvcc**（NVIDIA CUDA Compiler Driver）进行离线编译。nvcc 并非一个单一的编译器，而是一个 **编译器驱动程序**——它在内部协调多个子工具完成不同的编译阶段。

```bash
# 编译为 PTX 中间表示（文本格式，可读）
nvcc -ptx vector_add.cu -o vector_add.ptx

# 编译为 CUBIN 二进制（针对特定 GPU 架构）
nvcc -cubin vector_add.cu -arch=sm_80 -o vector_add.cubin

# 标准方式：生成包含多个架构代码的宿主可执行文件
nvcc vector_add.cu -o vector_add -arch=sm_80
```

nvcc 的第一步是将 `.cu` 文件中的**设备代码**（`__global__`、`__device__` 标记的函数）与**主机代码**（普通的 `main()` 函数等）分离。在此过程中，nvcc 还会处理一个重要的语法糖展开——`<<<grid, block>>>`。回顾 §1.2 中提到的 `vectorAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n)`，nvcc 会将其转换为一系列 Driver API 调用：首先根据 `<<<>>>` 中的参数计算 grid 和 block 维度，然后调用 `cudaConfigureCall` 设置执行配置，最后将 `vectorAdd` 的实际参数（`d_a, d_b, d_c, n`）打包并通过 `cudaSetupArgument` 逐个压入参数栈，最终通过 `cudaLaunch` 触发 kernel 执行。这一整套展开逻辑对开发者完全透明——你写的是简洁的三个尖括号，编译后变成了一组复杂的 Runtime/Driver API 调用。这便是开篇所谓"语法糖"的实质：nvcc 在编译期将友好的语法形式机械地替换为底层 API 调用序列，开发者无需手动管理这些繁琐的步骤。

分离后，设备代码被送往 GPU 编译器前端。这里需要说明：nvcc 的 GPU 前端是 NVIDIA 自研的闭源编译器前端，基于 EDG（Edison Design Group）C++ 前端引擎构建，并在此基础上扩展了对 CUDA C++ 特有语法（`__global__`、`__device__`、`<<<>>>` 等）的解析支持。值得注意的是，LLVM/Clang 也提供了另一条 CUDA 编译路径（`clang++ -x cuda`），能够直接将 CUDA 代码编译为 NVPTX（LLVM 后端的 PTX 目标），但本文以主流的 nvcc 路径为准。设备代码经过词法分析、语法分析、语义分析和优化后，生成 **PTX（Parallel Thread Execution，并行线程执行）**——这是一种文本格式的虚拟 ISA（指令集架构）汇编代码。

以下是对应的 `vectorAdd` kernel 生成的 PTX 代码（已简化以便阅读）：

```asm
// vector_add.ptx —— 简化后的 PTX 输出
.version 8.0                       // PTX ISA 版本
.target sm_80                       // 目标虚拟架构（sm_80 = Ampere 级）
.address_size 64

.visible .entry _Z9vectorAddPfS_S_i(   // kernel 入口函数（C++ name mangling）
    .param .u64 _Z9vectorAddPfS_S_i_param_0,  // 参数 a (float*)
    .param .u64 _Z9vectorAddPfS_S_i_param_1,  // 参数 b (float*)
    .param .u64 _Z9vectorAddPfS_S_i_param_2,  // 参数 c (float*)
    .param .u32 _Z9vectorAddPfS_S_i_param_3   // 参数 n (int)
) {
    .reg .s32  %r<5>;     // 声明 5 个 32 位有符号整数寄存器
    .reg .f32  %f<4>;     // 声明 4 个 32 位浮点寄存器
    .reg .b64  %rd<12>;   // 声明 12 个 64 位地址寄存器

    // 计算线程索引: index = threadIdx.x + blockIdx.x * blockDim.x
    mov.u32   %r1, %tid.x;        // r1 = threadIdx.x
    mov.u32   %r2, %ctaid.x;      // r2 = blockIdx.x
    mov.u32   %r3, %ntid.x;       // r3 = blockDim.x
    mad.lo.s32 %r4, %r2, %r3, %r1; // r4 = r2 * r3 + r1 (乘加指令)

    // 边界检查: if (index >= n) return
    ld.param.u32 %r5, [_Z9vectorAddPfS_S_i_param_3];  // 加载参数 n
    setp.ge.s32  %p1, %r4, %r5;  // p1 = (r4 >= r5)，设置条件谓词（谓词是一种布尔寄存器，只有"真/假"两个值，用来控制后续指令是否生效）
    @%p1 bra     L_END;           // 如果 p1 为真，跳转到 L_END

    // 加载 a[index] 和 b[index]
    cvta.to.global.u64 %rd2, %rd1;  // 将地址转换为全局内存地址空间
    mul.wide.s32 %rd4, %r4, 4;      // r4 * sizeof(float)
    add.s64       %rd5, %rd2, %rd4; // 基地址 + 偏移
    ld.global.f32 %f1, [%rd5];     // 从全局内存加载 a[index]

    // 同理加载 b[index]（省略详细指令）
    ld.global.f32 %f2, [%rd8];     // 加载 b[index]

    // 执行加法并写回
    add.f32  %f3, %f1, %f2;        // f3 = f1 + f2
    st.global.f32 [%rd11], %f3;   // 存储结果到 c[index]

L_END:
    ret;
}
```

PTX 有两个核心特征值得特别关注。其一，**它是虚拟 ISA 而非真实硬件指令**：PTX 中使用 `%r1`、`%f1` 这样的"虚拟寄存器"——它不指定物理寄存器编号，不包含指令的二进制编码，也不绑定任何具体的 SM 微架构。其二，**它提供前向兼容性**：一次编译生成的 PTX 可以被更新版本的 GPU 驱动在运行时 JIT（Just-in-Time，即时编译）为任何更新代 GPU 的机器码。这意味着为 Ampere 架构（sm_80）编译的程序，可以将 PTX 嵌入到二进制中，在 Hopper（sm_90）甚至未来未发布的 GPU 上运行——因为驱动可以在加载时实时编译 PTX。

PTX 在整个 CUDA 生态中的角色，十分类似于 **LLVM IR 在 CPU 编译器中的角色**：两者都是"虚拟 ISA 的中间表示层"，允许编译器前端输出一个与具体硬件解耦的中间形式，再由后端编译器将其映射到具体目标架构的物理指令。这种分阶段编译的设计，使得跨代硬件兼容成为可能。

### 1.4 编译链路第二站：ptxas 后端 —— PTX → SASS → CUBIN

PTX 生成完成后，**ptxas**（PTX Assembler）工具将 PTX 汇编为 **SASS（Shader Assembly）**——NVIDIA GPU 的真实机器指令，并封装为 **CUBIN**（CUDA Binary，CUDA 二进制）——一个 ELF 格式的二进制文件。

```bash
# 显式调用 ptxas 将 .ptx 编译为 .cubin
ptxas -arch=sm_80 vector_add.ptx -o vector_add.cubin

# nvcc 在生成二进制时自动调用 ptxas
nvcc -cubin vector_add.cu -arch=sm_80 -o vector_add.cubin
```

使用 `cuobjdump -sass` 可以反汇编 CUBIN 查看其中的 SASS 指令：

```asm
// cuobjdump -sass vector_add.cubin 的输出（简化）
code for sm_80                     // Ampere 架构专用
Function : _Z9vectorAddPfS_S_i

/* 地址     十六进制编码         SASS 指令                   注释                    */
/* 0000 */  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]   // 栈指针初始化
/* 0010 */  S2R        R0, SR_TID.X                    // R0 = threadIdx.x (读特殊寄存器)
/* 0020 */  S2R        R3, SR_CTAID.X                  // R3 = blockIdx.x
/* 0030 */  IMAD       R0, R3, c[0x0][0x0], R0        // R0 = blockIdx*blockDim+threadIdx
/* 0040 */  ISETP.GE.AND P0, R0, c[0x0][0x168], PT    // if index >= n, 设置谓词 P0
/* 0050 */  @P0 EXIT                                    // 若 P0 为真，本线程退出
/* 0060 */  SHF.L.S32.HIGH R2, RZ, 0x2, R0            // 计算字节偏移 (index * 4)
/* 0070 */  IADD3       R4, R2, c[0x0][0x160], RZ     // 计算 a[index] 的全局地址
/* 0080 */  LDG.E       R5, [R4]                        // 从全局内存加载 a[index]
/* 0090 */  IADD3       R6, R2, c[0x0][0x168], RZ     // 计算 b[index] 的全局地址
/* 00a0 */  LDG.E       R7, [R6]                        // 加载 b[index]
/* 00b0 */  FADD        R8, R5, R7                      // R8 = a[index] + b[index]
/* 00c0 */  IADD3       R9, R2, c[0x0][0x170], RZ     // 计算 c[index] 的全局地址
/* 00d0 */  STG.E       [R9], R8                        // 存储结果到 c[index]
/* 00e0 */  EXIT                                         // 线程退出
```

将 SASS 与前面的 PTX 逐行对比，可以清晰地观察到两者之间的关键差异：

- **物理寄存器替换虚拟寄存器**：PTX 中的 `%r1`、`%f1` 全部被替换为物理寄存器 `R0`-`R9`。这步称为**寄存器分配（Register Allocation）**——编译器决定了每个变量在硬件寄存器堆中的具体位置。
- **指令选择与编码**：PTX 的 `mov.u32` 映射为 `S2R`（读特殊寄存器），`mad.lo.s32` 映射为硬件指令 `IMAD`。每条 SASS 指令都有确定的二进制编码（左侧十六进制列）。
- **架构特异性**：SASS 中出现了多项 Ampere 架构特有的硬件特性，在其他代 GPU(如 Hopper 的 sm_90)上编码格式和可用指令都不同。具体表现为：

  1. **常量内存寻址** `c[0x0][0x28]`：kernel 参数和编译期常量存放在 GPU 的 constant memory 中，通过 `c[bank][offset]` 方式访问；相比之下，更老的 Fermi/Kepler 架构使用单独的 `cbank` 寄存器寻址，地址空间更小。

  2. **专用特殊寄存器** `SR_TID.X`：`SR` = Special Register，GPU 硬件为每个线程内置了一组只读寄存器，直接暴露线程索引、warp ID 等硬件状态；而在 CPU 上这些信息需要通过系统调用或读取特定 MSR[Model-Specific Register，CPU 内部的一组专用寄存器，用于控制硬件行为或读取底层状态，如当前 CPU 核编号、温度传感器等，需要特权指令 `RDMSR` 才能访问]才能获取。

  3. **谓词化执行** `@P0 EXIT`：`@P0` 表示"当谓词寄存器 P0 为真时，本条指令才生效"。它的巧妙之处在于**消除了分支跳转**：传统的 `bra L_END` 是"条件满足就跳转"，处理器需要猜测跳转方向，一旦猜错，流水线中已取入的指令全部作废重来——这就是"冲刷流水线"。而谓词化执行的做法是，不管条件是否满足，`EXIT` 这条指令都照常流入流水线，只是当 P0 为假时，指令的效果被静默丢弃(不写回状态)。指令流始终是顺序的，流水线永远不会被打断。CPU 中类似的概念是条件移动指令 `cmov`(Conditional Move)，x86 指令集中的条件赋值指令。例如 `cmovne eax, ebx` 表示"若上一条比较结果不相等，则将 ebx 的值复制到 eax，否则什么都不做"。它同样避免了分支跳转——用一条顺序执行的指令替代了"if-else + jmp"的组合——因此不会触发分支预测失败导致的流水线冲刷。但 `cmov` 仅适用于简单的值赋值场景，而 GPU 的谓词化执行覆盖面更广，几乎所有指令都可以带谓词。

  **不同 SM 版本之间的 SASS 二进制不兼容。**
- **SASS 才是 GPU 真正执行的内容**：PTX 只是中间表示，最终落到硬件上的是 SASS。因此在做性能分析时（如通过 Nsight Compute 检查 kernel 是否走了 Tensor Core 路径），观察 SASS 才是真实反映 GPU 行为的"真相层"。

### 1.5 FATBIN：多架构代码的统一容器

一个实际部署的 CUDA 程序面临的难题是：用户的 GPU 可能是 Ampere 架构，也可能是 Hopper 架构，程序应该怎么打包？把所有架构的 SASS 都编译一份打包进去，同时保留一份 PTX 作为兜底——这就是 **FATBIN（Fat Binary，胖二进制）** 的设计初衷。

```bash
# 为多种架构生成 FATBIN（多份 SASS + 一份 PTX 兜底）
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_52,code=compute_52 \
     vector_add.cu -o vector_add
```

这个命令生成了一个包含三种代码的 FATBIN：

1. **sm_80 CUBIN**：针对 Ampere 架构（如 A100）预编译的 SASS 机器码
2. **sm_90 CUBIN**：针对 Hopper 架构（如 H100）预编译的 SASS 机器码
3. **compute_52 PTX**：一份中间表示兜底代码，当用户的 GPU 架构超出了预编译 CUBIN 覆盖范围时（例如未来发布的 Rubin 架构），驱动可以在运行时 JIT 编译此 PTX

使用 `cuobjdump` 可以查看宿主可执行文件中嵌入的 FATBIN 内容：

```bash
cuobjdump vector_add
# 输出：
# Fatbin elf code:
# ====================
# arch = sm_80
# code version = [1,7]
# host = linux
# code for sm_80
#         Function : _Z9vectorAddPfS_S_i
#         ...
#
# Fatbin elf code:
# ====================
# arch = sm_90
# code for sm_90
#         Function : _Z9vectorAddPfS_S_i
#         ...
#
# Fatbin ptx code:
# ====================
# arch = compute_52
# ptx for compute_52
#         ...
```

FATBIN 在运行时的选择逻辑如下：如果存在与当前 GPU 架构**完全匹配**的 CUBIN，直接加载执行；如果存在**二进制兼容**的同主版本 CUBIN，加载执行；如果仅有 PTX，通过驱动内置的 JIT 编译器将 PTX 编译为目标架构的 SASS 后执行；如果什么都没有，报错 `"no kernel image is available for execution on the device"`。

这个机制的意义在于它解决了 GPU 硬件快速迭代带来的"前向兼容"问题——开发者在 2022 年为 A100（sm_80）编译的程序，可以在 2024 年发布的 H100（sm_90）上直接运行，只要编译时保留了 PTX 兜底代码。JIT 编译过程由 CUDA 驱动自动完成，对应用程序透明，且编译后的 SASS 会被缓存到磁盘（compute cache），不会在每次运行时重复编译。

那么既然 JIT 可以兜底，为什么还需要提前编译 CUBIN 呢？原因有二：其一，**首次启动延迟**——JIT 编译 PTX → SASS 需要几百毫秒到数秒（取决于 kernel 复杂度和数量），而加载预编译的 CUBIN 是即时的。对于需要快速启动的应用场景（如 Serverless GPU 推理），这个延迟不可接受。其二，**离线编译优化更充分**——离线 `ptxas` 不受启动时间约束，可以执行更激进的优化遍数（如更彻底的寄存器分配、指令调度重排），生成的 SASS 性能通常优于驱动 JIT 编译器快速生成的版本。因此，FATBIN 的最佳实践是"能预编译的架构尽量预编译，未来未知架构用 PTX 兜底"——兼顾了当下性能与未来兼容。

### 1.6 运行时路径：Runtime API 与 Driver API 的分工

所有离线编译的产物（FATBIN/CUBIN/PTX）最终需要在程序运行时被加载到 GPU 上执行。CUDA 提供两套 API 来完成这一过程，它们之间存在明确的分层关系。

**CUDA Runtime API**（`libcudart.so`）是高层封装，提供了 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel` 等开发者最常用的接口。它自动管理 CUDA context（设备的"进程"抽象）和 module（设备的"动态库"抽象）的创建与加载，使 GPU 编程的代码量大幅减少。以下是使用纯 Runtime API 完成向量加法的代码——注意这里不需要手动加载 PTX 文件，因为 kernel 在编译时就已经嵌入到了可执行文件中（FATBIN 机制），且 kernel 启动直接使用 `<<<>>>` 语法糖：

```cpp
// 使用 CUDA Runtime API —— 99% CUDA 开发者的日常写法
// 对比 Driver API：无需 cuInit/cuCtxCreate/cuModuleLoad/cuLaunchKernel
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1024;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;  // 主机端数组

    // Step 1: 分配设备内存——简洁的函数名，无需 CUdeviceptr 类型
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Step 2: 将数据从主机拷贝到设备
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Step 3: 启动 kernel——<<<>>> 语法糖，nvcc 自动展开为 cudaLaunchKernel
    vectorAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    // Step 4: 同步并取回结果
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 5: 清理——无需手动销毁 context
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

**CUDA Driver API**（`libcuda.so`）是底层接口，Runtime API 内部实际上就是通过调用 Driver API 来完成所有实际操作的。Driver API 暴露了 `cuInit`、`cuCtxCreate`、`cuModuleLoadData`、`cuLaunchKernel` 等更细粒度的控制接口，允许应用程序手动管理 context 和 module 的生命周期。以下代码展示了通过 Driver API 手动加载并执行一个 kernel 的完整流程：

```cpp
// 使用 CUDA Driver API 加载和执行 kernel
#include <cuda.h>  // Driver API 头文件，非 cuda_runtime.h

int main() {
    // Step 1: 初始化 Driver API（必须在任何其他 Driver API 调用之前）
    cuInit(0);

    // Step 2: 获取设备句柄并显式创建上下文（Runtime API 隐式完成此步）
    CUdevice   device;
    CUcontext  context;
    cuDeviceGet(&device, 0);       // 获取设备 0
    cuCtxCreate(&context, 0, device);  // 创建上下文并绑定到当前线程

    // Step 3: 加载模块——可以加载 .ptx 文件、.cubin 文件或内存中的 PTX 字符串
    CUmodule module;
    cuModuleLoad(&module, "vectorAdd.ptx");
    // 或从内存加载：cuModuleLoadData(&module, ptx_string);

    // Step 4: 按名称查找 kernel 函数
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "vectorAdd");

    // Step 5: 分配设备内存（Driver API 使用 CUdeviceptr 类型表示设备地址）
    CUdeviceptr d_a, d_b, d_c;
    size_t n = 1024;
    cuMemAlloc(&d_a, n * sizeof(float));
    cuMemAlloc(&d_b, n * sizeof(float));
    cuMemAlloc(&d_c, n * sizeof(float));

    // ... 通过 cuMemcpyHtoD 将初始化数据拷贝到设备 ...

    // Step 6: 设置 kernel 参数并启动
    void* args[] = { &d_a, &d_b, &d_c, &n };
    cuLaunchKernel(kernel,
        (n+255)/256, 1, 1,   // grid 维度 (x, y, z)
        256,          1, 1,   // block 维度 (x, y, z)
        0, NULL,              // 共享内存大小、流句柄
        args, NULL);          // kernel 参数、额外选项

    cuCtxSynchronize();       // 等待 GPU 完成

    // Step 7: 清理资源
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
}
```

Driver API 与 Runtime API 之间存在一个重要的隐蔽依赖：**Dark API**。Driver API 库 `libcuda.so` 中暴露了一个未公开的函数 `cuGetExportTable`，它接受一个 GUID（全局唯一标识符）参数，返回一张包含数十个未公开函数指针的表。这些"Dark API"被 CUDA Runtime（`libcudart`）和 NVIDIA 自有库（`cuBLAS`、`cuDNN` 等）内部大量使用。由于这些 API 没有文档、没有函数签名说明、没有参数类型定义，任何尝试重新实现 `libcuda.so` 的工作都只能通过**逆向工程**来逐函数推测其行为。在后续的兼容方案分析中，Dark API 将反复作为一个核心难题出现。

### 1.7 运行时编译：NVRTC 与 nvJitLink 的角色

并非所有 CUDA kernel 都是在编译时静态确定的。在 AI 编译器和框架中，经常需要在运行时动态生成 CUDA 代码然后即时编译执行——例如 JAX 和 Triton 编译器将高层算子表达转换为 CUDA C++ 代码字符串后，需要即时编译为可执行 kernel。这一需求由 **NVRTC（CUDA Runtime Compilation）** 库提供。

```cpp
// NVRTC：在运行时编译 CUDA C++ 字符串并加载执行
#include <nvrtc.h>
#include <cuda.h>

const char* src = R"(
    extern "C" __global__ void saxpy(float a, float* x, float* y, float* out, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) out[tid] = a * x[tid] + y[tid];
    }
)";

int main() {
    // Step 1: 创建 NVRTC 程序对象
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src, "saxpy.cu", 0, NULL, NULL);

    // Step 2: 指定目标架构并编译为 PTX
    const char* opts[] = { "--gpu-architecture=compute_80" };
    nvrtcResult res = nvrtcCompileProgram(prog, 1, opts);

    // Step 3: 获取编译后的 PTX 字符串
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Step 4: 通过 Driver API 加载 PTX 并执行
    CUmodule module;
    cuModuleLoadData(&module, ptx);     // 从内存字符串直接加载 PTX
    // ...后续与 1.6 节的 Driver API 流程相同...

    nvrtcDestroyProgram(&prog);
    delete[] ptx;
    return 0;
}
```

NVRTC 以库的形式（`libnvrtc.so`）工作，不需要启动外部 `nvcc` 进程，避免了进程创建、磁盘 I/O 等开销。输入端接受 CUDA C++ 字符串和编译选项，输出端提供 PTX 字符串（也可以直接输出 CUBIN）。

你可能会问：为什么要用字符串的形式写 kernel，然后运行时编译，而不是像 §1.2 那样正常写一个 `.cu` 文件离线编译？关键区别在于**编译时间点**。离线编译（nvcc）适用于 kernel 代码在构建时已经确定的场景；而 NVRTC 面向的是 kernel 代码**在运行时才知道长什么样**的场景。一个典型例子是 Triton 编译器：它根据运行时输入张量的形状（如 `(128, 256) @ (256, 512)` vs `(4096, 4096) @ (4096, 1024)`）和数据类型，动态生成不同的 CUDA C++ kernel 字符串——小矩阵用小 tile、少共享内存，大矩阵用大 tile、多级分块。这些优化决策无法在编译期预先做完，只能在运行时根据实际输入来"写代码"并即时编译。因此，NVRTC 的"字符串输入 → 即时编译"模式并不是一种退而求其次的用法，而是 JIT 编译器链中的标准工作方式。

从 CUDA 12.0 开始，NVRTC 还能生成 **LTO-IR（Link-Time Optimization Intermediate Representation，链接时优化中间表示）** 以配合 **nvJitLink** 运行时链接器，实现跨 kernel 模块的链接时优化——这一机制对 Triton 等 AI kernel 编译器至关重要，因为它允许在运行时将多个独立编译的 kernel 模块链接为一个高度优化的统一 kernel。

### 1.8 CUDA Graph 与算子融合：超越逐 Kernel 执行的优化

前面的讨论都建立在"逐 kernel 启动执行"的 eager 模式之上。但在现代 GPU 计算中，逐个 kernel 启动的开销常常超过实际计算本身——尤其是对于大量小规模 kernel 组成的模型（例如含有大量逐元素操作的 Transformer），CPU 端每次 kernel launch 的提交延迟、GPU 侧的线程块调度开销和中间张量的 HBM 读写构成了巨大的隐性浪费。这一矛盾催生了两个层面的优化技术：**算子融合** 和 **CUDA Graphs**。两者解决的是不同层面的问题，可以叠加使用。

#### 1.8.1 算子融合（Operator Fusion）—— 减少 HBM 读写

算子融合由框架编译器在离线或 JIT 阶段完成。以 `Conv2d → BatchNorm → ReLU` 这一常见组合为例，在 eager 模式下，这三个操作会产生三次 HBM 读写（每个操作的结果都要写回 HBM 再由下一个操作读取）。而融合后，编译器将其合并为一个 kernel——输入数据从 HBM 读取一次，在片上完成卷积、归一化和激活函数的全部计算，最终结果写回一次。在现代 GPU 上，HBM 带宽是主要瓶颈（H100 为 3.35 TB/s），而非计算吞吐（H100 的 BF16 算力达 1,979 TFLOPS），这种融合能带来数倍的性能提升。

#### 1.8.2 CUDA Graphs —— 消除 CPU-GPU 交互延迟

CUDA Graphs 解决的是完全不同的开销：kernel 提交延迟（launch overhead）。在 eager 模式下，每启动一个 kernel，都需要 CPU 准备参数、通过 Runtime/Driver API 提交给 GPU、GPU 调度器接受并排队。对于一个包含数百个小 kernel 的模型，每个 kernel 可能只跑几微秒，但提交开销就要 5-10 微秒——**提交开销比计算本身还大**。

CUDA Graphs 的思路是：把这些 kernel 之间的"CPU→GPU 提交往返"全部砍掉。它的使用分为三个步骤——**捕获、实例化、重放**：

```cpp
// CUDA Graph 的基本用法：捕获 → 实例化 → 重放

// Step 1: 开始捕获。stream 是 CUDA 的命令队列，所有提交到这个 stream 的操作
//         都会被"录下来"而非立即执行。（之前的示例中 <<<>>> 未指定 stream，
//         使用的是默认流 stream 0，这里为了图捕获需要显式传入 stream 参数）
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Step 2: 提交操作——这些操作不会被立即执行，而是记录到图结构中
kernel_A<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(..., stream);
kernel_B<<<grid, block, 0, stream>>>(...);

cudaStreamEndCapture(stream, &graph);          // 结束捕获，获得图句柄

// Step 3: 实例化——对图做预分析：提前分配临时内存、消除冗余同步点、
//         确定最优执行顺序。这个步骤只做一次，结果可以重复使用。
cudaGraphInstantiate(&instance, graph, 0);

// Step 4: 重放——之后每次调用，GPU 自己按图一口气跑完所有 kernel，
//         CPU 不需要逐个启动，只需要在最后等结果。
cudaGraphLaunch(instance, stream);
```

这里需要解释几个关键概念：

- **Stream（流）**：CUDA 中的命令队列。同一个 stream 里的操作严格按顺序执行，不同 stream 之间的操作可以并行。之前的示例代码中 `<<<>>>` 没有指定 stream 参数，实际走的是默认流（stream 0）。CUDA Graphs 依赖 stream 来"监听"你提交了哪些操作，所以这里必须显式传入。

- **实例化（Instantiate）**：把捕获到的图做一次预分析和优化——类似于编译器对代码做优化 pass。具体包括：提前分配好所有临时内存、确定 kernel 的执行顺序、消除不必要的同步点。实例化只需做一次，结果可以被反复重放。

- **重放（Replay / Launch）**：之后每次调用 `cudaGraphLaunch`，GPU 直接按优化好的计划一口气执行整张图。kernel A 算完自动触发 kernel B，不需要 CPU 介入说"该启动 B 了"。

#### 1.8.3 算子融合 vs CUDA Graphs：一张表说清区别

两者解决的是不同层面的瓶颈，可以叠加使用：

| | 算子融合 | CUDA Graphs |
|---|---|---|
| **优化目标** | 减少 HBM 读写次数 | 减少 CPU-GPU 交互延迟 |
| **怎么做** | 多个算子合并成一个 kernel | 多个 kernel 串成一张图，GPU 自主驱动 |
| **kernel 数量** | 减少（N→1） | 不变（N 个还是 N 个） |
| **CPU 参与** | 不变（每次仍需 CPU 启动） | 大幅减少（CPU 只启动一次图） |
| **类比** | 三个包裹打包成一个箱子 | 一次性把所有包裹放到快递柜，快递员自己按顺序取 |

两者叠加的典型场景是 PyTorch 2.0 的 `torch.compile`：Inductor 后端先用算子融合把可以合并的操作合成大 kernel，再把剩余无法融合的操作通过 CUDA Graphs 串起来一键重放。对于国产芯片的兼容方案而言，最关键的战场因此不在于逐 kernel 的 API 映射，而在于能否在 Inductor 后端上提供自有硬件的代码生成能力。

#### 1.8.4 CUDA Graphs 在 DSA 与 GPGPU 上的差异

CUDA Graphs 有一个反直觉的特征：在 **DSA（Domain-Specific Architecture，领域专用架构）** 中这一机制是天然契合的——DSA 编译器天生就是将整个计算图静态编译为执行计划，图捕获对其而言几乎不需要额外工作。而在 **GPGPU（General-Purpose GPU，通用 GPU）** 的动态 warp 调度模型下，图捕获反而需要额外的运行时机制来拦截和记录 kernel 提交、管理静态内存地址的复用。

> 如果你对 DSA 和 GPGPU 这两个概念还不熟悉，没关系，请接着往下看——本文第二章会专门从兼容 CUDA 生态的角度，详细对比这两种架构路线的本质差异。

### 1.9 全链路总览

将以上八个节的内容串联起来，可以得到 CUDA 软件栈从框架到硬件的完整执行链路。下面的表格按照**一次计算任务实际发生的先后顺序**排列。需要注意：实际执行有两条路径——**标准算子路径**（如 `torch.matmul`，框架直接调用现成的 cuBLAS，跳过编译阶段）和**自定义 kernel 路径**（如 `torch.compile` 或手写 `.cu` 文件，需要先经过图优化和编译）。下表的 ①→⑨ 覆盖了自定义 kernel 路径的完整流程；标准算子路径则直接从 ① 跳到 ⑧。

| 阶段 | 输入 | 输出 | 关键工具/库 | 开源状态 |
|------|------|------|-----------|---------|
| ① 框架层调度 | Python 框架代码 | 对算子库（cuBLAS/cuDNN/NCCL）的调用，或对自定义 kernel 的引用 | PyTorch dispatcher | 开源 |
| ② 图优化/算子融合 | 计算图 (FX Graph) | 融合后的图 + 生成的 kernel 代码 | `torch.compile` + **Inductor**（生成 Triton/C++ kernel）、TensorRT | 部分闭源 |
| ③ kernel 代码生成 | Inductor 生成的 Triton/C++ 或手写 `.cu` | CUDA C++ 源码（字符串或文件） | Inductor、Triton、手写 CUDA | 开源（手写）/闭源（工具） |
| ④ 前端编译 | CUDA C++ 源码 | PTX 文本/字符串 | nvcc GPU 前端、NVRTC | 闭源 |
| ⑤ 后端汇编 | PTX | SASS / CUBIN (ELF) | ptxas (离线) 或 驱动 JIT 编译器 | 闭源 |
| ⑥ FATBIN 打包 | 多个 CUBIN + PTX | FATBIN (嵌入宿主可执行文件或 `.so`) | nvcc --fatbin | 闭源 |
| ⑦ Runtime 加载 | FATBIN / CUBIN / PTX | 已加载到 GPU 上下文的 CUmodule | libcudart, libcuda (+ Dark API) | 闭源 |
| ⑧ 算子库执行 | 矩阵乘/卷积/通信调用，或已加载的自定义 kernel | GPU 计算结果 | cuBLAS, cuDNN, cuFFT (闭源); NCCL, CUTLASS (开源) | 部分闭源 |
| ⑨ GPU 硬件执行 | SASS 指令流 | 计算结果 | SM, Tensor Core, HBM, NVLink | 硬件 |

需要特别说明 **Inductor 的位置**：Inductor 是 PyTorch 2.0 `torch.compile` 的默认后端，工作在 **② 图优化阶段**。它负责两件事——(1) 对 FX 计算图做算子融合、调度等优化；(2) 生成底层 kernel 代码（默认生成 Triton kernel，也可生成 C++/CUTLASS）。Inductor 本身不直接编译 kernel，它生成的代码会交给 ③ 的 Triton/NVRTC 或 ④ 的编译工具链来完成编译。因此 Inductor 是连接"高层计算图"和"底层编译"的桥梁——它的输出是 ③ 的输入。

### 1.10 NCCL 在编译链路中的位置

前面的讨论主要聚焦于"单卡计算"的编译链路，但实际的大模型训练/推理场景中，多卡通信同样占据关键地位。这里集中说明 NCCL 的编译方式以及它与计算算子库的协作关系。关于通算融合的讨论将在 §1.11 中展开。

#### 1.10.1 NCCL 是如何编译的？和 nvcc 是什么关系？

NCCL 的编译同时使用了两种编译器，最终产物是单一的 `libnccl.so` 动态库：

- **CPU 端代码**（如 NCCL 的拓扑探测、通信算法选择、channel 管理）：用标准 C/C++ 编写，由 **GCC/Clang** 编译为 x86-64 机器码（`.o` 文件）。
- **GPU 端 kernel 代码**（如 NCCL 的 Reduce、Copy、AllReduce 等 GPU kernel）：用 CUDA C++ 编写，由 **nvcc** 编译为 CUBIN/PTX，以 FATBIN 格式嵌入。

两者通过链接器（ld）合并为一个 `libnccl.so` 文件。这里需要澄清一个技术细节：一个 `.so` 文件完全可以同时包含 CPU 机器码和 GPU 代码（CUBIN/PTX）。具体做法是将 CUBIN/PTX 以二进制数据的形式嵌入 `.so` 的数据段（`.rodata` 段），运行时 CUDA 驱动通过 `cuModuleLoadData` 从内存中的二进制数据直接加载，无需从磁盘读取单独的 `.cubin` 文件。这并非特殊机制——任何二进制数据都可以通过 `objcopy` 或链接器脚本嵌入 `.so`，关键只是加载方要知道如何解析嵌入的数据。

PyTorch 在运行时通过动态链接加载 `libnccl.so`。加载后，CPU 端代码直接在 Host 上执行，GPU 端 kernel 则由 CUDA 驱动从 `.so` 中提取 CUBIN/PTX 并加载到 GPU 上执行——与 §1.5-1.6 描述的 FATBIN 加载机制完全一致。cuBLAS 和 cuDNN 的编译方式与此相同，区别仅在于它们是闭源的，用户看不到源码和编译过程。以 cuBLAS 为例，PyTorch 调用 `cublasGemmEx` → `libcublas.so` 内部会调用 `libcudart.so`（Runtime API）来加载其嵌入的 GPU kernel、创建内部 CUDA stream 和 event → Runtime API 再调用 `libcuda.so`（Driver API）→ 最终进入 GPU 内核驱动。

#### 1.10.2 NCCL 与计算算子库（cuBLAS/cuDNN）是如何协作的？

两者的关系是"独立运作、框架协调"——NCCL 和 cuBLAS/cuDNN 之间没有直接的 API 调用关系，它们都是由 PyTorch 的分布式框架层（`torch.distributed`）统一调度和编排的独立组件。

PyTorch 框架对两者的调用可以分为两类场景：

**场景一：框架自动触发的通信。** 最常见的是数据并行（DDP, Distributed Data Parallel）中的梯度同步。AllReduce 是反向传播的必要步骤——每张卡必须拿到所有卡的梯度平均值，优化器才能正确更新参数。关键优化在于**执行时序**：PyTorch 在构建 DDP 模型时，会在计算图中为每个参数注册一个 hook——当某个参数的局部梯度计算完成时，框架立即对该参数的梯度启动 NCCL AllReduce，而不等待所有层的梯度都算完。这意味着：当参数 N 的梯度正在网络上做 AllReduce 通信时，参数 N-1 的反向计算可以同时在 GPU 上执行。两者在不同的 CUDA stream 上运行，互不阻塞，从而将通信延迟"隐藏"在计算时间之下（§1.11 会详细讨论这种重叠机制）。AllReduce 本身仍然是必须完成的——它只是被提前启动、与后续计算并行执行，以减少整体训练时间。这种自动插入对用户透明——你只需写标准的训练循环，PyTorch 在背后完成了所有 NCCL 调用。

**场景二：用户显式调用的通信。** 在张量并行（TP, Tensor Parallelism）或 MoE（Mixture of Experts）架构中，单次前向/反向传播本身就依赖集合通信。例如，TP 中的 `all_reduce` 用于在每层计算后将各 GPU 的部分结果求和；MoE 中的 `all_to_all` 用于将 token 路由到对应的 expert GPU。这些 NCCL 调用不是梯度同步的附属品，而是计算图的核心组成部分——如果 NCCL 不可用，这些模型的单次前向传播都无法完成。用户通常通过 `torch.distributed.all_reduce`、`torch.distributed.all_gather` 等 API 显式调用，PyTorch 内部将这些调用转发给 NCCL。

**两者的调度关系。** 无论是自动触发还是显式调用，最终都是由 PyTorch 的 distributed backend 模块将请求发给 NCCL。NCCL 和 cuBLAS 在 PyTorch 进程内部共享同一个 CUDA context，但使用不同的 CUDA stream——cuBLAS 的计算在计算 stream 上执行，NCCL 的通信在通信 stream 上执行。两个 stream 之间通过 CUDA event 进行同步：PyTorch 在需要确保通信完成后再继续计算的节点插入 `cudaStreamWaitEvent`，保证数据依赖关系不被破坏。这种多 stream 架构是实现通信-计算重叠的基础。

### 1.11 通算融合：通信算子与计算算子的融合优化

#### 1.11.1 通信算子是什么？

**通信算子**是指将集合通信操作（如 AllReduce、AllGather）本身抽象为一个"算子"，与计算算子（如 matmul、conv2d）在同一个计算图中统一表示和调度。例如，在 PyTorch 的计算图中，`all_reduce` 和 `matmul` 都是图中的节点，编译器可以对它们一视同仁地做优化。

**通算融合（Communication-Computation Overlap / Fusion）** 则是在此基础上的进一步优化，主要有两种形式：

- **通信与计算重叠（Overlap）**：将通信和计算放入不同的 CUDA stream，使它们在 GPU 上并行执行。例如，在反向传播中，当第 N 层的梯度还在做 AllReduce 通信时，第 N-1 层的反向计算可以同时进行。这种重叠能将通信延迟"隐藏"在计算时间之下，是分布式训练中最重要的性能优化手段之一。

- **通信与计算融合（Fusion）**：将通信操作和计算操作合并为同一个 GPU kernel。NCCL 的 AllReduce 可以分解为 ReduceScatter + AllGather 两个阶段。在某些场景中，后续计算不需要等待 AllReduce 完全结束——ReduceScatter 产出的分片规约结果已经足够让部分计算先开始，AllGather 可以在计算进行中或计算完成后再执行。

  一个实际例子是 Ring AllReduce 算法中的局部规约与计算融合。在 Ring ReduceScatter 过程中，每张 GPU 将数据切为多个 chunk，每个 chunk 依次沿环形链路传递：收到上一个邻居的 chunk → 与本地对应 chunk 做逐元素加法（规约）→ 发送给下一个邻居。这个"与本地 chunk 做逐元素加法"本质上是一个计算 kernel——如果此时 GPU 正在同时执行另一个计算任务（比如处理上一层反向传播的激活梯度），理论上可以将网络接收、规约加法和激活梯度计算融合进同一个 kernel，从而避免中间结果写回 HBM 再读出的开销。这种 kernel 级融合要求编译器能同时理解通信数据的到达时序和计算的数据依赖，目前主要是学术界和工业界实验室的前沿研究方向。

  另一个更贴近工程实践的"拆分-交错"例子是 FSDP（Fully Sharded Data Parallel）中的优化器步骤。FSDP 中每张卡只持有参数分片，反向传播后通过 ReduceScatter 获得梯度分片，优化器更新直接在该分片上执行，**不执行 AllGather**。AllGather 被推迟到下一轮前向传播中——当某层的前向计算需要完整参数时，才对该层的参数分片做 AllGather 拼回完整权重。这样 AllGather 的通信开销与下一层的前向计算重叠，且 AllGather 的结果直接喂给计算 kernel，中间不需要额外的 HBM 往返。

  这些技术的共同点是：编译器需要理解通信算子的数据依赖关系，识别哪些计算只需要部分通信结果即可开始，以及哪些通信结果可以被延迟消费。目前这一领域仍处于活跃研究阶段，是 AI 系统性能优化的前沿方向。

推动这些技术落地的关键基础设施之一是 **MLIR（Multi-Level Intermediate Representation，多层中间表示）**。传统的编译器（如 LLVM）只有一层抽象——太高了丢失硬件细节，太低了丢失领域语义。而 AI 编译器的优化链很长（计算图 → kernel → PTX → SASS），每一层向下转换时信息都会损失一部分，导致跨层优化（如通信与计算融合）难以实现。MLIR 的核心思想是**允许编译器同时维护多层 IR**，每层用不同的"Dialect"（方言）描述不同粒度的操作——例如 Linalg Dialect 描述矩阵运算、GPU Dialect 描述线程级并行、LLVM Dialect 对接底层机器码。编译器可以在不同 Dialect 之间做优化，而不需要把所有信息压扁到一层。这意味着通信-计算融合这样的跨层优化可以在合适的 Dialect 层上被表达和实现，而不必等待整个编译栈的重写。对于国产 AI 芯片的自研编译器而言，MLIR 正在成为事实上的标准基础设施——厂商只需实现自己硬件的 Dialect 即可，无需从零构建整个编译器。

---

## 二、从通用视角看兼容难度：DSA 与 GPGPU 的本质分歧

第一章以 CUDA 为主线，从 PyTorch 框架层一路拆解到了 GPU 硬件的 SASS 指令执行，覆盖了单卡计算、多卡通信以及编译优化的完整链路。但在实际产业中，"兼容 CUDA"并不是简单地把这套链路复制一份——不同的芯片架构，面临的障碍性质完全不同。要理解为什么，需要先厘清两种截然不同的硬件设计路线：**GPGPU** 和 **DSA**。

### 什么是 GPGPU？什么是 DSA？

**GPGPU（General-Purpose Graphics Processing Unit，通用图形处理器）** 顾名思义，是从 GPU 演变而来的通用并行计算架构。它的核心设计理念是 **SIMT（Single Instruction, Multiple Threads，单指令多线程）**——硬件将大量线程组织为 warp（NVIDIA 为 32 线程），由硬件 warp scheduler 在运行时动态调度到计算单元上执行。开发者看到的是一个"无限线程"的抽象：你写一个 kernel 函数，描述单个线程的行为，硬件自动将这份逻辑复制到成千上万个线程上并行运行。这种设计的最大优势是**通用性**——同一个硬件可以高效运行图形渲染、科学计算、AI 训练/推理等截然不同的负载。代价是硬件必须维护复杂的运行时调度逻辑（线程分配、缓存替换、分支发散处理等），这部分功耗和面积不直接贡献计算吞吐。NVIDIA GPU（包括 A100、H100）、AMD GPU（MI300X 等）、以及国产的摩尔线程 MUSA 和壁仞 BR100 都属于 GPGPU 路线。

**DSA（Domain-Specific Architecture，领域专用架构）** 则走向了完全相反的方向。这一概念由 2017 年图灵奖得主 John Hennessy 和 David Patterson 在《A New Golden Age for Computer Architecture》演讲中正式提出，核心理念是：放弃"一个架构通吃所有应用"的传统 CPU 设计思路，针对特定领域（如 AI 推理/训练）定制专用硬件，以换取更高的能效和确定性性能。DSA 的典型特征是将**复杂度从运行时转移到编译器**——没有动态线程调度器，没有运行时缓存替换策略，所有调度决策（数据放在哪块 SRAM、指令在哪个时钟周期发射、跨芯片通信在哪个拍发生）全部由编译器在离线阶段静态规划完成。硬件在运行时只负责按既定计划执行，几乎没有动态决策开销。华为昇腾的达芬奇架构（Ascend 910/950）、Google TPU（v1-v7）的脉动阵列、Groq LPU 的流水线结构，都是 DSA 路线的代表。

下表总结了两者的核心差异：

| | GPGPU | DSA |
|---|---|---|
| **全称** | General-Purpose GPU | Domain-Specific Architecture |
| **调度方式** | 硬件 warp scheduler 动态调度 | 编译器静态规划，硬件按计划执行 |
| **线程模型** | 有独立线程抽象（threadIdx/blockIdx） | 多数无线程概念，数据流静态编排 |
| **通用性** | 高——同一硬件跑各种负载 | 低——针对特定领域（如 AI）深度优化 |
| **能效比** | 中等（通用性开销大） | 高（专用化减少浪费） |
| **代表产品** | NVIDIA GPU、AMD GPU、摩尔线程、壁仞 | Google TPU、华为昇腾、Groq LPU |
| **对 CUDA 兼容性** | 可做 API 级兼容 | 存在语义级分歧，只能框架层兼容 |

有了这个基础认知，接下来分别分析两种路线在兼容 CUDA 生态时面临的核心挑战。你会发现：GPGPU 的障碍主要在**软件栈的工程规模**——每一层都要重实现但不存在根本性语义错配；DSA 的障碍则在**编程模型的语义鸿沟**——CUDA 的线程/warp/block 抽象无法直接映射到 DSA 的数据流/脉动阵列硬件。

### 2.1 GPGPU 路线的兼容挑战：软件栈规模，而非语义鸿沟

GPGPU 路线的硬件执行模型与 NVIDIA GPU 高度同构——都是 SIMT 并行计算架构，以 warp/线程为基本调度单位，拥有类似的 L1/L2 缓存层级和 HBM 显存体系。因此，GPGPU 路线在与 CUDA 生态兼容时，面临的主要挑战集中在**软件栈的工程规模**而非编程模型的语义错配。我们可以将其问题分解为以下几个层级：

**Runtime 和 Driver API 的完整复刻。** CUDA Runtime API（`libcudart.so`）提供了数百个函数（`cudaMalloc`、`cudaMemcpy`、`cudaStreamCreate`、`cudaEventRecord`、`cudaLaunchKernel`、`cudaGraphCreate`……），Driver API（`libcuda.so`）提供了更多底层函数（`cuInit`、`cuCtxCreate`、`cuModuleLoadData`……）。任何一个函数的行为偏差都可能导致现有程序运行错误。更棘手的是，`libcuda.so` 中还存在大量通过 `cuGetExportTable` 暴露的 Dark API——没有文档、没有签名、仅在特定 NVIDIA 库的内部运行时环境中被调用。一个可行的策略是"按需逆向"：追踪 PyTorch、vLLM 等主流框架实际触达的 Dark API 函数，逐函数通过反汇编和调试推测其语义并实现兼容版本，而不追求全量逆向。

**闭源算子库的 API 级干净室重实现。** cuBLAS、cuDNN、cuFFT、cuSPARSE、cuRAND 全是闭源动态库。这意味着无法像 NCCL 那样复用其开源算法实现——必须提供同名的自有 `.so` 文件（`libcublas.so`、`libcudnn.so` 等），使用 NVIDIA 公开的头文件（定义了全部函数签名和枚举）保证编译兼容，在内部用自研 kernel 实现完全相同的数学语义。这个过程被称为 **clean-room reimplementation（干净室重实现）**：API 和数学行为严格对标，内部实现完全独立。摩尔线程的 muDNN 对标 cuDNN、muBLAS 对标 cuBLAS，即采用此策略，并宣称关键算子的效率可达对应产品的 98% 以上。

这里需要区分两种不同层次的兼容策略——**"同名替换"** 和 **"改名替换"**：

- **同名替换（干净室重实现）**：提供一个函数名、符号、头文件完全对齐的自有库（如自研的 `libcublas.so`，内部导出 `cublasGemmEx` 等相同符号）。好处是**用户代码完全不用改，甚至已经编译好的二进制都能直接运行**——因为程序只按符号名 `cublasGemmEx` 链接，运行时动态链接到哪个 `.so` 由库搜索路径决定。代价是符号名与 NVIDIA 冲突，同一系统里不能同时存在两个 `libcublas.so`；且要精确复刻那些未文档化的内部行为（如 cuBLAS 内部创建的 stream 数量、event 同步语义）。

- **改名替换**：把函数名从 `cudaMalloc` 改成 `musaMalloc`、`cublasGemmEx` 改成 `musaGemmEx`。好处是符号不冲突，NVIDIA 和国产两套库可以共存；代价是**用户必须修改源码并重新编译**（哪怕只是 AST 级的函数名替换）。

两者的选择取决于目标用户群：如果面向的是有源码、愿意重新编译的开发者，改名替换（如摩尔线程的 `musa` 系列）更省事；如果要承接存量闭源二进制、做到零改动迁移，就必须做同名替换（干净室重实现）。这背后的工程量差异巨大——同名替换还要处理那些只在 NVIDIA 库内部使用的未公开行为，这也是 §2.1 反复强调"工程规模"的缘由之一。

这里解释一下上文提到的"AST 级替换"的含义。**AST（Abstract Syntax Tree，抽象语法树）** 是编译器在词法分析和语法分析之后，把源代码解析成的一种树状数据结构——源代码里的每个语法结构（变量声明、函数调用、表达式等）都对应树上的一个节点。例如 `cudaMalloc(&ptr, size)` 这行代码解析成 AST 后，就是一个"函数调用节点"，其下挂载"函数名 `cudaMalloc`"、"参数 `&ptr`"、"参数 `size`"三个子节点。所谓 AST 级替换，就是在编译器解析出这棵树之后、生成机器码之前，把树上的某些节点替换掉——比如把"函数名 `cudaMalloc`"节点换成"`musaMalloc`"。它比简单的文本查找替换更可靠，因为编译器已经理解了代码结构，能精确替换语义节点而不会误伤字符串或注释里的同名文字。摩尔线程之所以能用 AST 级替换来兼容 CUDA，是因为 MUSA 和 CUDA 的编程模型同构（都是 SIMT 线程模型），替换函数名即可、语义不变；而华为昇腾不能用这一招，是因为达芬奇架构与 CUDA 的编程模型根本不同，`__global__` 线程函数无法通过替换名字来变成 Ascend C 的 AI Core 函数，必须让开发者重写（这一点将在 §2.2 详细展开）。

**PTX 兼容层的取舍。** GPGPU 路线理论上可以在 PTX 层做二进制兼容——实现一个 PTX 到自有 ISA 的编译器，在运行时截获 FATBIN 中的 PTX 代码并 JIT 编译到自有硬件上执行（如 ZLUDA 在 AMD GPU 上的做法）。这一方案可以让已编译的 CUDA 二进制文件不经修改直接运行（前提是二进制中保留了 PTX 代码，而非仅有 SASS）。但 PTX 虽然号称"虚拟 ISA"，其设计仍然深度耦合于 NVIDIA 硬件——warp size=32 的线程模型、shared memory 的 bank 结构、Tensor Core 的 wmma/mma 指令，这些在另一个 GPU 微架构上不可能一一直接对应。PTX 级兼容的实际性能上限因此受限于"多对一"的硬件语义映射效率损耗。

总结来说，GPGPU 路线面临的是**巨大的工程规模问题**——软件栈的每一层都需要完整的重实现或映射，但不存在根本性的语义错配。摩尔线程 MUSA 对标的正是这条路，也是国产 GPU 中兼容程度最高的一家。

下表以摩尔线程 MUSA 为例，汇总了从 CUDA 迁移到 GPGPU 路线时，各层需要做的"函数名替换"（改名替换策略）：

| 层次 | CUDA（NVIDIA） | MUSA（摩尔线程） | 替换方式 |
|---|---|---|---|
| 设备指定 | `device="cuda"` | `device="musa"` | 字符串替换 |
| 内存分配 | `cudaMalloc` | `musaMalloc` | AST 级函数名替换 |
| 内存拷贝 | `cudaMemcpy` | `musaMemcpy` | AST 级函数名替换 |
| Kernel 启动 | `kernel<<<grid, block>>>` | `kernel<<<grid, block>>>`（语法不变） | 无需替换 |
| 集合通信后端 | `dist.init_process_group(backend="nccl")` | `backend="mccl"` | 字符串替换 |
| 线性代数库 | `cublasGemmEx`（`libcublas.so`） | `musaGemmEx`（`muBLAS`） | 函数名替换 + 库替换 |
| 深度神经网络库 | `cudnnConvolutionForward`（`libcudnn.so`） | `muDNN` 对应接口 | 函数名替换 + 库替换 |

可以看到，从 CUDA 到 MUSA 的迁移，本质上是**机械的函数名/字符串替换**，编程模型（SIMT 线程模型）完全不变。这正是 GPGPU 路线相对于 DSA 路线的核心优势——迁移可以自动化（摩尔线程的 MUSIFY 工具就是做这件事的），无需开发者理解新的编程范式。而 DSA 路线（如华为昇腾）在 kernel 层无法做这种机械替换，必须让开发者用全新的编程语言重写，这正是下一节要讨论的内容。

### 2.2 DSA 路线的兼容挑战：根本性的编程模型语义错配

与 GPGPU 的同构兼容不同，DSA 架构与 CUDA 在硬件执行模型上存在**根本性的语义分歧**。DSA（Domain-Specific Architecture）由 2017 年图灵奖得主 John Hennessy 和 David Patterson 提出，核心理念是放弃对所有应用都"通用但效率一般"的传统芯片设计，为特定领域（如 AI 推理）定制专用架构。华为昇腾的达芬奇架构、Google TPU 的脉动阵列、Groq LPU 的流水线结构，都是 DSA 路线的代表。

DSA 架构的核心特征是：**将复杂度从运行时转移到编译器**。GPGPU 依赖硬件动态调度器（warp scheduler）在运行时分配线程到 SM、处理缓存替换和同步；DSA 在编译器阶段就完成了所有调度决策——哪个张量放在哪块 SRAM、哪条指令在哪个时钟周期执行、跨芯片通信在哪一拍发生——硬件在运行时只负责按既定计划执行。

这一设计差异导致 DSA 与 CUDA 之间存在以下不可调和的语义分歧：

| CUDA/GPGPU 模型假设 | DSA 硬件现实 | 映射可行性 |
|---|---|---|
| 线程是第一类抽象：kernel 内用 threadIdx/blockIdx 组织并行 | 多数 DSA 没有独立线程概念，计算单元按数据流静态编排 | **难**：需在编译器中将线程语义转译为数据流调度 |
| warp=32 线程锁步执行，支持 warp 内 shuffle/vote/ballot 原语 | 无 warp 概念，计算单元间数据交换路径固定 | **极难**：warp 原语需软件模拟，性能损失大 |
| `__syncthreads()` 等动态同步原语 | 同步由编译器在静态调度时保证 | **可映射但不灵活** |
| 动态并行（device 端再启动 kernel） | 执行计划编译期固定，无法运行时再生成任务 | **原则上不可支持** |
| 任意控制流（数据依赖的分支、while 循环） | 部分 DSA 要求控制流可在编译期展开 | **不可支持（数据依赖控制流）** |
| 统一虚拟地址空间，运行时任意 malloc/free | 内存布局编译期静态规划 | **难**：需运行时管理层做语义等价转换 |

这解释了为什么纯粹的 DSA 架构（Google TPU、Groq LPU）没有试图直接兼容 CUDA 的线程级编程模型——它们在硬件设计之初就决定放弃这一兼容性，转而建立自己的全栈生态。华为昇腾走的则是"有限兼容"路线——在框架层（PyTorch PrivateUse1）和算子库层（AscendCL）提供 CUDA-like 的 API 体验，但在 kernel 编程层（算子开发）提供完全不同的 Ascend C 语言，要求开发者接受 SPMD + 显式数据流编程而非 CUDA 的线程模型。（**SPMD**，Single Program Multiple Data，单程序多数据，是一种并行编程范式：所有计算单元运行同一份代码，但各自处理不同的数据分片——这正是 DSA 架构"编译器静态规划、硬件按计划执行"的直接体现，与 CUDA 的 SIMT"每个线程执行同一指令但处理不同数据"在抽象层次上形成对比。）

#### 2.2.1 DSA 如何支持 PyTorch：哪些层次相同，哪些层次不同

上面的语义分歧表可能给人一个印象——DSA 架构与 CUDA 生态似乎处处格格不入。但实际情况并非如此：DSA 架构在**框架层和图编译层**与 CUDA 生态高度兼容，分歧主要集中在**最底层的 kernel 编程层**。一个 PyTorch 用户从 NVIDIA GPU 迁移到 DSA 硬件（如昇腾 NPU）时，绝大部分代码是无需改动的。

具体来说，DSA 的软件栈在以下层次与 CUDA 生态一致：

| 层次 | 是否一致 | 说明 |
|---|---|---|
| **框架层（PyTorch API）** | ✅ 基本一致 | `torch.nn`、`torch.optim`、`torch.distributed` 等高层 API 完全一样，用户代码几乎不改 |
| **图编译层（torch.compile）** | ✅ 一致 | 通过 `torch.compile` 的 Inductor 后端，DSA 厂商可以接管 PyTorch 的 FX 图，做自己的图优化 |
| **分布式框架层** | ✅ 一致 | `torch.distributed` 的 `all_reduce`、`all_gather` 等 API 相同，只是底层通信库从 NCCL 换成厂商自研库（如华为 HCCL） |
| **算子库层** | ⚠️ 需替换 | cuBLAS/cuDNN 这些闭源库不能直接用，需要厂商提供对标的算子库（如昇腾的 ATC 编译出的 .om 算子） |
| **kernel 编程层** | ❌ 本质不同 | CUDA 的线程级编程模型无法映射到 DSA 硬件，厂商提供完全不同的编程语言（如华为 Ascend C） |

**用户代码层面的对比**最能说明问题。以最典型的 PyTorch 使用场景为例——训练一个模型：

```python
# ===== NVIDIA GPU 上的代码 =====
import torch

model = MyModel()
model.to("cuda")                 # 迁移到 GPU
optimizer = torch.optim.Adam(model.parameters())

for x, y in dataloader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

```python
# ===== 华为昇腾 NPU 上的代码 =====
import torch
import torch_npu                 # 唯一新增：昇腾的 PyTorch 适配包

model = MyModel()
model.to("npu")                  # 仅此一处不同：cuda → npu
optimizer = torch.optim.Adam(model.parameters())

for x, y in dataloader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

两者的差异只有一行——`model.to("cuda")` 变成 `model.to("npu")`，外加 `import torch_npu`。所有上层 API（模型定义、损失函数、优化器、数据加载、训练循环）完全一致。这就是为什么"有限兼容"的 DSA 路线（华为昇腾）对绝大多数**应用开发者**是透明的——他们只使用框架 API，从不手写 kernel。

#### 2.2.2 DSA 的计算核心与 CUDA kernel 在代码层面的对比

差异真正体现在**手写算子**（自定义 kernel）的场景。当框架内置算子无法满足需求，需要开发者自己写底层计算逻辑时，CUDA 和 DSA 的编程模型就完全不同了。

以最简单的向量加法为例，先看 CUDA 的写法：

```cpp
// ===== CUDA kernel：SIMT 线程模型 =====
// 思维模式："每个线程处理一个元素，硬件自动分配线程"
__global__ void vecAdd_cuda(float* a, float* b, float* c, int n) {
    // threadIdx/blockIdx/blockDim 是 CUDA 的线程层次抽象
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
// 开发者只需描述"单个线程做什么"，并行度由 <<<grid, block>>> 指定
```

再看华为昇腾的 Ascend C 写法（DSA 的代表）：

```cpp
// ===== Ascend C kernel：SPMD + 显式数据流 =====
// 思维模式："每个 AI Core 处理一块数据，显式管理数据搬运和计算"
__global__ __aicore__ void vecAdd_ascendc(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int n)
{
    // 每个 AI Core 拿到自己的数据块范围
    int block_size = n / GET_BLOCK_NUM();
    int start = GET_BLOCK_IDX() * block_size;

    // 在 Unified Buffer（片上 SRAM）中申请缓冲区
    __local__ float local_a[256];
    __local__ float local_b[256];
    __local__ float local_c[256];

    for (int i = start; i < start + block_size; i += 256) {
        int chunk = min(256, start + block_size - i);

        // CopyIn：全局内存（HBM）→ Unified Buffer
        __memcpy(local_a, a + i, chunk * sizeof(float), MEMCPY_GM_TO_UB);
        __memcpy(local_b, b + i, chunk * sizeof(float), MEMCPY_GM_TO_UB);

        // Compute：在片上完成计算
        for (int j = 0; j < chunk; j++)
            local_c[j] = local_a[j] + local_b[j];

        // CopyOut：Unified Buffer → 全局内存
        __memcpy(c + i, local_c, chunk * sizeof(float), MEMCPY_UB_TO_GM);
    }
}
```

两者的核心差异可以总结为：

| | CUDA kernel | Ascend C（DSA） |
|---|---|---|
| **并行抽象** | 线程（thread/warp/block） | AI Core（每个 core 处理一块数据） |
| **片上数据搬运（HBM↔SRAM）** | 隐式——Shared Memory 的加载由编译器/硬件自动管理，开发者通常无需显式指定数据何时进出片上缓存 | 显式——开发者必须用 `__memcpy` 手动执行 CopyIn（HBM→UB）和 CopyOut（UB→HBM），硬件没有自动缓存，数据流完全由开发者规划 |
| **内存层次** | 透明（global/shared/register 由编译器分配） | 显式——Unified Buffer、L1/L0 缓存需开发者指定 |
| **同步方式** | `__syncthreads()` 等运行时同步 | 编译器静态调度，多数无需显式同步 |
| **编程范式** | SIMT（单线程视角） | SPMD + 三段式（CopyIn→Compute→CopyOut） |

这里需要特别澄清一个容易混淆的点：表格中"片上数据搬运"指的是 **device 内部** 的数据移动（GPU/NPU 芯片内的 HBM 显存 ↔ 片上 SRAM），它与 host 和 device 之间的大块数据搬运是**两个完全不同的层面**。后者在 CUDA 中对应 `cudaMemcpy`（CPU 内存 ↔ GPU 显存），在昇腾中对应 `aclrtMemcpy`（Host 内存 ↔ NPU 显存）——这一层**两个架构都有对应接口，语义完全一致**，不是差异所在。真正的差异在 device 内部：CUDA 的 Shared Memory 加载由编译器和硬件自动管理（数据什么时候从 HBM 进 shared memory、什么时候被换出，由硬件缓存策略决定，开发者大多无需关心）；而 DSA 架构（如昇腾）没有这种自动缓存，Unified Buffer 的数据进出必须由开发者用 `__memcpy` 显式地"搬进来（CopyIn）→ 算 → 搬出去（CopyOut）"。

这一差异的本质根源，正是前面表格说的"将复杂度从运行时转移到编译器"：CUDA 让开发者写"单线程逻辑"，硬件在运行时负责把线程分配到 SM、管理缓存；DSA 让开发者写"数据块逻辑"，编译器在离线阶段规划好数据流向和计算时序，硬件按计划执行。两者无法通过简单的函数名替换来桥接——这也是为什么华为昇腾必须提供全新的 Ascend C 语言，而不是像摩尔线程 MUSA 那样用 `cudaMalloc → musaMalloc` 的 AST 级替换（AST 的含义已在 §2.1 中解释）来兼容 CUDA。

对于自研芯片而言，这一层是关键的战略分水岭。需要特别澄清的是：**在"应用开发者只写 PyTorch、不碰 kernel"这个场景下，GPGPU 和 DSA 的兼容成本其实是打平的——两者都接近零。** 因为无论是摩尔线程的 `to("musa")` 还是华为昇腾的 `to("npu")`，对于上层用户来说都只是改一个设备名字符串，这一层的迁移摩擦对两种架构同样小。真正的差异出现在**算子开发者（需要手写 kernel）**这个群体：GPGPU 路线可以用 AST 级替换把存量 CUDA kernel 几乎原样迁移过来（`cudaMalloc → musaMalloc`），迁移成本极低；而 DSA 路线要求开发者用全新的编程范式（如 Ascend C 的 CopyIn→Compute→CopyOut）重写 kernel，学习成本和迁移成本显著更高。因此，**如果目标用户群包含大量的算子开发者或需要迁移存量 CUDA kernel 的团队，GPGPU 路线具有明显优势；如果目标用户群主要是应用开发者，两者的差别则微乎其微，此时硬件能效、成本等其它因素会成为更重要的决策依据。**

对于国产自研芯片而言，这是一个无法回避的战略抉择：**选择 GPGPU 路线，可以用 API 级兼容 + 库重写的工程规模换取对 CUDA 生态的最大覆盖（代价是硬件能效比不可能达到 DSA 的极致水平）；选择 DSA 路线，可以获得针对 AI 负载的确定性低延迟和极致能效，但必须接受与 CUDA 的线程级编程模型之间的语义不可兼容边界。**

---

## 三、案例一：华为昇腾 CANN —— DSA 架构的"有限兼容"实践

在第二章中，我们从宏观视角对比了 GPGPU 与 DSA 两条路线的本质分歧，并明确了 DSA 架构"将复杂度从运行时转移到编译器"的核心特征。但那些讨论停留在抽象层面——DSA 架构到底长什么样、它的软件栈如何运作、开发者实际会面对什么样的编程体验，这些都需要落到一个具体的案例上。

华为昇腾（Ascend）系列正是观察这些问题的绝佳样本。它是国产 DSA 路线中最具代表性、生态最成熟的一家，也是目前少数能与 NVIDIA CUDA 生态在"框架层兼容、kernel 层分化"上形成完整对照的国产方案。通过剖析昇腾的达芬奇架构和 CANN 软件栈，我们不仅能看到 DSA 架构如何以完全不同于 CUDA 的方式完成 AI 计算负载，更能反推出一个关键结论：**硬件架构的选择，从根本上决定了软件栈的构建方式和使用流程**——这正是本章希望传达的核心。

这一章会带你从零基础开始，建立对昇腾 CANN 全部所需背景知识的理解图谱。由于昇腾生态本身极其庞大（涉及硬件、编译器、算子库、框架适配、分布式等众多领域），本章不会在每个部分过度深入技术细节，而是力求帮你搭建一个"看得懂全景、知道该往哪里深挖"的框架。如果你想在某个具体领域深入（例如 CANN 内部的算子开发或 Ascend C 编程模型），可以再针对性地查阅官方文档。华为官方的 CANN 架构文档见：[CANN 基础架构](https://developer.huawei.com/consumer/cn/doc/hiai-GUIdes/cannkit-basic-architecture-0000002300080172)。

### 3.1 概念澄清：CANN、达芬奇架构与昇腾芯片

在进入细节之前，有必要先厘清几个容易混淆的名词——它们分别对应软件、硬件、产品三个层面，但经常被混用。

**CANN（Compute Architecture for Neural Networks，神经网络计算架构）** 是华为面向 AI 场景推出的**异构计算架构**，本质上是昇腾硬件之上的完整软件栈。它的定位对标 NVIDIA 的 CUDA 平台——向上承接 PyTorch、TensorFlow、MindSpore 等 AI 框架，向下驱动昇腾 NPU 硬件，中间提供算子库、图编译器、运行时、集合通信库等全套工具。CANN 于 2018 年发布，与昇腾硬件深度协同，是理解华为 AI 软件体系的核心。

**达芬奇（DaVinci）架构** 是昇腾 AI 处理器采用的**硬件计算架构**，由华为自研，2018 年首次商用。它专为 AI 计算负载（尤其是大规模矩阵乘加运算）设计，核心设计是 3D Cube 矩阵计算单元。注意区分：CANN 是**软件**，达芬奇是**硬件架构**——前者运行在后者之上。

**昇腾（Ascend）芯片** 是采用达芬奇架构的**具体产品**。华为通过"魔方"（Cube）单元的不同组合，衍生出面向不同场景的芯片：

| 芯片型号 | 定位 | 典型场景 |
|---|---|---|
| Ascend 310 系列 | 低功耗推理 | 边缘计算、端侧设备（摄像头、自动驾驶 MDC 等） |
| Ascend 910 系列 | 高性能训练 | 云端大模型训练、数据中心 |
| Ascend 950 系列 | 新一代推理优化 | 分为 950PR（预填充优化）与 950DT（解码优化） |

虽然这些芯片定位不同，但都基于同一个达芬奇架构，共享同一套 CANN 软件栈——这正是达芬奇架构"可扩展性"的体现：通过组合不同数量的 Cube 单元，就能衍生出面向不同功耗和算力需求的芯片，而软件栈无需推倒重来。

### 3.2 硬件基础：达芬奇架构与 AI Core 的计算单元

在了解了概念之后，我们进入达芬奇架构的硬件细节。达芬奇架构的 AI Core 硬件组织方式并非一成不变，而是随着代际演进发生过一次重要变化——根据 **Cube（矩阵）计算单元和 Vector（向量）计算单元是否同核部署**，可以分为"耦合架构"和"分离架构"两种模式：

**耦合架构**：Cube 计算单元和 Vector 计算单元**同核部署**在同一个 AI Core 上。计算单元和存储单元集中在一个核内，数据处理流向在核内完成，Cube 与 Vector 共享核内存储资源。这种架构下，数据不需要跨核传递，延迟更低，但单个核的复杂度更高。

**分离架构**：将 AI Core 拆分为两个**独立的核心**——**AI Cube（AIC，矩阵计算核心）** 和 **AI Vector（AIV，向量计算核心）**。每个核都有自己的 Scalar 单元，能独立加载代码段，实现矩阵计算与向量计算的解耦，在系统软件的统一下配合完成计算。AIC 与 AIV 之间通过 **Global Memory（全局内存）** 传递数据，相比耦合架构增加了 BT Buffer（BiasTable Buffer，存放 Bias）和 FP Buffer（Fixpipe Buffer，存放量化参数、ReLU 参数等）两个缓冲区。

两者的核心差异可以总结为：

| 对比维度 | 耦合架构 | 分离架构 |
|---|---|---|
| 核心部署 | Cube + Vector 同核 | Cube（AIC）与 Vector（AIV）分离为两个独立核 |
| 独立代码段 | 否 | 是（每个核独立加载） |
| 核间通信 | 核内直接共享存储 | 通过 Global Memory 传递 |
| 额外 Buffer | 无 | BT Buffer + FP Buffer |
| 适用产品 | 早期的 Atlas 推理/训练系列 | Atlas A2/A3 系列（如 Atlas 800I A2、Atlas 200I/500 A2） |

为什么要从耦合走向分离？核心驱动力是**算力密度与能效的进一步提升**。当 Cube 和 Vector 同核时，两者共享同一套控制逻辑和存储，虽然延迟低，但随着矩阵计算规模增大，Cube 单元需要更大的面积和更高的存储带宽，与 Vector 单元"挤在"一个核内会相互制约。将其拆分为独立的 AIC 和 AIV，可以让矩阵计算核心（AIC）独立扩展、专注极致吞吐，同时让向量计算核心（AIV）专注逐元素运算，两者通过统一调度协同工作。这是昇腾从早期代次（耦合）向 A2/A3 代次（分离）演进过程中的一个重要架构变化。

无论耦合还是分离，达芬奇架构内部都包含三种基本计算单元，它们的分工在两种架构下是一致的：

- **Cube Unit（矩阵计算单元）**：专做矩阵乘加运算。单个 Cube 每时钟周期可完成一个 16×16×16 的 FP16 矩阵乘加（等效 8,192 次 MAC），是达芬奇架构的计算主力，承担了 AI 负载中约 90% 的计算量。对标 NVIDIA 的 Tensor Core，但它是达芬奇架构的独立核心单元。
- **Vector Unit（向量计算单元）**：负责逐元素运算（ReLU、GELU、LayerNorm、Softmax 等激活函数和归一化），单次处理 128 个 FP16 或 64 个 FP32 元素。
- **Scalar Unit（标量计算单元）**：类似一个小型 CPU，负责循环控制、分支判断、地址计算，以及向 Cube/Vector 发射指令。

三类计算单元**并行独立工作**——Cube 在算矩阵乘的同时，Vector 可以并行处理激活函数，Scalar 准备下一批数据的地址。这种"按计算类型分离"的设计，正是 DSA 架构"专用化"哲学的硬件体现：矩阵运算这种 AI 负载里最密集、最规律的部分，被单独拎出来用最高效的专用硬件实现。

**为什么达芬奇要采用"按类型分离"的设计？** 答案在于能效比。AI 负载的计算高度集中于矩阵乘法，而矩阵乘法有着极其规整的数据流（数据在规则的网格中流动，不需要复杂的动态调度）。将这部分"确定性强"的计算用专用硬件（Cube）实现，可以把这部分硬件的利用率和能效做到极致——同等功耗下，矩阵乘吞吐远高于 NVIDIA 的通用 SM。代价则是灵活性：硬件不再能像 SM 那样"什么都能算"，必须依赖编译器（GE 图引擎）把算子合理地拆分、映射到不同的单元上。

存储层次同样体现了"功能绑定"的设计。达芬奇架构的存储不仅是多层级，更是**按功能划分的**——L0A/L0B/L0C 仅供 Cube 使用（分别缓存矩阵 A、B、C 的分块），UB（Unified Buffer，约 256KB/核）仅供 Vector 使用。这与 CUDA 中 Shared Memory 可被任意线程自由访问的模型完全不同。这一设计带来了更高的能效和确定性，但也意味着内存管理和数据流调度**必须由开发者或编译器显式规划**，无法依赖硬件自动缓存——这直接决定了后文将讨论的 Ascend C 编程模型为何必须采用"显式数据搬运"。

### 3.3 软件栈逐层映射：CANN 如何对标 CUDA

在理解了硬件基础之后，我们回到软件栈。CANN 作为昇腾的全栈软件平台，全面对标 CUDA Toolkit + CUDA-X 库 + GPU 驱动的组合功能。下面逐层对比两者，这是理解"有限兼容"策略的关键。

**第一层：AscendCL → CUDA Runtime/Driver API**

AscendCL 是 CANN 的统一编程接口，在 API 分类和调用模式上直接对标 CUDA Runtime API。下表列出了关键接口的对应关系：

| 功能域 | CUDA Runtime API | AscendCL 对应接口 | 兼容度 |
|---|---|---|---|
| 资源初始化 | `cudaSetDevice`, `cudaGetDeviceCount` | `aclrtSetDevice`, `aclrtGetDeviceCount` | 高 |
| 内存分配 | `cudaMalloc`, `cudaFree` | `aclrtMalloc`, `aclrtFree` | 高（语义完全对等）|
| 流管理 | `cudaStreamCreate`, `cudaStreamSynchronize` | `aclrtCreateStream`, `aclrtSynchronizeStream` | 高 |
| 事件管理 | `cudaEventCreate`, `cudaEventElapsedTime` | `aclrtCreateEvent`, `aclrtEventElapsedTime` | 高 |
| Kernel 执行 | `cudaLaunchKernel` 或 `<<<>>>` 语法糖 | **需将算子/模型离线编译为 .om 格式**，通过 `aclmdlExecute` 加载执行 | **本质不同** |
| 图捕获 | `cudaStreamBeginCapture`, `cudaGraphLaunch` | `aclmdlRICaptureMode_*` 等 RI 捕获模式 | 高（枚举基本对应）|

关键差异在于 kernel 执行模型：CUDA 允许主机代码中任意调用 `kernel<<<grid, block>>>(...)` 即时下发到 GPU；AscendCL 需要**先将算子或模型通过 ATC 工具离线编译为 .om 格式**（昇腾自有离线模型格式），再通过 `aclmdlExecute` 加载执行。开发者无法在 Python 中写一个 kernel 即时测试——必须走"编写→编译→加载→执行"的闭环。华为通过 CANN 兼容层提供 `cuda.h` 头文件映射，在特定 API 范围内实现编译兼容（`cudaMalloc` 被替换为 `aclrtMalloc`），但 kernel 启动模型不可直接映射。

**第二层：GE 图引擎 → CUDA Graphs + TensorRT + torch.compile**

GE（Graph Engine）是 CANN 中最具特色的组件，它同时扮演了 CUDA 生态中三个不同工具的角色：

- **TensorRT 的图优化角色**：GE 内置数十种融合 pass（编译器优化遍历，对计算图做等价变换以提升效率），可将 Conv→BN→ReLU 融合为单算子（中间结果在片上 UB 内流转不写 HBM），自动做常量折叠、死代码消除和公共子表达式消除。
- **torch.compile/Inductor 的角色**：GE 通过 `torch.compile` 直接对接 PyTorch——用户使用 `torch.compile(model, backend="inductor")` 后，昇腾后端自动接管 PyTorch 生成的 FX 图，转化为 AIR 中间表示，再执行全链路图优化。这使 PyTorch 用户从 eager 模式切换到图模式时几乎不需要代码改动。
- **CUDA Graphs 的角色**：GE 支持**图下沉（Graph Sinking）**——将整个优化后的计算图一次性下发到 NPU，NPU 侧自主完成所有执行，Host 仅等待最终结果。由于 NPU 的静态调度特性，图下沉能彻底消除 Host-Device 间的来回交互。

**第三层：Ascend C → CUDA C++ Kernel 编程**

这是 CANN 与 CUDA 差异最大的一层。在 NVIDIA 生态中，自定义 kernel 用 CUDA C++ 编写（`__global__` 函数）；在昇腾生态中，对应的是 **Ascend C**——一套基于 C++17 标准的算子编程语言，但其编程范式与 CUDA 截然不同。以下对比直观地展示了差异：

```cpp
// ---- CUDA C++: SIMT 线程模型 ----
// 思维模式："每个线程处理一个元素"
__global__ void vecAdd_cuda(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ---- Ascend C: SPMD + 显式数据搬运 ----
// 思维模式："每个 AI Core 处理一块数据，显式管理数据流"
__global__ __aicore__ void vecAdd_ascendc(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c, int n)
{
    int block_size = n / GET_BLOCK_NUM();   // 每个 AI Core 处理的数据量
    int start = GET_BLOCK_IDX() * block_size;

    __local__ float local_a[256];  // 在 Unified Buffer 中分配本地缓冲
    __local__ float local_b[256];
    __local__ float local_c[256];

    for (int i = start; i < start + block_size; i += 256) {
        int chunk = min(256, start + block_size - i);

        // CopyIn: 全局内存 → Unified Buffer
        __memcpy(local_a, a + i, chunk * sizeof(float), MEMCPY_GM_TO_UB);
        __memcpy(local_b, b + i, chunk * sizeof(float), MEMCPY_GM_TO_UB);

        // Compute: UB 内计算
        for (int j = 0; j < chunk; j++)
            local_c[j] = local_a[j] + local_b[j];

        // CopyOut: UB → 全局内存
        __memcpy(c + i, local_c, chunk * sizeof(float), MEMCPY_UB_TO_GM);
    }
}
```

Ascend C 的编程范式可以概括为 **CopyIn → Compute → CopyOut 三段式 + Tiling 多核拆分**。对于 CUDA 开发者而言，这是从"写线程逻辑"到"写显式数据流"的根本性编程习惯转换。不过对于绝大多数仅使用框架 API 而非手写 kernel 的应用开发者，这一差异被框架适配层（`torch_npu`）完全屏蔽。此外，Triton 编译器的 tile 级抽象（`tl.load`/`tl.dot`/`tl.store`）与 Ascend C 的三段式范式在逻辑上高度一致——Triton-昇腾后端移植后，大量现有 Triton kernel 可近零改写直接运行。

**第四层：HCCL → NCCL**

HCCL（Huawei Collective Communication Library）是 CANN 中的集合通信库，**在 API 层面与 NCCL 几乎没有区别**。两者支持完全相同的 12 种通信原语（AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll、Send/Recv 等），API 参数签名基本一致：

```cpp
// NCCL:  ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
// HCCL: HcclAllReduce(sendbuff, recvbuff, count, HCCL_FLOAT, HCCL_SUM, comm, stream);
```

在 PyTorch 分布式训练中，迁移只需修改一个字符串：

```python
# NCCL 后端（NVIDIA GPU）
dist.init_process_group(backend="nccl")

# HCCL 后端（昇腾 NPU）——仅此差异
dist.init_process_group(backend="hccl")
```

`torch.distributed.all_reduce` 等所有上层 API 调用在两种后端上行为完全一致。HCCL 底层自动检测昇腾集群的 HCCS（Huawei Cache Coherence System，对标 NVLink）和 RoCE（RDMA over Converged Ethernet，对标 InfiniBand）网络拓扑，选择最优通信算法。在 8×Ascend 910 集群上，128MB 数据 AllReduce 可达 78 GB/s（A100×8 NVLink 为 92 GB/s，约 85%）。

### 3.4 硬件架构如何塑造软件栈：从达芬奇看 DSA 的决定性影响

回顾上面四层的对比，我们可以提炼出一个贯穿始终的规律：**昇腾软件栈与 CUDA 软件栈的每一处差异，几乎都能追溯到"达芬奇 DSA 架构"与"NVIDIA GPGPU 架构"在硬件设计哲学上的根本区别。**（注意：这里用的是第二章的 DSA vs GPGPU 对比视角，而非 §3.2 里昇腾内部的"耦合/分离"术语。）

最直观的例子是编程模型的分化。NVIDIA 的 GPGPU 架构允许开发者写"单线程逻辑"（`__global__` 函数里只管 `threadIdx`），因为硬件有 warp scheduler 在运行时自动把线程分配到 SM、自动管理缓存替换。而达芬奇 DSA 架构没有这套动态调度——Cube/Vector/Scalar 三单元各司其职、存储功能绑定，硬件"不知道怎么调度"，所以这些决策必须前移到编译器或开发者手中。这就是为什么 Ascend C 必须让开发者显式地做 CopyIn→Compute→CopyOut，显式地指定数据放在 UB 还是 L0——**硬件的"不灵活"，转化成了编程模型的"显式"**。

再比如图编译器的地位差异。在 NVIDIA 生态中，TensorRT/torch.compile 是"可选优化"——不启用它们，eager 模式照样能跑。但在昇腾生态中，GE 图引擎是**必选的基础设施**——因为达芬奇架构要求算子被正确拆分映射到不同计算单元，这件事必须由编译器在编译期完成，硬件自己无能为力。这解释了为什么 §3.3 中 AscendCL 的 kernel 执行必须走"离线编译 .om → 加载执行"的闭环，而无法像 CUDA 那样即时下发。

这一规律对其它国产硬件厂商同样成立，可以作为一个分析框架：

- **软件栈的"上层"（框架适配、集合通信 API、图优化）大概率趋同**——因为无论硬件架构如何，厂商都想让 PyTorch 用户只改一个 device 字符串就能迁移，所以 `torch_npu` 这类适配层、HCCL 这类对标 NCCL 的通信库，在所有国产方案中都会以相似的形式存在。
- **软件栈的"底层"（kernel 编程语言、算子的编译模型）则取决于硬件架构**——如果厂商走 GPGPU 路线（硬件有动态调度），它可以像摩尔线程 MUSA 那样提供 CUDA-like 的线程编程模型，甚至用 AST 替换直接复用 CUDA 生态；如果厂商走 DSA 路线（硬件静态调度），它就必须像昇腾一样发明一套显式数据流编程语言（Ascend C 的同类物），因为硬件的计算单元组织和存储层次决定了编程抽象无法与 CUDA 对齐。
- **图编译器的"必要性"也由硬件决定**——GPGPU 可以把它当作可选优化，DSA 则必须把它当作核心基础设施。

因此，判断一个国产 AI 芯片软件栈的形态，最根本的线索不是它宣传的"兼容 CUDA 程度"，而是它的**底层硬件架构**：硬件有没有动态调度器、存储是否功能绑定、矩阵计算是不是独立单元——这些硬件特征，几乎一对一地决定了它的编程语言、编译流程和使用体验。

### 3.5 昇腾方案的"兼容与不兼容"边界总结

**可以实现"类 CUDA 体验"的区域**：PyTorch 用户代码（仅改 device 名→`"npu"` 或劫持 `"cuda"` 映射）、分布式训练初始化（仅改 backend 字符串）、标准模型推理（ONNX→ATC 转换一次）、torch.compile 优化（Inductor 后端自动接管）、Triton kernel（后端移植后零改写）。

**需要手动适配的区域**：自定义 CUDA C++ kernel（必须用 Ascend C 按 CopyIn→Compute→CopyOut 范式重写）、warp 级原语（`__shfl_sync` 等无直接映射，需重新设计算法）、TensorRT 优化模型需在昇腾上重新做 ATC+AMCT+AOE 全链路优化。

**不可实现的区域**：已编译的 CUDA 二进制（无源码的 PTX/CUBIN）、动态并行（device kernel 再启动 kernel）。这是 DSA 架构的固有边界——非投入更多工程资源能够突破。

---

## 四、案例二：Google TPU —— 完全自主生态的 DSA 路线

如果说华为昇腾是"在 DSA 硬件上尽可能兼容主流框架"的折中路线，Google TPU 则走向了完全不同的方向：**从指令集、编译器到框架，构建完全独立于 CUDA 的全栈自主生态。** 这种策略由 Google 特殊的商业定位决定——TPU 仅通过 Google Cloud 对外提供，不向第三方销售芯片，因此不需要"兼容 CUDA 以吸引外部开发者"。

### 4.1 TPU 的硬件与编译器栈

TPU 从 2015 年的第一代起就采用了**脉动阵列（Systolic Array）**作为核心计算单元——一种多个处理单元以规则网格排列、数据按固定节奏在单元间级联流动的专用结构。每一代 TPU 在脉动阵列尺寸、HBM 容量和互联拓扑上都有升级，但核心架构路径从未改变。最新的 Ironwood（TPU v7）单芯片的 BF16 算力已经达到数千 TFLOPS 级别。

与 CUDA 生态的全栈差异在编译器层最为显著。TPU 的编译链路是：

```
JAX / TensorFlow 用户代码
    ↓ (框架层)
XLA (Accelerated Linear Algebra) — Google 的全程序优化编译器
    ↓ (HLO → LLO → TPU 原生指令)
TPU Runtime — 加载和执行编译后的程序
    ↓
TPU 硬件 — 脉动阵列执行
```

这里的 **XLA** 是关键。XLA 接受高层框架（JAX 的 jit 编译函数、TensorFlow 的 graph）输出的 HLO（High-Level Optimizer，高层优化器 IR），执行从算子融合、内存布局优化到 TPU 原生指令代码生成的全链路编译。与 CUDA 的区别在于：**XLA 是一个完整的全程序编译器**——它不是像 NVCC 那样逐 `.cu` 文件编译单个 kernel，而是对整个模型的计算图进行全局优化后再统一生成代码。这恰好契合 DSA 的静态调度需求：编译器在离线阶段就掌握了全部张量的生命周期和所有运算的依赖关系，可以做出最优的资源分配决策。

### 4.2 TPU 的"不兼容"策略与开发者迁移体验

Google 没有为 TPU 提供任何 CUDA API 兼容层。开发者代码中不会出现 `cudaMalloc` 或 TPU 版的 `tpuMalloc`——取而代之的是通过 **JAX**（Google 开发的函数式自动微分 + XLA 编译框架）来表达计算，XLA 编译器自动将 JAX 操作转换为 TPU 指令：

```python
import jax
import jax.numpy as jnp

# TPU 上的矩阵乘法 —— 没有设备 API 调用，没有 cudaMalloc
@jax.jit                          # JIT 编译整个函数
def matmul_fn(x, y):
    return jnp.dot(x, y)

# 数据自动分配到 TPU 设备
x = jnp.ones((1024, 1024))       # 默认在 TPU 上创建（若 TPU 可用）
y = jnp.ones((1024, 1024))
z = matmul_fn(x, y)               # 单步执行，整个计算图被 XLA 编译优化
```

这种体验对习惯于 PyTorch + CUDA 的开发者来说，迁移成本体现在**编程范式而非 API 对接**上——JAX 是函数式编程风格（无副作用、纯函数转换），需要学习 `jit`、`vmap`（自动向量化）、`pmap`（SPMD 并行设备映射）等函数式原语，这与 PyTorch 的 imperative（命令式）风格完全不同。但对于已经在 JAX 生态中的用户（如 Google DeepMind 和许多学术研究团队），设备差异是完全透明的——代码不经修改即可在 TPU 或 GPU 上运行，XLA 负责后端适配。

### 4.3 TPU 对国产自研芯片的启示

TPU 路线的核心启示是：**如果预期的市场定位是"内部或封闭云服务"，且有足够的资源和时间建立自有框架生态（XLA + JAX 经过近十年开发），那么完全放弃 CUDA 兼容性是可行的。** 但这一策略的前提条件是拥有自己的深度学习框架生态——JAX 在 Google 内部和学术界的增长是 TPU 被广泛使用的前提。如果面向的是已有海量 PyTorch 代码的企业客户，完全自建生态的风险极高。

---

## 五、案例三：摩尔线程 MUSA —— GPGPU 路线的"极致 API 兼容"

摩尔线程是国产 GPU 中走 CUDA 兼容路线最坚定的一家。其策略可概括为：**让开发者写的 CUDA 代码几乎不改就能在自己的卡上跑起来。** 这一定位的硬件基础是 MUSA（Meta-computing Unified System Architecture，元计算统一系统架构）——一套从芯片架构到软件栈全栈自研的 GPGPU 体系。

### 5.1 MUSA 的分层兼容策略

MUSA 的兼容是在每一层都做对应映射，而非只在某一层"一刀切"：

**Runtime 层**：MUSA Runtime 提供 `musaMalloc`、`musaMemcpy`、`musaLaunchKernel` 等 API，语义与 CUDA Runtime 完全对应。MUSA SDK 5.1.0 对标 CUDA 12.8，兼容接口数达 761 个（涵盖驱动、运行时和核心数学库）。

**编译器层**：MUSA 编译器支持 CUDA C++ 扩展语法（`__global__`、`<<<>>>` 等），CUDA 源码经 **MUSIFY**（代码自动迁移工具）转换后即可编译为 MUSA 指令。MUSIFY 的转换实质上是一次**AST 级别的函数名替换**（`cudaMalloc` → `musaMalloc`），因为两者在编程模型上同构（都是 SIMT 线程模型），转换不涉及语义重构。

```cpp
// 原始 CUDA 代码  --MUSIFY 自动转换-->  MUSA 代码
// cudaMalloc(&ptr, size);               musaMalloc(&ptr, size);
// cudaMemcpy(dst, src, n, H2D);         musaMemcpy(dst, src, n, H2D);
// kernel<<<grid, block>>>(args);         kernel<<<grid, block>>>(args);  // 语法不变
// cudaFree(ptr);                        musaFree(ptr);
```

**加速库层**：摩尔线程提供 muDNN 对标 cuDNN（宣称 GEMM/FlashAttention 效率超过 98%）、muBLAS 对标 cuBLAS、MCCL 对标 NCCL（支持 AllReduce/AllGather 等全部原语，双机 RDMA 带宽利用率达 97%）。这一层的实际工作量极大——需要独立实现每个库中的全部核心 API 和底层 kernel，以满足数值精度和性能上的对标要求。摩尔线程采用 API 级干净室重实现的策略，即提供同名的自有 `.so` 文件并保证符号表和语义完全对应，但内部实现全自研。

**框架层**：MUSA 已正式成为 vLLM 官方后端，并合入 SGLang 官方主线。对 PyTorch 用户而言，迁移成本仅为将 `device="cuda"` 改为 `device="musa"`（或通过适配层自动劫持映射）。

### 5.2 MUSA 路线的优势与边界

MUSA 路线最大的优势在于开发者体验——从 CUDA 生态迁移的摩擦成本在国产方案中最低。但它仍然存在边界：**需要重新编译**（至少对 kernel 部分），不能直接运行已编译的 CUDA 二进制；对于使用了内联 PTX 汇编（`asm("...")`）的 CUDA 代码，MUSIFY 无法自动转换；部分 CUDA 特有的硬件特性（如特定代数 Tensor Core 的 wmma 指令）对应的 MUSA 硬件实现可能不完全一致，需要手动调整。

在行业采纳方面，摩尔线程的 MUSA 已被德锐特（德睿智药子公司）用于冷冻电镜数据处理软件 RELION5 的迁移，双机 RDMA 通信带宽利用率达到 97%。在 PyTorch 生态中，MUSA 已完整支持全部 3,194 个算子。对比华为昇腾的 CANN（需要开发者学习 Ascend C），MUSA 的"零学习成本迁移"策略在降低开发者心理门槛方面具有明显优势。

---

## 六、总结

从本系列分析中可以提炼出几条贯穿全文的核心结论。

**1. CUDA 不是一层 API，而是一个七层软件帝国。** 从 PyTorch 框架到 GPU 硬件，数据流经过了框架调度层、闭源算子库、Runtime/Driver API、PTX 中间表示、SASS 机器码、FATBIN 多架构打包、以及 Runtime 加载与执行。每一层都有独立的技术壁垒——闭源的 cuBLAS 和 cuDNN 的自动调优引擎、libcuda.so 中的数百个 Dark API、NVLink 与 InfiniBand 的私有互联协议、以及数十年来积累的特定架构 SASS 编码优化。理解这种多层次结构，是评估任何兼容方案可行性的前提。

**2. GPGPU 与 DSA 是两种不同的"难"。** GPGPU 路线（摩尔线程、壁仞）的主要障碍是**软件栈的巨量工程规模**——需要重写数百个 Runtime API、复刻闭源算子库的全部接口、自研编译器工具链——但不存在根本性的编程模型错配。DSA 路线（华为昇腾、Google TPU）的障碍是**语义级的分歧**——CUDA 的线程/warp/block 编程模型无法直接映射到 DSA 的数据流/脉动阵列架构，这是架构选择决定的固有边界，非投入更多工程资源可突破。DSA 厂商的可行策略不是"兼容 CUDA"，而是通过框架层（PyTorch PrivateUse1）和图编译层（torch.compile/Triton 后端移植）在更高的抽象层级上承接 AI 开发者生态。

**3. 三个案例展示了三种不同的兼容策略。** 华为昇腾 CANN 是在 DSA 硬件上走"有限兼容"路线——在框架层和 Runtime 层提供类 CUDA 体验，在 kernel 编程层做完全不同的 Ascend C 语言。Google TPU 是完全放弃 CUDA 兼容，转而通过 JAX + XLA 构建独立的全栈生态。摩尔线程 MUSA 是在 GPGPU 硬件上走"极致 API 兼容"路线——通过 MUSIFY 工具、自研编译器、自有加速库和完整 PyTorch 适配，将迁移成本压缩到改 device 名字符串的程度。三条路线各自匹配厂商的硬件架构选择、目标市场定位和生态策略——不存在普适的"最优解"，只有与自身禀赋匹配的"最适合解"。

**4. 产业趋势：图编译模式的普及对 DSA 有利。** PyTorch 2.0 的 `torch.compile` 默认启用图模式编译、vLLM 等推理框架在向编译派发模式演进，整个 AI 软件栈正在从逐 kernel 的 eager 执行转向全局图优化。图编译模式下，上层框架主动交出完整的子图结构，恰好是 DSA 编译器（如 CANN GE、Google XLA）的标准输入形态——无需在运行时动态转换线程语义。这一趋势客观上缩小了 DSA 架构在"迁移痛感"上与 GPGPU 路线的差距——但需要指出的是，对于存量 CUDA 源码（特别是涉及自定义 kernel 和 warp 级编程的代码），GPGPU 路线的直接兼容优势仍然不可替代。

---

*声明：本文技术分析基于公开资料整理，包括英伟达官方 CUDA 编程指南与编译器文档、华为昇腾 CANN 开发文档与社区技术文章、Google TPU 与 XLA/JAX 公开论文与技术博客、摩尔线程 MUSA 开发者文档、以及相关行业媒体的技术分析报告。所有代码示例均基于公开 API，为示意性而非生产级实现。文中涉及的各厂商技术参数截至 2026 年 7 月。*

*本文为个人博客技术笔记系列的第三篇。本系列从 AI 计算系统的不同层面递进展开：首篇 [《GPU、TPU 与 LPU：AI 时代计算架构的分野》](/notes/GPU_vs_TPU/) 从单芯片架构出发，对比了三种加速器的设计哲学与"通用性 vs 专用性"的权衡；第二篇 [《GPU 卡间互联协议深度调研》](/notes/GPU_Interconnect_Research/) 将视角扩展至芯片之间的物理互联，拆解了 scale-up 与 scale-out 两层的协议与交换芯片竞争格局。本文则进一步上移至软件栈层面，以 CUDA 为主线拆解从框架到硬件的全路径，并分析国产 AI 芯片在不同架构路线下兼容该生态的挑战与策略。三篇文章共同构成对 AI 计算系统"硬件架构 → 硬件互联 → 软件栈"的完整分析框架。*
