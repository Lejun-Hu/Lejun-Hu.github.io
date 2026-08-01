---
title: "【未完成】CUDA软件栈全路径拆解与国产AI芯片生态兼容深度分析"
permalink: /notes/cuda-ecosystem-compatibility/
date: 2026-08-15
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

> **阅读提示**：上面这段出现了几个可能不太熟悉的名词——eager 模式、图编译、Inductor、CUTLASS、Triton。不用担心，本文后续章节会逐一深入介绍它们。这里先给一个速览表，方便你快速建立印象：
>
> | 名词 | 简要含义 | 本文详细位置 |
> |------|---------|------------|
> | **eager 模式** | PyTorch 的默认执行方式：每写一行 Python，框架就立即调用底层算子执行，不做全局优化。类似于"解释执行"。 | §1.1 本节 |
> | **图编译模式** | 先将整个计算过程记录为一幅计算图，再对整图做优化（如算子融合），最后一次性生成并执行高效代码。 | §1.8 |
> | **cuBLAS** | NVIDIA 提供的闭源 GPU 线性代数加速库，是 PyTorch 在 CUDA 设备上执行矩阵乘法的默认后端，内部包含针对不同 GPU 架构高度手工调优的 kernel。 | §1.1 本节 |
> | **Inductor** | PyTorch 2.0 `torch.compile` 的默认后端编译器，负责将计算图转化为 Triton/C++ kernel。 | §1.8 |
> | **CUTLASS** | NVIDIA 开源的 CUDA C++ 模板库，提供了高度可定制的 GEMM/Conv 等算子模板，适合作为自定义 kernel 的性能参照。 | §1.1 表格 |
> | **Triton** | OpenAI 开源的 GPU kernel 编程语言与编译器，用 tile 级抽象替代 CUDA 的线程级编程，也是 Inductor 默认的 kernel 生成后端。 | §1.7, §3.2 |

这里的第一个关键概念是 **算子（Operator）**。在深度学习框架中，"算子"指代对一个或多个张量执行的特定数学运算——矩阵乘法（matmul）、卷积（conv2d）、逐元素加法（add）、归一化（layernorm）等均属于算子。PyTorch 中有超过 2,000 个 ATen 算子，每个算子需要针对不同设备的多个后端（CPU / CUDA / MPS / XPU 等）提供独立的实现。对于 CUDA 后端，绝大多数算子的实现并不是手写的 CUDA C++ kernel——PyTorch 团队会优先调用 NVIDIA 提供的**闭源高性能算子库**，因为经过二十年代代优化的闭源库性能远超通用 kernel 实现。具体来说：

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

NCCL 则是唯一在此层面开源的关键库（BSD 许可）。它提供了 `ncclAllReduce`、`ncclBroadcast`、`ncclAllGather` 等集合通信原语，支持 ring、tree、collnet 等多种算法变体。虽然 NCCL 源码公开，但其传输层深度绑定了 NVLink（GPU 间直连协议）、NVSwitch（交换芯片，将多颗 GPU 编织成全互联域）和 InfiniBand（跨节点 RDMA 网络协议）等 NVIDIA 自有互联硬件，要在不同硬件上移植必须重写整个传输层。

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
- **`cudaMalloc` / `cudaMemcpy` / `cudaFree`**：CUDA Runtime API，管理 GPU 显存的分配、主机-设备数据传输和释放。这些是 `libcudart.so` 动态库提供的高层封装。
- **`<<<grid, block>>>` 语法糖**：CUDA 特有的 kernel 启动语法，它会在编译时被 nvcc 展开为实际的 `cudaLaunchKernel` Runtime API 调用。
- **`cudaDeviceSynchronize`**：阻塞主机端直到 GPU 所有已提交任务完成，是最基本的同步原语。

### 1.3 编译链路第一站：nvcc 前端 —— CUDA C++ → PTX

上面的 `.cu` 源文件如何变成 GPU 可执行的代码？答案是通过 **nvcc**（NVIDIA CUDA Compiler Driver）进行离线编译。nvcc 并非一个单一的编译器，而是一个 **编译器驱动程序**——它在内部协调多个子工具完成不同的编译阶段。

```bash
# 编译为 PTX 中间表示（文本格式，可读）
nvcc -ptx vector_add.cu -o vector_add.ptx

# 编译为 CUBIN 二进制（针对特定 GPU 架构）
nvcc -cubin vector_add.cu -arch=sm_80 -o vector_add.cubin

# 标准方式：生成包含多个架构代码的宿主可执行文件
nvcc vector_add.cu -o vector_add -arch=sm_80
```

nvcc 的第一步是将 `.cu` 文件中的**设备代码**（`__global__`、`__device__` 标记的函数）与**主机代码**（普通的 `main()` 函数等）分离。设备代码被送往 GPU 编译器前端，经过词法分析、语法分析、语义分析和优化后，生成 **PTX（Parallel Thread Execution，并行线程执行）**——这是一种文本格式的虚拟 ISA（指令集架构）汇编代码。

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
    setp.ge.s32  %p1, %r4, %r5;  // p1 = (r4 >= r5)，设置条件谓词
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
- **架构特异性**：SASS 中出现了 `c[0x0][0x28]` 这样的常量内存寻址、`SR_TID.X` 这样的专用特殊寄存器、`@P0 EXIT` 这样的谓词化执行——这些都是 Ampere 架构特有的硬件特性，在其他代 GPU（如 Hopper 的 sm_90）上编码格式和可用指令都不同。**不同 SM 版本之间的 SASS 二进制不兼容。**
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

### 1.6 运行时路径：Runtime API 与 Driver API 的分工

所有离线编译的产物（FATBIN/CUBIN/PTX）最终需要在程序运行时被加载到 GPU 上执行。CUDA 提供两套 API 来完成这一过程，它们之间存在明确的分层关系。

**CUDA Runtime API**（`libcudart.so`）是高层封装，提供了 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel` 等开发者最常用的接口。它自动管理 CUDA context（设备的"进程"抽象）和 module（设备的"动态库"抽象）的创建与加载，使 GPU 编程的代码量大幅减少。

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

NVRTC 以库的形式（`libnvrtc.so`）工作，不需要启动外部 `nvcc` 进程，避免了进程创建、磁盘 I/O 等开销。输入端接受 CUDA C++ 字符串和编译选项，输出端提供 PTX 字符串（也可以直接输出 CUBIN）。从 CUDA 12.0 开始，NVRTC 还能生成 **LTO-IR（Link-Time Optimization Intermediate Representation，链接时优化中间表示）** 以配合 **nvJitLink** 运行时链接器，实现跨 kernel 模块的链接时优化——这一机制对 Triton 等 AI kernel 编译器至关重要，因为它允许在运行时将多个独立编译的 kernel 模块链接为一个高度优化的统一 kernel。

### 1.8 CUDA Graph 与算子融合：超越逐 Kernel 执行的优化

前面的讨论都建立在"逐 kernel 启动执行"的 eager 模式之上。但在现代 GPU 计算中，逐个 kernel 启动的开销常常超过实际计算本身——尤其是对于大量小规模 kernel 组成的模型（例如含有大量逐元素操作的 Transformer），CPU 端每次 kernel launch 的提交延迟、GPU 侧的线程块调度开销和中间张量的 HBM 读写构成了巨大的隐性浪费。这一矛盾催生了两个层面的优化技术：**算子融合** 和 **CUDA Graphs**。

**算子融合（Operator Fusion）** 由框架编译器在离线或 JIT 阶段完成。以 `Conv2d → BatchNorm → ReLU` 这一常见组合为例，在 eager 模式下，这三个操作会产生三次 HBM 读写（每个操作的结果都要写回 HBM 再由下一个操作读取）。而融合后，编译器将其合并为一个 kernel——输入数据从 HBM 读取一次，在片上完成卷积、归一化和激活函数的全部计算，最终结果写回一次。在现代 GPU 上，HBM 带宽是主要瓶颈（H100 为 3.35 TB/s），而非计算吞吐（H100 的 BF16 算力达 1,979 TFLOPS），这种融合能带来数倍的性能提升。

**CUDA Graphs** 则从另一个角度解决开销问题：它将一系列 CUDA 操作（kernel 启动、memcpy、event 等）捕获为一个有向无环图，之后通过一次 API 调用重放整个图。这消除了逐 kernel 启动的 CPU 端提交延迟，且允许 GPU 驱动提前规划整个执行序列的资源分配。

```cpp
// CUDA Graph 的基本用法：捕获 → 实例化 → 重放
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 在此之间提交的所有 CUDA 操作都会被记录到图中
kernel_A<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(..., stream);
kernel_B<<<grid, block, 0, stream>>>(...);

cudaStreamEndCapture(stream, &graph);          // 结束捕获，获得图句柄
cudaGraphInstantiate(&instance, graph, 0);     // 实例化（编译优化图结构）
cudaGraphLaunch(instance, stream);             // 一次调用重放全部操作
```

CUDA Graphs 的反直觉之处在于，在 DSA 架构中这一机制是天然契合的——DSA 编译器天生就是将整个计算图静态编译为执行计划，图捕获对其而言几乎不需要额外工作。而在 GPGPU 的动态 warp 调度模型下，图捕获反而需要额外的运行时机制来拦截和记录 kernel 提交、管理静态内存地址的复用。

从 PyTorch 2.0 开始，`torch.compile` 默认启用图模式编译，Inductor 后端结合 Triton 做 kernel 生成和 CUDA Graph 做图重放，将上述两个优化统一到一个编译流水线中。对于国产芯片的兼容方案而言，最关键的战场因此不在于逐 kernel 的 API 映射，而在能否在 Inductor 后端上提供自有硬件的代码生成能力。

### 1.9 全链路总览

将以上八个节的内容串联起来，可以得到 CUDA 软件栈从框架到硬件的完整执行链路：

| 阶段 | 输入 | 输出 | 关键工具/库 | 开源状态 |
|------|------|------|-----------|---------|
| ① 框架层调度 | Python 框架代码 | 对 cuBLAS/cuDNN/NCCL 等的函数调用 | PyTorch dispatcher | 开源 |
| ② 算子库执行 | 矩阵乘/卷积/通信调用 | GPU 计算结果 | cuBLAS, cuDNN, cuFFT (闭源); NCCL, CUTLASS (开源) | 部分闭源 |
| ③ CUDA kernel 编写 | `.cu` 源文件 | CUDA C++ AST | nvcc, NVRTC | 编译工具闭源 |
| ④ 前端编译 | CUDA C++ AST | PTX 文本/字符串 | nvcc GPU 前端, NVRTC | 闭源 |
| ⑤ 后端汇编 | PTX | SASS / CUBIN (ELF) | ptxas (离线) 或 驱动 JIT 编译器 | 闭源 |
| ⑥ FATBIN 打包 | 多个 CUBIN + PTX | FATBIN (嵌入宿主可执行文件) | nvcc --fatbin | 闭源 |
| ⑦ Runtime 加载 | FATBIN / CUBIN / PTX | 已加载到 GPU 上下文的 CUmodule | libcudart, libcuda (+ Dark API) | 闭源 |
| ⑧ 图优化/算子融合 | 计算图 (FX/HLO) | 融合后的图 + 优化 kernel | torch.compile, TensorRT, CUDA Graphs | 部分闭源 |
| ⑨ GPU 硬件执行 | SASS 指令流 | 计算结果 | SM, Tensor Core, HBM, NVLink | 硬件 |

---

## 二、从通用视角看兼容难度：DSA 与 GPGPU 的本质分歧

在了解了 CUDA 软件栈的完整结构之后，一个自然的问题是：如果我们要造一颗全新的 AI 芯片并希望兼容这套生态，面临的核心障碍是什么？

这个问题没有简单的"难"或"不难"的回答——因为不同的硬件架构路线，面临的障碍性质完全不同。本文将国产 AI 芯片的技术路线分为两大阵营：**GPGPU**（通用 GPU 路线）和 **DSA**（领域专用架构路线），分别分析它们与 CUDA 生态兼容的根本性挑战。

### 2.1 GPGPU 路线的兼容挑战：软件栈规模，而非语义鸿沟

GPGPU 路线的硬件执行模型与 NVIDIA GPU 高度同构——都是 SIMT（单指令多线程）并行计算架构，以 warp/线程为基本调度单位，拥有类似的 L1/L2 缓存层级和 HBM 显存体系。因此，GPGPU 路线在与 CUDA 生态兼容时，面临的主要挑战集中在**软件栈的工程规模**而非编程模型的语义错配。我们可以将其问题分解为以下几个层级：

**Runtime 和 Driver API 的完整复刻。** CUDA Runtime API（`libcudart.so`）提供了数百个函数（`cudaMalloc`、`cudaMemcpy`、`cudaStreamCreate`、`cudaEventRecord`、`cudaLaunchKernel`、`cudaGraphCreate`……），Driver API（`libcuda.so`）提供了更多底层函数（`cuInit`、`cuCtxCreate`、`cuModuleLoadData`……）。任何一个函数的行为偏差都可能导致现有程序运行错误。更棘手的是，`libcuda.so` 中还存在大量通过 `cuGetExportTable` 暴露的 Dark API——没有文档、没有签名、仅在特定 NVIDIA 库的内部运行时环境中被调用。一个可行的策略是"按需逆向"：追踪 PyTorch、vLLM 等主流框架实际触达的 Dark API 函数，逐函数通过反汇编和调试推测其语义并实现兼容版本，而不追求全量逆向。

**闭源算子库的 API 级干净室重实现。** cuBLAS、cuDNN、cuFFT、cuSPARSE、cuRAND 全是闭源动态库。这意味着无法像 NCCL 那样复用其开源算法实现——必须提供同名的自有 `.so` 文件（`libcublas.so`、`libcudnn.so` 等），使用 NVIDIA 公开的头文件（定义了全部函数签名和枚举）保证编译兼容，在内部用自研 kernel 实现完全相同的数学语义。这个过程被称为 **clean-room reimplementation（干净室重实现）**：API 和数学行为严格对标，内部实现完全独立。摩尔线程的 muDNN 对标 cuDNN、muBLAS 对标 cuBLAS，即采用此策略，并宣称关键算子的效率可达对应产品的 98% 以上。

**PTX 兼容层的取舍。** GPGPU 路线理论上可以在 PTX 层做二进制兼容——实现一个 PTX 到自有 ISA 的编译器，在运行时截获 FATBIN 中的 PTX 代码并 JIT 编译到自有硬件上执行（如 ZLUDA 在 AMD GPU 上的做法）。这一方案可以让已编译的 CUDA 二进制文件不经修改直接运行（前提是二进制中保留了 PTX 代码，而非仅有 SASS）。但 PTX 虽然号称"虚拟 ISA"，其设计仍然深度耦合于 NVIDIA 硬件——warp size=32 的线程模型、shared memory 的 bank 结构、Tensor Core 的 wmma/mma 指令，这些在另一个 GPU 微架构上不可能一一直接对应。PTX 级兼容的实际性能上限因此受限于"多对一"的硬件语义映射效率损耗。

总结来说，GPGPU 路线面临的是**巨大的工程规模问题**——软件栈的每一层都需要完整的重实现或映射，但不存在根本性的语义错配。摩尔线程 MUSA 对标的正是这条路，也是国产 GPU 中兼容程度最高的一家。

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

这解释了为什么纯粹的 DSA 架构（Google TPU、Groq LPU）没有试图直接兼容 CUDA 的线程级编程模型——它们在硬件设计之初就决定放弃这一兼容性，转而建立自己的全栈生态。华为昇腾走的则是"有限兼容"路线——在框架层（PyTorch PrivateUse1）和算子库层（AscendCL）提供 CUDA-like 的 API 体验，但在 kernel 编程层（算子开发）提供完全不同的 Ascend C 语言，要求开发者接受 SPMD + 显式数据流编程而非 CUDA 的线程模型。

对于国产自研芯片而言，这是一个无法回避的战略抉择：**选择 GPGPU 路线，可以用 API 级兼容 + 库重写的工程规模换取对 CUDA 生态的最大覆盖（代价是硬件能效比不可能达到 DSA 的极致水平）；选择 DSA 路线，可以获得针对 AI 负载的确定性低延迟和极致能效，但必须接受与 CUDA 的线程级编程模型之间的语义不可兼容边界。**

---

## 三、案例一：华为昇腾 CANN —— DSA 架构的"有限兼容"实践

华为昇腾系列（Ascend 310/910/950）采用自研的**达芬奇（DaVinci）架构**，是国产 AI 芯片中最具代表性的 DSA 路线案例。接下来逐一解析 CANN 软件栈的每一层是如何对标 CUDA 生态、实现了什么以及放弃了什么。

### 3.1 硬件基础：达芬奇 AI Core —— 与 NVIDIA SM 的截然不同

昇腾 AI 处理器的计算核心称为 **AI Core**，一颗芯片上通常有数十个 AI Core 独立执行计算任务。每个 AI Core 内部有三类功能严格划分的计算单元：

- **Cube Unit（矩阵计算单元）**：专做矩阵乘法。每时钟周期可完成一个 16×16×16 的 FP16 矩阵乘加运算（等效 8,192 次 MACs）。这是达芬奇架构的计算主力，对标 NVIDIA 的 Tensor Core 但面积占比更大、更专精。
- **Vector Unit（向量计算单元）**：负责逐元素运算（ReLU、GELU、LayerNorm、Softmax 等激活函数和归一化操作），单次处理 128 个 FP16 元素或 64 个 FP32 元素。
- **Scalar Unit（标量计算单元）**：类似一个小型 CPU，负责循环控制、分支判断、地址计算以及向 Cube/Vector 发射指令。

三类计算单元**并行独立工作**——Cube 在算矩阵乘的同时，Vector 可以并行处理激活函数，Scalar 准备下一批数据的地址。这与 NVIDIA SM 的设计哲学截然不同：SM 中的 CUDA Core 是通用计算单元，FP32/INT ALU、Tensor Core、Shared Memory 等资源由 warp scheduler 动态分配给活跃的 warp，不预设"哪类操作必须在哪个单元上执行"。

存储层次同样差异显著。达芬奇架构的存储层级不仅是多层级的，更是**功能绑定的**——L0A/L0B/L0C 仅供 Cube 使用，UB（Unified Buffer，约 256KB/核）仅供 Vector 使用。这与 CUDA 中 Shared Memory 可被任意线程自由访问的模型完全不同。这一设计带来更高的能效和确定性，但也意味着内存管理和数据流调度**必须由开发者或编译器显式规划**，而无法依赖硬件自动缓存。

### 3.2 软件栈逐层映射：CANN 如何对标 CUDA

CANN（Compute Architecture for Neural Networks）是昇腾的全栈软件平台，自身分为五层，全面对标 CUDA Toolkit + CUDA-X 库 + GPU 驱动的组合功能。

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

### 3.3 昇腾方案的"兼容与不兼容"边界总结

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
