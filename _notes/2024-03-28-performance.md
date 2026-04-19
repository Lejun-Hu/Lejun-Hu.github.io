---
title: "前端性能优化笔记"
date: 2024-03-28
category: "前端"
tags: ["Performance", "Web"]
description: "总结前端性能优化的核心策略和实践经验"
---

# 前端性能优化笔记

## 核心指标

- **LCP** (Largest Contentful Paint): 最大内容绘制，目标 < 2.5s
- **FID** (First Input Delay): 首次输入延迟，目标 < 100ms
- **CLS** (Cumulative Layout Shift): 累积布局偏移，目标 < 0.1

## 优化策略

### 1. 资源加载优化

- 使用 `preload` 预加载关键资源
- 图片懒加载：`loading="lazy"`
- 使用 WebP 格式替代 JPEG/PNG
- 启用 Gzip/Brotli 压缩

### 2. 代码优化

- 代码分割（Code Splitting）
- Tree Shaking 移除未使用代码
- 延迟加载非关键 JavaScript

### 3. 缓存策略

- 合理使用浏览器缓存
- 使用 Service Worker 实现离线缓存
- CDN 加速静态资源

## 推荐工具

- Lighthouse — Chrome 内置性能审计工具
- WebPageTest — 详细的性能测试报告
- Chrome DevTools Performance 面板
