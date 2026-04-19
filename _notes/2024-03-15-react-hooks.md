---
title: "React Hooks 最佳实践"
date: 2024-03-15
category: "前端"
tags: ["React", "JavaScript"]
description: "总结 React Hooks 的使用技巧和常见陷阱"
---

# React Hooks 最佳实践

## useEffect 依赖项管理

```jsx
// ❌ 错误：缺少依赖项
useEffect(() => {
  fetchData(userId);
}, []); // userId 变化时不会重新获取

// ✅ 正确：包含所有依赖
useEffect(() => {
  fetchData(userId);
}, [userId]);
```

## 自定义 Hooks

将重复逻辑抽取为自定义 Hook：

```jsx
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : initialValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}
```

## 常见陷阱

1. **闭包陷阱**：在异步操作中使用过期的 state
2. **无限循环**：useEffect 中 setState 导致重新渲染
3. **依赖数组遗漏**：导致副作用不按预期执行

## 推荐规则

- 使用 `eslint-plugin-react-hooks` 自动检查依赖项
- 将复杂逻辑拆分为多个 useEffect
- 使用 `useCallback` / `useMemo` 谨慎优化，避免过早优化
