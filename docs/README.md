# Docs Guide

`docs/` 现在采用“少量主入口 + 分层子目录”的结构。

## Primary entrypoints

- `development.md`
  - 目标态架构、接口约束、工程方向
- `progress.md`
  - 当前实现事实、验证证据、已知范围缺口
- `regression_matrix.md`
  - 紧凑验证矩阵
- `harness.md`
  - 当前 harness contract、参考实现决策、实现边界

## Directory map

```text
docs/
├── design-docs/
├── exec-plans/
│   ├── active/
│   └── completed/
├── product-specs/
├── references/
├── development.md
├── harness.md
├── progress.md
└── regression_matrix.md
```

## What goes where

- `design-docs/`
  - 稳定设计原则、架构索引、长期有效的设计判断
- `exec-plans/active/`
  - 需要跨多次会话保留的执行计划
- `exec-plans/completed/`
  - 已完成的审计、整改、准备文档
- `product-specs/`
  - 对外表面、CLI、训练 harness 能力说明
- `references/`
  - 本地 reference 代码映射、官方最佳实践摘录

## Rules

- Update `progress.md` whenever capability status or validation confidence changes.
- Update `development.md` when target architecture or public contracts change.
- Update `harness.md` when runtime boundary, env step contract, registry surface, or task geometry scope changes.
- Use `bd` for work tracking. `docs/` is not a task tracker.
