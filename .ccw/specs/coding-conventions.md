---
title: Coding Conventions
readMode: optional
priority: medium
category: general
scope: project
dimension: specs
keywords: [general, python, style, tensors, configs]
---

# Coding Conventions

- Target Python 3.11 and keep code compatible with the repo Ruff profile in `pyproject.toml`.
- Keep Ruff-clean imports and style under `E`, `F`, `I`, and `UP`, with a 100-character line limit.
- Use `from __future__ import annotations` in new Python modules.
- Add explicit type hints on public functions, methods, dataclass fields, and config surfaces.
- Use `cfg` for configuration objects. Do not introduce new public `config` synonyms.
- Use IsaacLab-style `@configclass` configs for environment, task, and sensor configuration objects.
- Declare `gymnasium.spaces.Box` or equivalent spaces explicitly in config and keep them synchronized with emitted tensors.
- Keep public environment names stable: `obs`, `state`, `reward`, `terminated`, `truncated`, `extras`.
- Keep public rollout and training tuple order stable: `obs, state, loss_terms, reward, extras`.
- Document every public tensor-returning API with shape, dtype, units when physical, and device semantics.
- Prefer `None` defaults for optional config or transform inputs instead of mutable constructed defaults.
- Keep comments short and about intent or invariants, not line-by-line narration.
