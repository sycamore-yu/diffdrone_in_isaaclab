# Core Beliefs

`diffaero_newton` 当前文档和代码应遵守以下信条：

1. 先写清 current reality，再写 target direction。
2. `loss` 和 `reward` 是两个通道，不是一个标量的两种叫法。
3. 数值推进属于 `dynamics/`，任务几何属于 `tasks/`，不要把两者塞回环境壳层。
4. 先保持 direct workflow 清晰，再考虑 manager-based abstraction。
5. 本地 `direct_rl_shim` 是过渡 runtime，不是 IsaacLab 全兼容层。
6. reference parity 只按任务语义和接口契约对齐，不按文件结构或工具链逐字复制。
7. 文档不能把 deferred 能力写成 supported surface。
8. 任何新能力都必须进入 registry、config 和验证门禁，而不是只落一段代码。
