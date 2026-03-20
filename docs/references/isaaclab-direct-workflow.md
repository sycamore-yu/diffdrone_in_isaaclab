# IsaacLab Direct Workflow Notes

来源：Context7 `IsaacLab` 文档摘要，聚焦自定义 `DirectRLEnv` 任务的推荐实践。

## 官方建议的关键点

### 1. 配置使用 `@configclass`

自定义 RL 环境配置通常继承 `DirectRLEnvCfg`，并显式声明：

- `decimation`
- `episode_length_s`
- `action_space`
- `observation_space`
- `state_space`
- `sim: SimulationCfg`
- `scene: InteractiveSceneCfg`

这与本仓想要的“配置即 contract”方向一致。

### 2. 显式声明 scene replication

官方示例明确使用：

```python
InteractiveSceneCfg(
    num_envs=...,
    env_spacing=...,
    replicate_physics=True,
)
```

含义不是装饰性配置，而是把“同构并行环境复制”作为标准工作流的一部分。

### 3. 语义未稳定前优先 direct workflow

IsaacLab 官方对 direct workflow 的定位很清楚：

- 由任务类直接实现 reward、observation、done/reset
- 代码路径更短，更容易保持梯度与状态边界清晰
- 比 manager-based workflow 更适合语义仍在快速收敛的任务

这和本仓当前阶段完全一致。

### 4. Step/reset contract 要显式

官方示例通常把下列方法拆开实现：

- `_pre_physics_step`
- `_apply_action`
- `_get_observations`
- `_get_rewards`
- `_get_dones`

本仓即使继续使用本地 shim，也应保持这一拆分习惯。

## 对本仓的直接结论

应继续保留：

- direct workflow
- 显式 config
- 显式 observation/action/state space

应避免：

- 过早抽象成 manager-based env
- 让 shim 伪装成完整 IsaacLab runtime
- 依赖运行期推断空间维度或隐式配置
