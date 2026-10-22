# Workspace v1 离线验收

验收通过。原始记录保存在 [workspace_acceptance_20261022_v5](../runs/workspace_acceptance_20261022_v5/report.md)，
计数和实现哈希见 [acceptance.json](../runs/workspace_acceptance_20261022_v5/acceptance.json)。

| 检查 | 结果 |
|---|---|
| 全量 pytest，包含现有回归 | 631 通过，0 失败，0 错误，0 跳过 |
| 真实 CLI | Docker 内 Codex CLI 0.162.1；工具、持久会话、指定 ID 恢复、压缩与重试计数通过 |
| 模拟完整生命周期 | 2 次独立重复、120 个模拟目标槽位；Stage1 使用缩减测试规模，Stage2 保持 5 轮、4 候选、5 共享场景 |
| 预算与故障注入 | 并发额度、重复提交、提交崩溃、执行前后断线、已保存结果恢复、未完成任务禁止重跑通过 |
| 隔离与完整性 | 阶段越权、目标/评分/其他运行/凭据访问、模型哈希修改、非法附着、非有限控制量和缺失证据检查通过 |
| 真实 MuJoCo 模型检查 | 原始模型行为逐步一致；几何、摩擦、时长和控制响应修改实际生效；不能应用到目标引擎 |
| Preparation 验收 | 独立 MuJoCo worker，编译、RobotAPI、实际几何、RGBD 检查通过；进程正常关闭 |
| 人工 `direct_plate` | MuJoCo 3.3.7、640×480，`released_placement_v1` 八场景 8/8 |
| 模拟推理请求，含故障与压力测试 | 1321 |
| **真实模型请求** | **0** |
| **Isaac 目标执行** | **0** |

测试明细见 [pytest.log](../runs/workspace_acceptance_20261022_v5/pytest.log) 和
[JUnit XML](../runs/workspace_acceptance_20261022_v5/pytest.xml)。其中故障注入夹具会故意修改哈希或留下中断；
这些目录用于检验拒绝行为，不应当作正式研究运行恢复。

人工能力的八场景是 `far_plate` 种子 42、9、10、11、12、13，以及 `far_to_near` 和
`shifted_single` 各种子 42。每次均保留独立策略、版本、动作轨迹及视频；八个视频的路径和内容哈希均不同。
详见[源域报告](../runs/workspace_acceptance_20261022_v5/supplied_source/report.md)。该运行的完整性检查
通过 18 项产物核对，无活动操作；验收结束后没有遗留仿真 worker 或研究容器。

本次实现清单的 SHA-256 为 `143fdeab23f053da3cdc1e2b2801c6229e9af42fbed8e0c40e14ff855bb3ca77`。
验收开始和结束时实现清单一致；后续源码变化应重新验收，不能沿用本记录证明新代码。
此前被中断或失败的验收目录保留，未与本次成功结果混合统计。

复验命令（必须使用新目录）：

```bash
.venv/bin/python scripts/build_research_image.py
.venv/bin/python scripts/accept_workspace.py --out runs/workspace_acceptance_NEW
.venv/bin/python -m simex verify --run runs/workspace_acceptance_NEW/supplied_source
```

本记录证明工程流程、隔离、恢复以及人工技能接入通过离线检查。自主 Stage1 学习效果、真实
MuJoCo → Isaac 的 Stage2 迁移效果和相对基线改善，仍需后续独立正式实验验证；本次没有启动这些实验。
