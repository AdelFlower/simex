# SimEX：Panda / MuJoCo → Isaac Sim 复现框架

新协议 `simex_workspace_v1` 覆盖 **Preparation → Stage1 能力发现 → Stage2 源模型代码校准与修复 → 冻结评估 → 报告**。
借鉴 EmbodiedEvo 的研究会话、工作区、经验记录、执行留档与恢复机制，保持 SimEX 的单轮无工具执行策略边界。

当前改造只做离线验收。模拟服务结果、人工 `direct_plate` 回归不能作为自主 Stage1 学习或 Stage2 迁移成功。
本次[离线验收记录](docs/offline-acceptance-workspace-v1.md)：631 项测试通过，人工技能 MuJoCo 八场景 8/8，真实模型请求与 Isaac 目标执行均为 0。
旧运行保持独立；历史说明见 [历史 README](docs/history/README-before-workspace-v1.md)。
方法与差异见 [协议说明](docs/workspace-protocol.md)。

## 环境与离线验收

沿用仓库 `.venv`、Panda 源资产、MuJoCo 3.3.7 和已安装的 Isaac Sim。研究容器不包含模拟器、目标代码或凭据。
固定 CLI 版本来自 `.tools/package-lock.json`，不是系统全局 `codex`。

```bash
cd .tools
npm ci
cd ..
.venv/bin/python scripts/build_research_image.py
.venv/bin/python scripts/accept_workspace.py --out runs/workspace_acceptance_NEW
```

验收脚本运行完整 pytest、真实 CLI 加模拟模型服务、故障注入，以及真实 MuJoCo 的
`direct_plate` 八场景释放放置回归。脚本禁止真实模型提供方和 Isaac 工作进程启动。
结果保存到指定的新目录；`acceptance.json` 明列真实模型请求和 Isaac 目标执行计数。
测试轨迹与单次执行视频保留在该目录内，不能并入正式学习统计。

单独检查代码：`.venv/bin/python -m pytest -q`。

## 新自主实验的启动配置

默认 [configs/workspace.yaml](configs/workspace.yaml)：4 个成员、每成员 20 周期，
每周期 28 探测、10 聚焦、5 回放场景。Stage2 每任务 5 轮、4 候选、5 个共享源场景。
三个任务、三种方法、5 次独立重复，共 300 个目标槽位。

必须准备以下输入，程序不会默认将缺失观测当作准确重建：

- 冻结的公开 RGBD NPZ，字段 `rgb`、`depth`、`intrinsics`、`camera_to_world`，640×480；禁止 pickle。
- 工作区文字说明和允许使用的源资产目录。
- 宿主模型服务的 Responses 地址 `model_gateway_url`，以及环境变量 `SIMEX_MODEL_API_KEY`。
- **显式 `model_request_limit` 或 `--model-request-limit`**。没有默认预算，不继承旧实验额度。

在新配置副本中填入这些字段。以下命令是后续正式实验入口，本次交付不执行它们：

```bash
.venv/bin/python -m simex prepare --config configs/my_workspace.yaml --run runs/my_new_study --model-request-limit 60000
.venv/bin/python -m simex discover --config configs/my_workspace.yaml --run runs/my_new_study --model-request-limit 60000
.venv/bin/python -m simex adapt --config configs/my_workspace.yaml --run runs/my_new_study --model-request-limit 60000
.venv/bin/python -m simex evaluate --config configs/my_workspace.yaml --run runs/my_new_study --model-request-limit 60000
```

数字 `60000` 只是显式配置示例，不是完成论文规模的保证；实际请求达到上限即停。
也可用 `run` 串行完成所有阶段和重复。冻结后配置、实现、输入或资产改变，都必须开新目录。
真实目标执行始终由 Isaac Sim 完成；源模型校准与候选筛选始终在 MuJoCo 完成。

## 人工能力诊断入口

[configs/workspace_supplied.yaml](configs/workspace_supplied.yaml) 只比较 `skill_transfer`、
`skill_stage2`，一次独立运行 45 个目标槽位。它明确使用人工能力，不验证 Stage1 自主发现。

```bash
# 只做 MuJoCo 八场景接入验收；没有模型请求和目标执行。
.venv/bin/python -m simex stage2 --config configs/workspace_supplied.yaml --run runs/my_supplied_check --source-only
```

两条入口共享 Stage2 的代码模型校准、固定脚本回放、四假设筛选和冻结评估。
源模型提供 XML、命令时长、逐步控制三个扩展点；真实物理步进、观测、RobotAPI 和评分由宿主管理。
后续正式运行需另建目录，配置模型服务并显式设定请求上限。

## 状态、审计与恢复

```bash
.venv/bin/python -m simex status --run runs/my_new_study
.venv/bin/python -m simex verify --run runs/my_new_study
.venv/bin/python -m simex report --run runs/my_new_study
.venv/bin/python -m simex stop --run runs/my_new_study
.venv/bin/python -m simex resume --run runs/my_new_study
```

`status`、`verify` 可读旧记录；旧格式的详细审计仍可使用 `scripts/verify_run.py`。
`scripts/status.py` 和 `scripts/verify_run.py` 也是新协议的兼容入口。
`resume` 只支持新协议，先核对冻结哈希与执行台账，不接受已开始但结果丢失的自动重跑。

研究工作区只读规则和公开证据，可写笔记与当前阶段草稿。持久研究代理通过
`status / submit / test / commit / finish` 操作；未测试草稿不能成为最终能力。
每个实际推理请求单独计数，包括会话内部请求、压缩和重试。执行策略生成器始终是新的单轮文本请求。

报告自动刷新并区分：预置技能源域验证、Stage1 流程与子能力证据、完整源任务成功、
Stage2 流程完成、目标成功和相对基线改善。每条执行记录关联视频、实际策略、能力版本、
源模型版本、引擎版本和 `released_placement_v1` 评分版本。

## 主要实现

| 文件 | 职责 |
|---|---|
| `simex/lab/protocol.py`, `records.py` | 冻结配置、版本、配对目标槽位、原子请求和操作台账 |
| `gateway.py`, `session.py`, `workspace.py` | 宿主模型网关、持久 CLI、阶段权限、证据与提交 |
| `preparation.py`, `discovery.py` | 公共观测初始化、独立源域能力发现和选拔 |
| `simex/stage2.py` | 两条入口共用的源模型校准、候选筛选和冻结评估 |
| `execution.py`, `reporting.py`, `campaign.py` | 物理执行去重、生命周期、审计和报告 |
| `simex/lab/mock.py` | 明确标记的离线模拟服务和物理测试替身 |

`ref/EmbodiedEvo` 是只读参考，没有运行时依赖。其值得借鉴的设计及未采用的差异，
见 [来源说明](docs/workspace-protocol.md#借鉴来源)。
