# SimEX 本地方法复现

这是根据随附论文实现的 **Franka 单臂、缩小预算的 MuJoCo → Isaac Sim 复现**。优化对象是 `toolbox.py` 与 `skill.md`；每次任务由模型生成新 Python 策略。实现不包含预训练 RL 策略或预写成功抓盘程序，也不承诺达到论文的成功率。

## 运行

本机依赖：RTX 4090、MuJoCo 3.3.7、Isaac Sim 4.5、`isaac-sim` conda 环境、bubblewrap、系统 ffmpeg、已有 Codex 登录。项目自带隔离 `.venv`，Codex 0.162.1 安装在 `.tools`，用于兼容本机配置的 `gpt-6-astra`。全局 CLI 不受影响。

安装脚本还需要已有 `uv`、Node/npm，默认借用 `motionbricks` 的 Python 3.10 创建虚拟环境；可用 `SIMEX_PYTHON` 指定解释器。脚本不会安装 Isaac、系统级依赖或本地 Franka USD。

```bash
# 重建项目依赖和固定版本 Panda 资产
bash scripts/setup.sh

# 双引擎、RGBD、动作、策略隔离、视频和模型连通检查
.venv/bin/simex doctor --run runs/preflight

# 完整小规模实验：有断点时从已有结果继续
.venv/bin/simex run --run runs/small_v2 --config configs/small.yaml

# 长实验后台启动或从断点恢复；已有进程时不会再次启动
bash scripts/run_background.sh runs/small_v2 configs/small.yaml

# 分阶段执行
.venv/bin/simex discover --run runs/small_v2 --config configs/small.yaml
.venv/bin/simex adapt --run runs/small_v2 --config configs/small.yaml
.venv/bin/simex evaluate --run runs/small_v2 --config configs/small.yaml
.venv/bin/simex report --run runs/small_v2
python3 scripts/status.py --run runs/small_v2
python3 scripts/verify_run.py --run runs/small_v2
```

审计前的 `runs/small_v1` 已停止，保留为旧版诊断记录。修复版默认使用 `runs/small_v2`；实现指纹变化后旧实验不能续跑，不能把两版结果拼接。

首次 Isaac 启动可能花几分钟编译 RTX 着色器；缓存后通常更快。`doctor` 中运动诊断没有搬盘，其 rollout 的任务分数为 `failed` 是预期，诊断本身另外检查 `passed`。

`configs/smoke.yaml` 是另一个更小的工程测试协议，使用单独运行目录；它的结果不能替代 `small.yaml`。手动 `rollout` 输出位于 `debug/`，不进入正式目标试验计数。

```bash
# 生成并执行一个源仿真策略，保留全部证据
.venv/bin/simex rollout --run runs/debug --engine mujoco --task far_plate

# 固定已有脚本回放，禁止再次生成策略
.venv/bin/simex rollout --run runs/debug --engine mujoco \
  --script path/to/policy.py --toolbox path/to/toolbox_dir --label replay
```

## 实验协议

Stage 1 使用两个独立成员，每成员三轮。被拒候选的代码、测试结果和拒绝理由会进入下一次修订提示，接受的基线保持独立。每轮 8 个探测任务、4 个聚焦任务、最多两次工具库修订和最多两个历史能力回放。代理生成任务、可执行 JSON 成功判据、代码修订及保留决定。仅根据冻结的源仿真选拔集选取成员。

独立源任务的策略生成最多并发三路，物理执行保持串行。选拔排除没有有效物理执行的成员，按成功数、有效执行数、有效执行平均步数依次比较；非法脚本的零步数不会获得选拔优势。

Stage 2 的三个任务分别从相同 Stage 1 结果开始，各最多 5 次 Isaac 试验。每次先固定脚本与当前工具库，在源仿真中校准；然后比较当前版本及两个候选，进行固定脚本回放和三个泛化任务测试，由代理选择修复。源模型校准每轮最多三次。同一任务后续轮次保留此前候选和选择理由；不同任务的适应历史不共享。非法工具库候选会消耗一次候选名额并记录失败，恢复时不会反复卡在同一提案。

目标场景重建只使用初始 RGBD 的颜色分割与反投影估计，保留标称几何与物理参数，不复制目标物体真值。可校准项是质量、摩擦、相机偏移和动作时长。`friction_scale` 统一缩放源仿真全场景的滑动摩擦（盘、桌面和夹指），使合成后的接触摩擦也按比例变化；`scale=1` 保留基线值。私有诊断记录实际接触参数，回归测试检查缩放的灵敏度。

| 项目 | small 配置 |
|---|---:|
| 独立研究运行 | 1 |
| 目标适应 | 3 任务 × 5 次 = 15 |
| 冻结评估 | 3 方法 × 3 任务 × 5 次 = 45 |
| 正式目标试验总数 | 60 |

三个方法为初始工具库 Direct CaP、Stage 1 直接迁移、完整 SimEX。共享评估种子，各次生成新策略；结果冻结后不反馈给优化代理。

任务是远侧单盘、远盘到近箱、箱体偏移后的单盘。绿色薄盘需抬升至少 5 cm，最终完整位于箱内且线速度低于 0.05 m/s，红盘需保持在箱外。该判据不检查持续稳定时间。没有物体吸附、瞬移或辅助固定约束。两引擎都使用真实电机、夹爪接触和重力；MuJoCo 中保留的指间耦合约束属于机器人本体。

当前冻结判据检查物体的位置、姿态、速度和历史抬升，不单独要求夹爪张开或退出接触。首轮已出现夹爪仍闭合但盘子满足箱内几何判据的记录，因此应将“几何入箱成功”和“完成释放放置”区分，不能只凭成功分数声称后者。实际审计注释保存在运行目录的 `audit_notes.json`。

## 接口与隔离

策略得到 `robot` 和 `tb` 两个对象。`RobotAPI` 只暴露：

```python
robot.observe(camera="overhead")  # RGB, metric depth, K, camera_to_world
robot.state()                    # joints, EE pose, gripper width, sim time
robot.ik(position, quaternion)    # scalar-first quaternion: w,x,y,z
robot.move_joints(joints, duration=1.0)
robot.move_ee(position, quaternion, duration=1.0)
robot.gripper(width, force=40.0, duration=0.5)
robot.wait(seconds)
```

米、弧度、世界 Z 向上；相机为 OpenCV 坐标。夹爪 `width` 是两指总开口，`force` 是每指的电机力上限（N），两端保持相同语义；该值不代表实际测得的接触力。策略单次动作时长为 0.01–15 s，`wait` 为 0–15 s；源控制时长校准在输入校验后应用，包含省略参数时的默认时长，`wait(0)` 不推进仿真。TCP 为夹指中心，手部局部偏移 `[0,0,0.1034]`，两个后端的关节顺序一致。

策略在无网络、最小文件挂载且清空继承环境的 bubblewrap 进程中执行。物理工作进程通过受信任中介只接受上述公开调用；评分器和真实物体状态不进入策略进程。Python AST 检查是辅助约束，实际隔离依赖操作系统命名空间和中介接口。

编码代理收到公开实验摘要及按实验分别标注的阶段图组。固定脚本回放和各候选的泛化场景均提供证据，不再全局截取四张图片；附件清单及哈希另行记录。项目、私有实现与运行数据在其挂载中隐藏。登录文件只读挂载，临时 CLI 状态在退出时销毁。所有策略生成是单次纯文本调用；检测到工具执行或非法策略即判失败。

## 结果与恢复

每个运行目录包含：

- `manifest.json`、`runtime.json`：预算、模型、版本、环境和冻结计划。
- `target_ledger.json`：预分配的目标试验及状态，防止重复执行。
- `discovery/`、`adaptation/`、`evaluation/`：各次脚本、候选、选择理由和结果。
- `frozen/`：初始、Stage 1 和各任务适应后的工具库及校验值。
- 每次 rollout 的 `public/`：MP4、关键帧、策略读取的 RGBD NPZ、机器人轨迹。
- 每次 rollout 的 `private/`：仅可信评估器使用的场景与真值；不作为代理证据。
- `agent_calls/`：逐次输入、输出、事件、实际图像附件及哈希、服务返回的 usage；超时保留已收到的部分输出。
- `report.md`、`report.json`：已完成/排定数量、成功率、适应曲线数据和失败状态。

相同命令会复用已完成结果。超时、非法策略、执行失败消耗排定试验；策略沙箱尚未连接时的启动故障、初始 RGBD 故障属于基础设施故障，可修复后恢复。连接后发生后端故障则将目标试验标为 `unresolved_infrastructure` 并停止研究；它保持未完成、不计入实测成功率，也不自动重跑。源侧基础设施中断也不会作为候选失败进入筛选。若崩溃后无法判断目标试验是否执行，流程停止并记录 `unresolved`，不会悄悄重跑。研究轮数、目标试验数、冻结工具库、场景种子和模型配置不可在恢复时更改；模型调用资源上限 `max_agent_calls` 可由操作者另行调整。

小规模实验仍可能调用模型数百次，默认上限 800 次。额度耗尽或基础设施错误会留下断点与明确错误，报告不会把未完成评估记成成功或零成功。

后台启动优先使用用户级 systemd 服务（名称保存在 `research.service`）；无可用用户服务时使用独立会话。进程号保存在 `research.pid`，控制台输出持续写入 `console.log`。后台脚本不会无限重试；进程因错误退出时，请先查看 `last_error.json` 和日志，再用相同启动命令恢复。`report.md` 在每个研究阶段完成或出错时更新，也可随时运行 `simex report` 手动刷新。

## 验证与复现差异

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest -q
# 独立真实 Isaac 测试（需要本机资产）
bash scripts/isaac_python.sh tests/smoke_isaac.py
.venv/bin/python tests/smoke_isaac_worker.py
```

新增回归测试覆盖拒绝后的反馈、非法候选恢复、每场景证据覆盖、摩擦灵敏度、夹爪力语义、任务参数边界和运行时故障分类。适配器测试使用假进程，不调用实验模型。

测试覆盖真实 MuJoCo RGBD/反投影、物理接触、IK不修改现场、动作、复位、录像、沙箱边界、严格评分及试验预算/冻结/回放协议。Isaac 的真实测试另行运行，避免常规测试重复占用 GPU。

与论文相比：单臂 Panda 替代双臂 YAM；三个可达改编任务替代七任务；两个成员/三轮替代四成员/二十轮；两个修复和三个筛选任务替代四个/五个；仅一次独立研究运行。相机、资产、控制器、初始工具库与模型也不同，因此报告用于检验方法链路和观察迁移现象，不与论文成功率作直接等价比较。

首版公开任务模板，留出评估场景种子；这与论文对代理隐藏完整评估任务定义的协议也有区别。

论文：随附 `2609.38982v1.pdf`。Panda 来源与许可证：`assets/panda/PROVENANCE.json`、`assets/panda/LICENSE`。本地 Isaac USD 不被重新分发，其路径在配置中指定。
