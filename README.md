# SimEX 本地方法复现

这是根据随附论文实现的 **Franka 单臂、缩小预算的 MuJoCo → Isaac Sim 复现**。优化对象是 `toolbox.py` 与 `skill.md`；每次任务由模型生成新 Python 策略。agent 的初始能力不包含预训练 RL 策略或预写抓盘流程。`direct_plate/` 是独立的固定控制工程演示，不注入 agent 的初始库，也不计为论文式发现结果。

## 运行

本机依赖：RTX 4090、MuJoCo 3.3.7、Isaac Sim 4.5、`isaac-sim` conda 环境、bubblewrap、系统 ffmpeg、已有 Codex 登录。项目自带隔离 `.venv`，Codex 0.162.1 安装在 `.tools`，用于兼容本机配置的 `gpt-6-astra`。全局 CLI 不受影响。

安装脚本还需要已有 `uv`、Node/npm，默认借用 `motionbricks` 的 Python 3.10 创建虚拟环境；可用 `SIMEX_PYTHON` 指定解释器。脚本不会安装 Isaac、系统级依赖或本地 Franka USD。

```bash
# 重建项目依赖和固定版本 Panda 资产
bash scripts/setup.sh

# Stage 1 预检：MuJoCo、RGBD、动作、策略隔离、视频和模型连通
.venv/bin/simex doctor --engine mujoco --run runs/preflight_v6

# 完整小规模实验：有断点时从已有结果继续
.venv/bin/simex run --run runs/small_v6 --config configs/small.yaml

# 只运行 Stage 1，完成后退出；已有进程时不会再次启动
bash scripts/run_background.sh runs/small_v6 configs/small.yaml discover
# 完整流程需显式选 run
# bash scripts/run_background.sh runs/small_v6 configs/small.yaml run

# 分阶段执行
.venv/bin/simex discover --run runs/small_v6 --config configs/small.yaml
.venv/bin/simex adapt --run runs/small_v6 --config configs/small.yaml
.venv/bin/simex evaluate --run runs/small_v6 --config configs/small.yaml
.venv/bin/simex report --run runs/small_v6
python3 scripts/status.py --run runs/small_v6
python3 scripts/verify_run.py --run runs/small_v6
```

2026-10-16 已停止 `small_v3` 的实验服务及全部子进程；该次 16 个源试验均未发出机械臂运动指令，记录保留在 `runs/small_v3`。随后 `small_v4` 在第一轮修订调用因连接超时退出：12 次源试验均有机械臂运动，但没有任务成功，正式目标试验尚未开始。两次运行记录保留。v5 的两个成员在完整任务选拔中均为 0/9；这些记录保留作诊断。本轮 v6 改进初始通用工具和 Stage 1 学习闭环，默认使用全新的 `runs/small_v6`，由操作者手动启动。不要把旧版工具库复制成新实验初始能力，也不要混接不同版本的结果。实现、阶段推理配置或依赖指纹变化后，旧实验不能续跑。

旧运行已整理为 `runs/archive/legacy_20261016/records.tar.gz`；归档保留逐次结果、策略、工具库版本、模型输入输出、全部视频/图组、关键故障 RGBD 和已有目标试验证据。`manifest.json` 列出保留文件及 SHA-256。重复 RGBD/截图和过时调试帧已删除，约 4.03 GiB 原始文件缩为 168 MiB 压缩归档。这些是不完整的诊断记录，不是复现成功率报告。

`discover` 只检查 MuJoCo 与模型，不要求 Isaac。`run` 进入适应阶段以及单独执行 `adapt` / `evaluate` 时再检查 Isaac；源引擎通过不会替代目标检查。`doctor --engine all` 可同时检查两者，`--offline` 不调用模型。首次 Isaac 启动可能花几分钟编译 RTX 着色器；缓存后通常更快。`doctor` 同时检查从初始 RGBD 出发的主动观察：`inspect` 必须实际移动、获得更细的图像采样。运动诊断没有搬盘，其 rollout 的任务分数为 `failed` 是预期，诊断本身另外检查 `passed`。

`configs/smoke.yaml` 是另一个更小的工程测试协议，使用单独运行目录；它的结果不能替代 `small.yaml`。手动 `rollout` 输出位于 `debug/`，不进入正式目标试验计数。

```bash
# 生成并执行一个源仿真策略，保留全部证据
.venv/bin/simex rollout --run runs/debug --engine mujoco --task far_plate

# 固定已有脚本回放，禁止再次生成策略
.venv/bin/simex rollout --run runs/debug --engine mujoco \
  --script path/to/policy.py --toolbox path/to/toolbox_dir --label replay
```

## 实验协议

Stage 1 使用两个独立成员，每成员三轮，每轮 8 个探测任务、4 个聚焦任务、最多两次工具库修订和最多两个历史能力回放。成员按轮次交替执行，在相同边界读取其他成员上一轮的简短摘要；不交换实现。各自的工具库、能力库和研究历史独立。GPU 上的物理执行仍然串行，这是对论文并行种群的工程改编。

每次 Stage 1 探测、聚焦测试和历史任务回放仍重新生成一次 CaP 策略，符合论文 §2.1、§2.3、附录 A.2；不能把历史成功脚本当作额外持久技能。历史脚本和计划仅作为实验诊断证据交给优化代理，帮助将成功机制提炼到 `toolbox.py` 和 `skill.md`。聚焦和修订代理能看到实际执行的代码、结果、图组及成功探测；候选评审还收到任务判据、当前/候选工具库及差分。被拒候选的代码、成绩和原因会传入下一次修订，最近两轮的研究摘要也会延续。

能力库保存任务、首次/最近成功的脚本和工具库哈希，以及最近实际观测结果。回放按通用判据结构覆盖和历史使用次数选取，轮内冻结，避免始终只选最后两个任务。新策略失败不会抹掉历史成功。聚焦分数下降或当前通过的回放变失败时不接受候选；若历史任务在当前与候选新策略中均失败，代理必须明确说明接受该不确定取舍的理由，否则保留当前版本。不同新策略的成绩变化不能单独证明工具库发生了因果退化；平分也不能单独证明改进。

代理生成任务、可执行 JSON 成功判据、修订及候选决定。仅根据预先冻结的源仿真选拔集选择成员，不访问目标结果。修订和评审提示中的执行代码属于同一成员的研究证据，不会传给随后的一次性策略调用。

v5 将研究决策与一次性策略生成分开：优化代理先看全部场景的结果摘要、运动统计和证据编号，再通过 JSON `evidence_requests` 请求特定日志、代码页或图片。中介只服务该次调用已获准的公开证据，不开放文件系统、隐藏目标参数或其他成员的实现。原始证据继续完整保留；相同文本按哈希去重，摘要中的省略明确标注并可分页读取。生成工具库修订时，若当前库的序列化内容不超过 30,000 字符，`toolbox.py` 和 `skill.md` 的完整内容固定保留在每轮输入中。更大的库切换为局部替换协议：先读取相关源码，再返回 `edits`，中介要求旧代码块唯一精确匹配，应用后再验证完整候选文件。这样无需从摘要重建代码，也不会覆盖未查看的原有函数；非法替换消耗候选名额，不自动重采样。

单次研究输入硬上限为 **64,000 字符（不是 token）和 4 张图片**；默认摘要上限 34,000 字符，完整工具库较大时继续压缩摘要。每个研究决定最多 4 轮读取，再返回最终决定；每轮读取最多 4 项、最多 16,000 个序列化字符，按剩余上下文空间进一步限制。最新读取页和最多 2,000 字笔记替换上一轮内容，不无限追加历史。每次读取调用都计入 `max_agent_calls`。这些是本地工程限制，并非论文给出的参数；当前实现是受限多轮研究代理，仍不等同于论文完整的自主 coding-agent session。

独立源任务的策略生成最多并发三路，物理执行保持串行。v6 选拔排除没有有效物理执行的成员，依次比较完整任务成功数、共同辅助任务成功数、有效执行数，最后按成员编号确定顺序；不再奖励更早失败。完整选拔集仍为三个任务各三个场景；额外四项共同辅助任务为两个场景的 5 cm 抬升和两个场景的 3 cm 位移，初始化时冻结，仅在成员工具库冻结后执行，不反馈给训练课程。

v6 每轮任务生成使用当前源场景模板，默认继承 30 g 盘子；若研究质量、摩擦或尺寸变化，提示要求在提案 `scene_change_reasons` 中按任务序号解释实验变量。中介按物体名称记录实际变更字段；缺少解释会标记 `missing_reason` 并传给聚焦分析，不因缺少备注重采样任务或中止研究。位置变化无需额外理由。各任务与模板的差异保存为 `probe_scene_variations.json` / `focused_scene_variations.json`。probe 成功在选择当轮 replay 前入库，聚焦基线和已接受修订的成功及时保存，避免当轮刚获得的能力遗漏回放保护。动作摘要区分闭合、张开、最低 TCP、闭合后的上升、拒绝和终止信息；这些仅是执行证据，不能证明抓稳。

当前 `small.yaml` 保留操作者选择的 Stage 1 `high`，Stage 2 与三个方法的冻结评估均为 `medium`。论文 Table 2 的 Stage 1 为 `max`；当前配置与其不同，需在报告中注明。模型名继续读取本机 Codex 配置；本机模型缓存已确认 `gpt-6-astra` 支持 `max`。这里不使用带自动委派的 `ultra`，以保持单次纯文本策略调用。Stage 1 单次调用超时为 1200 秒，其余默认 600 秒；推理等级和超时均记录在每次 `request.json` 并纳入运行指纹。[Codex 参数说明](https://learn.chatgpt.com/docs/config-file/config-reference#model_reasoning_effort)说明等级支持取决于模型及客户端。其他模型应在首次启动前配置其支持的等级。

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

编码代理收到公开实验摘要及按实验分别标注的阶段图组。每个场景保留证据索引，首轮最多附带四张图，其余图片由研究代理按编号读取；附件清单及哈希另行记录。项目、私有实现与运行数据在其挂载中隐藏。登录文件只读挂载，临时 CLI 状态在退出时销毁。所有策略生成是单次纯文本调用；检测到工具执行或非法策略即判失败。

初始工具库提供通用 `move_checked`、`move_path` 和 `visibility`。前两者保存绝对目标并检查实际 TCP/姿态误差；纠偏次数、偏移范围和可选几何 guard 均有界，接触阶段不能把受阻当作伺服偏差强行纠偏。后者区分相机射线上观测为空闲、测得表面、未知；自遮挡、视野外和缺失深度不能直接当成碰撞，也不能当成通行许可。它们没有盘子专用抓取点、搬运路径或释放顺序，三种方法共享相同初始版本。

v6 增加通用 `outline_points`、`fit_plane`、`fit_circle` 与 `move_segment`。圆拟合的半径范围由调用方提供，返回残差、内点比例与可见弧覆盖；局部遮挡时可结合表面平面拟合轮廓，避免直接把可见点中位数当完整物体中心。拟合残差和像素采样都不等于真实定位误差。

`move_checked` 保留原严格默认，新增 `accept_position_tolerance`；普通 API 跟踪失败时仍检查实测位姿，不可达、关节截断及 guard 拒绝等硬失败保留。`move_segment` 默认每 20 mm 一个绝对路点，自由运动最多两次、总计 25 mm 纠偏，先追求 3 mm，最终可按 15 mm 接受；接触/持物模式关闭纠偏。最后夹持对准应显式设置严格验收精度，15 mm 只适用于允许该误差的转移。姿态容差维持 0.05 rad。夹爪包装器开放宽度、力上限和时长；初始库没有预置 `direct_plate` 的抓取参数或整套策略。

固定策略提示明确：技能说明中的抓取保持等建议只适用于该技能的阶段，不能变成禁止完成上层任务的永久规则。中间技能成功不等于整体任务完成；策略仍须根据 RGBD 与公开状态选择后续动作，并遵守信息隔离。

v4 增加通用 `measure` 和空夹爪主动观察 `inspect`。后者用 RGBD 粗定位与公开的相机/TCP 位姿生成观察路点，分步检查 IK、关节变化、跟踪误差和时间预算，改善腕部图像采样。它不包含抓盘接触点、夹紧、抬升或搬运策略。观察路线根据已识别物体表面估计，适用于当前静态桌面场景；不声称完整覆盖所有机械臂连杆或不可见障碍。三种比较方法共享相同初始能力。

策略允许在仿真中进行有预算、带反馈的试探；不再要求先获得完整碰撞证明或精细盘沿尺寸，才允许非接触观察移动。视野未知仍保留为未知，不伪装成自由空间。像素采样、深度误差、接触精度分别处理；评分判据、真实接触和真值隔离保持原定义。Stage 1 的任务生成会参考已有能力安排前置实验；代理仍自行选择任务和判据。修订评审会检查新路径是否实际执行，零动作的代码扩展不能宣称解决了探索问题。

## 结果与恢复

每个运行目录包含：

- `manifest.json`、`runtime.json`：预算、模型、版本、环境和冻结计划。
- `target_ledger.json`：预分配的目标试验及状态，防止重复执行。
- `discovery/`、`adaptation/`、`evaluation/`：各次脚本、候选、选择理由和结果。
- Stage 1 每轮 `replay_records.json`、`capability_bank.json`、`research_history.json`：冻结回放选择、能力来源及跨轮经验；`accepted.json` 区分代理理由与程序回归检查原因。
- `frozen/`：初始、Stage 1 和各任务适应后的工具库及校验值。
- 每次 rollout 的 `public/`：MP4、关键帧、策略读取的 RGBD NPZ、机器人轨迹。
- 每次 rollout 的 `private/`：仅可信评估器使用的场景与真值；不作为代理证据。
- `agent_calls/`：逐次输入、输出、事件、实际图像附件及哈希、服务返回的 usage；超时保留已收到的部分输出。
- 各研究决定旁的 `*.context/`：证据索引、去重后的公开材料、摘要、逐轮请求和回答、连接重试记录；包含哈希校验。恢复时复用已完成读取轮次。
- `report.md`、`report.json`：已完成/排定数量、成功率、适应曲线数据和失败状态。

v6 报告单独列出 Stage 1 流程完成、基础能力成功和完整任务成功，并提供选拔中所有成功/失败的视频路径。被选中成员在九次完整任务选拔中至少成功一次，才标记 `full_task_verified=true`；流程结束而完整任务为零时显示“尚未验证完整入箱能力”。这沿用现有任务谓词，未额外把释放接触条件混入原评分。

相同命令会复用已完成结果。超时、非法策略、执行失败消耗排定试验；策略沙箱尚未连接时的启动故障、初始 RGBD 故障属于基础设施故障，可修复后恢复。连接后发生后端故障则将目标试验标为 `unresolved_infrastructure` 并停止研究；它保持未完成、不计入实测成功率，也不自动重跑。源侧基础设施中断也不会作为候选失败进入筛选。若崩溃后无法判断目标试验是否执行，流程停止并记录 `unresolved`，不会悄悄重跑。研究轮数、目标试验数、冻结工具库、场景种子和模型配置不可在恢复时更改；模型调用资源上限 `max_agent_calls` 可由操作者另行调整。

小规模实验仍可能调用模型数百次，默认上限 800 次。额度耗尽或基础设施错误会留下断点与明确错误，报告不会把未完成评估记成成功或零成功。

仅研究决策调用在超时或明确的连接断开且没有完成回答时自动重试一次，同样计入模型预算；再次失败则保留断点退出。无效 JSON、非法策略不自动重采样。策略生成和物理 rollout 没有新增重试；证据读取、重连和恢复都不会再次执行已有目标试验。控制台 `research_request` 显示输入字符数、附件数、原始材料体量和读取轮次，`request.json` 另外记录完整调用的字符/字节计数。

每次结果新增 `action_summary`，控制台的 `source_rollout_complete`、状态查询和报告也显示动作统计。`no_arm_commands` 表示没有调用机械臂运动；`arm_commands_without_measured_motion` 表示发出了指令但没有测得明显运动；`arm_motion_observed` 表示测得机械臂运动。仅等待和夹爪开合不计为机械臂运动，运动次数也不代表抓取成功。旧记录没有该字段时显示 `unavailable`，避免误推断。

后台启动优先使用用户级 systemd 服务（名称保存在 `research.service`）；无可用用户服务时使用独立会话。进程号保存在 `research.pid`，控制台输出持续写入 `console.log`。后台脚本不会无限重试；进程因错误退出时，请先查看 `last_error.json` 和日志，再用相同启动命令恢复。`report.md` 在每个研究阶段完成或出错时更新，也可随时运行 `simex report` 手动刷新。

本机使用 systemd 后台方式时，可停止整个实验及其子进程：

```bash
systemctl --user stop "$(cat runs/small_v6/research.service)"
```

## 验证与复现差异

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest -q
# 独立真实 Isaac 测试（需要本机资产）
bash scripts/isaac_python.sh tests/smoke_isaac.py
.venv/bin/python tests/smoke_isaac_worker.py

# 两个预先固定的源场景，一次性生成策略，验证是否真正发生机械臂运动。
# 最多两次策略生成；不启动 Stage 1/2，也不消耗正式目标试验。
.venv/bin/python scripts/validate_exploration.py --run runs/exploration_check_v4

# 只用历史材料检查新证据协议，不调用模型或仿真器。
.venv/bin/python scripts/inspect_research_context.py \
  --request runs/small_v4/agent_calls/00016_discover_revision/request.json \
  --out runs/context_packet_check

# 可选：两次真实模型调用验证“申请证据—读取—回答”，没有物理试验。
.venv/bin/python scripts/validate_research_context.py --run runs/context_read_check
```

新增回归测试覆盖拒绝后的反馈、非法候选恢复、每场景证据覆盖、摩擦灵敏度、夹爪力语义、任务参数边界和运行时故障分类。适配器测试使用假进程，不调用实验模型。

v3 另外验证优化代理收到真实脚本、历史能力来源与覆盖采样、双成员公平交换、断点恢复一致性、历史失败取舍、阶段推理设置、固定绝对目标和遮挡分类。完整协议假执行测试排定并核对全部 60 个目标试验，检查第五次适应后不增加验证试验、任务间独立以及评估不反馈给优化代理；它不证明物理任务成功率。

v6 无模型验证命令（输出目录必须为空）：

```bash
.venv/bin/python -m pytest -q
.venv/bin/simex doctor --offline --engine mujoco --run runs/preflight_v6_offline
.venv/bin/python scripts/validate_initial_toolbox.py --run runs/toolbox_v6_recheck --video
```

2026-10-19 工程验证：全量 **308 项测试通过**。保持 30 g、320×240 及原评分，在三个现有任务的 seed 42 上，将新通用几何/运动工具接入 `direct_plate` 固定流程，完整入箱及独立接触审计 **3/3 通过**；原固定流程在相同低分辨率条件下 **2/3 通过**，`far_to_near` 停在持物轮廓检查。全部六次结果和三视角视频在 [toolbox_final/summary.json](runs/validation_v6/toolbox_final/summary.json)，初次无视频记录也保留。视频放大为 1920×480 展示，控制器输入仍是 320×240。两种流程的几何/运动处理不同，此对照不能单独归因于某一个改动。

MuJoCo 离线预检通过，记录在 [doctor.json](runs/validation_v6/doctor/doctor.json)。这些验证没有模型调用，没有启动正式 Stage 1；固定流程成功证明基础工具可支持任务，不证明 agent 已自主发现技能，也不是未见场景成功率。正式实验请使用上方 `discover --run runs/small_v6` 命令。

v4 回归覆盖初始粗视图下真实移动并提高采样、观察预算、运动失败停止、目标不可见时停止、观察不自动开合夹爪，以及蓝色机器人区域污染箱体高度的感知故障。`doctor` 在两引擎实际执行主动观察，不以“API 可以调用”代替探索验证。验证记录在 `runs/validation_v4/`，包括发现错误时的失败记录；这些工程试验与正式研究结果分别保存。

v4 验证结果：全量测试 229 项及新增真实反馈回归 1 项通过，双引擎主动观察通过。两次新生成策略分别发出 31、27 条机械臂指令，均改善了视角，但仍未抓盘：它们关闭了自由空间反馈，在后续移动误差处停止。最终提示已区分自由空间纠偏与接触模式；没有继续采样模型筛成功。固定原脚本在最终初始库中回放仍失败。详细报告见 `runs/validation_v4/report.md`，不能把观察能力验证当作抓盘或迁移成功。

v5 验证结果：254 项全量测试通过。用 small_v4 实际失败请求离线重建，输入从 807,150 字符降至 52,476 字符（减少 93.50%），首轮附带 4 张图，全部 12 次测试和 12 张图的证据仍保留并可读取。两次真实模型工程调用验证了申请片段后再回答的通路，没有重启正式实验或运行新的目标试验。报告见 `runs/context_fix_v5/report.md`。这不证明外部连接永不超时，也不代表抓盘成功率已改善。

真实 MuJoCo 通用运动冒烟记录位于 `runs/validation_v3/motion/summary.json`。在无负载自由空间的四步下降测试中，使用上一步实测位置生成下一目标产生 18.929 mm 最终位置误差；新的绝对路点与有界反馈将四步误差控制在 0.779–0.965 mm。该验证没有抓盘，没有调用模型，也不能外推为接触抓取或目标迁移成功。

测试覆盖真实 MuJoCo RGBD/反投影、物理接触、IK不修改现场、动作、复位、录像、沙箱边界、严格评分及试验预算/冻结/回放协议。Isaac 的真实测试另行运行，避免常规测试重复占用 GPU。

与论文相比：单臂 Panda 替代双臂 YAM；三个可达改编任务替代七任务；两个成员/三轮替代四成员/二十轮；两个修复和三个筛选任务替代四个/五个；仅一次独立研究运行。相机、资产、控制器、初始工具库与模型也不同，因此报告用于检验方法链路和观察迁移现象，不与论文成功率作直接等价比较。

首版公开任务模板，留出评估场景种子；这与论文对代理隐藏完整评估任务定义的协议也有区别。

论文：随附 `2609.38982v1.pdf`。Panda 来源与许可证：`assets/panda/PROVENANCE.json`、`assets/panda/LICENSE`。本地 Isaac USD 不被重新分发，其路径在配置中指定。
