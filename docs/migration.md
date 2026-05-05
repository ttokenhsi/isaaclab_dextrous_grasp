# ViViDex（SAPIEN）→ IsaacLab 迁移说明

本文档记录把 [ViViDex](https://github.com/zerchen/vividex_sapien) 的 UR5 +
Allegro YCB relocate 任务从 SAPIEN 迁移到 IsaacLab 的 manager-based RL
框架时所做的设计、对齐与踩坑总结。文档刻意写成自包含形式，使
`/root/workspace/rl_grasp/isaaclab_dextrous_grasp` 这个包脱离原始
`vividex_sapien` 仓库后仍可独立使用。

IsaacLab 实现位于
`isaaclab_dextrous_grasp/tasks/allegro_relocate/`，已通过
`gym.register("Isaac-AllegroUR5-Relocate-v0", ...)` 注册。

---

## 1. 顶层映射

| ViViDex / SAPIEN                                                       | IsaacLab 对应                                                                                       |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `AllegroRelocateRLEnv`（gym + SAPIEN actor 系统）                       | `AllegroRelocateManagerEnv` ⊂ `ManagerBasedRLEnv`                                                   |
| YAML / Hydra 配置（`config/agent/ppo.yaml`）                            | `manager_env_cfg.py` + `agents/rsl_rl_ppo_cfg.py` 中的 `@configclass` Python 配置                    |
| `frame_skip = 10`，`sim_freq = 500`                                     | `decimation = 10`，`sim.dt = 1/200` ⇒ 等效 50 Hz 控制                                                |
| `compute_inverse_kinematics(...)`，damping=0.05，cart limit=1.0          | `IKHandAction` + `DifferentialIKController(command_type="velocity")`                                |
| 22 维动作 = 掌心速度 + 16 手指 qpos                                      | `IKHandActionCfg`（palm 6 维 + Allegro 16 维）→ `articulation.set_joint_position_target`            |
| 393 维 oracle observation                                              | `policy` group 下 4 个 `ObsTerm` 拼接                                                                |
| `get_reward` 中硬编码奖励                                               | 7 个 `RewTerm`（pregrasp / contact / object_track / fingertip_track / lift / ctrl / action）         |
| `is_done` 标志                                                          | 4 个 `DoneTerm`（`pregrasp_failure`、`object_too_far`、`lost_contact_in_imitate`、`time_out`）       |
| `reset` 通过 `randomize_trajectories` 重建轨迹                           | `EventTerm(mode="reset")` → `reset_trajectory_state`                                                |
| `check_actor_pair_contacts(palm, finger_parents, object)`              | 5 个 `ContactSensorCfg`，过滤到 `{ENV_REGEX_NS}/Object`                                              |
| `stable_baselines3.PPO`                                                | `rsl_rl.runners.OnPolicyRunner`（rsl-rl-lib ≥ 5.0.0）                                                |
| 单 env 步进 × `n_envs` Python 进程                                       | `InteractiveScene` 向量化步进（同一进程内 `num_envs` 个 env）                                          |

---

## 2. Action space（22 维，`∈ [-1, 1]`）

| 切片 | 含义                                                | ViViDex 来源                                              |
| ----- | --------------------------------------------------- | --------------------------------------------------------- |
| `[0:3]`   | 掌心世界系**线**速度命令                       | `vividex_sapien/.../base.py::compute_inverse_kinematics` |
| `[3:6]`   | 掌心世界系**角**速度命令                       | 同上                                                      |
| `[6:22]`  | Allegro 16-DoF 关节位置目标                    | `relocate_env.py::pre_step`                               |

`IKHandAction`（在 `mdp/actions.py`）**逐字节地**实现 ViViDex 基于速度
的控制循环，刻意没有走 IsaacLab `DifferentialIKController` 包装：

1. **缩放** `[-1, 1]` 到物理上限（`cart_lin_vel_limit=1.0`，
   `cart_ang_vel_limit=1.0`，Allegro 各关节 `joint_limits`）。
2. 在 `process_actions` 里**每个外层 step 解一次 DLS IK**：

       J ∈ ℝ^{6×6}  = jacobian(palm, arm_joints)        # 世界系
       arm_qvel     = Jᵀ (J Jᵀ + λ²I)⁻¹ · v_des          # λ = 0.05
       arm_qvel     = clip(arm_qvel, ±π)                 # ViViDex line 168
       arm_qpos_des = arm_qpos + arm_qvel · dt_ctrl

   这与 ViViDex 的 `get_arm_qvel` **位级一致**（已与
   `~1e-7` 精度对齐，参见 `scripts/test_ik.py` Test B）。我们**故意不**用
   `DifferentialIKController`：它会在每个 PhysX 子步针对一个过期的
   pose target 重新解 IK，更糟的是它没有暴露我们做速度前馈所需的
   `arm_qvel`。
3. 在 `apply_actions` 里**每个 PhysX 子步同时**写两个目标：

       articulation.set_joint_position_target([arm_qpos_des, hand_qpos_des])
       articulation.set_joint_velocity_target([arm_qvel,    0])

   速度前馈是**关键**：缺了它，从 ViViDex 抄来的高 `kd` PD 增益
   （手臂 `kd=40000`）会作为强阻尼制动到关节速度，IK 推导出的
   `arm_qpos_des` 永远跟不上。详见踩坑 #13。

两个副产品被缓存到 env 上供 reward 使用：

- `cartesian_error`：每 env 的 `‖Δpalm − v_des · dt_ctrl‖`，每个子步用
  `step_dt` 更新，使所有 decimation 子步走完后的**最终值**与 ViViDex
  的"外层 step"约定一致。
- `target_lin_vel`：未缩放的 3 维掌心线速度命令。

这与 `compute_inverse_kinematics` 在几何（世界系 Jacobian、相同阻尼）
和时序（50 Hz 外层步内一次 IK 解、`decimation=10` PhysX 子步）上完全
一致。

---

## 3. Observation space（393 维 oracle）

四个 obs term 都挂在 `policy` 这个观测 group 下，通过 action term 和
`AllegroRelocateManagerEnv` 里的 trajectory buffer 共享中间结果。

| Term            | Dim | 组成                                                                                                                                                |
| --------------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `robot_state`   | 330 | qpos (22) + qvel (22) + 22 个 link 状态；22 link 按**整段拼接**：22×3 pos、22×4 quat、22×3 lin-vel、22×3 ang-vel = 286 维                              |
| `object_state`  | 13  | object pos (3) + quat (4) + lin-vel (3) + ang-vel (3)                                                                                               |
| `goal_state`    | 42  | 未来 3 帧 `[orn(4) + trans(3)]` = 21 + palm−object 差 (3) + 4 fingertip−object 差 (12) + palm−target 差 (3) + object−target 差 (3)                     |
| `time_state`    | 8   | `[sin(k·t), cos(k·t)]`，`k ∈ {1, 4, 6, 8}`，`t = traj_step / imitate_steps`                                                                          |

`manager_env_cfg.py` 中的 `ROBOT_BODY_NAMES` 列出了 ViViDex
`joint_link_names` 引用的 22 个 link：6 条 arm link
（`shoulder_link`、`upper_arm_link`、…、`right_gripper_palm_link`）+ 16
个 Allegro 远端／中段 link。**整段式**拼接（先把所有 22 个 pos 接起来，
再拼所有 22 个 quat，依此类推）是个非常关键的细节；早先按 link 一
个一个 `(pos, quat, lv, av)` **交错**拼出来也是 393 维，**但顺序不
一样**，会破坏样本效率，且无法对齐已加载的 ViViDex policy。

`oracle_state` group 也通过
`obs_groups = {"actor": ["policy"], "critic": ["policy"]}` 暴露给
critic，这与 PPO 原版 actor / critic 共用同一观测的设置一致。

---

## 4. Reward（与 ViViDex `get_reward / 10` 对齐）

7 个 `RewTerm` 定义在 `mdp/rewards.py`。每步的值与 ViViDex
`get_reward` 中各项**完全相同**，全局 `/10` 的因子被折叠进
`manager_env_cfg.py` 中各项的 `weight`。这样 manager 日志就能保留
reward 各子项的可读性。

| RewTerm                | weight  | ViViDex 项                                                                                  |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `pregrasp`             | 1.0     | `step ≤ pregrasp_steps` 期间 `10 · exp(-10 · fingertip_err)`，否则 0                         |
| `contact`              | 0.05    | `step > pregrasp_steps` 期间 `0.5 · num_contacts`（palm + 4 指，最多 5），否则 0             |
| `object_track`         | 1.0     | imitate 期 `10 · exp(-50 · (com_err + 0.1 · rot_err))`                                      |
| `fingertip_track`      | 0.4     | imitate 期 `4 · exp(-10 · fingertip_err)`                                                    |
| `lift_bonus`           | 0.25    | imitate 期 `2.5 · 1{lift > 0.02}`                                                            |
| `controller_penalty`   | -100.0  | `-1e3 · cartesian_error²`                                                                   |
| `action_penalty`       | -0.001  | `-0.01 · ∑ clip(qvel, -1, 1)²`                                                              |

中间量（`com_err`、`rot_err`、`fingertip_err`、palm/finger 接触
boolean、`lift`）每步在 `_ensure_intermediates` 计算一次后缓存到 env。
`AllegroRelocateManagerEnv.step()` override 在每次环境 step 时使
缓存失效，保证 reward / termination / observation 在同一物理帧上
看到一致的中间值。

### Contact 项细节

`num_contacts` 是 **palm + 4 fingers** 5 个桶的命中计数，与 ViViDex
`sum(self.robot_object_contact)` 完全一致；那里
`robot_object_contact` 由
`check_actor_pair_contacts(palm + 4 finger parents, object)` 的 5 桶
`bincount` 产生（参见 `relocate_env.py:204-214`）。每 env 每步最多
`5`，未归一化的 contact reward 上限 `0.5 · 5 = 2.5`（再除 `/10` 后
`0.25`）。我们用 5 组 `ContactSensor`（每个桶各一组，桶内多个
phalange link 的力做 OR）取桶值，并用
`impulse_threshold = 1e-2 / dt` 去阈值化 `force_matrix_w` 的模长。
`force_matrix_w` 过滤到每 env 单一 body（`{ENV_REGEX_NS}/Object`），
确保跨所有 env 输出形状一致。

`num_finger_contacts = num_contacts − palm_bucket` 也单独缓存供
termination（`lost_contact_in_imitate`）使用，但**不**进入 contact
reward。

---

## 5. Termination（4 个 DoneTerm）

| DoneTerm                  | 触发条件                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| `pregrasp_failure`        | `step == pregrasp_steps + 1` 且 `‖hand_qpos − ref_hand_qpos‖ > 0.05`                     |
| `object_too_far`          | `‖object_pos − ref_object_pos‖ > 0.15`                                                   |
| `lost_contact_in_imitate` | `step > pregrasp_steps` 且 5 个接触桶全部为空                                            |
| `time_out`                | `episode_length_buf ≥ max_episode_length`（仅置 `truncated=True`，非 `terminated`）       |

与 `AllegroRelocateRLEnv.is_done` 完全等价。`time_out=True` 是
IsaacLab 暴露 Gymnasium `truncation` 标志的标准方式。

---

## 6. Events（reset）

`mdp/events.py::reset_trajectory_state` 注册为 `EventTerm(mode="reset")`。
对每个需要重置的 env：

1. 按 `task.trajectory_names` 采样一个轨迹 id。
2. 按 `task.stage` 采样平面位移 / yaw 旋转
   （0 = canonical，1 = `(x,y) ∈ U[0.30, 0.40]²`，2 = +`θ_z ∈ U(-π/12, π/12)`）。
3. 向量化地对轨迹（`object_translation`、`object_orientation`、
   `robot_jpos`）做 canonicalize，与 ViViDex 的
   `randomize_trajectories` **完全一致**。
4. 把 canonical 引用写到每 env buffer
   （`env._traj_object_pos`、`env._traj_robot_qpos`、…）。
5. 重置 `current_step / traj_step / pregrasp_failure_pending` 计数器。
6. 把机器人 articulation state 和 YCB 物体 root state 设为轨迹的 t=0
   帧。

这些 buffer 本身在 `AllegroRelocateManagerEnv.load_managers`（见 §8）
中、父类构造 manager 之前，**只分配一次**，使 obs/reward/done
回调在 dry-run 时就能找到已分配好的 buffer。

---

## 7. 课程（`task.stage`）

完全对齐 ViViDex 的 stage。这些旋转同时作用在轨迹**和**物体的初始
姿态上（在 `reset_trajectory_state` 内做），保证手部参考与随机化
后的物体位置一致：

```
stage 0 → (x, y, θz) = (0.35, 0.35, 0)
stage 1 → x, y ∼ U[0.30, 0.40]，θz = 0
stage 2 → x, y ∼ U[0.30, 0.40]，θz ∼ U(-π/12, +π/12)
```

---

## 8. 自定义 env 子类（`AllegroRelocateManagerEnv`）

默认 `ManagerBasedRLEnv` 在两个地方过于刚性，`manager_env.py` 中的
子类做了补丁：

1. **物体资产数据驱动。** ViViDex 根据轨迹的 `object_name` 选 YCB 物体。
   我们在 `__init__` 调用 `super().__init__` **之前**动态修改
   `cfg.scene.object` 来复现：
   - 在 `assets/ycb/<id>/textured_simple.obj` 下找 OBJ 路径。
   - 通过 `MeshConverter` 转 USD（`cache/usd` 缓存）。
   - 构建 `RigidObjectCfg`，并设 `activate_contact_sensors=True`，使
     contact sensor 管线能挂到物体 body 上。

2. **`ObsTerm` dry-run 需要轨迹 buffer 提前就位。** IsaacLab 的
   `ObservationManager` 构造时会用合成数据各调用一次每个 obs 函数推
   断 shape；我们的 `goal_state` 会读
   `env._traj_object_pos[:, env.traj_step + Δ]`，否则崩溃。所以在
   `load_managers` override 里：
   1. 先 resolve articulation 的 body / joint index。
   2. 用合理默认值分配 `_traj_*` 与计数器 buffer。
   3. 再 `super().load_managers()`。

3. **每步簿记**：override `step()`：
   - 递增 `current_step` 与 `traj_step`（`min(current_step, T-1)`）。
   - 让 `_intermediate_cache` 失效，使 reward / termination 始终基于
     post-step 状态重新推导中间量。

---

## 9. Contact sensor

按 IsaacLab 推荐的三步走：

1. **机器人 URDF spawn** 设 `activate_contact_sensors=True`，让
   `right_gripper_palm_link` 与各手指 parent link 暴露 contact-reporter
   API。
2. **物体 spawn** 设 `activate_contact_sensors=True`，让 YCB body 成为
   有效的 contact partner。
3. **5 个 `ContactSensorCfg`**，scene cfg 里挂到：
   - `right_gripper_palm_link`（palm，桶 0）
   - `right_gripper_link_15`（拇指 parent，桶 1）
   - `right_gripper_link_03`（食指 parent，桶 2）
   - `right_gripper_link_07`（中指 parent，桶 3）
   - `right_gripper_link_11`（无名指 parent，桶 4）
   每个都带 `filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]`，使
   `force_matrix_w` 形状是确定的 `(num_envs, num_bodies, 1, 3)`。

每 env 的 `num_contacts` 是 5 个桶里**力幅值超 `1e-2 / dt` 阈值**的桶
个数（**含 palm 桶**），这正是 contact reward 消费的量，与 ViViDex
`sum(self.robot_object_contact) * 0.5`（同样含 palm 桶，参见
`relocate_env.py:214`）一致。`lost_contact_in_imitate` 终止则消费
`num_finger_contacts = num_contacts − palm_bucket`。

---

## 10. PPO 配置（`agents/rsl_rl_ppo_cfg.py`）

对齐 `vividex_sapien/algos/rl/config/agent/ppo.yaml`：

| 字段                  | ViViDex          | IsaacLab 对应（rsl-rl 5.x）                                  |
| --------------------- | ---------------- | ------------------------------------------------------------- |
| `gamma`               | 0.95             | `RslRlPpoAlgorithmCfg.gamma=0.95`                             |
| `gae_lambda`          | 0.95             | `lam=0.95`                                                    |
| `learning_rate`       | 1e-5（固定）     | `learning_rate=1e-5`，`schedule="fixed"`                      |
| `ent_coef`            | 0.001            | `entropy_coef=0.001`                                          |
| `vf_coef`             | 0.5              | `value_loss_coef=0.5`                                         |
| `clip_range`          | 0.2              | `clip_param=0.2`，`use_clipped_value_loss=True`               |
| `n_steps`（每 env）   | 4096 / num_envs  | `num_steps_per_env=64`（默认 num_envs=64 ⇒ 4096）             |
| `batch_size`（mini）  | 256              | `num_mini_batches = (64 × 64) / 256 = 16`                     |
| `n_epochs`            | 5                | `num_learning_epochs=5`                                       |
| `net_arch=[256,128]`  | actor 与 critic  | `RslRlMLPModelCfg(hidden_dims=[256,128], activation="elu")`   |
| `log_std_init=-1.6`   | exp(-1.6)≈0.20   | `GaussianDistributionCfg(init_std=0.20, std_type="scalar")`   |
| `desired_kl=0.01`     | 无               | early-stopping 信号（rsl-rl 额外）                            |
| `max_grad_norm=1.0`   | 无               | 梯度裁剪（rsl-rl 额外）                                        |

rsl-rl ≥ 5.0.0 已删除老 `policy=...` 字段；我们改用独立的
`actor: RslRlMLPModelCfg` 与 `critic: RslRlMLPModelCfg`，配合
`obs_groups={"actor": ["policy"], "critic": ["policy"]}`。配置里也保留
了已废弃的 `stochastic`/`init_noise_std`/`noise_std_type`/
`state_dependent_std` 字段；IsaacLab 的
`handle_deprecated_rsl_rl_cfg(agent_cfg, version)` 会在运行时
（`scripts/train.py` 与 `scripts/play.py`）剥掉它们。

---

## 11. 训练脚本

- `scripts/train.py`：构建 env + `RslRlVecEnvWrapper`，跑
  `OnPolicyRunner.learn`。不用 Hydra，CLI override 直接拎进来，让本包
  保持"放进任何项目都能用"的状态。
- `scripts/play.py`：加载 checkpoint，做 `--num_steps` 步推理 rollout。
- `scripts/smoke_test.py`：零动作 rollout 加形状断言
  （`obs == (E, 393)`、`act == (E, 22)`），CI 健康检查用。

三者都是先调 `AppLauncher`，再做任何 IsaacLab / Isaac Sim 的 import
——这是 Omniverse 运行时的硬性要求。

---

## 12. 踩坑实录（以及修复方法）

下列问题在迁移过程中实际造成过故障，记录在此让后来人不必重新踩一遍：

1. **URDF mesh 文件名里的 `.` 会让 PXR 的 `SdfPath` 出错。**
   Allegro 的 `link_0.0.obj` 等文件触发
   `ValueError: Failed to convert MeshConfig`。修复：把
   `assets/robot/ur5_description/allegro_meshes/` 下的 `*.0.*` 改名为
   `*_0.*`，并同步 URDF 里的引用。Robotiq 那个超长名
   `robotiq_ft300-G-062-COUPLING_G-50-4M6-1D6_20181119.STL` 也压成
   `robotiq_ft300_coupling.STL`。

2. **`UrdfConverter` 默认 `merge_fixed_joints=True`，** 会在 USD 转换
   时把 `right_gripper_palm_link` 并入其 parent，于是 palm 上的
   `ContactSensor` 报 "no rigid bodies under prim"。修复：在机器人
   `UrdfFileCfg` 上显式设 `merge_fixed_joints=False`。4 个手指 parent
   link 不论怎么设都在；palm 是唯一需要这个开关的。

3. **`activate_contact_sensors` 必须**两侧都开**。** 机器人侧打开
   reporter API，物体侧让 YCB body 成为有效 contact partner。只开一
   侧会报 "could not find any bodies with contact reporter API"，或者
   即使可见接触也得到空的 `force_matrix_w`。

4. **`filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]`。** 用全局
   regex（`/World/envs/.*/Object`）会把所有 env 的 contact 聚合成单
   个 `force_matrix_w`，触发 "expected N, found 2" 断言。每 env 一个
   regex 解析为单一 filter body，形状才确定。

5. **`InteractiveSceneCfg.clone_in_fabric=True` 让复杂 articulation
   渲染崩。** PhysX 仿真还是对的，但视觉只有 `env_0` 出现。我们默认
   `clone_in_fabric=False`。

6. **`ManagerBasedRLEnv` 的实例化顺序。** 两个后果：
   - 物体的 `RigidObjectCfg` 必须**在 `super().__init__` 之前**写入
     `cfg.scene`，所以
     `AllegroRelocateManagerEnv.__init__` 里有动态打 patch 的代码。
   - `ObsTerm` 的 dry-run 在 `super().__init__ → load_managers` 内
     运行，会用任意状态触发每个 obs 函数；`goal_state` 读 `env._traj_*`，
     于是我们 override `load_managers`，先分配 buffer 再交给父类。

7. **每 env 的 `current_step`。** Reward / termination 都依赖它；我们
   在 `step()` override 里递增（PhysX 步进之后、manager 跑 reward /
   termination 之前），并在 `reset_trajectory_state` 中重置。

8. **轨迹 object_name。** `.npz` 里存的是
   `object_name="006_mustard_bottle"`。早先代码把前缀 `006_` 剥掉，
   导致路径 `assets/ycb/006_mustard_bottle/textured_simple.obj` 找不
   到。修复：保留完整名（与 `vividex_sapien/assets/ycb_models` 目录
   布局一致）。

9. **rsl-rl-lib 5.0.0 删了 `policy=`。** 旧 IsaacLab 示例还自带
   `RslRlPpoActorCriticCfg`，用了它会在
   `rsl_rl.algorithms.ppo.construct_algorithm` 深处抛
   `KeyError: 'class_name'`。修法：分开定义 `actor`/`critic` 两个
   `RslRlMLPModelCfg`，并调用 `handle_deprecated_rsl_rl_cfg` 把残留
   的 MISSING 字段剥掉。

10. **机器人 link 状态的"整段式"拼接。** 一开始 22 link 的状态按
    `(pos, quat, lv, av)` 一 link 一 link 拼，得到的也是 393 维但
    顺序与 ViViDex 不同。我们重新对照
    `vividex_sapien/.../base.py::get_oracle_state` 推导后改成整段
    拼接（先所有 22 个 pos，再所有 22 个 quat，依此类推）。

11. **机器人基座位姿必须对齐 `lab.ROBOT2BASE`。** ViViDex 中 UR5
    *不在*世界原点：每次 reset 都
    `self.robot.set_pose(Pose(p=ROBOT2BASE.p + root_offset))`，
    `ROBOT2BASE.p = (0.765, -0.09, 0)`。录制轨迹的
    `object_translation` 与 `robot_jpos` 在同一世界系下，所以
    canonical `(x, y) = (0.35, 0.35)` 在**机械臂基座系**里大约是
    `(-0.42, +0.44)`。把机器人钉死在 `(0, 0, 0)` 同时把物体放在
    `(0.35, 0.35)`，机械臂就出现在了物体的错误一侧（左侧而非
    后侧／右侧）。修法：设
    `ArticulationCfg.init_state.pos = ROBOT_BASE_POS = (0.765, -0.09,
    TABLE_TOP_Z)`，并把桌子重新摆到 `(0.5625, 0)`，让它同时覆盖
    机器人脚印与物体区域。

12. **PhysX `gpu_total_aggregate_pairs_capacity` 必须随 `num_envs`
    放大。** 4096 envs 时 PhysX 会打印
    `The application needs to increase
    PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to ~70k`，
    之后 broad-phase 会**静默丢失**接触。cfg 在 `__post_init__`
    里取的是 cfg-time 的 `num_envs`，但用户 CLI override 是在 cfg
    构造**之后**才发生，所以我们在
    `AllegroRelocateManagerEnv.__init__`（`super().__init__` 之前）
    重新算这个容量为 `max(16384, 64 × num_envs)`，给 4096 envs 实测
    17 pairs/env 留 ~2× 余量。

13. **隐式 PD 控制器里**速度前馈是必须的**。** ViViDex 的 SAPIEN
    actor 在每次 `step()` 前**同时**写
    `set_drive_target(arm_qpos_des)` 和
    `set_drive_velocity_target(arm_qvel)`，所以关节力矩是
    `kp · (q* − q) + kd · (q̇* − q̇)`。IsaacLab 的
    `DifferentialInverseKinematicsAction` 只写**位置**目标；配上从
    ViViDex 抄来的高 `kd = 40000`（坑 #14），缺失的 `q̇*` 让
    `kd · q̇` 直接变成强制动力矩，arm 最终只能跟踪到 IK 推导位移的
    ~17%（`scripts/test_ik.py` Test C）。我们因此自写
    `IKHandAction`，绕开 `DifferentialIKController`：
    `process_actions` 每外层 step 解一次 DLS IK，得到
    `(arm_qpos_des, arm_qvel)`，`apply_actions` 在每个 PhysX 子步
    通过 `set_joint_position_target` **和**
    `set_joint_velocity_target` 同时下发，并按 `step_dt`（不是
    `physics_dt`）更新 `cartesian_error`，让 step 完后的最终值与
    ViViDex 外层 step 约定一致。

14. **PD 增益必须照搬 SAPIEN 的 `set_drive_property`。** ViViDex
    手臂为 `(stiffness=200000, damping=40000, force_limit=500)`，
    手部为 `(stiffness=200, damping=60, force_limit=10)`。先前默认
    的 `(20000, 400, 300)`、`(damping=10)` 比正确值软了 10–100×（
    手臂）和 6×（手部），imitation tracking 项几步就饱和到
    `exp(−10 × large_err)` ≈ 0。现在
    `manager_env_cfg.py` 里完全照搬 ViViDex 数值。

15. **`body_pos_w` / `root_pos_w` 是*全局*仿真世界坐标，不是
    env-local。** `env_spacing=2.5`、2×2 网格布局下，env 0 原点是
    `(1.25, -1.25, 0)`，env 2 是 `(-1.25, -1.25, 0)`，依此类推。把
    env-local 的轨迹目标（如 `_traj_pregrasp[:, -1, 1:]`）从
    `body_pos_w` 里减出去会**混坐标系**，每 env 多出一个随
    `num_envs` 增长的常数偏移。我们这边的症状：
    `pregrasp_reward = 1e-11`、`object_track_reward = 0`，且
    所有非 env-0 的 env 都触发 `pregrasp_failure`。修法：所有用来
    与轨迹 tensor 比较（或并列馈送）的世界系量，都显式减
    `env.scene.env_origins`。涉及四处：

    - `mdp/rewards.py::_ensure_intermediates` —— `finger_pos`、
      `obj_pos`。
    - `mdp/observations.py::_gather_links_blockwise` —— `robot_state`
      用的 body link 位置（让 22-link 那段在所有并行 env 里完全一致）。
    - `mdp/observations.py::object_state` —— root 位置（朝向、线速度、
      角速度都是平移不变量，无需调整）。
    - `mdp/observations.py::goal_state` —— 在与 `_traj_target_pos`
      （已经 env-local）做差之前的 `palm_pos`、`obj_pos`、`finger_pos`。
      env 内部的差（`palm − obj`、`finger − obj`）本身就 env-origin
      不变，但仍然减一遍以保持一致。

    用 `/tmp/check_frames.py` 验证（4 envs，stage 0）：修复后
    `pre_err`、`obj_com_err`、palm/object 的 env-local 坐标在所有并
    行 env 上**位级一致**（散布 `< 1e-6 m`）。修复前 env 2、3 看到
    `pre_err ≈ 1.4 m` 而非 `0.28 m`。

16. **指尖运动学查询要打 `*_tip` *子* link，不是它的 parent。**
    ViViDex（`relocate_env.py:75-80, 196`）把 `(palm + 4 fingertip)`
    参考点存在 `robot_jpos[t, :, :]` 中，与
    `right_gripper_link_*_tip`（即 `*_tip` 子刚体）的 FK 位姿做比对。
    早先 `merge_fixed_joints=True` 让 `*_tip` 并入了 parent，
    `FINGER_BODY_NAMES` 因此设成了 parent link
    （`link_15 / 03 / 07 / 11`）。后来为支持 ContactSensor 切到
    `merge_fixed_joints=False` 之后，parent link 还在，但 tip 也活
    回来了，而我们忘了把指尖名字切回去。结果：`pre_err`、
    `fingertip_err` 与 `goal_state` 的 `hand_obj_dense_diff` 段沿
    远端 phalange 轴永久地差了 ~2 cm。验证：读 mustard 轨迹第 0 帧
    canonical 状态下的 `pre_err`：`0.290 m`（parent link）vs
    `0.268 m`（`*_tip`），刚好等于远端 phalange 长度。修法：
    `FINGER_BODY_NAMES` 改用 `right_gripper_link_*_tip`。

17. **每指 contact 桶要对 3-4 个 phalange 做 OR，仅 tip-parent 不
    够。** ViViDex 的
    `finger_contact_link_names + finger_contact_ids`
    （`relocate_env.py` 行 77、83）定义：
    - 拇指  = `link_15_tip ∪ link_15 ∪ link_14`           （3 个 link）
    - 食指  = `link_03_tip ∪ link_03 ∪ link_02 ∪ link_01` （4 个 link）
    - 中指  = `link_07_tip ∪ link_07 ∪ link_06 ∪ link_05` （4 个 link）
    - 无名指 = `link_11_tip ∪ link_11 ∪ link_10 ∪ link_09` （4 个 link）

    任何一个组成 link 与物体有非平凡接触，桶就触发。我们之前每个
    桶只在 parent link（`link_15` / `03` / `07` / `11`）挂了一个
    `ContactSensor`，于是 16 个 phalange 级接触面里有 12 个（4 个
    tip + 8 个近端 / 中段 phalange）被静默忽略了。这系统性地低估
    `num_contacts`，把 `contact_reward`（`0.5 × num_contacts`）压
    到正确值的 1/4 之多。

    PhysX 的 `create_rigid_contact_view` 不接受多 body sensor 配
    单 filter prim（filter list 必须满足
    `len(filter) == num_envs × num_bodies`，但单个 Object 只产生
    `num_envs` 个），所以最干净的修法是每 phalange 一个
    `ContactSensorCfg`：1 palm + 15 finger link = 16 个 sensor。
    这 15 个 finger sensor 在
    `AllegroRelocateManagerEnvCfg.__post_init__` 里编程式添加，命名
    `{thumb,index,middle,ring}_phalange_{0..3}`。reward 代码再在
    `mdp/rewards.py::_ensure_intermediates` 中按桶用 `torch.maximum`
    做 OR 还原成 5 桶并阈值化。

    阈值调整：ViViDex 阈值化的是每步**冲量**，
    `1e-2 N·s`（`sim_env/base.py:97`）；在 SAPIEN `sim_freq=500 Hz`
    下相当于约 5 N 的有效**力**。我们之前用 1 N（松了 5 倍），现已
    调到 5 N。

18. **四元数顺序一致（无 bug）。** 本轮审计中确认：npz 的
    `object_orientation` 是 `wxyz`（近单位旋转时第 0 分量 ≈ 1，已
    用 mustard-bottle 第 0 帧
    `[0.99974, -0.0091, 0.0073, 0.0195]` 直接验证）；IsaacLab 文档
    里 `RigidObjectData.root_quat_w` 也是 `wxyz`；自写的
    `_quat_mul_wxyz` 同样按 `wxyz` 处理。reward 中的旋转距离项
    `2·acos(|⟨q1, q2⟩|)/π` 只要两边惯例一致，对内积顺序无关。

19. **物体质量：density 驱动，不是固定 0.2 kg。** ViViDex
    （`utils/ycb_object_utils.py:118-122`）每个 YCB 物体加载时
    `density=1000 kg/m³`，让 SAPIEN 对凸分解体积积分得到每个物体
    的质量。我们之前在 `manager_env.py::_build_object_cfg` 的
    `MeshConverter` 与 spawn 时的 `UsdFileCfg` 里都硬写
    `MassPropertiesCfg(mass=0.2)`。mustard bottle 现在算出来约
    0.77 kg（density × 凸分解体积），先前的 0.2 kg 比真实 YCB 物体
    轻了约 4×，导致它一被指尖碰就明显飞起、合拢手指根本来不及（
    物体加速度比手指闭合还快）。
    修法：两处 `MassPropertiesCfg` 都换成
    `MassPropertiesCfg(density=1000.0)`，并清掉旧的 YCB USD 缓存
    （`cache/usd/ycb/`），让转换器重新把 mass 烘进缓存 USD。

20. **机器人 + 物体摩擦必须对齐 SAPIEN 的 `(1.5, 1.0)`。** ViViDex
    显式设置两个材料：
    - YCB 物体：`(static=1.5, dynamic=1.0, restitution=0.1)`
      （`utils/ycb_object_utils.py:120`）。
    - 机器人 collision shape：`(static=1.5, dynamic=1.0,
      restitution=0.01)` + 每个 link `min_patch_radius=0.02`、
      `patch_radius=0.04`（`utils/common_robot_utils.py:163-168`）。

    我们之前的设置下机器人和物体都继承 PhysX 默认
    `(0.5, 0.5, 0.0)`——**两边都滑了 3×**，手指能碰到物体却生不出
    足够 Coulomb 摩擦把它抬起。症状：policy 走到 pregrasp
    （`pre_err < 0.05`）、合拢手指，但物体死活不抬（`obj_lift ≈ 0`）。
    修法：在 env `__post_init__` 里设
    `cfg.sim.physics_material =
       RigidBodyMaterialCfg(static_friction=1.5,
                            dynamic_friction=1.0,
                            restitution=0.0)`，
    它会成为任何**没有自带物理材料绑定**的刚体的默认值。桌子保留
    自己的 `(1.0, 0.5, 0.01)` override，对应
    `sim_env/relocate_env.py:98`。

    用 `/tmp/check_physics.py` 验证：运行时物体质量 `0.7678 kg`
    （之前 `0.2 kg`），`cfg.sim.physics_material` 报告
    `(1.5, 1.0, 0.0)`。

---

## 13. TensorBoard 诊断指标

`AllegroRelocateManagerEnv.step()` 每步将 ViViDex 风格的诊断标量
写入 `extras["log"]["Metrics/<name>"]`。rsl_rl 的 logger 会在每个 PPO
iteration 里把它们沿 rollout 步求均值，最终落到 TensorBoard 的
`Metrics/` 分组下：

| TB 名称                       | 含义                                  | 对应 ViViDex 字段     |
| ----------------------------- | ------------------------------------- | --------------------- |
| `Metrics/control_error`       | DLS-IK Cartesian 残差（m）            | `control_error`       |
| `Metrics/hand_jpos_err`       | pregrasp 期 fingertip L2 误差（m）    | `hand_jpos_err`       |
| `Metrics/hand_mjpos_err`      | imitate 期 fingertip L2 误差（m）     | `hand_mjpos_err`      |
| `Metrics/obj_com_err`         | 物体 COM 跟踪误差（m）                | `obj_com_err`         |
| `Metrics/obj_rot_err`         | 物体旋转误差（归一化到 [0, 1]）       | （隐含在 reward 里）  |
| `Metrics/obj_lift`            | 物体相对初始高度的抬升量（m, ≥0）     | `obj_lift`            |
| `Metrics/num_finger_contacts` | 4 指中正在接触物体的桶数（max 4）     | （隐含在 reward 里）  |
| `Metrics/pregrasp_success`    | 各 env `_pregrasp_success` 的均值     | `pregrasp_success`    |
| `Metrics/imitate_steps`       | 每 env imitate 总步数（stage 0/1 恒值）| `imitate_steps`       |
| `Metrics/stage`               | 课程 stage（0/1/2）                   | `stage`               |

聚合粒度：`Metrics/X` 是 **per-step 各 env 均值再沿 rollout 求均值**
（与 `Episode_Reward/*` 沿 episode 求和后求均值的口径不同）。

---

## 14. 外部使用

要在其它项目里用本包：

```bash
pip install -e /root/workspace/rl_grasp/isaaclab_dextrous_grasp
```

然后：

```python
import gymnasium as gym
import isaaclab_dextrous_grasp  # 注册 gym ID

env = gym.make("Isaac-AllegroUR5-Relocate-v0", cfg=env_cfg)
```

env 遵守标准 IsaacLab manager-based 协议，可直接接入任何第三方
`RslRlVecEnvWrapper` / `RslRlOnPolicyRunnerCfg` 流水线。

---

## 15. 验证清单

在 `env_isaaclab` conda 环境中跑下列命令：

```bash
# 1. 形状健康检查 + 20 步零动作 rollout
python scripts/smoke_test.py --num_envs 4 --headless

# 2. 端到端 PPO 单 iteration
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 64 --headless --max_iterations 1
```

通过的话会打印 `obs.shape == (4, 393)`、`act dims == 22`，以及
rsl-rl runner banner 的 `Mean action std: 0.20`（iteration 0）。

---

## 16. 轨迹可视化

`scripts/visualize_trajectory.py` 用 headless 模式把任意一条
trajectory 录成 MP4：

- 物体每一控制步被运动学地"瞬移"到目标位姿
- 6 个 sphere marker 显示每帧目标 palm + 4 fingertip + 物体目标位置
- 机器人保持复位时的 pregrasp 初始姿态

```bash
python scripts/visualize_trajectory.py \
    --trajectory ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 \
    --num_steps 75 --video_dir /tmp/traj_vis \
    --cam_eye 0.59 0.09 0.29 --cam_lookat 0.35 0.35 0.12
```

`--cam_eye` / `--cam_lookat` 用 env-local 坐标，方便就近看抓取细节。
