# 常用命令速查

> 工作目录默认 `/root/workspace/rl_grasp/isaaclab_dextrous_grasp`，conda env `env_isaaclab`。

---

## 1. 训练 (`scripts/train.py`)

### 1.1 标配：4096 env，无视频

```bash
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 4096 --headless
```

- `TaskCfg.auto_curriculum=True` 默认开启：当滚动 buffer 里的 per-episode `pregrasp_success ≥ 0.95`（默认阈值）且累计 episode 数 ≥ `num_envs`（默认 min）后，env 会打印 `[CURRICULUM] stage X -> Y` 并强制 `_reset_idx` 全 env，下一步立即按新 stage 取样 (x, y, θ)，无需重启。`stage` 上限 `curriculum_max_stage=3`（0→1→2→3 三次升级）。
- 4 个 stage：0 = canonical；1 = U[0.30, 0.40] xy；2 = + ±15° yaw（以上对齐 ViViDex）；3 = U[0.20, 0.40] xy + ±30° yaw（默认值，由 `TaskCfg.stage3_xy_range` / `stage3_yaw_abs` 配置）。stage 3 的上界与 stage 1/2 一致（0.40），只把下界从 0.30 推到 0.20，让升级更平滑。
- 检查点写到 `logs/rsl_rl/allegro_ur5_relocate/<timestamp>/model_*.pt`，每 50 iter 一份。

### 1.2 边训练边录像（每 200 iter 一段近距离视频）

```bash
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 4096 --headless \
    --video --video_interval 200
```

- 视频在 `logs/rsl_rl/.../videos/train/rl-video-step-N.mp4`，N 是当前**总仿真步数**（不是 PPO iteration）。换算关系：`N = iteration × num_envs × num_steps_per_env`。
- 默认相机 env-local 等距 `(2,2,2)` → `(0,0,0)`，距离 ~3.5 m。要更近用 `--video_cam_eye 1.05 0.95 0.55 --video_cam_lookat 0.4 0.4 0.18`。

### 1.3 从已有 checkpoint 续训（同 stage）

```bash
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 4096 --headless \
    --checkpoint logs/rsl_rl/allegro_ur5_relocate/2026-05-06_16-05-49/model_900.pt
```

### 1.4 强制单 stage（关掉 auto curriculum）

不接 CLI，直接在 `manager_env_cfg.py` 里改 `TaskCfg.auto_curriculum = False`，
或在脚本里 `env_cfg.task.auto_curriculum = False`。配合 `--stage 1` / `--stage 2`
跑分段实验。`scripts/train.py` 在 `__post_init__` 之后才赋值 `task.stage`，
但 `mdp/events.reset_trajectory_state` 每次 reset 都从 `env.cfg.task.stage`
活读，所以 CLI 覆盖一定生效（不再像旧代码那样被 `__post_init__` 静默吞掉）。

### 1.5 小规模 debug（少 env、少 iter、看 stdout）

```bash
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 32 --headless --max_iterations 5
```

---

## 2. Play / 推理 (`scripts/play.py`)

### 2.1 拿 checkpoint 跑几个 episode

```bash
python scripts/play.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 4 --headless --num_steps 120 \
    --checkpoint logs/rsl_rl/allegro_ur5_relocate/2026-05-06_16-05-49/model_900.pt
```

### 2.2 录视频

```bash
python scripts/play.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 1 --headless \
    --video --num_steps 120 \
    --checkpoint logs/rsl_rl/allegro_ur5_relocate/2026-05-06_16-05-49/model_900.pt
```

- 视频默认存到 `<ckpt 所在目录>/videos/play/rl-video-step-0.mp4`。
- ⚠️ **注意：与 train 不能并行**。两边都开 PhysX 会出现 KVDB 锁竞争，play 第一步会卡死几分钟。先 `Ctrl-C` 训练，跑完 play (~30 s)，再重启训练。
- ⚠️ headless + video 必须 `enable_cameras`，`play.py` 已经在 `--video` 时自动打开。

### 2.3 不同 stage 测试 generalization

```bash
python scripts/play.py --task ... --stage 1 --num_envs 16 \
    --checkpoint logs/.../model_XXXX.pt --num_steps 120
```

---

## 3. 轨迹可视化 (`scripts/visualize_trajectory.py`)

跑一遍 imitator 目标轨迹（不带 policy，仅 kinematic replay）+ marker，验证轨迹本身是否合理：

```bash
python scripts/visualize_trajectory.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 1 --headless \
    --num_steps 75 \
    --video_dir logs/traj_vis \
    --trajectory ycb-011_banana-20200709-subject-01-20200709_145401 \
    --stage 0
```

视频出在 `logs/traj_vis/trajectory-step-0.mp4`（前缀可以用 `--name_prefix` 改）。

默认相机 `eye=(1.00, 0.95, 0.55)` → `lookat=(0.35, 0.35, 0.05)`，从机械臂
外侧 NE 方向俯视、lookat z 压到 0.05，扁平物体（banana / plate）也在画面
中央；以前默认 `eye=(0.59, 0.09, 0.29)` → `(0.35, 0.35, 0.12)` 站在机械臂
和物体之间，看到的是手腕背面挡住物体，且 lookat 偏高让 banana 落在画面
下边缘——升级前的视频是这个原因。

要看 stage 3 的远角，把 lookat 拉到 stage 3 范围中心：`--cam_lookat 0.30 0.30 0.05`。

---

## 4. Replay action 序列 (`scripts/play_replay.py`)

把 ViViDex 录下来的 action `.npz` 在 IsaacLab 里复演一遍，用来 diff 物理一致性。

```bash
python scripts/play_replay.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 16 --headless \
    --replay_actions /path/to/actions.npz \
    --no_terminations \
    --video
```

- `--no_terminations` 把 pregrasp_failure / object_too_far / lost_contact 三个早终止全 stub 掉，只剩 time_out，保证 replay 跑满 60 步。
- 视频走近距离镜头（`(1.05, 0.95, 0.55)` → `(0.4, 0.4, 0.18)`），方便和 SAPIEN 一侧 `record_replay_data.py` 的视频左右拼对比。

---

## 5. Smoke test / IK test

跑一次完整 env 构造 + 几步 zero action，验证迁移没炸：

```bash
python scripts/smoke_test.py --num_envs 4 --num_steps 80 --headless
```

只验证 IK 收敛（不依赖 policy）：

```bash
python scripts/test_ik.py --headless
```

---

## 6. TensorBoard

```bash
cd logs/rsl_rl
tensorboard --logdir=./ --bind_all --port 6006
```

关注的几条曲线：

- `Metrics/pregrasp_success` ：**已对齐 ViViDex Monitor 语义**（最近 `curriculum_history_size`（默认 num_envs）个 finished episode 的 0/1 均值，稳态 0.99-1.0），过 `curriculum_threshold`（默认 0.95）自动升 stage。源码：`manager_env._record_episode_success` 在每次 reset 时由 `mdp/events.reset_trajectory_state` 调用，把 `_pregrasp_success[env_ids]` 推进环形 buffer。
- `Metrics/pregrasp_success_timewise` ：旧口径，per-step 平均 sticky bool，ceiling=`(imitate_steps - pregrasp_steps) / imitate_steps ≈ 0.75`，仅做调试参考，不要看这条判断成功率。
- `Metrics/stage` ：实时读 `env.cfg.task.stage`。auto-curriculum 一旦升级会瞬时跳变（因为升级当 step 强制 reset 全部 env），不会有过渡期分数。
- `Metrics/obj_lift` ：物体抬升量（m），ViViDex success_10 阈值 0.05。
- `Metrics/num_finger_contacts` ：5 桶（4 指 + 掌）中接触数，max=5。
- `Episode_Termination/*` ：终止类型分布，最后应几乎全 `time_out`。
- `Train/mean_reward`、`Loss/entropy`：常规 PPO 监控。

---

## 7. 临时 / 一次性 diagnostic 脚本

历史上写过的 `/tmp/*.py`（已删，不在 repo 里），任务做完后随写随丢：

| 脚本 | 用途 |
|---|---|
| `/tmp/check_frames.py` | 验证 obs / reward 用的坐标系是 env-local |
| `/tmp/verify_audit_fixes.py` | 验证 `*_tip` link / 15 个 contact sensor / `pre_err` |
| `/tmp/check_physics.py` | runtime 验证物体质量、`sim.physics_material` |
| `/tmp/check_traj_z.py` | 验证 imitator 物体 z 轨迹是上升的 |
| `/tmp/diag_rollout.py` | 抓某个 ckpt 的 contact / lift / action 详情 |

需要时按 §10 §11 §12 在 `docs/migration.md` 里找模板复制。

---

## 8. 进程 / GPU / 杀任务

```bash
# 看哪个 python 在用 GPU
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 看 train.py 还在不在
ps -eo pid,stat,etime,pcpu,cmd | grep -E "train\.py|play\.py" | grep -v grep

# 杀一个 stuck 的 Isaac Sim
kill -9 <pid>; sleep 2; nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

---

## 9. 常见踩坑速查（详见 `docs/migration.md`）

| 现象 | 原因 | 修法 |
|---|---|---|
| `Metrics/pregrasp_success` 一直 ≤ 0.75 | 旧口径数学上限就是 45/60 | 看 `Metrics/pregrasp_success`（新版）应是 ~1.0 |
| 视频地面变白 | `num_envs ≥ 4096` 时 env_0 跑到 100m 默认地面之外 | 已修：地面改 500m × 500m |
| play.py 头几分钟没动静 | `--video` 触发 Replicator 懒加载 | 头一次 ~60 s 是正常的，等就行 |
| play.py 直接卡死 | 与 train.py 抢 PhysX KVDB | 先停 train，再跑 play |
| `pregrasp_failure` 居高不下 | PD 增益 / IK 速度前馈 / 接触阈值 | 见 migration.md §12 / §10 |
| 物体一碰就飞 | 质量被硬写成 0.2 kg | 已修：`density=1000` 让 PhysX 算质量 |
| `Metrics/stage` 一次跳两级 (0→2) | 升级时只清 buffer 没 force-reset，且 `min_episodes=200` 太低 | 已修：升级时 `_reset_idx(all)` + `min_episodes=num_envs`；详见 migration.md §12 #24 |
| 某些 YCB 物体（banana / gelatin_box / cracker_box ...）渲染时透明 / 看不见 | `assets/ycb/visual/<obj>/textured_simple.obj.mtl` 里 `Tr 1.000000` = 完全透明（YCB 数据集的已知 bug，16 个物体里 12 个是这个值，只有 mustard / tomato_soup / potted_meat 例外） | 已修：把所有 `.mtl` 的 `Tr 1.000000` 全改成 `Tr 0.000000`，并清空 `cache/usd/ycb/` 让 MeshConverter 用修好的 mtl 重新转换。下次启动训练 / visualize 自动重生成。 |
