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

- `auto_curriculum=True` 是默认开启的，看到 TB 里 `Metrics/pregrasp_success ≥ 0.95` 后会自动 `[CURRICULUM] stage 0 -> 1`，无需重启。
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
跑分段实验。

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
    --trajectory ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 \
    --stage 0
```

视频出在 `logs/traj_vis/<obj>_stage<N>-step-0.mp4`。

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

- `Metrics/pregrasp_success` ：**已对齐 ViViDex Monitor 语义**（最近 4096 集 episode 的 0/1 均值，稳态 0.99-1.0），过 0.95 自动升 stage。
- `Metrics/pregrasp_success_timewise` ：旧口径，ceiling=0.75，仅做调试参考，不要看这条判断成功率。
- `Metrics/stage` ：实时 stage（0/1/2），过渡期可能是分数（如 0.875）。
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
