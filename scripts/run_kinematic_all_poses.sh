#!/usr/bin/env bash
#
# Run the full kinematic-replay pipeline for all 3 mustard-bottle poses.
#
# For each pose this script:
#   1. Re-records the SAPIEN-side ViViDex rollout (60 steps) via
#      ``vividex_sapien/tools/record_replay_data.py``. The resulting npz
#      contains per-step (robot_qpos, object_pos, object_quat, ...).
#   2. Plays the npz back kinematically in IsaacLab via
#      ``scripts/play_kinematic.py`` (no policy, no physics control --
#      every frame is a direct PhysX teleport of joints + object pose).
#
# All 6 output mp4s are collected under
# ``logs/kinematic_replay/mustard_pose{1,2,3}_{sapien,kinematic}.mp4``.
#
# Usage::
#
#     conda activate env_isaaclab
#     bash scripts/run_kinematic_all_poses.sh
#
# Expected runtime: ~2.5 minutes (3x SAPIEN recording ~10s + 3x IsaacLab
# kinematic replay ~30s each, dominated by IsaacSim startup).

set -euo pipefail

LAB_ROOT="/root/workspace/rl_grasp/isaaclab_dextrous_grasp"
VIVI_ROOT="/root/workspace/rl_grasp/vividex_sapien"
VIVI_PYTHON="${VIVI_PYTHON:-/root/autodl-tmp/rl/bin/python}"

REPLAY_TMP="/tmp/replay"
OUT_DIR="${LAB_ROOT}/logs/kinematic_replay"
mkdir -p "${REPLAY_TMP}" "${OUT_DIR}"

# Trajectory name per pose. The IsaacLab ckpt dirs live under
# ``logs/mustard_bottle/poseN/``; each was trained on a separate
# norm_trajectories/*.npz episode.
declare -A TRAJ
TRAJ[1]="ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"
TRAJ[2]="ycb-006_mustard_bottle-20200908-subject-05-20200908_144439"
TRAJ[3]="ycb-006_mustard_bottle-20200928-subject-07-20200928_144226"

for pose in 1 2 3; do
    traj="${TRAJ[$pose]}"
    echo
    echo "##############################################################"
    echo "# POSE ${pose}   trajectory=${traj}"
    echo "##############################################################"

    # -----------------------------------------------------------------
    # 1) SAPIEN: re-run ViViDex ckpt, dump npz + mp4
    # -----------------------------------------------------------------
    echo "[1/2] recording SAPIEN rollout ..."
    (
        cd "${VIVI_ROOT}"
        PYTHONPATH=. "${VIVI_PYTHON}" tools/record_replay_data.py \
            --checkpoint_dir "${LAB_ROOT}/logs/mustard_bottle/pose${pose}" \
            --output_dir "${REPLAY_TMP}" \
            --num_steps 60 \
            2>&1 | tail -6
    )
    cp "${REPLAY_TMP}/vividex_${traj}_stage0.mp4" \
       "${OUT_DIR}/mustard_pose${pose}_sapien.mp4"

    # -----------------------------------------------------------------
    # 2) IsaacLab: kinematic teleport replay, dump mp4
    # -----------------------------------------------------------------
    echo "[2/2] running IsaacLab kinematic replay ..."
    work="/tmp/kinematic_${pose}"
    rm -rf "${work}"
    (
        cd "${LAB_ROOT}"
        python -u scripts/play_kinematic.py \
            --task Isaac-AllegroUR5-Relocate-v0 \
            --replay_data "${REPLAY_TMP}/actions_${traj}_stage0.npz" \
            --trajectory "${traj}" \
            --video --video_dir "${work}" \
            --name_prefix replay \
            --num_steps 60 \
            2>&1 | tail -3
    )
    cp "${work}/replay-step-0.mp4" \
       "${OUT_DIR}/mustard_pose${pose}_kinematic.mp4"
done

echo
echo "##############################################################"
echo "# DONE -- collected videos:"
echo "##############################################################"
ls -la "${OUT_DIR}"/mustard_pose*.mp4
