#!/usr/bin/env bash
#
# Run the kinematic-replay pipeline for ALL ViViDex objects x poses.
#
# Five objects, three poses each (15 pairs total):
#
#   large_clamp        pose1..3   (ycb-052_extra_large_clamp)
#   mug                pose1..3   (ycb-025_mug)
#   mustard_bottle     pose1..3   (ycb-006_mustard_bottle)
#   sugar_box          pose1..3   (ycb-004_sugar_box)
#   tomato_soup_can    pose1..3   (ycb-005_tomato_soup_can)
#
# For every (object, pose) the script:
#   1. Re-records the SAPIEN-side ViViDex rollout (60 steps) using
#      ``vividex_sapien/tools/record_replay_data.py`` and the ViViDex ckpt
#      at ``vividex_sapien/logs/<object>/pose<N>/restore_checkpoint.zip``.
#   2. Plays the resulting npz back kinematically in IsaacLab with
#      ``scripts/play_kinematic.py``. The script teleports robot joints
#      and the object pose directly into PhysX every frame, with no
#      policy and no physics control.
#   3. Symlinks the matching trajectory npz from ViViDex's
#      ``norm_trajectories/`` into IsaacLab's ``trajectories/`` if it's
#      not already there (needed by the env at reset time).
#
# Output: ``logs/kinematic_replay/<object>_pose<N>_{sapien,kinematic}.mp4``
# Failures on a single (object, pose) are logged but do not abort the
# loop -- you'll still get videos for everything that did succeed.
#
# Usage::
#
#     conda activate env_isaaclab
#     bash scripts/run_kinematic_all_objects.sh
#
#     # Or restrict to a subset:
#     OBJECTS="mug sugar_box" bash scripts/run_kinematic_all_objects.sh
#     POSES="1 2"           bash scripts/run_kinematic_all_objects.sh
#
# Expected runtime: ~25 minutes (15 IsaacLab cycles ~30s each + the
# corresponding SAPIEN runs ~10s each, dominated by IsaacSim startup).

set -u   # NOTE: no -e on purpose; we want to keep going on per-pose failures.

LAB_ROOT="/root/workspace/rl_grasp/isaaclab_dextrous_grasp"
VIVI_ROOT="/root/workspace/rl_grasp/vividex_sapien"
VIVI_PYTHON="${VIVI_PYTHON:-/root/autodl-tmp/rl/bin/python}"

REPLAY_TMP="/tmp/replay"
OUT_DIR="${LAB_ROOT}/logs/kinematic_replay"
TRAJ_DIR="${LAB_ROOT}/trajectories"
mkdir -p "${REPLAY_TMP}" "${OUT_DIR}" "${TRAJ_DIR}"

# -------------------------------------------------------------------
# (object, pose) -> trajectory_name mapping. Discovered by inspecting
# ``vividex_sapien/logs/<obj>/pose<N>/exp_config.yaml`` :: env.name.
# Keep these literal so the script doesn't depend on yq / yaml parsing.
# -------------------------------------------------------------------
declare -A TRAJ

TRAJ[large_clamp,1]="ycb-052_extra_large_clamp-20200709-subject-01-20200709_152843"
TRAJ[large_clamp,2]="ycb-052_extra_large_clamp-20200820-subject-03-20200820_144829"
TRAJ[large_clamp,3]="ycb-052_extra_large_clamp-20201002-subject-08-20201002_112816"

TRAJ[mug,1]="ycb-025_mug-20200709-subject-01-20200709_150949"
TRAJ[mug,2]="ycb-025_mug-20200928-subject-07-20200928_154547"
TRAJ[mug,3]="ycb-025_mug-20200820-subject-03-20200820_143304"

TRAJ[mustard_bottle,1]="ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"
TRAJ[mustard_bottle,2]="ycb-006_mustard_bottle-20200908-subject-05-20200908_144439"
TRAJ[mustard_bottle,3]="ycb-006_mustard_bottle-20200928-subject-07-20200928_144226"

TRAJ[sugar_box,1]="ycb-004_sugar_box-20200918-subject-06-20200918_113441"
TRAJ[sugar_box,2]="ycb-004_sugar_box-20200903-subject-04-20200903_104157"
TRAJ[sugar_box,3]="ycb-004_sugar_box-20200908-subject-05-20200908_143931"

TRAJ[tomato_soup_can,1]="ycb-005_tomato_soup_can-20200709-subject-01-20200709_142853"
TRAJ[tomato_soup_can,2]="ycb-005_tomato_soup_can-20201015-subject-09-20201015_143403"
TRAJ[tomato_soup_can,3]="ycb-005_tomato_soup_can-20200709-subject-01-20200709_142926"

OBJECTS="${OBJECTS:-large_clamp mug mustard_bottle sugar_box tomato_soup_can}"
POSES="${POSES:-1 2 3}"

declare -a SUCCESS=()
declare -a FAILED=()

for obj in ${OBJECTS}; do
    for pose in ${POSES}; do
        traj="${TRAJ[${obj},${pose}]:-}"
        if [[ -z "${traj}" ]]; then
            echo "[SKIP] no trajectory entry for ${obj} pose${pose}"
            continue
        fi

        tag="${obj}_pose${pose}"
        ckpt_dir="${VIVI_ROOT}/logs/${obj}/pose${pose}"

        echo
        echo "##############################################################"
        echo "# ${tag}   trajectory=${traj}"
        echo "##############################################################"

        if [[ ! -f "${ckpt_dir}/restore_checkpoint.zip" ]]; then
            echo "[FAIL] no restore_checkpoint.zip under ${ckpt_dir}"
            FAILED+=("${tag} (no ckpt)")
            continue
        fi

        # Make sure IsaacLab can find the trajectory npz.
        src_npz="${VIVI_ROOT}/norm_trajectories/${traj}.npz"
        dst_npz="${TRAJ_DIR}/${traj}.npz"
        if [[ ! -e "${dst_npz}" ]]; then
            if [[ -f "${src_npz}" ]]; then
                ln -sf "${src_npz}" "${dst_npz}"
                echo "[OK]   symlinked trajectory: ${dst_npz} -> ${src_npz}"
            else
                echo "[FAIL] no source trajectory npz at ${src_npz}"
                FAILED+=("${tag} (no traj)")
                continue
            fi
        fi

        # -----------------------------------------------------------------
        # 1) SAPIEN side: record rollout + side video
        # -----------------------------------------------------------------
        echo "[1/2] recording SAPIEN rollout ..."
        (
            cd "${VIVI_ROOT}"
            PYTHONPATH=. "${VIVI_PYTHON}" tools/record_replay_data.py \
                --checkpoint_dir "${ckpt_dir}" \
                --output_dir "${REPLAY_TMP}" \
                --num_steps 60 \
                2>&1 | tail -6
        )
        sapien_mp4="${REPLAY_TMP}/vividex_${traj}_stage0.mp4"
        sapien_npz="${REPLAY_TMP}/actions_${traj}_stage0.npz"
        if [[ ! -f "${sapien_mp4}" || ! -f "${sapien_npz}" ]]; then
            echo "[FAIL] SAPIEN stage missing outputs for ${tag}"
            FAILED+=("${tag} (sapien crash)")
            continue
        fi
        cp "${sapien_mp4}" "${OUT_DIR}/${tag}_sapien.mp4"

        # -----------------------------------------------------------------
        # 2) IsaacLab side: kinematic teleport replay
        # -----------------------------------------------------------------
        echo "[2/2] running IsaacLab kinematic replay ..."
        work="/tmp/kinematic_${tag}"
        rm -rf "${work}"
        (
            cd "${LAB_ROOT}"
            python -u scripts/play_kinematic.py \
                --task Isaac-AllegroUR5-Relocate-v0 \
                --replay_data "${sapien_npz}" \
                --trajectory "${traj}" \
                --video --video_dir "${work}" \
                --name_prefix replay \
                --num_steps 60 \
                2>&1 | tail -3
        )
        kine_mp4="${work}/replay-step-0.mp4"
        if [[ ! -f "${kine_mp4}" ]]; then
            echo "[FAIL] IsaacLab stage produced no mp4 for ${tag}"
            FAILED+=("${tag} (lab crash)")
            continue
        fi
        cp "${kine_mp4}" "${OUT_DIR}/${tag}_kinematic.mp4"
        SUCCESS+=("${tag}")
    done
done

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
echo
echo "##############################################################"
echo "# SUMMARY"
echo "##############################################################"
echo "Succeeded (${#SUCCESS[@]}):"
for s in "${SUCCESS[@]:-}"; do echo "  - ${s}"; done
echo
echo "Failed (${#FAILED[@]}):"
for f in "${FAILED[@]:-}"; do echo "  - ${f}"; done
echo
echo "Output dir: ${OUT_DIR}"
ls -la "${OUT_DIR}"/*.mp4 2>/dev/null | tail -40
