#!/bin/bash
# Train ViViDex state-based PPO policy for YCB objects that don't yet have ckpts.
# One invocation per (object, pose). Each task writes to logs/<short_name>/pose<i>/.
# train.py auto-resumes from restore_checkpoint.zip if interrupted.
#
# Usage:
#   bash scripts/train_new_objects.sh                  # train all objects below
#   bash scripts/train_new_objects.sh banana mug       # train only listed objects
#                                                     # (accepts short or NNN_ prefix)

set -euo pipefail

# Always run from the vividex_sapien repo root (parent of scripts/)
cd "$(dirname "$0")/.."

# (object_id_with_NNN_prefix  short_log_dir_name)
OBJECTS=(
  "011_banana:banana"
  "002_master_chef_can:master_chef_can"
  "003_cracker_box:cracker_box"
  "007_tuna_fish_can:tuna_fish_can"
  "008_pudding_box:pudding_box"
  "009_gelatin_box:gelatin_box"
  "010_potted_meat_can:potted_meat_can"
  "019_pitcher_base:pitcher_base"
  "021_bleach_cleanser:bleach_cleanser"
  "036_wood_block:wood_block"
  "061_foam_brick:foam_brick"
)

# Optional filter from CLI args; empty means "all"
FILTER=("$@")

want_object() {
  if [ "${#FILTER[@]}" -eq 0 ]; then return 0; fi
  local id="$1" short="$2"
  for f in "${FILTER[@]}"; do
    if [ "$f" = "$id" ] || [ "$f" = "$short" ]; then return 0; fi
  done
  return 1
}

# wandb offline so training never blocks on network
export WANDB_MODE="${WANDB_MODE:-offline}"
# SAPIEN Vulkan: prefer NVIDIA ICD when multiple ICDs exist (lavapipe vs nvidia).
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"

for entry in "${OBJECTS[@]}"; do
  obj_id="${entry%%:*}"
  short="${entry##*:}"

  if ! want_object "$obj_id" "$short"; then
    echo "[skip] $obj_id"
    continue
  fi

  # Pick 3 demos from 3 different subjects (subject-01/02/03 by default).
  mapfile -t demos < <(ls "norm_trajectories/ycb-${obj_id}-"*.npz \
      | xargs -n1 basename \
      | awk -F'-' '!seen[$5]++' \
      | head -3)

  if [ "${#demos[@]}" -lt 3 ]; then
    echo "[warn] $obj_id only has ${#demos[@]} unique-subject demos, skipping"
    continue
  fi

  for i in 0 1 2; do
    seq_name="${demos[$i]%.npz}"
    pose_idx=$((i + 1))
    out_dir="logs/${short}/pose${pose_idx}"

    # If a previous attempt wrote exp_config.yaml but no restore_checkpoint.zip,
    # train.py would assert. Clear that broken state so we restart cleanly.
    if [ -f "${out_dir}/exp_config.yaml" ] && [ ! -f "${out_dir}/restore_checkpoint.zip" ]; then
      echo "[clean] removing incomplete ${out_dir} (no restore_checkpoint.zip)"
      rm -rf "${out_dir}"
    fi

    mkdir -p "${out_dir}"
    echo "============================================================"
    echo "[train] ${short}/pose${pose_idx}  <-  ${seq_name}"
    echo "        out_dir=${out_dir}"
    echo "============================================================"

    python tools/train.py \
      env.name="${seq_name}" \
      env.norm_traj=True \
      hydra.run.dir="${out_dir}" \
      2>&1 | tee -a "${out_dir}/train.log"
  done
done

echo "All requested training tasks done."
