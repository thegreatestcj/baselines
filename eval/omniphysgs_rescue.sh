#!/bin/bash
# Re-evaluate a finished OmniPhysGS PhysON scene whose free rollout blew up numerically (the
# fitted stiffness needs a finer substep than the dataset's own): re-run only the rollout from
# the last checkpoint with sim.dt halved (default 1.0e-4), then the metrics and the overlay.
#
#   bash eval/omniphysgs_rescue.sh singleobject_heterogeneous 1_4 [1.0e-4] [extra fit.py k=v ...]
#
# Detect candidates with: grep -l "non_finite_from_frame\": [0-9]" OmniPhysGS/outputs/PhysON/*/*/metrics.json
# or a mean CD in the hundreds in results/physon_<subset>/omniphysgs/<scene>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
sub=${1:?subset}; s=${2:?scene}; dt=${3:-1.0e-4}; shift 3 2>/dev/null || shift $#
src="data/PhysON/$sub/$s"; out="outputs/PhysON/$sub/$s"; res="../results/physon_$sub/omniphysgs/$s"
cam=$(grep -o 'a_[0-9]*_0' "OmniPhysGS/$src/transforms_test.json" | head -n1 | cut -d_ -f2)
cd OmniPhysGS
"$OMNIPHYSGS_PY" fit.py --config "$src/config.yaml" --tag "PhysON/$sub/$s" --eval_only "sim.dt=$dt" "$@"
"$BASELINES_PY" ../eval/eval_scene.py --pred_plys "$out/particles/*.ply" --gt_plys "$src/point_clouds/*.ply" \
  --pred_frames "$out/renders_test/*.png" --gt_frames "$out/gt_test/*.png" --out "$res.json"
"$BASELINES_PY" ../eval/overlay_video.py --gt_rgba "$src/data/a_${cam}_*.png" --pred_mask "$out/renders_test_alpha/*.png" \
  --title OmniPhysGS --subtitle "$sub/$s (dt $dt)" --out "${res}_overlay"
echo "[rescue] $sub/$s re-evaluated at dt=$dt -> $res.json"
