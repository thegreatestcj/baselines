#!/usr/bin/env python
"""Generate a GIC predict-config for one novel-interaction variant.

Protocol (all methods identical): per scene exactly two variants, each drawn
from a range with the SCENE NAME as RNG seed — so every method gets the same
perturbation on a given scene, while different scenes get different ones
(no fixed constants; mirrors the datagen sampling policy).

  gravtilt:   gravity tilted by U[30,60] deg at azimuth U[0,360) deg,
              magnitude preserved.
  velperturb: identified v0's horizontal part rotated by U[90,270] deg and
              scaled by U[0.5,1.5]; if |v_horizontal| < 0.05 m/s, inject a
              sampled horizontal kick of U[0.3,0.8] m/s instead.
Material, geometry and boundaries stay at the identified/original values
({cid}-pred.json).

  python gen_novel_cfg.py <model_path> <template_predict_json> <variant> <cid>
prints the path of the generated config.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

model_path, template, variant, cid = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

scene = Path(model_path).name
seed = int(hashlib.sha256(f"{scene}:{variant}".encode()).hexdigest()[:8], 16)
rng = np.random.default_rng(seed)

pred = json.load(open(Path(model_path) / f"{cid}-pred.json"))
cfg = json.load(open(template))

phys = cfg["physics"]
phys["mat_params"] = pred["mat_params"]
phys["vel"] = list(pred["vel"])
phys["gravity"] = list(pred.get("gravity", phys.get("gravity", [0, -9.8, 0])))

if variant == "gravtilt":
    g = float(np.linalg.norm(phys["gravity"]))
    tilt = math.radians(rng.uniform(30, 60))
    azim = rng.uniform(0, 2 * math.pi)
    phys["gravity"] = [g * math.sin(tilt) * math.cos(azim),
                       -g * math.cos(tilt),
                       g * math.sin(tilt) * math.sin(azim)]
elif variant == "velperturb":
    vx, vy, vz = phys["vel"]
    h = math.hypot(vx, vz)
    if h < 0.05:
        kick = rng.uniform(0.3, 0.8)
        azim = rng.uniform(0, 2 * math.pi)
        phys["vel"] = [kick * math.cos(azim), vy, kick * math.sin(azim)]
    else:
        rot = math.radians(rng.uniform(90, 270))
        scale = rng.uniform(0.5, 1.5)
        c, s = math.cos(rot), math.sin(rot)
        phys["vel"] = [scale * (c * vx - s * vz), vy, scale * (s * vx + c * vz)]
else:
    sys.exit(f"unknown variant {variant}")

out = Path(model_path) / f"novel_{variant}.json"
json.dump(cfg, open(out, "w"), indent=4)
print(out)
