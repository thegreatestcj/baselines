# PhysON baselines — first single-scene runs (2026-09-25)

One scene per method, run end-to-end through the task queue (`gen_tasks.sh` →
`run_queue.sh`, see the top-level README). Every number below comes from the files
next to this README; the overlays are the standard presentation (left GT frame of
the held-out camera; right GT silhouette red, prediction cyan, overlap white, per-frame IoU).

## OmniPhysGS — `singleobject_heterogeneous_new/0_0` (bread, soft shell E=8e4 / hard core E=1.8e6, vortex force)

Files: `physon_singleobject_heterogeneous_new/omniphysgs/0_0.json`, `0_0_overlay.mp4`,
`0_0_overlay_frames.png`, `0_0_overlay_iou.json`. Run: 10 epochs × 6 stages × 10 steps
(600 Adam steps, 4 views/frame), 6.2 h on one H200, closed-form v0 + 30 refinement steps.

| | CD (10³ mm², 8192 samples) | EMD (m) | PSNR / SSIM / LPIPS (cam 1) | silhouette IoU |
|---|---|---|---|---|
| OmniPhysGS (default expert set, incl. fluid experts) | 10.95 | 0.149 | 24.2 / 0.930 / 0.287 | 0.60 |
| OmniPhysGS (solid expert set, `0_0_solid*`) | 10.19 | 0.132 | 25.2 / 0.954 / 0.135 | 0.79 |
| MOSIV paper's OmniPhysGS-RGB baseline on MOSIV's own benchmark (Table 1) | 11.79 | | PSNR 25.9 | |
| reference: physics oracle, GT soft-region params (E=8e4, ν=.38, Corotated, no plasticity) | 8.81 | | 25.4 / 0.966 | 0.75 |
| reference: physics oracle, GT hard-region params (E=1.8e6, ν=.30) | 5.29 | | 26.4 / 0.969 | 0.86 |

Per-frame CD of the default run: 1.1 (frame 0), 1.3, 2.5, 9.2, 16.5, 22.7 (frame 40), 31.1 (frame 47).
The fit assigned every particle to one Corotated expert (E=1.48e6, ν=0.42; GT core 1.8e6/0.30,
shell 8e4/0.38 — no second region was found) and 81 % of the particles to the Drucker–Prager
plasticity expert (+19 % fluid reset), so the object crumbles after contact (cyan debris in the
overlay from frame ~9 on). The straight-through expert mixture is the weak point on this data: it
collapsed to all-fluid in epoch 0 and only partly recovered. The `0_0_solid` variant uses upstream's
released solid expert set (Corotated ×2 + StVK / Identity ×2 + VonMises + Drucker–Prager) and keeps
Identity plasticity for 94 % of the particles and one Corotated expert (E=5.8e5, ν=0.385) for all
of them: no crumbling (IoU 0.98 until contact, 0.72 at the end), CD 1.1 before contact and 16–21
after it — the stiffness lands between the two GT regions and the two-region structure is not found. The references show what the same pipeline gives with the true
single-region parameters (no fitting): the observation/render path itself is sound (IoU 0.98 before
contact).

## MOSIV — `multiobject_heterogeneous_new/0_11` (apple elastic E=4.5e5 + bread plasticine E=6e5, σ_y=1.8e4; curl-gust force up to 8 m/s²)

Files: `physon_multiobject/mosiv/0_11.json`, `0_11_overlay.mp4`, `0_11_overlay_frames.png`,
`0_11_overlay_iou.json`; per-object numbers and the fitted parameters in
`MOSIV/output/physon/multiobject_heterogeneous_new/0_11/{prediction_metrics.json,0_11-pred.json}`.
Run: object-aware 3DGS 40k iterations, lifting (152k particles), 80 velocity iterations,
80 parameter iterations (upstream default 300; 2.3 min/iteration over 48 frames), 4.8 h on one H200.
The rollout re-simulates the fitted scene with GIC's CFL handling (dt halved on a CFL failure and
the rollout restarted, as in training; here 400 substeps/frame were needed).

| | CD all (10³ mm²) | CD apple | CD bread | EMD (m) | silhouette IoU (cam 0) |
|---|---|---|---|---|---|
| MOSIV | 0.57 | 0.56 | 0.46 | 0.037 | 0.81 |
| MOSIV paper, its own 2-object benchmark (observable frames, Table 1) | 1.26 | | | | PSNR 30.5 |

Per-frame CD (all / apple / bread): 0.47/0.39/0.47 (frame 0), 0.36/0.35/0.28 (8), 0.41/0.35/0.39 (16),
0.48/0.50/0.36 (24), 0.61/0.60/0.51 (32), 0.76/0.80/0.58 (40). IoU stays 0.83–0.86 until frame ~32 and
drops to 0.67 at frame 47 (the apple's late deformation/contact). Fitted parameters vs GT:

| object | fitted | GT |
|---|---|---|
| apple (elastic) | E=9.7e4, ν=−0.41, v0=(0.078, 0.006, −0.010) | E=4.5e5, ν=0.36, v0=(0.09, 0.015, 0) |
| bread (plasticine) | E=6.8e3, ν=0.14, σ_y=6.6e3, v0=(−0.086, 0.038, 0.066) | E=6e5, ν=0.34, σ_y=1.8e4, v0=(−0.09, −0.015, 0) |

Trajectories and silhouettes are recovered well; the material parameters themselves end far too
soft (and MOSIV's Poisson bound [−0.99, 0.5] lets ν go negative) — the motion here is dominated by
the known gust, so the geometry loss constrains stiffness weakly.

## Reproduce / next scenes

```bash
bash gen_tasks.sh mosiv_physon_mo omniphysgs_physon_het > tasks.txt   # all ready scenes
bash run_queue.sh <gpus> 1
```
`MOSIV_PHYSON_CONVERT_ARGS='--iter_cnt 80'` was used here (default 300 ≈ 18 h/scene);
`OMNIPHYSGS_PHYSON_ARGS='material.elasticity_physicals=[CorotatedElasticity,CorotatedElasticity,StVKElasticity] material.plasticity_physicals=[IdentityPlasticity,IdentityPlasticity,VonMisesPlasticity,DruckerPragerPlasticity]'`
reproduces the solid-expert variant.
