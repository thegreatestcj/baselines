# Vendored upstream repos

Third-party baseline code is vendored directly in this repo (their `.git` removed).
Local modifications are committed on top; diff against the commits below to see them.

| Dir | Upstream | Pinned commit |
|---|---|---|
| PAC-NeRF/ | https://github.com/xuan-li/PAC-NeRF.git | b613048557d0648e885697ececbef80297defac0 |
| GIC/ | https://github.com/Jukgei/gic.git | b523851ed4343109ff67b0ea123e0a154af2a40c |
| Spring-Gaus/ | https://github.com/Colmar-zlicheng/Spring-Gaus.git (+submodules) | 62a1bb5dbe83fe4396efa7048d3226754ac8fe1d |
| MASIV/ | https://github.com/Skaldak/MASIV.git | (to vendor after smoke run; currently at /scr/chujunta/gic_staging/MASIV, f05ff17) |

## Local modifications so far

- PAC-NeRF/lib/pac_nerf.py — `TI_DEVICE_MEMORY_GB` env var overrides taichi
  `device_memory_fraction` (H200s are shared; fraction-based preallocation OOMs).
- GIC/train_dynamic.py — same `TI_DEVICE_MEMORY_GB` override.
- Spring-Gaus/lib/models/gaus/render.py — rasterizer returns 3+ values in our
  installed diff_gaussian_rasterization; unpack with `*_`.
- Spring-Gaus/train.py — `torch.backends.cuda.preferred_linalg_library("magma")`
  (cusolverDnCreate fails on some shared GPUs).
