#!/usr/bin/env python
"""Summarise the PhysON baseline results into one table.

Walks results/physon_*/<method>/<scene>.json (eval/eval_scene.py output), the overlay IoU files
(<scene>_overlay_iou.json) and, when present, the methods' own parameter/metric dumps
(OmniPhysGS outputs/PhysON/<subset>/<scene>/params.json, MOSIV output/physon/<subset>/<scene>/
{prediction_metrics.json,<scene>-pred.json}) and prints a markdown table per method plus means;
also writes results/physon_summary.csv.

Metrics (all over the 48 observed frames, no future split in these subsets):
  CD    symmetric squared Chamfer distance vs the GT particles, 8192 samples, in 10^3 mm^2
        (the PAC-NeRF / MASIV / Spring-Gaus convention)
  EMD   earth mover's distance on 2048 samples, metres
  PSNR/SSIM/LPIPS  held-out camera renders vs GT frames (OmniPhysGS; MOSIV only renders silhouettes)
  IoU   silhouette IoU of the held-out camera (eval/overlay_video.py)
  per-object CD (MOSIV), fitted parameters vs GT (both) from the method dumps

  python eval/physon_summary.py [--results results]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fmt(v, nd=3):
    return "--" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()
    root = Path(args.results)
    rows = []
    for f in sorted(root.glob("physon_*/*/*.json")):
        if f.name.endswith("_overlay_iou.json") or f.name.endswith("_iou.json"):
            continue
        subset_dir, method, scene = f.parent.parent.name, f.parent.name, f.stem
        d = load(f) or {}
        part = (d.get("particles") or {}).get("all") or {}
        img = (d.get("images") or {}).get("all") or {}
        iou = load(f.parent / f"{scene}_overlay_iou.json") or {}
        row = dict(subset=subset_dir.replace("physon_", ""), method=method, scene=scene,
                   cd=part.get("cd"), emd=part.get("emd"), psnr=img.get("psnr"), ssim=img.get("ssim"), lpips=img.get("lpips"),
                   iou=iou.get("iou_mean"), iou_last=(iou.get("iou_per_frame") or [None])[-1])
        base_scene = scene.replace("_solid", "")
        if method == "omniphysgs":
            for sub in ("singleobject_heterogeneous_new", "singleobject_heterogeneous"):
                pj = load(ROOT / "OmniPhysGS" / "outputs" / "PhysON" / sub / scene / "params.json")
                if pj:
                    used = [e for e in pj["experts_elasticity"] if e.get("usage", 0) > 0.05]
                    row["fit"] = "; ".join(f"{e['name']} E={e.get('E', 0):.3g} nu={e.get('nu', 0):.2f} ({e['usage']:.0%})" for e in used)
                    row["fit"] += " | plast " + ", ".join(f"{e['name']}:{e['usage']:.0%}" for e in pj["experts_plasticity"] if e.get("usage", 0) > 0.05)
                    row["gt"] = "; ".join(f"{r['material_parameters'].get('kind')} E={r['material_parameters'].get('E', 0):.3g} nu={r['material_parameters'].get('nu', 0):.2f}"
                                          for r in pj.get("gt_regions", []) if "material_parameters" in r)
                    break
        elif method == "mosiv":
            pm = load(ROOT / "MOSIV" / "output" / "physon" / row["subset"] / scene / "prediction_metrics.json")
            if pm:
                ks = sorted(k for k in pm["mean"] if k.startswith("cd_obj"))
                row["cd_obj"] = " / ".join(fmt(pm["mean"].get(k)) for k in ks)
                fits = []
                for o in pm.get("fitted") or []:
                    mp = o.get("mat_params", {})
                    fits.append(f"{o['name']}: " + ", ".join(f"{k}={v:.3g}" for k, v in mp.items() if k not in ("material", "rho")))
                row["fit"] = "; ".join(fits)
            md = load(ROOT / "MOSIV" / "data" / "PhysON_mosiv" / row["subset"] / scene / "metadata.json")
            if md:
                keys = [k for k in md if k.startswith("obj") and k[3:].isdigit()]
                row["gt"] = "; ".join(f"{md[k]['geometry']}: " + ", ".join(f"{kk}={vv:.3g}" for kk, vv in md[k]["gt_material_parameters"].items() if isinstance(vv, (int, float)))
                                      for k in sorted(keys, key=lambda k: int(k[3:])))
        rows.append(row)
    if not rows:
        print("no PhysON results under", root)
        return
    cols = ["subset", "method", "scene", "cd", "emd", "psnr", "ssim", "lpips", "iou", "iou_last", "cd_obj", "fit", "gt"]
    with open(root / "physon_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        print(f"\n### {method} ({len(sub)} runs)\n")
        print("| scene | CD (10³ mm²) | EMD (m) | PSNR | SSIM | LPIPS | IoU mean | IoU last | per-object CD | fitted | GT |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            print("| " + " | ".join([r["scene"], fmt(r["cd"]), fmt(r["emd"]), fmt(r["psnr"], 2), fmt(r["ssim"]), fmt(r["lpips"]),
                                     fmt(r["iou"]), fmt(r["iou_last"], 2), r.get("cd_obj", "--"), r.get("fit", "--"), r.get("gt", "--")]) + " |")
        def mean(k):
            v = [r[k] for r in sub if isinstance(r.get(k), (int, float))]
            return sum(v) / len(v) if v else None
        print("| **mean** | " + " | ".join(fmt(mean(k)) for k in ("cd", "emd")) + " | " + fmt(mean("psnr"), 2) + " | " +
              " | ".join(fmt(mean(k)) for k in ("ssim", "lpips", "iou")) + " | " + fmt(mean("iou_last"), 2) + " | | | |")
    print(f"\nwrote {root / 'physon_summary.csv'}")


if __name__ == "__main__":
    main()
