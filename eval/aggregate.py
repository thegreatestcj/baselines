#!/usr/bin/env python
"""Aggregate per-scene metric JSONs into one summary table.

eval_scene.py writes one JSON per scene under results/<method>/<scene>.json.
This walks results/, averages each metric over scenes per (method, window),
and writes results/summary.csv plus a markdown table to stdout.

Scene names like "elastic_3" contribute their prefix as a material group, so
per-material means come out of the same pass (--by_group).

  python eval/aggregate.py [--results results] [--by_group]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def scene_group(name):
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=Path(__file__).parent.parent / "results")
    ap.add_argument("--by_group", action="store_true",
                    help="also break means down by scene-name prefix (material)")
    args = ap.parse_args()
    root = Path(args.results)

    # rows[(method, group, window)][metric] -> list of values
    rows = defaultdict(lambda: defaultdict(list))
    n_scenes = defaultdict(set)
    for f in sorted(root.glob("*/*.json")):
        method, scene = f.parent.name, f.stem
        data = json.load(open(f))
        for block in ("particles", "images"):
            if block not in data:
                continue
            for window in ("observable", "future", "all"):
                means = data[block].get(window)
                if not means:
                    continue
                for metric, v in means.items():
                    keys = [(method, "all", window)]
                    if args.by_group:
                        keys.append((method, scene_group(scene), window))
                    for k in keys:
                        rows[k][metric].append(v)
                        n_scenes[k].add(scene)

    metrics = sorted({m for r in rows.values() for m in r})
    out = root / "summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "group", "window", "n_scenes"] + metrics)
        for (method, group, window), vals in sorted(rows.items()):
            w.writerow([method, group, window, len(n_scenes[(method, group, window)])]
                       + [f"{sum(v)/len(v):.4f}" if (v := vals.get(m)) else ""
                          for m in metrics])

    hdr = ["method", "group", "window", "n"] + metrics
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for (method, group, window), vals in sorted(rows.items()):
        cells = [method, group, window, str(len(n_scenes[(method, group, window)]))]
        cells += [f"{sum(v)/len(v):.3f}" if (v := vals.get(m)) else "--" for m in metrics]
        print("| " + " | ".join(cells) + " |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
