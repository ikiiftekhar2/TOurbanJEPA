"""Build a tile manifest excluding flat tiles (aggressive: union-of-bottom-20%).

A tile is DROPPED if ANY of these is true:
  global_std       in bottom 20% (low overall pixel variance)
  mean_patch_std   in bottom 20% (low per-patch variance)
  edge_density     in bottom 20% (few edges)
  flat_patch_frac  in top 20%    (many flat patches)

Output: data/ortho/metadata/train_textured.txt  (one tile basename per line)
"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_PATH = "experiments/v4_phase1_5/tile_flatness.csv"
OUT_PATH = "data/ortho/metadata/train_textured.txt"
DROP_CUT = 0.20  # bottom-20% on lower-is-flat, top-20% on flat_patch_frac

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))
for r in rows:
    for k in ("global_std", "mean_patch_std", "flat_patch_frac", "edge_density"):
        r[k] = float(r[k])

gs = np.array([r["global_std"] for r in rows])
mps = np.array([r["mean_patch_std"] for r in rows])
ed = np.array([r["edge_density"] for r in rows])
flat = np.array([r["flat_patch_frac"] for r in rows])

gs_cut = np.percentile(gs, DROP_CUT * 100)
mps_cut = np.percentile(mps, DROP_CUT * 100)
ed_cut = np.percentile(ed, DROP_CUT * 100)
flat_cut = np.percentile(flat, (1 - DROP_CUT) * 100)

kept, dropped = [], []
for r in rows:
    drop = (r["global_std"] <= gs_cut
            or r["mean_patch_std"] <= mps_cut
            or r["edge_density"] <= ed_cut
            or r["flat_patch_frac"] >= flat_cut)
    (dropped if drop else kept).append(r)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    for r in kept:
        f.write(os.path.basename(r["path"]) + "\n")

print(f"thresholds: gs<={gs_cut:.4f} mps<={mps_cut:.4f} ed<={ed_cut:.4f} flat>={flat_cut:.4f}")
print(f"kept   {len(kept):>5}  ({100*len(kept)/len(rows):.1f}%)")
print(f"dropped {len(dropped):>4}  ({100*len(dropped)/len(rows):.1f}%)")
print(f"wrote {OUT_PATH}")
