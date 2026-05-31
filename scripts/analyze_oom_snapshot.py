"""Analyze a torch.cuda memory snapshot pickle: aggregate live (allocated)
blocks by the stack frame that allocated them, so we can see WHAT held the
memory at OOM. Usage: python scripts/analyze_oom_snapshot.py runs/oom_snapshot_*.pickle
"""
import pickle, sys, glob
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("runs/oom_snapshot_*.pickle"))[-1]
print(f"snapshot: {path}")
with open(path, "rb") as f:
    snap = pickle.load(f)

segments = snap.get("segments", [])
tot_reserved = sum(s.get("total_size", 0) for s in segments)
tot_alloc = 0
by_frame = defaultdict(lambda: [0, 0])  # frame -> [bytes, count]
big_blocks = []

for seg in segments:
    addr = seg.get("address")
    for blk in seg.get("blocks", []):
        if blk.get("state") != "active_allocated":
            continue
        sz = blk.get("size", 0)
        tot_alloc += sz
        frames = blk.get("frames", []) or []
        # pick the first non-torch-internal frame for attribution
        label = "?"
        for fr in frames:
            fn = fr.get("filename", "")
            name = fr.get("name", "")
            if "UrbanJEPA" in fn or "/src/" in fn:
                label = f"{fn.split('/')[-1]}:{fr.get('line')}:{name}"
                break
        else:
            if frames:
                fr = frames[0]
                label = f"{fr.get('filename','?').split('/')[-1]}:{fr.get('line')}:{fr.get('name')}"
        by_frame[label][0] += sz
        by_frame[label][1] += 1
        if sz > 50 * 1024 * 1024:
            big_blocks.append((sz, label, [f"{f.get('filename','').split('/')[-1]}:{f.get('line')}:{f.get('name')}" for f in frames[:6]]))

print(f"reserved total: {tot_reserved/1e9:.2f} GB | active_allocated: {tot_alloc/1e9:.2f} GB")
print("\n=== top allocators (by first UrbanJEPA/src frame) ===")
for label, (b, c) in sorted(by_frame.items(), key=lambda kv: -kv[1][0])[:20]:
    print(f"  {b/1e9:7.3f} GB  x{c:<5} {label}")

print("\n=== largest individual blocks (>50MB), with stack ===")
for sz, label, stack in sorted(big_blocks, reverse=True)[:15]:
    print(f"  {sz/1e6:8.1f} MB  {label}")
    for s in stack:
        print(f"        {s}")
