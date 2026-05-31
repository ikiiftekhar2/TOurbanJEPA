"""
System acoustic / thermal stress test.

- GPU: continuous large fp32 matmul on as much VRAM as fits, with an adaptive
  controller that grows the matmul size until temperature hits --target_temp
  (default 85 C) and holds it.
- CPU: a worker per physical core, each running matmul forever in fp32.
- RAM: holds ~half of free system memory in a non-paged tensor so the memory
  bus stays warm too.

Reads nvidia-smi + lm-sensors every --report_every seconds and prints a
single line so you can hear the fans and see the numbers at the same time.

Stop with Ctrl+C. Cleanly releases GPU memory + joins CPU workers.

USAGE:
  python scripts/stress_test.py --duration 120 --target_temp 85
"""

import argparse
import multiprocessing as mp
import signal
import subprocess
import sys
import time

import torch


def _cpu_worker(stop_evt):
    # Each worker hammers a 1024x1024 fp32 matmul tight loop. The PyTorch
    # MKL/OpenMP runtime will saturate the assigned core.
    torch.set_num_threads(1)
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    while not stop_evt.is_set():
        a = a @ b + 1e-9
        b = b @ a + 1e-9
        # Prevent overflow drift — re-randomize occasionally.
        if a.abs().max() > 1e6 or not torch.isfinite(a).all():
            a = torch.randn(1024, 1024)
            b = torch.randn(1024, 1024)


def _gpu_loop(target_temp, report_every, duration, ram_hold_gb):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Hold some pinned host RAM to keep the memory bus busy.
    if ram_hold_gb > 0:
        n = int(ram_hold_gb * (1 << 30) // 4)
        try:
            host_hog = torch.empty(n, dtype=torch.float32, pin_memory=True)
            host_hog.fill_(1.0)
            print(f"[ram] holding {ram_hold_gb:.1f} GB pinned host RAM")
        except RuntimeError as e:
            print(f"[ram] could not pin {ram_hold_gb} GB: {e}")

    # Pre-allocate two big square matrices on GPU; matmul A @ B repeatedly.
    # Start at ~2 GB each and grow until we hit target temp or run out of VRAM.
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free={free/1e9:.1f} GB total={total/1e9:.1f} GB")

    def make_pair(side):
        a = torch.randn((side, side), device=device, dtype=torch.float32)
        b = torch.randn((side, side), device=device, dtype=torch.float32)
        return a, b

    side = 8192
    a, b = make_pair(side)
    out = torch.empty_like(a)
    print(f"[gpu] start side={side} → ~{(side*side*4*3)/1e9:.1f} GB tensors")

    start = time.time()
    last_report = 0.0
    last_grow_attempt = 0.0
    while True:
        # Tight matmul loop — do 20 iterations between sync/report checks so the
        # kernel queue stays saturated.
        for _ in range(20):
            torch.matmul(a, b, out=out)
            a, out = out, a  # rotate so b stays warm too
        torch.cuda.synchronize()

        now = time.time()
        if now - start > duration:
            print(f"[gpu] duration {duration}s reached, exiting")
            break

        if now - last_report >= report_every:
            last_report = now
            gpu_stats = _nvidia_smi_oneline()
            cpu_temp = _cpu_temp_c()
            elapsed = int(now - start)
            print(f"[t={elapsed:>4}s] {gpu_stats} | cpu={cpu_temp:>5} | "
                  f"matmul_side={side}", flush=True)

            # Adaptive grow: if we're more than 4 C below target, grow tensor
            # to push more compute through. Cap to avoid OOM (leave ~2 GB).
            try:
                temp_c = int(gpu_stats.split("temp=")[1].split("C")[0])
            except Exception:
                temp_c = 0
            if temp_c and temp_c < target_temp - 4 and now - last_grow_attempt > 5:
                free_now, _ = torch.cuda.mem_get_info()
                if free_now > 4 * (1 << 30):  # >4 GB headroom
                    new_side = min(int(side * 1.1), 24576)
                    if new_side > side:
                        try:
                            del a, b, out
                            torch.cuda.empty_cache()
                            a, b = make_pair(new_side)
                            out = torch.empty_like(a)
                            side = new_side
                            last_grow_attempt = now
                            print(f"[gpu] grew side={side} "
                                  f"({(side*side*4*3)/1e9:.1f} GB)")
                        except torch.cuda.OutOfMemoryError:
                            print(f"[gpu] OOM growing to {new_side}, staying at {side}")
                            torch.cuda.empty_cache()
                            a, b = make_pair(side)
                            out = torch.empty_like(a)

    print(f"[gpu] final stats: {_nvidia_smi_oneline()}")


def _nvidia_smi_oneline() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,"
             "memory.used,clocks.current.graphics",
             "--format=csv,noheader,nounits"],
            timeout=3,
        ).decode().strip()
        t, fan, pwr, util, mem, clk = [s.strip() for s in out.split(",")]
        return (f"temp={t}C fan={fan}% pwr={pwr}W util={util}% "
                f"mem={mem}MiB clk={clk}MHz")
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def _cpu_temp_c() -> str:
    # Try kernel hwmon first (most reliable), fall back to `sensors`.
    import glob
    import os
    temps = []
    for p in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        try:
            with open(p) as f:
                v = float(f.read().strip()) / 1000.0
            if 20.0 < v < 120.0:
                temps.append(v)
        except (OSError, ValueError):
            pass
    if temps:
        return f"{max(temps):.0f}C(max)"
    try:
        out = subprocess.check_output(
            ["sensors", "-u"], timeout=3, stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            s = line.strip()
            if "_input:" in s:
                try:
                    v = float(s.split(":")[1].strip())
                    if 20.0 < v < 120.0:
                        temps.append(v)
                except (IndexError, ValueError):
                    pass
        return f"{max(temps):.0f}C(max)" if temps else "n/a"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=120,
                    help="Seconds to run before auto-stopping.")
    ap.add_argument("--target_temp", type=int, default=85,
                    help="GPU temp (C) the adaptive loop tries to reach.")
    ap.add_argument("--report_every", type=int, default=5)
    ap.add_argument("--cpu_workers", type=int, default=0,
                    help="0 = use all logical cores.")
    ap.add_argument("--ram_hold_gb", type=float, default=8.0,
                    help="Pin this much host RAM to keep memory bus busy. 0 disables.")
    args = ap.parse_args()

    n_cpu = args.cpu_workers or mp.cpu_count()
    print(f"[init] CPU workers={n_cpu}  target_temp={args.target_temp}C  "
          f"duration={args.duration}s  ram_hold={args.ram_hold_gb}GB")

    stop_evt = mp.Event()
    procs = [mp.Process(target=_cpu_worker, args=(stop_evt,), daemon=True)
             for _ in range(n_cpu)]
    for p in procs:
        p.start()
    print(f"[cpu] {len(procs)} workers spinning")

    def _sigint(signum, frame):
        print("\n[stop] Ctrl+C — shutting down")
        stop_evt.set()
        time.sleep(1)
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    try:
        _gpu_loop(args.target_temp, args.report_every, args.duration,
                  args.ram_hold_gb)
    finally:
        stop_evt.set()
        time.sleep(1)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2)
        torch.cuda.empty_cache()
        print("[done] clean shutdown")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
