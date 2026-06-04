"""
Phase 1 smoke test — JEPAConditionedRRDBNet integration.

Three checks:

  1. Forward shape: model produces (B, 3, 256, 256) on (B, 3, 256, 256) input.
  2. **Identity-at-init**: with zero-init injection layers, output MUST equal
     stock RRDBNet on the downsampled LR — bit-equivalent up to fp32 noise.
  3. Perturb test: after writing non-zero weights into the injection fuses,
     output must differ noticeably (proves injections are wired into compute).

We use synthetic JEPA features (random Gaussian, the right shape) to avoid
loading DINOv2 — this test is about the integration plumbing, not features.
"""

import argparse

import torch
import torch.nn.functional as F

from src.models.esrgan import build_pretrained_x4plus
from src.models.jepa_esrgan import JEPAConditionedRRDBNet


def fmt(x):
    return f"shape={tuple(x.shape)} range=[{x.min():.4f},{x.max():.4f}] " \
           f"mean={x.mean():.4f} std={x.std():.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--jepa_dim", type=int, default=768)
    p.add_argument("--token_grid_side", type=int, default=16)
    p.add_argument("--n_multi", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ----- Build the stock RRDBNet (reference) and the JEPA-conditioned wrapper -----
    stock = build_pretrained_x4plus(args.weights, device=device, eval_mode=True)
    rrdbnet_for_wrap = build_pretrained_x4plus(args.weights, device=device, eval_mode=True)
    model = JEPAConditionedRRDBNet(
        rrdbnet=rrdbnet_for_wrap,
        jepa_dim=args.jepa_dim,
        token_grid_side=args.token_grid_side,
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_inj = (sum(p.numel() for p in model.inject_bottleneck.parameters())
             + sum(p.numel() for p in model.inject_up1.parameters())
             + sum(p.numel() for p in model.inject_up2.parameters()))
    print(f"[model] total {n_params/1e6:.2f}M params  "
          f"|  JEPA injection {n_inj/1e3:.2f}K params  "
          f"|  RRDBNet {(n_params-n_inj)/1e6:.2f}M params")

    # ----- Synthetic inputs -----
    B = args.batch_size
    low_res = torch.rand(B, 3, 256, 256, device=device)
    N = args.token_grid_side * args.token_grid_side
    ctx_final = torch.randn(B, N, args.jepa_dim, device=device)
    ctx_multi = [torch.randn(B, N, args.jepa_dim, device=device) for _ in range(args.n_multi)]
    print(f"[input] low_res {fmt(low_res)}")
    print(f"[input] ctx_final {fmt(ctx_final)}")
    print(f"[input] ctx_multi[{len(ctx_multi)}] each {fmt(ctx_multi[0])}")

    # ===== Test 1 — forward shape =====
    with torch.no_grad():
        out = model(low_res, ctx_final, ctx_multi)
    print(f"[t1]    forward out {fmt(out)}")
    assert out.shape == (B, 3, 256, 256), f"bad shape {out.shape}"
    assert torch.isfinite(out).all(), "out has NaN/Inf"
    print("[t1]    PASS — shape (B,3,256,256), finite")

    # ===== Test 2 — identity-at-init =====
    # Stock RRDBNet on downsampled LR is the reference.
    lr_small = F.avg_pool2d(low_res, kernel_size=4, stride=4)
    with torch.no_grad():
        ref = stock(lr_small)
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    rel_err = (diff.sum() / ref.abs().sum().clamp(min=1e-12)).item()
    print(f"[t2]    max|out - stock(lr_small)| = {max_abs:.3e}  "
          f"|  rel L1 err = {rel_err:.3e}")
    # All convs in the injection paths are zero — residual is exactly 0.
    # The wrapped RRDBNet is a *separate* instance from `stock` but loaded
    # from the exact same weights, so their forward paths must match to fp32
    # numerical noise.
    TOL = 1e-4  # bf16-permissive; fp32 is typically <1e-5
    assert max_abs < TOL, f"identity-at-init violated: max|diff|={max_abs}"
    print(f"[t2]    PASS — identity-at-init holds (max|diff|<{TOL})")

    # ===== Test 3 — perturb test (injections actually do something) =====
    # Write non-zero values into the fuse layers and re-run.
    with torch.no_grad():
        for inj in (model.inject_bottleneck, model.inject_up1, model.inject_up2):
            torch.nn.init.normal_(inj.fuse.weight, std=0.05)
            torch.nn.init.normal_(inj.fuse.bias, std=0.05)
        out_perturbed = model(low_res, ctx_final, ctx_multi)
    delta = (out_perturbed - ref).abs().mean().item()
    print(f"[t3]    after perturbing fuse weights, mean|out_perturbed - stock| = {delta:.4e}")
    assert delta > 1e-3, f"injections appear disconnected from output (delta={delta})"
    print("[t3]    PASS — injection layers are connected to output")

    # ----- Quick gradient check: a backward pass through the model works -----
    model.train()
    # Re-zero the injection fuses so we test the gradient flow at init.
    for inj in (model.inject_bottleneck, model.inject_up1, model.inject_up2):
        torch.nn.init.zeros_(inj.fuse.weight)
        torch.nn.init.zeros_(inj.fuse.bias)

    low_res.requires_grad_(False)
    fake_target = torch.rand_like(out)
    out = model(low_res, ctx_final, ctx_multi)
    loss = F.l1_loss(out, fake_target)
    loss.backward()

    # Injection fuse weights should have non-zero gradient (residual path live).
    gb = model.inject_bottleneck.fuse.weight.grad.abs().mean().item()
    g1 = model.inject_up1.fuse.weight.grad.abs().mean().item()
    g2 = model.inject_up2.fuse.weight.grad.abs().mean().item()
    # RRDBNet conv_last should also have non-zero gradient.
    gl = model.rrdbnet.conv_last.weight.grad.abs().mean().item()
    print(f"[grad]  inject_bottleneck.fuse  : {gb:.3e}")
    print(f"[grad]  inject_up1.fuse         : {g1:.3e}")
    print(f"[grad]  inject_up2.fuse         : {g2:.3e}")
    print(f"[grad]  rrdbnet.conv_last       : {gl:.3e}")
    for name, g in [("inject_bottleneck", gb), ("inject_up1", g1),
                    ("inject_up2", g2), ("rrdbnet.conv_last", gl)]:
        assert g > 0, f"{name} has zero gradient — broken backward path"
    print("[grad]  PASS — all injection paths have live gradients")

    print("[ok]    Phase 1 smoke test passed.")


if __name__ == "__main__":
    main()
