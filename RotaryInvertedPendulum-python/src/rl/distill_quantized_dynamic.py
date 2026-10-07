"""QAT (quantisation-aware training) variant of distill.py for int16 DFP.

Produces an int16-deployable student MLP for FPGA/Microcontrollers.
- Activations and weights are quantised to 16-bit integers.
- Granular control over the fractional width (Q-format) of each layer.

Usage:
    python distill_quantised.py \
        --float-student runs/<run>/student.pt \
        --dataset       runs/<run>/dataset.npz \
        --out-dir       runs/<run>/quantised \
        --in-fw 11 --h1-fw 10 --h2-fw 10 --out-fw 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distill import StudentMLP


# ---------------------------------------------------------------------------
# FakeQuant — straight-through int16/int32 simulation
# ---------------------------------------------------------------------------

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor,
                max_int: float = 32767.0) -> torch.Tensor:
        scale = scale.clamp(min=1e-12)
        q = torch.round(x / scale).clamp(-max_int, max_int)
        return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None

def fake_quant_int16(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(x, scale, 32767.0)

def fake_quant_int32(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(x, scale, float(2**31 - 1))


# ---------------------------------------------------------------------------
# DFP Activation Observers — EMA of absolute-max locking to Power of 2
# ---------------------------------------------------------------------------

def compute_dfp_scale(max_abs: torch.Tensor, max_int: float = 32767.0):
    max_abs = max_abs.clamp(min=1e-8)
    frac_width = torch.floor(torch.log2(max_int / max_abs))
    scale = 2.0 ** (-frac_width)
    return scale, frac_width


class DFPTensorObserver(nn.Module):
    """Tracks running |max| for a whole tensor and provides a single DFP scale.
    If fixed_frac_width is provided, it locks to that Q-format.
    """
    def __init__(self, fixed_frac_width: int | None = None, ema_decay: float = 0.99):
        super().__init__()
        self.fixed_frac_width = fixed_frac_width
        self.ema_decay = ema_decay
        self.register_buffer("max_abs", torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_frac_width is None and self.training:
            with torch.no_grad():
                cur = x.detach().abs().max()
                if self.max_abs.item() <= 1e-3:
                    self.max_abs.copy_(cur)
                else:
                    self.max_abs.mul_(self.ema_decay).add_(cur, alpha=1.0 - self.ema_decay)
        return x

    @property
    def scale_and_frac(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fixed_frac_width is not None:
            fw = torch.tensor(float(self.fixed_frac_width), device=self.max_abs.device)
            scale = 2.0 ** (-fw)
            return scale, fw
        return compute_dfp_scale(self.max_abs, 32767.0)


# ---------------------------------------------------------------------------
# QAT student — int16 DFP Network
# ---------------------------------------------------------------------------

class QATStudent(nn.Module):
    def __init__(self, hidden: int = 128, obs_dim: int = 5, act_dim: int = 1,
                 in_fw: int | None = None,
                 h1_fw: int | None = None,
                 h2_fw: int | None = None,
                 out_fw: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)
        
        # Granular observers for each stage of the network
        self.obs_in = DFPTensorObserver(in_fw)
        self.obs_h1 = DFPTensorObserver(h1_fw)
        self.obs_h2 = DFPTensorObserver(h2_fw)
        self.obs_out = DFPTensorObserver(out_fw)  # Pre-tanh output quantizer
        
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @staticmethod
    def _q_weight_and_bias(weight: torch.Tensor, bias: torch.Tensor, x_scale: torch.Tensor):
        w_max = weight.abs().amax(dim=1, keepdim=True)
        w_scale, w_frac = compute_dfp_scale(w_max, 32767.0)
        w_q = fake_quant_int16(weight, w_scale)
        
        b_scale = (w_scale.squeeze(-1) * x_scale).clamp(min=1e-12)
        b_q = fake_quant_int32(bias, b_scale)
        
        return w_q, b_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 0 (Input)
        self.obs_in(x)
        s_obs, f_obs = self.obs_in.scale_and_frac
        x = fake_quant_int16(x, s_obs)

        # Layer 1
        w1_q, b1_q = self._q_weight_and_bias(self.fc1.weight, self.fc1.bias, s_obs)
        x = F.linear(x, w1_q, b1_q)
        x = F.relu(x)
        
        self.obs_h1(x)
        s_h1, f_h1 = self.obs_h1.scale_and_frac
        x = fake_quant_int16(x, s_h1)

        # Layer 2
        w2_q, b2_q = self._q_weight_and_bias(self.fc2.weight, self.fc2.bias, s_h1)
        x = F.linear(x, w2_q, b2_q)
        x = F.relu(x)
        
        self.obs_h2(x)
        s_h2, f_h2 = self.obs_h2.scale_and_frac
        x = fake_quant_int16(x, s_h2)

        # Layer 3 (Output pre-tanh)
        w3_q, b3_q = self._q_weight_and_bias(self.fc3.weight, self.fc3.bias, s_h2)
        x = F.linear(x, w3_q, b3_q)
        
        self.obs_out(x)
        s_out, f_out = self.obs_out.scale_and_frac
        x = fake_quant_int16(x, s_out)
        
        return x


# ---------------------------------------------------------------------------
# Warmstart helpers
# ---------------------------------------------------------------------------

def warmstart_from_float(qat: QATStudent, float_student: StudentMLP) -> None:
    qat.fc1.load_state_dict(float_student.fc1.state_dict())
    qat.fc2.load_state_dict(float_student.fc2.state_dict())
    qat.fc3.load_state_dict(float_student.fc3.state_dict())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_qat(
    qat: QATStudent, obs: np.ndarray, target: np.ndarray, *,
    epochs: int = 200, batch_size: int = 1024, lr: float = 3e-4,
    val_frac: float = 0.1, seed: int = 0, device: str = "cpu",
) -> dict:
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    idx = rng.permutation(n)
    n_val = int(val_frac * n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
    tgt_t = torch.from_numpy(target.astype(np.float32)).to(device)
    train_obs, train_tgt = obs_t[train_idx], tgt_t[train_idx]
    val_obs, val_tgt = obs_t[val_idx], tgt_t[val_idx]

    qat = qat.to(device)
    opt = torch.optim.Adam(qat.parameters(), lr=lr)
    n_train = train_obs.shape[0]

    last_val = float("nan")
    for epoch in range(1, epochs + 1):
        qat.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        seen = 0
        for i in range(0, n_train, batch_size):
            sel = perm[i : i + batch_size]
            x = train_obs[sel]
            y = train_tgt[sel]
            pred = qat(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * x.shape[0]
            seen += x.shape[0]
        train_loss = train_loss_sum / max(1, seen)

        qat.eval()
        with torch.no_grad():
            val_pred = qat(val_obs)
            val_loss = F.mse_loss(val_pred, val_tgt).item()
        last_val = val_loss
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[qat] epoch {epoch:3d}/{epochs}  "
                  f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")
    return {"val_mse": last_val}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_qat_checkpoint(qat: QATStudent, out_path: Path, val_mse: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    _, f_obs = qat.obs_in.scale_and_frac
    _, f_h1 = qat.obs_h1.scale_and_frac
    _, f_h2 = qat.obs_h2.scale_and_frac
    _, f_out = qat.obs_out.scale_and_frac

    payload = {
        "state_dict": qat.state_dict(),
        "hidden": qat.hidden,
        "obs_dim": qat.obs_dim,
        "act_dim": qat.act_dim,
        "val_mse": val_mse,
        "frac_widths": {
            "obs_in": int(f_obs.item()),
            "h1": int(f_h1.item()),
            "h2": int(f_h2.item()),
            "out": int(f_out.item()),
        },
    }
    torch.save(payload, out_path)
    print(f"[qat] saved -> {out_path}, val_mse={val_mse:.6f}")
    print(f"[qat] fractional widths (Q format bits):")
    print(f"[qat]   in={payload['frac_widths']['obs_in']} "
          f"h1={payload['frac_widths']['h1']} "
          f"h2={payload['frac_widths']['h2']} "
          f"out={payload['frac_widths']['out']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QAT distillation for FPGA with granular Q-formats")
    p.add_argument("--float-student", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    
    # Granular fractional width arguments
    p.add_argument("--in-fw", type=int, default=9, help="Input fractional width")
    p.add_argument("--h1-fw", type=int, default=9, help="Layer 1 fractional width")
    p.add_argument("--h2-fw", type=int, default=9, help="Layer 2 fractional width")
    p.add_argument("--out-fw", type=int, default=9, help="Output (pre-tanh) fractional width")
    
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = torch.load(str(args.float_student), map_location=args.device, weights_only=True)
    hidden = int(ckpt.get("hidden", 128))
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    float_student = StudentMLP(hidden=hidden, obs_dim=obs_dim, act_dim=act_dim)
    float_student.load_state_dict(ckpt["state_dict"])

    qat = QATStudent(
        hidden=hidden, obs_dim=obs_dim, act_dim=act_dim, 
        in_fw=args.in_fw, h1_fw=args.h1_fw, h2_fw=args.h2_fw, out_fw=args.out_fw
    )
    warmstart_from_float(qat, float_student)

    data = np.load(args.dataset)
    obs = np.asarray(data["obs"], dtype=np.float32)
    target = np.asarray(data["action_target"], dtype=np.float32)
    
    metrics = train_qat(
        qat, obs, target,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_qat_checkpoint(qat, args.out_dir / "student_quantised.pt", metrics["val_mse"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())