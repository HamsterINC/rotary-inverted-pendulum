"""
Fixed-point (Qm.n) dynamic-range analysis for an SB3 ActorCriticPolicy (.pth) checkpoint.

This script has two clearly separate parts:

  PART A - WEIGHTS
      Read directly from the .pth checkpoint. These are static numbers -
      exact, no simulation needed, independent of any input.

  PART B - ACTIVATIONS / ACCUMULATORS
      Do NOT exist in the checkpoint file. They are the intermediate values
      produced when data flows through the network at inference time, so
      they depend entirely on what observations you feed in. To get them
      we simulate a forward pass (numpy, matching the real Linear->Tanh
      architecture) over a batch of sample observations and record the
      min/max/percentiles seen at every stage.

Everything here is standard numpy - no torch/stable-baselines3 required,
since this environment doesn't have them installed. A tiny custom
unpickler (torch_loader.py) reads the torch.save zip/pickle format
directly to pull out raw tensors.

Qm.n CONVENTION USED BELOW: m includes the sign bit.
  e.g. Q8.8  = 16 bits total, magnitude range +-128,  resolution 1/256
       Q10.6 = 16 bits total, magnitude range +-512,  resolution 1/64
"""

import numpy as np
from torch_loader import load_pth_from_sb3_zip
from pathlib import Path

np.random.seed(0)

# Point this at your SB3 checkpoint .zip (the file you get from model.save()).
# No manual unzipping needed -- SB3's zip contains policy.pth internally,
# which is itself a zip (that's just how torch.save works); both layers are
# unwrapped in memory by load_pth_from_sb3_zip.
SB3_ZIP_PATH = 'Saved-runs/converted_model.zip'

# Observation space bounds pulled from the checkpoint's saved data.pkl
# (gymnasium Box: 6-dim, e.g. angle, two normalized rates, two large
# position/velocity terms, and another rate)
OBS_LOW  = np.array([-2.3561945, -1., -1., -200., -200., -1.])
OBS_HIGH = np.array([ 2.3561945,  1.,  1.,  200.,  200.,  1.])

N_SAMPLES = 30000

# Candidate fixed-point formats to sweep, as (magnitude_int_bits, frac_bits, label)
# total bits = magnitude_int_bits + frac_bits + 1 (sign) = 16 in all cases here
CANDIDATES = [
    (10, 5, "Q11.5"),
    (9,  6, "Q10.6"),
    (8,  7, "Q9.7"),
    (7,  8, "Q8.8"),
    (6,  9, "Q7.9"),
    (5, 10, "Q6.10"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def int_bits_needed(absmax, headroom=0):
    """Minimum magnitude integer bits (excludes sign) so 2^m > absmax."""
    if absmax <= 0:
        return 0
    return max(int(np.ceil(np.log2(absmax))), 0) + headroom


def simulate_fixed(x, magnitude_int_bits, frac_bits):
    """Quantize x to a Qm.n fixed-point grid and report the result + clip rate."""
    scale = 2.0 ** frac_bits
    max_val = 2 ** (magnitude_int_bits + frac_bits) - 1
    min_val = -max_val - 1
    scaled = x * scale
    clip_frac = np.mean((scaled > max_val) | (scaled < min_val)) * 100
    q = np.clip(np.round(scaled), min_val, max_val)
    return q / scale, clip_frac


def sqnr_db(x, xq):
    """Signal-to-quantization-noise ratio in dB."""
    signal_power = np.mean(x ** 2)
    error_power = np.mean((x - xq) ** 2)
    return np.inf if error_power == 0 else 10 * np.log10(signal_power / error_power)


def linear(x, w, b):
    return x @ w.T + b


def tanh(x):
    return np.tanh(x)


# ---------------------------------------------------------------------------
# Load weights (exact - read straight from the checkpoint)
# ---------------------------------------------------------------------------
model_path = Path(__file__).parent / "Saved_runs" / "converted_model.zip"
sd = load_pth_from_sb3_zip(model_path)



def g(name):
    return sd[name].astype(np.float64)


WEIGHTS = {
    'policy_net.0':  (g('mlp_extractor.policy_net.0.weight'), g('mlp_extractor.policy_net.0.bias')),
    'policy_net.2':  (g('mlp_extractor.policy_net.2.weight'), g('mlp_extractor.policy_net.2.bias')),
    'action_net':    (g('action_net.weight'),                 g('action_net.bias')),
    'value_net.0':   (g('mlp_extractor.value_net.0.weight'),  g('mlp_extractor.value_net.0.bias')),
    'value_net.2':   (g('mlp_extractor.value_net.2.weight'),  g('mlp_extractor.value_net.2.bias')),
    'value_net_out': (g('value_net.weight'),                  g('value_net.bias')),
}


# ---------------------------------------------------------------------------
# PART A: WEIGHT dynamic range (exact, from the checkpoint)
# ---------------------------------------------------------------------------

def analyze_weights():
    print("#" * 100)
    print("PART A: WEIGHTS  (exact values read from policy.pth - no simulation)")
    print("#" * 100)
    print(f"{'Layer':16s}{'|W|max':>10s}{'|bias|max':>11s}{'MinIntBits':>12s}")
    print("-" * 100)
    results = {}
    for name, (w, b) in WEIGHTS.items():
        absmax = max(np.abs(w).max(), np.abs(b).max())
        ib = int_bits_needed(absmax)
        results[name] = absmax
        print(f"{name:16s}{np.abs(w).max():10.4f}{np.abs(b).max():11.4f}{ib:12d}")

    print()
    print(f"{'Layer':16s}" + "".join(f"{c[2]:>9s}" for c in CANDIDATES) + "   <- SQNR (dB) per format")
    for name, (w, b) in WEIGHTS.items():
        allvals = np.concatenate([w.flatten(), b.flatten()])
        row = f"{name:16s}"
        for ib, fb, label in CANDIDATES:
            xq, clip = simulate_fixed(allvals, ib, fb)
            s = sqnr_db(allvals, xq)
            tag = f"{s:.1f}" + ("*" if clip > 0.01 else "")
            row += f"{tag:>9s}"
        print(row)
    print("(* => >0.01% of values clipped at that format)\n")
    return results


# ---------------------------------------------------------------------------
# PART B: ACTIVATION / ACCUMULATOR dynamic range
#         (simulated - these values do NOT exist in the .pth file; they only
#          appear when data is actually forward-passed through the network)
# ---------------------------------------------------------------------------

def simulate_activations():
    print("#" * 100)
    print("PART B: ACTIVATIONS / ACCUMULATORS  (simulated forward pass - depends on input data!)")
    print("#" * 100)

    # Sample observations across the declared Box bounds: uniform coverage
    # plus extra samples biased toward the corners/edges, since worst-case
    # dynamic range is driven by rare extreme inputs, not typical ones.
    obs_uniform = np.random.uniform(OBS_LOW, OBS_HIGH, size=(N_SAMPLES, 6))
    obs_edges = OBS_LOW + (OBS_HIGH - OBS_LOW) * np.random.choice(
        [0, 0.01, 0.99, 1.0], size=(N_SAMPLES, 6)
    )
    obs_all = np.vstack([obs_uniform, obs_edges])

    # NOTE: this is SYNTHETIC calibration data (sampled from the observation
    # space's declared bounds), not real rollout data. Real trajectories may
    # never visit the full range, so treat this as a safe worst-case rather
    # than the typical operating range. Swap `obs_all` for real logged
    # observations if you have them, for a tighter/more realistic estimate.

    records = []  # (layer_name, stage_description, flattened_values)

    def record(name, stage, vals):
        records.append((name, stage, vals.flatten()))

    record('input.observation', 'raw observation input', obs_all)

    # --- policy (actor) branch ---
    z1 = linear(obs_all, *WEIGHTS['policy_net.0'])
    record('policy_net.0', 'accumulator (pre-tanh)', z1)
    a1 = tanh(z1)
    record('policy_net.0', 'output (post-tanh)', a1)

    z2 = linear(a1, *WEIGHTS['policy_net.2'])
    record('policy_net.2', 'accumulator (pre-tanh)', z2)
    a2 = tanh(z2)
    record('policy_net.2', 'output (post-tanh)', a2)

    z3 = linear(a2, *WEIGHTS['action_net'])
    record('action_net', 'output (action mean)', z3)

    # --- value (critic) branch ---
    v1 = linear(obs_all, *WEIGHTS['value_net.0'])
    record('value_net.0', 'accumulator (pre-tanh)', v1)
    b1 = tanh(v1)
    record('value_net.0', 'output (post-tanh)', b1)

    v2 = linear(b1, *WEIGHTS['value_net.2'])
    record('value_net.2', 'accumulator (pre-tanh)', v2)
    b2 = tanh(v2)
    record('value_net.2', 'output (post-tanh)', b2)

    v3 = linear(b2, *WEIGHTS['value_net_out'])
    record('value_net_out', 'output (state value)', v3)

    print(f"{'Layer':16s}{'Stage':26s}{'AbsMax':>9s}{'P99.9':>9s}{'MinIntBits':>12s}")
    print("-" * 100)
    for name, stage, vals in records:
        absmax = np.max(np.abs(vals))
        p999 = np.quantile(np.abs(vals), 0.999)
        ib = int_bits_needed(absmax)
        print(f"{name:16s}{stage:26s}{absmax:9.3f}{p999:9.3f}{ib:12d}")

    print()
    print(f"{'Layer':16s}{'Stage':26s}" + "".join(f"{c[2]:>9s}" for c in CANDIDATES) + "  <- SQNR (dB)")
    for name, stage, vals in records:
        row = f"{name:16s}{stage:26s}"
        for ib, fb, label in CANDIDATES:
            xq, clip = simulate_fixed(vals, ib, fb)
            s = sqnr_db(vals, xq)
            tag = f"{s:.1f}" + ("*" if clip > 0.01 else "")
            row += f"{tag:>9s}"
        print(row)
    print("(* => >0.01% of values clipped at that format -- avoid for that layer)\n")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    weight_results = analyze_weights()
    activation_records = simulate_activations()