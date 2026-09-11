import torch
import numpy as np
import zipfile
import io

# Ensure deterministic execution (reproducible random weights for testing)
torch.manual_seed(42)

# =========================================================================
# 1. Fixed-Point Quantization Functions (PyTorch Vectorized)
# =========================================================================
def float_to_q8_8(tensor_or_val):
    """Converts floats into signed 16-bit Q8.8 integers inside a PyTorch tensor."""
    if not isinstance(tensor_or_val, torch.Tensor):
        tensor_or_val = torch.tensor(tensor_or_val, dtype=torch.float32)
    scaled = torch.round(tensor_or_val * 256.0).to(torch.int32)
    return torch.clamp(scaled, -32768, 32767)

def q8_8_to_float(tensor):
    """Converts a Q8.8 PyTorch tensor back to floating point for debugging."""
    sign_extended = torch.where(tensor >= 32768, tensor - 65536, tensor)
    return sign_extended.float() / 256.0

def saturate_and_shift_tensor(accumulated_32bit):
    """
    Emulates SystemVerilog's saturate_q16_16_to_q8_8.
    Performs floor division by 256 (>> 8 for signed Q16.16) and clamps to signed 16-bit.
    """
    shifted = torch.bitwise_right_shift(accumulated_32bit, 8)
    return torch.clamp(shifted, -32768, 32767)

def q8_8_relu(tensor):
    """Fixed-point ReLU: zeros out negative values, leaves positive Q8.8 values unchanged."""
    return torch.clamp_min(tensor, 0)

# =========================================================================
# 2. Define Inputs & Network Tensors (6-128-128-1)
# =========================================================================
# 6 input activations scaled to Q8.8 format
input_features = float_to_q8_8([-0.00390625,-0.0078125, 0.49609375 , 0.0, 0.0, 1.046875])

try:
    zip_path = "runs/ppo_2026-09-08_1707/best_model.zip"

    with zipfile.ZipFile(zip_path, 'r') as archive:
        with archive.open('policy.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=True)

    w1_raw = checkpoint['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1_raw = checkpoint['mlp_extractor.policy_net.0.bias'].detach().numpy()

    w2_raw = checkpoint['mlp_extractor.policy_net.2.weight'].detach().numpy()
    b2_raw = checkpoint['mlp_extractor.policy_net.2.bias'].detach().numpy()

    w3_raw = checkpoint['action_net.weight'].detach().numpy()
    b3_raw = checkpoint['action_net.bias'].detach().numpy()
except (FileNotFoundError, KeyError):
    print("[WARNING] Checkpoint not found or incompatible. Generating synthetic 6-128-128-1 weights.")
    w1_raw = torch.randn(128, 6) * 0.1
    b1_raw = torch.zeros(128)
    w2_raw = torch.randn(128, 128) * 0.1
    b2_raw = torch.zeros(128)
    w3_raw = torch.randn(1, 128) * 0.1
    b3_raw = torch.zeros(1)

# Quantize weights into Q8.8 integers
w1 = float_to_q8_8(w1_raw)  # (128, 6)
w2 = float_to_q8_8(w2_raw)  # (128, 128)
w3 = float_to_q8_8(w3_raw)  # (1, 128)

# Quantize biases into Q8.8 integers
b1 = float_to_q8_8(b1_raw)  # (128,)
b2 = float_to_q8_8(b2_raw)  # (128,)
b3 = float_to_q8_8(b3_raw)  # (1,)

# Layer 3 splits for Actor/Critic configurations
w3_actor  = w3
w3_critic = w3
b3_actor  = b3
b3_critic = b3

# =========================================================================
# 3. PyTorch Inference Engine Execution Loop
# =========================================================================
def run_pytorch_mlp_inference(critic_mode=False):
    # --- LAYER 1: (128, 6) @ (6,) -> (128,) ---
    l1_dot = torch.mv(w1, input_features)
    l1_acc = l1_dot + (b1 * 256)
    layer1_saturated = saturate_and_shift_tensor(l1_acc)
    layer1_out = q8_8_relu(layer1_saturated)

    print("\n--- Layer 1 Output Sample (First 5 Neurons) ---")
    for i in range(min(5, len(layer1_out))):
        val = int(layer1_out[i].item())
        print(f"Layer 1 Index {i} in Hex: {val & 0xFFFF:04x}")

    # --- LAYER 2: (128, 128) @ (128,) -> (128,) ---
    l2_dot = torch.mv(w2, layer1_out)
    l2_acc = l2_dot + (b2 * 256)
    layer2_saturated = saturate_and_shift_tensor(l2_acc)
    layer2_out = q8_8_relu(layer2_saturated)

    print("\n--- Layer 2 Output Sample (First 5 Neurons) ---")
    for i in range(min(5, len(layer2_out))):
        val = int(layer2_out[i].item())
        print(f"Layer 2 Index {i} in Hex: {val & 0xFFFF:04x}")

    # --- LAYER 3: (1, 128) @ (128,) -> (1,) (Linear / No ReLU) ---
    w3_selected = w3_critic if critic_mode else w3_actor
    b3_selected = b3_critic if critic_mode else b3_actor

    l3_dot = torch.mv(w3_selected, layer2_out)
    l3_acc = l3_dot + (b3_selected * 256)

    l3_out = saturate_and_shift_tensor(l3_acc)
    final_q_val = q8_8_relu(l3_out)
    return final_q_val.item()

# =========================================================================
# 4. Runtime Output Routine
# =========================================================================
if __name__ == "__main__":
    print("--- Running 6-128-128-1 Hardware-Emulated Inference via PyTorch ---")

    # Run Actor Mode Execution
    actor_res_q = run_pytorch_mlp_inference(critic_mode=False)
    print("\n===========================================================")
    print(f"Actor Hex Target  : 16'sh{actor_res_q & 0xFFFF:04x}")
    print(f"Actor Real Value  : {q8_8_to_float(torch.tensor(actor_res_q)):.4f}")
    print("===========================================================")

    # Run Critic Mode Execution
    critic_res_q = run_pytorch_mlp_inference(critic_mode=True)
    print("\n===========================================================")
    print(f"Critic Hex Target : 16'sh{critic_res_q & 0xFFFF:04x}")
    print(f"Critic Real Value : {q8_8_to_float(torch.tensor(critic_res_q)):.4f}")
    print("===========================================================")