import numpy as np
import torch
import zipfile
import io

def float_to_q_format(val, int_bits, frac_bits, total_bits=16):
    """
    Converts a float to a two's complement fixed-point hex string (Q format).
    
    Args:
        val (float): Floating point value to convert.
        int_bits (int): Number of integer bits (including sign bit).
        frac_bits (int): Number of fractional bits.
        total_bits (int): Total word width in bits (default: 16).
    """
    # 1. Scale floating point value by 2^(frac_bits)
    scaling_factor = 1 << frac_bits
    scaled_val = int(np.round(val * scaling_factor))
    
    # 2. Determine signed saturation limits for 2's complement
    min_val = -(1 << (total_bits - 1))
    max_val = (1 << (total_bits - 1)) - 1
    clamped_val = max(min_val, min(max_val, scaled_val))
    
    # 3. Convert negative numbers into 2's complement bit patterns
    if clamped_val < 0:
        clamped_val = (1 << total_bits) + clamped_val
        
    # 4. Format hex padded to total_bits / 4 digits
    hex_len = total_bits // 4
    return f"{clamped_val:0{hex_len}x}"

def main():
    zip_path = "Saved_runs/best_model.zip"

    with zipfile.ZipFile(zip_path, 'r') as archive:
        with archive.open('policy.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=True)

    # Extract weight matrices & biases
    w1 = checkpoint['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = checkpoint['mlp_extractor.policy_net.0.bias'].detach().numpy()

    w2 = checkpoint['mlp_extractor.policy_net.2.weight'].detach().numpy()
    b2 = checkpoint['mlp_extractor.policy_net.2.bias'].detach().numpy()

    w3 = checkpoint['action_net.weight'].detach().numpy()
    b3 = checkpoint['action_net.bias'].detach().numpy()

    # =========================================================================
    # CONFIGURATION: Set Fixed-Point (Q-Format) parameters per layer
    # Note: (int_bits + frac_bits) should equal total_bits (16 bits here)
    # Examples:
    #   Q8.8  -> int_bits=8,  frac_bits=8
    #   Q4.12 -> int_bits=4,  frac_bits=12
    #   Q2.14 -> int_bits=2,  frac_bits=14
    # =========================================================================
    LAYER_Q_CONFIGS = {
        'L1': {'int_bits': 8, 'frac_bits': 8},   # Layer 1 Q-format
        'L2': {'int_bits': 8, 'frac_bits': 8},   # Layer 2 Q-format
        'L3': {'int_bits': 8, 'frac_bits': 8},   # Layer 3 Q-format
    }

    # Dynamic Topology Discovery
    input_dim = w1.shape[1]      # Layer 1 inputs (e.g., 5)
    l1_units = w1.shape[0]       # Layer 1 hidden units (e.g., 64 or 256)
    l2_units = w2.shape[0]       # Layer 2 hidden units (e.g., 64 or 256)
    action_dim = w3.shape[0]     # Layer 3 output actions (e.g., 1)

    print("==================================================")
    print("           DETECTED NETWORK ARCHITECTURE          ")
    print("==================================================")
    print(f"  Layer 1 : {input_dim} -> {l1_units} | Q{LAYER_Q_CONFIGS['L1']['int_bits']}.{LAYER_Q_CONFIGS['L1']['frac_bits']}")
    print(f"  Layer 2 : {l2_units} -> {l2_units} | Q{LAYER_Q_CONFIGS['L2']['int_bits']}.{LAYER_Q_CONFIGS['L2']['frac_bits']}")
    print(f"  Layer 3 : {l2_units} -> {action_dim} | Q{LAYER_Q_CONFIGS['L3']['int_bits']}.{LAYER_Q_CONFIGS['L3']['frac_bits']}")
    print("==================================================\n")

    PE_BLOCK_SIZE = 4

    l1_row_tiles = int(np.ceil(l1_units / PE_BLOCK_SIZE))
    l2_row_tiles = int(np.ceil(l2_units / PE_BLOCK_SIZE))
    l3_row_tiles = int(np.ceil(action_dim / PE_BLOCK_SIZE))

    w3_padded = np.zeros((l3_row_tiles * PE_BLOCK_SIZE, l2_units))
    w3_padded[:action_dim, :] = w3

    b3_padded = np.zeros((l3_row_tiles * PE_BLOCK_SIZE,))
    b3_padded[:action_dim] = b3

    unified_bram_lines = []

    # =========================================================================
    # --- PROCESS LAYER 1 ---
    # =========================================================================
    q_l1 = LAYER_Q_CONFIGS['L1']
    for r_tile in range(l1_row_tiles):
        r3, r2, r1, r0 = r_tile*4 + 3, r_tile*4 + 2, r_tile*4 + 1, r_tile*4 + 0

        # Inject Bias Word
        b3_hex = float_to_q_format(b1[r3] if r3 < l1_units else 0.0, **q_l1)
        b2_hex = float_to_q_format(b1[r2] if r2 < l1_units else 0.0, **q_l1)
        b1_hex = float_to_q_format(b1[r1] if r1 < l1_units else 0.0, **q_l1)
        b0_hex = float_to_q_format(b1[r0] if r0 < l1_units else 0.0, **q_l1)
        unified_bram_lines.append(f"{b3_hex}{b2_hex}{b1_hex}{b0_hex}")

        # Stream columns
        for c in range(input_dim):
            w3_hex = float_to_q_format(w1[r3, c] if r3 < l1_units else 0.0, **q_l1)
            w2_hex = float_to_q_format(w1[r2, c] if r2 < l1_units else 0.0, **q_l1)
            w1_hex = float_to_q_format(w1[r1, c] if r1 < l1_units else 0.0, **q_l1)
            w0_hex = float_to_q_format(w1[r0, c] if r0 < l1_units else 0.0, **q_l1)
            unified_bram_lines.append(f"{w3_hex}{w2_hex}{w1_hex}{w0_hex}")

    l1_size = len(unified_bram_lines)

    # =========================================================================
    # --- PROCESS LAYER 2 ---
    # =========================================================================
    q_l2 = LAYER_Q_CONFIGS['L2']
    for r_tile in range(l2_row_tiles):
        r3, r2, r1, r0 = r_tile*4 + 3, r_tile*4 + 2, r_tile*4 + 1, r_tile*4 + 0

        # Inject Bias Word
        b3_hex = float_to_q_format(b2[r3] if r3 < l2_units else 0.0, **q_l2)
        b2_hex = float_to_q_format(b2[r2] if r2 < l2_units else 0.0, **q_l2)
        b1_hex = float_to_q_format(b2[r1] if r1 < l2_units else 0.0, **q_l2)
        b0_hex = float_to_q_format(b2[r0] if r0 < l2_units else 0.0, **q_l2)
        unified_bram_lines.append(f"{b3_hex}{b2_hex}{b1_hex}{b0_hex}")

        # Stream columns
        for c in range(l1_units):
            w3_hex = float_to_q_format(w2[r3, c] if r3 < l2_units else 0.0, **q_l2)
            w2_hex = float_to_q_format(w2[r2, c] if r2 < l2_units else 0.0, **q_l2)
            w1_hex = float_to_q_format(w2[r1, c] if r1 < l2_units else 0.0, **q_l2)
            w0_hex = float_to_q_format(w2[r0, c] if r0 < l2_units else 0.0, **q_l2)
            unified_bram_lines.append(f"{w3_hex}{w2_hex}{w1_hex}{w0_hex}")

    l2_size = len(unified_bram_lines) - l1_size

    # =========================================================================
    # --- PROCESS LAYER 3 ---
    # =========================================================================
    q_l3 = LAYER_Q_CONFIGS['L3']
    for r_tile in range(l3_row_tiles):
        r3, r2, r1, r0 = r_tile*4 + 3, r_tile*4 + 2, r_tile*4 + 1, r_tile*4 + 0

        # Inject Bias Word
        b3_hex = float_to_q_format(b3_padded[r3], **q_l3)
        b2_hex = float_to_q_format(b3_padded[r2], **q_l3)
        b1_hex = float_to_q_format(b3_padded[r1], **q_l3)
        b0_hex = float_to_q_format(b3_padded[r0], **q_l3)
        unified_bram_lines.append(f"{b3_hex}{b2_hex}{b1_hex}{b0_hex}")

        # Stream columns
        for c in range(l2_units):
            w3_hex = float_to_q_format(w3_padded[r3, c], **q_l3)
            w2_hex = float_to_q_format(w3_padded[r2, c], **q_l3)
            w1_hex = float_to_q_format(w3_padded[r1, c], **q_l3)
            w0_hex = float_to_q_format(w3_padded[r0, c], **q_l3)
            unified_bram_lines.append(f"{w3_hex}{w2_hex}{w1_hex}{w0_hex}")

    l3_size = len(unified_bram_lines) - l1_size - l2_size

    # Save Memory File
    filename = "unified_weights.mem"
    with open(filename, "w") as f:
        for hex_val in unified_bram_lines:
            f.write(f"{hex_val}\n")

    print(f"Successfully generated {filename} with {len(unified_bram_lines)} total entries.")
    print("--------------------------------------------------")
    print(f"Layer 1 Offset: 0 (Words: {l1_size})")
    print(f"Layer 2 Offset: {l1_size} (Words: {l2_size})")
    print(f"Layer 3 Offset: {l1_size + l2_size} (Words: {l3_size})")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()