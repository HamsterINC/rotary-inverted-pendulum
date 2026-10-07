import numpy as np
import torch
import io

def float_to_q_format(val, int_bits, frac_bits, total_bits=16):
    """
    Converts a float to a two's complement fixed-point hex string (Q format).
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

def compute_q_config(tensor, total_bits=16):
    """Automatically finds the maximum fractional bits to prevent overflow."""
    max_val = (1 << (total_bits - 1)) - 1
    max_abs = np.max(np.abs(tensor))
    max_abs = max(max_abs, 1e-8)
    
    frac_bits = int(np.floor(np.log2(max_val / max_abs)))
    int_bits = total_bits - frac_bits
    
    return {'int_bits': int_bits, 'frac_bits': frac_bits, 'total_bits': total_bits}

def main():
    # Update path to your quantised QAT model
    checkpoint_path = 'runs/async_FPGA/distill_h128_quantized_2/student_quantised.pt'
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=True)
    
    state_dict = checkpoint['state_dict']
    
    # Extract weight matrices & biases
    w1 = state_dict['fc1.weight'].numpy()
    b1 = state_dict['fc1.bias'].numpy()

    w2 = state_dict['fc2.weight'].numpy()
    b2 = state_dict['fc2.bias'].numpy()

    w3 = state_dict['fc3.weight'].numpy()
    b3 = state_dict['fc3.bias'].numpy()

    # Extract Activation Fractional Widths (saved by QAT)
    act_fw = checkpoint.get('frac_widths', {})
    in_fw = act_fw.get('obs_in', 11)   # default to 11 if not found
    h1_fw = act_fw.get('h1', 11)
    h2_fw = act_fw.get('h2', 11)
    out_fw = act_fw.get('out', 7)

    # Automatically compute Weight Q-formats
    q_w1 =  {'int_bits': 16-in_fw, 'frac_bits': in_fw, 'total_bits': 16} # compute_q_config(w1)
    q_w2 = {'int_bits': 16-h1_fw, 'frac_bits': h1_fw, 'total_bits': 16} # compute_q_config(w2)
    q_w3 = {'int_bits': 16-h2_fw, 'frac_bits': h2_fw, 'total_bits': 16} # compute_q_config(w3)

    # Dynamic Topology Discovery
    input_dim = w1.shape[1]      
    l1_units = w1.shape[0]       
    l2_units = w2.shape[0]       
    action_dim = w3.shape[0]     

    print("==================================================")
    print("           DETECTED NETWORK ARCHITECTURE          ")
    print("==================================================")
    print(f" Layer 1 : {input_dim} -> {l1_units}")
    print(f"   - Input Act: Q{16-in_fw}.{in_fw}")
    print(f"   - Weights  : Q{q_w1['int_bits']}.{q_w1['frac_bits']}")
    print(f"   -> MAC Accumulator naturally becomes Q-format with {in_fw + q_w1['frac_bits']} frac bits")
    print(f"   -> Required Shift to match H1 Activation: {in_fw + q_w1['frac_bits'] - h1_fw} bits right\n")

    print(f" Layer 2 : {l1_units} -> {l2_units}")
    print(f"   - H1 Act   : Q{16-h1_fw}.{h1_fw}")
    print(f"   - Weights  : Q{q_w2['int_bits']}.{q_w2['frac_bits']}")
    print(f"   -> Required Shift to match H2 Activation: {h1_fw + q_w2['frac_bits'] - h2_fw} bits right\n")

    print(f" Layer 3 : {l2_units} -> {action_dim}")
    print(f"   - H2 Act   : Q{16-h2_fw}.{h2_fw}")
    print(f"   - Weights  : Q{q_w3['int_bits']}.{q_w3['frac_bits']}")
    print(f"   -> Required Shift to match Output format: {h2_fw + q_w3['frac_bits'] - out_fw} bits right")
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
    for r_tile in range(l1_row_tiles):
        r3, r2, r1, r0 = r_tile*4 + 3, r_tile*4 + 2, r_tile*4 + 1, r_tile*4 + 0

        # Inject Bias Word (Uses weight scale. If you change biases to 32-bit in HW, change total_bits here)
        b3_hex = float_to_q_format(b1[r3] if r3 < l1_units else 0.0, **q_w1)
        b2_hex = float_to_q_format(b1[r2] if r2 < l1_units else 0.0, **q_w1)
        b1_hex = float_to_q_format(b1[r1] if r1 < l1_units else 0.0, **q_w1)
        b0_hex = float_to_q_format(b1[r0] if r0 < l1_units else 0.0, **q_w1)
        unified_bram_lines.append(f"{b3_hex}{b2_hex}{b1_hex}{b0_hex}")

        # Stream columns
        for c in range(input_dim):
            w3_hex = float_to_q_format(w1[r3, c] if r3 < l1_units else 0.0, **q_w1)
            w2_hex = float_to_q_format(w1[r2, c] if r2 < l1_units else 0.0, **q_w1)
            w1_hex = float_to_q_format(w1[r1, c] if r1 < l1_units else 0.0, **q_w1)
            w0_hex = float_to_q_format(w1[r0, c] if r0 < l1_units else 0.0, **q_w1)
            unified_bram_lines.append(f"{w3_hex}{w2_hex}{w1_hex}{w0_hex}")

    l1_size = len(unified_bram_lines)

    # =========================================================================
    # --- PROCESS LAYER 2 ---
    # =========================================================================
    for r_tile in range(l2_row_tiles):
        r3, r2, r1, r0 = r_tile*4 + 3, r_tile*4 + 2, r_tile*4 + 1, r_tile*4 + 0

        # Inject Bias Word
        b3_hex = float_to_q_format(b2[r3] if r3 < l2_units else 0.0, **q_w2)
        b2_hex = float_to_q_format(b2[r2] if r2 < l2_units else 0.0, **q_w2)
        b1_hex = float_to_q_format(b2[r1] if r1 < l2_units else 0.0, **q_w2)
        b0_hex = float_to_q_format(b2[r0] if r0 < l2_units else 0.0, **q_w2)
        unified_bram_lines.append(f"{b3_hex}{b2_hex}{b1_hex}{b0_hex}")

        # Stream columns
        for c in range(l1_units):
            w3_hex = float_to_q_format(w2[r3, c] if r3 < l2_units else 0.0, **q_w2)
            w2_hex = float_to_q_format(w2[r2, c] if r2 < l2_units else 0.0, **q_w2)
            w1_hex = float_to_q_format(w2[r1, c] if r1 < l2_units else 0.0, **q_w2)
            w0_hex = float_to_q_format(w2[r0, c] if r0 < l2_units else 0.0, **q_w2)
            unified_bram_lines.append(f"{w3_hex}{w2_hex}{w1_hex}{w0_hex}")

    l2_size = len(unified_bram_lines) - l1_size

    # =========================================================================
    # --- PROCESS LAYER 3 ---
    # =========================================================================
    for r_tile in range(l3_row_tiles):
        r3, r2, r1, r0 = r_tile*4 + 3, r_tile*4 + 2, r_tile*4 + 1, r_tile*4 + 0

        # Inject Bias Word
        b3_hex = float_to_q_format(b3_padded[r3], **q_w3)
        b2_hex = float_to_q_format(b3_padded[r2], **q_w3)
        b1_hex = float_to_q_format(b3_padded[r1], **q_w3)
        b0_hex = float_to_q_format(b3_padded[r0], **q_w3)
        unified_bram_lines.append(f"{b3_hex}{b2_hex}{b1_hex}{b0_hex}")

        # Stream columns
        for c in range(l2_units):
            w3_hex = float_to_q_format(w3_padded[r3, c], **q_w3)
            w2_hex = float_to_q_format(w3_padded[r2, c], **q_w3)
            w1_hex = float_to_q_format(w3_padded[r1, c], **q_w3)
            w0_hex = float_to_q_format(w3_padded[r0, c], **q_w3)
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