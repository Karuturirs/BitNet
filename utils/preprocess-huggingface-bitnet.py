from safetensors import safe_open
from safetensors.torch import save_file
import torch
import numpy as np

def unpack_weights(weight, scale):
    """
    Unpacks 2-bit weights from uint8 and scales them.
    This logic matches the requirements for BitNet-b1.58-2B-4T.
    """
    origin_shape = weight.shape
    # 4 weights per byte. 
    # Based on quantize_i2_s: (q0 << 6) | (q1 << 4) | (q2 << 2) | (q3 << 0)
    # q0 is bits 6-7, q1 is bits 4-5, q2 is bits 2-3, q3 is bits 0-1
    shift = torch.tensor([6, 4, 2, 0], dtype=torch.uint8).reshape((4, 1, 1))
    
    # Unpack bits
    # (4, rows, cols)
    unpacked = weight.unsqueeze(0).expand((4, *origin_shape)) >> shift
    unpacked = unpacked & 3
    
    # BitNet b1.58 ternary mapping (0, 1, 2 map to -1, 0, 1)
    unpacked = unpacked.float() - 1
    
    # Interleave correctly: (4, rows, cols) -> (rows, 4, cols) -> (rows*4, cols)
    # This puts q0, q1, q2, q3 in adjacent rows.
    unpacked = unpacked.permute(1, 0, 2).reshape((origin_shape[0] * 4, origin_shape[1]))
    
    # Apply scale (multiply)
    return unpacked * scale.float()

def quant_model(input_path, output_path):
    tensors = {}
    
    with safe_open(input_path, framework='pt') as f:
        # First pass: collect all scales
        scales = {}
        for name in f.keys():
            if name.endswith('.weight_scale'):
                scales[name.replace('.weight_scale', '')] = f.get_tensor(name)
        
        # Second pass: process tensors
        for name in f.keys():
            if name.endswith('.weight_scale'):
                continue
                
            tensor = f.get_tensor(name)
            
            # Check if this weight is packed (has a corresponding scale)
            base_name = name.replace('.weight', '')
            if base_name in scales:
                print(f'[INFO] Unpacking and scaling {name}')
                tensors[name] = unpack_weights(tensor, scales[base_name])
            else:
                # Keep other tensors (norms, embeddings) as is, but convert to float32
                tensors[name] = tensor.to(torch.float32)
    
    print(f'[INFO] Saving preprocessed model to {output_path}')
    save_file(tensors, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess BitNet Safetensors for GGUF conversion")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    quant_model(input_path=args.input, output_path=args.output)
