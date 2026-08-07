#!/usr/bin/env python3
"""
Compare two dump directories and find where tensors diverge.
Loads raw binary tensor files and reports the first layer that differs.
"""

import struct
import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


def load_raw_tensor(raw_file: str) -> Tuple[np.ndarray, str]:
    """
    Load a raw binary tensor file.
    Format: [ndim (int32)][dim0 (int64)][dim1 (int64)]...[data (float32)]
    Returns: (tensor as numpy array, error message or empty string)
    """
    try:
        with open(raw_file, 'rb') as f:
            # Read number of dimensions
            ndim_bytes = f.read(4)
            if len(ndim_bytes) < 4:
                return None, "File too short to read ndim"
            ndim = struct.unpack('<i', ndim_bytes)[0]
            
            # Read shape
            shape = []
            for _ in range(ndim):
                dim_bytes = f.read(8)
                if len(dim_bytes) < 8:
                    return None, f"File too short to read dimension"
                dim = struct.unpack('<q', dim_bytes)[0]
                shape.append(dim)
            
            # Calculate expected data size
            numel = int(np.prod(shape)) if shape else 1
            expected_bytes = numel * 4  # float32 = 4 bytes
            
            # Read data
            data_bytes = f.read(expected_bytes)
            if len(data_bytes) < expected_bytes:
                return None, f"Expected {expected_bytes} bytes, got {len(data_bytes)}"
            
            # Convert to numpy array
            data = np.frombuffer(data_bytes, dtype=np.float32, count=numel)
            tensor = data.reshape(shape)
            
            return tensor, ""
    except Exception as e:
        return None, str(e)


def load_tensors_from_dir(dump_dir: str) -> Dict[str, np.ndarray]:
    """Load all .raw files from a directory."""
    tensors = {}
    dump_path = Path(dump_dir)
    
    if not dump_path.exists():
        print(f"Error: Directory {dump_dir} does not exist")
        return tensors
    
    raw_files = sorted(dump_path.glob("*.raw"))
    print(f"Found {len(raw_files)} tensor files in {dump_dir}")
    
    for raw_file in raw_files:
        tensor, error = load_raw_tensor(str(raw_file))
        if error:
            print(f"Warning: {raw_file.name}: {error}")
        elif tensor is not None:
            tensors[raw_file.name] = tensor
    
    return tensors


def compare_tensors(t1: np.ndarray, t2: np.ndarray, name: str = "") -> Tuple[bool, str]:
    """
    Compare two tensors and return (are_equal, message).
    """
    # Ensure we have actual arrays
    if not isinstance(t1, np.ndarray) or not isinstance(t2, np.ndarray):
        return False, f"Not both arrays: {type(t1).__name__} vs {type(t2).__name__}"
    
    # Check shapes
    if t1.shape != t2.shape:
        return False, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    
    # Check dtypes
    if t1.dtype != t2.dtype:
        return False, f"Dtype mismatch: {t1.dtype} vs {t2.dtype}"
    
    # Compare
    if np.allclose(t1, t2, rtol=1e-4, atol=1e-5):
        return True, "Tensors match"
    
    # Compute differences
    diff = np.abs(t1 - t2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_different = np.sum(diff > 1e-5)
    
    msg = (f"Tensors differ - "
           f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
           f"num_different_elements={num_different}/{t1.size}")
    
    return False, msg


def main():
    if len(sys.argv) < 3:
        print("Usage: python first_layer_diff.py <dump_dir_1> <dump_dir_2>")
        print("\nCompares tensors between two dump directories and reports the first divergence.")
        sys.exit(1)
    
    dump_dir1 = sys.argv[1]
    dump_dir2 = sys.argv[2]
    
    print(f"Loading tensors from: {dump_dir1}")
    tensors1 = load_tensors_from_dir(dump_dir1)
    
    print(f"Loading tensors from: {dump_dir2}")
    tensors2 = load_tensors_from_dir(dump_dir2)
    
    if not tensors1 or not tensors2:
        print("Error: Could not load tensors from one or both directories")
        sys.exit(1)
    
    # Get common filenames
    names1 = set(tensors1.keys())
    names2 = set(tensors2.keys())
    
    print(f"\nTensors in dir1: {len(names1)}")
    print(f"Tensors in dir2: {len(names2)}")
    
    # Find common tensors
    common = sorted(names1 & names2)
    only_in_1 = sorted(names1 - names2)
    only_in_2 = sorted(names2 - names1)
    
    if only_in_1:
        print(f"\nOnly in dir1: {only_in_1}")
    if only_in_2:
        print(f"\nOnly in dir2: {only_in_2}")
    
    # Compare common tensors
    print(f"\nComparing {len(common)} common tensors:\n")
    
    first_diff = None
    all_match = True
    
    for i, name in enumerate(common, 1):
        t1 = tensors1[name]
        t2 = tensors2[name]
        
        match, msg = compare_tensors(t1, t2, name)
        
        status = "✓" if match else "✗"
        print(f"{i:3d}. {status} {name:40s} | {msg}")
        
        if not match and first_diff is None:
            first_diff = (i, name)
            all_match = False
    
    # Summary
    print("\n" + "="*80)
    if all_match:
        print("SUCCESS: All tensors match!")
    elif first_diff:
        idx, name = first_diff
        print(f"FIRST DIVERGENCE: Layer {idx} ({name})")
        print(f"  Dir1 shape: {tensors1[name].shape}, dtype: {tensors1[name].dtype}")
        print(f"  Dir2 shape: {tensors2[name].shape}, dtype: {tensors2[name].dtype}")
    else:
        print("MISMATCH: Different tensor sets between directories")


if __name__ == "__main__":
    main()
