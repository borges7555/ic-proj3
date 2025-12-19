import os
import time
import struct
import zlib
import bz2
import lzma
import zstandard as zstd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re

INPUT_FILE = "model.safetensors"
CHUNK_SIZE = 100 * 1024 * 1024  # Benchmark on 100 MB
TEST_DATA = b""
TEMP_INPUT = "benchmark_sample.bin"
TEMP_COMPRESSED = "benchmark_sample.zst"
TEMP_RESTORED = "benchmark_sample.restored"

def load_data():
    global TEST_DATA
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "rb") as f:
            TEST_DATA = f.read(CHUNK_SIZE)
    else:
        # Create dummy data if file doesn't exist (fallback)
        TEST_DATA = os.urandom(CHUNK_SIZE)
    
    # Write temp file for C++ benchmark
    with open(TEMP_INPUT, "wb") as f:
        f.write(TEST_DATA)

def cleanup():
    for f in [TEMP_INPUT, TEMP_COMPRESSED, TEMP_RESTORED]:
        if os.path.exists(f):
            os.remove(f)

def get_size(data):
    return len(data)

def benchmark_compressor(name, compress_func, decompress_func):
    # Compression
    start = time.time()
    compressed = compress_func(TEST_DATA)
    comp_time = time.time() - start
    
    size = len(compressed)
    ratio = len(TEST_DATA) / size if size > 0 else 0
    
    # Decompression
    start = time.time()
    decompressed = decompress_func(compressed)
    decomp_time = time.time() - start
    
    assert len(decompressed) == len(TEST_DATA)
    
    return {
        "name": name,
        "ratio": ratio,
        "comp_time": comp_time,
        "decomp_time": decomp_time,
        "size_mb": size / (1024*1024)
    }

def benchmark_cpp_compressor():
    # Run C++ executable
    # Usage: ./build/compressor <input> <compressed> <restored>
    cmd = ["./build/compressor", TEMP_INPUT, TEMP_COMPRESSED, TEMP_RESTORED]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse output for times and ratio
        # Expected output format:
        # Compression Completed:
        #   Original: ... MB
        #   Compressed: ... MB
        #   Ratio: 1.46x
        #   Time: 0.5962s
        # 
        # Starting decompression (C++)...
        # Decompression Completed in 0.2370s
        
        comp_time_match = re.search(r"Time: ([\d\.]+)s", output)
        decomp_time_match = re.search(r"Decompression Completed in ([\d\.]+)s", output)
        
        comp_time = float(comp_time_match.group(1)) if comp_time_match else 0.0
        decomp_time = float(decomp_time_match.group(1)) if decomp_time_match else 0.0
        
        compressed_size = os.path.getsize(TEMP_COMPRESSED)
        ratio = len(TEST_DATA) / compressed_size if compressed_size > 0 else 0
        
        return {
            "name": "Split+Zstd (C++)",
            "ratio": ratio,
            "comp_time": comp_time,
            "decomp_time": decomp_time,
            "size_mb": compressed_size / (1024*1024)
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ benchmark: {e}")
        print(e.stdout)
        print(e.stderr)
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Custom Strategy ---
def split_bytes(data):
    # Ensure even length for reshaping
    if len(data) % 2 != 0:
        data = data[:-1] # Drop last byte for simplicity in benchmark
        
    arr = np.frombuffer(data, dtype=np.uint8)
    bytes_even = arr[0::2].tobytes()
    bytes_odd = arr[1::2].tobytes()
    return bytes_even + bytes_odd

def unsplit_bytes(data):
    mid = len(data) // 2
    bytes_even = data[:mid]
    bytes_odd = data[mid:]
    
    reconstructed = np.empty(len(data), dtype=np.uint8)
    reconstructed[0::2] = np.frombuffer(bytes_even, dtype=np.uint8)
    reconstructed[1::2] = np.frombuffer(bytes_odd, dtype=np.uint8)
    return reconstructed.tobytes()

def compress_custom(data, level=3):
    split_data = split_bytes(data)
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(split_data)

def decompress_custom(data):
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(data)
    return unsplit_bytes(decompressed)

# --- Main ---
def main():
    load_data()
    print(f"Benchmarking on {len(TEST_DATA)/1024/1024:.2f} MB of data...")
    
    results = []
    
    # 1. Gzip
    results.append(benchmark_compressor("Gzip (L6)", 
        lambda d: zlib.compress(d, level=6), 
        lambda d: zlib.decompress(d)))
        
    # 2. Bzip2
    results.append(benchmark_compressor("Bzip2", 
        lambda d: bz2.compress(d), 
        lambda d: bz2.decompress(d)))
        
    # 3. LZMA (XZ)
    results.append(benchmark_compressor("LZMA", 
        lambda d: lzma.compress(d), 
        lambda d: lzma.decompress(d)))
        
    # 4. Zstd (Standard)
    for level in [1, 3, 9]:
        cctx = zstd.ZstdCompressor(level=level)
        dctx = zstd.ZstdDecompressor()
        results.append(benchmark_compressor(f"Zstd (L{level})", 
            lambda d: cctx.compress(d), 
            lambda d: dctx.decompress(d)))

    # 5. Custom (Split + Zstd) - C++
    cpp_result = benchmark_cpp_compressor()
    if cpp_result:
        results.append(cpp_result)

    # Print Table
    print(f"{'Algorithm':<25} | {'Ratio':<10} | {'Comp Time (s)':<15} | {'Decomp Time (s)':<15}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} | {r['ratio']:<10.2f} | {r['comp_time']:<15.4f} | {r['decomp_time']:<15.4f}")

    # Plotting
    names = [r['name'] for r in results]
    ratios = [r['ratio'] for r in results]
    comp_times = [r['comp_time'] for r in results]
    
    # Bar Chart: Compression Ratio
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, ratios, color='skyblue')
    plt.title('Compression Ratio Comparison')
    plt.ylabel('Ratio (Original / Compressed)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/ratio_comparison.png')
    
    # Scatter Plot: Time vs Ratio
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.scatter(r['comp_time'], r['ratio'], label=r['name'], s=100)
    
    plt.title('Compression Efficiency (Time vs Ratio)')
    plt.xlabel('Compression Time (s)')
    plt.ylabel('Compression Ratio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('images/efficiency_comparison.png')
    
    cleanup()

if __name__ == "__main__":
    main()
