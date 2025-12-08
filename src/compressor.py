import zstandard as zstd
import numpy as np
import os
import struct
import time

def compress_file(input_path, output_path, chunk_size=50*1024*1024, level=3):
    """
    Compresses a file using Byte-Plane Splitting + Zstd.
    Block format in output file:
    [Original Size (4 bytes)] [Compressed Size (4 bytes)] [Compressed Data]
    """
    cctx = zstd.ZstdCompressor(level=level)
    
    total_original = 0
    total_compressed = 0
    start_time = time.time()

    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            
            original_len = len(chunk)
            total_original += original_len

            # 1. Byte Splitting (Transformation)
            # If size is odd, keep the last byte separately
            if original_len % 2 != 0:
                data_even = chunk[:-1]
                last_byte = chunk[-1:]
            else:
                data_even = chunk
                last_byte = b''

            # Use numpy to quickly separate even and odd bytes
            arr = np.frombuffer(data_even, dtype=np.uint8)
            bytes_even = arr[0::2].tobytes()
            bytes_odd = arr[1::2].tobytes()
            
            # Concatenate: [Even Bytes] + [Odd Bytes] + [Leftover]
            transformed_data = bytes_even + bytes_odd + last_byte

            # 2. Zstd Compression
            compressed_data = cctx.compress(transformed_data)
            compressed_len = len(compressed_data)
            total_compressed += compressed_len

            # 3. Write Block (Header + Data)
            # Header: Original Size (4B) + Compressed Size (4B)
            f_out.write(struct.pack('<II', original_len, compressed_len))
            f_out.write(compressed_data)

    end_time = time.time()
    print(f"Compression Completed:")
    print(f"  Original: {total_original / 1024 / 1024:.2f} MB")
    print(f"  Compressed: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"  Ratio: {total_original / total_compressed:.2f}x")
    print(f"  Time: {end_time - start_time:.2f}s")

def decompress_file(input_path, output_path):
    """
    Decompresses a file created by the compress_file function.
    """
    dctx = zstd.ZstdDecompressor()
    
    start_time = time.time()
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        while True:
            # 1. Read Block Header (8 bytes)
            header = f_in.read(8)
            if not header:
                break
            
            original_len, compressed_len = struct.unpack('<II', header)
            
            # 2. Read Compressed Data
            compressed_data = f_in.read(compressed_len)
            
            # 3. Zstd Decompression
            transformed_data = dctx.decompress(compressed_data)
            
            # 4. Reverse Byte Splitting
            # Calculate where even bytes end
            # If original_len is odd, we have 1 leftover byte at the end
            split_len = original_len // 2
            
            bytes_even = transformed_data[:split_len]
            bytes_odd = transformed_data[split_len:split_len*2]
            
            # Reconstruct interleaved array
            # Create an empty array and fill it
            reconstructed = np.empty(split_len * 2, dtype=np.uint8)
            reconstructed[0::2] = np.frombuffer(bytes_even, dtype=np.uint8)
            reconstructed[1::2] = np.frombuffer(bytes_odd, dtype=np.uint8)
            
            f_out.write(reconstructed.tobytes())
            
            # Write the leftover byte if it exists
            if original_len % 2 != 0:
                f_out.write(transformed_data[-1:])

    end_time = time.time()
    print(f"Decompression Completed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    # Usage example
    INPUT_FILE = "../model.safetensors"
    COMPRESSED_FILE = "../model.safetensors.zst_split"
    RESTORED_FILE = "../model_restored.safetensors"

    if os.path.exists(INPUT_FILE):
        print("Starting compression...")
        compress_file(INPUT_FILE, COMPRESSED_FILE)
        
        print("\nStarting decompression...")
        decompress_file(COMPRESSED_FILE, RESTORED_FILE)
        
        # Simple verification
        orig_size = os.path.getsize(INPUT_FILE)
        rest_size = os.path.getsize(RESTORED_FILE)
        print(f"\nSize Verification: {'OK' if orig_size == rest_size else 'FAILED'}")
    else:
        print(f"File {INPUT_FILE} not found.")
