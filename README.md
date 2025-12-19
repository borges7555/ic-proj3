# IC Project 3 - LLM Weights Compression

## Description
This project implements a specialized compression algorithm for Large Language Model (LLM) weights stored in `model.safetensors` format (BF16). The strategy uses **Byte-Plane Splitting** combined with **Zstandard (Zstd)** to achieve high compression ratios with extremely fast processing times.

The solution exploits the structure of BF16 numbers (16-bit floating point) by separating the high-entropy mantissa bytes from the low-entropy exponent bytes. This allows the Zstd compressor to find more patterns in the exponent stream, significantly improving compression efficiency compared to standard tools.

## Requirements

### System Dependencies
- **C++ Compiler**: `g++`
- **Zstd Library**: `libzstd-dev` (Ubuntu/Debian) or `zstd` (Arch/Fedora)

### Python Dependencies (for Benchmarking)
- Python 3.x
- Packages listed in `requirements.txt`

## Build Instructions

To compile the C++ compressor, simply run:

```bash
make
```

This will create the executable `build/compressor`.

## Usage

To compress and decompress a file:

```bash
./build/compressor <input_file> <compressed_output> <restored_output>
```

**Example:**
```bash
./build/compressor model.safetensors model.zst model_restored.safetensors
```

The program will:
1. Compress `model.safetensors` to `model.zst`.
2. Decompress `model.zst` to `model_restored.safetensors`.
3. Report the compression ratio and execution times.

## Benchmarking

We provide a Python script to compare our solution against standard algorithms (Gzip, Bzip2, LZMA, Zstd).

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the benchmark:**
   ```bash
   python3 benchmark.py
   ```

This script will:
- Generate a temporary 100MB sample from the model (or random data if the model is missing).
- Test all algorithms.
- Generate comparison graphs in the `images/` directory.
- Print a summary table to the console.

## Report

The detailed project report is available in `ic-proj3-report.pdf`.

## Presentation

The presentation slides are available in `IC Project 3.pdf`

## Authors
- Henrique Cruz (103442)
- Diogo Borges (102954)
- Piotr Bartczak (130327)

**University of Aveiro - Information and Coding**
