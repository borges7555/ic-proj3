#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <zstd.h>

// Chunk size: 50 MB
const size_t CHUNK_SIZE = 50 * 1024 * 1024;

void compress_file(const std::string& input_path, const std::string& output_path, int level = 3) {
    std::ifstream f_in(input_path, std::ios::binary);
    std::ofstream f_out(output_path, std::ios::binary);

    if (!f_in.is_open() || !f_out.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        return;
    }

    std::vector<char> buffer(CHUNK_SIZE);
    size_t total_original = 0;
    size_t total_compressed = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (f_in) {
        f_in.read(buffer.data(), CHUNK_SIZE);
        std::streamsize bytes_read = f_in.gcount();
        if (bytes_read == 0) break;

        total_original += bytes_read;

        // 1. Byte Splitting (Transformation)
        std::vector<char> transformed_data(bytes_read);
        
        bool is_odd = (bytes_read % 2 != 0);
        size_t process_len = is_odd ? bytes_read - 1 : bytes_read;
        size_t half_size = process_len / 2;

        const char* src = buffer.data();
        char* dest_even = transformed_data.data();
        char* dest_odd = transformed_data.data() + half_size;

        for (size_t i = 0; i < process_len; i += 2) {
            *dest_even++ = src[i];
            *dest_odd++ = src[i+1];
        }

        if (is_odd) {
            transformed_data[bytes_read - 1] = src[bytes_read - 1];
        }

        // 2. Zstd Compression
        size_t bound = ZSTD_compressBound(bytes_read);
        std::vector<char> compressed_data(bound);

        size_t cSize = ZSTD_compress(compressed_data.data(), bound, transformed_data.data(), bytes_read, level);
        if (ZSTD_isError(cSize)) {
            std::cerr << "ZSTD error: " << ZSTD_getErrorName(cSize) << std::endl;
            return;
        }

        total_compressed += cSize;

        // 3. Write Block Header (Original Size, Compressed Size)
        // Using Little Endian for consistency with Python struct.pack('<II')
        // On x86/x64 systems, this is native.
        uint32_t orig_len_u32 = static_cast<uint32_t>(bytes_read);
        uint32_t comp_len_u32 = static_cast<uint32_t>(cSize);
        
        f_out.write(reinterpret_cast<const char*>(&orig_len_u32), sizeof(orig_len_u32));
        f_out.write(reinterpret_cast<const char*>(&comp_len_u32), sizeof(comp_len_u32));
        f_out.write(compressed_data.data(), cSize);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Compression Completed:" << std::endl;
    std::cout << "  Original: " << total_original / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Compressed: " << total_compressed / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Ratio: " << (double)total_original / total_compressed << "x" << std::endl;
    std::cout << "  Time: " << elapsed.count() << "s" << std::endl;
}

void decompress_file(const std::string& input_path, const std::string& output_path) {
    std::ifstream f_in(input_path, std::ios::binary);
    std::ofstream f_out(output_path, std::ios::binary);

    if (!f_in.is_open() || !f_out.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    while (f_in) {
        // 1. Read Header
        uint32_t orig_len_u32;
        uint32_t comp_len_u32;

        f_in.read(reinterpret_cast<char*>(&orig_len_u32), sizeof(orig_len_u32));
        if (f_in.gcount() == 0) break; // End of file
        f_in.read(reinterpret_cast<char*>(&comp_len_u32), sizeof(comp_len_u32));

        size_t original_len = orig_len_u32;
        size_t compressed_len = comp_len_u32;

        // 2. Read Compressed Data
        std::vector<char> compressed_data(compressed_len);
        f_in.read(compressed_data.data(), compressed_len);

        // 3. Decompress
        std::vector<char> transformed_data(original_len);
        size_t dSize = ZSTD_decompress(transformed_data.data(), original_len, compressed_data.data(), compressed_len);
        
        if (ZSTD_isError(dSize)) {
            std::cerr << "ZSTD error: " << ZSTD_getErrorName(dSize) << std::endl;
            return;
        }

        // 4. Reverse Byte Splitting
        std::vector<char> reconstructed(original_len);
        
        bool is_odd = (original_len % 2 != 0);
        size_t process_len = is_odd ? original_len - 1 : original_len;
        size_t half_size = process_len / 2;

        const char* src_even = transformed_data.data();
        const char* src_odd = transformed_data.data() + half_size;
        char* dest = reconstructed.data();

        for (size_t i = 0; i < half_size; ++i) {
            dest[2*i] = src_even[i];
            dest[2*i+1] = src_odd[i];
        }

        if (is_odd) {
            reconstructed[original_len - 1] = transformed_data[original_len - 1];
        }

        f_out.write(reconstructed.data(), original_len);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Decompression Completed in " << elapsed.count() << "s" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string input_file = "model.safetensors";
    std::string compressed_file = "model.safetensors.zst_split_cpp";
    std::string restored_file = "model_restored_cpp.safetensors";

    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) compressed_file = argv[2];
    if (argc >= 4) restored_file = argv[3];

    // Check if input file exists
    std::ifstream check(input_file);
    if (!check.good()) {
        std::cout << "File " << input_file << " not found." << std::endl;
        return 1;
    }
    check.close();

    std::cout << "Starting compression (C++)..." << std::endl;
    compress_file(input_file, compressed_file);

    std::cout << "\nStarting decompression (C++)..." << std::endl;
    decompress_file(compressed_file, restored_file);

    return 0;
}
