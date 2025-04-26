#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

// Cấu hình CUDA
#define BLOCK_SIZE 256          // Số thread trong một block
#define MAX_GRID_SIZE 65535     // Số block tối đa trong một grid
#define WARP_SIZE 32            // Kích thước warp
#define RATE 8                  // Rate của Ascon

// Debug flags
#define debug 0
#define STREAM_COUNT 4          // Số stream CUDA để chạy song song
#define USE_SHARED_MEMORY 1     // Sử dụng shared memory 
#define USE_PINNED_MEMORY 1     // Sử dụng pinned memory

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Device code 
__device__ __forceinline__ uint64_t rotr(uint64_t val, int r) {
    return (val >> r) | (val << (64 - r));
}

// Shared memory cho permutation
extern __shared__ uint64_t shared_state[];

__device__ void ascon_permutation_device(uint64_t *S, int rounds) {
    #pragma unroll
    for (int r = 12 - rounds; r < 12; r++) {
        // Round constant
        S[2] ^= (0xf0 - r * 0x10 + r * 0x1);
        
        // Substitution layer
        S[0] ^= S[4]; 
        S[4] ^= S[3]; 
        S[2] ^= S[1];
        
        uint64_t T0 = (S[0] ^ 0xFFFFFFFFFFFFFFFF) & S[1];
        uint64_t T1 = (S[1] ^ 0xFFFFFFFFFFFFFFFF) & S[2];
        uint64_t T2 = (S[2] ^ 0xFFFFFFFFFFFFFFFF) & S[3];
        uint64_t T3 = (S[3] ^ 0xFFFFFFFFFFFFFFFF) & S[4];
        uint64_t T4 = (S[4] ^ 0xFFFFFFFFFFFFFFFF) & S[0];
        
        S[0] ^= T1; 
        S[1] ^= T2; 
        S[2] ^= T3; 
        S[3] ^= T4; 
        S[4] ^= T0;
        
        S[1] ^= S[0]; 
        S[0] ^= S[4]; 
        S[3] ^= S[2]; 
        S[2] ^= 0xFFFFFFFFFFFFFFFF;
        
        // Linear diffusion layer
        S[0] ^= rotr(S[0], 19) ^ rotr(S[0], 28);
        S[1] ^= rotr(S[1], 61) ^ rotr(S[1], 39);
        S[2] ^= rotr(S[2],  1) ^ rotr(S[2],  6);
        S[3] ^= rotr(S[3], 10) ^ rotr(S[3], 17);
        S[4] ^= rotr(S[4],  7) ^ rotr(S[4], 41);
    }
}

// Kernel optimized for each block handling multiple plaintext blocks
__global__ void ascon_encrypt_advanced_kernel(const uint8_t *plaintext, uint8_t *ciphertext, 
                                             const uint8_t *key, const uint8_t *nonce,
                                             const uint8_t *associateddata, uint32_t adlen,
                                             uint32_t plaintext_length, uint32_t blocks_per_thread) {
    
    // Mỗi thread xử lý nhiều block
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;
    uint32_t block_size = RATE;
    
    // Khởi tạo state chung cho tất cả các thread trong một block
    if (USE_SHARED_MEMORY && threadIdx.x < 5) {
        shared_state[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Mỗi thread xử lý một tập hợp các block
    for (uint32_t block_index = tid; block_index < (plaintext_length + block_size - 1) / block_size; 
         block_index += total_threads) {
        
        uint32_t offset = block_index * block_size;
        if (offset >= plaintext_length) continue;
        
        // Từng thread khởi tạo trạng thái riêng
        uint64_t S[5];
        
        // IV
        S[0] = 0x80400c0600000000ULL; // IV for Ascon-128
        
        // Load key và nonce
        uint64_t key_low = 0, key_high = 0;
        uint64_t nonce_low = 0, nonce_high = 0;
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            key_low |= ((uint64_t)key[i]) << (i * 8);
            key_high |= ((uint64_t)key[i + 8]) << (i * 8);
            nonce_low |= ((uint64_t)nonce[i]) << (i * 8);
            nonce_high |= ((uint64_t)nonce[i + 8]) << (i * 8);
        }
        
        S[1] = key_low;
        S[2] = key_high;
        S[3] = nonce_low;
        S[4] = nonce_high;
        
        // Áp dụng permutation ban đầu
        ascon_permutation_device(S, 12);
        
        // Xử lý associated data nếu có
        if (adlen > 0) {
            // Tính padding
            uint32_t a_padding_len = block_size - (adlen % block_size) - 1;
            
            // Xử lý từng block của associated data
            for (uint32_t ad_block = 0; ad_block < (adlen + a_padding_len) / block_size + 1; ad_block++) {
                uint32_t ad_offset = ad_block * block_size;
                
                uint64_t ad_block_val = 0;
                
                if (ad_offset < adlen) {
                    // Số byte cần xử lý trong block hiện tại
                    uint32_t bytes_to_process = min(block_size, adlen - ad_offset);
                    
                    #pragma unroll
                    for (uint32_t i = 0; i < bytes_to_process; i++) {
                        ad_block_val |= ((uint64_t)associateddata[ad_offset + i]) << (i * 8);
                    }
                    
                    // Nếu là block cuối và cần padding
                    if (bytes_to_process < block_size) {
                        ad_block_val |= ((uint64_t)0x80) << (bytes_to_process * 8);
                    }
                }
                else if (ad_offset == adlen) {
                    // Block chỉ chứa padding
                    ad_block_val = 0x80;
                }
                
                S[0] ^= ad_block_val;
                ascon_permutation_device(S, 8);
            }
        }
        
        // Domain separation
        S[4] ^= 1;
        
        // Xử lý block plaintext
        uint32_t bytes_to_process = min(block_size, plaintext_length - offset);
        uint64_t p_block = 0;
        
        #pragma unroll
        for (uint32_t i = 0; i < bytes_to_process; i++) {
            p_block |= ((uint64_t)plaintext[offset + i]) << (i * 8);
        }
        
        // Nếu là block cuối và cần padding
        if (offset + block_size > plaintext_length) {
            p_block |= ((uint64_t)0x80) << (bytes_to_process * 8);
        }
        
        // Mã hóa block
        S[0] ^= p_block;
        
        // Ghi block ciphertext
        #pragma unroll
        for (uint32_t i = 0; i < bytes_to_process; i++) {
            ciphertext[offset + i] = (S[0] >> (i * 8)) & 0xFF;
        }
        
        // Nếu không phải block cuối, áp dụng permutation
        if (offset + block_size < plaintext_length) {
            ascon_permutation_device(S, 8);
        }
        else { // Block cuối, tạo tag
            // Finalization
            S[1] ^= key_low;
            S[2] ^= key_high;
            ascon_permutation_device(S, 12);
            S[3] ^= key_low;
            S[4] ^= key_high;
            
            // Ghi tag (16 bytes) vào cuối ciphertext
            #pragma unroll
            for (uint32_t i = 0; i < 8; i++) {
                ciphertext[plaintext_length + i] = (S[3] >> (i * 8)) & 0xFF;
                ciphertext[plaintext_length + 8 + i] = (S[4] >> (i * 8)) & 0xFF;
            }
        }
    }
}

// Host functions
void ascon_encrypt_gpu_advanced(uint8_t *ciphertext, const uint8_t *key, const uint8_t *nonce,
                             const uint8_t *associateddata, size_t adlen,
                             const uint8_t *plaintext, size_t plaintext_length) {
                              
    // Allocate device memory
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
    
    // Use pinned memory for better performance
    if (USE_PINNED_MEMORY) {
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&d_plaintext, plaintext_length));
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&d_ciphertext, plaintext_length + 16));
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&d_key, 16));
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&d_nonce, 16));
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&d_associateddata, adlen > 0 ? adlen : 1));
        
        memcpy(d_plaintext, plaintext, plaintext_length);
        memcpy(d_key, key, 16);
        memcpy(d_nonce, nonce, 16);
        if (adlen > 0) memcpy(d_associateddata, associateddata, adlen);
    }
    else {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, plaintext_length));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, plaintext_length + 16)); // +16 cho tag
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_plaintext, plaintext, plaintext_length, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_key, key, 16, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
        if (adlen > 0) CHECK_CUDA_ERROR(cudaMemcpy(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice));
    }
    
    // Tính toán số block và thread
    size_t num_blocks = (plaintext_length + RATE - 1) / RATE;
    int blocks_per_thread = 1;
    
    // Điều chỉnh số lượng threads và blocks tùy theo kích thước dữ liệu
    int threads_needed = (num_blocks + blocks_per_thread - 1) / blocks_per_thread;
    int thread_blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (thread_blocks > MAX_GRID_SIZE) {
        thread_blocks = MAX_GRID_SIZE;
        blocks_per_thread = (num_blocks + thread_blocks * BLOCK_SIZE - 1) / (thread_blocks * BLOCK_SIZE);
    }
    
    // Sử dụng streams để chạy song song
    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    size_t shared_mem_size = USE_SHARED_MEMORY ? 5 * sizeof(uint64_t) : 0;
    
    // Chạy kernel trên nhiều stream
    for (int i = 0; i < STREAM_COUNT; i++) {
        size_t stream_offset = i * (plaintext_length / STREAM_COUNT);
        size_t stream_size = (i == STREAM_COUNT - 1) ? 
                            plaintext_length - stream_offset : 
                            plaintext_length / STREAM_COUNT;
                            
        if (stream_size > 0) {
            ascon_encrypt_advanced_kernel<<<thread_blocks, BLOCK_SIZE, shared_mem_size, streams[i]>>>(
                d_plaintext + stream_offset, 
                d_ciphertext + stream_offset,
                d_key, d_nonce, d_associateddata, adlen,
                stream_size, blocks_per_thread
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
    }
    
    // Synchronize all streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }
    
    // Copy result back to host
    if (USE_PINNED_MEMORY) {
        memcpy(ciphertext, d_ciphertext, plaintext_length + 16);
    }
    else {
        CHECK_CUDA_ERROR(cudaMemcpy(ciphertext, d_ciphertext, plaintext_length + 16, cudaMemcpyDeviceToHost));
    }
    
    // Free resources
    if (USE_PINNED_MEMORY) {
        CHECK_CUDA_ERROR(cudaFreeHost(d_plaintext));
        CHECK_CUDA_ERROR(cudaFreeHost(d_ciphertext));
        CHECK_CUDA_ERROR(cudaFreeHost(d_key));
        CHECK_CUDA_ERROR(cudaFreeHost(d_nonce));
        CHECK_CUDA_ERROR(cudaFreeHost(d_associateddata));
    }
    else {
        CHECK_CUDA_ERROR(cudaFree(d_plaintext));
        CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
        CHECK_CUDA_ERROR(cudaFree(d_key));
        CHECK_CUDA_ERROR(cudaFree(d_nonce));
        CHECK_CUDA_ERROR(cudaFree(d_associateddata));
    }
    
    // Destroy streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}

// Demo function
void benchmark_gpu_advanced() {
    uint8_t key[16], nonce[16];
    uint8_t associateddata[] = "ASCON";
    uint8_t *plaintext;
    size_t plaintext_length;
    
    // Random key and nonce
    srand(time(NULL));
    for (int i = 0; i < 16; i++) {
        key[i] = rand() % 256;
        nonce[i] = rand() % 256;
    }
    
    // Read input data
    FILE *file = fopen("frame_0.txt", "r");
    if (!file) {
        perror("Failed to open plaintext file");
        exit(EXIT_FAILURE);
    }
    
    fseek(file, 0, SEEK_END);
    plaintext_length = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    plaintext = (uint8_t *)malloc(plaintext_length);
    if (!plaintext) {
        perror("Failed to allocate memory");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    fread(plaintext, 1, plaintext_length, file);
    fclose(file);
    
    // Allocate memory for ciphertext
    uint8_t *ciphertext = (uint8_t *)malloc(plaintext_length + 16);  // +16 for tag
    
    // Create GPU event for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Warm up the GPU
    ascon_encrypt_gpu_advanced(ciphertext, key, nonce, associateddata, sizeof(associateddata) - 1, 
                             plaintext, plaintext_length);
    
    // Measure encryption time
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Repeat test 10 times for better measurement
    for (int i = 0; i < 10; i++) {
        ascon_encrypt_gpu_advanced(ciphertext, key, nonce, associateddata, sizeof(associateddata) - 1, 
                                 plaintext, plaintext_length);
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    float encryption_time = milliseconds / 10.0f;  // Average over 10 runs
    
    // Save results
    FILE *output_file = fopen("ascon_advanced_gpu.txt", "w");
    if (!output_file) {
        perror("Failed to open output file");
        free(plaintext);
        free(ciphertext);
        exit(EXIT_FAILURE);
    }
    
    // Print key and nonce
    fprintf(output_file, "key: 0x");
    for (int i = 0; i < 16; i++) fprintf(output_file, "%02x", key[i]);
    fprintf(output_file, " (16 bytes)\n");
    
    fprintf(output_file, "nonce: 0x");
    for (int i = 0; i < 16; i++) fprintf(output_file, "%02x", nonce[i]);
    fprintf(output_file, " (16 bytes)\n");
    
    fprintf(output_file, "plaintext length: %zu bytes\n", plaintext_length);
    
    // Print a sample of ciphertext
    fprintf(output_file, "ciphertext (first 64 bytes): 0x");
    for (int i = 0; i < (plaintext_length < 64 ? plaintext_length : 64); i++) {
        fprintf(output_file, "%02x", ciphertext[i]);
    }
    fprintf(output_file, "...\n");
    
    fprintf(output_file, "tag: 0x");
    for (int i = 0; i < 16; i++) {
        fprintf(output_file, "%02x", ciphertext[plaintext_length + i]);
    }
    fprintf(output_file, " (16 bytes)\n");
    
    // Print performance metrics
    fprintf(output_file, "\n=== Performance Metrics (Advanced GPU) ===\n");
    fprintf(output_file, "Encryption time on GPU: %.3f ms\n", encryption_time);
    fprintf(output_file, "Data size: %zu bytes\n", plaintext_length);
    fprintf(output_file, "Throughput: %.2f MB/s\n", 
            (plaintext_length / (encryption_time / 1000.0)) / (1024.0 * 1024.0));
    
    fprintf(output_file, "\nCUDA Configuration:\n");
    fprintf(output_file, "- Threads per block: %d\n", BLOCK_SIZE);
    fprintf(output_file, "- Stream count: %d\n", STREAM_COUNT);
    fprintf(output_file, "- Using shared memory: %s\n", USE_SHARED_MEMORY ? "Yes" : "No");
    fprintf(output_file, "- Using pinned memory: %s\n", USE_PINNED_MEMORY ? "Yes" : "No");
    
    // Clean up
    fclose(output_file);
    free(plaintext);
    free(ciphertext);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    printf("Advanced GPU encryption completed in %.3f ms\n", encryption_time);
    printf("Results saved to ascon_advanced_gpu.txt\n");
}

// Main function
int main() {
    // Check for CUDA devices
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("Error: No CUDA-capable device found\n");
        return -1;
    }
    
    // Set device with highest compute capability
    int max_compute = 0;
    int max_device = 0;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, i));
        
        int compute_capability = deviceProp.major * 10 + deviceProp.minor;
        if (compute_capability > max_compute) {
            max_compute = compute_capability;
            max_device = i;
        }
    }
    
    CHECK_CUDA_ERROR(cudaSetDevice(max_device));
    
    // Get device properties
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, max_device));
    
    printf("Using GPU: %s (Compute %d.%d)\n", 
           deviceProp.name, deviceProp.major, deviceProp.minor);
    printf("- Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("- Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("- Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("- Shared Memory Per Block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
    printf("\n");
    
    // Run benchmark
    benchmark_gpu_advanced();
    
    return 0;
} 