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

// Align data to 128-byte blocks for coalesced memory access
#define ALIGN_128 __align__(128)

// Shared memory for ASCON state
__shared__ ALIGN_128 uint8_t shared_state[40]; // 320-bit ASCON state with padding to avoid bank conflicts

__device__ __host__ void zero_bytes_device(uint8_t *bytes, size_t n) {
    for (size_t i = 0; i < n; i++) {
        bytes[i] = 0;
    }
}

__device__ __host__ void ff_bytes_device(uint8_t *bytes, size_t n) {
    for (size_t i = 0; i < n; i++) {
        bytes[i] = 0xFF;
    }
}

__device__ __host__ void to_bytes_device(uint8_t *dest, const uint8_t *src, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}

__device__ __host__ uint64_t bytes_to_int_device(const uint8_t *bytes, size_t len) {
    uint64_t result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= ((uint64_t)bytes[i]) << (i * 8);
    }
    return result;
}

__device__ __host__ void bytes_to_state_device(uint64_t *state, const uint8_t *bytes) {
    for (size_t i = 0; i < 5; i++) {
        state[i] = bytes_to_int_device(bytes + 8 * i, 8);
    }
}

__device__ __host__ void int_to_bytes_device(uint8_t *bytes, uint64_t integer, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) {
        bytes[i] = (integer >> (i * 8)) & 0xFF;
    }
}

__device__ __host__ uint64_t rotr_device(uint64_t val, int r) {
    return (val >> r) | (val << (64 - r));
}

// Add the missing ascon_permutation_device function
__device__ __host__ void ascon_permutation_device(uint64_t *S, int rounds) {
    // Implement the permutation function
    for (int r = 12 - rounds; r < 12; r++) {
        // Add round constants
        S[2] ^= (0xf0 - r * 0x10 + r * 0x1);
        
        // Substitution layer (S-box)
        S[0] ^= S[4];
        S[4] ^= S[3];
        S[2] ^= S[1];
        
        uint64_t T0 = S[0] ^ (~S[1] & S[2]);
        uint64_t T1 = S[1] ^ (~S[2] & S[3]);
        uint64_t T2 = S[2] ^ (~S[3] & S[4]);
        uint64_t T3 = S[3] ^ (~S[4] & S[0]);
        uint64_t T4 = S[4] ^ (~S[0] & S[1]);
        
        S[0] = T0;
        S[1] = T1;
        S[2] = T2;
        S[3] = T3;
        S[4] = T4;
        
        // Linear diffusion layer
        S[0] ^= rotr_device(S[0], 19) ^ rotr_device(S[0], 28);
        S[1] ^= rotr_device(S[1], 61) ^ rotr_device(S[1], 39);
        S[2] ^= rotr_device(S[2], 1) ^ rotr_device(S[2], 6);
        S[3] ^= rotr_device(S[3], 10) ^ rotr_device(S[3], 17);
        S[4] ^= rotr_device(S[4], 7) ^ rotr_device(S[4], 41);
    }
}

// Main GPU functions
__device__ void ascon_initialize_device(uint64_t *S, const uint8_t *key, const uint8_t *nonce, 
                                       int a, int b, int rate, int taglen, uint8_t version) {
    // Khởi tạo IV
    uint8_t iv[40];
    zero_bytes_device(iv, 40);
    
    iv[0] = version;
    iv[1] = 0;
    iv[2] = (b << 4) + a;
    int_to_bytes_device(iv + 3, taglen, 2);
    iv[5] = rate;
    iv[6] = 0;
    iv[7] = 0;
    
    // Sao chép key và nonce
    to_bytes_device(iv + 8, key, 16);
    to_bytes_device(iv + 24, nonce, 16);
    
    // Chuyển đổi bytes thành trạng thái
    bytes_to_state_device(S, iv);
    
    // Áp dụng permutation
    ascon_permutation_device(S, a);
}

__device__ void ascon_process_associated_data_device(uint64_t *S, const uint8_t *associateddata, 
                                                   size_t adlen, int b, int rate) {
    if (adlen > 0) {
        // Tính padding
        size_t a_padding_len = rate - (adlen % rate) - 1;
        
        uint8_t a_padded[256]; // Giả sử AD không quá dài
        uint8_t a_padding[8]; // Giả sử rate tối đa là 8
        
        // Tạo padding
        a_padding[0] = 0x80;
        zero_bytes_device(a_padding + 1, rate - 1);
        
        // Sao chép AD và padding
        to_bytes_device(a_padded, associateddata, adlen);
        to_bytes_device(a_padded + adlen, a_padding, a_padding_len + 1);
        
        // Xử lý từng block
        for (size_t block = 0; block < adlen + a_padding_len + 1; block += rate) {
            S[0] ^= bytes_to_int_device(a_padded + block, rate);
            ascon_permutation_device(S, b);
        }
    }
    
    // Domain separation
    S[4] ^= 1;
}

__device__ void ascon_process_plaintext_device(uint64_t *S, uint8_t *ciphertext, 
                                             const uint8_t *plaintext, size_t ptlen, 
                                             int b, int rate) {
    // Tính padding
    size_t p_padding_len = rate - (ptlen % rate) - 1;
    
    uint8_t p_padded[1024]; // Dữ liệu plaintext có padding
    uint8_t p_padding[8]; // Padding
    
    // Tạo padding
    p_padding[0] = 0x80;
    zero_bytes_device(p_padding + 1, rate - 1);
    
    // Sao chép plaintext và padding
    to_bytes_device(p_padded, plaintext, ptlen);
    to_bytes_device(p_padded + ptlen, p_padding, p_padding_len + 1);
    
    // Xử lý từng block
    size_t c_len = 0;
    for (size_t block = 0; block < ptlen + p_padding_len + 1; block += rate) {
        S[0] ^= bytes_to_int_device(p_padded + block, rate);
        int_to_bytes_device(ciphertext + c_len, S[0], rate);
        c_len += rate;
        ascon_permutation_device(S, b);
    }
}

__device__ void ascon_process_ciphertext_device(uint64_t *S, uint8_t *plaintext, 
                                              const uint8_t *ciphertext, size_t ctlen, 
                                              int b, int rate) {
    // Tính padding
    size_t c_padding_len = rate - (ctlen % rate) - 1;
    
    uint8_t c_padded[1024]; // Dữ liệu ciphertext có padding
    uint8_t c_padding[8]; // Padding
    
    // Tạo padding
    c_padding[0] = 0x80;
    zero_bytes_device(c_padding + 1, rate - 1);
    
    // Sao chép ciphertext và padding
    to_bytes_device(c_padded, ciphertext, ctlen);
    to_bytes_device(c_padded + ctlen, c_padding, c_padding_len + 1);
    
    // Xử lý từng block
    size_t p_len = 0;
    for (size_t block = 0; block < ctlen + c_padding_len + 1; block += rate) {
        uint64_t c_block = bytes_to_int_device(c_padded + block, rate);
        int_to_bytes_device(plaintext + p_len, S[0] ^ c_block, rate);
        p_len += rate;
        S[0] = c_block;
        ascon_permutation_device(S, b);
    }
}

__device__ void ascon_finalize_device(uint64_t *S, uint8_t *tag, const uint8_t *key, 
                                    int a, int taglen) {
    // Key addition
    S[1] ^= bytes_to_int_device(key, 8);
    S[2] ^= bytes_to_int_device(key + 8, 8);
    
    // Permutation
    ascon_permutation_device(S, a);
    
    // Key addition
    S[3] ^= bytes_to_int_device(key, 8);
    S[4] ^= bytes_to_int_device(key + 8, 8);
    
    // Tạo tag
    int_to_bytes_device(tag, S[3], 8);
    int_to_bytes_device(tag + 8, S[4], 8);
}

// Host functions cho CPU
void zero_bytes(uint8_t *bytes, size_t n) {
    memset(bytes, 0, n);
}

void ff_bytes(uint8_t *bytes, size_t n) {
    memset(bytes, 0xFF, n);
}

void to_bytes(uint8_t *dest, const uint8_t *src, size_t len) {
    memcpy(dest, src, len);
}

uint64_t bytes_to_int(const uint8_t *bytes, size_t len) {
    uint64_t result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= ((uint64_t)bytes[i]) << (i * 8);
    }
    return result;
}

void bytes_to_state(uint64_t *state, const uint8_t *bytes) {
    for (size_t i = 0; i < 5; i++) {
        state[i] = bytes_to_int(bytes + 8 * i, 8);
    }
}

void int_to_bytes(uint8_t *bytes, uint64_t integer, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) {
        bytes[i] = (integer >> (i * 8)) & 0xFF;
    }
}

uint64_t rotr(uint64_t val, int r) {
    return (val >> r) | (val << (64 - r));
}

// Hàm sinh bytes ngẫu nhiên
void get_random_bytes(uint8_t *bytes, size_t num) {
    for (size_t i = 0; i < num; i++) {
        bytes[i] = rand() % 256;
    }
}

// Hàm in ra file
void demo_print(FILE *file, const char *text, const uint8_t *val, size_t length) {
    fprintf(file, "%s: 0x", text);
    for (size_t i = 0; i < length; i++) {
        fprintf(file, "%02x", val[i]);
    }
    fprintf(file, " (%zu bytes)\n", length);
}

// Hàm đọc từ file
void read_plaintext_from_file(const char *filename, uint8_t **plaintext, size_t *length) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open plaintext file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    *length = file_size;
    *plaintext = (uint8_t *)malloc(*length);
    if (!*plaintext) {
        perror("Failed to allocate memory for plaintext");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(*plaintext, 1, *length, file);
    fclose(file);
} 

// Optimized kernel for encryption using shared memory
__global__ void ascon_encrypt_kernel_optimized(const uint8_t *plaintext, uint8_t *ciphertext, 
                                               const uint8_t *key, const uint8_t *nonce,
                                               const uint8_t *associateddata, uint32_t adlen,
                                               uint32_t plaintext_length) {
    // Thread and block indices
    uint32_t tid = threadIdx.x;
    uint32_t block_offset = blockIdx.x * blockDim.x * RATE;

    // Load key and nonce into shared memory
    if (tid < 16) {
        shared_state[tid] = key[tid];
        shared_state[16 + tid] = nonce[tid];
    }
    __syncthreads();

    // Process plaintext in 128-byte aligned blocks
    for (uint32_t offset = block_offset + tid * RATE; offset < plaintext_length; offset += blockDim.x * RATE) {
        uint8_t local_block[RATE];
        if (offset < plaintext_length) {
            // Load plaintext into local memory
            for (int i = 0; i < RATE; i++) {
                local_block[i] = plaintext[offset + i];
            }

            // Perform encryption using shared state
            uint64_t S[5];
            ascon_initialize_device(S, shared_state, shared_state + 16, 12, 8, RATE, 16, 1);
            ascon_process_associated_data_device(S, associateddata, adlen, 8, RATE);
            S[0] ^= bytes_to_int_device(local_block, RATE);
            ascon_permutation_device(S, 8);
            int_to_bytes_device(ciphertext + offset, S[0], RATE);
        }
    }
}

// Placeholder for TensorRT/NVDLA integration
void integrate_tensorrt_nvdla() {
    // TensorRT/NVDLA integration for INT8 matrix operations
    // Example: Use TensorRT for ascon128_check or XOR operations
    // Example: Use NVDLA for plaintext-decrypted comparison
    // ...integration code...
}

// Optimized encryption function
void ascon_encrypt_gpu_optimized(uint8_t *ciphertext, const uint8_t *key, const uint8_t *nonce,
                                 const uint8_t *associateddata, size_t adlen,
                                 const uint8_t *plaintext, size_t plaintext_length) {
    // Allocate GPU memory
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, plaintext_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, plaintext_length + 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_plaintext, plaintext, plaintext_length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key, key, 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
    if (adlen > 0) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice));
    }

    // Launch optimized kernel
    int num_blocks = (plaintext_length + RATE - 1) / RATE;
    int thread_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (thread_blocks > MAX_GRID_SIZE) {
        thread_blocks = MAX_GRID_SIZE;
    }
    ascon_encrypt_kernel_optimized<<<thread_blocks, BLOCK_SIZE>>>(d_plaintext, d_ciphertext, 
                                                                  d_key, d_nonce, d_associateddata, 
                                                                  adlen, plaintext_length);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy results back to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(ciphertext, d_ciphertext, plaintext_length + 16, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_plaintext));
    CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
    CHECK_CUDA_ERROR(cudaFree(d_key));
    CHECK_CUDA_ERROR(cudaFree(d_nonce));
    CHECK_CUDA_ERROR(cudaFree(d_associateddata));
}

// Optimized decryption function
void ascon_decrypt_gpu_optimized(uint8_t *plaintext, const uint8_t *key, const uint8_t *nonce,
                                 const uint8_t *associateddata, size_t adlen,
                                 const uint8_t *ciphertext, size_t ciphertext_length) {
    // Allocate GPU memory
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, ciphertext_length - 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, ciphertext_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_ciphertext, ciphertext, ciphertext_length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key, key, 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
    if (adlen > 0) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice));
    }

    // Launch optimized kernel (reuse encryption kernel logic with adjustments)
    int num_blocks = (ciphertext_length - 16 + RATE - 1) / RATE;
    int thread_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (thread_blocks > MAX_GRID_SIZE) {
        thread_blocks = MAX_GRID_SIZE;
    }
    ascon_encrypt_kernel_optimized<<<thread_blocks, BLOCK_SIZE>>>(d_ciphertext, d_plaintext, 
                                                                  d_key, d_nonce, d_associateddata, 
                                                                  adlen, ciphertext_length - 16);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy results back to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(plaintext, d_plaintext, ciphertext_length - 16, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_plaintext));
    CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
    CHECK_CUDA_ERROR(cudaFree(d_key));
    CHECK_CUDA_ERROR(cudaFree(d_nonce));
    CHECK_CUDA_ERROR(cudaFree(d_associateddata));
}

// Update demo function to use optimized encryption
void demo_ascon_gpu_optimized() {
    uint8_t key[16], nonce[16];
    uint8_t associateddata[] = "ASCON";
    uint8_t *plaintext;
    size_t plaintext_length;

    // Generate random key and nonce
    get_random_bytes(key, 16);
    get_random_bytes(nonce, 16);

    // Read plaintext from file
    read_plaintext_from_file("frame_0.txt", &plaintext, &plaintext_length);

    // Allocate memory for ciphertext
    uint8_t *ciphertext = (uint8_t *)malloc(plaintext_length + 16);

    // Measure encryption time
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    ascon_encrypt_gpu_optimized(ciphertext, key, nonce, associateddata, 
                                sizeof(associateddata) - 1, plaintext, plaintext_length);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float encryption_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&encryption_time, start, stop));

    // Write results to file
    FILE *output_file = fopen("ascon_full_gpu_optimized.txt", "w");
    if (!output_file) {
        printf("Error: Unable to open output file\n");
        free(plaintext);
        free(ciphertext);
        exit(EXIT_FAILURE);
    }

    demo_print(output_file, "key", key, 16);
    demo_print(output_file, "nonce", nonce, 16);
    demo_print(output_file, "plaintext", plaintext, plaintext_length);
    demo_print(output_file, "ciphertext", ciphertext, plaintext_length);
    fprintf(output_file, "Encryption time on GPU (optimized): %.3f ms\n", encryption_time);

    fclose(output_file);
    free(plaintext);
    free(ciphertext);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("Optimized GPU encryption completed in %.3f ms\n", encryption_time);
}

// Update demo function to use optimized decryption
void demo_ascon_gpu_decrypt_optimized() {
    uint8_t key[16], nonce[16];
    uint8_t associateddata[] = "ASCON";
    uint8_t *plaintext, *decrypted_plaintext;
    size_t plaintext_length;

    // Generate random key and nonce
    get_random_bytes(key, 16);
    get_random_bytes(nonce, 16);

    // Read plaintext from file
    read_plaintext_from_file("frame_0.txt", &plaintext, &plaintext_length);

    // Allocate memory for ciphertext and decrypted plaintext
    uint8_t *ciphertext = (uint8_t *)malloc(plaintext_length + 16);
    decrypted_plaintext = (uint8_t *)malloc(plaintext_length);

    // Encrypt plaintext
    ascon_encrypt_gpu_optimized(ciphertext, key, nonce, associateddata, 
                                sizeof(associateddata) - 1, plaintext, plaintext_length);

    // Decrypt ciphertext
    ascon_decrypt_gpu_optimized(decrypted_plaintext, key, nonce, associateddata, 
                                sizeof(associateddata) - 1, ciphertext, plaintext_length + 16);

    // Write results to file
    FILE *output_file = fopen("ascon_decrypt_gpu_optimized.txt", "w");
    if (!output_file) {
        printf("Error: Unable to open output file\n");
        free(plaintext);
        free(ciphertext);
        free(decrypted_plaintext);
        exit(EXIT_FAILURE);
    }

    demo_print(output_file, "key", key, 16);
    demo_print(output_file, "nonce", nonce, 16);
    demo_print(output_file, "plaintext", plaintext, plaintext_length);
    demo_print(output_file, "ciphertext", ciphertext, plaintext_length + 16);
    demo_print(output_file, "decrypted plaintext", decrypted_plaintext, plaintext_length);

    fclose(output_file);
    free(plaintext);
    free(ciphertext);
    free(decrypted_plaintext);

    printf("Optimized GPU decryption completed.\n");
}

// Update main function to call both encryption and decryption demos
int main() {
    // Kiểm tra CUDA device
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("Lỗi: Không tìm thấy thiết bị CUDA\n");
        return -1;
    }
    
    // In thông tin GPU
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Sử dụng GPU: %s\n", deviceProp.name);
    
    // Chạy demo mã hóa
    demo_ascon_gpu_optimized();

    // Chạy demo giải mã
    demo_ascon_gpu_decrypt_optimized();
    
    return 0;
}
