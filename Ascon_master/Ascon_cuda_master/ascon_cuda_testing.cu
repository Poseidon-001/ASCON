#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

// Cấu hình CUDA
#define BLOCK_SIZE 512          // Số thread trong một block
#define MAX_GRID_SIZE 65535     // Số block tối đa trong một grid
#define WARP_SIZE 32            // Kích thước warp
#define RATE 8                  // Rate của Ascon

// Debug flags
#define debug 0
#define STREAM_COUNT 2          // Số stream CUDA để chạy song song
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

// Kernel chính được tối ưu hóa cho GPU
__global__ void ascon_encrypt_kernel(const uint8_t *plaintext, uint8_t *ciphertext, 
                                    const uint8_t *key, const uint8_t *nonce,
                                    const uint8_t *associateddata, uint32_t adlen,
                                    uint32_t plaintext_length) {
    // Sử dụng shared memory cho key và nonce
    __shared__ uint8_t s_key[16];
    __shared__ uint8_t s_nonce[16];
    
    // Load key và nonce vào shared memory
    if (threadIdx.x < 16) {
        s_key[threadIdx.x] = key[threadIdx.x];
        s_nonce[threadIdx.x] = nonce[threadIdx.x];
    }
    __syncthreads();
    
    // Mỗi thread xử lý một block dữ liệu
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;
    uint32_t block_size = RATE;
    
    // Mỗi thread xử lý một tập hợp các block
    for (uint32_t block_index = tid; block_index < (plaintext_length + block_size - 1) / block_size; 
         block_index += total_threads) {
        
        uint32_t offset = block_index * block_size;
        if (offset >= plaintext_length) continue;
        
        // Khởi tạo trạng thái
        uint64_t S[5] = {0};
        
        // Khởi tạo state với IV + key + nonce
        ascon_initialize_device(S, s_key, s_nonce, 12, 8, RATE, 16, 1);
        
        // Xử lý associated data
        ascon_process_associated_data_device(S, associateddata, adlen, 8, RATE);
        
        // Tính số byte cần xử lý trong block này
        uint32_t bytes_to_process = (offset + block_size <= plaintext_length) ? 
                                  block_size : (plaintext_length - offset);
        
        // Tạo một block plaintext tạm
        uint8_t p_block[16]; // Giả sử block_size <= 16
        for (uint32_t i = 0; i < bytes_to_process; i++) {
            p_block[i] = plaintext[offset + i];
        }
        
        // Nếu là block cuối và cần padding
        if (offset + block_size > plaintext_length) {
            p_block[bytes_to_process] = 0x80;
            for (uint32_t i = bytes_to_process + 1; i < block_size; i++) {
                p_block[i] = 0;
            }
        }
        
        // Mã hóa block
        S[0] ^= bytes_to_int_device(p_block, bytes_to_process);
        
        // Ghi ciphertext
        int_to_bytes_device(ciphertext + offset, S[0], bytes_to_process);
        
        // Nếu không phải block cuối, áp dụng permutation
        if (offset + block_size < plaintext_length) {
            ascon_permutation_device(S, 8);
        }
        else {
            // Block cuối, tạo tag
            uint8_t tag[16];
            ascon_finalize_device(S, tag, s_key, 12, 16);
            
            // Ghi tag vào cuối ciphertext
            for (int i = 0; i < 16; i++) {
                ciphertext[plaintext_length + i] = tag[i];
            }
        }
    }
}

// Kernel giải mã
__global__ void ascon_decrypt_kernel(uint8_t *plaintext, const uint8_t *ciphertext, 
                                    const uint8_t *key, const uint8_t *nonce,
                                    const uint8_t *associateddata, uint32_t adlen,
                                    uint32_t ciphertext_length, uint8_t *tag_verification) {
    
    // Ciphertext thực sự (không bao gồm tag)
    uint32_t actual_ciphertext_length = ciphertext_length - 16;
    
    // Mỗi thread xử lý một block dữ liệu
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;
    uint32_t block_size = RATE;
    
    // Tag là 16 bytes cuối cùng
    uint8_t expected_tag[16];
    if (tid == 0) {
        for (int i = 0; i < 16; i++) {
            expected_tag[i] = ciphertext[actual_ciphertext_length + i];
        }
    }
    __syncthreads();
    
    // Mỗi thread xử lý một tập hợp các block
    for (uint32_t block_index = tid; block_index < (actual_ciphertext_length + block_size - 1) / block_size; 
         block_index += total_threads) {
        
        uint32_t offset = block_index * block_size;
        if (offset >= actual_ciphertext_length) continue;
        
        // Khởi tạo trạng thái
        uint64_t S[5] = {0};
        
        // Khởi tạo state với IV + key + nonce
        ascon_initialize_device(S, key, nonce, 12, 8, RATE, 16, 1);
        
        // Xử lý associated data
        ascon_process_associated_data_device(S, associateddata, adlen, 8, RATE);
        
        // Tính số byte cần xử lý trong block này
        uint32_t bytes_to_process = (offset + block_size <= actual_ciphertext_length) ? 
                                   block_size : (actual_ciphertext_length - offset);
        
        // Lấy block ciphertext
        uint8_t c_block[16]; // Giả sử block_size <= 16
        for (uint32_t i = 0; i < bytes_to_process; i++) {
            c_block[i] = ciphertext[offset + i];
        }
        
        // Nếu là block cuối và cần padding
        if (offset + block_size > actual_ciphertext_length) {
            c_block[bytes_to_process] = 0x80;
            for (uint32_t i = bytes_to_process + 1; i < block_size; i++) {
                c_block[i] = 0;
            }
        }
        
        // Lưu trữ ciphertext để cập nhật state sau
        uint64_t c_val = bytes_to_int_device(c_block, bytes_to_process);
        
        // Giải mã block
        int_to_bytes_device(plaintext + offset, S[0] ^ c_val, bytes_to_process);
        
        // Cập nhật state với ciphertext và áp dụng permutation
        S[0] = c_val;
        
        // Nếu không phải block cuối, áp dụng permutation
        if (offset + block_size < actual_ciphertext_length) {
            ascon_permutation_device(S, 8);
        }
        else if (tid == 0) { // Thread 0 tính toán tag và xác minh
            // Finalization
            uint8_t computed_tag[16];
            ascon_finalize_device(S, computed_tag, key, 12, 16);
            
            // Xác minh tag
            *tag_verification = 0;
            for (int i = 0; i < 16; i++) {
                if (computed_tag[i] != expected_tag[i]) {
                    *tag_verification = 1; // Tag không khớp
                    break;
                }
            }
        }
    }
}

// Hàm mã hóa trên GPU
void ascon_encrypt_gpu(uint8_t *ciphertext, const uint8_t *key, const uint8_t *nonce,
                     const uint8_t *associateddata, size_t adlen,
                     const uint8_t *plaintext, size_t plaintext_length) {
    
    // Cấp phát bộ nhớ trên GPU
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, plaintext_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, plaintext_length + 16)); // +16 cho tag
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
    
    // Sao chép dữ liệu từ CPU sang GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_plaintext, plaintext, plaintext_length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key, key, 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
    if (adlen > 0) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice));
    }
    
    // Tính số block và thread
    int num_blocks = (plaintext_length + RATE - 1) / RATE;
    int thread_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (thread_blocks > MAX_GRID_SIZE) {
        thread_blocks = MAX_GRID_SIZE;
    }
    
    // Chạy kernel
    ascon_encrypt_kernel<<<thread_blocks, BLOCK_SIZE>>>(d_plaintext, d_ciphertext, 
                                                      d_key, d_nonce, d_associateddata, 
                                                      adlen, plaintext_length);
    
    // Chờ kernel hoàn thành
    cudaDeviceSynchronize();
    
    // Kiểm tra lỗi
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Sao chép kết quả từ GPU về CPU
    CHECK_CUDA_ERROR(cudaMemcpy(ciphertext, d_ciphertext, plaintext_length + 16, cudaMemcpyDeviceToHost));
    
    // Giải phóng bộ nhớ
    CHECK_CUDA_ERROR(cudaFree(d_plaintext));
    CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
    CHECK_CUDA_ERROR(cudaFree(d_key));
    CHECK_CUDA_ERROR(cudaFree(d_nonce));
    CHECK_CUDA_ERROR(cudaFree(d_associateddata));
}

// Hàm giải mã trên GPU
int ascon_decrypt_gpu(uint8_t *plaintext, const uint8_t *key, const uint8_t *nonce,
                     const uint8_t *associateddata, size_t adlen,
                     const uint8_t *ciphertext, size_t ciphertext_length) {
    
    if (ciphertext_length < 16) {
        fprintf(stderr, "Lỗi: Ciphertext phải chứa ít nhất 16 bytes cho tag\n");
        return 0;
    }
    
    size_t actual_length = ciphertext_length - 16; // Trừ đi độ dài tag
    
    // Cấp phát bộ nhớ trên GPU
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata, *d_tag_verification;
    uint8_t tag_verification = 0;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, actual_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, ciphertext_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tag_verification, sizeof(uint8_t)));
    
    // Sao chép dữ liệu từ CPU sang GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_ciphertext, ciphertext, ciphertext_length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key, key, 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tag_verification, &tag_verification, sizeof(uint8_t), cudaMemcpyHostToDevice));
    if (adlen > 0) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice));
    }
    
    // Tính số block và thread
    int num_blocks = (actual_length + RATE - 1) / RATE;
    int thread_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (thread_blocks > MAX_GRID_SIZE) {
        thread_blocks = MAX_GRID_SIZE;
    }
    
    // Chạy kernel
    ascon_decrypt_kernel<<<thread_blocks, BLOCK_SIZE>>>(d_plaintext, d_ciphertext, 
                                                      d_key, d_nonce, d_associateddata, 
                                                      adlen, ciphertext_length, d_tag_verification);
    
    // Chờ kernel hoàn thành
    cudaDeviceSynchronize();
    
    // Kiểm tra lỗi
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Sao chép kết quả từ GPU về CPU
    CHECK_CUDA_ERROR(cudaMemcpy(plaintext, d_plaintext, actual_length, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&tag_verification, d_tag_verification, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    // Giải phóng bộ nhớ
    CHECK_CUDA_ERROR(cudaFree(d_plaintext));
    CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
    CHECK_CUDA_ERROR(cudaFree(d_key));
    CHECK_CUDA_ERROR(cudaFree(d_nonce));
    CHECK_CUDA_ERROR(cudaFree(d_associateddata));
    CHECK_CUDA_ERROR(cudaFree(d_tag_verification));
    
    // Trả về 1 nếu xác thực thành công, 0 nếu thất bại
    return (tag_verification == 0) ? 1 : 0;
}

// Thêm trước hàm demo_ascon_gpu
void ascon_encrypt_gpu_pipelined(uint8_t *ciphertext, const uint8_t *key, const uint8_t *nonce,
                              const uint8_t *associateddata, size_t adlen,
                              const uint8_t *plaintext, size_t plaintext_length) {
    
    // Cấu hình streams
    const int num_streams = STREAM_COUNT;
    cudaStream_t streams[STREAM_COUNT];
    
    // Khởi tạo các streams
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    // Cấp phát bộ nhớ trên GPU
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
    
    // Sử dụng pinned memory để truyền dữ liệu nhanh hơn
    #if USE_PINNED_MEMORY
    uint8_t *h_key, *h_nonce, *h_ad;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_key, 16));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_nonce, 16));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_ad, adlen > 0 ? adlen : 1));
    
    // Sao chép dữ liệu vào pinned memory
    memcpy(h_key, key, 16);
    memcpy(h_nonce, nonce, 16);
    if (adlen > 0) {
        memcpy(h_ad, associateddata, adlen);
    }
    #else
    const uint8_t *h_key = key;
    const uint8_t *h_nonce = nonce;
    const uint8_t *h_ad = associateddata;
    #endif
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
    
    // Sao chép key, nonce và associateddata (data nhỏ) một lần
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_key, h_key, 16, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_nonce, h_nonce, 16, cudaMemcpyHostToDevice, streams[0]));
    if (adlen > 0) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_associateddata, h_ad, adlen, cudaMemcpyHostToDevice, streams[0]));
    }
    
    // Tạo các event để theo dõi
    cudaEvent_t computeDone, transferDone;
    CHECK_CUDA_ERROR(cudaEventCreate(&computeDone));
    CHECK_CUDA_ERROR(cudaEventCreate(&transferDone));
    
    // Bắt đầu truyền dữ liệu
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_plaintext, plaintext, plaintext_length, 
                                   cudaMemcpyHostToDevice, streams[0]));
    
    // Đánh dấu khi truyền xong
    CHECK_CUDA_ERROR(cudaEventRecord(transferDone, streams[0]));
    
    // Đợi truyền xong trước khi tính toán
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(streams[1], transferDone, 0));
    
    // Bắt đầu tính toán
    ascon_encrypt_kernel<<<thread_blocks, BLOCK_SIZE, 0, streams[1]>>>(
        d_plaintext, d_ciphertext, 
        d_key, d_nonce, d_associateddata, 
        adlen, plaintext_length);
    
    // Đánh dấu khi tính toán xong
    CHECK_CUDA_ERROR(cudaEventRecord(computeDone, streams[1]));
    
    // Đợi tính toán xong trước khi truyền kết quả về
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(streams[0], computeDone, 0));
    
    // Truyền kết quả về
    CHECK_CUDA_ERROR(cudaMemcpyAsync(ciphertext, d_ciphertext, plaintext_length + 16,
                                   cudaMemcpyDeviceToHost, streams[0]));
    
    // Giải phóng bộ nhớ
    CHECK_CUDA_ERROR(cudaFree(d_plaintext));
    CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
    CHECK_CUDA_ERROR(cudaFree(d_key));
    CHECK_CUDA_ERROR(cudaFree(d_nonce));
    CHECK_CUDA_ERROR(cudaFree(d_associateddata));
    
    #if USE_PINNED_MEMORY
    CHECK_CUDA_ERROR(cudaFreeHost(h_key));
    CHECK_CUDA_ERROR(cudaFreeHost(h_nonce));
    CHECK_CUDA_ERROR(cudaFreeHost(h_ad));
    #endif
    
    // Hủy các streams
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}

// Tương tự cho hàm decrypt
int ascon_decrypt_gpu_pipelined(uint8_t *plaintext, const uint8_t *key, const uint8_t *nonce,
                             const uint8_t *associateddata, size_t adlen,
                             const uint8_t *ciphertext, size_t ciphertext_length) {
    
    if (ciphertext_length < 16) {
        fprintf(stderr, "Lỗi: Ciphertext phải chứa ít nhất 16 bytes cho tag\n");
        return 0;
    }
    
    size_t actual_length = ciphertext_length - 16; // Trừ đi độ dài tag
    
    // Cấu hình streams
    const int num_streams = STREAM_COUNT;
    cudaStream_t streams[STREAM_COUNT];
    
    // Khởi tạo các streams
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    // Cấp phát bộ nhớ trên GPU
    uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
    uint8_t *d_tag_verification;
    uint8_t tag_verification = 0;
    
    #if USE_PINNED_MEMORY
    // Sử dụng page-locked memory cho toàn bộ dữ liệu
    uint8_t *h_plaintext, *h_ciphertext;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_plaintext, plaintext_length));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_ciphertext, plaintext_length + 16));
    
    // Sao chép dữ liệu vào pinned memory
    memcpy(h_plaintext, plaintext, plaintext_length);
    #endif
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, actual_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, ciphertext_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tag_verification, sizeof(uint8_t)));
    
    // Sao chép key, nonce, và tag_verification
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_key, key, 16, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_nonce, nonce, 16, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_tag_verification, &tag_verification, sizeof(uint8_t), 
                                  cudaMemcpyHostToDevice, streams[0]));
    if (adlen > 0) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_associateddata, associateddata, adlen, 
                                      cudaMemcpyHostToDevice, streams[0]));
    }
    
    // Tạo các event để theo dõi
    cudaEvent_t computeDone, transferDone;
    CHECK_CUDA_ERROR(cudaEventCreate(&computeDone));
    CHECK_CUDA_ERROR(cudaEventCreate(&transferDone));
    
    // Bắt đầu truyền dữ liệu
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_plaintext, plaintext, plaintext_length, 
                                   cudaMemcpyHostToDevice, streams[0]));
    
    // Đánh dấu khi truyền xong
    CHECK_CUDA_ERROR(cudaEventRecord(transferDone, streams[0]));
    
    // Đợi truyền xong trước khi tính toán
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(streams[1], transferDone, 0));
    
    // Bắt đầu tính toán
    ascon_encrypt_kernel<<<thread_blocks, BLOCK_SIZE, 0, streams[1]>>>(
        d_plaintext, d_ciphertext, 
        d_key, d_nonce, d_associateddata, 
        adlen, plaintext_length);
    
    // Đánh dấu khi tính toán xong
    CHECK_CUDA_ERROR(cudaEventRecord(computeDone, streams[1]));
    
    // Đợi tính toán xong trước khi truyền kết quả về
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(streams[0], computeDone, 0));
    
    // Truyền kết quả về
    CHECK_CUDA_ERROR(cudaMemcpyAsync(ciphertext, d_ciphertext, plaintext_length + 16,
                                   cudaMemcpyDeviceToHost, streams[0]));
    
    // Giải phóng bộ nhớ
    CHECK_CUDA_ERROR(cudaFree(d_plaintext));
    CHECK_CUDA_ERROR(cudaFree(d_ciphertext));
    CHECK_CUDA_ERROR(cudaFree(d_key));
    CHECK_CUDA_ERROR(cudaFree(d_nonce));
    CHECK_CUDA_ERROR(cudaFree(d_associateddata));
    CHECK_CUDA_ERROR(cudaFree(d_tag_verification));
    
    #if USE_PINNED_MEMORY
    // Giải phóng pinned memory
    CHECK_CUDA_ERROR(cudaFreeHost(h_plaintext));
    CHECK_CUDA_ERROR(cudaFreeHost(h_ciphertext));
    #endif
    
    // Hủy các streams
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
    
    // Trả về 1 nếu xác thực thành công, 0 nếu thất bại
    return (tag_verification == 0) ? 1 : 0;
}

// Thay đổi hàm demo để sử dụng phiên bản pipelined
void demo_ascon_gpu_pipelined() {
    uint8_t key[16], nonce[16];
    uint8_t associateddata[] = "ASCON";
    uint8_t *plaintext;
    size_t plaintext_length;
    
    // Tạo key và nonce ngẫu nhiên
    get_random_bytes(key, 16);
    get_random_bytes(nonce, 16);
    
    // Đọc plaintext từ file
    read_plaintext_from_file("frame_0.txt", &plaintext, &plaintext_length);
    
    // Cấp phát bộ nhớ cho ciphertext
    uint8_t *ciphertext = (uint8_t *)malloc(plaintext_length + 16);
    uint8_t *decrypted = (uint8_t *)malloc(plaintext_length);
    
    // Đo thời gian mã hóa
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Khởi động GPU
    ascon_encrypt_gpu_pipelined(ciphertext, key, nonce, associateddata, 
                              sizeof(associateddata) - 1, plaintext, plaintext_length);
    
    // Đo thời gian mã hóa
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    ascon_encrypt_gpu_pipelined(ciphertext, key, nonce, associateddata, 
                              sizeof(associateddata) - 1, plaintext, plaintext_length);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float encryption_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&encryption_time, start, stop));
    
    // Đo thời gian giải mã
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    int decrypt_success = ascon_decrypt_gpu_pipelined(decrypted, key, nonce, associateddata, 
                                                   sizeof(associateddata) - 1, ciphertext, 
                                                   plaintext_length + 16);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float decryption_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&decryption_time, start, stop));
    
    // Ghi kết quả
    FILE *output_file = fopen("ascon_pipeline_gpu.txt", "w");
    if (!output_file) {
        printf("Lỗi: Không thể mở file đầu ra\n");
        free(plaintext);
        free(ciphertext);
        free(decrypted);
        exit(EXIT_FAILURE);
    }
    
    // Ghi thông tin
    demo_print(output_file, "key", key, 16);
    demo_print(output_file, "nonce", nonce, 16);
    demo_print(output_file, "plaintext", plaintext, plaintext_length);
    demo_print(output_file, "ass.data", associateddata, sizeof(associateddata) - 1);
    demo_print(output_file, "ciphertext", ciphertext, plaintext_length);
    demo_print(output_file, "tag", ciphertext + plaintext_length, 16);
    demo_print(output_file, "decrypted", decrypted, plaintext_length);
    
    // Kiểm tra xem giải mã có đúng không
    int is_correct = 1;
    for (size_t i = 0; i < plaintext_length; i++) {
        if (plaintext[i] != decrypted[i]) {
            is_correct = 0;
            break;
        }
    }
    
    // Ghi thông tin hiệu suất
    fprintf(output_file, "\n=== Performance Metrics ===\n");
    fprintf(output_file, "Encryption time on GPU: %.3f ms\n", encryption_time);
    fprintf(output_file, "Decryption time on GPU: %.3f ms\n", decryption_time);
    fprintf(output_file, "Total time: %.3f ms\n", encryption_time + decryption_time);
    fprintf(output_file, "Data size: %zu bytes\n", plaintext_length);
    fprintf(output_file, "Encryption throughput: %.2f MB/s\n", 
            (plaintext_length / (encryption_time / 1000.0)) / (1024.0 * 1024.0));
    fprintf(output_file, "Decryption throughput: %.2f MB/s\n", 
            (plaintext_length / (decryption_time / 1000.0)) / (1024.0 * 1024.0));
    fprintf(output_file, "GPU Configuration: %d threads/block\n", BLOCK_SIZE);
    fprintf(output_file, "Decryption successful: %s\n", decrypt_success ? "Yes" : "No");
    fprintf(output_file, "Decrypted data matches original: %s\n", is_correct ? "Yes" : "No");
    
    // Ghi thêm thông tin về pipeline
    fprintf(output_file, "Pipeline configuration: %d streams\n", STREAM_COUNT);
    fprintf(output_file, "Pinned memory: %s\n", USE_PINNED_MEMORY ? "Enabled" : "Disabled");
    
    // Giải phóng bộ nhớ
    fclose(output_file);
    free(plaintext);
    free(ciphertext);
    free(decrypted);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    printf("Mã hóa GPU hoàn tất trong %.3f ms\n", encryption_time);
    printf("Giải mã GPU hoàn tất trong %.3f ms\n", decryption_time);
    printf("Tổng thời gian: %.3f ms\n", encryption_time + decryption_time);
    printf("Giải mã thành công: %s\n", decrypt_success ? "Có" : "Không");
    printf("Kết quả đã lưu vào ascon_pipeline_gpu.txt\n");
}

// Hàm main
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
    
    // Thiết lập cấu hình GPU
    printf("Cấu hình: %d threads/block, %d streams, %s pinned memory\n", 
          BLOCK_SIZE, STREAM_COUNT, USE_PINNED_MEMORY ? "có" : "không");
    
    // Thiết lập giới hạn shared memory
    CHECK_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024));
    
    // Thiết lập cache preference
    CHECK_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    
    // Chạy benchmark để so sánh hiệu suất
    demo_ascon_gpu_pipelined();
    
    return 0;
} 
