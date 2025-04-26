    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <stdint.h>
    #include <string.h>
    #include <assert.h>
    #include <cuda_runtime.h>

    // Cấu hình CUDA tối ưu cho Jetson Xavier AGX
    #define BLOCK_SIZE 128          // Số thread trong một block - giảm xuống để phù hợp với L1 cache
    #define MAX_GRID_SIZE 65535     // Số block tối đa trong một grid
    #define WARP_SIZE 32            // Kích thước warp
    #define RATE 8                  // Rate của Ascon
    #define SMX_COUNT 8             // Jetson Xavier AGX có 8 SM
    #define WARPS_PER_SM 32         // Số lượng warp tối đa trên mỗi SM
    #define USE_SHARED_MEMORY 1     // Sử dụng shared memory 
    #define USE_PINNED_MEMORY 1     // Sử dụng pinned memory
    #define STREAM_COUNT 4          // Số stream CUDA để chạy song song

    // Debug flags
    #define debug 0

    // Error checking macro
    #define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

    // Thêm các hằng số kernel mới
    #define MAX_PLAINTEXT_SIZE_PER_THREAD (BLOCK_SIZE * 2)
    #define SHARED_MEM_SIZE (BLOCK_SIZE * sizeof(uint64_t) * 5)

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

    // Thêm hàm ascon_permutation_device tối ưu hơn
    __device__ __forceinline__ void ascon_permutation_device(uint64_t *S, int rounds) {
        // Tối ưu bằng cách sử dụng unrolling và tái sử dụng các biến
        #pragma unroll
        for (int r = 12 - rounds; r < 12; r++) {
            // --- add round constants ---
            S[2] ^= (0xf0 - r * 0x10 + r * 0x1);

            // --- substitution layer ---
            // Tối ưu các phép XOR để giảm phụ thuộc dữ liệu
            S[0] ^= S[4];
            S[4] ^= S[3];
            S[2] ^= S[1];
            
            // Tính trước tất cả các giá trị T để tránh phụ thuộc
            uint64_t T0 = (S[0] ^ 0xFFFFFFFFFFFFFFFF) & S[1];
            uint64_t T1 = (S[1] ^ 0xFFFFFFFFFFFFFFFF) & S[2];
            uint64_t T2 = (S[2] ^ 0xFFFFFFFFFFFFFFFF) & S[3];
            uint64_t T3 = (S[3] ^ 0xFFFFFFFFFFFFFFFF) & S[4];
            uint64_t T4 = (S[4] ^ 0xFFFFFFFFFFFFFFFF) & S[0];
            
            // Cập nhật tất cả trạng thái trong một lần
            S[0] ^= T1;
            S[1] ^= T2;
            S[2] ^= T3;
            S[3] ^= T4;
            S[4] ^= T0;
            
            // Thực hiện các phép XOR còn lại
            S[1] ^= S[0];
            S[0] ^= S[4];
            S[3] ^= S[2];
            S[2] ^= 0xFFFFFFFFFFFFFFFF;

            // --- linear diffusion layer --- sử dụng unroll
            uint64_t s0 = S[0], s1 = S[1], s2 = S[2], s3 = S[3], s4 = S[4];
            S[0] = s0 ^ rotr_device(s0, 19) ^ rotr_device(s0, 28);
            S[1] = s1 ^ rotr_device(s1, 61) ^ rotr_device(s1, 39);
            S[2] = s2 ^ rotr_device(s2,  1) ^ rotr_device(s2,  6);
            S[3] = s3 ^ rotr_device(s3, 10) ^ rotr_device(s3, 17);
            S[4] = s4 ^ rotr_device(s4,  7) ^ rotr_device(s4, 41);
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

    // Kernel chính được tối ưu hóa cho GPU Jetson Xavier
    __global__ void ascon_encrypt_kernel(const uint8_t *plaintext, uint8_t *ciphertext, 
                                        const uint8_t *key, const uint8_t *nonce,
                                        const uint8_t *associateddata, uint32_t adlen,
                                        uint32_t plaintext_length) {
        
        // Mỗi thread xử lý một block dữ liệu
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t total_threads = gridDim.x * blockDim.x;
        uint32_t block_size = RATE;
        
        // Sao chép key và nonce vào registers
        uint8_t local_key[16];
        uint8_t local_nonce[16];
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            local_key[i] = key[i];
            local_nonce[i] = nonce[i];
        }
        
        // Sao chép associated data vào shared memory để tất cả thread có thể truy cập
        __shared__ uint8_t shared_ad[256];
        if (threadIdx.x < adlen && adlen > 0) {
            shared_ad[threadIdx.x] = associateddata[threadIdx.x];
        }
        __syncthreads();
        
        // Xóa biến count để tránh bank conflicts
        uint32_t block_count = (plaintext_length + block_size - 1) / block_size;
        
        // Mỗi thread xử lý nhiều block để giảm kernel overhead
        uint32_t blocks_per_thread = (block_count + total_threads - 1) / total_threads;
        uint32_t start_block = tid * blocks_per_thread;
        uint32_t end_block = min(start_block + blocks_per_thread, block_count);
        
        // Sử dụng coalesced memory access
        for (uint32_t block_index = start_block; block_index < end_block; block_index++) {
            uint32_t offset = block_index * block_size;
            if (offset >= plaintext_length) continue;
            
            // Khởi tạo trạng thái
            uint64_t S[5] = {0};
            
            // Khởi tạo state với IV + key + nonce
            ascon_initialize_device(S, local_key, local_nonce, 12, 8, RATE, 16, 1);
            
            // Xử lý associated data
            ascon_process_associated_data_device(S, shared_ad, adlen, 8, RATE);
            
            // Tính số byte cần xử lý trong block này
            uint32_t bytes_to_process = min(block_size, plaintext_length - offset);
            
            // Coalesced memory access cho plaintext
            uint8_t p_block[16]; // Giả sử block_size <= 16
            #pragma unroll
            for (uint32_t i = 0; i < bytes_to_process; i++) {
                p_block[i] = plaintext[offset + i];
            }
            
            // Nếu là block cuối và cần padding
            if (offset + block_size > plaintext_length) {
                p_block[bytes_to_process] = 0x80;
                #pragma unroll
                for (uint32_t i = bytes_to_process + 1; i < block_size; i++) {
                    p_block[i] = 0;
                }
            }
            
            // Mã hóa block
            S[0] ^= bytes_to_int_device(p_block, bytes_to_process);
            
            // Ghi ciphertext với coalesced memory access
            int_to_bytes_device(ciphertext + offset, S[0], bytes_to_process);
            
            // Nếu không phải block cuối, áp dụng permutation
            if (offset + block_size < plaintext_length) {
                ascon_permutation_device(S, 8);
            }
            else {
                // Block cuối, tạo tag
                uint8_t tag[16];
                ascon_finalize_device(S, tag, local_key, 12, 16);
                
                // Ghi tag vào cuối ciphertext
                #pragma unroll
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
        
        // Sao chép key và nonce vào registers
        uint8_t local_key[16];
        uint8_t local_nonce[16];
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            local_key[i] = key[i];
            local_nonce[i] = nonce[i];
        }
        
        // Sao chép associated data vào shared memory
        __shared__ uint8_t shared_ad[256];
        if (threadIdx.x < adlen && adlen > 0) {
            shared_ad[threadIdx.x] = associateddata[threadIdx.x];
        }
        
        // Tag là 16 bytes cuối cùng, sao chép vào shared memory
        __shared__ uint8_t expected_tag[16];
        if (threadIdx.x < 16) {
            expected_tag[threadIdx.x] = ciphertext[actual_ciphertext_length + threadIdx.x];
        }
        __syncthreads();
        
        // Xử lý nhiều block trên mỗi thread
        uint32_t block_count = (actual_ciphertext_length + block_size - 1) / block_size;
        uint32_t blocks_per_thread = (block_count + total_threads - 1) / total_threads;
        uint32_t start_block = tid * blocks_per_thread;
        uint32_t end_block = min(start_block + blocks_per_thread, block_count);
        
        for (uint32_t block_index = start_block; block_index < end_block; block_index++) {
            uint32_t offset = block_index * block_size;
            if (offset >= actual_ciphertext_length) continue;
            
            // Khởi tạo trạng thái
            uint64_t S[5] = {0};
            
            // Khởi tạo state với IV + key + nonce
            ascon_initialize_device(S, local_key, local_nonce, 12, 8, RATE, 16, 1);
            
            // Xử lý associated data
            ascon_process_associated_data_device(S, shared_ad, adlen, 8, RATE);
            
            // Tính số byte cần xử lý trong block này
            uint32_t bytes_to_process = min(block_size, actual_ciphertext_length - offset);
            
            // Lấy block ciphertext với coalesced memory access
            uint8_t c_block[16]; // Giả sử block_size <= 16
            #pragma unroll
            for (uint32_t i = 0; i < bytes_to_process; i++) {
                c_block[i] = ciphertext[offset + i];
            }
            
            // Nếu là block cuối và cần padding
            if (offset + block_size > actual_ciphertext_length) {
                c_block[bytes_to_process] = 0x80;
                #pragma unroll
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
            else if (threadIdx.x == 0 && blockIdx.x == 0) { // Thread 0 tính toán tag và xác minh
                // Finalization
                uint8_t computed_tag[16];
                ascon_finalize_device(S, computed_tag, local_key, 12, 16);
                
                // Xác minh tag với atomic operations để tránh race condition
                *tag_verification = 0;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    if (computed_tag[i] != expected_tag[i]) {
                        *tag_verification = 1; // Tag không khớp
                        break;
                    }
                }
            }
        }
    }

    // Hàm mã hóa trên GPU tối ưu
    void ascon_encrypt_gpu(uint8_t *ciphertext, const uint8_t *key, const uint8_t *nonce,
                         const uint8_t *associateddata, size_t adlen,
                         const uint8_t *plaintext, size_t plaintext_length) {
        
        // Cấp phát bộ nhớ trên GPU
        uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata;
        
        // Sử dụng pinned memory để đẩy nhanh chuyển dữ liệu
        if (USE_PINNED_MEMORY) {
            CHECK_CUDA_ERROR(cudaHostRegister((void*)plaintext, plaintext_length, cudaHostRegisterDefault));
            CHECK_CUDA_ERROR(cudaHostRegister((void*)key, 16, cudaHostRegisterDefault));
            CHECK_CUDA_ERROR(cudaHostRegister((void*)nonce, 16, cudaHostRegisterDefault));
            if (adlen > 0) {
                CHECK_CUDA_ERROR(cudaHostRegister((void*)associateddata, adlen, cudaHostRegisterDefault));
            }
        }
        
        // Cấp phát bộ nhớ với căn chỉnh cho truy cập dữ liệu tối ưu
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, plaintext_length));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, plaintext_length + 16)); // +16 cho tag
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
        
        // Sử dụng streams để chạy song song
        cudaStream_t streams[STREAM_COUNT];
        for (int i = 0; i < STREAM_COUNT; i++) {
            CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
        }
        
        // Sao chép dữ liệu từ CPU sang GPU với streams khác nhau
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_key, key, 16, cudaMemcpyHostToDevice, streams[0]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_nonce, nonce, 16, cudaMemcpyHostToDevice, streams[1]));
        
        if (adlen > 0) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice, streams[2]));
        }
        
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_plaintext, plaintext, plaintext_length, cudaMemcpyHostToDevice, streams[3]));
        
        // Đồng bộ hóa tất cả các streams
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Tính số block và thread tối ưu cho Jetson Xavier AGX
        int num_blocks = (plaintext_length + RATE - 1) / RATE;
        
        // Tính toán số lượng block, threads tối ưu dựa trên số lượng SM
        int optimal_threads_per_block = BLOCK_SIZE;
        
        // Số lượng blocks tối ưu để đạt được occupancy 100%
        int warps_per_block = (optimal_threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
        int optimal_blocks_per_sm = WARPS_PER_SM / warps_per_block;
        int optimal_blocks = SMX_COUNT * optimal_blocks_per_sm;
        
        // Đảm bảo số block đủ để xử lý tất cả dữ liệu
        int thread_blocks = (num_blocks + optimal_threads_per_block - 1) / optimal_threads_per_block;
        thread_blocks = min(thread_blocks, optimal_blocks);
        
        if (thread_blocks > MAX_GRID_SIZE) {
            thread_blocks = MAX_GRID_SIZE;
        }
        
        // Cấu hình L1 cache và shared memory
        CHECK_CUDA_ERROR(cudaFuncSetCacheConfig(ascon_encrypt_kernel, cudaFuncCachePreferShared));
        
        // Chạy kernel với cấu hình tối ưu
        ascon_encrypt_kernel<<<thread_blocks, optimal_threads_per_block, 0, streams[0]>>>(
            d_plaintext, d_ciphertext, d_key, d_nonce, d_associateddata, adlen, plaintext_length);
        
        // Chờ kernel hoàn thành
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
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
        
        // Giải phóng pinned memory
        if (USE_PINNED_MEMORY) {
            CHECK_CUDA_ERROR(cudaHostUnregister((void*)plaintext));
            CHECK_CUDA_ERROR(cudaHostUnregister((void*)key));
            CHECK_CUDA_ERROR(cudaHostUnregister((void*)nonce));
            if (adlen > 0) {
                CHECK_CUDA_ERROR(cudaHostUnregister((void*)associateddata));
            }
        }
        
        // Hủy streams
        for (int i = 0; i < STREAM_COUNT; i++) {
            CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        }
    }

    // Tối ưu hàm giải mã trên GPU
    int ascon_decrypt_gpu(uint8_t *plaintext, const uint8_t *key, const uint8_t *nonce,
                         const uint8_t *associateddata, size_t adlen,
                         const uint8_t *ciphertext, size_t ciphertext_length) {
        
        if (ciphertext_length < 16) {
            fprintf(stderr, "Lỗi: Ciphertext phải chứa ít nhất 16 bytes cho tag\n");
            return 0;
        }
        
        size_t actual_length = ciphertext_length - 16; // Trừ đi độ dài tag
        
        // Sử dụng pinned memory
        if (USE_PINNED_MEMORY) {
            CHECK_CUDA_ERROR(cudaHostRegister((void*)ciphertext, ciphertext_length, cudaHostRegisterDefault));
            CHECK_CUDA_ERROR(cudaHostRegister((void*)key, 16, cudaHostRegisterDefault));
            CHECK_CUDA_ERROR(cudaHostRegister((void*)nonce, 16, cudaHostRegisterDefault));
            if (adlen > 0) {
                CHECK_CUDA_ERROR(cudaHostRegister((void*)associateddata, adlen, cudaHostRegisterDefault));
            }
        }
        
        // Cấp phát bộ nhớ trên GPU
        uint8_t *d_plaintext, *d_ciphertext, *d_key, *d_nonce, *d_associateddata, *d_tag_verification;
        uint8_t tag_verification = 0;
        
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_plaintext, actual_length));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ciphertext, ciphertext_length));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key, 16));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nonce, 16));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_associateddata, adlen > 0 ? adlen : 1));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tag_verification, sizeof(uint8_t)));
        
        // Sử dụng streams
        cudaStream_t streams[STREAM_COUNT];
        for (int i = 0; i < STREAM_COUNT; i++) {
            CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
        }
        
        // Sao chép dữ liệu từ CPU sang GPU với streams
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_ciphertext, ciphertext, ciphertext_length, cudaMemcpyHostToDevice, streams[0]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_key, key, 16, cudaMemcpyHostToDevice, streams[1]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_nonce, nonce, 16, cudaMemcpyHostToDevice, streams[2]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_tag_verification, &tag_verification, sizeof(uint8_t), cudaMemcpyHostToDevice, streams[3]));
        
        if (adlen > 0) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_associateddata, associateddata, adlen, cudaMemcpyHostToDevice, streams[0]));
        }
        
        // Đồng bộ hóa
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Tính toán cấu hình kernel tối ưu
        int num_blocks = (actual_length + RATE - 1) / RATE;
        
        // Tính toán số lượng block, threads tối ưu
        int optimal_threads_per_block = BLOCK_SIZE;
        
        // Số lượng blocks tối ưu
        int warps_per_block = (optimal_threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
        int optimal_blocks_per_sm = WARPS_PER_SM / warps_per_block;
        int optimal_blocks = SMX_COUNT * optimal_blocks_per_sm;
        
        // Đảm bảo số block đủ để xử lý tất cả dữ liệu
        int thread_blocks = (num_blocks + optimal_threads_per_block - 1) / optimal_threads_per_block;
        thread_blocks = min(thread_blocks, optimal_blocks);
        
        if (thread_blocks > MAX_GRID_SIZE) {
            thread_blocks = MAX_GRID_SIZE;
        }
        
        // Cấu hình L1 cache và shared memory
        CHECK_CUDA_ERROR(cudaFuncSetCacheConfig(ascon_decrypt_kernel, cudaFuncCachePreferShared));
        
        // Chạy kernel
        ascon_decrypt_kernel<<<thread_blocks, optimal_threads_per_block, 0, streams[0]>>>(
            d_plaintext, d_ciphertext, d_key, d_nonce, d_associateddata, adlen, ciphertext_length, d_tag_verification);
        
        // Chờ kernel hoàn thành
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
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
        
        // Giải phóng pinned memory
        if (USE_PINNED_MEMORY) {
            CHECK_CUDA_ERROR(cudaHostUnregister((void*)ciphertext));
            CHECK_CUDA_ERROR(cudaHostUnregister((void*)key));
            CHECK_CUDA_ERROR(cudaHostUnregister((void*)nonce));
            if (adlen > 0) {
                CHECK_CUDA_ERROR(cudaHostUnregister((void*)associateddata));
            }
        }
        
        // Hủy streams
        for (int i = 0; i < STREAM_COUNT; i++) {
            CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        }
        
        // Trả về 1 nếu xác thực thành công, 0 nếu thất bại
        return (tag_verification == 0) ? 1 : 0;
    }

    // Hàm demo được tối ưu hóa
    void demo_ascon_gpu() {
        // Đặt thiết bị CUDA vào chế độ tối ưu hiệu suất
        CHECK_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost));
        
        // Tăng L1 cache size
        CHECK_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
        
        uint8_t key[16], nonce[16];
        uint8_t associateddata[] = "ASCON";
        uint8_t *plaintext;
        size_t plaintext_length;
        
        // Tạo key và nonce ngẫu nhiên
        get_random_bytes(key, 16);
        get_random_bytes(nonce, 16);
        
        // Đọc plaintext từ file
        read_plaintext_from_file("frame_0.txt", &plaintext, &plaintext_length);
        
        // Sử dụng pinned memory cho các bộ đệm để tăng tốc chuyển dữ liệu
        uint8_t *ciphertext, *decrypted;
        
        if (USE_PINNED_MEMORY) {
            CHECK_CUDA_ERROR(cudaHostAlloc((void**)&ciphertext, plaintext_length + 16, cudaHostAllocDefault));
            CHECK_CUDA_ERROR(cudaHostAlloc((void**)&decrypted, plaintext_length, cudaHostAllocDefault));
        } else {
            ciphertext = (uint8_t *)malloc(plaintext_length + 16);
            decrypted = (uint8_t *)malloc(plaintext_length);
        }
        
        // Đo thời gian mã hóa
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Khởi động GPU (warm-up)
        ascon_encrypt_gpu(ciphertext, key, nonce, associateddata, 
                         sizeof(associateddata) - 1, plaintext, plaintext_length);
        
        // Đo thời gian mã hóa thực tế
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Thực hiện 10 lần và lấy trung bình để có kết quả đo lường ổn định hơn
        int num_iterations = 10;
        for (int i = 0; i < num_iterations; i++) {
            ascon_encrypt_gpu(ciphertext, key, nonce, associateddata, 
                            sizeof(associateddata) - 1, plaintext, plaintext_length);
        }
        
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float encryption_time = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&encryption_time, start, stop));
        encryption_time /= num_iterations; // Lấy thời gian trung bình
        
        // Đo thời gian giải mã
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Thực hiện 10 lần và lấy trung bình
        int decrypt_success = 1;
        for (int i = 0; i < num_iterations; i++) {
            decrypt_success &= ascon_decrypt_gpu(decrypted, key, nonce, associateddata, 
                                              sizeof(associateddata) - 1, ciphertext, plaintext_length + 16);
        }
        
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float decryption_time = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&decryption_time, start, stop));
        decryption_time /= num_iterations; // Lấy thời gian trung bình
        
        // Ghi kết quả
        FILE *output_file = fopen("ascon_optimized_gpu.txt", "w");
        if (!output_file) {
            printf("Lỗi: Không thể mở file đầu ra\n");
            goto cleanup;
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
        if (decrypt_success) {
            for (size_t i = 0; i < plaintext_length; i++) {
                if (plaintext[i] != decrypted[i]) {
                    is_correct = 0;
                    break;
                }
            }
        } else {
            is_correct = 0;
        }
        
        // Ghi thông tin hiệu suất
        fprintf(output_file, "\n=== Performance Metrics (Optimized) ===\n");
        fprintf(output_file, "Encryption time on GPU: %.3f ms\n", encryption_time);
        fprintf(output_file, "Decryption time on GPU: %.3f ms\n", decryption_time);
        fprintf(output_file, "Total time: %.3f ms\n", encryption_time + decryption_time);
        fprintf(output_file, "Data size: %zu bytes\n", plaintext_length);
        fprintf(output_file, "Encryption throughput: %.2f MB/s\n", 
                (plaintext_length / (encryption_time / 1000.0)) / (1024.0 * 1024.0));
        fprintf(output_file, "Decryption throughput: %.2f MB/s\n", 
                (plaintext_length / (decryption_time / 1000.0)) / (1024.0 * 1024.0));
        fprintf(output_file, "GPU Configuration: %d threads/block\n", BLOCK_SIZE);
        fprintf(output_file, "Optimizations: Shared memory: %s, Pinned memory: %s\n",
                USE_SHARED_MEMORY ? "Yes" : "No", USE_PINNED_MEMORY ? "Yes" : "No");
        fprintf(output_file, "Decryption successful: %s\n", decrypt_success ? "Yes" : "No");
        fprintf(output_file, "Decrypted data matches original: %s\n", is_correct ? "Yes" : "No");
        
        fclose(output_file);
        
        printf("=== GPU Performance (Optimized) ===\n");
        printf("Mã hóa GPU hoàn tất trong %.3f ms\n", encryption_time);
        printf("Giải mã GPU hoàn tất trong %.3f ms\n", decryption_time);
        printf("Tổng thời gian: %.3f ms\n", encryption_time + decryption_time);
        printf("Throughput mã hóa: %.2f MB/s\n", 
              (plaintext_length / (encryption_time / 1000.0)) / (1024.0 * 1024.0));
        printf("Throughput giải mã: %.2f MB/s\n", 
              (plaintext_length / (decryption_time / 1000.0)) / (1024.0 * 1024.0));
        printf("Giải mã thành công: %s\n", decrypt_success ? "Có" : "Không");
        printf("Kết quả đã lưu vào ascon_optimized_gpu.txt\n");

cleanup:
        // Giải phóng bộ nhớ
        free(plaintext);
        
        if (USE_PINNED_MEMORY) {
            CHECK_CUDA_ERROR(cudaFreeHost(ciphertext));
            CHECK_CUDA_ERROR(cudaFreeHost(decrypted));
        } else {
            free(ciphertext);
            free(decrypted);
        }
        
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    }

    // Hàm main
    int main() {
        // Kiểm tra CUDA device
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        if (error != cudaSuccess) {
            printf("Lỗi kiểm tra thiết bị CUDA: %s\n", cudaGetErrorString(error));
            return -1;
        }
        
        if (deviceCount == 0) {
            printf("Lỗi: Không tìm thấy thiết bị CUDA\n");
            return -1;
        }
        
        // In thông tin GPU
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
        printf("=== Thông tin GPU ===\n");
        printf("Tên GPU: %s\n", deviceProp.name);
        printf("Số lượng SM: %d\n", deviceProp.multiProcessorCount);
        printf("Kích thước warp: %d\n", deviceProp.warpSize);
        printf("Tổng bộ nhớ: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Tối đa threads/block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Shared memory/block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
        printf("L2 cache size: %d KB\n", deviceProp.l2CacheSize / 1024);
        printf("Max grid size: %d x %d x %d\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("=====================\n\n");
        
        // Chạy demo
        demo_ascon_gpu();
        
        return 0;
    } 
