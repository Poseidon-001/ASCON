#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <vector>

// Định nghĩa ASCON-128
#define BLOCK_SIZE 256
#define NUM_BLOCKS 47000 // ~47.000 khối 64-bit (frame 1080p, ~3MB)
#define STATE_SIZE 5 // Trạng thái 320-bit (5 x 64-bit)
#define IV 0x80800c0800000000ULL // IV cho ASCON-128
#define RATE 8 // 64-bit (8 byte)
#define TAG_SIZE 16 // 128-bit tag

// Hàm S-box 5-bit
__device__ void sbox(uint64_t *state) {
    state[0] ^= state[4]; state[4] ^= state[3]; state[2] ^= state[1];
    uint64_t t0 = state[0]; uint64_t t1 = state[1]; uint64_t t2 = state[2];
    state[0] = t0 ^ (~t1 & state[2]); state[1] = t1 ^ (~t2 & state[3]);
    state[2] = t2 ^ (~state[3] & state[4]); state[3] ^= (~state[4] & t0);
    state[4] ^= (~t0 & t1);
    state[2] = ~state[2];
}

// Hàm p^12
__device__ void p12(uint64_t *state, cudaEvent_t sbox_start, cudaEvent_t sbox_end) {
    const uint64_t round_constants[12] = {
        0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b
    };
    for (int i = 0; i < 12; i++) {
        // Thêm hằng số
        state[2] ^= round_constants[i];
        // S-box
        if (threadIdx.x == 0) cudaEventRecord(sbox_start);
        sbox(state);
        if (threadIdx.x == 0) cudaEventRecord(sbox_end);
        // Linear diffusion
        state[0] ^= (state[0] >> 19) ^ (state[0] >> 28);
        state[1] ^= (state[1] >> 61) ^ (state[1] >> 39);
        state[2] ^= (state[2] >> 1) ^ (state[2] >> 6);
        state[3] ^= (state[3] >> 7) ^ (state[3] >> 41);
        state[4] ^= (state[4] >> 17) ^ (state[4] >> 59);
    }
}

// Kernel mã hóa ASCON-128
__global__ void ascon_enc(uint64_t *input, uint64_t *output, uint64_t *state, uint64_t ad, uint64_t key, uint64_t nonce, uint64_t *tag,
                          int num_blocks, cudaEvent_t init_event, cudaEvent_t ad_event, cudaEvent_t enc_event,
                          cudaEvent_t p12_event, cudaEvent_t sbox_start, cudaEvent_t sbox_end, cudaEvent_t fin_event) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Khởi tạo
    if (threadIdx.x == 0) {
        cudaEventRecord(init_event);
        state[0] = IV;
        state[1] = key;
        state[2] = nonce;
        state[3] = 0;
        state[4] = 0;
        p12(state, sbox_start, sbox_end);
        state[4] ^= key;
    }
    __syncthreads();

    // Xử lý AD
    if (threadIdx.x == 0) {
        cudaEventRecord(ad_event);
        state[0] ^= ad;
        p12(state, sbox_start, sbox_end);
    }
    __syncthreads();

    // Mã hóa
    if (idx < num_blocks) {
        cudaEventRecord(enc_event);
        cudaEventRecord(p12_event);
        output[idx] = input[idx] ^ state[0];
        state[0] ^= input[idx];
        p12(state, sbox_start, sbox_end);
    }
    __syncthreads();

    // Hoàn tất
    if (threadIdx.x == 0) {
        cudaEventRecord(fin_event);
        state[0] ^= key;
        p12(state, sbox_start, sbox_end);
        tag[0] = state[0] ^ key;
    }
}

int main() {
    // Khởi tạo
    const int num_blocks = NUM_BLOCKS;
    const int threads_per_block = BLOCK_SIZE;
    const int grid_size = (num_blocks + threads_per_block - 1) / threads_per_block;
    const size_t data_size = num_blocks * sizeof(uint64_t);
    const int num_frames = 100;

    // Cấp phát bộ nhớ
    uint64_t *h_input, *h_output, *d_input, *d_output, *d_state, *d_tag;
    h_input = (uint64_t *)malloc(data_size);
    h_output = (uint64_t *)malloc(data_size);
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size);
    cudaMalloc(&d_state, STATE_SIZE * sizeof(uint64_t));
    cudaMalloc(&d_tag, sizeof(uint64_t));

    // Khởi tạo dữ liệu
    for (int i = 0; i < num_blocks; i++) {
        h_input[i] = i;
    }
    uint64_t key = 0x123456789ABCDEF0ULL;
    uint64_t nonce = 0xFEDCBA9876543210ULL;
    uint64_t ad = 0xAAAAAAAAAAAAAAAAULL;

    // File CSV
    std::ofstream log("cuda_metrics.csv");
    log << "Frame,Total_Kernel_ms,Init_us,AD_us,Encrypt_us,P12_us,Sbox_us,Finalize_us,Transfer_ms,Sync_us\n";

    // Lưu thời gian để tính biến động
    std::vector<float> total_times;

    // Đo 100 frame
    for (int frame = 0; frame < num_frames; frame++) {
        cudaEvent_t start, stop, init_event, ad_event, enc_event, p12_event, sbox_start, sbox_end, fin_event, sync_event;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&init_event);
        cudaEventCreate(&ad_event);
        cudaEventCreate(&enc_event);
        cudaEventCreate(&p12_event);
        cudaEventCreate(&sbox_start);
        cudaEventCreate(&sbox_end);
        cudaEventCreate(&fin_event);
        cudaEventCreate(&sync_event);

        // Đo truyền dữ liệu
        cudaEventRecord(start);
        cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float transfer_ms;
        cudaEventElapsedTime(&transfer_ms, start, stop);

        // Đo tổng kernel
        cudaEventRecord(start);
        ascon_enc<<<grid_size, threads_per_block>>>(d_input, d_output, d_state, ad, key, nonce, d_tag, num_blocks,
                                                   init_event, ad_event, enc_event, p12_event, sbox_start, sbox_end, fin_event);
        cudaEventRecord(sync_event);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Lấy thời gian
        float total_ms, init_us, ad_us, enc_us, p12_us, sbox_us, fin_us, sync_us;
        cudaEventElapsedTime(&total_ms, start, sync_event);
        cudaEventElapsedTime(&sync_us, sync_event, stop);
        cudaEventElapsedTime(&init_us, init_event, ad_event);
        cudaEventElapsedTime(&ad_us, ad_event, enc_event);
        cudaEventElapsedTime(&enc_us, enc_event, p12_event);
        cudaEventElapsedTime(&p12_us, p12_event, sbox_start);
        cudaEventElapsedTime(&sbox_us, sbox_start, sbox_end);
        cudaEventElapsedTime(&fin_us, fin_event, stop);

        // Lưu thời gian
        total_times.push_back(total_ms);

        // Ghi log
        log << frame << "," << total_ms << "," << init_us << "," << ad_us << ","
            << enc_us << "," << p12_us << "," << sbox_us << "," << fin_us << ","
            << transfer_ms << "," << sync_us << "\n";

        // Giải phóng sự kiện
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(init_event);
        cudaEventDestroy(ad_event);
        cudaEventDestroy(enc_event);
        cudaEventDestroy(p12_event);
        cudaEventDestroy(sbox_start);
        cudaEventDestroy(sbox_end);
        cudaEventDestroy(fin_event);
        cudaEventDestroy(sync_event);
    }

    // Tính biến động
    float avg_ms = 0;
    for (float t : total_times) avg_ms += t;
    avg_ms /= num_frames;
    float std_ms = 0;
    for (float t : total_times) std_ms += (t - avg_ms) * (t - avg_ms);
    std_ms = sqrt(std_ms / num_frames);

    // Ghi biến động
    log << "Average_ms," << avg_ms << "\n";
    log << "Std_Dev_ms," << std_ms << "\n";

    // Giải phóng
    log.close();
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_state);
    cudaFree(d_tag);
    free(h_input);
    free(h_output);

    return 0;
}
