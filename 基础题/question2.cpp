#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>


// 编译
// hipcc question2.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N 1024
#define M 2024
#define P 512
#define TILE_SIZE 16  // 定义子块尺寸

// 优化后的kernel：采用共享内存分块算法
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double sum = 0.0;
    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < n && (t * TILE_SIZE + threadIdx.x) < m)
            As[threadIdx.y][threadIdx.x] = A[row * m + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
        
        if (col < p && (t * TILE_SIZE + threadIdx.y) < m)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * p + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    
    if (row < n && col < p)
        C[row * p + col] = sum;
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
    return;
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    return;
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    // CPU baseline
    matmul_cpu(A, B, C_ref);

    // 分配设备内存
    double *d_A, *d_B, *d_C;
    size_t bytes_A = N * M * sizeof(double);
    size_t bytes_B = M * P * sizeof(double);
    size_t bytes_C = N * P * sizeof(double);

    hipMalloc(&d_A, bytes_A);
    hipMalloc(&d_B, bytes_B);
    hipMalloc(&d_C, bytes_C);

    hipMemcpy(d_A, A.data(), bytes_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), bytes_B, hipMemcpyHostToDevice);

    // 设定kernel执行配置
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((P + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // 使用hipEvent进行计时
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);
    
    hipLaunchKernelGGL(matmul_kernel, blocks, threads, 0, 0, d_A, d_B, d_C, N, M, P);
    
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    float elapsed;
    hipEventElapsedTime(&elapsed, start, stop);
    std::cout << "[HIP] Kernel execution time: " << elapsed << " ms" << std::endl;
    
    hipMemcpy(C.data(), d_C, bytes_C, hipMemcpyDeviceToHost);
    
    if (validate(C_ref, C)) {
       std::cout << "[HIP] Valid: 1" << std::endl;
    } else {
       std::cout << "[HIP] Valid: 0" << std::endl;
    }
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    // 注：可通过以下工具对该优化核进行性能分析和调试：
    // 1. rocm-smi：命令如 "rocm-smi --showuse --showtemp" 用于监控GPU利用率、功耗及温度；
    // 2. hipprof：使用 "hipprof" 工具采集kernel执行时间和资源使用情况，可生成柱状图和折线图与CPU baseline进行对比；
    // 3. hipgdb：利用 "hipgdb ./outputfile_dcu" 对kernel进行调试，查找潜在瓶颈。
    // 利用上述工具可以直观展示和验证本次优化带来的性能提升。
    return 0;
}