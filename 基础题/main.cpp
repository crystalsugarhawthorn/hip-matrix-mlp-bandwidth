#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <hip/hip_runtime.h>
#include <fstream>  // 新增

//hipcc -fopenmp -O2 -o outputfile_compare main.cpp

// 定义矩阵尺寸和tile尺寸
#define N 1024
#define M 2048
#define P 512
#define TILE_SIZE 16

// ...existing code for CPU methods...

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (auto& x : mat)
        x = dist(gen);
}

// CPU baseline实现
void matmul_baseline(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

// OpenMP方法
void matmul_openmp(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

// 子块（Block Tiling）方法
void matmul_block_tiling(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int block_size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < P; j += block_size) {
            for (int k = 0; k < M; k += block_size) {
                int i_max = std::min(i + block_size, N);
                int j_max = std::min(j + block_size, P);
                int k_max = std::min(k + block_size, M);
                for (int ii = i; ii < i_max; ++ii)
                    for (int jj = j; jj < j_max; ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < k_max; ++kk)
                            sum += A[ii * M + kk] * B[kk * P + jj];
                        C[ii * P + jj] += sum;
                    }
            }
        }
}

// 其他方式：利用OpenMP simd
void matmul_other(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

// 验证函数
bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

// ---------------------- 下面为DCU（HIP）部分 ----------------------

// HIP kernel：采用共享内存分块算法
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

// DCU方法，返回kernel执行时间（ms）
float matmul_dcu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    size_t bytes_A = N * M * sizeof(double);
    size_t bytes_B = M * P * sizeof(double);
    size_t bytes_C = N * P * sizeof(double);
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes_A);
    hipMalloc(&d_B, bytes_B);
    hipMalloc(&d_C, bytes_C);
    
    hipMemcpy(d_A, A.data(), bytes_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), bytes_B, hipMemcpyHostToDevice);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((P + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);
    
    hipLaunchKernelGGL(matmul_kernel, blocks, threads, 0, 0, d_A, d_B, d_C, N, M, P);
    
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    float elapsed;
    hipEventElapsedTime(&elapsed, start, stop);
    
    hipMemcpy(C.data(), d_C, bytes_C, hipMemcpyDeviceToHost);
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    return elapsed;
}

// ---------------------- 主函数 ----------------------
int main() {
    // 为各方法分配矩阵并初始化，同一数据用于所有方法
    std::vector<double> A_cpu(N * M), B_cpu(M * P);
    init_matrix(A_cpu);
    init_matrix(B_cpu);
    std::vector<double> C_baseline(N * P, 0), C_openmp(N * P, 0);
    std::vector<double> C_block(N * P, 0), C_other(N * P, 0);
    std::vector<double> C_dcu(N * P, 0), C_ref(N * P, 0);
    
    // CPU baseline
    auto t1 = std::chrono::high_resolution_clock::now();
    matmul_baseline(A_cpu, B_cpu, C_baseline);
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_baseline = std::chrono::duration<double, std::milli>(t2 - t1).count();
    C_ref = C_baseline;  // 用CPU baseline作为参照
    
    // OpenMP
    t1 = std::chrono::high_resolution_clock::now();
    matmul_openmp(A_cpu, B_cpu, C_openmp);
    t2 = std::chrono::high_resolution_clock::now();
    double time_openmp = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Block tiling, 注意：C_block需先置零
    std::fill(C_block.begin(), C_block.end(), 0);
    t1 = std::chrono::high_resolution_clock::now();
    matmul_block_tiling(A_cpu, B_cpu, C_block, 64);
    t2 = std::chrono::high_resolution_clock::now();
    double time_block = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Other (simd)
    t1 = std::chrono::high_resolution_clock::now();
    matmul_other(A_cpu, B_cpu, C_other);
    t2 = std::chrono::high_resolution_clock::now();
    double time_other = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // DCU方法
    float time_dcu = matmul_dcu(A_cpu, B_cpu, C_dcu);
    
    // 计算加速比（baseline作为1倍）
    double speedup_openmp = time_baseline / time_openmp;
    double speedup_block  = time_baseline / time_block;
    double speedup_other  = time_baseline / time_other;
    double speedup_dcu    = time_baseline / time_dcu;
    
    // 输出性能数据（CSV格式）
    std::ofstream ofs("d:\\南开\\南开\\计算机组成原理\\先导杯\\基础题\\perform.csv");
    ofs << "Method,Time(ms),Speedup" << std::endl;
    ofs << "Baseline," << time_baseline << ",1" << std::endl;
    ofs << "OpenMP," << time_openmp << "," << speedup_openmp << std::endl;
    ofs << "BlockTiling," << time_block << "," << speedup_block << std::endl;
    ofs << "Other_SIMD," << time_other << "," << speedup_other << std::endl;
    ofs << "DCU," << time_dcu << "," << speedup_dcu << std::endl;
    ofs.close();
    
    // 同时在终端输出
    std::cout << "Method,Time(ms),Speedup" << std::endl;
    std::cout << "Baseline," << time_baseline << ",1" << std::endl;
    std::cout << "OpenMP," << time_openmp << "," << speedup_openmp << std::endl;
    std::cout << "BlockTiling," << time_block << "," << speedup_block << std::endl;
    std::cout << "Other_SIMD," << time_other << "," << speedup_other << std::endl;
    std::cout << "DCU," << time_dcu << "," << speedup_dcu << std::endl;
    
    // 提示：利用rocm-smi、hipprof、hipgdb等工具可进一步分析DCU方法；
    // 生成的CSV数据可用柱状图和折线图展示CPU方法与DCU方法的性能对比。
    return 0;
}
