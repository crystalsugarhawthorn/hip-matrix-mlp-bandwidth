#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <omp.h> // OpenMP 支持

#define BATCH 1024 // 批处理大小
#define I 10       // 输入层神经元数量
#define H 20       // 隐藏层神经元数量
#define O 5        // 输出层神经元数量
#define TILE_SIZE 16 // 块划分大小

// 编译指令：
// hipcc -fopenmp -o mlp main.cpp
// 执行指令：
// ./mlp 或者 hipprof ./mlp

// 矩阵乘法核函数：使用块划分（Block Tiling）优化
__global__ void matmul_tiled_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ReLU 激活函数核函数：将输入矩阵中的负值置为 0
__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
}

// 随机初始化矩阵或向量，值范围为 [-1, 1]
void random_init(std::vector<double>& mat) {
    #pragma omp parallel for // 使用 OpenMP 并行化初始化
    for (int i = 0; i < mat.size(); ++i) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
}

int main() {
    // 主机端（CPU）数据分配
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_Y(BATCH * O);

    // 随机初始化输入、权重和偏置
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // 设备端（GPU）数据指针
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;

    // 一次性分配设备端内存
    size_t total_memory = BATCH * I * sizeof(double) + I * H * sizeof(double) + H * sizeof(double) +
                          BATCH * H * sizeof(double) + H * O * sizeof(double) + O * sizeof(double) +
                          BATCH * O * sizeof(double);
    double* d_memory_pool;
    hipMalloc(&d_memory_pool, total_memory);

    // 内存池分配
    d_X = d_memory_pool;
    d_W1 = d_X + BATCH * I;
    d_B1 = d_W1 + I * H;
    d_H = d_B1 + H;
    d_W2 = d_H + BATCH * H;
    d_B2 = d_W2 + H * O;
    d_Y = d_B2 + O;

    // 一次性传输数据到设备端
    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    // 隐藏层计算：H = X * W1
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((H + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<gridDim, blockDim>>>(d_X, d_W1, d_H, BATCH, H, I);

    // 添加偏置并应用 ReLU 激活函数
    int hidden_size = BATCH * H;
    dim3 reluGrid((hidden_size + 255) / 256);
    relu_kernel<<<reluGrid, 256>>>(d_H, hidden_size);

    // 输出层计算：Y = H * W2
    gridDim = dim3((O + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<gridDim, blockDim>>>(d_H, d_W2, d_Y, BATCH, O, H);

    // 添加输出层偏置
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);
    #pragma omp parallel for simd // 使用 OpenMP 和 SIMD 优化偏置加法
    for (int i = 0; i < BATCH; ++i) {
        for (int j = 0; j < O; ++j) {
            h_Y[i * O + j] += h_B2[j];
        }
    }

    // 打印部分输出结果
    for (int i = 0; i < 5; ++i) {
        std::cout << "Output[" << i << "]: ";
        for (int j = 0; j < O; ++j)
            std::cout << h_Y[i * O + j] << " ";
        std::cout << std::endl;
    }

    // 释放设备端内存
    hipFree(d_memory_pool);

    return 0;
}
