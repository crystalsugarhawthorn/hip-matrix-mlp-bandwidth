#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include <chrono> // 增加计时库

// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
//  mpic++ -fopenmp -o outputfile question1.cpp

// 运行 baseline
// ./outputfile baseline

// 运行 OpenMP
// ./outputfile openmp

// 运行 子块并行优化
// ./outputfile block

// 运行 MPI（假设 4 个进程）
// mpirun -np 4 ./outputfile mpi --allow-run-as-root --allow-run-as-root-confirm

// 运行 MPI（假设 4 个进程）
// ./outputfile other

// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
    std::cout << "matmul_openmp methods..." << std::endl;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size) {
    std::cout << "matmul_block_tiling methods..." << std::endl;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < P; j += block_size) {
            for (int k = 0; k < M; k += block_size) {
                int i_max = std::min(i + block_size, N);
                int j_max = std::min(j + block_size, P);
                int k_max = std::min(k + block_size, M);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int kk = k; kk < k_max; ++kk) {
                            C[ii * P + jj] += A[ii * M + kk] * B[kk * P + jj];
                        }
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0) {
        std::cout << "matmul_mpi methods..." << std::endl;
    }
    double start_time = MPI_Wtime(); // 开始计时
    
    // 计算每个进程负责的行数
    int base = N / size;
    int rest = N % size;
    int local_N = base + (rank < rest ? 1 : 0);

    // 计算散发和收集数据的偏移量
    std::vector<int> sendcounts(size), displs(size);
    for (int i = 0, offset = 0; i < size; ++i) {
        sendcounts[i] = (base + (i < rest ? 1 : 0)) * M;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    std::vector<double> A_local((base + (rank < rest ? 1 : 0)) * M);
    std::vector<double> B(M * P);
    std::vector<double> C_local((base + (rank < rest ? 1 : 0)) * P, 0);
    
    if (rank == 0) {
        std::vector<double> A(N * M), B_full(M * P);
        // 初始化矩阵A和B
        init_matrix(A, N, M);
        init_matrix(B_full, M, P);
        B = B_full;
        // Scatter A的行数据到各个进程
        MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     A_local.data(), A_local.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     A_local.data(), A_local.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    // 所有进程共同调用B的广播
    MPI_Bcast(B.data(), B.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 局部矩阵乘法：A_local * B = C_local
    for (int i = 0; i < local_N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k)
                sum += A_local[i * M + k] * B[k * P + j];
            C_local[i * P + j] = sum;
        }
    
    // 添加MPI Barrier确保所有进程完成局部计算
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector<int> recvcounts(size), displsC(size);
    for (int i = 0, offset = 0; i < size; ++i) {
        int rows = base + (i < rest ? 1 : 0);
        recvcounts[i] = rows * P;
        displsC[i] = offset;
        offset += rows * P;
    }
    if (rank == 0) {
        std::vector<double> C(N * P);
        MPI_Gatherv(C_local.data(), C_local.size(), MPI_DOUBLE,
                    C.data(), recvcounts.data(), displsC.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        double end_time = MPI_Wtime(); // 结束计时
        std::cout << "[MPI] Done. Time: " << (end_time - start_time) << " s" << std::endl;
        std::cout.flush();
    } else {
        MPI_Gatherv(C_local.data(), C_local.size(), MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::cout.flush();
    }
}

// 方式4: 其他方式 （主要修改函数）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    std::cout << "Other methods..." << std::endl;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        matmul_mpi(N, M, P);
        MPI_Finalize();
        return 0;
    }

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);
    // 计算baseline结果用于验证
    matmul_baseline(A, B, C_ref, N, M, P);

    if (mode == "baseline") {
        auto start = std::chrono::high_resolution_clock::now();
        // 用C存储baseline计算结果
        matmul_baseline(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[Baseline] Done. Time: " << diff.count() << " s\n";
    } else if (mode == "openmp") {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P)
                  << " Time: " << diff.count() << " s\n";
    } else if (mode == "block") {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_block_tiling(A, B, C, N, M, P, 64);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P)
                  << " Time: " << diff.count() << " s\n";
    } else if (mode == "other") {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P)
                  << " Time: " << diff.count() << " s\n";
    } else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi]" << std::endl;
    }
        // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}