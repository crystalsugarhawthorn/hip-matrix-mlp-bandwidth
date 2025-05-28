#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>

// 编译文件
// hipcc main.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数
#define INPUT_DIM 10
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1
#define BATCH_SIZE 256
#define EPOCHS 200
#define LEARNING_RATE 1e-4

// 添加偏置核函数
__global__ void add_bias(double* A, const double* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        A[idx] += bias[col];
    }
}

// 矩阵乘法核函数：计算 A^T * B
__global__ void matmul_transpose_A(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < K && col < N) {
        double sum = 0.0;
        #pragma unroll
        for (int m = 0; m < M; ++m)
            sum += A[m * K + row] * B[m * N + col];
        C[row * N + col] = sum;
    }
}

// 矩阵乘法核函数：计算 A * B^T
__global__ void matmul_transpose_B(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N) {
        double sum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[col * K + k];
        C[row * N + col] = sum;
    }
}

// ReLU 反向传播核函数
__global__ void compute_relu_backward(double* delta, const double* activ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        delta[idx] = activ[idx] > 0.0 ? delta[idx] : 0.0;
}

// 对矩阵每列求和，计算偏置梯度
__global__ void sum_bias(const double* matrix, double* bias_grad, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i)
            sum += matrix[i * cols + j];
        bias_grad[j] = sum;
    }
}

// 矩阵乘法核函数
__global__ void matmul(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N) {
        double sum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// 添加偏置并应用 ReLU 激活函数
__global__ void add_bias_and_relu(double* A, const double* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        A[idx] += bias[col];
        A[idx] = fmax(0.0, A[idx]);
    }
}

// 计算输出梯度
__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = pred[idx] - target[idx];
    }
}

// 计算均方误差损失
__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        atomicAdd(loss, diff * diff);
    }
}

// SGD 参数更新
__global__ void sgd_update(double* params, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        params[idx] -= lr * grad[idx];
}

// 数据加载和数据集生成辅助函数
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::ifstream f(filename);
    std::vector<double> data;
    if (f.is_open()) {
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        content.erase(remove(content.begin(), content.end(), '['), content.end());
        content.erase(remove(content.begin(), content.end(), ']'), content.end());
        std::istringstream iss(content);
        double num;
        char comma;
        while (iss >> num) {
            data.push_back(num);
            iss >> comma;
        }
    }
    return data;
}

void create_dataset(const std::vector<double>& data, std::vector<double>& X, std::vector<double>& y) {
    int num_samples = data.size() - INPUT_DIM;
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j)
            X.push_back(data[i + j]);
        y.push_back(data[i + INPUT_DIM]);
    }
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    return;
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
    return;
}

// ----------------------------- Main -------------------------------
int main() {
    // 加载数据并生成数据集
    std::string json_file = "./starlink_bw.json";
    auto bandwidth_data = load_json_bandwidth(json_file);
    double min_val, max_val;
    normalize_data(bandwidth_data, min_val, max_val);
    std::vector<double> X, y;
    create_dataset(bandwidth_data, X, y);
    int total_samples = y.size();
    int train_samples = static_cast<int>(total_samples * 0.8);
    int num_batches = train_samples / BATCH_SIZE;

    // 初始化权重和偏置
    std::default_random_engine eng;
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    std::vector<double> h_W1(INPUT_DIM * HIDDEN_DIM), h_b1(HIDDEN_DIM);
    std::vector<double> h_W2(HIDDEN_DIM * OUTPUT_DIM), h_b2(OUTPUT_DIM);
    for (auto& w : h_W1) w = dist(eng);
    for (auto& b : h_b1) b = dist(eng);
    for (auto& w : h_W2) w = dist(eng);
    for (auto& b : h_b2) b = dist(eng);

    // 设备端内存分配
    double *d_X, *d_hidden, *d_output, *d_W1, *d_b1, *d_W2, *d_b2, *d_error, *d_loss, *d_y;
    size_t size_X = BATCH_SIZE * INPUT_DIM * sizeof(double);
    size_t size_hidden = BATCH_SIZE * HIDDEN_DIM * sizeof(double);
    size_t size_output = BATCH_SIZE * OUTPUT_DIM * sizeof(double);
    size_t size_W1 = INPUT_DIM * HIDDEN_DIM * sizeof(double);
    size_t size_b1 = HIDDEN_DIM * sizeof(double);
    size_t size_W2 = HIDDEN_DIM * OUTPUT_DIM * sizeof(double);
    size_t size_b2 = OUTPUT_DIM * sizeof(double);
    size_t size_error = size_output;

    hipMalloc(&d_X, size_X);
    hipMalloc(&d_hidden, size_hidden);
    hipMalloc(&d_output, size_output);
    hipMalloc(&d_W1, size_W1);
    hipMalloc(&d_b1, size_b1);
    hipMalloc(&d_W2, size_W2);
    hipMalloc(&d_b2, size_b2);
    hipMalloc(&d_error, size_error);
    hipMalloc(&d_loss, sizeof(double));
    hipMalloc(&d_y, BATCH_SIZE * OUTPUT_DIM * sizeof(double));

    // 声明并分配缺失的设备变量
    double *d_grad_W2, *d_dA1, *d_grad_W1, *d_grad_b2, *d_grad_b1;

    size_t size_grad_W2 = HIDDEN_DIM * OUTPUT_DIM * sizeof(double);
    size_t size_grad_W1 = INPUT_DIM * HIDDEN_DIM * sizeof(double);
    size_t size_grad_b2 = OUTPUT_DIM * sizeof(double);
    size_t size_grad_b1 = HIDDEN_DIM * sizeof(double);
    size_t size_dA1 = BATCH_SIZE * HIDDEN_DIM * sizeof(double);

    hipMalloc(&d_grad_W2, size_grad_W2);
    hipMalloc(&d_grad_W1, size_grad_W1);
    hipMalloc(&d_grad_b2, size_grad_b2);
    hipMalloc(&d_grad_b1, size_grad_b1);
    hipMalloc(&d_dA1, size_dA1);

    // 创建 HIP 流
    hipStream_t stream;
    hipStreamCreate(&stream);

    // 声明 vecBlock
    dim3 vecBlock(256);

    // 训练过程
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss = 0.0;
        for (int b = 0; b < num_batches; ++b) {
            // 异步拷贝本批次输入数据到设备
            hipMemcpyAsync(d_X, X.data() + b * BATCH_SIZE * INPUT_DIM, size_X, hipMemcpyHostToDevice, stream);

            // 前向传播：隐含层 = ReLU( X * W1 + b1 )
            dim3 launchGrid1((HIDDEN_DIM + 15) / 16, (BATCH_SIZE + 15) / 16);
            matmul<<<launchGrid1, dim3(16, 16), 0, stream>>>(d_X, d_W1, d_hidden, BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);
            dim3 launchVecGrid1((BATCH_SIZE * HIDDEN_DIM + 255) / 256);
            add_bias_and_relu<<<launchVecGrid1, dim3(256), 0, stream>>>(d_hidden, d_b1, BATCH_SIZE, HIDDEN_DIM);

            // 前向传播：输出层 = 隐含层 * W2 + b2
            dim3 launchGrid2((OUTPUT_DIM + 15) / 16, (BATCH_SIZE + 15) / 16);
            matmul<<<launchGrid2, dim3(16, 16), 0, stream>>>(d_hidden, d_W2, d_output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);
            dim3 launchVecGrid2((BATCH_SIZE * OUTPUT_DIM + 255) / 256);
            add_bias<<<launchVecGrid2, dim3(256), 0, stream>>>(d_output, d_b2, BATCH_SIZE, OUTPUT_DIM);

            // 将本批次目标值拷贝到已分配的 d_y（异步拷贝）
            std::vector<double> batch_y(BATCH_SIZE * OUTPUT_DIM);
            memcpy(batch_y.data(), y.data() + b * BATCH_SIZE, size_output);
            hipMemcpyAsync(d_y, batch_y.data(), size_output, hipMemcpyHostToDevice, stream);

            // 初始化 d_loss 并计算 Loss（均方误差）
            double init_zero = 0.0;
            hipMemcpyAsync(d_loss, &init_zero, sizeof(double), hipMemcpyHostToDevice, stream);
            launchVecGrid2 = dim3(((BATCH_SIZE * OUTPUT_DIM) + 255) / 256);
            compute_mse_loss<<<launchVecGrid2, dim3(256), 0, stream>>>(d_output, d_y, d_loss, BATCH_SIZE * OUTPUT_DIM);

            // 同步流确保 Loss 计算完成，再拷贝回 host
            hipStreamSynchronize(stream);
            hipMemcpy(&epoch_loss, d_loss, sizeof(double), hipMemcpyDeviceToHost);
            epoch_loss /= (BATCH_SIZE * OUTPUT_DIM);

            // 计算输出梯度
            compute_output_grad<<<launchVecGrid2, dim3(256), 0, stream>>>(d_output, d_y, d_error, BATCH_SIZE * OUTPUT_DIM);

            // ----------------- 反向传播 & 参数更新 -----------------
            dim3 launchGrid3((OUTPUT_DIM + 15) / 16, (HIDDEN_DIM + 15) / 16);
            matmul_transpose_A<<<launchGrid3, dim3(16, 16), 0, stream>>>(d_hidden, d_error, d_grad_W2, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);

            dim3 launchGrid4((HIDDEN_DIM + 15) / 16, (BATCH_SIZE + 15) / 16);
            matmul_transpose_B<<<launchGrid4, dim3(16, 16), 0, stream>>>(d_error, d_W2, d_dA1, BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM);

            dim3 launchVecGrid3((BATCH_SIZE * HIDDEN_DIM + 255) / 256);
            compute_relu_backward<<<launchVecGrid3, vecBlock, 0, stream>>>(d_dA1, d_hidden, BATCH_SIZE * HIDDEN_DIM);

            dim3 launchGrid5((HIDDEN_DIM + 15) / 16, (INPUT_DIM + 15) / 16);
            matmul_transpose_A<<<launchGrid5, dim3(16, 16), 0, stream>>>(d_X, d_dA1, d_grad_W1, BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);

            int numThreads = 256;
            int numBlocksInt = (OUTPUT_DIM + numThreads - 1) / numThreads;
            sum_bias<<<numBlocksInt, numThreads, 0, stream>>>(d_error, d_grad_b2, BATCH_SIZE, OUTPUT_DIM);
            numBlocksInt = (HIDDEN_DIM + numThreads - 1) / numThreads;
            sum_bias<<<numBlocksInt, numThreads, 0, stream>>>(d_dA1, d_grad_b1, BATCH_SIZE, HIDDEN_DIM);

            int total_W1 = INPUT_DIM * HIDDEN_DIM;
            int total_W2 = HIDDEN_DIM * OUTPUT_DIM;
            int total_b1 = HIDDEN_DIM;
            int total_b2 = OUTPUT_DIM;
            dim3 sgdGrid1((total_W1 + vecBlock.x - 1) / vecBlock.x);
            sgd_update<<<sgdGrid1, vecBlock, 0, stream>>>(d_W1, d_grad_W1, LEARNING_RATE, total_W1);
            dim3 sgdGrid2((total_b1 + vecBlock.x - 1) / vecBlock.x);
            sgd_update<<<sgdGrid2, vecBlock, 0, stream>>>(d_b1, d_grad_b1, LEARNING_RATE, total_b1);
            dim3 sgdGrid3((total_W2 + vecBlock.x - 1) / vecBlock.x);
            sgd_update<<<sgdGrid3, vecBlock, 0, stream>>>(d_W2, d_grad_W2, LEARNING_RATE, total_W2);
            dim3 sgdGrid4((total_b2 + vecBlock.x - 1) / vecBlock.x);
            sgd_update<<<sgdGrid4, vecBlock, 0, stream>>>(d_b2, d_grad_b2, LEARNING_RATE, total_b2);
        }
        hipStreamSynchronize(stream);
        std::cout << "[Epoch " << epoch + 1 << "] Loss: " << epoch_loss << std::endl;
    }

    // 推理过程
    int test_samples = total_samples - train_samples;
    int test_batches = test_samples / BATCH_SIZE;
    std::vector<double> h_test_X(test_samples * INPUT_DIM);
    std::vector<double> h_test_y(test_samples * OUTPUT_DIM);
    for (int i = 0; i < test_samples; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j)
            h_test_X[i * INPUT_DIM + j] = X[(train_samples + i) * INPUT_DIM + j];
        h_test_y[i] = y[train_samples + i];
    }
    std::vector<double> predictions(test_samples * OUTPUT_DIM, 0.0);

    auto infer_start = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < test_batches; ++b) {
        hipMemcpyAsync(d_X, h_test_X.data() + b * BATCH_SIZE * INPUT_DIM, size_X, hipMemcpyHostToDevice, stream);
        dim3 launchGrid1((HIDDEN_DIM + 15) / 16, (BATCH_SIZE + 15) / 16);
        matmul<<<launchGrid1, dim3(16, 16), 0, stream>>>(d_X, d_W1, d_hidden, BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);
        dim3 launchVecGrid1((BATCH_SIZE * HIDDEN_DIM + 255) / 256);
        add_bias_and_relu<<<launchVecGrid1, dim3(256), 0, stream>>>(d_hidden, d_b1, BATCH_SIZE, HIDDEN_DIM);
        dim3 launchGrid2((OUTPUT_DIM + 15) / 16, (BATCH_SIZE + 15) / 16);
        matmul<<<launchGrid2, dim3(16, 16), 0, stream>>>(d_hidden, d_W2, d_output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);
        dim3 launchVecGrid2((BATCH_SIZE * OUTPUT_DIM + 255) / 256);
        add_bias<<<launchVecGrid2, dim3(256), 0, stream>>>(d_output, d_b2, BATCH_SIZE, OUTPUT_DIM);
        hipMemcpyAsync(predictions.data() + b * BATCH_SIZE * OUTPUT_DIM, d_output, size_output, hipMemcpyDeviceToHost, stream);
    }
    hipStreamSynchronize(stream);
    auto infer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> infer_duration = infer_end - infer_start;
    double infer_time_ms = infer_duration.count() * 1000;
    double throughput = (test_batches * BATCH_SIZE) / infer_duration.count();

    double mse = 0.0, mae = 0.0;
    for (int i = 0; i < test_samples; ++i) {
        double diff = predictions[i] - h_test_y[i];
        mse += diff * diff;
        mae += fabs(diff);
    }
    mse /= test_samples;
    mae /= test_samples;

    // 输出评价指标（中文）
    std::cout << "推理时间: " << infer_time_ms << " 毫秒" << std::endl;
    std::cout << "吞吐量: " << throughput << " 样本/秒" << std::endl;
    std::cout << "测试均方误差 (MSE): " << mse << std::endl;
    std::cout << "测试平均绝对误差 (MAE): " << mae << std::endl;

    // 保存预测结果及真实值到 CSV 文件（用于趋势匹配分析）
    std::ofstream ofs("predictions.csv");
    if (ofs.is_open()) {
        ofs << "序号,预测值,真实值\n";
        for (int i = 0; i < test_samples; ++i)
            ofs << i << "," << predictions[i] << "," << h_test_y[i] << "\n";
        ofs.close();
        std::cout << "预测结果已保存到 predictions.csv" << std::endl;
    }
    else {
        std::cout << "无法保存预测结果至 CSV 文件。" << std::endl;
    }

    // 释放设备内存
    hipFree(d_X);
    hipFree(d_hidden);
    hipFree(d_output);
    hipFree(d_W1);
    hipFree(d_b1);
    hipFree(d_W2);
    hipFree(d_b2);
    hipFree(d_error);
    hipFree(d_loss);
    hipFree(d_y);
    hipFree(d_grad_W2);
    hipFree(d_grad_W1);
    hipFree(d_grad_b2);
    hipFree(d_grad_b1);
    hipFree(d_dA1);
    hipStreamDestroy(stream);

    return 0;
}
