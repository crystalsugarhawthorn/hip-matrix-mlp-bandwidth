# 计算机组成原理先导杯项目

## 项目概述
本项目包含基础题和进阶题两部分，主要涉及高性能计算和机器学习领域：
- 基础题：矩阵乘法优化实现（CPU/GPU多种版本）
- 进阶题1：多层感知机(MLP)实现
- 进阶题2：基于MLP的带宽预测模型

## 目录结构
```
.
├── 基础题/
│   ├── main.cpp            # 矩阵乘法主程序（含多种优化实现）
│   ├── question1.cpp      # MPI实现版本
│   ├── question2.cpp      # HIP/DCU实现版本
│   ├── performance_plot_combined.py  # 性能可视化脚本
├── 进阶题1/
│   ├── main.cpp            # MLP实现
│   ├── analyze.py          # 性能分析脚本
├── 进阶题2/
│   ├── main.cpp            # 带宽预测模型
│   ├── analyze.py          # 性能分析脚本
│   ├── visualize_predictions.py  # 预测结果可视化
│   └── starlink_bw.json    # 带宽数据集
├── README.md               # 项目说明文档
└── 实验报告.pdf            #项目分析报告
```

## 基础题 - 矩阵乘法优化

### 编译运行
```bash
# 基础版本
hipcc -fopenmp -O2 -o matmul main.cpp
./matmul

# OpenMP版本
./matmul openmp

# Block Tiling版本
./matmul block

# MPI版本 (4进程)
mpirun -np 4 ./matmul mpi

# HIP/DCU版本
hipcc question2.cpp -o outputfile_dcu
./outputfile_dcu
```

### 性能分析
```bash
# 使用hipprof工具分析
hipprof ./outputfile_dcu

# 生成性能图表
python performance_plot_combined.py
```

## 进阶题1 - 多层感知机

### 编译运行
```bash
hipcc -fopenmp -o mlp main.cpp
./mlp

# 性能分析
hipprof ./mlp
```

### 性能可视化
```bash
python analyze.py
```

## 进阶题2 - 带宽预测

### 编译运行
```bash
hipcc main.cpp -o mlp_full_dcu
./mlp_full_dcu

# 可视化预测结果
python visualize_predictions.py
```

## 性能分析工具
项目支持以下性能分析工具：
1. `rocm-smi` - 监控GPU状态
2. `hipprof` - HIP性能分析
3. `hipgdb` - HIP调试工具

## 结果展示
- 基础题：对比不同实现的执行时间和加速比
- 进阶题1：分析kernel执行时间和HIP API调用占比
- 进阶题2：展示预测值与真实值的趋势匹配图和其它相关数据

## 注意事项
1. 运行HIP/DCU版本需要安装ROCm环境
2. MPI版本需要OpenMPI支持
3. 可视化脚本需要matplotlib库
