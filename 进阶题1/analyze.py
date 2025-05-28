import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
kernel_data = pd.read_csv("d:\\南开\\南开\\计算机组成原理\\先导杯\\进阶题1\\kernel.csv")
hiptrace_data = pd.read_csv("d:\\南开\\南开\\计算机组成原理\\先导杯\\进阶题1\\hiptrace.csv")

# 数据清理与转换
kernel_data = kernel_data[kernel_data["Name"] != "Total"]  # 排除总计行
kernel_data["TotalDurationMs"] = kernel_data["TotalDurationNs"] / 1e6  # 转换为毫秒

hiptrace_data = hiptrace_data[hiptrace_data["Name"] != "Total"]  # 排除总计行
hiptrace_data["TotalDurationMs"] = hiptrace_data["TotalDurationNs"] / 1e6  # 转换为毫秒

# 截断长字符串
def truncate_label(label, max_length=10):
    return label if len(label) <= max_length else label[:max_length] + "..."

# 创建大图
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Kernel and HIP API Execution Analysis", fontsize=18, fontweight="bold", color="darkblue", y=0.98)

# Kernel 执行时间占比（饼图）
axes[0, 0].pie(kernel_data["Percentage"], labels=[truncate_label(name) for name in kernel_data["Name"]],
               autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, textprops={'fontsize': 10})
axes[0, 0].set_title("Kernel Execution Time Percentage", fontsize=14, fontweight="bold", pad=15)
axes[0, 0].add_patch(plt.Rectangle((-1.1, -1.1), 2.2, 2.2, fill=False, edgecolor="black", lw=1.5, transform=axes[0, 0].transAxes))

# Kernel 总执行时间（柱状图，使用对数坐标轴）
axes[0, 1].bar(range(len(kernel_data)), kernel_data["TotalDurationMs"], color=plt.cm.Paired.colors[:len(kernel_data)])
axes[0, 1].set_xlabel("Kernel Name", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Total Duration (ms, log scale)", fontsize=12, fontweight="bold")
axes[0, 1].set_title("Kernel Total Execution Time", fontsize=14, fontweight="bold", pad=15)
axes[0, 1].set_yscale("log")  # 使用对数坐标轴
axes[0, 1].set_xticks(range(len(kernel_data)))
axes[0, 1].set_xticklabels([truncate_label(name) for name in kernel_data["Name"]], rotation=30, ha="right", fontsize=10)
axes[0, 1].add_patch(plt.Rectangle((-0.1, -0.1), 1.2, 1.2, fill=False, edgecolor="black", lw=1.5, transform=axes[0, 1].transAxes))

# HIP API 调用时间占比（饼图）
explode = [0.1 if i == hiptrace_data["TotalDurationMs"].idxmax() else 0 for i in range(len(hiptrace_data))]
axes[1, 0].pie(hiptrace_data["Percentage"], labels=[truncate_label(name) for name in hiptrace_data["Name"]],
               autopct='%1.1f%%', startangle=140, explode=explode, colors=plt.cm.Set3.colors, textprops={'fontsize': 10})
axes[1, 0].set_title("HIP API Execution Time Percentage", fontsize=14, fontweight="bold", pad=15)
axes[1, 0].add_patch(plt.Rectangle((-1.1, -1.1), 2.2, 2.2, fill=False, edgecolor="black", lw=1.5, transform=axes[1, 0].transAxes))

# HIP API 总执行时间（柱状图，使用对数坐标轴）
axes[1, 1].bar(range(len(hiptrace_data)), hiptrace_data["TotalDurationMs"], color=plt.cm.Set3.colors[:len(hiptrace_data)])
axes[1, 1].set_xlabel("HIP API Name", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Total Duration (ms, log scale)", fontsize=12, fontweight="bold")
axes[1, 1].set_title("HIP API Total Execution Time", fontsize=14, fontweight="bold", pad=15)
axes[1, 1].set_yscale("log")  # 使用对数坐标轴
axes[1, 1].set_xticks(range(len(hiptrace_data)))
axes[1, 1].set_xticklabels([truncate_label(name) for name in hiptrace_data["Name"]], rotation=30, ha="right", fontsize=10)
axes[1, 1].add_patch(plt.Rectangle((-0.1, -0.1), 1.2, 1.2, fill=False, edgecolor="black", lw=1.5, transform=axes[1, 1].transAxes))

# 调整布局并保存图表
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 增加子图之间的间距
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，避免标题重叠
plt.savefig("d:\\南开\\南开\\计算机组成原理\\先导杯\\进阶题1\\execution_analysis_combined.png", dpi=300)
plt.show()

# 输出 Kernel 总结数据
kernel_summary = {
    "Total Kernels": len(kernel_data),
    "Total Execution Time (ms)": kernel_data["TotalDurationMs"].sum(),
    "Average Execution Time (ms)": kernel_data["TotalDurationMs"].mean(),
    "Longest Kernel": kernel_data.loc[kernel_data["TotalDurationMs"].idxmax(), "Name"],
    "Longest Kernel Time (ms)": kernel_data["TotalDurationMs"].max()
}

print("Kernel Execution Summary:")
for key, value in kernel_summary.items():
    print(f"{key}: {value}")

# 输出 HIP API 总结数据
hiptrace_summary = {
    "Total HIP API Calls": len(hiptrace_data),
    "Total Execution Time (ms)": hiptrace_data["TotalDurationMs"].sum(),
    "Average Execution Time (ms)": hiptrace_data["TotalDurationMs"].mean(),
    "Longest HIP API": hiptrace_data.loc[hiptrace_data["TotalDurationMs"].idxmax(), "Name"],
    "Longest HIP API Time (ms)": hiptrace_data["TotalDurationMs"].max()
}

print("\nHIP API Execution Summary:")
for key, value in hiptrace_summary.items():
    print(f"{key}: {value}")
