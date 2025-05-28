import matplotlib.pyplot as plt
import csv

# 配置中文显示与负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

methods = []
times = []
speedups = []

with open("d:\\南开\\南开\\计算机组成原理\\先导杯\\基础题\\perform.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        methods.append(row["Method"])
        times.append(float(row["Time(ms)"]))
        speedups.append(float(row["Speedup"]))

fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()  # 共享x轴的第二坐标轴

# 设置背景颜色
ax1.set_facecolor('#f7f7f7')
fig.patch.set_facecolor('#ffffff')

# 绘制左侧执行时间的柱状图（对数刻度）
bars = ax1.bar(methods, times, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"],
               alpha=0.85, width=0.6)
ax1.set_yscale("log")
ax1.set_xlabel("方法", fontsize=14, fontweight='bold')
ax1.set_ylabel("执行时间 (ms) (对数刻度)", fontsize=14, fontweight='bold')
ax1.set_title("矩阵乘法执行时间与加速比对比", fontsize=18, fontweight='bold')
ax1.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

# 为柱状图添加数据标签（略微偏上）
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval*1.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, color='black')

# 绘制右侧加速比折线图（对数刻度），更换颜色使其直观
ax2.plot(methods, speedups, marker='o', linestyle='-', color='#17becf', linewidth=2, markersize=8, label="加速比")
ax2.set_ylabel("加速比（相对于Baseline）", fontsize=14, fontweight='bold')
ax2.set_yscale("log")
ax2.tick_params(axis='y', labelcolor='#17becf')
for i, s in enumerate(speedups):
    ax2.text(i, s*1.05, f"{s:.2f}", ha='center', va='bottom', fontsize=12, color='#17becf')

# 设置x轴标签旋转45度以便查看
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=12)

# 添加图例，并将图例放置在图表外侧（右侧）
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, frameon=True)

# 美化坐标轴边框
for spine in ax1.spines.values():
    spine.set_edgecolor('gray')
for spine in ax2.spines.values():
    spine.set_edgecolor('gray')

plt.subplots_adjust(right=0.8)  # 调整右侧边距以留出图例区域
fig.tight_layout()
plt.savefig("d:\\南开\\南开\\计算机组成原理\\先导杯\\基础题\\performance_comparison_combined_enhanced.png", dpi=300)
plt.show()
