import matplotlib.pyplot as plt
import csv

# 添加中文字体支持配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件中的数据
csv_file = r"d:\\南开\\南开\\计算机组成原理\\先导杯\\进阶题2\\predictions.csv"
indices = []
predictions = []
ground_truth = []

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过标题行，此行为：序号,预测值,真实值
    for row in reader:
        indices.append(int(row[0]))
        predictions.append(float(row[1]))
        ground_truth.append(float(row[2]))

# 绘制预测趋势匹配图
plt.figure(figsize=(10, 6))
plt.plot(indices, predictions, label="预测值", color="blue")
plt.plot(indices, ground_truth, label="真实值", color="red")
plt.xlabel("序号")
plt.ylabel("带宽")
plt.title("带宽预测趋势匹配度")
plt.legend()
plt.grid(True)
# 保存图像到指定位置
plt.savefig(r"d:\\南开\\南开\\计算机组成原理\\先导杯\\进阶题2\\predictions_plot.png")
plt.show()
