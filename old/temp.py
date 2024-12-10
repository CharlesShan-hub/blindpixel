import matplotlib.pyplot as plt

# 示例数据
x = range(10)
y1 = [xi**2 for xi in x]  # 第一个折线图的数据
y2 = [2**xi for xi in x]  # 第二个折线图的数据

fig, ax1 = plt.subplots()

# 绘制第一个折线图
color = 'tab:red'
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个坐标轴
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Y2 data', color=color)  
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 用于调整布局
plt.show()
