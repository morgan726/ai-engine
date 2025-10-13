import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 生成更像马鞍的双曲抛物面数据，扩大y的范围，让马鞍的凹陷更明显
x = np.linspace(-2, 2, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2/4 - Y**2/9  # 调整系数，让马鞍形态更突出

# 创建图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制马鞍面，设置透明度，让内部曲线更易观察
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6, 
                       linewidth=0, antialiased=True)

# 添加颜色条表示高度
fig.colorbar(surf, shrink=0.5, aspect=5, label='Height (Z-axis)')

# 绘制第一条测地线 L（沿x轴方向的测地线，y=0）
t1 = np.linspace(-2, 2, 100)
x1 = t1
y1 = np.zeros_like(t1)
z1 = x1**2/4 - y1**2/9
ax.plot(x1, y1, z1, 'b-', linewidth=3, label='Geodesic L')

# 定义点 P（在L外，y=2的位置）
p_x, p_y, p_z = 0, 2, (0)**2/4 - (2)**2/9
ax.scatter(p_x, p_y, p_z, color='red', s=100, label='Point P')

# 绘制过点P的第一条平行线（测地线，设计为沿y方向偏移且不与L相交的曲线）
t2 = np.linspace(-2, 2, 100)
x2 = t2
y2 = 2 * np.ones_like(t2)
z2 = x2**2/4 - y2**2/9
ax.plot(x2, y2, z2, 'g-', linewidth=3, label='Parallel Geodesic 1 (through P)')

# 绘制过点P的第二条平行线（测地线，另一条不与L相交的曲线）
t3 = np.linspace(-2, 2, 100)
x3 = -t3
y3 = 2 * np.ones_like(t3)
z3 = x3**2/4 - y3**2/9
ax.plot(x3, y3, z3, 'g--', linewidth=3, label='Parallel Geodesic 2 (through P)')

# 设置坐标轴标签和标题
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_zlabel('Z-axis', fontsize=12)
ax.set_title('Hyperbolic Axiom on the Saddle Surface', fontsize=15)

# 设置视角，从合适角度观察马鞍和曲线
ax.view_init(elev=35, azim=30)

# 添加图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()