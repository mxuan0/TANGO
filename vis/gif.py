import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 假设有以下端点坐标数据
# 每一行代表一个时刻，每一列代表一个棍子的两个端点
X = np.array([
    [[0, 1], [1, 2], [2, 3]],
    [[1, 2], [2, 3], [3, 4]],
    [[2, 3], [3, 4], [4, 5]]
])
Y = X**2  # 仅作为示例

fig, ax = plt.subplots()
lines = [ax.plot([], [], lw=2)[0] for _ in range(X.shape[1])]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i, line in enumerate(lines):
        line.set_data(X[frame, i], Y[frame, i])
    return lines

ani = FuncAnimation(fig, update, frames=len(X), init_func=init, blit=True)

plt.xlim(0, 5)
plt.ylim(0, 25)
plt.pause(0.1)
ani.save('filename.gif', writer='pillow')
plt.show()