import numpy as np
import matplotlib.pyplot as plt
import torch
# 加载.npy文件
loc_pen = np.load('/home/zijiehuang/wanjia/LG-ODE/data/pendulum/loc_train_pendulum3.npy')

loc_penold = np.load('/home/zijiehuang/wanjia/LG-ODE/data/pendulum_old2000/loc_train_pendulum3.npy')
loc_pennew = np.load('/home/zijiehuang/wanjia/LG-ODE/data/pendulum_new2000/loc_train_pendulum3.npy')

# loc_theta = np.load('/home/zijiehuang/wanjia/LG-ODE/data/pendulum_test/loc_theta_train_pendulum3.npy')
# times=np.load('/home/zijiehuang/wanjia/LG-ODE/data/pendulum_test/times_train_pendulum3.npy')

print('loc_pen shape:', loc_pen.shape)

print('loc_pentest shape:', loc_penold.shape)
print('loc_pennew shape:', loc_pennew.shape)

# print('times shape:', times.shape)
# print(times)


group_data=loc[0]
fig, ax = plt.subplots()


for ball_index in range(3):
    pred_ball_trajectory = group_data[ball_index]

    ax.scatter(pred_ball_trajectory[:, 0], pred_ball_trajectory[:, 1], label=f'Ball {ball_index + 1}')
    print(pred_ball_trajectory[0, 0], pred_ball_trajectory[0, 1])
    ax.plot(pred_ball_trajectory[0, 0], pred_ball_trajectory[0, 1], 'd',label=f'Ball {ball_index + 1}')
ax.set_aspect('equal')
# plt.title('Trajectories of 3 Balls ')
plt.xlim(-3, 3)
plt.ylim(-3, 0)
ax.plot(0,0, 'd')
#
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.legend()
plt.grid(True)
plt.show()
