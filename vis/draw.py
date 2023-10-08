import numpy as np
import matplotlib.pyplot as plt
import torch
# 加载.npy文件
data = np.load('/home/zijiehuang/wanjia/LG-ODE/data/spring_external/loc_test_springs_external5.npy')
print('shape : ', data.shape)


# Plot for
# plt.figure()  # Create a new figure
# group_ball=data[90]
# # print(group_ball)
# print('group_ball shape: ' ,group_ball.shape)
# for ball_index in range(5):
#     pred_ball_trajectory = group_ball[ball_index]
#     x=5.*np.cos(pred_ball_trajectory[:, 0])
#     y=5.*np.sin(pred_ball_trajectory[:, 0])
#     plt.scatter(x,y,  label=f'Ball {ball_index + 1}')

group_pred=data[0]
plt.figure()  # Create a new figure
for ball_index in range(5):
    pred_ball_trajectory = group_pred[ball_index]
    plt.scatter(pred_ball_trajectory[:, 0], pred_ball_trajectory[:, 1], label=f'Ball {ball_index + 1}')




plt.title('Trajectories of 5 Balls (PRED)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xlim(-0.4, 0.1)
# plt.ylim(-0.4, 0.3)
# plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()



