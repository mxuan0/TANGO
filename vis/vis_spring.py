import numpy as np
import matplotlib.pyplot as plt
import torch

interaction_strength=.1


def mse(mu, pred):
    # 取最后一维的前两个分量
    # mu_selected = mu[:, :, :2]
    # pred_selected = pred[:, :, :2]
    mu_selected = mu
    pred_selected = pred
    # 计算MSE
    return (mu_selected - pred_selected) ** 2

def compute_mse(mu,pred):
    log_prob = mse(mu, pred)
    res = torch.mean(log_prob, dim=(-1, -2, -3))  # 【n_traj_sample, n_traj], average among features.
    return res

def mape(mu, pred):
    # 避免除以零的情况
    epsilon = 1e-8

    # 计算 MAPE
    absolute_percentage_error = abs(mu - pred) / (abs(mu) + epsilon)
    mape_value = 100.0 * absolute_percentage_error.mean()
    return mape_value

def compute_mape(mu,pred):
    log_prob = mape(mu, pred)
    res = torch.mean(log_prob, dim=(-1, -2, -3))  # 【n_traj_sample, n_traj], average among features.
    return res


def find_best(mu,pred):
    n_traj, n_ball, n_timepoints, n_dims = mu.size()
    assert (pred.size()[-1] == n_dims)
    t_best=0
    best_mse = compute_mse(mu[0], pred[0])
    for t in range(1, n_traj):
        now_mse = compute_mse(mu[t],pred[t])
        if  now_mse <= best_mse:
            best_mse = now_mse
            t_best=t
    return t_best



# 加载.npy文件

data_gt = np.load('/home/zijiehuang/wanjia/LG-ODE/visdata/simple_spring_60/observe_ratio_train0.40_test0.40/train_cut20000_test_cut5000/reverse_f_lambda0.50_reverse_gt_lambda0.00/'
               'groundtruth_trajectory.npy')
data_f = np.load('/home/zijiehuang/wanjia/LG-ODE/visdata/forced_spring_60/observe_ratio_train0.40_test0.40/train_cut20000_test_cut5000/reverse_f_lambda0.50_reverse_gt_lambda0.00/'
               'forward_trajectory.npy')
data_r = np.load('/home/zijiehuang/wanjia/LG-ODE/visdata/forced_spring_60/observe_ratio_train0.40_test0.40/train_cut20000_test_cut5000/reverse_f_lambda0.50_reverse_gt_lambda0.00/'
               'reverse_trajectory.npy')
edges=np.load('/home/zijiehuang/wanjia/LG-ODE/data/forced_spring/edges_test_springs_external5.npy')
print('gt shape : ',data_gt.shape)
print('f shape : ',data_f.shape)
print('r shape : ',data_r.shape)
print("edges shape:",edges.shape)

data_gt_tensor = torch.from_numpy(data_gt)
data_gt_draw = data_gt_tensor.view(392, 5, 60, 4)

data_f_tensor = torch.from_numpy(data_f)
data_f_draw = data_f_tensor.view(392, 5, 60, 4)

data_r_tensor = torch.from_numpy(data_r)
data_r_draw = data_r_tensor.view(392, 5, 60, 4)


print('gt final shape : ',data_gt_draw.shape)
print('f final shape : ',data_f_draw.shape)
print('r final shape : ',data_r_draw.shape)


mu=data_gt_draw
pred=data_r_draw
group_index = find_best(mu,pred) # find best trajectory
group_mu = mu[group_index]
group_pred = pred[group_index]
# goup_edges=

print('best trajectory index',group_index )

# 获取数据的最大和最小值以设置统一的轴范围
all_data = np.concatenate([group_mu, group_pred], axis=0)
x_min, x_max = all_data[:, :, 0].min(), all_data[:, :, 0].max()
y_min, y_max = all_data[:, :, 1].min(), all_data[:, :, 1].max()
print(x_min, x_max,y_min, y_max)

# Plot for group_mu
plt.figure()  # Create a new figure
for ball_index in range(5):
    mu_ball_trajectory = group_mu[ball_index]
    plt.scatter(mu_ball_trajectory[:, 0], mu_ball_trajectory[:, 1], label=f'Ball {ball_index + 1}')

plt.title('Trajectories of 5 Balls (MU)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xlim(-0.4, 0.1)
plt.ylim(-0.4, 0.3)
# plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

# Plot for group_pred
plt.figure()  # Create a new figure
for ball_index in range(5):
    pred_ball_trajectory = group_pred[ball_index]
    plt.scatter(pred_ball_trajectory[:, 0], pred_ball_trajectory[:, 1], label=f'Ball {ball_index + 1}')


plt.title('Trajectories of 5 Balls (PRED)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xlim(-0.4, 0.1)
plt.ylim(-0.4, 0.3)
# plt.xlim(-0.4, 0.1)
# plt.ylim(-0.4, 0.3)
# plt.axis('equal')
plt.legend()
plt.grid(True)


loc_mu=group_mu[:, :, :2].view(60,5,2).cpu().detach().numpy()
vel_mu=group_mu[:, :, 2:].view(60,5,2).cpu().detach().numpy()
loc_pred=group_pred[:, :, :2].view(60,5,2).cpu().detach().numpy()
vel_pred=group_pred[:, :, 2:].view(60,5,2).cpu().detach().numpy()

print('loc_mu:',loc_mu.shape)
print('loc_mu[0]:',loc_mu.shape[0])
plt.figure()
axes = plt.gca()
# axes.set_xlim([0., 2.])
# axes.set_ylim([0., 2.])
def _energy(loc, vel, edges):
    # disables division by zero warning, since I fix it with fill_diagonal
    with np.errstate(divide='ignore'):

        K = 0.5 * (vel ** 2).sum()
        U = 0
        for i in range(loc.shape[0]):
            for j in range(loc.shape[0]):
                if i != j:
                    r = loc[i,:] - loc[ j,:]
                    dist = np.sqrt((r ** 2).sum())
                    U += 0.5 * interaction_strength * edges[
                        i, j] * (dist ** 2) / 2
        print('U:',U )
        print('K:', K)
        return U + K


mu_energies = [_energy(loc_mu[i,:, :], vel_mu[i, :,:], edges) for i in
            range(loc_mu.shape[0])]
# pred_energies = [_energy(loc_pred[i, :], vel_pred[i, :], edges) for i in
#             range(loc_pred.shape[0])]

plt.plot(mu_energies)
# plt.plot(pred_energies)

plt.title('Energy (simple)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()
plt.show()

# Plot for energy