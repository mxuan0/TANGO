import numpy as np
import matplotlib.pyplot as plt
import os
from synthetic_sim_pendulum import PendulumSim
import pickle
# import torch

paint_res = 300  # TODO change to 300 when publish
label_font = 24
markersize = 18
tick_font = 24
line_width = 3
markers = ['o', 's', 'v', 'D', 'h', 'H', 'd', '*', 'p', 'P', 'X', 'x', '+', '|', '_', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', 'd', 'D', 'v', '^', '<', '>']
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000"]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.frameon"] = False
plt.rcParams['figure.dpi'] = paint_res
plt.rcParams.update({'figure.autolayout': True})


def _convert_xy_to_theta(linear_loc):
    eps = 1e-6
    # assume input linear_loc are [T,3,2] np array (de normalized)
    # convert to loc (in theta)
    rod1_vec = linear_loc[:, 0, :]
    rod2_vec = linear_loc[:, 1, :] - linear_loc[:, 0, :]
    rod3_vec = linear_loc[:, 2, :] - linear_loc[:, 1, :]
    rod1_len = np.linalg.norm(rod1_vec, axis=-1)
    rod2_len = np.linalg.norm(rod2_vec, axis=-1)
    rod3_len = np.linalg.norm(rod3_vec, axis=-1)
    sin_theta1 = rod1_vec[:, 0] / rod1_len
    cos_theta1 = -rod1_vec[:, 1] / rod1_len
    sin_theta2 = rod2_vec[:, 0] / rod2_len
    cos_theta2 = -rod2_vec[:, 1] / rod2_len
    sin_theta3 = rod3_vec[:, 0] / rod3_len
    cos_theta3 = -rod3_vec[:, 1] / rod3_len

    # tan_theta1 = sin_theta1 / cos_theta1
    # tan_theta2 = sin_theta2 / cos_theta2
    # tan_theta3 = sin_theta3 / cos_theta3

    # theta1 = np.arctan(tan_theta1)
    # theta2 = np.arctan(tan_theta2)
    # theta3 = np.arctan(tan_theta3)

    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    theta3 = np.arctan2(sin_theta3, cos_theta3)

    # # Ensure theta is in the range [0, 2π)
    # if theta1 < 0:
    #     theta1 += 2 * np.pi

    # find where each rod len (shape is (T,)) is smaller than eps
    tiny_rod1_idx = np.where(rod1_len < eps)[0]
    tiny_rod2_idx = np.where(rod2_len < eps)[0]
    tiny_rod3_idx = np.where(rod3_len < eps)[0]

    theta1[tiny_rod1_idx] = 0.0
    theta2[tiny_rod2_idx] = 0.0
    theta3[tiny_rod3_idx] = 0.0

    thetas = np.stack([theta1, theta2, theta3], axis=-1)
    # print(thetas.shape)
    # exit(1)

    return thetas


def _energy_theta(loc, vel):
    g = 9.8
    stick_length = 1.0
    stick_mass = 1.0
    # disables division by zero warning, since I fix it with fill_diagonal
    with np.errstate(divide='ignore'):
        U = -stick_mass * stick_length * g / 2 * (5 * np.cos(loc[0]) + 3 * np.cos(loc[1]) + 1 * np.cos(loc[2]))
        K = stick_mass * stick_length * stick_length / 6 * (9 * vel[1] * vel[0] * np.cos(loc[0] - loc[1]) + 3 * vel[2] * vel[0] * np.cos(loc[0] - loc[2]) +
                                                            3 * vel[2] * vel[1] * np.cos(loc[1] - loc[2]) + 7 * vel[0] * vel[0] + 4 * vel[1] * vel[1] + 1 * vel[2] * vel[2])

        U_1 = -stick_mass * stick_length * g / 2 * (1 * np.cos(loc[0]))
        U_2 = -stick_mass * stick_length * g / 2 * (2 * np.cos(loc[0]) + 1 * np.cos(loc[1]))
        U_3 = -stick_mass * stick_length * g / 2 * (2 * np.cos(loc[0]) + 2 * np.cos(loc[1]) + 1 * np.cos(loc[2]))

        K_1 = stick_mass * stick_length * stick_length / 6 * (vel[0] * vel[0])
        K_2 = stick_mass * stick_length * stick_length * (vel[0] * vel[0] / 2 + vel[1] * vel[1] / 6 + vel[1] * vel[0] * np.cos(loc[0] - loc[1]) / 2)
        K_3 = stick_mass * stick_length * stick_length * (vel[0] * vel[0] / 2 + vel[1] * vel[1] / 2 + vel[2] * vel[2] / 6 + vel[1] * vel[0] * np.cos(loc[0] - loc[1]) +
                                                          vel[2] * vel[0] * np.cos(loc[0] - loc[2]) / 2 + vel[2] * vel[1] * np.cos(loc[1] - loc[2]) / 2)

        # print('U: ', U)
        # print('U_1+U_2+U_3: ', U_1 + U_2 + U_3)
        # print('K: ', K)
        # print('K_1+K_2+K_3: ', K_1 + K_2 + K_3)
        # print('energy:', U + K)
        # exit(1)

        return U + K, U, K, np.array([K_1, K_2, K_3]), np.array([U_1, U_2, U_3])


def _energy(linear_loc, linear_vel):
    # assume input linear_loc and linear_vel are [3,2] np array (de-normalized)
    # convert to loc (in theta) and vel (in theta)
    rod1_vec = linear_loc[0]
    rod2_vec = linear_loc[1] - linear_loc[0]
    rod3_vec = linear_loc[2] - linear_loc[1]
    rod1_len = np.linalg.norm(rod1_vec)
    rod2_len = np.linalg.norm(rod2_vec)
    rod3_len = np.linalg.norm(rod3_vec)
    sin_theta1 = rod1_vec[0] / rod1_len
    cos_theta1 = -rod1_vec[1] / rod1_len
    sin_theta2 = rod2_vec[0] / rod2_len
    cos_theta2 = -rod2_vec[1] / rod2_len
    sin_theta3 = rod3_vec[0] / rod3_len
    cos_theta3 = -rod3_vec[1] / rod3_len
    ang_vel1 = (linear_vel[0, 0] * cos_theta1 + linear_vel[0, 1] * sin_theta1) / rod1_len
    ang_vel2 = ((linear_vel[1, 0] - linear_vel[0, 0]) * cos_theta2 + (linear_vel[1, 1] - linear_vel[0, 1]) * sin_theta2) / rod2_len
    ang_vel3 = ((linear_vel[2, 0] - linear_vel[1, 0]) * cos_theta3 + (linear_vel[2, 1] - linear_vel[1, 1]) * sin_theta3) / rod3_len
    g = 9.8
    stick_length = 1.0
    stick_mass = 1.0
    # disables division by zero warning, since I fix it with fill_diagonal
    with np.errstate(divide='ignore'):
        U = -stick_mass * stick_length * g / 2 * (5 * cos_theta1 + 3 * cos_theta2 + 1 * cos_theta3)
        K = stick_mass * stick_length * stick_length / 6 * (9 * ang_vel2 * ang_vel1 * (cos_theta1 * cos_theta2 + sin_theta1 * sin_theta2) + 3 * ang_vel3 * ang_vel1 *
                                                            (cos_theta1 * cos_theta3 + sin_theta1 * sin_theta3) + 3 * ang_vel3 * ang_vel2 *
                                                            (cos_theta2 * cos_theta3 + sin_theta2 * sin_theta3) + 7 * ang_vel1 * ang_vel1 + 4 * ang_vel2 * ang_vel2 + 1 * ang_vel3 * ang_vel3)

        print('U: ', U)
        print('K: ', K)
        print('energy:', U + K)

        return U + K, U, K


def gen_trajtory(dir, initial_thetas=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    sim = PendulumSim()
    loc, vel, loc_theta, vel_theta, edges = sim.sample_trajectory(T=T, sample_freq=sample_freq, initial_thetas=initial_thetas)
    # dump to a folder traj/initial_thetas_T_sample_freq
    # create dir if necessary
    cache_dir = os.path.join(dir, f'traj_{initial_thetas[0, 0]}_{initial_thetas[0, 1]}_{initial_thetas[0, 2]}_{T}_{sample_freq}')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # use pickle to save all loc, vel, loc_theta, vel_theta, edges
    pickle.dump([loc, vel, loc_theta, vel_theta, edges], open(os.path.join(cache_dir, 'data.pkl'), 'wb'))


def plot_trajtory_full(dir, initial_thetas=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir = os.path.join(dir, f'traj_{initial_thetas[0, 0]}_{initial_thetas[0, 1]}_{initial_thetas[0, 2]}_{T}_{sample_freq}')
    loc, vel, loc_theta, vel_theta, edges = pickle.load(open(os.path.join(dir, cache_dir, 'data.pkl'), 'rb'))

    for t in range(loc.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # end point pos is [0,0]
        end_pos = np.array([0, 0])
        plot_end, = ax.plot(end_pos[0], end_pos[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)
        plots.append(plot_end)
        legends.append('Joint Locations')

        for joint_idx in range(loc.shape[-1]):
            ball_data = loc[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc.shape[-1]):
            ball_data = loc[t, :, joint_idx]
            prev_ball_data = loc[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 4, color=colors[joint_idx], zorder=1)
            plots.append(plot_rod)
            legends.append(f'Rod {joint_idx + 1}')

        ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
        ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
        ax.xaxis.offsetText.set_fontsize(label_font)
        ax.set_xlabel(r'X [m]', fontsize=label_font)
        ax.set_ylabel(r'Y [m]', fontsize=label_font)
        # plot in [3x3] box
        # # set the x lenght to be the same as y length
        plt.xlim(-3.25, 3.25)
        plt.ylim(-3, 1.0)
        ax.grid(True, linestyle='--', linewidth=1.5)

        # dump to the same cache dir
        plt.savefig(os.path.join(dir, cache_dir, f'frame{t}.pdf'), transparent=False, dpi=paint_res, bbox_inches="tight")


def plot_trajtory_compare(dir, initial_thetas1=np.full((1, 3), np.pi / 2), initial_thetas2=np.full((1, 3), np.pi / 2), initial_thetas3=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir1 = os.path.join(dir, f'traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{T}_{sample_freq}')
    loc1, vel1, loc_theta1, vel_theta1, edges1 = pickle.load(open(os.path.join(dir, cache_dir1, 'data.pkl'), 'rb'))
    cache_dir2 = os.path.join(dir, f'traj_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{T}_{sample_freq}')
    loc2, vel2, loc_theta2, vel_theta2, edges2 = pickle.load(open(os.path.join(dir, cache_dir2, 'data.pkl'), 'rb'))
    cache_dir3 = os.path.join(dir, f'traj_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}')
    loc3, vel3, loc_theta3, vel_theta3, edges3 = pickle.load(open(os.path.join(dir, cache_dir3, 'data.pkl'), 'rb'))

    for t in range(loc1.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # end point pos is [0,0]
        end_pos = np.array([0, 0])
        plot_end, = ax.plot(end_pos[0], end_pos[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)
        plots.append(plot_end)
        legends.append('Joint Locations')

        # plot the first traj in solid line

        # plot the joints
        for joint_idx in range(loc1.shape[-1]):
            ball_data = loc1[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc1.shape[-1]):
            ball_data = loc1[t, :, joint_idx]
            prev_ball_data = loc1[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 4, color=colors[joint_idx], zorder=9)
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('Original initial condition')

        # plot the second traj in dashed line

        # plot the joints
        for joint_idx in range(loc2.shape[-1]):
            ball_data = loc2[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='left', linewidth=line_width, color='k', zorder=8)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc2.shape[-1]):
            ball_data = loc2[t, :, joint_idx]
            prev_ball_data = loc2[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 2, color=colors[joint_idx], zorder=7, linestyle='--')
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('w/ 1e-3 perturbation')

        # plot the third traj in dotdash line

        # plot the joints
        for joint_idx in range(loc3.shape[-1]):
            ball_data = loc3[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, color='k', zorder=6)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc3.shape[-1]):
            ball_data = loc3[t, :, joint_idx]
            prev_ball_data = loc3[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 2, color=colors[joint_idx], zorder=5, linestyle='dotted')
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('w/ 1e-2 perturbation')

        ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends) // 2)
        ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
        ax.xaxis.offsetText.set_fontsize(label_font)
        ax.set_xlabel(r'X [m]', fontsize=label_font)
        ax.set_ylabel(r'Y [m]', fontsize=label_font)
        # plot in [3x3] box
        # # set the x lenght to be the same as y length
        plt.xlim(-3.25, 3.25)
        plt.ylim(-3, 1.0)
        ax.grid(True, linestyle='--', linewidth=1.5)

        # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
        cache_dir = os.path.join(
            dir,
            f'compare_traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}'
        )
        # create dir if necessary
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # dump to the same cache dir
        plt.savefig(os.path.join(dir, cache_dir, f'frame{t}.pdf'), transparent=False, dpi=paint_res, bbox_inches="tight")


def plot_rod_eng(dir, initial_thetas1=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir1 = os.path.join(dir, f'traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{T}_{sample_freq}')
    loc1, vel1, loc_theta1, vel_theta1, edges1 = pickle.load(open(os.path.join(dir, cache_dir1, 'data.pkl'), 'rb'))

    T, U, K, K_rods, U_rods = [], [], [], [], []
    # gen energy for each traj
    for t in range(loc1.shape[0]):
        t, u, k, k_rods, u_rods = _energy_theta(loc1[t, 0], loc1[t, 0])
        T.append(t)
        U.append(u)
        K.append(k)
        K_rods.append(k_rods)
        U_rods.append(u_rods)
    T = np.array(T)
    U = np.array(U)
    K = np.array(K)
    K_rods = np.array(K_rods)
    U_rods = np.array(U_rods)

    # print(K_rods.shape)
    # exit(1)

    # min_T, max_T = np.min(T), np.max(T)
    # min_U, max_U = np.min(U), np.max(U)
    # min_K, max_K = np.min(K), np.max(K)

    # max_eng = max(max_T, max_U, max_K)
    # min_eng = min(min_T, min_U, min_K)

    lines = ['-', '--', '-.']
    for t in range(loc1.shape[0] - 1, loc1.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # plot the last joint theta log
        lines = ['-', '--', '-.']
        for t in range(loc1.shape[0] - 1, loc1.shape[0]):
            # clear the plot
            ax.cla()
            plots = []
            legends = []

            # plot the last joint theta log
            lines = ['-', '--', '-.']
            for rod_idx in range(3):
                # tmpT = T[:t + 1]
                tmpU = U_rods[:t + 1, rod_idx]
                tmpK = K_rods[:t + 1, rod_idx]
                tmpT = tmpU + tmpK

                frames = np.arange(t + 1)

                plot_U, = ax.plot(frames,
                                  tmpU,
                                  marker=markers[rod_idx * 3 + 0],
                                  markersize=markersize,
                                  markevery=50,
                                  fillstyle='none',
                                  linewidth=line_width,
                                  linestyle=lines[rod_idx],
                                  color=colors[0])
                plot_K, = ax.plot(frames,
                                  tmpK,
                                  marker=markers[rod_idx * 3 + 1],
                                  markersize=markersize,
                                  markevery=50,
                                  fillstyle='none',
                                  linewidth=line_width,
                                  linestyle=lines[rod_idx],
                                  color=colors[1])
                plot_T, = ax.plot(frames,
                                  tmpT,
                                  marker=markers[rod_idx * 3 + 2],
                                  markersize=markersize,
                                  markevery=50,
                                  fillstyle='none',
                                  linewidth=line_width,
                                  linestyle=lines[rod_idx],
                                  color=colors[2])

                plots.append(plot_U)
                plots.append(plot_K)
                plots.append(plot_T)
                if rod_idx == 0:
                    legends.append(rf'Potential Energy of Rod$_{rod_idx+1}$')
                    legends.append(rf'Kinetic Energy of Rod$_{rod_idx+1}$')
                    legends.append(rf'Total Energy of Rod$_{rod_idx+1}$')
                else:
                    legends.append(rf'Rod$_{rod_idx+1}$')
                    legends.append(rf'Rod$_{rod_idx+1}$')
                    legends.append(rf'Rod$_{rod_idx+1}$')

            # # print total energy
            # tot_T = T[:t + 1]
            # plot_T, = ax.plot(frames, tot_T, fillstyle='none', linewidth=line_width, linestyle=':', color='k')
            # plots.append(plot_T)
            # # legends.append(r'Total Energy')

            ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=3)
            ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel(r'Time steps', fontsize=label_font)
            ax.set_ylabel(r'Energies', fontsize=label_font)
            ax.set_xlim([0, loc1.shape[0]])
            ax.set_ylim([-26, 16])
            ax.grid(True, linestyle='--', linewidth=1.5)

            plt.savefig(os.path.join(dir, cache_dir1, f'plot_eng_rods_{t}.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def plot_eng_compare(dir, initial_thetas1=np.full((1, 3), np.pi / 2), initial_thetas2=np.full((1, 3), np.pi / 2), initial_thetas3=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir1 = os.path.join(dir, f'traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{T}_{sample_freq}')
    loc1, vel1, loc_theta1, vel_theta1, edges1 = pickle.load(open(os.path.join(dir, cache_dir1, 'data.pkl'), 'rb'))
    cache_dir2 = os.path.join(dir, f'traj_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{T}_{sample_freq}')
    loc2, vel2, loc_theta2, vel_theta2, edges2 = pickle.load(open(os.path.join(dir, cache_dir2, 'data.pkl'), 'rb'))
    cache_dir3 = os.path.join(dir, f'traj_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}')
    loc3, vel3, loc_theta3, vel_theta3, edges3 = pickle.load(open(os.path.join(dir, cache_dir3, 'data.pkl'), 'rb'))

    # loc1_t0, vel1_t0, loc_theta1_t0, vel_theta1_t0, edges1_t0 = loc1[0], vel1[0], loc_theta1[0], vel_theta1[0], edges1[0]
    # print(loc_theta1_t0.shape)
    # print(vel_theta1_t0.shape)
    # print(loc1_t0.shape)
    # print(vel1_t0.shape)
    # T, U, K = _energy_theta(loc_theta1_t0.T, vel_theta1_t0.T)
    # T2, U2, K2 = _energy(loc1_t0.T, vel1_t0.T)
    # exit(1)

    T_theta1, U_theta1, K_theta1 = [], [], []
    T_theta2, U_theta2, K_theta2 = [], [], []
    T_theta3, U_theta3, K_theta3 = [], [], []
    # gen energy for each traj
    for t in range(loc1.shape[0]):
        T1, U1, K1 = _energy_theta(loc_theta1[t, 0], vel_theta1[t, 0])
        T2, U2, K2 = _energy_theta(loc_theta2[t, 0], vel_theta2[t, 0])
        T3, U3, K3 = _energy_theta(loc_theta3[t, 0], vel_theta3[t, 0])
        T_theta1.append(T1)
        U_theta1.append(U1)
        K_theta1.append(K1)
        T_theta2.append(T2)
        U_theta2.append(U2)
        K_theta2.append(K2)
        T_theta3.append(T3)
        U_theta3.append(U3)
        K_theta3.append(K3)
    T_theta1 = np.array(T_theta1)
    U_theta1 = np.array(U_theta1)
    K_theta1 = np.array(K_theta1)
    T_theta2 = np.array(T_theta2)
    U_theta2 = np.array(U_theta2)
    K_theta2 = np.array(K_theta2)
    T_theta3 = np.array(T_theta3)

    T_stacked = np.vstack([T_theta1, T_theta2, T_theta3]).T
    U_stacked = np.vstack([U_theta1, U_theta2, U_theta3]).T
    K_stacked = np.vstack([K_theta1, K_theta2, K_theta3]).T

    min_T, max_T = np.min(T_stacked), np.max(T_stacked)
    min_U, max_U = np.min(U_stacked), np.max(U_stacked)
    min_K, max_K = np.min(K_stacked), np.max(K_stacked)

    max_eng = max(max_T, max_U, max_K)
    min_eng = min(min_T, min_U, min_K)

    # print(T_stacked.shape)
    # exit(1)

    for t in range(loc1.shape[0] - 1, loc1.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # plot the last joint theta log
        lines = ['-', '--', '-.']
        for theta_idx in range(3):
            # theta_idx = -1
            # print(loc_theta1.shape)
            theta_T = T_stacked[:t + 1, theta_idx]
            theta_U = U_stacked[:t + 1, theta_idx]
            theta_K = K_stacked[:t + 1, theta_idx]

            frames = np.arange(t + 1)

            plot_T, = ax.plot(frames, theta_T, marker=markers[theta_idx * 3 + 0], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='-', color=colors[theta_idx])
            plot_U, = ax.plot(frames, theta_U, marker=markers[theta_idx * 3 + 1], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='--', color=colors[theta_idx])
            plot_K, = ax.plot(frames, theta_K, marker=markers[theta_idx * 3 + 2], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='-.', color=colors[theta_idx])

            if theta_idx == 0:
                legends.append(r'Original initial condition: Total Energy')
                legends.append(r'Potential Energy')
                legends.append(r'Kinetic Energy')
            elif theta_idx == 1:
                legends.append(r'w/ 1e-3 perturbation: Total Energy')
                legends.append(r'Potential Energy')
                legends.append(r'Kinetic Energy')
            else:
                legends.append(r'w/ 1e-2 perturbation: Total Energy')
                legends.append(r'Potential Energy')
                legends.append(r'Kinetic Energy')

            plots.append(plot_T)
            plots.append(plot_U)
            plots.append(plot_K)

            ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=3)
            ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel(r'Time steps', fontsize=label_font)
            ax.set_ylabel(r'Energies', fontsize=label_font)
            ax.set_xlim([0, loc1.shape[0]])
            ax.set_ylim([min_eng, max_eng])
            ax.grid(True, linestyle='--', linewidth=1.5)

            # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
            cache_dir = os.path.join(
                dir,
                f'compare_traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}'
            )
            # create dir if necessary
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            plt.savefig(os.path.join(dir, cache_dir, f'compare_eng3_{t}.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def plot_theta_gts_compare(dir, initial_thetas1=np.full((1, 3), np.pi / 2), initial_thetas2=np.full((1, 3), np.pi / 2), initial_thetas3=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir1 = os.path.join(dir, f'traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{T}_{sample_freq}')
    loc1, vel1, loc_theta1, vel_theta1, edges1 = pickle.load(open(os.path.join(dir, cache_dir1, 'data.pkl'), 'rb'))
    cache_dir2 = os.path.join(dir, f'traj_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{T}_{sample_freq}')
    loc2, vel2, loc_theta2, vel_theta2, edges2 = pickle.load(open(os.path.join(dir, cache_dir2, 'data.pkl'), 'rb'))
    cache_dir3 = os.path.join(dir, f'traj_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}')
    loc3, vel3, loc_theta3, vel_theta3, edges3 = pickle.load(open(os.path.join(dir, cache_dir3, 'data.pkl'), 'rb'))

    min_theta1 = np.min(loc_theta1[:, 0, :])
    max_theta1 = np.max(loc_theta1[:, 0, :])
    min_theta2 = np.min(loc_theta2[:, 0, :])
    max_theta2 = np.max(loc_theta2[:, 0, :])
    min_theta3 = np.min(loc_theta3[:, 0, :])
    max_theta3 = np.max(loc_theta3[:, 0, :])

    min_theta = min(min_theta1, min_theta2, min_theta3)
    max_theta = max(max_theta1, max_theta2, max_theta3)

    for t in range(loc1.shape[0] - 1, loc1.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # plot the last joint theta log
        lines = ['-', '--', '-.']
        for joint_idx in range(loc1.shape[-1]):
            # joint_idx = -1
            # print(loc_theta1.shape)
            theta1 = loc_theta1[:t + 1, 0, joint_idx]
            theta2 = loc_theta2[:t + 1, 0, joint_idx]
            theta3 = loc_theta3[:t + 1, 0, joint_idx]

            # mod thetas with [0,2 pi)
            # theta1 = np.mod((theta1 + np.pi), 2 * np.pi) - np.pi
            # theta2 = np.mod((theta2 + np.pi), 2 * np.pi) - np.pi
            # theta3 = np.mod((theta3 + np.pi), 2 * np.pi) - np.pi

            frames = np.arange(t + 1)

            plot1, = ax.plot(frames, theta1, marker=markers[joint_idx * 3 + 0], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='-', color=colors[joint_idx])
            plot2, = ax.plot(frames, theta2, marker=markers[joint_idx * 3 + 1], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='--', color=colors[joint_idx])
            plot3, = ax.plot(frames, theta3, marker=markers[joint_idx * 3 + 2], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='-.', color=colors[joint_idx])

            if joint_idx == 0:
                legends.append(r'Original initial condition: ' + rf'$\theta_{joint_idx}$')
                legends.append(r'w/ 1e-3 perturbation: ' + rf'$\theta_{joint_idx}$')
                legends.append(r'w/ 1e-2 perturbation: ' + rf'$\theta_{joint_idx}$')
            else:
                legends.append(rf'$\theta_{joint_idx}$')
                legends.append(rf'$\theta_{joint_idx}$')
                legends.append(rf'$\theta_{joint_idx}$')

            plots.append(plot1)
            plots.append(plot2)
            plots.append(plot3)

            ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=3)
            ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel(r'Time steps', fontsize=label_font)
            ax.set_ylabel(r'Joint ' + r'$\theta$', fontsize=label_font)
            ax.set_xlim([0, loc1.shape[0]])
            ax.set_ylim([min_theta, max_theta])
            ax.grid(True, linestyle='--', linewidth=1.5)

            # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
            cache_dir = os.path.join(
                dir,
                f'compare_traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}'
            )
            # create dir if necessary
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            plt.savefig(os.path.join(dir, cache_dir, f'compare_theta3_{t}.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def plot_trajtory_learned(dir, model_name, traj_idx=0):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder dir/pendulum/pendulum_{model_name}
    # under this folder there are forward_trajectory.npy and groundtruth_trajectory.npy
    folder_name = os.path.join(dir, 'pendulum', f'pendulum_{model_name}')
    forward_traj = np.load(os.path.join(folder_name, 'forward_trajectory.npy'))
    groundtruth_traj = np.load(os.path.join(folder_name, 'groundtruth_trajectory.npy'))

    # read normalizer from dir/pendulum/normalizer.json
    import json
    with open(os.path.join(dir, 'pendulum', 'normalizer.json'), 'r') as f:
        normalizer = json.load(f)

    min_loc = normalizer['min_loc']
    max_loc = normalizer['max_loc']
    min_vel = normalizer['min_vel']
    max_vel = normalizer['max_vel']

    # reshape to [-1,3,60,2,2]
    forward_traj = forward_traj.reshape(-1, 3, 60, 4)
    groundtruth_traj = groundtruth_traj.reshape(-1, 3, 60, 4)
    # print(forward_traj.shape)
    # print(groundtruth_traj.shape)
    # exit()

    # choose traj_idx at 0th dim
    forward_traj = forward_traj[traj_idx]
    groundtruth_traj = groundtruth_traj[traj_idx]

    # de-normalize
    min_vec = np.array([min_loc, min_loc, min_vel, min_vel])
    max_vec = np.array([max_loc, max_loc, max_vel, max_vel])
    forward_traj = (forward_traj + 1) / 2 * (max_vec - min_vec) + min_vec
    groundtruth_traj = (groundtruth_traj + 1) / 2 * (max_vec - min_vec) + min_vec

    # TODO get rid of this time for
    # TODO in the for joint, ax.scatter their location
    # NOTE consider using different alpha for scatter points
    # for t in range(forward_traj.shape[1]):
    # clear the plot
    ax.cla()
    plots = []
    legends = []

    # calcualte the alpha as transparency
    frames = np.arange(forward_traj.shape[1])
    alpha = frames / forward_traj.shape[1] * 0.8 + 0.2
    # scatte the joints location
    for joint_idx in range(forward_traj.shape[0]):
        ball_data = forward_traj[joint_idx, :, :2]
        # plot_i0 is only plotting the last time snapshot for the sake of alpha for legends
        plot_i0 = ax.scatter(ball_data[-1, 0], ball_data[-1, 1], marker=markers[joint_idx], s=4 * markersize, linewidth=line_width, color=colors[joint_idx], alpha=1)
        plot_i = ax.scatter(ball_data[:, 0], ball_data[:, 1], marker=markers[joint_idx], s=4 * markersize, linewidth=line_width, color=colors[joint_idx], alpha=alpha)
        plots.append(plot_i0)
        legends.append(f'Joint {joint_idx}')

        # # end point pos is [0,0]
        # end_pos = np.array([0, 0])
        # plot_end, = ax.plot(end_pos[0], end_pos[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)
        # plots.append(plot_end)
        # legends.append('Joint Locations')

        # plot the first traj in solid line

        # # plot the joints
        # for joint_idx in range(forward_traj.shape[0]):
        #     ball_data = forward_traj[joint_idx, t, :2]
        #     # # de-normalize
        #     # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
        #     ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)

        ### NOTE below are commented rod and GT plots

        # # plot the rigid rod (thick line) between each ball
        # # for 0, its the end point and ball 1
        # for joint_idx in range(forward_traj.shape[0]):
        #     ball_data = forward_traj[joint_idx, t, :2]
        #     # # de-normalize
        #     # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
        #     prev_ball_data = forward_traj[joint_idx - 1, t, :2] if joint_idx > 0 else end_pos
        #     # if joint_idx > 0:
        #     #     # de-normalize
        #     #     prev_ball_data = (prev_ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
        #     plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 4, color=colors[joint_idx], zorder=9)
        #     if joint_idx == 0:
        #         plots.append(plot_rod)
        #         legends.append('Our model')

        # # plot the second traj in dashed line

        # # plot the joints
        # for joint_idx in range(groundtruth_traj.shape[0]):
        #     ball_data = groundtruth_traj[joint_idx, t, :2]
        #     # # de-normalize
        #     # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
        #     ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='left', linewidth=line_width, color='k', zorder=8)

        # # plot the rigid rod (thick line) between each ball
        # # for 0, its the end point and ball 1
        # for joint_idx in range(groundtruth_traj.shape[0]):
        #     ball_data = groundtruth_traj[joint_idx, t, :2]
        #     # # de-normalize
        #     # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
        #     prev_ball_data = groundtruth_traj[joint_idx - 1, t, :2] if joint_idx > 0 else end_pos
        #     # if joint_idx > 0:
        #     #     # de-normalize
        #     #     prev_ball_data = (prev_ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
        #     plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 2, color=colors[joint_idx], zorder=7, linestyle='--')
        #     if joint_idx == 0:
        #         plots.append(plot_rod)
        #         legends.append('GT')

    ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.xaxis.offsetText.set_fontsize(label_font)
    ax.set_xlabel(r'X [m]', fontsize=label_font)
    ax.set_ylabel(r'Y [m]', fontsize=label_font)
    # plot in [3x3] box
    # # set the x lenght to be the same as y length
    plt.xlim(-3.25, 3.25)
    plt.ylim(-3, 1.0)
    ax.grid(True, linestyle='--', linewidth=1.5)

    # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
    cache_dir = os.path.join(dir, f'learned_model_traj_{model_name}')
    # create dir if necessary
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # dump to the same cache dir
    plt.savefig(os.path.join(dir, cache_dir, f'traj_{traj_idx}.pdf'), transparent=False, dpi=paint_res, bbox_inches="tight")


def plot_theta_learned_gt_compare(dir, our_model, prev_model, traj_idx=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    # read from the folder dir/pendulum/pendulum_{model_name}
    # under this folder there are forward_trajectory.npy and groundtruth_trajectory.npy
    folder_name = os.path.join(dir, 'pendulum', f'pendulum_{our_model}')
    forward_traj = np.load(os.path.join(folder_name, 'forward_trajectory.npy'))
    groundtruth_traj = np.load(os.path.join(folder_name, 'groundtruth_trajectory.npy'))
    # print(groundtruth_traj.shape)
    # mask = np.load(os.path.join(folder_name, 'mask.npy'), allow_pickle=True)

    # print(mask.shape)
    # exit(1)

    # load the prev model
    folder_name_prev = os.path.join(dir, 'pendulum', f'pendulum_{prev_model}')
    forward_traj_prev = np.load(os.path.join(folder_name_prev, 'forward_trajectory.npy'))

    # read normalizer from dir/pendulum/normalizer.json
    import json
    with open(os.path.join(dir, 'pendulum', 'normalizer.json'), 'r') as f:
        normalizer = json.load(f)

    min_loc = normalizer['min_loc']
    max_loc = normalizer['max_loc']
    min_vel = normalizer['min_vel']
    max_vel = normalizer['max_vel']

    min_vec = np.array([min_loc, min_loc])
    max_vec = np.array([max_loc, max_loc])

    # reshape to [-1,3,60,4]
    forward_traj = forward_traj.reshape(-1, 3, 60, 4)
    groundtruth_traj = groundtruth_traj.reshape(-1, 3, 60, 4)
    forward_traj_prev = forward_traj_prev.reshape(-1, 3, 60, 4)
    # print(forward_traj.shape)
    # print(groundtruth_traj.shape)
    # exit()

    # choose traj_idx at 0th dim; shape is [3,T,4]
    # permute to [T,3,4]
    # get only the [..,:2] begining dim: for linear loc
    forward_traj = forward_traj[traj_idx].transpose(1, 0, 2)[..., :2]
    groundtruth_traj = groundtruth_traj[traj_idx].transpose(1, 0, 2)[..., :2]
    forward_traj_prev = forward_traj_prev[traj_idx].transpose(1, 0, 2)[..., :2]

    # de-normalize
    forward_traj = (forward_traj + 1) / 2 * (max_vec - min_vec) + min_vec
    groundtruth_traj = (groundtruth_traj + 1) / 2 * (max_vec - min_vec) + min_vec
    forward_traj_prev = (forward_traj_prev + 1) / 2 * (max_vec - min_vec) + min_vec

    # convert traj (x,y) into thetas
    forward_traj_theta = _convert_xy_to_theta(forward_traj)
    # print('1')
    groundtruth_traj_theta = _convert_xy_to_theta(groundtruth_traj)
    # print('1')
    forward_traj_prev_theta = _convert_xy_to_theta(forward_traj_prev)
    # print('1')

    # print(forward_traj_theta.shape)
    # exit(1)

    # min_theta1 = np.min(loc_theta1[:, 0, :])
    # max_theta1 = np.max(loc_theta1[:, 0, :])
    # min_theta2 = np.min(loc_theta2[:, 0, :])
    # max_theta2 = np.max(loc_theta2[:, 0, :])
    # min_theta3 = np.min(loc_theta3[:, 0, :])
    # max_theta3 = np.max(loc_theta3[:, 0, :])

    # min_theta = min(min_theta1, min_theta2, min_theta3)
    # max_theta = max(max_theta1, max_theta2, max_theta3)

    for t in range(forward_traj_theta.shape[0] - 1, forward_traj_theta.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # plot the last joint theta log
        lines = ['-', '--', '-.']
        for joint_idx in range(forward_traj_theta.shape[-1]):
            # joint_idx = -1
            # print(loc_theta1.shape)
            theta1 = groundtruth_traj_theta[:t + 1, joint_idx]
            theta2 = forward_traj_theta[:t + 1, joint_idx]
            theta3 = forward_traj_prev_theta[:t + 1, joint_idx]

            frames = np.arange(t + 1)

            plot1, = ax.plot(frames, theta1, marker=markers[joint_idx * 3 + 0], markersize=markersize, markevery=4, fillstyle='none', linewidth=line_width, linestyle='-', color=colors[joint_idx])
            plot2, = ax.plot(frames, theta2, marker=markers[joint_idx * 3 + 1], markersize=markersize, markevery=4, fillstyle='none', linewidth=line_width, linestyle='--', color=colors[joint_idx])
            plot3, = ax.plot(frames, theta3, marker=markers[joint_idx * 3 + 2], markersize=markersize, markevery=4, fillstyle='none', linewidth=line_width, linestyle='-.', color=colors[joint_idx])

            if joint_idx == 0:
                legends.append(r'GT: ' + rf'$\theta_{joint_idx}$')
                legends.append(r'TANGO: ' + rf'$\theta_{joint_idx}$')
                legends.append(r'LGODE: ' + rf'$\theta_{joint_idx}$')
            else:
                legends.append(rf'$\theta_{joint_idx}$')
                legends.append(rf'$\theta_{joint_idx}$')
                legends.append(rf'$\theta_{joint_idx}$')

            plots.append(plot1)
            plots.append(plot2)
            plots.append(plot3)

            ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=3)
            ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel(r'Time steps', fontsize=label_font)
            ax.set_ylabel(r'Joint ' + r'$\theta$', fontsize=label_font)
            ax.set_xlim([0, forward_traj_theta.shape[0]])
            # ax.set_ylim([min_theta, max_theta])
            ax.grid(True, linestyle='--', linewidth=1.5)

            # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
            cache_dir = os.path.join(dir, f'compare_traj_{our_model}_{prev_model}')
            # create dir if necessary
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            plt.savefig(os.path.join(dir, cache_dir, f'compare_learned_thetas_traj{traj_idx}.pdf'), transparent=False, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    eps = 1e-3
    theta1 = np.full((1, 3), np.pi / 2)
    theta2 = np.array([[np.pi / 2 + eps, np.pi / 2 - eps, np.pi / 2 + eps]])
    theta3 = np.array([[np.pi / 2 + 10 * eps, np.pi / 2 - 10 * eps, np.pi / 2 - 10 * eps]])

    ### sample some trajtories
    # gen_trajtory('.', initial_thetas=theta1)
    # gen_trajtory('.', initial_thetas=theta2)
    # gen_trajtory('.', initial_thetas=theta3)

    ### plot the traj without perturbation
    plot_trajtory_full('.', initial_thetas=theta1)

    ### plot the traj comparisons with perturbation
    plot_trajtory_compare('.', initial_thetas1=theta1, initial_thetas2=theta2, initial_thetas3=theta3)
    plot_theta_gts_compare('.', initial_thetas1=theta1, initial_thetas2=theta2, initial_thetas3=theta3)
    plot_rod_eng('.', initial_thetas1=theta1)

    ### plot the learned results
    ### plot the theta comparison between learned, a previous work, and groundtruth
    for i in range(1, 5000, 100):  # TODO change
        plot_trajtory_learned('.', '60_DCODE_ob0.40_rflambda100.00', traj_idx=i)
        plot_trajtory_learned('.', '60_LGODE_ob0.40_rflambda0.00', traj_idx=i)
        plot_theta_learned_gt_compare('.', '60_DCODE_ob0.40_rflambda100.00', '60_LGODE_ob0.40_rflambda0.00', traj_idx=i)

    pass
