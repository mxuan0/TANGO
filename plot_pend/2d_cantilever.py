import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

paint_res = 300
label_font = 18
markersize = 14
tick_font = 18
line_width = 3
markers = ['o', 's', 'v', 'D', 'h', 'H', 'd', '*']
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000"]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.frameon"] = False
plt.rcParams['figure.dpi'] = paint_res
plt.rcParams.update({'figure.autolayout': True})
frame_dt = 0.02


def plot_eng(dir):
    # in dir there are multiple folders containing res with different resolution
    # real dir is dir/f'2d_cantilever_quality{q}_flip'
    # with quality in [2.0 8.0]
    # in each folder then read the file energies.npz
    # and plot it with legend
    # for the last plot legend is 'Ref'
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    qualities = [2.0, 8.0]
    plots = []
    legends = []
    for i, q in enumerate(qualities):
        data = np.load(os.path.join(dir, f'2d_cantilever_quality{q}_flip', 'energies.npz'), allow_pickle=True)
        UE = data['pot']
        KE = data['kin']
        GE = data['gra']
        TE = data['tot']
        t = np.arange(0, UE.shape[0]) * frame_dt
        color = colors[i] if i == 0 else colors[-1]
        plot_ke, = ax.plot(t, UE, marker=markers[i * 4 + 0], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, color=color)
        plot_ue, = ax.plot(t, KE, marker=markers[i * 4 + 1], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, linestyle="dashed", color=color)
        plot_ge, = ax.plot(t, GE, marker=markers[i * 4 + 2], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, linestyle="dotted", color=color)
        plot_te, = ax.plot(t, TE, marker=markers[i * 4 + 3], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, linestyle="dashdot", color=color)
        if i == 0:
            legends.append(r'Potential')
            legends.append(r'Kinetic')
            legends.append(r'Gravational')
            legends.append(r'Total')
        else:
            legends.append(r'Ref Potential')
            legends.append(r'Ref Kinetic')
            legends.append(r'Ref Gravational')
            legends.append(r'Ref Total')

        plots.append(plot_ke)
        plots.append(plot_ue)
        plots.append(plot_ge)
        plots.append(plot_te)

    ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends) // 2)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'Time [s]', fontsize=label_font)
    ax.set_ylabel(r'Energy [J]', fontsize=label_font)
    ax.set_xlim([0, 4.0])
    ax.set_ylim([-250, 300])
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.savefig(os.path.join(dir, 'energies.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def plot_Y_displacement(dir):
    # in dir there are multiple folders containing res with different resolution
    # real dir is dir/f'2d_cantilever_quality{q}_flip'
    # in each folder then read the file sample_p_disy_log.npy
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    qualities = [0.5, 1.0, 2.0, 4.0, 8.0]
    plots = []
    legends = []
    for i, q in enumerate(qualities):
        data = np.load(os.path.join(dir, f'2d_cantilever_quality{q}_flip', 'sample_p_disy_log.npy'))
        t = np.arange(0, data.shape[0]) * frame_dt
        linestyle = '-' if i != len(qualities) - 1 else 'dashed'
        color = colors[i] if i != len(qualities) - 1 else colors[-1]
        plot_y_displacement, = ax.plot(t, data, marker=markers[i], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, linestyle=linestyle, color=color)

        plots.append(plot_y_displacement)
        if i == 0:
            legends.append(r'$dx^{-1}=$' + rf'{q:.1f}')
        elif i != len(qualities) - 1:
            legends.append(rf'{int(q)}')
        else:
            legends.append(r'Ref')

    # # plot rigid recover which is a horizontal line at 12.56
    # plot_rigid_recover, = ax.plot([0, 3], [-12.56, -12.56], linewidth=line_width, linestyle='dashed', color=colors[-1])
    # plots.append(plot_rigid_recover)
    # legends.append(r'Rigid body recover')

    ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'Time [s]', fontsize=label_font)
    ax.set_ylabel(r'Y Displacement [m]', fontsize=label_font)
    ax.set_xlim([0, 4.])
    ax.set_ylim([-1.6, 0.5])
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.savefig(os.path.join(dir, 'y_displacement.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def plot_Y_displacement_rotated(dir):
    # in dir there are multiple folders containing res with different resolution
    # real dir is dir/f'2d_cantilever_quality{q}_flip'
    # in each folder then read the file sample_p_disy_log.npy
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    # qualities = [0.5, 1.0, 2.0, 4.0, 8.0]
    # q_all is the quality for all rotation angles
    q_all = 4.0
    q_ref = 8.0  # the reference quality for the reference solution
    angels = [0, 15, 30, 45]
    plots = []
    legends = []
    for i, r in enumerate(angels):
        if i == 0:
            # no rotation
            data = np.load(os.path.join(dir, f'2d_cantilever_quality{q_all}_flip', 'sample_p_disy_log.npy'))
        else:
            # rotation
            data = np.load(os.path.join(dir, f'2d_cantilever_rotated_{r}_quality{q_all}_flip', 'sample_p_disy_log.npy'))
        t = np.arange(0, data.shape[0]) * frame_dt
        linestyle = '-'
        color = colors[i]
        plot_y_displacement, = ax.plot(t, data, marker=markers[i], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, linestyle=linestyle, color=color)

        plots.append(plot_y_displacement)
        if i == 0:
            legends.append(r'Rotated 0' + r'${}^{\circ}}$')
        else:
            legends.append(rf'{r}' + r'${}^{\circ}}$')

    # plot high resolution reference solution without any rotation
    data = np.load(os.path.join(dir, f'2d_cantilever_quality{q_ref}_flip', 'sample_p_disy_log.npy'))
    t = np.arange(0, data.shape[0]) * frame_dt
    linestyle = 'dashed'
    color = colors[-1]
    plot_y_displacement, = ax.plot(t, data, marker=markers[-1], markersize=markersize, markevery=10, fillstyle='none', linewidth=line_width, linestyle=linestyle, color=color)
    plots.append(plot_y_displacement)
    legends.append(r'Ref')

    ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'Time [s]', fontsize=label_font)
    ax.set_ylabel(r'Y Displacement [m]', fontsize=label_font)
    ax.set_xlim([0, 4.])
    ax.set_ylim([-1.6, 0.5])
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.savefig(os.path.join(dir, 'y_displacement_rotated.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def convergence_test(dir):
    # in dir there are multiple folders containing res with different resolution
    # real dir is dir/f'2d_cantilever_quality{q}_flip'
    # with quality in [0.5 1.0 2.0 4.0 8.0 16.0]
    # the last resolution is the reference one
    # in each folder then dive into res folder and read there are f{particle{}.vtk} files
    # create initial particle location by reading geo_meta.pkl in the dir
    # for time in range
    # read particle data at each time step
    # use initial particle location to get analytical solution
    # compare each to the analytical solution, get the summed errors and summed sample number
    # when it's done; ave the errors record it for this resolution
    # repeat for all resolutions; then plot and save fig
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ref_dx = 0.5
    qualities = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    plots = []
    legends = []

    # get the ref
    data_ref = np.load(os.path.join(dir, f'2d_cantilever_quality{qualities[-1]}_flip', 'sample_p_disy_log.npy'))
    # remove from quality list
    qualities = qualities[:-1]
    dxs = ref_dx / np.array(qualities)

    angels = [0, 15, 30, 45]
    for j, r in enumerate(angels):
        RMSEs = []
        for i, q in enumerate(qualities):
            cache_dir = f'2d_cantilever_rotated_{r}_quality{q}_flip' if j != 0 else f'2d_cantilever_quality{q}_flip'
            data = np.load(os.path.join(dir, cache_dir, 'sample_p_disy_log.npy'))
            diff = data[:, 0] - data_ref[:, 0]
            RMSE = np.sqrt(np.mean(np.square(diff)))
            RMSEs.append(RMSE)

        RMSEs = np.array(RMSEs)
        plot_e_x, = ax.plot(dxs, RMSEs, marker=markers[j], markersize=markersize, markevery=1, fillstyle='none', linewidth=line_width, linestyle="-", color=colors[j])

        plots.append(plot_e_x)
        if j == 0:
            legends.append(r'Rotated ' + rf'{r}' + r'${}^{\circ}$')
        else:
            legends.append(rf'{r}' + r'${}^{\circ}$')
    # draw 2nd order line
    lg_x0 = np.log10(3e-1)
    lg_x1 = 1.0 + lg_x0
    lg_y0 = np.log10(2e-3)
    lg_y1 = 2.0 + lg_y0
    # uniform sample lgx and lgy; res = 200
    lgxs = np.linspace(lg_x0, lg_x1, 200)
    lgys = np.linspace(lg_y0, lg_y1, 200)
    xs = np.power(10, lgxs)
    ys = np.power(10, lgys)
    plot_2nd_order, = ax.plot(xs, ys, linewidth=line_width, linestyle="--", color=colors[-1])
    plots.append(plot_2nd_order)
    legends.append(r'$2^{nd}$' + r' order')

    ax.legend(plots, legends, loc='upper right', fontsize=label_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'Typical element size[m]', fontsize=label_font)
    ax.set_ylabel(r'RMSE', fontsize=label_font)
    # set log 10 sytle
    ax.set_xscale('log')
    ax.set_yscale('log')
    # reverse x axis
    ax.set_xlim([5, 1e-2])
    ax.set_ylim([1e-3, 0.75])
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.savefig(os.path.join(dir, 'convergence.pdf'), transparent=False, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_Y_displacement('.')
    plot_Y_displacement_rotated('.')
    plot_eng('.')
    convergence_test('.')
