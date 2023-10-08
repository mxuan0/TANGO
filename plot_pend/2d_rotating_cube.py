import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

paint_res = 300
label_font = 18
markersize = 18
tick_font = 18
line_width = 3
markers = ['o', 's', 'v', 'D', 'h', 'H', 'd', '*']
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000"]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.frameon"] = False
plt.rcParams['figure.dpi'] = paint_res
plt.rcParams.update({'figure.autolayout': True})
frame_dt = 0.01


def plot_linear_momentum(dir):
    # in dir there are multiple folders containing res with different resolution
    # real dir is dir/f'2d_rotating_cube_quality{q}_apic'
    # with quality in [0.5 1.0 2.0 4.0 8.0]
    # in each folder then read the file energies.npz
    # and plot it with legend
    # for the last plot legend is 'Ref'
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    qualities = [1.0]
    plots = []
    legends = []
    for i, q in enumerate(qualities):
        data = np.load(os.path.join(dir, f'2d_rotating_cube_quality{q}_apic', 'linear_momentum.npz'), allow_pickle=True)
        p = np.abs(data['p'])
        g = np.abs(data['g'])
        steps = np.arange(0, p.shape[0]) * 200
        steps = steps.astype(int)
        plot_px, = ax.plot(steps, p[:, 0], marker=markers[0], markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, color=colors[0])
        plot_py, = ax.plot(steps, p[:, 1], marker=markers[1], markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, color=colors[1])
        plot_gx, = ax.plot(steps, g[:, 0], marker=markers[2], markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, linestyle='dashed', color=colors[0])
        plot_gy, = ax.plot(steps, g[:, 1], marker=markers[3], markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, linestyle='dashed', color=colors[1])
        legends.append(r'$\left| \Sigma P_{x} \right|$' + r' on Particles')
        legends.append(r'$\left| \Sigma P_{y} \right|$' + r' on Particles')
        legends.append(r'$\left| \Sigma P_{x} \right|$' + r' on Grids')
        legends.append(r'$\left| \Sigma P_{y} \right|$' + r' on Grids')
        plots.append(plot_px)
        plots.append(plot_py)
        plots.append(plot_gx)
        plots.append(plot_gy)

    ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends) // 2)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.xaxis.offsetText.set_fontsize(label_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'Time Steps', fontsize=label_font)
    ax.set_ylabel(r'Linear Momentum [kg m/s]', fontsize=label_font)
    # set y log
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim([1e-20, 1e-13])
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.savefig(os.path.join(dir, 'linear_momentum.pdf'), transparent=False, dpi=300, bbox_inches="tight")


def plot_affine_momentum(dir):
    # in dir there are multiple folders containing res with different resolution
    # real dir is dir/f'2d_rotating_cube_quality{q}_apic'
    # with quality in [0.5 1.0 2.0 4.0 8.0]
    # in each folder then read the file energies.npz
    # and plot it with legend
    # for the last plot legend is 'Ref'
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    qualities = [1.0]
    plots = []
    legends = []
    for i, q in enumerate(qualities):
        data = np.load(os.path.join(dir, f'2d_rotating_cube_quality{q}_apic', 'affine_momentum.npz'), allow_pickle=True)
        p = np.abs(data['p'])
        g = np.abs(data['g'])
        steps = np.arange(0, p.shape[0]) * 200
        steps = steps.astype(int)
        plot_px, = ax.plot(steps, np.abs(p[:, 0] - 2e-3), marker=markers[0], markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, color=colors[0])
        plot_gx, = ax.plot(steps, np.abs(g[:, 0] - 2e-3), marker=markers[1], markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, linestyle='dashed', color=colors[1])
        legends.append(r'$\left| \Sigma \left(L-L_{0}\right) \right|$' + ' on Particles')
        legends.append(r'$\left| \Sigma \left(L-L_{0}\right) \right|$' + ' on Grids')
        plots.append(plot_px)
        plots.append(plot_gx)

    ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.xaxis.offsetText.set_fontsize(label_font)
    ax.yaxis.offsetText.set_fontsize(label_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel(r'Time Steps', fontsize=label_font)
    ax.set_ylabel(r'Affine Momentum [kg m' + r'${}^{2}$' + r'/s]', fontsize=label_font)
    # set y log
    ax.set_yscale('log')
    ax.set_ylim([1e-8, 1e-6])
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.savefig(os.path.join(dir, 'affine_momentum.pdf'), transparent=False, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_linear_momentum('.')
    plot_affine_momentum('.')
