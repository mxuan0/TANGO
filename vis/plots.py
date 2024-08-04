import matplotlib.pyplot as plt 
import torch
import numpy as np
import pandas as pd


DCODE_SIMPLE_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_93488_simple_spring_60_DCGODE_obratio_0.6_rflambda_0.5_epoch_101_mse_0.011395840719342232_mape_10.495017194747925.ckpt'
DCODE_DAMPED_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_40754_damped_spring_60_DCGODE_obratio_0.6_rflambda_0.1_epoch_77_mse_0.006552300415933132_mape_3.0879559993743895.ckpt'

LGODE_SIMPLE_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_88415_simple_spring_60_LGODE_obratio_0.6_rflambda_0.0_epoch_45_mse_0.017428847402334212_mape_13.2711993932724.ckpt'
LGODE_DAMPED_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_45617_damped_spring_60_LGODE_obratio_0.6_rflambda_0.0_epoch_38_mse_0.009717862121760845_mape_3.9367954969406127.ckpt'

TRSODE_SIMPLE_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_99657_simple_spring_60_TRSODE_obratio_0.6_rflambda_10.0_epoch_77_mse_0.036784544587135315_mape_11.900006699562073.ckpt'
TRSODE_DAMPED_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_48873_damped_spring_60_TRSODE_obratio_0.6_rflambda_0.5_epoch_100_mse_0.1412089243531227_mape_6.647007274627685.ckpt'

HODEN_SIMPLE_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_18053_forced_spring_60_HAM_obratio_0.6_rflambda_0.0_epoch_113_mse_0.03003889471292496_mape_5.482403993606567.ckpt'
HODEN_DAMPED_SPRING_CKPT = '../../wanjia/LG-ODE/experiments/experiment_13339_damped_spring_60_HAM_obratio_0.6_rflambda_0.0_epoch_95_mse_0.08795016333460808_mape_5.106032133102417.ckpt'

SIMPLE_SPRING_BY_LENGTH = {
        'TANGO' : [0.009295, 0.008351, 0.008339, 0.009684, 0.012483], #0.011396],
        'LG-ODE' : [0.01071, 0.009695, 0.010099, 0.012589, 0.017474], #0.017429]
    }

DAMPED_SPRING_BY_LENGTH = {
        'TANGO' : [0.007755, 0.006703, 0.006292, 0.00654, 0.007409], #0.006552],
        'LG-ODE' : [0.008429, 0.007419, 0.007276, 0.008081, 0.009929], #0.009718]
    }

PENDULUM_BY_LENGTH = {
        'TANGO' : [0.009309, 0.009723, 0.010039, 0.010858, 0.012823], #0.012526],
        'LG-ODE' : [0.013601, 0.013446, 0.013464, 0.014336, 0.015718], #0.014536]
    }

EXP_DICT_BY_LENGTH = {
    'simple_spring': SIMPLE_SPRING_BY_LENGTH,
    'damped_spring': DAMPED_SPRING_BY_LENGTH,
    'pendulum': PENDULUM_BY_LENGTH
}

TENSORBOARD_DATA_DIR = 'tensorboard_data_exported'


def varying_pred_length_plot(exp, main_method_name='TANGO', baseline_name='LG-ODE'):
    result_dict = EXP_DICT_BY_LENGTH[exp]
    x_range = np.arange(len(result_dict[main_method_name]))
    fig, ax = plt.subplots()

    
    ax.plot(x_range, result_dict[main_method_name], marker='s', label=main_method_name, color='tab:blue')
    ax.plot(x_range, result_dict[baseline_name], marker='o', label=baseline_name, color='tab:red')

    custom_xticklabels = ['20', '30', '40', '50', '60'] 
    ax.set_xticks(x_range)
    ax.set_xticklabels(custom_xticklabels)
    ax.set_xlabel('MSE')
    ax.set_xlabel('Number of Extrapolation Steps')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plot/%s_BY_LENGTH.png'%exp, dpi=250)


def reverse_forward_mse_plot(exp, main_method_name='TANGO', baseline_name='LG-ODE'):
    val_main_method = pd.read_csv("%s/%s_%s.csv" % (TENSORBOARD_DATA_DIR, main_method_name, exp))['Value'].tolist()
    val_baseline_method = pd.read_csv("%s/%s_%s.csv" % (TENSORBOARD_DATA_DIR, baseline_name, exp))['Value'].tolist()
    
    
    warmup = 20
    cut = min(len(val_main_method), len(val_baseline_method))
    val_main_method = val_main_method[warmup:cut]
    val_baseline_method = val_baseline_method[warmup:cut]
    
    x_range = np.arange(len(val_main_method))
    fig, ax1 = plt.subplots()

    line1, = ax1.plot(x_range, val_main_method, color='tab:blue', label=main_method_name)
    ax1.set_ylabel('%s Reverse Forward mse'%main_method_name, color='tab:blue')

    # Create the second y-axis
    ax2 = ax1.twinx()

    # Plot the second dataset on the second y-axis
    line2, = ax2.plot(x_range, val_baseline_method, color='tab:red')
    ax2.set_ylabel('%s Reverse Forward mse'%baseline_name, color='tab:red')

    ax1.set_xlabel('Number of Epochs')
    handles = [line1, line2]
    labels = [main_method_name, baseline_name]

    # Add a legend
    plt.legend(handles, labels)

    plt.tight_layout()
    plt.savefig('plot/%s_reverse_forward_mse.png'%exp, dpi=250)


varying_pred_length_plot('simple_spring')
varying_pred_length_plot('damped_spring')
varying_pred_length_plot('pendulum')

reverse_forward_mse_plot('simple_spring')
reverse_forward_mse_plot('damped_spring')
reverse_forward_mse_plot('pendulum')
    