import h5py
import numpy as np
import matplotlib.pyplot as plt

def get_dataset_names(h5_file_path):
    dataset_names = []

    # 打开.h5文件
    with h5py.File(h5_file_path, 'r') as f:
        # 获取所有数据集的名称
        dataset_names = list(f.keys())

    return dataset_names

def rs_train_loss(h5_file_path):
    loss_data = []  # 用于存储每一轮次的loss数据

    # 打开.h5文件
    with h5py.File(h5_file_path, 'r') as f:
        # 读取包含loss数据的数据集
        dataset = f['rs_train_loss']  # 假设数据集名称为'loss_data'

        # 将数据集转换为NumPy数组
        loss_data = np.array(dataset)

    return loss_data

def rs_train_acc(h5_file_path):
    loss_data = []  # 用于存储每一轮次的loss数据

    # 打开.h5文件
    with h5py.File(h5_file_path, 'r') as f:
        # 读取包含loss数据的数据集
        dataset = f['rs_train_acc']  # 假设数据集名称为'loss_data'

        # 将数据集转换为NumPy数组
        loss_data = np.array(dataset)

    return loss_data

# dataset_names = get_dataset_names('E:/Propy/DONE/results/Mnist_DONE_1_0.03_1.0_0.02_32u_0b_40_0.h5')
# print("Dataset names:", dataset_names)

def plot_loss(a, b, c, d, e, f):
    x_axis = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis, a[19:100:10], 'o-', color='#4169E1', alpha=0.8, linewidth=1, label='DONE')
    plt.plot(x_axis, b[19:100:10], '*-', color='#FF6600', alpha=0.8, linewidth=1, label='NEWTON')
    plt.plot(x_axis, c[19:100:10], 'v-', color='#CC0033', alpha=0.8, linewidth=1, label='FEDL')
    plt.plot(x_axis, d[19:100:10], 'x-', color='#006633', alpha=0.8, linewidth=1, label='GD')
    plt.plot(x_axis, e[19:100:10], 'p-', color='#CC33FF', alpha=0.8, linewidth=1, label='GT')
    plt.plot(x_axis, f[19:100:10], 'P-', color='#FF1493', alpha=0.8, linewidth=1, label='DANE')
    # 显示标签
    plt.legend(loc="upper right")
    plt.xlabel('Global rounds')
    plt.ylabel('Training loss')
    plt.show()

def plot_acc(a, b, c, d, e, f):
    x_axis = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis, a[19:100:10], 'o-', color='#4169E1', alpha=0.8, linewidth=1, label='DONE')
    plt.plot(x_axis, b[19:100:10], '*-', color='#FF6600', alpha=0.8, linewidth=1, label='NEWTON')
    plt.plot(x_axis, c[19:100:10], 'v-', color='#CC0033', alpha=0.8, linewidth=1, label='FEDL')
    plt.plot(x_axis, d[19:100:10], 'x-', color='#006633', alpha=0.8, linewidth=1, label='GD')
    plt.plot(x_axis, e[19:100:10], 'p-', color='#CC33FF', alpha=0.8, linewidth=1, label='GT')
    plt.plot(x_axis, f[19:100:10], 'P-', color='#FF1493', alpha=0.8, linewidth=1, label='DANE')
    # 显示标签
    plt.legend(loc="upper right")
    plt.xlabel('Global rounds')
    plt.ylabel('Training accuracy')
    plt.show()

# MNIST
MNIST_DONE_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Mnist_DONE_1_0.03_1.0_0.02_32u_0b_40_0.h5')    #替换成自己的路径
MNIST_FEDL_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Mnist_FEDL_0.04_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_NEWTON_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Mnist_Newton_1_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_GT_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Mnist_GT_1_0.03_1.0_1.0_32u_0b_40_0.h5')
MNIST_GD_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Mnist_GD_0.2_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_DANE_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Mnist_DANE_0.04_0.03_1.0_0.02_32u_0b_40_0.h5')

MNIST_DONE_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Mnist_DONE_1_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_FEDL_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Mnist_FEDL_0.04_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_NEWTON_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Mnist_Newton_1_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_GT_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Mnist_GT_1_0.03_1.0_1.0_32u_0b_40_0.h5')
MNIST_GD_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Mnist_GD_0.2_0.03_1.0_0.02_32u_0b_40_0.h5')
MNIST_DANE_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Mnist_DANE_0.04_0.03_1.0_0.02_32u_0b_40_0.h5')

# Nist
NIST_DONE_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Nist_DONE_1_0.01_1.0_0.02_32u_0b_40_0.h5')
NIST_FEDL_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Nist_FEDL_0.02_0.03_1.0_0.02_32u_0b_40_0.h5')
NIST_NEWTON_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Nist_Newton_1_0.01_1.0_0.02_32u_0b_40_0.h5')
NIST_GT_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Nist_GT_1_0.01_1.0_1.0_32u_0b_40_0.h5')
NIST_GD_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Nist_GD_0.02_0.03_1.0_0.02_32u_0b_20_0.h5')
NIST_DANE_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/Nist_DANE_0.02_0.03_1.0_0.02_32u_0b_40_0.h5')

NIST_DONE_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Nist_DONE_1_0.01_1.0_0.02_32u_0b_40_0.h5')
NIST_FEDL_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Nist_FEDL_0.02_0.03_1.0_0.02_32u_0b_40_0.h5')
NIST_NEWTON_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Nist_Newton_1_0.01_1.0_0.02_32u_0b_40_0.h5')
NIST_GT_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Nist_GT_1_0.01_1.0_1.0_32u_0b_40_0.h5')
NIST_GD_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Nist_GD_0.02_0.03_1.0_0.02_32u_0b_20_0.h5')
NIST_DANE_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/Nist_DANE_0.02_0.03_1.0_0.02_32u_0b_40_0.h5')

# HumanActivity
HumanActivity_DONE_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/human_activity_DONE_1_0.01_1.0_0.02_30u_0b_40_0.h5')
HumanActivity_FEDL_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/human_activity_FEDL_0.05_0.03_1.0_0.02_30u_0b_40_0.h5')
HumanActivity_NEWTON_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/human_activity_Newton_1_0.01_1.0_0.02_30u_0b_40_0.h5')
HumanActivity_GT_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/human_activity_GT_1_0.01_1.0_1.0_30u_0b_40_0.h5')
HumanActivity_GD_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/human_activity_GD_0.1_0.03_1.0_0.02_30u_0b_20_0.h5')
HumanActivity_DANE_loss_data = rs_train_loss('E:/Propy/DONE/DONE/results/human_activity_DANE_0.05_0.03_1.0_0.02_30u_0b_40_0.h5')

HumanActivity_DONE_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/human_activity_DONE_1_0.01_1.0_0.02_30u_0b_40_0.h5')
HumanActivity_FEDL_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/human_activity_FEDL_0.05_0.03_1.0_0.02_30u_0b_40_0.h5')
HumanActivity_NEWTON_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/human_activity_Newton_1_0.01_1.0_0.02_30u_0b_40_0.h5')
HumanActivity_GT_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/human_activity_GT_1_0.01_1.0_1.0_30u_0b_40_0.h5')
HumanActivity_GD_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/human_activity_GD_0.1_0.03_1.0_0.02_30u_0b_20_0.h5')
HumanActivity_DANE_acc_data = rs_train_acc('E:/Propy/DONE/DONE/results/human_activity_DANE_0.05_0.03_1.0_0.02_30u_0b_40_0.h5')

# MNIST
plot_loss(MNIST_DONE_loss_data, MNIST_NEWTON_loss_data, MNIST_FEDL_loss_data, MNIST_GD_loss_data, MNIST_GT_loss_data, MNIST_DANE_loss_data)
plot_acc(MNIST_DONE_acc_data, MNIST_NEWTON_acc_data, MNIST_FEDL_acc_data, MNIST_GD_acc_data, MNIST_GT_acc_data, MNIST_DANE_acc_data)

# Nist
plot_loss(NIST_DONE_loss_data, NIST_NEWTON_loss_data, NIST_FEDL_loss_data, NIST_GD_loss_data, NIST_GT_loss_data, NIST_DANE_loss_data)
plot_acc(NIST_DONE_acc_data, NIST_NEWTON_acc_data, NIST_FEDL_acc_data, NIST_GD_acc_data, NIST_GT_acc_data, NIST_DANE_acc_data)

# HumanActivity
plot_loss(HumanActivity_DONE_loss_data, HumanActivity_NEWTON_loss_data, HumanActivity_FEDL_loss_data, HumanActivity_GD_loss_data, HumanActivity_GT_loss_data, HumanActivity_DANE_loss_data)
plot_acc(HumanActivity_DONE_acc_data, HumanActivity_NEWTON_acc_data, HumanActivity_FEDL_acc_data, HumanActivity_GD_acc_data, HumanActivity_GT_acc_data, HumanActivity_DANE_acc_data)
