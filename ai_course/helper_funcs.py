import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from subprocess import check_call
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm


def plot_decision_tree(decision_tree, feature_names, class_names):
    with open("tree.dot", 'w') as f:
        export_graphviz(decision_tree,
                        max_depth=3,
                        out_file=f,
                        impurity=True,
                        feature_names=feature_names,
                        class_names=class_names,
                        rounded=True,
                        filled=True)
    check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
    img = Image.open("tree.png")
    img.save("tree.png")


def least_square_analytical(x, y):
    """
    This function calculates the analytical values of m and b using the formula above

    inputs:
    x,y = coordinates of data point. both are numpy array

    outputs:
    m_analytical: the analytical value of m given by the formula above
    b_analytical: the analytical value of b given by the formula above
    """
    n = len(x)
    m_analytical = (n * x.dot(y.T) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x) ** 2)
    b_analytical = (np.sum(y) - m_analytical * np.sum(x)) / n
    return m_analytical, b_analytical


def plot_least_square_results(x_noisy, y_noisy, loss_history, m_history, b_history):
    """
    This function plots the result of gradient descent for the least square problem.

    inputs:
    :param x_noisy, y_noisy = coordinates of data point. both are numpy array
    :param loss_history: a list containing all loss values after each optimization update
    :param m_history: a list containing all m values after each optimization update
    :param b_history: a list containing all b values after each optimization update
    """
    b_best = b_history[-1]
    m_best = m_history[-1]
    x = np.arange(x_noisy.min(), x_noisy.max())
    y_best = m_best * x + b_best

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(b_history, linewidth=2)
    plt.xlabel("iterations")
    plt.ylabel("b")

    plt.subplot(2, 2, 2)
    plt.plot(m_history, linewidth=2)
    plt.xlabel("iterations")
    plt.ylabel("m")

    plt.subplot(2, 2, 3)
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.xscale("log")
    plt.yscale("log")

    plt.subplot(2, 2, 4)
    plt.plot(x_noisy, y_noisy, 'ro', linestyle='', markersize=5)
    plt.plot(x, y_best, 'g--', linewidth=3, label='optimized line')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


def animate_least_square_final_results(x_noisy, y_noisy, m_history, b_history):
    """
    This function animates the result of gradient descent for the least square problem.
    This visualization is partially inspired by the work in
    https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/

    inputs:
    :param x_noisy, y_noisy = coordinates of data point. both are numpy array
    :param m_history: a list containing all m values after each optimization update
    :param b_history: a list containing all b values after each optimization update
    """
    x = np.arange(x_noisy.min(), x_noisy.max())
    y_init = m_history[0] * x + b_history[0]

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # plot a scatter and the initial line.
    ax.scatter(x_noisy, y_noisy)
    line, = ax.plot(x, y_init, 'r-', linewidth=2)

    def update_func(i):
        label = 'timestep {0}'.format(i)
        line.set_ydata(m_history[i] * x + b_history[i])
        ax.set_xlabel(label)
        ax.set_xlim([-1, 10])
        return line, ax

    return fig, update_func


def animate_least_square_loss(x_noisy, y_noisy, loss_func, m_history, b_history):
    """
    This function animates the result of gradient descent for the least square problem.
    This visualization is partially inspired by the work in
    https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/

    inputs:
    :param x_noisy, y_noisy = coordinates of data point. both are numpy array
    :param loss_func: the function that computes loss
    :param m_history: a list containing all m values after each optimization update
    :param b_history: a list containing all b values after each optimization update
    """
    m_best, b_best = m_history[-1], b_history[-1]

    mvec = np.linspace(-2, 8, 100)
    bvec = np.linspace(-2, 5, 70)
    C = np.zeros((100, 70))
    for i, m in enumerate(mvec):
        for j, b in enumerate(bvec):
            C[i, j] = loss_func(x_noisy, y_noisy, m, b)[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.set_tight_layout(True)
    M, B = np.meshgrid(mvec, bvec)
    ax.contourf(M, B, C.T, cmap=cm.coolwarm)
    ax.plot(m_best, b_best, 'y*', markersize=15)
    point, = ax.plot(m_history[0], b_history[0], 'r*')

    def update_func(i):
        label = 'timestep {0}'.format(i)
        point.set_xdata(m_history[0:i])
        point.set_ydata(b_history[0:i])
        point.set_marker('o')
        point.set_linestyle('--')
        ax.set_xlabel("m")
        ax.set_ylabel("b")
        # ax.set_xlim([-1,10])
        return point, ax

    return fig, update_func

