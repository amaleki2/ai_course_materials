import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from subprocess import check_call
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.colors import ListedColormap


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
    x = np.array([x_noisy.min(), x_noisy.max()])
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
    x = np.array([x_noisy.min(), x_noisy.max()])
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
        ax.set_xlim([x[0], x[1]])
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
    ax.contourf(M, B, C.T, 50, cmap=cm.coolwarm)
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
        ax.set_xlim([x_noisy.min(), x_noisy.max()])
        return point, ax

    return fig, update_func


def import_simple_classification_data(plot=True, n_data=200):
    """
    A simple function to generate a classification problem with a nonlinear decision boundary

    inputs:
    :param plot: boolean, to plot or not
    :param n_data: number of data points
    """
    X = np.random.random((n_data, 2)) * 2
    y = X[:, 0] ** 2 - 2 * X[:, 0] + X[:, 1] - 0.5 > 0

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], s=40, c=y)
        plt.gca().axis("equal")

    return X, y


def import_spiral_classification_data(plot=True, n_data=100, n_dim=2, n_class=3):
    """
    A simple function to generate a classification problem with a nonlinear decision boundary

    inputs:
    :param plot: boolean, to plot or not
    :param n_data: number of data points
    :param n_dim: number of data features (dimensions)
    :param n_class: number of classes
    """
    X = np.zeros((n_data * n_class, n_dim))  # data matrix (each row = single example)
    y = np.zeros(n_data * n_class, dtype='uint8')  # class labels
    for j in range(n_class):
        ix = range(n_data * j, n_data * (j + 1))
        r = np.linspace(0.0, 1, n_data)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, n_data) + np.random.randn(n_data) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t) * 2]
        y[ix] = j
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    return X, y


def classifier_region_contour(model, X, y, model_name=None):
    """
    function to draw classification regions

    inputs:
    :param model: classification model
    :param X: data
    :param y: labels
    :param model_name: name of the model
    """
    num_classes = np.unique(y).shape[0]
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:num_classes])
    cmap_bold = ListedColormap(color_list_bold[0:num_classes])

    x_min, y_min = X.min(axis=0) - 0.05
    x_max, y_max = X.max(axis=0) + 0.05

    n_data = 200
    x2, y2 = np.meshgrid(np.linspace(x_min, x_max, n_data), np.linspace(y_min, y_max, n_data))
    preds = model.predict(np.c_[x2.ravel(), y2.ravel()])
    preds = preds.reshape(x2.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(x2, y2, preds, cmap=cmap_light, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=50, edgecolor='black')
    plt.xlim(x_min - 0.5, x_max + 0.5)
    plt.ylim(y_min - 0.5, y_max + 0.5)
    if model_name is not None:
        plt.title(f"classification region for {model_name} model")


def generate_kmeans_data(dim=2, n_samples=100, n_clusters=3, eps=0.3):
    X = np.zeros((n_samples * n_clusters, dim))
    cluster_centers = np.random.random((n_clusters, dim)) * 2
    for i in range(n_clusters):
        X[i * n_samples:(i + 1) * n_samples, :] += cluster_centers[i, :]
    X += np.random.random((n_samples * n_clusters, dim)) * eps
    np.random.shuffle(X)
    return X
