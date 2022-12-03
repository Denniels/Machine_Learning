#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: opti_graphs.py
Author: Ignacio Soto Zamorano / Ignacio Loayza Campos
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com / ignacio1505[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for optimization lecture - ADL
"""

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import LambdaCallback
from keras import optimizers

sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False)

#######################################################################
#                     scypython gradiente descent                     #
#######################################################################

def cost_func(theta1, y, x):
    import numpy as np
    theta1 = np.atleast_2d(np.asarray(theta1))
    return np.sum((y-hypothesis(x, theta1))**2, axis=1)

def hypothesis(x, theta1):
    return theta1*x


def gradient_1d(alpha = 1):
    """
    El código utilizado para esta visualización
        está basado en el ejemplo de grandiente descendente de la página de Scipy.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # The data to fit
    m = 20
    intercept_true = 0.5
    x = np.linspace(-1,1,m)
    y = intercept_true * x

    # The plot: LHS is the data, RHS will be the cost function.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    ax[0].scatter(x, y, marker='x', s=40, color='k')

    # First construct a grid of beta parameter pairs and their corresponding
    # cost function values.
    beta_grid = np.linspace(-0.2,1,50)
    J_grid = cost_func(beta_grid[:,np.newaxis], y, x)

    # The cost function as a function of its single parameter, beta.
    ax[1].plot(beta_grid, J_grid, 'k')

    # Take N steps with learning rate alpha down the steepest gradient,
    # starting at beta = 0.
    N = 5
    beta = [0]
    J = [cost_func(beta[0],y,x)[0]]
    for j in range(N-1):
        last_beta = beta[-1]
        this_beta = last_beta - alpha / m * np.sum((hypothesis(x, last_beta) - y) * x)
        beta.append(this_beta)
        J.append(cost_func(this_beta, y, x))

    # Annotate the cost function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    colors = ['b', 'g', 'm', 'c', 'orange']
    ax[0].plot(x, hypothesis(x, beta[0]), color=colors[0], lw=2,
               label=r'$\beta_1 = {:.3f}$'.format(beta[0]))
    for j in range(1,N):
        ax[1].annotate('', xy=(beta[j], J[j]), xytext=(beta[j-1], J[j-1]),
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')
        ax[0].plot(x, hypothesis(x, beta[j]), color=colors[j], lw=2, label=r'$\beta = {:.3f}$'.format(beta[j]))

    # Labels, titles and a legend.
    ax[1].scatter(beta, J, c=colors, s=40, lw=0)
    ax[1].set_xlim(-0.2,1)
    ax[1].set_xlabel(r'$\beta$')
    ax[1].set_ylabel(r'$J(\beta)$')
    ax[1].set_title('Cost function')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_title('Data and fit')
    ax[0].legend(loc='upper left', fontsize='small')

    plt.tight_layout()


def cost_func_2(theta0, theta1, y, x):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.sum((y-hypothesis_2(x, theta0, theta1))**2, axis=2)

def hypothesis_2(x, theta0, theta1):
    """Our "hypothesis function", a straight line."""
    return theta0 + theta1*x



def gradient_2d(alpha = 1):
    import numpy as np
    import matplotlib.pyplot as plt

    # The data to fit
    m = 20
    intercept_true = 2
    beta_true = 0.5
    x = np.linspace(-1,1,m)
    y = intercept_true + beta_true * x

    # The plot: LHS is the data, RHS will be the cost function.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    ax[0].scatter(x, y, marker='x', s=40, color='k')

    # First construct a grid of (theta0, theta1) parameter pairs and their
    # corresponding cost function values.
    intercept_grid = np.linspace(-1,4,101)
    beta_grid = np.linspace(-5,5,101)
    J_grid = cost_func_2(intercept_grid[np.newaxis,:,np.newaxis], beta_grid[:,np.newaxis,np.newaxis], y,x)

    # A labeled contour plot for the RHS cost function
    X, Y = np.meshgrid(intercept_grid, beta_grid)
    contours = ax[1].contour(X, Y, J_grid, 30)
    ax[1].clabel(contours)
    # The target parameter values indicated on the cost function contour plot
    ax[1].scatter([intercept_true]*2,[intercept_true]*2,s=[50,10], color=['k','w'])

    # Take N steps with learning rate alpha down the steepest gradient,
    # starting at (intercept, beta) = (0, 0).
    N = 5
#    alpha = 1.7
    beta = [np.array((0,0))]
    J = [cost_func_2(*beta[0],y,x)[0]]
    for j in range(N-1):
        last_beta = beta[-1]
        this_beta = np.empty((2,))
        this_beta[0] = last_beta[0] - alpha / m * np.sum((hypothesis_2(x, *last_beta) - y))
        this_beta[1] = last_beta[1] - alpha / m * np.sum((hypothesis_2(x, *last_beta) - y) * x)
        beta.append(this_beta)
        J.append(cost_func_2(*this_beta,y, x))


    # Annotate the cost function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    colors = ['b', 'g', 'm', 'c', 'orange']
    ax[0].plot(x, hypothesis_2(x, *beta[0]), color=colors[0], lw=2,
               label=r'$intercept = {:.3f}, \theta_1 = {:.3f}$'.format(*beta[0]))
    for j in range(1,N):
        ax[1].annotate('', xy=beta[j], xytext=beta[j-1],
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')
        ax[0].plot(x, hypothesis_2(x, *beta[j]), color=colors[j], lw=2,
                   label=r'$intercept = {:.3f}, \beta = {:.3f}$'.format(*beta[j]))
        ax[1].scatter(*zip(*beta), c=colors, s=40, lw=0)

        # Labels, titles and a legend.
        ax[1].set_xlabel(r'$intercept$')
        ax[1].set_ylabel(r'$\beta$')
        ax[1].set_title('Cost function')
        ax[0].set_xlabel(r'$x$')
        ax[0].set_ylabel(r'$y$')
        ax[0].set_title('Data and fit')
        axbox = ax[0].get_position()

        ax[0].legend(loc='upper right', bbox_to_anchor=(-0.2, 1), fontsize = 10)

        plt.show()
############################################################

def visualize_border(x, y, title = "", model = None):
    fig = plt.figure(figsize=(12,6))
    plt.scatter(x[:,0], x[:,1], s = 50, c=y, cmap=plt.cm.winter)
    h = .02 #step size?
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max()+1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if model is not None:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, cmap = plt.cm.Paired)

    plt.title(title)
    plt.show()

def simple_network(batch, X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(20, input_dim=20, kernel_initializer='uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    weights = []
    print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: weights.append(model.layers[0].get_weights()))

    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=0, save_best_only=False)
    model.fit(X_train, y_train, batch_size=1, 
              epochs=100, verbose=0, validation_data=(X_test, y_test), callbacks=[print_weights])
    w = [item[0] for item in [item[0] for item in [item[0] for item in weights]]]
    return w



#######################################################################
#                           Geron batch-GD                            #
#######################################################################

def artificial_points():
    """TODO: Docstring for artificial_points.
    :returns: TODO

    """
    X = 2 * np.random.rand(100, 1)
    y = 4  +3 * X + np.random.randn(100, 1)
    return X, y


def numpy_least_squares(X, y):
    """TODO: Docstring for numpy_least_squares.

    :X: TODO
    :y: TODO
    :returns: TODO

    """
    tmpX_beta = np.c_[np.ones((100, 1)), X]
    theta_svd, _, _, _ = np.linalg.lstsq(tmpX_beta, y, rcond=1e-6)
    return theta_svd


def batch_gd_plot(X, y, theta, alpha, theta_path=None):
    """TODO: Docstring for batch_gd_plot.

    :X: TODO
    :y: TODO
    :theta: TODO
    :alpha: TODO
    :theta_path: TODO
    :returns: TODO

    """
    np.random.seed(42)
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    X_b = np.c_[np.ones((100, 1)), X]
    m = len(X_b)
    theta_holder = []
    iters = 1000
    for iteration in range(iters):
        if iteration < 10:
            y_hat = X_new_b.dot(theta)
            if iteration > 0 and theta_path is None:
                plt.plot(X_new, y_hat, color='dodgerblue', linestyle='-', linewidth=1)
            elif iteration <= 0 and theta_path is None:
                plt.plot(X_new, y_hat, color='tomato', linestyle='--', linewidth=1)
        nablas = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - alpha * nablas
        if theta_path is not None:
            theta_holder.append(theta)
    if theta_path is not None:
        return theta_holder
    else:
        plt.scatter(X, y, color='slategrey', marker='o')
        plt.xlabel("$x_1$")
        plt.axis([0, 2, 0, 15])
        plt.title(r'$\alpha = {}$'.format(alpha))


def stochastic_gd_plot(X, y, n_epochs=50, theta_path=None):
    """TODO: Docstring for stochastic_gd_plot.

    :X: TODO
    :y: TODO
    :n_epochs: TODO
    :returns: TODO

    """
    np.random.seed(42)
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    X_b = np.c_[np.ones((100, 1)), X]
    m = len(X_b)

    theta_holder = []
    learn_schedule = lambda x: 5 / (x + 50)
    theta = np.random.randn(2, 1)

    for e in range(n_epochs):
        for i in range(m):
            if e == 0 and i < 20:
                y_hat = X_new_b.dot(theta)
                if i > 0 and theta_path is None:
                    plt.plot(X_new, y_hat,color='dodgerblue', linestyle='-', linewidth=1)
                elif i <= 0 and theta_path is None:
                    plt.plot(X_new, y_hat, color='tomato', linestyle='--', linewidth=1)
            rand_index = np.random.randint(m)
            x_i = X_b[rand_index:rand_index + 1]
            y_i = y[rand_index:rand_index + 1]
            nablas = 2 * x_i.T.dot(x_i.dot(theta) - y_i)
            alpha = learn_schedule(e * m + i)
            theta = theta - alpha * nablas

            if theta_path is not None:
                theta_holder.append(theta)

    if theta_path is not None:
        return theta_holder
    else:
        plt.scatter(X, y, color='slategrey', marker='o')


def mini_batch_gd_plot(X, y, theta, alpha, n_iterations=50, minibatch_size=20, theta_path=None):
    """TODO: Docstring for mini_batch_gd_plot.

    :X: TODO
    :y: TODO
    :theta: TODO
    :alpha: TODO
    :theta_path: TODO
    :returns: TODO

    """

    np.random.seed(42)
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    X_b = np.c_[np.ones((100, 1)), X]
    m = len(X_b)
    theta_holder = []

    learn_schedule = lambda x: 200/ (x + 1000)
    theta = np.random.randn(2, 1)

    t = 0
    for e in range(n_iterations):
        shuffle_index = np.random.permutation(m)
        shuffle_X_b = X_b[shuffle_index]
        shuffle_y = y[shuffle_index]
        for i in range(0, m, minibatch_size):
            t += 1
            x_i = shuffle_X_b[i:i+minibatch_size]
            y_i = shuffle_y[i:i+minibatch_size]
            nablas = 2/minibatch_size * x_i.T.dot(x_i.dot(theta) - y_i)
            alpha = learn_schedule(t)
            theta = theta - alpha * nablas
            theta_holder.append(theta)
    return theta_holder

def compare_gd_strategies(batch_gd, stochastic_gd, minibatch_gd):
    """TODO: Docstring for compare_gd_strategies.
    :returns: TODO

    """
    bgd = np.array(batch_gd)
    sgd = np.array(stochastic_gd)
    mbgd = np.array(minibatch_gd)

    plt.plot(sgd[:, 0], sgd[:, 1], label="Estocástica", linewidth=1, alpha=.5, marker='*')
    plt.plot(mbgd[:, 0], mbgd[:, 1], label="MiniBatch", linewidth=1, alpha=.5, marker="^")
    plt.plot(bgd[:, 0], bgd[:, 1], label="Batch", linewidth=1, alpha=.5, marker='o')
    plt.xlim(3, 4.5)
    plt.ylim(1, 4)

    plt.xlabel(r'$\theta_{0}$')
    plt.ylabel(r'$\theta_{1}$')
    plt.legend()

def global_surface(L=.1, n = 400):
    """TODO: Docstring for global_surface.
    :returns: TODO

    """
    x = np.linspace(-L, L, n)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.exp((X**2 + Y**2))
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, Z, rstride=60, cstride=40, cmap='coolwarm_r')
    ax.set_xticks([-L, -1, 0, 1, L])
    ax.set_yticks([-L, -1, 0, 1, L])
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlabel(r'$\theta_{0}$')
    ax.set_ylabel(r'$\theta_{1}$')
    ax.set_zlabel(r'$J(\theta)$')

def saddle_surface(angles=8, radius=24):
    """TODO: Docstring for saddle_surface.

    :angles: TODO
    :radius: TODO
    :returns: TODO

    """
    each_radius = np.linspace(0, 1.0, radius)
    each_angle = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    each_angle = np.repeat(each_angle[..., np.newaxis], radius, axis=1)
    x_axis = np.append(0, (each_radius * np.cos(each_angle)).flatten())
    y_axis = np.append(0, (each_radius * np.sin(each_angle)).flatten())
    z = np.sin((-x_axis * y_axis))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x_axis, y_axis, z, linewidth=.2, antialiased=True, cmap='coolwarm_r')
    ax.set_yticks([-1, 0, 0, 0, 1])
    ax.set_xticks([-1, 0, 0, 0, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlabel(r'$\theta_{0}$')
    ax.set_ylabel(r'$\theta_{1}$')
    ax.set_zlabel(r'$J(\theta)$')

def max_min_surface():
    """TODO: Docstring for max_min_surface.
    :returns: TODO

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.7)
    ax.plot_surface(X, Y, Z, cmap='coolwarm_r')
    ax.set_xlabel(r'$\theta_{0}$')
    ax.set_ylabel(r'$\theta_{1}$')
    ax.set_zlabel(r'$J(\theta)$')
