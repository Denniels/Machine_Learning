#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec9_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for Gradient Boosting Trees - ADL
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xgboost
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, mean_squared_error, roc_curve, auc, classification_report


# Feed data
df = pd.read_csv('Datasets/spamdata_esl_colnames.csv').drop(columns='Unnamed: 0')

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, 'word_freq_george':'capital_run_length_total'],
                                                    df['spam'], test_size=.33, random_state=11238)
model_list = ['GradientBoostingClassifier', 'GradientBoostingRegressor']
fetch_lims = lambda x: [np.floor(np.min(x)), np.ceil(np.max(x))]
count_valid_model_class = lambda x: [True if re.search(x, i, re.IGNORECASE) else False for i in model_list].count(True)

markers = ['^', 'o']
tree = DecisionTreeClassifier(max_depth=1)



def get_mu_sigma(train_vector, test_vector):
    """TODO: Docstring for get_mu_sigma.

    :train_vector: TODO
    :test_vector: TODO
    :returns: TODO

    """
    return np.mean(train_vector, axis=1), np.std(train_vector,axis=1), np.mean(test_vector, axis=1), np.std(test_vector, axis=1)

def generate_mesh_grid(df, x1, x2):
    """TODO: Docstring for generate_mesh_grid.

    :df: TODO
    :x1: TODO
    :x2: TODO
    :returns: TODO

    """
    tmp_X = df.loc[:, [x1, x2]]
    tmp_x, tmp_y = np.meshgrid(
        np.linspace(np.min(tmp_X[x1]), np.max(tmp_X[x1]), num=100),
        np.linspace(np.min(tmp_X[x2]), np.max(tmp_X[x2]), num=100)
    )

    joint_xy = np.c_[
        tmp_x.ravel(), tmp_y.ravel()
    ]

    return tmp_x, tmp_y, joint_xy


def loss_functions():
    """TODO: Docstring for loss_functions.
    :returns: TODO

    """
    def huber_loss(y_true, y_pred):
        """TODO: Docstring for huber_loss.

        :y_true: TODO
        :y_pred: TODO
        :returns: TODO

        """
        z = y_pred * y_true
        loss = -4 * z
        loss[z >= -1] = (1 - z[z >= -1]) ** 2
        loss[z >= 1] = 0
        return loss

    x_axis = np.linspace(-4, 4, 100)

    plt.plot([-4, 0, 0, 4], [1, 1, 0, 0], label="Sharp", lw=3)
    plt.plot(x_axis, np.where(x_axis < 1, 1 - x_axis, 0), label='Hinge', lw = 3)
    plt.plot(x_axis, x_axis ** 2, label='Squared', lw=3)
    plt.plot(x_axis, np.where(x_axis < 1, 1 - x_axis, 0) ** 2, label='Squared hinge', lw = 3)
    plt.plot(x_axis, huber_loss(x_axis, 1), label='Huber', lw=3)
    plt.ylim(0, 10)
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))





def plot_class_report(y_test, y_hat, classes_labels):
    """TODO: Docstring for plot_class_report.

    :y_test: TODO
    :y_hat: TODO
    :returns: TODO

    """
    # extraemos la métrica. Ya que es un string, generamos un split por los saltos de línea
    tmp_report = classification_report(y_test, y_hat).split()
    # Excluímos un listado de palabras definidas de esta lista
    tmp_report = [i for i in tmp_report if i not in ['avg', '/', 'precision', 'recall', 'f1-score', 'support']]
    # Generamos un reshape con 5 columnas y n/5 filas.
    tmp_report = pd.DataFrame(np.array(tmp_report).reshape(int(len(tmp_report)/ 5),5))
    # Agregamos etiquetas a cada columna
    tmp_report.columns = ['class', 'Precision', 'Recall', 'F1', 'N']
    # declaramos class como índice
    tmp_report = tmp_report.set_index('class')
    tmp_report.index = np.append(classes_labels, 'total')
    # Transformamos todas las cifras a float
    tmp_report = tmp_report.applymap(lambda x: float(x))

    # Para cada columna de la base, extraemos la posición, nombre y datos
    for index, (colname, serie) in enumerate(tmp_report.iteritems()):
        # Si no es la última columna
        if index + 1 is not tmp_report.shape[0]:
            # generamos una lista vacía para guardar etiquetas
            ticks_holder = []
            # para cada una de las tres métricas reportadas creamos un subplot
            plt.subplot(3, 1, index + 1)
            # comparamos la métrica para cada clase estimada
            plt.barh(range(tmp_report.shape[0] - 1), serie.drop('total'),color= color_palette_sequential[index + 1])
            # extraemos el valor y nombre de la clase estimada
            for i, v  in enumerate(serie.drop('total')):
                # y la guardamos como un string en nuestra lista vacía
                ticks_holder.append("{}: {}".format(tmp_report.index[i], v))
            # Agregamos el benchmark global
            plt.axvline(serie.iloc[-1], color='tomato', linestyle='--', lw=4)
            # Declaramos el valor promedio
            plt.title("{}: Average = {}".format(colname,serie.iloc[-1]), fontsize=16)
            # Ajustamos el gráfico
            plt.tight_layout()
            # Agregamos las etiquetas construídas.
            plt.yticks(range(tmp_report.shape[0] - 1), ticks_holder, fontsize=14)

def train_test_over_params(model, params, X_train, X_test, y_train, y_test, plot_mean=True):
    """TODO: Docstring for train_test_over_params.
    :model: TODO
    :params: TODO
    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :returns: TODO
    """
    tmp_train, tmp_test = [], []
    values = list(params.values())[0]
    hyperparam = str(list(params.keys())[0])


    for i in values:
        params_spec = {hyperparam: i}
        tmp_model = model.set_params(**params_spec).fit(X_train, y_train)
        tmp_train.append(1 - accuracy_score(y_train, tmp_model.predict(X_train)))
        tmp_test.append(1 - accuracy_score(y_test, tmp_model.predict(X_test)))

    plt.plot(values, tmp_train, '.-',color='dodgerblue', label='Train', lw=2, alpha=.5)
    plt.plot(values, tmp_test,'.-', color='tomato', label='Test', lw=2, alpha=.5)

    if plot_mean is True:
        plt.axhline(np.mean(tmp_train), lw=.5, linestyle="--", color="dodgerblue", label="Mean Train Error: {}".format(round(np.mean(tmp_train), 3)))
        plt.axhline(np.mean(tmp_test), lw=.5,linestyle='--', color="tomato", label="Mean Test Error: {}".format(round(np.mean(tmp_test), 3)))

    plt.legend()
    plt.title(hyperparam)
    plt.ylabel("Error Rate")
    # tmp_best_score = tmp_test[np.max(tmp_test)]
    # plt.axvline(tmp_best_score, color='slategrey',
    # linestyle='--',
    # label="Best {} on test: {}".format(hyperparam, round(tmp_best_score, 3)))



def plot_response_boundaries(model, X_train, y_train, X_testing, y_testing, x_mesh_train, y_mesh_train):
    tmp_model = model.fit(X_train, y_train)
    model_Z_density = tmp_model.predict(np.c_[x_mesh_train.ravel(), y_mesh_train.ravel()]).reshape(x_mesh_train.shape)
    tmp_test_error = 1 - accuracy_score(y_testing, tmp_model.predict(X_testing))
    plt.contour(x_mesh_train, y_mesh_train, model_Z_density,cmap='coolwarm', zorder=1)
    for i in y_train.unique():
        plt.scatter(X_train[y_train == i].iloc[:, 0],
                    X_train[y_train == i].iloc[:, 1],
                    marker=markers[i], alpha=.3, color='grey', label=i, zorder=5)
    plt.legend()
    return round(tmp_test_error, 3)


X_mat = df.loc[:, ['word_freq_george', 'capital_run_length_total']]
# Para reescalar atributos, tomaremos el logaritmo más una pequeña suavización
X_mat = X_mat.apply(lambda x: np.log(x + 0.01), axis=1)
X_mat['y'] = df['spam']
x_mesh, y_mesh, joint_xy = generate_mesh_grid(X_mat,'word_freq_george', 'capital_run_length_total')
X_tr, X_te, y_tr, y_te = train_test_split(X_mat.loc[:, ['word_freq_george', 'capital_run_length_total']],
                                          X_mat['y'], random_state=11238, test_size=.33)
def n_estimators_hyperparams():
    """TODO: Docstring for n_estimators_hyperparams.
    :returns: TODO

    """
    gs = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(gs[0, :])
    train_test_over_params(model=AdaBoostClassifier(base_estimator=tree, random_state=11238, n_estimators=100),
                           params= {'n_estimators': np.linspace(1, 1000, 100, endpoint=True, dtype=int)},
                           X_train = X_tr, X_test = X_te, y_train = y_tr, y_test=y_te)


    ax4 = plt.subplot(gs[-1, 0])
    tmp_score = plot_response_boundaries(AdaBoostClassifier(base_estimator=tree,
                                                            random_state=11238,n_estimators=1),
                                         X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
    plt.title( r'n_estimators'+": {}\nTest Error: {}".format(1, tmp_score), fontsize=10)
    plt.xticks(());plt.yticks(());
    ax5 = plt.subplot(gs[-1, -1])

    tmp_score = plot_response_boundaries(AdaBoostClassifier(base_estimator=tree,
                                                            random_state=11238,n_estimators=600),
                                         X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
    plt.title( r'n_estimators'+": {}\nTest Error: {}".format(600, tmp_score),fontsize=10)
    plt.xticks(());plt.yticks(());
    ax6 = plt.subplot(gs[-1, -2])
    tmp_score = plot_response_boundaries(AdaBoostClassifier(base_estimator=tree,
                                                            random_state=11238,n_estimators=120),
                                         X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
    plt.title( r'n_estimators'+": {}\nTest Error: {}".format(190, tmp_score), fontsize=10)
    plt.xticks(());plt.yticks(());
    plt.tight_layout()

def learning_rate_hyperparams():
    """TODO: Docstring for learning_rate_hyperparams.
    :returns: TODO

    """
    gs = gridspec.GridSpec(2, 3)
    plt.subplot(gs[0, :])
    train_test_over_params(model=AdaBoostClassifier(base_estimator=tree, random_state=11238, n_estimators=200),
                           params= {'learning_rate': np.linspace(0.001, 10, 50, endpoint=True, dtype=float)},
                           X_train = X_tr, X_test = X_te, y_train = y_tr, y_test=y_te)
    plt.xticks(np.linspace(0.01, 10,endpoint=True, dtype=float).round(2), rotation=90)
    plt.subplot(gs[-1, 0])
    tmp_score = plot_response_boundaries(AdaBoostClassifier(base_estimator=tree,
                                                            random_state=11238,
                                                            n_estimators = 200,
                                                            learning_rate=0.01),
                                         X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
    plt.title( r'learning_rate'+": {}\nTest Error: {}".format(0.01, tmp_score), fontsize=10)
    plt.xticks(());plt.yticks(());
    plt.subplot(gs[-1, -1])

    tmp_score = plot_response_boundaries(AdaBoostClassifier(base_estimator=tree,
                                                            random_state=11238,
                                                            n_estimators = 200,
                                                            learning_rate=2.05),
                                         X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
    plt.title( r'learning_rate'+": {}\nTest Error: {}".format(2.05, tmp_score), fontsize=10)
    plt.xticks(());plt.yticks(());
    plt.subplot(gs[-1, -2])
    tmp_score = plot_response_boundaries(AdaBoostClassifier(base_estimator=tree,
                                                            random_state=11238,
                                                            n_estimators = 200,
                                                            learning_rate=1.23),
                                         X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
    plt.title( r'learning_rate'+": {}\nTest Error: {}".format(1.23, tmp_score),fontsize=10)
    plt.xticks(());plt.yticks(());
    plt.tight_layout()


def plot_predictions(model, X, y, axes, label=None, style='tomato', data_style='.', data_label=None):
    """TODO: Docstring for plot_predictions.

    :model: TODO
    :X: TODO
    :y: TODO
    :axes: TODO
    :label: TODO
    :style: TODO
    :data_style: TODO
    :data_label: TODO
    :returns: TODO

    """
    x_1 = np.linspace(axes[0], axes[1], 500)
    y_hat = sum(i.predict(x_1.reshape(-1, 1)) for i in model)
    plt.plot(X[:,0], y, data_style, label=data_label)
    plt.plot(x_1, y_hat, style, linewidth=2, label=label)
    if label is True or data_label is True:
        plt.legend()


def gboost_early_stoping():
    """TODO: Docstring for gboost_early_stoping.
    :returns: TODO

    """
    X_train, X_test, y_train, y_test = train_test_split()
    tmp_gradient_boosting = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=11238).fit(X_train, y_train)
    errors = [mean_squared_error(y_test, i) for i in tmp_gradient.predict(X_test)]
    best_estimator = np.argmin(errors)
    tmp_best_gb = GradientBoostingRegressor(max_depth=2, n_estimators=base_estimator, random_state=11238).fit(X_train, y_train)
    minimal_error = np.min(errors)

    plt.subplot(1, 2, 1)
    plt.plot(errors, '.')
    plt.plot([base_estimator, best_estimator], [0, minimal_error], '--')
    plt.plot([0, 120], [minimal_error, minimal_error], 'o')
    plt.plot(best_estimator, minimal_error)
    plt.plot(best_estimator)

    plt.subplot(1, 2, 2)
    plt.plot(plot_predictions) # onhold!


def plot_roc(model=tree, y_true=y_test, X_test=X_test, model_label=None):
    """TODO: Docstring for plot_roc.

    :model: TODO
    :y_true: TODO
    :X_test: TODO
    :model_label: TODO
    :returns: TODO

    """
    tmp_y_pred = model.predict_proba(X_test)[:, 1]
    false_positive_rates, true_positive_rates, _ = roc_curve(y_test, tmp_y_pred)
    store_auc = auc(false_positive_rates, true_positive_rates)
    if model_label is not None:
        tmp_label = "{}: {}".format(model_label, round(store_auc,3))
    else:
        tmp_label = None
    plt.plot(false_positive_rates, true_positive_rates, label=tmp_label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def adaboost_weak_learner_behavior():

    X_mat = df.loc[:, ['word_freq_george', 'capital_run_length_total']]
    # Para reescalar atributos, tomaremos el logaritmo más una pequeña suavización
    X_mat = X_mat.apply(lambda x: np.log(x + 0.01), axis=1)
    X_mat['y'] = df['spam']
    x_mesh, y_mesh, joint_xy = generate_mesh_grid(X_mat,'word_freq_george', 'capital_run_length_total')
    X_tr, X_te, y_tr, y_te = train_test_split(X_mat.loc[:, ['word_freq_george', 'capital_run_length_total']],
                                              X_mat['y'], random_state=11238, test_size=.33)

    adaboost_demo_model = AdaBoostClassifier(random_state=11238).fit(X_tr, y_tr)
    x_mesh, y_mesh, joint_xy = generate_mesh_grid(X_mat,'word_freq_george', 'capital_run_length_total')
    for index, value in enumerate(np.random.choice(range(50), size=6)):
        plt.subplot(2, 3, index + 1)
        Z = adaboost_demo_model.estimators_[value].predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
        tmp_error_test = 1 - accuracy_score(y_te, adaboost_demo_model.estimators_[value].predict(X_te))
        Z = Z.reshape(x_mesh.shape)
        plt.contour(x_mesh, y_mesh, Z, colors='orange')
        plt.title("Árbol: {}\n Max Depth: 1\n Test Error: {}".format(value, round(tmp_error_test, 3)), fontsize=10)
        for i in X_mat['y'].unique():
            plt.scatter(X_mat[X_mat['y'] == i]['word_freq_george'],
                        X_mat[X_mat['y'] == i]['capital_run_length_total'],
                        marker=markers[i], alpha=.2, color='grey')

        plt.tight_layout()

def plot_adaptation(model, n_estimators):
    for i in n_estimators:
        z = model.estimators_[i].predict(joint_xy).reshape(x_mesh.shape)
        plt.contour(x_mesh, y_mesh, z, zorder=100, colors='orange')
    for i in X_mat['y'].unique():
        plt.scatter(X_mat[X_mat['y'] == i]['word_freq_george'],
                    X_mat[X_mat['y'] == i]['capital_run_length_total'],
                    alpha=.7, color='grey', zorder=10)
        plt.title("n_est: {}".format(len(n_estimators)))


def adaboost_adaptive_behavior():
    adaboost_demo_model = AdaBoostClassifier(random_state=11238, n_estimators=100).fit(X_tr, y_tr)
    plt.subplot(1, 4, 1)
    plot_adaptation(adaboost_demo_model, [1])
    plt.subplot(1, 4, 2)
    plot_adaptation(adaboost_demo_model, [1, 2, 3, 5])
    plt.subplot(1, 4, 3)
    plot_adaptation(adaboost_demo_model, [1, 2, 3, 5, 10, 20, 50])
    plt.subplot(1, 4, 4)
    tmp_z = adaboost_demo_model.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()]).reshape(x_mesh.shape)
    plt.contour(x_mesh, y_mesh, tmp_z, zorder=1000, colors='orange')
    for i in X_mat['y'].unique():
        plt.scatter(X_mat[X_mat['y'] == i]['word_freq_george'],
                X_mat[X_mat['y'] == i]['capital_run_length_total'],
                alpha=.7, color='grey', zorder=100)
    plt.title('AdaBoost Bounds')
    plt.tight_layout()






from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 2*X[:, 0]**2 + 0.05 * np.random.randn(100)

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

def plot_predictions_gboost(regressors, X, y, axes, label=None, line_style=['tomato', '--'], data_style=['dodgerblue', '.'], data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style[1], color=data_style[0], label=data_label)
    plt.plot(x1, y_pred, color=line_style[0], linestyle=line_style[1], linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center")
    plt.axis(axes)

def gboost_stage_one():
    plt.subplot(1, 2, 1)
    plot_predictions_gboost([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", line_style=['forestgreen', '-'], data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Predicción en los residuos")
    plt.subplot(1, 2, 2)
    plot_predictions_gboost([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
    plt.ylabel("$y$", rotation=0)
    plt.title("Predicción GBoost")
    
def gboost_stage_two():
    plt.subplot(1, 2, 1)
    plot_predictions_gboost([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", line_style=['forestgreen', '-.'], data_style=['slategrey', '.'], data_label="Residuos")
    plt.ylabel("$y - h_1(x_1)$")
    plt.title("Predicción en los residuos")
    plt.subplot(1, 2, 2)
    plot_predictions_gboost([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    plt.ylabel("$y$")
    plt.title("Predicción GBoost")

def gboost_stage_three():
    plt.subplot(1, 2, 1)
    plot_predictions_gboost([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", line_style=['forestgreen', '-.'], data_style=['slategrey', '.'])
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$")
    plt.xlabel("$x_1$")
    plt.title("Predicción en los residuos")
    plt.subplot(1, 2, 2)
    plot_predictions_gboost([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$")
    plt.title("Predicción GBoost")


def gboost_subsample_hyperparams():
    """TODO: Docstring for subsample_hyperparams.
    :returns: TODO

    """

    for index, value in enumerate([.1, .5, 1]):
        plt.subplot(1, 3, index + 1)
        tmp_score = plot_response_boundaries(GradientBoostingClassifier(subsample=value),
                                                 X_tr, y_tr, X_te,y_te,x_mesh,y_mesh)
        plt.title( r'subsample'+": {}\nTest Error: {}".format(value, tmp_score),fontsize=10)
        plt.xticks(());plt.yticks(());
        plt.tight_layout()


def profile_loss_function(model, X_test, y_test):
    # create empty array to hold estimators
    tmp_loss_storage = np.empty(model.n_estimators)
    # identify optimal estimator
    get_optimal_estimators = lambda x: np.where(x == np.min(x))
    # for each estimator, extract its decision function 
    for index, individual_estimate in enumerate(model.staged_decision_function(X_test)):
        # register loss function in empty array
        tmp_loss_storage[index] = model.loss_(y_test, individual_estimate)

    tmp_optimal = get_optimal_estimators(tmp_loss_storage)

    return tmp_loss_storage, tmp_optimal


def gboost_sampling_hyperparams(X_train,X_test, y_train, y_test, params=[.5, .8, 1]):
    """TODO: Docstring for gboost_sampling_hyperparams.
    :returns: TODO

    """
    colors = ['tomato', 'dodgerblue', 'purple', 'orange']
    # for each hyperparameter
    for n, i in enumerate(params):
        # train Gradient boosting
        tmp_model = GradientBoostingClassifier(n_estimators=1000, subsample=i).fit(X_train, y_train)
        # store loss function and optimal estimator
        gb_loss, optim = profile_loss_function(tmp_model, X_test, y_test)
        # plot loss function 
        plt.plot(gb_loss, label=f'Subsample:{i} = {round(np.min(gb_loss), 3)}', lw=2, color=colors[n])
        # signal optimal estimator
        plt.axvline(optim, linestyle='--',lw=1, color=colors[n])
    plt.title('1000 Estimadores\nSubsample variante')
    plt.xlabel('Estimadores')
    plt.ylabel('Test BinomialDeviance')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))


def gboost_learning_hyperparams(X_train, X_test, y_train, y_test, params=[0.01, 0.1, 0.5, 1]):
    """TODO: Docstring for gboost_learning_hyperparams.

    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :params: TODO
    :0.1: TODO
    :0.5: TODO
    :1]: TODO
    :returns: TODO

    """
    colors = ['tomato', 'dodgerblue', 'purple', 'orange']
    for n, i in enumerate(params):
        tmp_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=i).fit(X_train, y_train)
        gb_loss, optim = profile_loss_function(tmp_model, X_test, y_test)
        plt.plot(gb_loss, label=f'Learning Rate: {i} = {round(np.min(gb_loss), 3)}', color=colors[n])
        plt.axvline(optim, linestyle='--', lw=1, color=colors[n])
    plt.title('6000 Estimadores\nTasa de aprendizaje variante')
    plt.xlabel('Estimadores')
    plt.ylabel('Test BinomialDeviance')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))

def adaboost_classification_behavior(pr_list, final_pr, observation):
    """TODO: Docstring for adaboost_classification_behavior.

    :pr_list: TODO
    :final_pr: TODO
    :observation: TODO
    :returns: TODO

    """
    spec_obs = pd.DataFrame([i[observation] for i in pr_list])
    colors = ['dodgerblue', 'tomato']
    most_likely_class = np.where(final_pr[observation] == np.max(final_pr[observation]))[0][0]

    for index, (classname, serie) in enumerate(spec_obs.iteritems()):
        plt.plot(serie, 'o-', alpha=.3, lw=.5, label=f'Class: {classname}', color=colors[index])
        plt.axhline(final_pr[observation][index], color=colors[index])
    plt.legend()
    return most_likely_class

