#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

plt.style.use("seaborn")

def plot_classification_report(y_true, y_hat):
    """
    plot_classification_report: Genera una visualización de los puntajes reportados con la función `sklearn.metrics.classification_report`.

    Parámetros de ingreso:
        - y_true: Un vector objetivo de validación.
        - y_hat: Un vector objetivo estimado en función a la matriz de atributos de validación y un modelo entrenado.

    Retorno:
        - Un gráfico generado con matplotlib.pyplot

    """
    colors = ['dodgerblue', 'tomato', 'purple', 'orange']
    avg_p, avg_r, avg_f1 = 0, 0, 0
    class_labels = np.unique(y_true)
    
    for i in class_labels:
        p = precision_score(y_true, y_hat, pos_label=i)
        r = recall_score(y_true, y_hat, pos_label=i)
        f = f1_score(y_true, y_hat, pos_label=i)
        avg_p += p
        avg_r += r
        avg_f1 += f
        plt.scatter(p, 1, marker='x', color=colors[i])
        plt.scatter(r, 2, marker='x', color=colors[i])
        plt.scatter(f, 3, marker='x',color=colors[i], label=f'Class: {i}')
        
    avg_p /= len(class_labels)
    avg_r /= len(class_labels)
    avg_f1 /= len(class_labels)

    plt.scatter([avg_p, avg_r, avg_f1], [1, 2, 3], marker='o', color='forestgreen', label='Avg')
    plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])


def grid_plot_batch(df, cols, plot_type):

    """
    grid_plot_batch: Genera una grilla matplotlib para cada conjunto de variables.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame
        - cols: cantidad de columnas en la grilla.
        - plot_type: tipo de gráfico a generar. Puede ser una instrucción genérica de matplotlib o seaborn.

    Retorno:
        - Una grilla generada con plt.subplots y las instrucciones dentro de cada celda.

    """
    # calcular un aproximado a la cantidad de filas
    rows = int(np.ceil(df.shape[1] / cols))

    # para cada columna
    for index, colname in enumerate(df.columns):
        plt.subplot(rows, cols, index + 1)
        
        if "seaborn" in plot_type.__module__:
            plot_type(df, x=colname)
        elif "matplotlib" in plot_type.__module__:
            plot_type(df[colname])
        plt.tight_layout()

def identify_high_correlations(df, threshold=.7):
    """
    identify_high_correlations: Genera un reporte sobre las correlaciones existentes entre variables, condicional a un nivel arbitrario.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame, por lo general es la base de datos a trabajar.
        - threshold: Nivel de correlaciones a considerar como altas. Por defecto es .7.

    Retorno:
        - Un pd.DataFrame con los nombres de las variables y sus correlaciones
    """

    # extraemos la matriz de correlación con una máscara booleana
    tmp = df.corr().mask(abs(df.corr()) < .7, df)
    # convertimos a long format
    tmp = pd.melt(tmp)
    # agregamos una columna extra que nos facilitará los cruces entre variables
    tmp['var2'] = list(df.columns) * len(df.columns)
    # reordenamos
    tmp = tmp[['variable', 'var2', 'value']].dropna()
    # eliminamos valores duplicados
    tmp = tmp[tmp['value'].duplicated()]
    # eliminamos variables con valores de 1 
    return tmp[tmp['value'] < 1.00]

def plot_roc(model, y_true, X_test, model_label=None):
    """TODO: Docstring for plot_roc.

    :model: TODO
    :y_true: TODO
    :X_test: TODO
    :model_label: TODO
    :returns: TODO

    """
    class_pred = model.predict_proba(X_test)[:1]
    false_positive_rates, true_positive_rates, _ = roc_curve(y_true, class_pred)
    store_auc = auc(false_positive_rates, true_positive_rate)

    if model_label is not None:
        tmp_label = f'{model_label}: {round(store_auc, 3)}'
    else:
        tmp_label = None
    plt.plot(false_positive_rates, true_positive_rates, label=tmp_label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def grid_plot_batch_def(df, cols, plot_type):

    """
    grid_plot_batch: Genera una grilla matplotlib para cada conjunto de variables.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame
        - cols: cantidad de columnas en la grilla.
        - plot_type: tipo de gráfico a generar. Puede ser una instrucción genérica de matplotlib o seaborn.

    Retorno:
        - Una grilla generada con plt.subplots y las instrucciones dentro de cada celda.

    """
    # calcular un aproximado a la cantidad de filas
    rows = np.ceil(df.shape[1] / cols)

    # para cada columna
    for index, (colname, serie) in enumerate(df.iteritems()):
        plt.subplot(int(rows), cols, index + 1)
        plot_type(serie)
        plt.tight_layout()



def weighting_schedule(voting_ensemble, X_train, X_test, y_train, y_test, weights_dict, plot_scheme=True, plot_performance=True):
    """TODO: Docstring for weighting_schedule.

    :voting_ensemble: TODO
    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :weights_dict: TODO
    :plot_scheme: TODO
    :plot_performance: TODO
    :returns: TODO

    """

    def weight_scheme():
        """TODO: Docstring for weight_scheme.
        :returns: TODO

        """
        weights = pd.DataFrame(weights_dict)
        weights['model'] = [i[0] for i in voting_ensemble.estimators]
        weights = weights.set_index('model')
        sns.heatmap(weights, annot=True, cmap='Blues', cbar=False)
        plt.title('Esquema de Ponderación')

    def weight_performance():
        """TODO: Docstring for weight_performance.
        :returns: TODO

        """

        n_scheme = len(weights_dict)
        f1_metrics, accuracy = [], []
        f1_metrics_train, accuracy_train = [], []

        for i in weights_dict:
            model = voting_ensemble.set_params(weights=weights_dict[i]).fit(X_train, y_train)
            tmp_model_yhat = model.predict(X_test)
            tmp_model_yhat_train = model.predict(X_train)
            f1_metrics.append(f1_score(y_test, tmp_model_yhat).round(3))
            f1_metrics_train.append(f1_score(y_train, tmp_model_yhat_train).round(3))
            accuracy.append(accuracy_score(y_test, tmp_model_yhat).round(3))
            accuracy_train.append(accuracy_score(y_train, tmp_model_yhat_train).round(3))
        plt.plot(range(n_scheme), accuracy, 'o', color='tomato', alpha=.5, label='Exactitud-Test')
        plt.plot(range(n_scheme), f1_metrics, 'x', color='tomato', alpha=.5, label='F1-Test')
        plt.plot(range(n_scheme), accuracy_train, 'o', color='dodgerblue', alpha=.5, label='Exactitud-Train')
        plt.plot(range(n_scheme), f1_metrics_train, 'x', color='dodgerblue', alpha=.5, label='F1-Train')
        plt.xticks(ticks=range(n_scheme), labels=list(weights_dict.keys()), rotation=90)
        plt.title('Desempeño en Train/Test')
        plt.legend(loc='center left', bbox_to_anchor=(1, .5))


    if plot_scheme is True and plot_performance is True:
        plt.subplot(1, 2, 1)
        weight_scheme()
        plt.subplot(1, 2, 2)
        weight_performance()

    else:
        if plot_scheme is True:
            weight_scheme()
        elif plot_performance is True:
            weight_performance()


def committee_voting(voting_ensemble, X_train, X_test, y_train, y_test):
    """TODO: Docstring for committee_voting.

    :voting_ensemble: TODO
    :returns: TODO

    """
    # iniciar dataframe vacio para guardar valores
    individual_preds = pd.DataFrame()
    # preservamos la lista de tuplas
    voting_estimators = voting_ensemble.estimators
    # para cada iterador en la lista de tuplas
    for i in voting_estimators:
        # generamos la estimación específica
        individual_preds[i[0]] = i[1].fit(X_train, y_train).predict(X_test)
    # extraemos los votos individuales de cada clasificador
    individual_preds['votes_n'] = individual_preds.loc[:, voting_estimators[0][0]:voting_estimators[-1][0]].apply(np.sum, axis=1)
    # generamos la predicción del ensamble heterogéneo
    individual_preds['Majority'] = voting_ensemble.set_params(weights=None).predict(X_test)

    # iniciamos un contenedor vacío
    tmp_holder = pd.DataFrame()
    # buscamos el cruce entre cada predicción existente a nivel de modelo
    for i in np.unique(individual_preds['votes_n']):
        # y la predicción a nivel de comité
        for j in np.unique(individual_preds['Majority']):
            # separamos los casos que cumplan con ambas condiciones
            tmp_subset = individual_preds[np.logical_and(
                individual_preds['votes_n'] == i,
                individual_preds['Majority'] == j
            )]
            # extraemos la cantidad de casos existentes
            tmp_rows_n = tmp_subset.shape[0]
            # Si la cantidad de casos existentes es mayor a cero
            if tmp_rows_n > 0:
                # registramos la importancia del clasificador RESPECTO A LA CANTIDAD DE CASOS EXISTENTES.
                tmp_holder[f'Votes: {i} / Class: {j}'] = round(tmp_subset.apply(sum) / tmp_rows_n, 3)
    # transpose
    tmp_holder = tmp_holder.T
    # Eliminamos columnas redundantes del dataframe
    tmp_holder = tmp_holder.drop(columns=['votes_n', 'Majority'])
    # visualizamos la matriz resultante
    sns.heatmap(tmp_holder, annot=True, cmap='coolwarm_r', cbar=False)