#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: func.py
Author: Daniel Mardones
Email: daniel[dot]mardones[dot]s[at]gmail[dot]com
Github: https://github.com/Denniels
Description: Funciones para Desafío - Expansiones basales
"""

#import argparse
#import time
#import os
#from collections import Counter
#import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import KFold
import sklearn.linear_model as lm
#from scipy.stats import norm
#from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import warnings
warnings.filterwarnings(action="ignore")
colors = ["tomato", "darkgoldenrod", "limegreen", "dodgerblue", "sienna", "slategray"]

def summary_drop(data):

    """Funcion que entrega un summary de un dataframe completo.
    Args:
        data (DataFrame): Conjunto de datos
    Returns:
        pd.DataFrame: con todas las variables del df
    """
    tipos = pd.DataFrame({'tipo': data.dtypes},index=data.columns)
    na = pd.DataFrame({'nulos': data.isna().sum()}, index=data.columns)
    na_prop = pd.DataFrame({'nulos_porces':data.isna().sum()/data.shape[0]}, index=data.columns)
    ceros = pd.DataFrame({'ceros':[data.loc[data[col]==0,col].shape[0] for col in data.columns]}, index= data.columns)
    ceros_prop = pd.DataFrame({'ceros_porces':[data.loc[data[col]==0,col].shape[0]/data.shape[0] for col in data.columns]}, index= data.columns)
    summary = data.describe(include='all').T

    summary['dist_IQR'] = summary['75%'] - summary['25%']
    summary['limit_inf'] = summary['25%'] - summary['dist_IQR']*1.5
    summary['limit_sup'] = summary['75%'] + summary['dist_IQR']*1.5

    summary['outliers'] = data.apply(lambda x: sum(np.where((x<summary['limit_inf'][x.name]) | (x>summary['limit_sup'][x.name]),1 ,0)) if x.name in summary['limit_inf'].dropna().index else 0)

    return pd.concat([tipos, na, na_prop, ceros, ceros_prop, summary], axis=1).sort_values('tipo')

def form_model(df, var_obj):
    """Modelo logit con todos sus atributos.
    Args:
        df (dataframe): Conjunto de datos
        var_obj (string): variable objetivo
    Returns:
        string: formula del modelo
    """
    base_formula = f'{var_obj} ~ '
    for col in df.columns:
        if col != var_obj:
            base_formula += f'{col} + '
    return base_formula[:-3]

def predict(df, var_obj):
    """Función que automatiza las predicciones por LogisticRegression.

    Args:
        df (dataframe): dataframe con todas las variables a introducir 
        en el modelo, incluida la V.O.
        var_obj (str): variable objetivo

    Returns:
            array: vector de prueba y vector de predicciones
    """
    # separando matriz de atributos de vector objetivo
    # utilizamos dataframe con variables significativas
    mat_atr = df.drop(var_obj, axis=1)
    vec_obj = df[var_obj]
    # split de conjuntos de entrenamiento vs prueba
    X_train, X_test, y_train, y_test = train_test_split(mat_atr, vec_obj, test_size = .33, random_state = 15820)
    # estandarizamos conjunto de entrenamiento
    X_train_std = StandardScaler().fit_transform(X_train)
    X_test_std = StandardScaler().fit_transform(X_test)
    # ajustamos modelo sin alterar hiperparámetros
    modelo_x =  LogisticRegression().fit(X_train_std, y_train)
    # prediccion de clases y probabilidad
    y_hat = modelo_x.predict(X_test_std)
    return modelo_x, y_test, y_hat

def report_scores(y_predict, y_validate):
    """Calcula el error cuadrático medio y el r2 score entre dos vectores. 
    El primero, el vector de valores predecidos por el
    conjunto de prueba, y el segundo, el vector objetivo original.

    Args:
        y_predict (vector): vector de valores predecidos
        y_validate (vector): vector de valores verdaderos
    """
    mse = mean_squared_error(y_validate, y_predict)
    r2 = r2_score(y_validate, y_predict).round(2)
    print(f'Error cuadrático medio: {mse}')
    print(f'R2: {r2}')

def dist_box(data):
    """Funcion que imprime grafico de las distribuciones de un 
    dataframe completo.
    Args:
        data (DataFrame): Conjunto de datos
    Returns:
        distplots: con todas las variables del df
    """
    Name=data.name.upper()
    fig,(ax_box,ax_dis) = plt.subplots(nrows=2,sharex=True,gridspec_kw = {"height_ratios": (.25, .75)},figsize=(8, 5))
    mean=data.mean()
    median=data.median()
    mode=data.mode().tolist()[0]
    sns.set_theme(style="white")
    fig.suptitle("Distribucion " + Name, fontsize=18, fontweight='bold')#+ Name
    sns.boxplot(x=data,showmeans=True, orient='h',color="tan",ax=ax_box)
    ax_box.set(xlabel='')

    sns.despine(top=True,right=True,left=True) 
    sns.distplot(data,kde=False,color='red',ax=ax_dis)
    ax_dis.axvline(mean, color='r', linestyle='--',linewidth=2)
    ax_dis.axvline(median, color='g', linestyle='-',linewidth=2)
    plt.legend({'Media':mean,'Mediana':median})

def plot_hist(df, var):
    """Funcion que imprime grafico de las distribuciones de un 
    dataframe completo.
    Args:
        data (DataFrame): Conjunto de datos
    Returns:
        histplots: con todas las variables del df
    """
    df[var].hist()
    
    plt.axvline(df[var].mean(), label = 'Media', color = 'orange')
    plt.axvline(np.median(df[var]), label = 'Mediana', color = 'green')

    plt.legend()
    plt.title(var)
    plt.show()

def polynomial_degrees(m=50):
    """TODO: Docstring for polynomial_degrees.
    :returns: TODO

    """
    scatter_kws = {'color':'slategrey'}
    line_kws= {'color': 'tomato', 'linewidth': 3}
    np.random.seed(11238)
    X_mat = 3 * np.random.rand(m, 1) / 3.0
    y = 1 - 3 * X_mat + np.random.randn(m, 1) / 1

    plt.subplot(2, 3, 1)
    sns.regplot(X_mat[:, 0], y[:, 0], order=1, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    y_lim = plt.ylim()
    plt.title(r'$y = \beta_{0} + \beta_{1}X_{1} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 2)
    sns.regplot(X_mat[:, 0], y[:, 0], order=3, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{3} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)


    plt.subplot(2, 3, 3)
    sns.regplot(X_mat[:, 0], y[:, 0], order=5, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{5} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 4)
    sns.regplot(X_mat[:, 0], y[:, 0], order=7, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{7} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 5)
    sns.regplot(X_mat[:, 0], y[:, 0], order=10, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{10} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 6)
    sns.regplot(X_mat[:, 0], y[:, 0], order=20, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{20} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)
    plt.tight_layout()

def cv_error(x_train,y_train,k, method = 'OLS', alpha = 1):
    """Funcion que calcula el error de la validacion cruzada.
        Args:
        x_train: Matris de atributos 
        y_train: vector objetivo
        k: Kfolds
        method: False = 'OLS', modelo lineal 'OLS', 'ridge', 'lasso', 'enet' 
        Returns:
            r2_score, mean_squared_error, median_absolute_error
    """

    # Entrnar matris de atributos y vector objetivo
    Xm, ym = x_train.to_numpy(), y_train.to_numpy()

    #Definir KFold division manual
    kf = KFold(n_splits=k)
    # set rmse a 0
    rmse_cv = 0
    # cueficientes
    coef_v = []

    # Seleccion del modelo
    if method is 'OLS':
        method_type = lm.LinearRegression(fit_intercept=False)
    elif method is 'ridge':
        method_type = lm.Ridge(alpha=alpha, fit_intercept=False)
    elif method is 'lasso':
        method_type = lm.Lasso(alpha = alpha, fit_intercept=False)
    elif method is 'enet':
        method_type = lm.ElasticNet(alpha=alpha, fit_intercept=False)
    else:
        # error de argumento no valido
        raise TypeError("Argumento 'method' no valido")

    # para cada variable
    for train_index, validation_index in kf.split(Xm):
        # instaciar modelo
        method_type = method_type

        method_type.fit(Xm[train_index], ym[train_index])

        coef_v.append(method_type.coef_)
        # predicciones
        yhat_validation = method_type.predict(Xm[validation_index])
        # actualizar metricas
        rmse_cv += np.mean(np.power(yhat_validation - ym[validation_index], 2))

    # crear dataframe de coeficientes
    coefs_dataframe = pd.DataFrame(
        # fpara cada matriz interna, reorganizar los coeficientes
        [[fold[x] for x in range(x_train.shape[1])] for fold in coef_v]
    )

    coefs_dataframe.columns = list(x_train.columns)
    return coefs_dataframe, rmse_cv

def report_metrics(model, x_test, y_test):
    """Funcion que imprime las metricas de un modelo.
        Args:
        model (modelo): modelo lineal
        Returns:
            r2_score, mean_squared_error, median_absolute_error
    """
    preds = model.predict(x_test)
    print(f'''
    Test R2: {r2_score(y_test, preds)}
    Test MSE: {mean_squared_error(y_test, preds)}
    Test Median Absolute Error: {median_absolute_error(y_test, preds)}''')