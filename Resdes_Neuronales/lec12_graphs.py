#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: lec12_graphs.py
Author: Ignacio Soto Zamorano / Ignacio Loayza Campos
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com / ignacio1505[at]gmail[dot]com
Github: https://github.com/ignaciosotoz / https://github.com/tattoedeer
Description: Ancilliary files for Tensors and Perceptron lecture - ADL
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
# importamos de manera explícita el optimizador de Gradiente Estocástica
from keras.optimizers import SGD
#importamos de forma explícita la estructura básica
from keras.models import Sequential
# importamos de forma explícita la definición de capas densas (fully connected) 
from keras.layers import Dense

seed = hash("Desafio LATAM es lolein")%2^32



def circles(n = 2000, stddev = 0.05):
    generator = check_random_state(seed)

    linspace = np.linspace(0, 2 * np.pi, n // 2 + 1)[:-1]
    outer_circ_x = np.cos(linspace)
    outer_circ_y = np.sin(linspace)
    inner_circ_x = outer_circ_x * .3
    inner_circ_y = outer_circ_y * .3

    X = np.vstack((np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n // 2, dtype=np.intp), np.ones(n // 2, dtype = np.intp)])
    X += generator.normal(scale = stddev, size = X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = seed)
    return X_train,y_train,X_test,y_test

def plot_classifier(clf, X_train, Y_train, X_test, Y_test, model_type):
    # Generamos los parámetros de nuestro canvas
    f, axis = plt.subplots(1, 1, sharex = "col", sharey = "row", figsize = (12,8))
    # Representamos los datos de entrenamiento
    axis.scatter(X_train[:,0], X_train[:,1], s = 30, c = Y_train, zorder = 10, cmap = "autumn")
    # Representamos los datos de validación
    axis.scatter(X_test[:,0], X_test[:,1], s = 20, c = Y_test, zorder = 10, cmap ="winter")
    # generamos una grilla multidimensional
    XX, YY = np.mgrid[-2:2:200j, -2:2:200j]
    # Si el modelo es una variante de árbol
    if model_type == "tree":
        # La densidad probabilística se obtendrá de la siguiente forma
        Z = clf.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:,0]
    # si el modelo es una variante de una red neuronal artificial
    elif model_type == "ann":
        # Obtendremos las clases predichas 
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    else:
        # de lo contrario generaremos una excepción.
        raise ValueError("model type not supported")
    Z = Z.reshape(XX.shape)
    Zplot = Z >= 0.5
    axis.pcolormesh(XX, YY, Zplot, cmap = "Purples")
    axis.contour(XX, YY, Z, alpha = 1, colors = ["k", "k", "k"], linestyles = ["--", "-", "--"], levels = [-2, 0, 2])
    plt.show()


def one_layer_network(X_train, y_train, neurons = 1, input_init = "uniform", input_activation = "relu", hidden_init = "uniform", hidden_activation = "sigmoid", loss = "binary_crossentropy", verbosity = 0):

    """
    X_train: matriz de atributos de entrenamiento
    y_train: vector objetivo de entrenamiento
    input_init: inicializador de las capas de entrada
    input_activation: forma de activación (por defecto es Rectified Linear Unit)
    hidden_init: inicializador de las capas escondidas
    hidden_activation: forma de activación (por defecto es Sigmoide)
    loss: forma de obtención de la medida de pérdida, por defecto es binary_crossentropy

    """
    # Definimos una serie de capas lineales como arquitectura
    model = Sequential()
    # Añadimos una capa densa (neuronas completamente conectadas) con la cantidad de atributos en nuestra matriz de entrenamiento
    model.add(Dense(neurons, input_dim = X_train.shape[1], kernel_initializer = input_init, activation = input_activation))
    # Añadimos una capa densa con 1 neurona para representar el output
    model.add(Dense(1, kernel_initializer = hidden_init, activation = hidden_activation))
    # compilamos los elementos necesarios, implementando gradiente estocástica y midiendo exactitud de las predicciones como norma de minimización
    model.compile(optimizer = SGD(lr = 1), loss = loss, metrics = ["accuracy"])
    # entrenamos el modelo
    model.fit(X_train, y_train, epochs = 50, batch_size = 100, verbose = verbosity)
    return model

def evaluate_network(net, X_train, y_train, X_test, y_test, show_results = True):
    scores = net.evaluate(X_test, y_test)
    test_acc = scores[1]
    if show_results:
        print("\r"+ " "*60 + "\rAccuracy: %f" % test_acc)
        plot_classifier(net, X_train, y_train, X_test, y_test, "ann")
    return test_acc
