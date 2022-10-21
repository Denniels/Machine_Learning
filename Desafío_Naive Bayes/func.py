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
import warnings
warnings.filterwarnings(action="ignore")
colors = ["tomato", "darkgoldenrod", "limegreen", "dodgerblue", "sienna", "slategray"]


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
    fig.suptitle("Distribución para "+ Name  , fontsize=18, fontweight='bold')
    sns.boxplot(x=data,showmeans=True, orient='h',color="tan",ax=ax_box)
    ax_box.set(xlabel='')

    sns.despine(top=True,right=True,left=True) 
    sns.distplot(data,kde=False,color='red',ax=ax_dis)
    ax_dis.axvline(mean, color='r', linestyle='--',linewidth=2)
    ax_dis.axvline(median, color='g', linestyle='-',linewidth=2)
    plt.legend({'Media':mean,'Mediana':median})

    