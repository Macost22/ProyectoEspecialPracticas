#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:06:56 2019

@author: bessel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:46:47 2019

@author: bessel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:27:13 2019

@author: bessel
"""

# Visualize training history
# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import plot_model
from matplotlib import pyplot
import numpy as np
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from keras.optimizers import SGD
from keras.utils import to_categorical


# Cargar los datos
dataframe = pandas.read_csv("/home/bessel/Desktop/DatosIA/DatabaseNoRepos012.csv", header=0)
 
#Encontrar correlación de los datos mediante el metodo Pearson
Correlacion_data= dataframe.corr(method="pearson")

#Graficar matriz de correlación de los datos
corrmat = dataframe.corr(method='spearman') 
f, ax = pyplot.subplots(figsize =(19, 15)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 


#Determinar si los datos se encuentran balancedos
count_class= dataframe.groupby('Clase').size()

#Descripción estadistica de los datos
Describe_Data= dataframe.describe()

#Inclinacion de la distribución Gaussiana
Inclinacion = dataframe.skew()


##Distribución

g = sns.pairplot(dataframe, height=3, diag_kind="kde", vars=["feature63", "Clase"])
#dataframe.hist(column='feature135', color='steelblue', edgecolor='black', linewidth=1.0,
#           xlabelsize=8, ylabelsize=8, grid=False)    
#pyplot.tight_layout(rect=(0, 0, 1.2, 1.2)) 
##
#dataframe.hist(column='feature66', color='steelblue', edgecolor='black', linewidth=1.0,
#           xlabelsize=8, ylabelsize=8, grid=False)    
#pyplot.tight_layout(rect=(0, 0, 1.2, 1.2)) 

#dataframe.plot(kind='density', subplots=True, layout=(139,139), sharex=False)
#pyplot.show()


#g = sns.pairplot(dataframe, hue="Clase")
#g = sns.pairplot(dataframe, height=3, diag_kind="kde", vars=["feature66", "feature135",  "feature120"])


#Cajas
dataframe.plot(kind = 'box', subplots = True, layout = (139,139), sharex = False,sharey = False)
pyplot.show()

#Separa los datos que corresponden a las características y a las Etiquetas
dataset = dataframe.values
X = dataset[0:8330,0:138].astype(float)
Yn = dataset[0:8330:,138]
Y = np_utils.to_categorical(Yn)
scaler = Normalizer('l2').fit(X)
X_normalized = scaler.transform(X)

#Separar los datos entrenamiento y validación 60-40
#Separar los datos entrenamiento y validación 60-40
X_train, X_test, y_train, y_test = train_test_split(X_normalized, Yn, test_size=0.4, random_state=42)

##balancear datos con SMOTE
sm = SMOTE(random_state=12, ratio = 1.0)
X_train1, Y_train1 = sm.fit_sample(X_train, y_train)
##Convertir en vectores binarios y_train y y_test
y_train1 = np_utils.to_categorical(Y_train1)
y_test1 = np_utils.to_categorical(y_test)


#Balancear datos con UNDERSAMPLING

#print(sorted(Counter(y_train).items()))
cc = ClusterCentroids(random_state=0)
X_train2, Y_train2 = cc.fit_sample(X_train, y_train)
#print(sorted(Counter(y_resampled).items()) 
y_train2 = np_utils.to_categorical(Y_train2)
y_test2 = np_utils.to_categorical(y_test)