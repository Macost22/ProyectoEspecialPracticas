#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:27:13 2019

@author: bessel
"""
#Tomado de: https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/
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

#Separa los datos que corresponden a las características y a las Etiquetas
dataset = dataframe.values
X = dataset[0:8330,0:138].astype(float)
Yn = dataset[0:8330:,138]
Y = np_utils.to_categorical(Yn)
scaler = Normalizer('l2').fit(X)
X_normalized = scaler.transform(X)

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
X_train2, y_train2 = cc.fit_sample(X_train, y_train)
#print(sorted(Counter(y_resampled).items()) 
y_train2 = np_utils.to_categorical(y_train2)
y_test2 = np_utils.to_categorical(y_test)
 
# define baseline model

# study of mlp learning curves given different number of nodes for multi-class classification




# fit model with given number of nodes, returns test set accuracy
def evaluate_model(n_nodes, X_train1, y_train1, X_test, y_test1):
	# configure the model based on the data
	n_input, n_classes = X_train1.shape[1], y_train1.shape[1]
	# define model
	model = Sequential()
	model.add(Dense(n_nodes, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(n_classes, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# fit model on train set
	history = model.fit(X_train1, y_train1, epochs=100, verbose=0)
	# evaluate model on test set
	_, test_acc = model.evaluate(X_test, y_test1, verbose=0)
	return history, test_acc

# prepare dataset

# evaluate model and plot learning curve with given number of nodes
num_nodes = [5, 10, 15, 20, 25]
for n_nodes in num_nodes:
	# evaluate model with a given number of nodes
	history, result = evaluate_model(n_nodes, X_train1, y_train1, X_test, y_test1)
	# summarize final test set accuracy
	print('nodes=%d: %.3f' % (n_nodes, result))
	# plot learning curve
	pyplot.plot(history.history['loss'], label=str(n_nodes))

    
# show the plot
pyplot.ylabel("Pérdida")
pyplot.xlabel("Época")
pyplot.legend()
pyplot.show()