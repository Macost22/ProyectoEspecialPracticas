import pandas 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
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
import scikitplot as skplt
from sklearn.metrics import classification_report
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
X1, X_test, y1, y_test = train_test_split(X_normalized, Yn, test_size=0.4, random_state=42)
sm = SMOTE(random_state=12, ratio = 1.0)
X_train, y_train = sm.fit_sample(X1, y1)
#X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.4, random_state=1)##balancear datos con SMOTE

##Convertir en vectores binarios y_train y y_test






 
model = DecisionTreeClassifier(criterion='entropy')
model.fitted = model.fit(X_train, y_train)
model.predictions = model.fitted.predict(X_test)
y_probas = model.fitted.predict_proba(X_test)
 
print(confusion_matrix(y_test, model.predictions))
print(accuracy_score(y_test, model.predictions))
 
predicted = cross_validation.cross_val_predict(model, X_normalized, Yn, cv=10)
print(accuracy_score(Yn , predicted))


skplt.metrics.plot_confusion_matrix(y_test, model.predictions, title='Matriz de confusión', figsize=(13,13),cmap='Blues')
pyplot.show()

report = classification_report(y_test, model.predictions)
print(report)
#, figsize=(7,7)
skplt.metrics.plot_roc(y_test, y_probas, figsize=(14,14))
pyplot.show()

