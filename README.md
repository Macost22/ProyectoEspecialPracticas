# ProyectoEspecialPracticas
Implementación de algoritmos de aprendizaje automático para reconocimiento de gestos dinámicos de las manos

## ExportData.m
Permite organizar los datos en formato CSV y eliminar los "REPOS" de la base de datos LMDHG, tomada de: https://www-intuidoc.irisa.fr/english-leap-motion-dynamic-hand-gesture-lmdhg-database/

## DataFile1.mat
Datos utilizados para organizar la base de datos.

## DatabaseNorepos012.csv
Base de datos final generada en MATLAB con el cambio de variables categórica a numérico, cuenta con 13 clases y se eliminaron las que corresponden a estados de Reposición.

## DataAnalisisPrepro.py
Corresponde al codigo en el que se implementó análisis estadístico, visualización y preprocesamiento de la base de datos DAtabaseNorepos.

## NNneuronas.py
Implementación en python de una red neuronal en la que se puede cambiar el numero de neuronas y evaluar la precisión del modelo con datos de prueba.

## NNcapas.py
Implementación en python de una red neuronal en la que se puede cambiar el numero de capas y evaluar la precisión del modelo con datos de prueba.

## MLPfromScratch.py
Implementación python de una perceptron multicapa generalizable sin librerías para clasificación multiclase.

## ArbolBosqueSVMKNN.py
Implementación de Arbol de decisión, bosque aleatorio, máquina de vector de soporte y k vecino más cercano  ṕara clasificar la base de datos LMDHG y generar métricas para validación del modelo.




