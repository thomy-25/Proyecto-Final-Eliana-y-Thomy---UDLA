# Proyecto-Final-Eliana-y-Thomy---UDLA
Proyecto final de grado correspondiente a la maestría en Inteligencia de Negocios y Ciencia de Datos. El Objetivo es desarrollar un modelo de aprendizaje automático capaz de estimar el riesgo sistémico del Banco de Loja en función de variables macroeconómicas del Ecuador.
# Proyecto de Análisis de Riesgo Crediticio Sistémico

Este repositorio contiene el código fuente del proyecto de análisis de riesgo crediticio sistémico en el sistema financiero ecuatoriano, basado en indicadores macroeconómicos y aplicado al Banco de Loja. El análisis incluye modelos de aprendizaje automático supervisado no paramétricos tales como knn, árbol de decisión y redes neuronales perceptrón multicapa (MLP)

## Archivos

- `Proyecto_revisado.ipynb`: Notebook principal con el desarrollo del análisis.
- `base_datos_extendida.xlsx`: Conjunto de datos utilizado para entrenamiento y evaluación de los modelos. Notar que es un archivo .xlsx y no .csv

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes bibliotecas (no es necesario instalarlas previamente, dentro del código hay líneas específicas que importan estas librerías):

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


Este proyecto fue desarrollado en Google Colab, por lo que, para poder replicarlo adecuadamente, se recomienda hacer eso de esta herramienta, a fin de evitar errores de reproducción. Asegúrate de adaptar las rutas de los archivos correctamente.

## Instrucciones para Ejecutar el Proyecto en Google Colabs

1. **Subir los archivos** `Proyecto_revisado.ipynb` y `base_datos_extendida.xlsx` a una carpeta guardada en Google Drive.
2. Abrir el archivo "Proyecto_revisado.ipynb" con Google Colab.
3. Una vez dentro del código en Google Colab, lo primero que se debe ejecutar, previo a cualquier otra celda, es la celda #2 correspondiente a **Montar Google Drive**.

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Proporciona los permisos de acceso, esperar a que se conecten las aplicaciones. 
5. Ejecutar la celda 3 que contiene las librerías necesarias para ejecutar el código. 
6. **Cargar el archivo Excel** con pandas:

```python
df = pd.read_excel('/content/drive/MyDrive/ruta_al_archivo/base_datos_extendida.xlsx')
```

NOTA: RECUERDA, ES UN ARCHIVO .xlsx. Asegúrate de revisar esto antes de ejecutar. 

7. **Ejecutar todas las celdas del notebook** siguiendo el orden original para:
   - Procesar los datos.
   - Aplicar técnicas de limpieza, discretización y normalización.
   - Entrenar modelos de aprendizaje automático.
   - Evaluar y visualizar métricas de rendimiento. 
   - Optimizar los modelos.

## Resultados Esperados

El proyecto compara el desempeño de tres modelos predictivos en función de varias bases de datos derivadas de la inicial sobre las que se aplicaron diversas combinaciones de tratamiento de datos. Al final se generan gráficos de evaluación y predicciones sobre los datos test.

## Licencia

Este proyecto se publica bajo una licencia académica para fines educativos.

## Autores
Eliana Marisabel Herrera Diaz
Thomy Enrique Cedeño Bazurto
