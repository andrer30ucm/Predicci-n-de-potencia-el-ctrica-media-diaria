"""IMPORTAMOS PAQUETES"""
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import math
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import IPython
import IPython.display

from IPython.display import display
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import boxcox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.moment_helpers import cov2corr

from IPython.display import display

"""TESTEAMOS LA DETECCION DE LA GPU"""
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')
    
"""Este archivo contiene 2075259 mediciones recogidas en una casa ubicada en 
Sceaux (7 km de París, Francia) entre diciembre de 2006 y noviembre de 2010 
(47 meses).
Notas: 
1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3)
representa la energía activa consumida cada minuto (en vatios hora) en el hogar
por los equipos eléctricos no medidos en las submedidas 1, 2 y 3.
2.El conjunto de datos contiene algunos valores faltantes en las mediciones 
(casi el 1,25% de las filas). Todas las marcas de tiempo del calendario están
presentes en el conjunto de datos, pero para algunas marcas de tiempo, faltan
los valores de medición: un valor faltante está representado por la ausencia de
valor entre dos separadores de atributos de punto y coma consecutivos. 
Por ejemplo, el conjunto de datos muestra valores faltantes el 28 de abril 
de 2007.

1.fecha: Fecha en formato dd/mm/aaaa
2.hora: hora en formato hh:mm:ss
3.global_active_power: potencia activa global promedio por minuto del hogar 
(en kilovatios)
4.global_reactive_power: potencia reactiva promedio por minuto global del 
hogar (en kilovatios)
5.voltaje: voltaje promedio por minuto (en voltios)
6.global_intensity: intensidad de corriente promedio por minuto global del 
hogar (en amperios)
7.sub_metering_1: submedición de energía N°1 (en vatios-hora de energía activa).
 Corresponde a la cocina, que contiene principalmente lavavajillas, horno y 
 microondas (los fogones no son eléctricos sino de gas).
8.sub_metering_2: submedición de energía N° 2 (en vatios-hora de energía 
 activa). Corresponde al lavadero, que contiene lavadora, secadora, frigorífico
 y luz.
9.sub_metering_3: submedición de energía nº 3 (en vatios-hora de energía activa).
 Corresponde a un termo eléctrico y a un aire acondicionado.
"""  


"""IMPORTAMOS LA TABLA DE LA SERIE, CSV"""
tabla = pd.read_csv(("C:/Users/Usuario/Desktop/Asignaturas Máster/TFM/"
                       "Ejemplos-Series Temporales/"
                       "household_power_consumption.txt"),
                      index_col=None, na_values=["?"], delimiter=';',
                      dtype={'Global_active_power': float, 
                             'Global_reactive_power': float, 'Voltage': float, 
                             'Global_intensity': float, 'Sub_metering_1': float,
                             'Sub_metering_2': float, 'Sub_metering_3': float})

# Convertir las columnas de fecha y hora a formato datetime
tabla.loc[:, 'Date'] = pd.to_datetime(tabla['Date'], dayfirst=True)
tabla.loc[:, 'Time'] = pd.to_datetime(tabla['Time']).dt.time

# Establecer la nueva columna como el índice del DataFrame
tabla.set_index('Date', inplace=True)

"""TRATAMOS MISSING PROPAGANDO VALORES HACIA DELANTE"""
tabla = tabla.ffill()

"""CREAMOS LA TABLA DIARIA CON LA MEDIA DE CADA DIA"""
tabla = tabla.drop('Time', axis=1)
tabla_diaria = tabla.groupby(tabla.index).mean()

"""TRABAJAMOS CON LA TABLA DIARIA"""
n = len(tabla_diaria)
train = tabla_diaria[0:int(n*0.8)]

"""DIBUJAMOS LA TABLA"""
tabla_diaria.plot(y='Global_active_power')

# %% DEFINICION DE FUNCIONES PARA ARIMA

"""BOX-JENKINS, ARIMA"""

"""FUNCIONES Y RESULTADOS DEL AJUSTE DE LOS MODELOS"""
def resumen(modelo, target):
    
    display(modelo.summary())
    covs = modelo.cov_params()
    display(pd.DataFrame(cov2corr(covs), columns = covs.columns, index = covs.index))
    fig, axarray = plt.subplots(1,2)
    acf = plot_acf(modelo.resid, ax = axarray[0], lags = 35)
    pacf = plot_pacf(modelo.resid, ax = axarray[1], lags = 35, method = "ols")
    
    fig, axarray = plt.subplots(1,2)
    acf2 = plot_acf(modelo.resid, ax = axarray[0], lags = 35)
    pacf2 = plot_pacf(modelo.resid, ax = axarray[1], lags = 35, method = "ols")

    
    testRB = acorr_ljungbox(modelo.resid, lags = 35)
    print(testRB)

# ======================================================================= 
"""TRANSFORMACION BOX-COX"""
bxcx = boxcox(train['Global_active_power'])
lbd = bxcx[1] #lambda = 0.6601 HACEMOS RAIZ CUADRADA
train.loc[:, ['Sqrt_GAP']] = np.sqrt(train.loc[:, ['Global_active_power']
                                               ]).values.tolist()
tabla_diaria.loc[:, ['Sqrt_GAP']] = \
    np.sqrt(tabla_diaria.loc[:, ['Global_active_power']]).values.tolist()
# =======================================================================

# =======================================================================
"""ESTACIONARIEDAD EN MEDIA: DICKEY-FULLER"""
# Test de estacionariedad
result = adfuller(train.Sqrt_GAP)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# ADF Statistic: -3.335003
# p-value: 0.013385
# Critical Values:
# 	1%: -3.436
# 	5%: -2.864
# 	10%: -2.568
# =======================================================================

# =======================================================================
"""ACF Y PACF"""
fig, axarray = plt.subplots(1,2)
acfplot = plot_acf(train.Sqrt_GAP, ax = axarray[0], lags = 15)
pacf = plot_pacf(train.Sqrt_GAP, ax = axarray[1], lags = 15)
# =======================================================================

# =======================================================================
"""PRIMER MODELO"""
modelo = SARIMAX(train.Sqrt_GAP, trend='n', 
                  order = (1,0,0), 
                  seasonal_order = (0,0,0,0), enforce_stationarity=False).fit()
resumen(modelo, train.Sqrt_GAP)

#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1          0.9857      0.005    212.413      0.000       0.977       0.995
# sigma2         0.0301      0.001     28.271      0.000       0.028       0.032
# =======================================================================

# =======================================================================
"""SEGUNDO MODELO"""
modelo = SARIMAX(train.Sqrt_GAP, trend='n', 
                  order = (1,1,0), 
                  seasonal_order = (0,0,0,0), enforce_stationarity=False).fit()
resumen(modelo, train.Sqrt_GAP)

#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1         -0.3139      0.026    -12.221      0.000      -0.364      -0.264
# sigma2         0.0273      0.001     29.121      0.000       0.026       0.029
# =======================================================================

# =======================================================================
"""TERCER MODELO"""
modelo = SARIMAX(train.Sqrt_GAP, trend='n', 
                  order = (2,1,0), 
                  seasonal_order = (0,0,0,0), enforce_stationarity=False).fit()
resumen(modelo, train.Sqrt_GAP)

#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1         -0.4348      0.023    -18.640      0.000      -0.480      -0.389
# ar.L2         -0.3774      0.024    -15.606      0.000      -0.425      -0.330
# sigma2         0.0234      0.001     29.560      0.000       0.022       0.025
# =======================================================================

# =======================================================================
"""CUARTO MODELO"""
modelo = SARIMAX(train.Sqrt_GAP, trend='n', 
                  order = (2,1,0), 
                  seasonal_order = (0,0,1,7), enforce_stationarity=False).fit()
resumen(modelo, train.Sqrt_GAP)

#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1         -0.4339      0.024    -18.445      0.000      -0.480      -0.388
# ar.L2         -0.3518      0.024    -14.552      0.000      -0.399      -0.304
# ma.S.L7        0.1382      0.027      5.125      0.000       0.085       0.191
# sigma2         0.0225      0.001     29.793      0.000       0.021       0.024
# =======================================================================

# =======================================================================
"""QUINTO MODELO"""
modelo = SARIMAX(train.Sqrt_GAP, trend='n', 
                  order = (2,1,0), 
                  seasonal_order = (2,0,2,7), enforce_stationarity=False).fit()
resumen(modelo, train.Sqrt_GAP)

#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1         -0.4500      0.023    -19.265      0.000      -0.496      -0.404
# ar.L2         -0.3119      0.023    -13.321      0.000      -0.358      -0.266
# ar.S.L7        0.5780      0.082      7.045      0.000       0.417       0.739
# ar.S.L14       0.3796      0.079      4.799      0.000       0.225       0.535
# ma.S.L7       -0.5038      0.084     -5.971      0.000      -0.669      -0.338
# ma.S.L14      -0.3676      0.077     -4.761      0.000      -0.519      -0.216
# sigma2         0.0203      0.001     30.923      0.000       0.019       0.022
# =======================================================================


# =======================================================================
"""DataFrame DE REGRESORES"""
exogenas = pd.DataFrame({'Sub_metering_1':tabla_diaria.Sub_metering_1, 
                         'Sub_metering_2':tabla_diaria.Sub_metering_2, 
                         'Sub_metering_3':tabla_diaria.Sub_metering_3})
exogenas.loc[:, 'Global_intensity'] = tabla_diaria.Global_intensity
exogenas.loc[:, 'Global_reactive_power'] = tabla_diaria.Global_reactive_power
exogenas.loc[:, 'Voltage'] = tabla_diaria.Voltage

exogenas_train = exogenas[0:int(n*0.8)]
exogenas_test = exogenas[int(n*0.8):]

test = tabla_diaria[int(n*0.8):]

# =======================================================================
"""SEXTO MODELO. EXOGENAS"""
modelo = SARIMAX(train.Sqrt_GAP, trend='n', 
                  order = (2,1,0),
                  exog = exogenas_train,
                  seasonal_order = (2,0,2,7), enforce_stationarity=False,
                  freq='D').fit()
resumen(modelo, train.Sqrt_GAP)

#                             coef    std err          z      P>|z|      [0.025      0.975]
# -----------------------------------------------------------------------------------------
# Sub_metering_1            0.0008      0.001      1.061      0.289      -0.001       0.002
# Sub_metering_2            0.0016      0.000      3.278      0.001       0.001       0.003
# Sub_metering_3            0.0050      0.000     11.281      0.000       0.004       0.006
# Global_intensity          0.1026      0.001    119.482      0.000       0.101       0.104
# Global_reactive_power    -0.0197      0.039     -0.511      0.609      -0.095       0.056
# Voltage                   0.0030      0.001      2.910      0.004       0.001       0.005
# ar.L1                    -0.3873      0.019    -20.882      0.000      -0.424      -0.351
# ar.L2                    -0.2725      0.020    -13.594      0.000      -0.312      -0.233
# ar.S.L7                   0.1008      0.213      0.473      0.636      -0.317       0.519
# ar.S.L14                  0.1538      0.061      2.523      0.012       0.034       0.273
# ma.S.L7                  -0.0631      0.215     -0.294      0.769      -0.484       0.358
# ma.S.L14                 -0.1009      0.067     -1.516      0.130      -0.231       0.030
# sigma2                    0.0006   1.52e-05     40.849      0.000       0.001       0.001
# =======================================================================

# %%
"""SARIMA(2,1,0)X(2,0,2)_7, MEJOR MODELO SIN EXOGENAS"""
pred_acumulada = []
vector_real = []
for date in test.index:
    if date == test.index[-1]:
        break
    modelo = SARIMAX(tabla_diaria.Sqrt_GAP[:date], trend='n', 
                      order = (2,1,0), 
                      seasonal_order = (2,0,2,7), enforce_stationarity=False,
                      freq='D').fit()
    
    prediccion_ARIMA = modelo.forecast(steps=1).values # Predecimos a horizonte 1
    prediccion_ARIMA = prediccion_ARIMA[0]**2
    valor_real = tabla_diaria.Global_active_power[date + pd.Timedelta(days=1)]
    if valor_real == tabla_diaria.Global_active_power['2010-07-29 00:00:00']:
        print('valor predicho: ', prediccion_ARIMA)
        print('valor real: ', valor_real)
        print('error: ', abs(valor_real - prediccion_ARIMA))
    
    vector_real.append(valor_real)
    pred_acumulada.append(prediccion_ARIMA) # Vector de predicciones a un dia

sarimax_mape=(100*mean_absolute_percentage_error(vector_real[:],
                                                 pred_acumulada[:]))

sarimax_mse = mean_squared_error(vector_real[:], pred_acumulada[:])

plt.figure(figsize=(10, 6))
plt.plot(vector_real[:], label='Real - Test')
plt.plot( pred_acumulada[:], label='Predicción - Test', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Valores')
plt.title('Predicción vs Valores Reales')
plt.legend()
plt.show()

# %%
"""SARIMAX(2,1,0)X(2,0,2)_7, MEJOR MODELO CON EXOGENAS"""
pred_acumulada = []
vector_real = []
exog_lag1 = exogenas.copy().shift(1).bfill()
for date in test.index:
    if date == test.index[-1]:
        break
    modelo = SARIMAX(tabla_diaria.Sqrt_GAP[:date], trend='n', 
                      exog=exog_lag1[:date], order = (2,1,0), 
                      seasonal_order = (2,0,2,7), enforce_stationarity=False,
                      freq='D').fit()
    
    prediccion_ARIMA = modelo.forecast(steps=1, exog=exogenas[date:date]).values # Predecimos a horizonte 1
    prediccion_ARIMA = prediccion_ARIMA[0]**2
    
    valor_real = tabla_diaria.Global_active_power[date + pd.Timedelta(days=1)]
    
    vector_real.append(valor_real)
    pred_acumulada.append(prediccion_ARIMA) # Vector de predicciones a un dia

sarimax_mape=(100*mean_absolute_percentage_error(vector_real[:],
                                                 pred_acumulada[:]))

sarimax_mse = mean_squared_error(vector_real[:], pred_acumulada[:])

plt.figure(figsize=(10, 6))
plt.plot(vector_real[:], label='Real - Test')
plt.plot( pred_acumulada[:], label='Predicción - Test', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Valores')
plt.title('Predicción vs Valores Reales')
plt.legend()
plt.show()




# %%
"""LSTM DIARIA. DEFINICION DE FUNCIONES"""
from keras.models import Sequential
from keras.layers import Activation, Dense, Attention, Bidirectional
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten
from keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

"""FUNCION QUE GENERA FRAGMENTOS DE INPUTS DE TEST Y TRAINING DE LONGITUD WINDOW_LEN"""
def fragmentos(training_set, test_set, window_len, variable):
  with tf.device('/device:GPU:0'):
      # WINDOW_LEN ES EL TAMAÑO DE LOS FRAGMENTOS QUE SE USAN PARA EL ENTRENAMIENTO
        
      # Inicializar LSTM_training_inputs como una lista vacía
      LSTM_training_inputs = []
    
      # Recorremos el conjunto de entrenamiento para crear los fragmentos
      for i in range(len(training_set) - window_len):
          temp_set = training_set.iloc[i:(i + window_len), :].copy()  # Este es el fragmento actual
          LSTM_training_inputs += [temp_set]  # Usamos `+=` para agregar sin `append()`
        
      LSTM_training_outputs = (training_set[variable][window_len:].values)
      
      # PEGAMOS LOS DATOS DE TRAIN Y DE TEST
      training_test_set = pd.concat([training_set, test_set], axis=0, ignore_index=False)
    
      LSTM_test_inputs = []
      
      # CREAMOS LOS FRAGMENTOS DEL TEST  
      for i in range(len(test_set)):
          temp_set = training_test_set.iloc[
              len(training_set)-window_len+i:(len(training_set)+i), :].copy()
          LSTM_test_inputs += [temp_set]  # Usamos `+=` para agregar sin `append()`
        
      LSTM_test_outputs = (test_set[variable][0:].values)
      
      # CONVERTIMOS A ARRAY PARA QUE LO LEA LSTM
      LSTM_training_inputs_list_array = [np.array(dataframe) for dataframe in LSTM_training_inputs]
      LSTM_training_inputs_array = np.array(LSTM_training_inputs_list_array)
      
      LSTM_test_inputs_list_array = [np.array(dataframe) for dataframe in LSTM_test_inputs]
      LSTM_test_inputs_array = np.array(LSTM_test_inputs_list_array)
      
      # DEVUELVE ARRAYS PARA ENTRENAR LA RED Y TESTEARLA
      return LSTM_training_inputs_array, LSTM_training_outputs, \
          LSTM_test_inputs_array, LSTM_test_outputs

"""FUNCION QUE DEFINE CRITERIO DE PARADA, COMPILA Y AJUSTA EL MODELO"""
def compile_and_fit(model, epochs, LSTM_training_inputs_array, LSTM_training_outputs,
                    patience=50):
  with tf.device('/device:GPU:0'):
      # CRITERIO DE PARADA: SI NO MEJORA AL MENOS DELTA_MIN SE PARA
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min',
                                                        verbose=1,
                                                        min_delta = 0.0001,
                                                        restore_best_weights = True
                                                        )

      model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate=0.001),
                    metrics=[keras.metrics.MeanSquaredError()])
      
      # EPOCHS = NUMERO DE ITERACIONES SOBRE LOS PATRONES DE ENTRENAMIENTO
      history = model.fit(x = LSTM_training_inputs_array, y = LSTM_training_outputs, 
                          epochs=epochs, validation_split = 0.20, verbose = 1, 
                          callbacks=[early_stopping], shuffle=False)
      # SHUFFLE = FALSE PARA QUE NO DESORDENE LOS DATOS
      return history


"""FUNCION QUE DIBUJA EL HISTORICO DE ENTRENAMIENTO"""
def plot_graphs(history, metric):
  plt.figure(figsize=(10, 6))
  plt.plot(history.history[metric], label='train')
  plt.plot(history.history['val_'+metric], label='test')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend()
  plt.show()

# %%
"""RED (CNN+LSTM) CON EXOGENAS"""

# ======= DIVISION EN TRAIN Y TEST DE LA VARIABLE TARGET ======= #
train_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][:int(n*0.8)])
test_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][int(n*0.8):])
# ============================================================== #

# TRANSFORMACION RAIZ CUADRADA DE TARGET
train_rn.loc[:, ['Global_active_power']] = np.sqrt(train_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()
test_rn.loc[:, ['Global_active_power']] = np.sqrt(test_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()


# ============= DATAFRAME DE VARIABLES EXOGENAS ================ #
# DIFERENCIAMOS
exogenas_rn = exogenas.copy().diff()
exogenas_rn = exogenas_rn.bfill()

# DIVIDIMOS EN TRAIN Y TEST
n = len(exogenas_rn)
exogenas_rn_train = exogenas_rn[0:int(n*0.8)]
exogenas_rn_test = exogenas_rn[int(n*0.8):]
# ============================================================== #


# ================= DIFERENCIAMOS TRAIN Y TEST ================= #
train_rn = train_rn.diff()
train_rn = train_rn.bfill()
test_noDif = test_rn.copy()
test_rn = test_rn.diff()
test_rn = test_rn.bfill()
# ============================================================== #

# =============== ESCALAMOS CON MINMAXSCALER =================== #
scaler = MinMaxScaler(feature_range=(0, 1))

# ESCALAMOS CADA VARIABLE EXOGENA, HACIENDO FIT SOLO DE LA PARTE DE TRAIN
scaler.fit(pd.DataFrame(exogenas_rn_train.Sub_metering_1))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Sub_metering_1))
exogenas_rn_train.loc[:, 'Sub_metering_1'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Sub_metering_1))
exogenas_rn_test.loc[:, 'Sub_metering_1'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Sub_metering_2))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Sub_metering_2))
exogenas_rn_train.loc[:, 'Sub_metering_2'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Sub_metering_2))
exogenas_rn_test.loc[:, 'Sub_metering_2'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Sub_metering_3))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Sub_metering_3))
exogenas_rn_train.loc[:, 'Sub_metering_3'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Sub_metering_3))
exogenas_rn_test.loc[:, 'Sub_metering_3'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Global_intensity))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Global_intensity))
exogenas_rn_train.loc[:, 'Global_intensity'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Global_intensity))
exogenas_rn_test.loc[:, 'Global_intensity'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Global_reactive_power))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Global_reactive_power))
exogenas_rn_train.loc[:, 'Global_reactive_power'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Global_reactive_power))
exogenas_rn_test.loc[:, 'Global_reactive_power'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Voltage))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Voltage))
exogenas_rn_train.loc[:, 'Voltage'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Voltage))
exogenas_rn_test.loc[:, 'Voltage'] = normalizados


# ESCALAMOS LA VARIABLE TARGET, HACIENDO FIT SOLO DE LA PARTE DE TRAIN
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_rn)
normalizados  = scaler.transform(train_rn)
train_rn.loc[:, 'Global_active_power'] = normalizados
normalizados  = scaler.transform(test_rn)
test_rn.loc[:, 'Global_active_power'] = normalizados

# CONCATENAMOS TARGET Y EXOGENAS PARA TRAIN Y PARA TEST
train_rn_exog = pd.concat([train_rn, exogenas_rn_train], axis= 1)
test_rn_exog = pd.concat([test_rn, exogenas_rn_test], axis= 1)
tabla_rn = pd.concat([train_rn_exog, test_rn_exog], axis=0)
# ============================================================== #

# =========== CONTROL DE ENTRENAMIENTO DETERMINISTA ============ #
import random

SEED = 123
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

tf.config.experimental.enable_op_determinism()
# ============================================================== #

# ========= FRAGMENTOS PARA EL ENTRENAMIENTO DE LA RED ========= #
window_len = 7 # AJUSTE DE LA VENTANA DE TIEMPO
LSTM_training_inputs_array, LSTM_training_outputs, LSTM_test_inputs_array, \
    LSTM_test_outputs = fragmentos(train_rn_exog, test_rn_exog, window_len, 'Global_active_power')

# ESTRUCTURA DE LA RED NEURONAL CNN+LSTM
redCNNLSTM = Sequential()

# Sección CNN
redCNNLSTM.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
                input_shape=(LSTM_training_inputs_array.shape[1], 7), 
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))
redCNNLSTM.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same',
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

redCNNLSTM.add(MaxPooling1D(pool_size=2))
redCNNLSTM.add(Conv1D(filters=16, kernel_size=7, activation='relu', padding='same',
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
                      ))
redCNNLSTM.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
                      ))

redCNNLSTM.add(Dense(units=100, activation='relu', use_bias= True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

# Sección LSTM
redCNNLSTM.add(LSTM(200, return_sequences=True, use_bias = True, bias_initializer="zeros",
              input_shape=(LSTM_training_inputs_array.shape[1], 1), 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redCNNLSTM.add(LayerNormalization())

redCNNLSTM.add(LSTM(50, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redCNNLSTM.add(LSTM(100, return_sequences=False, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redCNNLSTM.add(Dense(100, activation='relu', use_bias= True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

# Neurona de salida
redCNNLSTM.add(Dense(units=1, activation='linear', use_bias=True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

redCNNLSTM.summary()
# ============================================================== #

# =============== ENTRENAMIENTO DE LA RED ====================== #
history_redCNNLSTM = compile_and_fit(redCNNLSTM, 10, LSTM_training_inputs_array, 
                               LSTM_training_outputs)

plot_graphs(history_redCNNLSTM, 'loss')
# ============================================================== #
    
 # %%
 # ======================== PREDICCION ========================== #
 # PRIMERO PREDECIMOS Y DESPUES DESHACEMOS EL ESCALAMIENTO
pred_acumulada = []
vector_real = []
for t in range(1, len(test_noDif.Global_active_power)):
     corte = len(train_rn.Global_active_power) + t
     ventana = tabla_rn[corte-window_len:corte]
     valor_real = np.array(tabla_diaria.Global_active_power[corte:corte+1])[0]
     ventana = np.array(ventana).reshape(1,7,7)
     pred_redLSTM = redCNNLSTM.predict(ventana) # prediccion para t+1
    
     pred_redLSTM_desescalada = scaler.inverse_transform(pred_redLSTM)[0][0]
     pred_desdiferenciada = pred_redLSTM_desescalada + test_noDif.Global_active_power[t-1] # yhat(t+1) + y(t)
     pred_destransformada = pred_desdiferenciada**2
     error = abs(valor_real - pred_destransformada)
    
     pred_acumulada.append(pred_destransformada) # Vector de predicciones a un dia
     vector_real.append(valor_real)

pred_acumulada = np.array(pred_acumulada)
# CALCULAMOS ERRORES DESHACIENDO LA DIFERENCIA Y ELEVANDO AL CUADRADO
# POR LA TRANSFORMACION RAIZ CUADRADA
red_CNNLSTMEXOG_mape = 100*mean_absolute_percentage_error(vector_real, pred_acumulada)

# ============================================================== #

# ======================== GRAFICOS ============================ #
plt.figure(figsize=(10, 6))
plt.plot(vector_real, label='Real - Test')
plt.plot(pred_acumulada, label='Predicción - Test', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Valores')
plt.title(f'Red: Predicción vs Valores Reales. MAPE = {red_CNNLSTMEXOG_mape}')
plt.legend()
plt.show()
# ================================================================



# %%
"""RED (CELDAS LSTM + VARIABLES EXOGENAS)"""

# ======= DIVISION EN TRAIN Y TEST DE LA VARIABLE TARGET ======= #
train_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][:int(n*0.8)])
test_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][int(n*0.8):])
# ============================================================== #

# TRANSFORMACION RAIZ CUADRADA DE TARGET
train_rn.loc[:, ['Global_active_power']] = np.sqrt(train_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()
test_rn.loc[:, ['Global_active_power']] = np.sqrt(test_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()

# ============= DATAFRAME DE VARIABLES EXOGENAS ================ #
# DIFERENCIAMOS
exogenas_rn = exogenas.copy().diff()
exogenas_rn = exogenas_rn.bfill()

# DIVIDIMOS EN TRAIN Y TEST
n = len(exogenas_rn)
exogenas_rn_train = exogenas_rn[0:int(n*0.8)]
exogenas_rn_test = exogenas_rn[int(n*0.8):]
# ============================================================== #

# ================= DIFERENCIAMOS TRAIN Y TEST ================= #
train_rn = train_rn.diff()
train_rn = train_rn.bfill()
test_noDif = test_rn.copy()
test_rn = test_rn.diff()
test_rn = test_rn.bfill()
# ============================================================== #

# =============== ESCALAMOS CON MINMAXSCALER =================== #
scaler = MinMaxScaler(feature_range=(0, 1))

# ESCALAMOS CADA VARIABLE EXOGENA, HACIENDO FIT SOLO DE LA PARTE DE TRAIN
scaler.fit(pd.DataFrame(exogenas_rn_train.Sub_metering_1))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Sub_metering_1))
exogenas_rn_train.loc[:, 'Sub_metering_1'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Sub_metering_1))
exogenas_rn_test.loc[:, 'Sub_metering_1'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Sub_metering_2))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Sub_metering_2))
exogenas_rn_train.loc[:, 'Sub_metering_2'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Sub_metering_2))
exogenas_rn_test.loc[:, 'Sub_metering_2'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Sub_metering_3))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Sub_metering_3))
exogenas_rn_train.loc[:, 'Sub_metering_3'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Sub_metering_3))
exogenas_rn_test.loc[:, 'Sub_metering_3'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Global_intensity))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Global_intensity))
exogenas_rn_train.loc[:, 'Global_intensity'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Global_intensity))
exogenas_rn_test.loc[:, 'Global_intensity'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Global_reactive_power))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Global_reactive_power))
exogenas_rn_train.loc[:, 'Global_reactive_power'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Global_reactive_power))
exogenas_rn_test.loc[:, 'Global_reactive_power'] = normalizados

scaler.fit(pd.DataFrame(exogenas_rn_train.Voltage))
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_train.Voltage))
exogenas_rn_train.loc[:, 'Voltage'] = normalizados
normalizados = scaler.transform(pd.DataFrame(exogenas_rn_test.Voltage))
exogenas_rn_test.loc[:, 'Voltage'] = normalizados


# ESCALAMOS LA VARIABLE TARGET, HACIENDO FIT SOLO DE LA PARTE DE TRAIN
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_rn)
normalizados  = scaler.transform(train_rn)
train_rn.loc[:, 'Global_active_power'] = normalizados
normalizados  = scaler.transform(test_rn)
test_rn.loc[:, 'Global_active_power'] = normalizados

# CONCATENAMOS TARGET Y EXOGENAS PARA TRAIN Y PARA TEST
train_rn_exog = pd.concat([train_rn, exogenas_rn_train], axis= 1)
test_rn_exog = pd.concat([test_rn, exogenas_rn_test], axis= 1)
tabla_rn = pd.concat([train_rn_exog, test_rn_exog], axis=0)
# ============================================================== #

# =========== CONTROL DE ENTRENAMIENTO DETERMINISTA ============ #
import random


SEED = 123
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

tf.config.experimental.enable_op_determinism()
# ============================================================== #


# ========= FRAGMENTOS PARA EL ENTRENAMIENTO DE LA RED ========= #
window_len = 7 # AJUSTE DE LA VENTANA DE TIEMPO
LSTM_training_inputs_array, LSTM_training_outputs, LSTM_test_inputs_array, \
    LSTM_test_outputs = fragmentos(train_rn_exog, test_rn_exog, window_len, 'Global_active_power')

# ESTRUCTURA DE LA RED NEURONAL LSTM
redLSTMEXOG = Sequential()


redLSTMEXOG.add(LSTM(200, return_sequences=True, use_bias = True, bias_initializer="zeros",
              input_shape=(LSTM_training_inputs_array.shape[1], 7), 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTMEXOG.add(LayerNormalization())

redLSTMEXOG.add(LSTM(50, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTMEXOG.add(LSTM(25, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTMEXOG.add(LSTM(50, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTMEXOG.add(LSTM(100, return_sequences=False, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))


redLSTMEXOG.add(Dense(100, activation='relu', use_bias= True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

# Neurona de salida
redLSTMEXOG.add(Dense(units=1, activation='linear', use_bias=True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

redLSTMEXOG.summary()
# ============================================================== #

# =============== ENTRENAMIENTO DE LA RED ====================== #
history_redLSTMEXOG = compile_and_fit(redLSTMEXOG, 50, LSTM_training_inputs_array, 
                               LSTM_training_outputs)


plot_graphs(history_redLSTMEXOG, 'loss')
# ============================================================== #
    
# %%
# ======================== PREDICCION ========================== #
# PRIMERO PREDECIMOS Y DESPUES DESHACEMOS EL ESCALAMIENTO
pred_acumulada = []
vector_real = []
for t in range(1, len(test_noDif.Global_active_power)):
     corte = len(train_rn.Global_active_power) + t
     ventana = tabla_rn[corte-window_len:corte]
     valor_real = np.array(tabla_diaria.Global_active_power[corte:corte+1])[0]
     ventana = np.array(ventana).reshape(1,7,7)
     pred_redLSTM = redLSTMEXOG.predict(ventana) # prediccion para t+1
    
     pred_redLSTM_desescalada = scaler.inverse_transform(pred_redLSTM)[0][0]
     pred_desdiferenciada = pred_redLSTM_desescalada + test_noDif.Global_active_power[t-1] # yhat(t+1) + y(t)
     pred_destransformada = pred_desdiferenciada**2
     error = abs(valor_real - pred_destransformada)
    
     pred_acumulada.append(pred_destransformada) # Vector de predicciones a un dia
     vector_real.append(valor_real)
 
pred_acumulada = np.array(pred_acumulada)
# CALCULAMOS ERRORES DESHACIENDO LA DIFERENCIA Y ELEVANDO AL CUADRADO
# POR LA TRANSFORMACION RAIZ CUADRADA
red_LSTMEXOG_mape = 100*mean_absolute_percentage_error(vector_real, pred_acumulada)
# ============================================================== #

# ======================== GRAFICOS ============================ #
plt.figure(figsize=(10, 6))
plt.plot(vector_real, label='Real - Test')
plt.plot(pred_acumulada, label='Predicción - Test', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Valores')
plt.title(f'Red: Predicción vs Valores Reales. MAPE = {red_LSTMEXOG_mape}')
plt.legend()
plt.show()
# ================================================================

# %%
"""RED (CNN+LSTM) SIN EXOGENAS"""
    
# ======= DIVISION EN TRAIN Y TEST DE LA VARIABLE TARGET ======= #
train_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][:int(n*0.8)])
test_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][int(n*0.8):])

# ============================================================== #

# TRANSFORMACION RAIZ CUADRADA DE TARGET
train_rn.loc[:, ['Global_active_power']] = np.sqrt(train_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()
test_rn.loc[:, ['Global_active_power']] = np.sqrt(test_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()

# ================= DIFERENCIAMOS TRAIN Y TEST ================= #
train_rn = train_rn.diff()
train_rn = train_rn.bfill()
test_noDif = test_rn.copy()
test_rn = test_rn.diff()
test_rn = test_rn.bfill()
# ============================================================== #

# =============== ESCALAMOS CON MINMAXSCALER =================== #
scaler = MinMaxScaler(feature_range=(0, 1))


# ESCALAMOS LA VARIABLE TARGET, HACIENDO FIT SOLO DE LA PARTE DE TRAIN
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_rn)
normalizados  = scaler.transform(train_rn)
train_rn.loc[:, 'Global_active_power'] = normalizados
normalizados  = scaler.transform(test_rn)
test_rn.loc[:, 'Global_active_power'] = normalizados
# ============================================================== #

# =========== CONTROL DE ENTRENAMIENTO DETERMINISTA ============ #
import random

   
SEED = 123
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

tf.config.experimental.enable_op_determinism()
# ============================================================== #

# ========= FRAGMENTOS PARA EL ENTRENAMIENTO DE LA RED ========= #
window_len = 7 # AJUSTE DE LA VENTANA DE TIEMPO
LSTM_training_inputs_array, LSTM_training_outputs, LSTM_test_inputs_array, \
    LSTM_test_outputs = fragmentos(train_rn, test_rn, window_len, 'Global_active_power')

# ESTRUCTURA DE LA RED NEURONAL CNN+LSTM
redCNNLSTM = Sequential()

# Sección CNN
redCNNLSTM.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
                input_shape=(LSTM_training_inputs_array.shape[1], 1), 
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))
redCNNLSTM.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same',
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))
# redCNNLSTM.add(Dropout(0.5))
redCNNLSTM.add(MaxPooling1D(pool_size=2))
redCNNLSTM.add(Conv1D(filters=16, kernel_size=7, activation='relu', padding='same',
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
                      ))
redCNNLSTM.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
                      ))

redCNNLSTM.add(Dense(units=100, activation='relu', use_bias= True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

# Sección LSTM
redCNNLSTM.add(LSTM(200, return_sequences=True, use_bias = True, bias_initializer="zeros",
              input_shape=(LSTM_training_inputs_array.shape[1], 1), 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redCNNLSTM.add(LayerNormalization())

redCNNLSTM.add(LSTM(50, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redCNNLSTM.add(LSTM(100, return_sequences=False, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redCNNLSTM.add(Dense(100, activation='relu', use_bias= True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

# Neurona de salida
redCNNLSTM.add(Dense(units=1, activation='linear', use_bias=True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

redCNNLSTM.summary()
# ============================================================== #

# =============== ENTRENAMIENTO DE LA RED ====================== #
history_redCNNLSTM = compile_and_fit(redCNNLSTM, 10, LSTM_training_inputs_array, 
                               LSTM_training_outputs)

plot_graphs(history_redCNNLSTM, 'loss')
# ============================================================== #


 # %%
 # ======================== PREDICCION ========================== #
 # PRIMERO PREDECIMOS Y DESPUES DESHACEMOS EL ESCALAMIENTO
pred_acumulada = []
vector_real = []
for t in range(1, len(test_noDif.Global_active_power)):
     corte = len(train_rn.Global_active_power) + t
     ventana = tabla_rn[corte-window_len:corte]
     valor_real = np.array(tabla_diaria.Global_active_power[corte:corte+1])[0]
     ventana = np.array(ventana).reshape(1,7,1)
     pred_redCNNLSTM = redCNNLSTM.predict(ventana) # prediccion para t+1
    
     pred_redCNNLSTM_desescalada = scaler.inverse_transform(pred_redCNNLSTM)[0][0]
     pred_desdiferenciada = pred_redCNNLSTM_desescalada + test_noDif.Global_active_power[t-1] # yhat(t+1) + y(t)
     pred_destransformada = pred_desdiferenciada**2
     error = abs(valor_real - pred_destransformada)
    
     pred_acumulada.append(pred_destransformada) # Vector de predicciones a un dia
     vector_real.append(valor_real)

pred_acumulada = np.array(pred_acumulada)
# CALCULAMOS ERRORES DESHACIENDO LA DIFERENCIA Y ELEVANDO AL CUADRADO
# POR LA TRANSFORMACION RAIZ CUADRADA
red_CNNLSTM_mape = 100*mean_absolute_percentage_error(vector_real, pred_acumulada)

# ============================================================== #

# ======================== GRAFICOS ============================ #
plt.figure(figsize=(10, 6))
plt.plot(vector_real, label='Real - Test')
plt.plot(pred_acumulada, label='Predicción - Test', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Valores')
plt.title(f'Red: Predicción vs Valores Reales. MAPE = {red_CNNLSTM_mape}')
plt.legend()
plt.show()
# ================================================================

"""NO DETERMINISTA"""


# %%
"""RED (CELDAS LSTM SIN VARIABLES EXOGENAS)"""
    
# ======= DIVISION EN TRAIN Y TEST DE LA VARIABLE TARGET ======= #
train_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][:int(n*0.8)])
test_rn = pd.DataFrame(tabla_diaria.loc[:, 'Global_active_power'][int(n*0.8):])

# ============================================================== #

# TRANSFORMACION RAIZ CUADRADA DE TARGET
train_rn.loc[:, ['Global_active_power']] = np.sqrt(train_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()
test_rn.loc[:, ['Global_active_power']] = np.sqrt(test_rn.loc[:, ['Global_active_power']
                                               ]).values.tolist()

# ================= DIFERENCIAMOS TRAIN Y TEST ================= #
test_noDif = test_rn.copy()
# ============================================================== #

# =============== ESCALAMOS CON MINMAXSCALER =================== #
scaler = MinMaxScaler(feature_range=(0, 1))

# ESCALAMOS LA VARIABLE TARGET, HACIENDO FIT SOLO DE LA PARTE DE TRAIN
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_rn)
normalizados  = scaler.transform(train_rn)
train_rn.loc[:, 'Global_active_power'] = normalizados
normalizados  = scaler.transform(test_rn)
test_rn.loc[:, 'Global_active_power'] = normalizados
tabla_rn = pd.concat([train_rn, test_rn], axis=0)
# ============================================================== #

# =========== CONTROL DE ENTRENAMIENTO DETERMINISTA ============ #
import random

SEED = 123
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"


tf.config.experimental.enable_op_determinism()
# ============================================================== #


# ========= FRAGMENTOS PARA EL ENTRENAMIENTO DE LA RED ========= #
window_len = 7 # AJUSTE DE LA VENTANA DE TIEMPO
LSTM_training_inputs_array, LSTM_training_outputs, LSTM_test_inputs_array, \
    LSTM_test_outputs = fragmentos(train_rn, test_rn, window_len, 'Global_active_power')

# ESTRUCTURA DE LA RED NEURONAL LSTM
redLSTM = Sequential()

redLSTM.add(LSTM(200, return_sequences=True, use_bias = True, bias_initializer="zeros",
              input_shape=(LSTM_training_inputs_array.shape[1], 1), 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTM.add(LayerNormalization())

redLSTM.add(LSTM(50, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTM.add(LSTM(25, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTM.add(LSTM(50, return_sequences=True, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))

redLSTM.add(LSTM(100, return_sequences=False, use_bias = True, bias_initializer="zeros", 
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123),
              recurrent_initializer=tf.keras.initializers.orthogonal(seed=123),seed=123))


redLSTM.add(Dense(100, activation='relu', use_bias= True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

# Neurona de salida
redLSTM.add(Dense(units=1, activation='linear', use_bias=True, bias_initializer="zeros",
               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=123)))

redLSTM.summary()
# ============================================================== #

# =============== ENTRENAMIENTO DE LA RED ====================== #
history_redLSTM = compile_and_fit(redLSTM, 120, LSTM_training_inputs_array, 
                               LSTM_training_outputs)

plot_graphs(history_redLSTM, 'loss')
# ============================================================== #
# ============================================================== #
 # %%
 # ======================== PREDICCION ========================== #
 # PRIMERO PREDECIMOS Y DESPUES DESHACEMOS EL ESCALAMIENTO
pred_acumulada = []
vector_real = []
for t in range(1, len(test_noDif.Global_active_power)):
    corte = len(train_rn.Global_active_power) + t
    ventana = tabla_rn[corte-window_len:corte]
    valor_real = np.array(tabla_diaria.Global_active_power[corte:corte+1])[0]
    ventana = np.array(ventana).reshape(1,7,1)
    pred_redLSTM = redLSTM.predict(ventana) # prediccion para t+1
   
    pred_redLSTM_desescalada = scaler.inverse_transform(pred_redLSTM)[0][0]
    pred_destransformada = pred_redLSTM_desescalada**2
    error = abs(valor_real - pred_destransformada)
    
    pred_acumulada.append(pred_destransformada) # Vector de predicciones a un dia
    vector_real.append(valor_real)

pred_acumulada = np.array(pred_acumulada)
# CALCULAMOS ERRORES DESHACIENDO LA DIFERENCIA Y ELEVANDO AL CUADRADO
# POR LA TRANSFORMACION RAIZ CUADRADA
red_LSTM_mape = 100*mean_absolute_percentage_error(vector_real, pred_acumulada)

# ============================================================== #

# ======================== GRAFICOS ============================ #
plt.figure(figsize=(10, 6))
plt.plot(vector_real, label='Real - Test')
plt.plot(pred_acumulada, label='Predicción - Test', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Valores')
plt.title(f'Red: Predicción vs Valores Reales. MAPE = {red_LSTM_mape}')
plt.legend()
plt.show()
# ================================================================
    
