"""HELPER MODULE"""

"""FUNCIONES DE BOX-JENKINS"""
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

# RESUMEN DEL MODELO 
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
    
# PREDICCION ROLLING A HORIZONTE 1
def rolling_pred(tabla_diaria: pd.DataFrame, 
                 test: pd.DataFrame, exogenas: pd.DataFrame,
                 use_exog: bool = False):
    pred_acumulada = []
    vector_real = []
    exog_lag1 = exogenas.copy().shift(1).bfill()
    
    if use_exog:
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
    else:
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
            
            vector_real.append(valor_real)
            pred_acumulada.append(prediccion_ARIMA) # Vector de predicciones a un dia

    sarimax_mape=(100*mean_absolute_percentage_error(vector_real[:],
                                                     pred_acumulada[:]))

    plt.figure(figsize=(10, 6))
    plt.plot(vector_real[:], label='Real - Test')
    plt.plot( pred_acumulada[:], label='Predicción - Test', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Valores')
    plt.title('Predicción vs Valores Reales')
    plt.legend()
    plt.show()
    
    return sarimax_mape


"""FUNCIONES DE REDES NEURONALES"""
from keras.models import Sequential
from keras.layers import Activation, Dense, Attention, Bidirectional
from keras.layers import LSTM, LeakyReLU, GRU, Conv1D, MaxPooling1D, Flatten
from keras.layers import Dropout
from tensorflow.keras.layers import Reshape, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, TimeDistributed
from keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.regularizers import l2

# FUNCION QUE GENERA FRAGMENTOS DE INPUTS DE TEST Y TRAINING DE LONGITUD WINDOW_LEN
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

# FUNCION QUE DEFINE CRITERIO DE PARADA, COMPILA Y AJUSTA EL MODELO
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


# FUNCION QUE DIBUJA EL HISTORICO DE ENTRENAMIENTO
def plot_graphs(history, metric):
  plt.figure(figsize=(10, 6))
  plt.plot(history.history[metric], label='train')
  plt.plot(history.history['val_'+metric], label='test')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend()
  plt.show()
  
# FUNCION QUE HACE TRANSFORMACIONES SOBRE LAS VARIABLES TARGET Y EXOGENAS
def transform(target_series: pd.Series, exogenas: pd.DataFrame, 
              use_exog: bool = False, dif: bool = False):
    # ======= DIVISION EN TRAIN Y TEST DE LA VARIABLE TARGET ======= #
    n = len(target_series)
    train_rn = pd.DataFrame(target_series[:int(n*0.8)])
    test_rn = pd.DataFrame(target_series[int(n*0.8):])
    # ============================================================== #
    
    # TRANSFORMACION RAIZ CUADRADA DE TARGET
    train_rn.loc[:, ['Global_active_power']] = np.sqrt(train_rn.loc[:, ['Global_active_power']
                                                   ]).values.tolist()
    test_rn.loc[:, ['Global_active_power']] = np.sqrt(test_rn.loc[:, ['Global_active_power']
                                                   ]).values.tolist()
    
    
    if use_exog:
        # ============= DATAFRAME DE VARIABLES EXOGENAS ================ #
        if dif:
            # DIFERENCIAMOS
            exogenas_rn = exogenas.copy().diff()
            exogenas_rn = exogenas_rn.bfill()
        else:
            exogenas_rn = exogenas.copy()
    
        # DIVIDIMOS EN TRAIN Y TEST
        exogenas_rn_train = exogenas_rn[0:int(n*0.8)]
        exogenas_rn_test = exogenas_rn[int(n*0.8):]
        # ============================================================== #
    
    test_noDif = test_rn.copy()
    if dif:
        # ================= DIFERENCIAMOS TRAIN Y TEST ================= #
        train_rn = train_rn.diff()
        train_rn = train_rn.bfill()
        
        test_rn = test_rn.diff()
        test_rn = test_rn.bfill()
        # ============================================================== #
    
    # =============== ESCALAMOS CON MINMAXSCALER =================== #
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    if use_exog:
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
    scaler.fit(train_rn)
    normalizados  = scaler.transform(train_rn)
    train_rn.loc[:, 'Global_active_power'] = normalizados
    normalizados  = scaler.transform(test_rn)
    test_rn.loc[:, 'Global_active_power'] = normalizados
    
    if use_exog:
        # CONCATENAMOS TARGET Y EXOGENAS PARA TRAIN Y PARA TEST
        train_rn_exog = pd.concat([train_rn, exogenas_rn_train], axis= 1)
        test_rn_exog = pd.concat([test_rn, exogenas_rn_test], axis= 1)
        tabla_rn = pd.concat([train_rn_exog, test_rn_exog], axis=0)
    else:
        tabla_rn = pd.concat([train_rn, test_rn], axis=0)
        # ============================================================== #
        
    if use_exog:
        return tabla_rn, train_rn_exog, test_rn_exog, test_noDif, scaler
    else:
        return tabla_rn, train_rn, test_rn, test_noDif, scaler
    
def nn_rolling_pred(target_series: pd.Series, tabla_rn: pd.DataFrame, 
                    train_rn: pd.DataFrame, test_noDif: pd.DataFrame,
                    window_len: int, nn, scaler, use_exog: bool, dif: bool):
    # ======================== PREDICCION ========================== #
    # PRIMERO PREDECIMOS Y DESPUES DESHACEMOS EL ESCALAMIENTO
    pred_acumulada = []
    vector_real = []
    for t in range(1, len(test_noDif.Global_active_power)):
        corte = len(train_rn.Global_active_power) + t
        ventana = tabla_rn[corte-window_len:corte]
        valor_real = np.array(target_series[corte:corte+1])[0]
        if use_exog:
            ventana = np.array(ventana).reshape(1,7,7)
        else:
            ventana = np.array(ventana).reshape(1,7,1)
        pred_red = nn.predict(ventana) # prediccion para t+1
       
        pred_desescalada = scaler.inverse_transform(pred_red)[0][0]
        if dif:
            pred_desdiferenciada = pred_desescalada + test_noDif.Global_active_power[t-1] # yhat(t+1) + y(t)
            pred_destransformada = pred_desdiferenciada**2
        else:
            pred_destransformada = pred_desescalada**2
        
       
        pred_acumulada.append(pred_destransformada) # Vector de predicciones a un dia
        vector_real.append(valor_real)

    pred_acumulada = np.array(pred_acumulada)
    
    red_mape = 100*mean_absolute_percentage_error(vector_real, pred_acumulada)
   
    # ============================================================== #
   
    # ======================== GRAFICOS ============================ #
    plt.figure(figsize=(10, 6))
    plt.plot(vector_real, label='Real - Test')
    plt.plot(pred_acumulada, label='Predicción - Test', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Valores')
    plt.title(f'Red: Predicción vs Valores Reales. MAPE = {red_mape}')
    plt.legend()
    plt.show()
    # ================================================================
   
    return red_mape