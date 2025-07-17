"""RED LSTM"""

from helper import fragmentos, compile_and_fit, plot_graphs
from helper import transform, nn_rolling_pred

import pandas as pd
import numpy as np
import tensorflow as tf

import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.layers import LayerNormalization


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


"""POR EL MOMENTO TRABAJAMOS CON LA TABLA DIARIA"""
n = len(tabla_diaria)
train = tabla_diaria[0:int(n*0.8)]

"""DIBUJAMOS LA TABLA"""
tabla_diaria.plot(y='Global_active_power')

# EXOGENAS
exogenas = pd.DataFrame({'Sub_metering_1':tabla_diaria.Sub_metering_1, 
                         'Sub_metering_2':tabla_diaria.Sub_metering_2, 
                         'Sub_metering_3':tabla_diaria.Sub_metering_3})
exogenas.loc[:, 'Global_intensity'] = tabla_diaria.Global_intensity
exogenas.loc[:, 'Global_reactive_power'] = tabla_diaria.Global_reactive_power
exogenas.loc[:, 'Voltage'] = tabla_diaria.Voltage


"""RED CON VARIABLES EXOGENAS"""

# ================= PROCESAMIENTO DE LOS DATOS ================= #
tabla_rn, train_rn_exog, test_rn_exog, test_noDif, scaler = \
    transform(target_series=tabla_diaria.Global_active_power, 
              exogenas=exogenas, use_exog=True, dif=True)
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

# ============ PREDICCION ROLLING Y ERROR MAPE ================= #
LSTMEXOG_mape = nn_rolling_pred(target_series=tabla_diaria.Global_active_power,
                                tabla_rn=tabla_rn, train_rn=train_rn_exog, 
                                test_noDif=test_noDif, window_len=7, 
                                nn=redLSTMEXOG, scaler=scaler, use_exog=True,
                                dif=True)
# ============================================================== #


"""RED SIN VARIABLES EXOGENAS"""

# ================= PROCESAMIENTO DE LOS DATOS ================= #
tabla_rn, train_rn, test_rn, test_noDif, scaler = \
    transform(target_series=tabla_diaria.Global_active_power, 
              exogenas=exogenas, use_exog=False, dif=False)
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

# ============ PREDICCION ROLLING Y ERROR MAPE ================= #
LSTM_mape = nn_rolling_pred(target_series=tabla_diaria.Global_active_power,
                                tabla_rn=tabla_rn, train_rn=train_rn, 
                                test_noDif=test_noDif, window_len=7, 
                                nn=redLSTM, scaler=scaler, use_exog=False,
                                dif=False)
# ============================================================== #