"""ARIMA"""

from helper import resumen, rolling_pred

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.statespace.sarimax import SARIMAX

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


"""SARIMA(2,1,0)X(2,0,2)_7 SIN EXOGENAS"""
sarimax_mape = rolling_pred(tabla_diaria=tabla_diaria, 
                 test=test, exogenas=exogenas)
# %%
"""SARIMA(2,1,0)X(2,0,2)_7 CON EXOGENAS"""
sarimax_mape_wExog = rolling_pred(tabla_diaria=tabla_diaria, 
                 test=test, exogenas=exogenas, use_exog=True)

