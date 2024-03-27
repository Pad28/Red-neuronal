import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

data = pd.read_csv("diabetes.csv");

#Se cambia el nombre de las columnas
data.columns  = ['Embarazos', 'Glucosa', 'Presión arterial', 'Grosor de la piel', 'Insulina', 'IMC', 'DiabetesPedigríFunción', 'Edad', 'Resultado']
#Se eliminan las columnas que no son significantes para la prediccion
data = data.drop(columns=['Grosor de la piel', 'Embarazos','Edad'])

all_data = data.drop(['Resultado'], axis=1)
est = all_data.div(all_data.max())
all_inputs = est.iloc[:, :].values
all_output = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_output, test_size=1/3)

L = 0.05 # Indice de aprendizaje
n = X_train.shape[0]

# Pesos y bias aleatorios
w_hidden = np.random.rand(5, 5)
w_output = np.random.rand(1, 5)

b_hidden = np.random.rand(5, 1)
b_output = np.random.rand(1, 1)

# Funciones de activacion
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1+np.exp(-x))

def forward_prop(x):
  Z1 = w_hidden @ x + b_hidden
  A1 = relu(Z1)
  Z2 = w_output @ A1 + b_output
  A2 = logistic(Z2)
  return Z1, A1, Z2, Z2

# Dericadas de las funciones de activación
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

def backward_prop(Z1, A1, Z2, A2, X, Y):
  dC_dA2 = 2 * A2 - 2 * Y
  dA2_dZ2 = d_logistic(Z2)
  dZ2_dA1 = w_output
  dZ2_dW2 = A1
  dZ2_dB2 = 1
  dA1_dZ1 = d_relu(Z1)
  dZ1_dW1 = X
  dZ1_dB1 = 1
  dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T
  dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2
  dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1
  dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T
  dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1
  return dC_dW1, dC_dB1, dC_dW2, dC_dB2

for i in range(100_000):
  idx = np.random.choice(n, 1, replace=False)
  X_sample = X_train[idx].transpose()
  Y_sample = Y_train[idx]

  Z1, A1, Z2, A2 = forward_prop(X_sample)

  dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

  w_hidden -= L * dW1
  b_hidden -= L * dB1
  w_output -= L * dW2
  b_output -= L * dB2

test_predictions = forward_prop(X_test.transpose())[3]
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
print("Exactitud: ", accuracy)

def predict_probability(glucosa, presion_arterial, insulina, imc, dpf):
  glucosa_est = glucosa / (data['Glucosa'].max())
  presion_arterial_est = presion_arterial / (data['Presión arterial'].max())
  insulina_est = insulina / (data['Insulina'].max())
  imc_est = imc / (data['IMC'].max())
  dpf_est = dpf / (data['DiabetesPedigríFunción'].max())

  X = np.array([[glucosa_est, presion_arterial_est, insulina_est, imc_est, dpf_est]]).transpose()
  Z1, A1, Z2, A2 = forward_prop(X)
  return A2

def predict_diabetes(glucosa, presion_arterial, insulina, imc, dpf):
  output_values = predict_probability(glucosa, presion_arterial, insulina, imc, dpf)
  print("Valor de salida: ", output_values)
  return True if output_values > 0.5 else False

import json
with open("data.json", "r") as archivo:
    datos_json = json.load(archivo)

result = predict_diabetes(
    datos_json['Glucosa'], 
    datos_json['Presion arterial'], 
    datos_json['Insulina'],
    datos_json['IMC'],
    datos_json['DiabetesPedigríFunción']
)
# result = predict_diabetes(189, 60, 846, 30.1, 398)
print(result)

prediccion = { "Resultado": result }
with open("data.json", "w") as archivo:
    json.dump(prediccion, archivo)