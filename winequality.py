import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#importando os dados
dataset = pd.read_csv('winequality-red.csv')#extensão csv

#ordenando as variáveis 
X = dataset.iloc[:,[7, 10]]
Y = dataset.iloc[:, 11] 

x = X.values.reshape(-1, 2)
y = Y.values.reshape(-1, 1)

randomForestRegressor = RandomForestRegressor(random_state = 0)
randomForestRegressor.fit(x, y)
