
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

car_df = pd.read_csv('/content/Car_Purchasing_Data.csv', encoding='ISO-8859-1')

car_df

car_df.head(5)

car_df.tail(5)

sns.pairplot(car_df)

X = car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'], axis=1)

X

Y= car_df['Car Purchase Amount']

Y

X.shape

Y.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled

scaler= MinMaxScaler()
Y_scaled= scaler.fit_transform(Y.values.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.25)

X_train.shape

X_test.shape

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, Y_train, epochs=100, batch_size=25, verbose=1, validation_split=0.2)

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progression During Training')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

#Gender Age Annual salary Credit card debt net worth
X_test = np.array([[1,50,50000,10000,600000]])
y_predict = model.predict(X_test)

print('Expected Purchase Amount', y_predict)
