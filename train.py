import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 비트코인 데이터 로드 및 전처리
data = pd.read_csv('btc_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

features = data[['Open', 'High', 'Low', 'Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# 데이터셋 생성
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 3])
    return np.array(X), np.array(y)

time_step = 30
X, y = create_dataset(scaled_features, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# LSTM 모델 정의
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(tf.keras.layers.LSTM(50, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, batch_size=1, epochs=20)

# 모델 저장
model.save('train20_3.h5')
