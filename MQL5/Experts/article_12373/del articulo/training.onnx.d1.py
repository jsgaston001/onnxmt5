# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

from datetime import datetime
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sys import argv

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# we will save generated onnx-file near the our script
data_path=argv[0]
last_index=data_path.rfind("\\")+1
data_path=data_path[0:last_index]
print("data path to save onnx model",data_path)

# input parameters
inp_model_name = "model.eurusd.D1.10.onnx"
inp_history_size = 10
inp_start_date = datetime(2003, 1, 1, 0)
inp_end_date = datetime(2023, 1, 1, 0)

# get data from client terminal
eurusd_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_D1, inp_start_date, inp_end_date)
df = pd.DataFrame(eurusd_rates)


#
# collect dataset subroutine
#
def collect_dataset(df: pd.DataFrame, history_size: int):
    """
    Collect dataset for the following regression problem:
    - input: history_size consecutive H1 bars;
    - output: close price for the next bar.

    :param df: D1 bars for a range of dates
    :param history_size: how many bars should be considered for making a prediction
    :return: features and labels
    """
    n = len(df)
    xs = []
    ys = []
    for i in tqdm(range(n - history_size)):
        w = df.iloc[i: i + history_size + 1]

        x = w[['open', 'high', 'low', 'close']].iloc[:-1].values
        y = w.iloc[-1]['close']
        xs.append(x)
        ys.append(y)

    X = np.array(xs)
    y = np.array(ys)
    return X, y
###


# get prices
X, y = collect_dataset(df, history_size=inp_history_size)

# normalize prices
m = X.mean(axis=1, keepdims=True)
s = X.std(axis=1, keepdims=True)
X_norm = (X - m) / s
y_norm = (y - m[:, 0, 3]) / s[:, 0, 3]

# split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=0)

# define model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(inp_history_size, 4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model training for 50 epochs
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001)
history = model.fit(X_train, y_train, epochs=50, verbose=2, validation_split=0.15, callbacks=[lr_reduction])

# model evaluation
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"test_loss={test_loss:.3f}")
print(f"test_mae={test_mae:.3f}")

# save model to onnx
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

# finish
mt5.shutdown()