import pandas as pd
import numpy as np
import datetime as dt
import os
from enum import Enum
from sklearn import preprocessing
from collections import deque
import random
import tensorflow as tf
import time

from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def classify(current: float, future: float) -> int:
    if future > current:
        return 1

    return 0

# here we basically preprocess the data
# ... in order to have some proper training data
def preprocess_df(df):
    df = df.drop("Future", 1) # fuck that future shit

    for c in df.columns:
        if c != "Target":
            df[c] = df[c].pct_change()
            df.dropna(inplace=True)
            df[c] = preprocessing.scale(df[c].values)
    
    df.dropna(inplace=True)

    sequential_data = []
    prev_data       = deque(maxlen=SEQ_PERIOD)

    for v in df.values:
        prev_data.append([n for n in v[:-1]])

        if len(prev_data) == SEQ_PERIOD:
            sequential_data.append([np.array(prev_data), v[-1]])
    
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0: # sell
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys  = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    return np.array(X), y



main_df = pd.DataFrame()
ratios  = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

SEQ_PERIOD    = 60 # units to look at
FUTURE_PERIOD = 3  # units
CHOSEN_CRYPTO = "LTC-USD"

EPOCHS = 10
BATCH_SIZE = 64
NAME = f"Zubinator-{SEQ_PERIOD}-{FUTURE_PERIOD}@{int(time.time())}"

for r in ratios:
    data_path = F"crypto/crypto_data/{ r }.csv"
    df        = pd.read_csv(data_path, names=["time", "low", "high", "open", "close", "volume"])

    df.rename(columns={
        "close":  F"{ r } Close",
        "volume": F"{ r } Volume",
    }, inplace=True) 

    df.set_index("time", inplace=True)

    df = df[[F"{ r } Close", F"{ r } Volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

main_df['Future'] = main_df[F"{CHOSEN_CRYPTO} Close"].shift(-FUTURE_PERIOD)
main_df['Target'] = list(map(classify, main_df[F"{CHOSEN_CRYPTO} Close"], main_df["Future"]))

main_df.dropna(inplace=True)

times = sorted(main_df.index.values)
last5 = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last5)]
main_df            = main_df[(main_df.index < last5)]



train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True)) # LONG SHORT TERM MEMORY
model.add(Dropout(0.2)) # adding forget
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

board = TensorBoard(log_dir=F"logs/{NAME}")

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"))

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[board, checkpoint]
)