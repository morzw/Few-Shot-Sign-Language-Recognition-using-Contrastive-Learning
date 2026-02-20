import keras as k
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, GRU, LSTM, Lambda
from keras.models import Sequential


def create_embedding_model(model_name: str, input_shape):
    if model_name == "gru":
        model = Sequential()
        model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(64, return_sequences=False)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation=None))
        model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
        return model

    if model_name == "lstm":
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, activation="relu", input_shape=input_shape))
        model.add(LSTM(192, return_sequences=True, activation="relu"))
        model.add(Dropout(0.1))
        model.add(LSTM(64, return_sequences=False, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(50, activation=None))
        model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
        return model

    if model_name == "bi_lstm":
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation=None))
        model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
        return model

    raise ValueError("Unsupported model_name. Use one of: lstm, bi_lstm, gru.")


def load_model(model_path: str):
    return k.models.load_model(
        model_path,
        custom_objects={
            "Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss,
        },
    )
