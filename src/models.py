import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="TrafficDL")
class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)

    def get_config(self):
        return super().get_config()


def build_vanilla_lstm(input_shape=(24, 25)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="Vanilla_LSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_bilstm(input_shape=(24, 25)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="Stacked_BiLSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_champion_model(input_shape=(24, 25), name="Conv1D_BiLSTM_Attention") -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    context = TemporalAttention()(x)
    x = layers.Dense(64, activation="relu")(context)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model