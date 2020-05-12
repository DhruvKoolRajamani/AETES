import os, sys

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, LSTM, TimeDistributed, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Activation, ZeroPadding2D, Flatten, Bidirectional, Conv2DTranspose, Input

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs, MODEL_DIR, create_model_directory, path


def get_conv_encoder_model(input_shape=(257, 8, 1), l2_strength=0.0):
  """
    Get the Encoder Model

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """

  NFFT = (input_shape[0] - 1) * 2
  STACKED_FRAMES = input_shape[1]

  # Conv2D with 32 kernels and ReLu, 3x3in time
  inputs = Input(shape=input_shape, name='encoder_input')
  x = inputs

  # -----
  x = ZeroPadding2D(((4, 4), (0, 0)))(x)
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 8],
    strides=[1,
             1],
    padding='valid',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  skip0 = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(skip0)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # -----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  skip1 = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(skip1)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = x + skip1
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = x + skip0
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = SpatialDropout2D(0.2)(x)
  x = Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same')(x)

  Encoder = tf.keras.Model(inputs=inputs, outputs=[x], name='Encoder')
  Encoder.summary()

  return Encoder


def get_encoder_model(input_shape=(None, 1566, 257)):
  """
    Get the Encoder Model
    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """

  NFFT = (input_shape[2] - 1) * 2
  SAMPLES = input_shape[1]
  BATCH_SIZE = input_shape[0]

  # Conv2D with 32 kernels and ReLu, 3x3in time
  input_layer = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
  il_expand_dims = tf.expand_dims(input_layer, axis=1)
  enc_C1D_1 = TimeDistributed(Conv1D(filters=32, kernel_size=3, strides=2, use_bias=True, name='Enc_Conv_1'))(il_expand_dims)
  enc_BN_1 = TimeDistributed(BatchNormalization(name='Enc_Batch_Norm_1'))(enc_C1D_1)
  enc_Act_1 = TimeDistributed(Activation("relu", name='Enc_ReLU_1'))(enc_BN_1)
  enc_C1D_2 = TimeDistributed(Conv1D(filters=32, kernel_size=3, strides=2, use_bias=True, name='Enc_Conv_2'))(enc_Act_1)
  enc_BN_2 = TimeDistributed(BatchNormalization(name='Enc_Batch_Norm_2'))(enc_C1D_2)
  enc_Act_2 = TimeDistributed(Activation("relu", name='Enc_ReLU_2'))(enc_BN_2)

  # ConvLSTM1D -> Try and make this Bidirectional
  # int_input_layer = tf.reshape(tf.expand_dims(enc_Act_2, axis=1), [-1, enc_Act_2.shape[1], enc_Act_2.shape[2], 1], name='Enc_Expand_Dims')
  ConvLSTM1D = Conv2D(1, (1, 3), use_bias=False, name='Enc_ConvLSTM1D', data_format='channels_first')(enc_Act_2)
  print(ConvLSTM1D.shape)
  int_C1DLSTM_out = tf.squeeze(ConvLSTM1D, axis=[1])

  # 3 Stacked Bidirectional LSTMs
  enc_BiLSTM_1 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_1')(int_C1DLSTM_out)
  # enc_BiLSTM_2 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_2')(enc_BiLSTM_1)
  # enc_BiLSTM_3 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_3')(enc_BiLSTM_2)

  # Linear Projection into NFFT/2 and batchnorm and ReLU
  enc_Dense_1 = Dense(NFFT // 8, name='Enc_Linear_Projection')(enc_BiLSTM_1)
  enc_BN_3 = BatchNormalization(name='Enc_Batch_Norm_3')(enc_Dense_1)
  enc_Act_3 = Activation("relu", name='Enc_ReLU_3')(enc_BN_3)

  encoder = tf.keras.Model(inputs=input_layer, outputs=[enc_Act_3], name='Encoder')

  # Begin DeConvolution
  deConv_input_expand_dims = tf.reshape(tf.expand_dims(enc_Act_3, axis=1), [-1, 1, enc_Act_3.shape[1], enc_Act_3.shape[2]])
  DeC1D_filters = enc_Act_3.shape[2]
  Act = deConv_input_expand_dims
  for i in range(2):
    DeC1D = Conv2DTranspose(
      filters=DeC1D_filters * 2,
      kernel_size=(1,
                   3),
      strides=(1,
               2),
      data_format='channels_last',
      output_padding=(0,
                      1),
      padding='valid',
      name='DeConv1D_{}'.format(i + 1)
    )(Act)
    # DeC1D = TimeDistributed()
    BN = TimeDistributed(BatchNormalization(name='DeConv_Batch_norm_{}'.format(i + 1)))(DeC1D)
    Act = TimeDistributed(Activation("relu", name='DeConv_ReLU_{}'.format(i + 1)))(BN)
    DeC1D_filters *= 2

  # DeConvReshape = Conv2D(filters=1, kernel_size=(1, 1), data_format='channels_first', name='DC1D_Reshape')(Act)
  int_DeConv_out = tf.squeeze(Act, axis=[1])
  # Linear Projection into NFFT/2 and batchnorm and ReLU
  deConv_Dense_1 = Dense(NFFT//2 + 1, name='DeConv_Linear_Projection')(int_DeConv_out)
  deConv_BN_3 = BatchNormalization(name='DeConv_Batch_Norm_{}'.format(i + 1))(deConv_Dense_1)
  # deConv_Act_3 = Activation("tanh", name='DeConv_Tanh')(deConv_BN_3)
  output_layer = deConv_BN_3
  # if input_layer.shape[1] > output_layer.shape[1]:
  #   shape = [input_layer.shape[1] - output_layer.shape[1], output_layer.shape[2]]
  #   zero_padding = tf.zeros(shape, dtype=output_layer.dtype)
  #   output_layer = tf.reshape(tf.concat([output_layer, zero_padding], 1), input_layer.shape)

  ConvDeConvModel = tf.keras.Model(inputs=input_layer, outputs=[output_layer], name='ConvDeConv')
  ConvDeConvModel.summary()

  return ConvDeConvModel


def gen_model(input_shape=(None, 1566, 257)):
  """
    Define the model architecture

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """
  # Encoder = get_conv_encoder_model(input_shape)
  Encoder = get_encoder_model(input_shape)

  # model = Concatenate()([Encoder])

  return Encoder


def model_load(model_name='speech2speech'):
  """ Load a saved model if present """
  json_file = open(path(MODEL_DIR, model_name, 'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  # Load model weights
  model = model_from_json(loaded_model_json)
  model.load_weights(path(MODEL_DIR, model_name, 'model.h5'))

  return model


def save_model(model, model_name='speech2speech'):

  model_json = model.to_json()
  if not os.path.isdir(path(MODEL_DIR, model_name)):
    create_model_directory(model_name)
  with open(path(MODEL_DIR, model_name, 'model.json'), 'w') as json_file:
    json_file.write(model_json)

  model.save_weights(path(MODEL_DIR, model_name, 'model.h5'))

  return model