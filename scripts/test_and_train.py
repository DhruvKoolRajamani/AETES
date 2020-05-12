import os, argparse, sys, shutil, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
utilsdir = currentdir + '/utils'
sys.path.insert(0, parentdir)
sys.path.insert(1, utilsdir)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime, random

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import librosa, sounddevice

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs, MODEL_DIR, create_model_directory, path

from dataset import load_dataset, generate_dataset, get_padded_stft, pad
from model import save_model, model_load, gen_model, get_encoder_model, get_conv_encoder_model

from copy import deepcopy

# Global vars to change here
NFFT = 256
HOP_LENGTH = NFFT // 4
FS = 16000
STACKED_FRAMES = 8
CHUNK = 200
max_val = 1

# Model Parameters
BATCH_SIZE = 8
EPOCHS = 500
METRICS = ['mse', 'accuracy', tf.keras.metrics.RootMeanSquaredError()]
LOSS = 'mse'
OPTIMIZER = 'adam'


class BatchIdCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


def normalize_sample(X):
  """ Normalize the sample """

  if X.shape[1] != NFFT//2 + 1:
    for i, x in enumerate(X):
      # axis = 0 is along column. For Conv I'm not transposing the array -> Each row is a frequence, each column is time
      # Makes sense to normalize of every time frame and not every frequency bin.
      X_norm = librosa.util.normalize(x, axis=1)
      X[i] = X_norm
  else:
    X = librosa.util.normalize(X, axis=1)

  return X


def l2_norm(vector):
  return np.square(vector)


def SDR(denoised, cleaned, eps=1e-7):  # Signal to Distortion Ratio
  a = l2_norm(denoised)
  b = l2_norm(denoised - cleaned)
  a_b = a / b
  return np.mean(10 * np.log10(a_b + eps))


def play_sound(x, y_pred, y, fs=NFFT):
  sounddevice.play(x, fs)
  plt.plot(x)
  plt.show()

  sounddevice.play(y_pred, fs)
  plt.plot(y_pred)
  plt.show()

  sounddevice.play(y, fs)
  plt.plot(y)
  plt.show()

  return


def test_and_train(model_name='speech2speech', retrain=True):
  """ 
    Test and/or train on given dataset 

    @param model_name name of model to save.
    @param retrain True if retrain, False if load from pretrained model
  """

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(
                                                                        'raw',
                                                                        nfft=NFFT,
                                                                        hop_len=HOP_LENGTH,
                                                                        fs=FS,
                                                                        stacked_frames=STACKED_FRAMES,
                                                                        chunk=CHUNK
                                                                      )
  model = None

  X_train_norm = normalize_sample(X_train)
  X_val_norm = normalize_sample(X_val)
  X_test_norm = normalize_sample(X_test)

  y_train_norm = normalize_sample(y_train)
  y_val_norm = normalize_sample(y_val)
  y_test_norm = normalize_sample(y_test)

  print("X shape:", X_train_norm.shape, "y shape:", y_train_norm.shape)

  # Xtn_strided = stride_over(X_train_norm)
  # Xvn_strided = stride_over(X_val_norm)
  # Xten_strided = stride_over(X_test_norm)

  # Xtn_reshape = Xtn_strided
  # Xvn_reshape = Xvn_strided
  # Xten_reshape = Xten_strided

  # ytn_reshape = y_train_norm.reshape(-1, NFFT//2 + 1, 1, 1)
  # yvn_reshape = y_val_norm.reshape(-1, NFFT//2 + 1, 1, 1)
  # yten_reshape = y_test_norm.reshape(-1, NFFT//2 + 1, 1, 1)

  # train_dataset = tf.data.Dataset.from_tensor_slices((Xtn_reshape,
  #                                                     ytn_reshape)).batch(X_train_norm.shape[1]).shuffle(X_train.shape[0]).repeat()
  # val_dataset = tf.data.Dataset.from_tensor_slices((Xvn_reshape, yvn_reshape)).batch(X_val_norm.shape[1]).repeat(1)

  # train_dataset = tf.data.Dataset.from_tensor_slices((X_train_norm, y_train_norm)).batch(BATCH_SIZE).shuffle(BATCH_SIZE).repeat()
  # val_dataset = tf.data.Dataset.from_tensor_slices((X_val_norm, y_val_norm)).batch(BATCH_SIZE).repeat(1)

  # print(list(train_dataset.as_numpy_iterator())[0])

  # Scale the sample X and get the scaler
  # scaler = scale_sample(X)

  # Check if model already exists and retrain is not being called again
  if (os.path.isfile(os.path.join(MODEL_DIR, model_name, 'model.json')) and not retrain):
    model = model_load(model_name)
    # Compile the model
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
  else:
    if not os.path.isdir(os.path.join(MODEL_DIR, model_name)):
      create_model_directory(model_name)

    baseline_val_loss = None

    model = None

    # model = gen_model(tuple(Xtn_reshape.shape[1:]))
    model = gen_model(tuple(X_train_norm.shape[1:]))
    print('Created Model...')

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    print('Metrics for Model...')

    # print(list(train_dataset.as_numpy_iterator())[0])

    tf.keras.utils.plot_model(model, show_shapes=True, dpi=96, to_file=os.path.join(MODEL_DIR, model_name, 'model.png'))
    print(model.metrics_names)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if (os.path.isfile(path(MODEL_DIR, model_name))):
      model.load_weights(path(MODEL_DIR, model_name))
      baseline_val_loss = model.evaluate(X_val_norm, y_val_norm)[0]
      print(baseline_val_loss)
      early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        baseline=baseline_val_loss
      )

    log_dir = os.path.join(LOGS_DIR, 'files', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

    model_checkpoint_callback = ModelCheckpoint(
      monitor='val_loss',
      filepath=os.path.join(MODEL_DIR,
                            model_name,
                            'model.h5'),
      save_best_only=True,
      save_weights_only=False,
      mode='min'
    )

    # fit the keras model on the dataset
    cbs = [early_stopping_callback, model_checkpoint_callback]  # tensorboard_callback

    model.fit(
      X_train_norm,
      y_train_norm,
      epochs=EPOCHS,
      validation_data=(X_val_norm,
                       y_val_norm),
      verbose=1,
      callbacks=cbs,
      batch_size=BATCH_SIZE
    )
    print('Model Fit...')

    model = save_model(model, model_name)

  # model = model_load(model_name)

  [loss, mse, accuracy, rmse] = model.evaluate(X_test_norm, y_test_norm, verbose=0)  # _, mse, accuracy =
  print('Testing accuracy: {}, Testing MSE: {}, Testing Loss: {}, Testing RMSE: {}'.format(accuracy * 100, mse, loss, rmse))

  # # Randomly pick 1 test
  idx = 32
  print(idx)
  X = X_test_norm[idx]
  y = y_test_norm[idx]
  # y = y_test_norm[idx].reshape(-1, NFFT//2 + 1)
  # min_y, max_y = np.min(y_test_norm[idx]), np.max(y_test_norm[idx])
  # min_x, max_x = np.min(y_test_norm[idx]), np.max(y_test_norm[idx])
  # print("MinY: {}\tMaxY{}".format(min_y, max_y))
  # print("MinX: {}\tMaxX{}".format(min_x, max_x))

  X = np.expand_dims(X, axis=0)
  # X = stride_over(X)

  # mean = np.mean(X)
  # std = np.std(X)
  # X = (X-mean) / std

  print(X.shape)

  # y_pred = model.predict(X)
  y_pred = np.squeeze(model.predict(X), axis=0)
  # y_pred = y_pred.reshape(-1, NFFT//2 + 1)

  print(y.shape)
  print(y_pred.shape)

  y = y.T
  y_pred = y_pred.T
  X_test_norm = X_test_norm[idx].T

  # GriffinLim Vocoder
  output_sound = librosa.core.griffinlim(y_pred)
  input_sound = librosa.core.griffinlim(X_test_norm)
  target_sound = librosa.core.griffinlim(y)

  # Play and plot all
  play_sound(input_sound, output_sound, target_sound, FS)

  if not os.path.isdir(os.path.join(MODEL_DIR, model_name, 'audio_output')):
    create_model_directory(os.path.join(model_name, 'audio_output'))

  librosa.output.write_wav(path(MODEL_DIR, model_name, 'audio_output', 'input.wav'), input_sound, sr=FS, norm=True)
  librosa.output.write_wav(path(MODEL_DIR, model_name, 'audio_output', 'target.wav'), target_sound, sr=FS, norm=True)
  librosa.output.write_wav(path(MODEL_DIR, model_name, 'audio_output', 'predicted.wav'), output_sound, sr=FS, norm=True)

  return


def prepare_input_features(stft_features):
  # Phase Aware Scaling: To avoid extreme differences (more than
  # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
  noisySTFT = np.concatenate([stft_features[:, 0:STACKED_FRAMES - 1], stft_features], axis=1)
  stftSegments = np.zeros((NFFT//2 + 1, STACKED_FRAMES, noisySTFT.shape[1] - STACKED_FRAMES + 1))

  for index in range(noisySTFT.shape[1] - STACKED_FRAMES + 1):
    stftSegments[:, :, index] = noisySTFT[:, index:index + STACKED_FRAMES]
  return stftSegments


if __name__ == '__main__':
  clear_logs()
  print("Cleared Tensorboard Logs...")
  test_and_train(model_name='ConvLSTM', retrain=False)
  # print(timeit.timeit(generate_dataset, number=1))
  # generate_dataset(truth_type='raw')
