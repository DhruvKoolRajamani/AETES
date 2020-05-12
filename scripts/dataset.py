import os, argparse, sys, shutil

import time, timeit, datetime, random

import numpy as np

from sklearn.model_selection import train_test_split

from add_echoes import add_echoes
from add_noise import add_noise, add_noise_gaussian
from preprocessing import process_sentence, download_corpus

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs, MODEL_DIR, create_model_directory, path

from copy import deepcopy

import librosa

# Global vars to change here
NFFT = 256
HOP_LENGTH = NFFT // 4
FS = 16000
STACKED_FRAMES = 8
CHUNK = 200
max_val = 1


def pad(data, length):
  return librosa.util.fix_length(data, length)


def get_fvt_stft(data_stft):
  data_stft_T = []
  for sentence in data_stft:
    data_stft_T.append(sentence.T)
  data_stft_T = np.array(data_stft_T)
  return data_stft_T


def get_padded_stft(data_stft):
  padded_stft = []
  for sentence in data_stft:
    padding = np.zeros((STACKED_FRAMES, NFFT//2 + 1), dtype=data_stft.dtype)
    padded_stft.append(np.concatenate((padding, sentence), axis=0))

  padded_stft = np.array(padded_stft)
  return padded_stft


def save_distributed_files(truth_type=None, corpus=None):
  """ 
  Save input and processed files for tests
  
  @param truth_type None for both, 'raw' for non processed and 'eq' for equalized
  """
  print("Started saving chunks of data")
  corpus_len = len(corpus)
  max_val = 0
  max_stft_len = 0
  # Find maximum length of time series data to pad
  for i in range(corpus_len):
    if len(corpus[i].data) > max_val:
      max_val = len(corpus[i].data)

  is_eq = False
  is_raw = False
  is_both = True

  if truth_type == 'eq':
    is_eq = True
    is_both = False
  elif truth_type == 'raw':
    is_raw = True
    is_both = False

  X = []
  y_eq = None
  y_raw = None

  if is_eq or is_both:
    y_eq = []
  if is_raw or is_both:
    y_raw = []

  memory_counter = 0
  pad_length = 0
  total_time = 0
  # Get each sentence from corpus and add random noise/echos/both to the input
  # and preprocess the output. Also pad the signals to the max_val
  for i in range(corpus_len):
    start = datetime.datetime.now()

    pad_length = max_val + NFFT//2
    if pad_length % STACKED_FRAMES != 0:
      pad_length += (STACKED_FRAMES - (pad_length%STACKED_FRAMES))

    # Original data in time domain
    data_orig_td = corpus[i].data.astype(np.float64)
    yi = pad(deepcopy(data_orig_td), pad_length)
    # Sampling frequency
    fs = corpus[i].fs

    # Pad transformed signals
    echosample = add_echoes(data_orig_td)
    noisesample = add_noise(data_orig_td, path(NOISE_DIR, "RainNoise.flac"))
    orig_sample = data_orig_td

    echosample = pad(echosample, pad_length)
    noisesample = pad(noisesample, pad_length)
    orig_sample = pad(orig_sample, pad_length)

    # Equalize data for high frequency hearing loss
    data_eq = None
    if is_eq or is_both:
      data_eq, _ = process_sentence(yi, fs=fs)
      yi_stft_eq = librosa.core.stft(data_eq, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
      yi_stft_eq = librosa.util.normalize(yi_stft_eq, axis=0)
      y_eq.append(np.abs(yi_stft_eq).T)

    # Use non processed input and pad as well
    data_raw = None
    if is_raw or is_both:
      data_raw = deepcopy(yi)
      yi_stft_raw = librosa.core.stft(data_raw, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
      yi_stft_raw = librosa.util.normalize(yi_stft_raw, axis=0)
      y_raw.append(np.abs(yi_stft_raw).T)

    #randomise which sample is input
    rand = random.randint(0, 1)
    random_sample_stft = None
    if rand == 0:
      random_sample_stft = librosa.core.stft(noisesample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
    else:
      random_sample_stft = librosa.core.stft(orig_sample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)

    max_stft_len = random_sample_stft.shape[1]
    random_sample_stft = librosa.util.normalize(random_sample_stft, axis=0)
    X.append(np.abs(random_sample_stft).T)

    # print("Padded {}".format(i))
    dt = datetime.datetime.now() - start
    total_time += dt.total_seconds() * 1000
    avg_time = total_time / (i+1)
    if (i % CHUNK == CHUNK - 1):
      print("Time taken for {}: {}ms".format(i, (i+1) * avg_time))
      print("Saving temp npy file to CHUNK {}".format(memory_counter))
      # Convert to np arrays
      size = 0
      if is_eq or is_both:
        y_eq_temp = np.array(y_eq)
        size += sys.getsizeof(y_eq_temp)

      if is_raw or is_both:
        y_raw_temp = np.array(y_raw)
        size += sys.getsizeof(y_raw_temp)

      X_temp = np.array(X)
      size += sys.getsizeof(X_temp)

      print("Memory used: {}".format(size / (1024*1024)))

      # Save files
      np.save(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(memory_counter)), X_temp, allow_pickle=True)

      if is_eq or is_both:
        np.save(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(memory_counter)), y_eq_temp, allow_pickle=True)
      if is_raw or is_both:
        np.save(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(memory_counter)), y_raw_temp, allow_pickle=True)

      X = []
      y_eq = None
      y_raw = None

      if is_eq or is_both:
        y_eq = []
      if is_raw or is_both:
        y_raw = []

      memory_counter += 1

  if corpus_len % CHUNK > 0:
    # Convert to np arrays
    if is_eq or is_both:
      y_eq_temp = np.array(y_eq)

    if is_raw or is_both:
      y_raw_temp = np.array(y_raw)

    X_temp = np.array(X)
    end_len = len(X)

    # Save temp files
    np.save(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(memory_counter)), X_temp, allow_pickle=True)

    if is_eq or is_both:
      np.save(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(memory_counter)), y_eq_temp, allow_pickle=True)
    if is_raw or is_both:
      np.save(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(memory_counter)), y_raw_temp, allow_pickle=True)
    print("Saved blocks {}:{}".format(0, memory_counter*CHUNK + end_len))
    memory_counter += 1

  memory_counter = np.array(memory_counter)
  max_stft_len = np.array(max_stft_len)
  corpus_len = np.array(corpus_len)

  np.save(os.path.join(PP_DATA_DIR, "model", "memory_counter"), memory_counter)
  np.save(os.path.join(PP_DATA_DIR, "model", "max_stft_len"), max_stft_len)
  np.save(os.path.join(PP_DATA_DIR, "model", "corpus_len"), corpus_len)

  return memory_counter, max_stft_len, truth_type, corpus_len


def concatenate_files(truth_type=None, delete_flag=False):
  """
  Save the distributed files.
  """
  is_eq = False
  is_raw = False
  is_both = True

  if truth_type == 'eq':
    is_eq = True
    is_both = False
  elif truth_type == 'raw':
    is_raw = True
    is_both = False

  memory_counter = np.load(os.path.join(PP_DATA_DIR, "model", "memory_counter.npy"))

  max_stft_len = np.load(os.path.join(PP_DATA_DIR, "model", "max_stft_len.npy"))

  corpus_len = np.load(os.path.join(PP_DATA_DIR, "model", "corpus_len.npy"))

  X = []
  y_eq = None
  y_raw = None
  if is_eq or is_both:
    y_eq = []
  if is_raw or is_both:
    y_raw = []

  end = 0
  for file_i in range(memory_counter):
    x = np.load(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(file_i)))
    if is_eq or is_both:
      y_eq_ = np.load(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(file_i)))
    if is_raw or is_both:
      y_raw_ = np.load(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(file_i)))
    for i in range(x.shape[0]):
      X.append(x[i])
      if is_eq or is_both:
        y_eq.append(y_eq_[i])
      if is_raw or is_both:
        y_raw.append(y_raw_[i])

    print("Loaded blocks {}".format(file_i))

  X = np.array(X)
  print("Loaded blocks {}:{}".format(end, X.shape[0]))

  if y_eq is None:
    y_raw = np.array(y_raw)
    np.savez_compressed(os.path.join(PP_DATA_DIR, "model", "speech"), inputs=X, truths_raw=y_raw)
  elif y_raw is None:
    y_eq = np.array(y_eq)
    np.savez_compressed(os.path.join(PP_DATA_DIR, "model", "speech"), inputs=X, truths_eq=y_eq)
  else:
    y_raw = np.array(y_raw)
    y_eq = np.array(y_eq)
    np.savez_compressed(os.path.join(PP_DATA_DIR, "model", "speech"), inputs=X, truths_raw=y_raw, truths_eq=y_eq)
  print("Saved speech.npz")

  if delete_flag:
    print("Deleting temp files")

    os.remove(os.path.join(PP_DATA_DIR, "model", "memory_counter.npy"))
    print("Deleted memory_counter")

    os.remove(os.path.join(PP_DATA_DIR, "model", "corpus_len.npy"))
    print("Deleted corpus_len")

    os.remove(os.path.join(PP_DATA_DIR, "model", "max_stft_len.npy"))
    print("Deleted max_stft_len")

    for file_i in range(memory_counter):
      os.remove(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(file_i)))
      print("Deleted inputs_{}".format(file_i))
      if is_raw or is_both:
        os.remove(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(file_i)))
        print("Deleted truths_raw_{}".format(file_i))
      if is_eq or is_both:
        os.remove(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(file_i)))
        print("Deleted truths_eq_{}".format(file_i))

  if is_eq and not is_both:
    return X, y_eq, None
  elif is_raw and not is_both:
    return X, y_raw, None

  return X, y_raw, y_eq


def generate_dataset(truth_type, delete_flag=False, speaker=[]):
  corpus = download_corpus(speaker=speaker)
  processed_data_path = os.path.join(PP_DATA_DIR, 'model')
  if not os.path.exists(processed_data_path):
    print("Creating preprocessed/model")
    create_preprocessed_dataset_directories()
  memory_counter, max_stft_len, truth_type, corpus_len = save_distributed_files(truth_type, corpus)
  X, y_raw, y_eq = concatenate_files(truth_type, delete_flag)
  print("Completed generating dataset")

  if y_eq is None:
    return X, y_raw, None
  elif y_raw is None:
    return X, None, y_eq

  return X, y_raw, y_eq


def stride_over(data):
  """
    Stride over the dataset to maintain latency.

    @param data [sentences, samples, bins, 1]
  """
  data_out = []

  size = 0
  for idx, sentence in enumerate(data):
    data_stacked_frames = []
    for i in range(sentence.shape[0] - STACKED_FRAMES):
      data_stacked_frames.append(sentence[i:i + STACKED_FRAMES].T)
      size += sys.getsizeof(data_stacked_frames)
    data_out.append(np.array(data_stacked_frames))
    if idx % 100 == 0:
      print("Memory used: {}".format(idx * size / (1024**3)))
      size = 0

  data_out = np.reshape(np.array(data_out), [-1, NFFT//2 + 1, STACKED_FRAMES, 1])
  return data_out


def load_dataset(truth_type, nfft=256, hop_len=256 // 4, fs=16000, stacked_frames=8, chunk=200):
  """ 
  Load a dataset and return the normalized test and train datasets
   appropriately split test and train sets.
  
  @param truth_type 'raw' for non processed and 'eq' for equalized
  @param **kwargs Dictionary of constants
  """
  global NFFT
  global HOP_LENGTH
  global FS
  global STACKED_FRAMES
  global CHUNK

  NFFT = nfft
  HOP_LENGTH = hop_len
  FS = fs
  STACKED_FRAMES = stacked_frames
  CHUNK = chunk

  X = None
  y = None

  X_train = None
  X_test = None
  X_val = None

  y_test = None
  y_train = None
  y_val = None

  Inputs = None
  Targets_eq = None
  Targets_raw = None
  y_raw = None
  y_eq = None

  if not os.path.isfile(os.path.join(PP_DATA_DIR, 'model', 'speech.npz')):
    Inputs, Targets_raw, Targets_eq = generate_dataset(truth_type, delete_flag=True, speaker=['clb'])
  else:
    SPEECH = np.load(os.path.join(PP_DATA_DIR, 'model', 'speech.npz'))
    Inputs = SPEECH['inputs'].astype(np.float32)
    try:
      if SPEECH['truths_raw'] is not None:
        Targets_raw = SPEECH['truths_{}'.format('raw')].astype(np.float32)
    except Exception as ex:
      print(ex)
      Targets_raw

    try:
      if SPEECH['truths_eq'] is not None:
        Targets_eq = SPEECH['truths_{}'.format('eq')].astype(np.float32)
    except Exception as ex:
      print(ex)
      Targets_eq = None

  X = Inputs
  if Targets_raw is not None:
    y_raw = Targets_raw
  if Targets_eq is not None:
    y_eq = Targets_eq

  if truth_type == 'raw':
    y = y_raw
  elif truth_type == 'eq':
    y = y_eq

  if truth_type is None:
    # Lets start with raw
    y = y_raw

  print(X.shape, y.shape)

  # Generate training and testing set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # Generate validation set
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)