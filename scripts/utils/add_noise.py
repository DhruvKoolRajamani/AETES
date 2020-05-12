import soundfile as sf
import numpy as np

from copy import deepcopy


def add_noise(data_new, data_noise_path, SNR=6):

  data_noise = sf.read(data_noise_path)

  j = 0
  for i in range(len(data_new)):
    data_new[i] = data_new[i] + data_noise[0][j] * SNR  #6 is noise strength

    j = j + 2

  #sf.write('new_file.flac', data_new, samplerate)

  return (data_new)


def add_noise_gaussian(sample):
  noise = np.random.normal(0, 1, sample.size)
  return sample + noise


#path = "84-121123-0000.flac"
#path = "arctic_a0001.wav"
#path_noise = "rain_noise.flac"

#data, samplerate = sf.read(path) ##Main File
#data_noise, samplerate_noise = sf.read(path_noise)
#print(samplerate,samplerate_noise)

#add_noise(data,samplerate, data_noise,samplerate_noise)
