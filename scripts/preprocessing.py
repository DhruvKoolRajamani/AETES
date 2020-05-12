import os, argparse, sys, shutil
from pathlib import Path

import datetime
import time

import numpy as np

from scipy.interpolate import interp1d

import scipy.io as sio
import scipy.signal
from scipy.signal import spectrogram, periodogram, welch, wiener
from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.datasets import Sample, Dataset, CMUArcticCorpus
from pyroomacoustics.datasets import CMUArcticSentence

import librosa
from librosa.core import power_to_db, db_to_power, amplitude_to_db, stft, istft, fft_frequencies, db_to_amplitude, power_to_db, db_to_power
from librosa.display import specshow
import librosa.filters
import librosa.effects

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
import constants

from copy import deepcopy


def get_audiogram(x, y, order='cubic'):
  """
  'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 
  'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline 
  interpolation of zeroth, first, second or third order; 'previous' and 'next' 
  simply return the previous or next value of the point) or as an integer 
  specifying the order of the spline interpolator to use. Default is 'linear'
  """
  X = sorted(x)
  Y = [i for _, i in sorted(zip(x, y))]

  if float(X[0]) is not 0.:
    tempX = X
    tempY = Y
    X = [0.]
    Y = [tempY[0]]
    X.extend(tempX)
    Y.extend(tempY)

  if float(X[-1]) is not 8000.:
    X.extend([8000.])
    Y.extend([Y[-1]])

  z = interp1d(X, Y, kind=order)

  return z, (X, Y)


def process_audiogram(x_audiogram, y_audiogram, freq, plot=False):

  audiogram_eq, (x_audiogram, y_audiogram) = get_audiogram(x_audiogram, y_audiogram, 'linear')

  x_range_audiogram = freq[freq >= np.min(x_audiogram)]
  x_range_audiogram = x_range_audiogram[x_range_audiogram <= np.max(x_audiogram)]

  audiogram = np.array([audiogram_eq(x) for x in x_range_audiogram])
  audiogram = (x_range_audiogram, audiogram)

  # if plot:
  #   plot_audiogram(audiogram)

  return audiogram


def get_fig_nums():
  num = plt.get_fignums()
  if not num:
    return 0
  else:
    return num[-1]


def plot_audiogram(source, audiogram, modulated=None, extra_fram_idx_arg=None):
  f = plt.figure(get_fig_nums() + 1)

  if extra_fram_idx_arg is not None:
    plt.title('Audiograms: {}'.format(extra_fram_idx_arg))
  else:
    plt.title('Audiograms')
  plt.grid(True)

  plt.plot(source[0], source[1], 'r-')
  plt.plot(audiogram[0], audiogram[1], 'b-')
  if modulated is not None:
    plt.plot(modulated[0], modulated[1], 'g-')

  plt.legend(['source', 'audiogram', 'modulated'] if modulated is not None else ['source', 'audiogram'])

  plt.ylim((-40, 120))
  plt.xlim((0, 8 * 1e3))
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Amplitude [dB]')
  # plt.show()


def butter_filter(data, cutoff, fs, order, btype='low'):
  normal_cutoff = cutoff / (fs//2)  # is this even required?
  # Get the filter coefficients
  b, a = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False)
  y = scipy.signal.filtfilt(b, a, data)
  return y


def design_filter(order, cutoff, btype='lowpass', ftype='butter', fs=None, freqs=(512 // 2), rp=None, rs=None):
  b, a = scipy.signal.iirfilter(order, cutoff, btype=btype, analog=False, ftype=ftype, output='ba', fs=fs, rp=rp, rs=rs)

  a = 1 if a is None else a
  w, h = scipy.signal.freqz(b, a, worN=freqs, fs=fs)

  return w, h


def apply_filter(data, filter_vals, scale=None):

  dtype = np.float

  if np.any(np.iscomplex(data)) or np.any(np.iscomplex(filter_vals)):
    dtype = np.complex

  data_T = data.T
  data_T_filtered = np.zeros(shape=data_T.shape, dtype=dtype)

  for idx, sample in enumerate(data_T):

    h_new = np.multiply(filter_vals, sample)
    # h_new = scipy.signal.convolve(sample, filter_vals, 'same')
    # for i, h in enumerate(h_new):
    # if abs(h) < 0:
    #   h_new[i] = 0.

    if scale is not None:
      min_data = np.min(h_new)
      breath_indices = h_new < (min_data + 12.5)
      h_new[breath_indices] = min_data

      h_new = np.multiply(scale, h_new)

    data_T_filtered[idx], phase = librosa.core.magphase(h_new)

  data_return = data_T_filtered.T

  return data_return


def nlfc(data_orig_mod, freq, n, db_ref, start_freq, compression_ratio, compression_frequency, compression_nfft, compress=True):
  order = 32
  ftype = 'butter'
  rp = None
  rs = None
  if db_ref is not None:
    ftype = 'cheby2'
    rp = 0.1
    rs = 80
  # Lowpass filter
  w_lp, h_lp = design_filter(order=order, cutoff=[start_freq], fs=fs, freqs=freq, ftype=ftype, rp=rp, rs=rs)
  data_low_pass = apply_filter(data_orig_mod, h_lp)

  plt.figure(get_fig_nums() + 1)
  plt.title('low pass')
  specshow(amplitude_to_db(data_low_pass) if db_ref is None else data_low_pass, x_axis='time', y_axis='linear')
  plt.colorbar()

  # Highpass filter
  w_hp, h_hp = design_filter(order=order, cutoff=[start_freq], btype='highpass', fs=fs, freqs=freq, ftype=ftype, rp=rp, rs=rs)
  data_high_pass = apply_filter(data_orig_mod, h_hp)

  plt.figure(get_fig_nums() + 1)
  plt.title('high pass before')
  specshow(amplitude_to_db(data_high_pass) if db_ref is None else data_high_pass, x_axis='time', y_axis='linear')
  plt.colorbar()

  # Convert signals back to time domain
  data_high_pass_td = istft(data_high_pass, center=center, length=n)

  # Resample time-domain signal
  data_high_pass_td = librosa.core.resample(data_high_pass_td, fs, compression_frequency)
  n2 = len(data_high_pass_td)

  # FFT with 172 Hz bins and 1.45ms rate
  resample_freqs = fft_frequencies(sr=compression_frequency, n_fft=compression_nfft)
  data_hp_padded = librosa.util.fix_length(data_high_pass_td, n2 + compression_nfft//2)

  adf = lambda frq_list: abs(frq_list - start_freq)
  sf_idx = np.where(freq == min(freq, key=adf))[0][0]

  modulated_carrier_freqs_dict = {}
  modulated_carrier_freqs = []

  for i, f in enumerate(freq[freq < freq[sf_idx]]):
    f_idx = sf_idx + i
    adf_in = lambda frq_list: abs(frq_list - f)
    idx = np.where(resample_freqs == min(resample_freqs, key=adf_in))[0][0]

    modulated_carrier_freqs_dict.update({(idx, resample_freqs[idx]): (idx, resample_freqs[idx])})
    modulated_carrier_freqs.append(idx)

  for i, f in enumerate(freq[freq >= freq[sf_idx]]):
    f_idx = sf_idx + i
    adf_in = lambda frq_list: abs(frq_list - f)
    idx = np.where(resample_freqs == min(resample_freqs, key=adf_in))[0][0]

    f_out = start_freq**(1 - compression_ratio) * f**(compression_ratio)
    adf_out = lambda frq_list: abs(frq_list - f_out)
    frq_idx = np.where(resample_freqs == min(resample_freqs, key=adf_out))[0][0]

    modulated_carrier_freqs_dict.update({(idx, resample_freqs[idx]): (frq_idx, resample_freqs[frq_idx])})
    modulated_carrier_freqs.append(frq_idx)

  data_hp_resampled = stft(data_hp_padded, n_fft=512, center=center)
  modulated_carrier = np.zeros(data_hp_resampled.T.shape, dtype=np.complex)
  data_hp_resampled_T = data_hp_resampled.T

  for time_idx, sample in enumerate(data_hp_resampled_T):
    modulated_carrier[time_idx] = sample[modulated_carrier_freqs]

  modulated_carrier = modulated_carrier.T
  modulated_carrier_td = istft(modulated_carrier, center=center, length=n2)

  # Resample time-domain signal
  data_high_pass_td_new = librosa.core.resample(modulated_carrier_td, compression_frequency, fs)

  # Pad the data since istft will drop any data in the last frame if samples are
  # less than n_fft.
  data_pad_new = librosa.util.fix_length(data_high_pass_td_new, n + n_fft//2)

  data_high_pass_modulated = stft(data_pad_new, n_fft=n_fft, center=center)

  # Set theory yay!
  # data_double = data_high_pass_modulated - data_low_pass
  data_stacked = data_high_pass_modulated if compress else data_high_pass + data_low_pass
  if db_ref is not None:
    dmag, dphase = librosa.core.magphase(data_stacked)
    dmag = db_to_amplitude(dmag, ref=db_ref)
    data_stacked = dmag * dphase

  plt.figure(get_fig_nums() + 1)
  plt.title('high pass after')
  specshow(amplitude_to_db(data_high_pass_modulated) if db_ref is None else data_high_pass_modulated, x_axis='time', y_axis='linear')
  plt.colorbar()
  # plt.show()

  plt.figure(get_fig_nums() + 1)
  plt.title('stacked')
  specshow(amplitude_to_db(data_stacked) if db_ref is None else data_stacked, x_axis='time', y_axis='linear')
  plt.colorbar()

  return data_stacked


def eq(data_db, eq_freqs, audiogram, sr, n_fft, db_ref, data_amp, phase, plot=False):

  data_raw = deepcopy(data_amp)
  # mag, phase = librosa.core.magphase(data_raw)

  data = deepcopy(data_db)

  if db_ref is None:
    data = amplitude_to_db(data, ref=np.max)

  min_data = np.min(data)

  # Add 80 to magnitude of data
  # breath_indices = data < (min_data + 12.5)
  # data[breath_indices] = min_data
  data += abs(min_data)

  # half everything?
  data_halved = data / 2

  if plot:
    plt.figure(get_fig_nums() + 1)
    plt.title('data halved')
    specshow(amplitude_to_db(data_halved) if db_ref is None else data_halved, x_axis='time', y_axis='linear')
    plt.colorbar()

  eq_freqs = np.array(eq_freqs)
  sample_rates = np.array([sr for _ in eq_freqs])

  fb_sos, _ = librosa.filters._multirate_fb(eq_freqs, sample_rates, Q=25.0, passband_ripple=0.01, stopband_attenuation=80)
  max_scaling = np.max(audiogram[1])
  scaled_audiogram = audiogram[1] / 120.
  fb = []
  if plot:
    plt.figure(get_fig_nums() + 1)
    plt.title('filterbank')
  for sos in fb_sos:
    freqs, fb_filter = scipy.signal.sosfreqz(np.array(sos), n_fft//2 + 1, fs=sr)
    fb.append(fb_filter)
    if plot:
      plt.plot(freqs, np.abs(fb_filter))

  fb = np.array(fb)
  fb_smoothened = np.zeros((fb.shape[1], ), dtype=np.float)
  for filt in fb:
    fb_smoothened += abs(filt)

  data_halved_complete = data_halved
  filtered_data = apply_filter(data_halved_complete, abs(fb_smoothened), scaled_audiogram)
  # fb_mag, fb_phase = librosa.core.magphase(filtered_data)
  fb_mag = filtered_data
  fb_mag += data_halved

  if plot:
    plt.figure(get_fig_nums() + 1)
    plt.title('filtered')
    specshow(amplitude_to_db(abs(fb_mag)) if db_ref is None else abs(fb_mag), x_axis='time', y_axis='linear')
    plt.colorbar()

  db_shift = 65
  max_filtered_dt = np.max(fb_mag)
  # print(max_filtered_dt - db_shift)
  fb_mag -= (max_filtered_dt - db_shift)

  if plot:
    plt.figure(get_fig_nums() + 1)
    plt.title('final output')
    specshow(amplitude_to_db(abs(fb_mag)) if db_ref is None else abs(fb_mag), x_axis='time', y_axis='linear')
    plt.colorbar()

  # Lowpass filter
  start_freq = 4500
  order = 32
  ftype = 'butter'
  rp = None
  rs = None
  if db_ref is not None:
    ftype = 'cheby2'
    rp = 0.01
    rs = 80

  w_lp, h_lp = design_filter(order=order, cutoff=[start_freq], fs=sr, freqs=n_fft//2 + 1, ftype=ftype, rp=rp, rs=rs)
  data_low_pass = apply_filter(fb_mag, abs(h_lp))

  dlp_min = np.min(data_low_pass)
  remove_bacground = data_low_pass < 30
  data_low_pass[remove_bacground] = dlp_min

  if plot:
    plt.figure(get_fig_nums() + 1)
    plt.title('low pass')
    specshow(amplitude_to_db(abs(data_low_pass)) if db_ref is None else abs(data_low_pass), x_axis='time', y_axis='linear')
    plt.colorbar()

  data_out = db_to_amplitude(data_low_pass, ref=1)
  data_out_noisy = scipy.signal.wiener(data_out, mysize=[n_fft, 3], noise=0.01)

  return data_out_noisy


def process_sentence(data, fs, n_fft=512, center=True, plot=False):

  # Default settings for speech analysis
  # n_fft = 512 to provide 25ms-35ms samples
  # (https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc)
  n = len(data)

  # Pad the data since istft will drop any data in the last frame if samples are
  # less than n_fft.
  data_pad = librosa.util.fix_length(data, n + n_fft//2)
  # data_pad = data

  # Get the frequency distribution
  freq = fft_frequencies(sr=fs, n_fft=n_fft)

  # Get the equation and freq, db array from the audiogram provided
  x_audiogram = [125, 250, 500, 1000, 1500, 2000, 2400, 2800, 3000]
  y_audiogram = [10, 15, 0, -10, -30, -35, -40, -50, -60, -70]
  # y_audiogram = [0, 0, 0, 0, 0, 0, 0]
  audiogram = process_audiogram(x_audiogram, y_audiogram, freq, plot)

  # Preemphasis to increase amplitude of high frequencies
  # data_emph_filt = librosa.effects.preemphasis(data_pad)

  # Perform the stft, separate magnitude and save phase for later (important)
  data_pad_stft = stft(data_pad, n_fft=n_fft, center=center)
  mag, phase = librosa.core.magphase(data_pad_stft)

  db_ref = np.max

  # Consider using frequencies of phonomes.
  # eq_freqs = [125, 250, 500, 1000, 1500, 2000, 4000]
  eq_freqs = librosa.filters.mel_frequencies(n_mels=12, fmin=100., fmax=5000., htk=True)
  # mel_fb = librosa.filters.mel(
  #   fs,
  #   n_fft,
  #   n_mels=8,
  #   fmin=315.,
  #   fmax=8000.,
  #   norm=None
  # )
  # mel_fb_smoothened = np.zeros((mel_fb.shape[1]))

  # for bands in mel_fb:
  #   mel_fb_smoothened += bands

  # fb_filtered_raw = apply_filter(mag, mel_fb_smoothened)
  # fb_filtered = fb_filtered_raw

  # Normalize to 60db
  fb_filtered = mag
  if db_ref is not None:
    fb_filtered = amplitude_to_db(mag, ref=db_ref)

  # Multiply new magnitude with saved phase to reconstruct sentence
  data_orig_mod = mag * phase  # mag_inv

  data_proc_mag = eq(fb_filtered, eq_freqs, audiogram, fs, n_fft, db_ref, mag, phase, plot)

  data_proc_mag_td = istft(data_proc_mag, center=center, length=n)
  if plot:
    librosa.output.write_wav(
      os.path.join(constants.PP_DATA_DIR,
                   "audio",
                   'preprocessed_unfiltered_magnitude.wav'),
      data_proc_mag_td,
      fs,
      norm=True
    )

  data_proc_magphase_td = istft(data_proc_mag * phase, center=center, length=n)
  if plot:
    librosa.output.write_wav(
      os.path.join(constants.PP_DATA_DIR,
                   "audio",
                   'preprocessed_unfiltered_magphase.wav'),
      data_proc_magphase_td,
      fs,
      norm=True
    )

  data_proc_griffinlim_td = librosa.core.griffinlim(data_proc_mag)
  if plot:
    librosa.output.write_wav(
      os.path.join(constants.PP_DATA_DIR,
                   "audio",
                   'preprocessed_unfiltered_griffinlim.wav'),
      data_proc_griffinlim_td,
      fs,
      norm=True
    )

  data_proc = data_proc_mag * phase

  # # compression parameters
  # start_freq = 2400.
  # compression_ratio = 1. / 2.  #:1
  # compression_frequency = 12000
  # compression_nfft = 512

  # data_stacked = nlfc(
  #   data_orig_mod,
  #   freq,
  #   n,
  #   db_ref,
  #   start_freq,
  #   compression_ratio,
  #   compression_frequency,
  #   compression_nfft,
  #   compress=False
  # )

  # Perform the inverse stft
  data_mod = istft(data_proc, center=center, length=n)

  # Denoising

  denoised_signal = data_mod
  # denoised_signal = pra.denoise.apply_subspace(data_mod, frame_len=64, mu=2, lookback=20, skip=1, thresh=0.85, data_type='float64')

  # denoised_signal = pra.denoise.apply_iterative_wiener(data_mod, frame_len=n_fft, lpc_order=12, iterations=2, alpha=0.8, thresh=0.05)

  if plot:
    plt.figure(get_fig_nums() + 1)
    plt.title('final output time domain')
    plt.plot(denoised_signal)

  # Normalize
  denoised_signal = librosa.util.normalize(denoised_signal)
  if plot:
    librosa.output.write_wav(os.path.join(constants.PP_DATA_DIR, "audio", 'preprocessed_filtered.wav'), denoised_signal, fs, norm=True)

  return denoised_signal, audiogram


def obtain_fft_in_db(data, n_fft):
  # Viewing the fft of the entire signal
  data, phase = librosa.core.magphase(fft(data, n=n_fft//2 + 1))
  data = librosa.util.normalize(data)
  data = amplitude_to_db(data)
  data += 120

  return data


def download_corpus(download_flag=True, speaker=[]):
  # Download the corpus, be patient
  corpus = None
  if os.path.exists(ARCTIC_DIR):
    if os.path.isfile(os.path.join(ARCTIC_DIR, 'cmu_us_aew_arctic/wav/arctic_a0001.wav')):
      download_flag = False
    if len(speaker) < 1:
      corpus = CMUArcticCorpus(basedir=ARCTIC_DIR, download=download_flag)
    else:
      corpus = CMUArcticCorpus(basedir=ARCTIC_DIR, download=download_flag, speaker=speaker)
  else:
    create_arctic_directory()
    corpus = CMUArcticCorpus(basedir=ARCTIC_DIR, download=download_flag)

  return corpus


def save(data, filename, fs, norm=True):
  librosa.output.write_wav(filename, data, fs, norm)

  return


def gen_processed_and_save_wav(corpus, play=False):
  # Save input and processed files for tests
  if not os.path.exists(constants.WAVE_DIR):
    constants.create_saved_wav_directory()

  folder_dir = os.path.join(DATA_DIR, constants.WAVE_DIR)
  # play = False
  for sentence_idx in range(10):
    orig_data, proc_data = test(corpus, sentence_idx, play=play)

    exp_dir = '{}'.format(sentence_idx)
    exp_dir = os.path.join(folder_dir, exp_dir)

    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    else:
      shutil.rmtree(exp_dir)
      os.makedirs(exp_dir)

    orig_path = os.path.join(exp_dir, '{}_orig'.format(sentence_idx))
    proc_path = os.path.join(exp_dir, '{}_proc'.format(sentence_idx))

    orig_data_shifted = orig_data  # .astype(np.float64)
    proc_data_shifted = proc_data  # .astype(np.float64)

    print("Writing exp: {} to {}".format(sentence_idx, exp_dir))
    librosa.output.write_wav(orig_path + '.wav', orig_data_shifted, corpus[sentence_idx].fs, norm=True)
    librosa.output.write_wav(proc_path + '.wav', proc_data_shifted, corpus[sentence_idx].fs, norm=True)

    # from pydub import AudioSegment
    # print("Converting to mp3")
    # # orig_sound = AudioSegment(orig_data_shifted.tobytes())
    # orig_sound = AudioSegment.from_wav(orig_path + '.wav')
    # orig_sound.export(orig_path + '.mp3', format='mp3')

    # proc_sound = AudioSegment.from_wav(proc_path + '.wav')
    # proc_sound.export(proc_path + '.mp3', format='mp3')


def test(corpus, sentence_idx=1, n_fft=512, center=True, play=False):
  # Default settings for speech analysis
  # n_fft = 512 to provide 25ms-35ms samples
  # (https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc)

  start = datetime.datetime.now()

  # Get the timeseries and sampling frequency
  data = deepcopy(corpus[sentence_idx].data).astype(np.float64)
  data_raw = deepcopy(data)
  fs = corpus[sentence_idx].fs
  print(corpus.sentences[sentence_idx])

  if play:
    plt.figure(get_fig_nums() + 1)
    plt.title('input signal')
    plt.plot(data_raw)

  librosa.output.write_wav(os.path.join(constants.PP_DATA_DIR, "audio", 'raw.wav'), data, fs, norm=True)

  # Get the frequency distribution
  freq = fft_frequencies(sr=fs, n_fft=n_fft)

  data_mod, audiogram = process_sentence(data, fs, n_fft=n_fft, center=True, plot=play)

  # librosa.feature.chroma_cqt(

  # source = obtain_fft_in_db(corpus[sentence_idx].data, n_fft)
  # modulated = obtain_fft_in_db(data_mod, n_fft)
  # plot_audiogram((freq, source), audiogram, (freq, modulated))
  dt = datetime.datetime.now() - start
  print("Time taken: {}".format(dt.total_seconds() * 1000))
  if play:
    corpus[sentence_idx].data = data_mod
    corpus[sentence_idx].play()
    plt.show()

  plt.close('all')

  return data_raw, data_mod


if __name__ == '__main__':

  corpus = download_corpus(speaker=['clb'])

  test(corpus, sentence_idx=1, play=True)
  # gen_processed_and_save_wav(corpus, False)