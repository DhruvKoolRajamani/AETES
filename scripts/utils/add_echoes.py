import matplotlib.pyplot as plt
import pyroomacoustics as pra
import sounddevice
import numpy as np

def add_echoes(sample):
    room = pra.ShoeBox([20, 20, 20], fs=16000, absorption=0.45, max_order=17)

    room.add_source([10.5, 3.73, 1.76], signal=sample, delay=1.3)

    R = np.c_[[6.3, 4.87, 1.2],[6.3, 4.93, 1.2]]

    mic_array = pra.MicrophoneArray(R, room.fs)

    room.add_microphone_array(mic_array)

    room.compute_rir()

    room.simulate()

    return room.mic_array.signals[1,:]
