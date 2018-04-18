import numpy as np
from signals import Signal
from signals.generator import gaussian_white_noise
from signals.generator import multi_tone
from signals.utils import DataSet
from systems.volterra import Volterra
from models.volterra import Kernels

# system define
def generate_data():

    system = Volterra()

    # order 1
    system.set_kernel((0, ), 1.0)
    system.set_kernel((10, ), 0.2)
    system.set_kernel((20, ), 0.03)
    # order 2
    system.set_kernel((0, 0), 0.03)
    system.set_kernel((10, 20), 0.01)
    # order 3
    system.set_kernel((10, 20, 10), 0.001)
    system.set_kernel((0, 20, 10), 0.0005)
    # order 4
    system.set_kernel((0, 20, 10, 30), 0.0001)
    system.set_kernel((0, 10, 10, 10), 0.0001)
    # order 5
    system.set_kernel((0, 10, 20, 10, 30), 0.00001)
    system.set_kernel((0, 10, 20, 20, 30), 0.00001)
    # order 6
    system.set_kernel((0, 10, 20, 20, 30, 20), 0.000001)
    system.set_kernel((0, 10, 20, 20, 30, 30), 0.000001)
    # order 7
    system.set_kernel((0, 10, 20, 20, 30, 30, 20), 0.0000001)
    system.set_kernel((0, 10, 20, 20, 30, 30, 10), 0.0000001)
    # order 8
    system.set_kernel((0, 10, 20, 20, 30, 30, 10, 10), 0.0000001)
    system.set_kernel((0, 10, 20, 20, 30, 30, 10, 0), 0.0000001)
    # order 9
    system.set_kernel((0, 10, 20, 20, 30, 30, 10, 0, 10), 0.0000001)
    system.set_kernel((0, 10, 20, 20, 30, 30, 10, 0, 10), 0.0000001)


    sig_length = 100000
    intensity = 1
    fs = 20000
    noise = gaussian_white_noise(intensity, sig_length, fs)
    system_output_tra = system(noise)

    tra_features = noise[:80000]
    tra_targets = system_output_tra[:80000]

    val_features = noise[80000:]
    val_targets = system_output_tra[80000:]

    freqs = [5000, 6000]
    mulit_tone_signal = multi_tone(freqs, fs, 3)
    systemoutput_test = system(mulit_tone_signal)


    train_set = DataSet(tra_features, tra_targets, memory_depth=30)
    val_set = DataSet(val_features, val_targets, memory_depth=30)
    test_set = DataSet(mulit_tone_signal, systemoutput_test, memory_depth=30)

    return train_set, val_set, test_set


