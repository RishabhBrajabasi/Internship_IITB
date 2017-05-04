"""
Frequency response.py
Observe the frequency response of the different kind of filters with different order.

Author: Rishabh Brajabasi
Dtae: 4th May 2017
"""
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def cheby1_bandpass(low_cut, high_cut, fs, order=5, ripple=0.1):
    nqy = 0.5 * fs
    low = low_cut / nqy
    high = high_cut / nqy
    b, a = signal.cheby1(order, ripple, [low, high], 'bandpass')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def cheby1_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby1_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def cheby2_bandpass(low_cut, high_cut, fs, order=5, ripple=5):
    nqy = 0.5 * fs
    low = low_cut / nqy
    high = high_cut / nqy
    b, a = signal.cheby2(order, ripple, [low, high], 'bandpass')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#
fs = 16000.0
lowcut = 300.0
highcut = 2500.0
plt.figure(1)
plt.clf()
#----------------------------------------------------------------------------------------------------------------------#
for order in [9]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = signal.freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#----------------------------------------------------------------------------------------------------------------------#
for order in [8]:
    for ripple in [0.01]:
        b, a = cheby1_bandpass(lowcut, highcut, fs, order=order, ripple=ripple)
        w, h = signal.freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d " % order)
#----------------------------------------------------------------------------------------------------------------------#
for order in [6]:
    for ripple in [10]:
        b, a = cheby2_bandpass(lowcut, highcut, fs, order=order, ripple=ripple)
        w, h = signal.freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="ripple = %d" % ripple)
#----------------------------------------------------------------------------------------------------------------------#
plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
plt.show()
#----------------------------------------------------------------------------------------------------------------------#
