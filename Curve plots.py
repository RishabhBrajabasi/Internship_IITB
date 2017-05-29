from __future__ import division

import math
import csv
import glob
import os
import re
import sys
import win32api
import winsound
from datetime import datetime
from shutil import copyfile
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

startTime = datetime.now()  # To calculate the run time of the code.
#----------------------------------------------------------------------------------------------------------------------#
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
#----------------------------------------------------------------------------------------------------------------------#
def butter_low_pass(low_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    b, a = butter(order, low, btype='low')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_low_pass_filter(data_low, lowcut, fs, order=5):
    b, a = butter_low_pass(lowcut, fs, order=order)
    y = lfilter(b, a, data_low)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def butter_band_pass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_band_pass_filter(data_band, lowcut, highcut, fs, order=5):
    b, a = butter_band_pass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data_band)
    return y
#----------------------------------------------------------------------------------------------------------------------#

file_no = '42'
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + '.wav'
textgridFA = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + 'FA.TextGrid'
text_grid_1 = open(textgridFA, 'r')
data_1 = text_grid_1.read()
time_1 = []
counter = 0
#----------------------------------------------------------------------------------------------------------------------#
for m in re.finditer('text = "', data_1):
    if data_1[m.start() - 33] == '=':
        time_1.append(float(
            data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] + data_1[m.start() - 29] +
            data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
        time_1.append(float(
            data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] +
            data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] +
            data_1[m.start() - 5]))
    else:
        time_1.append(float(
            data_1[m.start() - 33] + data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] +
            data_1[m.start() - 29] + data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
        time_1.append(float(
            data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] +
            data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] +
            data_1[m.start() - 5]))
#----------------------------------------------------------------------------------------------------------------------#
    if data_1[m.start() + 9] == '"':
        time_1.append((data_1[m.start() + 8]))
    elif data_1[m.start() + 10] == '"':
        time_1.append((data_1[m.start() + 8] + data_1[m.start() + 9]))
    else:
        time_1.append((data_1[m.start() + 8] + data_1[m.start() + 9] + data_1[m.start() + 10]))
    time_1.append(counter)
    counter += 1

fs, audio_data = wavfile.read(audio_file)  # Extract the sampling frequency and the data points of the audio file.
audio_data_e = audio_data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
audio_data_e = butter_band_pass_filter(audio_data_e, 300, 2500, fs, order=6)  # Filtering the data.

data_e = []
for sample_1 in range(len(audio_data_e)):
    if audio_data_e[sample_1] < 0:
        data_e.append(audio_data_e[sample_1] * -1)  # Calculating absolute of the data.
    else:
        data_e.append(audio_data_e[sample_1])

analytic_signal = hilbert(data_e)
amplitude_envelope = np.abs(analytic_signal)

amplitude_envelope = moving_average(amplitude_envelope, 480)  # Calculating envelope of the audio file
amplitude_envelope = butter_low_pass_filter(amplitude_envelope, 100, fs, 5)  # Filtering out the high frequency ripples in the curve.
amplitude_envelope = moving_average(amplitude_envelope, 20)  # Smoothing the curve.

max1 = max(amplitude_envelope)  # Finding the maximum of the curve.
for sample_2 in range(len(amplitude_envelope)):
    amplitude_envelope[sample_2] = amplitude_envelope[sample_2] / max1  # Normalizing the curve

amplitude_envelope = amplitude_envelope.tolist()

data_e = moving_average(data_e, 480)  # Calculating envelope of the audio file
data_e = butter_low_pass_filter(data_e, 100, fs, 5)  # Filtering out the high frequency ripples in the curve.
data_e = moving_average(data_e, 20)  # Smoothing the curve.
data_e = data_e.tolist()

max2 = max(data_e)  # Finding the maximum of the curve.
for sample_2 in range(len(data_e)):
    data_e[sample_2] = data_e[sample_2] / max1  # Normalizing the curve

window_dur = 30
hop_dur = 5
threshold_smooth = 120

data_ste = butter_band_pass_filter(audio_data, 300, 2500, fs, order=6)  # Filter to remove fricatives
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data_ste) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data_ste = np.concatenate((data_ste, zero_array))
length = len(data_ste)  # Finding length of the actual data

st_energy = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = data_ste[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    st_energy.append(sum(frame ** 2))  # Calculating the short term energy
max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energy[i] = st_energy[i] / max_st_energy  # Normalizing the curve
    # ----------------------------------------------------------------------------------------------------------------------#
# if len(st_energy) < threshold_smooth:
#     st_energy = st_energy
# else:
#     st_energy = moving_average(st_energy, 20)

# plt.figure('Envelope')
# plt.plot(data_e)
#
# plt.figure('Short term energy')
# plt.plot(st_energy)
#
# plt.figure('Audio file')
# plt.plot(audio_data)

plt.figure('Compare')

plt.subplot(311)
plt.plot(audio_data)
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j]*fs, min(audio_data), max(audio_data), linestyles=':', color='black', linewidth='2.0')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2]*fs, min(audio_data), time_1[j], fontsize=10, color='black', rotation=0)  # Syllable Labels
plt.xlim(0, len(audio_data))

plt.subplot(312)
plt.plot(amplitude_envelope, 'blue')
plt.plot(data_e, 'red')
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j]*fs, min(amplitude_envelope), max(amplitude_envelope), linestyles=':', color='black', linewidth='2.0')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2]*fs, min(amplitude_envelope), time_1[j], fontsize=10, color='black', rotation=0)  # Syllable Labels
plt.xlim(0, len(amplitude_envelope))

plt.subplot(313)
plt.plot(st_energy)
plt.xlim(0, len(st_energy))

plt.show()
