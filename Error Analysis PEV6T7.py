from __future__ import division

import math
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

#----------------------------------------------------------------------------------------------------------------------#
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def filter_bank(o_data, low_pass, high_pass, fs, order_of_filter, window_dur, hop_dur):
    atad = butter_bandpass_filter(o_data, low_pass, high_pass, fs, order_of_filter)
    window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
    hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
    window_type = np.hanning(window_size)  # Window type: Hanning (by default)
    no_frames = int(math.ceil(len(atad) / (float(hop_size))))  # Determining the number of frames
    zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
    atad = np.concatenate((atad, zero_array))
    st_energy = []
    for i in range(no_frames):  # Calculating frame wise short term energy
        frame = atad[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
        st_energy.append(sum(frame ** 2))  # Calculating the short term energy
    max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
    for i in range(no_frames):
        st_energy[i] = st_energy[i] / max_st_energy  # Normalizing the curve
    return st_energy, atad
#----------------------------------------------------------------------------------------------------------------------#
file_no = '1372'
audio_file ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + '.wav'
textgridFA = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + 'FA.TextGrid'
textgridPE = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + 'PE.TextGrid'

window_dur=50
hop_dur = 7
threshold_smooth = 120
#----------------------------------------------------------------------------------------------------------------------#
fs, data_f = wavfile.read(audio_file)  # Reading data from wav file in an array
data_f = data_f / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]



fs, datao = wavfile.read(audio_file)  # Reading data from wav file in an array
datao = datao / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
length_o = len(datao)
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(datao) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
datao = np.concatenate((datao, zero_array))
length = len(datao)  # Finding length of the actual data

st_energyo = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = datao[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    st_energyo.append(sum(frame ** 2))  # Calculating the short term energy
max_st_energy = max(st_energyo)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energyo[i] = st_energyo[i]/max_st_energy  # Normalizing the curve




fs, data1 = wavfile.read(audio_file)  # Reading data from wav file in an array
data1 = data1 / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
data = butter_bandpass_filter(data1, 300, 2500, fs, order=6)
length_o = len(data)
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
x_values = np.arange(0, len(data), 1) / float(fs)
#----------------------------------------------------------------------------------------------------------------------#
noise_energy = 0  # Initializing noise energy
energy = [0] * length  # Initializing list energy
for bit in range(length):
    energy[bit] = data[bit] * data[bit]  # Squaring each point of the data to calculate noise energy
for ne in range(0, 800):
    noise_energy += energy[ne]  # Taking the first 800 samples of the original sound file
noise_energy /= 800  # Averaging the square of the first 800 noise samples
#----------------------------------------------------------------------------------------------------------------------#
st_energy = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    st_energy.append(sum(frame ** 2))  # Calculating the short term energy
max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energy[i] = st_energy[i]/max_st_energy  # Normalizing the curve

original_st_energy = st_energy
#----------------------------------------------------------------------------------------------------------------------#
if len(st_energy) < threshold_smooth:
    st_energy = st_energy
else:
    st_energy = moving_average(st_energy, 20)

#----------------------------------------------------------------------------------------------------------------------#
peak = []  # Initializing list
count_of_peaks = 0  # Initializing no of peaks
for p in range(len(st_energy)):
    if p == 0:  # First element
        if st_energy[p] > st_energy[p + 1]:  # If the first element is greater than the succeeding element it is a peak.
            peak.append(st_energy[p])  # Append the energy level of the peak
            count_of_peaks += 1  # Increment count
        else:
            peak.append(0)  # Else append a zero
    elif p == len(st_energy) - 1:  # Last element
        if st_energy[p] > st_energy[p - 1]:  # If the last element is greater than the preceding element it is a peak.
            peak.append(st_energy[p])  # Append the energy level of the peak
            count_of_peaks += 1  # Increment count
        else:
            peak.append(0)  # Else append a zero
    else:  # All the other elements
        if st_energy[p] > st_energy[p + 1] and st_energy[p] > st_energy[p - 1]:  # If the element is greater than the element preceding and succeeding it, it is a peak.
            peak.append(st_energy[p])  # Append the energy level of the peak
            count_of_peaks += 1  # Increment count
        else:
            peak.append(0)  # Else append a zero
#----------------------------------------------------------------------------------------------------------------------#
threshold = 0.01 + 0.04 * (noise_energy + (sum(peak) / count_of_peaks))  # The threshold which eliminates minor peaks.
#----------------------------------------------------------------------------------------------------------------------#
count_of_peaks_threshold = 0
peak_threshold = []
location_peak = []
for p in range(len(peak)):
    if threshold < peak[p]:  # If the peak value is greater than the threshold
        peak_threshold.append(peak[p])  # Append the energy level to a new list
        count_of_peaks_threshold += 1  # Increment count
        location_peak.append(p)  # Make note of the location of the peak
    else:
        peak_threshold.append(0)  # Else append zero
#----------------------------------------------------------------------------------------------------------------------#
valley = []
count_of_valleys = 0
location_valley = []
for p in range(len(st_energy)):
    if p == 0:  # For the first element
        if st_energy[p] < st_energy[p + 1]:  # If the first element is lesser than the succeeding element
            valley.append(st_energy[p])  # Append the energy level of the valley
            count_of_valleys += 1  # Increment the count
            location_valley.append(p)  # Make note of the position of the valley
        else:
            valley.append(0)  # Else append zero
    elif p == len(st_energy) - 1:  # For the last element
        if st_energy[p] < st_energy[p - 1]:  # If the last element is lesser than the preceding element
            valley.append(st_energy[p])  # Append the energy level of the valley
            count_of_valleys += 1  # Increment the count
            location_valley.append(p)  # Make note of the position of the valley
        else:
            valley.append(0)  # Else append zero
    else:
        if st_energy[p] < st_energy[p + 1] and st_energy[p] < st_energy[
                    p - 1]:  # If the element is lesser than the element preceding and succeeding it
            valley.append(st_energy[p])  # Append the energy level of the valley
            count_of_valleys += 1  # Increment the count
            location_valley.append(p)  # Make note of the position of the valley
        else:
            valley.append(0)  # Else append zero
#----------------------------------------------------------------------------------------------------------------------#
location = location_peak + location_valley  # Combing the list of the location of all the peaks and valleys
location.sort()  # Sorting it so that each peak has a valley to it's left and right
ripple_valley = []
ripple_peak = []
ripple = []
# What we need is only the valleys to the left and right of the peak. The other valleys are not important
for k in range(len(location_peak)):
    q = location.index(location_peak[k])  # Extracting the location of the peak
    if location_peak[k] == len(peak) - 1:  # If the peak is the last element of the short term energy curve
        ripple.append(location[q - 1])  # The location of the valley before the last peak is added
        ripple_valley.append(location[q - 1])  # The location of the valley before the last peak is added
        ripple.append(location[q])  # The location of the peak is added
        ripple_peak.append(location[q])  # The location of the peak is added
        ripple.append(location[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
        ripple_valley.append(location[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
    elif location_peak[k] == 0:  # If the peak is the first element of the short term energy curve
        ripple.append(location[q + 1])  # The location of the valley after the first peak is added
        ripple_valley.append(location[q + 1])  # The location of the valley after the first peak is added
        ripple.append(location[q])  # The location of the peak is added
        ripple_peak.append(location[q])  # The location of the peak is added
        ripple.append(location[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
        ripple_valley.append(location[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
    else:  # For every other element
        ripple.append(location[q - 1])  # The location of the valley before the peak is added
        ripple_valley.append(location[q - 1])  # The location of the valley before the peak is added
        ripple.append(location[q])  # The location of the peak is added
        ripple_peak.append(location[q])  # The location of the peak is added
        ripple.append(location[q + 1])  # The location of the valley after the peak is added
        ripple_valley.append(location[q + 1])  # The location of the valley after the peak is added
#----------------------------------------------------------------------------------------------------------------------#
value_valley =[]
for i in range(len(ripple_valley)):
    value_valley.append(st_energy[ripple_valley[i]])

ripple_value = []
for k in range(1, len(ripple), 3):
    ripple_value.append((st_energy[ripple[k]] - st_energy[ripple[k + 1]]) / (st_energy[ripple[k]] - st_energy[ripple[k - 1]]))
#----------------------------------------------------------------------------------------------------------------------#
ripple_value = []
for k in range(1, len(ripple), 3):
    ripple_value.append(
        (st_energy[ripple[k]] - st_energy[ripple[k + 1]]) / (st_energy[ripple[k]] - st_energy[ripple[k - 1]]))

loc = []
for k in range(len(ripple_value)):
    loc.append(location_peak[ripple_value.index(ripple_value[k])])
#----------------------------------------------------------------------------------------------------------------------#
for k in range(len(ripple_value)):
    if k != len(ripple_value) - 1:
        if location_peak[ripple_value.index(ripple_value[k + 1])] - location_peak[ripple_value.index(ripple_value[k])] < 20:
            if ripple_value[k] > 3.0 and ripple_value[k + 1] < 1.4 or ripple_value[k] > 1.02 and ripple_value[k + 1] < 0.3:
                v1 = st_energy[location_peak[ripple_value.index(ripple_value[k])]]
                v2 = st_energy[location_peak[ripple_value.index(ripple_value[k + 1])]]
                if v1 >= v2:
                    loc.remove(location_peak[ripple_value.index(ripple_value[k + 1])])
                else:
                    loc.remove(location_peak[ripple_value.index(ripple_value[k])])
    else:
        if ripple_value[k] > 3.0:
            loc.remove(location_peak[ripple_value.index(ripple_value[k])])
#----------------------------------------------------------------------------------------------------------------------#
peak_threshold[:] = []
for j in range(no_frames):
    if j in loc:
        peak_threshold.append(st_energy[loc.index(j)])
    else:
        peak_threshold.append(0)
#----------------------------------------------------------------------------------------------------------------------#
text_grid_1 = open(textgridFA, 'r')  # Open the FA TextGrid
text_grid_2 = open(textgridPE, 'r')  # Open the TextGrid created by the script
data_1 = text_grid_1.read()  # Read and assign the content of the FA TextGrid to data_1
data_2 = text_grid_2.read()  # Read and assign the content of the created TextGrid to data_2
time_1 = []  # Creating an empty list to record time
time_2 = []
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
#----------------------------------------------------------------------------------------------------------------------#
for m in re.finditer('"Vowel"', data_2):
    time_2.append(float(
        data_2[m.start() - 34] + data_2[m.start() - 33] + data_2[m.start() - 32] + data_2[m.start() - 31] +
        data_2[m.start() - 30] + data_2[m.start() - 29]))
    time_2.append(float(
        data_2[m.start() - 17] + data_2[m.start() - 16] + data_2[m.start() - 15] + data_2[m.start() - 14] +
        data_2[m.start() - 13] + data_2[m.start() - 12]))




st_energy_1, f_data_1 = filter_bank(data_f, 200,400,fs,6,window_dur,hop_dur)
st_energy_2, f_data_2 = filter_bank(data_f, 400,630,fs,6,window_dur,hop_dur)
st_energy_3, f_data_3 = filter_bank(data_f, 630,920,fs,6,window_dur,hop_dur)
st_energy_4, f_data_4 = filter_bank(data_f, 920,1270,fs,6,window_dur,hop_dur)
st_energy_5, f_data_5 = filter_bank(data_f, 1270,1720,fs,6,window_dur,hop_dur)
st_energy_6, f_data_6 = filter_bank(data_f, 1720,2320,fs,6,window_dur,hop_dur)
st_energy_7, f_data_7 = filter_bank(data_f, 2320,3200,fs,6,window_dur,hop_dur)
st_energy_8, f_data_8 = filter_bank(data_f, 3200,7000,fs,6,window_size,hop_dur)

#---------------------------------------------------------------------------------------------------------------------#
plt.subplot(311)
plt.plot(x_values, data)  # The Original Data
plt.xlim(0,x_values[-1])  # Limiting it to fixed range for representational purposes
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j], min(data)+0.30*min(data), max(data), 'black')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2], min(data)+0.28*min(data), time_1[j], fontsize=15, color='green', rotation=0)  # Syllable Labels
for j in range(len(time_2)):
    plt.vlines(time_2[j], min(data), max(data), 'red')  # Vowel Boundaries
for j in range(0, len(time_2), 2):
    plt.text(time_2[j], max(data), 'Vowel', fontsize=12, color='red')  # Vowel Label
for j in range(0,len(time_2),2):  # Bounding arrows for Vowel
    plt.arrow(time_2[j], max(data), (time_2[j + 1] - time_2[j])-0.01, 0, head_width=0.005, head_length=0.01,color='red')
    plt.arrow(time_2[j+1], max(data), -(time_2[j + 1] - time_2[j]) + 0.01, 0, head_width=0.005, head_length=0.01,color='red')
for j in range(0,len(time_1),4):  # Bounding arrows for Syllable
    plt.arrow(time_1[j], min(data)+0.30*min(data), (time_1[j + 1] - time_1[j])-0.01, 0, head_width=0.005, head_length=0.01)
    plt.arrow(time_1[j+1], min(data)+0.30*min(data), -(time_1[j + 1] - time_1[j]) + 0.01, 0, head_width=0.005, head_length=0.01)
plt.xlabel('Time (In seconds)')
plt.ylabel('Amplitude')
plt.title('Sound Waveform',color='blue')

plt.subplot(312)
plt.plot(st_energy)  # Smoothed Short term energy
plt.plot(original_st_energy)
plt.plot(st_energyo)
for i in range(len(location_peak)):
    plt.scatter(location_peak[i], st_energy[location_peak[i]], color='red', label='Peak')
plt.scatter(ripple_valley, value_valley, color='green', label='Valley')
for j in range(len(location_peak)):
    plt.text(location_peak[j], st_energy[location_peak[j]], str(round(ripple_value[j], 2)))
for j in range(len(loc)):
    plt.vlines(loc[j], min(st_energy), max(st_energy), 'black')  # Vowel Centres
plt.xlim(0,len(st_energy))  # Limiting it to fixed range for representational purposes

plt.subplot(313)
plt.plot(st_energy_1,'red',label='[200-400]')
plt.plot(st_energy_2,'orange',label='[400-630]')
plt.plot(st_energy_3,'yellow',label='[630-920]')
plt.plot(st_energy_8,'black',label='[3200-7500]')
plt.plot(st_energy_4,'green',label='[920-1270]')
plt.plot(st_energy_5,'blue',label='[1270-1720]')
plt.plot(st_energy_6,'indigo',label='[1720-2320]',ls='dotted')
plt.plot(st_energy_7,'violet',label='[2320-3200]',ls='dashed')

plt.xlim(0,len(st_energy_1))
# plt.legend()
plt.xlabel('No. of frames')
plt.ylabel('Normalised Magnitude')
plt.title('Short Term Energy')
plt.show()
plt.show()


