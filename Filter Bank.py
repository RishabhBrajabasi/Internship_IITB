from scipy.io import wavfile
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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

file_no = '27'

audio_file ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_M12\\' + file_no + '.wav'
textgridFA = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_M12\\' + file_no + 'FA.TextGrid'

window = 50
hop = 7

fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]

st_energy_1, f_data_1 = filter_bank(data, 200,400,fs,6,window,hop)
st_energy_2, f_data_2 = filter_bank(data, 400,630,fs,6,window,hop)
st_energy_3, f_data_3 = filter_bank(data, 630,920,fs,6,window,hop)
st_energy_4, f_data_4 = filter_bank(data, 920,1270,fs,6,window,hop)
st_energy_5, f_data_5 = filter_bank(data, 1270,1720,fs,6,window,hop)
st_energy_6, f_data_6 = filter_bank(data, 1720,2320,fs,6,window,hop)
st_energy_7, f_data_7 = filter_bank(data, 2320,3200,fs,6,window,hop)

# st_energy_sum = [0] * len(st_energy_1)
# for i in range(len(st_energy_1)):
#     st_energy_sum[i] = st_energy_1[i] + st_energy_2[i] + st_energy_3[i] + st_energy_4[i] + st_energy_5[i] + st_energy_6[i] + st_energy_7[i]
# max_st_energy_sum = max(st_energy_sum)  # Maximum value of Short term energy curve
# for i in range(len(st_energy_sum)):
#     st_energy_sum[i] = st_energy_sum[i] / max_st_energy_sum  # Normalizing the curve


window_size = int(window * fs * 0.001)  # Converting window length to samples
hop_size = int(hop * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
x_values = np.arange(0, len(data), 1) / float(fs)

st_energy = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    st_energy.append(sum(frame ** 2))  # Calculating the short term energy
max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energy[i] = st_energy[i] / max_st_energy  # Normalizing the curve


# plt.subplot(8,1,1)
# plt.plot(data)
# plt.subplot(8,1,2)
# plt.plot(f_data_1)
# plt.subplot(8,1,3)
# plt.plot(f_data_2)
# plt.subplot(8,1,4)
# plt.plot(f_data_3)
# plt.subplot(8,1,5)
# plt.plot(f_data_4)
# plt.subplot(8,1,6)
# plt.plot(f_data_5)
# plt.subplot(8,1,7)
# plt.plot(f_data_6)
# plt.subplot(8,1,8)
# plt.plot(f_data_7)
# plt.show()

# plt.subplot(111)
# plt.plot(st_energy_1,'red',label='[200-400]')
# plt.plot(st_energy_2,'orange',label='[400-630]')
# plt.plot(st_energy_3,'yellow',label='[630-920]')
# plt.plot(st_energy_4,'green',label='[920-1270]')
# plt.plot(st_energy_5,'blue',label='[1270-1720]')
# plt.plot(st_energy_6,'indigo',label='[1720-2320]')
# plt.plot(st_energy_7,'violet',label='[2320-3200]')
# plt.legend()
# plt.show()

#----------------------------------------------------------------------------------------------------------------------#
text_grid_1 = open(textgridFA, 'r')  # Open the FA TextGrid
data_FA = text_grid_1.read()  # Read and assign the content of the FA TextGrid to data_1
time_1 = []  # Creating an empty list to record time
counter = 0
#----------------------------------------------------------------------------------------------------------------------#
for m in re.finditer('text = "', data_FA):
    if data_FA[m.start() - 33] == '=':
        time_1.append(float(
            data_FA[m.start() - 32] + data_FA[m.start() - 31] + data_FA[m.start() - 30] + data_FA[m.start() - 29] +
            data_FA[m.start() - 28] + data_FA[m.start() - 27] + data_FA[m.start() - 26]))
        time_1.append(float(
            data_FA[m.start() - 13] + data_FA[m.start() - 12] + data_FA[m.start() - 11] + data_FA[m.start() - 10] +
            data_FA[m.start() - 9] + data_FA[m.start() - 8] + data_FA[m.start() - 7] + data_FA[m.start() - 6] +
            data_FA[m.start() - 5]))
    else:
        time_1.append(float(
            data_FA[m.start() - 33] + data_FA[m.start() - 32] + data_FA[m.start() - 31] + data_FA[m.start() - 30] +
            data_FA[m.start() - 29] + data_FA[m.start() - 28] + data_FA[m.start() - 27] + data_FA[m.start() - 26]))
        time_1.append(float(
            data_FA[m.start() - 13] + data_FA[m.start() - 12] + data_FA[m.start() - 11] + data_FA[m.start() - 10] +
            data_FA[m.start() - 9] + data_FA[m.start() - 8] + data_FA[m.start() - 7] + data_FA[m.start() - 6] +
            data_FA[m.start() - 5]))
#----------------------------------------------------------------------------------------------------------------------#
    if data_FA[m.start() + 9] == '"':
        time_1.append((data_FA[m.start() + 8]))
    elif data_FA[m.start() + 10] == '"':
        time_1.append((data_FA[m.start() + 8] + data_FA[m.start() + 9]))
    else:
        time_1.append((data_FA[m.start() + 8] + data_FA[m.start() + 9] + data_FA[m.start() + 10]))

    time_1.append(counter)
    counter += 1
#----------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
plt.subplot(211)
plt.plot(x_values,data)  # The Original Data
plt.xlim(0,x_values[-1])  # Limiting it to fixed range for representational purposes
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j], min(data)+0.30*min(data), max(data), 'black')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2], min(data)+0.28*min(data), time_1[j], fontsize=15, color='green', rotation=0)  # Syllable Labels
for j in range(0,len(time_1),4):  # Bounding arrows for Syllable
    plt.arrow(time_1[j], min(data)+0.30*min(data), (time_1[j + 1] - time_1[j])-0.01, 0, head_width=0.005, head_length=0.01)
    plt.arrow(time_1[j+1], min(data)+0.30*min(data), -(time_1[j + 1] - time_1[j]) + 0.01, 0, head_width=0.005, head_length=0.01)
plt.xlabel('Time (In seconds)')
plt.ylabel('Amplitude')
plt.title('Sound Waveform',color='blue')
plt.subplot(212)
plt.plot(st_energy_1,'red',label='[200-400]')
plt.plot(st_energy_2,'orange',label='[400-630]')
plt.plot(st_energy_3,'yellow',label='[630-920]')
plt.plot(st_energy_4,'green',label='[920-1270]')
plt.plot(st_energy_5,'blue',label='[1270-1720]')
plt.plot(st_energy_6,'indigo',label='[1720-2320]',ls='dotted')
plt.plot(st_energy_7,'violet',label='[2320-3200]',ls='dashed')
plt.xlim(0,len(st_energy_1))
plt.legend()
plt.xlabel('No. of frames')
plt.ylabel('Normalised Magnitude')
plt.title('Short Term Energy')
plt.show()
#---------------------------------------------------------------------------------------------------------------------#
