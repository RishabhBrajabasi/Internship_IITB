from scipy.signal import butter, lfilter
from scipy.io import wavfile
import math
import numpy as np
import matplotlib.pyplot as plt


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

audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_M6\\17.wav'
window_dur = 50
hop_dur = 7

fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
x_values = np.arange(0, len(data), 1) / float(fs)

st_energy = []
st_energy_filter = []
data_filter = butter_bandpass_filter(data,100,1100,fs,7)

# plt.plot(x_values,data)
# plt.plot(x_values,data_filter)


for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    st_energy.append(sum(frame ** 2))
norm_max_square = max(st_energy)

for i in range(no_frames):
    st_energy[i] = st_energy[i] / norm_max_square


for i in range(no_frames):
    frame = data_filter[i * hop_size:i * hop_size + window_size] * window_type
    st_energy_filter.append(sum(frame ** 2))
norm_max_square_filter = max(st_energy_filter)

for i in range(no_frames):
    st_energy_filter[i] = st_energy_filter[i] / norm_max_square_filter

# plt.subplot(211)
plt.plot(st_energy)
# plt.subplot(212)
plt.plot(st_energy_filter)




plt.show()