from scipy.io import wavfile
import math
import numpy as np
import matplotlib.pyplot as plt

audio_file_1 = 'C:\Users\Mahe\Desktop\\sh.wav'
audio_file_2 = 'C:\Users\Mahe\Desktop\\d.wav'
window_dur = 25
hop_dur = 5


fs, data = wavfile.read(audio_file_2)  # Reading data from wav file in an array
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
x_values = np.arange(0, len(data), 1) / float(fs)


#
# plt.figure('Sound Waveform')
# plt.plot(x_values, data)
# plt.xlabel('Time in seconds')
# plt.ylabel('Amplitude')
# plt.title('Sound Waveform')
# plt.show()
#######################################################################################################################
# sums = 0
# gamma = [0]*len(data)
# big_gamma = [0]*len(data)
# for h in range(0,len(data)-1):
#     for t in range(0,len(data)-h):
#         sums = sums + data[t+h]*data[t]
#     gamma[h] = (sums/len(data))
#     sums = 0
#     big_gamma[h] = (gamma[h]/(len(data)-h))/(gamma[0]/len(data))
#     # gamma[:]= []
# # print gamma
# plt.plot(big_gamma)
# plt.show()
# st_energy = [0] * no_frames
# auto = []
# conv = []
# for i in range(no_frames):
#     frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
#     sums = 0
#     gamma = [0] * len(frame)
#     big_gamma = [0] * len(frame)
#     for h in range(0, len(frame) - 1):
#         for t in range(0, len(frame) - h):
#             sums = sums + frame[t + h] * frame[t]
#         gamma[h] = (sums / len(frame))
#         sums = 0
#         big_gamma[h] = (gamma[h] / (len(frame) - h)) / (gamma[0] / len(frame))

    # auto.append(max(big_gamma))
plt.subplot(211)
plt.plot((np.correlate(data, data,'full')))
plt.subplot(212)
plt.plot((np.convolve(data, data)))
plt.show()
# plt.subplot(211)
# plt.plot(auto)
# plt.subplot(212)
# plt.plot(conv)
# plt.show()

plt.plot(big_gamma)
plt.show()