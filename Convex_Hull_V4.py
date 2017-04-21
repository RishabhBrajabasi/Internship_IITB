from scipy.io import wavfile
import math
import matplotlib.pyplot as plt
import numpy as np

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation CH V2\Analyze\Vowel_Evaluation_V2_I1\\6.wav'
window_dur = 30
hop_dur = 5

fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
x_values = np.arange(0, len(data), 1) / float(fs)


gamma = []
sums = 0
auto = []
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    # frame_1 = frame.tolist()
    # frame_1.reverse()
    # for h in range(0,len(frame)-1):
    #     for t in range(0,len(frame)-h):
    #         sums = sums + frame[t+h]*frame[t]
    #     gamma.append(sums/len(frame))
    #     sums = 0
    auto.append(max(np.correlate(frame,frame)))
    gamma[:] = []

max_st_energy = max(auto)  # Maximum value of Short term energy curve
for i in range(no_frames):
    auto[i] = auto[i] / max_st_energy  # Normalizing the curve

seg = int(no_frames/200)
rem = no_frames%200
print seg, rem, no_frames
print auto
for i in range(0,seg-1):
    print i*200, (i+1)*200
    maxi = max(auto[i*200:(i+1)*200])
    for j in range(i*200,(i+1)*200):
        auto[j] = auto[j]/maxi
print auto
# for i in range(no_frames):


st_energy = []
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    st_energy.append(sum(frame ** 2))


# gamma = []
# sum = 0
# for h in range(0,len(data)-1):
#     for t in range(0, len(data) - h):
#         sum = sum + data[t+h]*data[t]
#     gamma.append(sum)
#     # print len(gamma)
#     sum = 0


plt.subplot(311)
plt.plot(data)
plt.subplot(312)
plt.plot(auto)
plt.subplot(313)
plt.plot(st_energy)
plt.show()

