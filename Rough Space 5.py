# ---------------------------------------------------------------------------------------------------------------------#
from scipy.io import wavfile
import numpy as np
import math


# ---------------------------------------------------------------------------------------------------------------------#
def location(arg1):
    arg3 = []
    for k in range(len(arg1)):
        if arg1[k] != 0:
            arg3.append(k)
    return arg3
# ---------------------------------------------------------------------------------------------------------------------#
audio_file = "F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_5\InputTestFile.wav"
window_dur = 30  # Duration of window in milliseconds
hop_dur = 10  # Hop duration in milliseconds
fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hamming(window_size)  # Window type: Hamming (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
noise_energy = 0
energy = [0] * length

# Squaring each data point
for j in range(length):
    energy[j] = data[j] * data[j]

# Calculating noise energy
for j in range(0, 800):  # energy
    noise_energy += energy[j]
noise_energy /= 800
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
st_energy = [0] * no_frames
maximum = [0] * no_frames
frame_number = [0] * no_frames
start = [0] * no_frames
stop = [0] * no_frames

# Calculating frame wise short term energy
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    frame_number[i] = i
    start[i] = i * hop_size
    stop[i] = i * hop_size + window_size

    st_energy[i] = sum(frame ** 2)

# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
peak = []

for j in range(no_frames):

    if j == 0:
        if st_energy[j] > st_energy[j + 1]:
            peak.append(st_energy[j])
        else:
            peak.append(0)

    elif j == no_frames - 1:
        if st_energy[j] > st_energy[j - 1]:
            peak.append(st_energy[j])
        else:
            peak.append(0)

    else:
        if st_energy[j] > st_energy[j + 1] and st_energy[j] > st_energy[j - 1]:
            peak.append(st_energy[j])
        else:
            peak.append(0)
# ---------------------------------------------------------------------------------------------------------------------#
# Finding the valleys
valley = []

for j in range(no_frames):
    if j == 0:
        if st_energy[j] < st_energy[j + 1]:
            valley.append(st_energy[j])

        else:
            peak.append(0)
    elif j == no_frames - 1:
        if st_energy[j] < st_energy[j - 1]:
            valley.append(st_energy[j])

        else:
            peak.append(0)
    else:
        if st_energy[j] < st_energy[j + 1] and st_energy[j] < st_energy[j - 1]:
            valley.append(st_energy[j])

        else:
            peak.append(0)
# ---------------------------------------------------------------------------------------------------------------------#
location_peak = location(peak)
location_valley = location(valley)
print "Number of peaks                        :", len(location_peak)
print "Number of valleys                      :", len(location_valley)
# ---------------------------------------------------------------------------------------------------------------------#
peak_1 = peak[:]
valley_1 = valley[:]
location_peak_1 = location_peak[:]
location_valley_1 = location_valley[:]
# ---------------------------------------------------------------------------------------------------------------------#
threshold = 0.04 * (noise_energy + (sum(peak) / len(location_peak_1)))
# ---------------------------------------------------------------------------------------------------------------------#
for j in range(len(peak)):
    if threshold < peak[j]:
        peak.pop(j)
        peak.insert(j, 0)

for j in range(len(peak) - 10):
    if peak[j] is not 0:
        for i in range(1, 10):
            if peak[j] < peak[j + i]:
                peak.pop(j)
                peak.insert(j, 0)

location_peak = location(peak)
location_valley = location(valley)
# ---------------------------------------------------------------------------------------------------------------------#
peak_2 = peak[:]
valley_2 = valley[:]
location_peak_2 = location_peak[:]
location_valley_2 = location_valley[:]
# ---------------------------------------------------------------------------------------------------------------------#
location = location_peak + location_valley
location.sort()
ripple = []


# for k in range(len(location_peak)):
#     q = location.index(location_peak[k])
#     print q
# #     ripple.append(location[q-1])
# #     ripple.append(location[q])
# #     ripple.append(location[q+1])
# # #
# # # ripple_value = []
# # for k in range(1, len(ripple), 3):
# #     ripple_value.append((st_energy[ripple[k]]-st_energy[ripple[k+1]])/(st_energy[ripple[k]]-st_energy[ripple[k-1]]))
# #
# # for q in range(len(ripple_value)-1):
# #     if ((ripple_value[q] > 3) and (ripple_value[q+1] < 1.4)) or ((ripple_value[q] > 1.02) and (ripple_value[q+1] < 0.3)):
# #         if peak[location_peak[q]] > peak[location_peak[q+1]]:
# #             peak.pop(location_peak[q+1])
# #             peak.insert(location_peak[q+1], 0)
# #
# #         if peak[location_peak[q]] < peak[location_peak[q + 1]]:
# #             peak.pop(location_peak[q])
# #             peak.insert(location_peak[q], 0)
