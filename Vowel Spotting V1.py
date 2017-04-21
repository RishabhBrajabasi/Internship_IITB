from scipy.io import wavfile
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

# Provide the absolute path of wav file as argument to script  (input argument)
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Spotting V1\InputTestFile.wav'

window_dur = 30  # Duration of window in milliseconds
hop_dur = 10  # Hop duration in milliseconds

text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel Spotting V1\Data\Mark.csv',
                   'w')  # Output file for storing features

fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
#  Returns Sample rate and data read from  wav file

data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]

window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hamming(window_size)  # Window type: Hamming (by default)

no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames

zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))

length = len(data)  # Finding length of the actual data

# Work on Data
energy = [0] * length

# Calculating Noise Energy for use in threshold operations
noise_energy = 0
for j in range(0, 800):  # energy
    noise_energy += energy[j]
noise_energy /= 800

# Work on Frames
st_energy = [0] * no_frames
maximum = [0] * no_frames

# When you want to plot all the frames, Run it sparingly
# fileNameTemplate = r'C:\Users\Mahe\Desktop\Hopethisworks\Plot{0:02d}.png'

frame_number = [0] * no_frames
start = [0] * no_frames
stop = [0] * no_frames

for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    frame_number[i] = i
    start[i] = i * hop_size
    stop[i] = i * hop_size + window_size

    # Do frame wise processing here
    # ---------- For example: Computing short time energy -------------------
    st_energy[i] = sum(frame ** 2)
    maximum[i] = max(frame)


count_of_peaks = 0
count_of_valleys = 0

# Find the peaks[Maxima]
peak = []
for j in range(1, no_frames - 1):
    if st_energy[j] > st_energy[j + 1] and st_energy[j] > st_energy[j - 1]:
        peak.append(st_energy[j])
        count_of_peaks += 1
    else:
        peak.append(0)

# Find the valleys[Minima]
valley = []
for j in range(1, no_frames - 1):
    if st_energy[j] < st_energy[j + 1] and st_energy[j] < st_energy[j - 1]:
        valley.append(st_energy[j])
        count_of_valleys += 1
    else:
        valley.append(0)

# Threshold operation to eliminate peaks
threshold = 0.04 * (noise_energy + (sum(peak) / count_of_peaks))


# Calculating peaks which are greater than the threshold
count_of_maxi = 0
maxi = []

for j in range(len(peak)):
    if threshold < peak[j]:
        maxi.append(peak[j])
        count_of_maxi += 1
    else:
        maxi.append(0)

# Removing adjacent peaks
for j in range(len(maxi) - 10):
    if maxi[j] is not 0:
        for i in range(1, 10):
            if maxi[j] < maxi[j + i]:
                maxi[j] = 0

# The start and end of the regions of interest
mark = []
for j in range(len(maxi)):
    if maxi[j] is not 0:
        mark.append(j * hop_size)
        mark.append(j * hop_size + window_size)

work = [0] * length
for j in range(0, len(mark) - 1):
    work.insert(mark[j], 1)


go = [0] * length

for k in range(0, len(mark), 2):
    for j in range(mark[k], mark[k + 1]):
        go.pop(j)
        go.insert(j, 1)


data = data * go

for i in range(0, len(mark), 2):
    text_file_1.write(str(mark[i] * 0.0000625) + "\t" + str(mark[i+1]*0.0000625) + "\t" + str(i) + "\n")

# Storing enhanced wavefile (Just to show input data array is written as output wav file)
wavfile.write("F:\Projects\Active Projects\Project Intern_IITB\Vowel Spotting V1\OutputTestFile.wav", fs, data)


