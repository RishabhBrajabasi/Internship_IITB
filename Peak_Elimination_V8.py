from scipy.io import wavfile
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


audiofile ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_Repeat_2\\17.wav'
textgridFA = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_Repeat_2\\17FA.TextGrid'
textgridPE = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_Repeat_2\\17PE.TextGrid'



window_dur = 50
hop_dur = 7
threshold_smooth =100
fs, data = wavfile.read(audiofile)
data = data / float(2 ** 15)
window_size = int(window_dur * fs * 0.001)
hop_size = int(hop_dur * fs * 0.001)
window_type = np.hanning(window_size)
no_frames = int(math.ceil(len(data) / (float(hop_size))))
zero_array = np.zeros(window_size)
data = np.concatenate((data, zero_array))
length = len(data)
x_values = np.arange(0, len(data), 1) / float(fs)

st_energy = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    st_energy.append(sum(frame ** 2))  # Calculating the short term energy
max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energy[i] = st_energy[i] / max_st_energy  # Normalizing the curve
    # ----------------------------------------------------------------------------------------------------------------------#
original = st_energy
if len(st_energy) < threshold_smooth:
    st_energy = st_energy
else:
    st_energy = moving_average(st_energy, 15)
    # st_energy = savitzky_golay(st_energy,51,5)

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
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
#----------------------------------------------------------------------------------------------------------------------#
listing = []
print time_1
print time_2
for outer in range(0, len(time_2), 2):
    for inner in range(0, len(time_1), 4):
        if time_1[inner] <= time_2[outer] < time_1[inner + 1] and time_1[inner] < time_2[outer + 1] <= time_1[
                    inner + 1]:
            listing.append(time_1[inner + 2])
            listing.append(time_1[inner + 3])

for outer in range(0, len(time_2), 2):
    for inner in range(0, len(time_1) - 4, 4):
        if time_1[inner] < time_2[outer] < time_1[inner + 1] and time_1[inner + 4] < time_2[outer + 1] < time_1[
                    inner + 5]:
            listing.append(time_1[inner + 2])
            listing.append(time_1[inner + 3])
            listing.append(time_1[inner + 6])
            listing.append(time_1[inner + 7])

count = 0
vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']

already_here = []
for vowel_sound in range(0, len(listing), 2):
    if listing[vowel_sound] in vowel_data and listing[vowel_sound + 1] not in already_here:
        count += 1
        already_here.append(listing[vowel_sound + 1])

plt.plot(x_values, data)
plt.xlim(0,x_values[-1])
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j], min(data)+0.30*min(data), max(data), 'black')
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2], min(data)+0.28*min(data), time_1[j], fontsize=15, color='green', rotation=0)
for j in range(len(time_2)):
    plt.vlines(time_2[j], min(data)+0.2*min(data), max(data), 'red')
for j in range(0,len(time_2),2):
    try:
        plt.arrow(time_2[j], min(data)+0.2*min(data), (time_2[j + 1] - time_2[j])-0.01, 0, head_width=0.005, head_length=0.01,color='red')
        plt.arrow(time_2[j+1], min(data)+0.2*min(data), -(time_2[j + 1] - time_2[j]) + 0.01, 0, head_width=0.005, head_length=0.01,color='red')
    except:
        print "Meh"
for j in range(0,len(time_1),4):
    try:
        plt.arrow(time_1[j], min(data)+0.30*min(data), (time_1[j + 1] - time_1[j])-0.01, 0, head_width=0.005, head_length=0.01)
        plt.arrow(time_1[j+1], min(data)+0.30*min(data), -(time_1[j + 1] - time_1[j]) + 0.01, 0, head_width=0.005, head_length=0.01)
    except:
        print "Meh"
for j in range(0, len(time_2), 2):
    plt.text(time_2[j], min(data)+0.2*min(data), 'Vowel', fontsize=12, color='red')
plt.show()

plt.subplot(211)
plt.plot(x_values, data)
plt.xlim(0,x_values[-1])
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j], min(data)+0.30*min(data), max(data), 'black')
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2], min(data)+0.28*min(data), time_1[j], fontsize=15, color='green', rotation=0)
plt.subplot(212)
plt.plot(st_energy)
plt.plot(original)
plt.xlim(0,len(st_energy))

plt.show()
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#