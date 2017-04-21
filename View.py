from scipy.io import wavfile
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter



file_no = '56'

audiofile ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_6\\' + file_no + '.wav'
textgridFA = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_6\\' + file_no + 'FA.TextGrid'
textgridPE = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_6\\' + file_no + 'PE.TextGrid'

#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)

    ax.cla()
    ax.plot(plots[curr_pos][0], plots[curr_pos][1])
    for j in range(0, len(time_1), 4):
        plt.vlines(time_1[j], min(data), max(data), 'black')  # Syllable Boundaries
    for j in range(2, len(time_1), 4):
        plt.text(time_1[j - 2], min(data), time_1[j], fontsize=15, color='green', rotation=0)  # Syllable Labels

    plt.xlabel(labels[curr_pos])
    plt.xlim(0,x_values[-1])
    plt.ylim(min(data), max(data))
    fig.canvas.draw()
#----------------------------------------------------------------------------------------------------------------------#




#----------------------------------------------------------------------------------------------------------------------#
fs, data = wavfile.read(audiofile)  # Reading data from wav file in an array
data_0 = butter_bandpass_filter(data, 300, 2300, fs, order=6)
data_1 = butter_bandpass_filter(data, 300, 500, fs, order=6)
data_2 = butter_bandpass_filter(data, 2300, 7500, fs, order = 6)

x_values = np.arange(0, len(data), 1) / float(fs)
x_values_0 = np.arange(0, len(data_0), 1) / float(fs)
x_values_1 = np.arange(0, len(data_1), 1) / float(fs)
x_values_2 = np.arange(0, len(data_2), 1) / float(fs)

text_grid_1 = open(textgridFA, 'r')  # Open the FA TextGrid
text_1 = text_grid_1.read()
time_1 = []
counter = 0
for m in re.finditer('text = "', text_1):
    if text_1[m.start() - 33] == '=':
        time_1.append(float(
            text_1[m.start() - 32] + text_1[m.start() - 31] + text_1[m.start() - 30] + text_1[m.start() - 29] +
            text_1[m.start() - 28] + text_1[m.start() - 27] + text_1[m.start() - 26]))
        time_1.append(float(
            text_1[m.start() - 13] + text_1[m.start() - 12] + text_1[m.start() - 11] + text_1[m.start() - 10] +
            text_1[m.start() - 9] + text_1[m.start() - 8] + text_1[m.start() - 7] + text_1[m.start() - 6] +
            text_1[m.start() - 5]))
    else:
        time_1.append(float(
            text_1[m.start() - 33] + text_1[m.start() - 32] + text_1[m.start() - 31] + text_1[m.start() - 30] +
            text_1[m.start() - 29] + text_1[m.start() - 28] + text_1[m.start() - 27] + text_1[m.start() - 26]))
        time_1.append(float(
            text_1[m.start() - 13] + text_1[m.start() - 12] + text_1[m.start() - 11] + text_1[m.start() - 10] +
            text_1[m.start() - 9] + text_1[m.start() - 8] + text_1[m.start() - 7] + text_1[m.start() - 6] +
            text_1[m.start() - 5]))
    if text_1[m.start() + 9] == '"':
        time_1.append((text_1[m.start() + 8]))
    elif text_1[m.start() + 10] == '"':
        time_1.append((text_1[m.start() + 8] + text_1[m.start() + 9]))
    else:
        time_1.append((text_1[m.start() + 8] + text_1[m.start() + 9] + text_1[m.start() + 10]))

    time_1.append(counter)
    counter += 1

plots = [(x_values,data), (x_values_0,data_0), (x_values_1,data_1), (x_values_2,data_2)]
labels =['Unfiltered Data','[300 - 2300]','[300 - 3500]', '[2300 - 7500]']

curr_pos = 0
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.plot(x_values,data)
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j], min(data), max(data), 'black')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2], min(data), time_1[j], fontsize=15, color='green', rotation=0)  # Syllable Labels
plt.xlim(0,x_values[-1])
plt.ylim(min(data), max(data))
plt.xlabel('Unfiltered data')
plt.show()