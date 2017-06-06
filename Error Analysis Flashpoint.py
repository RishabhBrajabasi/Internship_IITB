"""
Error Analysis Flashpoint.py
Input:
1. Path of audio file and TextGrid file to be analyzed.
2. File number to be analyzed.
Output:
1. Image of the filtered audio data along with force aligned and flashpoint boundaries.
2. Image of the envelope with force aligned boundaries, flashpoint boundaries, peak and corresponding flashpoint information.
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, hilbert

file_no = '17'
path = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\'

audio_file = path + file_no + '.wav'
textgridFA = path + file_no + 'FA.TextGrid'


"""
Reading force aligned TextGrid data to determine boundaries.
"""
text_grid_1 = open(textgridFA, 'r')
tg_data_1 = text_grid_1.read()
time_1 = []
counter = 0
#----------------------------------------------------------------------------------------------------------------------#
for m in re.finditer('text = "', tg_data_1):
    if tg_data_1[m.start() - 33] == '=':
        time_1.append(float(
            tg_data_1[m.start() - 32] + tg_data_1[m.start() - 31] + tg_data_1[m.start() - 30] + tg_data_1[m.start() - 29] +
            tg_data_1[m.start() - 28] + tg_data_1[m.start() - 27] + tg_data_1[m.start() - 26]))
        time_1.append(float(
            tg_data_1[m.start() - 13] + tg_data_1[m.start() - 12] + tg_data_1[m.start() - 11] + tg_data_1[m.start() - 10] +
            tg_data_1[m.start() - 9] + tg_data_1[m.start() - 8] + tg_data_1[m.start() - 7] + tg_data_1[m.start() - 6] +
            tg_data_1[m.start() - 5]))
    else:
        time_1.append(float(
            tg_data_1[m.start() - 33] + tg_data_1[m.start() - 32] + tg_data_1[m.start() - 31] + tg_data_1[m.start() - 30] +
            tg_data_1[m.start() - 29] + tg_data_1[m.start() - 28] + tg_data_1[m.start() - 27] + tg_data_1[m.start() - 26]))
        time_1.append(float(
            tg_data_1[m.start() - 13] + tg_data_1[m.start() - 12] + tg_data_1[m.start() - 11] + tg_data_1[m.start() - 10] +
            tg_data_1[m.start() - 9] + tg_data_1[m.start() - 8] + tg_data_1[m.start() - 7] + tg_data_1[m.start() - 6] +
            tg_data_1[m.start() - 5]))
#----------------------------------------------------------------------------------------------------------------------#
    if tg_data_1[m.start() + 9] == '"':
        time_1.append((tg_data_1[m.start() + 8]))
    elif tg_data_1[m.start() + 10] == '"':
        time_1.append((tg_data_1[m.start() + 8] + tg_data_1[m.start() + 9]))
    else:
        time_1.append((tg_data_1[m.start() + 8] + tg_data_1[m.start() + 9] + tg_data_1[m.start() + 10]))
    time_1.append(counter)
    counter += 1
#----------------------------------------------------------------------------------------------------------------------#
"""

"""
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
"""

"""
fs, audio_data = wavfile.read(audio_file)  # Extract the sampling frequency and the data points of the audio file.
audio_data = audio_data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
audio_data = butter_band_pass_filter(audio_data, 300, 2500, fs, order=6)  # Filtering the data.

analytic_signal = hilbert(audio_data)
amplitude_envelope = np.abs(analytic_signal)
amplitude_envelope = moving_average(amplitude_envelope, 800)  # Calculating envelope of the audio file
amplitude_envelope = butter_low_pass_filter(amplitude_envelope, 100, fs, 5)  # Filtering out the high frequency ripples in the curve.
amplitude_envelope = moving_average(amplitude_envelope, 20)  # Smoothing the curve.

max1 = max(amplitude_envelope)  # Finding the maximum of the curve.
for sample_2 in range(len(amplitude_envelope)):
    amplitude_envelope[sample_2] = amplitude_envelope[sample_2] / max1  # Normalizing the curve
data1 = amplitude_envelope.tolist()
#----------------------------------------------------------------------------------------------------------------------#
"""
Find all 1-point maxima's in the envelope of the audio file. Maxima's which have a magnitude above the threshold are
marked as peaks. The magnitude of the peak and the location of the peak are stored.
"""
threshold_peak = 0.1  # Minimum value for a maxima to qualify as a peak.
peak_1 = []
location_peak_1 = []
for p in range(len(data1)):
    if p == 0:
        peak_1.append(0)
    elif p == len(data1) - 1:
        peak_1.append(0)
    else:
        if data1[p] > data1[p + 1] and data1[p] > data1[p - 1] and data1[p] >= threshold_peak:
            peak_1.append(data1[p])  # Finding value of the peak.
            location_peak_1.append(p)  # Finding location of the peak.
        else:
            peak_1.append(0)
#----------------------------------------------------------------------------------------------------------------------#
"""
Find flash points[called flash points after the DC comics character flash]. They indicate points on the envelope
of the audio file which lie at magnitude of 0.85 times the magnitude of the peak. There is one such point to the
peak and one such point to the right of the peak. If any other peaks lie within the region between these two flash
points then they are removed.

Format of 'the_list_1' :
Location of flash point to the left of the peak || Location of peak || Location of flash point to the right of the peak || tracker variable || Magnitude of peak
"""
the_list_1 = [[]]
count = 0
for l1 in range(len(location_peak_1)):
    value = data1[location_peak_1[l1]] * 0.85
    for ele in range(location_peak_1[l1], -1, -1):
        if data1[ele] < value:
            flash_1 = ele  # Finding the first flash point, to the left of the peak.
            break
    for ele in range(location_peak_1[l1], len(data1), 1):
        if data1[ele] < value:
            flash_2 = ele  # Finding the second flash point, to the right of the peak.
            break
    the_list_1.append([flash_1, location_peak_1[l1], flash_2, count, data1[location_peak_1[l1]]])
    count += 1
the_list_1.pop(0)

descending_order_of_peaks = sorted(the_list_1, key=lambda student: student[4], reverse=1)  # Sorting the list of peaks in descending order

remove = []  # List of peaks which are to be removed.
for l in range(len(descending_order_of_peaks)):
    for k in range(l + 1, len(descending_order_of_peaks)):
        if descending_order_of_peaks[l][0] < descending_order_of_peaks[k][1] < descending_order_of_peaks[l][2] and descending_order_of_peaks[k][3] not in remove:
            remove.append(descending_order_of_peaks[k][3])

the_list_2 = [[]]  # Shortlisted peak candidates
for iteration in the_list_1:
    if iteration[3] not in remove:
        the_list_2.append(iteration)
the_list_2.pop(0)
#----------------------------------------------------------------------------------------------------------------------#
"""
Between a pair of two peaks, the point with the least magnitude is found and is marked as a valley.
The staring point and the last point of the envelope are also marked as valleys. If the valley is located within
the region of the flash points of the peak, the peak is removed. Once the shortlisted peaks are obtained, valleys
between peaks are found again as mentioned earlier.
"""
valley_1 = [0]
for v1 in range(len(the_list_2) - 1):
    minimum = min(data1[the_list_2[v1][1]:the_list_2[v1 + 1][1]])  # Region between the two peaks
    valley_1.append(data1.index(minimum))
valley_1.append(len(data1) - 1)

the_list_3 = [[]]
fresh_count = 0  # Tracker variable to aid in removing peaks later.
for l3 in range(len(the_list_2)):
    the_list_3.append([valley_1[l3], the_list_2[l3][0], the_list_2[l3][1], the_list_2[l3][2], valley_1[l3 + 1], fresh_count])
    fresh_count += 1
the_list_3.pop(0)

remove[:] = []
for cc in the_list_3:
    if cc[3] > cc[4] or cc[0] > cc[1]:
        remove.append(cc[5])

peak_2 = []
for sf in the_list_3:
    if sf[5] not in remove:
        peak_2.append(sf[2])

valley_2 = [0]
for v in range(len(peak_2) - 1):
    minimum = min(data1[peak_2[v]:peak_2[v + 1]])
    valley_2.append(data1.index(minimum))
valley_2.append(len(data1) - 1)
#----------------------------------------------------------------------------------------------------------------------#
"""
Re-calculating flash points for the remaining peaks. Using flash point and valley information to determine
suitable boundary for start and end of vowel. Flash points are calculated as points having a magnitude of
0.707 times the magnitude of the peak.

Format of 'the_list_3':
Location of valley to the left of peak||Location of flash point to the left of the peak||Location of Peak||
Location of flash point to the right of the peak||Location of the valley to the right of the peak
"""

peak_list = []
the_list_4 = [[]]
for l3 in range(len(peak_2)):
    value = data1[peak_2[l3]] * 0.707
    for ele in range(peak_2[l3], -1, -1):
        if data1[ele] < value:
            flash_1 = ele
            break
    for ele in range(peak_2[l3], len(data1), 1):
        if data1[ele] < value:
            flash_2 = ele
            break
    the_list_4.append([valley_2[l3], flash_1, peak_2[l3], flash_2, valley_2[l3 + 1]])
    peak_list.append(peak_2[l3])

the_list_4.pop(0)
print len(the_list_4)
boundary = [[]]
for b in the_list_4:
    if b[0] < b[1] < b[3] < b[4]:  # valley_left < flash_left < flash_right < valley_right
        boundary.append([b[1], b[3]])  # flash_left <--> flash_right

    elif b[0] < b[1] and b[3] > b[4]:  # valley_left < flash_left and flash_right > valley_right
        boundary.append([b[1], b[4]])  # flash_left <--> valley_right

    elif b[0] > b[1] and b[3] < b[4]:  # valley_left > flash_left and flash_right < valley_right
        boundary.append([b[0], b[3]])  # valley_left <--> flash_right

    elif b[0] > b[1] and b[3] > b[4]:  # valley_left < flash_left and flash_right > valley_right
        boundary.append([b[0], b[4]])  # valley_left <--> valley_right
    else:
        boundary.append([b[0], b[4]])  # valley_left <--> valley_right
boundary.pop(0)
#----------------------------------------------------------------------------------------------------------------------#
"""
Subplot 1
Audio Data + Flashpoint boundaries + Syllable boundaries(Force aligned boundaries) + Syllable label.
Subplot 2
Envelope + Peak + Flashpoint's(0.85) + Flashpoint Boundaries + Syllable Boundaries(Force aligned boundaries) + Syllable label.
"""

plt.subplot(211)
for j in range(len(boundary)):
    plt.vlines(boundary[j][0], min(audio_data), max(audio_data), color='red', linewidth='2.0')  # Flashpoint boundaries
    plt.vlines(boundary[j][1], min(audio_data), max(audio_data), color='red', linewidth='2.0')  # Flashpoint boundaries
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j] * fs, min(audio_data), max(audio_data), linestyles=':', color='black', linewidth='2.0')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2] * fs, min(audio_data), time_1[j], fontsize=10, color='black', rotation=0)  # Syllable Labels
plt.plot(audio_data, 'black', label='Audio')
plt.xlabel('')
plt.ylabel('Normalised Magnitude')
# plt.vlines(boundary[0][1], min(audio_data), max(audio_data), color='red', linewidth='2.0', label='Vowel Boundary')  # Flashpoint boundaries
# plt.legend(loc='best')

plt.subplot(212)
plt.plot(data1)
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j] * fs, min(data1), max(data1), linestyles=':', color='black', linewidth='2.0')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2] * fs, min(data1), time_1[j], fontsize=10, color='black', rotation=0)  # Syllable Labels

for j in range(len(boundary)):
    plt.vlines(boundary[j][0], min(data1), max(data1), color='red', linewidth='2.0')  # Flashpoint Boundaries
    plt.vlines(boundary[j][1], min(data1), max(data1), color='red', linewidth='2.0')  # Flashpoint Boundaries

for p in range(len(the_list_4)):
    plt.scatter(the_list_4[p][2], data1[the_list_4[p][2]], color='green')  # Peak information
for pfp in range(len(the_list_3)):
    if the_list_3[pfp][2] in peak_list:
        plt.hlines(0.85 * data1[the_list_3[pfp][2]], the_list_3[pfp][1], the_list_3[pfp][3], color='black', linewidth='2.0') # Flashpoint

# plt.vlines(time_1[0] * fs, min(data1), max(data1), linestyles=':', color='black', linewidth='2.0', label='Force alignment boundaries')  # Syllable Boundaries
# plt.vlines(boundary[0][0] * fs, min(data1), max(data1), color='red', linewidth='2.0', label='Algorithm Boundaries')  # Syllable Boundaries
# plt.legend(loc='best')
plt.xlim(0, len(data1))
plt.ylim(0, max(data1))
plt.xlabel('')
plt.ylabel('Amplitude')
plt.show()

